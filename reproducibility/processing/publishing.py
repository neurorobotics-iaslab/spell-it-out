"""Export reviewed Spell-IT-OUT preprocessing as a BIDS derivative."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shutil
from typing import Iterable

import mne
import pandas as pd

from .bids_sidecars import (
    apply_sidecar_plan,
    plan_sidecar_migration,
    read_channels_metadata,
    read_eeg_metadata,
)


DERIVATIVE_NAME = "preprocessing"
DESCRIPTION_LABEL = "preproc"
RUN_PATTERN = re.compile(
    r"^(?P<base>"
    r"(?P<subject>sub-[0-9A-Za-z]+)_"
    r"(?P<session>ses-[0-9A-Za-z]+)_"
    r"task-(?P<task>[0-9A-Za-z]+)_"
    r"run-(?P<run>[0-9A-Za-z]+)"
    r")_desc-infomax-cleaned_raw\.fif$"
)
RAW_EVENTS_PATTERN = re.compile(
    r"^(?P<base>"
    r"(?P<subject>sub-[0-9A-Za-z]+)_"
    r"(?P<session>ses-[0-9A-Za-z]+)_"
    r"task-(?P<task>[0-9A-Za-z]+)_"
    r"run-(?P<run>[0-9A-Za-z]+)"
    r")_events\.tsv$"
)
EEG_EXTENSIONS = (".vhdr", ".edf", ".bdf", ".set")
CHANNEL_PLAN_COLUMNS = (
    "channel",
    "condition_decision",
    "interpolate_triggering_runs",
    "exclude_triggering_runs",
)
_PREPROCESSED_README = """\
# Spell-IT-OUT preprocessed EEG

Cleaned continuous EEG, one `desc-preproc` recording for each raw run. This is
the data used for the analyses in the paper.

## Processing

1. Per run: AF7 and AF8 dropped (often poor quality); EEG and EOG resampled
   to 512 Hz and band-pass filtered at 1–100 Hz (zero-phase FIR).
2. Per run: blink candidates (from FP1, FP2, FPz and EOG) and muscle
   candidates (from high-frequency activity) detected with MNE, plus PyPREP
   channel-quality evidence. Everything was reviewed by hand.
3. Per subject and task: channel decisions of the three runs merged, so every
   run gets the same interpolations. 50 Hz notch filter, then average reference.
4. Extended Infomax ICA fitted on the three runs concatenated, leaving out
   `BAD_*` spans, with as many components as the data rank allows. Components
   were labelled with ICLabel and decided with these rules, in order:
   reject muscle/eye/heart/line noise with p ≥ 0.80; review channel noise with
   p ≥ 0.70; keep brain with p ≥ 0.60; keep components explaining < 0.4 % of
   the variance; review everything else by hand. The removed components are
   listed in the `Description` of each `*_eeg.json`.
5. The cleaned signal was split back into the original runs.

Sampling rate, filters and reference of each file are in its `*_eeg.json`;
interpolated channels are marked in `*_channels.tsv`.

## Annotations

Besides the experimental events, `*_events.tsv` contains the reviewed
artifact annotations. They mark time spans; no samples were removed.

- `BLINK`: reviewed blinks. Not treated as bad data.
- `BAD_MUSCLE`, `BAD_AMPLITUDE`, `BAD_MOVEMENT`, `BAD_MANUAL`: spans to
  exclude from analyses.

## qc/

The complete review record: artifact candidates and decisions, channel
evidence and decisions, the channel plan used for ICA, component decisions and
ICA fit metadata. These tables use a project-specific format and are listed in
`.bidsignore`. See `qc/README`.
"""


@dataclass(frozen=True, slots=True)
class ExportRecord:
    """Inputs needed to publish one preprocessed run."""

    subject: str
    session: str
    task: str
    run: str
    base: str
    cleaned_raw: Path
    raw_eeg: Path
    raw_events: Path
    annotations_qc: Path
    channels_qc: Path
    components: Path
    channel_plan: Path
    ica_metadata: Path

    @property
    def task_base(self) -> str:
        return f"{self.subject}_{self.session}_task-{self.task}"


@dataclass(frozen=True, slots=True)
class RunInspection:
    """Validated metadata retained without keeping signal data in memory."""

    record: ExportRecord
    sfreq: float
    n_times: int
    highpass: float
    lowpass: float
    channel_names: tuple[str, ...]
    channel_types: tuple[str, ...]
    interpolated_channels: tuple[str, ...]
    excluded_channels: tuple[str, ...]
    rejected_components: tuple[int, ...]
    n_components: int
    line_noise_method: str
    line_frequency_hz: float
    final_events: pd.DataFrame


def _json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return value


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _read_tsv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", keep_default_na=False)


def _read_channel_plan(path: Path) -> pd.DataFrame:
    """Read a channel plan, accepting legacy zero-byte empty plans."""
    try:
        return _read_tsv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=CHANNEL_PLAN_COLUMNS)


def _label_filter(values: Iterable[str] | None, prefix: str) -> set[str] | None:
    if values is None:
        return None
    labels = set()
    for value in values:
        label = str(value)
        if label.startswith(prefix):
            label = label[len(prefix) :]
        if label.isdigit():
            label = f"{int(label):02d}"
        labels.add(label)
    return labels


def _find_raw_eeg(raw_bids_root: Path, base: str, eeg_dir: Path) -> Path:
    matches = [eeg_dir / f"{base}_eeg{extension}" for extension in EEG_EXTENSIONS]
    existing = [path for path in matches if path.is_file()]
    if len(existing) != 1:
        rendered = ", ".join(str(path) for path in existing) or "none"
        raise FileNotFoundError(
            f"Expected one raw BIDS EEG data file for {base}, found {rendered} "
            f"below {raw_bids_root}."
        )
    return existing[0]


def discover_export_records(
    raw_bids_root: Path,
    working_derivatives: Path,
    *,
    subjects: Iterable[str] | None = None,
    tasks: Iterable[str] | None = None,
    runs: Iterable[str] | None = None,
) -> list[ExportRecord]:
    """Discover complete final runs and their review/provenance inputs."""
    raw_bids_root = raw_bids_root.resolve()
    working_derivatives = working_derivatives.resolve()
    subject_filter = _label_filter(subjects, "sub-")
    task_filter = None if tasks is None else {str(value) for value in tasks}
    run_filter = _label_filter(runs, "run-")

    records: list[ExportRecord] = []
    candidates = sorted(
        working_derivatives.glob(
            "sub-*/ses-*/eeg/*_desc-infomax-cleaned_raw.fif"
        )
    )
    for cleaned_raw in candidates:
        match = RUN_PATTERN.match(cleaned_raw.name)
        if match is None:
            raise ValueError(f"Unexpected cleaned-run filename: {cleaned_raw}")
        values = match.groupdict()
        subject_label = values["subject"].removeprefix("sub-")
        run_label = values["run"]
        if subject_filter is not None and subject_label not in subject_filter:
            continue
        if task_filter is not None and values["task"] not in task_filter:
            continue
        if run_filter is not None and run_label not in run_filter:
            continue

        eeg_dir = raw_bids_root / values["subject"] / values["session"] / "eeg"
        working_eeg_dir = cleaned_raw.parent
        task_base = (
            f"{values['subject']}_{values['session']}_task-{values['task']}"
        )
        base = values["base"]
        record = ExportRecord(
            subject=values["subject"],
            session=values["session"],
            task=values["task"],
            run=run_label,
            base=base,
            cleaned_raw=cleaned_raw,
            raw_eeg=_find_raw_eeg(raw_bids_root, base, eeg_dir),
            raw_events=eeg_dir / f"{base}_events.tsv",
            annotations_qc=working_eeg_dir / f"{base}_desc-annotationsqc.tsv",
            channels_qc=working_eeg_dir / f"{base}_desc-channelqc.tsv",
            components=working_eeg_dir / f"{task_base}_desc-infomax_components.tsv",
            channel_plan=(
                working_eeg_dir / f"{task_base}_desc-infomax-input_channels.tsv"
            ),
            ica_metadata=working_eeg_dir / f"{task_base}_desc-infomax.json",
        )
        missing = [
            path
            for path in (
                record.raw_events,
                record.annotations_qc,
                record.channels_qc,
                record.components,
                record.channel_plan,
                record.ica_metadata,
            )
            if not path.is_file()
        ]
        if missing:
            raise FileNotFoundError(
                f"Missing export input(s) for {base}: "
                + ", ".join(str(path) for path in missing)
            )
        records.append(record)

    if not records:
        raise FileNotFoundError(
            "No desc-infomax-cleaned_raw.fif files match the requested selection "
            f"below {working_derivatives}."
        )

    selected_raw_bases = set()
    for events_path in raw_bids_root.glob("sub-*/ses-*/eeg/*_events.tsv"):
        match = RAW_EVENTS_PATTERN.match(events_path.name)
        if match is None:
            continue
        values = match.groupdict()
        subject_label = values["subject"].removeprefix("sub-")
        if subject_filter is not None and subject_label not in subject_filter:
            continue
        if task_filter is not None and values["task"] not in task_filter:
            continue
        if run_filter is not None and values["run"] not in run_filter:
            continue
        selected_raw_bases.add(values["base"])
    cleaned_bases = {record.base for record in records}
    missing_cleaned = sorted(selected_raw_bases.difference(cleaned_bases))
    unexpected_cleaned = sorted(cleaned_bases.difference(selected_raw_bases))
    if missing_cleaned or unexpected_cleaned:
        details = []
        if missing_cleaned:
            details.append("missing cleaned runs: " + ", ".join(missing_cleaned))
        if unexpected_cleaned:
            details.append(
                "cleaned runs without raw BIDS events: " + ", ".join(unexpected_cleaned)
            )
        raise RuntimeError("Raw/preprocessed run inventory mismatch; " + "; ".join(details))
    return records


def _require_columns(table: pd.DataFrame, columns: set[str], path: Path) -> None:
    missing = sorted(columns.difference(table.columns))
    if missing:
        raise ValueError(f"{path} is missing columns: {', '.join(missing)}")


def _validate_decisions(
    table: pd.DataFrame,
    column: str,
    allowed: set[str],
    path: Path,
) -> None:
    values = table[column].astype(str).str.strip().str.lower()
    invalid = sorted(set(values).difference(allowed))
    if invalid:
        raise ValueError(
            f"{path} contains invalid {column} values: "
            + ", ".join(repr(value) for value in invalid)
        )
    if "unreviewed" in set(values):
        count = int(values.eq("unreviewed").sum())
        raise RuntimeError(f"{path} still contains {count} unreviewed decision(s).")


def effective_preprocessed_events(
    experimental: pd.DataFrame,
    annotations_qc: pd.DataFrame,
    *,
    recording_duration: float | None = None,
) -> pd.DataFrame:
    """Combine experiment events with the artifact annotations used downstream."""
    _require_columns(experimental, {"onset", "duration", "trial_type"}, Path("events.tsv"))
    _require_columns(
        annotations_qc,
        {
            "onset",
            "duration",
            "annotation_source",
            "review_status",
            "reviewed_description",
        },
        Path("desc-annotationsqc.tsv"),
    )
    events = experimental.copy()
    statuses = annotations_qc["review_status"].astype(str).str.strip().str.lower()
    descriptions = annotations_qc["reviewed_description"].astype(str).str.strip()
    effective = annotations_qc.loc[
        statuses.eq("approved_bad")
        | (statuses.eq("keep") & descriptions.str.upper().eq("BLINK"))
    ].copy()

    rows: list[dict[str, object]] = []
    for row in effective.itertuples(index=False):
        onset = float(row.onset)
        duration = float(row.duration)
        if recording_duration is not None:
            clipped_onset = max(0.0, onset)
            clipped_stop = min(recording_duration, onset + duration)
            if clipped_stop <= clipped_onset:
                continue
            onset = clipped_onset
            duration = clipped_stop - clipped_onset
        values: dict[str, object] = {column: "n/a" for column in events.columns}
        values["onset"] = onset
        values["duration"] = duration
        values["trial_type"] = str(row.reviewed_description)
        if "event_source" in values:
            values["event_source"] = str(row.annotation_source)
        rows.append(values)
    if rows:
        events = pd.concat([events, pd.DataFrame(rows)], ignore_index=True)
    # The stimuli belong to the linked raw dataset. Keeping its relative
    # ``stim_file`` value here would make validators search inside the
    # derivative dataset and report a missing file.
    if "stim_file" in events:
        events["stim_file"] = "n/a"
    events["onset"] = pd.to_numeric(events["onset"], errors="raise")
    events["duration"] = pd.to_numeric(events["duration"], errors="raise")
    events["_stable_order"] = range(len(events))
    events = events.sort_values(["onset", "_stable_order"], kind="stable")
    return events.drop(columns="_stable_order").reset_index(drop=True)


def _read_raw(path: Path, *, preload: bool) -> mne.io.BaseRaw:
    readers = {
        ".vhdr": mne.io.read_raw_brainvision,
        ".edf": mne.io.read_raw_edf,
        ".bdf": mne.io.read_raw_bdf,
        ".set": mne.io.read_raw_eeglab,
        ".fif": mne.io.read_raw_fif,
    }
    try:
        reader = readers[path.suffix.lower()]
    except KeyError as error:
        raise ValueError(f"Unsupported EEG extension: {path}") from error
    return reader(path, preload=preload, verbose="error")


def _validate_event_bounds(events: pd.DataFrame, duration: float, path: Path) -> None:
    onsets = pd.to_numeric(events["onset"], errors="raise")
    durations = pd.to_numeric(events["duration"], errors="raise")
    invalid = events.index[
        onsets.lt(0) | durations.lt(0) | (onsets + durations).gt(duration + 1e-6)
    ].tolist()
    if invalid:
        shown = ", ".join(str(int(index) + 2) for index in invalid[:10])
        raise ValueError(f"Out-of-bounds event row(s) in {path}: {shown}")


def inspect_record(record: ExportRecord) -> RunInspection:
    """Validate one run without retaining its signal array."""
    experimental = _read_tsv(record.raw_events)
    annotations = _read_tsv(record.annotations_qc)
    channels = _read_tsv(record.channels_qc)
    components = _read_tsv(record.components)
    plan = _read_channel_plan(record.channel_plan)
    metadata = _json(record.ica_metadata)

    _require_columns(
        annotations,
        {"review_status", "reviewed_description", "annotation_source"},
        record.annotations_qc,
    )
    _validate_decisions(
        annotations,
        "review_status",
        {"keep", "approved_bad", "dismissed", "unreviewed"},
        record.annotations_qc,
    )
    _require_columns(channels, {"channel", "channel_type", "final_decision"}, record.channels_qc)
    eeg_channels = channels.loc[
        channels["channel_type"].astype(str).str.upper().eq("EEG")
    ]
    _validate_decisions(
        eeg_channels,
        "final_decision",
        {"retain", "interpolate", "exclude_from_ica", "unreviewed"},
        record.channels_qc,
    )
    _require_columns(components, {"component", "final_decision"}, record.components)
    _validate_decisions(
        components,
        "final_decision",
        {"keep", "reject", "unreviewed"},
        record.components,
    )
    _require_columns(
        plan,
        set(CHANNEL_PLAN_COLUMNS),
        record.channel_plan,
    )

    cleaned = _read_raw(record.cleaned_raw, preload=False)
    source = _read_raw(record.raw_eeg, preload=False)
    cleaned_duration = cleaned.n_times / float(cleaned.info["sfreq"])
    source_duration = source.n_times / float(source.info["sfreq"])
    tolerance = max(1 / float(cleaned.info["sfreq"]), 1 / float(source.info["sfreq"]))
    if abs(cleaned_duration - source_duration) > tolerance:
        raise ValueError(
            f"Duration mismatch for {record.base}: raw BIDS={source_duration:.9f}s, "
            f"cleaned={cleaned_duration:.9f}s."
        )

    final_events = effective_preprocessed_events(
        experimental,
        annotations,
        recording_duration=cleaned_duration,
    )
    _validate_event_bounds(final_events, cleaned_duration + tolerance, record.raw_events)

    source_runs = {str(value) for value in metadata.get("source_runs", [])}
    if f"run-{record.run}" not in source_runs:
        raise ValueError(
            f"{record.ica_metadata} does not list run-{record.run} in source_runs."
        )
    interpolated = tuple(
        str(value) for value in metadata.get("condition_interpolated_channels", [])
    )
    excluded = tuple(
        str(value) for value in metadata.get("condition_excluded_channels", [])
    )
    missing_interpolated = sorted(set(interpolated).difference(cleaned.ch_names))
    present_excluded = sorted(set(excluded).intersection(cleaned.ch_names))
    if missing_interpolated:
        raise ValueError(
            f"Interpolated channels missing from {record.cleaned_raw}: "
            + ", ".join(missing_interpolated)
        )
    if present_excluded:
        raise ValueError(
            f"Condition-excluded channels remain in {record.cleaned_raw}: "
            + ", ".join(present_excluded)
        )

    component_ids = pd.to_numeric(components["component"], errors="raise").astype(int)
    decisions = components["final_decision"].astype(str).str.strip().str.lower()
    rejected = tuple(sorted(component_ids.loc[decisions.eq("reject")].tolist()))
    line_noise = metadata.get("line_noise", {})
    if not isinstance(line_noise, dict):
        raise ValueError(f"Invalid line_noise object in {record.ica_metadata}.")

    inspection = RunInspection(
        record=record,
        sfreq=float(cleaned.info["sfreq"]),
        n_times=int(cleaned.n_times),
        highpass=float(cleaned.info["highpass"]),
        lowpass=float(cleaned.info["lowpass"]),
        channel_names=tuple(cleaned.ch_names),
        channel_types=tuple(cleaned.get_channel_types()),
        interpolated_channels=interpolated,
        excluded_channels=excluded,
        rejected_components=rejected,
        n_components=len(components),
        line_noise_method=str(line_noise.get("method", "unknown")),
        line_frequency_hz=float(line_noise.get("frequency_hz", 50.0)),
        final_events=final_events,
    )
    del cleaned, source
    return inspection


def inspect_records(records: Iterable[ExportRecord]) -> list[RunInspection]:
    """Validate all selected runs and condition-wide consistency."""
    inspections = [inspect_record(record) for record in records]
    groups: dict[tuple[str, str, str], list[RunInspection]] = {}
    for inspection in inspections:
        key = (
            inspection.record.subject,
            inspection.record.session,
            inspection.record.task,
        )
        groups.setdefault(key, []).append(inspection)
    for key, group in groups.items():
        reference = group[0]
        for current in group[1:]:
            compared = (
                current.channel_names,
                current.interpolated_channels,
                current.excluded_channels,
                current.rejected_components,
                current.n_components,
            )
            expected = (
                reference.channel_names,
                reference.interpolated_channels,
                reference.excluded_channels,
                reference.rejected_components,
                reference.n_components,
            )
            if compared != expected:
                raise ValueError(
                    "Condition-level ICA/channel metadata differs between runs for "
                    + "_".join(key)
                )
    return inspections


def _package_version(distribution: str, fallback: str = "unknown") -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return fallback


def _raw_link(output_root: Path, raw_bids_root: Path) -> str:
    try:
        return Path(os.path.relpath(raw_bids_root, output_root)).as_posix()
    except ValueError:
        # Windows cannot express a relative path between drive letters.
        return raw_bids_root.resolve().as_uri()


def write_derivative_scaffold(
    output_root: Path,
    raw_bids_root: Path,
    *,
    pipeline_version: str | None = None,
    code_url: str | None = None,
) -> None:
    """Write dataset-level BIDS derivative documentation."""
    source = _json(raw_bids_root / "dataset_description.json")
    generated = {
        "Name": DERIVATIVE_NAME,
        "Version": pipeline_version or _package_version("spellitout", "0.1.1"),
        "Description": (
            "Spell-IT-OUT EEG preprocessing with reviewed artifact and channel "
            "quality control and condition-level Extended Infomax ICA."
        ),
    }
    if code_url:
        generated["CodeURL"] = code_url
    description = {
        "Name": "Spell-IT-OUT preprocessed EEG",
        "BIDSVersion": source.get("BIDSVersion", "1.11.1"),
        "DatasetType": "derivative",
        "DatasetLinks": {"raw": _raw_link(output_root, raw_bids_root)},
        "GeneratedBy": [
            generated,
            {
                "Name": "MNE-Python",
                "Version": mne.__version__,
                "Description": "Signal processing, ICA, and BrainVision export.",
            },
            {
                "Name": "PyPREP",
                "Version": _package_version("pyprep"),
                "Description": "Automatic channel-quality evidence.",
            },
            {
                "Name": "MNE-ICALabel",
                "Version": _package_version("mne-icalabel"),
                "Description": "ICA component classification used for automatic decisions.",
            },
            {
                "Name": "Manual",
                "Description": (
                    "Human review of artifact spans, channel decisions, and ICA "
                    "component decisions."
                ),
            },
        ],
    }
    for key in ("License", "Authors", "Acknowledgements", "HowToAcknowledge"):
        if key in source:
            description[key] = source[key]
    source_dataset = {"URL": _raw_link(output_root, raw_bids_root)}
    if source.get("DatasetDOI"):
        source_dataset["DOI"] = source["DatasetDOI"]
    if source.get("DatasetVersion"):
        source_dataset["Version"] = source["DatasetVersion"]
    description["SourceDatasets"] = [source_dataset]
    _write_json(output_root / "dataset_description.json", description)

    (output_root / "README").write_text(_PREPROCESSED_README, encoding="utf-8")
    (output_root / "CHANGES").write_text(
        f"1.0.0 {date.today().isoformat()}\n"
        "- Initial publication export of reviewed preprocessed EEG.\n",
        encoding="utf-8",
    )
    # Validator 3.0.2 does not yet accept the derivative descriptions.tsv
    # schema. The desc-preproc meaning is therefore documented in README and
    # the complete local EEG sidecars instead.
    # Validator 3.x tests directory nodes without a trailing slash and their
    # contents separately, so both patterns are required.
    (output_root / ".bidsignore").write_text(
        "qc\nqc/**\n",
        encoding="utf-8",
    )


def _event_sidecar(
    raw_bids_root: Path,
    task: str,
    inspections: Iterable[RunInspection],
) -> dict:
    source_path = raw_bids_root / f"task-{task}_events.json"
    sidecar = _json(source_path)
    sidecar["Description"] = (
        "Experimental events copied from the linked raw dataset, plus retained "
        "blink annotations and reviewer-approved BAD_* artifact spans. stim_file "
        "is n/a here because presented stimuli belong to the linked raw dataset."
    )
    trial_type = sidecar.setdefault("trial_type", {})
    levels = trial_type.setdefault("Levels", {})
    descriptions = sorted(
        {
            str(value)
            for inspection in inspections
            for value in inspection.final_events["trial_type"]
            if str(value) == "BLINK" or str(value).startswith("BAD_")
        }
    )
    for description in descriptions:
        if description == "BLINK":
            levels[description] = "Reviewed blink annotation retained for analysis."
        else:
            levels[description] = (
                "Reviewer-approved artifact span; omitted by analyses that reject "
                "MNE BAD_* annotations."
            )
    event_source = sidecar.setdefault("event_source", {})
    source_levels = event_source.setdefault("Levels", {})
    source_levels.update(
        {
            "auto": "Automatic artifact detector followed by human review.",
            "manual_qc_review": "Annotation added manually during QC review.",
        }
    )
    return sidecar


def _software_filters(group: list[RunInspection]) -> dict:
    first = group[0]
    methods = sorted({inspection.line_noise_method for inspection in group})
    frequencies = sorted({inspection.line_frequency_hz for inspection in group})
    return {
        "BandpassFilter": {
            "HighpassCutoffHz": first.highpass,
            "LowpassCutoffHz": first.lowpass,
        },
        "LineNoiseRemoval": {
            "Method": methods[0] if len(methods) == 1 else methods,
            "FrequenciesHz": frequencies,
        },
    }


def write_inherited_sidecars(
    output_root: Path,
    raw_bids_root: Path,
    inspections: Iterable[RunInspection],
) -> None:
    """Write dataset-level event schemas and custom-channel column descriptions.

    Complete EEG metadata are written alongside each recording, then shared
    within subject/session/task only when all values are identical.
    """
    by_task: dict[str, list[RunInspection]] = {}
    for inspection in inspections:
        by_task.setdefault(inspection.record.task, []).append(inspection)
    for task, group in sorted(by_task.items()):
        _write_json(
            output_root / f"task-{task}_desc-{DESCRIPTION_LABEL}_events.json",
            _event_sidecar(raw_bids_root, task, group),
        )
        _write_json(
            output_root / f"task-{task}_desc-{DESCRIPTION_LABEL}_channels.json",
            {
                "Description": (
                    "Channel metadata for the final continuous preprocessed EEG."
                ),
                "interpolated": {
                    "Description": (
                        "Whether the published channel signal was interpolated "
                        "condition-wide after reviewed run-level channel QC."
                    ),
                    "Levels": {
                        "true": "Channel was interpolated.",
                        "false": "Channel was not interpolated.",
                    },
                }
            },
        )


def _channel_counts(types: Iterable[str]) -> dict[str, int]:
    values = [str(value).lower() for value in types]
    mapping = {
        "EEGChannelCount": "eeg",
        "EOGChannelCount": "eog",
        "ECGChannelCount": "ecg",
        "EMGChannelCount": "emg",
        "MiscChannelCount": "misc",
        "TriggerChannelCount": "stim",
    }
    return {field: values.count(channel_type) for field, channel_type in mapping.items()}


def _run_sidecar(inspection: RunInspection, raw_bids_root: Path) -> dict:
    source = inspection.record.raw_eeg.relative_to(raw_bids_root).as_posix()
    rejected = list(inspection.rejected_components)
    return {
        **read_eeg_metadata(raw_bids_root, inspection.record.raw_eeg),
        "SamplingFrequency": inspection.sfreq,
        "EEGReference": "average",
        "SoftwareFilters": _software_filters([inspection]),
        "PowerLineFrequency": inspection.line_frequency_hz,
        "RecordingType": "continuous",
        "Sources": [f"bids:raw:{source}"],
        "Description": (
            "Continuous preprocessed EEG after reviewed channel harmonization and "
            "condition-level Extended Infomax ICA. Rejected components: "
            + (", ".join(str(value) for value in rejected) if rejected else "none")
            + "."
        ),
        **_channel_counts(inspection.channel_types),
    }


def _status_description(channel: str, inspection: RunInspection, plan: pd.DataFrame) -> str:
    if channel not in inspection.interpolated_channels:
        return "n/a"
    rows = plan.loc[plan["channel"].astype(str).eq(channel)]
    runs: list[str] = []
    if not rows.empty:
        value = str(rows.iloc[0]["interpolate_triggering_runs"])
        runs = [item for item in value.split(";") if item]
    suffix = f" Triggered by reviewed decisions in {', '.join(runs)}." if runs else ""
    return "Interpolated condition-wide after reviewed channel QC." + suffix


def final_channels_table(
    inspection: RunInspection,
    raw_channels: pd.DataFrame,
    plan: pd.DataFrame,
) -> pd.DataFrame:
    """Create a standard channels.tsv for the published signal."""
    _require_columns(raw_channels, {"name", "type", "units"}, Path("channels.tsv"))
    raw_channels = raw_channels.set_index("name", drop=False)
    rows = []
    for channel, channel_type in zip(
        inspection.channel_names, inspection.channel_types
    ):
        if channel in raw_channels.index:
            row = raw_channels.loc[channel].to_dict()
        else:
            row = {
                "name": channel,
                "type": channel_type.upper(),
                "units": "V" if channel_type.lower() in {"eeg", "eog", "ecg", "emg"} else "n/a",
            }
        row["type"] = channel_type.upper()
        row["low_cutoff"] = inspection.highpass
        row["high_cutoff"] = inspection.lowpass
        row["sampling_frequency"] = inspection.sfreq
        row["status"] = "good"
        row["status_description"] = _status_description(channel, inspection, plan)
        row["interpolated"] = (
            "true" if channel in inspection.interpolated_channels else "false"
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _copy_coordinates(record: ExportRecord, raw_bids_root: Path, output_root: Path) -> None:
    source_dir = raw_bids_root / record.subject / record.session / "eeg"
    target_dir = output_root / record.subject / record.session / "eeg"
    coordinate_base = f"{record.subject}_{record.session}"
    electrodes_source = source_dir / f"{coordinate_base}_electrodes.tsv"
    coordsystem_source = source_dir / f"{coordinate_base}_coordsystem.json"
    for source in (electrodes_source, coordsystem_source):
        if not source.is_file():
            raise FileNotFoundError(f"Missing coordinate metadata: {source}")
    shutil.copy2(electrodes_source, target_dir / electrodes_source.name)
    coordsystem = _json(coordsystem_source)
    coordsystem["Description"] = (
        "Electrode coordinate system propagated unchanged from the linked raw dataset."
    )
    coordsystem["Sources"] = [
        "bids:raw:"
        + coordsystem_source.relative_to(raw_bids_root).as_posix()
    ]
    _write_json(target_dir / coordsystem_source.name, coordsystem)
    _write_json(
        target_dir / f"{coordinate_base}_electrodes.json",
        {
            "Description": (
                "Electrode coordinates propagated unchanged from the linked raw dataset."
            ),
            "Sources": [
                "bids:raw:"
                + electrodes_source.relative_to(raw_bids_root).as_posix()
            ],
        },
    )


def _set_export_annotations(raw: mne.io.BaseRaw, events: pd.DataFrame) -> None:
    raw.set_meas_date(None)
    raw.info["subject_info"] = None
    raw.info["experimenter"] = None
    raw.set_annotations(
        mne.Annotations(
            onset=pd.to_numeric(events["onset"], errors="raise").to_numpy(float),
            duration=pd.to_numeric(events["duration"], errors="raise").to_numpy(float),
            description=events["trial_type"].astype(str).to_numpy(),
            orig_time=None,
        )
    )


def _write_qc_documentation(qc_root: Path) -> None:
    qc_root.mkdir(parents=True, exist_ok=True)
    (qc_root / "README").write_text(
        "Spell-IT-OUT preprocessing audit trail\n"
        "======================================\n\n"
        "These project-specific tables preserve the complete human-review record.\n"
        "annotationsqc: automatic candidates and manual artifact annotations with "
        "review decisions.\n"
        "channelqc: PyPREP evidence and final manual channel decisions.\n"
        "channelplan: condition-level interpolation/exclusion plan used for ICA.\n"
        "icadecisions: ICLabel evidence and final component decisions.\n"
        "icaprovenance: exact ICA preparation, rank, convergence, and fit metadata.\n\n"
        "Paths referring to minproc or ICA-input checkpoints identify working "
        "pipeline artifacts that are reproducible but not included in this release.\n",
        encoding="utf-8",
    )
    schemas = {
        "annotationsqc.json": {
            "Description": "Complete artifact-annotation review table.",
            "DecisionColumn": "review_status",
        },
        "channelqc.json": {
            "Description": "Automatic channel evidence and human review table.",
            "DecisionColumn": "final_decision",
        },
        "channelplan.json": {
            "Description": "Condition-level channel harmonization used for ICA.",
        },
        "icadecisions.json": {
            "Description": "ICA component evidence and final review decisions.",
            "DecisionColumn": "final_decision",
        },
        "icaprovenance.json": {
            "Description": "Exact Extended Infomax configuration and fit metadata."
        },
    }
    for name, schema in schemas.items():
        _write_json(qc_root / name, schema)


def _copy_qc(record: ExportRecord, output_root: Path) -> None:
    target = output_root / "qc" / record.subject / record.session / "eeg"
    target.mkdir(parents=True, exist_ok=True)
    mapping = {
        record.annotations_qc: target / f"{record.base}_desc-annotationsqc.tsv",
        record.channels_qc: target / f"{record.base}_desc-channelqc.tsv",
        record.components: target / f"{record.task_base}_desc-icadecisions.tsv",
        record.channel_plan: target / f"{record.task_base}_desc-channelplan.tsv",
        record.ica_metadata: target / f"{record.task_base}_desc-icaprovenance.json",
    }
    for source, destination in mapping.items():
        shutil.copy2(source, destination)


_VHDR_CHANNEL = re.compile(r"^Ch(\d+)=([^,]*),[^,]*,([^,]*),(.*)$", re.MULTILINE)


def write_vhdr_filter_settings(
    vhdr: Path, highpass: float, lowpass: float, notch_hz: float | None,
) -> None:
    """Record the software filters in the BrainVision header.

    pybv writes no filter information, so readers such as MNE would report the
    data as unfiltered. MNE reads highpass/lowpass from the "Channels" table of
    the [Comment] section (BrainVision Recorder layout), which is written here.
    Re-running replaces a previously written table.
    """
    text = vhdr.read_text(encoding="utf-8")
    head, marker, comment = text.partition("[Comment]")
    if not marker:
        raise ValueError(f"No [Comment] section in {vhdr}")
    comment = comment.split("\nChannels\n", 1)[0].rstrip("\n")
    notch = "Off" if notch_hz is None else f"{notch_hz:g}"
    rows = [
        "", "Channels", "--------",
        "#  Name  Phys. Chn.  Resolution / Unit  Low Cutoff [Hz]  High Cutoff [Hz]  Notch [Hz]",
    ] + [
        f"{number}  {name}  {number}  {resolution} {unit}  {highpass:g}  {lowpass:g}  {notch}"
        for number, name, resolution, unit in _VHDR_CHANNEL.findall(head)
    ]
    vhdr.write_text(head + marker + comment + "\n" + "\n".join(rows) + "\n", encoding="utf-8")


def export_run(
    inspection: RunInspection,
    raw_bids_root: Path,
    output_root: Path,
    *,
    overwrite: bool,
) -> None:
    """Export one validated FIF as BrainVision plus BIDS sidecars."""
    record = inspection.record
    output_dir = output_root / record.subject / record.session / "eeg"
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{record.base}_desc-{DESCRIPTION_LABEL}"
    vhdr = output_dir / f"{stem}_eeg.vhdr"

    raw = _read_raw(record.cleaned_raw, preload=True)
    _set_export_annotations(raw, inspection.final_events)
    mne.export.export_raw(
        vhdr,
        raw,
        fmt="brainvision",
        overwrite=overwrite,
        verbose="error",
    )
    del raw
    write_vhdr_filter_settings(
        vhdr, inspection.highpass, inspection.lowpass, inspection.line_frequency_hz,
    )

    inspection.final_events.to_csv(
        output_dir / f"{stem}_events.tsv",
        sep="\t",
        index=False,
        na_rep="n/a",
        lineterminator="\n",
    )
    raw_channels = read_channels_metadata(raw_bids_root, record.raw_eeg)
    plan = _read_channel_plan(record.channel_plan)
    final_channels_table(inspection, raw_channels, plan).to_csv(
        output_dir / f"{stem}_channels.tsv",
        sep="\t",
        index=False,
        na_rep="n/a",
        lineterminator="\n",
    )
    _write_json(output_dir / f"{stem}_eeg.json", _run_sidecar(inspection, raw_bids_root))
    _copy_coordinates(record, raw_bids_root, output_root)
    _copy_qc(record, output_root)


def export_preprocessed_dataset(
    inspections: Iterable[RunInspection],
    raw_bids_root: Path,
    output_root: Path,
    *,
    overwrite: bool = False,
    pipeline_version: str | None = None,
    code_url: str | None = None,
) -> None:
    """Write the complete publication derivative after successful validation."""
    inspections = list(inspections)
    raw_bids_root = raw_bids_root.resolve()
    output_root = output_root.resolve()
    if output_root == raw_bids_root:
        raise ValueError("The derivative output cannot be the raw BIDS root itself.")
    existing = [path for path in output_root.rglob("*") if path.is_file()] if output_root.exists() else []
    if existing and not overwrite:
        raise FileExistsError(
            f"Derivative output already contains files: {output_root}. "
            "Pass overwrite=True only after checking the existing export."
        )

    validate_source_sidecars(inspections, raw_bids_root)
    output_root.mkdir(parents=True, exist_ok=True)
    write_derivative_scaffold(
        output_root,
        raw_bids_root,
        pipeline_version=pipeline_version,
        code_url=code_url,
    )
    write_inherited_sidecars(output_root, raw_bids_root, inspections)
    _write_qc_documentation(output_root / "qc")
    for inspection in inspections:
        export_run(
            inspection,
            raw_bids_root,
            output_root,
            overwrite=overwrite,
        )
    # Preserve run-specific provenance and metadata; compact only truly equal
    # sidecars. Also absorbs obsolete inherited EEG JSON from earlier exports.
    apply_sidecar_plan(plan_sidecar_migration(output_root))


def validate_source_sidecars(inspections: Iterable[RunInspection], raw_bids_root: Path) -> None:
    """Check both old and migrated raw metadata layouts before exporting data."""
    for inspection in inspections:
        _run_sidecar(inspection, raw_bids_root)
        final_channels_table(
            inspection,
            read_channels_metadata(raw_bids_root, inspection.record.raw_eeg),
            _read_channel_plan(inspection.record.channel_plan),
        )
