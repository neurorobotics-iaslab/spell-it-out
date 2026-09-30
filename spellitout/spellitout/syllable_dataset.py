"""Build efficient, provenance-rich syllable datasets from Spell-IT-OUT BIDS.

The generated HDF5 files are BIDS-derived analysis artifacts, not standard
BIDS EEG files.  Every sample retains the BIDS entities and event metadata
needed to trace it back to its continuous source recording.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import h5py
import mne
import numpy as np
import pandas as pd
from scipy.signal import resample_poly
import soundfile as sf

from .dataset import RunData, SpellItOutDataset


SCHEMA_VERSION = "1.0"
REQUIRED_CONFIG_KEYS = {"source", "datasets", "eeg", "preprocessing", "audio", "hdf5"}
REQUIRED_EVENT_COLUMNS = {
    "onset",
    "duration",
    "trial_type",
    "trial_index",
    "token",
    "token_index",
}
@dataclass(frozen=True, slots=True)
class DatasetSpec:
    """One logical sample collection described by the build configuration."""

    name: str
    description_label: str
    task: str
    event_type: str
    include_audio: bool
    pair_reference_event_type: str | None = None


@dataclass(frozen=True, slots=True)
class BuildPaths:
    """Resolved input and output roots."""

    bids_root: Path
    source_root: Path
    preprocessed_root: Path
    output_root: Path


def load_config(path: str | Path) -> dict[str, Any]:
    """Load and minimally validate a syllable-dataset JSON configuration."""
    path = Path(path)
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON configuration {path}: {error}") from error
    if not isinstance(config, dict):
        raise ValueError("The build configuration must contain a JSON object.")
    missing = REQUIRED_CONFIG_KEYS - config.keys()
    if missing:
        raise ValueError(f"Configuration is missing: {', '.join(sorted(missing))}")
    if config["source"] not in {"raw", "preprocessed"}:
        raise ValueError("'source' must be 'raw' or 'preprocessed'.")
    if not isinstance(config["datasets"], list) or not config["datasets"]:
        raise ValueError("'datasets' must be a non-empty list.")
    tmin = float(config["eeg"]["tmin"])
    tmax = float(config["eeg"]["tmax"])
    if tmin >= tmax:
        raise ValueError("eeg.tmin must be smaller than eeg.tmax.")
    for item in config["datasets"]:
        for key in ("name", "description_label", "task", "event_type", "include_audio"):
            if key not in item:
                raise ValueError(f"Dataset entry is missing '{key}'.")
        if item["task"] not in {"silent", "overt"}:
            raise ValueError(f"Unsupported task in dataset {item['name']!r}.")
        if item["include_audio"] and item["task"] != "overt":
            raise ValueError("Audio samples are only supported for the overt task.")
    return config


def resolve_config(
    config: dict[str, Any],
    *,
    source: str | None = None,
    eeg_tmin: float | None = None,
    eeg_tmax: float | None = None,
) -> dict[str, Any]:
    """Return a deep copy with explicit CLI overrides applied."""
    resolved = json.loads(json.dumps(config))
    if source is not None:
        if source not in {"raw", "preprocessed"}:
            raise ValueError("Source override must be 'raw' or 'preprocessed'.")
        resolved["source"] = source
    if eeg_tmin is not None:
        resolved["eeg"]["tmin"] = float(eeg_tmin)
    if eeg_tmax is not None:
        resolved["eeg"]["tmax"] = float(eeg_tmax)
    if float(resolved["eeg"]["tmin"]) >= float(resolved["eeg"]["tmax"]):
        raise ValueError("The resolved EEG window has tmin >= tmax.")
    return resolved


def config_digest(config: dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def dataset_specs(config: dict[str, Any]) -> list[DatasetSpec]:
    specs = []
    names: set[str] = set()
    labels: set[str] = set()
    for item in config["datasets"]:
        spec = DatasetSpec(
            name=str(item["name"]),
            description_label=str(item["description_label"]),
            task=str(item["task"]),
            event_type=str(item["event_type"]),
            include_audio=bool(item["include_audio"]),
            pair_reference_event_type=item.get("pair_reference_event_type"),
        )
        if spec.name in names or spec.description_label in labels:
            raise ValueError("Dataset names and description labels must be unique.")
        names.add(spec.name)
        labels.add(spec.description_label)
        specs.append(spec)
    return specs


def read_phonetic_categories(path: str | Path) -> tuple[list[str], dict[str, np.ndarray]]:
    """Read the syllable-to-phonetic-feature lookup table."""
    path = Path(path)
    table = pd.read_csv(path, sep="\t", keep_default_na=False)
    if "phonesyll" not in table or len(table.columns) < 2:
        raise ValueError(f"Invalid phonetic category table: {path}")
    if table["phonesyll"].duplicated().any():
        duplicates = sorted(table.loc[table["phonesyll"].duplicated(), "phonesyll"].unique())
        raise ValueError(f"Duplicate syllables in {path}: {duplicates}")
    categories = list(table.columns[1:])
    try:
        values = table[categories].astype(np.uint8).to_numpy()
    except ValueError as error:
        raise ValueError(f"Phonetic categories must be binary integers in {path}.") from error
    if not np.isin(values, (0, 1)).all():
        raise ValueError(f"Phonetic categories must contain only zero and one: {path}")
    lookup = {str(token): row for token, row in zip(table["phonesyll"], values)}
    return categories, lookup


def _index(value: Any, field: str, context: str) -> int:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid {field} {value!r} in {context}.") from error
    if not number.is_integer():
        raise ValueError(f"Non-integral {field} {value!r} in {context}.")
    return int(number)


def _event_key(row: pd.Series, context: str) -> tuple[int, int]:
    return (
        _index(row["trial_index"], "trial_index", context),
        _index(row["token_index"], "token_index", context),
    )


def validate_event_table(events: pd.DataFrame, context: str) -> None:
    missing = REQUIRED_EVENT_COLUMNS - set(events.columns)
    if missing:
        raise ValueError(f"{context} is missing event columns: {', '.join(sorted(missing))}")


def bad_intervals(events: pd.DataFrame) -> np.ndarray:
    """Return finite, positive-duration BAD_* intervals as [start, stop]."""
    descriptions = events["trial_type"].astype(str)
    bad = events.loc[descriptions.str.startswith("BAD_")]
    if bad.empty:
        return np.empty((0, 2), dtype=float)
    onset = pd.to_numeric(bad["onset"], errors="coerce").to_numpy(float)
    duration = pd.to_numeric(bad["duration"], errors="coerce").to_numpy(float)
    valid = np.isfinite(onset) & np.isfinite(duration) & (duration > 0)
    return np.column_stack((onset[valid], onset[valid] + duration[valid]))


def overlaps_bad(start: float, stop: float, intervals: np.ndarray) -> bool:
    """Whether a half-open time interval intersects any BAD_* interval."""
    if intervals.size == 0:
        return False
    return bool(np.any((start < intervals[:, 1]) & (stop > intervals[:, 0])))


def _source_audio_path(bids_root: Path, run: RunData) -> Path:
    session = f"ses-{run.session}" if run.session else None
    folder = bids_root / "sourcedata" / run.subject_id
    if session:
        folder /= session
    folder /= "audio"
    base = f"{run.subject_id}"
    if session:
        base += f"_{session}"
    base += f"_task-{run.task}_run-{run.run}"
    path = folder / f"{base}_audio.wav"
    if not path.is_file():
        raise FileNotFoundError(f"Audio file not found: {path}")
    return path


def _bad_events_for_run(
    run: RunData,
    *,
    mode: str,
    preprocessed_dataset: SpellItOutDataset,
) -> pd.DataFrame:
    if mode == "none":
        return run.events.iloc[0:0]
    if mode == "source":
        return run.events
    if mode == "preprocessed":
        return preprocessed_dataset.get_run(
            subject=run.subject,
            session=run.session or "01",
            task=run.task,
            run=run.run or "01",
            preload=False,
        ).events
    raise ValueError("eeg.bad_annotation_source must be source, preprocessed, or none.")


def _pick_and_preprocess(raw: mne.io.BaseRaw, config: dict[str, Any]) -> mne.io.BaseRaw:
    """Copy, select channels, and apply the configured optional preprocessing."""
    raw = raw.copy().load_data()
    eeg = config["eeg"]
    picks = eeg.get("picks", "eeg")
    if picks == "eeg":
        indices = mne.pick_types(raw.info, eeg=True, exclude=[])
        if len(indices) == 0:
            raise ValueError("Source recording has no EEG channels.")
        raw.pick(indices)
    elif isinstance(picks, list) and picks:
        missing = sorted(set(picks) - set(raw.ch_names))
        if missing:
            raise ValueError(f"Requested EEG channels are missing: {', '.join(missing)}")
        raw.pick(picks)
    else:
        raise ValueError("eeg.picks must be 'eeg' or a non-empty channel-name list.")

    prep = config["preprocessing"]
    l_freq = prep.get("l_freq")
    h_freq = prep.get("h_freq")
    if l_freq is not None or h_freq is not None:
        raw.filter(l_freq=l_freq, h_freq=h_freq, picks="all", verbose="error")
    notch = prep.get("notch_freqs", [])
    if notch:
        raw.notch_filter(np.asarray(notch, dtype=float), picks="all", verbose="error")
    target_sfreq = prep.get("resample_sfreq")
    if target_sfreq is not None and not np.isclose(float(target_sfreq), raw.info["sfreq"]):
        raw.resample(float(target_sfreq), verbose="error")
    reference = prep.get("reference")
    if reference == "average":
        raw.set_eeg_reference("average", projection=False, verbose="error")
    elif isinstance(reference, list) and reference:
        raw.set_eeg_reference(reference, projection=False, verbose="error")
    elif reference is not None:
        raise ValueError("preprocessing.reference must be null, 'average', or a channel list.")
    return raw


def _epoch_offsets(sfreq: float, tmin: float, tmax: float) -> tuple[np.ndarray, np.ndarray]:
    first = int(round(tmin * sfreq))
    last = int(round(tmax * sfreq))
    offsets = np.arange(first, last + 1, dtype=np.int64)
    return offsets, offsets.astype(float) / sfreq


def _apply_epoch_operations(data: np.ndarray, times: np.ndarray, eeg: dict[str, Any]) -> np.ndarray:
    detrend = eeg.get("detrend")
    if detrend is not None:
        if detrend not in (0, 1):
            raise ValueError("eeg.detrend must be null, 0, or 1.")
        from scipy.signal import detrend as scipy_detrend

        data = scipy_detrend(data, axis=-1, type="constant" if detrend == 0 else "linear")
    baseline = eeg.get("baseline")
    if baseline is not None:
        if not isinstance(baseline, list) or len(baseline) != 2:
            raise ValueError("eeg.baseline must be null or [start, stop].")
        lo = times[0] if baseline[0] is None else float(baseline[0])
        hi = times[-1] if baseline[1] is None else float(baseline[1])
        mask = (times >= lo) & (times <= hi)
        if not mask.any():
            raise ValueError("The requested baseline contains no EEG samples.")
        data = data - data[..., mask].mean(axis=-1, keepdims=True)
    return data


def _audio_segment(
    audio: np.ndarray,
    native_sfreq: int,
    onset: float,
    duration: float,
    config: dict[str, Any],
) -> tuple[np.ndarray, int, int, int]:
    start = int(round(onset * native_sfreq))
    stop = int(round((onset + duration) * native_sfreq))
    if start < 0 or stop > len(audio) or stop <= start:
        raise ValueError(
            f"Audio span [{onset:.6f}, {onset + duration:.6f}) is outside the recording."
        )
    segment = np.asarray(audio[start:stop])
    policy = config["audio"].get("channel_policy", "mono")
    if segment.ndim == 2:
        if policy == "mono":
            segment = segment.mean(axis=1)
        elif policy == "first":
            segment = segment[:, 0]
        else:
            raise ValueError("audio.channel_policy must be 'mono' or 'first'.")
    target = config["audio"].get("sampling_frequency", "native")
    output_sfreq = int(native_sfreq)
    if target != "native":
        output_sfreq = int(target)
        if output_sfreq <= 0:
            raise ValueError("audio.sampling_frequency must be 'native' or positive.")
        if output_sfreq != native_sfreq:
            ratio = Fraction(output_sfreq, native_sfreq).limit_denominator()
            segment = resample_poly(segment, ratio.numerator, ratio.denominator)
    return segment.astype(np.float32, copy=False), output_sfreq, start, stop


class _ShardWriter:
    """Append homogeneous samples to one subject/dataset HDF5 shard."""

    def __init__(
        self,
        path: Path,
        *,
        spec: DatasetSpec,
        subject: str,
        session: str,
        channel_names: list[str],
        times: np.ndarray,
        sfreq: float,
        category_names: list[str],
        config: dict[str, Any],
        digest: str,
    ):
        self.path = path
        self.temp_path = path.with_suffix(path.suffix + ".tmp")
        path.parent.mkdir(parents=True, exist_ok=True)
        if self.temp_path.exists():
            self.temp_path.unlink()
        self.file = h5py.File(self.temp_path, "w")
        self.file.attrs.update(
            schema_name="Spell-IT-OUT syllable dataset",
            schema_version=SCHEMA_VERSION,
            format_status="BIDS-derived non-standard HDF5",
            dataset_name=spec.name,
            subject=f"sub-{subject}",
            session=f"ses-{session}",
            task=spec.task,
            event_type=spec.event_type,
            source_type=config["source"],
            build_config_sha256=digest,
            build_config_json=json.dumps(config, sort_keys=True),
        )
        strings = h5py.string_dtype("utf-8")
        eeg = self.file.create_group("eeg")
        eeg.attrs["units"] = config["eeg"].get("units", "V")
        eeg.attrs["sampling_frequency"] = sfreq
        eeg.create_dataset("times", data=times)
        eeg.create_dataset("channel_names", data=np.asarray(channel_names, dtype=object), dtype=strings)
        h5 = config["hdf5"]
        compression = h5.get("compression", "gzip")
        compression_opts = h5.get("compression_level", 4) if compression == "gzip" else None
        self.eeg = eeg.create_dataset(
            "data",
            shape=(0, len(channel_names), len(times)),
            maxshape=(None, len(channel_names), len(times)),
            chunks=(1, len(channel_names), len(times)),
            dtype=np.dtype(config["eeg"].get("dtype", "float32")),
            compression=compression,
            compression_opts=compression_opts,
            shuffle=bool(h5.get("shuffle", True)),
        )
        samples = self.file.create_group("samples")
        self.sample_columns: dict[str, h5py.Dataset] = {}
        string_columns = ("id", "run", "syllable", "stimulus_id", "source_eeg", "source_events")
        int_columns = ("trial_index", "token_index")
        float_columns = ("event_onset", "event_duration")
        for name in string_columns:
            self.sample_columns[name] = samples.create_dataset(name, (0,), maxshape=(None,), dtype=strings)
        for name in int_columns:
            self.sample_columns[name] = samples.create_dataset(name, (0,), maxshape=(None,), dtype=np.int32)
        for name in float_columns:
            self.sample_columns[name] = samples.create_dataset(name, (0,), maxshape=(None,), dtype=np.float64)
        labels = self.file.create_group("labels")
        labels.create_dataset("phonetic_category_names", data=np.asarray(category_names, dtype=object), dtype=strings)
        self.categories = labels.create_dataset(
            "phonetic_categories",
            shape=(0, len(category_names)),
            maxshape=(None, len(category_names)),
            dtype=np.uint8,
        )
        self.audio_data = self.audio_offsets = self.audio_sfreq = None
        self.audio_start = self.audio_stop = None
        if spec.include_audio:
            audio = self.file.create_group("audio")
            audio.attrs["units"] = "relative amplitude"
            self.audio_data = audio.create_dataset(
                "data", shape=(0,), maxshape=(None,), dtype=np.float32,
                compression=compression, compression_opts=compression_opts,
                shuffle=bool(h5.get("shuffle", True)),
            )
            self.audio_offsets = audio.create_dataset("offsets", data=np.asarray([0], dtype=np.int64), maxshape=(None,))
            self.audio_sfreq = audio.create_dataset("sampling_frequency", (0,), maxshape=(None,), dtype=np.int32)
            self.audio_start = audio.create_dataset("source_start_sample", (0,), maxshape=(None,), dtype=np.int64)
            self.audio_stop = audio.create_dataset("source_stop_sample", (0,), maxshape=(None,), dtype=np.int64)

    @property
    def count(self) -> int:
        return int(self.eeg.shape[0])

    def append(
        self,
        eeg_data: np.ndarray,
        records: list[dict[str, Any]],
        category_data: np.ndarray,
        audio_segments: list[tuple[np.ndarray, int, int, int]] | None,
    ) -> None:
        n = len(records)
        if n == 0:
            return
        old = self.count
        new = old + n
        self.eeg.resize(new, axis=0)
        self.eeg[old:new] = eeg_data
        self.categories.resize(new, axis=0)
        self.categories[old:new] = category_data
        for name, dataset in self.sample_columns.items():
            dataset.resize(new, axis=0)
            dataset[old:new] = [record[name] for record in records]
        if audio_segments is not None:
            assert self.audio_data is not None and self.audio_offsets is not None
            assert self.audio_sfreq is not None and self.audio_start is not None and self.audio_stop is not None
            lengths = np.asarray([len(segment[0]) for segment in audio_segments], dtype=np.int64)
            flat = np.concatenate([segment[0] for segment in audio_segments]) if lengths.sum() else np.empty(0, np.float32)
            audio_old = int(self.audio_data.shape[0])
            self.audio_data.resize(audio_old + len(flat), axis=0)
            self.audio_data[audio_old:] = flat
            offsets = audio_old + np.cumsum(lengths)
            self.audio_offsets.resize(new + 1, axis=0)
            self.audio_offsets[old + 1:new + 1] = offsets
            for dataset, index in ((self.audio_sfreq, 1), (self.audio_start, 2), (self.audio_stop, 3)):
                dataset.resize(new, axis=0)
                dataset[old:new] = [segment[index] for segment in audio_segments]

    def close(self, *, commit: bool) -> None:
        self.file.attrs["sample_count"] = self.count
        self.file.close()
        if commit:
            self.temp_path.replace(self.path)
        elif self.temp_path.exists():
            self.temp_path.unlink()


def _selected_subjects(dataset: SpellItOutDataset, subjects: Iterable[str] | None) -> list[str]:
    available = set(dataset.subjects)
    if subjects is None:
        return sorted(available)
    selected = []
    for value in subjects:
        label = str(value).removeprefix("sub-")
        if label.isdigit():
            label = f"{int(label):02d}"
        if label not in available:
            raise ValueError(f"Subject sub-{label} is not present in the selected source.")
        selected.append(label)
    return sorted(set(selected))


def _sample_id(run: RunData, spec: DatasetSpec, trial: int, token_index: int) -> str:
    session = run.session or "01"
    return (
        f"sub-{run.subject}_ses-{session}_task-{run.task}_run-{run.run}_"
        f"trial-{trial:02d}_token-{token_index:03d}_desc-{spec.description_label}"
    )


def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _write_metadata(paths: BuildPaths, config: dict[str, Any], digest: str) -> None:
    metadata = {
        "Name": "Spell-IT-OUT syllable-segment datasets",
        "Format": "BIDS-derived HDF5 (non-standard analysis artifact)",
        "SchemaVersion": SCHEMA_VERSION,
        "SourceBIDSRoot": str(paths.bids_root),
        "SourceType": config["source"],
        "BuildConfigSHA256": digest,
        "Description": (
            "Fixed-window EEG segments aligned to syllable events, with phonetic "
            "labels and duration-matched audio for overt production samples."
        ),
    }
    (paths.output_root / "dataset_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    (paths.output_root / "build_config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )
    (paths.output_root / "README.md").write_text(
        "# Spell-IT-OUT syllable datasets\n\n"
        "This directory is a **BIDS-derived, non-standard HDF5 analysis artifact**. "
        "It is not an EEG datatype accepted by the BIDS Validator. Each shard stores "
        "source BIDS entities, event times, the resolved build configuration, and its "
        "SHA-256 digest. See `manifest.tsv` for sample-level provenance and "
        "`exclusions.tsv` for omitted candidate samples.\n",
        encoding="utf-8",
    )


def build_syllable_datasets(
    paths: BuildPaths,
    config: dict[str, Any],
    phonetic_categories_path: str | Path,
    *,
    mode: str = "validate",
    subjects: Iterable[str] | None = None,
    selected_datasets: Iterable[str] | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Validate inputs or create subject-level HDF5 shards and audit tables."""
    if mode not in {"validate", "apply"}:
        raise ValueError("mode must be 'validate' or 'apply'.")
    config = json.loads(json.dumps(config))
    specs = dataset_specs(config)
    if selected_datasets is not None:
        wanted = set(selected_datasets)
        specs = [spec for spec in specs if spec.name in wanted]
        missing = wanted - {spec.name for spec in specs}
        if missing:
            raise ValueError(f"Unknown dataset selection: {', '.join(sorted(missing))}")
        config["datasets"] = [
            item for item in config["datasets"] if item["name"] in wanted
        ]
    source_dataset = SpellItOutDataset(paths.source_root)
    preprocessed_dataset = SpellItOutDataset(paths.preprocessed_root)
    chosen_subjects = _selected_subjects(source_dataset, subjects)
    config["selection"] = {"subjects": [f"sub-{value}" for value in chosen_subjects]}
    category_names, category_lookup = read_phonetic_categories(phonetic_categories_path)
    digest = config_digest(config)

    if mode == "apply":
        if paths.output_root.exists() and any(paths.output_root.iterdir()) and not overwrite:
            raise FileExistsError(
                f"Output directory is not empty: {paths.output_root}. Pass --overwrite."
            )
        paths.output_root.mkdir(parents=True, exist_ok=True)

    manifest: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    counts = {spec.name: {"candidates": 0, "accepted": 0, "excluded": 0} for spec in specs}
    eeg_cfg = config["eeg"]
    tmin = float(eeg_cfg["tmin"])
    tmax = float(eeg_cfg["tmax"])
    bad_mode = str(eeg_cfg.get("bad_annotation_source", "source"))

    for subject in chosen_subjects:
        writers: dict[str, _ShardWriter] = {}
        try:
            tasks = sorted({spec.task for spec in specs})
            for run in source_dataset.iter_runs(subjects=subject, tasks=tasks, preload=(mode == "apply")):
                context = str(run.events_path)
                validate_event_table(run.events, context)
                run_specs = [spec for spec in specs if spec.task == run.task]
                if not run_specs:
                    continue
                bad_events = _bad_events_for_run(
                    run, mode=bad_mode, preprocessed_dataset=preprocessed_dataset
                )
                intervals = bad_intervals(bad_events)
                raw = _pick_and_preprocess(run.raw, config) if mode == "apply" else run.raw
                if mode == "validate":
                    # Validation accounts for configured resampling without loading data.
                    target = config["preprocessing"].get("resample_sfreq")
                    sfreq = float(target or raw.info["sfreq"])
                    if eeg_cfg.get("picks", "eeg") == "eeg":
                        channel_names = [raw.ch_names[index] for index in mne.pick_types(raw.info, eeg=True, exclude=[])]
                    else:
                        channel_names = list(eeg_cfg["picks"])
                    recording_end = (run.raw.n_times - 1) / run.raw.info["sfreq"]
                else:
                    sfreq = float(raw.info["sfreq"])
                    channel_names = list(raw.ch_names)
                    recording_end = (raw.n_times - 1) / sfreq
                offsets, times = _epoch_offsets(sfreq, tmin, tmax)

                audio_data = None
                audio_sfreq = None
                audio_frames = None
                audio_path = None
                if any(spec.include_audio for spec in run_specs):
                    audio_path = _source_audio_path(paths.bids_root, run)
                    info = sf.info(str(audio_path))
                    audio_sfreq = int(info.samplerate)
                    audio_frames = int(info.frames)
                    if mode == "apply":
                        audio_data, loaded_sfreq = sf.read(str(audio_path), dtype="float32", always_2d=False)
                        if loaded_sfreq != audio_sfreq:
                            raise RuntimeError(f"Audio sampling frequency changed while reading {audio_path}.")

                for spec in run_specs:
                    selected = run.events.loc[run.events["trial_type"].astype(str) == spec.event_type].copy()
                    selected_keys = [_event_key(row, context) for _, row in selected.iterrows()]
                    if len(selected_keys) != len(set(selected_keys)):
                        raise ValueError(
                            f"Duplicate (trial_index, token_index) keys for {spec.event_type} in {context}."
                        )
                    reference = None
                    if spec.pair_reference_event_type:
                        reference = run.events.loc[
                            run.events["trial_type"].astype(str) == spec.pair_reference_event_type
                        ]
                        production_keys = {_event_key(row, context) for _, row in selected.iterrows()}
                        for _, cue in reference.iterrows():
                            key = _event_key(cue, context)
                            if key not in production_keys:
                                exclusions.append({
                                    "dataset": spec.name, "subject": run.subject_id,
                                    "session": f"ses-{run.session or '01'}", "task": run.task,
                                    "run": f"run-{run.run}", "trial_index": key[0],
                                    "token_index": key[1], "syllable": str(cue["token"]),
                                    "reason": "missing_production_annotation",
                                })

                    accepted_rows: list[tuple[pd.Series, int, int, float, float]] = []
                    for _, row in selected.iterrows():
                        counts[spec.name]["candidates"] += 1
                        trial, token_index = _event_key(row, context)
                        token = str(row["token"]).strip()
                        reason = None
                        try:
                            onset = float(row["onset"])
                            duration = float(row["duration"])
                        except (TypeError, ValueError):
                            onset = duration = np.nan
                            reason = "invalid_event_time"
                        epoch_start, epoch_stop = onset + tmin, onset + tmax
                        if reason is None and (not np.isfinite(onset) or not np.isfinite(duration)):
                            reason = "invalid_event_time"
                        elif reason is None and token not in category_lookup:
                            reason = "unmapped_syllable"
                        elif reason is None and (epoch_start < 0 or epoch_stop > recording_end):
                            reason = "eeg_window_out_of_bounds"
                        elif reason is None and eeg_cfg.get("exclude_bad_overlap", True) and overlaps_bad(epoch_start, epoch_stop, intervals):
                            reason = "overlaps_bad_annotation"
                        elif reason is None and spec.include_audio:
                            assert audio_path is not None and audio_sfreq is not None
                            assert audio_frames is not None
                            audio_start = int(round(onset * audio_sfreq))
                            audio_stop = int(round((onset + duration) * audio_sfreq))
                            if duration <= 0 or audio_start < 0 or audio_stop > audio_frames:
                                reason = "audio_window_out_of_bounds"
                        if reason is not None:
                            counts[spec.name]["excluded"] += 1
                            exclusions.append({
                                "dataset": spec.name, "subject": run.subject_id,
                                "session": f"ses-{run.session or '01'}", "task": run.task,
                                "run": f"run-{run.run}", "trial_index": trial,
                                "token_index": token_index, "syllable": token, "reason": reason,
                            })
                            continue
                        accepted_rows.append((row, trial, token_index, onset, duration))

                    if not accepted_rows:
                        continue
                    counts[spec.name]["accepted"] += len(accepted_rows)
                    if mode == "validate":
                        continue

                    session = run.session or "01"
                    if spec.name not in writers:
                        filename = (
                            f"sub-{subject}_ses-{session}_task-{spec.task}_"
                            f"desc-{spec.description_label}_dataset.h5"
                        )
                        output = paths.output_root / f"sub-{subject}" / f"ses-{session}" / "eeg" / filename
                        if output.exists() and not overwrite:
                            raise FileExistsError(f"Output already exists: {output}")
                        writers[spec.name] = _ShardWriter(
                            output, spec=spec, subject=subject, session=session,
                            channel_names=channel_names, times=times, sfreq=sfreq,
                            category_names=category_names, config=config, digest=digest,
                        )
                    writer = writers[spec.name]
                    if list(writer.file["eeg/channel_names"].asstr()[:]) != channel_names:
                        raise ValueError(f"Channel order differs between runs in {writer.path}.")
                    if not np.allclose(writer.file["eeg/times"][:], times):
                        raise ValueError(f"EEG sampling grid differs between runs in {writer.path}.")

                    eeg_batch = []
                    records = []
                    categories = []
                    audio_batch = [] if spec.include_audio else None
                    for row, trial, token_index, onset, duration in accepted_rows:
                        center = raw.time_as_index(onset, use_rounding=True)[0]
                        indices = center + offsets
                        segment = raw.get_data(start=int(indices[0]), stop=int(indices[-1] + 1))
                        segment = _apply_epoch_operations(segment, times, eeg_cfg)
                        token = str(row["token"]).strip()
                        sample_id = _sample_id(run, spec, trial, token_index)
                        source_eeg = _relative(Path(run.bids_path.fpath), paths.bids_root)
                        source_events = _relative(run.events_path, paths.bids_root)
                        record = {
                            "id": sample_id, "run": f"run-{run.run}", "syllable": token,
                            "trial_index": trial, "token_index": token_index,
                            "event_onset": onset, "event_duration": duration,
                            "stimulus_id": str(row.get("stimulus_id", "n/a")),
                            "source_eeg": source_eeg, "source_events": source_events,
                        }
                        eeg_batch.append(segment)
                        records.append(record)
                        categories.append(category_lookup[token])
                        hdf5_index = writer.count + len(records) - 1
                        manifest_row = {
                            "sample_id": sample_id, "dataset": spec.name,
                            "subject": run.subject_id, "session": f"ses-{session}",
                            "task": run.task, "run": f"run-{run.run}",
                            "trial_index": trial, "token_index": token_index,
                            "syllable": token, "event_type": spec.event_type,
                            "onset": onset, "duration": duration,
                            "hdf5_path": _relative(writer.path, paths.output_root),
                            "hdf5_index": hdf5_index,
                        }
                        manifest_row.update({name: int(value) for name, value in zip(category_names, category_lookup[token])})
                        manifest.append(manifest_row)
                        if audio_batch is not None:
                            assert audio_data is not None and audio_sfreq is not None
                            audio_batch.append(_audio_segment(audio_data, audio_sfreq, onset, duration, config))
                    writer.append(
                        np.asarray(eeg_batch, dtype=eeg_cfg.get("dtype", "float32")),
                        records,
                        np.asarray(categories, dtype=np.uint8),
                        audio_batch,
                    )
        except Exception:
            for writer in writers.values():
                writer.close(commit=False)
            raise
        else:
            for writer in writers.values():
                writer.close(commit=True)

    if mode == "apply":
        _write_metadata(paths, config, digest)
        pd.DataFrame(manifest).to_csv(paths.output_root / "manifest.tsv", sep="\t", index=False)
        exclusion_columns = (
            "dataset", "subject", "session", "task", "run", "trial_index",
            "token_index", "syllable", "reason",
        )
        pd.DataFrame(exclusions, columns=exclusion_columns).to_csv(
            paths.output_root / "exclusions.tsv", sep="\t", index=False
        )
    return {
        "mode": mode,
        "subjects": len(chosen_subjects),
        "counts": counts,
        "missing_production_annotations": sum(
            row["reason"] == "missing_production_annotation" for row in exclusions
        ),
        "config_sha256": digest,
    }
