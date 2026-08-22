"""Time-frequency ERD/ERS analysis on condition-level preprocessing outputs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import logging
from pathlib import Path

import mne
import numpy as np
import pandas as pd

ERD_BANDS = {
    "Theta": (4.0, 8.0),
    "Alpha": (8.0, 12.0),
    "Low Beta": (12.0, 16.0),
    "High Beta": (16.0, 30.0),
    "Low Gamma": (30.0, 45.0),
}
TRIAL_WINDOWS = {
    "Listening": (-6.5, -3.5),
    "Baseline": (-3.0, 0.0),
    "Sentence repetition": (1.0, 4.0),
    "Repetition end": (6.0, 7.0),
    "Dots": (7.0, 8.8),
    "Syllable repetition": (8.8, 21.2),
}


@dataclass(frozen=True)
class ERDERSConfig:
    """Reproducible choices for the full-trial ERD/ERS analysis."""

    input_stage: str = "amica_input"
    event_name: str = "repetition_cue"
    tmin: float = -10.0
    tmax: float = 25.0
    resample_sfreq: float = 256.0
    fmin: float = 2.0
    fmax: float = 98.0
    fstep: float = 2.0
    n_cycles_divisor: float = 2.0
    apply_csd: bool = True


def _bad_intervals(raw: mne.io.BaseRaw) -> list[tuple[float, float, str]]:
    return [
        (float(onset), float(onset + duration), str(description))
        for onset, duration, description in zip(
            raw.annotations.onset,
            raw.annotations.duration,
            raw.annotations.description,
        )
        if str(description).upper().startswith("BAD")
    ]


def _union_overlap_duration(
    start: float,
    stop: float,
    intervals: list[tuple[float, float, str]],
) -> tuple[float, list[str]]:
    clipped = sorted(
        (max(start, interval_start), min(stop, interval_stop), description)
        for interval_start, interval_stop, description in intervals
        if interval_stop > start and interval_start < stop
    )
    if not clipped:
        return 0.0, []
    total = 0.0
    current_start, current_stop = clipped[0][:2]
    descriptions = [clipped[0][2]]
    for interval_start, interval_stop, description in clipped[1:]:
        descriptions.append(description)
        if interval_start <= current_stop:
            current_stop = max(current_stop, interval_stop)
        else:
            total += current_stop - current_start
            current_start, current_stop = interval_start, interval_stop
    total += current_stop - current_start
    return total, sorted(set(descriptions))


def epoch_badness(
    raw: mne.io.BaseRaw,
    events: np.ndarray,
    tmin: float,
    tmax: float,
) -> pd.DataFrame:
    """Report BAD overlap per event, counting overlapping spans only once."""
    intervals = _bad_intervals(raw)
    sfreq = float(raw.info["sfreq"])
    rows = []
    for index, event in enumerate(events):
        event_time = (int(event[0]) - raw.first_samp) / sfreq
        overlap, descriptions = _union_overlap_duration(
            event_time + tmin,
            event_time + tmax,
            intervals,
        )
        rows.append(
            {
                "epoch_index": index,
                "event_sample": int(event[0]),
                "event_time": event_time,
                "bad_overlap_seconds": overlap,
                "bad_descriptions": ";".join(descriptions),
                "tfr_available": False,
                "drop_reason": "",
            }
        )
    return pd.DataFrame(rows)


def annotations_table(raw: mne.io.BaseRaw) -> pd.DataFrame:
    """Serialise all raw annotations alongside a subject-level TFR checkpoint."""
    return pd.DataFrame(
        {
            "onset": raw.annotations.onset,
            "duration": raw.annotations.duration,
            "description": raw.annotations.description,
        }
    )


def _apply_ica_decisions(raw: mne.io.BaseRaw, ica_path: Path, components_path: Path) -> mne.io.BaseRaw:
    ica = mne.preprocessing.read_ica(ica_path, verbose="error")
    table = pd.read_csv(components_path, sep="\t")
    if "final_decision" not in table:
        raise ValueError(f"ICA table lacks final_decision: {components_path}")
    unreviewed = table.index[table["final_decision"] == "unreviewed"].tolist()
    if unreviewed:
        raise RuntimeError(
            f"Cannot use cleaned input: {len(unreviewed)} ICA component decisions remain unreviewed."
        )
    ica.exclude = table.index[table["final_decision"] == "reject"].tolist()
    cleaned = raw.copy().load_data()
    ica.apply(cleaned, verbose="error")
    return cleaned


def load_condition_input(
    path: Path,
    input_stage: str,
    ica_path: Path | None = None,
    components_path: Path | None = None,
) -> mne.io.BaseRaw:
    """Load AMICA input or apply reviewed ICA decisions in memory."""
    raw = mne.io.read_raw_fif(path, preload=True, verbose="error")
    if input_stage == "amica_input":
        return raw
    if input_stage != "cleaned":
        raise ValueError("input_stage must be 'amica_input' or 'cleaned'.")
    if ica_path is None or components_path is None:
        raise ValueError("cleaned input requires an ICA FIF and component TSV.")
    return _apply_ica_decisions(raw, ica_path, components_path)


def compute_subject_tfr(
    raw: mne.io.BaseRaw,
    config: ERDERSConfig,
    logger: logging.Logger,
) -> tuple[mne.time_frequency.EpochsTFR, pd.DataFrame]:
    """Compute and retain every extractable subject-level epoch TFR."""
    events, event_ids = mne.events_from_annotations(
        raw,
        event_id={config.event_name: 1},
        verbose="error",
    )
    if config.event_name not in event_ids or len(events) == 0:
        raise RuntimeError(f"No {config.event_name!r} events found.")
    report = epoch_badness(
        raw,
        events,
        config.tmin,
        config.tmax,
    )
    epochs = mne.Epochs(
        raw,
        events,
        event_id={config.event_name: 1},
        tmin=config.tmin,
        tmax=config.tmax,
        picks="eeg",
        baseline=None,
        reject_by_annotation=False,
        preload=True,
        metadata=report,
        verbose="error",
    )
    available_indices = epochs.selection
    report.loc[available_indices, "tfr_available"] = True
    report.loc[~report["tfr_available"], "drop_reason"] = "TOO_SHORT"
    if not len(epochs):
        raise RuntimeError("No full-length epochs are available for TFR computation.")
    if config.apply_csd:
        epochs = mne.preprocessing.compute_current_source_density(epochs, verbose="error")
    freqs = np.arange(config.fmin, config.fmax + config.fstep / 2, config.fstep)
    tfr = epochs.compute_tfr(
        method="morlet",
        freqs=freqs,
        n_cycles=freqs / config.n_cycles_divisor,
        use_fft=True,
        return_itc=False,
        average=False,
        n_jobs=-1,
        verbose="error",
    )
    logger.info("%s: computed TFRs for %d/%d full-length epochs", config.event_name, len(epochs), len(events))
    return tfr, report


def subject_tfr_paths(output_root: Path, subject: str, condition: str) -> dict[str, Path]:
    directory = output_root / subject / "ses-01" / "eeg"
    base = f"{subject}_ses-01_task-{condition}_desc-erders"
    return {
        "tfr": directory / f"{base}_tfr.h5",
        "epochs": directory / f"{base}_epochs.tsv",
        "annotations": directory / f"{base}_annotations.tsv",
        "metadata": directory / f"{base}.json",
    }


def subject_checkpoint_matches(
    paths: dict[str, Path],
    config: ERDERSConfig,
    input_path: Path,
) -> bool:
    """Return whether an existing subject checkpoint matches its calculation input."""
    if not all(path.exists() for path in paths.values()):
        return False
    metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    configuration = json.loads(json.dumps(asdict(config)))
    return (
        metadata.get("configuration") == configuration
        and metadata.get("input_mtime_ns") == input_path.stat().st_mtime_ns
        and metadata.get("input_path") == str(input_path)
    )


def write_subject_tfr_outputs(
    output_dir: Path,
    subject: str,
    condition: str,
    tfr: mne.time_frequency.EpochsTFR,
    epoch_report: pd.DataFrame,
    annotations: pd.DataFrame,
    config: ERDERSConfig,
    input_path: Path,
    overwrite: bool,
) -> dict[str, Path]:
    paths = subject_tfr_paths(output_dir, subject, condition)
    paths["tfr"].parent.mkdir(parents=True, exist_ok=True)
    if not overwrite and any(path.exists() for path in paths.values()):
        raise FileExistsError(f"ERD/ERS output exists for {subject} {condition}; pass --overwrite.")
    mne.time_frequency.write_tfrs(paths["tfr"], [tfr], overwrite=overwrite)
    epoch_report.to_csv(paths["epochs"], sep="\t", index=False)
    annotations.to_csv(paths["annotations"], sep="\t", index=False)
    metadata = {
        "schema_version": 2,
        "subject": subject,
        "condition": condition,
        "input_stage": config.input_stage,
        "input_path": str(input_path),
        "input_mtime_ns": input_path.stat().st_mtime_ns,
        "configuration": asdict(config),
        "bands_hz": ERD_BANDS,
        "plot_windows_s": TRIAL_WINDOWS,
        "normalization": "none; apply baseline during plotting",
        "bad_span_policy": "all extractable epochs retained; BAD_* overlap is recorded in epochs.tsv",
    }
    paths["metadata"].write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return paths
