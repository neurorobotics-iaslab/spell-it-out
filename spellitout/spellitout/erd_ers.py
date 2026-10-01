"""Time-frequency ERD/ERS analysis on condition-level preprocessing outputs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import logging
from pathlib import Path
from collections.abc import Sequence

import mne
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ERDERSConfig:
    """Reproducible choices for the full-trial ERD/ERS analysis."""

    input_stage: str = "ica_input"
    event_name: str = "repetition_cue"
    tmin: float = -10.0
    tmax: float =  21.2
    resample_sfreq: float = 128.0
    fmin: float = 2.0
    fmax: float = 60.0
    fstep: float = 2.0
    n_cycles_divisor: float = 2.0
    apply_csd: bool = True
    n_jobs: int = 1


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


def load_condition_input(
    paths: Path | Sequence[Path],
    input_stage: str,
) -> mne.io.BaseRaw:
    """Load a condition-level input or concatenate cleaned run-level inputs."""
    input_paths = [paths] if isinstance(paths, Path) else list(paths)
    if not input_paths:
        raise ValueError("At least one ERD/ERS input path is required.")
    if input_stage == "ica_input":
        if len(input_paths) != 1:
            raise ValueError("ica_input requires exactly one condition-level raw path.")
        return mne.io.read_raw_fif(input_paths[0], preload=True, verbose="error")
    if input_stage != "cleaned":
        raise ValueError("input_stage must be 'ica_input' or 'cleaned'.")
    runs = [
        mne.io.read_raw_fif(path, preload=True, verbose="error")
        for path in input_paths
    ]
    return mne.concatenate_raws(runs)


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
    if not np.isclose(epochs.info["sfreq"], config.resample_sfreq):
        epochs.resample(config.resample_sfreq, npad="auto", verbose="error")
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
        n_jobs=config.n_jobs,
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
    input_paths: Path | Sequence[Path],
) -> bool:
    """Return whether an existing subject checkpoint matches its calculation input."""
    if not all(path.exists() for path in paths.values()):
        return False
    metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    configuration = json.loads(json.dumps(asdict(config)))
    source_paths = [input_paths] if isinstance(input_paths, Path) else list(input_paths)
    return (
        metadata.get("configuration") == configuration
        and metadata.get("input_paths") == [str(path) for path in source_paths]
        and metadata.get("input_mtimes_ns")
        == [path.stat().st_mtime_ns for path in source_paths]
    )


def write_subject_tfr_outputs(
    output_dir: Path,
    subject: str,
    condition: str,
    tfr: mne.time_frequency.EpochsTFR,
    epoch_report: pd.DataFrame,
    annotations: pd.DataFrame,
    config: ERDERSConfig,
    input_paths: Path | Sequence[Path],
    overwrite: bool,
) -> dict[str, Path]:
    paths = subject_tfr_paths(output_dir, subject, condition)
    paths["tfr"].parent.mkdir(parents=True, exist_ok=True)
    if not overwrite and any(path.exists() for path in paths.values()):
        raise FileExistsError(f"ERD/ERS output exists for {subject} {condition}; pass --overwrite.")
    mne.time_frequency.write_tfrs(paths["tfr"], [tfr], overwrite=overwrite)
    epoch_report.to_csv(paths["epochs"], sep="\t", index=False)
    annotations.to_csv(paths["annotations"], sep="\t", index=False)
    source_paths = [input_paths] if isinstance(input_paths, Path) else list(input_paths)
    metadata = {
        "schema_version": 3,
        "subject": subject,
        "condition": condition,
        "input_stage": config.input_stage,
        "input_paths": [str(path) for path in source_paths],
        "input_mtimes_ns": [path.stat().st_mtime_ns for path in source_paths],
        "configuration": asdict(config),
        "normalization": "none; apply baseline during plotting",
        "bad_span_policy": "all extractable epochs retained; BAD_* overlap is recorded in epochs.tsv",
    }
    paths["metadata"].write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return paths
