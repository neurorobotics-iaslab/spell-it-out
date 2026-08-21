"""Time-frequency ERD/ERS analysis on condition-level preprocessing outputs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import logging
from pathlib import Path

import mne
import numpy as np
import pandas as pd

from .plotting import plot_erd_grid


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
    baseline: tuple[float, float] = (-3.0, -0.1)
    resample_sfreq: float = 256.0
    fmin: float = 2.0
    fmax: float = 98.0
    fstep: float = 2.0
    n_cycles_divisor: float = 2.0
    max_bad_duration: float | None = 0.5
    apply_csd: bool = True
    vlim_db: float | None = 3.0


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
    max_bad_duration: float | None,
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
        rejected = max_bad_duration is not None and overlap > max_bad_duration
        rows.append(
            {
                "epoch_index": index,
                "event_sample": int(event[0]),
                "event_time": event_time,
                "bad_overlap_seconds": overlap,
                "bad_descriptions": ";".join(descriptions),
                "kept": not rejected,
                "drop_reason": "BAD_duration" if rejected else "",
            }
        )
    return pd.DataFrame(rows)


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
) -> tuple[mne.time_frequency.AverageTFR, pd.DataFrame]:
    """Compute one subject-level, baseline-corrected average TFR."""
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
        config.max_bad_duration,
    )
    kept_indices = report.index[report["kept"]].to_numpy()
    if len(kept_indices) == 0:
        raise RuntimeError("No epochs remain after BAD-duration rejection.")
    epochs = mne.Epochs(
        raw,
        events[kept_indices],
        event_id={config.event_name: 1},
        tmin=config.tmin,
        tmax=config.tmax,
        picks="eeg",
        baseline=None,
        reject_by_annotation=False,
        preload=True,
        verbose="error",
    )
    if config.apply_csd:
        epochs = mne.preprocessing.compute_current_source_density(epochs, verbose="error")
    freqs = np.arange(config.fmin, config.fmax + config.fstep / 2, config.fstep)
    tfr = epochs.compute_tfr(
        method="morlet",
        freqs=freqs,
        n_cycles=freqs / config.n_cycles_divisor,
        use_fft=True,
        return_itc=False,
        average=True,
        n_jobs=-1,
        verbose="error",
    )
    tfr.apply_baseline(config.baseline, mode="logratio", verbose="error")
    report.loc[kept_indices, "kept"] = True
    report["subject_epoch_count"] = len(epochs)
    logger.info("%s: retained %d/%d epochs", config.event_name, len(epochs), len(events))
    return tfr, report


def run_grand_average(
    subjects: dict[str, mne.io.BaseRaw],
    config: ERDERSConfig,
    logger: logging.Logger,
) -> tuple[mne.time_frequency.AverageTFR, pd.DataFrame, dict[str, mne.time_frequency.AverageTFR]]:
    subject_tfrs = {}
    reports = []
    for subject, raw in subjects.items():
        tfr, report = compute_subject_tfr(raw, config, logger)
        subject_tfrs[subject] = tfr
        report.insert(0, "subject", subject)
        reports.append(report)
    if not subject_tfrs:
        raise RuntimeError("No subject TFRs were computed.")
    data = np.stack([tfr.data for tfr in subject_tfrs.values()], axis=0)
    grand = mne.time_frequency.AverageTFRArray(
        subject_tfrs[next(iter(subject_tfrs))].info,
        data.mean(axis=0),
        subject_tfrs[next(iter(subject_tfrs))].times,
        subject_tfrs[next(iter(subject_tfrs))].freqs,
        nave=len(subject_tfrs),
        method="morlet",
    )
    return grand, pd.concat(reports, ignore_index=True), subject_tfrs


def write_erd_ers_outputs(
    output_dir: Path,
    condition: str,
    grand: mne.time_frequency.AverageTFR,
    epoch_report: pd.DataFrame,
    subject_tfrs: dict[str, mne.time_frequency.AverageTFR],
    config: ERDERSConfig,
    input_paths: dict[str, Path],
    overwrite: bool,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    base = f"task-{condition}_desc-erders"
    paths = {
        "tfr": output_dir / f"{base}_tfr.h5",
        "epochs": output_dir / f"{base}_epochs.tsv",
        "metadata": output_dir / f"{base}.json",
        "figure": output_dir / f"{base}.png",
    }
    if not overwrite and any(path.exists() for path in paths.values()):
        raise FileExistsError(f"ERD/ERS output exists for {condition}; pass --overwrite.")
    mne.time_frequency.write_tfrs(paths["tfr"], [grand], overwrite=overwrite)
    epoch_report.to_csv(paths["epochs"], sep="\t", index=False)
    fig = plot_erd_grid(
        grand,
        ERD_BANDS,
        TRIAL_WINDOWS,
        vlim=(None if config.vlim_db is None else (-config.vlim_db, config.vlim_db)),
        cmap="RdBu_r",
        title=f"{condition.capitalize()}: Grand-average ERD/ERS (dB)",
        cbar_label="ERD / ERS (dB)",
    )
    fig.savefig(paths["figure"], dpi=300)
    metadata = {
        "schema_version": 1,
        "condition": condition,
        "subjects": sorted(subject_tfrs),
        "n_subjects": len(subject_tfrs),
        "input_stage": config.input_stage,
        "input_paths": {key: str(value) for key, value in input_paths.items()},
        "configuration": asdict(config),
        "bands_hz": ERD_BANDS,
        "plot_windows_s": TRIAL_WINDOWS,
        "normalization": "10*log10(power / mean baseline power)",
        "subject_weighting": "equal subject weighting",
        "bad_span_policy": "reject if union of BAD_* overlap exceeds max_bad_duration",
    }
    paths["metadata"].write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return paths
