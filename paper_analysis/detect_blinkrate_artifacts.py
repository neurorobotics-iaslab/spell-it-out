"""Detect muscle and rapid amplitude artifacts in the original EEG recordings.

The original FIFs are loaded into memory and resampled to 512 Hz for analysis;
the 110--140 Hz muscle band is preserved. Subject-specific muscle thresholds
are calibrated against reviewed muscle candidates where possible. Amplitude
thresholds use a robust empirical tail because reviewed amplitude examples
are too sparse for supervised calibration.

Outputs are confined to paper_analysis/blinkrate/artifact_detection and do not
change raw data or reviewed annotations.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from scipy.ndimage import minimum_filter1d
from scipy.signal import resample_poly


LOG = logging.getLogger("blinkrate_artifacts")
DEFAULT_ROOT = Path("paper_analysis/blinkrate/artifact_detection")
MIN_REVIEWED_PER_CLASS = 10
EEG_EXCLUDE = {"AF7", "AF8"}
INTERVAL_COLUMNS = (
    "subject", "task", "run", "detector", "onset", "duration", "threshold",
    "threshold_unit", "source_file",
)
THRESHOLD_COLUMNS = (
    "subject", "muscle_threshold_z", "muscle_method", "muscle_balanced_accuracy",
    "muscle_baseline_accuracy", "n_confirmed_muscle", "n_dismissed_muscle",
    "amplitude_threshold_uv_per_sample", "amplitude_method", "amplitude_quantile",
    "n_runs_processed", "n_muscle_intervals", "n_amplitude_intervals",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-path", type=Path,
        default=Path(os.environ["MNE_DATASETS_SPELLITOUT_DATA_PATH"])
        if "MNE_DATASETS_SPELLITOUT_DATA_PATH" in os.environ else None,
        help="Original dataset root; defaults to MNE_DATASETS_SPELLITOUT_DATA_PATH.",
    )
    parser.add_argument("--review-path", type=Path, default=Path("derivatives/preprocessing"))
    parser.add_argument("--output-path", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--conditions", nargs="+", choices=("covert", "overt"), default=("covert", "overt"))
    parser.add_argument("--subjects", type=int, nargs="+", default=None)
    parser.add_argument("--runs", type=int, nargs="+", default=(1, 2, 3))
    parser.add_argument("--analysis-sfreq", type=float, default=512.0)
    parser.add_argument("--amplitude-quantile", type=float, default=0.9999)
    parser.add_argument("--amplitude-sample-seconds", type=float, default=30.0)
    return parser.parse_args()


def _original_paths(data_path: Path, subject: str, conditions: tuple[str, ...], runs: tuple[int, ...]):
    for task in conditions:
        for run in runs:
            stem = f"{subject}_ses-01_task-{task}_run-{run:02d}"
            path = data_path / task / subject / "ses-01" / "eeg" / f"{stem}_eeg.fif"
            if path.is_file():
                yield task, run, path
            else:
                LOG.warning("No original recording: %s", path)


def _eeg_names(info: mne.Info) -> list[str]:
    picks = mne.pick_types(info, eeg=True, eog=False, exclude="bads")
    return [info["ch_names"][index] for index in picks if info["ch_names"][index] not in EEG_EXCLUDE]


def _amplitude_sample(path: Path, seconds: float, sfreq: float) -> np.ndarray:
    """Sample the per-channel 5 ms derivative floor across the whole run."""
    raw = mne.io.read_raw_fif(path, preload=False, verbose="error")
    names = _eeg_names(raw.info)
    if not names:
        raise ValueError(f"No usable EEG channels in {path}")
    native_sfreq = float(raw.info["sfreq"])
    ratio = native_sfreq / sfreq
    if abs(ratio - round(ratio)) > 1e-9:
        raise ValueError(f"Expected integer resampling ratio in {path}: {ratio}")
    samples_per_window = min(raw.n_times, max(2, int(round(seconds / 3 * native_sfreq))))
    starts = np.linspace(0, max(0, raw.n_times - samples_per_window), 3, dtype=int)
    n_samples = max(1, int(np.ceil(0.005 * sfreq)))
    samples = []
    for start in starts:
        block = raw.get_data(picks=names, start=int(start), stop=int(start + samples_per_window))
        block = resample_poly(block, up=1, down=int(round(ratio)), axis=1)
        derivative = np.abs(np.diff(block, axis=1))
        floor = minimum_filter1d(derivative, size=n_samples, axis=1, mode="constant", cval=0)
        # Equidistant sampling avoids storing millions of calibration samples.
        samples.append(floor[:, ::16].ravel())
    raw.close()
    return np.concatenate(samples)


def _reviewed_muscle_scores(
    score: np.ndarray, sfreq: float, table_path: Path,
) -> tuple[list[float], list[float]]:
    if not table_path.is_file():
        return [], []
    table = pd.read_csv(table_path, sep="\t")
    required = {"trial_type", "review_status", "reviewed_description", "onset", "duration"}
    if not required.issubset(table.columns):
        LOG.warning("Review table lacks muscle columns: %s", table_path)
        return [], []
    approved, dismissed = [], []
    for row in table.itertuples(index=False):
        if row.trial_type != "MUSCLE_CANDIDATE":
            continue
        positive = row.review_status == "approved_bad" and row.reviewed_description == "BAD_MUSCLE"
        negative = row.review_status == "dismissed"
        if not (positive or negative):
            continue
        start = max(0, int(np.floor(float(row.onset) * sfreq)))
        stop = min(len(score), int(np.ceil((float(row.onset) + float(row.duration)) * sfreq)))
        if stop <= start:
            continue
        value = np.nanmax(score[start:stop])
        if np.isfinite(value):
            (approved if positive else dismissed).append(float(value))
    return approved, dismissed


def choose_muscle_threshold(
    confirmed: list[float], dismissed: list[float],
) -> tuple[float, str, float, float]:
    """Maximize balanced accuracy on reviewed candidates; prefer conservative ties."""
    if len(confirmed) < MIN_REVIEWED_PER_CLASS or len(dismissed) < MIN_REVIEWED_PER_CLASS:
        return 4.0, "default_insufficient_reviewed_examples", np.nan, np.nan
    positive = np.asarray(confirmed)
    negative = np.asarray(dismissed)
    values = np.unique(np.r_[positive, negative])
    middle = (values[:-1] + values[1:]) / 2
    # A decision only changes between adjacent observed scores. Searching
    # these boundaries avoids an arbitrary upper limit on the z-score.
    grid = np.unique(np.r_[2.5, 4.0, middle, values[-1] + 1e-6])
    grid = grid[grid >= 2.5]
    scores = np.asarray([
        0.5 * (np.mean(positive >= threshold) + np.mean(negative < threshold))
        for threshold in grid
    ])
    best = float(np.nanmax(scores))
    # Within 0.5 percentage points of the best score, prefer the threshold
    # with fewer false alarms rather than a more permissive one.
    eligible = np.flatnonzero(scores >= best - 0.005)
    chosen = float(grid[eligible[-1]])
    baseline = 0.5 * (np.mean(positive >= 4.0) + np.mean(negative < 4.0))
    return chosen, "reviewed_candidate_balanced_accuracy", float(scores[eligible[-1]]), float(baseline)


def score_intervals(
    score: np.ndarray, sfreq: float, threshold: float,
    *, merge_gap_seconds: float = 0.1, min_duration_seconds: float = 0.1,
) -> list[tuple[float, float]]:
    """Threshold MNE's muscle score, joining very short gaps."""
    active = np.asarray(score > threshold, dtype=bool) & np.isfinite(score)
    edges = np.diff(np.r_[False, active, False].astype(np.int8))
    starts = np.flatnonzero(edges == 1)
    stops = np.flatnonzero(edges == -1)
    if not len(starts):
        return []
    intervals: list[list[int]] = [[int(starts[0]), int(stops[0])]]
    max_gap = int(round(merge_gap_seconds * sfreq))
    for start, stop in zip(starts[1:], stops[1:]):
        if int(start) - intervals[-1][1] <= max_gap:
            intervals[-1][1] = int(stop)
        else:
            intervals.append([int(start), int(stop)])
    return [
        (start / sfreq, (stop - start) / sfreq)
        for start, stop in intervals
        if (stop - start) / sfreq >= min_duration_seconds
    ]


def _write_outputs(root: Path, intervals: list[dict], thresholds: list[dict]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(intervals, columns=INTERVAL_COLUMNS).to_csv(
        root / "artifact_intervals.tsv", sep="\t", index=False,
    )
    pd.DataFrame(thresholds, columns=THRESHOLD_COLUMNS).to_csv(
        root / "subject_thresholds.tsv", sep="\t", index=False,
    )


def main(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    if args.data_path is None or not args.data_path.is_dir():
        raise FileNotFoundError("Provide --data-path pointing to the original dataset root.")
    if args.analysis_sfreq <= 2 * 140:
        raise ValueError("--analysis-sfreq must exceed 280 Hz to retain the 110--140 Hz muscle band.")
    if not 0 < args.amplitude_quantile < 1:
        raise ValueError("--amplitude-quantile must be between 0 and 1.")
    if args.amplitude_sample_seconds <= 0:
        raise ValueError("--amplitude-sample-seconds must be positive.")
    subjects = args.subjects or sorted({
        int(path.name.removeprefix("sub-"))
        for task in args.conditions
        for path in (args.data_path / task).glob("sub-*")
        if path.is_dir() and path.name.removeprefix("sub-").isdigit()
    })
    if not subjects:
        raise FileNotFoundError(f"No subjects found under {args.data_path}")
    all_intervals: list[dict] = []
    all_thresholds: list[dict] = []
    for number in subjects:
        subject = f"sub-{number:02d}"
        paths = list(_original_paths(args.data_path, subject, tuple(args.conditions), tuple(args.runs)))
        if not paths:
            LOG.warning("Skipping %s: no original recordings", subject)
            continue
        # Quantile calibration uses windows from the beginning, middle, and
        # end of every available original run.
        calibration = np.concatenate([
            _amplitude_sample(path, args.amplitude_sample_seconds, args.analysis_sfreq)
            for _, _, path in paths
        ])
        amplitude_threshold = float(np.quantile(calibration[np.isfinite(calibration)], args.amplitude_quantile))
        if amplitude_threshold <= 0:
            raise ValueError(f"Amplitude calibration produced a non-positive threshold for {subject}")
        LOG.info("%s: amplitude threshold %.2f µV/sample", subject, amplitude_threshold * 1e6)
        run_data = []
        confirmed: list[float] = []
        dismissed: list[float] = []
        for task, run, path in paths:
            LOG.info("Analyzing %s task-%s run-%02d", subject, task, run)
            raw = mne.io.read_raw_fif(path, preload=False, verbose="error")
            raw.pick(_eeg_names(raw.info)).load_data()
            raw.resample(args.analysis_sfreq, verbose="error")
            muscle_annotations, score = mne.preprocessing.annotate_muscle_zscore(
                raw, threshold=4.0, ch_type="eeg", filter_freq=(110, 140),
                min_length_good=0.1, verbose="error",
            )
            del muscle_annotations
            amp_annotations, amp_bads = mne.preprocessing.annotate_amplitude(
                raw, peak=amplitude_threshold, min_duration=0.005,
                bad_percent=100.0, picks="eeg", verbose="error",
            )
            if amp_bads:
                LOG.warning("%s task-%s run-%02d: amplitude detector marked channels bad: %s", subject, task, run, amp_bads)
            raw.close()
            stem = f"{subject}_ses-01_task-{task}_run-{run:02d}"
            review_path = args.review_path / subject / "ses-01" / "eeg" / f"{stem}_desc-qc_annotations.tsv"
            pos, neg = _reviewed_muscle_scores(score, args.analysis_sfreq, review_path)
            confirmed.extend(pos)
            dismissed.extend(neg)
            run_data.append((task, run, path, score, amp_annotations))
        muscle_threshold, method, accuracy, baseline = choose_muscle_threshold(confirmed, dismissed)
        subject_muscle_count = 0
        subject_amplitude_count = 0
        for task, run, path, score, amp_annotations in run_data:
            for onset, duration in score_intervals(score, args.analysis_sfreq, muscle_threshold):
                all_intervals.append({
                    "subject": subject, "task": task, "run": f"{run:02d}",
                    "detector": "muscle", "onset": onset, "duration": duration,
                    "threshold": muscle_threshold, "threshold_unit": "z_score",
                    "source_file": str(path.relative_to(args.data_path)),
                })
                subject_muscle_count += 1
            for onset, duration in zip(amp_annotations.onset, amp_annotations.duration):
                all_intervals.append({
                    "subject": subject, "task": task, "run": f"{run:02d}",
                    "detector": "amplitude", "onset": float(onset), "duration": float(duration),
                    "threshold": amplitude_threshold * 1e6,
                    "threshold_unit": "microvolt_per_sample",
                    "source_file": str(path.relative_to(args.data_path)),
                })
                subject_amplitude_count += 1
        all_thresholds.append({
            "subject": subject, "muscle_threshold_z": muscle_threshold,
            "muscle_method": method, "muscle_balanced_accuracy": accuracy,
            "muscle_baseline_accuracy": baseline,
            "n_confirmed_muscle": len(confirmed), "n_dismissed_muscle": len(dismissed),
            "amplitude_threshold_uv_per_sample": amplitude_threshold * 1e6,
            "amplitude_method": "subject_empirical_quantile_of_5ms_derivative_floor",
            "amplitude_quantile": args.amplitude_quantile,
            "n_runs_processed": len(run_data), "n_muscle_intervals": subject_muscle_count,
            "n_amplitude_intervals": subject_amplitude_count,
        })
        _write_outputs(args.output_path, all_intervals, all_thresholds)
        LOG.info("%s: muscle z=%.2f (%s); %d muscle and %d amplitude intervals",
                 subject, muscle_threshold, method, subject_muscle_count, subject_amplitude_count)

    args.output_path.mkdir(parents=True, exist_ok=True)
    (args.output_path / "method.json").write_text(json.dumps({
        "Description": "Exploratory muscle and rapid amplitude detection for blink-rate QC.",
        "Source": "Original task/run EEG FIFs, resampled in memory to analysis_sfreq; no 100 Hz low-pass.",
        "Muscle": "MNE annotate_muscle_zscore EEG 110-140 Hz score; threshold selected per subject by balanced accuracy against approved/dismissed MUSCLE_CANDIDATE review rows; 4.0 fallback if fewer than 10 per class; merge gaps <=100 ms; discard spans <100 ms.",
        "Amplitude": "MNE annotate_amplitude peak detector; threshold per subject from empirical quantile of minimum absolute adjacent-sample EEG change across 5 ms, sampled at the beginning, middle, and end of each raw run. This detects rapid changes, not sustained voltage offsets. No reliable reviewed amplitude labels were available for supervised tuning.",
        "analysis_sfreq": args.analysis_sfreq,
        "muscle_band_hz": [110, 140],
        "amplitude_quantile": args.amplitude_quantile,
        "amplitude_sample_seconds_per_run": args.amplitude_sample_seconds,
        "excluded_eeg_channels": sorted(EEG_EXCLUDE),
        "original_data_path": str(args.data_path),
    }, indent=2), encoding="utf-8")
    LOG.info("Saved results to %s", args.output_path)


if __name__ == "__main__":
    main(parse_args())
