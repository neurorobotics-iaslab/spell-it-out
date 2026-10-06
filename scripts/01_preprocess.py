"""Create run-level, minimally processed Spell-IT-OUT EEG derivatives.

This is stage 1 of the refactored workflow. It does not run PREP, interpolate
channels, concatenate runs, or fit ICA. Those stages consume the explicit QC
outputs produced here.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import sys

sys.path.extend([".", ".."]) 

from spellitout import SpellItOutDataset
from spellitout.preprocessing import (
    MinimalPreprocessingConfig,
    build_minimal_run,
    ensure_derivative_scaffold,
    write_minimal_run,
)
from spellitout.utils import get_run_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", choices=("overt", "covert"), required=True)
    parser.add_argument("--subjects", type=int, nargs="+", default=None)
    parser.add_argument("--runs", type=int, nargs="+", default=(1, 2, 3))
    parser.add_argument("--data-path", type=Path, default=None)
    parser.add_argument(
        "--derivatives-path",
        type=Path,
        default=Path("derivatives/preprocessing"),
    )
    parser.add_argument("--drop-channels", nargs="*", default=("AF7", "AF8"))
    parser.add_argument("--resample-sfreq", type=float, default=512.0)
    parser.add_argument("--l-freq", type=float, default=1.0)
    parser.add_argument("--h-freq", type=float, default=100.0)
    parser.add_argument("--no-blink-annotations", action="store_true")
    parser.add_argument("--blink-channels", nargs="*", default=())
    parser.add_argument(
        "--blink-threshold-uv",
        "--blink-threshold",
        dest="blink_threshold_uv",
        type=float,
        default=None,
        help="Optional peak threshold for MNE blink detection, in microvolts. Omit to use MNE's automatic threshold.",
    )
    parser.add_argument(
        "--blink-l-freq",
        type=float,
        default=1.0,
        help="Low-cut frequency (Hz) used only by the blink detector (default: 1 Hz).",
    )
    parser.add_argument(
        "--blink-h-freq",
        type=float,
        default=10.0,
        help="High-cut frequency (Hz) used only by the blink detector (default: 10 Hz).",
    )
    parser.add_argument("--no-muscle-candidates", action="store_true")
    parser.add_argument("--muscle-threshold", type=float, default=4.0)
    parser.add_argument("--muscle-merge-gap", type=float, default=0.10)
    parser.add_argument("--muscle-min-duration", type=float, default=0.10)
    parser.add_argument(
        "--peak-uv",
        type=float,
        default=None,
        help="Optional amplitude-candidate threshold in microvolts.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    logger = logging.getLogger("spellitout.minproc")
    config = MinimalPreprocessingConfig(
        drop_channels=tuple(args.drop_channels),
        resample_sfreq_hz=args.resample_sfreq,
        l_freq_hz=args.l_freq,
        h_freq_hz=args.h_freq,
        detect_blinks=not args.no_blink_annotations,
        blink_channels=tuple(args.blink_channels),
        blink_threshold_volts=(
            None
            if args.blink_threshold_uv is None
            else args.blink_threshold_uv * 1e-6
        ),
        blink_l_freq_hz=args.blink_l_freq,
        blink_h_freq_hz=args.blink_h_freq,
        detect_muscle_candidates=not args.no_muscle_candidates,
        muscle_zscore_threshold=args.muscle_threshold,
        muscle_merge_gap_seconds=args.muscle_merge_gap,
        muscle_min_duration_seconds=args.muscle_min_duration,
        amplitude_peak_volts=(None if args.peak_uv is None else args.peak_uv * 1e-6),
    )
    ensure_derivative_scaffold(args.derivatives_path)
    dataset = SpellItOutDataset(args.data_path)
    for run in dataset.iter_runs(
        subjects=args.subjects,
        tasks=args.condition,
        runs=args.runs,
        preload=True,
    ):
        if run.session is None or run.run_number is None:
            raise ValueError(f"Recording lacks a session or run entity: {run.bids_path}")
        output_dir = (
            args.derivatives_path / run.subject_id / f"ses-{run.session}" / "eeg"
        )
        base = get_run_id(run.subject_id, run.task, run.run_number)
        logger.info("Minimal preprocessing: %s", base)
        minimal_raw, candidates = build_minimal_run(run.raw, config, logger)
        paths = write_minimal_run(
            minimal_raw,
            candidates,
            output_dir,
            base,
            config,
            source={
                "subject": run.subject_id,
                "task": run.task,
                "run": f"run-{run.run}",
            },
            overwrite=args.overwrite,
        )
        logger.info("Saved %s", paths["raw"])


if __name__ == "__main__":
    main(parse_args())
