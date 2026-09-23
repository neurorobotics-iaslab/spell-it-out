"""Compute and save all subject-level full-trial ERD/ERS epoch TFRs."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from spellitout.erd_ers import (
    ERDERSConfig,
    annotations_table,
    compute_subject_tfr,
    load_condition_input,
    subject_checkpoint_matches,
    subject_tfr_paths,
    write_subject_tfr_outputs,
)
from spellitout.preprocessing import register_derivative_description
from spellitout.utils import get_run_id, get_subject_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", choices=("overt", "silent"), required=True)
    parser.add_argument("--subjects", type=int, nargs="+", required=True)
    parser.add_argument("--runs", type=int, nargs="+", default=(1, 2, 3))
    parser.add_argument("--derivatives-path", type=Path, default=Path("derivatives/preprocessing"))
    parser.add_argument("--output-path", type=Path, default=Path("derivatives/erders"))
    parser.add_argument("--input-stage", choices=("ica_input", "cleaned"), default="ica_input")
    parser.add_argument("--tmin", type=float, default=-10.0)
    parser.add_argument("--tmax", type=float, default=22.0)
    parser.add_argument("--resample-sfreq", type=float, default=128.0)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--no-csd", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    logger = logging.getLogger("spellitout.erders")
    config = ERDERSConfig(
        input_stage=args.input_stage,
        tmin=args.tmin,
        tmax=args.tmax,
        resample_sfreq=args.resample_sfreq,
        apply_csd=not args.no_csd,
        n_jobs=args.n_jobs,
    )
    if args.n_jobs == 0:
        raise ValueError("--n-jobs cannot be zero.")
    if config.resample_sfreq <= 2 * config.fmax:
        raise ValueError("--resample-sfreq must be greater than twice the 60 Hz TFR maximum.")
    register_derivative_description(
        args.output_path,
        "erders",
        "Subject-level full-trial ERD/ERS epoch TFR checkpoints with BAD-span evidence.",
    )
    for number in args.subjects:
        subject = get_subject_id(number)
        eeg_dir = args.derivatives_path / subject / "ses-01" / "eeg"
        base = f"{subject}_ses-01_task-{args.condition}"
        if args.input_stage == "cleaned":
            input_paths = [
                eeg_dir
                / f"{get_run_id(subject, args.condition, run_number)}_desc-infomax-cleaned_raw.fif"
                for run_number in args.runs
            ]
        else:
            input_paths = [eeg_dir / f"{base}_desc-infomax-input_raw.fif"]
        missing_inputs = [path for path in input_paths if not path.exists()]
        if missing_inputs:
            raise FileNotFoundError(
                f"Missing {args.input_stage} ERD/ERS input(s): "
                + ", ".join(str(path) for path in missing_inputs)
            )
        paths = subject_tfr_paths(args.output_path, subject, args.condition)
        cache_matches = not args.overwrite and subject_checkpoint_matches(
            paths, config, input_paths
        )
        if cache_matches:
            logger.info("Reusing subject TFR checkpoint for %s.", subject)
            continue
        raw = load_condition_input(input_paths, args.input_stage)
        if args.resample_sfreq and raw.info["sfreq"] != args.resample_sfreq:
            raw.resample(args.resample_sfreq, verbose="error")
        tfr, report = compute_subject_tfr(raw, config, logger)
        paths = write_subject_tfr_outputs(
            args.output_path,
            subject,
            args.condition,
            tfr,
            report,
            annotations_table(raw),
            config,
            input_paths,
            args.overwrite or any(path.exists() for path in paths.values()),
        )
        logger.info("Saved subject TFR checkpoint: %s", paths["tfr"])


if __name__ == "__main__":
    main(parse_args())
