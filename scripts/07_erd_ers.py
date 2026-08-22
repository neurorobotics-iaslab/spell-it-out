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
from spellitout.utils import get_subject_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", choices=("overt", "silent"), required=True)
    parser.add_argument("--subjects", type=int, nargs="+", required=True)
    parser.add_argument("--derivatives-path", type=Path, default=Path("derivatives/preprocessing"))
    parser.add_argument("--output-path", type=Path, default=Path("derivatives/erders"))
    parser.add_argument("--input-stage", choices=("amica_input", "cleaned"), default="amica_input")
    parser.add_argument("--tmin", type=float, default=-10.0)
    parser.add_argument("--tmax", type=float, default=25.0)
    parser.add_argument("--resample-sfreq", type=float, default=256.0)
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
    )
    register_derivative_description(
        args.output_path,
        "erders",
        "Subject-level full-trial ERD/ERS epoch TFR checkpoints with BAD-span evidence.",
    )
    for number in args.subjects:
        subject = get_subject_id(number)
        eeg_dir = args.derivatives_path / subject / "ses-01" / "eeg"
        base = f"{subject}_ses-01_task-{args.condition}"
        input_path = eeg_dir / f"{base}_desc-ica-input_raw.fif"
        ica_path = eeg_dir / f"{base}_desc-amica_ica.fif"
        components_path = eeg_dir / f"{base}_desc-amica_components.tsv"
        if not input_path.exists():
            raise FileNotFoundError(f"Missing AMICA input: {input_path}")
        paths = subject_tfr_paths(args.output_path, subject, args.condition)
        cache_matches = not args.overwrite and subject_checkpoint_matches(paths, config, input_path)
        if cache_matches:
            logger.info("Reusing subject TFR checkpoint for %s.", subject)
            continue
        raw = load_condition_input(
            input_path,
            args.input_stage,
            ica_path=ica_path if args.input_stage == "cleaned" else None,
            components_path=components_path if args.input_stage == "cleaned" else None,
        )
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
            input_path,
            args.overwrite or any(path.exists() for path in paths.values()),
        )
        logger.info("Saved subject TFR checkpoint: %s", paths["tfr"])


if __name__ == "__main__":
    main(parse_args())
