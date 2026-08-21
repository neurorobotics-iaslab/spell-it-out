"""Compute full-trial grand-average ERD/ERS figures and checkpoints."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from spellitout.erd_ers import ERDERSConfig, load_condition_input, run_grand_average, write_erd_ers_outputs
from spellitout.preprocessing import register_derivative_description
from spellitout.utils import get_subject_id


def optional_float(value: str) -> float | None:
    if value.strip().lower() in {"none", "auto"}:
        return None
    return float(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", choices=("overt", "silent"), required=True)
    parser.add_argument("--subjects", type=int, nargs="+", required=True)
    parser.add_argument("--derivatives-path", type=Path, default=Path("derivatives/preprocessing"))
    parser.add_argument("--output-path", type=Path, default=Path("derivatives/erders"))
    parser.add_argument("--input-stage", choices=("amica_input", "cleaned"), default="amica_input")
    parser.add_argument("--tmin", type=float, default=-10.0)
    parser.add_argument("--tmax", type=float, default=25.0)
    parser.add_argument("--max-bad-duration", type=float, default=0.5)
    parser.add_argument("--resample-sfreq", type=float, default=256.0)
    parser.add_argument("--no-csd", action="store_true")
    parser.add_argument(
        "--vlim-db",
        type=optional_float,
        default=3.0,
        help="Symmetric color limit in dB; use 'none' for automatic limits.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    if args.max_bad_duration < 0:
        raise ValueError("--max-bad-duration must be non-negative.")
    if args.vlim_db is not None and args.vlim_db <= 0:
        raise ValueError("--vlim-db must be positive or 'none'.")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    logger = logging.getLogger("spellitout.erders")
    config = ERDERSConfig(
        input_stage=args.input_stage,
        tmin=args.tmin,
        tmax=args.tmax,
        max_bad_duration=args.max_bad_duration,
        resample_sfreq=args.resample_sfreq,
        apply_csd=not args.no_csd,
        vlim_db=args.vlim_db,
    )
    register_derivative_description(
        args.output_path,
        "erders",
        "Full-trial subject-weighted grand-average ERD/ERS time-frequency results.",
    )
    subjects = {}
    input_paths = {}
    for number in args.subjects:
        subject = get_subject_id(number)
        eeg_dir = args.derivatives_path / subject / "ses-01" / "eeg"
        base = f"{subject}_ses-01_task-{args.condition}"
        input_path = eeg_dir / f"{base}_desc-ica-input_raw.fif"
        ica_path = eeg_dir / f"{base}_desc-amica_ica.fif"
        components_path = eeg_dir / f"{base}_desc-amica_components.tsv"
        if not input_path.exists():
            raise FileNotFoundError(f"Missing AMICA input: {input_path}")
        raw = load_condition_input(
            input_path,
            args.input_stage,
            ica_path=ica_path if args.input_stage == "cleaned" else None,
            components_path=components_path if args.input_stage == "cleaned" else None,
        )
        if args.resample_sfreq and raw.info["sfreq"] != args.resample_sfreq:
            raw.resample(args.resample_sfreq, verbose="error")
        subjects[subject] = raw
        input_paths[subject] = input_path
    grand, report, subject_tfrs = run_grand_average(subjects, config, logger)
    paths = write_erd_ers_outputs(
        args.output_path,
        args.condition,
        grand,
        report,
        subject_tfrs,
        config,
        input_paths,
        args.overwrite,
    )
    logger.info("Saved ERD/ERS outputs: %s", paths["figure"])


if __name__ == "__main__":
    main(parse_args())
