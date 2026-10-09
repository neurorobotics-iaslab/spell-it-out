"""Run per-recording PREP channel QC without interpolating channels.

This stage reads the ``01_preprocess`` outputs and produces the canonical,
editable channel-review table. It does not alter the EEG derivative and does
not make final channel decisions.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import mne
from mne_bids import BIDSPath

from reproducibility.processing.preprocessing import (
    PrepQCConfig,
    register_derivative_description,
    run_prep_channel_qc,
    write_prep_qc,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", choices=("overt", "covert"), required=True)
    parser.add_argument("--subjects", type=int, nargs="+", required=True)
    parser.add_argument("--runs", type=int, nargs="+", default=(1, 2, 3))
    parser.add_argument(
        "--derivatives-path",
        type=Path,
        default=Path("derivatives/preprocessing"),
    )
    parser.add_argument("--line-frequency", type=float, default=50.0)
    parser.add_argument("--random-state", type=int, default=435656)
    parser.add_argument("--no-ransac", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    logger = logging.getLogger("spellitout.channel_qc")
    register_derivative_description(
        args.derivatives_path,
        "prepqc",
        (
            "Run-level PREP channel-quality evidence; no final PREP interpolation. "
            "The accompanying channel-QC table requires review before harmonization."
        ),
    )
    config = PrepQCConfig(
        line_frequency_hz=args.line_frequency,
        random_state=args.random_state,
        ransac=not args.no_ransac,
    )
    failed_runs = []
    for subject_number in args.subjects:
        subject = f"sub-{subject_number:02d}"
        output_dir = args.derivatives_path / subject / "ses-01" / "eeg"
        for run_number in args.runs:
            base = BIDSPath(subject=subject.removeprefix("sub-"), session="01", task=args.condition, run=f"{run_number:02d}", suffix="eeg").basename.removesuffix("_eeg")
            minproc_path = output_dir / f"{base}_desc-minproc_raw.fif"
            if not minproc_path.exists():
                raise FileNotFoundError(
                    f"Missing {minproc_path}. Run 01_preprocess.py before 02_channel_qc.py."
                )
            logger.info("PREP channel QC: %s", base)
            raw = mne.io.read_raw_fif(minproc_path, preload=True, verbose="error")
            report = run_prep_channel_qc(raw, config, logger)
            paths = write_prep_qc(
                raw,
                report,
                output_dir,
                base,
                minproc_path,
                overwrite=args.overwrite,
            )
            logger.info("Saved %s", paths["channels"])
            if report["status"] != "completed":
                failed_runs.append(base)
    if failed_runs:
        raise RuntimeError(
            "PREP channel QC failed for " + ", ".join(failed_runs)
            + ". Failure reports were written; inspect them before continuing."
        )


if __name__ == "__main__":
    main(parse_args())
