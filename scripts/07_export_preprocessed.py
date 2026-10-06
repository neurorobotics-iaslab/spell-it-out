"""Validate or export reviewed preprocessed EEG as a BIDS derivative.

The default ``validate`` mode is read-only. ``apply`` writes a publication
derivative below the raw BIDS dataset unless ``--output-path`` is supplied.

Examples
--------
python scripts/07_export_preprocessed.py validate
python scripts/07_export_preprocessed.py validate --subjects 01 --tasks covert --runs 01
python scripts/07_export_preprocessed.py apply
python scripts/07_export_preprocessed.py apply --overwrite
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from spellitout.publishing import (
    discover_export_records,
    export_preprocessed_dataset,
    inspect_records,
)


DEFAULT_BIDS_ROOT = Path("data/spellitout")
DEFAULT_WORKING_DERIVATIVES = Path("derivatives/preprocessing")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("validate", "apply"), nargs="?", default="validate")
    parser.add_argument("--bids-root", type=Path, default=DEFAULT_BIDS_ROOT)
    parser.add_argument(
        "--working-derivatives",
        type=Path,
        default=DEFAULT_WORKING_DERIVATIVES,
        help="Current FIF/checkpoint directory (default: derivatives/preprocessing).",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help=(
            "Derivative dataset root. By default writes to "
            "<bids-root>/derivatives/preprocessing."
        ),
    )
    parser.add_argument("--subjects", nargs="+", default=None)
    parser.add_argument("--tasks", nargs="+", choices=("overt", "covert"), default=None)
    parser.add_argument("--runs", nargs="+", default=None)
    parser.add_argument("--pipeline-version", default=None)
    parser.add_argument("--code-url", default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logger = logging.getLogger("spellitout.export_preprocessed")
    bids_root = args.bids_root.expanduser().resolve()
    working = args.working_derivatives.expanduser().resolve()
    output = (
        args.output_path.expanduser().resolve()
        if args.output_path is not None
        else bids_root / "derivatives" / "preprocessing"
    )
    if not (bids_root / "dataset_description.json").is_file():
        raise FileNotFoundError(f"BIDS dataset root not found: {bids_root}")
    if not working.is_dir():
        raise FileNotFoundError(f"Working derivatives not found: {working}")
    if output.resolve() == working.resolve():
        raise ValueError(
            "Output must differ from the working derivatives directory; the export "
            "never rewrites the FIF/checkpoint tree in place."
        )

    records = discover_export_records(
        bids_root,
        working,
        subjects=args.subjects,
        tasks=args.tasks,
        runs=args.runs,
    )
    logger.info("Validating %d preprocessed run(s).", len(records))
    inspections = inspect_records(records)
    tasks = sorted({inspection.record.task for inspection in inspections})
    subjects = sorted({inspection.record.subject for inspection in inspections})
    total_seconds = sum(
        inspection.n_times / inspection.sfreq for inspection in inspections
    )
    logger.info(
        "Validation passed: %d subject(s), task(s) %s, %.2f recording hours.",
        len(subjects),
        ", ".join(tasks),
        total_seconds / 3600,
    )
    if args.mode == "validate":
        logger.info("Read-only validation complete. Planned output: %s", output)
        return

    export_preprocessed_dataset(
        inspections,
        bids_root,
        output,
        overwrite=args.overwrite,
        pipeline_version=args.pipeline_version,
        code_url=args.code_url,
    )
    logger.info("BIDS preprocessing derivative exported to %s", output)


if __name__ == "__main__":
    main(parse_args())
