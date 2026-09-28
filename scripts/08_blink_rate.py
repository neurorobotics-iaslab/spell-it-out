"""Compute reviewed spontaneous blink rates by task, run, and canonical phase."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from spellitout.blink_rate import (
    BLINKRATE_COLUMNS,
    PHASES,
    run_blink_rate_rows,
    write_blinkrate_sidecar,
)
from spellitout.preprocessing import ensure_derivative_scaffold
from spellitout.review_qc import discover_review_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=f"Output phases: {', '.join(PHASES)}.",
    )
    parser.add_argument("--conditions", nargs="+", choices=("silent", "overt"), default=("silent", "overt"))
    parser.add_argument(
        "--subjects",
        type=int,
        nargs="+",
        default=None,
        help="Subject numbers to process (default: every subject found in derivatives-path).",
    )
    parser.add_argument("--runs", type=int, nargs="+", default=(1, 2, 3))
    parser.add_argument("--derivatives-path", type=Path, default=Path("derivatives/preprocessing"))
    parser.add_argument("--output-path", type=Path, default=Path("derivatives/blinkrate"))
    parser.add_argument(
        "--exclude-bad-spans",
        action="store_true",
        help="Remove approved BAD_* spans from phase duration and blink counting.",
    )
    parser.add_argument(
        "--min-blink-separation-ms",
        type=float,
        default=150.0,
        help="Minimum centre-to-centre distance for separate blinks (default: 150 ms).",
    )
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    if args.min_blink_separation_ms < 0:
        raise ValueError("--min-blink-separation-ms must be non-negative.")
    subjects = args.subjects
    if subjects is None:
        subjects = sorted(
            int(path.name.removeprefix("sub-"))
            for path in args.derivatives_path.glob("sub-*")
            if path.is_dir() and path.name.removeprefix("sub-").isdigit()
        )
        if not subjects:
            raise FileNotFoundError(
                f"No subject directories found below {args.derivatives_path}."
            )
    rows: list[dict[str, object]] = []
    logger = logging.getLogger("spellitout.blink_rate")
    for condition in args.conditions:
        for subject in subjects:
            try:
                records = discover_review_records(
                    args.derivatives_path,
                    condition,
                    [subject],
                    args.runs,
                )
            except FileNotFoundError as error:
                logger.warning(
                    "Skipping sub-%02d task %s: %s",
                    subject,
                    condition,
                    error,
                )
                continue
            for record in records:
                rows.extend(
                    run_blink_rate_rows(
                        record,
                        exclude_bad_spans=args.exclude_bad_spans,
                        minimum_blink_separation_seconds=args.min_blink_separation_ms / 1000.0,
                    )
                )

    ensure_derivative_scaffold(args.output_path)
    write_blinkrate_sidecar(args.output_path)
    table = pd.DataFrame(rows, columns=BLINKRATE_COLUMNS)
    for (subject, session), group in table.groupby(["subject", "session"], sort=True):
        output_dir = args.output_path / subject / session / "eeg"
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"{subject}_{session}_desc-blinkrate.tsv"
        group.to_csv(path, sep="\t", index=False)
        print(f"Wrote {path}")


if __name__ == "__main__":
    main(parse_args())
