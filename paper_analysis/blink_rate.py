"""Compute and aggregate reviewed blink rates for the paper.

This replaces the former two-step ``08_blink_rate.py`` and ``blinkrate.py``
workflow. It reads the  BIDS preprocessing derivative and writes consolidated
analysis tables plus article-ready wide TSV tables below ``paper_analysis/results``.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from spellitout.blink_rate import (
    BLINKRATE_COLUMNS,
    PHASES,
    run_blink_rate_rows,
    write_blinkrate_sidecar,
)
from spellitout import SpellItOutDataset


DEFAULT_OUTPUT = Path("paper_analysis/results/blink_rate")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=("covert", "overt"),
        default=("covert", "overt"),
    )
    parser.add_argument(
        "--subjects",
        type=int,
        nargs="+",
        default=None,
        help="Subjects to process; default: all subjects in derivatives-path.",
    )
    parser.add_argument("--runs", type=int, nargs="+", default=(1, 2, 3))
    parser.add_argument(
        "--derivatives-path",
        type=Path,
        default=Path("data/spellitout/derivatives/preprocessing"),
        help="Published BIDS preprocessing derivative root (not its qc/ subdirectory).",
    )
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT)
    bad_span_policy = parser.add_mutually_exclusive_group()
    bad_span_policy.add_argument(
        "--exclude-bad-spans",
        dest="exclude_bad_spans",
        action="store_true",
        help=(
            "Remove approved BAD_* spans from phase duration and blink counting "
            "(default)."
        ),
    )
    bad_span_policy.add_argument(
        "--include-bad-spans",
        dest="exclude_bad_spans",
        action="store_false",
        help="Keep approved BAD_* spans in phase duration and blink counting.",
    )
    parser.set_defaults(exclude_bad_spans=True)
    parser.add_argument(
        "--min-blink-separation-ms",
        type=float,
        default=350.0,
        help=(
            "Minimum centre-to-centre distance for separate blinks "
            "(default: 350 ms, matching the paper analysis)."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _subjects(args: argparse.Namespace) -> list[int]:
    if args.subjects is not None:
        return sorted(set(args.subjects))
    subjects = [int(subject) for subject in SpellItOutDataset(args.derivatives_path).subjects]
    if not subjects:
        raise FileNotFoundError(
            f"No subject directories found below {args.derivatives_path}."
        )
    return subjects


def compute_run_table(args: argparse.Namespace) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    logger = logging.getLogger("spellitout.paper.blink_rate")
    dataset = SpellItOutDataset(args.derivatives_path)
    if dataset.dataset_type != "derivative":
        raise ValueError("--derivatives-path must point to the final preprocessing derivative.")
    subjects = _subjects(args)
    for condition in args.conditions:
        for subject in subjects:
            try:
                records = list(dataset.iter_runs(
                    subjects=subject, tasks=condition, runs=args.runs, preload=False,
                ))
            except FileNotFoundError as error:
                logger.warning(
                    "Skipping sub-%02d task %s: %s", subject, condition, error
                )
                continue
            for record in records:
                base = (
                    f"{record.subject_id}_ses-{record.session}_task-{record.task}"
                    f"_run-{record.run}"
                )
                qc_path = (
                    dataset.root / "qc" / record.subject_id / f"ses-{record.session}"
                    / "eeg" / f"{base}_desc-annotationsqc.tsv"
                )
                annotations_qc = (
                    pd.read_csv(qc_path, sep="\t", keep_default_na=False)
                    if qc_path.is_file() else None
                )
                try:
                    rows.extend(
                        run_blink_rate_rows(
                            record,
                            annotations_qc=annotations_qc,
                            exclude_bad_spans=args.exclude_bad_spans,
                            minimum_blink_separation_seconds=(
                                args.min_blink_separation_ms / 1000.0
                            ),
                        )
                    )
                finally:
                    record.raw.close()
    if not rows:
        raise RuntimeError("No blink-rate rows were generated.")
    return pd.DataFrame(rows, columns=BLINKRATE_COLUMNS)


def summarise_subject_phase(run_table: pd.DataFrame) -> pd.DataFrame:
    grouped = run_table.groupby(
        ["subject", "task", "phase"], sort=True, dropna=False
    )
    summary = grouped.agg(
        mean_blink_rate_per_min=("blink_rate_per_min", "mean"),
        median_blink_rate_per_min=("blink_rate_per_min", "median"),
        std_blink_rate_per_min=("blink_rate_per_min", "std"),
        min_blink_rate_per_min=("blink_rate_per_min", "min"),
        max_blink_rate_per_min=("blink_rate_per_min", "max"),
        n_runs=("run", "nunique"),
        n_complete_runs=("review_complete", "sum"),
        total_blinks=("blink_count", "sum"),
        total_valid_duration_s=("valid_duration_s", "sum"),
    ).reset_index()
    summary["n_incomplete_runs"] = (
        summary["n_runs"] - summary["n_complete_runs"]
    )
    summary["weighted_blink_rate_per_min"] = (
        summary["total_blinks"] / summary["total_valid_duration_s"] * 60.0
    ).where(summary["total_valid_duration_s"] > 0)
    return summary


def article_table(summary: pd.DataFrame, condition: str) -> pd.DataFrame:
    """Return the displayed subject-by-phase table for one task."""
    table = summary.loc[summary["task"] == condition].pivot(
        index="subject", columns="phase", values="mean_blink_rate_per_min"
    )
    return table.reindex(columns=[phase for phase in PHASES if phase in table])


def _print_tables(
    tables: dict[str, pd.DataFrame],
) -> None:
    for condition, table in tables.items():
        print(f"\nMean blink rate per minute — {condition}")
        print(
            table.to_string(float_format=lambda value: f"{value:.1f}")
            if not table.empty
            else "No rows."
        )


def main(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    if args.min_blink_separation_ms < 0:
        raise ValueError("--min-blink-separation-ms must be non-negative.")
    outputs = {
        "runs": args.output_path / "run_level.tsv",
        "summary": args.output_path / "subject_phase.tsv",
        "method": args.output_path / "method.json",
    }
    article_outputs = {
        condition: args.output_path / f"mean_blink_rate_task-{condition}.tsv"
        for condition in args.conditions
    }
    existing = [
        path
        for path in (*outputs.values(), *article_outputs.values())
        if path.exists()
    ]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Blink-rate output already exists; pass --overwrite: "
            + ", ".join(str(path) for path in existing)
        )
    run_table = compute_run_table(args)
    summary = summarise_subject_phase(run_table)
    article_tables = {
        condition: article_table(summary, condition)
        for condition in args.conditions
    }
    args.output_path.mkdir(parents=True, exist_ok=True)
    run_table.to_csv(outputs["runs"], sep="\t", index=False)
    summary.to_csv(outputs["summary"], sep="\t", index=False)
    for condition, table in article_tables.items():
        table.to_csv(
            article_outputs[condition],
            sep="\t",
            index=True,
            float_format="%.1f",
        )
    method_path = write_blinkrate_sidecar(args.output_path, "method.json")
    method = json.loads(method_path.read_text(encoding="utf-8"))
    method["Parameters"] = {
        "conditions": list(args.conditions),
        "subjects": _subjects(args),
        "runs": list(args.runs),
        "exclude_bad_spans": args.exclude_bad_spans,
        "minimum_blink_separation_ms": args.min_blink_separation_ms,
        "preprocessed_bids_root": str(args.derivatives_path.resolve()),
    }
    method["ArticleTables"] = {
        "Description": (
            "Mean blink rate per minute across runs, by subject and phase."
        ),
        "Rounding": "One decimal place for article reporting.",
        "Files": [path.name for path in article_outputs.values()],
    }
    method_path.write_text(json.dumps(method, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {outputs['runs']}")
    print(f"Saved {outputs['summary']}")
    print(f"Saved {outputs['method']}")
    for path in article_outputs.values():
        print(f"Saved {path}")
    _print_tables(article_tables)


if __name__ == "__main__":
    main(parse_args())
