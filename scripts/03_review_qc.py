"""Review automatic artifact candidates and PREP channel evidence in Tk.

The application is the only human decision gate before ICA preparation.  It
autosaves review decisions to the existing ``desc-qc_annotations.tsv`` and
``desc-channelqc.tsv`` files; it never changes a raw FIF file.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from spellitout.preprocessing import register_derivative_description
from spellitout.review_qc import QCReviewApplication, discover_review_records


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
    parser.add_argument("--reviewer", default="")
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    import tkinter as tk

    records = discover_review_records(
        args.derivatives_path,
        args.condition,
        args.subjects,
        args.runs,
    )
    register_derivative_description(
        args.derivatives_path,
        "reviewqc",
        (
            "Human review of automatic artifact candidates and run-level PREP "
            "channel evidence. Decisions are stored in desc-qc_annotations.tsv "
            "and desc-channelqc.tsv; no additional signal derivative is written."
        ),
    )
    root = tk.Tk()
    QCReviewApplication(root, records, reviewer=args.reviewer)
    root.mainloop()


if __name__ == "__main__":
    main(parse_args())
