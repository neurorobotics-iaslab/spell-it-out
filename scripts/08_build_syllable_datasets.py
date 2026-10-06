"""Validate or build BIDS-derived HDF5 syllable datasets.

Examples
--------
python scripts/08_build_syllable_datasets.py validate
python scripts/08_build_syllable_datasets.py validate --subjects 01 --datasets overt_production
python scripts/08_build_syllable_datasets.py apply
python scripts/08_build_syllable_datasets.py apply --source raw --eeg-tmin -0.2 --eeg-tmax 0.8
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from spellitout.syllable_dataset import (
    BuildPaths,
    build_syllable_datasets,
    load_config,
    resolve_config,
)


DEFAULT_BIDS_ROOT = Path("data/spellitout")
DEFAULT_CONFIG = Path("configs/syllable_dataset_default.json")
DEFAULT_PHONETICS = Path("spellitout/spellitout/resources/phonetic_categories.tsv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("validate", "apply"), nargs="?", default="validate")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--bids-root", type=Path, default=DEFAULT_BIDS_ROOT)
    parser.add_argument(
        "--preprocessed-root", type=Path, default=None,
        help="Default: <bids-root>/derivatives/preprocessing.",
    )
    parser.add_argument(
        "--output-path", type=Path, default=None,
        help="Default: <bids-root>/derivatives/syllable-datasets.",
    )
    parser.add_argument("--phonetic-categories", type=Path, default=DEFAULT_PHONETICS)
    parser.add_argument("--source", choices=("raw", "preprocessed"), default=None)
    parser.add_argument("--subjects", nargs="+", default=None)
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--eeg-tmin", type=float, default=None)
    parser.add_argument("--eeg-tmax", type=float, default=None)
    parser.add_argument(
        "--exclude-bad-overlap", action=argparse.BooleanOptionalAction, default=None,
        help="Reject EEG windows overlapping BAD_* spans.",
    )
    parser.add_argument(
        "--bad-annotation-source", choices=("auto", "source", "preprocessed", "none"),
        default=None,
        help="auto: reviewed derivative events for raw EEG, source events for preprocessed EEG.",
    )
    parser.add_argument(
        "--filter-phase", choices=("zero", "causal"), default=None,
        help="Band-pass/notch mode: zero-phase FIR or forward-only Butterworth IIR. Resampling is unchanged.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logger = logging.getLogger("spellitout.syllable_dataset")
    bids_root = args.bids_root.expanduser().resolve()
    preprocessed_root = (
        args.preprocessed_root.expanduser().resolve()
        if args.preprocessed_root is not None
        else bids_root / "derivatives" / "preprocessing"
    )
    output_root = (
        args.output_path.expanduser().resolve()
        if args.output_path is not None
        else bids_root / "derivatives" / "syllable-datasets"
    )
    config = resolve_config(
        load_config(args.config), source=args.source,
        eeg_tmin=args.eeg_tmin, eeg_tmax=args.eeg_tmax,
        exclude_bad_overlap=args.exclude_bad_overlap,
        bad_annotation_source=args.bad_annotation_source,
        filter_phase=args.filter_phase,
    )
    source_root = bids_root if config["source"] == "raw" else preprocessed_root
    paths = BuildPaths(
        bids_root=bids_root, source_root=source_root,
        preprocessed_root=preprocessed_root, output_root=output_root,
    )
    logger.info("%s source: %s", config["source"].capitalize(), source_root)
    result = build_syllable_datasets(
        paths, config, args.phonetic_categories,
        mode=args.mode, subjects=args.subjects,
        selected_datasets=args.datasets, overwrite=args.overwrite,
    )
    for name, counts in result["counts"].items():
        logger.info(
            "%s: %d candidates, %d accepted, %d excluded.",
            name, counts["candidates"], counts["accepted"], counts["excluded"],
        )
    logger.info(
        "Missing production annotations recorded separately: %d.",
        result["missing_production_annotations"],
    )
    if args.mode == "validate":
        logger.info("Read-only validation complete. Planned output: %s", output_root)
    else:
        logger.info("HDF5 dataset written to %s", output_root)
    logger.info("Resolved configuration SHA-256: %s", result["config_sha256"])


if __name__ == "__main__":
    main(parse_args())
