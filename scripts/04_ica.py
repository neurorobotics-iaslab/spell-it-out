"""Build reviewed condition-level ICA input and fit Extended Infomax."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import mne

from spellitout.ica import (
    InfomaxConfig,
    fit_extended_infomax,
    prepare_ica_input,
    reviewed_run_decisions,
    write_infomax_outputs,
)
from spellitout.preprocessing import register_derivative_description
from spellitout.utils import get_run_id, get_subject_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", choices=("overt", "silent"), required=True)
    parser.add_argument("--subjects", type=int, nargs="+", required=True)
    parser.add_argument("--runs", type=int, nargs="+", default=(1, 2, 3))
    parser.add_argument(
        "--derivatives-path", type=Path, default=Path("derivatives/preprocessing")
    )
    parser.add_argument("--line-noise-method", choices=("notch", "prep"), default="notch")
    parser.add_argument("--line-frequency", type=float, default=50.0)
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--random-state", type=int, default=435656)
    parser.add_argument(
        "--rank-tolerance",
        type=float,
        default=1e-7,
        help="Relative singular-value tolerance used to estimate effective EEG rank.",
    )
    parser.add_argument(
        "--max-unmixing-condition",
        type=float,
        default=1e6,
        help="Reject decompositions with a larger unmixing condition number.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    logger = logging.getLogger("spellitout.infomax")
    config = InfomaxConfig(
        line_noise_method=args.line_noise_method,
        line_frequency_hz=args.line_frequency,
        max_iter=args.max_iter,
        random_state=args.random_state,
        rank_tolerance=args.rank_tolerance,
        max_unmixing_condition=args.max_unmixing_condition,
    )
    register_derivative_description(
        args.derivatives_path,
        "extended-infomax",
        "Condition-level reviewed ICA input and Extended Infomax decomposition; "
        "component decisions remain unreviewed.",
    )
    for subject_number in args.subjects:
        subject = get_subject_id(subject_number)
        eeg_dir = args.derivatives_path / subject / "ses-01" / "eeg"
        raws, annotation_tables, channel_tables = {}, {}, {}
        run_names = []
        for run_number in args.runs:
            base = get_run_id(subject, args.condition, run_number)
            raw_path = eeg_dir / f"{base}_desc-minproc_raw.fif"
            annotations_path = eeg_dir / f"{base}_desc-qc_annotations.tsv"
            channels_path = eeg_dir / f"{base}_desc-channelqc.tsv"
            for path in (raw_path, annotations_path, channels_path):
                if not path.exists():
                    raise FileNotFoundError(
                        f"Missing {path}; complete Stages 1–3 before Extended Infomax."
                    )
            run = f"run-{run_number:02d}"
            annotation_tables[run], channel_tables[run] = reviewed_run_decisions(
                annotations_path, channels_path, run
            )
            raws[run] = mne.io.read_raw_fif(raw_path, preload=True, verbose="error")
            run_names.append(run)

        pooled, interpolated, excluded, plan = prepare_ica_input(
            raws, annotation_tables, channel_tables, config, logger
        )
        base = f"{subject}_ses-01_task-{args.condition}"
        expected = [
            eeg_dir / f"{base}_desc-infomax-input_raw.fif",
            eeg_dir / f"{base}_desc-infomax-input_channels.tsv",
            eeg_dir / f"{base}_desc-infomax_ica.fif",
            eeg_dir / f"{base}_desc-infomax_components.tsv",
            eeg_dir / f"{base}_desc-infomax.json",
        ]
        if not args.overwrite and any(path.exists() for path in expected):
            raise FileExistsError(
                f"Extended Infomax output exists for {base}; pass --overwrite to replace it."
            )

        # Preserve the prepared input before the potentially long optimization.
        input_path = eeg_dir / f"{base}_desc-infomax-input_raw.fif"
        pooled.save(input_path, overwrite=args.overwrite)
        ica, n_components, rank_estimate, n_fit_samples, unmixing_condition = (
            fit_extended_infomax(pooled, config)
        )
        paths = write_infomax_outputs(
            eeg_dir,
            base,
            pooled,
            ica,
            plan,
            config,
            run_names,
            interpolated,
            excluded,
            rank_estimate,
            n_fit_samples,
            unmixing_condition,
            overwrite=args.overwrite,
            write_ica_input=False,
        )
        logger.info(
            "Extended Infomax complete for %s: %d components, rank %d, condition %.3e; %s",
            subject,
            n_components,
            rank_estimate,
            unmixing_condition,
            paths["ica"],
        )


if __name__ == "__main__":
    main(parse_args())
