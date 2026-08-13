"""Build reviewed condition-level ICA input and fit AMICA in one invocation."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import mne

from spellitout.ica import (
    AMICAConfig,
    fit_amica,
    prepare_amica_input,
    reviewed_run_decisions,
    write_amica_outputs,
)
from spellitout.preprocessing import register_derivative_description
from spellitout.utils import get_run_id, get_subject_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", choices=("overt", "silent"), required=True)
    parser.add_argument("--subjects", type=int, nargs="+", required=True)
    parser.add_argument("--runs", type=int, nargs="+", default=(1, 2, 3))
    parser.add_argument("--derivatives-path", type=Path, default=Path("derivatives/preprocessing"))
    parser.add_argument("--line-noise-method", choices=("notch", "prep"), default="notch")
    parser.add_argument("--line-frequency", type=float, default=50.0)
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    logger = logging.getLogger("spellitout.amica")
    config = AMICAConfig(
        line_noise_method=args.line_noise_method,
        line_frequency_hz=args.line_frequency,
        max_iter=args.max_iter,
        device=args.device,
    )
    register_derivative_description(
        args.derivatives_path,
        "amica",
        "Condition-level reviewed ICA input and AMICA decomposition; component decisions remain unreviewed.",
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
                    raise FileNotFoundError(f"Missing {path}; complete Stages 1–3 before AMICA.")
            run = f"run-{run_number:02d}"
            annotation_tables[run], channel_tables[run] = reviewed_run_decisions(
                annotations_path, channels_path, run
            )
            raws[run] = mne.io.read_raw_fif(raw_path, preload=True, verbose="error")
            run_names.append(run)
        pooled, interpolated, excluded, plan = prepare_amica_input(
            raws, annotation_tables, channel_tables, config, logger
        )
        base = f"{subject}_ses-01_task-{args.condition}"
        expected = [
            eeg_dir / f"{base}_desc-ica-input_raw.fif",
            eeg_dir / f"{base}_desc-ica-input_channels.tsv",
            eeg_dir / f"{base}_desc-amica_ica.fif",
            eeg_dir / f"{base}_desc-amica_components.tsv",
            eeg_dir / f"{base}_desc-amica.json",
        ]
        if not args.overwrite and any(path.exists() for path in expected):
            raise FileExistsError(f"AMICA output exists for {base}; pass --overwrite to replace it.")
        # Persist the repaired/referenced checkpoint before the potentially long fit.
        input_path = eeg_dir / f"{base}_desc-ica-input_raw.fif"
        pooled.save(input_path, overwrite=args.overwrite)
        model, ica, n_components, rank_estimate, n_fit_samples = fit_amica(pooled, config)
        paths = write_amica_outputs(
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
            overwrite=args.overwrite,
            write_ica_input=False,
        )
        logger.info("AMICA complete for %s: %s", subject, paths["ica"])
        del model


if __name__ == "__main__":
    main(parse_args())
