"""Apply reviewed condition-level ICA decisions and save cleaned run-level raws."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import mne
import pandas as pd

from spellitout.preprocessing import register_derivative_description
from spellitout.utils import get_run_id, get_subject_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", choices=("overt", "silent"), required=True)
    parser.add_argument("--subjects", type=int, nargs="+", required=True)
    parser.add_argument(
        "--runs",
        type=int,
        nargs="+",
        default=(1, 2, 3),
        help="Run numbers, in concatenation order (default: 1 2 3).",
    )
    parser.add_argument(
        "--decomposition",
        choices=("infomax", "amica"),
        default="infomax",
        help="Reviewed decomposition to apply (default: infomax).",
    )
    parser.add_argument(
        "--derivatives-path",
        type=Path,
        default=Path("derivatives/preprocessing"),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def reviewed_exclusions(table: pd.DataFrame, n_components: int) -> list[int]:
    """Validate a component-review table and return components marked reject."""
    required = {"component", "final_decision"}
    missing_columns = required.difference(table.columns)
    if missing_columns:
        raise ValueError(
            "ICA component table lacks required column(s): "
            + ", ".join(sorted(missing_columns))
        )

    components = pd.to_numeric(table["component"], errors="raise")
    if not components.map(lambda value: float(value).is_integer()).all():
        raise ValueError("ICA component identifiers must be integers.")
    components = components.astype(int)
    if components.duplicated().any():
        duplicates = sorted(components[components.duplicated()].unique())
        raise ValueError(f"Duplicate ICA component row(s): {duplicates}")

    expected = set(range(n_components))
    observed = set(components)
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        details = []
        if missing:
            details.append(f"missing {missing}")
        if extra:
            details.append(f"unexpected {extra}")
        raise ValueError("ICA component table does not match the decomposition: " + "; ".join(details))

    decisions = table["final_decision"].astype(str).str.strip().str.lower()
    unreviewed = components[decisions.eq("unreviewed")].tolist()
    if unreviewed:
        raise RuntimeError(
            f"Cannot apply ICA: {len(unreviewed)} component decision(s) remain "
            f"unreviewed: {unreviewed}"
        )
    invalid = sorted(set(decisions) - {"keep", "reject"})
    if invalid:
        raise ValueError(
            "Invalid final_decision value(s): "
            + ", ".join(repr(value) for value in invalid)
        )
    return sorted(components[decisions.eq("reject")].tolist())


def input_paths(
    eeg_dir: Path,
    base: str,
    decomposition: str,
) -> tuple[Path, Path, Path]:
    """Resolve the method-specific Stage 4/5 inputs."""
    input_desc = "infomax-input" if decomposition == "infomax" else "ica-input"
    raw_path = eeg_dir / f"{base}_desc-{input_desc}_raw.fif"
    ica_path = eeg_dir / f"{base}_desc-{decomposition}_ica.fif"
    components_path = eeg_dir / f"{base}_desc-{decomposition}_components.tsv"
    return raw_path, ica_path, components_path


def split_condition_input(raw: mne.io.BaseRaw) -> list[mne.io.BaseRaw]:
    """Split an MNE-concatenated raw at technical boundary annotations."""
    sfreq = float(raw.info["sfreq"])
    boundary_onsets = sorted(
        {
            float(onset)
            for onset, description in zip(
                raw.annotations.onset, raw.annotations.description
            )
            if str(description).strip().lower() == "bad boundary"
        }
    )
    boundary_samples = [
        int(raw.time_as_index(onset, use_rounding=True)[0])
        for onset in boundary_onsets
    ]
    invalid = [sample for sample in boundary_samples if sample <= 0 or sample >= raw.n_times]
    if invalid:
        raise RuntimeError(f"Invalid concatenation boundary sample(s): {invalid}")
    if boundary_samples != sorted(set(boundary_samples)):
        raise RuntimeError("Concatenation boundary annotations are not unique and ordered.")

    starts = [0, *boundary_samples]
    stops = [*boundary_samples, raw.n_times]
    runs = []
    for start, stop in zip(starts, stops):
        if stop <= start:
            raise RuntimeError(f"Empty run between concatenated samples {start} and {stop}.")
        run = mne.io.RawArray(
            raw.get_data(start=start, stop=stop),
            raw.info.copy(),
            first_samp=0,
            verbose="error",
        )
        segment_start = start / sfreq
        segment_stop = stop / sfreq
        last_sample_time = segment_start + (stop - start - 1) / sfreq
        onsets, durations, descriptions = [], [], []
        for onset, duration, description in zip(
            raw.annotations.onset,
            raw.annotations.duration,
            raw.annotations.description,
        ):
            if str(description).strip().lower() in {"bad boundary", "edge boundary"}:
                continue
            annotation_start = float(onset) - float(raw.first_time)
            annotation_stop = annotation_start + float(duration)
            if duration == 0:
                if not segment_start <= annotation_start < segment_stop:
                    continue
                clipped_start = annotation_start
                clipped_stop = annotation_start
            else:
                if annotation_stop <= segment_start or annotation_start >= segment_stop:
                    continue
                clipped_start = max(annotation_start, segment_start)
                clipped_stop = min(annotation_stop, last_sample_time)
                if clipped_stop < clipped_start:
                    continue
            onsets.append(clipped_start - segment_start)
            durations.append(clipped_stop - clipped_start)
            descriptions.append(str(description))
        run.set_annotations(
            mne.Annotations(
                onset=onsets,
                duration=durations,
                description=descriptions,
                orig_time=run.info.get("meas_date"),
            )
        )
        runs.append(run)
    return runs


def main(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    logger = logging.getLogger("spellitout.apply_ica")
    register_derivative_description(
        args.derivatives_path,
        "ica-cleaned",
        "Run-level EEG after application of manually reviewed condition-level ICA decisions.",
    )

    for subject_number in args.subjects:
        subject = get_subject_id(subject_number)
        base = f"{subject}_ses-01_task-{args.condition}"
        eeg_dir = args.derivatives_path / subject / "ses-01" / "eeg"
        raw_path, ica_path, components_path = input_paths(
            eeg_dir, base, args.decomposition
        )
        for path in (raw_path, ica_path, components_path):
            if not path.exists():
                raise FileNotFoundError(f"Missing ICA application input: {path}")
        output_paths = [
            eeg_dir
            / f"{get_run_id(subject, args.condition, run_number)}_desc-{args.decomposition}-cleaned_raw.fif"
            for run_number in args.runs
        ]
        existing = [path for path in output_paths if path.exists()]
        if existing and not args.overwrite:
            raise FileExistsError(
                "Cleaned run-level output(s) exist; pass --overwrite to replace them: "
                + ", ".join(str(path) for path in existing)
            )

        raw = mne.io.read_raw_fif(raw_path, preload=True, verbose="error")
        ica = mne.preprocessing.read_ica(ica_path, verbose="error")
        table = pd.read_csv(components_path, sep="\t")
        excluded = reviewed_exclusions(table, int(ica.n_components_))

        missing_channels = sorted(set(ica.ch_names).difference(raw.ch_names))
        if missing_channels:
            raise RuntimeError(
                "ICA decomposition contains channels absent from its input raw: "
                + ", ".join(missing_channels)
            )

        runs = split_condition_input(raw)
        if len(runs) != len(args.runs):
            raise RuntimeError(
                f"The condition-level input contains {len(runs)} run segment(s), but "
                f"--runs specifies {len(args.runs)} value(s): {list(args.runs)}."
            )

        ica.exclude = excluded
        for run_number, run, output_path in zip(args.runs, runs, output_paths):
            cleaned = run.copy()
            ica.apply(cleaned, verbose="error")
            previous_description = cleaned.info.get("description") or ""
            summary = (
                f"Reviewed condition-level {args.decomposition} ICA applied; rejected "
                f"components: {excluded if excluded else 'none'}. Source decisions: "
                f"{components_path.name}."
            )
            cleaned.info["description"] = f"{previous_description} {summary}".strip()
            cleaned.save(output_path, overwrite=args.overwrite)
            logger.info(
                "Saved %s run-%02d after rejecting %d/%d component(s): %s",
                base,
                run_number,
                len(excluded),
                ica.n_components_,
                output_path,
            )


if __name__ == "__main__":
    main(parse_args())
