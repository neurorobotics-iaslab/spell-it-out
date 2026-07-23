"""Prepare continuous EEG and fit AMICA separately for each condition.

This script intentionally does *not* use AutoReject.  AutoReject repairs
individual epochs, which makes its interpolated output inappropriate as AMICA
input.  AMICA is instead fitted on continuous, PREP-cleaned data pooled across
the three runs of one subject *within one condition*.  Existing ``BAD_*``
annotations are omitted from the fit without altering the retained samples.

The output is a BIDS-derivative-like directory containing one preprocessed
continuous file per run, one condition-level AMICA solution, its ICLabel
probabilities, and a JSON record of all choices made here.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import torch
from amica import AMICA
from mne_icalabel.iclabel import iclabel_label_components
from pyprep.prep_pipeline import PrepPipeline

sys.path.extend([".", ".."]) 

from spellitout.spellitout import SpellItOut
from spellitout.utils import get_run_id

DROP_CHANNELS = ("AF7", "AF3", "AF4", "AF8")
SFREQ = 512.0
LINE_FREQ = 50.0
ICA_FILTER = (1.0, 100.0)


def setup_logger(log_path: Path) -> logging.Logger:
    """Create an independent, reproducible per-subject/condition log."""
    logger = logging.getLogger(f"amica.{log_path.parent.name}.{log_path.parent.parent.name}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
    for handler in (logging.FileHandler(log_path, mode="w"), logging.StreamHandler()):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def run_prep(raw: mne.io.BaseRaw, logger: logging.Logger) -> tuple[mne.io.BaseRaw, list[str]]:
    """Run PREP once; PREP's line-noise removal is the only notch operation."""
    prep_params = {
        "ref_chs": "eeg",
        "reref_chs": "eeg",
        "line_freqs": np.arange(LINE_FREQ, raw.info["sfreq"] / 2, LINE_FREQ),
    }
    prep = PrepPipeline(raw, prep_params, raw.get_montage())
    prep.fit()
    interpolated = list(prep.interpolated_channels)
    logger.info("PREP interpolated channels: %s", interpolated or "none")
    return prep.raw, interpolated


def preprocess_run(raw: mne.io.BaseRaw, logger: logging.Logger) -> tuple[mne.io.BaseRaw, dict[str, list[str]]]:
    """Return a continuous 1--100 Hz, 512 Hz, PREP-cleaned ICA input."""
    raw = raw.copy()
    to_drop = [channel for channel in DROP_CHANNELS if channel in raw.ch_names]
    if to_drop:
        raw.drop_channels(to_drop)

    # Resampling performs anti-alias filtering before reducing the data rate.
    raw.resample(SFREQ, verbose="error")

    # PREP is run on the continuous data.  Its ``line_freqs`` handles 50/100 Hz
    # line noise, so do not apply an additional notch filter here.
    raw.filter(l_freq=ICA_FILTER[0], h_freq=ICA_FILTER[1], picks="eeg", verbose="error")
    raw, prep_interpolated = run_prep(raw, logger)
    # PyPREP's final ``interpolated_channels`` list does not necessarily cover
    # every channel retained in MNE's ``info['bads']`` after its iterative
    # robust-reference procedure.  Make the output unambiguous: explicitly
    # repair every residual flag, then reset it before run concatenation.
    residual_bads = list(raw.info["bads"])
    final_interpolated = []
    if residual_bads:
        logger.info("Final interpolation of residual PREP flags: %s", residual_bads)
        raw.interpolate_bads(reset_bads=True, verbose="error")
        final_interpolated = residual_bads
    if raw.info["bads"]:
        raise RuntimeError("Residual bad-channel flags remain after final interpolation.")
    raw.info["description"] = (
        "Resampled to 512 Hz; 1-100 Hz filtered; PREP line-noise removal, "
        "bad-channel interpolation, and robust reference."
    )
    return raw, {
        "prep_reported_interpolated": prep_interpolated,
        "residual_bad_flags": residual_bads,
        "final_interpolated": final_interpolated,
    }


def fit_amica(raw: mne.io.BaseRaw, n_components: int, max_iter: int, device: str):
    """Fit AMICA to EEG channels while omitting explicitly BAD_* spans."""
    data = raw.get_data(units="uV", reject_by_annotation="omit").T
    if data.size == 0:
        raise RuntimeError("No samples remain for AMICA after BAD annotations were omitted.")
    model = AMICA(device=device, n_components=n_components, max_iter=max_iter, verbose=2)
    model.fit(data)
    return model, model.to_mne(info=raw.info)


def label_components(raw: mne.io.BaseRaw, ica: mne.preprocessing.ICA) -> pd.DataFrame:
    """Store all ICLabel probabilities; component rejection remains manual."""
    torch.set_default_dtype(torch.float32)
    probabilities = iclabel_label_components(raw, ica, inplace=True, backend="torch")
    names = ["brain", "muscle artifact", "eye blink", "heart beat", "line noise", "channel noise", "other"]
    labels = pd.DataFrame(probabilities, columns=names)
    labels.insert(0, "component", np.arange(len(labels)))
    labels["predicted_label"] = labels[names].idxmax(axis=1)
    return labels


def save_component_plots(
    ica: mne.preprocessing.ICA,
    labels: pd.DataFrame,
    output_dir: Path,
    base: str,
    n_rows: int = 8,
    n_columns: int = 4,
) -> None:
    """Save labelled ICA topographies in pages suitable for quick inspection."""
    components_per_figure = n_rows * n_columns
    n_components = len(labels)
    for page, start in enumerate(range(0, n_components, components_per_figure), start=1):
        component_indices = np.arange(start, min(start + components_per_figure, n_components))
        fig, axes = plt.subplots(n_rows, n_columns, figsize=(2*n_columns, 2.4*n_rows))
        axes = axes.ravel()
        ica.plot_components(
            picks=component_indices,
            axes=axes[:len(component_indices)],
            cmap="jet",
            nrows=n_rows,
            ncols=n_columns,
            show=False,
        )
        for axis, (_, component) in zip(axes, labels.iloc[component_indices].iterrows()):
            predicted_label = component["predicted_label"]
            probability = component[predicted_label]
            axis.set_title(
                f"{axis.get_title()}\n{predicted_label} ({probability:.2f})",
                fontsize=9,
            )
        for axis in axes[len(component_indices):]:
            axis.set_visible(False)
        fig.savefig(
            output_dir / f"{base}_desc-amica_components-{page:02d}.png",
            dpi=150,
            bbox_inches="tight",
        )
        # plt.show()
        plt.close(fig)


def harmonize_channels_across_runs(
    cleaned_runs: dict[str, mne.io.BaseRaw],
    channel_reports: dict[str, dict[str, list[str]]],
    logger: logging.Logger,
) -> list[str]:
    """Give every run the same interpolated-channel set and EEG reference.

    ICA requires a stable sensor mixing geometry.  PREP's run-specific channel
    decisions are retained for QC, then their union is interpolated in every
    run so that a physical channel in one run is not compared with a virtual
    channel in another.
    """
    common_bad_channels = sorted({
        channel
        for report in channel_reports.values()
        for channels in report.values()
        for channel in channels
    })
    logger.info(
        "Condition-level common interpolation set (%d channels): %s",
        len(common_bad_channels), common_bad_channels or "none",
    )
    for run_name, raw in cleaned_runs.items():
        available_channels = [channel for channel in common_bad_channels if channel in raw.ch_names]
        missing_channels = sorted(set(common_bad_channels) - set(available_channels))
        if missing_channels:
            raise RuntimeError(
                f"{run_name} lacks channels required for common interpolation: "
                + ", ".join(missing_channels)
            )
        if available_channels:
            raw.info["bads"] = available_channels
            raw.interpolate_bads(reset_bads=True, verbose="error")
        # PREP's robust reference can vary with each run's initially usable
        # channels.  Apply one explicit final reference after harmonization.
        raw.set_eeg_reference("average", projection=False, verbose="error")
        raw.info["description"] = (
            (raw.info.get("description") or "")
            + " Condition-level common channel interpolation and average EEG reference applied."
        )
        channel_reports[run_name]["condition_common_interpolated"] = available_channels
    return common_bad_channels


def process_subject(subject: str, runs: dict, output_root: Path, args: argparse.Namespace) -> None:
    eeg_dir = output_root / subject / "ses-01" / "eeg"
    eeg_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(eeg_dir / f"{subject}_task-{args.condition}_desc-amica.log")
    logger.info("Processing %s, condition %s", subject, args.condition)

    cleaned_runs = {}
    run_ids = {}
    channel_reports = {}
    for run_name, (raw, _) in runs.items():
        run_no = int(run_name.split("-")[-1])
        run_id = get_run_id(subject, args.condition, run_no)

        logger.info("Preprocessing %s", run_name)
        clean_raw, channel_report = preprocess_run(raw, logger)
        channel_reports[run_name] = channel_report
        cleaned_runs[run_name] = clean_raw
        run_ids[run_name] = run_id

    common_bad_channels = harmonize_channels_across_runs(cleaned_runs, channel_reports, logger)
    for run_name, clean_raw in cleaned_runs.items():
        clean_raw.save(
            eeg_dir / f"{run_ids[run_name]}_desc-prep_raw.fif",
            overwrite=args.overwrite,
        )

    # One AMICA model per subject x condition, fitted over all available runs.
    # Keep EOG in the saved run-level raws for artifact validation, but do not
    # include it in the EEG decomposition or its rank calculation.
    pooled_raw = mne.concatenate_raws(list(cleaned_runs.values()))
    pooled_eeg = pooled_raw.copy().pick("eeg")
    mne_rank = mne.compute_rank(pooled_eeg, tol="auto", tol_kind="absolute")["eeg"]
    # PREP applies an average reference.  This imposes one exact linear
    # constraint across EEG channels, so AMICA must receive at most N - 1
    # components even when MNE's numerical tolerance reports full rank.
    reference_rank = len(pooled_eeg.ch_names) - 1
    rank = min(mne_rank, reference_rank)
    logger.info(
        "Fitting AMICA to %d EEG channels at rank %d "
        "(MNE estimate %d; average-reference cap %d)",
        len(pooled_eeg.ch_names), rank, mne_rank, reference_rank,
    )
    model, ica = fit_amica(pooled_eeg, rank, args.max_iter, args.device)

    base = f"{subject}_ses-01_task-{args.condition}"
    ica.save(eeg_dir / f"{base}_desc-amica_ica.fif", overwrite=args.overwrite)
    labels = label_components(pooled_eeg, ica)
    labels.to_csv(eeg_dir / f"{base}_desc-amica_iclabels.tsv", sep="\t", index=False)
    save_component_plots(ica, labels, eeg_dir, base)
    logger.info("Saved labelled component plots to %s", eeg_dir)

    metadata = {
        "condition": args.condition,
        "runs": list(runs),
        "sampling_frequency_hz": SFREQ,
        "filter_hz": {"highpass": ICA_FILTER[0], "lowpass": ICA_FILTER[1]},
        "line_noise_method": "PREP line_freqs (50 and 100 Hz); no extra notch filter",
        "dropped_channels": list(DROP_CHANNELS),
        "amica_channel_types": ["eeg"],
        "retained_non_eeg_channels": ["eog"],
        "channel_interpolation_report": channel_reports,
        "condition_common_interpolation_channels": common_bad_channels,
        "final_reference": "average EEG reference after condition-level channel harmonization",
        "amica": {
            "n_components": int(rank),
            "mne_rank_estimate": int(mne_rank),
            "average_reference_rank_cap": int(reference_rank),
            "max_iter": args.max_iter,
            "device": args.device,
        },
        "bad_span_policy": "Existing BAD_* annotations omitted from AMICA fitting; no AutoReject or window-wise interpolation.",
    }
    (eeg_dir / f"{base}_desc-preproc.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    report_rows = []
    for run_name, report in channel_reports.items():
        for stage, channels in report.items():
            report_rows.append({
                "run": run_name,
                "stage": stage,
                "n_channels": len(channels),
                "channels": ",".join(channels),
            })
    pd.DataFrame(report_rows).to_csv(
        eeg_dir / f"{base}_desc-channel_interpolation.tsv", sep="\t", index=False
    )
    logger.info("AMICA completed after %s iterations", getattr(model, "n_iter_", "unknown"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", choices=("overt", "silent"), required=True)
    parser.add_argument("--data-path", type=Path, default=Path("data/spellitout"))
    parser.add_argument("--derivatives-path", type=Path, default=Path("derivatives/preprocessing"))
    parser.add_argument("--subjects", type=int, nargs="+", default=None)
    parser.add_argument("--runs", type=int, nargs="+", default=(1, 2, 3))
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    dataset = SpellItOut(
        arguments.condition,
        subjects=arguments.subjects,
        runs=list(arguments.runs),
        data_path=arguments.data_path,
    )
    output = arguments.derivatives_path / f"task-{arguments.condition}"
    for subject_id, subject_runs in dataset.get_data().items():
        process_subject(subject_id, subject_runs, output, arguments)
