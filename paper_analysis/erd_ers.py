"""Compute cached epoch TFRs and create the paper ERD/ERS figure.

The workflow supports raw and published preprocessed EEG. Full epoch-level TFR
checkpoints are intentionally retained so that plotting can be repeated with
different ``--max-bad-duration`` values without recomputing Morlet transforms.
Raw EEG uses the final BAD annotations from the published preprocessing
derivative, while its signal samples remain otherwise raw.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import mne
import numpy as np
import pandas as pd

from spellitout import SpellItOutDataset
from spellitout.erd_ers import ERDERSConfig, compute_subject_tfr


DEFAULT_BIDS_ROOT = Path("data/spellitout")
DEFAULT_OUTPUT = Path("paper_analysis/results/erd_ers")
CONDITIONS = ("overt", "covert")
ERD_BANDS = {
    "Theta": (4.0, 8.0),
    "Alpha": (8.0, 12.0),
    "Low Beta": (12.0, 16.0),
    "High Beta": (16.0, 30.0),
    "Low Gamma": (30.0, 45.0),
}
TRIAL_WINDOWS = {
    "LISTEN.": (-6.5, -3.5),
    "REP.": (1.0, 5.0),
    "REP. END": (6.0, 7.0),
    "DOTS": (7.0, 8.8),
    "SYLL.": (8.8, 20.8),
}

DEFAULT_BASELINE = (-3.0, -0.1)

def optional_float(value: str) -> float | None:
    if value.lower() in {"none", "null", "all"}:
        return None
    return float(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("compute", "plot", "all"), nargs="?", default="all")
    parser.add_argument("--source", choices=("raw", "preprocessed"), default="preprocessed")
    parser.add_argument("--bids-root", type=Path, default=DEFAULT_BIDS_ROOT)
    parser.add_argument(
        "--preprocessed-root",
        type=Path,
        default=None,
        help="Default: <bids-root>/derivatives/preprocessing.",
    )
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--subjects", type=int, nargs="+", default=None)
    parser.add_argument("--runs", type=int, nargs="+", default=(1, 2, 3))
    parser.add_argument("--tmin", type=float, default=-10.0)
    parser.add_argument("--tmax", type=float, default=22.0)
    parser.add_argument("--resample-sfreq", type=float, default=128.0)
    parser.add_argument("--fmin", type=float, default=2.0)
    parser.add_argument("--fmax", type=float, default=60.0)
    parser.add_argument("--fstep", type=float, default=2.0)
    parser.add_argument("--n-cycles-divisor", type=float, default=2.0)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--no-csd", action="store_true")
    parser.add_argument(
        "--max-bad-duration",
        type=optional_float,
        default=0.5,
        help="Maximum BAD_* overlap in seconds; use 'none' to retain every epoch.",
    )
    parser.add_argument("--baseline", type=float, nargs=2, default=DEFAULT_BASELINE)
    parser.add_argument(
        "--vlim-db",
        type=optional_float,
        default=None,
        help="Symmetric color limit in dB; use 'none' for automatic limits.",
    )
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--overwrite-checkpoints",
        action="store_true",
        help="Recompute existing subject/condition EpochsTFR checkpoints.",
    )
    parser.add_argument(
        "--redraw",
        action="store_true",
        help="Redraw from matching grand-average caches without reading subject checkpoints.",
    )
    return parser.parse_args()


def _roots(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    bids_root = args.bids_root.expanduser().resolve()
    preprocessed = (
        args.preprocessed_root.expanduser().resolve()
        if args.preprocessed_root is not None
        else bids_root / "derivatives" / "preprocessing"
    )
    source = bids_root if args.source == "raw" else preprocessed
    return bids_root, preprocessed, source


def _subjects(dataset: SpellItOutDataset, requested: list[int] | None) -> list[str]:
    if requested is None:
        return list(dataset.subjects)
    available = set(dataset.subjects)
    selected = [f"{number:02d}" for number in requested]
    missing = sorted(set(selected) - available)
    if missing:
        raise ValueError(f"Subjects absent from {dataset.root}: {', '.join(missing)}")
    return sorted(set(selected))


def _config(args: argparse.Namespace) -> ERDERSConfig:
    return ERDERSConfig(
        input_stage=args.source,
        event_name="repetition_cue",
        tmin=args.tmin,
        tmax=args.tmax,
        resample_sfreq=args.resample_sfreq,
        fmin=args.fmin,
        fmax=args.fmax,
        fstep=args.fstep,
        n_cycles_divisor=args.n_cycles_divisor,
        apply_csd=not args.no_csd,
        n_jobs=args.n_jobs,
    )


def source_output_root(args: argparse.Namespace) -> Path:
    return (
        args.output_path / f"source-{args.source}"
        / f"event-{_config(args).event_name}"
        / f"runs-{'-'.join(str(run) for run in sorted(set(args.runs)))}"
        / f"window-{_number_label(args.tmin)}-{_number_label(args.tmax)}s"
        / (f"freq-{_number_label(args.fmin)}-{_number_label(args.fmax)}Hz_"
           f"step-{_number_label(args.fstep)}Hz_sfreq-{_number_label(args.resample_sfreq)}Hz")
        / (f"cycles-divisor-{_number_label(args.n_cycles_divisor)}_"
           f"csd-{'off' if args.no_csd else 'on'}_jobs-{args.n_jobs}")
    )


def _check_method(root: Path, args: argparse.Namespace) -> None:
    path = root / "method.json"
    if not path.exists():
        if (root / "epochs_tfr").exists():
            raise RuntimeError(f"Checkpoint configuration is missing: {path}")
        return
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if (metadata["Configuration"] != asdict(_config(args))
            or metadata.get("Runs") != sorted(set(args.runs))):
        raise RuntimeError(f"ERD/ERS checkpoint configuration differs from requested parameters: {path}")


def checkpoint_path(root: Path, subject: str, condition: str) -> Path:
    return root / "epochs_tfr" / f"sub-{subject}_task-{condition}_epochs-tfr.h5"


def _ensure_head_coordinate_montage(raw: mne.io.BaseRaw) -> None:
    """Relabel the published head-based RAS montage when BIDS reports Other.

    The dataset coordinate sidecars explicitly describe the electrode and
    fiducial coordinates as MNE ``standard_1020`` positions transformed to a
    head-based RAS frame. BIDS represents that custom description as ``Other``;
    MNE-BIDS consequently reads it as ``unknown``, which prevents the automatic
    sphere fit required by CSD. Rebuilding the montage changes only its frame
    metadata and preserves every coordinate value.
    """
    montage = raw.get_montage()
    if montage is None:
        raise RuntimeError("CSD requires an EEG montage, but none is available.")
    positions = montage.get_positions()
    frame = positions["coord_frame"]
    if frame == "head":
        return
    if frame != "unknown":
        raise RuntimeError(
            f"CSD requires head coordinates; found montage frame {frame!r}."
        )
    head_montage = mne.channels.make_dig_montage(
        ch_pos=positions["ch_pos"],
        nasion=positions["nasion"],
        lpa=positions["lpa"],
        rpa=positions["rpa"],
        hsp=positions["hsp"],
        hpi=positions["hpi"],
        coord_frame="head",
    )
    raw.set_montage(head_montage, on_missing="raise")


def _condition_raw(
    source_dataset: SpellItOutDataset,
    annotation_dataset: SpellItOutDataset,
    *,
    subject: str,
    condition: str,
    runs: list[int],
    source: str,
) -> tuple[mne.io.BaseRaw, list[Path], list[Path]]:
    loaded = list(
        source_dataset.iter_runs(
            subjects=subject,
            tasks=condition,
            runs=runs,
            preload=True,
        )
    )
    if len(loaded) != len(runs):
        raise RuntimeError(
            f"Expected {len(runs)} runs for sub-{subject} task-{condition}, "
            f"found {len(loaded)}."
        )
    signal_paths = [Path(run.bids_path.fpath) for run in loaded]
    annotation_paths = []
    raws = []
    for run in loaded:
        raw = run.raw
        if source == "raw":
            reviewed = annotation_dataset.get_run(
                subject=subject,
                session=run.session or "01",
                task=condition,
                run=run.run or "01",
                preload=False,
            )
            annotations = reviewed.raw.annotations.copy()
            annotations.crop(
                tmin=raw.first_time,
                tmax=raw.times[-1],
                emit_warning=False,
                use_orig_time=False,
            )
            raw.set_annotations(annotations)
            annotation_paths.append(reviewed.events_path)
        else:
            annotation_paths.append(run.events_path)
        _ensure_head_coordinate_montage(raw)
        raws.append(raw)
    return mne.concatenate_raws(raws), signal_paths, annotation_paths


def _write_method(
    root: Path,
    args: argparse.Namespace,
    config: ERDERSConfig,
) -> None:
    _check_method(root, args)
    root.mkdir(parents=True, exist_ok=True)
    metadata = {
        "Description": "Epoch-level Morlet TFR checkpoints for the paper ERD/ERS analysis.",
        "SignalSource": args.source,
        "AnnotationSource": (
            "published preprocessing derivative"
            if args.source == "raw"
            else "signal source events"
        ),
        "Configuration": asdict(config),
        "Runs": sorted(set(args.runs)),
        "BandsHz": ERD_BANDS,
        "PlotWindowsSeconds": TRIAL_WINDOWS,
        "Normalization": "Per-epoch dB baseline correction during aggregation.",
        "BadSpanPolicy": (
            "All extractable epoch TFRs are cached; max_bad_duration is applied during plotting."
        ),
    }
    (root / "method.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )


def compute_checkpoints(args: argparse.Namespace) -> None:
    logger = logging.getLogger("spellitout.paper.erd_ers")
    bids_root, preprocessed_root, source_root = _roots(args)
    source_dataset = SpellItOutDataset(source_root)
    annotation_dataset = SpellItOutDataset(preprocessed_root)
    subjects = _subjects(source_dataset, args.subjects)
    config = _config(args)
    root = source_output_root(args)
    _write_method(root, args, config)
    manifest_rows: list[dict[str, Any]] = []
    old_manifest = root / "checkpoints.tsv"
    if old_manifest.exists():
        manifest_rows = pd.read_csv(old_manifest, sep="\t").to_dict("records")

    for subject in subjects:
        for condition in CONDITIONS:
            output = checkpoint_path(root, subject, condition)
            if output.exists() and not args.overwrite_checkpoints:
                logger.info("Reusing %s", output)
                continue
            raw, signal_paths, annotation_paths = _condition_raw(
                source_dataset,
                annotation_dataset,
                subject=subject,
                condition=condition,
                runs=list(args.runs),
                source=args.source,
            )
            tfr, report = compute_subject_tfr(raw, config, logger)
            output.parent.mkdir(parents=True, exist_ok=True)
            mne.time_frequency.write_tfrs(
                output, [tfr], overwrite=True, verbose="error"
            )
            manifest_rows = [
                row
                for row in manifest_rows
                if not (
                    str(row.get("subject")) == f"sub-{subject}"
                    and row.get("task") == condition
                )
            ]
            manifest_rows.append(
                {
                    "subject": f"sub-{subject}",
                    "task": condition,
                    "source": args.source,
                    "checkpoint": output.as_posix(),
                    "n_events": len(report),
                    "n_tfr_epochs": len(tfr),
                    "sampling_frequency": tfr.info["sfreq"],
                    "signal_files": ";".join(path.as_posix() for path in signal_paths),
                    "annotation_files": ";".join(
                        path.as_posix() for path in annotation_paths
                    ),
                }
            )
            logger.info("Saved %s", output)
            del tfr, raw
    if manifest_rows:
        pd.DataFrame(manifest_rows).sort_values(["subject", "task"]).to_csv(
            old_manifest, sep="\t", index=False
        )


def read_subject_tfr(path: Path, subject: str) -> mne.time_frequency.EpochsTFR:
    if not path.exists():
        raise FileNotFoundError(f"Missing TFR checkpoint for {subject}: {path}")
    loaded = mne.time_frequency.read_tfrs(path, verbose="error")
    if isinstance(loaded, mne.time_frequency.EpochsTFR):
        return loaded
    if isinstance(loaded, list) and len(loaded) == 1:
        loaded = loaded[0]
    if not isinstance(loaded, mne.time_frequency.EpochsTFR):
        raise RuntimeError(
            f"Expected one EpochsTFR in {path}, got {type(loaded).__name__}."
        )
    return loaded


def subject_average(
    tfr: mne.time_frequency.EpochsTFR,
    max_bad_duration: float | None,
    baseline: tuple[float, float],
    subject: str,
) -> tuple[mne.time_frequency.AverageTFR, np.ndarray]:
    if tfr.metadata is None or "bad_overlap_seconds" not in tfr.metadata:
        raise RuntimeError(f"BAD-overlap metadata is missing for {subject}.")
    overlap = tfr.metadata["bad_overlap_seconds"].to_numpy(float)
    keep = (
        np.ones(len(tfr), dtype=bool)
        if max_bad_duration is None
        else overlap <= max_bad_duration
    )
    if not keep.any():
        raise RuntimeError(f"No epochs remain for {subject} at this BAD threshold.")
    baseline_mask = (tfr.times >= baseline[0]) & (tfr.times <= baseline[1])
    if not baseline_mask.any():
        raise ValueError(f"Baseline {baseline} does not overlap the TFR time axis.")
    average_data = np.zeros(tfr.data.shape[1:], dtype=np.float64)
    for epoch_index in np.flatnonzero(keep):
        power = tfr.data[epoch_index]
        if np.any(power <= 0):
            raise RuntimeError("TFR power must be positive for dB conversion.")
        epoch_db = 10.0 * np.log10(power)
        epoch_db -= epoch_db[:, :, baseline_mask].mean(axis=2, keepdims=True)
        average_data += epoch_db
    average_data /= int(keep.sum())
    average = mne.time_frequency.AverageTFRArray(
        tfr.info,
        average_data,
        tfr.times,
        tfr.freqs,
        nave=int(keep.sum()),
        method="morlet",
    )
    return average, keep


def grand_average(
    subject_averages: list[mne.time_frequency.AverageTFR],
) -> mne.time_frequency.AverageTFR:
    if not subject_averages:
        raise RuntimeError("No subject averages are available.")
    first = subject_averages[0]
    return mne.time_frequency.AverageTFRArray(
        first.info,
        np.stack([average.data for average in subject_averages]).mean(axis=0),
        first.times,
        first.freqs,
        nave=len(subject_averages),
        method="morlet",
    )


def _number_label(value: float | None) -> str:
    if value is None:
        return "none"
    return f"{value:g}".replace("-", "m").replace(".", "p")


def analysis_label(args: argparse.Namespace) -> str:
    cohort = (
        "all"
        if args.subjects is None
        else "-".join(f"{subject:02d}" for subject in sorted(set(args.subjects)))
    )
    return (
        f"badmax-{_number_label(args.max_bad_duration)}_"
        f"baseline-{_number_label(args.baseline[0])}-{_number_label(args.baseline[1])}_"
        f"cohort-{cohort}"
    )


def grand_path(root: Path, args: argparse.Namespace, condition: str) -> Path:
    return root / "grand_averages" / analysis_label(args) / f"task-{condition}_grandaverage-tfr.h5"


def read_cached_grand(path: Path) -> mne.time_frequency.AverageTFR:
    if not path.exists():
        raise FileNotFoundError(f"Grand-average cache not found: {path}")
    loaded = mne.time_frequency.read_tfrs(path, verbose="error")
    if isinstance(loaded, list) and len(loaded) == 1:
        loaded = loaded[0]
    if not isinstance(loaded, mne.time_frequency.AverageTFR):
        raise RuntimeError(f"Expected an AverageTFR in {path}.")
    return loaded


def _common_vlim(
    grands: list[mne.time_frequency.AverageTFR],
    requested_db: float | None,
) -> tuple[tuple[float, float], str]:
    values = []
    for grand in grands:
        for fmin, fmax in ERD_BANDS.values():
            frequency_mask = (grand.freqs >= fmin) & (grand.freqs <= fmax)
            for tmin, tmax in TRIAL_WINDOWS.values():
                time_mask = (grand.times >= tmin) & (grand.times <= tmax)
                values.append(
                    np.nanmean(
                        grand.data[:, frequency_mask, :][:, :, time_mask],
                        axis=(1, 2),
                    )
                )
    rendered = np.concatenate(values)
    if requested_db is None:
        bound = float(np.nanmax(np.abs(rendered)))
        if not np.isfinite(bound) or bound == 0:
            bound = 1.0
        limits = (-bound, bound)
    else:
        limits = (-requested_db, requested_db)
    below = float(np.nanmin(rendered)) < limits[0]
    above = float(np.nanmax(rendered)) > limits[1]
    extend = "both" if below and above else "min" if below else "max" if above else "neither"
    return limits, extend


def plot_comparison(
    grands: dict[str, mne.time_frequency.AverageTFR],
    vlim: tuple[float, float],
    colorbar_extend: str,
) -> plt.Figure:
    conditions = (("overt", "Overt speech"), ("covert", "Covert speech"))
    bands = list(ERD_BANDS.items())
    windows = list(TRIAL_WINDOWS.items())
    n_bands, n_windows = len(bands), len(windows)
    figure = plt.figure(figsize=(17, 10), facecolor="white")
    grid = GridSpec(
        n_bands,
        n_windows * 2,
        figure=figure,
        left=0.12,
        right=0.98,
        top=0.88,
        bottom=0.16,
        hspace=0.08,
        wspace=0.025,
    )
    axes = np.empty((n_bands, n_windows * 2), dtype=object)
    images = []
    for condition_index, (condition, _) in enumerate(conditions):
        grand = grands[condition]
        for band_index, (_, (fmin, fmax)) in enumerate(bands):
            frequency_mask = (grand.freqs >= fmin) & (grand.freqs <= fmax)
            for window_index, (window_name, (tmin, tmax)) in enumerate(windows):
                column = condition_index * n_windows + window_index
                axis = figure.add_subplot(grid[band_index, column])
                axes[band_index, column] = axis
                time_mask = (grand.times >= tmin) & (grand.times <= tmax)
                values = np.nanmean(
                    grand.data[:, frequency_mask, :][:, :, time_mask], axis=(1, 2)
                )
                image, _ = mne.viz.plot_topomap(
                    values,
                    grand.info,
                    axes=axis,
                    cmap="RdBu_r",
                    vlim=vlim,
                    contours=4,
                    extrapolate="head",
                    show=False,
                )
                images.append(image)
                axis.set_xticks([])
                axis.set_yticks([])
                if band_index == 0:
                    axis.set_title(window_name, fontsize=12, fontweight="bold", pad=9)
    for band_index, (band_name, (fmin, fmax)) in enumerate(bands):
        position = axes[band_index, 0].get_position()
        figure.text(
            0.105,
            position.y0 + position.height / 2,
            f"{band_name}\n{fmin:.0f}–{fmax:.0f} Hz",
            ha="right",
            va="center",
            fontsize=12,
            fontweight="bold",
        )
    overt_right = axes[0, n_windows - 1].get_position().x1
    covert_left = axes[0, n_windows].get_position().x0
    figure.add_artist(
        plt.Line2D(
            [(overt_right + covert_left) / 2] * 2,
            [axes[-1, 0].get_position().y0, axes[0, 0].get_position().y1],
            transform=figure.transFigure,
            color="0.45",
            linewidth=1.2,
        )
    )
    for condition_index, (_, title) in enumerate(conditions):
        first = axes[0, condition_index * n_windows].get_position()
        last = axes[0, condition_index * n_windows + n_windows - 1].get_position()
        figure.text(
            (first.x0 + last.x1) / 2,
            0.925,
            title,
            ha="center",
            va="center",
            fontsize=17,
            fontweight="bold",
        )
    colorbar_axis = figure.add_axes([0.30, 0.075, 0.42, 0.024])
    colorbar = figure.colorbar(
        images[-1], cax=colorbar_axis, orientation="horizontal", extend=colorbar_extend
    )
    colorbar.set_label("ERD / ERS (dB)", fontsize=13, labelpad=10)
    colorbar.ax.tick_params(labelsize=11)
    return figure


def plot_results(args: argparse.Namespace) -> None:
    _, _, source_root = _roots(args)
    dataset = SpellItOutDataset(source_root)
    subjects = _subjects(dataset, args.subjects)
    root = source_output_root(args)
    grands: dict[str, mne.time_frequency.AverageTFR] = {}
    _check_method(root, args)
    selection_rows = []
    for condition in CONDITIONS:
        saved_grand = grand_path(root, args, condition)
        if args.redraw:
            grands[condition] = read_cached_grand(saved_grand)
            continue
        averages = []
        for subject in subjects:
            tfr = read_subject_tfr(
                checkpoint_path(root, subject, condition), f"sub-{subject}"
            )
            average, keep = subject_average(
                tfr,
                args.max_bad_duration,
                tuple(args.baseline),
                f"sub-{subject} task-{condition}",
            )
            averages.append(average)
            overlap = tfr.metadata["bad_overlap_seconds"].to_numpy(float)
            for epoch_index, (duration, included) in enumerate(zip(overlap, keep)):
                selection_rows.append(
                    {
                        "subject": f"sub-{subject}",
                        "task": condition,
                        "epoch_index": epoch_index,
                        "bad_overlap_seconds": duration,
                        "included": bool(included),
                        "max_bad_duration": args.max_bad_duration,
                    }
                )
            del tfr
        grands[condition] = grand_average(averages)
        saved_grand.parent.mkdir(parents=True, exist_ok=True)
        mne.time_frequency.write_tfrs(
            saved_grand, [grands[condition]], overwrite=True, verbose="error"
        )
    if not args.redraw:
        selection_path = grand_path(root, args, CONDITIONS[0]).parent / "epoch_selection.tsv"
        pd.DataFrame(selection_rows).to_csv(selection_path, sep="\t", index=False)
    vlim, extend = _common_vlim(list(grands.values()), args.vlim_db)
    figure = plot_comparison(grands, vlim, extend)
    figure_dir = root / "figures" / analysis_label(args)
    figure_dir.mkdir(parents=True, exist_ok=True)
    output = figure_dir / "erd_ers_overt-vs-covert.png"
    figure.savefig(output, dpi=args.dpi, bbox_inches="tight", pad_inches=0.04)
    figure.savefig(output.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.04)
    plt.close(figure)
    print(f"Saved {output}")


def main(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    if args.n_jobs == 0:
        raise ValueError("--n-jobs cannot be zero.")
    if args.resample_sfreq <= 2 * args.fmax:
        raise ValueError("--resample-sfreq must be greater than twice --fmax.")
    if args.max_bad_duration is not None and args.max_bad_duration < 0:
        raise ValueError("--max-bad-duration must be non-negative or none.")
    if args.vlim_db is not None and args.vlim_db <= 0:
        raise ValueError("--vlim-db must be positive or none.")
    if args.redraw and args.mode == "compute":
        raise ValueError("--redraw is only valid in plot or all mode.")
    if args.mode in {"compute", "all"}:
        compute_checkpoints(args)
    if args.mode in {"plot", "all"}:
        plot_results(args)


if __name__ == "__main__":
    main(parse_args())
