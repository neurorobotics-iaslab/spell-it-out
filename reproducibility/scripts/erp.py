"""Compute cached subject ERPs and create the paper comparison figure.

The workflow supports raw and published preprocessed BIDS EEG. Computation
writes one compact ROI-average checkpoint per subject, task, and event. Plotting
reads only those checkpoints, so figures can be redrawn without loading EEG.
Raw EEG uses the final annotations from the published preprocessing derivative
while retaining the original signal samples.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import mne
import numpy as np
import pandas as pd

from spellitout import SpellItOutDataset


DEFAULT_BIDS_ROOT = Path("data/spellitout")
DEFAULT_OUTPUT = Path("reproducibility/results/erp")
DEFAULT_CONDITIONS = ("overt", "covert")
DEFAULT_EVENTS = ("pre_syllabification_dot", "syllabification_cue")
DEFAULT_CHANNELS = ("O1", "OZ", "O2")
CONDITION_LABELS = {"overt": "Overt Speech", "covert": "Covert Speech"}
EVENT_ALIASES = {
    "sentence_cue": "listening_cue",
    "rhythm_dots": "pre_syllabification_dot",
    "syllable_cue": "syllabification_cue",
}
N170_WINDOW_MS = (140.0, 250.0)
EVENT_NAMES = {
    "listening_cue": "LISTEN.",
    "pre_syllabification_dot": "DOTS",
    "syllabification_cue": "SYLL.",
}

def optional_float(value: str) -> float | None:
    if value.lower() in {"none", "null", "off"}:
        return None
    return float(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode", choices=("compute", "plot", "all"), nargs="?", default="all"
    )
    parser.add_argument(
        "--source", choices=("raw", "preprocessed"), default="preprocessed"
    )
    parser.add_argument("--bids-root", type=Path, default=DEFAULT_BIDS_ROOT)
    parser.add_argument(
        "--preprocessed-root",
        type=Path,
        default=None,
        help="Default: <bids-root>/derivatives/preprocessing.",
    )
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=DEFAULT_CONDITIONS,
        default=DEFAULT_CONDITIONS,
    )
    parser.add_argument(
        "--events",
        nargs="+",
        default=DEFAULT_EVENTS,
        help="Annotation descriptions, in figure-row order.",
    )
    parser.add_argument("--subjects", type=int, nargs="+", default=None)
    parser.add_argument("--runs", type=int, nargs="+", default=(1, 2, 3))
    parser.add_argument("--tmin", type=float, default=-0.2)
    parser.add_argument("--tmax", type=float, default=0.6)
    parser.add_argument(
        "--plot-zero",
        type=float,
        default=0.0,
        help="Event-relative time displayed as zero; affects plotting only.",
    )
    parser.add_argument(
        "--baseline",
        type=float,
        nargs=2,
        default=None,
        metavar=("START", "END"),
        help="Optional baseline interval; default: no baseline correction.",
    )
    parser.add_argument(
        "--lowpass",
        type=optional_float,
        default=30.0,
        help="Low-pass applied to an in-memory copy; use 'none' to disable.",
    )
    parser.add_argument(
        "--reject-blinks",
        action="store_true",
        help="Treat reviewed BLINK annotations as BAD_BLINK during epoching.",
    )
    parser.add_argument(
        "--channels", nargs="+", default=DEFAULT_CHANNELS
    )
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--overwrite-checkpoints",
        action="store_true",
        help="Recompute existing subject/task/event checkpoints.",
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
    selected = sorted({f"{number:02d}" for number in requested})
    missing = sorted(set(selected) - available)
    if missing:
        raise ValueError(f"Subjects absent from {dataset.root}: {', '.join(missing)}")
    return selected


def _baseline(args: argparse.Namespace) -> tuple[float, float] | None:
    return None if args.baseline is None else tuple(args.baseline)


def _checkpoint_configuration(args: argparse.Namespace) -> dict[str, object]:
    return {
        "source": args.source,
        "conditions": list(args.conditions),
        "events": [EVENT_ALIASES.get(event, event) for event in args.events],
        "runs": sorted(set(args.runs)),
        "tmin": args.tmin,
        "tmax": args.tmax,
        "baseline": _baseline(args),
        "lowpass": args.lowpass,
        "reject_blinks": args.reject_blinks,
        "channels": list(args.channels),
    }


def source_output_root(args: argparse.Namespace) -> Path:
    baseline = (
        "none" if args.baseline is None
        else "-".join(_number_label(value) for value in args.baseline)
    )
    processing = (
        f"window-{_number_label(args.tmin)}-{_number_label(args.tmax)}s_"
        f"lowpass-{_number_label(args.lowpass)}_baseline-{baseline}_"
        f"blinks-{'reject' if args.reject_blinks else 'keep'}"
    )
    events = "__".join(EVENT_ALIASES.get(event, event) for event in args.events)
    return (
        args.output_path / f"source-{args.source}"
        / f"tasks-{'-'.join(args.conditions)}" / f"events-{events}"
        / f"runs-{'-'.join(str(run) for run in sorted(set(args.runs)))}"
        / f"channels-{'-'.join(args.channels)}" / processing
    )


def _number_label(value: float | None) -> str:
    return "none" if value is None else str(float(value)).replace("-", "m").replace(".", "p")


def _check_method(root: Path, args: argparse.Namespace) -> None:
    path = root / "method.json"
    if not path.exists():
        if (root / "subject_erps").exists():
            raise RuntimeError(f"Checkpoint configuration is missing: {path}")
        return
    stored = json.loads(path.read_text(encoding="utf-8"))["Configuration"]
    expected = json.loads(json.dumps(_checkpoint_configuration(args)))
    if stored != expected:
        raise RuntimeError(f"ERP checkpoint configuration differs from requested parameters: {path}")


def _slug(value: str) -> str:
    return value.replace("_", "-")


def checkpoint_path(
    root: Path, subject: str, condition: str, event: str
) -> Path:
    return (
        root
        / "subject_erps"
        / f"sub-{subject}_task-{condition}_event-{_slug(event)}_erp.npz"
    )


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
    annotation_paths: list[Path] = []
    raws: list[mne.io.BaseRaw] = []
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
        raws.append(raw)
    return (
        mne.concatenate_raws(raws, preload=True, verbose="error"),
        signal_paths,
        annotation_paths,
    )


def _mark_blinks_bad(raw: mne.io.BaseRaw) -> int:
    descriptions = [str(description) for description in raw.annotations.description]
    blink_mask = np.isin(
        np.char.upper(np.asarray(descriptions)), ("BLINK", "MANUAL_BLINK")
    )
    if not blink_mask.any():
        return 0
    descriptions = [
        "BAD_BLINK" if is_blink else description
        for description, is_blink in zip(descriptions, blink_mask)
    ]
    raw.set_annotations(
        mne.Annotations(
            onset=raw.annotations.onset,
            duration=raw.annotations.duration,
            description=descriptions,
            orig_time=raw.annotations.orig_time,
        )
    )
    return int(blink_mask.sum())


def _event_epochs(
    raw: mne.io.BaseRaw,
    channels: list[str],
    event: str,
    tmin: float,
    tmax: float,
    baseline: tuple[float, float] | None,
) -> mne.Epochs:
    source_event = EVENT_ALIASES.get(event, event)
    events, event_ids = mne.events_from_annotations(
        raw, event_id={source_event: 1}, verbose="error"
    )
    if source_event not in event_ids or len(events) == 0:
        raise RuntimeError(f"No {source_event!r} annotations found.")
    return mne.Epochs(
        raw,
        events,
        event_id={source_event: 1},
        tmin=tmin,
        tmax=tmax,
        baseline=baseline,
        picks=channels,
        reject_by_annotation=True,
        preload=True,
        verbose="error",
    )


def _write_checkpoint(
    path: Path,
    *,
    times: np.ndarray,
    data: np.ndarray,
    roi_channels: list[str],
    n_epochs: int,
    rejected_blinks: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        times=np.asarray(times, dtype=np.float64),
        data=np.asarray(data, dtype=np.float64),
        roi_channels=np.asarray(roi_channels, dtype="U"),
        n_epochs=np.asarray(n_epochs, dtype=np.int64),
        rejected_blinks=np.asarray(rejected_blinks, dtype=np.int64),
    )


def _write_method(root: Path, args: argparse.Namespace) -> None:
    _check_method(root, args)
    configuration = _checkpoint_configuration(args)
    root.mkdir(parents=True, exist_ok=True)
    metadata = {
        "Description": "Subject-level ROI ERP checkpoints for the paper.",
        "SignalSource": args.source,
        "AnnotationSource": (
            "published preprocessing derivative"
            if args.source == "raw"
            else "signal source events"
        ),
        "Configuration": configuration,
        "CheckpointContents": (
            "Per-subject ROI mean across retained epochs and ROI channels, "
            "time vector, channel names, epoch count, and blink-span count."
        ),
        "BadSpanPolicy": "Epochs overlapping BAD_* annotations are rejected.",
        "Normalization": (
            "No baseline correction"
            if args.baseline is None
            else "MNE baseline correction"
        ),
    }
    (root / "method.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )


def compute_checkpoints(args: argparse.Namespace) -> None:
    logger = logging.getLogger("spellitout.paper.erp")
    _, preprocessed_root, source_root = _roots(args)
    source_dataset = SpellItOutDataset(source_root)
    annotation_dataset = SpellItOutDataset(preprocessed_root)
    subjects = _subjects(source_dataset, args.subjects)
    root = source_output_root(args)
    _write_method(root, args)
    manifest_path = root / "checkpoints.tsv"
    manifest_rows: list[dict[str, Any]] = []
    if manifest_path.exists():
        manifest_rows = pd.read_csv(manifest_path, sep="\t").to_dict("records")

    resolved_events = [EVENT_ALIASES.get(event, event) for event in args.events]
    for subject in subjects:
        for condition in args.conditions:
            outputs = {
                event: checkpoint_path(root, subject, condition, event)
                for event in resolved_events
            }
            pending = [
                event
                for event, path in outputs.items()
                if args.overwrite_checkpoints or not path.exists()
            ]
            if not pending:
                logger.info(
                    "Reusing all checkpoints for sub-%s task-%s", subject, condition
                )
                continue
            raw, signal_paths, annotation_paths = _condition_raw(
                source_dataset,
                annotation_dataset,
                subject=subject,
                condition=condition,
                runs=list(args.runs),
                source=args.source,
            )
            rejected_blinks = _mark_blinks_bad(raw) if args.reject_blinks else 0
            roi_channels = [
                channel
                for channel in args.channels
                if channel in raw.ch_names
            ]
            if not roi_channels:
                raise RuntimeError(
                    f"None of the requested ROI channels is available for sub-{subject}."
                )
            if args.lowpass is not None:
                raw.filter(
                    l_freq=None,
                    h_freq=args.lowpass,
                    picks=roi_channels,
                    verbose="error",
                )
            for event in pending:
                epochs = _event_epochs(
                    raw,
                    roi_channels,
                    event,
                    args.tmin,
                    args.tmax,
                    _baseline(args),
                )
                if not len(epochs):
                    raise RuntimeError(
                        f"No clean {event} epochs remain for "
                        f"sub-{subject} task-{condition}."
                    )
                data = epochs.get_data(copy=False).mean(axis=(0, 1))
                output = outputs[event]
                _write_checkpoint(
                    output,
                    times=epochs.times,
                    data=data,
                    roi_channels=roi_channels,
                    n_epochs=len(epochs),
                    rejected_blinks=rejected_blinks,
                )
                manifest_rows = [
                    row
                    for row in manifest_rows
                    if not (
                        str(row.get("subject")) == f"sub-{subject}"
                        and row.get("task") == condition
                        and row.get("event") == event
                    )
                ]
                manifest_rows.append(
                    {
                        "subject": f"sub-{subject}",
                        "task": condition,
                        "event": event,
                        "source": args.source,
                        "checkpoint": output.as_posix(),
                        "n_epochs": len(epochs),
                        "roi_channels": ";".join(roi_channels),
                        "rejected_blink_spans": rejected_blinks,
                        "signal_files": ";".join(
                            path.as_posix() for path in signal_paths
                        ),
                        "annotation_files": ";".join(
                            path.as_posix() for path in annotation_paths
                        ),
                    }
                )
                logger.info("Saved %s (%d epochs)", output, len(epochs))
            del raw
    if manifest_rows:
        pd.DataFrame(manifest_rows).sort_values(
            ["subject", "task", "event"]
        ).to_csv(manifest_path, sep="\t", index=False)


def _read_checkpoint(path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing ERP checkpoint: {path}. Run in compute or all mode first."
        )
    with np.load(path, allow_pickle=False) as archive:
        return {
            "times": archive["times"].copy(),
            "data": archive["data"].copy(),
            "roi_channels": tuple(archive["roi_channels"].tolist()),
            "n_epochs": int(archive["n_epochs"].item()),
            "rejected_blinks": int(archive["rejected_blinks"].item()),
        }


def _panel_summary(checkpoints: list[dict[str, object]]) -> dict[str, object]:
    if len(checkpoints) < 2:
        raise RuntimeError(
            "At least two subject checkpoints are required for an ERP panel."
        )
    times = np.asarray(checkpoints[0]["times"])
    for checkpoint in checkpoints[1:]:
        if not np.array_equal(times, checkpoint["times"]):
            raise RuntimeError("ERP checkpoint time axes do not match.")
    subject_data = np.stack([checkpoint["data"] for checkpoint in checkpoints])
    mean = subject_data.mean(axis=0)
    sem = subject_data.std(axis=0, ddof=1) / np.sqrt(len(subject_data))
    common_channels = set(checkpoints[0]["roi_channels"])
    for checkpoint in checkpoints[1:]:
        common_channels &= set(checkpoint["roi_channels"])
    requested_order = tuple(checkpoints[0]["roi_channels"])
    return {
        "times": times,
        "mean": mean,
        "confidence": 1.96 * sem,
        "n_subjects": len(subject_data),
        "median_clean_epochs": int(
            np.median([checkpoint["n_epochs"] for checkpoint in checkpoints])
        ),
        "roi_channels": tuple(
            channel for channel in requested_order if channel in common_channels
        ),
    }


def _analysis_label(args: argparse.Namespace) -> str:
    cohort = (
        "all"
        if args.subjects is None
        else "-".join(f"{subject:02d}" for subject in sorted(set(args.subjects)))
    )
    zero = f"{args.plot_zero:g}".replace("-", "m").replace(".", "p")
    return f"cohort-{cohort}_plotzero-{zero}"


def plot_comparison(
    panels: dict[tuple[str, str], dict[str, object]],
    args: argparse.Namespace,
    output_dir: Path,
) -> tuple[Path, Path]:
    all_bounds: list[np.ndarray] = []
    for panel in panels.values():
        all_bounds.extend(
            (
                panel["mean"] - panel["confidence"],
                panel["mean"] + panel["confidence"],
            )
        )
    amplitudes = np.concatenate(all_bounds) * 1e6
    padding = max(0.25, 0.06 * np.ptp(amplitudes))
    y_limits = (
        float(np.nanmin(amplitudes) - padding),
        float(np.nanmax(amplitudes) + padding),
    )

    events = [EVENT_ALIASES.get(event, event) for event in args.events]
    figure, axes = plt.subplots(
        len(events),
        len(args.conditions),
        figsize=(5.0 * len(args.conditions), 2.5 * len(events)),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    for row, event in enumerate(events):
        for column, condition in enumerate(args.conditions):
            axis = axes[row, column]
            print (panels.keys(),events)

            panel = panels[(event, condition)]
            milliseconds = (panel["times"] - args.plot_zero) * 1000
            axis.axvspan(*N170_WINDOW_MS, color="#7f7f7f", alpha=0.10, zorder=0)
            axis.fill_between(
                milliseconds,
                (panel["mean"] - panel["confidence"]) * 1e6,
                (panel["mean"] + panel["confidence"]) * 1e6,
                color="#f58518",
                alpha=0.28,
                linewidth=0,
            )
            axis.plot(
                milliseconds,
                panel["mean"] * 1e6,
                color="#e45756",
                linewidth=2.3,
            )
            axis.axvline(0, color="black", linewidth=0.8, linestyle="--")
            axis.axhline(0, color="black", linewidth=0.65)
            axis.set_ylim(*y_limits)
            axis.grid(alpha=0.22)
            axis.text(
                0.98,
                0.06,
                f"n = {panel['n_subjects']}\n"
                f"median clean epochs = {panel['median_clean_epochs']}",
                transform=axis.transAxes,
                ha="right",
                va="bottom",
                fontsize=8,
            )
            if row == 0:
                axis.set_title(
                    CONDITION_LABELS.get(
                        condition, condition.replace("_", " ").title()
                    ),
                    fontsize=13,
                    fontweight="bold",
                )
            if column == 0:
                event_label = EVENT_NAMES.get(event, event)
                axis.set_ylabel(
                    rf"$\mathbf{{{event_label}}}$" + "\nAmplitude (µV)",
                    fontweight="normal",
                )
            if row == len(events) - 1:
                axis.set_xlabel("Time (ms)")

    figure.legend(
        handles=[
            Patch(
                facecolor="#f58518", alpha=0.28, label="95% CI across subjects"
            ),
            Line2D(
                [0], [0], color="#e45756", linewidth=2.3, label="Group mean"
            ),
            Patch(
                facecolor="#7f7f7f",
                alpha=0.10,
                label=(
                    f"ERP window "
                    f"({N170_WINDOW_MS[0]:g}–{N170_WINDOW_MS[1]:g} ms)"
                ),
            ),
        ],
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.02),
    )
    figure.tight_layout(rect=(0, 0.10, 1, 1))

    output_dir.mkdir(parents=True, exist_ok=True)
    condition_slug = "-".join(args.conditions)
    event_slug = "-".join(_slug(event) for event in events)
    stem = output_dir / (
        f"task-{condition_slug}_desc-{event_slug}_erp-comparison"
    )
    png_path, pdf_path = stem.with_suffix(".png"), stem.with_suffix(".pdf")
    figure.savefig(png_path, dpi=args.dpi, bbox_inches="tight")
    figure.savefig(pdf_path, bbox_inches="tight")
    plt.close(figure)
    return png_path, pdf_path


def plot_results(args: argparse.Namespace) -> None:
    _, _, source_root = _roots(args)
    dataset = SpellItOutDataset(source_root)
    subjects = _subjects(dataset, args.subjects)
    root = source_output_root(args)
    _check_method(root, args)
    panels: dict[tuple[str, str], dict[str, object]] = {}
    cohort_rows: list[dict[str, object]] = []
    events = [EVENT_ALIASES.get(event, event) for event in args.events]
    for event in events:
        for condition in args.conditions:
            checkpoints = []
            for subject in subjects:
                checkpoint = _read_checkpoint(
                    checkpoint_path(root, subject, condition, event)
                )
                checkpoints.append(checkpoint)
                cohort_rows.append(
                    {
                        "subject": f"sub-{subject}",
                        "task": condition,
                        "event": event,
                        "n_epochs": checkpoint["n_epochs"],
                        "roi_channels": ";".join(checkpoint["roi_channels"]),
                    }
                )
            panels[(event, condition)] = _panel_summary(checkpoints)

    figure_dir = root / "figures" / _analysis_label(args)
    png_path, pdf_path = plot_comparison(panels, args, figure_dir)
    pd.DataFrame(cohort_rows).to_csv(
        figure_dir / "cohort.tsv", sep="\t", index=False
    )
    print(f"Saved {png_path}")
    print(f"Saved {pdf_path}")


def _validate(args: argparse.Namespace) -> None:
    if args.tmin >= args.tmax:
        raise ValueError("--tmin must be lower than --tmax.")
    if not args.tmin <= args.plot_zero <= args.tmax:
        raise ValueError("--plot-zero must fall within the epoch window.")
    baseline = _baseline(args)
    if baseline is not None and (
        baseline[0] >= baseline[1]
        or baseline[0] < args.tmin
        or baseline[1] > args.tmax
    ):
        raise ValueError(
            "--baseline must be an increasing interval within the epoch window."
        )
    if args.lowpass is not None and args.lowpass <= 0:
        raise ValueError("--lowpass must be positive or none.")
    if not args.runs:
        raise ValueError("At least one run is required.")
    if len(set(args.conditions)) != len(args.conditions):
        raise ValueError("--conditions contains duplicates.")
    resolved_events = [EVENT_ALIASES.get(event, event) for event in args.events]
    if len(set(resolved_events)) != len(resolved_events):
        raise ValueError("--events contains duplicate or equivalent event names.")


def main(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    _validate(args)
    if args.mode in {"compute", "all"}:
        compute_checkpoints(args)
    if args.mode in {"plot", "all"}:
        plot_results(args)


if __name__ == "__main__":
    main(parse_args())
