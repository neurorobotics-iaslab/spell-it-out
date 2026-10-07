"""Plot per-channel mean and standard deviation for overt and covert EEG.

The output mirrors the layout of the channel-characterization figure used as
reference: channels are grouped by participant, vertical dotted lines mark
participant boundaries, and the two experimental conditions occupy two rows.

All available runs for a participant/condition are pooled.  Statistics are
computed incrementally, so the complete recordings are never loaded into RAM
at the same time.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import mne
from mne_bids import BIDSPath, read_raw_bids
import numpy as np

from spellitout import SpellItOutDataset


DEFAULT_OUTPUT = Path("paper_analysis/results/channel_statistics")

MICROVOLTS_PER_VOLT = 1e6
CONDITIONS = ("overt", "covert")
CONDITION_COLORS = {"overt": "#e66101", "covert": "#5e3c99"}


@dataclass(frozen=True)
class ChannelStatistics:
    """Summary values for one participant and one condition."""

    channels: tuple[str, ...]
    mean_uv: np.ndarray
    std_uv: np.ndarray
    n_samples: int
    n_runs: int


def _merge_moments(
    count: int,
    mean: np.ndarray,
    m2: np.ndarray,
    block: np.ndarray,
) -> tuple[int, np.ndarray, np.ndarray]:
    """Merge a channels-by-samples block into running Welford moments."""
    block_count = block.shape[1]
    if block_count == 0:
        return count, mean, m2

    block_mean = block.mean(axis=1)
    block_m2 = np.square(block - block_mean[:, np.newaxis]).sum(axis=1)
    if count == 0:
        return block_count, block_mean, block_m2

    total = count + block_count
    delta = block_mean - mean
    merged_mean = mean + delta * (block_count / total)
    merged_m2 = m2 + block_m2 + np.square(delta) * count * block_count / total
    return total, merged_mean, merged_m2


def compute_channel_statistics(
    files: Iterable[BIDSPath], chunk_seconds: float
) -> ChannelStatistics | None:
    """Pool EEG samples across files and return channel-wise statistics."""
    files = tuple(files)
    if not files:
        return None

    channels: tuple[str, ...] | None = None
    count = 0
    mean = np.empty(0, dtype=float)
    m2 = np.empty(0, dtype=float)

    for file_path in files:
        raw = read_raw_bids(file_path, extra_params={"preload": False}, verbose="error")
        picks = mne.pick_types(raw.info, eeg=True, eog=False, exclude=[])
        run_channels = tuple(raw.ch_names[index] for index in picks)

        if channels is None:
            channels = run_channels
            mean = np.zeros(len(channels), dtype=float)
            m2 = np.zeros(len(channels), dtype=float)
        elif run_channels != channels:
            raise ValueError(
                f"EEG channel order differs in {file_path}.\n"
                f"Expected: {channels}\nFound: {run_channels}"
            )

        chunk_samples = max(1, int(round(chunk_seconds * raw.info["sfreq"])))
        for start in range(0, raw.n_times, chunk_samples):
            stop = min(start + chunk_samples, raw.n_times)
            block = raw.get_data(picks=picks, start=start, stop=stop)
            count, mean, m2 = _merge_moments(count, mean, m2, block)
        raw.close()

    assert channels is not None
    variance = m2 / count if count else np.full(len(channels), np.nan)
    return ChannelStatistics(
        channels=channels,
        mean_uv=mean * MICROVOLTS_PER_VOLT,
        std_uv=np.sqrt(variance) * MICROVOLTS_PER_VOLT,
        n_samples=count,
        n_runs=len(files),
    )


def collect_statistics(
    data_root: Path, subjects: tuple[str, ...], chunk_seconds: float
) -> dict[str, dict[str, ChannelStatistics | None]]:
    """Collect condition/participant summaries from a BIDS dataset."""
    recordings = SpellItOutDataset(data_root).recordings
    results: dict[str, dict[str, ChannelStatistics | None]] = {}
    for condition in CONDITIONS:
        results[condition] = {}
        for subject in subjects:
            files = [path for path in recordings
                     if path.task == condition and f"sub-{path.subject}" == subject]
            print(f"{condition:6s} {subject}: {len(files)} run(s)", flush=True)
            results[condition][subject] = compute_channel_statistics(
                files, chunk_seconds
            )
    return results


def infer_channel_slots(
    statistics: dict[str, dict[str, ChannelStatistics | None]],
) -> int:
    """Return the common channel count, including a width for missing data."""
    counts = {
        len(summary.channels)
        for condition in CONDITIONS
        for summary in statistics[condition].values()
        if summary is not None
    }
    if not counts:
        raise RuntimeError("No BIDS EEG recordings were found.")
    if len(counts) != 1:
        raise ValueError(f"Participants do not share one EEG channel count: {counts}")
    return counts.pop()


def common_y_limits(
    statistics: dict[str, dict[str, ChannelStatistics | None]],
    ymin: float | None,
    ymax: float | None,
) -> tuple[float, float]:
    """Use shared, data-driven limits unless the user supplies fixed limits."""
    values = [
        values
        for condition in CONDITIONS
        for summary in statistics[condition].values()
        if summary is not None
        for values in (summary.mean_uv, summary.std_uv)
    ]
    finite = np.concatenate(values)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise RuntimeError("All computed statistics are non-finite.")

    data_min, data_max = float(finite.min()), float(finite.max())
    span = max(data_max - data_min, 1.0)
    return (
        ymin if ymin is not None else min(0.0, data_min - 0.05 * span),
        ymax if ymax is not None else data_max + 0.05 * span,
    )


def plot_statistics(
    statistics: dict[str, dict[str, ChannelStatistics | None]],
    subjects: tuple[str, ...],
    output: Path,
    ymin: float | None,
    ymax: float | None,
) -> None:
    """Create and save the two-row overt/covert characterization figure."""
    channel_slots = infer_channel_slots(statistics)
    y_limits = common_y_limits(statistics, ymin, ymax)
    total_slots = len(subjects) * channel_slots

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(20, 9),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )

    for axis, condition in zip(axes, CONDITIONS):
        color = CONDITION_COLORS[condition]
        for subject_index, subject in enumerate(subjects):
            left = subject_index * channel_slots
            summary = statistics[condition][subject]
            if summary is not None:
                x = left + np.arange(len(summary.channels))
                axis.scatter(
                    x,
                    summary.std_uv,
                    s=5,
                    color=color,
                    alpha=0.75,
                    linewidths=0,
                    label="Standard deviation" if subject_index == 0 else None,
                    zorder=3,
                )
                axis.plot(
                    x,
                    summary.mean_uv,
                    color="black",
                    linewidth=0.8,
                    label="Mean" if subject_index == 0 else None,
                    zorder=4,
                )
            if subject_index:
                axis.axvline(left - 0.5, color="0.55", linestyle=":", linewidth=0.7)

        axis.axhline(0, color="0.75", linewidth=0.6, zorder=1)
        axis.set_xlim(-0.5, total_slots - 0.5)
        axis.set_ylim(*y_limits)
        axis.set_ylabel("Amplitude (µV)")
        axis.set_title(condition.capitalize(), loc="left", fontweight="bold")
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(loc="upper right", frameon=True, ncols=2)

    centers = np.arange(len(subjects)) * channel_slots + (channel_slots - 1) / 2
    axes[-1].set_xticks(centers, subjects, rotation=90)
    axes[-1].set_xlabel("Channels grouped by participant")
    fig.suptitle("Channel-wise EEG mean and standard deviation", fontweight="bold")

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    if output.suffix.lower() != ".pdf":
        fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=project_root / "data" / "spellitout" / "derivatives" / "preprocessing",
        help="BIDS dataset root; defaults to the final preprocessing derivative.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root / DEFAULT_OUTPUT / "channel_statistics_overt_covert.png",
        help="Output image path; a PDF copy is also written for non-PDF output.",
    )
    parser.add_argument(
        "--chunk-seconds",
        type=float,
        default=60.0,
        help="Amount of each recording loaded per computation block (default: 60).",
    )
    parser.add_argument("--ymin", type=float, default=None, help="Optional lower y limit in µV.")
    parser.add_argument("--ymax", type=float, default=None, help="Optional upper y limit in µV.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.chunk_seconds <= 0:
        raise ValueError("--chunk-seconds must be greater than zero.")
    if args.ymin is not None and args.ymax is not None and args.ymin >= args.ymax:
        raise ValueError("--ymin must be smaller than --ymax.")

    subjects = tuple(f"sub-{subject}" for subject in SpellItOutDataset(args.data_root).subjects)
    statistics = collect_statistics(args.data_root, subjects, args.chunk_seconds)
    plot_statistics(statistics, subjects, args.output, args.ymin, args.ymax)
    print(f"Saved figure to {args.output}")
    if args.output.suffix.lower() != ".pdf":
        print(f"Saved vector copy to {args.output.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
