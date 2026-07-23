"""Compute SEBR with subject/run-level QC outputs and reproducible derivatives.

This version keeps the original SEBR definition, but improves technical
validation by:

- detecting blinks per run instead of only on concatenated subject data,
- recording which blink channel source was used for each run,
- saving long-form trial and blink tables,
- writing a per-subject log and JSON metadata file,
- using a derivatives layout that mirrors the other preprocessing scripts.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

sys.path.append('.')
from spellitout.spellitout import SpellItOut

PIPELINE_VERSION = "2.0"
DEFAULT_SFREQ = 256
DEFAULT_DERIVATIVES_PATH = Path("derivatives/SEBR")

PHASES_OF_INTEREST = {
    "sentence_audio": {"save_name": "listening", "duration": 7.0},
    "repetition_cue": {"save_name": "sentence pronunciation", "duration": 5.0},
    "syllable_cue": {"save_name": "syllable pronunciation", "duration": 0.6},
}


def setup_logger(log_path: Path) -> logging.Logger:
    """Create a dedicated subject log file and mirror it to stdout."""
    logger = logging.getLogger(f"sebr.{log_path.stem}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False
    formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")

    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def format_channel_list(channels: list[str]) -> str:
    return ",".join(channels) if channels else ""


def detect_blinks(raw: mne.io.BaseRaw, logger: logging.Logger) -> tuple[np.ndarray, dict[str, object]]:
    """Detect blink events using the most appropriate available frontal source.

    Preference order is EOG, then the frontal EEG set used in the original
    script, then smaller frontal subsets if needed.
    """
    candidates: list[tuple[str, list[str]]] = [
        ("EOG", ["EOG"]),
        ("frontal_triplet", ["FP1", "FP2", "FPZ"]),
        ("frontal_pair", ["FP1", "FP2"]),
        ("frontal_left", ["FP1"]),
        ("frontal_right", ["FP2"]),
        ("frontal_mid", ["FPZ"]),
    ]

    attempts: list[dict[str, object]] = []
    for source_name, candidate_channels in candidates:
        present_channels = [channel for channel in candidate_channels if channel in raw.ch_names]
        if not present_channels:
            attempts.append({"source": source_name, "channels": candidate_channels, "n_events": 0, "status": "missing"})
            continue

        channel_arg: str | list[str]
        if len(present_channels) == 1:
            channel_arg = present_channels[0]
        else:
            channel_arg = present_channels

        try:
            eog_events = mne.preprocessing.find_eog_events(
                raw,
                ch_name=channel_arg,
                l_freq=1.0,
                h_freq=10.0,
                verbose=False,
            )
        except Exception as error:
            logger.warning("Blink detection failed for %s (%s): %s", source_name, present_channels, error)
            attempts.append({
                "source": source_name,
                "channels": present_channels,
                "n_events": 0,
                "status": f"error: {error}",
            })
            continue

        attempts.append({
            "source": source_name,
            "channels": present_channels,
            "n_events": int(len(eog_events)),
            "status": "ok",
        })
        logger.info("Blink detector %s with %s found %d events", source_name, present_channels, len(eog_events))

        if len(eog_events) > 0:
            metadata = {
                "source": source_name,
                "channels": present_channels,
                "n_events": int(len(eog_events)),
                "attempts": attempts,
            }
            return eog_events[:, 0].astype(int), metadata

    metadata = {
        "source": "none",
        "channels": [],
        "n_events": 0,
        "attempts": attempts,
    }
    logger.warning("No blink detector produced any events for this run")
    return np.empty((0,), dtype=int), metadata


def summarize_phase(
    subject_id: str,
    run_name: str,
    raw: mne.io.BaseRaw,
    blink_samples: np.ndarray,
    phase_name: str,
    phase_info: dict[str, object],
    task_events: np.ndarray,
    ev_map: dict[str, int],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Return trial rows and one phase summary row for a single run."""
    trial_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []

    if phase_name not in ev_map:
        return trial_rows, summary_rows

    trigger_id = ev_map[phase_name]
    phase_events = task_events[task_events[:, 2] == trigger_id]
    duration_sec = float(phase_info["duration"])
    duration_samples = int(round(duration_sec * raw.info["sfreq"]))

    total_blinks = 0
    blink_count_values: list[int] = []

    for trial_index, event in enumerate(phase_events, start=1):
        start_sample = int(event[0])
        end_sample = start_sample + duration_samples
        blinks_in_trial = blink_samples[(blink_samples >= start_sample) & (blink_samples < end_sample)]
        n_blinks = int(len(blinks_in_trial))
        total_blinks += n_blinks
        blink_count_values.append(n_blinks)

        trial_rows.append({
            "subject": subject_id,
            "run": run_name,
            "phase": phase_name,
            "condition": phase_info["save_name"],
            "trial_index": trial_index,
            "start_sample": start_sample,
            "end_sample": end_sample,
            "duration_s": duration_sec,
            "n_blinks": n_blinks,
            "blink_rate_bpm": (n_blinks / (duration_sec / 60.0)) if duration_sec > 0 else np.nan,
            "sfreq_hz": float(raw.info["sfreq"]),
        })

    num_trials = int(len(phase_events))
    total_time_min = (num_trials * duration_sec) / 60.0 if num_trials > 0 else 0.0
    sebr = (total_blinks / total_time_min) if total_time_min > 0 else np.nan

    summary_rows.append({
        "subject": subject_id,
        "run": run_name,
        "phase": phase_name,
        "condition": phase_info["save_name"],
        "n_trials": num_trials,
        "total_blinks": total_blinks,
        "analyzed_time_s": num_trials * duration_sec,
        "analyzed_time_min": total_time_min,
        "sebr_blinks_per_min": sebr,
        "mean_blinks_per_trial": float(np.mean(blink_count_values)) if blink_count_values else np.nan,
        "median_blinks_per_trial": float(np.median(blink_count_values)) if blink_count_values else np.nan,
    })
    return trial_rows, summary_rows


def process_subject(subject: int, dataset: SpellItOut, output_root: Path) -> None:
    data = dataset.get_data()
    subject_id = f"sub-{subject:02d}"
    subject_runs = data[subject_id]

    eeg_dir = output_root / subject_id / "ses-01" / "eeg"
    eeg_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(eeg_dir / f"{subject_id}_task-sebr_desc-qc.log")
    logger.info("Processing %s with %d runs", subject_id, len(subject_runs))

    blink_rows: list[dict[str, object]] = []
    trial_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    run_rows: list[dict[str, object]] = []

    for run_name, (raw, _) in tqdm(subject_runs.items(), desc=f"Subject {subject_id}", leave=False):
        run_raw = raw.copy()
        run_raw.load_data()
        run_raw.resample(DEFAULT_SFREQ, verbose=False)
        logger.info("Run %s: resampled to %d Hz", run_name, DEFAULT_SFREQ)

        blink_samples, blink_meta = detect_blinks(run_raw, logger)
        task_events, ev_map = mne.events_from_annotations(run_raw, verbose="error")

        # Create per-run QC figures directory
        figures_dir = eeg_dir / "figures"
        figures_dir.mkdir(exist_ok=True)

        def plot_run_blinks(
            raw_run: mne.io.BaseRaw,
            blink_idx: np.ndarray,
            out_dir: Path,
            run_label: str,
            blink_meta: dict[str, object],
            max_examples: int = 3,
        ):
            sfreq = raw_run.info["sfreq"]
            # Choose channels for plotting preferentially
            pref = ["EOG", "FP1", "FP2", "FPZ"]
            channels = [ch for ch in pref if ch in raw_run.ch_names]
            if not channels:
                # take up to 6 EEG channels
                channels = [raw_run.ch_names[i] for i in range(min(6, len(raw_run.ch_names)))]

            # Overview plot: first 30 seconds with blink markers
            try:
                overview_tmax = min(30.0, raw_run.times[-1])
            except Exception:
                overview_tmax = 30.0
            try:
                fig, ax = plt.subplots(len(channels), 1, figsize=(10, 2.2 * len(channels)), sharex=True)
                if len(channels) == 1:
                    ax = [ax]
                data, times = raw_run.copy().pick(channels).crop(tmin=0.0, tmax=overview_tmax).get_data(return_times=True,units={'eeg':'uV','eog':'uV'})
                for i_ch, ch in enumerate(channels):
                    ax[i_ch].plot(times, data[i_ch])
                    ax[i_ch].set_ylabel(f"{ch} (uV)")
                    for bs in blink_idx:
                        t = float(bs / sfreq)
                        if 0.0 <= t <= overview_tmax:
                            ax[i_ch].axvline(t, color="r", alpha=0.6)
                ax[-1].set_xlabel("Time (s)")
                fig.suptitle(f"{subject_id} {run_label} overview (first {overview_tmax}s) - red=blinks")
                # annotate detector metadata
                try:
                    fig.text(
                        0.01,
                        0.99,
                        f"Detector: {blink_meta.get('source','unknown')} channels: {format_channel_list(blink_meta.get('channels',[]))}",
                        ha="left",
                        va="top",
                        fontsize=8,
                    )
                except Exception:
                    pass
                fig.tight_layout(rect=[0, 0.03, 1, 0.95])
                fig_path = out_dir / f"{subject_id}_{run_label}_overview.png"
                fig.savefig(fig_path, dpi=150)
                plt.close(fig)
            except Exception as e:
                logger.warning("Failed to create overview plot for %s: %s", run_label, e)

            # Blink snippets
            if len(blink_idx) == 0:
                # Save a generic snippet from start
                try:
                    fig, ax = plt.subplots(len(channels), 1, figsize=(10, 2.2 * len(channels)), sharex=True)
                    if len(channels) == 1:
                        ax = [ax]
                    data, times = raw_run.copy().pick(channels).crop(tmin=0.0, tmax=min(2.0, raw_run.times[-1])).get_data(return_times=True)
                    for i_ch in range(len(channels)):
                        ax[i_ch].plot(times, data[i_ch])
                        ax[i_ch].set_ylabel(f"{channels[i_ch]} (uV)")
                    ax[-1].set_xlabel("Time (s)")
                    fig.suptitle(f"{subject_id} {run_label} no detected blinks - example segment")
                    try:
                        fig.text(
                            0.01,
                            0.99,
                            f"Detector: {blink_meta.get('source','unknown')} channels: {format_channel_list(blink_meta.get('channels',[]))}",
                            ha="left",
                            va="top",
                            fontsize=8,
                        )
                    except Exception:
                        pass
                    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
                    fig_path = out_dir / f"{subject_id}_{run_label}_no_blinks.png"
                    fig.savefig(fig_path, dpi=150)
                    plt.close(fig)
                except Exception as e:
                    logger.warning("Failed to create no-blink snippet for %s: %s", run_label, e)
                return

            # pick evenly spaced examples across detected blinks
            indices = np.linspace(0, len(blink_idx) - 1, min(max_examples, len(blink_idx))).astype(int)
            half_window = 0.6
            for ii, idx in enumerate(indices, start=1):
                bs = int(blink_idx[idx])
                center = bs / sfreq
                tmin = max(0.0, center - half_window)
                tmax = min(raw_run.times[-1], center + half_window)
                try:
                    fig, ax = plt.subplots(len(channels), 1, figsize=(8, 2.2 * len(channels)), sharex=True)
                    if len(channels) == 1:
                        ax = [ax]
                    data, times = raw_run.copy().pick(channels).crop(tmin=tmin, tmax=tmax).get_data(return_times=True)
                    for i_ch in range(len(channels)):
                        ax[i_ch].plot(times, data[i_ch])
                        # Plot vertical marker relative to the cropped snippet start
                        rel_center = center - tmin
                        ax[i_ch].axvline(rel_center, color="r", linestyle="--")
                        ax[i_ch].set_ylabel(f"{channels[i_ch]} (uV)")
                    ax[-1].set_xlabel("Time (s)")
                    fig.suptitle(f"{subject_id} {run_label} blink example {ii} @ {center:.2f}s")
                    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
                    fig_path = out_dir / f"{subject_id}_{run_label}_blink-{ii}.png"
                    fig.savefig(fig_path, dpi=150)
                    plt.close(fig)
                except Exception as e:
                    logger.warning("Failed to create blink snippet for %s index %d: %s", run_label, idx, e)

            # Overlay of many blink snippets (normalized) for quick amplitude inspection
            try:
                if len(blink_idx) > 0:
                    max_overlay = 30
                    overlay_events = blink_idx[:: max(1, int(max(1, len(blink_idx) / max_overlay)))]
                    overlay_events = overlay_events[:max_overlay]
                    snippets = []
                    half_window = 0.6
                    target_times = np.linspace(-half_window, half_window, int(round(2 * half_window * sfreq)) + 1)
                    first_channel = channels[0]
                    for bs in overlay_events:
                        center = float(bs) / sfreq
                        tmin = max(0.0, center - half_window)
                        tmax = min(raw_run.times[-1], center + half_window)
                        data_seg, times_seg = raw_run.copy().pick([first_channel]).crop(tmin=tmin, tmax=tmax).get_data(return_times=True)
                        ch_data = data_seg[0]
                        rel = times_seg - center
                        # interpolate onto target_times
                        try:
                            interp = np.interp(target_times, rel, ch_data)
                            # normalize by peak-to-peak
                            p2p = np.ptp(interp) if np.ptp(interp) != 0 else 1.0
                            snippets.append(interp / p2p)
                        except Exception:
                            continue
                    if snippets:
                        arr = np.vstack(snippets)
                        mean = arr.mean(axis=0)
                        std = arr.std(axis=0)
                        fig, ax = plt.subplots(1, 1, figsize=(6, 3.5))
                        for row in arr:
                            ax.plot(target_times, row, color="C0", alpha=0.15)
                        ax.plot(target_times, mean, color="C1", lw=2, label="mean (normalized)")
                        ax.fill_between(target_times, mean - std, mean + std, color="C1", alpha=0.25)
                        ax.axvline(0.0, color="r", linestyle="--", label="detected center")
                        ax.set_xlabel("Time (s) relative to detected blink center")
                        ax.set_ylabel(f"Normalized amplitude ({first_channel})")
                        ax.set_title(f"{subject_id} {run_label} blink overlay (n={arr.shape[0]})")
                        try:
                            fig.text(
                                0.01,
                                0.98,
                                f"Detector: {blink_meta.get('source','unknown')} channels: {format_channel_list(blink_meta.get('channels',[]))}",
                                ha="left",
                                va="top",
                                fontsize=8,
                            )
                        except Exception:
                            pass
                        ax.legend(loc="upper right")
                        out_path = out_dir / f"{subject_id}_{run_label}_blink_overlay.png"
                        fig.tight_layout()
                        fig.savefig(out_path, dpi=150)
                        plt.close(fig)
            except Exception as e:
                logger.warning("Failed to create blink overlay for %s: %s", run_label, e)

        # produce QC figures for this run
        try:
            plot_run_blinks(run_raw, blink_samples, figures_dir, run_name, blink_meta, max_examples=3)
        except Exception as e:
            logger.warning("Plotting QC failed for %s: %s", run_name, e)

        for blink_sample in blink_samples:
            blink_rows.append({
                "subject": subject_id,
                "run": run_name,
                "blink_sample": int(blink_sample),
                "blink_time_s": float(blink_sample / run_raw.info["sfreq"]),
                "sfreq_hz": float(run_raw.info["sfreq"]),
                "detector_source": blink_meta["source"],
                "detector_channels": format_channel_list(blink_meta["channels"]),
            })

        run_rows.append({
            "subject": subject_id,
            "run": run_name,
            "sfreq_hz": float(run_raw.info["sfreq"]),
            "n_blinks": int(len(blink_samples)),
            "detector_source": blink_meta["source"],
            "detector_channels": format_channel_list(blink_meta["channels"]),
            "detector_attempts": json.dumps(blink_meta["attempts"], ensure_ascii=True),
            "n_annotations": int(len(task_events)),
        })

        for phase_name, phase_info in PHASES_OF_INTEREST.items():
            trial_part, summary_part = summarize_phase(
                subject_id=subject_id,
                run_name=run_name,
                raw=run_raw,
                blink_samples=blink_samples,
                phase_name=phase_name,
                phase_info=phase_info,
                task_events=task_events,
                ev_map=ev_map,
            )
            trial_rows.extend(trial_part)
            summary_rows.extend(summary_part)

        logger.info(
            "Run %s: %d blinks detected, %d annotated events, %d phase summary rows",
            run_name,
            len(blink_samples),
            len(task_events),
            sum(1 for _ in PHASES_OF_INTEREST if _ in ev_map),
        )

    if not trial_rows:
        logger.warning("No SEBR rows were generated for %s", subject_id)

    trial_df = pd.DataFrame(trial_rows)
    run_df = pd.DataFrame(run_rows)
    blink_df = pd.DataFrame(blink_rows)

    if not summary_rows:
        raise RuntimeError(f"No phase summaries were generated for {subject_id}")

    summary_df = (
        trial_df.groupby(["subject", "phase", "condition"], as_index=False)
        .agg(
            n_trials=("trial_index", "count"),
            total_blinks=("n_blinks", "sum"),
            analyzed_time_s=("duration_s", "sum"),
            mean_blinks_per_trial=("n_blinks", "mean"),
            median_blinks_per_trial=("n_blinks", "median"),
        )
    )
    summary_df["analyzed_time_min"] = summary_df["analyzed_time_s"] / 60.0
    summary_df["sebr_blinks_per_min"] = summary_df["total_blinks"] / summary_df["analyzed_time_min"]
    summary_df["pipeline_version"] = PIPELINE_VERSION

    if not blink_df.empty:
        blink_df = blink_df.sort_values(["run", "blink_sample"], kind="stable").reset_index(drop=True)
    if not trial_df.empty:
        trial_df = trial_df.sort_values(["run", "phase", "trial_index"], kind="stable").reset_index(drop=True)
    if not run_df.empty:
        run_df = run_df.sort_values(["run"], kind="stable").reset_index(drop=True)

    subject_metadata = {
        "pipeline": "compute_SEBR_V2.py",
        "pipeline_version": PIPELINE_VERSION,
        "subject": subject_id,
        "mode": dataset.condition,
        "sampling_frequency_hz": DEFAULT_SFREQ,
        "phase_definitions": PHASES_OF_INTEREST,
        "blink_detection": {
            "preference_order": ["EOG", "frontal_triplet", "frontal_pair", "frontal_left", "frontal_right", "frontal_mid"],
            "l_freq_hz": 1.0,
            "h_freq_hz": 10.0,
            "run_level_detection": True,
        },
        "outputs": {
            "summary_tsv": f"{subject_id}_task-{dataset.condition}_desc-sebr_summary.tsv",
            "trial_tsv": f"{subject_id}_task-{dataset.condition}_desc-sebr_trials.tsv",
            "run_tsv": f"{subject_id}_task-{dataset.condition}_desc-sebr_runs.tsv",
            "blink_tsv": f"{subject_id}_task-{dataset.condition}_desc-sebr_blinks.tsv",
            "log": f"{subject_id}_task-sebr_desc-qc.log",
        },
        "run_count": int(len(subject_runs)),
        "notes": "Blink counting uses a half-open interval [start, end) to avoid edge double-counting.",
    }

    summary_path = eeg_dir / f"{subject_id}_task-{dataset.condition}_desc-sebr_summary.tsv"
    trial_path = eeg_dir / f"{subject_id}_task-{dataset.condition}_desc-sebr_trials.tsv"
    run_path = eeg_dir / f"{subject_id}_task-{dataset.condition}_desc-sebr_runs.tsv"
    blink_path = eeg_dir / f"{subject_id}_task-{dataset.condition}_desc-sebr_blinks.tsv"
    metadata_path = eeg_dir / f"{subject_id}_task-{dataset.condition}_desc-sebr_metadata.json"

    summary_df.to_csv(summary_path, sep="\t", index=False)
    trial_df.to_csv(trial_path, sep="\t", index=False)
    run_df.to_csv(run_path, sep="\t", index=False)
    blink_df.to_csv(blink_path, sep="\t", index=False)
    metadata_path.write_text(json.dumps(subject_metadata, indent=2), encoding="utf-8")

    logger.info("Saved summary to %s", summary_path)
    logger.info("Saved trial table to %s", trial_path)
    logger.info("Saved run table to %s", run_path)
    logger.info("Saved blink table to %s", blink_path)
    logger.info("Saved metadata to %s", metadata_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("overt", "silent"), default="silent")
    parser.add_argument("--data-path", type=Path, default=None)
    parser.add_argument("--derivatives-path", type=Path, default=DEFAULT_DERIVATIVES_PATH)
    parser.add_argument("--subjects", type=int, nargs="+", default=None)
    parser.add_argument("--runs", type=int, nargs="+", default=(1, 2, 3))
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    subjects = list(args.subjects) if args.subjects is not None else list(range(1, 26))
    if args.mode == "overt":
        subjects = [subject for subject in subjects if subject not in {16, 24, 25}]

    dataset = SpellItOut(
        args.mode,
        subjects=subjects,
        runs=list(args.runs),
        data_path=args.data_path,
    )
    output_root = args.derivatives_path / f"task-{args.mode}"
    output_root.mkdir(parents=True, exist_ok=True)

    for subject in tqdm(subjects, desc=f"Mode {args.mode}"):
        process_subject(subject, dataset, output_root)


if __name__ == "__main__":
    main()