"""Stage-oriented preprocessing helpers for Spell-IT-OUT EEG derivatives.

The module deliberately separates run-level evidence generation from
condition-level repair and ICA.  ``build_minimal_run`` is non-destructive with
respect to channel quality: it never interpolates channels, rereferences EEG,
or removes ICA components.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import logging
from pathlib import Path
from typing import Any, Iterable

import mne
import numpy as np
import pandas as pd


QC_CANDIDATE_DESCRIPTIONS = {
    "BLINK",
    "MUSCLE_CANDIDATE",
    "AMPLITUDE_CANDIDATE",
}


@dataclass(frozen=True)
class MinimalPreprocessingConfig:
    """Parameters for the run-level minimal-processing stage."""

    drop_channels: tuple[str, ...] = ("AF7", "AF8")
    resample_sfreq_hz: float = 512.0
    l_freq_hz: float | None = 1.0
    h_freq_hz: float | None = 100.0
    detect_blinks: bool = True
    blink_channels: tuple[str, ...] = ()
    blink_pre_seconds: float = 0.25
    blink_post_seconds: float = 0.50
    detect_muscle_candidates: bool = True
    muscle_zscore_threshold: float = 4.0
    muscle_band_hz: tuple[float, float] = (110.0, 140.0)
    muscle_merge_gap_seconds: float = 0.10
    muscle_min_duration_seconds: float = 0.10
    amplitude_peak_volts: float | None = None


@dataclass(frozen=True)
class PrepQCConfig:
    """Parameters for run-level PREP channel-quality assessment."""

    line_frequency_hz: float = 50.0
    random_state: int = 435656
    reject_by_annotation: str = "omit"
    ransac: bool = True


def _empty_annotations(raw: mne.io.BaseRaw) -> mne.Annotations:
    return mne.Annotations(
        onset=[],
        duration=[],
        description=[],
        orig_time=raw.annotations.orig_time,
    )


def _annotations_from_windows(
    raw: mne.io.BaseRaw,
    onsets: Iterable[float],
    durations: Iterable[float],
    description: str,
) -> mne.Annotations:
    onsets = list(onsets)
    durations = list(durations)
    if not onsets:
        return _empty_annotations(raw)
    return mne.Annotations(
        onset=onsets,
        duration=durations,
        description=[description] * len(onsets),
        orig_time=raw.annotations.orig_time,
    )


def _relabel_annotations(
    raw: mne.io.BaseRaw,
    annotations: mne.Annotations,
    description: str,
) -> mne.Annotations:
    return _annotations_from_windows(
        raw,
        annotations.onset,
        annotations.duration,
        description,
    )


def _merge_annotation_windows(
    raw: mne.io.BaseRaw,
    annotations: mne.Annotations,
    description: str,
    merge_gap_seconds: float,
    min_duration_seconds: float,
) -> mne.Annotations:
    """Merge nearby spans, then retain only spans long enough for continuous QC."""
    if len(annotations) == 0:
        return _empty_annotations(raw)
    windows = sorted(
        (float(onset), float(onset + duration))
        for onset, duration in zip(annotations.onset, annotations.duration)
    )
    merged: list[tuple[float, float]] = []
    current_start, current_end = windows[0]
    for start, end in windows[1:]:
        if start <= current_end + merge_gap_seconds:
            current_end = max(current_end, end)
        else:
            merged.append((current_start, current_end))
            current_start, current_end = start, end
    merged.append((current_start, current_end))
    kept = [
        (start, end)
        for start, end in merged
        if end - start >= min_duration_seconds
    ]
    return _annotations_from_windows(
        raw,
        (start for start, _ in kept),
        (end - start for start, end in kept),
        description,
    )
def detect_qc_candidates(
    raw: mne.io.BaseRaw,
    config: MinimalPreprocessingConfig,
    logger: logging.Logger,
) -> mne.Annotations:
    """Generate reviewable annotations on the original sampling grid.

    Candidate labels intentionally do not begin with ``BAD``.  They therefore
    remain available to PREP and ICA until a reviewer promotes a span to a
    confirmed ``BAD_*`` annotation.
    """
    candidates = _empty_annotations(raw)

    if config.detect_blinks:
        channels = config.blink_channels or ("EOG", "FP1", "FPZ", "FP2")
        if channels:
            events = mne.preprocessing.find_eog_events(
                raw,
                ch_name=channels,
                reject_by_annotation=False,
                verbose="error",
            )
            onsets = (
                (events[:, 0] - raw.first_samp) / raw.info["sfreq"]
                - config.blink_pre_seconds
            )
            onsets = np.clip(onsets, 0.0, None)
            durations = np.repeat(
                config.blink_pre_seconds + config.blink_post_seconds,
                len(onsets),
            )
            candidates += _annotations_from_windows(raw, onsets, durations, "BLINK")
            logger.info("Detected %d blink candidates using %s", len(events), channels)
        else:
            logger.warning("Blink detection skipped: no configured EOG or frontal channels found.")

    if config.detect_muscle_candidates:
        muscle_high = config.muscle_band_hz[1]
        if raw.info["sfreq"] / 2 > muscle_high:
            muscle_annotations, _ = mne.preprocessing.annotate_muscle_zscore(
                raw,
                threshold=config.muscle_zscore_threshold,
                ch_type="eeg",
                filter_freq=config.muscle_band_hz,
                verbose="error",
            )
            muscle_candidates = _merge_annotation_windows(
                raw,
                muscle_annotations,
                "MUSCLE_CANDIDATE",
                config.muscle_merge_gap_seconds,
                config.muscle_min_duration_seconds,
            )
            candidates += muscle_candidates
            logger.info(
                "Detected %d muscle-artifact candidates; retained %d after merging "
                "within %.0f ms and discarding spans under %.0f ms",
                len(muscle_annotations),
                len(muscle_candidates),
                config.muscle_merge_gap_seconds * 1000,
                config.muscle_min_duration_seconds * 1000,
            )
        else:
            logger.warning(
                "Muscle detection skipped: Nyquist frequency %.1f Hz is below %.1f Hz.",
                raw.info["sfreq"] / 2,
                muscle_high,
            )

    if config.amplitude_peak_volts is not None:
        amplitude_result = mne.preprocessing.annotate_amplitude(
            raw,
            peak=config.amplitude_peak_volts,
            picks="eeg",
            verbose="error",
        )
        amplitude_annotations = amplitude_result[0]
        candidates += _relabel_annotations(
            raw,
            amplitude_annotations,
            "AMPLITUDE_CANDIDATE",
        )
        logger.info("Detected %d amplitude candidates", len(amplitude_annotations))

    return candidates


def build_minimal_run(
    raw: mne.io.BaseRaw,
    config: MinimalPreprocessingConfig,
    logger: logging.Logger,
) -> tuple[mne.io.BaseRaw, mne.Annotations]:
    """Create a minimally processed continuous run without channel repair."""
    raw = raw.copy().load_data()
    candidates = detect_qc_candidates(raw, config, logger)
    # to_drop = [channel for channel in config.drop_channels]
    if any([not(channel in raw.ch_names) for channel in config.drop_channels]):
        logger.warning("Some channels in 'drop_channels' are not present in the raw data.")
    if config.drop_channels:
        raw.drop_channels(config.drop_channels)
    raw.resample(config.resample_sfreq_hz, verbose="error")
    if config.l_freq_hz is not None or config.h_freq_hz is not None:
        raw.filter(
            l_freq=config.l_freq_hz,
            h_freq=config.h_freq_hz,
            picks="eeg",
            verbose="error",
        )
    raw.info["description"] = (
        "Minimally processed continuous derivative: channel exclusions, resampling, "
        "and filtering only. No rereferencing, channel interpolation, or ICA removal."
    )
    return raw, candidates


def _candidate_table(candidates: mne.Annotations) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "onset": candidates.onset,
            "duration": candidates.duration,
            "trial_type": candidates.description,
            "annotation_source": "auto",
            "review_status": "unreviewed",
            "reviewed_description": "",
            "reviewer": "",
            "reviewed_at": "",
            "review_notes": "",
        }
    )


def write_minimal_run(
    raw: mne.io.BaseRaw,
    candidates: mne.Annotations,
    output_dir: Path,
    base: str,
    config: MinimalPreprocessingConfig,
    source: dict[str, Any],
    overwrite: bool,
) -> dict[str, Path]:
    """Write the signal checkpoint and canonical annotation-review table."""
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / f"{base}_desc-minproc_raw.fif"
    raw.save(raw_path, overwrite=overwrite)

    candidates_path = output_dir / f"{base}_desc-qc_annotations.tsv"
    _candidate_table(candidates).to_csv(candidates_path, sep="\t", index=False)

    paths = {
        "raw": raw_path,
        "annotations": candidates_path,
    }
    return paths


def ensure_derivative_scaffold(output_root: Path) -> None:
    """Create root-level files documenting this derivative dataset."""
    output_root.mkdir(parents=True, exist_ok=True)
    description_path = output_root / "dataset_description.json"
    if not description_path.exists():
        description_path.write_text(
            json.dumps(
                {
                    "Name": "Spell-IT-OUT preprocessing derivatives",
                    "BIDSVersion": "1.10.0",
                    "DatasetType": "derivative",
                    "GeneratedBy": [
                        {
                            "Name": "spellitout.preprocessing",
                            "Description": "Stage-oriented EEG preprocessing workflow",
                        }
                    ],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    descriptions_path = output_root / "descriptions.tsv"
    if not descriptions_path.exists():
        pd.DataFrame(
            [
                {
                    "desc_id": "minproc",
                    "description": (
                        "Run-level minimal preprocessing: channel exclusions, "
                        "resampling, filtering, and reviewable QC annotations; "
                        "no rereference, interpolation, or ICA."
                    ),
                }
            ]
        ).to_csv(descriptions_path, sep="\t", index=False)


def register_derivative_description(
    output_root: Path,
    desc_id: str,
    description: str,
) -> None:
    """Add or update one documented derivative label at the dataset root."""
    ensure_derivative_scaffold(output_root)
    path = output_root / "descriptions.tsv"
    descriptions = pd.read_csv(path, sep="\t")
    descriptions = descriptions.loc[descriptions["desc_id"] != desc_id]
    descriptions = pd.concat(
        [descriptions, pd.DataFrame([{"desc_id": desc_id, "description": description}])],
        ignore_index=True,
    )
    descriptions.to_csv(path, sep="\t", index=False)


def _pyprep_channel_list(channels: Any) -> list[str]:
    if channels is None:
        return []
    if isinstance(channels, str):
        channels = [channels]
    return sorted({str(channel) for channel in channels})


def _pyprep_detector_report(report: Any) -> dict[str, list[str]]:
    if report is None:
        return {}
    if not isinstance(report, dict):
        return {"bad_all": _pyprep_channel_list(report)}
    normalized = {
        str(method): _pyprep_channel_list(channels)
        for method, channels in report.items()
    }
    if "bad_all" not in normalized:
        normalized["bad_all"] = sorted(
            {
                channel
                for channels in normalized.values()
                for channel in channels
            }
        )
    return dict(sorted(normalized.items()))


def _metric_map(channels: list[str], values: Any) -> dict[str, float | bool | None]:
    values = np.asarray(values)
    if len(channels) != len(values):
        return {channel: None for channel in channels}
    output: dict[str, float | bool | None] = {}
    for channel, value in zip(channels, values):
        if isinstance(value, (bool, np.bool_)):
            output[channel] = bool(value)
        elif np.isfinite(value):
            output[channel] = float(value)
        else:
            output[channel] = None
    return output


def _preinterpolation_metrics(
    prep: Any,
    config: PrepQCConfig,
    noisy_channels_class: Any,
) -> dict[str, Any]:
    """Capture compact, threshold-relevant PREP metrics without redoing RANSAC."""
    raw = prep.raw_eeg.copy()
    raw._data = prep.EEG_before_interpolation.copy()
    raw.info["bads"] = []
    detector = noisy_channels_class(
        raw,
        random_state=config.random_state,
        ransac=False,
        correlation=True,
        reject_by_annotation=config.reject_by_annotation,
    )
    detector.find_all_bads(ransac=False, correlation=True)
    channels = [str(channel) for channel in detector.ch_names_original]
    usable_channels = [channels[index] for index in np.flatnonzero(detector.usable_idx)]
    extra = detector._extra_info
    correlation = extra["bad_by_correlation"]
    psd = extra["bad_by_psd"]
    correlation_median = _metric_map(channels, correlation["median_max_correlations"])
    correlation_fraction = _metric_map(channels, correlation["bad_window_fractions"])
    psd_max_z = _metric_map(channels, psd["psd_zscore"])
    low_z = _metric_map(usable_channels, psd["zscore_low"])
    mid_z = _metric_map(usable_channels, psd["zscore_mid"])
    high_z = _metric_map(usable_channels, psd["zscore_high"])
    excess_band = _metric_map(usable_channels, psd["bad_by_band"])
    one_over_f = _metric_map(usable_channels, psd["bad_by_1f_violation"])
    return {
        "source": (
            "PSD and correlation were recomputed on PREP's rereferenced "
            "EEG_before_interpolation. RANSAC was not recomputed."
        ),
        "correlation": {
            "window_seconds": 1.0,
            "minimum_maximum_correlation": 0.4,
            "bad_window_fraction_threshold": 0.01,
            "per_channel": {
                channel: {
                    "median_maximum_correlation": correlation_median[channel],
                    "bad_window_fraction": correlation_fraction[channel],
                }
                for channel in channels
            },
        },
        "psd": {
            "frequency_bands_hz": {
                "low": [1.0, 15.0],
                "mid": [15.0, 30.0],
                "high": [30.0, 45.0],
            },
            "robust_zscore_threshold": 3.0,
            "per_channel": {
                channel: {
                    "maximum_absolute_band_zscore": psd_max_z[channel],
                    "low_band_zscore": low_z.get(channel),
                    "mid_band_zscore": mid_z.get(channel),
                    "high_band_zscore": high_z.get(channel),
                    "flagged_by_excess_band_power": excess_band.get(channel),
                    "flagged_by_1_over_f_violation": one_over_f.get(channel),
                }
                for channel in channels
            },
        },
    }


def run_prep_channel_qc(
    raw: mne.io.BaseRaw,
    config: PrepQCConfig,
    logger: logging.Logger,
) -> dict[str, Any]:
    """Run PREP per run, collect evidence, and deliberately skip interpolation."""
    import inspect
    from importlib.metadata import version

    from pyprep.find_noisy_channels import NoisyChannels
    from pyprep.prep_pipeline import PrepPipeline

    if "reject_by_annotation" not in inspect.signature(PrepPipeline).parameters:
        raise RuntimeError(
            "Channel QC requires PyPREP >= 0.6 because it uses "
            "reject_by_annotation='omit'."
        )
    if "interpolate_bads" not in inspect.signature(PrepPipeline.robust_reference).parameters:
        raise RuntimeError(
            "Channel QC requires PyPREP >= 0.7 because final interpolation must "
            "be disabled during this evidence-only stage."
        )
    montage = raw.get_montage()
    if montage is None:
        raise RuntimeError("PREP channel QC requires a valid EEG montage.")

    raw = raw.copy().load_data()
    prep_params = {
        "ref_chs": "eeg",
        "reref_chs": "eeg",
        "line_freqs": np.arange(
            config.line_frequency_hz,
            raw.info["sfreq"] / 2,
            config.line_frequency_hz,
        ),
    }
    prep = PrepPipeline(
        raw,
        prep_params,
        montage,
        ransac=config.ransac,
        random_state=config.random_state,
        reject_by_annotation=config.reject_by_annotation,
    )
    prep.remove_line_noise()
    try:
        prep.robust_reference(interpolate_bads=False)
    except ValueError as error:
        logger.error("PREP robust reference failed: %s", error)
        return {
            "status": "failed",
            "failure": str(error),
            "pyprep_version": version("pyprep"),
            "configuration": asdict(config),
            "interpolation_policy": (
                "PREP final interpolation was never reached because robust "
                "referencing failed. No channel decision may be made from this run."
            ),
            "noisy_channels_original": {},
            "noisy_channels_before_interpolation": {},
            "proposed_bad_channels": [],
            "preinterpolation_detector_metrics": {
                "available": False,
                "reason": "Robust reference failed before pre-interpolation metrics existed.",
            },
        }
    before = _pyprep_detector_report(prep.noisy_channels_before_interpolation)
    report = {
        "status": "completed",
        "pyprep_version": version("pyprep"),
        "configuration": asdict(config),
        "interpolation_policy": (
            "PREP final interpolation deliberately skipped. These flags are "
            "evidence for review, not automatic channel decisions."
        ),
        "noisy_channels_original": _pyprep_detector_report(prep.noisy_channels_original),
        "noisy_channels_before_interpolation": before,
        "proposed_bad_channels": before.get("bad_all", []),
        "preinterpolation_detector_metrics": _preinterpolation_metrics(
            prep,
            config,
            NoisyChannels,
        ),
    }
    logger.info("PREP proposed channels for review: %s", report["proposed_bad_channels"])
    return report


def _annotation_summary(raw: mne.io.BaseRaw) -> dict[str, Any]:
    descriptions = [str(description) for description in raw.annotations.description]
    bad_indices = [
        index
        for index, description in enumerate(descriptions)
        if description.lower().startswith("bad")
    ]
    return {
        "n_annotations": len(raw.annotations),
        "n_bad_annotations": len(bad_indices),
        "bad_duration_seconds": float(sum(raw.annotations.duration[index] for index in bad_indices)),
        "descriptions": sorted(set(descriptions)),
    }


def prep_qc_channel_table(raw: mne.io.BaseRaw, report: dict[str, Any]) -> pd.DataFrame:
    """Return one auditable review row per channel without auto-rejecting it."""
    by_method = report["noisy_channels_before_interpolation"]
    metrics = report["preinterpolation_detector_metrics"]
    correlation = metrics.get("correlation", {}).get("per_channel", {})
    psd = metrics.get("psd", {}).get("per_channel", {})
    rows = []
    for index, channel in enumerate(raw.ch_names):
        methods = sorted(
            method
            for method, channels in by_method.items()
            if method not in {"bad_all", "bad_by_manual"} and channel in channels
        )
        rows.append(
            {
                "channel": channel,
                "channel_type": mne.channel_type(raw.info, index).upper(),
                "pyprep_status": report["status"],
                "pyprep_version": report.get("pyprep_version", ""),
                "automatic_status": (
                    "qc_failed"
                    if report["status"] == "failed"
                    else "flagged" if methods else "not_flagged"
                ),
                "automatic_methods": ";".join(methods),
                "automatic_proposed_bad": channel in report.get("proposed_bad_channels", []),
                "correlation_bad_window_fraction": correlation.get(channel, {}).get(
                    "bad_window_fraction"
                ),
                "correlation_median_maximum": correlation.get(channel, {}).get(
                    "median_maximum_correlation"
                ),
                "psd_low_band_zscore": psd.get(channel, {}).get("low_band_zscore"),
                "psd_mid_band_zscore": psd.get(channel, {}).get("mid_band_zscore"),
                "psd_high_band_zscore": psd.get(channel, {}).get("high_band_zscore"),
                "psd_1_over_f_violation": psd.get(channel, {}).get(
                    "flagged_by_1_over_f_violation"
                ),
                "manual_decision": "unreviewed",
                "final_decision": "unreviewed",
                "decision_reason": "",
                "reviewer": "",
                "reviewed_at": "",
                "review_notes": "",
            }
        )
    return pd.DataFrame(rows)


def write_prep_qc(
    raw: mne.io.BaseRaw,
    report: dict[str, Any],
    output_dir: Path,
    base: str,
    source_minproc: Path,
    overwrite: bool,
) -> dict[str, Path]:
    """Write the canonical, editable channel-review table for one run."""
    output_dir.mkdir(parents=True, exist_ok=True)
    channels_path = output_dir / f"{base}_desc-channelqc.tsv"
    if not overwrite and channels_path.exists():
        raise FileExistsError(f"QC output exists for {base}; pass overwrite=True to replace it.")
    table = prep_qc_channel_table(raw, report)
    table.insert(0, "source_minproc", str(source_minproc))
    table.insert(1, "pyprep_random_state", report.get("configuration", {}).get("random_state"))
    table.insert(
        2,
        "pyprep_reject_by_annotation",
        report.get("configuration", {}).get("reject_by_annotation", ""),
    )
    table.insert(3, "pyprep_ransac", report.get("configuration", {}).get("ransac"))
    table.to_csv(channels_path, sep="\t", index=False)
    return {"channels": channels_path}
