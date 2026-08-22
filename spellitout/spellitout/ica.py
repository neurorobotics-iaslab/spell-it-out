"""Build reviewed AMICA input and fit a single condition-level decomposition."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import logging
from pathlib import Path
from typing import Any, Iterable

import mne
import numpy as np
import pandas as pd

from .review_qc import approved_bad_annotations, normalise_annotation_table, normalise_channel_table


ICLABEL_CLASSES = (
    "brain",
    "muscle artifact",
    "eye blink",
    "heart beat",
    "line noise",
    "channel noise",
    "other",
)


@dataclass(frozen=True)
class AMICAConfig:
    """Explicit choices affecting the decomposition, not the immutable minproc FIF."""

    line_noise_method: str = "notch"
    line_frequency_hz: float = 50.0
    max_iter: int = 2000
    device: str = "cuda"
    random_state: int = 435656


def _require_review_complete(annotations: pd.DataFrame, channels: pd.DataFrame, run: str) -> None:
    unreviewed_annotations = annotations.index[annotations["review_status"] == "unreviewed"].tolist()
    eeg = channels.loc[channels["channel_type"].str.upper() == "EEG"]
    unreviewed_channels = eeg.loc[eeg["final_decision"] == "unreviewed", "channel"].tolist()
    if unreviewed_annotations or unreviewed_channels:
        details = []
        if unreviewed_annotations:
            details.append(f"{len(unreviewed_annotations)} annotation(s)")
        if unreviewed_channels:
            details.append(f"channels {', '.join(unreviewed_channels)}")
        raise RuntimeError(f"Review is incomplete for {run}: " + "; ".join(details))


def reviewed_run_decisions(
    annotations_path: Path,
    channels_path: Path,
    run: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read and validate the two canonical review tables for a run."""
    annotations = normalise_annotation_table(pd.read_csv(annotations_path, sep="\t"))
    channels = normalise_channel_table(pd.read_csv(channels_path, sep="\t"))
    _require_review_complete(annotations, channels, run)
    return annotations, channels


def add_approved_bad_annotations(raw: mne.io.BaseRaw, annotations: pd.DataFrame) -> mne.io.BaseRaw:
    """Attach only confirmed global artifacts; candidates and blinks remain usable."""
    raw = raw.copy()
    approved = approved_bad_annotations(annotations)
    if approved.empty:
        return raw
    bad = mne.Annotations(
        onset=approved["onset"].to_numpy(float),
        duration=approved["duration"].to_numpy(float),
        description=approved["reviewed_description"].to_numpy(str),
        orig_time=raw.annotations.orig_time,
    )
    raw.set_annotations(raw.annotations + bad)
    return raw


def _line_frequencies(raw: mne.io.BaseRaw, line_frequency_hz: float) -> list[float]:
    limit = min(float(raw.info["lowpass"]), raw.info["sfreq"] / 2)
    return [
        float(frequency)
        for frequency in np.arange(line_frequency_hz, limit, line_frequency_hz)
    ]


def remove_line_noise(raw: mne.io.BaseRaw, config: AMICAConfig) -> mne.io.BaseRaw:
    """Remove line noise without altering channel decisions or applying PREP reference."""
    raw = raw.copy().load_data()
    frequencies = _line_frequencies(raw, config.line_frequency_hz)
    if not frequencies:
        return raw
    if config.line_noise_method == "notch":
        raw.notch_filter(freqs=frequencies, picks="eeg", verbose="error")
        return raw
    if config.line_noise_method != "prep":
        raise ValueError("line_noise_method must be 'notch' or 'prep'.")

    from pyprep.prep_pipeline import PrepPipeline

    montage = raw.get_montage()
    if montage is None:
        raise RuntimeError("PREP line-noise removal requires a valid EEG montage.")
    prep = PrepPipeline(
        raw,
        {"ref_chs": "eeg", "reref_chs": "eeg", "line_freqs": frequencies},
        montage,
        ransac=False,
        random_state=435656,
        reject_by_annotation="omit",
    )
    prep.remove_line_noise()
    eeg_names = raw.copy().pick("eeg").ch_names
    cleaned = prep.raw_eeg.get_data(picks=eeg_names)
    raw._data[mne.pick_channels(raw.ch_names, eeg_names)] = cleaned
    return raw


def condition_channel_plan(channel_tables: dict[str, pd.DataFrame]) -> tuple[list[str], list[str], pd.DataFrame]:
    """Create stable condition-wide interpolation/exclusion sets from reviewed runs."""
    interpolate: dict[str, list[str]] = {}
    exclude: dict[str, list[str]] = {}
    for run, table in channel_tables.items():
        eeg = table.loc[table["channel_type"].str.upper() == "EEG"]
        for channel in eeg.loc[eeg["final_decision"] == "interpolate", "channel"]:
            interpolate.setdefault(channel, []).append(run)
        for channel in eeg.loc[eeg["final_decision"] == "exclude_from_ica", "channel"]:
            exclude.setdefault(channel, []).append(run)
    excluded = sorted(exclude)
    interpolated = sorted(set(interpolate).difference(excluded))
    rows = []
    for channel in sorted(set(interpolate) | set(exclude)):
        rows.append(
            {
                "channel": channel,
                "condition_decision": "exclude_from_ica" if channel in exclude else "interpolate",
                "interpolate_triggering_runs": ";".join(sorted(interpolate.get(channel, []))),
                "exclude_triggering_runs": ";".join(sorted(exclude.get(channel, []))),
            }
        )
    return interpolated, excluded, pd.DataFrame(rows)


def prepare_amica_input(
    runs: dict[str, mne.io.BaseRaw],
    annotations: dict[str, pd.DataFrame],
    channel_tables: dict[str, pd.DataFrame],
    config: AMICAConfig,
    logger: logging.Logger,
) -> tuple[mne.io.BaseRaw, list[str], list[str], pd.DataFrame]:
    """Apply reviewed decisions, harmonize sensors, reference, then concatenate."""
    interpolated, excluded, plan = condition_channel_plan(channel_tables)
    prepared = []
    for run, raw in runs.items():
        current = add_approved_bad_annotations(raw, annotations[run])
        current = remove_line_noise(current, config)
        missing = sorted(set(interpolated + excluded).difference(current.ch_names))
        if missing:
            raise RuntimeError(f"{run} lacks condition-level decision channel(s): {', '.join(missing)}")
        if excluded:
            current.drop_channels(excluded)
        if interpolated:
            current.info["bads"] = interpolated
            current.interpolate_bads(reset_bads=True, verbose="error")
        current.set_eeg_reference("average", projection=False, verbose="error")
        prepared.append(current)
    pooled = mne.concatenate_raws(prepared)
    logger.info(
        "Prepared AMICA input: %d run(s), %d interpolated channel(s), %d excluded channel(s)",
        len(prepared), len(interpolated), len(excluded),
    )
    return pooled, interpolated, excluded, plan


def fit_amica(raw: mne.io.BaseRaw, config: AMICAConfig):
    """Fit AMICA on EEG samples, omitting confirmed BAD spans only at fit time."""
    eeg = raw.copy().pick("eeg")
    rank_estimate = mne.compute_rank(eeg, tol="auto", tol_kind="absolute")["eeg"]
    rank_cap = len(eeg.ch_names) - 1
    n_components = min(rank_estimate, rank_cap)
    if n_components < 2:
        raise RuntimeError("AMICA requires at least two effective EEG dimensions.")
    data = eeg.get_data(units="uV", reject_by_annotation="omit").T
    if not len(data):
        raise RuntimeError("No samples remain for AMICA after omitting reviewed BAD_* spans.")
    from amica import AMICA

    model = AMICA(
        device=config.device,
        n_components=n_components,
        max_iter=config.max_iter,
        random_state=config.random_state,
        verbose=2,
    )
    model.fit(data)
    ica = model.to_mne(info=eeg.info)
    unmixing_rank = np.linalg.matrix_rank(ica.unmixing_matrix_)
    if unmixing_rank != n_components:
        raise RuntimeError(
            "AMICA produced a numerically singular decomposition "
            f"(unmixing rank {unmixing_rank}/{n_components}). "
            "No AMICA ICA/component outputs were written; retry Stage 4 with a different "
            "random seed or device."
        )
    return model, ica, n_components, rank_estimate, len(data)


def component_energy_fractions(mixing: np.ndarray, sources: np.ndarray) -> np.ndarray:
    """Compute normalized sensor-space energy contributed by each IC.

    This is the MATLAB-like relative contribution metric: each component's
    reconstructed sensor signal is assigned a variance/energy, then the
    energies are normalized across components. It is not MNE's PVAF metric.
    """
    mixing = np.asarray(mixing, dtype=float)
    sources = np.asarray(sources, dtype=float)
    if mixing.ndim != 2 or sources.ndim != 2:
        raise ValueError("mixing must be channels x components and sources components x samples")
    if mixing.shape[1] != sources.shape[0]:
        raise ValueError("mixing and sources contain different numbers of components")
    energies = np.empty(sources.shape[0], dtype=float)
    for component in range(sources.shape[0]):
        contribution = mixing[:, component, None] * sources[component][None, :]
        energies[component] = float(np.var(contribution, axis=1, ddof=1).mean())
    total = float(energies.sum())
    if not np.isfinite(total) or total <= 0:
        raise RuntimeError("Could not compute positive component energies for AMICA output.")
    return energies / total


def component_review_table(raw: mne.io.BaseRaw, ica: mne.preprocessing.ICA) -> pd.DataFrame:
    """Create component evidence and editable decisions after AMICA fit."""
    try:
        import torch
        from mne_icalabel.iclabel import iclabel_label_components
    except ImportError as error:
        raise RuntimeError(
            "ICLabel is required for Stage 4. Install mne-icalabel and PyTorch."
        ) from error
    torch.set_default_dtype(torch.float32)
    probabilities = iclabel_label_components(raw, ica, inplace=True, backend="torch")
    labels = pd.DataFrame(probabilities, columns=ICLABEL_CLASSES)
    labels.insert(0, "component", np.arange(len(labels)))
    eeg = raw.copy().pick("eeg")
    sources = ica.get_sources(eeg).get_data(reject_by_annotation="omit")
    contributions = component_energy_fractions(ica.get_components(), sources)
    labels.insert(
        1,
        "variance_contribution",
        contributions,
    )
    labels["predicted_label"] = labels[list(ICLABEL_CLASSES)].idxmax(axis=1)
    labels["predicted_probability"] = labels.apply(
        lambda row: float(row[row["predicted_label"]]), axis=1
    )
    labels["manual_decision"] = "unreviewed"
    labels["final_decision"] = "unreviewed"
    labels["reviewer"] = ""
    labels["reviewed_at"] = ""
    labels["review_notes"] = ""
    return labels


def write_amica_outputs(
    output_dir: Path,
    base: str,
    pooled: mne.io.BaseRaw,
    ica: mne.preprocessing.ICA,
    plan: pd.DataFrame,
    config: AMICAConfig,
    source_runs: Iterable[str],
    interpolated: list[str],
    excluded: list[str],
    rank_estimate: int,
    n_fit_samples: int,
    overwrite: bool,
    write_ica_input: bool = True,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "ica_input": output_dir / f"{base}_desc-ica-input_raw.fif",
        "channels": output_dir / f"{base}_desc-ica-input_channels.tsv",
        "ica": output_dir / f"{base}_desc-amica_ica.fif",
        "components": output_dir / f"{base}_desc-amica_components.tsv",
        "metadata": output_dir / f"{base}_desc-amica.json",
    }
    targets = paths.values() if write_ica_input else [
        path for key, path in paths.items() if key != "ica_input"
    ]
    if not overwrite and any(path.exists() for path in targets):
        raise FileExistsError(f"AMICA output exists for {base}; pass overwrite=True to replace it.")
    if write_ica_input:
        pooled.save(paths["ica_input"], overwrite=overwrite)
    plan.to_csv(paths["channels"], sep="\t", index=False)
    ica.save(paths["ica"], overwrite=overwrite)
    components = component_review_table(pooled, ica)
    components.to_csv(paths["components"], sep="\t", index=False)
    metadata = {
        "schema_version": 1,
        "source_runs": list(source_runs),
        "line_noise": {"method": config.line_noise_method, "frequency_hz": config.line_frequency_hz},
        "approved_bad_span_policy": "Omitted from AMICA fitting; retained as annotations in the ICA-input FIF.",
        "condition_interpolated_channels": interpolated,
        "condition_excluded_channels": excluded,
        "reference": "average EEG reference after condition-level channel harmonization",
        "amica": {**asdict(config), "n_components": int(len(components)), "mne_rank_estimate": rank_estimate, "n_fit_samples": n_fit_samples},
    }
    paths["metadata"].write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return paths
