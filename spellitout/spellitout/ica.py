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
    n_mixtures: int = 3
    learning_rate: float = 0.05
    do_newton: bool = True
    max_unmixing_condition: float = 1e6


@dataclass(frozen=True)
class InfomaxConfig:
    """Choices affecting the Extended Infomax decomposition."""

    line_noise_method: str = "notch"
    line_frequency_hz: float = 50.0
    max_iter: int = 2000
    w_change: float = 1e-12
    random_state: int = 435656
    rank_tolerance: float = 1e-7
    max_unmixing_condition: float = 1e6


class _InfomaxStepCapture(logging.Handler):
    """Keep the final unrounded Infomax step diagnostic in memory."""

    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self.last: dict[str, int | float] | None = None

    def emit(self, record: logging.LogRecord) -> None:
        if not (
            isinstance(record.msg, str)
            and record.msg.startswith("step %d - lrate")
            and isinstance(record.args, tuple)
            and len(record.args) == 4
        ):
            return
        step, learning_rate, weight_change, angle = record.args
        self.last = {
            "iterations_executed": int(step),
            "final_learning_rate": float(learning_rate),
            "final_w_change": float(weight_change),
            "final_angle_deg": float(angle),
        }


def _infomax_convergence_diagnostics(
    config: InfomaxConfig,
    reported_iterations: int,
    last_step: dict[str, int | float] | None,
) -> dict[str, Any]:
    """Interpret MNE Infomax termination without trusting ``n_iter_`` alone."""
    if last_step is None:
        return {
            "iterations_executed": None,
            "final_learning_rate": None,
            "final_w_change": None,
            "final_angle_deg": None,
            "stopped_before_max_iter": None,
            "converged": None,
            "convergence_status": "diagnostics_unavailable",
            "convergence_note": "No Infomax step diagnostic was captured.",
        }

    diagnostics = dict(last_step)
    executed = int(diagnostics["iterations_executed"])
    final_change = float(diagnostics["final_w_change"])
    stopped_early = executed < config.max_iter
    change_met = executed > 2 and final_change < config.w_change

    if stopped_early and reported_iterations == config.max_iter:
        # MNE assigns step=max_iter when w_change is met, even when the actual
        # step is much smaller. The captured log retains the real step.
        status = "w_change"
        converged = True
    elif stopped_early:
        # MNE shortens its local max_iter after enough successive small-angle
        # updates. w_change may happen to be satisfied on the same final step.
        status = "small_angle_and_w_change" if change_met else "small_angle"
        converged = True
    elif change_met:
        status = "w_change_at_max_iter"
        converged = True
    else:
        status = "max_iter_reached"
        converged = False

    diagnostics.update(
        {
            "stopped_before_max_iter": stopped_early,
            "converged": converged,
            "convergence_status": status,
            "convergence_note": (
                "iterations_executed and final optimization values were captured "
                "from the MNE Infomax step logger before its ambiguous n_iter_ "
                "conversion."
            ),
        }
    )
    return diagnostics


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


def remove_line_noise(
    raw: mne.io.BaseRaw,
    config: AMICAConfig | InfomaxConfig,
) -> mne.io.BaseRaw:
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
    config: AMICAConfig | InfomaxConfig,
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
        "Prepared ICA input: %d run(s), %d interpolated channel(s), %d excluded channel(s)",
        len(prepared), len(interpolated), len(excluded),
    )
    return pooled, interpolated, excluded, plan


def prepare_ica_input(
    runs: dict[str, mne.io.BaseRaw],
    annotations: dict[str, pd.DataFrame],
    channel_tables: dict[str, pd.DataFrame],
    config: AMICAConfig | InfomaxConfig,
    logger: logging.Logger,
) -> tuple[mne.io.BaseRaw, list[str], list[str], pd.DataFrame]:
    """Prepare the condition-level input shared by ICA estimators."""
    return prepare_amica_input(runs, annotations, channel_tables, config, logger)


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
        n_mixtures=config.n_mixtures,
        lrate=config.learning_rate,
        do_newton=config.do_newton,
        verbose=2,
    )
    model.fit(data)
    ica = model.to_mne(info=eeg.info)
    singular_values = np.linalg.svd(ica.unmixing_matrix_, compute_uv=False)
    unmixing_rank = np.linalg.matrix_rank(ica.unmixing_matrix_)
    unmixing_condition = float(singular_values[0] / singular_values[-1])
    if unmixing_rank != n_components:
        raise RuntimeError(
            "AMICA produced a numerically singular decomposition "
            f"(unmixing rank {unmixing_rank}/{n_components}). "
            "No AMICA ICA/component outputs were written; retry Stage 4 with a different "
            "random seed or device."
        )
    if not np.isfinite(unmixing_condition) or unmixing_condition > config.max_unmixing_condition:
        raise RuntimeError(
            "AMICA produced an ill-conditioned decomposition "
            f"(condition number {unmixing_condition:.3e}; maximum allowed "
            f"{config.max_unmixing_condition:.3e}). No AMICA ICA/component "
            "outputs were written; retry Stage 4 with different optimisation settings."
        )
    return model, ica, n_components, rank_estimate, len(data), unmixing_condition


def fit_extended_infomax(raw: mne.io.BaseRaw, config: InfomaxConfig):
    """Fit Extended Infomax, omitting confirmed BAD spans only at fit time."""
    eeg = raw.copy().pick("eeg")
    rank_estimate = mne.compute_rank(
        eeg,
        tol=config.rank_tolerance,
        tol_kind="relative",
    )["eeg"]
    # The average reference applied during input preparation removes one EEG
    # dimension, even when the numerical rank estimate is overly optimistic.
    rank_cap = len(eeg.ch_names) - 1
    n_components = min(rank_estimate, rank_cap)
    if n_components < 2:
        raise RuntimeError("Extended Infomax requires at least two effective EEG dimensions.")
    n_fit_samples = eeg.get_data(reject_by_annotation="omit").shape[1]
    if not n_fit_samples:
        raise RuntimeError(
            "No samples remain for Extended Infomax after omitting reviewed BAD_* spans."
        )
    ica = mne.preprocessing.ICA(
        n_components=n_components,
        method="infomax",
        fit_params={
            "extended": True,
            "w_change": config.w_change,
            # Required for MNE to emit the per-step values captured below.
            "verbose": True,
        },
        max_iter=config.max_iter,
        random_state=config.random_state,
    )
    capture = _InfomaxStepCapture()
    mne_logger = mne.utils.logger
    existing_handlers = tuple(mne_logger.handlers)
    previous_handler_levels = {handler: handler.level for handler in existing_handlers}
    mne_logger.addHandler(capture)
    for handler in existing_handlers:
        handler.setLevel(logging.WARNING)
    try:
        ica.fit(eeg, reject_by_annotation=True, verbose="info")
    finally:
        mne_logger.removeHandler(capture)
        for handler, level in previous_handler_levels.items():
            handler.setLevel(level)
    convergence = _infomax_convergence_diagnostics(
        config,
        int(ica.n_iter_),
        capture.last,
    )
    singular_values = np.linalg.svd(ica.unmixing_matrix_, compute_uv=False)
    unmixing_rank = np.linalg.matrix_rank(ica.unmixing_matrix_)
    unmixing_condition = float(singular_values[0] / singular_values[-1])
    if unmixing_rank != n_components:
        raise RuntimeError(
            "Extended Infomax produced a numerically singular decomposition "
            f"(unmixing rank {unmixing_rank}/{n_components}). No ICA/component "
            "outputs were written; retry Stage 4 with another random seed."
        )
    if not np.isfinite(unmixing_condition) or unmixing_condition > config.max_unmixing_condition:
        raise RuntimeError(
            "Extended Infomax produced an ill-conditioned decomposition "
            f"(condition number {unmixing_condition:.3e}; maximum allowed "
            f"{config.max_unmixing_condition:.3e}). No ICA/component outputs were written."
        )
    return (
        ica,
        n_components,
        rank_estimate,
        n_fit_samples,
        unmixing_condition,
        convergence,
    )


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
    # For component k, var(a[:, k] * s[k, :]) averaged over channels is
    # exactly var(s[k, :]) * mean(a[:, k] ** 2). Computing this separable
    # expression avoids materialising a channels-by-samples reconstruction for
    # every component.
    source_variances = np.var(sources, axis=1, ddof=1)
    mean_squared_mixing = np.mean(np.square(mixing), axis=0)
    energies = source_variances * mean_squared_mixing
    total = float(energies.sum())
    if not np.isfinite(total) or total <= 0:
        raise RuntimeError("Could not compute positive component energies for ICA output.")
    return energies / total


AUTO_BRAIN_THRESHOLD = 0.60
AUTO_ARTIFACT_THRESHOLD = 0.80
AUTO_CHANNEL_REVIEW_THRESHOLD = 0.70
AUTO_CONTRIBUTION_THRESHOLD = 0.004
AUTO_REJECT_CLASSES = (
    "muscle artifact",
    "eye blink",
    "heart beat",
    "line noise",
)


def automatic_component_decisions(table: pd.DataFrame) -> pd.DataFrame:
    """Apply the fixed, reviewer-independent ICA component policy.

    High-confidence artifacts take precedence over automatic keeps. Channel
    noise above its threshold is deliberately forced to manual review because
    a globally small contribution can still be large in one channel.
    """
    required = {
        "brain",
        "channel noise",
        "variance_contribution_good_samples",
        *AUTO_REJECT_CLASSES,
    }
    missing = sorted(required.difference(table.columns))
    if missing:
        raise ValueError(
            "Cannot apply automatic ICA policy; missing columns: "
            + ", ".join(missing)
        )
    evidence = table[list(required)].apply(pd.to_numeric, errors="coerce")
    if evidence.isna().any().any():
        invalid = sorted(evidence.columns[evidence.isna().any()].tolist())
        raise ValueError(
            "Cannot apply automatic ICA policy; non-numeric evidence in: "
            + ", ".join(invalid)
        )

    artifact_probabilities = evidence[list(AUTO_REJECT_CLASSES)]
    reject = artifact_probabilities.ge(AUTO_ARTIFACT_THRESHOLD).any(axis=1)
    force_review = evidence["channel noise"].ge(AUTO_CHANNEL_REVIEW_THRESHOLD)
    brain_keep = evidence["brain"].ge(AUTO_BRAIN_THRESHOLD)
    low_contribution_keep = evidence[
        "variance_contribution_good_samples"
    ].lt(AUTO_CONTRIBUTION_THRESHOLD)
    keep = (brain_keep | low_contribution_keep) & ~reject & ~force_review

    result = pd.DataFrame(index=table.index)
    result["final_decision"] = "unreviewed"
    result["decision_source"] = "unreviewed"
    result["decision_rule"] = ""
    result.loc[reject, ["final_decision", "decision_source"]] = (
        "reject",
        "automatic",
    )
    for component_index in result.index[reject]:
        artifact_class = artifact_probabilities.loc[component_index].idxmax()
        result.loc[component_index, "decision_rule"] = (
            f"{artifact_class} >= {AUTO_ARTIFACT_THRESHOLD:.0%}"
        )
    result.loc[keep, ["final_decision", "decision_source"]] = (
        "keep",
        "automatic",
    )
    result.loc[keep & brain_keep, "decision_rule"] = (
        f"brain >= {AUTO_BRAIN_THRESHOLD:.0%}"
    )
    result.loc[keep & ~brain_keep & low_contribution_keep, "decision_rule"] = (
        "good-sample contribution < "
        f"{AUTO_CONTRIBUTION_THRESHOLD:.1%}"
    )
    return result


def component_review_table(raw: mne.io.BaseRaw, ica: mne.preprocessing.ICA) -> pd.DataFrame:
    """Create component evidence and editable decisions after an ICA fit."""
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
    source_raw = ica.get_sources(eeg)
    contributions = component_energy_fractions(
        ica.get_components(), source_raw.get_data()
    )
    good_sample_contributions = component_energy_fractions(
        ica.get_components(),
        source_raw.get_data(reject_by_annotation="omit"),
    )
    labels.insert(
        1,
        "variance_contribution",
        contributions,
    )
    labels.insert(
        2,
        "variance_contribution_good_samples",
        good_sample_contributions,
    )
    labels["predicted_label"] = labels[list(ICLABEL_CLASSES)].idxmax(axis=1)
    labels["predicted_probability"] = labels.apply(
        lambda row: float(row[row["predicted_label"]]), axis=1
    )
    labels["manual_decision"] = "unreviewed"
    labels["final_decision"] = "unreviewed"
    labels["decision_source"] = "unreviewed"
    labels["decision_rule"] = ""
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
    unmixing_condition: float,
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
        "amica": {
            **asdict(config),
            "n_components": int(len(components)),
            "mne_rank_estimate": rank_estimate,
            "n_fit_samples": n_fit_samples,
            "unmixing_condition": unmixing_condition,
        },
    }
    paths["metadata"].write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return paths


def write_infomax_outputs(
    output_dir: Path,
    base: str,
    pooled: mne.io.BaseRaw,
    ica: mne.preprocessing.ICA,
    plan: pd.DataFrame,
    config: InfomaxConfig,
    source_runs: Iterable[str],
    interpolated: list[str],
    excluded: list[str],
    rank_estimate: int,
    n_fit_samples: int,
    unmixing_condition: float,
    convergence: dict[str, Any],
    overwrite: bool,
    write_ica_input: bool = True,
) -> dict[str, Path]:
    """Write Extended Infomax equivalents of all Stage 4 AMICA outputs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "ica_input": output_dir / f"{base}_desc-infomax-input_raw.fif",
        "channels": output_dir / f"{base}_desc-infomax-input_channels.tsv",
        "ica": output_dir / f"{base}_desc-infomax_ica.fif",
        "components": output_dir / f"{base}_desc-infomax_components.tsv",
        "metadata": output_dir / f"{base}_desc-infomax.json",
    }
    targets = paths.values() if write_ica_input else [
        path for key, path in paths.items() if key != "ica_input"
    ]
    if not overwrite and any(path.exists() for path in targets):
        raise FileExistsError(
            f"Extended Infomax output exists for {base}; pass overwrite=True to replace it."
        )
    if write_ica_input:
        pooled.save(paths["ica_input"], overwrite=overwrite)
    plan.to_csv(paths["channels"], sep="\t", index=False)
    ica.save(paths["ica"], overwrite=overwrite)
    components = component_review_table(pooled, ica)
    components.to_csv(paths["components"], sep="\t", index=False)
    metadata = {
        "schema_version": 1,
        "source_runs": list(source_runs),
        "line_noise": {
            "method": config.line_noise_method,
            "frequency_hz": config.line_frequency_hz,
        },
        "approved_bad_span_policy": (
            "Omitted from Extended Infomax fitting; retained as annotations in the "
            "Infomax-input FIF."
        ),
        "condition_interpolated_channels": interpolated,
        "condition_excluded_channels": excluded,
        "reference": "average EEG reference after condition-level channel harmonization",
        "infomax": {
            **asdict(config),
            "method": "infomax",
            "extended": True,
            "n_components": int(len(components)),
            "mne_rank_estimate": rank_estimate,
            "n_fit_samples": n_fit_samples,
            "n_iterations": int(ica.n_iter_),
            **convergence,
            "unmixing_condition": unmixing_condition,
        },
    }
    paths["metadata"].write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return paths
