import logging
from pathlib import Path
from types import SimpleNamespace

import mne
import numpy as np
import pandas as pd

from spellitout.ica import (
    InfomaxConfig,
    _infomax_convergence_diagnostics,
    automatic_component_decisions,
    component_energy_fractions,
    condition_channel_plan,
)
from spellitout.erd_ers import ERDERSConfig, compute_subject_tfr, epoch_badness
from spellitout.blink_rate import _reviewed_blinks, enforce_minimum_blink_separation

from spellitout.preprocessing import (
    MinimalPreprocessingConfig,
    PrepQCConfig,
    _merge_annotation_windows,
    build_minimal_run,
    ensure_derivative_scaffold,
    run_prep_channel_qc,
    write_prep_qc,
    write_minimal_run,
)
from spellitout.review_qc import (
    QCReviewApplication,
    ReviewRecord,
    approved_bad_annotations,
    read_review_tables,
    write_review_tables,
)


def test_infomax_convergence_uses_captured_iteration_not_ambiguous_mne_count():
    config = InfomaxConfig(max_iter=100, w_change=1e-8)
    common = {
        "final_learning_rate": 1e-4,
        "final_angle_deg": 70.0,
    }

    stopped_by_change = _infomax_convergence_diagnostics(
        config,
        reported_iterations=100,
        last_step={
            **common,
            "iterations_executed": 25,
            "final_w_change": 1e-9,
        },
    )
    hit_limit = _infomax_convergence_diagnostics(
        config,
        reported_iterations=100,
        last_step={
            **common,
            "iterations_executed": 100,
            "final_w_change": 1e-5,
        },
    )

    assert stopped_by_change["converged"] is True
    assert stopped_by_change["convergence_status"] == "w_change"
    assert hit_limit["converged"] is False
    assert hit_limit["convergence_status"] == "max_iter_reached"


def test_qc_shortcut_is_not_blocked_by_windows_keyboard_state():
    application = QCReviewApplication.__new__(QCReviewApplication)
    calls = []

    result = application.run_shortcut(
        SimpleNamespace(
            state=0x8,
            widget=SimpleNamespace(winfo_class=lambda: "Treeview"),
        ),
        lambda: calls.append("invoked"),
    )

    assert calls == ["invoked"]
    assert result == "break"


def test_qc_shortcut_is_disabled_while_editing_text():
    application = QCReviewApplication.__new__(QCReviewApplication)
    calls = []

    result = application.run_shortcut(
        SimpleNamespace(
            state=0,
            widget=SimpleNamespace(winfo_class=lambda: "TEntry"),
        ),
        lambda: calls.append("invoked"),
    )

    assert calls == []
    assert result is None


def test_erd_epoch_badness_unions_overlapping_bad_spans():
    raw = mne.io.RawArray(
        np.zeros((1, 1000)),
        mne.create_info(["C3"], sfreq=100, ch_types="eeg"),
        verbose="error",
    )
    raw.set_annotations(
        mne.Annotations(
            onset=[1.0, 1.05],
            duration=[0.10, 0.15],
            description=["BAD_MUSCLE", "BAD_AMPLITUDE"],
        )
    )
    events = np.array([[150, 0, 1]])

    report = epoch_badness(raw, events, tmin=-1.0, tmax=1.0)

    assert np.isclose(report.loc[0, "bad_overlap_seconds"], 0.20)
    assert not bool(report.loc[0, "tfr_available"])


def test_erd_tfr_is_resampled_before_computation():
    sfreq = 100.0
    time = np.arange(int(sfreq * 10)) / sfreq
    raw = mne.io.RawArray(
        np.sin(2 * np.pi * 8 * time)[None, :] * 1e-6,
        mne.create_info(["C3"], sfreq=sfreq, ch_types="eeg"),
        verbose="error",
    )
    raw.set_annotations(mne.Annotations([5.0], [0.0], ["repetition_cue"]))
    config = ERDERSConfig(
        tmin=-1.0,
        tmax=1.0,
        resample_sfreq=25.0,
        fmin=4.0,
        fmax=10.0,
        fstep=6.0,
        n_cycles_divisor=2.0,
        apply_csd=False,
    )

    tfr, report = compute_subject_tfr(raw, config, logging.getLogger(__name__))

    assert tfr.info["sfreq"] == 25.0
    assert len(tfr) == 1
    assert bool(report.loc[0, "tfr_available"])


def test_unreviewed_blinks_are_counted_and_reported():
    annotations = pd.DataFrame(
        {
            "onset": [1.0, 3.0, 5.0],
            "duration": [0.75, 0.75, 0.75],
            "trial_type": ["BLINK", "BLINK", "BLINK"],
            "reviewed_description": ["", "BLINK", "BLINK"],
            "review_status": ["unreviewed", "keep", "dismissed"],
        }
    )
    centres, n_unreviewed, n_dismissed = _reviewed_blinks(annotations)

    assert len(centres) == 2
    assert n_unreviewed == 1
    assert n_dismissed == 1


def test_blink_minimum_separation_suppresses_only_close_centres():
    centres, suppressed = enforce_minimum_blink_separation(
        np.array([1.00, 1.10, 1.15, 1.31, 2.00]),
        0.150,
    )

    assert np.allclose(centres, [1.00, 1.15, 1.31, 2.00])
    assert suppressed == 1


def test_minimal_preprocessing_keeps_candidates_and_writes_checkpoint(tmp_path):
    channels = ["FP1", "FPZ", "FP2", "C3", "C4", "T7", "EOG1"]
    channel_types = ["eeg"] * 6 + ["eog"]
    data = np.random.default_rng(1).normal(0, 1e-6, size=(len(channels), 512 * 20))
    data[-1, 1500:1600] += 100e-6
    raw = mne.io.RawArray(
        data,
        mne.create_info(channels, sfreq=512, ch_types=channel_types),
        verbose="error",
    )
    raw.set_annotations(mne.Annotations([1.0], [0.1], ["trial_start"]))
    config = MinimalPreprocessingConfig(
        drop_channels=("T7",),
        detect_muscle_candidates=False,
    )

    minimal_raw, candidates = build_minimal_run(raw, config, logging.getLogger(__name__))

    assert "T7" not in minimal_raw.ch_names
    assert "trial_start" in minimal_raw.annotations.description
    assert "BLINK" in candidates.description
    assert not any(description.startswith("BAD") for description in candidates.description)

    ensure_derivative_scaffold(tmp_path)
    paths = write_minimal_run(
        minimal_raw,
        candidates,
        tmp_path / "sub-01" / "ses-01" / "eeg",
        "sub-01_ses-01_task-silent_run-01",
        config,
        source={"subject": "sub-01", "task": "silent", "run": "run-1"},
        overwrite=True,
    )

    assert (tmp_path / "dataset_description.json").is_file()
    assert (tmp_path / "descriptions.tsv").is_file()
    assert all(path.is_file() for path in paths.values())


def test_prep_qc_writes_evidence_without_interpolating(tmp_path):
    channels = [
        "Fp1", "Fpz", "Fp2", "AF3", "AF4", "F7", "F3", "Fz", "F4", "F8",
        "FC5", "FC1", "FC2", "FC6", "T7", "C3", "Cz", "C4", "T8", "CP5",
        "CP1", "CP2", "CP6", "P7", "P3", "Pz", "P4", "P8", "O1", "Oz", "O2",
    ]
    sfreq = 512
    time = np.arange(sfreq * 30) / sfreq
    sources = np.stack(
        [
            20e-6 * np.sin(2 * np.pi * 10 * time),
            10e-6 * np.sin(2 * np.pi * 6 * time + 0.3),
            5e-6 * np.sin(2 * np.pi * 20 * time + 0.1),
        ]
    )
    rng = np.random.default_rng(2)
    data = (1 + rng.normal(scale=0.1, size=(len(channels), 3))) @ sources
    data += rng.normal(scale=1e-8, size=data.shape)
    raw = mne.io.RawArray(
        data,
        mne.create_info(channels, sfreq=sfreq, ch_types="eeg"),
        verbose="error",
    )
    raw.set_montage("standard_1020")

    report = run_prep_channel_qc(
        raw,
        PrepQCConfig(ransac=False),
        logging.getLogger(__name__),
    )

    assert report["status"] == "completed"
    assert report["proposed_bad_channels"] == report[
        "noisy_channels_before_interpolation"
    ]["bad_all"]
    assert "C3" in report["preinterpolation_detector_metrics"]["correlation"]["per_channel"]
    assert raw.info["bads"] == []

    paths = write_prep_qc(
        raw,
        report,
        tmp_path,
        "sub-01_ses-01_task-silent_run-01",
        tmp_path / "input.fif",
        overwrite=True,
    )
    assert all(path.is_file() for path in paths.values())


def test_review_tables_store_only_approved_bad_spans(tmp_path):
    annotation_path = tmp_path / "sub-01_desc-qc_annotations.tsv"
    channel_path = tmp_path / "sub-01_desc-channelqc.tsv"
    pd.DataFrame(
        {
            "onset": [1.0, 4.0],
            "duration": [0.5, 2.0],
            "trial_type": ["BLINK", "MUSCLE_CANDIDATE"],
            "annotation_source": ["auto"] * 2,
            "review_status": ["keep", "approved_bad"],
            "reviewed_description": ["BLINK", "BAD_MUSCLE"],
        }
    ).to_csv(annotation_path, sep="\t", index=False)
    pd.DataFrame(
        {
            "channel": ["C3"],
            "channel_type": ["EEG"],
            "automatic_status": ["flagged"],
            "automatic_methods": ["bad_by_correlation"],
        }
    ).to_csv(channel_path, sep="\t", index=False)
    record = ReviewRecord("run", Path("input.fif"), annotation_path, channel_path)

    annotations, channels = read_review_tables(record)
    assert approved_bad_annotations(annotations).to_dict("records") == [
        {"onset": 4.0, "duration": 2.0, "reviewed_description": "BAD_MUSCLE"}
    ]
    channels.loc[0, "final_decision"] = "retain"
    write_review_tables(record, annotations, channels)

    saved_annotations, saved_channels = read_review_tables(record)
    assert "reviewed_at" in saved_annotations
    assert saved_channels.loc[0, "final_decision"] == "retain"


def test_muscle_windows_merge_before_short_spans_are_discarded():
    raw = mne.io.RawArray(
        np.zeros((1, 1000)),
        mne.create_info(["C3"], sfreq=1000, ch_types="eeg"),
        verbose="error",
    )
    annotations = mne.Annotations(
        onset=[1.00, 1.15, 3.00],
        duration=[0.06, 0.06, 0.09],
        description=["BAD_muscle"] * 3,
    )

    merged = _merge_annotation_windows(
        raw,
        annotations,
        "MUSCLE_CANDIDATE",
        merge_gap_seconds=0.10,
        min_duration_seconds=0.10,
    )

    assert len(merged) == 1
    assert merged.description[0] == "MUSCLE_CANDIDATE"
    assert np.isclose(merged.onset[0], 1.0)
    assert np.isclose(merged.duration[0], 0.21)


def test_condition_channel_plan_exclusion_overrides_interpolation():
    columns = {
        "channel": ["C3", "C4", "EOG1"],
        "channel_type": ["EEG", "EEG", "EOG"],
        "automatic_status": ["not_flagged"] * 3,
        "automatic_methods": [""] * 3,
        "final_decision": ["interpolate", "retain", "unreviewed"],
    }
    run_1 = pd.DataFrame(columns)
    run_2 = pd.DataFrame({**columns, "final_decision": ["exclude_from_ica", "interpolate", "unreviewed"]})

    interpolated, excluded, plan = condition_channel_plan({"run-01": run_1, "run-02": run_2})

    assert interpolated == ["C4"]
    assert excluded == ["C3"]
    assert plan.set_index("channel").loc["C3", "condition_decision"] == "exclude_from_ica"


def test_component_energy_fractions_are_nonnegative_and_sum_to_one():
    mixing = np.array([[1.0, 0.2], [0.3, 2.0]])
    sources = np.array([
        np.sin(np.linspace(0, 20, 1000)),
        0.5 * np.cos(np.linspace(0, 20, 1000)),
    ])
    fractions = component_energy_fractions(mixing, sources)
    reconstructed_energies = np.array([
        np.var(mixing[:, component, None] * sources[component][None, :], axis=1, ddof=1).mean()
        for component in range(sources.shape[0])
    ])
    expected = reconstructed_energies / reconstructed_energies.sum()
    assert np.all(fractions >= 0)
    assert np.isclose(fractions.sum(), 1.0)
    assert np.allclose(fractions, expected)


def test_automatic_component_policy_applies_priority_and_manual_review_veto():
    table = pd.DataFrame(
        {
            "brain": [0.05, 0.05, 0.60, 0.10, 0.30],
            "muscle artifact": [0.80, 0.05, 0.05, 0.10, 0.20],
            "eye blink": [0.05] * 5,
            "heart beat": [0.0] * 5,
            "line noise": [0.0] * 5,
            "channel noise": [0.05, 0.70, 0.05, 0.10, 0.20],
            "variance_contribution_good_samples": [0.001, 0.001, 0.02, 0.003, 0.01],
        }
    )

    decisions = automatic_component_decisions(table)

    assert decisions.final_decision.tolist() == [
        "reject",
        "unreviewed",
        "keep",
        "keep",
        "unreviewed",
    ]
    assert decisions.decision_source.tolist() == [
        "automatic",
        "unreviewed",
        "automatic",
        "automatic",
        "unreviewed",
    ]
    assert decisions.loc[0, "decision_rule"] == "muscle artifact >= 80%"
    assert decisions.loc[2, "decision_rule"] == "brain >= 60%"
    assert decisions.loc[3, "decision_rule"] == "good-sample contribution < 0.4%"
