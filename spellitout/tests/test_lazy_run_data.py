"""Run discovery must not open EEG files or read event tables."""

from importlib import import_module
from types import SimpleNamespace
from unittest.mock import Mock

from mne_bids import BIDSPath
import pandas as pd
import pytest

from spellitout import RunData, SpellItOutDataset


@pytest.fixture
def recording(monkeypatch, tmp_path):
    (tmp_path / "dataset_description.json").write_text("{}", encoding="utf-8")
    path = BIDSPath(
        root=tmp_path, subject="22", session="01", task="covert", run="02",
        datatype="eeg", suffix="eeg", extension=".vhdr",
    )
    dataset = SpellItOutDataset(tmp_path)
    monkeypatch.setattr(dataset, "_matching_paths", lambda **kwargs: [path])
    module = import_module(RunData.__module__)
    return dataset, path, module


@pytest.mark.parametrize("preload", [False, True])
def test_listing_and_repr_never_load_content(recording, monkeypatch, preload):
    dataset, _, module = recording
    reader = Mock(side_effect=AssertionError("Listing must not open the EEG"))
    csv_reader = Mock(side_effect=AssertionError("Listing must not read events"))
    monkeypatch.setattr(module, "read_raw_bids", reader)
    monkeypatch.setattr(module.pd, "read_csv", csv_reader)

    runs = list(dataset.iter_runs(subjects=22, tasks="covert", runs=2, preload=preload))
    assert len(runs) == 1
    run = runs[0]
    assert run.subject_id == "sub-22"
    assert run.run_number == 2
    assert run.run_id == "sub-22_ses-01_task-covert_run-02"
    assert run.events_path.name == run.run_id + "_events.tsv"
    assert "RunData" in repr(run)
    reader.assert_not_called()
    csv_reader.assert_not_called()


@pytest.mark.parametrize("preload", [False, True])
def test_get_run_and_raw_load_once_with_requested_preload(recording, monkeypatch, preload):
    dataset, path, module = recording
    raw = SimpleNamespace(preload=preload)
    reader = Mock(return_value=raw)
    monkeypatch.setattr(module, "read_raw_bids", reader)
    csv_reader = Mock(side_effect=AssertionError("Raw cache must not read the TSV itself"))
    monkeypatch.setattr(module.pd, "read_csv", csv_reader)

    run = dataset.get_run(subject=22, task="covert", run=2, preload=preload)
    reader.assert_not_called()
    assert run.raw is raw
    assert run.raw is raw
    assert run.raw.preload is preload
    reader.assert_called_once_with(path, extra_params={"preload": preload}, verbose="error")
    csv_reader.assert_not_called()


def test_events_load_once_without_opening_raw(recording, monkeypatch):
    dataset, _, module = recording
    run = dataset.get_run(subject=22, task="covert", run=2)
    run.events_path.parent.mkdir(parents=True)
    # "na" is a syllable, not a missing-value marker.
    run.events_path.write_text(
        "onset\tduration\ttrial_type\ttoken\n0\t0.6\tsyllabification_cue\tna\n",
        encoding="utf-8",
    )
    reader = Mock(side_effect=AssertionError("Events must not open the EEG"))
    csv_reader = Mock(wraps=module.pd.read_csv)
    monkeypatch.setattr(module, "read_raw_bids", reader)
    monkeypatch.setattr(module.pd, "read_csv", csv_reader)

    first = run.events
    assert run.events is first
    assert first.loc[0, "token"] == "na"
    csv_reader.assert_called_once()
    reader.assert_not_called()


def test_missing_events_error_is_deferred_and_can_be_retried(recording):
    dataset, _, _ = recording
    run = dataset.get_run(subject=22, task="covert", run=2)
    with pytest.raises(FileNotFoundError, match="Events table not found"):
        _ = run.events

    run.events_path.parent.mkdir(parents=True)
    run.events_path.write_text("onset\tduration\ttrial_type\n", encoding="utf-8")
    assert run.events.empty
    assert run.events is run.events


def test_events_validation_is_deferred_and_bad_tables_are_not_cached(recording):
    dataset, _, _ = recording
    run = dataset.get_run(subject=22, task="covert", run=2)
    run.events_path.parent.mkdir(parents=True)
    run.events_path.write_text("onset\n0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duration, trial_type"):
        _ = run.events

    run.events_path.write_text("onset\tduration\ttrial_type\n", encoding="utf-8")
    assert run.events.empty


def test_failed_raw_loading_can_be_retried(recording, monkeypatch):
    dataset, _, module = recording
    raw = object()
    reader = Mock(side_effect=[OSError("temporarily unavailable"), raw])
    monkeypatch.setattr(module, "read_raw_bids", reader)
    run = dataset.get_run(subject=22, task="covert", run=2)
    with pytest.raises(OSError, match="temporarily unavailable"):
        _ = run.raw
    assert run.raw is raw
    assert run.raw is raw
    assert reader.call_count == 2


def test_supplied_objects_and_assignments_do_not_trigger_loading(recording, monkeypatch):
    _, path, module = recording
    reader = Mock(side_effect=AssertionError("A supplied raw must be reused"))
    monkeypatch.setattr(module, "read_raw_bids", reader)
    raw = object()
    events = pd.DataFrame()
    run = RunData(path, raw=raw, events=events)
    assert run.raw is raw
    assert run.events is events
    replacement_raw = object()
    replacement_events = pd.DataFrame({"onset": [0]})
    run.raw = replacement_raw
    run.events = replacement_events
    assert run.raw is replacement_raw
    assert run.events is replacement_events
    reader.assert_not_called()
