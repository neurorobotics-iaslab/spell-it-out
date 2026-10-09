"""Compact metadata must preserve values and remain readable by MNE-BIDS."""

import json
from types import SimpleNamespace

import mne
import numpy as np
import pandas as pd
from mne_bids import BIDSPath, read_raw_bids
from mne_bids.path import _find_matching_sidecar
import pytest

from reproducibility.processing.bids_sidecars import (
    apply_sidecar_plan, plan_sidecar_migration,
    read_channels_metadata, read_eeg_metadata,
)
from reproducibility.processing.publishing import _run_sidecar, export_preprocessed_dataset


METADATA = {
    "TaskName": "Covert speech", "EEGReference": "CPz", "SamplingFrequency": 2048,
    "PowerLineFrequency": 50, "SoftwareFilters": "n/a",
}
CHANNELS = "name\ttype\tunits\nCz\tEEG\tV\nEOG\tEOG\tV\n"


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def raw_dataset(tmp_path):
    root = tmp_path / "raw"
    write_json(root / "dataset_description.json", {"Name": "Test", "BIDSVersion": "1.11.1"})
    write_json(root / "task-covert_eeg.json", METADATA)
    (root / "channels.tsv").write_text(CHANNELS, encoding="utf-8")
    directory = root / "sub-22/ses-01/eeg"
    directory.mkdir(parents=True)
    recordings = [directory / f"sub-22_ses-01_task-covert_run-{run:02d}_eeg.vhdr"
                  for run in (1, 2, 3)]
    for recording in recordings:
        recording.write_bytes(b"signal placeholder: never opened or rewritten")
    return root, recordings


def test_validate_is_read_only_and_apply_is_lossless_and_idempotent(raw_dataset, tmp_path):
    root, recordings = raw_dataset
    original_files = {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}
    plan = plan_sidecar_migration(root)
    assert plan.recordings == 3
    assert len(plan.writes) == 2
    assert {path: path.read_bytes() for path in original_files} == original_files
    backup = apply_sidecar_plan(plan, backup_root=tmp_path / "backups")
    assert (backup / "originals/channels.tsv").read_text() == CHANNELS
    assert json.loads((backup / "journal.json").read_text())["state"] == "complete"
    assert not (root / "channels.tsv").exists()
    assert not (root / "task-covert_eeg.json").exists()
    for recording in recordings:
        assert recording.read_bytes() == original_files[recording]
        assert read_eeg_metadata(root, recording) == METADATA
        assert read_channels_metadata(root, recording)["type"].tolist() == ["EEG", "EOG"]
        bids_path = BIDSPath(root=root, subject="22", session="01", task="covert",
                             run=recording.name.split("run-")[1][:2], datatype="eeg", suffix="eeg")
        assert _find_matching_sidecar(bids_path, suffix="eeg", extension=".json").name == "sub-22_ses-01_task-covert_eeg.json"
        assert _find_matching_sidecar(bids_path, suffix="channels", extension=".tsv").name == "sub-22_ses-01_task-covert_channels.tsv"
    repeated = plan_sidecar_migration(root)
    assert not repeated.writes and not repeated.removals


def test_run_specific_provenance_receives_complete_json(raw_dataset):
    root, recordings = raw_dataset
    expected = []
    for index, recording in enumerate(recordings):
        specific = {"Sources": [f"bids:raw:run-{index+1}"], "EEGChannelCount": 1}
        write_json(recording.with_suffix(".json"), specific)
        expected.append({**METADATA, **specific})
    apply_sidecar_plan(plan_sidecar_migration(root))
    for recording, metadata in zip(recordings, expected):
        assert json.loads(recording.with_suffix(".json").read_text()) == metadata
        assert read_eeg_metadata(root, recording) == metadata


def test_different_channel_tables_are_not_collapsed(raw_dataset):
    root, recordings = raw_dataset
    specific = recordings[1].with_name(recordings[1].name.replace("_eeg.vhdr", "_channels.tsv"))
    specific.write_text(CHANNELS.replace("Cz\tEEG\tV", "C3\tEEG\tV"), encoding="utf-8")
    apply_sidecar_plan(plan_sidecar_migration(root))
    assert [read_channels_metadata(root, path).iloc[0]["name"] for path in recordings] == ["Cz", "C3", "Cz"]
    assert not (recordings[0].parent / "sub-22_ses-01_task-covert_channels.tsv").exists()
    assert len(list(recordings[0].parent.glob("*_channels.tsv"))) == 3


def test_subject_groups_are_not_merged(raw_dataset):
    root, recordings = raw_dataset
    other = root / "sub-23/ses-01/eeg/sub-23_ses-01_task-covert_run-01_eeg.vhdr"
    other.parent.mkdir(parents=True)
    other.write_bytes(b"another subject")
    write_json(other.with_suffix(".json"), {"EEGReference": "Cz"})
    apply_sidecar_plan(plan_sidecar_migration(root))
    assert read_eeg_metadata(root, recordings[0])["EEGReference"] == "CPz"
    assert read_eeg_metadata(root, other)["EEGReference"] == "Cz"


def test_missing_metadata_and_concurrent_changes_fail_without_writes(raw_dataset):
    root, recordings = raw_dataset
    plan = plan_sidecar_migration(root)
    write_json(root / "task-covert_eeg.json", {**METADATA, "EEGReference": "Cz"})
    with pytest.raises(RuntimeError, match="changed after validation"):
        apply_sidecar_plan(plan)
    assert not (recordings[0].parent / "sub-22_ses-01_task-covert_eeg.json").exists()
    (root / "task-covert_eeg.json").unlink()
    with pytest.raises(FileNotFoundError, match="EEG JSON"):
        plan_sidecar_migration(root)


def test_backups_cannot_pollute_dataset(raw_dataset):
    root, _ = raw_dataset
    with pytest.raises(ValueError, match="outside"):
        apply_sidecar_plan(plan_sidecar_migration(root), backup_root=root / "backup")


def test_export_run_metadata_reads_migrated_raw_sidecars(raw_dataset):
    root, recordings = raw_dataset
    apply_sidecar_plan(plan_sidecar_migration(root))
    record = SimpleNamespace(raw_eeg=recordings[0])
    inspection = SimpleNamespace(
        record=record, sfreq=512., highpass=1., lowpass=100.,
        line_noise_method="notch", line_frequency_hz=50.,
        rejected_components=(0, 3), channel_types=("eeg", "eog"),
    )
    metadata = _run_sidecar(inspection, root)
    assert metadata["TaskName"] == METADATA["TaskName"]
    assert metadata["EEGReference"] == "average"
    assert metadata["SamplingFrequency"] == 512
    assert metadata["EOGChannelCount"] == 1
    assert metadata["Sources"] == ["bids:raw:" + recordings[0].relative_to(root).as_posix()]
    assert metadata["SoftwareFilters"]["BandpassFilter"]["HighpassCutoffHz"] == 1


def test_apply_rolls_back_partial_sidecar_writes(raw_dataset, monkeypatch):
    from reproducibility.processing import bids_sidecars

    root, _ = raw_dataset
    original = {path.relative_to(root): path.read_bytes()
                for path in root.rglob("*") if path.is_file()}
    write = bids_sidecars._atomic_write
    calls = 0

    def fail_once(path, content):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("test interrupted migration")
        return write(path, content)

    monkeypatch.setattr(bids_sidecars, "_atomic_write", fail_once)
    with pytest.raises(OSError, match="interrupted migration"):
        apply_sidecar_plan(plan_sidecar_migration(root))
    assert {path.relative_to(root): path.read_bytes()
            for path in root.rglob("*") if path.is_file()} == original


def test_export_produces_complete_run_json_and_shared_channel_table(raw_dataset, monkeypatch):
    from reproducibility.processing import publishing

    root, recordings = raw_dataset
    apply_sidecar_plan(plan_sidecar_migration(root))
    write_json(root / "task-covert_events.json", {"trial_type": {"Levels": {}}})
    plan = root.parent / "channel_plan.tsv"
    plan.write_text("channel\tinterpolate_triggering_runs\n", encoding="utf-8")
    inspections = []
    for recording in recordings:
        record = SimpleNamespace(
            raw_eeg=recording, subject="sub-22", session="ses-01", task="covert",
            base=recording.stem.removesuffix("_eeg"), cleaned_raw=recording,
            channel_plan=plan,
        )
        inspections.append(SimpleNamespace(
            record=record, sfreq=512., n_times=1024, highpass=1., lowpass=100.,
            line_noise_method="notch", line_frequency_hz=50.,
            rejected_components=(0,), channel_types=("eeg", "eog"),
            channel_names=("Cz", "EOG"), interpolated_channels=(),
            final_events=pd.DataFrame({"onset": [0.], "duration": [.2],
                                       "trial_type": ["repetition_cue"]}),
        ))
    monkeypatch.setattr(publishing, "_read_raw", lambda *args, **kwargs: mne.io.RawArray(
        np.zeros((2, 1024)), mne.create_info(["Cz", "EOG"], 512, ["eeg", "eog"]),
        verbose="error",
    ))
    monkeypatch.setattr(publishing, "_copy_coordinates", lambda *args: None)
    monkeypatch.setattr(publishing, "_copy_qc", lambda *args: None)
    output = root / "derivatives/preprocessing"
    export_preprocessed_dataset(inspections, root, output)
    directory = output / "sub-22/ses-01/eeg"
    assert len(list(directory.glob("*_channels.tsv"))) == 1
    assert (directory / "sub-22_ses-01_task-covert_desc-preproc_channels.tsv").is_file()
    assert len(list(directory.glob("*_eeg.json"))) == 3
    assert not list(output.glob("*_eeg.json"))
    for inspection in inspections:
        stem = inspection.record.base + "_desc-preproc_eeg"
        metadata = json.loads((directory / (stem + ".json")).read_text())
        assert metadata["TaskName"] == METADATA["TaskName"]
        assert metadata["Sources"] == ["bids:raw:" + inspection.record.raw_eeg.relative_to(root).as_posix()]
        path = BIDSPath(root=output, subject="22", session="01", task="covert",
                        run=inspection.record.base.split("run-")[1], description="preproc",
                        suffix="eeg", datatype="eeg", extension=".vhdr", check=False)
        loaded = read_raw_bids(path, verbose="error")
        assert loaded.get_channel_types() == ["eeg", "eog"]
        assert loaded.info["line_freq"] == 50
        assert loaded.info["sfreq"] == 512
        loaded.close()
