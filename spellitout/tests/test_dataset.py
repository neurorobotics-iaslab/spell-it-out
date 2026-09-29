from pathlib import Path

import pytest

from spellitout import SpellItOutDataset


TEST_DATASET = Path(__file__).resolve().parents[2] / "data" / "spellitout_bids_test"


def test_dataset_root_is_required(monkeypatch):
    monkeypatch.delenv("MNE_DATASETS_SPELLITOUT_DATA_PATH", raising=False)
    with pytest.raises(ValueError, match="Dataset root not provided"):
        SpellItOutDataset()


def test_non_bids_directory_is_rejected(tmp_path):
    with pytest.raises(FileNotFoundError, match="dataset_description.json"):
        SpellItOutDataset(tmp_path)


@pytest.mark.skipif(
    not (TEST_DATASET / "dataset_description.json").is_file(),
    reason="Local BIDS integration-test dataset is unavailable.",
)
def test_bids_discovery_and_loading():
    dataset = SpellItOutDataset(TEST_DATASET)

    assert dataset.subjects == ("01",)
    assert dataset.tasks == ("overt", "silent")
    assert len(dataset.recordings) == 2

    overt = dataset.get_run(subject=1, task="overt", run=1, preload=False)
    assert overt.subject == "01"
    assert overt.subject_id == "sub-01"
    assert overt.session == "01"
    assert overt.task == "overt"
    assert overt.run == "01"
    assert overt.run_number == 1
    assert overt.run_id == "sub-01_ses-01_task-overt_run-01"
    assert not overt.raw.preload
    assert {"onset", "duration", "trial_type"}.issubset(overt.events.columns)
    assert len(overt.raw.annotations) == len(overt.events)
    assert overt.audio_path is not None and overt.audio_path.is_file()
    assert overt.textgrid_path is not None and overt.textgrid_path.is_file()

    silent = dataset.get_run(subject="sub-01", task="silent", run="run-01")
    assert silent.audio_path is None
    assert silent.textgrid_path is None


@pytest.mark.skipif(
    not (TEST_DATASET / "dataset_description.json").is_file(),
    reason="Local BIDS integration-test dataset is unavailable.",
)
def test_iter_runs_filters_and_uses_stable_order():
    dataset = SpellItOutDataset(TEST_DATASET)
    runs = list(
        dataset.iter_runs(
            subjects=[1],
            tasks=["silent", "overt"],
            runs=[1],
            preload=False,
        )
    )
    assert [run.run_id for run in runs] == [
        "sub-01_ses-01_task-overt_run-01",
        "sub-01_ses-01_task-silent_run-01",
    ]
