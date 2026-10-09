from importlib import import_module
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


def test_matching_paths_excludes_nested_derivatives(monkeypatch, tmp_path):
    (tmp_path / "dataset_description.json").write_text("{}", encoding="utf-8")
    direct = tmp_path / "sub-01" / "eeg" / "sub-01_task-covert_eeg.vhdr"
    derivative = (
        tmp_path / "derivatives" / "preprocessing" / "sub-01" / "eeg"
        / "sub-01_task-covert_desc-preproc_eeg.vhdr"
    )

    class FakePath:
        def __init__(self, fpath, description):
            self.fpath = fpath
            self.description = description

    monkeypatch.setattr(
        import_module(SpellItOutDataset.__module__),
        "find_matching_paths",
        lambda *args, **kwargs: [
            FakePath(derivative, "preproc"), FakePath(direct, None)
        ],
    )
    dataset = SpellItOutDataset(tmp_path)
    matches = dataset._matching_paths()
    assert [path.fpath for path in matches] == [direct]


@pytest.mark.skipif(
    not (TEST_DATASET / "dataset_description.json").is_file(),
    reason="Local BIDS integration-test dataset is unavailable.",
)
def test_bids_discovery_and_loading():
    dataset = SpellItOutDataset(TEST_DATASET)

    assert dataset.subjects == ("01",)
    assert dataset.tasks == ("overt", "covert")
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

    covert = dataset.get_run(subject="sub-01", task="covert", run="run-01")
    assert covert.audio_path is None
    assert covert.textgrid_path is None


@pytest.mark.skipif(
    not (TEST_DATASET / "dataset_description.json").is_file(),
    reason="Local BIDS integration-test dataset is unavailable.",
)
def test_iter_runs_filters_and_uses_stable_order():
    dataset = SpellItOutDataset(TEST_DATASET)
    runs = list(
        dataset.iter_runs(
            subjects=[1],
            tasks=["covert", "overt"],
            runs=[1],
            preload=False,
        )
    )
    assert [run.run_id for run in runs] == [
        "sub-01_ses-01_task-overt_run-01",
        "sub-01_ses-01_task-covert_run-01",
    ]
