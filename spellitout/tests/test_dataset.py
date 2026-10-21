from importlib import import_module
from pathlib import Path

import pytest

from spellitout import SpellItOutDataset


TEST_DATASET = Path(__file__).resolve().parents[2] / "data" / "spellitout"


def test_dataset_root_is_required(monkeypatch, tmp_path):
    monkeypatch.delenv("MNE_DATASETS_SPELLITOUT_DATA_PATH", raising=False)
    monkeypatch.setattr("mne.get_config", lambda key, *args, **kwargs: None)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    with pytest.raises(FileNotFoundError, match="download_dataset"):
        SpellItOutDataset()


def test_default_root_is_mne_data(monkeypatch, tmp_path):
    monkeypatch.setattr("mne.get_config", lambda key, *args, **kwargs: None)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    root = tmp_path / "mne_data" / "spellitout"
    root.mkdir(parents=True)
    (root / "dataset_description.json").write_text('{"Name": "t", "BIDSVersion": "1.11.1"}')
    assert SpellItOutDataset().root == root.resolve()


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

    assert "01" in dataset.subjects
    assert dataset.tasks == ("covert", "overt")
    sub01 = [path for path in dataset.recordings if path.subject == "01"]
    assert {(path.task, path.run) for path in sub01} >= {("covert", "01"), ("overt", "01")}

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
    # Speech recordings live in sourcedata/, which the public release omits.
    if (TEST_DATASET / "sourcedata").is_dir():
        assert overt.audio_path is not None and overt.audio_path.is_file()
        assert overt.textgrid_path is not None and overt.textgrid_path.is_file()
    else:
        assert overt.audio_path is None and overt.textgrid_path is None

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
        "sub-01_ses-01_task-covert_run-01",
        "sub-01_ses-01_task-overt_run-01",
    ]


def test_missing_source_audio_gives_none(tmp_path):
    from mne_bids import BIDSPath

    (tmp_path / "dataset_description.json").write_text('{"Name": "t", "BIDSVersion": "1.11.1"}')
    dataset = SpellItOutDataset(tmp_path)
    path = BIDSPath(root=tmp_path, subject="01", session="01", task="overt", run="01",
                    datatype="eeg", suffix="eeg", extension=".vhdr")
    assert dataset._source_paths(path) == (None, None)
