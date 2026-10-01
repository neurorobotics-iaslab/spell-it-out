import json

import h5py
import numpy as np
import pandas as pd
import pytest

from spellitout.syllable_dataset import (
    BuildPaths,
    DatasetSpec,
    _ShardWriter,
    bad_intervals,
    config_digest,
    load_config,
    overlaps_bad,
    read_phonetic_categories,
    write_syllable_dataset_description,
)


def _config():
    return {
        "source": "preprocessed",
        "datasets": [{
            "name": "silent_cue", "description_label": "silentcue",
            "task": "silent", "event_type": "syllabification_cue",
            "include_audio": False,
        }],
        "eeg": {"tmin": -0.3, "tmax": 0.6, "dtype": "float32", "units": "V"},
        "preprocessing": {},
        "audio": {},
        "hdf5": {"compression": "gzip", "compression_level": 1, "shuffle": True},
    }


def test_config_validation_and_digest_are_deterministic(tmp_path):
    config = _config()
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    loaded = load_config(path)
    assert loaded == config
    assert config_digest(loaded) == config_digest(json.loads(json.dumps(config)))

    config["eeg"]["tmin"] = 1
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="tmin"):
        load_config(path)


def test_bad_overlap_uses_bad_prefix_and_half_open_intervals():
    events = pd.DataFrame({
        "trial_type": ["BLINK", "BAD_MUSCLE", "BAD_manual"],
        "onset": [1.0, 2.0, 4.0],
        "duration": [0.75, 0.5, 1.0],
    })
    intervals = bad_intervals(events)
    assert intervals.tolist() == [[2.0, 2.5], [4.0, 5.0]]
    assert overlaps_bad(1.9, 2.1, intervals)
    assert not overlaps_bad(2.5, 3.0, intervals)


def test_phonetic_categories_reject_duplicates(tmp_path):
    path = tmp_path / "phones.tsv"
    path.write_text("phonesyll\tA\npa\t1\npa\t0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate syllables"):
        read_phonetic_categories(path)


def test_hdf5_shard_roundtrip(tmp_path):
    config = _config()
    spec = DatasetSpec("silent_cue", "silentcue", "silent", "syllabification_cue", False)
    path = tmp_path / "sample.h5"
    writer = _ShardWriter(
        path, spec=spec, subject="01", session="01", channel_names=["C3", "C4"],
        times=np.array([-0.5, 0.0, 0.5]), sfreq=2.0, category_names=["VOWEL"],
        config=config, digest=config_digest(config),
    )
    writer.append(
        np.arange(6, dtype=np.float32).reshape(1, 2, 3),
        [{
            "id": "sample-1", "run": "run-01", "syllable": "a",
            "stimulus_id": "sentence-1", "source_eeg": "source.vhdr",
            "source_events": "events.tsv", "trial_index": 1, "token_index": 2,
            "event_onset": 3.0, "event_duration": 0.2,
        }],
        np.array([[1]], dtype=np.uint8),
        None,
    )
    writer.close(commit=True)
    with h5py.File(path, "r") as handle:
        assert handle.attrs["sample_count"] == 1
        assert handle["eeg/data"].shape == (1, 2, 3)
        assert handle["samples/syllable"].asstr()[0] == "a"
        assert handle["labels/phonetic_categories"][:].tolist() == [[1]]


def test_syllable_derivative_description_has_generated_by_and_links(tmp_path):
    bids_root = tmp_path / "bids"
    preprocessing_root = bids_root / "derivatives" / "preprocessing"
    output_root = bids_root / "derivatives" / "syllable-datasets"
    output_root.mkdir(parents=True)
    preprocessing_root.mkdir(parents=True)
    (bids_root / "dataset_description.json").write_text(
        json.dumps(
            {
                "Name": "Raw dataset",
                "BIDSVersion": "1.11.1",
                "DatasetType": "raw",
                "License": "CC0",
            }
        ),
        encoding="utf-8",
    )
    paths = BuildPaths(
        bids_root=bids_root,
        source_root=preprocessing_root,
        preprocessed_root=preprocessing_root,
        output_root=output_root,
    )

    description_path = write_syllable_dataset_description(paths)

    description = json.loads(description_path.read_text(encoding="utf-8"))
    assert description["DatasetType"] == "derivative"
    assert description["GeneratedBy"][0]["Name"] == "spellitout.syllable_dataset"
    assert description["DatasetLinks"] == {
        "raw": "../..",
        "preprocessing": "../preprocessing",
    }
    assert description["SourceDatasets"] == [
        {"URL": "../.."},
        {"URL": "../preprocessing"},
    ]
    assert description["License"] == "CC0"
