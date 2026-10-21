import json
from types import SimpleNamespace

import h5py
import mne
import numpy as np
import pandas as pd
import pytest

from reproducibility.processing.syllable_dataset import (
    BuildPaths,
    build_syllable_datasets,
    DatasetSpec,
    _ShardWriter,
    _bad_events_for_run,
    _pick_and_preprocess,
    bad_intervals,
    config_digest,
    load_config,
    overlaps_bad,
    read_phonetic_categories,
    resolve_config,
    write_syllable_dataset_description,
)


def _config():
    return {
        "source": "preprocessed",
        "datasets": [{
            "name": "covert_cue", "description_label": "covertcue",
            "task": "covert", "event_type": "syllabification_cue",
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


def test_audio_is_opt_in_and_overrides_do_not_mutate_config():
    config = _config()
    config["datasets"][0]["include_audio"] = True
    default = resolve_config(config)
    assert not default["datasets"][0]["include_audio"]
    assert default["audio"] == {"enabled": False, "extract_embeddings": False}
    assert config["datasets"][0]["include_audio"]
    assert config["audio"] == {}

    enabled = resolve_config(config, audio=True, embeddings=True)
    assert enabled["datasets"][0]["include_audio"]
    assert enabled["audio"]["extract_embeddings"]
    disabled = resolve_config(enabled, audio=False)
    assert not disabled["datasets"][0]["include_audio"]
    assert not disabled["audio"]["extract_embeddings"]
    with pytest.raises(ValueError, match="requires audio.enabled"):
        resolve_config(config, embeddings=True)


@pytest.mark.parametrize("mode,no_audio", [("apply", False), ("apply", True), ("validate", False)])
def test_production_embeddings_and_audio_bypass(tmp_path, monkeypatch, mode, no_audio):
    from reproducibility.processing import syllable_dataset as module

    config = _config()
    config["datasets"] = [{
        "name": "overt_production", "description_label": "overtproduction",
        "task": "overt", "event_type": "syllabification_syllable",
        "include_audio": True,
    }, {
        "name": "overt_cue", "description_label": "overtcue", "task": "overt",
        "event_type": "syllabification_cue", "include_audio": False,
    }]
    config = resolve_config(config, audio=not no_audio, embeddings=not no_audio)
    raw = mne.io.RawArray(np.zeros((1, 500)), mne.create_info(["C3"], 100, "eeg"), verbose="error")
    events = pd.DataFrame({
        "onset": [1., 2., 1., 2.], "duration": [.1, .2, .6, .6],
        "trial_type": ["syllabification_syllable"] * 2 + ["syllabification_cue"] * 2,
        "trial_index": [1] * 4, "token_index": [1, 2, 1, 2], "token": ["a"] * 4,
    })
    run = SimpleNamespace(
        raw=raw, events=events, events_path=tmp_path / "events.tsv",
        subject="01", subject_id="sub-01", session="01", task="overt", run="01",
        bids_path=SimpleNamespace(fpath=tmp_path / "source.vhdr"),
    )
    monkeypatch.setattr(module, "SpellItOutDataset", lambda _: SimpleNamespace(
        subjects=["01"], iter_runs=lambda **kwargs: iter([run]),
    ))
    calls = []
    def locate(*args):
        calls.append("locate")
        return tmp_path / "audio.wav"
    def read(*args, **kwargs):
        calls.append("read")
        return np.arange(5000, dtype=np.float32), 1000
    def extract(waveform, sample_rate):
        calls.append((len(waveform), sample_rate))
        return np.full((len(waveform) // 100, 3), waveform[0], dtype=np.float32)
    def factory():
        calls.append("model")
        return SimpleNamespace(extract_embedding=extract, model=SimpleNamespace(
            config=SimpleNamespace(_name_or_path="test-checkpoint", _commit_hash="revision"),
        ))
    monkeypatch.setattr(module, "_source_audio_path", locate)
    monkeypatch.setattr(module.sf, "info", lambda _: SimpleNamespace(samplerate=1000, frames=5000))
    monkeypatch.setattr(module.sf, "read", read)
    monkeypatch.setattr(module, "_create_embedding_extractor", factory)
    categories = tmp_path / "categories.tsv"
    categories.write_text("phonesyll\tVOWEL\na\t1\n", encoding="utf-8")
    (tmp_path / "dataset_description.json").write_text('{"Name":"test"}', encoding="utf-8")
    paths = BuildPaths(tmp_path, tmp_path, tmp_path, tmp_path / "output")
    result = build_syllable_datasets(paths, config, categories, mode=mode)
    assert result["counts"]["overt_production"]["accepted"] == 2
    if no_audio:
        assert calls == []
    elif mode == "validate":
        assert calls == ["locate"]
        assert not paths.output_root.exists()
    else:
        assert calls == ["locate", "read", "model", (100, 1000), (200, 1000)]
    if mode == "apply":
        production = next(paths.output_root.rglob("*desc-overtproduction_dataset.h5"))
        assert production.parent == paths.output_root / "overt_production"
        with h5py.File(production, "r") as handle:
            assert handle["eeg/data"].shape[0] == 2
            if no_audio:
                assert "audio" not in handle and "audio_embeddings" not in handle
            else:
                assert handle["audio/offsets"][:].tolist() == [0, 100, 300]
                assert handle["audio_embeddings/offsets"][:].tolist() == [0, 1, 3]
                np.testing.assert_array_equal(handle["audio_embeddings/data"][:], [[1000]*3, [2000]*3, [2000]*3])
                assert handle["audio_embeddings"].attrs["model"] == "test-checkpoint"
        with h5py.File(next(paths.output_root.rglob("*desc-overtcue_dataset.h5")), "r") as handle:
            assert "audio" not in handle and "audio_embeddings" not in handle


def test_processing_overrides_preserve_original_config_and_resampling():
    config = _config()
    config["preprocessing"]["resample_sfreq"] = 512
    resolved = resolve_config(
        config, source="raw", exclude_bads=True,
        bad_annotation_source="auto", filter_phase="causal",
    )
    assert resolved["source"] == "raw"
    assert resolved["eeg"]["exclude_bad_overlap"] is True
    assert resolved["eeg"]["bad_annotation_source"] == "auto"
    assert resolved["preprocessing"]["filter_phase"] == "causal"
    assert resolved["preprocessing"]["resample_sfreq"] == 512
    assert "filter_phase" not in config["preprocessing"]
    assert resolve_config(resolved, exclude_bads=False)["eeg"]["exclude_bad_overlap"] is False
    with pytest.raises(ValueError, match="filter_phase"):
        resolve_config(config, filter_phase="invalid")
    with pytest.raises(ValueError, match="bad_annotation_source"):
        resolve_config(config, bad_annotation_source="invalid")


def test_raw_signal_can_use_reviewed_bad_events():
    raw_events = pd.DataFrame({
        "trial_type": ["syllabification_cue"], "onset": [2.0], "duration": [0.6],
    })
    reviewed_events = pd.DataFrame({
        "trial_type": ["BAD_MUSCLE"], "onset": [2.1], "duration": [0.2],
    })
    run = SimpleNamespace(events=raw_events, subject="01", session="01", task="overt", run="02")
    calls = []

    def get_run(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(events=reviewed_events)

    derivative = SimpleNamespace(get_run=get_run)
    events = _bad_events_for_run(run, mode="preprocessed", preprocessed_dataset=derivative)
    assert overlaps_bad(1.7, 2.6, bad_intervals(events))
    assert not overlaps_bad(1.7, 2.6, bad_intervals(raw_events))
    assert calls == [{"subject": "01", "session": "01", "task": "overt", "run": "02", "preload": False}]
    assert _bad_events_for_run(run, mode="none", preprocessed_dataset=None).empty


def test_causal_filters_do_not_precede_impulse_and_leave_source_unchanged():
    data = np.zeros((1, 4096))
    data[0, 2048] = 1e-6
    raw = mne.io.RawArray(data, mne.create_info(["C3"], 512, "eeg"), verbose="error")
    config = _config()
    config["preprocessing"] = {
        "l_freq": 2, "h_freq": 70, "notch_freqs": [50, 100],
        "filter_phase": "causal", "resample_sfreq": None,
    }
    filtered = _pick_and_preprocess(raw, config).get_data()
    assert np.all(filtered[:, :2048] == 0)
    assert np.any(filtered[:, 2048:] != 0)
    np.testing.assert_array_equal(raw.get_data(), data)


def test_resampling_still_uses_standard_mne_method():
    raw = mne.io.RawArray(
        np.random.default_rng(42).normal(size=(1, 4096)) * 1e-6,
        mne.create_info(["C3"], 2048, "eeg"), verbose="error",
    )
    config = _config()
    config["preprocessing"] = {"filter_phase": "causal", "resample_sfreq": 512}
    actual = _pick_and_preprocess(raw, config)
    expected = raw.copy().resample(512, verbose="error")
    assert actual.info["sfreq"] == 512
    np.testing.assert_array_equal(actual.get_data(), expected.get_data())


def test_resampling_precedes_causal_bandpass_and_notch():
    raw = mne.io.RawArray(
        np.random.default_rng(42).normal(size=(1, 4096)) * 1e-6,
        mne.create_info(["C3"], 2048, "eeg"), verbose="error",
    )
    config = _config()
    config["preprocessing"] = {
        "filter_phase": "causal", "resample_sfreq": 512,
        "l_freq": 2, "h_freq": 70, "notch_freqs": [50],
    }
    options = {"method": "iir", "iir_params": {"order": 4, "ftype": "butter"}, "phase": "forward"}
    expected = raw.copy().resample(512, verbose="error")
    expected.filter(2, 70, picks="all", verbose="error", **options)
    expected.notch_filter([50], picks="all", verbose="error", **options)
    actual = _pick_and_preprocess(raw, config)
    np.testing.assert_array_equal(actual.get_data(), expected.get_data())


def test_phonetic_categories_reject_duplicates(tmp_path):
    path = tmp_path / "phones.tsv"
    path.write_text("phonesyll\tA\npa\t1\npa\t0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate syllables"):
        read_phonetic_categories(path)


def test_hdf5_shard_roundtrip(tmp_path):
    config = _config()
    spec = DatasetSpec("covert_cue", "covertcue", "covert", "syllabification_cue", False)
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
