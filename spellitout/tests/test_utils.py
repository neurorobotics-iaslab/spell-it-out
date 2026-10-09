"""Generic helpers retained in the dataset-access package."""

import numpy as np

from spellitout import utils


def test_make_montage_keeps_requested_channels():
    montage = utils.make_montage(["C3", "CZ", "C4"])
    assert montage.ch_names == ["C3", "CZ", "C4"]


def test_epoch_data_extracts_requested_window():
    data = np.arange(20, dtype=float).reshape(2, 10)
    epochs, times = utils.epoch_data(data, [{"onset": 0.5}], (-0.2, 0.3), 10)
    np.testing.assert_array_equal(epochs[0], data[:, 3:8])
    np.testing.assert_allclose(times, [-0.2, -0.1, 0.0, 0.1, 0.2])


def test_preparation_only_helpers_are_not_in_core():
    for name in ("get_subject_id", "get_run_id", "reset_mne_first_samp"):
        assert not hasattr(utils, name)
