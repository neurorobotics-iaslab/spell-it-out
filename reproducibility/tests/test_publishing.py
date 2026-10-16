import mne
import numpy as np

from reproducibility.processing.publishing import write_vhdr_filter_settings


def test_brainvision_export_keeps_filter_settings(tmp_path):
    info = mne.create_info(["FP1", "Cz", "EOG"], 512.0, ["eeg", "eeg", "eog"])
    raw = mne.io.RawArray(np.random.default_rng(0).normal(0, 1e-5, (3, 1024)), info, verbose="error")
    vhdr = tmp_path / "run_eeg.vhdr"
    mne.export.export_raw(vhdr, raw, fmt="brainvision", verbose="error")

    write_vhdr_filter_settings(vhdr, 1.0, 100.0, 50.0)
    write_vhdr_filter_settings(vhdr, 1.0, 100.0, 50.0)  # idempotent

    assert vhdr.read_text(encoding="utf-8").count("\nChannels\n") == 1
    reread = mne.io.read_raw_brainvision(vhdr, verbose="error")
    assert reread.info["highpass"] == 1.0
    assert reread.info["lowpass"] == 100.0
    assert reread.ch_names == raw.ch_names
