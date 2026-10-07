"""Blink-rate analysis must depend on published events, not working checkpoints."""

from pathlib import Path

import mne
from mne_bids import BIDSPath
import numpy as np
import pandas as pd

from spellitout.blink_rate import run_blink_rate_rows
from spellitout.dataset import RunData


def published_run():
    raw = mne.io.RawArray(
        np.zeros((1, 100)), mne.create_info(["Cz"], 10, "eeg"), verbose="error"
    )
    events = pd.DataFrame({
        "onset": [0., 4., 8., 9., 1.1, 1.2, 5.1, 5.],
        "duration": [4., 4., .4, .4, .2, .2, .2, 1.],
        "trial_type": ["listening_cue", "repetition_cue", "syllabification_cue",
                       "syllabification_cue", "BLINK", "MANUAL_BLINK", "BLINK",
                       "BAD_MUSCLE"],
    })
    raw.set_annotations(mne.Annotations(
        events.onset, events.duration, events.trial_type
    ))
    return RunData(
        bids_path=BIDSPath(subject="02", session="01", task="covert", run="01",
                           description="preproc", suffix="eeg", extension=".vhdr",
                           datatype="eeg", check=False),
        raw=raw, events=events, events_path=Path("unused_events.tsv"),
    )


def test_final_events_control_counts_and_bad_span_exclusion():
    run = published_run()
    # A kept QC candidate absent from the published events must not be added.
    qc = pd.DataFrame({
        "onset": [2., 3.], "duration": [.2, .2],
        "trial_type": ["BLINK", "BLINK"],
        "reviewed_description": ["BLINK", "BLINK"],
        "review_status": ["keep", "dismissed"],
    })
    rows = run_blink_rate_rows(run, annotations_qc=qc, exclude_bad_spans=True,
                               minimum_blink_separation_seconds=.35)
    whole_run = rows[0]
    assert whole_run["subject"] == "sub-02"
    assert whole_run["task"] == "covert"
    assert whole_run["run"] == "01"
    assert whole_run["phase_duration_s"] == 10.
    assert whole_run["valid_duration_s"] == 9.
    assert whole_run["blink_count"] == 1
    assert whole_run["n_dismissed_blinks"] == 1
    assert whole_run["n_blinks_suppressed_min_separation"] == 1
    assert whole_run["review_complete"]
    assert rows[-1]["n_syllabification_cues"] == 2


def test_published_run_works_without_qc_or_fif():
    rows = run_blink_rate_rows(published_run(), exclude_bad_spans=False,
                               minimum_blink_separation_seconds=.35)
    assert rows[0]["blink_count"] == 2
    assert np.isnan(rows[0]["n_dismissed_blinks"])
    assert rows[0]["review_complete"]
