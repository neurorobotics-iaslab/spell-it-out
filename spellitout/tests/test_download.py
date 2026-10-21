import pytest

from spellitout.download import include_patterns

glob = pytest.importorskip("openneuro._glob")

# Mirrors the published layout (two subjects are enough).
FILES = [
    "README", "dataset_description.json", "participants.tsv", "participants.json",
    "task-overt_events.json", "task-covert_events.json",
    "stimuli/wordlist.tsv", "stimuli/audio/sentence-1738.wav",
]
for s in ("01", "02"):
    FILES += [
        f"sub-{s}/ses-01/eeg/sub-{s}_ses-01_task-overt_run-01_eeg.eeg",
        f"sub-{s}/ses-01/eeg/sub-{s}_ses-01_task-overt_run-01_events.tsv",
        f"derivatives/preprocessing/sub-{s}/ses-01/eeg/sub-{s}_ses-01_task-overt_run-01_desc-preproc_eeg.eeg",
        f"derivatives/preprocessing/qc/sub-{s}/ses-01/eeg/sub-{s}_ses-01_task-overt_desc-icadecisions.tsv",
        f"derivatives/syllable-datasets/overt_cue/sub-{s}_ses-01_task-overt_desc-overtcue_dataset.h5",
        f"derivatives/syllable-datasets/covert_cue/sub-{s}_ses-01_task-covert_desc-covertcue_dataset.h5",
    ]
FILES += [
    "derivatives/preprocessing/README", "derivatives/preprocessing/CHANGES",
    "derivatives/preprocessing/.bidsignore", "derivatives/preprocessing/dataset_description.json",
    "derivatives/preprocessing/task-overt_desc-preproc_events.json",
    "derivatives/preprocessing/qc/README", "derivatives/preprocessing/qc/icadecisions.json",
    "derivatives/syllable-datasets/README.md", "derivatives/syllable-datasets/manifest.tsv",
    "derivatives/syllable-datasets/dataset_description.json",
]


# openneuro-py always downloads these root files, whatever the patterns.
ALWAYS = {"README", "dataset_description.json", "participants.tsv", "participants.json"}


def selected(parts, subjects=None):
    matches = glob.glob_filter(FILES, include_patterns(parts, subjects))
    assert all(matches.values()), "every pattern must match, or openneuro-py raises"
    return {f for found in matches.values() for f in found} | ALWAYS


def test_default_is_raw_only():
    files = selected(("raw",))
    assert files == {f for f in FILES if not f.startswith("derivatives/")}


def test_subject_selection_keeps_metadata():
    files = selected(("raw", "preprocessed", "syllable-datasets"), subjects=[2])
    assert not any("sub-01" in f for f in files)
    assert {f for f in FILES if "sub-02" in f} <= files
    for metadata in ("stimuli/wordlist.tsv", "derivatives/preprocessing/.bidsignore",
                     "derivatives/preprocessing/qc/README",
                     "derivatives/syllable-datasets/manifest.tsv"):
        assert metadata in files


def test_each_part_selects_only_its_folder():
    assert selected("preprocessed") == ALWAYS | {f for f in FILES if f.startswith("derivatives/preprocessing/")}
    assert selected("syllable-datasets") == ALWAYS | {f for f in FILES if f.startswith("derivatives/syllable-datasets/")}


@pytest.mark.parametrize("parts,subjects", [(("raw",), [26]), (("raw",), ["x"]), ((), None), (("audio",), None)])
def test_invalid_selection(parts, subjects):
    with pytest.raises(ValueError):
        include_patterns(parts, subjects)
