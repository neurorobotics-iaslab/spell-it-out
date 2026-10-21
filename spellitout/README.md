# spellitout

Python package to load the Spell-IT-OUT EEG dataset, built on MNE-BIDS, plus
the PhonItalia tables used to build the stimuli. Preprocessing and paper
analyses live in [`reproducibility/`](../reproducibility/README.md), not here.

## Installation

Python ≥ 3.10. From the repository root:

```bash
python -m pip install ./spellitout              # data access
python -m pip install "./spellitout[download]"  # also downloading from OpenNeuro
```

## Downloading the dataset

`download_dataset()` fetches the data from
[OpenNeuro ds008968](https://openneuro.org/datasets/ds008968) with
`openneuro-py`, pinned to snapshot 1.0.0. Choose what you need: the whole
dataset is about 71 GiB.

```python
from spellitout import download_dataset

root = download_dataset()                                  # raw EEG, events, stimuli (50 GiB)
root = download_dataset(parts=["syllable-datasets"])       # HDF5 epochs only (9 GiB)
root = download_dataset(parts=["raw", "preprocessed"], subjects=[1, 2])
```

`parts` can combine `"raw"`, `"preprocessed"` (12 GiB) and
`"syllable-datasets"`. Files already downloaded are skipped, so an interrupted
download can simply be restarted.

## Opening the dataset

Without arguments, `download_dataset()` and `SpellItOutDataset()` use the same
folder: `MNE_DATASETS_SPELLITOUT_DATA_PATH` if set (as an environment variable
or in the MNE configuration), otherwise `~/mne_data/spellitout`. You can also
pass any folder that contains `dataset_description.json`.

```python
from spellitout import SpellItOutDataset

dataset = SpellItOutDataset("path/to/spellitout")
dataset.subjects       # ('01', ..., '25')
dataset.tasks          # ('covert', 'overt')
dataset.participants   # participants.tsv
dataset.wordlist       # sentences: id, sentence, sentence_syll
```

The preprocessed derivative opens the same way:
`SpellItOutDataset("path/to/spellitout/derivatives/preprocessing")`.

## Runs

`get_run()` selects one recording, `iter_runs()` loops over several. Filters
accept numbers or BIDS labels (`1`, `"01"`, `"sub-01"`) and lists of them.

```python
run = dataset.get_run(subject=1, task="overt", run=1)

for run in dataset.iter_runs(subjects=[1, 2], tasks="covert", preload=True):
    print(run.run_id, run.raw)
```

Both return a `RunData`. Nothing is read from disk until you ask for it:
`raw` and `events` are loaded on first access and then cached, so listing runs
or reading their paths is free.

| Attribute | Content |
|---|---|
| `subject`, `session`, `task`, `run` | BIDS labels (`"01"`, `"overt"`, ...) |
| `subject_id`, `run_id`, `run_number` | `"sub-01"`, `"sub-01_ses-01_task-overt_run-01"`, `1` |
| `raw` | `mne.io.Raw` with the events as annotations; `preload=True` loads the samples too |
| `events` | `events.tsv` as a DataFrame |
| `bids_path`, `events_path` | file locations |
| `audio_path`, `textgrid_path` | participant's speech and its alignment, from `sourcedata/`; `None` when not available (the public release does not include them) |

The event columns are described in the dataset README. One thing to know:
`na` is a valid syllable, so if you read an `events.tsv` yourself with pandas,
pass `keep_default_na=False`.

## Phonetic resources

The PhonItalia tables ship with the package:

```python
from spellitout.it_phonetics import (
    load_phonitalia_phones,
    load_phonitalia_syllables,
    load_phonitalia_wordforms,
    split_into_phonemes,
    string2phonlist,
)

split_into_phonemes("il medico")   # 'il mE.di.ko'
string2phonlist("mE")              # Phoneme objects with their phonetic categories
```

`spellitout.utils` has two small helpers: `make_montage()` builds a 10–20
montage for a channel subset, and `epoch_data()` cuts fixed windows from a
NumPy array around event onsets.
