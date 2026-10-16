# spellitout

Python package to load the Spell-IT-OUT EEG dataset, built on MNE-BIDS, plus
the PhonItalia tables used to build the stimuli. Preprocessing and paper
analyses live in [`reproducibility/`](../reproducibility/README.md), not here.

## Installation

Python ≥ 3.10. From the repository root:

```bash
python -m pip install ./spellitout
```

## Opening the dataset

Pass the folder that contains `dataset_description.json`, or set the
`MNE_DATASETS_SPELLITOUT_DATA_PATH` environment variable and call
`SpellItOutDataset()` with no arguments.

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
| `audio_path`, `textgrid_path` | participant's speech and its alignment; `None` for covert runs |

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
