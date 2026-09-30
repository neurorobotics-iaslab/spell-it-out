# spellitout

A Python package for working with the Spell-IT-OUT EEG dataset and related utilities.

## Installation

```bash
pip install .
```

## Loading the BIDS dataset

`SpellItOutDataset` wraps MNE-BIDS while preserving Spell-IT-OUT-specific event,
audio, and TextGrid access. Pass the root containing `dataset_description.json`,
or set the `MNE_DATASETS_SPELLITOUT_DATA_PATH` environment variable.

```python
from spellitout import SpellItOutDataset

dataset = SpellItOutDataset("/path/to/spellitout")
run = dataset.get_run(subject=1, task="overt", run=1, preload=False)

print(run.raw)
print(run.events)
print(run.audio_path)
```

For sequential processing, iterate over matching runs so that only one recording
needs to be resident at a time:

```python
for run in dataset.iter_runs(
    subjects=[1, 2],
    tasks="silent",
    runs=[1, 2, 3],
    preload=True,
):
    print(run.run_id, run.raw)
```

Numeric software-event markers recorded by ROS-Neuro are exposed separately:

```python
from spellitout import EventCode

assert int(EventCode.REPETITION) == 782
```

## phonItalia resources

The packaged phonItalia tables can be loaded directly as Pandas DataFrames:

```python
from spellitout.it_phonetics import (
    load_phonitalia_phones,
    load_phonitalia_syllables,
    load_phonitalia_wordforms,
)

wordforms = load_phonitalia_wordforms()
phones = load_phonitalia_phones()
syllables = load_phonitalia_syllables()
```

The word-pronunciation lookup used by `split_into_phonemes()` is loaded lazily
on its first call and then cached for the remainder of the process.
