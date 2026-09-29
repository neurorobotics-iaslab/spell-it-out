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
