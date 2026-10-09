# spellitout

A core Python package for accessing the Spell-IT-OUT EEG dataset and its
phonetic resources. Processing, QC/ICA review, publication exports, and paper
analyses are provided separately under `reproducibility/` in the repository.

Generic helpers `make_montage()` and `epoch_data()` remain available in
`spellitout.utils`. Subject and run identifiers are exposed by `RunData`.

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

`get_run()` and `iter_runs()` return lightweight `RunData` descriptors. Listing
run IDs and paths does not open EEG recordings or read event tables. `run.raw`
opens the recording on first access; `run.events` independently reads and
validates the events TSV. Both results are cached for subsequent accesses.
`preload=True` loads EEG samples only when `run.raw` is first accessed.
Missing files and malformed events are reported when their content is accessed,
not while listing runs. Printing a descriptor does not load its content.

```python
for run in dataset.iter_runs(tasks="covert"):
    print(run.run_id, run.events_path)  # Metadata only; no raw/event loading.
```

For sequential processing, iterate over matching runs
```python
for run in dataset.iter_runs(
    subjects=[1, 2],
    tasks="covert",
    runs=[1, 2, 3],
    preload=True,
):
    print(run.run_id, run.raw)
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
