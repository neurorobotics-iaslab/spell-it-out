# Spell-IT-OUT

Spell-IT-OUT is an EEG dataset for imagined and overt production of Italian
syllables. The canonical raw dataset follows BIDS and stores the `overt` and
`covert` tasks below one dataset root.

## Dataset access

The local Python package provides a BIDS-native loader built on MNE-BIDS:

```python
from spellitout import SpellItOutDataset

dataset = SpellItOutDataset("data/spellitout")
run = dataset.get_run(subject=1, task="covert", run=1)
print(run.raw, run.events)
```

## Preprocessing derivatives

The staged preprocessing workflow is documented in
[docs/preprocessing_workflow.md](docs/preprocessing_workflow.md). Its first
stage creates independent run-level minimal-processing derivatives without
channel interpolation or ICA:

```bash
python scripts/01_preprocess.py \
  --condition covert \
  --subjects 1 \
  --data-path data/spellitout \
  --overwrite
```

The data location is provided through `--data-path` or the existing
`MNE_DATASETS_SPELLITOUT_DATA_PATH` environment variable.

After all QC and ICA decisions have been reviewed, validate and create the
publication-ready BIDS derivative with:

```bash
python scripts/07_export_preprocessed.py validate
python scripts/07_export_preprocessed.py apply
```

## Syllable-segment datasets

Stage 8 builds efficient HDF5 shards containing fixed-window EEG samples,
phonetic labels, and duration-matched audio for overt productions. Validate the
complete input inventory without writing anything, then apply the canonical
configuration:

```bash
python scripts/08_build_syllable_datasets.py validate
python scripts/08_build_syllable_datasets.py apply
```

The defaults are defined in
[`configs/syllable_dataset_default.json`](configs/syllable_dataset_default.json).
CLI options can select raw or preprocessed continuous data and override the EEG
window. The HDF5 outputs are explicitly **BIDS-derived analysis artifacts**, not
standard BIDS EEG files; each sample remains traceable through the manifest and
source BIDS entities.

The default configuration applies an additional 2–70 Hz band-pass filter, a
50 Hz notch filter, and resampling to 512 Hz before extracting EEG windows.
These choices are stored with the complete resolved configuration and its
SHA-256 digest in every HDF5 shard.
