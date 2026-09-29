# Spell-IT-OUT

Spell-IT-OUT is an EEG dataset for imagined and overt production of Italian
syllables. The canonical raw dataset follows BIDS and stores the `overt` and
`silent` tasks below one dataset root.

## Dataset access

The local Python package provides a BIDS-native loader built on MNE-BIDS:

```python
from spellitout import SpellItOutDataset

dataset = SpellItOutDataset("data/spellitout_bids")
run = dataset.get_run(subject=1, task="silent", run=1)
print(run.raw, run.events)
```

## Preprocessing derivatives

The staged preprocessing workflow is documented in
[docs/preprocessing_workflow.md](docs/preprocessing_workflow.md). Its first
stage creates independent run-level minimal-processing derivatives without
channel interpolation or ICA:

```bash
python scripts/01_preprocess.py \
  --condition silent \
  --subjects 1 \
  --data-path data/spellitout_bids \
  --overwrite
```

The data location is provided through `--data-path` or the existing
`MNE_DATASETS_SPELLITOUT_DATA_PATH` environment variable.
