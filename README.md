# spell-it-out
Spell-IT-Out, an EEG dataset for inner and overt spelling of Italian syllables

## Preprocessing derivatives

The refactored preprocessing workflow is documented in
[docs/preprocessing_workflow.md](docs/preprocessing_workflow.md). Its first
executable stage creates independent, run-level minimal-processing derivatives
without channel interpolation or ICA:

```bash
python scripts/01_minproc.py --condition silent --subjects 1 --overwrite
```

The data location is provided through `--data-path` or the existing
`MNE_DATASETS_SPELLITOUT_DATA_PATH` environment variable.
