# Paper processing and analysis

This directory contains the processing pipeline and paper analyses. Use the
entry points under `scripts/` in this directory. The obsolete script copies in
the repository-level `scripts/` and `reproducibility_old/` have been removed.
Administrative dataset-preparation and validation scripts remain at the
repository level; older results remain in `reproducibility_old/` pending review.

## Layout

- `scripts/`: numbered preprocessing entry points and paper-analysis commands.
- `processing/`: preprocessing, QC review, ICA, publishing and syllable extraction.
- `analysis/`: blink rate, ERD/ERS, plotting and phonetic statistics.
- `configs/`: default syllable-extraction configuration.
- `tests/`: tests of the copied processing and analysis code.
- `results/`: newly generated paper outputs (not versioned).

The installable `spellitout` package supplies dataset access and phonetic
resources, plus generic montage and epoching utilities. Processing and analysis
modules and their tests now live here rather than in the core package.

## Installation and execution

Run from the repository root:

```powershell
python -m pip install ./spellitout
python -m pip install -r reproducibility/requirements.txt
python -m reproducibility.scripts.01_preprocess --help
python -m reproducibility.scripts.08_build_syllable_datasets validate
python -m reproducibility.scripts.erd_ers --help
python -m pytest reproducibility/tests
```

Installation of `spellitout` is required. Launch the paper scripts as modules
from the repository root, for example `python -m reproducibility.scripts.erp --help`.
Use `python -m reproducibility.scripts.<script_name>` without the `.py` extension,
including for numbered stages. Direct execution with
`python reproducibility/scripts/<script_name>.py` is not supported: running from
the repository root alone does not ensure that Python can import `reproducibility`.


**Example**
```powershell
python -m reproducibility.scripts.01_preprocess --condition covert --subjects 1 --runs 1 2 3 --blink-channels FP1 EOG --blink-l-freq 1.5 --blink-threshold 65
```

Processing stages preserve their existing arguments, data formats and output
locations. Paper analyses write to `reproducibility/results/`; they retain their
output-path options for selecting another location.

Previous results are not copied from `reproducibility_old/results/`. To reuse
existing checkpoints, pass the corresponding old output path to an analysis
command rather than recomputing it. Moving those outputs is a separate decision.

The syllable configuration remains unchanged: preprocessed data by default (raw
data can be selected with `--source raw`), resampling to 512 Hz,
2–70 Hz band-pass and 50 Hz notch filtering with the configured causal mode,
and rejection of windows overlapping reviewed BAD_* spans. Standard resampling
is unchanged and should not be described as causal.

For `overtproduction`, stage 08 also calls `EmbeddingExtractor.extract_embedding`
on each duration-matched audio segment. The optional `torch`, `torchaudio` and
`transformers` dependencies and the extractor's Hugging Face checkpoint must be
available for an audio-enabled build. Validation does not load the model.
Embedding sequences are stored without temporal pooling as float32 arrays in
`audio_embeddings/data` (frames × features), with `audio_embeddings/offsets`
mapping each EEG sample to its frame interval, just like the audio offsets.
Group attributes record the model, revision, extraction method and hidden-state
index. The current extractor returns `hidden_states[0]`, not the penultimate
transformer layer; its output is stored unchanged. These are derived audio
representations, not a guarantee of voice anonymization.

To build EEG/labels only, without reading WAVs or loading the embedding model:

```powershell
python -m reproducibility.scripts.08_build_syllable_datasets apply --no-audio
```

`--no-audio` omits both `audio` and `audio_embeddings` groups and skips audio
availability/duration checks. Missing production annotations are still reported.

Administrative preparation and migration scripts remain outside this directory.
In particular, `prepare_data_V3.py` is not part of this paper pipeline copy.

## EEG sidecar migration and export

From the repository root, first inspect the metadata-only migration:

```powershell
python -m scripts.migrate_bids_sidecars validate
python -m scripts.migrate_bids_sidecars apply
```

By default both `data/spellitout` and its existing `derivatives/preprocessing`
dataset are selected. Use `--datasets raw` or `--datasets preprocessed` to
select one; `--bids-root` and `--preprocessed-root` accept other locations.
Validation is read-only. Apply backs up changed/deleted sidecars under
`backups/bids_sidecars/` with a journal before writing; `--backup-root` changes
the backup location. Signal, event, audio, coordinate and QC files are untouched.
No migration is needed on a repeat run once the new layout is canonical.

Raw metadata shared across runs live under each subject/session's `eeg/`:
`sub-XX_ses-01_task-<task>_eeg.json` and
`sub-XX_ses-01_task-<task>_channels.tsv`. Distinct values are never collapsed.

Stage 07 accepts either the old or migrated raw layout. It exports complete
EEG JSON next to recordings, including filters and reference, so MNE-BIDS does
not need to merge root-level metadata. Run-specific `Sources` are preserved,
so preprocessed EEG JSON normally remain run-specific. Identical channel tables
are shared as `sub-XX_ses-01_task-<task>_desc-preproc_channels.tsv`.
