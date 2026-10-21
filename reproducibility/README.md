# Reproducibility

Everything needed to go from the raw BIDS dataset to the published derivatives
and to the figures and tables of the paper: the preprocessing pipeline with its
review GUIs, the syllable-dataset builder, and the analysis scripts.

## Setup

From the repository root:

```bash
python -m pip install ./spellitout
python -m pip install -r reproducibility/requirements.txt
```

Run every script as a module from the repository root:

```bash
python -m reproducibility.scripts.01_preprocess --help
```

Tests: `python -m pytest reproducibility/tests`.

## Layout

| Folder | Content |
|---|---|
| `scripts/` | entry points: numbered pipeline stages and paper analyses |
| `processing/` | preprocessing, QC review, ICA, export and syllable-dataset code |
| `analysis/` | blink rate, ERD/ERS, phonetic statistics, plotting helpers |
| `configs/` | default syllable-dataset configuration |
| `results/` | outputs of the paper analyses (not versioned) |

## Pipeline

| Stage | What it does | Output |
|---|---|---|
| `00_validate_bids` | runs the official BIDS Validator | report only |
| `01_preprocess` | minimal per-run preprocessing and artifact candidates | `desc-minproc_raw.fif`, `desc-annotationsqc.tsv` |
| `02_channel_qc` | PyPREP channel-quality evidence | `desc-channelqc.tsv` |
| `03_review_qc` | GUI: review artifacts and channels, run by run | the two tables above |
| `04_ica` | merges channel decisions, fits Extended Infomax | `desc-infomax*` files |
| `05_review_ica` | GUI: review ICA components | `desc-infomax_components.tsv` |
| `06_apply_ica` | removes rejected components, splits back into runs | `desc-infomax-cleaned_raw.fif` |
| `07_export_preprocessed` | validates and publishes the cleaned EEG as BIDS | `data/spellitout/derivatives/preprocessing/` |
| `08_build_syllable_datasets` | builds the HDF5 syllable datasets | `data/spellitout/derivatives/syllable-datasets/` |

Stages 01–06 write working files to `derivatives/preprocessing/` at the
repository root; they are large and not published. The raw dataset is never
modified, review tables are the only record of human decisions, and no stage
overwrites existing outputs without `--overwrite`. Each stage records its
configuration in the derivative metadata.

The examples below process subject 1, covert task; use `--condition overt` for
the other task. The raw dataset is found through `--data-path` or the
`MNE_DATASETS_SPELLITOUT_DATA_PATH` environment variable.

### 00 – Validate the BIDS dataset

```bash
python -m reproducibility.scripts.00_validate_bids data/spellitout
python -m reproducibility.scripts.00_validate_bids data/spellitout/derivatives/preprocessing
```

Uses an installed `bids-validator` if there is one, otherwise the pinned
official Deno package (install Deno from <https://deno.com/>). Arguments after
`--` go to the validator, e.g. `-- --verbose`.

### 01 – Minimal preprocessing

```bash
python -m reproducibility.scripts.01_preprocess --condition covert --subjects 1 --data-path data/spellitout
```

Per run: drops AF7 and AF8, resamples to 512 Hz, band-passes EEG and EOG at
1–100 Hz, and detects candidate blinks (from FP1, FPz, FP2 and the EOG by default;
`--blink-channels` selects others),
muscle artifacts and optionally high-amplitude spans. Nearby muscle candidates
are merged and those shorter than 100 ms dropped. Candidates are only
suggestions until reviewed in stage 03.

### 02 – Channel quality

```bash
python -m reproducibility.scripts.02_channel_qc --condition covert --subjects 1
```

Runs the PyPREP detectors (RANSAC included; `--no-ransac` to skip) and records
their evidence. No channel is interpolated or removed here.

### 03 – Review artifacts and channels

```bash
python -m reproducibility.scripts.03_review_qc --condition covert --subjects 1 --reviewer <name>
```

A Tk GUI to accept or dismiss each candidate, add manual annotations, and
decide each channel. Display filters and rereferencing only affect what you
see. Approved blinks become `BLINK`; other approved spans become `BAD_*`. All
annotations and channels of a run must be decided before stage 04.

### 04 – ICA

```bash
python -m reproducibility.scripts.04_ica --condition covert --subjects 1
```

Merges the channel decisions of the three runs so they all get the same
interpolations, applies a 50 Hz notch (`--line-noise-method prep` for PREP
instead), interpolates bad channels, sets the average reference, and
concatenates the runs. Extended Infomax is fitted leaving out `BAD_*` spans,
with the number of components set by the data rank; ICLabel probabilities and
variance contributions are computed for each component. The ICA input is saved
before the fit; `desc-infomax.json` records rank, convergence and the
full configuration.

### 05 – Review components

```bash
python -m reproducibility.scripts.05_review_ica --condition covert --subject 1 --reviewer <name>
```

Shows topography, spectrum, time course, ICLabel probabilities and explained
variance of each component. An automatic policy can pre-fill the decisions,
applied in order: reject muscle/eye/heart/line noise with p ≥ 0.80, review
channel noise with p ≥ 0.70, keep brain with p ≥ 0.60, keep components
explaining < 0.4 % of the variance. Everything else is decided by hand.

### 06 – Apply ICA

```bash
python -m reproducibility.scripts.06_apply_ica --condition covert --subjects 1
```

Needs every component marked `keep` or `reject`. Removes the rejected ones and
splits the recording back into single runs.

### 07 – Export the preprocessed derivative

```bash
python -m reproducibility.scripts.07_export_preprocessed validate
python -m reproducibility.scripts.07_export_preprocessed apply
```

`validate` is read-only and checks that every selected run has its cleaned
FIF, completed reviews, ICA decisions and matching raw events (`--subjects`,
`--tasks` and `--runs` restrict the selection). `apply` writes BrainVision
`desc-preproc` recordings with complete sidecars. Events include the
experimental events plus blinks and approved `BAD_*` spans, and `channels.tsv`
marks interpolated channels. The review tables and ICA metadata go to `qc/`,
which is listed in `.bidsignore`. Working checkpoints are not exported.

### 08 – Syllable datasets

```bash
python -m reproducibility.scripts.08_build_syllable_datasets validate
python -m reproducibility.scripts.08_build_syllable_datasets apply
```

Builds one HDF5 file per subject for each dataset in the configuration:

| Dataset | Task | EEG aligned to |
|---|---|---|
| `covert_cue` | covert | `syllabification_cue` |
| `overt_cue` | overt | `syllabification_cue` |
| `overt_production` | overt | `syllabification_syllable` (speech onset) |

Defaults, from
[`configs/syllable_dataset_default.json`](configs/syllable_dataset_default.json):

- raw EEG (`--source preprocessed` for the cleaned one);
- resampling to 512 Hz, then 2–70 Hz band-pass and 50 Hz notch with causal
  4th-order Butterworth filters (`--filter-phase zero` for zero-phase FIR).
- window from −0.3 to 0.6 s (`--eeg-tmin`, `--eeg-tmax`);
- windows overlapping a reviewed `BAD_*` span are dropped
  (`--no-exclude-bads` keeps them). Raw EEG takes its `BAD_*` spans from the
  preprocessed derivative, which must therefore exist;
- no audio.

The configuration can also add rereferencing, detrending or baseline
correction to each epoch.

Audio is opt-in and needs the participants' speech recordings in
`sourcedata/`, which are not part of the public release. `--audio` adds the speech segment of each `overt_production`
sample, cut to the annotated syllable duration. `--embeddings` (requires
`--audio`) also stores Wav2Vec2 embeddings of each segment. These come from
`facebook/wav2vec2-large-xlsr-53-italian` at a pinned revision, taken from
`hidden_states[0]`, the input of the transformer encoder. They need PyTorch
and `transformers`, which are not in the requirements (see the main README).

Each sample keeps its subject, run, trial, syllable and event time, and
`manifest.tsv` links it back to the source recording. `exclusions.tsv` lists
the events that were left out and why (`BAD_*` overlap, missing production
annotation, out of bounds). The resolved configuration and its SHA-256 are
stored in every file, so changing any option gives a different digest.
The HDF5 layout is described in the README that the builder writes next to
the files.

## Paper analyses

| Script | What it produces | Output in `results/` |
|---|---|---|
| `blink_rate` | blink rate per run and trial phase, from the published events and QC tables | `blink_rate/` |
| `erp` | occipital ERPs (O1, Oz, O2) for countdown dots vs syllable cues, overt and covert | `erp/` |
| `erd_ers` | Morlet time–frequency power (after CSD) and ERD/ERS topographies by trial phase | `erd_ers/` |
| `channel_statistics` | per-channel mean and standard deviation across participants | `channel_statistics/` |
| `signal_quality` | summary of QC decisions and syllable availability, without reading EEG | `signal_quality/<timestamp>/` |
| `make_images` | stimulus figures: syllable word cloud, phoneme frequencies vs PhonItalia, phonetic category matrix, example EEG | `other_images/` |

`erp` and `erd_ers` work on the raw or the preprocessed EEG (`--source`) and
run in two steps, `compute` and `plot` (`all` does both). The computed
checkpoints stay on disk, so figures can be redrawn with different rejection
settings without recomputing. For example, `erd_ers plot --max-bad-duration 0.5`
changes how much `BAD_*` overlap an epoch may have. Output folders are named
after the settings that produced them, and the saved `method.json` is checked
before a checkpoint is reused.

`make_images` has no options and reads the dataset from
`MNE_DATASETS_SPELLITOUT_DATA_PATH`.
