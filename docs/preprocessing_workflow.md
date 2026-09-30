# Preprocessing workflow

The preprocessing pipeline consumes the canonical Spell-IT-OUT BIDS dataset
and writes BIDS-like derivatives below `derivatives/`. Raw BIDS files are never
modified.

The stages deliberately separate automatic evidence, human decisions, ICA
fitting, and component application:

1. `01_preprocess.py` creates one minimally processed FIF per run and an
   editable artifact-candidate table.
2. `02_channel_qc.py` calculates PREP channel-quality evidence without
   interpolating channels.
3. `03_review_qc.py` is the run-level human review gate for artifacts and
   channels.
4. `04_ica.py` harmonizes reviewed channels across runs, creates the
   condition-level ICA input, and fits Extended Infomax.
5. `05_review_ica.py` reviews ICLabel evidence and component decisions.
6. `06_apply_ica.py` applies the completed component decisions and restores
   separate cleaned run-level FIF files.
7. `07_export_preprocessed.py` validates and exports the publication-ready
   preprocessing derivative in BrainVision format.
8. `08_build_syllable_datasets.py` creates the configurable BIDS-derived HDF5
   datasets used for syllable-level machine-learning experiments.

Stages 1–6 form the reviewed preprocessing sequence. Stage 7 publishes its
continuous output and Stage 8 derives the configurable syllable datasets.
ERD/ERS and blink-rate analyses are paper-specific utilities under
`per_paper/`, not stages of the dataset-production pipeline.

## Input dataset

The input must be the unified BIDS root containing both `task-overt` and
`task-silent` recordings:

```text
dataset_description.json
task-overt_eeg.json
task-silent_eeg.json
channels.tsv
sub-01/
  ses-01/
    eeg/
      sub-01_ses-01_task-overt_run-01_eeg.vhdr
      sub-01_ses-01_task-overt_run-01_events.tsv
      ...
```

Pass the root with `--data-path`, or set
`MNE_DATASETS_SPELLITOUT_DATA_PATH`. The package uses MNE-BIDS to resolve raw
recordings, inherited metadata, channel types, and event annotations.

The raw dataset can be checked independently with:

```bash
python scripts/00_validate_bids.py data/spellitout_bids
```

## Stage 1: minimal run-level preprocessing

```bash
python scripts/01_preprocess.py \
  --condition silent \
  --subjects 1 \
  --runs 1 2 3 \
  --data-path data/spellitout_bids \
  --overwrite
```

For each run, Stage 1:

- drops the configured permanently excluded channels (`AF7` and `AF8` by
  default);
- applies the configured EEG/EOG band-pass filter;
- resamples to 512 Hz by default;
- detects blink, muscle, and optional high-amplitude candidates;
- writes `desc-minproc_raw.fif` and `desc-qc_annotations.tsv`.

Candidate annotations are evidence, not exclusions. Blinks remain available
for ICA. Nearby muscle candidates are merged and spans shorter than the
configured minimum duration are discarded before the table is written.

## Stage 2: channel-quality evidence

```bash
python scripts/02_channel_qc.py \
  --condition silent \
  --subjects 1 \
  --runs 1 2 3 \
  --overwrite
```

Stage 2 runs PREP channel detectors on each minimally processed run and writes
`desc-channelqc.tsv`. RANSAC is enabled by default and can be disabled with
`--no-ransac`. This stage records detector evidence and margins but never
interpolates or removes channels from the saved FIF.

## Stage 3: run-level review

```bash
python scripts/03_review_qc.py \
  --condition silent \
  --subjects 1 \
  --reviewer Tommaso
```

The Tk application edits only the two canonical review tables:

- `desc-qc_annotations.tsv` for artifact candidates and approved `BAD_*`
  spans;
- `desc-channelqc.tsv` for per-run channel decisions.

Display filters and rereferencing in the GUI operate on an in-memory copy and
never change the saved signal. Every annotation and EEG-channel decision must
be complete before Stage 4.

## Stage 4: Extended Infomax

```bash
python scripts/04_ica.py \
  --condition silent \
  --subjects 1 \
  --runs 1 2 3 \
  --overwrite
```

Stage 4:

- builds the union of reviewed interpolation/exclusion decisions across the
  selected runs;
- applies the same sensor geometry to every run;
- removes line noise with a notch filter by default, or PREP when requested
  with `--line-noise-method prep`;
- interpolates reviewed bad channels and removes channels explicitly marked
  `exclude_from_ica`;
- applies the average EEG reference;
- concatenates the runs and retains technical boundary annotations;
- omits approved `BAD_*` samples only while fitting Extended Infomax;
- estimates the effective rank and limits the component count accordingly;
- calculates ICLabel probabilities and sensor-space component contributions.

The main outputs are:

```text
desc-infomax-input_raw.fif
desc-infomax-input_channels.tsv
desc-infomax_ica.fif
desc-infomax_components.tsv
desc-infomax.json
```

The condition-level input checkpoint is saved before the potentially long ICA
fit. The metadata JSON records rank, convergence diagnostics, fit sample count,
condition number, configuration, and channel harmonization decisions.

## Stage 5: component review

```bash
python scripts/05_review_ica.py \
  --condition silent \
  --subject 1 \
  --reviewer Tommaso
```

The GUI displays the component topography, full-recording PSD, continuous or
event-averaged activity, ICLabel probabilities, and variance contribution.
The optional fixed automatic policy can initialize decisions; remaining
components must be reviewed manually. Decisions are saved in
`desc-infomax_components.tsv`. The ICA input and fitted decomposition are not
modified.

## Stage 6: apply reviewed ICA decisions

```bash
python scripts/06_apply_ica.py \
  --condition silent \
  --subjects 1 \
  --runs 1 2 3 \
  --overwrite
```

Stage 6 requires every component to be marked `keep` or `reject`. It applies
the rejected-component list to the concatenated input, splits the result at
the technical run boundaries, and writes:

```text
sub-*_ses-01_task-*_run-*_desc-infomax-cleaned_raw.fif
```

## Paper utility: ERD/ERS checkpoints

Use the reviewed ICA input directly:

```bash
python per_paper/07_erd_ers.py \
  --condition silent \
  --subjects 1 2 3 \
  --input-stage ica_input \
  --overwrite
```

Or use the run-level ICA-cleaned outputs from Stage 6:

```bash
python per_paper/07_erd_ers.py \
  --condition silent \
  --subjects 1 2 3 \
  --input-stage cleaned \
  --overwrite
```

The stage epochs `repetition_cue`, resamples to 128 Hz by default, computes
Morlet TFRs, and applies CSD unless `--no-csd` is passed. Full-length epochs
are retained even when they overlap `BAD_*` spans; overlap evidence is stored
for selection during later analysis.

## Paper utility: blink-rate summaries

```bash
python per_paper/08_blink_rate.py \
  --conditions silent overt \
  --subjects 1 2 3
```

The utility reads reviewed run-level annotations and writes subject/session tables
below `derivatives/blinkrate`. Dismissed blinks are excluded. The
`review_complete` column identifies runs with unresolved blink decisions.
Approved `BAD_*` spans remain in the denominator by default; pass
`--exclude-bad-spans` to remove them from both duration and blink counting.

## Stage 7: publication export

Run the read-only preflight first:

```bash
python scripts/07_export_preprocessed.py validate
```

The validator requires a final ICA-cleaned FIF, completed annotation and
channel reviews, completed component decisions, ICA metadata, and matching raw
BIDS events for every selected run. A limited selection can be checked with
`--subjects`, `--tasks`, and `--runs`.

After validation succeeds, create the derivative:

```bash
python scripts/07_export_preprocessed.py apply
```

By default, the export is written to
`data/spellitout_bids/derivatives/preprocessing`. Final continuous EEG is
stored as `desc-preproc_eeg` BrainVision data. Standard `events.tsv` files
contain experimental events plus retained blinks and approved `BAD_*` spans;
standard `channels.tsv` files describe the final channel set and interpolation
status.

The complete project-specific audit trail is copied below `qc/`: annotation
reviews, channel reviews, the condition-level channel plan, ICA component
decisions, and ICA fit metadata. That directory is listed in `.bidsignore` so
the files remain distributed without being misinterpreted as standard BIDS
tables. Large reproducible checkpoints (`minproc`, concatenated ICA input, and
the fitted ICA FIF) are not exported.

The exporter never changes the working checkpoint tree. It refuses to write
into a non-empty output unless `--overwrite` is explicitly supplied.

## Stage 8: syllable-segment HDF5 datasets

Run a read-only validation of the canonical sample definition:

```bash
python scripts/08_build_syllable_datasets.py validate
```

Then generate subject-level shards:

```bash
python scripts/08_build_syllable_datasets.py apply
```

The canonical configuration creates `silent_cue`, `overt_cue`, and
`overt_production` datasets. EEG is aligned to the relevant syllable event;
overt-production samples also contain audio cropped to the annotated production
duration. Epochs overlapping `BAD_*` spans are excluded, while `BLINK` alone is
retained. Missing production annotations and all other exclusions are recorded
in `exclusions.tsv` rather than synthesized.

The configuration can select raw or published preprocessed BIDS recordings and
can add filtering, notch filtering, resampling, rereferencing, detrending, or
baseline correction. These operations are applied to an in-memory copy and are
stored verbatim, with a SHA-256 digest, in each HDF5 file. The output is
BIDS-derived but deliberately not presented as a validator-compatible BIDS EEG
datatype.

The distributed default configuration applies a 2–70 Hz band-pass filter, a
50 Hz notch filter, and resampling to 512 Hz before window extraction. Changing
the JSON configuration produces a different recorded configuration digest.

## Reproducibility and overwrite policy

All stages refuse to replace existing primary outputs unless `--overwrite` is
explicitly supplied. Derivative metadata records the producing stage and its
configuration. Review TSVs are the authoritative human-decision artifacts;
the raw BIDS dataset and minimally processed FIF files remain immutable during
review.
