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
`paper_analysis/`, not stages of the dataset-production pipeline.

## Input dataset

The input must be the unified BIDS root containing both `task-overt` and
`task-covert` recordings:

```text
dataset_description.json
task-overt_eeg.json
task-covert_eeg.json
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
python scripts/00_validate_bids.py data/spellitout
```

## Stage 1: minimal run-level preprocessing

```bash
python scripts/01_preprocess.py \
  --condition covert \
  --subjects 1 \
  --runs 1 2 3 \
  --data-path data/spellitout \
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
  --condition covert \
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
  --condition covert \
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
  --condition covert \
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
  --condition covert \
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
  --condition covert \
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

## Paper utility: ERD/ERS checkpoints and plots

Compute full epoch-level TFR checkpoints from the published preprocessed data:

```bash
python paper_analysis/erd_ers.py compute --source preprocessed
```

The raw BIDS signals can be analysed with the same reviewed annotations and a
separate set of checkpoints:

```bash
python paper_analysis/erd_ers.py compute --source raw
```

Plotting is independent of computation, so the allowed `BAD_*` overlap can be
changed without recomputing the Morlet transforms:

```bash
python paper_analysis/erd_ers.py plot \
  --source preprocessed \
  --max-bad-duration 0.5
```

Outputs are written below
`paper_analysis/results/erd_ers/source-<source>/`, in readable directories
identifying the event, selected runs, time window, frequencies, sampling rate,
wavelet cycle divisor, CSD, and worker count. The configuration in `method.json`
is checked before checkpoint reuse. Epochs are resampled to 128 Hz by
default before the TFR is computed, and CSD is applied unless `--no-csd` is
passed. All extractable epochs are retained; `BAD_*` overlap is recorded in
the checkpoint metadata and applied only during plotting.

## Paper utility: ERP checkpoints and plots

Compute compact subject-level occipital ROI checkpoints from the published
preprocessed data and then create the comparison figure:

```bash
python paper_analysis/erp.py compute --source preprocessed
python paper_analysis/erp.py plot --source preprocessed
```

Both operations can be requested together, and the raw BIDS signals are also
supported:

```bash
python paper_analysis/erp.py all --source raw
```

The defaults compare `pre_syllabification_dot` and `syllabification_cue` in
the overt and covert tasks, using the O1/Oz/O2 ROI from -0.2 to 0.6 seconds.
Epochs overlapping `BAD_*` annotations are rejected. Reviewed blink spans are
retained by default and can additionally be rejected with `--reject-blinks`.
The in-memory 30 Hz low-pass can be disabled with `--lowpass none`.
Checkpoints and figures are stored below `paper_analysis/results/erp` in
readable directories identifying source, tasks, events, runs, channels, epoch
window, low-pass, baseline, and blink rejection. The saved `method.json`
configuration is checked before checkpoint reuse. Figure directories identify
the selected subjects and plotted zero time explicitly.

## Paper utility: blink-rate summaries

```bash
python paper_analysis/blink_rate.py \
  --conditions covert overt \
  --overwrite
```

The utility reads reviewed run-level annotations and writes `run_level.tsv`,
`subject_phase.tsv`, `method.json`, and one article-ready wide table per task
(`mean_blink_rate_task-<task>.tsv`) below `paper_analysis/results/blink_rate`.
Dismissed blinks are excluded. The
`review_complete` column identifies runs with unresolved blink decisions.
Approved `BAD_*` spans are excluded from both duration and blink counting by
default; pass `--include-bad-spans` to retain them.

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
`data/spellitout/derivatives/preprocessing`. Final continuous EEG is
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

The canonical configuration creates `covert_cue`, `overt_cue`, and
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

The distributed default configuration first resamples to 512 Hz, then applies
a 2–70 Hz band-pass filter and a 50 Hz notch filter before window extraction. Changing
the JSON configuration produces a different recorded configuration digest.

`--exclude-bad-overlap` / `--no-exclude-bad-overlap` overrides
`eeg.exclude_bad_overlap` (enabled by default). With
`eeg.bad_annotation_source="auto"`, raw EEG uses the corresponding published
preprocessing events for reviewed `BAD_*` spans; preprocessed EEG uses its own
events. Raw-based rejection therefore requires the preprocessing derivative,
but disabling rejection does not. `--bad-annotation-source` can explicitly
override this selection.

`--filter-phase causal` (JSON: `preprocessing.filter_phase="causal"`) applies
forward-only fourth-order Butterworth IIR band-pass and notch filters to the
continuous signal before epoch extraction. The default `zero` retains MNE's
zero-phase FIR filters. Causal filters introduce phase delay; they do not make
an already preprocessed source causal. Resampling remains unchanged and is not
causal, so this option does not make the entire pipeline online-compatible.
The resolved configuration records the operation order in
`preprocessing.operation_order` so that the new order has a distinct digest.

```bash
python scripts/08_build_syllable_datasets.py validate --source raw --exclude-bad-overlap --filter-phase causal
```

## Reproducibility and overwrite policy

All stages refuse to replace existing primary outputs unless `--overwrite` is
explicitly supplied. Derivative metadata records the producing stage and its
configuration. Review TSVs are the authoritative human-decision artifacts;
the raw BIDS dataset and minimally processed FIF files remain immutable during
review.
