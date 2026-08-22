# Preprocessing workflow

The preprocessing derivative is deliberately staged. Each stage has an
inspectable output and must not silently make the next stage's scientific
decision.

1. `01_preprocess` is run independently for each recording run. It creates a
   continuous derivative with channel exclusions, resampling, EEG/EOG filtering, and
   non-BAD QC candidates (`BLINK`, `MUSCLE_CANDIDATE`,
   `AMPLITUDE_CANDIDATE`). It never rereferences, interpolates channels, or
   applies ICA.
2. `02_channel_qc` creates the per-run PREP evidence columns in the editable
   channel-QC table. It never interpolates channels.
3. `03_review_qc` is the human review gate. It updates only
   `desc-qc_annotations.tsv` and `desc-channelqc.tsv`: global artifact candidates
   can be approved or dismissed, and every channel receives a per-run decision.
4. `04_amica` or `04_ica` calculates the approved condition-level bad-channel union,
   interpolates that same set in every run, applies line-noise removal and the
   final common average reference, saves the concatenated ICA input, and fits
   AMICA or Extended Infomax in one invocation.
5. `05_review_ica` reviews either decomposition and records manual decisions;
   applying selected components remains a later explicit operation.
6. `06_blink_rate` summarizes reviewed spontaneous blink counts and rates for
   each run and task phase in one subject-level BIDS TSV.
7. `07_erd_ers` computes and saves all full-trial epoch TFRs per subject,
   including BAD-span evidence; paper figures are made separately from those
   checkpoints.

The source raw dataset is immutable. Each derivative sidecar records source
identity, configuration, and decisions required to reproduce the next stage.

## Running stage 1

```bash
python scripts/01_preprocess.py \
  --condition silent \
  --subjects 1 \
  --data-path /path/to/dataset \
  --overwrite
```

The default output root is `derivatives/preprocessing`. Automatic artifact
detections are candidates, not exclusions. Do not approve blinks as `BAD_*`;
they should remain available for ICA to learn ocular components.
Automatic muscle candidates separated by at most 100 ms are merged, and spans
shorter than 100 ms after merging are discarded before the review table is
written.

To update checkpoints generated before EOG filtering was included, first run a
dry run, then explicitly replace only the affected minproc FIFs:

```bash
python scripts/01_filter_existing_eog.py
python scripts/01_filter_existing_eog.py --overwrite
```

The migration uses the Stage 1 1--100 Hz band-pass by default; set `--l-freq`
and `--h-freq` to match a non-default original Stage 1 invocation.

## Running stage 2

```bash
python scripts/02_channel_qc.py --condition silent --subjects 1 --overwrite
```

This writes `desc-channelqc.tsv`, which holds the PREP detector evidence,
PSD/correlation margins, and the reviewer’s final channel decision. It
intentionally does not write a repaired EEG file.

## Running stage 3

```bash
python scripts/03_review_qc.py --condition silent --subjects 1
```

The Tk review application autosaves changes to the two canonical TSVs. It does
not modify the minimally processed raw FIF.

Its display-only controls can apply high-pass, low-pass, notch, and average
reference operations to a temporary in-memory copy for visual inspection. They
never alter the saved FIF, review TSVs, PREP evidence, or later ICA input.

## Running blink-rate summaries

```bash
python scripts/06_blink_rate.py \
  --conditions silent overt \
  --subjects 1 2 3
```

The output is written under `derivatives/blinkrate`, with one TSV per
subject/session containing all tasks, runs, and the `run`, `listening`,
`sentence_repetition`, and `syllabation` rows. Blink annotations marked
`keep` and still `unreviewed` are counted; dismissed annotations are excluded.
The `review_complete` column identifies runs that still contain unreviewed
blink annotations. Approved `BAD_*` spans are retained in the denominator by
default; add `--exclude-bad-spans` to remove them from both duration and blink
counting. Stage 6 uses a 150 ms minimum centre-to-centre blink separation by
default; change it with `--min-blink-separation-ms`. This affects only the
summary counts, never the reviewed annotation TSVs.

## Running ERD/ERS analysis

```bash
python scripts/07_erd_ers.py \
  --condition silent \
  --subjects 1 2 3 \
  --input-stage amica_input \
  --overwrite
```

Stage 7 epochs `repetition_cue` from -10 to 25 seconds, computes Morlet TFRs
from 2 to 98 Hz, and applies CSD by default. It retains every full-length epoch,
including epochs that overlap `BAD_*` spans. Each subject checkpoint contains an
epoch-level TFR, an epoch table with the unioned BAD-overlap duration, and a TSV
copy of all source annotations. Events at recording boundaries are retained in
the epoch table with `tfr_available=False` and `TOO_SHORT` rather than silently
discarded.

The default `amica_input` mode is useful before ICA review is complete. After
all component decisions are available, use `--input-stage cleaned`; this mode
applies only components marked `reject` in memory and does not create a fake
cleaned FIF.

Create the paper figure separately, choosing the BAD-overlap criterion at plot
time:

```bash
python per_paper/plot_erd_ers.py \
  --condition silent \
  --subjects 1 2 3 \
  --max-bad-duration 0.5 \
  --vlim-db none
```

The plotter selects eligible epochs, averages within each subject, applies the
-3 to -0.1 second baseline as a dB log-ratio, then averages subjects equally.

## Running stage 4

```bash
python scripts/04_amica.py --condition silent --subjects 1 --device cuda

# Extended Infomax alternative
python scripts/04_ica.py --condition silent --subjects 1
```

All selected runs must have completed annotation and EEG-channel decisions.
The default line-noise method is a 50 Hz notch; use `--line-noise-method prep`
to use only PREP's line-noise procedure instead. The script writes the
condition-level ICA-input checkpoint before starting the fit. Extended
Infomax outputs use `desc-infomax*`, so they can coexist with AMICA outputs.

## Running stage 5

```bash
python scripts/05_review_ica.py --condition silent --subject 1 --reviewer Tommaso

# Review Extended Infomax instead
python scripts/05_review_ica.py --condition silent --subject 1 --decomposition infomax --reviewer Tommaso
```

This GUI records component keep/reject decisions in the selected decomposition's
component TSV; it does not apply them to a cleaned EEG derivative.
