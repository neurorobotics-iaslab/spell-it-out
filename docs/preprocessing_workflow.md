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
4. `04_amica` calculates the approved condition-level bad-channel union,
   interpolates that same set in every run, applies line-noise removal and the
   final common average reference, saves the concatenated ICA input, and fits
   AMICA in one invocation.
5. `05_review_ica` reviews AMICA components and records manual decisions;
   applying selected components remains a later explicit operation.
6. `06_blink_rate` summarizes reviewed spontaneous blink counts and rates for
   each run and task phase in one subject-level BIDS TSV.
7. `07_erd_ers` computes subject-weighted, full-trial ERD/ERS from the
   condition-level AMICA input or from an in-memory ICA-cleaned copy.

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
  --max-bad-duration 0.5 \
  --overwrite
```

Stage 7 epochs `repetition_cue` from -10 to 25 seconds, uses a -3 to -0.1
second baseline, computes Morlet TFRs from 2 to 98 Hz, applies CSD by default,
and reports baseline-normalised values as dB log-ratios. `BAD_*` spans are
unioned within each epoch and an epoch is rejected only when their total overlap
exceeds `--max-bad-duration`. Subjects contribute equally to the grand average.
Use `--vlim-db none` to let the plot choose symmetric limits automatically from
the grand-average data.

The default `amica_input` mode is useful before ICA review is complete. After
all component decisions are available, use `--input-stage cleaned`; this mode
applies only components marked `reject` in memory and does not create a fake
cleaned FIF.

## Running stage 4

```bash
python scripts/04_amica.py --condition silent --subjects 1 --device cuda
```

All selected runs must have completed annotation and EEG-channel decisions.
The default line-noise method is a 50 Hz notch; use `--line-noise-method prep`
to use only PREP's line-noise procedure instead. The script writes the
condition-level ICA-input checkpoint before starting the AMICA fit.

## Running stage 5

```bash
python scripts/05_review_ica.py --condition silent --subject 1 --reviewer Tommaso
```

This GUI records component keep/reject decisions in the AMICA component TSV;
it does not apply them to a cleaned EEG derivative.
