# Preprocessing workflow

The preprocessing derivative is deliberately staged. Each stage has an
inspectable output and must not silently make the next stage's scientific
decision.

1. `01_minproc` is run independently for each recording run. It creates a
   continuous derivative with channel exclusions, resampling, filtering, and
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

The source raw dataset is immutable. Each derivative sidecar records source
identity, configuration, and decisions required to reproduce the next stage.

## Running stage 1

```bash
python scripts/01_minproc.py \
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
