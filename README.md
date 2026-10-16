# Spell-IT-OUT

Code for **Spell-IT-OUT**, an EEG dataset of overt and covert (imagined)
production of Italian sentences, syllable by syllable. 25 participants
listened to 20 sentences, repeated them, and then produced them one syllable
at a time at a fixed pace, three times aloud and three times in their mind.
The dataset contains raw and preprocessed EEG, the overt speech with
hand-corrected word and syllable alignment, and ready-to-use syllable epochs
for machine learning.

**Data:** TODO OpenNeuro link. The dataset README there describes the protocol,
the files and the events in detail.

## What is in this repository

| Folder | Content |
|---|---|
| [`spellitout/`](spellitout/) | Python package to load the dataset ([README](spellitout/README.md)) |
| [`reproducibility/`](reproducibility/) | Preprocessing pipeline, syllable-dataset builder, and the analyses and figures of the paper ([README](reproducibility/README.md)) |

## Installation

Python ≥ 3.10. From the repository root:

```bash
python -m pip install ./spellitout                        # data access only
python -m pip install -r reproducibility/requirements.txt # pipeline and paper analyses
```

The optional audio embeddings of the syllable datasets also need PyTorch and
`transformers`. They are not in the requirements because the right PyTorch
build depends on your OS and GPU: install it following
<https://pytorch.org/get-started/locally/>, then `python -m pip install transformers`.

## Quick start

```python
from spellitout import SpellItOutDataset

dataset = SpellItOutDataset("path/to/spellitout")
run = dataset.get_run(subject=1, task="overt", run=1)

run.raw          # mne.io.Raw, loaded on first access
run.events       # pandas DataFrame from events.tsv
run.audio_path   # participant's speech (overt runs only)
```

The same class opens the preprocessed derivative:
`SpellItOutDataset("path/to/spellitout/derivatives/preprocessing")`.

## License

The code is released under the [MIT license](LICENSE); the dataset under CC0.

## Citation

If you use the dataset or the code, please cite:

Cortecchia, T., Tortora, S., Tonin, L. *Spell-It-Out, a speech EEG dataset for
overt and covert production of Italian syllables.* TODO: journal and DOI.

This publication is part of the project PNRR-NGEU which has received funding
from the MUR – DM 118/2023.
