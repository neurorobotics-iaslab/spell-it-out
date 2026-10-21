# Spell-IT-OUT

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23536164.svg)](https://doi.org/10.5281/zenodo.23536164)

Code for **Spell-IT-OUT**, an EEG dataset of overt and covert (imagined)
production of Italian sentences, syllable by syllable. 25 participants
listened to 20 sentences, repeated them, and then produced them one syllable
at a time at a fixed pace, three times aloud and three times in their mind.
The dataset contains raw and preprocessed EEG, hand-corrected word and
syllable timing of the overt speech, and ready-to-use syllable epochs for
machine learning.

**Data:** [OpenNeuro ds008968](https://openneuro.org/datasets/ds008968),
doi:[10.18112/openneuro.ds008968.v1.0.0](https://doi.org/10.18112/openneuro.ds008968.v1.0.0).
The dataset README there describes the protocol, the files and the events in
detail. The participants' voice recordings are not part of the public release.

**Code:** archived on Zenodo, doi:[10.5281/zenodo.23536164](https://doi.org/10.5281/zenodo.23536164)
(all versions); v0.1.0, the version used in the paper, is
doi:[10.5281/zenodo.23536165](https://doi.org/10.5281/zenodo.23536165).

## What is in this repository

| Folder | Content |
|---|---|
| [`spellitout/`](spellitout/) | Python package to load the dataset ([README](spellitout/README.md)) |
| [`reproducibility/`](reproducibility/) | Preprocessing pipeline, syllable-dataset builder, and the analyses and figures of the paper ([README](reproducibility/README.md)) |

## Installation

Python ≥ 3.10. From the repository root:

```bash
python -m pip install "./spellitout[download]"            # data access and download
python -m pip install -r reproducibility/requirements.txt # pipeline and paper analyses
```

The optional audio embeddings of the syllable datasets also need PyTorch and
`transformers`. They are not in the requirements because the right PyTorch
build depends on your OS and GPU: install it following
<https://pytorch.org/get-started/locally/>, then `python -m pip install transformers`.

## Quick start

```python
from spellitout import SpellItOutDataset, download_dataset

download_dataset(subjects=[1])   # raw data of subject 1 into ~/mne_data/spellitout
dataset = SpellItOutDataset()
run = dataset.get_run(subject=1, task="overt", run=1)

run.raw          # mne.io.Raw, loaded on first access
run.events       # pandas DataFrame from events.tsv
```

The same class opens the preprocessed derivative:
`SpellItOutDataset("path/to/spellitout/derivatives/preprocessing")`.

## License

The code is released under the [MIT license](LICENSE); the dataset under CC0.

## Citation

If you use the dataset or the code, please cite the paper:

Cortecchia, T., Tortora, S., Kiesner, J., Menegatti, E., Tonin, L. *Spell-It-Out, a speech EEG dataset for
overt and covert production of Italian syllables.* (in preparation)

and the dataset:

Cortecchia, T., Tortora, S., Kiesner, J., Menegatti, E., Tonin, L. *Spell-IT-Out* (version 1.0.0).
OpenNeuro. doi:[10.18112/openneuro.ds008968.v1.0.0](https://doi.org/10.18112/openneuro.ds008968.v1.0.0)

and, if you use the code, the release you used, e.g.:

Cortecchia, T., Tortora, S., Kiesner, J., Menegatti, E., Tonin, L. *Spell-IT-Out: data access and
reproducibility code* (v0.1.0). Zenodo. doi:[10.5281/zenodo.23536165](https://doi.org/10.5281/zenodo.23536165)

This publication is part of the project PNRR-NGEU which has received funding
from the MUR – DM 118/2023.
