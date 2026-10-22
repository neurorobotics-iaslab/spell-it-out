"""Download the Spell-IT-OUT dataset from OpenNeuro."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import mne

DEFAULT_DATA_PATH_ENV = "MNE_DATASETS_SPELLITOUT_DATA_PATH"
OPENNEURO_DATASET = "ds008968"
DEFAULT_VERSION = "1.1.0"
PARTS = ("raw", "preprocessed", "syllable-datasets")
N_SUBJECTS = 25

_PREPROCESSED = "derivatives/preprocessing"
_SYLLABLES = "derivatives/syllable-datasets"


def default_data_path() -> Path:
    """Where the dataset lives when no root is given.

    ``MNE_DATASETS_SPELLITOUT_DATA_PATH`` (environment variable or MNE
    configuration) if set, otherwise ``~/mne_data/spellitout``.
    """
    configured = mne.get_config(DEFAULT_DATA_PATH_ENV)
    if configured:
        return Path(configured).expanduser()
    return Path.home() / "mne_data" / "spellitout"


def _subject_labels(subjects: int | str | Iterable[int | str] | None) -> list[str] | None:
    if subjects is None:
        return None
    if isinstance(subjects, (int, str)):
        subjects = [subjects]
    labels = []
    for value in subjects:
        label = str(value).removeprefix("sub-")
        if not label.isdigit() or not 1 <= int(label) <= N_SUBJECTS:
            raise ValueError(f"Unknown subject {value!r}; subjects are 1-{N_SUBJECTS}.")
        labels.append(f"{int(label):02d}")
    return sorted(set(labels))


def include_patterns(
    parts: str | Iterable[str] = ("raw",),
    subjects: int | str | Iterable[int | str] | None = None,
) -> list[str]:
    """OpenNeuro ``include`` patterns for the requested parts and subjects.

    Root metadata (README, dataset_description, participants) is always
    downloaded by OpenNeuro, so every part yields a valid BIDS root.
    """
    parts = [parts] if isinstance(parts, str) else list(parts)
    unknown = sorted(set(parts) - set(PARTS))
    if unknown or not parts:
        raise ValueError(f"parts must be chosen from {PARTS}, got {parts}.")
    labels = _subject_labels(subjects)
    patterns: list[str] = []
    if "raw" in parts:
        # Leading "/" anchors at the root; otherwise "sub-*" would also match
        # the subject folders inside derivatives/.
        patterns += ["/*.json", "/*.tsv", "/stimuli"]
        patterns += [f"/sub-{label}" for label in labels] if labels else ["/sub-*"]
    if "preprocessed" in parts:
        # Dotfiles are only downloaded when named explicitly.
        patterns.append(f"{_PREPROCESSED}/.bidsignore")
        if labels:
            patterns += [
                f"{_PREPROCESSED}/README", f"{_PREPROCESSED}/CHANGES",
                f"{_PREPROCESSED}/*.json",
                f"{_PREPROCESSED}/qc/README", f"{_PREPROCESSED}/qc/*.json",
            ]
            for label in labels:
                patterns += [f"{_PREPROCESSED}/sub-{label}", f"{_PREPROCESSED}/qc/sub-{label}"]
        else:
            patterns.append(_PREPROCESSED)
    if "syllable-datasets" in parts:
        if labels:
            patterns.append(f"{_SYLLABLES}/*.*")
            patterns += [f"{_SYLLABLES}/*/sub-{label}_*" for label in labels]
        else:
            patterns.append(_SYLLABLES)
    return patterns


def download_dataset(
    path: str | Path | None = None,
    *,
    version: str = DEFAULT_VERSION,
    parts: str | Iterable[str] = ("raw",),
    subjects: int | str | Iterable[int | str] | None = None,
    **openneuro_kwargs: Any,
) -> Path:
    """Download Spell-IT-OUT from OpenNeuro and return the dataset root.

    Parameters
    ----------
    path
        Target folder. Default: :func:`default_data_path`.
    version
        OpenNeuro snapshot. Pinned by default for reproducibility.
    parts
        Any of ``"raw"`` (EEG, events and stimuli; the default),
        ``"preprocessed"`` (cleaned EEG and review tables) and
        ``"syllable-datasets"`` (HDF5 epochs). The full dataset is about 71 GiB:
        raw 50 GiB, preprocessed 12 GiB, syllable datasets 9 GiB.
    subjects
        Subjects to download, e.g. ``[1, 2]``. Default: all 25.
    **openneuro_kwargs
        Passed to :func:`openneuro.download`, e.g. ``max_concurrent_downloads``.

    Files that are already complete are not downloaded again. Private
    snapshots require ``openneuro-py login`` with an account that has access.
    """
    try:
        import openneuro
    except ImportError:
        raise ImportError(
            "Downloading needs openneuro-py: pip install 'spellitout[download]'"
        ) from None
    root = Path(path).expanduser() if path is not None else default_data_path()
    root.mkdir(parents=True, exist_ok=True)
    openneuro.download(
        dataset=OPENNEURO_DATASET,
        tag=version,
        target_dir=root,
        include=include_patterns(parts, subjects),
        **openneuro_kwargs,
    )
    return root
