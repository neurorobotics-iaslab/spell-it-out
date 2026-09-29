"""BIDS-native access to the Spell-IT-OUT dataset."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
import os
from pathlib import Path

import mne
from mne_bids import BIDSPath, find_matching_paths, read_raw_bids
import pandas as pd


DEFAULT_DATA_PATH_ENV = "MNE_DATASETS_SPELLITOUT_DATA_PATH"
EEG_EXTENSIONS = (".vhdr", ".edf", ".bdf", ".set")
REQUIRED_EVENT_COLUMNS = ("onset", "duration", "trial_type")


def _label(value: str | int, entity: str) -> str:
    """Normalize a BIDS entity label without its ``<entity>-`` prefix."""
    prefix = f"{entity}-"
    label = str(value)
    if label.startswith(prefix):
        label = label[len(prefix) :]
    if not label:
        raise ValueError(f"Empty BIDS {entity} label.")
    if entity in {"sub", "ses", "run"} and label.isdigit():
        label = f"{int(label):02d}"
    return label


def _labels(
    values: str | int | Iterable[str | int] | None,
    entity: str,
) -> list[str] | None:
    if values is None:
        return None
    if isinstance(values, (str, int)):
        values = [values]
    return [_label(value, entity) for value in values]


@dataclass(slots=True)
class RunData:
    """One loaded BIDS EEG recording and its Spell-IT-OUT metadata."""

    bids_path: BIDSPath
    raw: mne.io.BaseRaw
    events: pd.DataFrame
    events_path: Path
    audio_path: Path | None = None
    textgrid_path: Path | None = None

    @property
    def subject(self) -> str:
        return str(self.bids_path.subject)

    @property
    def subject_id(self) -> str:
        return f"sub-{self.subject}"

    @property
    def session(self) -> str | None:
        return self.bids_path.session

    @property
    def task(self) -> str:
        return str(self.bids_path.task)

    @property
    def run(self) -> str | None:
        return self.bids_path.run

    @property
    def run_number(self) -> int | None:
        return None if self.run is None else int(self.run)

    @property
    def run_id(self) -> str:
        return self.bids_path.basename.rsplit("_", maxsplit=1)[0]


class SpellItOutDataset:
    """Discover and load Spell-IT-OUT recordings from a local BIDS root."""

    def __init__(self, root: str | Path | None = None):
        if root is None:
            root = os.environ.get(DEFAULT_DATA_PATH_ENV)
        if root is None:
            raise ValueError(
                "Dataset root not provided and "
                f"{DEFAULT_DATA_PATH_ENV} is not set."
            )
        self.root = Path(root).expanduser().resolve()
        if not self.root.is_dir():
            raise FileNotFoundError(f"BIDS dataset root not found: {self.root}")
        description = self.root / "dataset_description.json"
        if not description.is_file():
            raise FileNotFoundError(
                f"BIDS dataset_description.json not found: {description}"
            )

    def _matching_paths(
        self,
        *,
        subjects: str | int | Iterable[str | int] | None = None,
        sessions: str | int | Iterable[str | int] | None = None,
        tasks: str | Iterable[str] | None = None,
        runs: str | int | Iterable[str | int] | None = None,
    ) -> list[BIDSPath]:
        paths = find_matching_paths(
            self.root,
            subjects=_labels(subjects, "sub"),
            sessions=_labels(sessions, "ses"),
            tasks=_labels(tasks, "task"),
            runs=_labels(runs, "run"),
            datatypes="eeg",
            suffixes="eeg",
            extensions=EEG_EXTENSIONS,
            check=True,
        )
        return sorted(paths, key=lambda path: str(path.fpath))

    @property
    def recordings(self) -> tuple[BIDSPath, ...]:
        """All raw EEG recordings in stable BIDS-path order."""
        return tuple(self._matching_paths())

    @property
    def subjects(self) -> tuple[str, ...]:
        return tuple(sorted({str(path.subject) for path in self.recordings}))

    @property
    def tasks(self) -> tuple[str, ...]:
        return tuple(sorted({str(path.task) for path in self.recordings}))

    @property
    def participants(self) -> pd.DataFrame:
        path = self.root / "participants.tsv"
        if not path.is_file():
            raise FileNotFoundError(f"Participants table not found: {path}")
        return pd.read_csv(path, sep="\t", keep_default_na=False)

    @property
    def wordlist(self) -> pd.DataFrame:
        path = self.root / "stimuli" / "wordlist.tsv"
        if not path.is_file():
            raise FileNotFoundError(f"Word list not found: {path}")
        return pd.read_csv(
            path,
            sep="\t",
            header=None,
            names=("id", "sentence", "sentence_syll"),
            keep_default_na=False,
        )

    def _events_path(self, bids_path: BIDSPath) -> Path:
        return Path(
            bids_path.copy().update(suffix="events", extension=".tsv").fpath
        )

    def _source_paths(self, bids_path: BIDSPath) -> tuple[Path | None, Path | None]:
        session = f"ses-{bids_path.session}" if bids_path.session else None
        source_dir = self.root / "sourcedata" / f"sub-{bids_path.subject}"
        if session is not None:
            source_dir /= session
        source_dir /= "audio"
        run_id = bids_path.basename.rsplit("_", maxsplit=1)[0]
        audio = source_dir / f"{run_id}_audio.wav"
        textgrid = source_dir / f"{run_id}_annotations.TextGrid"
        return (
            audio if audio.is_file() else None,
            textgrid if textgrid.is_file() else None,
        )

    def _load(self, bids_path: BIDSPath, *, preload: bool) -> RunData:
        events_path = self._events_path(bids_path)
        if not events_path.is_file():
            raise FileNotFoundError(f"Events table not found: {events_path}")
        events = pd.read_csv(events_path, sep="\t", keep_default_na=False)
        missing = [column for column in REQUIRED_EVENT_COLUMNS if column not in events]
        if missing:
            raise ValueError(
                f"{events_path} is missing required columns: {', '.join(missing)}"
            )
        raw = read_raw_bids(
            bids_path,
            extra_params={"preload": preload},
            verbose="error",
        )
        audio_path, textgrid_path = self._source_paths(bids_path)
        return RunData(
            bids_path=bids_path,
            raw=raw,
            events=events,
            events_path=events_path,
            audio_path=audio_path,
            textgrid_path=textgrid_path,
        )

    def iter_runs(
        self,
        *,
        subjects: str | int | Iterable[str | int] | None = None,
        sessions: str | int | Iterable[str | int] | None = None,
        tasks: str | Iterable[str] | None = None,
        runs: str | int | Iterable[str | int] | None = None,
        preload: bool = False,
    ) -> Iterator[RunData]:
        """Yield matching recordings one at a time."""
        paths = self._matching_paths(
            subjects=subjects,
            sessions=sessions,
            tasks=tasks,
            runs=runs,
        )
        if not paths:
            raise FileNotFoundError("No EEG recordings match the requested BIDS entities.")
        for path in paths:
            yield self._load(path, preload=preload)

    def get_run(
        self,
        *,
        subject: str | int,
        task: str,
        run: str | int,
        session: str | int = "01",
        preload: bool = False,
    ) -> RunData:
        """Load exactly one recording selected by its BIDS entities."""
        paths = self._matching_paths(
            subjects=subject,
            sessions=session,
            tasks=task,
            runs=run,
        )
        if not paths:
            raise FileNotFoundError(
                "No EEG recording found for "
                f"sub-{_label(subject, 'sub')}, ses-{_label(session, 'ses')}, "
                f"task-{task}, run-{_label(run, 'run')}."
            )
        if len(paths) != 1:
            matches = "\n".join(f"- {path.fpath}" for path in paths)
            raise RuntimeError(f"Expected one EEG recording, found:\n{matches}")
        return self._load(paths[0], preload=preload)
