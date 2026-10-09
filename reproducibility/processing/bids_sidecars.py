"""Resolve and relocate EEG/channel sidecars without touching signal files."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

from mne_bids import get_bids_path_from_fname
import pandas as pd


EEG_EXTENSIONS = {".vhdr", ".edf", ".bdf", ".set"}


def _entities(path: Path) -> dict[str, str]:
    return dict(part.split("-", 1) for part in path.stem.split("_")[:-1] if "-" in part)


def matching_sidecars(root: Path, recording: Path, suffix: str, extension: str) -> list[Path]:
    """Return applicable metadata in increasing inheritance specificity."""
    root, recording = root.resolve(), recording.resolve()
    recording.relative_to(root)
    entities = _entities(recording)
    directories = [recording.parent, *recording.parent.parents]
    directories = list(reversed(directories[:directories.index(root) + 1]))
    result = []
    for directory in directories:
        applicable = []
        for path in directory.glob(f"*{suffix}{extension}"):
            if path.stem.split("_")[-1] != suffix:
                continue
            candidate = _entities(path)
            if all(entities.get(key) == value for key, value in candidate.items()):
                applicable.append(path)
        # Tied specificity is ambiguous, rather than resolved by filesystem order.
        sizes = [len(_entities(path)) for path in applicable]
        if len(sizes) != len(set(sizes)):
            raise ValueError(f"Ambiguous {suffix} metadata for {recording}: {applicable}")
        result.extend(sorted(applicable, key=lambda path: len(_entities(path))))
    return result


def read_eeg_metadata(root: Path, recording: Path) -> dict:
    paths = matching_sidecars(root, recording, "eeg", ".json")
    if not paths:
        raise FileNotFoundError(f"EEG JSON metadata not found for {recording}")
    metadata = {}
    for path in paths:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(value, dict):
            raise ValueError(f"Expected a JSON object: {path}")
        metadata.update(value)
    required = {"TaskName", "EEGReference", "SamplingFrequency", "PowerLineFrequency", "SoftwareFilters"}
    missing = sorted(required.difference(metadata))
    if missing:
        raise ValueError(f"Incomplete EEG metadata for {recording}: {', '.join(missing)}")
    return metadata


def read_channels_metadata(root: Path, recording: Path) -> pd.DataFrame:
    paths = matching_sidecars(root, recording, "channels", ".tsv")
    if not paths:
        raise FileNotFoundError(f"Channels TSV not found for {recording}")
    table = pd.read_csv(paths[-1], sep="\t", dtype=str, keep_default_na=False)
    if not {"name", "type", "units"}.issubset(table.columns):
        raise ValueError(f"Required channel columns missing in {paths[-1]}")
    if table["name"].duplicated().any():
        raise ValueError(f"Duplicate channel names in {paths[-1]}")
    return table


def _json_text(value: dict) -> str:
    return json.dumps(value, indent=2, ensure_ascii=False) + "\n"


@dataclass
class SidecarPlan:
    root: Path
    recordings: int
    groups: int
    writes: dict[Path, str]
    removals: set[Path]
    snapshots: dict[Path, bytes | None]

    def check_unchanged(self) -> None:
        for path, original in self.snapshots.items():
            current = path.read_bytes() if path.exists() else None
            if current != original:
                raise RuntimeError(f"Sidecar changed after validation: {path}")


def plan_sidecar_migration(root: Path) -> SidecarPlan:
    """Plan a lossless, idempotent migration of one raw or derivative dataset.

    Identical metadata are shared within subject/session/task/description.
    Otherwise every run receives a complete sidecar: MNE-BIDS does not merge
    a partial run JSON with less-specific JSON files.
    """
    root = root.resolve()
    if not (root / "dataset_description.json").is_file():
        raise FileNotFoundError(f"Dataset description not found below {root}")
    recordings = sorted(path for path in root.glob("sub-*/ses-*/eeg/*")
                        if path.suffix.lower() in EEG_EXTENSIONS)
    if not recordings:
        raise FileNotFoundError(f"No published EEG recordings below {root}")
    grouped = {}
    sources = set()
    for recording in recordings:
        bids_path = get_bids_path_from_fname(recording, check=False)
        if bids_path.acquisition or bids_path.recording or bids_path.processing:
            raise ValueError(f"Additional recording entities require explicit handling: {recording}")
        base = f"sub-{bids_path.subject}_ses-{bids_path.session}_task-{bids_path.task}"
        if bids_path.description:
            base += f"_desc-{bids_path.description}"
        for suffix, extension in (("eeg", ".json"), ("channels", ".tsv")):
            sources.update(matching_sidecars(root, recording, suffix, extension))
        metadata = read_eeg_metadata(root, recording)
        table = read_channels_metadata(root, recording)
        grouped.setdefault((recording.parent, base), []).append((recording, metadata, table))
    targets = {}
    for (directory, base), group in grouped.items():
        for suffix in ("eeg", "channels"):
            values = [entry[1] if suffix == "eeg" else entry[2] for entry in group]
            same = all(value == values[0] if suffix == "eeg" else value.equals(values[0])
                       for value in values[1:])
            selected = [(directory / base, values[0])] if same else [
                (recording.with_suffix(""), value)
                for (recording, _, _), value in zip(group, values)
            ]
            for stem, value in selected:
                # Recording stems include the EEG suffix; shared stems do not.
                name = stem.name.removesuffix("_eeg") if not same else stem.name
                extension = ".json" if suffix == "eeg" else ".tsv"
                target = directory / f"{name}_{suffix}{extension}"
                text = _json_text(value) if suffix == "eeg" else value.to_csv(
                    sep="\t", index=False, lineterminator="\n")
                targets[target] = text
    removals = sources.difference(targets)
    watched = sources | set(targets)
    snapshots = {path: path.read_bytes() if path.exists() else None for path in watched}
    writes = {path: text for path, text in targets.items()
              if snapshots[path] != text.encode("utf-8")}
    return SidecarPlan(root, len(recordings), len(grouped), writes, removals, snapshots)


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".sidecar-", delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(content)
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def apply_sidecar_plan(plan: SidecarPlan, *, backup_root: Path | None = None) -> Path | None:
    """Apply a preflighted plan; signal/event/coordinate files are never touched."""
    plan.check_unchanged()
    touched = set(plan.writes) | plan.removals
    for path in touched:
        path.resolve().relative_to(plan.root)
    backup = None
    journal = None
    if backup_root is not None and touched:
        backup_root = backup_root.resolve()
        if backup_root == plan.root or plan.root in backup_root.parents:
            raise ValueError("Backups must be outside the migrated dataset.")
        backup = backup_root / (plan.root.name + "_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
        backup.mkdir(parents=True, exist_ok=False)
        for path in sorted(touched):
            if plan.snapshots[path] is not None:
                destination = backup / "originals" / path.relative_to(plan.root)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, destination)
        journal = {
            "dataset_root": str(plan.root), "state": "prepared",
            "created": [str(path.relative_to(plan.root)) for path in sorted(plan.writes)
                        if plan.snapshots[path] is None],
            "writes": [str(path.relative_to(plan.root)) for path in sorted(plan.writes)],
            "removals": [str(path.relative_to(plan.root)) for path in sorted(plan.removals)],
            "original_sha256": {str(path.relative_to(plan.root)): hashlib.sha256(content).hexdigest()
                                for path, content in plan.snapshots.items() if content is not None},
        }
        _atomic_write(backup / "journal.json", _json_text(journal).encode("utf-8"))
    try:
        for path, text in sorted(plan.writes.items()):
            _atomic_write(path, text.encode("utf-8"))
        for path in sorted(plan.removals):
            path.unlink()
    except Exception:
        # Roll back this dataset's sidecars, including newly created targets.
        for path in sorted(touched):
            original = plan.snapshots[path]
            if original is None:
                path.unlink(missing_ok=True)
            else:
                _atomic_write(path, original)
        if journal is not None:
            journal["state"] = "rolled_back"
            _atomic_write(backup / "journal.json", _json_text(journal).encode("utf-8"))
        raise
    if journal is not None:
        journal["state"] = "complete"
        _atomic_write(backup / "journal.json", _json_text(journal).encode("utf-8"))
    return backup
