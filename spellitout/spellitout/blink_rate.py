"""Reviewed-annotation spontaneous blink-rate summaries."""

from __future__ import annotations

import json
import re
from pathlib import Path

import mne
import numpy as np
import pandas as pd

from .review_qc import ReviewRecord, read_review_tables


PHASES = ("run", "sentence_listening", "sentence_repetition", "syllabification")
_LABEL_RE = re.compile(r"(?P<key>sub|ses|task|run)-(?P<value>[^_]+)")


def _record_entities(record: ReviewRecord) -> dict[str, str]:
    entities = {}
    for match in _LABEL_RE.finditer(record.label):
        key, value = match.group("key"), match.group("value")
        entities[key] = f"{key}-{value}" if key in {"sub", "ses"} else value
    return entities


def _event_intervals(raw: mne.io.BaseRaw, description: str) -> list[tuple[float, float]]:
    intervals = []
    for onset, duration, label in zip(
        raw.annotations.onset,
        raw.annotations.duration,
        raw.annotations.description,
    ):
        if str(label) == description:
            start = float(onset)
            end = start + max(0.0, float(duration))
            intervals.append((start, end))
    return intervals


def _syllabification_intervals(
    raw: mne.io.BaseRaw,
    merge_gap_seconds: float = 1.0,
) -> tuple[list[tuple[float, float]], int]:
    """Merge each consecutive syllable-cue block into one phase interval."""
    events = sorted(
        (float(onset), float(onset + duration))
        for onset, duration, label in zip(
            raw.annotations.onset,
            raw.annotations.duration,
            raw.annotations.description,
        )
        if str(label) == "syllabification_cue"
    )
    if not events:
        return [], 0
    blocks: list[tuple[float, float]] = []
    start, end = events[0]
    for cue_start, cue_end in events[1:]:
        if cue_start - end <= merge_gap_seconds:
            end = max(end, cue_end)
        else:
            blocks.append((start, end))
            start, end = cue_start, cue_end
    blocks.append((start, end))
    return blocks, len(events)


def phase_intervals(raw: mne.io.BaseRaw) -> tuple[dict[str, list[tuple[float, float]]], int]:
    syllabification, n_syllabification_cues = _syllabification_intervals(raw)
    return {
        "run": [(0.0, float(raw.times[-1]))],
        "sentence_listening": _event_intervals(raw, "listening_cue"),
        "sentence_repetition": _event_intervals(raw, "repetition_cue"),
        "syllabification": syllabification,
    }, n_syllabification_cues


def subtract_intervals(
    intervals: list[tuple[float, float]],
    excluded: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    """Subtract possibly overlapping excluded spans from phase intervals."""
    result = intervals[:]
    for excluded_start, excluded_end in excluded:
        next_result: list[tuple[float, float]] = []
        for start, end in result:
            if excluded_end <= start or excluded_start >= end:
                next_result.append((start, end))
                continue
            if start < excluded_start:
                next_result.append((start, min(end, excluded_start)))
            if excluded_end < end:
                next_result.append((max(start, excluded_end), end))
        result = [(start, end) for start, end in next_result if end > start]
    return result


def _reviewed_blinks(annotations: pd.DataFrame) -> tuple[np.ndarray, int, int]:
    descriptions = annotations["reviewed_description"].astype(str)
    trial_types = annotations["trial_type"].astype(str)
    # Unreviewed automatic candidates retain BLINK only in trial_type; once
    # reviewed, the final description is copied to reviewed_description.
    blink_rows = annotations.loc[descriptions.eq("BLINK") | trial_types.eq("BLINK")]
    counted = blink_rows.loc[blink_rows["review_status"].ne("dismissed")]
    centres = (
        counted["onset"].to_numpy(dtype=float)
        + counted["duration"].to_numpy(dtype=float) / 2.0
    )
    n_unreviewed = int((blink_rows["review_status"] == "unreviewed").sum())
    n_dismissed = int((blink_rows["review_status"] == "dismissed").sum())
    return centres, n_unreviewed, n_dismissed


def enforce_minimum_blink_separation(
    centres: np.ndarray,
    minimum_separation_seconds: float,
) -> tuple[np.ndarray, int]:
    """Suppress later events whose centres are closer than the threshold."""
    if minimum_separation_seconds < 0:
        raise ValueError("minimum_separation_seconds must be non-negative")
    centres = np.sort(np.asarray(centres, dtype=float))
    if len(centres) < 2 or minimum_separation_seconds == 0:
        return centres, 0
    kept = [float(centres[0])]
    for centre in centres[1:]:
        if centre - kept[-1] + 1e-12 >= minimum_separation_seconds:
            kept.append(float(centre))
    return np.asarray(kept, dtype=float), len(centres) - len(kept)


def run_blink_rate_rows(
    record: ReviewRecord,
    *,
    exclude_bad_spans: bool = False,
    minimum_blink_separation_seconds: float = 0.150,
) -> list[dict[str, object]]:
    """Compute four phase rows for one reviewed run."""
    entities = _record_entities(record)
    raw = mne.io.read_raw_fif(record.raw_path, preload=False, verbose="error")
    annotations, _ = read_review_tables(record)
    intervals_by_phase, n_syllabification_cues = phase_intervals(raw)
    blink_centres, n_unreviewed, n_dismissed = _reviewed_blinks(annotations)
    blink_centres, n_suppressed = enforce_minimum_blink_separation(
        blink_centres,
        minimum_blink_separation_seconds,
    )
    bad_spans = []
    if exclude_bad_spans:
        bad_rows = annotations.loc[
            annotations["review_status"].eq("approved_bad")
            & annotations["reviewed_description"].astype(str).str.startswith("BAD_")
        ]
        bad_spans = [
            (float(row.onset), float(row.onset + row.duration))
            for row in bad_rows.itertuples()
        ]

    rows = []
    for phase in PHASES:
        intervals = intervals_by_phase[phase]
        valid_intervals = subtract_intervals(intervals, bad_spans) if exclude_bad_spans else intervals
        duration = sum(end - start for start, end in intervals)
        valid_duration = sum(end - start for start, end in valid_intervals)
        blink_count = sum(
            1
            for centre in blink_centres
            if any(start <= centre < end for start, end in valid_intervals)
        )
        rows.append(
            {
                "subject": entities.get("sub", ""),
                "session": entities.get("ses", ""),
                "task": entities.get("task", ""),
                "run": entities.get("run", ""),
                "phase": phase,
                "phase_duration_s": duration,
                "valid_duration_s": valid_duration,
                "blink_count": blink_count,
                "blink_rate_per_min": (
                    blink_count / valid_duration * 60.0 if valid_duration > 0 else np.nan
                ),
                "n_intervals": len(intervals),
                "n_syllabification_cues": (
                    n_syllabification_cues if phase == "syllabification" else 0
                ),
                "review_complete": n_unreviewed == 0,
                "n_unreviewed_blinks": n_unreviewed,
                "n_dismissed_blinks": n_dismissed,
                "n_blinks_suppressed_min_separation": n_suppressed,
                "minimum_blink_separation_s": minimum_blink_separation_seconds,
                "bad_spans_excluded": exclude_bad_spans,
            }
        )
    return rows


BLINKRATE_COLUMNS = (
    "subject", "session", "task", "run", "phase", "phase_duration_s",
    "valid_duration_s", "blink_count", "blink_rate_per_min", "n_intervals",
    "n_syllabification_cues", "review_complete", "n_unreviewed_blinks",
    "n_dismissed_blinks", "bad_spans_excluded",
    "n_blinks_suppressed_min_separation", "minimum_blink_separation_s",
)


def write_blinkrate_sidecar(
    output_root: Path,
    filename: str = "desc-blinkrate.json",
) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / filename
    metadata = {
        "Description": "Reviewed spontaneous eye-blink counts and rates by run and task phase.",
        "blink_count": "Number of reviewed BLINK annotation centres within the phase.",
        "blink_rate_per_min": "blink_count divided by valid_duration_s, multiplied by 60.",
        "n_syllabification_cues": (
            "Number of syllabification_cue events contributing to the "
            "syllabification phase; zero for all other phases."
        ),
        "review_complete": "False when one or more BLINK annotations in the run remain unreviewed.",
        "bad_spans_excluded": "Whether approved BAD_* spans were removed from duration and blink counting.",
        "PhaseDefinitions": {
            "run": "Complete run interval.",
            "sentence_listening": "Each listening_cue interval, aggregated per run.",
            "sentence_repetition": "Each repetition_cue interval, aggregated per run.",
            "syllabification": "Each consecutive syllabification_cue block, aggregated per run.",
        },
        "CountRule": "BLINK annotations with review_status keep or unreviewed are counted; dismissed annotations are excluded.",
        "MinimumBlinkSeparation": "Events with centres closer than the configured minimum separation are counted once; source annotations are not modified.",
    }
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return path
