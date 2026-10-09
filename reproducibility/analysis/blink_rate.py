"""Reviewed-annotation spontaneous blink-rate summaries."""

from __future__ import annotations

import json
from pathlib import Path

import mne
import numpy as np
import pandas as pd

from spellitout import RunData


from reproducibility.processing.intervals import PHASES, phase_intervals, subtract_intervals


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
    record: RunData,
    *,
    annotations_qc: pd.DataFrame | None = None,
    exclude_bad_spans: bool = False,
    minimum_blink_separation_seconds: float = 0.150,
) -> list[dict[str, object]]:
    """Compute phase rates from published events; QC supplies review diagnostics."""
    raw = record.raw
    intervals_by_phase, n_syllabification_cues = phase_intervals(raw)
    events = record.events
    labels = events["trial_type"].astype(str).str.upper()
    blink_rows = events.loc[labels.isin(("BLINK", "MANUAL_BLINK"))]
    blink_centres = (
        blink_rows["onset"].to_numpy(dtype=float)
        + blink_rows["duration"].to_numpy(dtype=float) / 2.0
    )
    # Published events contain accepted annotations only. Retain audit counts
    # when the publication includes its complete QC table.
    n_unreviewed = 0
    n_dismissed = np.nan
    if annotations_qc is not None:
        _, n_unreviewed, n_dismissed = _reviewed_blinks(annotations_qc)
    blink_centres, n_suppressed = enforce_minimum_blink_separation(
        blink_centres,
        minimum_blink_separation_seconds,
    )
    bad_spans = []
    if exclude_bad_spans:
        bad_rows = events.loc[labels.str.startswith("BAD_")]
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
                "subject": record.subject_id,
                "session": f"ses-{record.session}" if record.session else "",
                "task": record.task,
                "run": record.run or "",
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
        "blink_count": "Number of published BLINK or MANUAL_BLINK event centres within the phase.",
        "blink_rate_per_min": "blink_count divided by valid_duration_s, multiplied by 60.",
        "n_syllabification_cues": (
            "Number of syllabification_cue events contributing to the "
            "syllabification phase; zero for all other phases."
        ),
        "review_complete": "False when the published QC table reports unreviewed blink candidates; otherwise published events are treated as reviewed.",
        "n_dismissed_blinks": "Dismissed blink candidates in the published QC table; unavailable when that table is absent.",
        "bad_spans_excluded": "Whether approved BAD_* spans were removed from duration and blink counting.",
        "PhaseDefinitions": {
            "run": "Complete run interval.",
            "sentence_listening": "Each listening_cue interval, aggregated per run.",
            "sentence_repetition": "Each repetition_cue interval, aggregated per run.",
            "syllabification": "Each consecutive syllabification_cue block, aggregated per run.",
        },
        "CountRule": "Count BLINK and MANUAL_BLINK from the final desc-preproc_events.tsv; QC candidates are never added to the published events.",
        "MinimumBlinkSeparation": "Events with centres closer than the configured minimum separation are counted once; source annotations are not modified.",
    }
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return path
