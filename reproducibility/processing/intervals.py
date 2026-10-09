"""Shared event-phase intervals for review and paper analyses."""

from __future__ import annotations

import mne


PHASES = ("run", "sentence_listening", "sentence_repetition", "syllabification")


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
        "run": [(0.0, raw.n_times / float(raw.info["sfreq"]))],
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
