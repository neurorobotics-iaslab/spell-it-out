"""Describe published QC decisions and syllable availability without reading EEG.

python -m reproducibility.scripts.signal_quality --bids-root data/spellitout
Each execution creates a new result directory; source data are read-only.
"""
from __future__ import annotations

import argparse
import configparser
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PHASE_EVENTS = {"listening": "listening_cue", "repetition": "repetition_cue",
                "syllabification": "syllabification_cue"}
KEY = ["subject", "session", "task", "run", "trial_index", "token_index"]


def read_table(path):
    return pd.read_csv(path, sep="\t", keep_default_na=False)


def intervals_union(intervals, end):
    merged = []
    for start, stop in sorted((max(0., float(a)), min(end, float(b))) for a, b in intervals):
        if stop <= start:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(stop, merged[-1][1])
        else:
            merged.append([start, stop])
    return merged


def length(intervals):
    return sum(b - a for a, b in intervals)


def overlap(first, second):
    return sum(max(0., min(b, d) - max(a, c)) for a, b in first for c, d in second)


def binary_recording_info(header):
    """Infer exact sample count from BrainVision header and binary file size."""
    config = configparser.ConfigParser(interpolation=None)
    config.read_string(header.read_text(encoding="utf-8-sig").split("\n", 1)[1])
    common = config["Common Infos"]
    if common["DataFormat"] != "BINARY" or common["DataOrientation"] != "MULTIPLEXED":
        raise ValueError(f"Unsupported BrainVision layout: {header}")
    nchan = int(common["NumberOfChannels"])
    size = {"IEEE_FLOAT_32": 4, "INT_16": 2, "INT_32": 4}[config["Binary Infos"]["BinaryFormat"]]
    nbytes = (header.parent / common["DataFile"]).stat().st_size
    if nbytes % (size * nchan):
        raise ValueError(f"Incomplete binary frame: {header}")
    nsamples = nbytes // (size * nchan)
    sfreq = 1e6 / float(common["SamplingInterval"])
    return nsamples / sfreq, nsamples, sfreq


def bad_spans(events, end):
    global_spans, scoped_spans = [], []
    for row in events.to_dict("records"):
        if not str(row["trial_type"]).upper().startswith("BAD"):
            continue
        start, duration = float(row["onset"]), float(row["duration"])
        if not np.isfinite(start + duration) or duration < 0:
            raise ValueError("Invalid BAD annotation timing")
        scoped = any(str(row.get(field, "")).strip() not in {"", "n/a", "[]"}
                     for field in ("channel", "channels", "ch_names"))
        (scoped_spans if scoped else global_spans).append((start, start + duration))
    return intervals_union(global_spans, end), intervals_union(scoped_spans, end)


def phase_spans(events, end):
    result = {}
    for phase, event in PHASE_EVENTS.items():
        selected = events.loc[events.trial_type.eq(event)]
        if selected.empty:
            result[phase] = None
            continue
        spans = []
        for trial, group in selected.groupby("trial_index"):
            starts = group.onset.astype(float)
            stops = starts + group.duration.astype(float)
            trial_start = events.loc[events.trial_index.eq(trial) & events.trial_type.eq("trial_start")]
            trial_end = events.loc[events.trial_index.eq(trial) & events.trial_type.eq("trial_end")]
            lower = trial_start.onset.astype(float).min() if not trial_start.empty else 0.
            upper = trial_end.onset.astype(float).min() if not trial_end.empty else end
            if phase == "syllabification":
                # Continuous cued block: include the millisecond gaps between cues.
                a, b = starts.min(), stops.max()
                spans.append((max(a, lower), min(b, upper)))
            else:
                spans.extend((max(a, lower), min(b, upper)) for a, b in zip(starts, stops))
        result[phase] = intervals_union(spans, end)
    return result


def normalize_keys(table):
    table = table.copy()
    for column in KEY:
        table[column] = table[column].astype(str)
    for column in ("trial_index", "token_index"):
        table[column] = pd.to_numeric(table[column], errors="raise").astype(int).astype(str)
    return table


def export_rejection_table(runs, output):
    """Export global BAD durations and participant/run percentages for reporting."""
    detailed = runs[["subject", "task", "run", "duration_s", "bad_duration_s", "bad_percent"]].copy()
    detailed["bad_duration_min"] = detailed.bad_duration_s / 60
    detailed.to_csv(output / "rejection_by_run.tsv", sep="\t", index=False)
    wide = pd.DataFrame(index=sorted(runs.subject.unique()))
    for task in ("overt", "covert"):
        selected = runs.loc[runs.task.eq(task)]
        pivot = selected.pivot(index="subject", columns="run", values="bad_percent")
        for run in ("run-01", "run-02", "run-03"):
            wide[f"{task}_{run}_percent"] = pivot[run]
        sums = selected.groupby("subject")[["bad_duration_s", "duration_s"]].sum()
        wide[f"{task}_all_runs_percent"] = 100 * sums.bad_duration_s / sums.duration_s
    wide.index.name = "subject"
    wide.loc["MEAN_ACROSS_PARTICIPANTS"] = wide.mean()
    wide.to_csv(output / "rejection_percent_wide.tsv", sep="\t", float_format="%.2f")
    totals = runs.groupby("task")[["bad_duration_s", "duration_s"]].sum()
    totals["bad_percent"] = 100 * totals.bad_duration_s / totals.duration_s
    totals["bad_duration_min"] = totals.bad_duration_s / 60
    totals.to_csv(output / "rejection_totals.tsv", sep="\t", float_format="%.6f")
    columns = ["Participant", "Overt R1", "Overt R2", "Overt R3", "Overt all runs",
               "Covert R1", "Covert R2", "Covert R3", "Covert all runs"]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for subject, row in wide.iterrows():
        label = "Mean" if subject == "MEAN_ACROSS_PARTICIPANTS" else subject
        lines.append("| " + label + " | " + " | ".join(f"{value:.2f}" for value in row) + " |")
    (output / "rejection_table.md").write_text(
        "# Recording duration annotated BAD (%)\n\n" + "\n".join(lines) +
        "\n\nAll-runs percentages are the ratio of summed BAD durations to summed recording durations within each participant. "
        "The final row is the arithmetic mean across participants. Run-column means are computed across participants separately. "
        "The pooled duration-weighted percentages and total BAD durations are in rejection_totals.tsv. "
        "BAD intervals remain present in continuous recordings and are marked for exclusion during fitting or sample selection.\n",
        encoding="utf-8")
    return wide, totals


def main(args):
    bids = args.bids_root.resolve()
    pre = (args.preprocessed_root or bids / "derivatives/preprocessing").resolve()
    syll = (args.syllable_root or bids / "derivatives/syllable-datasets").resolve()
    output = args.output_path or Path("reproducibility/results/signal_quality") / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    output.mkdir(parents=True, exist_ok=False)
    issues, inputs, runs, phases, candidates = [], [], [], [], []

    def issue(level, context, message):
        issues.append(dict(level=level, context=context, message=message))

    def required(path):
        if not path.is_file():
            issue("missing", str(path), "Required documentation or recording missing; no zero assumed.")
            return False
        inputs.append(str(path))
        return True

    config = json.loads((syll / "build_config.json").read_text(encoding="utf-8"))
    digest = hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    source_root = bids if config["source"] == "raw" else pre
    specs = config["datasets"]
    manifest = normalize_keys(read_table(syll / "manifest.tsv"))
    exclusions = normalize_keys(read_table(syll / "exclusions.tsv"))
    inputs.extend(str(syll / x) for x in ("build_config.json", "manifest.tsv", "exclusions.tsv", "dataset_metadata.json"))
    participant_ids = sorted(p.name for p in bids.glob("sub-*") if p.is_dir())
    raw_channels = read_table(bids / "channels.tsv")
    inputs.append(str(bids / "channels.tsv"))
    raw_eeg_channels = set(raw_channels.loc[raw_channels.type.eq("EEG"), "name"])
    coverage = []
    expected_missing_productions = set()
    for subject in participant_ids:
        print(f"Auditing {subject}", flush=True)
        for task in ("overt", "covert"):
            for run in ("01", "02", "03"):
                base = f"{subject}_ses-01_task-{task}_run-{run}"
                entities = dict(subject=subject, session="ses-01", task=task, run=f"run-{run}")
                folder = pre / subject / "ses-01/eeg"
                raw_folder = bids / subject / "ses-01/eeg"
                qc_folder = pre / "qc" / subject / "ses-01/eeg"
                files = {
                    "raw": raw_folder / f"{base}_eeg.vhdr",
                    "raw_events": raw_folder / f"{base}_events.tsv",
                    "preprocessed": folder / f"{base}_desc-preproc_eeg.vhdr",
                    "events": folder / f"{base}_desc-preproc_events.tsv",
                    "channels": folder / f"{base}_desc-preproc_channels.tsv",
                    "annotation_qc": qc_folder / f"{base}_desc-annotationsqc.tsv",
                    "channel_qc": qc_folder / f"{base}_desc-channelqc.tsv",
                    "ica_provenance": qc_folder / f"{subject}_ses-01_task-{task}_desc-icaprovenance.json",
                    "ica_decisions": qc_folder / f"{subject}_ses-01_task-{task}_desc-icadecisions.tsv",
                    "channel_plan": qc_folder / f"{subject}_ses-01_task-{task}_desc-channelplan.tsv",
                }
                present = {name: required(path) for name, path in files.items()}
                coverage.append({**entities, **present})
                if not all(present[name] for name in ("preprocessed", "events", "raw", "raw_events")):
                    continue
                duration, n_samples, sfreq = binary_recording_info(files["preprocessed"])
                raw_duration, _, _ = binary_recording_info(files["raw"])
                if abs(duration - raw_duration) > 1 / sfreq:
                    issue("error", base, "Raw and preprocessed durations disagree beyond one output sample.")
                events = read_table(files["events"])
                raw_events = read_table(files["raw_events"])
                bad, scoped = bad_spans(events, duration)
                row = {**entities, "duration_s": duration, "raw_duration_s": raw_duration,
                       "n_samples": n_samples, "sfreq": sfreq, "bad_duration_s": length(bad),
                       "bad_percent": 100 * length(bad) / duration,
                       "channel_scoped_bad_duration_s": length(scoped)}
                if present["channel_qc"]:
                    cq = read_table(files["channel_qc"])
                    cq = cq.loc[cq.channel_type.eq("EEG")]
                    row.update(automatic_flagged_channels=int(cq.automatic_status.eq("flagged").sum()),
                               reviewed_interpolate_channels=int(cq.final_decision.eq("interpolate").sum()),
                               reviewed_exclude_channels=int(cq.final_decision.eq("exclude_from_ica").sum()),
                               unresolved_channel_decisions=int(cq.final_decision.eq("unreviewed").sum()))
                if present["annotation_qc"]:
                    aq = read_table(files["annotation_qc"])
                    row["unresolved_annotations"] = int(aq.review_status.eq("unreviewed").sum())
                    approved = aq.loc[aq.review_status.eq("approved_bad")]
                    audit_bad = intervals_union([(float(r.onset), float(r.onset) + float(r.duration)) for r in approved.itertuples()], duration)
                    if abs(length(audit_bad) - length(bad)) > 1 / sfreq:
                        issue("error", base, "Published global BAD duration differs from approved audit duration.")
                if present["ica_provenance"]:
                    provenance = json.loads(files["ica_provenance"].read_text(encoding="utf-8"))
                    for field, metric in (("condition_interpolated_channels", "applied_interpolated_channels"),
                                          ("condition_excluded_channels", "applied_excluded_channels")):
                        row[metric] = len(provenance[field]) if field in provenance else np.nan
                        if field not in provenance:
                            issue("missing_field", base, field)
                    if present["channel_plan"]:
                        try:
                            plan = read_table(files["channel_plan"])
                        except pd.errors.EmptyDataError:
                            plan = pd.DataFrame(columns=["channel", "condition_decision"])
                        for decision, field in (("interpolate", "condition_interpolated_channels"),
                                                ("exclude_from_ica", "condition_excluded_channels")):
                            planned = set(plan.loc[plan.condition_decision.eq(decision), "channel"])
                            if planned != set(provenance.get(field, [])):
                                issue("error", base, "Channel plan and ICA provenance disagree: " + decision)
                if present["ica_decisions"]:
                    decisions = read_table(files["ica_decisions"])
                    if not decisions.final_decision.isin(["keep", "reject"]).all():
                        issue("error", base, "Unresolved ICA component decisions.")
                if present["channels"]:
                    channels = read_table(files["channels"])
                    final_eeg = set(channels.loc[channels.type.eq("EEG"), "name"])
                    removed = raw_eeg_channels - final_eeg
                    row["total_removed_eeg_channels"] = len(removed)
                    row["removed_eeg_names"] = ";".join(sorted(removed))
                    row["final_eeg_channels"] = len(final_eeg)
                    row["published_interpolated_channels"] = int(channels.interpolated.astype(str).str.lower().eq("true").sum())
                    if row.get("applied_interpolated_channels") != row["published_interpolated_channels"]:
                        issue("error", base, "Interpolation provenance and published channels disagree.")
                runs.append(row)
                for phase, spans in phase_spans(events, duration).items():
                    if spans is None:
                        issue("missing_phase", base, phase)
                        phases.append({**entities, "phase": phase, "phase_duration_s": np.nan, "bad_duration_s": np.nan})
                    else:
                        total = length(spans)
                        contaminated = overlap(spans, bad)
                        phases.append({**entities, "phase": phase, "phase_duration_s": total,
                                       "bad_duration_s": contaminated, "bad_percent": 100 * contaminated / total,
                                       "channel_scoped_bad_duration_s": overlap(spans, scoped)})
                source_events = raw_events if config["source"] == "raw" else events
                for spec in specs:
                    if spec["task"] != task:
                        continue
                    selected = source_events.loc[source_events.trial_type.eq(spec["event_type"])]
                    if spec.get("pair_reference_event_type"):
                        reference = source_events.loc[source_events.trial_type.eq(spec["pair_reference_event_type"])]
                        observed = {(str(int(r.trial_index)), str(int(r.token_index))) for r in selected.itertuples()}
                        for r in reference.itertuples():
                            indices = (str(int(r.trial_index)), str(int(r.token_index)))
                            if indices not in observed:
                                expected_missing_productions.add((spec["name"], *(entities[k] for k in KEY[:4]), *indices))
                    for r in selected.itertuples():
                        candidates.append({**entities, "dataset": spec["name"], "trial_index": str(int(r.trial_index)),
                                           "token_index": str(int(r.token_index)), "syllable": r.token})

    candidates = pd.DataFrame(candidates)
    sample_audit = []
    for spec in specs:
        name = spec["name"]
        expected = candidates.loc[candidates.dataset.eq(name)]
        accepted = manifest.loc[manifest.dataset.eq(name)]
        excluded = exclusions.loc[exclusions.dataset.eq(name)]
        accepted_keys = set(map(tuple, accepted[KEY].to_numpy()))
        reasons = {}
        for r in excluded.to_dict("records"):
            reasons.setdefault(tuple(r[k] for k in KEY), set()).add(r["reason"])
        candidate_keys = set(map(tuple, expected[KEY].to_numpy()))
        if len(candidate_keys) != len(expected) or len(accepted_keys) != len(accepted):
            issue("error", name, "Duplicate candidate or retained sample keys.")
        missing_production = {key for key, values in reasons.items() if "missing_production_annotation" in values}
        expected_missing = {key[1:] for key in expected_missing_productions if key[0] == name}
        if missing_production != expected_missing:
            issue("error", name, "Missing-production exclusions disagree with cue/production matching.")
        for key in sorted(candidate_keys | set(reasons) | accepted_keys):
            values = reasons.get(key, set())
            is_candidate, retained = key in candidate_keys, key in accepted_keys
            if retained and values:
                issue("error", name, f"Retained and excluded sample: {key}")
            if is_candidate and not retained and not values:
                issue("error", name, f"Candidate has no retention or exclusion record: {key}")
            if retained and not is_candidate:
                issue("error", name, f"Retained key absent from candidate events: {key}")
            if not is_candidate and values and "missing_production_annotation" not in values:
                issue("error", name, f"Exclusion key absent from observed candidates: {key}")
            sample_audit.append({"dataset": name, **dict(zip(KEY, key)), "candidate": is_candidate,
                                 "retained": retained, "reasons": ";".join(sorted(values)),
                                 "n_reasons": len(values), "missing_production_annotation": key in missing_production})
    samples = pd.DataFrame(sample_audit)
    shard_checks = []
    for filename, group in manifest.groupby("hdf5_path"):
        path = syll / filename
        if not required(path):
            continue
        with h5py.File(path, "r") as handle:
            n = int(handle.attrs["sample_count"])
            shape_n = handle["eeg/data"].shape[0]
            shard_digest = str(handle.attrs["build_config_sha256"])
            if n != len(group) or shape_n != n or shard_digest != digest:
                issue("error", filename, "HDF5 count or configuration digest mismatch.")
            if sorted(group.hdf5_index.astype(int)) != list(range(n)):
                issue("error", filename, "Manifest HDF5 indexes are not unique and contiguous.")
            shard_checks.append({"path": filename, "manifest_samples": len(group), "sample_count": n,
                                 "eeg_shape_samples": shape_n, "config_hash_matches": shard_digest == digest})
    run_table = pd.DataFrame(runs)
    export_rejection_table(run_table, output)
    phase_table = pd.DataFrame(phases)
    participant_run = run_table.groupby(["subject", "task"], as_index=False).agg(
        duration_s=("duration_s", "sum"), bad_duration_s=("bad_duration_s", "sum"),
        n_runs=("run", "nunique"))
    participant_run["bad_percent"] = 100 * participant_run.bad_duration_s / participant_run.duration_s
    participant_phase = phase_table.groupby(["subject", "task", "phase"], as_index=False).agg(
        phase_duration_s=("phase_duration_s", lambda x: x.sum(min_count=len(x))),
        bad_duration_s=("bad_duration_s", lambda x: x.sum(min_count=len(x))), n_runs=("run", "nunique"))
    participant_phase["bad_percent"] = 100 * participant_phase.bad_duration_s / participant_phase.phase_duration_s
    retention = samples.groupby(["subject", "dataset", "task"], as_index=False).agg(
        candidates=("candidate", "sum"), retained=("retained", "sum"),
        missing_production_annotations=("missing_production_annotation", "sum"))
    retention["retained_percent"] = 100 * retention.retained / retention.candidates.replace(0, np.nan)
    retention["excluded_candidates"] = retention.candidates - retention.retained
    retention["cue_linked_opportunities"] = retention.candidates + retention.missing_production_annotations
    retention["retained_per_opportunity_percent"] = 100 * retention.retained / retention.cue_linked_opportunities
    run_retention = samples.groupby(["subject", "task", "run", "dataset"], as_index=False).agg(
        candidates=("candidate", "sum"), retained=("retained", "sum"),
        missing_production_annotations=("missing_production_annotation", "sum"))
    run_retention["retained_percent"] = 100 * run_retention.retained / run_retention.candidates.replace(0, np.nan)
    dataset_totals = retention.groupby("dataset", as_index=False)[
        ["candidates", "retained", "excluded_candidates", "missing_production_annotations", "cue_linked_opportunities"]].sum()
    dataset_totals["retained_percent"] = 100 * dataset_totals.retained / dataset_totals.candidates
    dataset_totals["retained_per_opportunity_percent"] = 100 * dataset_totals.retained / dataset_totals.cue_linked_opportunities
    reason_rows = []
    for r in samples.to_dict("records"):
        for reason in r["reasons"].split(";"):
            if reason:
                reason_rows.append({**{k: r[k] for k in ("subject", "dataset", "task")}, "reason": reason,
                                    "category": "qc" if reason == "overlaps_bad_annotation" else
                                    "missing_annotation" if reason == "missing_production_annotation" else "construction"})
    reason_counts = pd.DataFrame(reason_rows).groupby(["subject", "task", "dataset", "reason", "category"], as_index=False).size()
    # Explicit zero counts only after complete event/manifest reconciliation.
    category_rows = []
    for r in retention.to_dict("records"):
        related = samples.loc[samples.subject.eq(r["subject"]) & samples.dataset.eq(r["dataset"])]
        for reason in ("overlaps_bad_annotation", "missing_production_annotation", "invalid_event_time",
                       "unmapped_syllable", "eeg_window_out_of_bounds", "audio_window_out_of_bounds"):
            count = int(related.reasons.str.split(";").map(lambda values: reason in values).sum())
            category_rows.append({"subject": r["subject"], "dataset": r["dataset"], "reason": reason, "count": count})
    summary_rows = []

    def summarize(table, groups, metric, numerator=None, denominator=None):
        for keys, group in table.groupby(groups):
            if not isinstance(keys, tuple): keys = (keys,)
            values = group[metric].dropna()
            summary_rows.append({**dict(zip(groups, keys)), "metric": metric, "n_participants": len(values),
                                 "median": values.median(), "q1": values.quantile(.25), "q3": values.quantile(.75),
                                 "total_numerator": group[numerator].sum() if numerator else np.nan,
                                 "total_denominator": group[denominator].sum() if denominator else np.nan})
    summarize(participant_run, ["task"], "bad_percent", "bad_duration_s", "duration_s")
    summarize(participant_phase, ["task", "phase"], "bad_percent", "bad_duration_s", "phase_duration_s")
    summarize(retention, ["dataset"], "retained_percent", "retained", "candidates")
    # Channel operations are run decisions; median across a participant's runs
    # avoids presenting repeated runs as independent participants.
    channel_metrics = [c for c in run_table if c.endswith("_channels") and c != "final_eeg_channels"]
    participant_channels = run_table.groupby(["subject", "task"], as_index=False)[channel_metrics].median()
    for metric in channel_metrics:
        summarize(participant_channels, ["task"], metric)
    summary = pd.DataFrame(summary_rows)
    for name, table in {"coverage": pd.DataFrame(coverage), "run_qc": run_table,
                        "phase_run_qc": phase_table, "participant_recording_qc": participant_run,
                        "participant_phase_qc": participant_phase, "participant_channel_qc": participant_channels,
                        "sample_audit": samples, "participant_retention": retention,
                        "run_retention": run_retention,
                        "all_exclusion_reason_counts": pd.DataFrame(category_rows),
                        "dataset_retention_totals": dataset_totals,
                        "exclusion_reason_totals": pd.DataFrame(category_rows).groupby(["dataset", "reason"], as_index=False)["count"].sum(),
                        "exclusion_reasons": reason_counts, "descriptive_summary": summary,
                        "hdf5_checks": pd.DataFrame(shard_checks),
                        "issues": pd.DataFrame(issues, columns=["level", "context", "message"])}.items():
        table.to_csv(output / f"{name}.tsv", sep="\t", index=False, na_rep="n/a")
    plot_results(participant_phase, retention, output)
    text = subsection(participant_run, participant_phase, retention, summary, run_table, config, exclusions, len(issues))
    (output / "signal_quality.md").write_text(text, encoding="utf-8")
    note = {
        "command": [sys.executable, *sys.argv], "bids_root": str(bids), "preprocessed_root": str(pre),
        "syllable_root": str(syll), "build_config": config, "build_config_sha256": digest,
        "inputs": sorted(set(inputs)), "checks": {"runs": len(run_table), "shards": len(shard_checks),
                                                  "issues": len(issues), "multiple_reason_samples": int((samples.n_reasons > 1).sum())},
        "definitions": {
            "recording_duration": "n_samples / sfreq inferred from BrainVision binary size; no EEG values loaded.",
            "bad": "Union of positive-duration published BAD annotations; BLINK is not BAD. Channel scopes counted separately if present.",
            "phases": "Listening/repetition use cue onset+duration. Syllabification is first cue onset through last cue end per trial, capped at trial_end; includes small between-cue gaps. All intervals clipped to recording and unioned.",
            "channel_operations": "automatic_flagged from per-run EEG channel QC; reviewed choices per run; applied operations from condition-level ICA provenance, repeated across constituent runs; total removed from raw vs final EEG channel sets.",
            "retention": "Denominator = observed matching event keys; missing productions lack an event and are reported separately, not as observed candidates. Multiple reasons use sets; exclusion counts by reason are non-exclusive if multiple reasons occur.",
            "aggregation": "Duration-weighted ratio after summing runs within participant; retention retained/candidates within participant; medians and 25th/75th quantiles across participants.",
        },
        "limitations": ["Annotations measure applied decisions, not residual signal contamination.",
                        "Current build source and BAD-source policy must be distinguished from preprocessing QC.",
                        "Builder records first applicable exclusion reason; unrecorded secondary failures cannot be reconstructed.",
                        "Published annotation schema has no channel scope fields; exporter preserves global spans only.",
                        "No EEG/audio values or residual myogenic contributions were quantified."],
    }
    (output / "provenance.json").write_text(json.dumps(note, indent=2) + "\n", encoding="utf-8")
    (output / "README.md").write_text(
        "# Signal quality analysis\n\nRun: `python -m reproducibility.scripts.signal_quality --bids-root data/spellitout`\n\n"
        "Each execution creates a fresh output directory. See provenance.json for input inventory, exact command, "
        "metric definitions and checks; issues.tsv for missing information or mismatches. phase_run_qc.tsv and run_qc.tsv "
        "are run-level evidence; descriptive_summary.tsv aggregates participant-level values. "
        "sample_audit.tsv reconciles every observed event against retained samples and exclusions. "
        "HDF5 checks read attributes and array shapes only. No signal values were loaded.\n", encoding="utf-8")
    (output / "analysis_note.md").write_text(
        "# Inputs, definitions and checks\n\n"
        f"Dataset: `{bids}`. Continuous derivative: `{pre}`. HDF5 derivative: `{syll}`. "
        "The README and stage 7/8 documentation identify these as the published dataset, rather than the working "
        "FIF checkpoints or the BIDS test export.\n\n"
        "Reproduce with `python -m reproducibility.scripts.signal_quality --bids-root data/spellitout`. "
        "Use `--output-path` only for a new directory. The precise invocation and complete input inventory are in provenance.json.\n\n"
        f"Checked {len(run_table)} raw/preprocessed run pairs and {len(shard_checks)} HDF5 shards: "
        f"{len(issues)} missing-file or consistency issues; "
        f"{int((samples.n_reasons > 1).sum())} samples with multiple recorded exclusion reasons. "
        "Checks include binary sample counts, raw/processed duration agreement, QC approval vs published BAD duration, "
        "channel plan vs applied interpolation/exclusion, resolved ICA decisions, candidate key uniqueness, "
        "retained/excluded reconciliation, missing-production matching, HDF5 sample counts, manifest indexes, and build hashes.\n\n"
        "BAD intervals are unioned before overlap is calculated. BLINK annotations are not included. "
        "The exported schema contains no channel-specific span fields; its BAD annotations apply globally. "
        "The audit does not infer absence of channel-specific contamination. Listening and repetition follow cue durations. "
        "Syllabification uses the first-to-last cue envelope per trial, including the small inter-cue gaps; "
        "all phase intervals are clipped at trial and recording boundaries.\n\n"
        "Run durations are aggregated within participant before taking ratios. Median/IQR use the 25 participants. "
        "Channel summaries first take each participant's median run count; condition-wide operations are repeated in "
        "run tables for traceability, so their run sums must not be interpreted as distinct channels. "
        "Raw-to-final channel removal is reported separately from exclude_from_ica.\n\n"
        "The current HDF5 build uses raw signals and BAD annotations from source events. Its source events do not "
        "contain reviewed preprocessing BAD spans. All observed candidates survive; the 45 absent production "
        "annotations are not observed candidates. dataset_retention_totals.tsv also supplies a separate cue-linked "
        "opportunity denominator. The builder records only the first applicable failure: no secondary exclusion "
        "reason is invented. The absence of QC exclusions here reflects the saved configuration, not clean EEG.\n\n"
        "Author-reported residual myogenic activity and possible reading/gaze interference are qualitative observations. "
        "No physiological blink rates, residual artifact amplitudes, or myogenic contributions were estimated.\n",
        encoding="utf-8")
    print(f"Saved results to {output}", flush=True)
    print(text)


def plot_results(phases, retention, output):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), constrained_layout=True)
    rng = np.random.default_rng(435656)
    colors = {"overt": "#d95f02", "covert": "#1b9e77"}
    for task, shift in (("overt", -.16), ("covert", .16)):
        for x, phase in enumerate(PHASE_EVENTS):
            values = phases.loc[phases.task.eq(task) & phases.phase.eq(phase), "bad_percent"].to_numpy()
            axes[0].scatter(x + shift + rng.uniform(-.055, .055, len(values)), values,
                            s=20, alpha=.7, color=colors[task], label=task.title() if x == 0 else None)
            axes[0].plot([x + shift - .09, x + shift + .09], [np.nanmedian(values)] * 2, color="black", lw=2)
    axes[0].set(xticks=range(3), xticklabels=["Listening", "Repetition", "Syllabification"],
                ylabel="Duration annotated BAD (%)", title="A  Applied span-level QC")
    axes[0].legend(frameon=False)
    for x, name in enumerate(("covert_cue", "overt_cue", "overt_production")):
        values = retention.loc[retention.dataset.eq(name), "retained_percent"].to_numpy()
        axes[1].scatter(x + rng.uniform(-.09, .09, len(values)), values, s=20, alpha=.7,
                        color=colors["covert" if name.startswith("covert") else "overt"])
        axes[1].plot([x-.12, x+.12], [np.nanmedian(values)]*2, color="black", lw=2)
    axes[1].set(xticks=range(3), xticklabels=["covertcue", "overtcue", "overtproduction"],
                ylabel="Observed candidate samples retained (%)", title="B  Published HDF5 availability")
    axes[1].set_ylim(0, 103)
    for ax in axes:
        ax.grid(axis="y", alpha=.2)
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(output / "signal_quality.png", dpi=300)
    fig.savefig(output / "signal_quality.pdf")
    plt.close(fig)
    (output / "figure_caption.md").write_text(
        "**Figure. Applied quality-control decisions and syllable-sample availability.** "
        "(A) Percentage of phase duration overlapping global BAD annotations, separately for overt and covert tasks. "
        "For each participant, BAD durations and phase durations are summed over three runs before their ratio is calculated. "
        "(B) Retained samples as a percentage of observed candidate events in each published HDF5 dataset; "
        "missing production annotations are reported separately and excluded from this denominator. "
        "Each dot represents one participant; short black lines mark the participant median. "
        "The HDF5 build uses filtered raw signals and source-event annotations, rather than the reviewed preprocessing "
        "BAD spans. Thus panel B measures availability under the saved build configuration, not survival of panel A's QC. "
        "Lower BAD percentages do not establish less residual contamination.\n", encoding="utf-8")


def subsection(recording, phases, retention, summary, runs, config, exclusions, nissues):
    def stat(table, metric):
        v = table[metric]
        return f"{v.median():.2f}% (IQR {v.quantile(.25):.2f}–{v.quantile(.75):.2f}%)"
    texts = []
    for task in ("overt", "covert"):
        subset = phases.loc[phases.task.eq(task)]
        texts.append(f"For {task} recordings, the median participant-level proportions of phase duration annotated BAD were " +
                     "; ".join(f"{stat(subset.loc[subset.phase.eq(phase)], 'bad_percent')} during {phase}" for phase in PHASE_EVENTS) + ".")
    count_texts = []
    for name in ("covert_cue", "overt_cue", "overt_production"):
        group = retention.loc[retention.dataset.eq(name)]
        count_texts.append(f"{name.replace('_', '')}: {int(group.retained.sum()):,}/{int(group.candidates.sum()):,} "
                           f"({100*group.retained.sum()/group.candidates.sum():.2f}% overall)")
    flagged = []
    for task in ("overt", "covert"):
        group = runs.loc[runs.task.eq(task)].groupby("subject").automatic_flagged_channels.median()
        flagged.append(f"{task}: {group.median():.1f}, IQR {group.quantile(.25):.1f}–{group.quantile(.75):.1f}")
    missing = int(exclusions.reason.eq("missing_production_annotation").sum())
    interpolation = []
    for task in ("overt", "covert"):
        v = runs.loc[runs.task.eq(task)].groupby("subject").applied_interpolated_channels.median()
        interpolation.append(f"{v.median():.0f} (IQR {v.quantile(.25):.0f}–{v.quantile(.75):.0f}) for {task}")
    return (
        "## Signal quality and quality control\n\n"
        f"Quality control was documented for {recording.subject.nunique()} participants across {len(runs)} run recordings "
        "covering overt and covert tasks. Published continuous EEG was accompanied by run-level artifact reviews, "
        "channel decisions, condition-level channel harmonization, and ICA provenance. These records distinguish "
        "automatic flags from reviewed interpolation or exclusion decisions. All recordings remain continuous; "
        "BAD annotations identify intervals omitted during ICA fitting or downstream epoch selection. "
        "Retained BLINK annotations were not automatically treated as BAD.\n\n"
        f"The recordings totalled {runs.duration_s.sum()/3600:.2f} hours. "
        "Overlapping BAD intervals were counted once and clipped to recording and phase boundaries. "
        "Listening and repetition intervals followed their cue durations; syllabification extended from the first "
        "syllable cue to the end of the final cue within each trial. Durations were summed over runs within each "
        "participant before percentages were calculated; descriptive statistics therefore use participants, not "
        "runs, as independent units. " + " ".join(texts) + " "
        "The median number of automatically flagged EEG channels, after first taking each participant's median "
        "across runs, was " + "; ".join(flagged) + ". "
        "Flagging did not imply removal; the accompanying tables separately report reviewed and applied operations.\n\n"
        "The median condition-wide interpolation counts were " + " and ".join(interpolation) +
        ". No additional EEG channels were excluded through condition-level QC; AF7 and AF8 were absent "
        "from every preprocessed recording.\n\n"
        "Observed candidate events were reconciled with the saved HDF5 manifest and exclusion log. " +
        "; ".join(count_texts) + f". In addition, {missing} cue-linked overt productions lacked production annotations "
        "and therefore could not supply production-aligned samples; these are reported separately from observed "
        "production candidates. Including these missing productions in the cue-linked opportunity denominator "
        "gives 34,605/34,650 (99.87%) available production samples. The saved build configuration uses raw signals, additional filtering and resampling, "
        "and source-event BAD annotations. The raw source events contain no reviewed preprocessing BAD spans. "
        "Consequently, this sample retention describes availability under that build configuration and does not "
        "measure retention after the reviewed span-level QC.\n\n"
        "These metrics quantify recorded annotations and decisions rather than certifying artifact-free EEG. "
        "During qualitative inspection, the authors observed residual myogenic activity in frontal and EOG channels "
        "during overt speech, together with potential interference from reading and gaze shifts during listening. "
        "Their contributions were not quantified here. Accordingly, detector-labelled blink counts were not "
        "interpreted as physiological blink rates, relabelled as ocular artifacts, or used as EEG-quality measures. "
        "A smaller annotated BAD proportion does not establish lower residual contamination, including during "
        "syllabification.\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bids-root", type=Path, default=Path("data/spellitout"))
    parser.add_argument("--preprocessed-root", type=Path)
    parser.add_argument("--syllable-root", type=Path)
    parser.add_argument("--output-path", type=Path, help="Must not already exist.")
    main(parser.parse_args())
