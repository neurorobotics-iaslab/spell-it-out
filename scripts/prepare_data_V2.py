#%%
 
import sys
import json
from pathlib import Path

import mne
import textgrid

import numpy as np
import pandas as pd

from spellitout import Event, SpellItOut
from spellitout.utils import make_montage, reset_mne_first_samp

RAW_DATA_PATH = Path(r"D:\SHARED\PhD\data\speech_imagery_recordings")
OUT_PATH = Path(r"D:\SHARED\PhD\data\spell-it-out\data\spellitout")

PAD = 0.5 #[s]
SILENT_CUE_LEAD_IN = 3.0  # seconds;

# Columns used in every BIDS events.tsv file.  The first two columns are
# required by BIDS and must remain in this order.
BIDS_EVENT_COLUMNS = [
    "onset", "duration", "trial_type", "trial_index", "phase",
    "trigger_code", "event_source", "stimulus_id", "stim_file", "word",
    "word_index", "syllable", "syllable_index", "annotation_tier",
]
global wordlist

# MISSING SUBJECTS (I still have to annotate them) : 16(j9), 24(l7), 25(l9)

def _phase_to_trial_type(token_type, phase):
    """Map TextGrid tier/context labels to the controlled BIDS vocabulary."""
    phase = str(phase).strip().lower()
    listening_phases = {"cue", "listening", "listen", "sentence_cue"}
    if token_type == "word":
        return "word_listening" if phase in listening_phases else "word_production"
    return "syllable_listening" if phase in listening_phases else "syllable_production"


def _sentence_number_from_code(code, sentence_ids):
    """Support both 1401-based and legacy Event.CUE_1-based sentence codes."""
    candidates = (code - 1400, code - Event.CUE_1 + 1)
    return next((candidate for candidate in candidates if candidate in sentence_ids), None)


def _stimulus_id_from_code(code):
    """Return the on-disk stimulus ID (for example, 3138 -> sentence-1738)."""
    return f"sentence-{code - 1400:03d}"


def parse_silent_stimulus_events(trigger_events, stimuli_path):
    """Project word/syllable timings from stimulus TextGrids onto silent runs.

    The alignment formula is inherited from the previous silent-data pipeline:
    the sentence audio ends three seconds before the end of its cue trigger.
    """
    events = []
    trial_index = 0
    triggers = trigger_events.sort_values("onset").reset_index(drop=True)

    for _, trigger in triggers.iterrows():
        code = int(trigger["trigger_code"])
        if code == Event.TRIAL_START:
            trial_index += 1
            continue
        if _sentence_number_from_code(code, wordlist["id"].tolist()) is None:
            continue

        stimulus_id = _stimulus_id_from_code(code)
        textgrid_path = stimuli_path / f"{stimulus_id}_annotations.TextGrid"
        if not textgrid_path.exists():
            print(f"Missing stimulus annotation: {textgrid_path}")
            continue

        try:
            tg = textgrid.TextGrid.fromFile(str(textgrid_path))
        except Exception as exc:
            print(f"Error parsing stimulus TextGrid {textgrid_path}: {exc}")
            continue

        phase_durations = [
            interval.maxTime - interval.minTime
            for tier in tg.tiers if tier.name.lower() == "phase"
            for interval in tier.intervals if interval.mark
        ]
        if not phase_durations:
            print(f"No phase interval in stimulus annotation: {textgrid_path}")
            continue
        audio_duration = max(phase_durations)
        playback_start = (
            float(trigger["onset"]) + float(trigger["duration"])
            - SILENT_CUE_LEAD_IN - audio_duration
        )

        for tier in tg.tiers:
            token_type = tier.name.lower()
            if token_type not in {"word", "syllable"}:
                continue
            for interval in tier.intervals:
                if not interval.mark:
                    continue
                events.append({
                    "type": token_type,
                    "start": playback_start + interval.minTime,
                    "duration": interval.maxTime - interval.minTime,
                    "description": interval.mark.strip(),
                    "phase": "listening",
                    "sequence_id": trial_index,
                })
    return events

def parse_textgrid(filepath: Path):
    """
    Parses TextGrid to extract words and syllables (in syllabification phase).
    """
    parsed_events = []
    try:
        tg = textgrid.TextGrid.fromFile(str(filepath))
        
        # 1. Map context (Phase and Trial)
        phases = []
        trial_map = []
        
        for tier in tg.tiers:
            if tier.name.lower() == "phase":
                for interval in tier.intervals:
                    if interval.mark:
                        phases.append((interval.minTime, interval.maxTime, interval.mark))
            elif tier.name.lower() == "trial":
                for interval in tier.intervals:
                    if interval.mark:
                        trial_map.append((interval.minTime, interval.maxTime, interval.mark))

        def get_context(t_mid):
            curr_phase = "unknown"
            curr_trial = "unknown"
            margin = 1
            for start, end, mark in phases:
                if start - margin <= t_mid <= end + margin:
                    curr_phase = mark
                    break
            for start, end, mark in trial_map:
                if start - margin <= t_mid <= end + margin:
                    curr_trial = mark
                    break
            return curr_phase, curr_trial

        # 2. Extract events
        for tier in tg.tiers:
            tier_name = tier.name.lower()
            
            # A. Standard Words
            if tier_name == "word":
                for interval in tier.intervals:
                    if not interval.mark: continue
                    
                    t_mid = (interval.minTime + interval.maxTime) / 2
                    phase, trial = get_context(t_mid)
                    
                    parsed_events.append({
                        "type": "word",
                        "start": interval.minTime,
                        "duration": interval.maxTime - interval.minTime,
                        "description": interval.mark,
                        "phase": phase,
                        "sequence_id": trial,
                    })

            # B. Syllables 
            elif tier_name == "syllable":
                for interval in tier.intervals:
                    if not interval.mark: continue
                    
                    t_mid = (interval.minTime + interval.maxTime) / 2
                    phase, trial = get_context(t_mid)
                    parsed_events.append({
                        "type": "syllable",
                        "start": interval.minTime,
                        "duration": interval.maxTime - interval.minTime,
                        "description": interval.mark,
                        "phase": phase,
                        "sequence_id": trial
                    })

    except Exception as e:
        print(f"Error parsing TextGrid {filepath}: {e}")
        
    return parsed_events

def process_trigger_events(raw):
    annotations = {'trigger_code':[],'onset':[],'duration':[]}
    for ev in set([int(annot['description']) for annot in raw.annotations if int(annot['description']) <= Event.OFFSET]):
        start_annotations = [annot for annot in raw.annotations if int(annot['description']) == ev]
        end_annotations = [annot for annot in raw.annotations if int(annot['description']) == ev + Event.OFFSET]
        if len(end_annotations) != len(start_annotations):
            print(f"Event {ev} has {len(start_annotations)} start annotations but {len(end_annotations)} end annotations!")
            continue
        for start, end in zip(start_annotations, end_annotations):
            annotations['trigger_code'].append(ev)
            annotations['onset'].append(start['onset'])
            annotations['duration'].append(end['onset'] - start['onset'])
    trigger_events = pd.DataFrame({
        'onset': annotations['onset'],
        'duration': annotations['duration'],
        'trigger_code': annotations['trigger_code'],
    })
    trigger_events["event_source"] = "gdf_trigger"
    return trigger_events


def make_bids_events(trigger_events, word_events, condition):
    """Combine GDF and TextGrid annotations into one BIDS-compliant table."""
    sentences = wordlist.set_index("id")
    trigger_rows = []
    trial_index = 0
    current_stimulus_id = pd.NA

    # Sort first, because trial and stimulus identity are carried forward.
    triggers = trigger_events.sort_values("onset").reset_index(drop=True)
    for i, row in triggers.iterrows():
        code = int(row.trigger_code)
        event = {
            "onset": row.onset,
            "duration": row.duration,
            "trial_type": "n/a",
            "trial_index": trial_index if trial_index else pd.NA,
            "phase": "n/a",
            "trigger_code": code,
            "event_source": "gdf_trigger",
            "stimulus_id": current_stimulus_id,
            "stim_file": "n/a",
            "word": "n/a",
            "word_index": pd.NA,
            "syllable": "n/a",
            "syllable_index": pd.NA,
            "annotation_tier": "n/a",
        }

        if code == Event.TRIAL_START:
            trial_index += 1
            current_stimulus_id = pd.NA
            event.update(trial_type="trial_start", phase="trial_start",
                         trial_index=trial_index, stimulus_id=current_stimulus_id)
        elif code == Event.FIXATION:
            event.update(trial_type="fixation", phase="fixation")
        elif (sentence_number := _sentence_number_from_code(code, sentences.index)) is not None:
            current_stimulus_id = _stimulus_id_from_code(code)
            event.update(
                trial_type="sentence_audio", phase="listening",
                stimulus_id=current_stimulus_id,
                stim_file=f"{current_stimulus_id}.wav",
            )
        elif code == Event.REPETITION:
            event.update(trial_type="repetition_cue", phase="repetition")
        elif code == Event.DOTS:
            event.update(trial_type="rhythm_dots", phase="pre_syllable_production")
        elif code == Event.SYLLABATION:
            event.update(trial_type="syllable_cue", phase="syllable_production")
        elif code == Event.TRIAL_END:
            event.update(trial_type="trial_end", phase="trial_end")
        else:
            # Retain unexpected hardware markers without inventing a task label.
            event.update(trial_type="unclassified_trigger")
        
        trigger_rows.append(event)

    trial_stimulus_ids = {
        row["trial_index"]: row["stimulus_id"]
        for row in trigger_rows
        if row["trial_type"] == "sentence_audio" and pd.notna(row["trial_index"])
    }
    textgrid_rows = []
    token_counters = {}
    for event in sorted(word_events, key=lambda item: item["start"]):
        trial = pd.to_numeric(event["sequence_id"], errors="coerce")
        trial = int(trial) if pd.notna(trial) else pd.NA
        token_type = event["type"]
        counter_key = (trial, token_type)
        token_counters[counter_key] = token_counters.get(counter_key, 0) + 1
        row = {
            "onset": event["start"], "duration": event["duration"],
            "trial_type": _phase_to_trial_type(token_type, event["phase"]),
            "trial_index": trial, "phase": event["phase"],
            "trigger_code": pd.NA,
            "event_source": f"textgrid_{token_type}",
            "stimulus_id": trial_stimulus_ids.get(trial, "n/a"), "stim_file": "n/a",
            "word": "n/a", "word_index": pd.NA,
            "syllable": "n/a", "syllable_index": pd.NA,
            "annotation_tier": f"{token_type}_{event['phase']}",
        }
        if token_type == "word":
            row.update(word=event["description"], word_index=token_counters[counter_key])
        else:
            row.update(syllable=event["description"], syllable_index=token_counters[counter_key])
        textgrid_rows.append(row)

    events_df = pd.DataFrame(trigger_rows + textgrid_rows, columns=BIDS_EVENT_COLUMNS)
    events_df = events_df.sort_values("onset", kind="stable").reset_index(drop=True)
    return events_df


def events_sidecar():
    """Metadata shared by all run-level events.tsv files."""
    return {
        "trial_type": {"Description": "Controlled event category.", "Levels": {
            "trial_start": "Start of a trial.", "fixation": "Fixation period.",
            "sentence_audio": "Presentation of the sentence audio stimulus.",
            "repetition_cue": "Cue to repeat the sentence.",
            "rhythm_dots": "Rhythm dots marker.", 
            "syllable_cue": "Cue for syllable production.",
            "word_production": "Observed spoken word.", 
            "syllable_production": "Observed spoken syllable.",
            "word_listening": "Word timing during listening.", 
            "syllable_listening": "Syllable timing during listening.",
            "trial_end": "End of a trial.",
            "unclassified_trigger": "Hardware marker not assigned to a task event."}},
        "trial_index": {"Description": "One-based trial number within the run."},
        "phase": {"Description": "Broader task phase containing the event."},
        "trigger_code": {"Description": "Original numeric GDF trigger code."},
        "event_source": {"Description": "Annotation source used to create this row."},
        "stimulus_id": {"Description": "Stable identifier of the sentence stimulus."},
        "stim_file": {"Description": "Path to the presented stimulus relative to the dataset stimuli/audio directory."},
        "word": {"Description": "Orthographic word token from the TextGrid annotation."},
        "word_index": {"Description": "One-based word position within the trial."},
        "syllable": {"Description": "Syllable token from the TextGrid annotation."},
        "syllable_index": {"Description": "One-based syllable position within the trial."},
        "annotation_tier": {"Description": "TextGrid tier/context used for the annotation."},
    }

def prepare_raw(filepath,subj_id):
    raw = mne.io.read_raw_gdf(filepath,preload=True, verbose=False)
    raw.apply_function(lambda x: x*1e-6, channel_wise=False, verbose='error')
    raw.pick('eeg')
    raw.drop_channels(ch_names=['M1','M2'])
    raw.set_channel_types({'EOG': 'eog'})
    # raw.set_meas_date(None)
    raw.info['subject_info']['his_id'] = subj_id

    # get montage of eeg channels
    eeg_channels = [raw.info['ch_names'][i] for i in mne.pick_types(raw.info, eeg=True, exclude=[])]
    montage = make_montage(eeg_channels)
    raw.set_montage(montage)
    events,_ = mne.events_from_annotations(raw)

    start_at = events[0][0] / raw.info['sfreq'] - PAD
    end_at = events[-1][0] / raw.info['sfreq'] + PAD

    cropped = reset_mne_first_samp(raw.copy().crop(tmin=start_at, tmax=end_at))

    return cropped

def process(subj_id, raw_path, annot_path=None, audio_path=None):
    raw = prepare_raw(raw_path,subj_id)
    trigger_events = process_trigger_events(raw)
    condition = "overt" if "overt" in raw_path.name.lower() else "silent"
    if condition == "overt":
        word_events = parse_textgrid(annot_path)
    else:
        word_events = parse_silent_stimulus_events(trigger_events, OUT_PATH / "stimuli/audio")
    events_df = make_bids_events(trigger_events, word_events, condition)
    raw.set_annotations(None)
    return raw,events_df

#%%

if __name__ == "__main__":
    raw_data_path = Path(RAW_DATA_PATH)
    recordings_path = raw_data_path / "recordings"
    out_path = Path(OUT_PATH)

    subjects = sorted([subject.name for subject in recordings_path.iterdir() if subject.is_dir()])


    # cp stimuli and annotations to out_path
    (out_path / "stimuli" / "audio").mkdir(parents=True, exist_ok=True)
    for stim_file in (raw_data_path / "stimuli").iterdir():
        out_file = out_path / "stimuli" / "audio" / stim_file.name
        if not out_file.exists():
            out_file.write_bytes(stim_file.read_bytes())
    # cop wordlist.tsv to out_path
    out_wordlist_path = out_path / "stimuli" / "wordlist.tsv"
    if not out_wordlist_path.exists():
        out_wordlist_path.write_bytes((raw_data_path /  "wordlist.tsv").read_bytes())
    (out_path / "stimuli").mkdir(parents=True, exist_ok=True)

    wordlist = SpellItOut('overt', data_path=out_path).get_wordlist()

    for subjno, subject in enumerate(subjects):
        # if subjno!=21: continue
        subj_path = recordings_path / subject
        overt_runs = sorted(list(subj_path.rglob("*overt.gdf")))
        silent_runs = sorted(list(subj_path.rglob("*silent.gdf")))
        
        new_subj_id = f"sub-{subjno+1:02d}"
        print(f"Subject: {new_subj_id} ({subject})")
        print(f"  Overt runs:")
        for run_no,orig_gdf_path in enumerate(overt_runs):
            
            run_id = orig_gdf_path.stem.split(".gdf")[0]
            old_audio_path = orig_gdf_path.parent / f"{run_id}.audio.wav"
            old_annot_path = orig_gdf_path.parent / f"{run_id}.annotations.TextGrid"

            new_run_id = f"{new_subj_id}_ses-01_task-overt_run-{run_no+1:02d}"

            if not old_audio_path.exists() or not old_annot_path.exists():
                print(f"  Skipping run {run_no+1}: {run_id}")
                continue

            subj_out_path = out_path / 'overt' / new_subj_id / 'ses-01'
            eeg_path = subj_out_path / 'eeg'
            audio_path = subj_out_path / 'audio'
            subj_out_path.mkdir(parents=True, exist_ok=True)
            eeg_path.mkdir(parents=True, exist_ok=True)
            audio_path.mkdir(parents=True, exist_ok=True)

            new_gdf_path = eeg_path /f"{new_run_id}_eeg.fif"
            events_path = eeg_path / f"{new_run_id}_events.tsv"
            new_audio_path = audio_path / f"{new_run_id}_audio.wav"
            new_annot_path = audio_path / f"{new_run_id}_annotations.TextGrid"

            print(f"    {run_no+1}): {run_id}")
            print(f"      GDF: {orig_gdf_path.name} --> {new_gdf_path.name} + {events_path.name}")
            print(f"      Audio: {old_audio_path.name} --> {new_audio_path.name}")
            print(f"      VAD: {old_annot_path.name} --> {new_annot_path.name}")

            raw,events_df = process(new_subj_id, orig_gdf_path, old_annot_path, old_audio_path)

            raw.save(new_gdf_path, overwrite=True)
            events_df.to_csv(events_path, sep='\t', index=False, na_rep='n/a')
            new_annot_path.write_bytes(old_annot_path.read_bytes())
            new_audio_path.write_bytes(old_audio_path.read_bytes())
    
        print(f"  Silent runs:")
        for run_no,orig_gdf_path in enumerate(silent_runs):
            run_id = orig_gdf_path.stem.split(".gdf")[0]
            new_run_id = f"{new_subj_id}_ses-01_task-silent_run-{run_no+1:02d}"

            subj_out_path = out_path / 'silent' / new_subj_id / 'ses-01'
            subj_out_path.mkdir(parents=True, exist_ok=True)
            eeg_path = subj_out_path / 'eeg'
            eeg_path.mkdir(parents=True, exist_ok=True)

            new_gdf_path = eeg_path / f"{new_run_id}_eeg.fif"
            events_path = eeg_path / f"{new_run_id}_events.tsv"
            
            print(f"    {run_no+1}): {run_id}")
            print(f"      GDF: {orig_gdf_path.name} --> {new_gdf_path.name} + {events_path.name}")

            raw, events_df = process(new_subj_id, orig_gdf_path)
            raw.save(new_gdf_path, overwrite=True)
            events_df.to_csv(events_path, sep='\t', index=False, na_rep='n/a')
        print("_____________________________\n")

    for condition in ("overt", "silent"):
        events_json_path = out_path / condition / f"task-{condition}_events.json"
        events_json_path.parent.mkdir(parents=True, exist_ok=True)
        events_json_path.write_text(json.dumps(events_sidecar(), indent=4), encoding='utf-8')
