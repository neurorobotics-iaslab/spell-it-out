import mne
import numpy as np
import pandas as pd
from pathlib import Path
import sys

from tqdm import tqdm
sys.path.append('.')  
from spellitout.spellitout import SpellItOut


data_path = Path(r"D:\SHARED\PhD\data\Spell-IT-OUT_V2\data\spellitout")
data_path = None
task = 'silent'
subjects = range(1, 26)
if task == 'overt': 
    for i in [16,24,25]: subjects.remove(i)

# Define the phases, their names, and epoch durations in seconds
phases_of_interest = {
    'sentence_audio': {'save_name': 'listening', 'duration': 7.0},
    'repetition_cue': {'save_name': 'sentence pronunciation', 'duration': 5.0},
    'syllable_cue':   {'save_name': 'syllable pronunciation', 'duration': 0.6}
}

# List to store the final statistics
sebr_results = []

for subject in tqdm(subjects):
    dataset = SpellItOut(task, subjects=[subject],data_path=data_path)
    data = dataset.get_data()
    subject_id = f"sub-{subject:02d}"

    raws_sub = []
    runs = data[subject_id]
    # 1. Load and prepare data for the CURRENT subject
    for run_id, (raw,events) in tqdm(runs.items(), desc=f"Processing subject {subject_id}",leave=False):
        raw.load_data()
        raw.resample(256, verbose=False)  
        raws_sub.append(raw)
        
    # Concatenate runs for this subject to compute overall task stats
    raw_sub = mne.concatenate_raws(raws_sub)
    
    # 2. Detect EOG events (Blinks)
    # If the 'EOG' channel is noisy, you can pass ch_name=['FP1', 'FP2', 'FPZ'] instead
    # eog_events = mne.preprocessing.find_eog_events(raw_sub, ch_name='EOG', verbose=False)
    eog_events = mne.preprocessing.find_eog_events(raw_sub, ch_name=['FP1', 'FP2', 'FPZ'], verbose=False)
    blink_samples = eog_events[:, 0] # Extract just the sample indices of the blinks
    
    # 3. Extract task events based on annotations
    task_events, ev_map = mne.events_from_annotations(raw_sub, verbose='error')
    sfreq = raw_sub.info['sfreq']
    
    # 4. Compute SEBR for each phase
    for ann_name, phase_info in phases_of_interest.items():
        # Check if this annotation exists for the subject
        if ann_name not in ev_map:
            print(f"Warning: Annotation '{ann_name}' not found for Subject {subject_id}.")
            continue
            
        trigger_id = ev_map[ann_name]
        
        # Filter task events to only keep the trials for this phase
        phase_events = task_events[task_events[:, 2] == trigger_id]
        
        duration_sec = phase_info['duration']
        duration_samples = int(duration_sec * sfreq)
        
        total_blinks = 0
        num_trials = len(phase_events)
        
        # Count blinks that fall into the epoch window for each trial
        for event in phase_events:
            start_sample = event[0]
            end_sample = start_sample + duration_samples
            
            # Find how many blink timestamps fall between start and end sample
            blinks_in_trial = np.sum((blink_samples >= start_sample) & (blink_samples <= end_sample))
            total_blinks += blinks_in_trial
            
        # 5. Calculate Average Blink Rate per Minute
        if num_trials > 0:
            # Total time spent in this phase (in minutes)
            total_time_min = (num_trials * duration_sec) / 60.0
            sebr = total_blinks / total_time_min
        else:
            sebr = np.nan
            total_time_min = 0.0
            
        # Store the values
        sebr_results.append({
            'Subject': subject_id,
            'Condition': phase_info['save_name'],
            'SEBR (blinks/min)': sebr,
            'Total Blinks': total_blinks,
            'Total Trials': num_trials,
            'Analyzed Time (min)': round(total_time_min, 2)
        })

# 6. Convert to a DataFrame for easy saving/viewing
df_sebr = pd.DataFrame(sebr_results)
print(df_sebr)

# You can easily save this to CSV for your paper
df_sebr.to_csv('derivatives/SEBR_statistics.csv', index=False)