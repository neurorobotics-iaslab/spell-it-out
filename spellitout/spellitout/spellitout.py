import os
from pathlib import Path

import mne
import pandas as pd

from .utils import get_run_id, get_subject_id, make_annotations_from_events


class Event:
    OFFSET = 0X8000
    TRIAL_START = 1
    FIXATION = 786
    CUE_1 = 3138
    CUE_2 = 3139
    CUE_3 = 3140
    CUE_4 = 3141
    CUE_5 = 3142
    CUE_6 = 3143
    CUE_7 = 3144
    CUE_8 = 3145
    CUE_9 = 3146
    CUE_10 = 3147
    CUE_11 = 3148
    CUE_12 = 3149
    CUE_13 = 3150
    CUE_14 = 3151
    CUE_15 = 3152
    CUE_16 = 3153
    CUE_17 = 3154
    CUE_18 = 3155
    CUE_19 = 3156
    CUE_20 = 3157
    REPETITION = 782
    DOTS = 780
    SYLLABATION = 781
    TRIAL_END = 897

class SpellItOut:
    def __init__(self, 
                 condition, 
                 subjects = None, 
                 runs = [1,2,3], 
                 data_path = None):
        self.condition = condition
        self.runs = runs

        if data_path is None:
            data_path = os.environ.get('MNE_DATASETS_SPELLITOUT_DATA_PATH', None)
            if data_path is None:
                raise ValueError("Data path not provided and MNE_DATASETS_SPELLITOUT_DATA_PATH environment variable not set.")
        self.data_path = Path(data_path)
            
        if subjects is None or subjects == 'all':
            self.subjects = [int(sub.stem.split('-')[-1]) for sub in (self.data_path/ self.condition ).glob(f'sub-*')]
        else:
            self.subjects = [int(sub) for sub in subjects]

    def get_data(self,  preload=True, return_events=True, return_audio=True):
        out = dict()
        for subjno in self.subjects:
            subject_id = get_subject_id(subjno)

            if not (self.data_path / self.condition / subject_id).exists():
                raise ValueError(f"Data for condition {self.condition} not available for subject {subject_id}. Data path: {self.data_path / self.condition / subject_id}")
            
            out[subject_id] = dict()
            for run in self.runs:
                run_id = get_run_id(subject_id, self.condition, run)
                fif_path = self.data_path / self.condition / subject_id / 'ses-01' / 'eeg'/ f"{run_id}_eeg.fif"
                ev_path = self.data_path / self.condition / subject_id / 'ses-01' / 'eeg'/ f"{run_id}_events.tsv"
                audio_path = self.data_path / self.condition / subject_id / 'ses-01'/ 'audio'  / f"{run_id}_audio.wav"

                raw = mne.io.read_raw_fif(fif_path, preload=preload, verbose='error')
                events_df = pd.read_csv(ev_path, sep='\t')
                raw.set_annotations(make_annotations_from_events(events_df))
                out[subject_id][f'run-{run}'] = (raw, events_df)
            
        return out
    
    def get_wordlist(self):
        sentences_df = pd.read_csv(self.data_path / "stimuli" / "wordlist.tsv", sep='\t', names=['id','sentence','sentence_syll'])
        return sentences_df
    
    def download(self):
        """
        Downloads the Spell-IT-OUT dataset from the specified URL and extracts it to the data_path.
        """
        import io
        import zipfile

        import requests

        raise NotImplementedError("The download method is not available yet")
        # But it should be something along the lines of:
        url = "https://zenodo.org/record/1234567/files/spellitout_dataset.zip"  # Replace with actual URL
        response = requests.get(url)
        if response.status_code == 200:
            with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                z.extractall(self.data_path)
            print(f"Dataset downloaded and extracted to {self.data_path}")
        else:
            raise Exception(f"Failed to download dataset. Status code: {response.status_code}")
    


