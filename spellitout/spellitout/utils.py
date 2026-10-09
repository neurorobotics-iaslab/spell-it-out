import numpy as np
from mne.channels import make_dig_montage, make_standard_montage

def make_montage(subset_channels=None, montage_name='standard_1020'):

    """ Create a dig montage with a (optional) subset of 10-20 standard EEG channels.
    Args:
        subset_channels (list,optional): List of channel names to include in the montage.
    Returns:
        montage (mne.channels.DigMontage): Montage object with the specified channels.
    """

    full_montage = make_standard_montage(montage_name)

    if subset_channels is None:
        return full_montage

    all_positions = full_montage.get_positions()['ch_pos']
    subset_positions = {ch.upper(): pos for ch, pos in all_positions.items() if ch.upper() in subset_channels}

    # Order the subset positions based on the order in subset_channels
    subset_positions = {ch: subset_positions[ch] for ch in subset_channels}

    return make_dig_montage(ch_pos=subset_positions, coord_frame='head')

def epoch_data(data, events, epoch, sfreq):
    """
    Epochs the continuous data based on event onsets and specified epoch window.
    Args:
        data: np.array of shape (n_channels, n_times)
        events: list of dictionaries with 'onset' keys
        epoch: tuple of (start_offset, end_offset) in seconds
    returns: tuple of (epoched_data, times)
             epoched_data shape: (n_events, n_channels, n_times_in_epoch)
    """
    epoched = []
    
    n_channels = data.shape[0]
    n_times_total = data.shape[1]
    
    offset_start = int(epoch[0] * sfreq)
    offset_end = int(epoch[1] * sfreq)
    epoch_length = offset_end - offset_start
    
    for i, event in enumerate(events):
        start = int(event['onset'] * sfreq) + offset_start
        end = int(event['onset'] * sfreq) + offset_end
        
        if start < 0 or end > n_times_total:
            print(f"Warning: Epoch for event {i} (onset: {event['onset']}s) falls outside data boundaries. Zero-padding applied.")
            
        current_epoch = np.zeros((n_channels, epoch_length))

        data_start = max(0, start)
        data_end = min(n_times_total, end)
        
        epoch_start = max(0, -start)
        epoch_end = epoch_start + (data_end - data_start)
        
        if data_start < data_end:
            current_epoch[:, epoch_start:epoch_end] = data[:, data_start:data_end]
            
        epoched.append(current_epoch)
        
    times = np.arange(offset_start, offset_end) / sfreq
    return np.array(epoched), times

