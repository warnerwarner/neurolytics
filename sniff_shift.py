"""
A couple of functions to translate recordings from time to sniff basis (counting sniffs not time)
"""
import numpy as np
from tqdm import tqdm

def spikes_to_sb(spike_times, resp_peaks):
    """
    Translatate spike times into sniff basis

    Args:
        spike_times (array): Times of spikes
        resp_peaks (array): Times of exhalation peaks

    Returns:
        spikes_sb (array): Spike times in sniff basis
    """
    spikes_sb = []
    print('Finding spikes in sb')
    for i in tqdm(range(len(resp_peaks) - 1), leave=False):
        sniff_spikes = spike_times[(spike_times >= resp_peaks[i]) & (spike_times < resp_peaks[i+1])]
        sniff_spikes = [(j - resp_peaks[i])/(resp_peaks[i+1] - resp_peaks[i]) for j in sniff_spikes]
        sniff_spikes = [j + i for j in sniff_spikes]
        spikes_sb.append(sniff_spikes)
    spikes_sb = np.hstack(spikes_sb)
    return spikes_sb


def starts_to_sb(trial_starts, trial_ends, resp_peaks):
    """
    Shifts trial starts into sniff basis

    Args:
        trial_starts (array): Times of trial starts
        trial_ends (array): Times of trial ends
        resp_peaks (array): Times of exhulation peaks

    Returns:
        trials_sb (array): Trial starts in sniff basis
        trial_offs_sb (array): Trial ends in sniff basis
    """
    trials_sb = []
    trial_offs_sb = []
    print('Finding trial starts in sb')
    for i in tqdm(range(len(resp_peaks) - 1), leave=False):
        sniff_trials = trial_starts[(trial_starts >= resp_peaks[i]) & (trial_starts < resp_peaks[i+1])]
        sniff_trials = [(j - resp_peaks[i])/(resp_peaks[i+1] - resp_peaks[i]) for j in sniff_trials]
        sniff_trials = [j + i for j in sniff_trials]
        trials_sb.append(sniff_trials)

        sniff_ends = trial_ends[(trial_ends >= resp_peaks[i]) & (trial_ends < resp_peaks[i+1])]
        sniff_ends = [(j - resp_peaks[i])/(resp_peaks[i+1] - resp_peaks[i]) for j in sniff_ends]
        sniff_ends = [j + i for j in sniff_ends]

        trial_offs_sb.append(sniff_ends)
    trials_sb = np.hstack(trials_sb)
    trial_offs_sb = np.hstack(trial_offs_sb)
    print(trials_sb[:10])
    print(trial_offs_sb[:10])
    return trials_sb, trial_offs_sb