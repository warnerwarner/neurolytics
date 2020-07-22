import numpy as np

def spikes_to_sb(spike_times, resp_peaks):
    spikes_sb = []
    for i in range(len(resp_peaks) - 1):
        sniff_spikes = spike_times[(spike_times > resp_peaks[i]) & (spike_times < resp_peaks[i+1])]
        sniff_spikes = [(j - resp_peaks[i])/(resp_peaks[i+1] - resp_peaks[i]) for j in sniff_spikes]
        sniff_spikes = [j + i for j in sniff_spikes]
        spikes_sb.append(sniff_spikes)
    spikes_sb = np.hstack(spikes_sb)
    return spikes_sb


def starts_to_sb(trial_starts, trial_ends, resp_peaks):
    trials_sb = []
    trial_offs_sb = []
    for i in range(len(resp_peaks) - 1):
        sniff_trials = trial_starts[(trial_starts > resp_peaks[i]) & (trial_starts < resp_peaks[i+1])]
        sniff_trials = [(j - resp_peaks[i])/(resp_peaks[i+1] - resp_peaks[i]) for j in sniff_trials]
        sniff_trials = [j + i for j in sniff_trials]
        trials_sb.append(sniff_trials)

        sniff_ends = trial_ends[(trial_ends > resp_peaks[i]) & (trial_ends < resp_peaks[i+1])]
        sniff_trials = [(j - resp_peaks[i])/(resp_peaks[i+1] - resp_peaks[i]) for j in sniff_ends]
        sniff_trials = [j + i for j in sniff_trials]

        trial_offs_sb.append(sniff_trials)
    trials_sb = np.hstack(trials_sb)
    trial_offs_sb = np.hstack(trial_offs_sb)
    return trials_sb, trial_offs_sb