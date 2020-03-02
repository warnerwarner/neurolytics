from scipy.signal import find_peaks
from threshold_recording import bandpass_data
import numpy as np
from copy import deepcopy
from tqdm import tqdm



def make_sniff_basis(unit_recording, resp_trace):
    print('Bandpassing respiration')
    bp_resp = bandpass_data(resp_trace, highcut=100, lowcut=1)
    peaks = find_peaks(bp_resp, height=np.std(bp_resp), prominence=np.std(bp_resp))[0]

    unit_recording_sb = deepcopy(unit_recording)
    if hasattr(unit_recording, 'trial_starts'):
        print('Trial starts found!')
        print('Converting to sniff basis')
        trial_starts = unit_recording.trial_starts
        if trial_starts is not None:
            trials_sb = []
            for i in range(len(peaks) - 1):
                sniff_trials = trial_starts[(trial_starts > peaks[i]) & (trial_starts < peaks[i+1])]
                sniff_trials = [(j - peaks[i])/(peaks[i+1] - peaks[i]) for j in sniff_trials]
                sniff_trials = [j + i for j in sniff_trials]
                trials_sb.append(sniff_trials)
            trials_sb = np.hstack(trials_sb)
            unit_recording_sb.trial_starts = trials_sb
    else:
        print('No trial starts found')

    clusters = unit_recording.get_all_clusters()
    clusters_sb = []
    print('Converting clusters to sniff basis')
    for cluster in tqdm(clusters):
        #print('Cluster %d' % cluster.get_cluster_num())
        cluster_sb = make_cluster_sniff_basis(cluster, peaks)
        clusters_sb.append(cluster_sb)
    unit_recording_sb._set_clusters(clusters_sb)

    return unit_recording_sb


def make_cluster_sniff_basis(cluster, peaks):
    spike_times = cluster.get_spike_times()
    spike_times = np.hstack(spike_times)
    spikes_sb = []
    for i in range(len(peaks) - 1):

        sniff_spikes = spike_times[(spike_times > peaks[i]) & (spike_times < peaks[i+1])]
        sniff_spikes = [(j - peaks[i])/(peaks[i+1] - peaks[i]) for j in sniff_spikes]
        sniff_spikes = [j + i for j in sniff_spikes]
        spikes_sb.append(sniff_spikes)
    spikes_sb = np.hstack(spikes_sb)
    clus_sb = deepcopy(cluster)
    clus_sb._set_spike_times(spikes_sb)
    return clus_sb
