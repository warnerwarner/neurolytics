from unit_recording import Unit_Recording
from scipy.signal import find_peaks
import spiking
import numpy as np
import openephys as oe
from threshold_recording import bandpass_data
import os


class Binary_recording(Unit_Recording):

    def __init__(self, home_dir, channel_count, trial_names, *, trig_chan='100_ADC6.continuous', trial_length=0.12):
        Unit_Recording.__init__(self, home_dir, channel_count)
        self.trig_chan = trig_chan
        self.trial_names = trial_names
        self.trial_length = trial_length
        self.trial_starts = None
        self.trial_ends = None
        self.resp_peaks = None

    def set(self, *, resp_channel='100_ADC1.continuous', sniff_avg=True):
        print('Finding trial names...')
        self._extract_trial_names()
        print('Finding trial starts...')
        self._find_trial_starts()
        print('Finding respiration peaks...')
        self._find_respiration_peaks(resp_channel=resp_channel)
        if sniff_avg==True:
            self._find_sniff_avg(1)

    def _extract_trial_names(self):
        trial_names = self.trial_names
        if isinstance(trial_names, str):
            print('Extracting trial names')
            with open(trial_names, 'r') as f:
                lines = f.readlines()
                trial_lines = [i[:-1] for i in lines if ':' not in i]
                self.trial_names = trial_lines
                print('Found %d trials in trial name file' % len(trial_lines))
        else:
            print('trials are already trials')
        #print(len(trial_lines))


    def get_unique_trial_names(self):
        return list(set(self.trial_names))

    def get_unique_trial_starts(self, trial_name):
        try:
            assert trial_name in self.trial_names
        except (AssertionError):
            raise ValueError('Trial name not in trial names')
        return [j for i, j in zip(self.trial_names, self.trial_starts) if i == trial_name]

    def get_cluster_trial_response(self, trial_name, cluster_num, *, pre_trial_window=0.5, post_trial_window=0.5, real_time=True):
        cluster = self.get_cluster(cluster_num)
        cluster_spikes = cluster.spike_times
        starts = self.get_unique_trial_starts(trial_name)
        cluster_trial_spikes = []
        sniff_basis = self.sniff_basis

        for start in starts:
            if not sniff_basis:
                window_start = start - pre_trial_window*self.fs
                window_end = start + (post_trial_window+self.trial_length)*self.fs
            else:
                window_start = int(start - pre_trial_window)
                window_end = int(start + post_trial_window)
            trial_spikes = cluster_spikes[(cluster_spikes >= window_start) & (cluster_spikes <= window_end)]
            trial_spikes = [i - start for i in trial_spikes]
            if real_time:
                trial_spikes = [i/self.fs for i in trial_spikes]
            cluster_trial_spikes.append(trial_spikes)
        return cluster_trial_spikes

    def get_all_trial_response(self, trial_name, *, only_good=True, pre_trial_window=0.5, post_trial_window=0.5):
        if only_good:
            clusters = self.get_good_clusters()
        else:
            clusters = self.get_non_noise_clusters()

        all_cluster_responses = [self.get_cluster_trial_response(trial_name,
                                                                 cluster.cluster_num,
                                                                 pre_trial_window=pre_trial_window,
                                                                 post_trial_window=post_trial_window) for cluster in clusters]
        return all_cluster_responses



    def get_binned_trial_response(self, trial_name, cluster_num, *, pre_trial_window=0.5, post_trial_window=0.5, real_time=True, bin_size=0.01, baselined=True):

        cluster = self.get_cluster(cluster_num)

        if baselined:
            assert cluster.sniff_lock_spikes is not None, "No sniff locked spikes for cluster"
            assert self.resp_peaks is not None, "No respiration peaks"
            assert not self.sniff_basis, 'Sniff basis baseline not implemented as yet'

        cluster_spikes = cluster.spike_times
        sniff_locked_spikes = cluster.sniff_lock_spikes
        starts = self.get_unique_trial_starts(trial_name)
        cluster_trial_spikes = []
        resp_peaks = self.resp_peaks
        #base_hist = np.histogram(cluster.sniff_lock_spikes, bins=np.arange(0, 1.01, 0.01))
        for start in starts:

            window_start = int(start - pre_trial_window*self.fs)
            window_end = int(start + (self.trial_length + post_trial_window)*self.fs)
            trial_spikes = cluster_spikes[(cluster_spikes >= window_start) & (cluster_spikes <= window_end)]
            trial_spikes = [i - start for i in trial_spikes]
            if real_time:
                trial_spikes = [i/self.fs for i in trial_spikes]
            true_y, true_x = np.histogram(trial_spikes, bins=np.arange(-pre_trial_window, self.trial_length+post_trial_window, bin_size))

            if baselined:
                trial_peaks = resp_peaks[(resp_peaks > window_start-pre_trial_window*self.fs) & (resp_peaks < window_end+post_trial_window*self.fs)]
                trial_peaks = [(i-start)/self.fs for i in trial_peaks]
                pre_peaks = resp_peaks[(resp_peaks < start) & (resp_peaks > window_start)]
                pre_spikes = cluster_spikes[(cluster_spikes < start) & (cluster_spikes > window_start)]
                faux_trial_spikes = []
                for i, j in zip(trial_peaks[:-1], trial_peaks[1:]):
                    faux_sniff_spikes = [k*(j-i) + i for k in sniff_locked_spikes]
                    faux_trial_spikes.append(faux_sniff_spikes)
                faux_trial_spikes = np.hstack(faux_trial_spikes)
                fauxy, fauxx = np.histogram(faux_trial_spikes, bins=np.arange(-pre_trial_window, self.trial_length+post_trial_window, bin_size))
                if sum(true_y[:int(pre_trial_window/bin_size)]) == 0:
                    cf = 0
                else:
                    cf = 1/(sum(fauxy[:int(pre_trial_window/bin_size)])/sum(true_y[:int(pre_trial_window/bin_size)]))
                baselined_resp = true_y - fauxy*cf
                true_y = baselined_resp
            cluster_trial_spikes.append(true_y/bin_size)
            #cluster_trial_spikes.append(trial_spikes)
        return true_x, cluster_trial_spikes
