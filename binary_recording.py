from unit_recording import Unit_Recording

import spiking
import numpy as np
import openephys as oe
import os


class Binary_recording(Unit_Recording):

    def __init__(self, home_dir, channel_count, trial_names, *, trig_chan='100_ADC6.continuous', trial_length=0.12):
        Unit_Recording.__init__(self, home_dir, channel_count)
        self.trig_chan = trig_chan
        self.trial_names = trial_names
        self.trial_length = trial_length
        self.trial_starts = None
        self.trial_ends = None
        self.repeats = []
        self._find_trial_starts()

    def _set_trial_starts(self, trial_starts):
        self.trial_starts = np.array(trial_starts)

    def _set_trial_ends(self, trial_ends):
        self.trial_ends = trial_ends

    def get_trig_chan(self):
        return self.trig_chan

    def _find_trial_starts(self):
        print('Finding trial starts using trigger of %s' % self.get_trig_chan())
        trig = oe.loadContinuous2(os.path.join(self.get_home_dir(), self.get_trig_chan()))['data']
        fs = self.get_fs()
        trial_length = self.get_trial_length()*fs
        prev_trial = -trial_length
        trial_starts = []
        for index, val in enumerate(trig):
            if val > 2 and index - prev_trial > trial_length:
                trial_starts.append(index)
                prev_trial = index
        try:
            assert len(trial_starts) == len(self.get_trial_names())
        except (AssertionError):
            raise ValueError('Different lengths of trial starts (%d) and trial names (%d)' % (len(trial_starts),
                                                                                              len(self.get_trial_names())))

        trial_ends = [i+self.get_trial_length()*self.get_fs() for i in trial_starts]
        self._set_trial_ends(trial_ends)
        self._set_trial_starts(trial_starts)

    def get_trial_names(self):
        return self.trial_names

    def get_trial_starts(self):
        return self.trial_starts

    def get_trial_ends(self):
        return self.trial_ends

    def get_trial_length(self):
        return self.trial_length

    def get_unique_trial_names(self):
        return list(set(self.get_trial_names()))

    def get_unique_trial_starts(self, trial_name):
        return [j for i, j in zip(self.get_trial_names(), self.get_trial_starts()) if i == trial_name]

    def get_cluster_trial_response(self, trial_name, cluster_num, *, pre_trial_window=0.5, post_trial_window=0.5):
        cluster = self.get_cluster(cluster_num)
        cluster_spikes = cluster.get_spike_times()
        starts = self.get_unique_trial_starts(trial_name)
        cluster_trial_spikes = []
        sniff_basis = self.get_if_sniff_basis()
        for start in starts:
            if not sniff_basis:
                window_start = start - pre_trial_window*self.get_fs()
                window_end = start + (post_trial_window+self.get_trial_length())*self.get_fs()
            else:
                window_start = int(start - pre_trial_window)
                window_end = int(start + post_trial_window)
            trial_spikes = cluster_spikes[(cluster_spikes >= window_start) & (cluster_spikes <= window_end)]
            trial_spikes = [i - start for i in trial_spikes]
            cluster_trial_spikes.append(trial_spikes)
        return cluster_trial_spikes

    def get_all_trial_response(self, trial_name, *, only_good=True, pre_trial_window=0.5, post_trial_window=0.5):
        if only_good:
            clusters = self.get_good_clusters()
        else:
            clusters = self.get_non_noise_clusters()

        all_cluster_responses = [self.get_cluster_trial_response(trial_name,
                                                                 cluster.get_cluster_num(),
                                                                 pre_trial_window=pre_trial_window,
                                                                 post_trial_window=post_trial_window) for cluster in clusters]
        return all_cluster_responses
