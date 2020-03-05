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

    def set(self, *, resp_channel='100_ADC1.continous'):
        print('Finding trial names...')
        self._extract_trial_names()
        print('Finding trial starts...')
        self._find_trial_starts()
        print('Finding respiration peaks...')
        self._find_respiration_peaks(resp_channel=resp_channel)

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


    def _find_trial_starts(self):
        if os.path.isfile(os.path.join(self.home_dir, 'trial_starts.npy')):
            print('Found file starts')
            self.trial_starts = np.load(os.path.join(self.home_dir, 'trial_starts.npy'))
        if os.path.isfile(os.path.join(self.home_dir, 'trial_ends.npy')):
            print('Found file ends')
            self.trial_ends = np.load(os.path.join(self.home_dir, 'trial_ends.npy'))

        if self.trial_starts is None or self.trial_ends is None:
            print('No starts or ends found')
            print('Finding trial starts using trigger of %s' % self.trig_chan)
            trig = oe.loadContinuous2(os.path.join(self.home_dir, self.trig_chan))['data']
            fs = self.fs
            trial_length = self.trial_length*fs
            prev_trial = -trial_length
            trial_starts = []
            for index, val in enumerate(trig):
                if val > 2 and index - prev_trial > trial_length:
                    trial_starts.append(index)
                    prev_trial = index
            try:
                assert len(trial_starts) == len(self.trial_names)
                print('Found same number of trial starts as trial names')
            except (AssertionError):
                raise ValueError('Different lengths of trial starts (%d) and trial names (%d)' % (len(trial_starts),
                                                                                                  len(self.trial_names)))

            trial_ends = [i+self.trial_length*self.fs for i in trial_starts]
            self.trial_ends = np.array(trial_ends)
            self.trial_starts = np.array(trial_starts)
            print('Saving starts and ends')
            np.save(os.path.join(self.home_dir, 'trial_starts.npy'), trial_starts)
            np.save(os.path.join(self.home_dir, 'trial_ends.npy'), trial_starts)

    def _find_respiration_peaks(self, *, resp_channel='100_ADC1.continuous'):
        if os.path.isfile(os.path.join(self.home_dir, 'respiration_peaks.npy')):
            print('Respiration peaks found')
            respiration_peaks = np.load(os.path.join(self.home_dir, 'respiration_peaks.npy'))
        else:
            print('Finding respiration peaks')
            resp = oe.loadContinuous2(os.path.join(self.home_dir, resp_channel))['data']
            bp_data = bandpass_data(resp, highcut=100, lowcut=1)
            respiration_peaks = find_peaks(bp_data, height=np.std(bp_data), prominence=np.std(bp_data))[0]
            np.save(os.path.join(self.home_dir, 'respiration_peaks.npy'), respiration_peaks)
        self.resp_peaks = respiration_peaks

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

    def get_all_sniff_lock_avg(self, pre_trial_window, resp_trace, bp_resp=True):
        if bp_resp:
            print('Bandpassing respiration...')
            bp_resp = bandpass_data(resp_trace, lowcut=1, highcut=100)
        else:
            bp_resp = resp_trace

        cluster_nums = []
        cluster_sniff_lockeds = []
        resp_peaks = find_peaks(bp_resp, height=np.std(bp_resp), prominence=np.std(bp_resp))[0]
        np.save(os.path.join(home_dir, 'resp_peaks.npy'), resp_peaks)
        print("found peaks!")
        for cluster in self.clusters:
            print('Finding sniff locked average for cluster %d' % cluster.cluster_num)
            cluster_nums.append(cluster.cluster_num)
            cluster_sniff_lockeds.append(self.get_sniff_lock_avg(cluster.cluster_num, pre_trial_window, resp_peaks))

        cluster_sniff_lockeds = np.array(cluster_sniff_lockeds)
        cluster_nums = np.array(cluster_nums)
        np.save(os.path.join(self.home_dir, 'sniff_cluster_nums.npy'), cluster_nums)
        np.save(os.path.join(self.home_dir, 'sniff_locked_spikes.npy'), cluster_sniff_lockeds)




    def get_sniff_lock_avg(self, cluster_num, pre_trial_window, resp_peaks,):

        trial_starts = self.trial_starts
        spike_times = self.get_cluster(cluster_num).spike_times
        single_sniff_spikes = []
        for start in trial_starts:
            pre_trial_peaks = resp_peaks[(resp_peaks > start - pre_trial_window*self.fs) & (resp_peaks < start)]
            for i, j in zip(pre_trial_peaks[:-1], pre_trial_peaks[1:]):
                sniff_spikes = spike_times[(spike_times >= i) & (spike_times < j)]
                sniff_spikes = [(k-i)/(j-i) for k in sniff_spikes]
                single_sniff_spikes.append(sniff_spikes)
        single_sniff_spikes = np.hstack(single_sniff_spikes)
        return single_sniff_spikes


    def get_baslined_trial_response(self, trial_name, cluster_num, *, pre_trial_window=0.5, post_trial_window=0.5, real_time=True):
        '''
        Unfinished!
        '''

        cluster = self.get_cluster(cluster_num)

        assert cluster.sniff_lock_spikes is not None, "No sniff locked spikes for cluster"
        assert self.resp_peaks is not None, "No respiration peaks"
        assert not self.sniff_basis, 'Sniff basis baseline not implemented as yet'

        cluster_spikes = cluster.spike_times
        sniff_locked_spikes = cluster.sniff_lock_spikes
        starts = self.get_unique_trial_starts(trial_name)
        cluster_trial_spikes = []
        resp_peaks = self.resp_peaks
        base_hist = np.histogram(cluster.sniff_lock_spikes, bins=np.arange(0, 1.01, 0.01))
        for start in starts:

            window_start = int(start - pre_trial_window*self.fs)
            window_end = int(start + post_trial_window*self.fs)
            trial_peaks = resp_peaks[(resp_peaks > window_start) & (resp_peaks < window_end)]
            trial_peaks = trial_peaks[(i-start)/self.fs for i in trial_peaks]
            pre_peaks = resp_peaks[(resp_peaks < start) & (resp_peaks > window_start)]
            pre_spikes = cluster_spikes[(cluster_spikes < start) & (cluster_spikes > window_start)]
            cf = len(pre_spikes)/(len(pre_peaks) - 1)
            faux_trial_spikes = []
            for i, j in zip(trial_peaks[:-1], trial_peaks[1:]):
                faux_sniff_spikes = [k*(j-i) + i for k in sniff_locked_spikes]
                faux_trial_spikes.append(faux_sniff_spikes)
            faux_trial_spikes = np.hstack(faux_trial_spikes)
            fauxy, fauxx = np.histogram(faux_trial_spikes, bins=np.arange(-pre_trial_window, self.trial_length+post_trial_window, 0.01))
            trial_spikes = cluster_spikes[(cluster_spikes >= window_start) & (cluster_spikes <= window_end)]
            trial_spikes = [i - start for i in trial_spikes]
            if real_time:
                trial_spikes = [i/self.fs for i in trial_spikes]
            cluster_trial_spikes.append(trial_spikes)
        return cluster_trial_spikes
