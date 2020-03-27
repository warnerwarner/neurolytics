
import os
import numpy as np
from spiking import Cluster
import csv
from threshold_recording import Threshold_Recording, bandpass_data
import openephys as oe
from scipy.signal import find_peaks


class Unit_Recording(Threshold_Recording):

    def __init__(self, home_dir, channel_count, trial_length, *, fs=30000, dat_name='100_CHs.dat', conversion_factor=0.195,  sniff_basis=False):
        Threshold_Recording.__init__(self, home_dir, channel_count, fs=fs, dat_name=dat_name, conversion_factor=conversion_factor)
        self.channel_map = np.load(os.path.join(home_dir, 'channel_map.npy'))
        self.channel_positions = np.load(os.path.join(home_dir, 'channel_positions.npy'))
        self.clusters = self._find_clusters()
        self.sniff_basis = sniff_basis
        self.trial_length = trial_length

    def _find_clusters(self):
        home_dir = self.home_dir
        spike_clusters = np.load(os.path.join(home_dir, 'spike_clusters.npy'))
        spike_templates = np.load(os.path.join(home_dir, 'spike_templates.npy'))
        spike_times = np.load(os.path.join(home_dir, 'spike_times.npy'))
        channel_map = self.channel_map
        templates = np.load(os.path.join(home_dir, 'templates.npy'))
        cluster_tsv = os.path.join(home_dir, 'cluster_group.tsv')
        tsv_read = csv.reader(open(cluster_tsv, 'r'), delimiter='\t')
        if os.path.isfile(os.path.join(home_dir, 'sniff_locked_spikes.npy')):
            print('Found sniff spikes')
            sniff_locked_spikes = np.load(os.path.join(home_dir, 'sniff_locked_spikes.npy'), allow_pickle=True)
            sniff_cluster_nums = np.load(os.path.join(home_dir, 'sniff_cluster_nums.npy'))
            sniff_spikes = True
        else:
            sniff_spikes = False
        next(tsv_read)
        clusters = []
        for cluster_row in tsv_read:
        # Find the cluster number and label
            cluster_num = int(cluster_row[0])
            c_label = cluster_row[1]
            if sniff_spikes:
                c_sniff_spikes = sniff_locked_spikes[(sniff_cluster_nums == cluster_num)][0]
            else:
                c_sniff_spikes = None

            # Find the times and templates
            c_times = spike_times[(spike_clusters == cluster_num)]
            c_temps_index = spike_templates[(spike_clusters == cluster_num)]
            c_temp_index = list(set(np.hstack(c_temps_index)))[0]
            c_temp = templates[c_temp_index]
            maxes = [max(abs(i)) for i in c_temp.T]
            max_chan = channel_map[np.argmax(maxes)]
            # Create a cluster and add to the ClusterSet
            cluster = Cluster(cluster_num, c_times, home_dir, c_label, c_temp_index,
                              c_temp, max_chan, sniff_lock_spikes=c_sniff_spikes)
            clusters.append(cluster)
        return clusters

    def _set_clusters(self, clusters):
        self.clusters = clusters

    def add_cluster(self, cluster):
        try:
            assert isinstance(cluster, Cluster)
            self.clusters.append(cluster)
        except(AssertionError):
            raise TypeError('New cluster must be of a Cluster object')

    def get_cluster(self, cluster_num):
        cluster = [i for i in self.clusters if i.cluster_num == cluster_num]
        return cluster[0]

    def get_good_clusters(self):
        clusters = [i for i in self.clusters if i.label == 'good']
        return clusters

    def get_non_noise_clusters(self):
        clusters = [i for i in self.clusters if i.label == 'good' or i.label == 'mua']
        return clusters

    def set(self,  pre_trial_window, *, resp_channel='100_ADC1.continuous'):
        print('Finding trial starts')
        self._find_trial_starts(self.trial_length)
        print('Finding respiration peaks')
        self._find_respiration_peaks(resp_channel)
        print('Finding sniff locked avgs')
        self._find_all_sniff_lock_avg(pre_trial_window)
        

    def _find_trial_starts(self, trial_length):
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
            trial_length = trial_length*fs
            prev_trial = -trial_length
            trial_starts = []
            for index, val in enumerate(trig):
                if val > 2 and index - prev_trial > trial_length:
                    trial_starts.append(index)
                    prev_trial = index

            trial_ends = [i+trial_length for i in trial_starts]
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
            print('Finding respiration peaks from raw file')
            resp = oe.loadContinuous2(os.path.join(self.home_dir, resp_channel))['data']
            bp_data = bandpass_data(resp, highcut=100, lowcut=1)
            respiration_peaks = find_peaks(bp_data, height=np.std(bp_data), prominence=np.std(bp_data))[0]
            np.save(os.path.join(self.home_dir, 'respiration_peaks.npy'), respiration_peaks)
        self.resp_peaks = respiration_peaks

    def _find_all_sniff_lock_avg(self, pre_trial_window):
        # if bp_resp:
        #     print('Bandpassing respiration...')
        #     bp_resp = bandpass_data(resp_trace, lowcut=1, highcut=100)
        # else:
        #     bp_resp = resp_trace
        if not os.path.isfile(os.path.join(self.home_dir, 'sniff_cluster_nums.npy')):
            cluster_nums = []
            cluster_sniff_lockeds = []
            resp_peaks = self.resp_peaks
            #resp_peaks = find_peaks(bp_resp, height=np.std(bp_resp), prominence=np.std(bp_resp))[0]
            np.save(os.path.join(self.home_dir, 'resp_peaks.npy'), resp_peaks)
            print("found peaks!")
            for cluster in self.clusters:
                print('Finding sniff locked average for cluster %d' % cluster.cluster_num)
                cluster_nums.append(cluster.cluster_num)
                cluster_sniff_lockeds.append(self.get_sniff_lock_avg(cluster.cluster_num, pre_trial_window, resp_peaks))

            cluster_sniff_lockeds = np.array(cluster_sniff_lockeds)
            cluster_nums = np.array(cluster_nums)
            np.save(os.path.join(self.home_dir, 'sniff_cluster_nums.npy'), cluster_nums)
            np.save(os.path.join(self.home_dir, 'sniff_locked_spikes.npy'), cluster_sniff_lockeds)
        else:
            print('Found sniff locked avgs')



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

if __name__ == '__main__':
    cluster = Cluster([[1], [3], [4]], None, None, None, None, None, None, np.array([[10, 1, 5], [1, 3, 5]]))
    eh = [1]
    try:
        assert isinstance(eh, Cluster)
    except(AssertionError):
        raise AssertionError('Need to add Cluster object')
