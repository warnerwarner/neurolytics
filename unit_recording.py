
import os
import numpy as np
from spiking import Cluster
import csv
from recording import Recording, bandpass_data
import openephys as oe
from scipy.signal import find_peaks, resample
from tqdm import tqdm
import matplotlib.pyplot as plt
from plotter import Plotter

class Unit_Recording(Recording):
    def __init__(self, home_dir, channel_count, trial_length, *, fs=30000, dat_name='100_CHs.dat', 
                 resp_channel='100_ADC1.continuous', trig_chan='100_ADC6.continuous', conversion_factor=0.195,
                 sniff_basis=False, sniff_locked=False):
        """
        Unit recording - inherets from the Threshold_Recording object, the main object for a lot of different experiment
        types. Holds information both on the recording variables, and on the units isolated from the recording

        Args:
            home_dir (str): Home directory of the recording
            channel_count (int): Number of channels present in the recording
            trial_length (float): Length of the trials presented
            fs (int, optional): Sampling rate of the experiment. Defaults to 30000.
            dat_name (str, optional): Name of the dat file associated with the experiment. Defaults to '100_CHs.dat'.
            conversion_factor (float, optional): The conversion between bits and uV. Defaults to 0.195.
            sniff_basis (bool, optional): Is the recording in sniff basis. Defaults to False.
            sniff_locked (bool, optional): Is the recording sniff locked (?) - dont know where this is from. Defaults to False.
        """
        Recording.__init__(self, home_dir, channel_count, fs=fs, resp_channel=resp_channel, dat_name=dat_name, conversion_factor=conversion_factor)
        self.channel_map = np.load(os.path.join(home_dir, 'channel_map.npy'))
        self.channel_positions = np.load(os.path.join(home_dir, 'channel_positions.npy'))
        self.sniff_basis = sniff_basis
        self.trial_length = trial_length
        self.trig_chan =  trig_chan
        self.trial_names = []
        self.resp_peaks = self.find_respiration_peaks()
        self.trial_starts, self.trial_ends = self.find_trial_starts()
        self.resp_trace = self.find_respiration_trace()
        self.clusters = self.find_clusters()


    def find_clusters(self):
        """
        Finds clusters from files present in the home directory

        Returns:
            (list): List of the clusters
        """
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
            window = input('No sniff locked average, please enter (s) a window prior to trials to be used')
            sniff_spikes = False
        next(tsv_read)
        clusters = []
        all_sniff_locks = []
        cluster_nums = []
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
                              c_temp, max_chan)
            if sniff_spikes:
                c_sniff_spikes = sniff_locked_spikes[(sniff_cluster_nums == cluster_num)][0]
            else:
                c_sniff_spikes = self.get_sniff_lock_avg(cluster, window)
                all_sniff_locks.append(c_sniff_spikes)
                cluster_nums.append(cluster_num)
            cluster.sniff_lock_spikes = c_sniff_spikes
            clusters.append(cluster)
        if not sniff_spikes:
            cluster_sniff_lockeds = np.array(all_sniff_locks)
            cluster_nums = np.array(cluster_nums)
            np.save(os.path.join(self.home_dir, 'sniff_cluster_nums.npy'), cluster_nums)
            np.save(os.path.join(self.home_dir, 'sniff_locked_spikes.npy'), cluster_sniff_lockeds)

        return clusters

    def add_cluster(self, cluster):
        """
        Adds a cluster to the list of clusters in the Unit_Recording

        Args:
            cluster (Cluster): The cluster to add

        Raises:
            TypeError: Throws a TypeError if the cluster is not a Cluster object
        """
        try:
            assert isinstance(cluster, Cluster)
            self.clusters.append(cluster)
        except(AssertionError):
            raise TypeError('New cluster must be of a Cluster object')

    def get_cluster(self, cluster_num):
        """
        Gets a cluster from the list of clusters in the Unit_Recording

        Args:
            cluster_num (int): The number associated with the cluster

        Returns:
            (Cluster): The requested Cluster object
        """
        cluster = [i for i in self.clusters if i.cluster_num == cluster_num]
        return cluster[0]

    def get_good_clusters(self):
        """
        Gets all the 'good' clusters in an experiment

        Returns:
            list: List of all clusters labelled as 'good'
        """
        clusters = [i for i in self.clusters if i.label == 'good']
        return clusters

    def get_non_noise_clusters(self):
        """
        Gets all the 'good' and 'MUA' clusters in an experiment

        Returns:
            list: The clusters that aren't noise or MUA
        """
        clusters = [i for i in self.clusters if i.label == 'good' or i.label == 'mua']
        return clusters

    def find_trial_starts(self):
        trial_starts = None
        trial_ends = None
        if os.path.isfile(os.path.join(self.home_dir, 'trial_starts.npy')):
            print('Found file starts')
            trial_starts = np.load(os.path.join(self.home_dir, 'trial_starts.npy'))

        if os.path.isfile(os.path.join(self.home_dir, 'trial_ends.npy')):
            print('Found file ends')
            trial_ends = np.load(os.path.join(self.home_dir, 'trial_ends.npy'))

        if trial_starts is None or trial_ends is None:
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

            trial_ends = [i+trial_length for i in trial_starts]
            trial_ends = np.array(trial_ends)
            trial_starts = np.array(trial_starts)
            print('Saving starts and ends')
            np.save(os.path.join(self.home_dir, 'trial_starts.npy'), trial_starts)
            np.save(os.path.join(self.home_dir, 'trial_ends.npy'), trial_starts)
        return trial_starts, trial_ends    

    def find_respiration_peaks(self):
        resp_channel = self.resp_channel
        if os.path.isfile(os.path.join(self.home_dir, 'respiration_peaks.npy')):
            print('Respiration peaks found')
            respiration_peaks = np.load(os.path.join(self.home_dir, 'respiration_peaks.npy'))
        else:
            print('Finding respiration peaks from raw file')
            resp = oe.loadContinuous2(os.path.join(self.home_dir, resp_channel))['data']
            bp_data = bandpass_data(resp, highcut=100, lowcut=1)
            respiration_peaks = find_peaks(bp_data, height=np.std(bp_data), prominence=np.std(bp_data))[0]
            np.save(os.path.join(self.home_dir, 'respiration_peaks.npy'), respiration_peaks)
        return respiration_peaks

    def find_respiration_trace(self):
        resp_channel = self.resp_channel
        if os.path.isfile(os.path.join(self.home_dir, 'respiration_trace.npy')):
            print('Respiration trace found')
            respiration_trace = np.load(os.path.join(self.home_dir, 'respiration_trace.npy'))
        else:
            print('Finding respiration trace')
            resp = oe.loadContinuous2(os.path.join(self.home_dir, resp_channel))['data']
            resp_snippets = [resample(resp[i:j], 10000) for i, j in tqdm(zip(self.resp_peaks[:-1], self.resp_peaks[1:]))]
            respiration_trace= np.mean(resp_snippets, axis=0)
            np.save(os.path.join(self.home_dir, 'respiration_trace.npy'), respiration_trace)
        return respiration_trace

    def get_sniff_lock_avg(self, cluster, pre_trial_window):
        if isinstance(cluster, (int, float)):
            cluster = self.get_cluster(cluster)            
        trial_starts = self.trial_starts
        resp_peaks = self.resp_peaks
        spike_times = cluster.spike_times
        single_sniff_spikes = []
        for start in trial_starts:
            pre_trial_peaks = resp_peaks[(resp_peaks > start - pre_trial_window*self.fs) & (resp_peaks < start)]
            for i, j in zip(pre_trial_peaks[:-1], pre_trial_peaks[1:]):
                sniff_spikes = spike_times[(spike_times >= i) & (spike_times < j)]
                sniff_spikes = [(k-i)/(j-i) for k in sniff_spikes]
                single_sniff_spikes.append(sniff_spikes)
        single_sniff_spikes = np.hstack(single_sniff_spikes)
        return single_sniff_spikes

    def get_unique_trial_names(self):
        return list(set(self.trial_names))

    def get_unique_trial_starts(self, trial_name):
        try:
            assert trial_name in self.trial_names  # Don't know why this throws a warning, it does work
        except (AssertionError):
            raise ValueError('Trial name not in trial names')
        return [j for i, j in zip(self.trial_names, self.trial_starts) if i == trial_name]

    def get_cluster_trial_response(self, trial_name, cluster, *, pre_trial_window=None, post_trial_window=None, real_time=True):
        if isinstance(cluster, (int, float)):
            cluster = self.get_cluster(cluster)
        cluster_spikes = cluster.spike_times
        starts = self.get_unique_trial_starts(trial_name)
        cluster_trial_spikes = []
        sniff_basis = self.sniff_basis
        if pre_trial_window is None:
            pre_trial_window = 2*self.trial_length
        if post_trial_window is None:
            post_trial_window = 2*self.trial_length
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

    def get_all_binned_trial_response(self, trial_name, *, only_good=True, pre_trial_window=None, post_trial_window=None, bin_size=0.01, baselined=False):
        if only_good:
            clusters = self.get_good_clusters()
        else:
            clusters = self.get_non_noise_clusters()

        if baselined:
            all_cluster_responses = [self.get_binned_trial_response(trial_name,
                                                                    i.cluster_num,
                                                                    pre_trial_window=pre_trial_window,
                                                                    post_trial_window=post_trial_window,
                                                                    bin_size=bin_size,
                                                                    baselined=baselined)[1] for i in tqdm(clusters, leave=False)]
        else:
            all_cluster_responses = [self.get_binned_trial_response(trial_name,
                                                                    i.cluster_num,
                                                                    pre_trial_window=pre_trial_window,
                                                                    post_trial_window=post_trial_window,
                                                                    bin_size=bin_size,
                                                                    baselined=baselined)[1] for i in clusters]
        xs, _ = self.get_binned_trial_response(trial_name, clusters[0].cluster_num, pre_trial_window=pre_trial_window,
                                               post_trial_window=post_trial_window, bin_size=bin_size, baselined=False)
        return xs, all_cluster_responses




    def get_binned_trial_response(self, trial_name, cluster, *, pre_trial_window=None, post_trial_window=None, real_time=True, bin_size=0.01, baselined=True):
        if isinstance(cluster, (int, float)):
            cluster = self.get_cluster(cluster)

        if pre_trial_window is None:
            pre_trial_window = 2*self.trial_length
        if post_trial_window is None:
            post_trial_window = 2*self.trial_length

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
            true_y, true_x = np.histogram(trial_spikes, bins=np.arange(-1*pre_trial_window, self.trial_length+post_trial_window+bin_size, bin_size))
            
            if baselined:
                assert pre_trial_window > 0, 'Cannot apply baseline subtraction with no baseline'
                trial_peaks = resp_peaks[(resp_peaks > window_start-pre_trial_window*self.fs) & (resp_peaks < window_end+post_trial_window*self.fs)]
                trial_peaks = [(i-start)/self.fs for i in trial_peaks]
                peak_diffs = np.diff(trial_peaks)
                fin_peak = trial_peaks[:-1]
                faux_trial_spikes = np.outer(peak_diffs, sniff_locked_spikes) + np.array(fin_peak)[:, np.newaxis]
                faux_trial_spikes = np.hstack(faux_trial_spikes)
                fauxy, fauxx = np.histogram(faux_trial_spikes, bins=np.arange(-1*pre_trial_window, self.trial_length+post_trial_window+bin_size, bin_size))
                if sum(true_y[:int(pre_trial_window/bin_size)]) == 0:
                    cf = 0
                else:
                    cf = 1/(sum(fauxy[:int(pre_trial_window/bin_size)])/sum(true_y[:int(pre_trial_window/bin_size)]))
                baselined_resp = true_y - fauxy*cf
                true_y = baselined_resp
            cluster_trial_spikes.append(true_y/bin_size)
            #cluster_trial_spikes.append(trial_spikes)
        return true_x, cluster_trial_spikes
    
    def cluster_plot(self, cluster, **kwargs):
        plots = Plotter()
        plots.plot(cluster, *kwargs)

# if __name__ == '__main__':
#     cluster = Cluster([[1], [3], [4]], None, None, None, None, None, None, np.array([[10, 1, 5], [1, 3, 5]]))
#     eh = [1]
#     try:
#         assert isinstance(eh, Cluster)
#     except(AssertionError):
#         raise AssertionError('Need to add Cluster object')
