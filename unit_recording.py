
import os
import numpy as np
from spiking import Cluster
import csv
from threshold_recording import Threshold_Recording

class Unit_Recording(Threshold_Recording):

    def __init__(self, home_dir, channel_count, *, fs=30000, dat_name='100_CHs.dat', conversion_factor=0.195, sniff_basis=False):
        Threshold_Recording.__init__(self, home_dir, channel_count, fs=fs, dat_name=dat_name, conversion_factor=conversion_factor)
        self.channel_map = np.load(os.path.join(home_dir, 'channel_map.npy'))
        self.channel_positions = np.load(os.path.join(home_dir, 'channel_positions.npy'))
        self.clusters = self._find_clusters()
        self.sniff_basis = sniff_basis

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
                c_sniff_spikes = sniff_locked_spikes[(sniff_cluster_nums == cluster_num)]
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
                              c_temp, max_chan, sniff_lock_spikes=c_sniff_spikes[0])
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



if __name__ == '__main__':
    cluster = Cluster([[1], [3], [4]], None, None, None, None, None, None, np.array([[10, 1, 5], [1, 3, 5]]))
    eh = [1]
    try:
        assert isinstance(eh, Cluster)
    except(AssertionError):
        raise AssertionError('Need to add Cluster object')
