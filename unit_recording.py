
import os
import numpy as np
from spiking import Cluster
import csv
from threshold_recording import Threshold_Recording

class Unit_Recording(Threshold_Recording):

    def __init__(self, home_dir, channel_count, *, fs=30000, dat_name='100_CHs.dat', conversion_factor=0.195):
        Threshold_Recording.__init__(self, home_dir, channel_count, fs=fs, dat_name=dat_name, conversion_factor=conversion_factor)
        self.channel_map = np.load(os.path.join(home_dir, 'channel_map.npy'))
        self.channel_positions = np.load(os.path.join(home_dir, 'channel_positions.npy'))
        self.clusters = self._find_clusters()

    def _find_clusters(self):
        home_dir = self.get_home_dir()
        spike_clusters = np.load(os.path.join(home_dir, 'spike_clusters.npy'))
        spike_templates = np.load(os.path.join(home_dir, 'spike_templates.npy'))
        spike_times = np.load(os.path.join(home_dir, 'spike_times.npy'))
        channel_map = self.get_channel_map()
        templates = np.load(os.path.join(home_dir, 'templates.npy'))
        cluster_tsv = os.path.join(home_dir, 'cluster_group.tsv')
        tsv_read = csv.reader(open(cluster_tsv, 'r'), delimiter='\t')
        next(tsv_read)
        clusters = []
        for cluster_row in tsv_read:
        # Find the cluster number and label
            cluster_num = int(cluster_row[0])
            c_label = cluster_row[1]

            # Find the times and templates
            c_times = spike_times[(spike_clusters == cluster_num)]
            c_temps_index = spike_templates[(spike_clusters == cluster_num)]
            c_temp_index = list(set(np.hstack(c_temps_index)))[0]
            c_temp = templates[c_temp_index]
            maxes = [max(abs(i)) for i in c_temp.T]
            max_chan = channel_map[np.argmax(maxes)]
            # Create a cluster and add to the ClusterSet
            cluster = Cluster(cluster_num, c_times, home_dir, c_label, c_temp_index, c_temp, max_chan)
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

    def get_clusters(self):
        return self.clusters

    def get_all_clusters(self):
        return self.clusters

    def get_cluster(self, cluster_num):
        cluster = [i for i in self.get_all_clusters() if i.get_cluster_num() == cluster_num]
        return cluster[0]

    def get_good_clusters(self):
        clusters = [i for i in self.get_all_clusters() if i.get_label() == 'good']
        return clusters

    def get_non_noise_clusters(self):
        clusters = [i for i in self.get_all_clusters() if i.get_label() == 'good' or i.get_label() == 'mua']
        return clusters

    def get_channel_map(self):
        return self.channel_map

if __name__ == '__main__':
    cluster = Cluster([[1], [3], [4]], None, None, None, None, None, None, np.array([[10, 1, 5], [1, 3, 5]]))
    eh = [1]
    try:
        assert isinstance(eh, Cluster)
    except(AssertionError):
        raise AssertionError('Need to add Cluster object')
