from recording import Recording
import os
import numpy as np
from cluster import Cluster
import csv

class Unit_Recording(Recording):

    def __init__(self, home_dir, channel_count, fs):
        Recording.__init__(self, home_dir, channel_count, fs)
        self.clusters = self._set_clusters()

    def _set_clusters(self):
        home_dir = self.get_home_dir()
        channel_map = np.load(os.path.join(home_dir, 'channel_map.npy'))
        channel_positions = np.load(os.path.join(home_dir, 'channel_positions.npy'))
        spike_clusters = np.load(os.path.join(home_dir, 'spike_clusters.npy'))
        spike_templates = np.load(os.path.join(home_dir, 'spike_templates.npy'))
        spike_times = np.load(os.path.join(home_dir, 'spike_times.npy'))
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

            # Create a cluster and add to the ClusterSet
            cluster = Cluster(channel_map, channel_positions, cluster_num, c_times, home_dir, c_label, c_temp_index, c_temp)
            clusters.append(cluster)
        return clusters

    def add_cluster(self, cluster):
        try:
            assert isinstance(cluster, Cluster)
            self.clusters.append(cluster)
        except(AssertionError):
            raise TypeError('New cluster must be of a Cluster object')

    def get_clusters(self):
        return self.clusters

    def get_cluster(self, cluster_num):
        cluster = [i for i in self.get_all_clusters() if i.get_cluster_num() == cluster_num]
        return cluster[0]

    def get_good_clusters(self):
        clusters = [i for i in self.get_all_clusters() if i.get_label() == 'good']
        return clusters



cluster = Cluster([[1], [3], [4]], None, None, None, None, None, None, np.array([[10, 1, 5], [1, 3, 5]]))
eh = [1]
try:
    assert isinstance(eh, Cluster)
except(AssertionError):
    raise AssertionError('Need to add Cluster object')
