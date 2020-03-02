import numpy as np
import matplotlib.pyplot as plt

class Spiking():
    '''
    Superclass for any object that contains spikes

    Hold spike times, and recording directory of spiking object
    '''
    def __init__(self, times, recording_dir):
        self.times = times
        self.recording_dir = recording_dir
        self.amplitudes = None

    def get_spike_times(self):
        return self.times

    def get_recording_dir(self):
        return self.recording_dir

    def _set_amplitude(self, amplitudes):
        self.amplitudes = amplitudes

    def get_amplitudes(self):
        return self.amplitudes

    def get_firing_rate(self, exp_length, *, bin_size=1):
        ys, xs = np.histogram(self.get_spike_times(), bins=np.arange(0, exp_length, 1))
        return xs[:-1], ys/bin_size

    def plot_firing_rate(self, exp_length, *, ax=None, bin_size=1):
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        xs, ys = self.get_firing_rate(exp_length, bin_size=bin_size)
        ax.plot(xs, ys)

    def _set_spike_times(self, spike_times):
        self.times = spike_times

class Cluster(Spiking):
    '''
    Holds information about Kilosorted clusters, a subclass of Spiking

    Information includes cluster number, spiking times, the recording dir, cluster label and templates
    '''

    def __init__(self, cluster_num, times, recording_dir, label, template_ind, template, max_chan):
        Spiking.__init__(self, times, recording_dir)
        self.cluster_num = cluster_num
        self.label = label
        self.template_ind = template_ind
        self.template = template
        self.max_chan = max_chan

    def get_label(self):
        return self.label

    def get_cluster_num(self):
        return self.cluster_num

    def get_max_chan(self):
        return self.max_chan



class ThresholdCrossings(Spiking):
    '''
    Represent detected threshold crossings on a single channel

    Contains the spike times, recording directory, the channel number in the orignal recording, and the threshold used
    '''

    def __init__(self, times, recording_dir, channel_num, threshold):
        Spiking.__init__(self, times, recording_dir)
        self.channel_num = channel_num
        self.threshold = threshold

    def get_channel_num(self):
        return self.channel_num
