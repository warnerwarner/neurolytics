from spiking import Cluster
from unit_recording import Unit_Recording
import matplotlib.pyplot as plt
import os
import numpy as np
from tqdm import tqdm
from scipy.signal import resample

class Plotter():

    def __init__(self):
        None
    def plot(self, objs):
        if isinstance(objs, Cluster):
            self.cluster_plots(objs)
        else:
            print('Only cluster plotting so far, soz')


    def cluster_plots(self, cluster, *, sniff_lock=True, fr_bin=60, spikes_shown=100, pre_spike_waveform=30, post_spike_waveform=60, sniff_voltage=False, **kwargs):
        '''
        Constucts a 2 x 2 set of figures that can be used to decribe the attributes of a given cluster. 

        Arguments:
        cluster - The cluster to use for construction

        Optional arguments:
        sniff_lock - I dont know what this does... but I'll leave it in for now
        fr_bin - The bin size (in seconds) for the firing rate plot
        spikes_shown - The number of example spikes to plot in the waveform plot
        pre_spike_waveform - The number of samples to take prior to the spike time
        post_spike_waveform - The number of samples to take post to the spike time        
        '''

        # If the cluster is a cluster then uses it, otherwise finds the cluster from the number
        fig, ax = plt.subplots(2, 2, **kwargs) # Make some plots 
        # The waveform plot
        print('Getting waveforms')
        self.waveform_plot(cluster, ax=ax[0, 0], spikes_shown=spikes_shown, pre_window=pre_spike_waveform, post_window=post_spike_waveform)
        
        # The firing rate
        print('Getting firing rate')
        self.firing_rate_plot(cluster, ax=ax[1, 0])

        # The autocorrelogram
        print('Getting autocorr')
        self.autocorrelogram_plot(cluster, ax=ax[0, 1])

        # The phase
        print('Getting phase')
        self.phase_plot(cluster, ax=ax[1, 1], sniff_voltage=sniff_voltage)

        return fig, ax




    def waveform_plot(self, cluster, *, ax = None, spikes_shown=100, pre_window=1, post_window=2, zeroing='first', channel='max'):
        '''
        Constructs a unit waveform plot, calls the get_unit_waveforms to find the waveforms

        Arguments:
        cluster - The cluster to construct the waveforms from
        Optional arguments:
        ax - The axis to plot on, if none will constuct their own
        spikes_shown - Number of spikes to 
        '''
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)

        waveforms = cluster.get_waveforms(pre_window=pre_window*30, post_window=post_window*30, zeroing=zeroing)
        if channel != 'max':
            waveforms = waveforms[:, :, channel]
        else:
            waveforms = waveforms[:, :, cluster.max_chan]
        xs = np.arange(-pre_window, post_window, waveforms.shape[1])
        if len(waveforms) <= spikes_shown:
            print('Too many spikes requested, reducing %d-->%d' % (spikes_shown, len(waveforms)))
            spikes_shown = len(waveforms)
        for i in range(spikes_shown):
            snip = waveforms[int(i*len(waveforms)/spikes_shown)]
            ax.plot(xs, snip*0.195, color='lightgray')
        ax.plot(xs, np.mean(waveforms, axis=0)*0.195, color='r')
        ax.set_ylabel('Voltage ($\mu$V)')
        ax.set_xlabel('Time (ms)')
        ax.set_xlim(-1, 2)

    def autocorrelogram_plot(self, cluster, *, ax=None, bin_size=1, window_size=50):
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        spike_times = cluster.spike_times

        diffs = []
        for i in tqdm(range(len(spike_times)), leave=False):
            diff = spike_times[i+1:] - spike_times[i]
            diffs.append(diff[diff <= bin_size*30*window_size])
        diffs = np.concatenate(diffs)
        diffs = np.concatenate([diffs, diffs * - 1])

        hists = np.histogram(diffs, bins=np.arange(-window_size*30, 30*(window_size+bin_size), bin_size*30))
        bar_lim = window_size - bin_size/2
        ax.bar(hists[1][:-1]/30, hists[0]/len(spike_times), width=bin_size, align='edge')
        ax.set_xlabel('Inter-spike interval (ms)')
        ax.set_ylabel('Normalised spike probability')
        ax.set_xlim(-bar_lim, bar_lim)

    def firing_rate_plot(self, cluster, *, ax=None, bin_size=60, min_base=True):
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)

        rec_length = cluster.get_recording_length()
        xs, fr = cluster.get_firing_rate(rec_length, bin_size=bin_size)
        if min_base:
            xs = xs/60
            xlim = rec_length/60
            basis = 'mins'
        else:
            xlim = rec_length
            basis = 's'
        ax.plot(xs, fr)
        ax.set_ylabel('Firing rate (Hz)')
        ax.set_xlabel('Time (%s)' % basis)
        ax.set_xlim(0, xlim)

    def phase_plot(self, cluster, *, ax=None, bin_num=100, sniff_voltage=False):
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        
        xs = np.arange(0, 1, 1/bin_num)

        if os.path.isfile(os.path.join(cluster.recording_dir, 'respiration_trace.npy')):
            resp_trace = np.load(os.path.join(cluster.recording_dir, 'respiration_trace.npy'))
            ax2 = ax.twinx()
            if not sniff_voltage:
                ax2.set_yticks([])
            else:
                ax2.set_ylabel('Respiration signal (V)')

            ax2.plot(xs, resample(resp_trace, bin_num), color='r')

        ax.plot(xs, np.histogram(cluster.sniff_lock_spikes, bins=np.arange(0, 1 + 1/bin_num, 1/bin_num))[0]/len(cluster.sniff_lock_spikes))
        ax.set_ylabel('Normalised spike probability')
        ax.set_xlabel('Sniff cycle phase')
        ax.set_xlim(0, 1)
        ax.set_ylim(0)
        
