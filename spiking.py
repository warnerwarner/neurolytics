import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

class Spiking():
    '''
    Superclass for any object that contains spikes

    Hold spike times, and recording directory of spiking object
    '''
    def __init__(self, spike_times, recording_dir, *, dat_name='100_CHs.dat'):
        """
        Args:
            spike_times (array of ints): Times of spikes. 
            recording_dir (str): Directory of the recording the Spiking is from
            dat_name (str, optional): Name of the dat file associated with the recording. Defaults to '100_CHs.dat'.
        """
        self.spike_times = spike_times
        self.recording_dir = recording_dir
        self.amplitudes = None
        self.dat_name = dat_name

    def get_firing_rate(self, exp_length, *, bin_size=1, fs=30000):
        """
        Finds the firing rate

        Args:
            exp_length (int): Length of the experiment (in samples)
            bin_size (int, optional): Size of the bin to use (in seconds). Defaults to 1.
            fs (int, optional): Sampling rate of the recording. Defaults to 30000.

        Returns:
            xs (array of floats): The time series associated with the firing values
            ys (array of floats): Instanteous firing rate
        """
        ys, xs = np.histogram(self.spike_times/fs, bins=np.arange(0, exp_length, bin_size))
        return xs[:-1], ys/bin_size

    def plot_firing_rate(self, exp_length, *, ax=None, bin_size=1):
        """
        Depreciated. Plot the firing rate (don't use)

        Args:
            exp_length (int): Length of experiment (in samples)
            ax (Axis, optional): Axis to plot on. Defaults to None.
            bin_size (int, optional): Size of bin for histogram. Defaults to 1.
        """
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        xs, ys = self.get_firing_rate(exp_length, bin_size=bin_size)
        ax.plot(xs, ys)
    
    def get_waveforms(self, *, data=None, chan_count=32, pre_window=30, post_window=60, zeroing='first'):
        """
        Gets all the waveforms from a raw recording, very fast (at least on CAMP)

        Args:
        cluster (int or Cluster): Which cluster to find the spikes from, can be a Cluster object, or a cluster number
        pre_window (float, optional): The window size prior to the spike time to include, in ms. Defaults to 1
        post_window (float, optional): The window size post to the spike time to include, in ms. Defaults to 2
        zeroing (str, optional): Method for removing offset, can be. Defaults to 'first':
            - 'first', sets the first value from each spike snippet to zero, default
            - 'mean', sets the mean of each spike to zero
            - 'median', sets the median of each spike to zero
            - None, does nothing
        """
        # Check if the cluster is a number, and if it is change it to an Cluster
        spike_times = self.spike_times

        # Open the data
        if data is None:
            data = open(os.path.join(self.recording_dir, self.dat_name), 'rb')
        prev_loc = 0
        waveforms = []

        # Find the size of the chunk (in samples) to extract
        chunk_size = int(chan_count*(pre_window + post_window))

        # Offset is the size before the spike to find, this is in bytes
        offset_size = pre_window * 2 * chan_count
        for i in tqdm(spike_times, leave=False):

            # Loads in the chunk of data from the binary file
            chunk = np.fromfile(data, count = chunk_size, dtype=np.int16, offset=int(64*i - offset_size) - prev_loc)
            # Find current location
            prev_loc = data.tell()
            # Append the chunk to the list
            waveforms.append(chunk)

        # Convert to an array, reshape, and change to 32 bit not 16 (stops overflows) 
        waveforms = np.array(waveforms)
        new_shape = (len(waveforms), int(chunk_size/chan_count), chan_count)
        waveforms = waveforms.reshape(new_shape, order='C')
        waveforms = waveforms.astype(np.int32) 
        
        # zero options, e.g. remove the offset for the snippets
        if zeroing == 'first':
            waveforms = waveforms - waveforms[:, 0, :][:, np.newaxis]  # Set the first value to zero
        elif zeroing == 'mean':
            waveforms = waveforms - np.mean(waveforms, axis=1)[:, np.newaxis]  # Remove the mean across the whole spike
        elif zeroing == 'median':
            waveforms = waveforms - np.median(waveforms, axis=1)[:, np.newaxis]  # Remove the median value from each spike
        elif zeroing is not None:
            print('Misunderstood zeroing method, waveforms wont be zeroed')
        data.close()  # Probs dont need to close, but might as well be good
        return waveforms
    
    def get_recording_length(self, *, chan_count=32, fs=30000):
        """
        Get the length of a recording

        Args:
            chan_count (int, optional): Number of channels. Defaults to 32.
            fs (int, optional): Sampling rate. Defaults to 30000.

        Returns:
            recording length (float): Length of recording (in seconds)
        """
        data = np.memmap(os.path.join(self.recording_dir, self.dat_name), dtype=np.int16)
        return len(data)/chan_count/fs


class Cluster(Spiking):
    '''
    Holds information about Kilosorted clusters, a subclass of Spiking

    Information includes cluster number, spiking times, the recording dir, cluster label and templates
    '''

    def __init__(self, cluster_num, times, recording_dir, label, template_ind,
                 template, max_chan, *, sniff_lock_spikes=None, dat_name='100_CHs.dat'):
        """
        Args:
            cluster_num (int): Cluster number
            times (array of ints): Detected spike times
            recording_dir (str): Location of the recording directory
            label (str): Label of the unit, can be 'good', 'MUA', or 'noise'
            template_ind (int): Template indicie(s) associated with cluster
            template (arrays of floats): Template for cluster
            max_chan (int): Largest channel
            sniff_lock_spikes (array of floats, optional): All spikes as a fraction of sniff phase. Defaults to None.
            dat_name (str, optional): Name of dat that Cluster was derived from. Defaults to '100_CHs.dat'.
        """
        Spiking.__init__(self, times, recording_dir, dat_name=dat_name)
        self.cluster_num = cluster_num
        self.label = label
        self.template_ind = template_ind
        self.template = template
        self.max_chan = max_chan
        self.sniff_lock_spikes = sniff_lock_spikes


class ThresholdCrossings(Spiking):
    '''
    Represent detected threshold crossings on a single channel

    Contains the spike times, recording directory, the channel number in the orignal recording, and the threshold used
    '''

    def __init__(self, times, recording_dir, channel_num, threshold, *, 
                 spike_thresholds=None, dat_name='100_CHs.dat'):
        """
        Args:
            times (array of ints): Threshold crossing times
            recording_dir (str): Location of recording directory
            channel_num (int): Number of channels in recording
            threshold (float): The threshold for spikes
            spike_thresholds (array of floats, optional): The amount of times over threshold a spike was. Defaults to None.
            dat_name (str, optional): Name of the dat file. Defaults to '100_CHs.dat'.
        """
        Spiking.__init__(self, times, recording_dir, dat_name=dat_name)
        self.channel_num = channel_num
        self.threshold = threshold
        self.spike_thresholds = spike_thresholds
