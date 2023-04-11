from unit_recording import Unit_Recording
from scipy.signal import find_peaks
import spiking
import numpy as np
import matplotlib.pyplot as plt
import openephys as oe
from threshold_recording import bandpass_data
import os
from scipy.stats import norm
import joined_recording as jr

class Binary_recording(Unit_Recording):
    '''
    Contains functions and data relevent to binary_recordings, inherets from Unit_Recording
    '''

    def __init__(self, home_dir, channel_count, trial_names, *, trig_chan='100_ADC6.continuous', trial_length=0.12, **kwargs):
        """
        Args:
            home_dir (str): Location of the home directory of the experiment
            channel_count (int): Number of channels in the recording
            trial_names (str or list): File location, or list of trial names
            trig_chan (str, optional): Channel as trial trigger. Defaults to '100_ADC6.continuous'.
            trial_length (float, optional): Length of trials in seconds. Defaults to 0.12.
        """
        Unit_Recording.__init__(self, home_dir, channel_count, trial_length, **kwargs)
        self.trig_chan = trig_chan
        self.trial_names = trial_names
        self.extract_trial_names()

    def extract_trial_names(self):
        '''
        Find the trial names of experiments from the saved trial file
        '''
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

    
    def get_guassian_response(self, trial_name, cluster, *, pre_trial_window=None, post_trial_window=None, sd_denom=50):
        """
        Gets responses of a cluster to a trial in and then convolves it with a guassian around each spike
        Args:
            trial_name (str): The name of the trial to collect responses for
            cluster (int or Cluster): The cluster, if an int calls get_cluster to find the corresponding Cluster
            pre_trial_window (float, optional): Size of the window before the trial start to consider. If None then
                                                takes 2*trial length. Defaults to None.
            post_trial_window (float, optional): Size of the window after the trial ends to consider. If None then takes
                                                 2*trial length. Defaults to None.
            sd_denom (int, optional): The denomination for the standard deviation for each Guassian. Smaller values lead
                                      to larger guassians and vice versa. Defaults to 50.

        Returns:
            xs (array): Time points for x, trial starts at 0
            guasses (array): Array of arrays containing guassian convolutions for each trial
            trial_ends (array): Times trials end for each trial
        """
        guasses = []
        # Converts to a Cluster if needs to.
        if isinstance(cluster, (int, float)):
            cluster = self.get_cluster(cluster)

        # Find the spike times, ends and starts
        trial_spikes = self.get_cluster_trial_response(trial_name, 
                                                       cluster,
                                                       pre_trial_window=pre_trial_window,
                                                       post_trial_window=post_trial_window)
        trial_ends = self.get_unique_trial_ends(trial_name)
        trial_starts = self.get_unique_trial_starts(trial_name)
        trial_ends = [i-j for i, j in zip(trial_ends, trial_starts)]

        # Construct the x axis values, hard coded currently as 1/1000 points per time interval (second or sniff)
        xs = np.arange(-1*pre_trial_window, post_trial_window+self.trial_length, 1/1000)

        # Runs through each trial
        for trial in trial_spikes:
            # Generates a guassian over the full x range for each spike
            trial_guasses = [norm(spike, 1/sd_denom).pdf(xs) for spike in trial] 
            trial_guasses.append(np.zeros(len(xs)))  # Make one series of zeros in case there are no spikes at all =
            guasses.append(np.mean(trial_guasses, axis=0))
        return xs, guasses, trial_ends

def binary_plotter(bin_val, *, ax=None, num_of_bins=5, bin_width=20):
    bin_val = str(bin(bin_val)[2:])
    while len(bin_val) < num_of_bins:
        bin_val = '0' + bin_val
    
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    pulses = [np.ones(bin_width)*int(i) for i in bin_val]
    onset = np.zeros(bin_width)
    offset = np.zeros(bin_width)
    output = np.hstack([onset, np.hstack(pulses), offset])
    plt.plot(output)
    plt.axvspan(bin_width, len(output)-bin_width, alpha=0.3, color='gray')
    return output