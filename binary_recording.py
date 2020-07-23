from unit_recording import Unit_Recording
from scipy.signal import find_peaks
import spiking
import numpy as np
import openephys as oe
from threshold_recording import bandpass_data
import os
from scipy.stats import norm


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
        guasses = []
        if isinstance(cluster, (int, float)):
            cluster = self.get_cluster(cluster)
        trial_spikes = self.get_cluster_trial_response(trial_name, cluster, pre_trial_window=pre_trial_window, post_trial_window=post_trial_window)
        trial_ends = self.get_unique_trial_ends(trial_name)
        trial_starts = self.get_unique_trial_starts(trial_name)
        trial_ends = [i-j for i, j in zip(trial_ends, trial_starts)]
        xs = np.arange(-1*pre_trial_window, post_trial_window+self.trial_length, 1/1000)
        for trial in trial_spikes:
            trial_guasses = [norm(spike, 1/sd_denom).pdf(xs) for spike in trial]
            trial_guasses.append(np.zeros(len(xs)))
            guasses.append(np.mean(trial_guasses, axis=0))
        return xs, guasses, trial_ends