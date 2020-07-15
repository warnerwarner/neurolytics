from unit_recording import Unit_Recording
from scipy.signal import find_peaks
import spiking
import numpy as np
import openephys as oe
from threshold_recording import bandpass_data
import os


class Binary_recording(Unit_Recording):
    '''
    Contains functions and data relevent to binary_recordings, inherets from Unit_Recording
    '''

    def __init__(self, home_dir, channel_count, trial_names, *, trig_chan='100_ADC6.continuous', trial_length=0.12, **kwargs):
        Unit_Recording.__init__(self, home_dir, channel_count, trial_length, **kwargs)
        self.trig_chan = trig_chan
        self.trial_names = trial_names
        self.trial_starts = None
        self.trial_ends = None

    def set(self, *, resp_channel='100_ADC1.continuous', sniff_avg=True):
        '''
        Sets all the non saved variables
        '''
        print('Finding trial names...')
        self._extract_trial_names()
        super().set(1)

    def _extract_trial_names(self):
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