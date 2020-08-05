from unit_recording import Unit_Recording
import os
import numpy as np
import openephys as oe
import pickle
from threshold_recording import bandpass_data
from scipy.signal import find_peaks


class Correlation_Recording(Unit_Recording):
    """
    Represents a single Correlation Recording experiment
    """

    def __init__(self, home_dir, channel_count, trialbank_loc, *, trial_length=2, trig_chan='100_ADC6.continuous', **kwargs):
        """
        Args:
            home_dir (str): Location of the home directory for the experiment
            channel_count (int): Number of channels used in the recording
            trialbank_loc (str): Location of the trialbank pickle file
            trial_length (int, optional): Length of the trials. Defaults to 2.
            trig_chan (str, optional): Trigger channel, used to find trial starts. Defaults to '100_ADC6.continuous'.
        """

        Unit_Recording.__init__(self, home_dir, channel_count, trial_length, **kwargs)
        self.trialbank_loc = trialbank_loc
        self.trial_length = trial_length
        self.trig_chan = trig_chan
        self.extract_trial_names()

    def extract_trial_names(self):
        """
        Extracts the trial names from the trialbank pickle into an array
        """
        trialbank = pickle.Unpickler(open(self.trialbank_loc, 'rb')).load()
        trial_names = [i[-1] for i in trialbank]
        repeats = int(len(self.trial_starts)/len(trial_names))
        try:
            assert len(self.trial_starts) % len(trial_names) == 0.0
        except (AssertionError):
            raise ValueError('Length of trial starts is not fully divisible by number of found trial types (%f)' % (len(self.trial_starts) % (len(trial_names))))

        full_trial_names = []
        for i in range(repeats):
            for j in trial_names:
                full_trial_names.append(j)
        self.trial_names = full_trial_names
        self.repeats = repeats
        print('Found %d trial names, and %d repeats' % (len(full_trial_names), repeats))
