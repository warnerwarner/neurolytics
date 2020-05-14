from unit_recording import Unit_Recording
import os
import numpy as np
import openephys as oe
import pickle
from threshold_recording import bandpass_data
from scipy.signal import find_peaks


class Correlation_Recording(Unit_Recording):

    def __init__(self, home_dir, channel_count, trialbank_loc, *, trial_length=2, trig_chan='100_ADC6.continuous', **kwargs):
        Unit_Recording.__init__(self, home_dir, channel_count, trial_length, **kwargs)
        self.trialbank_loc = trialbank_loc
        self.trial_length = trial_length
        self.trig_chan = trig_chan
        self.trial_starts = None
        self.trial_ends = None
        self.trial_names = None
        self.repeats = None
        self.resp_peaks = None

    def set(self):
        print('finding trial starts')
        self._find_trial_starts()
        print('extracting trial names')
        self._extract_trial_names()
        print('Finding respiration peaks')
        self._find_respiration_peaks()
        print('Finding sniff subtracted')
        self._find_all_sniff_lock_avg(self.trial_length*2)

    def _extract_trial_names(self):
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
