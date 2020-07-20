# Class to combine between experiments - still a work in progress
# Assumes that all experiments added are of the same type

from unit_recording import Unit_Recording
from pydoc import locate
import numpy as np
from tqdm import tqdm
import warnings

class JoinedRecording():
    '''
    Holds and connects multiple Recording objects together
    '''

    def __init__(self, recordings):
        self.recordings = recordings
        self.recordings_type = type(recordings[0])
        self.num_of_recordings = len(recordings)
        self.unit_count = None
        # If the recording is a subclass of Unit_recording then adds in good units
        if issubclass(self.recordings_type, locate('unit_recording.Unit_Recording')):
            units = [i.get_good_clusters() for i in recordings]
            units = np.concatenate(units)
            self.unit_count = len(units)
        trial_names = []
        for recording in recordings:
            if hasattr(recording, 'trial_names'):
                trial_names.append(recording.trial_names)

        trial_names = list(set(np.concatenate(trial_names)))
        self.trial_names = trial_names

    def get_binned_trial_response(self, trial_name, *, pre_trial_window=None, post_trial_window=None, bin_size=0.01, baselined=False, w_nans=False, w_bootstrap=False, saved_trial=False):
        '''
        Returns binned responses from units across all recordings for a specific trial (if the trial is present in the recording).
        Arguments:
        trial_names: The name of the trial
        
        Optional arguments:
        pre_trial_window: The window prior to the trial to include
        post_trial_window: The windows post the trial to include
        bin_size: Size of the bins to use
        baselined: Should the response be baselined subtracted, only works if there is a pre_trial_windows > 0
        w_nans: Even mismatched repeat sizes by inserting nans into the arrays
        w_bootstrap: Even mismatched repeat sizes by bootstrapping data
        saved_trial: Should a trial be kept out of the bootstrapping, useful for classifiers
        '''
        
        assert trial_name in self.trial_names, 'Trial is not in any recording'
        rec_responses = []
        for recording in self.recordings:
            #print(recording)
            if trial_name in recording.get_unique_trial_names():
                xs, rec_response = recording.get_all_binned_trial_response(trial_name,
                                                                           pre_trial_window=pre_trial_window,
                                                                           post_trial_window=post_trial_window,
                                                                           bin_size=bin_size,
                                                                           baselined=baselined)
                rec_response = np.array(rec_response)
                rec_response = np.rollaxis(rec_response, axis=1)
                rec_responses.append(np.array(rec_response))
        # Check to make sure that the trials have all come out with the same response length
        assert len(set([i.shape[-1] for i in rec_responses])) == 1, 'Trials have different time lengths'

        # Find the maximum number of repeats across all experiments
        max_reps = max(set([i.shape[0] for i in rec_responses]))
        repeat_lengths = [len(i) for i in rec_responses]


        # I think some of these conditions are redundent but they work alright
        if len(set(repeat_lengths)) != 1 and not w_bootstrap:  # If there is a mismatch and bootstrapped is not used then will throw up a warning
            warnings.warn('Mismatch in repeat lengths, cannot make full numpy array')
            mismatched_repeats = True
        else:
            mismatched_repeats = False
            w_nans = False
        if mismatched_repeats:
            if w_nans and w_bootstrap:
                warnings.warn('Both nans and bootstrap selected, choosing bootstrap')
                w_nans = False

        ### Fill up empty repeats with nans so can make a fully numpy array - not really that useful
        if w_nans:
            for index, i in enumerate(rec_responses):
                while i.shape[0] < max_reps:
                    numps = np.empty(i.shape[1:])
                    numps[:] = np.NaN
                    i = np.r_[i, [numps]]
                rec_responses[index] = i
            for recording_index, recording in enumerate(self.recordings):
                if trial_name not in recording.get_unique_trial_names():
                    empty_response_shape = rec_responses[0].shape
                    empty_response_shape = (empty_response_shape[0], len(recording.get_good_clusters()), empty_response_shape[2])
                    nump_response = np.empty(empty_response_shape)
                    nump_response[:] = np.NaN
                    rec_responses.insert(recording_index, nump_response)
            rec_responses = np.concatenate(rec_responses, axis=1)

        ### Bootstrap all the data to make it an even length
        if w_bootstrap:
            bootstrap_size = max(repeat_lengths) * 2
            if saved_trial:  # If there is an trial to be kept out, reduces the bootstrapping size by 
                bootstrap_size -= 2
            resampled_responses = []
            all_saved_trials = []
            for rec in rec_responses:
                if saved_trial:
                    saved_index = np.random.randint(len(rec))
                    saved_trial = rec[i]
                    all_saved_trials.append(saved_trial)
                    rec = np.array(rec)[np.arange(len(rec)) != saved_index]
                resampled_rec = [rec[i] for i in np.random.randint(0, len(rec), size=bootstrap_size)]
                resampled_responses.append(resampled_rec)
            rec_responses = np.concatenate(resampled_responses, axis=1)
        else:
            try:
                rec_responses = np.concatenate(rec_responses, axis=1)
            except:
                None
        # Response returned is of a shape exps x repeats x units x time
        if saved_trial:
            return xs[:-1], rec_responses, all_saved_trials
        return xs[:-1], rec_responses
    
    def saved_repeat_bootstrapping(self, rec_responses, saved_trial_size=1):
        #### Trying to bootstrap save a different way
        saved_trials = []
        bootstrapped_trials = []
        repeat_lengths = [len(i) for i in rec_responses]
        bootstrap_len = 2*np.max(repeat_lengths) - 2 * saved_trial_size
        for i in rec_responses:
            saved_index = np.random.randint(len(i))
            saved_trials.append(i[saved_index])
            resta_trials = np.array(i)[np.arange(len(i)) != saved_index]
            bootstrapped = [np.array(resta_trials[i]) for i in np.random.randint(len(resta_trials), size=bootstrap_len)]
            bootstrapped_trials.append(bootstrapped)
        bootstrapped_trials = np.concatenate(bootstrapped_trials, axis=1)
        saved_trials = np.concatenate(saved_trials, axis=0)
        return bootstrapped_trials, saved_trials        

    def get_multi_trial_response(self, trial_names, *, pre_trial_window=None, post_trial_window=None, bin_size=0.01, baselined=False):
        assert all(i in self.trial_names for i in trial_names), "One or more trial names not present in any recording"
        trial_response = [self.get_binned_trial_response(i,
                                                         pre_trial_window=pre_trial_window,
                                                         post_trial_window=post_trial_window,
                                                         bin_size=bin_size,
                                                         baselined=baselined)[1] for i in trial_names]
        return trial_response



    def get_trial_response(self, trial_name, pre_trial_window=0.5, post_trial_window=0.5):
        assert trial_name in self.trial_names, 'Trial is not in any recording'
        rec_responses = []
        for recording in self.recordings:
            #print(recording)
            if trial_name in recording.get_unique_trial_names():
                all_cluster_spikes = recording.get_all_trial_response(trial_name, pre_trial_window=pre_trial_window, post_trial_window=post_trial_window)
                rec_responses.append(all_cluster_spikes)
        return rec_responses
    
    def get_single_cluster_trial_response(self, trial_name, pre_trial_window=0.5, post_trial_window=0.5):
        assert trial_name in self.trial_names, 'Trial is not in any recording'

