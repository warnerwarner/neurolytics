# Class to combine between experiments - still a work in progress
# Assumes that all experiments added are of the same type

from unit_recording import Unit_Recording
from pydoc import locate
import numpy as np
import warnings

class JoinedRecording():

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

    def get_binned_trial_response(self, trial_name, *, pre_trial_window=None, post_trial_window=None, baselined=False, w_nans=False, w_bootstrap=True):
        assert trial_name in self.trial_names, 'Trial is not in any recording'
        rec_responses = []
        for recording in self.recordings:
            #print(recording)
            if trial_name in recording.get_unique_trial_names():
                xs, rec_response = recording.get_all_binned_trial_response(trial_name,
                                                                           pre_trial_window=pre_trial_window,post_trial_window=post_trial_window, baselined=baselined)
                rec_response = np.array(rec_response)
                rec_response = np.rollaxis(rec_response, axis=1)
                rec_responses.append(np.array(rec_response))

        assert len(set([i.shape[-1] for i in rec_responses])) == 1, 'Trials have different time lengths'

        max_reps = max(set([i.shape[0] for i in rec_responses]))
        repeat_lengths = [len(i) for i in rec_responses]
        if len(set(repeat_lengths)) != 1 and not w_bootstrap:
            raise warnings.warn('Mismatch in repeat lengths, cannot make full numpy array')
            mismatched_repeats = True
        else:
            mismatched_repeats = False
            w_nans = False
        if mismatched_repeats:
            if w_nans and w_bootstrap:
                raise warnings.warn('Both nans and bootstrap selected, choosing bootstrap')
                w_nans = False

        if w_nans:
            for index, i in enumerate(rec_responses):
                while i.shape[0] < max_reps:
                    #print(i.shape)
                    numps = np.empty(i.shape[1:])
                    numps[:] = np.NaN
                    #print(numps.shape)
                    i = np.r_[i, [numps]]
                    #print(i.shape)
                rec_responses[index] = i
            for recording_index, recording in enumerate(self.recordings):
                if trial_name not in recording.get_unique_trial_names():
                    empty_response_shape = rec_responses[0].shape
                    #print(empty_response_shape)
                    empty_response_shape = (empty_response_shape[0], len(recording.get_good_clusters()), empty_response_shape[2])
                    nump_response = np.empty(empty_response_shape)
                    nump_response[:] = np.NaN
                    rec_responses.insert(recording_index, nump_response)
            rec_responses = np.concatenate(rec_responses, axis=1)

        if w_bootstrap:
            bootstrap_size = max(repeat_lengths) * 2
            resampled_responses = []
            for rec in rec_responses:
                resampled_rec = [rec[i] for i in np.random.randint(0, len(rec), size=bootstrap_size)]
                resampled_responses.append(resampled_rec)
            rec_responses = np.concatenate(resampled_responses, axis=1)


        # Response returned is of a shape exps x repeats x units x time
        return xs[:-1], rec_responses


    def get_multi_trial_response(self, trial_names, *, pre_trial_window=None, post_trial_window=None, baselined=False):
        assert all(i in self.trial_names for i in trial_names), "One or more trial names not present in any recording"
        trial_response = [self.get_binned_trial_response(trial_name,
                                                         pre_trial_window=pre_trial_window,
                                                         post_trial_window=post_trial_window,
                                                         baselined=baselined)[1] for i in trial_names]

        return trial_response
