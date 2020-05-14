# Class to combine between experiments - still a work in progress
# Assumes that all experiments added are of the same type

from unit_recording import Unit_Recording
from pydoc import locate
import numpy as np

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







    def get_binned_trial_response(self, trial_name, *, pre_trial_window=None, post_trial_window=None, baselined=False):
        assert trial_name in self.trial_names, 'Trial is not in any recording'
        unit_responses = []
        for recording in self.recordings:
            good_clusters = recording.get_good_clusters()
            if trial_name in recording.get_unique_trial_names():
                for i in good_clusters:
                    true_x, unit_resp = recording.get_binned_trial_response(trial_name,
                                                                    i.cluster_num,
                                                                    pre_trial_window=pre_trial_window,
                                                                    post_trial_window=post_trial_window,
                                                                    baselined=False)
                    unit_responses.append(unit_resp)
            else:
                for i in good_clusters:
                    unit_responses.append([np.nan])

        repeat_range = [len(i) for i in unit_responses]
        most_reps = max(repeat_range)
        for unit in unit_responses:
            while len(unit) < most_reps:
                unit.append(np.nan)
        return true_x[:-1], unit_responses
