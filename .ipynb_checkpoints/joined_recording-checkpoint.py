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
        """
        Args:
            recordings (array): Array of Recording objectes, can be of any type
        """
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

    def get_binned_trial_response(self, trial_name, *, pre_trial_window=None, post_trial_window=None, bin_size=0.01, baselined=False, 
                                  w_nans=False, w_bootstrap=False, saved_trial=False, bootstrap_limit=100):
        """
        Gets the binned responses of all units to a trial

        Args:
            trial_name (str): Name of trial
            pre_trial_window (float, optional): The length of window to take prior to the trial. Defaults to None.
            post_trial_window (float, optional): The length of the window to take post the end of the trial. Defaults to None.
            bin_size (float, optional): Size of bins. Defaults to 0.01.
            baselined (bool, optional): Subtract the baseline predicted activity. Defaults to False.
            w_nans (bool, optional): Fill mismatched repeats with nans. Defaults to False.
            w_bootstrap (bool, optional): Fill mismatched repeats using bootstrap. Defaults to False.
            saved_trial (bool, optional): Leave a trial out of any bootstrapping. Defaults to False.
            bootstrap_limit(int, optional): How many times the maximum number of repeats should all trials be bootstrapped to.
                                            Greater values increase data size but decrease random sampling error. Default to 100.

        Returns:
            xs (array): Time values corresponding to the trial responses
            rec_responses (array): Array of array of binned responses for each cluster during each trial
            all_saved_trials (array, optional): If saved_trial is true then also returns the saved trials
        """        
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
            #rec_responses = np.concatenate(rec_responses, axis=1)

        ### Bootstrap all the data to make it an even length
        if w_bootstrap:
            if bootstrap_limit < max(repeat_lengths):
                print('Bootstrapping limit too small, increasing to repeat limit')
                bootstrap_limit = max(repeat_lengths)
            bootstrap_size = bootstrap_limit
            resampled_responses = []
            all_saved_trials = []
            for rec in rec_responses:
                if saved_trial:
                    saved_index = np.random.randint(len(rec))
                    saved_t = rec[saved_index]
                    all_saved_trials.append(saved_t)
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

    def get_multi_trial_response(self, trial_names, *, super_bootstrap_limit=None, super_saved_trial=False, **kwargs):
        assert all(i in self.trial_names for i in trial_names), "One or more trial names not present in any recording"
        trial_response = [self.get_binned_trial_response(i, **kwargs)[1] for i in trial_names]
        # Runs if the groups of responses need to be bootstrapped together
        if super_bootstrap_limit is not None:
            conj_trial_response = trial_response[0]

            for exp in trial_response[1:]:
                conj_trial_response = [np.concatenate([i, j]) for i, j in zip(conj_trial_response, exp)]

            res_rec = []
            all_saved_trials = []
            for rec in conj_trial_response:
                if super_saved_trial:
                    saved_index = np.random.randint(len(rec))
                    saved_trial = rec[i]
                    all_saved_trials.append(saved_trial)
                    rec = np.array(rec)[np.arange(len(rec)) != saved_index]
                bs_rec = [rec[i] for i in np.random.randint(0, len(rec), size=super_bootstrap_limit)]
                empt_rec = []
                for i in range(len(bs_rec[0])):
                    empt_rec.append([x[i] for x in bs_rec])
                res_rec.append(empt_rec)
            res_rec = np.concatenate(res_rec)    
            trial_response = res_rec

        if super_saved_trial:
            return trial_response, all_saved_trials
        return trial_response



    def get_trial_response(self, trial_name, pre_trial_window=0.5, post_trial_window=0.5):
        """
        Gets the response of all clusters to a trial type (in spike times)
        Args:
            trial_name ([type]): [description]
            pre_trial_window (float, optional): [description]. Defaults to 0.5.
            post_trial_window (float, optional): [description]. Defaults to 0.5.

        Returns:
            [type]: [description]
        """
        assert trial_name in self.trial_names, 'Trial is not in any recording'
        rec_responses = []
        for recording in self.recordings:
            #print(recording)
            if trial_name in recording.get_unique_trial_names():
                all_cluster_spikes = recording.get_all_trial_response(trial_name, pre_trial_window=pre_trial_window, post_trial_window=post_trial_window)
                rec_responses.append(all_cluster_spikes)
        return rec_responses
    
    def get_single_cluster_trial_response(self, trial_name, cluster_index, *, pre_trial_window=0.5, post_trial_window=0.5):
        """
        Returns a single cluster response to a passed trial
        Args:
            trial_name (string): The name of the trial
            cluster_index (int): The index of the cluster
            pre_trial_window (float, optional): Window before the trial to include. Defaults to 0.5.
            post_trial_window (float, optional): Window after the trial to include. Defaults to 0.5.
        """
        assert trial_name in self.trial_names, 'Trial is not in any recording'
        
        cluster = np.concatenate([rec.get_good_clusters() for rec in self.recordings])[cluster_index]
        recording_index = [index for index, rec in enumerate(self.recordings) for cluster in rec.get_good_clusters()][cluster_index]
        rec = self.recordings[recording_index]
        resp = rec.get_cluster_trial_response(trial_name, cluster, pre_trial_window=pre_trial_window, post_trial_window=post_trial_window)
        return resp
        
    def get_single_unit(self, cluster_index):
        cluster = np.concatenate([rec.get_good_clusters() for rec in self.recordings])[cluster_index]
        return cluster