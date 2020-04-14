from unit_recording import Unit_Recording
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
import numpy as np
import random

class Classifier():
    '''
    Classifies responses to different trials using a vairety of classifiers and techniques
    '''

    def __init__(self, *, scale=None, test_size=1, pre_trial_window=None,
                 post_trial_window=None, bin_size=0.01, C=1000):
        self.scale = scale
        self.type = None
        self.varience = 1
        self.trial_spikes = None
        self.recordings = None
        self.test_size = test_size
        self.pre_trial_window = pre_trial_window
        self.post_trial_window = post_trial_window
        self.bin_size = bin_size
        self.C = C
        self.unit_response = None
        self.y_var = None
        self.svm = None
        self.X_test=None
        self.X_train = None
        self.y_test = None
        self.y_train = None
        self.pca = None
        self.accuracy = None
        self.shuffle = False
        self.num_of_units = 0
        self.window_start = None
        self.window_end = None

    def make_pca_responses(self, n_components, trial_names, *, baseline=True):
        '''
        Creates an array of neural responses to trials in PC space

        Arguments:
        n_components - The number of components for the PCA
        trial_names - Names of the trials in the experiment to apply PCA to

        Optional arguments:
        baseline - Should the responses be baseline subtracted (to remove expected sniff locked activity), default True
        '''
        pca = PCA(n_components=n_components)
        full_pcad_responses = []
        full_trial_names = []
        trial_responses = []
        all_trial_repeats = []

        # Runs through all the passed trials
        # Then for each trial runs through each recording
        # Then runs through each good cluster in each recording
        for trial in trial_names:
            trial_repeats = []
            for recording in self.recordings:
                for cluster in recording.get_good_clusters():
                    # Default values for pre/post trial window is None which will set them to 2*trial length
                    binned_trial_response = recording.get_binned_trial_response(trial,
                                                                                cluster.cluster_num,
                                                                                post_trial_window=self.post_trial_window,
                                                                                pre_trial_window=self.pre_trial_window,
                                                                                bin_size=self.bin_size,
                                                                                baselined=baseline)
                    trial_responses.append(binned_trial_response[1])  # The binned_trial_response is both the x and y values so discard x
                trial_repeats.append(len(recording.get_unique_trial_starts(trial)))  # Find how many repeats there are of the trial

            # Make sure that the number of repeats are equal across experiments - might change in the future
            assert len(set(trial_repeats)) != 0, 'Length of repeats varies between experiments...'
            # Add in the length of repeats
            all_trial_repeats.append(trial_repeats[0])
            for i in range(trial_repeats[0]):
                full_trial_names.append(trial)
        # Find the number of units
        self.num_of_units = sum([len(i.get_good_clusters()) for i in self.recordings])

        # Join all the responses together
        joined_response = np.concatenate(trial_responses, axis=0)

        # Apply the pca
        pcad_response = pca.fit_transform(joined_response)
        rearranged_trial = []
        num_of_trials = sum(all_trial_repeats)

        # Rearrange the responses to be in the trial x unit*pcs format
        for i in range(int(len(pcad_response)/num_of_trials)):
            rearranged_trial.append(pcad_response[int(num_of_trials*i):int(num_of_trials*(i+1))].T)
        rearranged_trial = np.concatenate(rearranged_trial).T
        full_pcad_responses.append(rearranged_trial)
        full_pcad_responses = np.concatenate(full_pcad_responses, axis=0)

        self.unit_response = full_pcad_responses
        self.y_var = full_trial_names
        self.type = 'PCA'
        self.pca = pca


    def pca_classifier(self, n_components, trial_names, *, baseline=True, shuffle=False):
        '''
        Run a classifier on PCA data

        Arguments:
        n_components - number of components for the pca
        trial_names - names of the trials to be analysed

        Optional arguments:
        baseline - remove the baseline sniff locked activity, default true
        shuffle - shuffle the labels, default false
        '''

        # Make the response if not already done so
        if self.unit_response is None:
            self.make_pca_responses(n_components, trial_names, baseline=baseline)

        # Should the responses be scaled in anyway


        # If the test size is < 1 then treated as fraction of trials, and greater treated as number of trials
        if self.test_size < 1:
            test_size = self.test_size
        else:
            test_size = self.test_size/len(self.unit_response)

        if shuffle:
            self.shuffle = True
            random.shuffle(self.y_var)

        X_train, X_test, y_train, y_test = train_test_split(self.unit_response, self.y_var, test_size=test_size)

        if self.scale is not None:
            if self.scale == 'standard':
                scaler = StandardScaler(with_mean=False, with_std=True)
            elif self.scale == 'minmax':
                scaler = MinMaxScaler()
            else:
                raise ValueError('Scalar type incorrect, must be standard, minmax, or None')
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)


        svm = LinearSVC(C=self.C)
        svm.fit(X_train, y_train)
        self.svm = svm
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def make_unit_response(self, trial_names, *, baseline=False, window_start=None, window_end=None, return_resp=False):
        '''
        Make a unit response matrix in the form trials x unit*timepoints

        Arguments:
        trial_names - trials to be used in construction
        Optional arguments:
        baseline - should the sniff locked expected activity be subtracted, defualt True
        window_start - The window start of response, default None
        window_end - The window end of the response to be considered, default None
        '''

        all_trial_responses = []
        y_var = []

        # Run through all the trials,
        # then the recordings
        # then the clusters in each recording
        for trial in trial_names:
            trial_responses = []
            for recording in self.recordings:
                for cluster in recording.get_good_clusters():
                    binned_trial_response = recording.get_binned_trial_response(trial,
                                                                                cluster.cluster_num,
                                                                                post_trial_window=self.post_trial_window,
                                                                                pre_trial_window=self.pre_trial_window,
                                                                                bin_size=self.bin_size,
                                                                                baselined=baseline)
                    trial_responses.append(binned_trial_response[1])
                    if self.pre_trial_window is None:  # Set to the default used in the get_binned_trial_response
                        self.pre_trial_window = recording.trial_length*2
            all_trial_responses.append(trial_responses)
            for i in range(len(binned_trial_response[1])):
                y_var.append(trial)
        self.num_of_units = sum([len(i.get_good_clusters()) for i in self.recordings])
        trial_responses = np.concatenate(np.concatenate(all_trial_responses, axis=1), axis=1)
        self.unit_response = trial_responses
        self.y_var = y_var
        if return_resp:
            return trial_responses

    def make_difference_response(self, trial_names_odour, trial_names_blanks, *, baseline=False, window_start=None, window_end=None):
        '''
        Sets the unit_response to be the difference between odour and blank trials - currently uses the difference between the odour and a random blank, not the average
        '''
        blank_response = self.make_unit_response(trial_names_blanks, baseline=baseline, window_start=window_start, window_end=window_end, return_resp=True)
        odour_response = self.make_unit_response(trial_names_odour, baseline=baseline, window_start=window_start, window_end=window_end, return_resp=True)
        difference = odour_response - blank_response
        self.unit_response = difference


    def window_classifier(self, trial_names, window_start, window_end, *, baseline=False, shuffle=False, sub_units=None):
        if self.unit_response is None:
            self.make_unit_response(trial_names, baseline=baseline)

        bin_start = int((self.pre_trial_window + window_start)/self.bin_size)
        bin_end = int((self.pre_trial_window + window_end)/self.bin_size)
        num_of_bins = int(len(self.unit_response[0])/self.num_of_units)

        window_unit_response = []
        for i in range(self.num_of_units):
            window_unit_response.append(self.unit_response[:, bin_start+num_of_bins*i:bin_end+num_of_bins*i])

        window_unit_response = np.sum(window_unit_response, axis=2).T

        # Runs if sub units is set to an int, only uses a random subsection of units
        if sub_units is not None:
            assert type(sub_units) == int, 'sub_units needs to be int'
            np.random.shuffle(window_unit_response.T)
            window_unit_response = window_unit_response[:, :sub_units]

        if self.test_size < 1:
            test_size = self.test_size
        else:
            test_size = self.test_size/len(window_unit_response)

        if shuffle:
            self.shuffle = True
            random.shuffle(self.y_var)


        X_train, X_test, y_train, y_test = train_test_split(window_unit_response, self.y_var, test_size=test_size)

        if self.scale is not None:
            if self.scale == 'standard':
                scaler = StandardScaler()
            elif self.scale == 'minmax':
                scaler = MinMaxScaler()
            elif self.scale == 'robust':
                scaler = RobustScaler()
            else:
                raise ValueError('Scalar type incorrect, must be standard, minmax, or None')
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        svm = LinearSVC(C=self.C)
        svm.fit(X_train, y_train)
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.svm = svm
        self.window_start = window_start
        self.window_end = window_end


    def find_accuracy(self):
        assert self.svm is not None, 'Please classify first'
        correct = 0
        for i, j in zip(self.svm.predict(self.X_test), self.y_test):
            if i == j:
                correct += 1
        self.accuracy = correct/len(self.y_test)
        return self.accuracy
