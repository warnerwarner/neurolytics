from unit_recording import Unit_Recording
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import numpy as np
import random

class Classifier():

    def __init__(self, *, kernel='linear', scale=None, test_size=1, pre_trial_window=None,
                 post_trial_window=None, bin_size=0.01, C=1000):
        self.kernel = kernel
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

        pca = PCA(n_components=n_components)
        full_pcad_responses = []
        full_trial_names = []
        trial_responses = []
        all_trial_repeats = []
        for trial in trial_names:
            trial_repeats = []
            for recording in self.recordings:
                for cluster in recording.get_good_clusters():
                    binned_trial_response = recording.get_binned_trial_response(trial,
                                                                                cluster.cluster_num,
                                                                                post_trial_window=self.post_trial_window,
                                                                                pre_trial_window=self.pre_trial_window,
                                                                                bin_size=self.bin_size,
                                                                                baselined=baseline)
                    trial_responses.append(binned_trial_response[1])
                trial_repeats.append(len(recording.get_unique_trial_starts(trial)))
                self.num_of_units += len(cluster.get_good_clusters())
            assert len(set(trial_repeats)) != 0, 'Length of repeats varies between experiments...'
            all_trial_repeats.append(trial_repeats[0])
            for i in range(trial_repeats[0]):
                full_trial_names.append(trial)
        joined_response = np.concatenate(trial_responses, axis=0)
        pcad_response = pca.fit_transform(joined_response)
        rearranged_trial = []
        num_of_trials = sum(all_trial_repeats)
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
        self.make_pca_responses(n_components, trial_names, baseline=baseline)
        if self.scale is not None:
            if self.scale == 'standard':
                scaler = StandardScaler(with_mean=False, with_std=True)
            elif self.scale == 'minmax':
                scaler = MinMaxScaler()
            else:
                raise ValueError('Scalar type incorrect, must be standard, minmax, or None')
                pcad_response = scaler.fit_transform(self.unit_response)
        else:
            pcad_response = self.unit_response
        if self.test_size < 1:
            test_size = self.test_size
        else:
            test_size = self.test_size/len(pcad_response)

        if shuffle:
            self.shuffle = True
            random.shuffle(self.y_var)

        X_train, X_test, y_train, y_test = train_test_split(pcad_response, self.y_var, test_size=test_size)

        svm = SVC(C=self.C, kernel=self.kernel)
        svm.fit(X_train, y_train)
        self.svm = svm
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def make_unit_response(self, trial_names, *, baseline=True, window_start=None, window_end=None):

        all_trial_responses = []
        y_var = []
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
                    if self.pre_trial_window is None:
                        self.pre_trial_window = recording.trial_length*2
            all_trial_responses.append(trial_responses)
            for i in range(len(binned_trial_response[1])):
                y_var.append(trial)
        self.num_of_units = sum([len(i.get_good_clusters()) for i in self.recordings])
        print(self.num_of_units)
        trial_responses = np.concatenate(np.concatenate(all_trial_responses, axis=1), axis=1)
        self.unit_response = trial_responses
        self.y_var = y_var

    def window_classifier(self, trial_names, window_start, window_end, *, baseline=True, shuffle=False):
        if self.unit_response is None:
            self.make_unit_response(trial_names, baseline=baseline)

        bin_start = int((self.pre_trial_window + window_start)/self.bin_size)
        bin_end = int((self.pre_trial_window + window_end)/self.bin_size)
        num_of_bins = int(len(self.unit_response[0])/self.num_of_units)

        window_unit_response = []
        for i in range(self.num_of_units):
            window_unit_response.append(self.unit_response[:, bin_start+num_of_bins*i:bin_end+num_of_bins*i])

        window_unit_response = np.sum(window_unit_response, axis=2).T

        if self.test_size < 1:
            test_size = self.test_size
        else:
            test_size = self.test_size/len(window_unit_response)

        if shuffle:
            self.shuffle = True
            random.shuffle(self.y_var)

        X_train, X_test, y_train, y_test = train_test_split(window_unit_response, self.y_var, test_size=test_size)

        svm = SVC(C=self.C, kernel=self.kernel)
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
        print('Accuracy is', self.accuracy)