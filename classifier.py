'''
TODO - PCA classifier up into train and test data
'''

from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
import numpy as np
import random
from sklearn.exceptions import ConvergenceWarning
import warnings
warnings.filterwarnings('ignore', category=ConvergenceWarning)
 
class Classifier():
    '''
    Classifies responses to different trials using a vairety of classifiers and techniques
    '''

    def __init__(self, *, scale='standard', test_size=1, pre_trial_window=None,
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
        self.X_test = None
        self.X_train = None
        self.y_test = None
        self.y_train = None
        self.pca = None
        self.accuracy = None
        self.shuffle = False
        self.num_of_units = 0
        self.window_start = None
        self.window_end = None
        self.trial_names = None


    def reassign_trial_label(self, trial_name, new_name):
        assert trial_name in self.trial_names, 'Trial to change is not present in trial_names'
        assert self.unit_response is not None, 'Please construct/define unit response'
        if new_name not in self.trial_names:
            print('Warning: new name is not already in trial names')

        y_var = self.y_var
        reassigned_y_var = [new_name if i == trial_name else i for i in y_var]
        self.y_var = reassigned_y_var
        trial_names = np.array(self.trial_names)
        if new_name in trial_names:
            old_response = self.unit_response[np.where(trial_names == trial_name)[0][0]]
            new_response = self.unit_response[np.where(trial_names == new_name)[0][0]]
            combined_response = np.concatenate([old_response, new_response])
            self.unit_response[np.where(trial_names == trial_name)[0][0]] = combined_response
            del self.unit_response[np.where(trial_names == new_name)[0][0]]
            trial_names = np.delete(trial_names, np.where(trial_names == trial_name)[0][0])
            #trial_names[np.where(trial_names == trial_name)[0][0]] = new_name
        else:
            trial_names[np.where(trial_names == trial_name)[0][0]] = new_name
        self.trial_names = list(trial_names)


    def make_pcad_response(self, n_components, trial_names, *, trace_start=0, window_size=None, baseline=True, reassign_y_var=None):
        '''
        Create and return a pcad unit response

        Arguments:
        n_components - Number of components for the PCA
        trial_names - The names of the trials to be used
        Optional arguments:
        window_size - If the unit response should be averaged over a window before it is PCAd, default None
        baseline - Should the response be baselined - passed to the make_unit_response function, default True
        reassign_y_var - Should any trial labels be reassigned, takes a 2d array, where each value has the name of a trial
                         type. Converts all of the second label to the first label, default None
        '''

        # Make a unit response if not already done so
        if self.unit_response is None:
            self.make_unit_response(trial_names, baseline=baseline)
        else:  # Check that the trial names being passed were used to build the unit response
            assert [i in self.trial_names for i in trial_names], 'Trial name passed not in classifiers response'

        bin_start =  int((self.pre_trial_window + trace_start) / self.bin_size)
        # Changing the shape to be PCAd
        pcad_response = []
        trial_responses = [np.concatenate(i) for i in self.unit_response]
        combined_response = np.concatenate(trial_responses)
        combined_response = combined_response[:, bin_start:]

        # Apply a rolling average window across the response if required
        if window_size is not None:
            assert isinstance(window_size, int), 'Window size must be an int'
            windowed_response = np.cumsum(combined_response, dtype=float, axis=-1)
            windowed_response[:, window_size:] = windowed_response[:, window_size:] - windowed_response[:, :-window_size]
            windowed_response = windowed_response[:, :1 - window_size] / window_size
            combined_response = windowed_response

        # If the number of components passed is greater than the number of features then reduces the components to n_features
        if n_components > combined_response.shape[1]:
            print('n_components greater than number of features, reducing to maximum num of features (%d-->%d)' % (n_components, combined_response.shape[1]))
            n_components = combined_response.shape[1]

        # Constructing y_var
        y_var = []
        for j in trial_names:
            trial_index = self.trial_names.index(j)
            trial_response = self.unit_response[trial_index]
            for i in range(trial_response.shape[0]):
                y_var.append(j)
        if reassign_y_var is not None:
            for i in reassign_y_var:
                y_var = [i[0] if j == i[1] else j for j in y_var]

        # Make an instance of a PCA and fit and transform the data
        pca = PCA(n_components=n_components)
        pcad_response = pca.fit_transform(combined_response)
        reordered_pcad = []
        trial_num = sum([i.shape[0] for i in self.unit_response])
        for i in range(trial_num):
            reordered_pcad.append(pcad_response[i*self.num_of_units:(i+1)*self.num_of_units])
        reordered_pcad = np.array(reordered_pcad)
        self.pca = pca
        # print(reordered_pcad.shape)
        return reordered_pcad, y_var

    def pca_classifier(self, pcad_response, y_var):
        if len(pcad_response.shape) > 2:
            pcad_full_response = np.concatenate(np.stack(pcad_response, axis=2), axis=0).T
        else:
            pcad_full_response = pcad_response

        X_train, X_test, y_train, y_test = train_test_split(pcad_full_response, y_var, test_size=self.test_size)

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
        self.y_var = y_var



    def full_pca_classifier(self, n_components, trial_names, *,  baseline=True, shuffle=False, reassign_y_var=None, single_components=None):
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
            self.make_unit_response(trial_names, baseline=baseline)
        else:
            assert [i in self.trial_names for i in trial_names], 'Trial name passed not in classifiers response'

        reordered_pcad, y_var = self.make_pcad_response(n_components, trial_names, baseline=baseline, reassign_y_var=reassign_y_var)

        pcad_full_response = np.concatenate(np.stack(reordered_pcad, axis=2), axis=0).T

        if single_components is not None:
            if isinstance(single_components, int):
                pcad_full_response = pcad_full_response[:, single_components].reshape(-1, 1)
            else:
                pcad_full_response = pcad_full_response[:, single_components]

        self.pca_classifier(pcad_full_response, y_var)

    def make_unit_response(self, trial_names, *, baseline=False, return_resp=False):
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
                clusters_responses = recording.get_all_binned_trial_response(trial,
                                                                             pre_trial_window=self.pre_trial_window,
                                                                             post_trial_window=self.post_trial_window,
                                                                             baselined=baseline)

                if self.pre_trial_window is None:  # Set to the default used in the get_binned_trial_response
                    self.pre_trial_window = recording.trial_length*2
                if self.post_trial_window is None:
                    self.post_trial_window = recording.trial_length*2
                trial_responses.append(clusters_responses[1])
            trial_responses = np.concatenate(trial_responses, axis=0)
            trial_responses = np.rollaxis(trial_responses, axis=1)  # Roll the axis so that the matrix now goes trials x units x time
            all_trial_responses.append(trial_responses)
            print(len(all_trial_responses))
            for i in range(len(trial_responses)):  # Append as many repeats as there were of the trial to the y_var variable
                y_var.append(trial)
        print(len(all_trial_responses))
        self.unit_response = all_trial_responses  # Now the response is trial_type x repeats x units x time
        self.num_of_units = sum([len(i.get_good_clusters()) for i in self.recordings])
        self.y_var = y_var
        self.trial_names = trial_names
        if return_resp:
            return all_trial_responses

    def make_difference_response(self, trial_names_odour, trial_names_blanks, *, baseline=False):
        '''
        Sets the unit_response to be the difference between odour and blank trials - currently uses the difference between the odour and a random blank, not the average
        '''
        blank_response = self.make_unit_response(trial_names_blanks, baseline=baseline, return_resp=True)
        odour_response = self.make_unit_response(trial_names_odour, baseline=baseline, return_resp=True)
        difference = []
        for i, j in zip(odour_response, blank_response):
            difference.append(i - j)
        self.unit_response = difference

    def window_classifier(self, trial_names, window_start, window_end, *, baseline=False, shuffle=False, sub_units=None, reassign_y_var=None):
        if self.unit_response is None:
            self.make_unit_response(trial_names, baseline=baseline)
        else:
            assert [i in self.trial_names for i in trial_names], 'Trial name passed not in classifiers response'

        bin_start = int((self.pre_trial_window + window_start)/self.bin_size)
        bin_end = int((self.pre_trial_window + window_end)/self.bin_size)

        window_unit_response = []
        y_var = []
        for trial in trial_names:
            trial_index = self.trial_names.index(trial)
            trial_response = self.unit_response[trial_index]
            summed_trial_response = np.sum(trial_response[:, :, bin_start:bin_end], axis=-1)
            window_unit_response.append(summed_trial_response)
            for i in range(len(summed_trial_response)):
                y_var.append(trial)
        if reassign_y_var:
            for i in reassign_y_var:
                y_var = [i[0] if j == i[1] else j for j in y_var]
        full_response = np.concatenate(window_unit_response, axis=0)

        # Runs if sub units is set to an int, only uses a random subsection of units
        if sub_units is not None:
            if sub_units > self.num_of_units:
                print('Sub unit count too high, reducing to number of units')
                sub_units = self.num_of_units
            random_units = []
            while len(random_units) < sub_units:
                r = np.random.randint(0, self.num_of_units)
                if r not in random_units:
                    random_units.append(r)
            random_units = np.array(random_units)
            full_response = full_response[:, random_units]

        if shuffle:
            self.shuffle = True
            random.shuffle(y_var)

        self.y_var = y_var
        X_train, X_test, y_train, y_test = train_test_split(full_response, y_var, test_size=self.test_size)

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
        self.y_var = y_var

    def find_accuracy(self):
        assert self.svm is not None, 'Please classify first'
        correct = 0
        for i, j in zip(self.svm.predict(self.X_test), self.y_test):
            if i == j:
                correct += 1
        self.accuracy = correct/len(self.y_test)
        return self.accuracy
