import numpy as np
import os
from spiking import ThresholdCrossings
import time
from scipy import signal
import matplotlib.pyplot as plt

class Recording():
    '''
    A bit of a mess right now, base for recording objects
    Holds general information about a recording

    Arguments:
    home_dir - The directory of the recording, expected to contain the binary recording file
    channel_count - The number of channels in the recording
    fs=30000 - the sampling rate
    dat_name=100_CHs.dat - the name of the dat file containing the raw recording
    conversion_factor=0.195 - the bitvolt conversion factor from the raw data
    '''

    def __init__(self, home_dir, channel_count, *, fs=30000, dat_name='100_CHs.dat',
                 resp_channel='100_ADC6.continuous', conversion_factor=0.195):
        self.home_dir = home_dir
        self.channel_count = channel_count
        self.fs = fs
        self.dat_name = dat_name
        self.conversion_factor = conversion_factor
        self.data = self.load_dat()
        self.rec_length = len(self.data[0])/self.fs
        self.resp_channel = resp_channel

    def load_dat(self):
        '''
        Load the dat data file in as a memmap
        '''

        data = np.memmap(os.path.join(self.home_dir, self.dat_name), dtype=np.int16)
        try:
            assert len(data) % self.channel_count == 0
        except (AssertionError):
            raise ValueError("Channel count and data length not comptable...")

        data = data.reshape(self.channel_count, int(len(data)/self.channel_count), order='F')
        return data



def bandpass_data(data, *, lowcut=300, highcut=6000, fs=30000, order=3):
    '''
    Bandpass data using a butterworth filter

    data - Raw data to filter - can be ndarray
    lowcut=300 - Lowcut frequency
    highcut=6000 - Highcut frequency
    fs=30000 - Sampling rate
    order=3 - Order for the filter
    '''
    nyq = 0.5*fs
    low = lowcut/nyq
    high = highcut/nyq
    b, a = signal.butter(3, [low, high], analog=False, btype='band')
    y = signal.filtfilt(b, a, data)
    return y

