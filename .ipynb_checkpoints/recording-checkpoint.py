import numpy as np
import os
from spiking import ThresholdCrossings
import time
from scipy import signal
import matplotlib.pyplot as plt

class Recording():
    '''
    Superclass for other recording classes, can be used as a default
    '''

    def __init__(self, home_dir, channel_count, *, fs=30000, dat_name='100_CHs.dat',
                 resp_channel='100_ADC6.continuous', conversion_factor=0.195):
        """
        Args:
            home_dir (str): Location of the directory the recording data is in
            channel_count (int): Number of channels used in the recording
            fs (int, optional): Sampling rate of the recording. Defaults to 30000.
            dat_name (str, optional): Name of the dat file of the recording. Defaults to '100_CHs.dat'.
            resp_channel (str, optional): Name of the channel containing the respiration trace. Defaults to '100_ADC6.continuous'.
            conversion_factor (float, optional): Value to convert between binary to volts. Defaults to 0.195.
        """
        self.home_dir = home_dir
        self.channel_count = channel_count
        self.fs = fs
        self.dat_name = dat_name
        self.conversion_factor = conversion_factor
        self.data = self.load_dat()
        self.rec_length = len(self.data[0])/self.fs
        self.resp_channel = resp_channel

    def load_dat(self):
        """
        Loads the dat into a memmap
        """

        data = np.memmap(os.path.join(self.home_dir, self.dat_name), dtype=np.int16)
        try:
            assert len(data) % self.channel_count == 0
        except (AssertionError):
            raise ValueError("Channel count and data length not comptable...")

        data = data.reshape(self.channel_count, int(len(data)/self.channel_count), order='F')
        return data



def bandpass_data(data, *, lowcut=300, highcut=6000, fs=30000, order=3):
    """
    Bandpass data using a butterworth filter

    Args:
        data (array): Data to bandpass
        lowcut (int, optional): Lowcut value for filter. Defaults to 300.
        highcut (int, optional): Highcut value for the filter. Defaults to 6000.
        fs (int, optional): Sampling rate of the data. Defaults to 30000.
        order (int, optional): Order of the butterworth filter, high is a stronger filtering. Defaults to 3.

    Returns:
        y (array): The bandpassed signal
    """
    nyq = 0.5*fs
    low = lowcut/nyq
    high = highcut/nyq
    b, a = signal.butter(3, [low, high], analog=False, btype='band')
    y = signal.filtfilt(b, a, data)
    return y

