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

    def __init__(self, home_dir, channel_count, *, fs=30000, dat_name='100_CHs.dat', conversion_factor=0.195):
        self.home_dir = home_dir
        self.channel_count = channel_count
        self.fs = fs
        self.dat_name = dat_name
        self.conversion_factor = conversion_factor
        self.data = self._load_dat()
        self.rec_length = len(self.data[0])/self.fs

    def _load_dat(self):
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

if __name__ == '__main__':

    rec = RecordingBase("/Volumes/lab-schaefera/working/warnert/Recordings/jULIE recordings - 2019/Deep cortex recording/191017/2019-10-17_16-22-56/", 16, dat_name="2019-10-17_16-22-56_trimmed.dat")


    rec.set_threshold_crossings()
    tc = rec.get_tcs()[0]
    tc.set_amplitude(rec.get_data())
    print(tc.get_spike_times()[5])
    print(rec.get_data().shape)
    print(len(tc.get_amplitudes()))
    fig = plt.figure()
    ax = fig.add_subplot(16, 1, 1)
    ax.plot([1, 2])
    ax = fig.add_subplot(16, 1, 2)
    ax.plot([1, 2])
    plt.tight_layout()
    rec.plot_all_firing_rates_tcs()
    fig.set_size_inches(1, 20)

    plt.xlabel('Time (s)')
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111)
    print(type(ax))
    ys, xs = np.histogram(tc.get_spike_times(), bins=np.arange(0, rec.get_rec_length(), 1))
    ax.plot(xs[:-1], ys/1)
    ax.set_xlim(0, rec.get_rec_length())
    ax.set_ylim(0, )
    ax.set_ylabel('Firing rate (Hz)')
    ax.set_xlabel('Time (s)')
    plt.show()
