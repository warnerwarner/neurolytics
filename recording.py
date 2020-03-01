import numpy as np
import os
from spiking import ThresholdCrossings
import time
from scipy import signal
import matplotlib.pyplot as plt

class RecordingBase():
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
        self.rec_length = len(self.data[0])/self.get_fs()
        self.threshold_crossings = []
        self.clusters = []

    def get_home_dir(self):
        return self.home_dir

    def get_dat_name(self):
        return self.dat_name

    def get_channel_count(self):
        return self.channel_count

    def get_fs(self):
        return self.fs

    def get_data(self):
        return self.data

    def get_conversion_factor(self):
        return self.conversion_factor

    def get_rec_length(self):
        return self.rec_length

    def _load_dat(self):
        '''
        Load the dat data file in as a memmap
        '''

        data = np.memmap(os.path.join(self.get_home_dir(), self.get_dat_name()), dtype=np.int16)
        try:
            assert len(data) % self.get_channel_count() == 0
        except (AssertionError):
            raise ValueError("Channel count and data length not comptable...")

        data = data.reshape(self.get_channel_count(), int(len(data)/self.get_channel_count()), order='F')
        return data

    def _set_threshold_crossings(self, tcs):
        self.threshold_crossings = tcs

    def set_threshold_crossings(self, *, pol='neg', lim=4, inter_spike_window=1, method='quian'):
        '''
        Set threshold crossings using the RecordingBase's data then creates ThresholdCrossing objects

        pol=neg - the polarity of the spikes, can be neg, pos, or both
        lim=4 - how many times over the std to set the threshold
        inter_spike_window=1 - the window in milliseconds between concurrent spikes, spike between this are lost ¯|_(ツ)_|¯
        method=quian - method to calculate the threshold, can be quain (median/0.6745 of data), std, or rms
        '''

        print('Bandpassing data, this make take some time...')
        bp_data = bandpass_data(self.get_data())
        print('Threshold set by %s' % method)
        if method == 'std':
            thresholds = np.std(bp_data, axis=1)
        elif method == 'quian':
            thresholds = np.median(abs(bp_data)/0.6745, axis=1)
        elif method == 'rms':
            thresholds = np.sqrt(np.mean(bp_data**2, axis=1))
        else:
            raise ValueError('Incorrect threshold crossing method, try std, quian, or rms')

        print('Searching for %s spikes' % pol)

        if pol == 'neg':
            bp_data = -bp_data
        elif pol == 'pos':
            bp_data = bp_data
        elif pol == 'both':
            bp_data = abs(bp_data)

        fs = self.get_fs()

        isw = inter_spike_window*fs/1000
        tcs = []
        chan_count = 0
        times = []
        for chan, threshold in zip(bp_data, thresholds):
            print('Finding spikes on chan %d...' % chan_count)
            chan_spikes = []
            st = time.time()
            prev_spike = -isw
            for time_index, val in enumerate(chan):
                if val > lim*threshold and time_index - prev_spike > isw:
                    spike_snip = chan[time_index:int(time_index+isw)]
                    spike_peak = np.argmax(spike_snip)
                    chan_spikes.append((time_index+spike_peak)/self.get_fs())  # Set it in seconds
                    prev_spike = spike_peak + time_index
            tt = time.time() - st
            times.append(tt)
            print('Found %d spikes on chan %d in %f s' % (len(chan_spikes), chan_count, tt))
            tc = ThresholdCrossings(chan_spikes, self.home_dir, chan_count, threshold*self.get_conversion_factor())
            tcs.append(tc)
            chan_count += 1
        self._set_threshold_crossings(tcs)
        print('Threshold crossings found and set!')

    def get_tcs(self):
        return self.threshold_crossings

    def set_tc_amplitudes(self, channel_num, *, amplitude_type='minmax', pre_spike_window=1, post_spike_window=2):
        '''
        Set amplitudes for single threshold crossing channel

        channel_num - The channel number associated with the threshold crossing object
        amplitude_type=minmax - The way to measure the amplitude of the spike, can be minmax (the difference between highest and lowest point),
        median (the difference between the maximum spike point and the median of the spike), or first (the difference between the value at t=0 on the spike)
        pre_spike_window=1 - The window (in ms) to take before the spike peak
        post_spike_window=2 - The window (in ms) to take after the spike peak
        '''
        try:
            assert len(self.get_tcs()) > 0
        except(AssertionError):
            raise RuntimeError('Set Tcs before finding their amplitudes')

        # Get all the variables
        tc = self.get_tcs()[channel_num]
        spike_times = tc.get_spike_times()
        data = self.get_data()
        cf = self.get_conversion_factor()

        all_amps = []
        for i in spike_times:
            pre_spike = int(i - pre_spike_window/1000)  # Convert to seconds, which the tcs are in
            post_spike = int(i + post_spike_window/1000)
            spike = data[channel_num, pre_spike*self.get_fs():post_spike*self.get_fs()]  # Convert back to samples to access the data

            # Choose the amplitude type
            if amplitude_type == 'minmax':
                amplitude = max(spike) - min(spike)
            elif amplitude_type == 'median':
                amplitude = abs(spike[pre_spike_window*self.get_fs()/1000] - np.median(spike))
            elif amplitude_type == 'first':
                amplitude  = abs(spike[pre_spike_window*self.get_fs()/1000] - spike[0])
            else:
                raise ValueError('Incorrect amplitude_type, can be minmax, median, or first')

            # Convert it with the conversion factor to microvolts
            all_amps.append(amplitude*cf)
        tc._set_amplitude(all_amps)

    def set_all_tcs_amplitudes(self, *, amplitude_type='minmax', pre_spike_window=1, post_spike_window=2):
        '''
        Setting amplitude for all threshold crossings - calls set_tc_amplitudes

        amplitude_type=minmax - The way to measure the amplitude of the spike, can be minmax (the difference between highest and lowest point),
        median (the difference between the maximum spike point and the median of the spike), or first (the difference between the value at t=0 on the spike)
        pre_spike_window=1 - The window (in ms) to take before the spike peak
        post_spike_window=2 - The window (in ms) to take after the spike peak
        '''
        print('Finding amplitudes with %s' % amplitude_type)
        for chan_num in range(self.channel_count()):
            self.set_tc_amplitudes(chan_num, amplitude_type, pre_spike_window, post_spike_window)

    def plot_firing_rate(self, spiking_obj, *, ax=None, bin_size=1, start=0, end=None):
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        spiking_obj.plot_firing_rate(self.get_rec_length(), ax=ax, bin_size=bin_size)
        if end is None:
            end = self.get_rec_length()
        ax.set_xlim(start, end)

    def plot_all_firing_rates_tcs(self, *, bin_size=1, start=0, end=None):
        '''
        Unfinished
        '''
        #fig = plt.figure(figsize=(10, 5))
        fig = plt.figure(figsize=(self.get_channel_count()/2, self.get_channel_count()))
        tcs = self.get_tcs()
        ax1 = fig.add_subplot(111)
        #ax.plot([0, 0], [0, 100])
        ax1.grid(True)

        ax.set_xlim(0, self.get_rec_length())
        ax.set_yticklabels([])
        ax.set_xlabel('Time (s)')
        for chan in range(self.get_channel_count()):
            ax = fig.add_subplot(self.get_channel_count(), 1, chan+1)
            self.plot_firing_rate(tcs[chan],  ax=ax, start=start, end=end)
            ax.axis('off')







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
    sos = signal.butter(3, [low, high], analog=False, btype='band',
                        output='sos')
    y = signal.sosfiltfilt(sos, data)
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
