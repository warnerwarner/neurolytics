from recording import Recording
import numpy as np
import os
import openephys as oe
from scipy import signal
import matplotlib.pyplot as plt
from tqdm import tqdm


class Threshold_Crossing(Recording):

    def __init__(self, home_dir, channel_count, *, fs=30000, dat_name='100_CHs.dat', conversion_factor=0.195):
        Recording.__init__(self, home_dir, channel_count, fs)
        self.dat_name = dat_name
        self.conversion_factor=conversion_factor
        self.data = self._load_dat()
        self.threshold_crossings = None
        self.amplitudes = None

    def _load_dat(self):
        data = np.memmap(os.path.join(self.get_home_dir(), self.get_dat_name()), dtype=np.int16)
        try:
            assert len(data) % self.get_channel_count() == 0
        except (AssertionError):
            raise ValueError("Channel count and data length not comptable...")

        data = data.reshape(self.get_channel_count(), int(len(data)/self.get_channel_count()), order='F')
        return data

    def get_dat_name(self):
        return self.dat_name

    def get_data(self):
        return self.data



    def set_threshold_crossings(self, *, pol='neg', lim=4, inter_spike_window=1, method='quian', tqdm_on=True):
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
        spikes = []
        chan_count = 0
        for chan, threshold in zip(bp_data, thresholds):
            print('Finding spikes on chan %d...' % chan_count)
            chan_spikes = []
            prev_spike = -isw
            for time_index, val in tqdm(enumerate(chan)):
                if val > lim*threshold and time_index - prev_spike > isw:
                    #print(time_index)
                    spike_snip = chan[time_index:int(time_index+isw)]
                    spike_peak = np.argmax(spike_snip)
                    chan_spikes.append(time_index+spike_peak)
                    prev_spike = spike_peak + time_index
            spikes.append(chan_spikes)
            print('Found %d spikes on chan %d' % (len(chan_spikes), chan_count))
            chan_count += 1
        self.threshold_crossings = spikes
        print('Threshold crossings found and set!')

    def set_amplitudes(self, *, amplitude_type='minmax', pre_spike_window=1, post_spike_window=2):
        try:
            assert self.get_threshold_crossings is not None
        except (AssertionError):
            raise RuntimeError('Run set_threshold_crossings first')


        print('Finding amplitudes with %s' % amplitude_type)
        tcs = self.get_threshold_crossings()
        data = self.get_data()
        all_amps = []
        for chan_index, chan in enumerate(tcs):
            print('Finding amplitudes for chan %d' % chan_index)
            chan_amps = []
            for tc in chan:
                pre_spike = tc - int(pre_spike_window*self.get_fs()/1000)
                post_spike = tc + int(post_spike_window*self.get_fs()/1000)
                spike = data[chan_index, pre_spike:post_spike]
                if amplitude_type == 'minmax':
                    amplitude = max(spike) - min(spike)
                elif amplitude_type == 'median':
                    amplitude = abs(spike[pre_spike_window] - np.median(spike))
                elif amplitude_type == 'first':
                    amplitude = abs(spike[pre_spike_window] - spike[0])
                else:
                    raise ValueError('Incorrect amplitude type, try minmax, median, or first')
                chan_amps.append(amplitude)
            all_amps.append(chan_amps)
        self.amplitudes = all_amps

    def get_amplitudes(self, chan_num):
        return self.amplitudes[chan_num]

    def get_threshold_crossings(self):
        return self.threshold_crossings

    def get_firing_rate(self, chan_num, *, bin_size=1):
        tcs = self.get_threshold_crossings()[chan_num]
        data = self.get_data()
        histy, histx = np.histogram(tcs, bins=np.arange(0, len(data[0]), bin_size*self.get_fs()))
        histx = histx/self.get_fs()
        histy = histy/bin_size
        return histx[:-1], histy


    def plot_firing_rate(self, chan_num, *, start=0, end=None, bin_size=1):
        xs, firing_rate = self.get_firing_rate(chan_num, bin_size=bin_size)
        plt.plot(xs, firing_rate)
        plt.ylabel('Firing rate (Hz)')
        plt.xlabel('Time (s)')
        if end is None:
            plt.xlim(start, max(xs))
        else:
            plt.xlim(start, end)


def bandpass_data(data, *, lowcut=300, highcut=6000, fs=30000, order=3):
    nyq = 0.5*fs
    low = lowcut/nyq
    high = highcut/nyq
    sos = signal.butter(3, [low, high], analog=False, btype='band',
                        output='sos')
    y = signal.sosfiltfilt(sos, data)
    return y


tc = Threshold_Crossing("/Volumes/lab-schaefera/working/warnert/Recordings/jULIE recordings - 2019/Deep cortex recording/191017/2019-10-17_16-19-40/", 16, dat_name="2019-10-17_16-19-40_trimmed.dat")
tc.set_threshold_crossings()
print(len(tc.get_threshold_crossings()[-1]))
x, y  = tc.get_firing_rate(1)
tc.set_amplitudes()

plt.plot(x, y)
tc.plot_firing_rate(1, bin_size=0.3)

test = oe.loadContinuous2("/Volumes/lab-schaefera/working/warnert/Recordings/jULIE recordings - 2019/Deep cortex recording/191017/2019-10-17_16-11-58/100_CH16.continuous")
data = tc.get_data()
bp_data = bandpass_data(data[:, :100000])
print(bp_data.shape)
plt.plot(data[0, :1000]*0.195)
plt.plot(test['data'][:1000])
print(np.std(bp_data[0]), np.median(abs(bp_data[0])/0.6745), np.sqrt(np.mean([i**2 for i in bp_data[0]])))
print(np.median(abs(bp_data)/0.6745, axis=1).shape)
print(np.sqrt(np.mean([i**2 for i in bp_data[0]])), np.sqrt(np.mean(bp_data**2, axis=1))[0])
