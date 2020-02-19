import numpy as np
import os
from spiking import ThresholdCrossings
import time
from scipy import signal

class RecordingBase():
    '''
    A bit of a mess right now, base for recording objects
    Holds general information about a recording
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

    def _load_dat(self):
        '''
        Load the dat data file in
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

    def set_threshold_crossings(self, *, pol='neg', lim=4, inter_spike_window=1, method='quian', tqdm_on=True):
        '''
        Set threshold crossings using the RecordingBase's data then creates ThresholdCrossing objects
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
                    #print(time_index)
                    spike_snip = chan[time_index:int(time_index+isw)]
                    spike_peak = np.argmax(spike_snip)
                    chan_spikes.append((time_index+spike_peak)/self.get_fs())
                    prev_spike = spike_peak + time_index
            tt = time.time() - st
            times.append(tt)
            print('Found %d spikes on chan %d in %f s' % (len(chan_spikes), chan_count, tt))
            tc = ThresholdCrossings(chan_spikes, self.home_dir, chan_count, threshold)
            tcs.append(tc)
            chan_count += 1
        self._set_threshold_crossings(tcs)
        print('Threshold crossings found and set!')

    def get_threshold_crossings(self):
        return self.threshold_crossings

    def set_tc_amplitudes(self):
        tcs = self.get_threshold_crossings()
        for tc in tcs:
            tc.set_amplitude(self.get_data())


    def set_tc_amplitudes(self, *, amplitude_type='minmax', pre_spike_window=1, post_spike_window=2):
        '''
        Setting amplitude for threshold crossings - currently UNFINISHED
        '''
        print('Finding amplitudes with %s' % amplitude_type)
        spike_times = self.get_spike_times()
        all_amps = []
        for i in spike_times:

            pre_spike = int((i - pre_spike_window/1000)*fs)
            post_spike = int((i + post_spike_window/1000)*fs)
            spike = data[max_chan, pre_spike:post_spike]
            if amplitude_type == 'minmax':
                amplitude = max(spike) - min(spike)
            elif amplitude_type == 'median':
                amplitude = abs(spike[pre_spike_window] - np.median(spike))
            elif amplitude_type == 'first':
                amplitude = abs(spike[pre_spike_window] - spike[0])
            else:
                raise ValueError('Incorrect amplitude type, try minmax, median, or first')
            all_amps.append(amplitude)
        self.amplitudes = all_amps

def bandpass_data(data, *, lowcut=300, highcut=6000, fs=30000, order=3):
    nyq = 0.5*fs
    low = lowcut/nyq
    high = highcut/nyq
    sos = signal.butter(3, [low, high], analog=False, btype='band',
                        output='sos')
    y = signal.sosfiltfilt(sos, data)
    return y


rec = RecordingBase("/Volumes/lab-schaefera/working/warnert/Recordings/jULIE recordings - 2019/Deep cortex recording/191017/2019-10-17_16-22-56/", 16, dat_name="2019-10-17_16-22-56_trimmed.dat")


rec.set_threshold_crossings()
tc = rec.get_threshold_crossings()[0]
tc.set_amplitude(rec.get_data())
print(tc.get_spike_times()[5])
print(rec.get_data().shape)
print(len(tc.get_amplitudes()))
