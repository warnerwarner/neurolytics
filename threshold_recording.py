import recording
import numpy as np

import openephys as oe
from scipy import signal
import matplotlib.pyplot as plt
import time
import seaborn as sns
from spiking import ThresholdCrossings


class Threshold_Recording(recording.Recording):
    '''
    Finds and holds information about threshold crossings
    '''

    def __init__(self, home_dir, channel_count, *, fs=30000, dat_name='100_CHs.dat', conversion_factor=0.195):
        recording.Recording.__init__(self, home_dir, channel_count, fs=fs, dat_name=dat_name, conversion_factor=conversion_factor)
        self.threshold_crossings = []

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



    def plot_crossing_heatmap(self, *, bin_size=1, chans='All', scale=None, cmap='plasma'):
        frs = []
        if chans == 'All':
            chans = range(self.get_channel_count())
        frs = [self.get_firing_rate(chan_num, bin_size=bin_size)[1] for chan_num in chans]
        frs = np.array(frs)
        print(frs.shape)
        if scale == 'log10':
            print('Scale set to log10')
            frs = np.log10(frs)
            frs[(frs == -np.inf)] = 0
        plt.figure(figsize=(10, 5))
        ax = sns.heatmap(frs, cmap=cmap)
        ax.invert_yaxis()
        plt.xlabel('Time (s)')
        plt.ylabel('Channels')




def bandpass_data(data, *, lowcut=300, highcut=6000, fs=30000, order=3):
    nyq = 0.5*fs
    low = lowcut/nyq
    high = highcut/nyq
    sos = signal.butter(3, [low, high], analog=False, btype='band',
                        output='sos')
    y = signal.sosfiltfilt(sos, data)
    return y



if __name__ == '__main__':

    tc = Threshold_Crossing("/Volumes/lab-schaefera/working/warnert/Recordings/jULIE recordings - 2019/Deep cortex recording/191017/2019-10-17_16-19-40/", 16, dat_name="2019-10-17_16-19-40_trimmed.dat")
    tc.set_threshold_crossings()
    print(len(tc.get_threshold_crossings()[-1]))
    x, y  = tc.get_firing_rate(1)
    tc.set_amplitudes()
    tc.plot_events()
    plt.plot(x, y)
    tc.plot_firing_rate(1, bin_size=0.1)
    tc.plot_crossing_heatmap()
    frs = [tc.get_firing_rate(chan_num)[1] for chan_num in range(tc.get_channel_count())]
    frs = np.array(frs)
    print(np.array(frs).shape)
    plt.figure(figsize=(10, 4))
    plt.imshow(frs[:, :30], )
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.figure(figsize=(10, 5))
    ax = sns.heatmap(log_10_frs, cmap='plasma')
    ax.invert_yaxis()
    log_10_frs = np.log10(frs)
    print(np.max(frs), np.max(log_10_frs))
    log_10_frs[(log_10_frs == -np.inf)] = 0
    print(np.min(log_10_frs))
    plt.xlabel('Time (s)')
    plt.ylabel('Channel')
    print(10**-0.08)
    test = oe.loadContinuous2("/Volumes/lab-schaefera/working/warnert/Recordings/jULIE recordings - 2019/Deep cortex recording/191017/2019-10-17_16-11-58/100_CH16.continuous")
    data = tc.get_data()
    bp_data = bandpass_data(data[:, :100000])
    print(bp_data.shape)
    plt.plot(data[0, :1000]*0.195)
    plt.plot(test['data'][:1000])
    print(np.std(bp_data[0]), np.median(abs(bp_data[0])/0.6745), np.sqrt(np.mean([i**2 for i in bp_data[0]])))
    print(np.median(abs(bp_data)/0.6745, axis=1).shape)
    print(np.sqrt(np.mean([i**2 for i in bp_data[0]])), np.sqrt(np.mean(bp_data**2, axis=1))[0])
