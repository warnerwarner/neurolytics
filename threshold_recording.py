import recording
import numpy as np

import openephys as oe
from scipy import signal
import matplotlib.pyplot as plt
import time
import os
from spiking import ThresholdCrossings
from copy import deepcopy
from tqdm import tqdm


class Threshold_Recording(recording.Recording):
    '''
    Finds and holds information about threshold crossings
    '''

    def __init__(self, home_dir, channel_count, *, fs=30000, dat_name='100_CHs.dat', conversion_factor=0.195):
        recording.Recording.__init__(self, home_dir, channel_count, fs=fs, dat_name=dat_name, conversion_factor=conversion_factor)
        self.threshold_crossings = []
        self.unique_spikes = []
        self.unique_spikes_window = None
        self.unique_spikes_chans = []

    def set(self, *, find_amps=True, tc_time_name='threshold_crossings.npy', tc_amp_name='threshold_amplitudes.npy',
            tc_chan_name='threshold_channel.npy', threshold_name='channel_threshold_indv.npy',
            tc_spike_threshold_name="spike_thresholds.npy", **kwargs):
        self.threshold_crossings = []
        if os.path.isfile(os.path.join(self.home_dir, tc_time_name)):
            print('Found tc times!')
            tcs = np.load(os.path.join(self.home_dir, tc_time_name))
            chans = np.load(os.path.join(self.home_dir, tc_chan_name))
            thresholds = np.load(os.path.join(self.home_dir, threshold_name))
            if os.path.isfile(os.path.join(self.home_dir, tc_amp_name)):
                print('Found tc amps!')
                amps = np.load(os.path.join(self.home_dir, tc_amp_name))
            else:
                print('Found no amps')
                amps = None
                if find_amps:
                    self.set_all_tcs_amplitudes(*kwargs)
            if os.path.isfile(os.path.join(self.home_dir, tc_spike_threshold_name)):
                print('Found spike thresholds!')
                spike_thresholds = np.load(os.path.join(self.home_dir, tc_spike_threshold_name))
            else:
                spike_thresholds = None
            for chan in range(self.channel_count):
                chan_tcs = tcs[(chans == chan)]
                threshold = thresholds[chan]
                chan_spike_thresholds = spike_thresholds[(chans == chan)]
                tc = ThresholdCrossings(chan_tcs,
                                        self.home_dir,
                                        self.channel_count,
                                        threshold*self.conversion_factor,
                                        spike_thresholds=chan_spike_thresholds)
                if amps is not None:
                    tc.amplitudes = amps[(chans == chan)]
                self.threshold_crossings.append(tc)
        else:
            print('No previous threshold crossing files found, finding thresholds from scratch')
            self.set_threshold_crossings()
            if find_amps:
                self.set_all_tcs_amplitudes(*kwargs)
        print('Threshold crossings found and set!')





    def set_threshold_crossings(self, *, pol='neg', lim=4, inter_spike_window=1, method='quian', bp_indiv_chans=False, return_bp=False):
        '''
        Set threshold crossings using the RecordingBase's data then creates ThresholdCrossing objects

        pol=neg - the polarity of the spikes, can be neg, pos, or both
        lim=4 - how many times over the std to set the threshold
        inter_spike_window=1 - the window in milliseconds between concurrent spikes, spike between this are lost ¯|_(ツ)_|¯
        method=quian - method to calculate the threshold, can be quain (median/0.6745 of data), std, or rms
        '''
        self.threshold_crossings = []
        print('Bandpassing data, this make take some time...')
        bp_data, whitened_data = bandpass_data(self.data, indiv_chans=bp_indiv_chans, preprocess=True)
        print('Threshold set by %s' % method)
        if method == 'std':
            thresholds = np.std(whitened_data, axis=1)
        elif method == 'quian':
            thresholds = []
            for chan in whitened_data:
                thresholds.append(np.median(abs(chan)/0.6745))
            thresholds = np.array(thresholds)
        elif method == 'rms':
            thresholds = np.sqrt(np.mean(whitened_data**2, axis=1))
        else:
            raise ValueError('Incorrect threshold crossing method, try std, quian, or rms')

        print('Searching for %s spikes' % pol)

        if pol == 'neg':
            whitened_data = -whitened_data
        elif pol == 'pos':
            whitened_data = whitened_data
        elif pol == 'both':
            whitened_data = abs(whitened_data)

        fs = self.fs

        isw = inter_spike_window*fs/1000
        tcs = []
        chan_count = 0
        times = []
        all_spikes = []
        all_thresholds = []
        all_chans = []
        all_spike_thresholds = []
        for chan, threshold in zip(whitened_data, thresholds):
            print('Finding spikes on chan %d...' % chan_count)
            chan_spikes = []
            chan_chans = []
            st = time.time()
            prev_spike = 0
            chan_spike_threshold = []
            for time_index, val in enumerate(chan):
                if val > lim*threshold and time_index - prev_spike > isw:

                    spike_snip = chan[time_index:int(time_index+isw)]
                    spike_peak = np.argmax(spike_snip)
                    post_spike_snip = chan[int(time_index+spike_peak):int(time_index+spike_peak+isw)]
                    if any(post_spike_snip) < lim*threshold:
                        chan_spikes.append(time_index+spike_peak)
                        prev_spike = spike_peak + time_index
                        chan_chans.append(chan_count)
                        chan_spike_threshold.append(max(spike_snip)/threshold)
            tt = time.time() - st
            times.append(tt)
            print('Found %d spikes on chan %d in %f s' % (len(chan_spikes), chan_count, tt))
            tc = ThresholdCrossings(chan_spikes, self.home_dir, chan_count,
                                    lim*threshold*self.conversion_factor,
                                    spike_thresholds=chan_spike_threshold)
            tcs.append(tc)
            chan_count += 1
            all_spikes.append(chan_spikes)
            all_chans.append(chan_chans)
            all_thresholds.append(threshold)
            all_spike_thresholds.append(chan_spike_threshold)
        all_chans = np.concatenate(all_chans)
        all_spikes = np.concatenate(all_spikes)
        all_spike_thresholds = np.concatenate(all_spike_thresholds)
        # all_thresholds = np.concatenate(all_thresholds)
        np.save(os.path.join(self.home_dir, 'threshold_crossings.npy'), all_spikes)
        np.save(os.path.join(self.home_dir, 'threshold_channel.npy'), all_chans)
        np.save(os.path.join(self.home_dir, 'channel_threshold_indv.npy'), all_thresholds)
        np.save(os.path.join(self.home_dir, 'spike_thresholds.npy'), all_spike_thresholds)
        self.threshold_crossings = tcs
        if return_bp:
            return bp_data

    def set_tc_amplitudes(self, channel_num, *, amplitude_type='minmax', pre_spike_window=1, post_spike_window=2,
                          bp_data=False):
        '''
        Set amplitudes for single threshold crossing channel

        channel_num - The channel number associated with the threshold crossing object
        amplitude_type=minmax - The way to measure the amplitude of the spike, can be minmax (the difference between highest and lowest point),
        median (the difference between the maximum spike point and the median of the spike), or first (the difference between the value at t=0 on the spike)
        pre_spike_window=1 - The window (in ms) to take before the spike peak
        post_spike_window=2 - The window (in ms) to take after the spike peak
        '''
        try:
            assert len(self.threshold_crossings) > 0
        except(AssertionError):
            raise RuntimeError('Set Tcs before finding their amplitudes')

        # Get all the variables
        tc = self.threshold_crossings[channel_num]
        spike_times = tc.spike_times
        if bp_data is None:
            print('Using raw data for amplitudes')
            data = self.data
        elif bp_data is True:
            print('Bandpassing data for amplitudes')
            data = bandpass_data(self.data)
        else:
            print('Using passed data for amplitudes')
            data = bp_data
        cf = self.conversion_factor

        all_amps = []
        for i in spike_times:
            pre_spike = i - pre_spike_window/1000*self.fs  # Find the start of the spike in samples
            post_spike = i + post_spike_window/1000*self.fs
            #print(pre_spike*self.fs, post_spike*self.fs)
            spike = data[channel_num, int(pre_spike):int(post_spike)]
            spike = spike.astype(np.int32)
            #print(pre_spike*self.fs, post_spike*self.fs)
            #print(spike)
            # Choose the amplitude type
            if len(spike) == 0:
                print(pre_spike, post_spike)
            if amplitude_type == 'minmax':
                amplitude = max(spike) - min(spike)
            elif amplitude_type == 'median':
                amplitude = abs(spike[pre_spike_window*self.fs/1000] - np.median(spike))
            elif amplitude_type == 'first':
                amplitude  = abs(spike[pre_spike_window*self.fs/1000] - spike[0])
            else:
                raise ValueError('Incorrect amplitude_type, can be minmax, median, or first')

            # Convert it with the conversion factor to microvolts
            all_amps.append(amplitude*cf)
        tc.amplitudes = all_amps

    def set_all_tcs_amplitudes(self, *, amplitude_type='minmax', pre_spike_window=1, post_spike_window=2, bp_data=False):
        '''
        Setting amplitude for all threshold crossings - calls set_tc_amplitudes

        amplitude_type=minmax - The way to measure the amplitude of the spike, can be minmax (the difference between highest and lowest point),
        median (the difference between the maximum spike point and the median of the spike), or first (the difference between the value at t=0 on the spike)
        pre_spike_window=1 - The window (in ms) to take before the spike peak
        post_spike_window=2 - The window (in ms) to take after the spike peak
        '''
        print('Finding amplitudes with %s' % amplitude_type)
        for chan_num in range(self.channel_count):
            print('Finding amplitudes for channel %d' % chan_num)
            self.set_tc_amplitudes(chan_num, amplitude_type=amplitude_type,
                                   pre_spike_window=pre_spike_window,
                                   post_spike_window=post_spike_window,
                                   bp_data=bp_data)

        all_amps = [i.amplitudes for i in self.threshold_crossings]
        np.save(os.path.join(self.home_dir, 'threshold_amplitudes.npy'), np.concatenate(all_amps))

    def plot_firing_rate(self, spiking_obj, *, ax=None, bin_size=1, start=0, end=None):
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        spiking_obj.plot_firing_rate(self.rec_length, ax=ax, bin_size=bin_size)
        if end is None:
            end = self.rec_length
        ax.set_xlim(start, end)

    def tc_waveforms(self, times_channel, *, pre_spike_window=1, post_spike_window=2, data_channel=None, bp_data=None):
        tc_times = self.threshold_crossings[times_channel].spike_times
        if bp_data is None:
            data = self.data
        elif bp_data is True:
            data = bandpass_data(self.data)
        else:
            data = bp_data
        spikes = []
        if data_channel is None:
            data_channel = times_channel
        elif data_channel == 'all':
            data_channel = range(self.channel_count)
        for spike_time in tc_times:
            spike = data[data_channel][int(spike_time-pre_spike_window/1000*self.fs):int(spike_time+post_spike_window/1000*self.fs)]
            if len(spike.shape) == 1:
                spike = spike - spike[0]
            else:
                spike = [i - i[0] for i in spike]
            spikes.append(spike*self.conversion_factor)

        return spikes

    def find_unique_spikes(self, *, isw=30, start_time=0, end_time=None,
                           chan_spike_matrix=True, out_dir=None,
                           unique_spikes_name='unique_spike_times.npy',
                           unique_channel_name='unique_spike_channels.npy'):
        unique_spike_times = deepcopy(self.threshold_crossings[0].spike_times)

        start_time = int(start_time*self.fs)
        if end_time is None:
            end_time = int(self.rec_length*self.fs)
        else:
            end_time = int(end_time*self.fs)

        unique_spike_times = unique_spike_times[(unique_spike_times >= start_time) & (unique_spike_times < end_time)]

        chan_spikes = np.zeros((len(unique_spike_times), self.channel_count))
        chan_spikes[:, 0] = 1
        print('Unique spikes found:%d' % len(unique_spike_times))

        for chan_index, chan in enumerate(self.threshold_crossings[1:]):
            cut_spikes = chan.spike_times[(chan.spike_times >= start_time) & (chan.spike_times < end_time)]
            for spike in tqdm(cut_spikes, leave=True):
                insert_index = np.searchsorted(unique_spike_times, spike, side='left')
                if insert_index != 0:
                    prev_dist = spike - unique_spike_times[insert_index -1]
                else:
                    prev_dist = np.inf

                if insert_index != len(unique_spike_times):
                    next_dist = unique_spike_times[insert_index] - spike
                else:
                    next_dist = np.inf
                if next_dist >= isw and prev_dist >= isw:
                    unique_spike_times = np.insert(unique_spike_times, insert_index, spike)
                    chan_binary = np.zeros(self.channel_count)
                    chan_binary[chan_index+1] = 1
                    chan_spikes = np.insert(chan_spikes, insert_index, chan_binary, axis=0)
                else:
                    chan_spikes[insert_index-1][chan_index+1] = 1
            print('Searched %d, found %d' % (chan_index+2, len(unique_spike_times)))
        self.unique_spikes = unique_spike_times
        self.unique_spikes_chans = chan_spikes
        self.unique_spikes_window = isw/self.fs*1000
        if out_dir is None:
            out_dir = self.home_dir
        np.save(os.path.join(out_dir, str(start_time)+'_'+str(end_time)+unique_spikes_name), unique_spike_times)
        np.save(os.path.join(out_dir, str(start_time)+'_'+str(end_time)+unique_channel_name), chan_spikes)



    # def plot_all_firing_rates_tcs(self, *, bin_size=1, start=0, end=None):
    #     '''
    #     Unfinished
    #     '''
    #     #fig = plt.figure(figsize=(10, 5))
    #     fig = plt.figure(figsize=(self.channel_count/2, self.channel_count))
    #     tcs = self.threshold_crossings
    #     ax1 = fig.add_subplot(111)
    #     #ax.plot([0, 0], [0, 100])
    #     ax1.grid(True)

    #     ax.set_xlim(0, self.get_rec_length())
    #     ax.set_yticklabels([])
    #     ax.set_xlabel('Time (s)')
    #     for chan in range(self.get_channel_count()):
    #         ax = fig.add_subplot(self.get_channel_count(), 1, chan+1)
    #         self.plot_firing_rate(tcs[chan],  ax=ax, start=start, end=end)
    #         ax.axis('off')



    # def plot_crossing_heatmap(self, *, bin_size=1, chans='All', scale=None, cmap='plasma'):
    #     '''
    #     Unfinished
    #     '''
    #     frs = []
    #     if chans == 'All':
    #         chans = range(self.get_channel_count())
    #     frs = [self.get_firing_rate(chan_num, bin_size=bin_size)[1] for chan_num in chans]
    #     frs = np.array(frs)
    #     print(frs.shape)
    #     if scale == 'log10':
    #         print('Scale set to log10')
    #         frs = np.log10(frs)
    #         frs[(frs == -np.inf)] = 0
    #     plt.figure(figsize=(10, 5))
    #     ax = sns.heatmap(frs, cmap=cmap)
    #     ax.invert_yaxis()
    #     plt.xlabel('Time (s)')
    #     plt.ylabel('Channels')




def bandpass_data(data, *, lowcut=300, highcut=6000, fs=30000, order=3, indiv_chans=False, preprocess=False):
    nyq = 0.5*fs
    low = lowcut/nyq
    high = highcut/nyq
    sos = signal.butter(3, [low, high], analog=False, btype='band', output='sos')
    if indiv_chans:
        print('Bandpassing individual channels')
        bp_data = []
        for index, i in enumerate(data):
            st = time.time()
            y = signal.sosfiltfilt(sos, i)
            bp_data.append(y)
            print('Bandpassed channel %d out of %d in' % (index, len(data)), time.time()-st)
        bp_data = np.array(bp_data)
    else:
        bp_data = signal.sosfiltfilt(sos, data)
    if preprocess:
        print('Preproccing data')
        y_process = preprocess_data(bp_data)
        return bp_data, y_process
    else:
        return bp_data


def preprocess_data(data):
    median_data = np.median(data, axis=0)
    referenced_data = data - median_data
    median_data = None
    print('CARed data')
    covariance_matrix = []
    for i in range(len(data)):
        row_cov = []
        for j in range(len(data)):
            row_cov.append(np.cov(referenced_data[i], referenced_data[j])[0, 1])
        covariance_matrix.append(row_cov)
    covariance_matrix = np.array(covariance_matrix)
    U, S, V = np.linalg.svd(covariance_matrix)
    whitening_const = 1e-5
    print('Found covariance_matrix')
    wzca = np.dot(U, np.dot(np.diag(1.0/np.sqrt(S + whitening_const)), V))
    covariance_matrix = None
    row_cov = None
    U = None
    S = None
    V = None
    whitened_data = np.dot(wzca, referenced_data)
    print('Whitened data')
    return whitened_data



# if __name__ == '__main__':

#     tc = Threshold_Crossing("/Volumes/lab-schaefera/working/warnert/Recordings/jULIE recordings - 2019/Deep cortex recording/191017/2019-10-17_16-19-40/", 16, dat_name="2019-10-17_16-19-40_trimmed.dat")
#     tc.set_threshold_crossings()
#     print(len(tc.get_threshold_crossings()[-1]))
#     x, y  = tc.get_firing_rate(1)
#     tc.set_amplitudes()
#     tc.plot_events()
#     plt.plot(x, y)
#     tc.plot_firing_rate(1, bin_size=0.1)
#     tc.plot_crossing_heatmap()
#     frs = [tc.get_firing_rate(chan_num)[1] for chan_num in range(tc.get_channel_count())]
#     frs = np.array(frs)
#     print(np.array(frs).shape)
#     plt.figure(figsize=(10, 4))
#     plt.imshow(frs[:, :30], )
#     plt.colorbar(fraction=0.046, pad=0.04)
#     plt.figure(figsize=(10, 5))
#     ax = sns.heatmap(log_10_frs, cmap='plasma')
#     ax.invert_yaxis()
#     log_10_frs = np.log10(frs)
#     print(np.max(frs), np.max(log_10_frs))
#     log_10_frs[(log_10_frs == -np.inf)] = 0
#     print(np.min(log_10_frs))
#     plt.xlabel('Time (s)')
#     plt.ylabel('Channel')
#     print(10**-0.08)
#     test = oe.loadContinuous2("/Volumes/lab-schaefera/working/warnert/Recordings/jULIE recordings - 2019/Deep cortex recording/191017/2019-10-17_16-11-58/100_CH16.continuous")
#     data = tc.get_data()
#     bp_data = bandpass_data(data[:, :100000])
#     print(bp_data.shape)
#     plt.plot(data[0, :1000]*0.195)
#     plt.plot(test['data'][:1000])
#     print(np.std(bp_data[0]), np.median(abs(bp_data[0])/0.6745), np.sqrt(np.mean([i**2 for i in bp_data[0]])))
#     print(np.median(abs(bp_data)/0.6745, axis=1).shape)
#     print(np.sqrt(np.mean([i**2 for i in bp_data[0]])), np.sqrt(np.mean(bp_data**2, axis=1))[0])
