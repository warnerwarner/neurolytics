
import os
import numpy as np
from spiking import Cluster
import csv
from threshold_recording import Threshold_Recording, bandpass_data
import openephys as oe
from scipy.signal import find_peaks, resample
from tqdm import tqdm
import matplotlib.pyplot as plt

class Unit_Recording(Threshold_Recording):

    def __init__(self, home_dir, channel_count, trial_length, *, fs=30000, dat_name='100_CHs.dat', conversion_factor=0.195,  sniff_basis=False, sniff_locked=False):
        Threshold_Recording.__init__(self, home_dir, channel_count, fs=fs, dat_name=dat_name, conversion_factor=conversion_factor)
        self.channel_map = np.load(os.path.join(home_dir, 'channel_map.npy'))
        self.channel_positions = np.load(os.path.join(home_dir, 'channel_positions.npy'))
        self.clusters = self._find_clusters()
        self.sniff_basis = sniff_basis
        self.trial_length = trial_length
        self.trig_chan =  None
        self.trial_names = []
        self.resp_peaks = None
        self.resp_trace = None

    def _find_clusters(self):
        home_dir = self.home_dir
        spike_clusters = np.load(os.path.join(home_dir, 'spike_clusters.npy'))
        spike_templates = np.load(os.path.join(home_dir, 'spike_templates.npy'))
        spike_times = np.load(os.path.join(home_dir, 'spike_times.npy'))
        channel_map = self.channel_map
        templates = np.load(os.path.join(home_dir, 'templates.npy'))
        cluster_tsv = os.path.join(home_dir, 'cluster_group.tsv')
        tsv_read = csv.reader(open(cluster_tsv, 'r'), delimiter='\t')
        if os.path.isfile(os.path.join(home_dir, 'sniff_locked_spikes.npy')):
            print('Found sniff spikes')
            sniff_locked_spikes = np.load(os.path.join(home_dir, 'sniff_locked_spikes.npy'), allow_pickle=True)
            sniff_cluster_nums = np.load(os.path.join(home_dir, 'sniff_cluster_nums.npy'))
            sniff_spikes = True
        else:
            sniff_spikes = False
        next(tsv_read)
        clusters = []
        for cluster_row in tsv_read:
        # Find the cluster number and label
            cluster_num = int(cluster_row[0])
            c_label = cluster_row[1]
            if sniff_spikes:
                c_sniff_spikes = sniff_locked_spikes[(sniff_cluster_nums == cluster_num)][0]
            else:
                c_sniff_spikes = None

            # Find the times and templates
            c_times = spike_times[(spike_clusters == cluster_num)]
            c_temps_index = spike_templates[(spike_clusters == cluster_num)]
            c_temp_index = list(set(np.hstack(c_temps_index)))[0]
            c_temp = templates[c_temp_index]
            maxes = [max(abs(i)) for i in c_temp.T]
            max_chan = channel_map[np.argmax(maxes)]
            # Create a cluster and add to the ClusterSet
            cluster = Cluster(cluster_num, c_times, home_dir, c_label, c_temp_index,
                              c_temp, max_chan, sniff_lock_spikes=c_sniff_spikes)
            clusters.append(cluster)
        return clusters

    def _set_clusters(self, clusters):
        self.clusters = clusters

    def add_cluster(self, cluster):
        try:
            assert isinstance(cluster, Cluster)
            self.clusters.append(cluster)
        except(AssertionError):
            raise TypeError('New cluster must be of a Cluster object')
        plt.figure()

    def get_cluster(self, cluster_num):
        cluster = [i for i in self.clusters if i.cluster_num == cluster_num]
        return cluster[0]

    def get_good_clusters(self):
        clusters = [i for i in self.clusters if i.label == 'good']
        return clusters

    def get_non_noise_clusters(self):
        clusters = [i for i in self.clusters if i.label == 'good' or i.label == 'mua']
        return clusters

    def set(self,  pre_trial_window, *, resp_channel='100_ADC1.continuous'):
        print('Finding trial starts')
        self._find_trial_starts()
        print('Finding respiration peaks')
        self._find_respiration_peaks(resp_channel=resp_channel)
        print('Finding sniff locked avgs')
        self._find_all_sniff_lock_avg(pre_trial_window)
        print('Finding respiration trace')
        self._find_respiration_trace(resp_channel=resp_channel)



    def _find_trial_starts(self):
        if os.path.isfile(os.path.join(self.home_dir, 'trial_starts.npy')):
            print('Found file starts')
            self.trial_starts = np.load(os.path.join(self.home_dir, 'trial_starts.npy'))
        if os.path.isfile(os.path.join(self.home_dir, 'trial_ends.npy')):
            print('Found file ends')
            self.trial_ends = np.load(os.path.join(self.home_dir, 'trial_ends.npy'))

        if self.trial_starts is None or self.trial_ends is None:
            print('No starts or ends found')
            print('Finding trial starts using trigger of %s' % self.trig_chan)
            trig = oe.loadContinuous2(os.path.join(self.home_dir, self.trig_chan))['data']
            fs = self.fs
            trial_length = self.trial_length*fs
            prev_trial = -trial_length
            trial_starts = []
            for index, val in enumerate(trig):
                if val > 2 and index - prev_trial > trial_length:
                    trial_starts.append(index)
                    prev_trial = index

            trial_ends = [i+trial_length for i in trial_starts]
            self.trial_ends = np.array(trial_ends)
            self.trial_starts = np.array(trial_starts)
            print('Saving starts and ends')
            np.save(os.path.join(self.home_dir, 'trial_starts.npy'), trial_starts)
            np.save(os.path.join(self.home_dir, 'trial_ends.npy'), trial_starts)        

    def _find_respiration_peaks(self, *, resp_channel='100_ADC1.continuous'):
        if os.path.isfile(os.path.join(self.home_dir, 'respiration_peaks.npy')):
            print('Respiration peaks found')
            respiration_peaks = np.load(os.path.join(self.home_dir, 'respiration_peaks.npy'))
        else:
            print('Finding respiration peaks from raw file')
            resp = oe.loadContinuous2(os.path.join(self.home_dir, resp_channel))['data']
            bp_data = bandpass_data(resp, highcut=100, lowcut=1)
            respiration_peaks = find_peaks(bp_data, height=np.std(bp_data), prominence=np.std(bp_data))[0]
            np.save(os.path.join(self.home_dir, 'respiration_peaks.npy'), respiration_peaks)
        self.resp_peaks = respiration_peaks

    def _find_respiration_trace(self, *, resp_channel='100_ADC1.continuous'):
        if os.path.isfile(os.path.join(self.home_dir, 'respiration_trace.npy')):
            print('Respiration trace found')
            respiration_trace = np.load(os.path.join(self.home_dir, 'respiration_trace.npy'))
        else:
            print('Finding respiration trace')
            resp = oe.loadContinuous2(os.path.join(self.home_dir, resp_channel))['data']
            resp_snippets = [resample(resp[i:j], 10000) for i, j in tqdm(zip(self.resp_peaks[:-1], self.resp_peaks[1:]))]
            respiration_trace= np.mean(resp_snippets, axis=0)
            np.save(os.path.join(self.home_dir, 'respiration_trace.npy'), respiration_trace)
        self.resp_trace = respiration_trace

    def _find_all_sniff_lock_avg(self, pre_trial_window):
        # if bp_resp:
        #     print('Bandpassing respiration...')
        #     bp_resp = bandpass_data(resp_trace, lowcut=1, highcut=100)
        # else:
        #     bp_resp = resp_trace
        if not os.path.isfile(os.path.join(self.home_dir, 'sniff_cluster_nums.npy')):
            cluster_nums = []
            cluster_sniff_lockeds = []
            resp_peaks = self.resp_peaks
            #resp_peaks = find_peaks(bp_resp, height=np.std(bp_resp), prominence=np.std(bp_resp))[0]
            np.save(os.path.join(self.home_dir, 'resp_peaks.npy'), resp_peaks)
            print("found peaks!")
            for cluster in self.clusters:
                print('Finding sniff locked average for cluster %d' % cluster.cluster_num)
                cluster_nums.append(cluster.cluster_num)
                cluster_sniff_lockeds.append(self.get_sniff_lock_avg(cluster.cluster_num, pre_trial_window, resp_peaks))

            cluster_sniff_lockeds = np.array(cluster_sniff_lockeds)
            cluster_nums = np.array(cluster_nums)
            np.save(os.path.join(self.home_dir, 'sniff_cluster_nums.npy'), cluster_nums)
            np.save(os.path.join(self.home_dir, 'sniff_locked_spikes.npy'), cluster_sniff_lockeds)
        else:
            print('Found sniff locked avgs')



    def get_sniff_lock_avg(self, cluster, pre_trial_window, resp_peaks):
        if isinstance(cluster, (int, float)):
            cluster = self.get_cluster(cluster)            
        trial_starts = self.trial_starts
        spike_times = cluster.spike_times
        single_sniff_spikes = []
        for start in trial_starts:
            pre_trial_peaks = resp_peaks[(resp_peaks > start - pre_trial_window*self.fs) & (resp_peaks < start)]
            for i, j in zip(pre_trial_peaks[:-1], pre_trial_peaks[1:]):
                sniff_spikes = spike_times[(spike_times >= i) & (spike_times < j)]
                sniff_spikes = [(k-i)/(j-i) for k in sniff_spikes]
                single_sniff_spikes.append(sniff_spikes)
        single_sniff_spikes = np.hstack(single_sniff_spikes)
        return single_sniff_spikes





    def get_unique_trial_names(self):
        return list(set(self.trial_names))

    def get_unique_trial_starts(self, trial_name):
        try:
            assert trial_name in self.trial_names  # Don't know why this throws a warning, it does work
        except (AssertionError):
            raise ValueError('Trial name not in trial names')
        return [j for i, j in zip(self.trial_names, self.trial_starts) if i == trial_name]

    def get_cluster_trial_response(self, trial_name, cluster, *, pre_trial_window=None, post_trial_window=None, real_time=True):
        if isinstance(cluster, (int, float)):
            cluster = self.get_cluster(cluster)
        cluster_spikes = cluster.spike_times
        starts = self.get_unique_trial_starts(trial_name)
        cluster_trial_spikes = []
        sniff_basis = self.sniff_basis
        if pre_trial_window is None:
            pre_trial_window = 2*self.trial_length
        if post_trial_window is None:
            post_trial_window = 2*self.trial_length
        for start in starts:
            if not sniff_basis:
                window_start = start - pre_trial_window*self.fs
                window_end = start + (post_trial_window+self.trial_length)*self.fs
            else:
                window_start = int(start - pre_trial_window)
                window_end = int(start + post_trial_window)
            trial_spikes = cluster_spikes[(cluster_spikes >= window_start) & (cluster_spikes <= window_end)]
            trial_spikes = [i - start for i in trial_spikes]
            if real_time:
                trial_spikes = [i/self.fs for i in trial_spikes]
            cluster_trial_spikes.append(trial_spikes)
        return cluster_trial_spikes

    def get_all_trial_response(self, trial_name, *, only_good=True, pre_trial_window=0.5, post_trial_window=0.5):
        if only_good:
            clusters = self.get_good_clusters()
        else:
            clusters = self.get_non_noise_clusters()

        all_cluster_responses = [self.get_cluster_trial_response(trial_name,
                                                                 cluster.cluster_num,
                                                                 pre_trial_window=pre_trial_window,
                                                                 post_trial_window=post_trial_window) for cluster in clusters]
        return all_cluster_responses

    def get_all_binned_trial_response(self, trial_name, *, only_good=True, pre_trial_window=None, post_trial_window=None, bin_size=0.01, baselined=False):
        if only_good:
            clusters = self.get_good_clusters()
        else:
            clusters = self.get_non_noise_clusters()

        if baselined:
            all_cluster_responses = [self.get_binned_trial_response(trial_name,
                                                                    i.cluster_num,
                                                                    pre_trial_window=pre_trial_window,
                                                                    post_trial_window=post_trial_window,
                                                                    bin_size=bin_size,
                                                                    baselined=baselined)[1] for i in tqdm(clusters, leave=False)]
        else:
            all_cluster_responses = [self.get_binned_trial_response(trial_name,
                                                                    i.cluster_num,
                                                                    pre_trial_window=pre_trial_window,
                                                                    post_trial_window=post_trial_window,
                                                                    bin_size=bin_size,
                                                                    baselined=baselined)[1] for i in clusters]
        xs, _ = self.get_binned_trial_response(trial_name, clusters[0].cluster_num, pre_trial_window=pre_trial_window,
                                               post_trial_window=post_trial_window, bin_size=bin_size, baselined=False)
        return xs, all_cluster_responses




    def get_binned_trial_response(self, trial_name, cluster, *, pre_trial_window=None, post_trial_window=None, real_time=True, bin_size=0.01, baselined=True):
        if isinstance(cluster, (int, float)):
            cluster = self.get_cluster(cluster)

        if pre_trial_window is None:
            pre_trial_window = 2*self.trial_length
        if post_trial_window is None:
            post_trial_window = 2*self.trial_length

        if baselined:
            assert cluster.sniff_lock_spikes is not None, "No sniff locked spikes for cluster"
            assert self.resp_peaks is not None, "No respiration peaks"
            assert not self.sniff_basis, 'Sniff basis baseline not implemented as yet'

        cluster_spikes = cluster.spike_times
        sniff_locked_spikes = cluster.sniff_lock_spikes
        starts = self.get_unique_trial_starts(trial_name)
        cluster_trial_spikes = []
        resp_peaks = self.resp_peaks
        #base_hist = np.histogram(cluster.sniff_lock_spikes, bins=np.arange(0, 1.01, 0.01))
        for start in starts:
            window_start = int(start - pre_trial_window*self.fs)
            window_end = int(start + (self.trial_length + post_trial_window)*self.fs)
            trial_spikes = cluster_spikes[(cluster_spikes >= window_start) & (cluster_spikes <= window_end)]
            trial_spikes = [i - start for i in trial_spikes]
            if real_time:
                trial_spikes = [i/self.fs for i in trial_spikes]
            true_y, true_x = np.histogram(trial_spikes, bins=np.arange(-1*pre_trial_window, self.trial_length+post_trial_window+bin_size, bin_size))
            
            if baselined:
                assert pre_trial_window > 0, 'Cannot apply baseline subtraction with no baseline'
                trial_peaks = resp_peaks[(resp_peaks > window_start-pre_trial_window*self.fs) & (resp_peaks < window_end+post_trial_window*self.fs)]
                trial_peaks = [(i-start)/self.fs for i in trial_peaks]
                peak_diffs = np.diff(trial_peaks)
                fin_peak = trial_peaks[:-1]
                faux_trial_spikes = np.outer(peak_diffs, sniff_locked_spikes) + np.array(fin_peak)[:, np.newaxis]
                faux_trial_spikes = np.hstack(faux_trial_spikes)
                fauxy, fauxx = np.histogram(faux_trial_spikes, bins=np.arange(-1*pre_trial_window, self.trial_length+post_trial_window+bin_size, bin_size))
                if sum(true_y[:int(pre_trial_window/bin_size)]) == 0:
                    cf = 0
                else:
                    cf = 1/(sum(fauxy[:int(pre_trial_window/bin_size)])/sum(true_y[:int(pre_trial_window/bin_size)]))
                baselined_resp = true_y - fauxy*cf
                true_y = baselined_resp
            cluster_trial_spikes.append(true_y/bin_size)
            #cluster_trial_spikes.append(trial_spikes)
        return true_x, cluster_trial_spikes


    ##############################
    ########### BEWARE ###########
    ######## NOT DONE YET ########
    ### ENTER AT YOUR OWN RISK ###
    ##############################
    def cluster_plots(self, cluster, *, sniff_lock=True, fr_bin=60, spikes_shown=100, pre_spike_waveform=1, post_spike_waveform=2, **kwargs):
        '''
        Constucts a 2 x 2 set of figures that can be used to decribe the attributes of a given cluster. 

        Arguments:
        cluster - The cluster to use for construction

        Optional arguments:
        sniff_lock - I dont know what this does... but I'll leave it in for now
        fr_bin - The bin size (in seconds) for the firing rate plot
        spikes_shown - The number of example spikes to plot in the waveform plot
        pre_spike_waveform - The number of samples to take prior to the spike time
        post_spike_waveform - The number of samples to take post to the spike time        
        '''

        # If the cluster is a cluster then uses it, otherwise finds the cluster from the number
        if isinstance(cluster, (int, float)):
            cluster = self.get_cluster(cluster)
        fig, ax = plt.subplots(2, 2, **kwargs) # Make some plots 
        # The waveform plot
        print('Getting waveforms')
        self.waveform_plot(cluster, ax=ax[0, 0], spikes_shown=spikes_shown, pre_window=pre_spike_waveform, post_window=post_spike_waveform)
        
        # The firing rate
        print('Getting firing rate')
        self.firing_rate_plot(cluster, ax=ax[1, 0])

        # The autocorrelogram
        print('Getting autocorr')
        self.autocorrelogram_plot(cluster, ax=ax[0, 1])

        # The phase
        print('Getting phase')
        self.phase_plot(cluster, ax=ax[1, 1])

        return fig, ax

    def get_unit_waveforms(self, cluster, *, pre_window=1, post_window=2, zeroing='first'):
        '''
        Gets all the waveforms from a raw recording, very fast (at least on CAMP)

        Arguments:
        cluster: Which cluster to find the spikes from, can be a Cluster object, or a cluster number
        Optional arguments:
        pre_window: The window size prior to the spike time to include, in ms, default 1
        post_window: The window size post to the spike time to include, in ms, default 2
        zeroing: Method for removing offset, can be:
            - 'first', sets the first value from each spike snippet to zero, default
            - 'mean', sets the mean of each spike to zero
            - 'median', sets the median of each spike to zero
            - None, does nothing
        '''
        # Check if the cluster is a number, and if it is change it to an Cluster
        if isinstance(cluster, (int, float)):
            cluster = self.get_cluster(cluster)
        spike_times = cluster.spike_times

        # Open the data
        data = open(os.path.join(self.home_dir, self.dat_name), 'rb')
        prev_loc = 0
        waveforms = []

        # Find the size of the chunk (in samples) to extract
        chunk_size = int(self.channel_count*(pre_window + post_window) * self.fs/1000)

        # Offset is the size before the spike to find, this is in bytes
        offset_size = pre_window*self.fs/1000 * 2 * self.channel_count
        for i in tqdm(spike_times, leave=False):

            # Loads in the chunk of data from the binary file
            chunk = np.fromfile(data, count = chunk_size, dtype=np.int16, offset=int(64*i - offset_size) - prev_loc)
            # Find current location
            prev_loc = data.tell()
            # Append the chunk to the list
            waveforms.append(chunk)

        # Convert to an array, reshape, and change to 32 bit not 16 (stops overflows) 
        waveforms = np.array(waveforms)
        new_shape = (len(waveforms), int(chunk_size/self.channel_count), self.channel_count)
        waveforms = waveforms.reshape(new_shape, order='C')
        waveforms = waveforms.astype(np.int32) 
        
        # zero options, e.g. remove the offset for the snippets
        if zeroing == 'first':
            waveforms = waveforms - waveforms[:, 0, :][:, np.newaxis]  # Set the first value to zero
        elif zeroing == 'mean':
            waveforms = waveforms - np.mean(waveforms, axis=1)[:, np.newaxis]  # Remove the mean across the whole spike
        elif zeroing == 'median':
            waveforms = waveforms - np.median(waveforms, axis=1)[:, np.newaxis]  # Remove the median value from each spike
        elif zeroing is not None:
            print('Misunderstood zeroing method, waveforms wont be zeroed')
        data.close()  # Probs dont need to close, but might as well be good
        return waveforms


    def waveform_plot(self, cluster, *, ax = None, spikes_shown=100, pre_window=1, post_window=2, zeroing='first', channel='max'):
        '''
        Constructs a unit waveform plot, calls the get_unit_waveforms to find the waveforms

        Arguments:
        cluster - The cluster to construct the waveforms from
        Optional arguments:
        ax - The axis to plot on, if none will constuct their own
        spikes_shown - Number of spikes to 
        '''
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        if isinstance(cluster, (int, float)):
            cluster = self.get_cluster(cluster)

        waveforms = self.get_unit_waveforms(cluster, pre_window=pre_window, post_window=post_window, zeroing=zeroing)
        if channel != 'max':
            waveforms = waveforms[:, :, channel]
        else:
            waveforms = waveforms[:, :, cluster.max_chan]
        xs = np.arange(-pre_window, post_window, 1000/self.fs)
        if len(waveforms) <= spikes_shown:
            print('Too many spikes requested, reducing %d-->%d' % (spikes_shown, len(waveforms)))
            spikes_shown = len(waveforms)
        for i in range(spikes_shown):
            snip = waveforms[int(i*len(waveforms)/spikes_shown)]
            ax.plot(xs, snip*0.195, color='lightgray')
        ax.plot(xs, np.mean(waveforms, axis=0)*0.195, color='r')
        ax.set_ylabel('Voltage ($\mu$V)')
        ax.set_xlabel('Time (ms)')
        ax.set_xlim(-1, 2)

    def autocorrelogram_plot(self, cluster, *, ax=None, bin_size=1, window_size=50):
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        if isinstance(cluster, (int, float)):
            cluster = self.get_cluster(cluster)

        spike_times = cluster.spike_times

        diffs = []
        for i in tqdm(range(len(spike_times)), leave=False):
            diff = spike_times[i+1:] - spike_times[i]
            diffs.append(diff[diff <= bin_size*30*window_size])
        diffs = np.concatenate(diffs)
        diffs = np.concatenate([diffs, diffs * - 1])

        hists = np.histogram(diffs, bins=np.arange(-window_size*30, 30*(window_size+bin_size), bin_size*30))
        bar_lim = window_size - bin_size/2
        ax.bar(hists[1][:-1]/30, hists[0]/len(spike_times), width=bin_size, align='edge')
        ax.set_xlabel('Inter-spike interval (ms)')
        ax.set_ylabel('Normalised spike probability')
        ax.set_xlim(-bar_lim, bar_lim)
    
    def firing_rate_plot(self, cluster, *, ax=None, bin_size=60, min_base=True):
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        if isinstance(cluster, (int, float)):
            cluster = self.get_cluster(cluster)
        
        xs, fr = cluster.get_firing_rate(self.rec_length, bin_size=bin_size)
        if min_base:
            xs = xs/60
            xlim = self.rec_length/60
            basis = 'mins'
        else:
            xlim = self.rec_length
            basis = 's'
        ax.plot(xs, fr)
        ax.set_ylabel('Firing rate (Hz)')
        ax.set_xlabel('Time (%s)' % basis)
        ax.set_xlim(0, xlim)

    def phase_plot(self, cluster, *, ax=None, bin_num=100):
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        if isinstance(cluster, (int, float)):
            cluster = self.get_cluster(cluster)
        
        xs = np.arange(0, 1, 1/bin_num)
        ax2 = ax.twinx()
        ax2.set_yticks([])
        ax2.plot(xs, resample(self.resp_trace, bin_num), color='r')
        ax.plot(xs, np.histogram(cluster.sniff_lock_spikes, bins=np.arange(0, 1 + 1/bin_num, 1/bin_num))[0]/len(cluster.sniff_lock_spikes))
        ax.set_ylabel('Normalised spike probability')
        ax.set_xlabel('Sniff cycle phase')
        ax.set_xlim(0, 1)
        ax.set_ylim(0)


            



# if __name__ == '__main__':
#     cluster = Cluster([[1], [3], [4]], None, None, None, None, None, None, np.array([[10, 1, 5], [1, 3, 5]]))
#     eh = [1]
#     try:
#         assert isinstance(eh, Cluster)
#     except(AssertionError):
#         raise AssertionError('Need to add Cluster object')
