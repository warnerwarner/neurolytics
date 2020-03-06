import numpy as np
from tqdm import tqdm


class Cluster():
    def __init__(self, channel_map, channel_positions, cluster_num, times, recording_dir, label, template_ind, template):
        '''
        Arguments:
        channel_map: Map between the data file and the templates
        cluster_num: The number assigned to the cluster
        times: Recorded spike times for the cluster
        recording_dir: The directory the cluster was originally derived from
        label: The label assigned to the cluster, can be good, mua, or noise
        template_ind: The template index of the cluster
        template: The template assinged to the cluster from Kilosort
        '''
        self.channel_map = channel_map
        self.channel_positions = channel_positions
        self.cluster_num = cluster_num
        self.times = times
        self.recording_dir = recording_dir
        self.label = label
        self.template_ind = template_ind
        self.template = template
        self.max_chan = self._find_max_chan()
        self.amplitudes = None
        self.spikes_sniff_basis = None

    def __str__(self):
        '''
        Return string to print
        '''
        return self.get_label() + ' cluster ' + str(self.get_cluster_num())

    def _find_max_chan(self):
        '''
        Find the maximum template channel - used for amplitude calculations
        '''
        maxes = [max(abs(i)) for i in self.template.T]
        max_chan = self.channel_map[np.argmax(maxes)]
        return max_chan[0]

    def get_channel_map(self):
        return self.channel_map

    def get_channel_positions(self):
        return self.channel_positions

    def get_cluster_num(self):
        return self.cluster_num

    def get_spike_times(self):
        return self.times

    def get_recording_dir(self):
        return self.recording_dir

    def get_label(self):
        return self.label

    def get_template_ind(self):
        return self.template_ind

    def get_template(self):
        return self.template

    def get_max_chan(self):
        return self.max_chan

    def disp_info(self):
        '''
        Displays info on the cluster
        '''
        return 'Cluster: %d\nLabel: %s\nSpike count: %d\nMax chan: %d\nDir: %s\n' % (self.get_cluster_num(),
                                                                                     self.get_label(),
                                                                                     len(self.get_spike_times()),
                                                                                     self.get_max_chan(),
                                                                                     self.get_recording_dir())

    def _set_amplitudes(self, amplitudes):
        '''
        Set the cluster's amplitudes
        '''
        self.amplitudes = amplitudes

    def find_amplitudes(self, data, *, conversion_rate=0.195, pre_spike_window=30, post_spike_window=60, tqdm_on=True):
        '''
        Find the amplitudes of spikes assigned to the cluster using the difference between the min and max values of the spike

        Arguments:
        data: The data (C x t) the cluster was extracted from

        Optional arguments:
        conversion_rate: The conversion rate to transform from bits to volts, default 0.195
        pre_spike_window: The window before the spike time to take, in samples, default 30
        post_spike_window: The window after the spike time to take, in samples, default 60
        tqdm_on: If tqdm should be used to keep track of the amplitude assigning
        '''
        best_chan = data[self.get_max_chan()]
        amps = []
        if tqdm_on:
            for spike in tqdm(self.get_spike_times()):
                spike = best_chan[int(spike - pre_spike_window):int(spike + post_spike_window)]
                amp = max(spike) - min(spike)
                amps.append(amp*conversion_rate)
        else:
            for spike in self.get_spike_times():
                spike = best_chan[int(spike - pre_spike_window):int(spike + post_spike_window)]
                amp = max(spike) - min(spike)
                amps.append(amp*conversion_rate)
        self._set_amplitudes(amps)

    def get_amplitudes(self):
        return self.amplitudes

    def _set_spikes_sniff_basis(self, spikes_sniff_basis):
        self.spikes_sniff_basis = spikes_sniff_basis

    def find_spikes_sniff_basis(self, respiration_peaks):
        clus_spikes = self.get_spike_times()
        spikes_sb = []
        for i in range(len(respiration_peaks) - 1):
            sniff_spikes = clus_spikes[(clus_spikes > respiration_peaks[i]) & (clus_spikes < respiration_peaks[i+1])]
            sniff_spikes = [(j - respiration_peaks[i])/(respiration_peaks[i+1] - respiration_peaks[i]) for j in sniff_spikes]
            sniff_spikes = [j + i for j in sniff_spikes]
            spikes_sb.append(sniff_spikes)
        spikes_sb = np.hstack(spikes_sb)
        self._set_spikes_sniff_basis(spikes_sb)

    def get_spikes_sniff_basis(self):
        return self.spikes_sniff_basis

    def get_trial_spike_times(self, trial_starts, *, trial_length=2, pre_trial_window=2, post_trial_window=2, fs=30000):
        spike_times = self.get_spike_times()
        trial_spike_times = []
        for start in trial_starts:
            single_trial_spike_times = spike_times[(spike_times > fs*(start-pre_trial_window)) & (spike_times < fs*(start+trial_length+post_trial_window))]
            trial_spike_times.append(single_trial_spike_times)
        return trial_spike_times

    def get_trial_spike_times_sb(self, trial_starts_sb, *, pre_window_sniffs=5, post_window_sniffs=10):
        spike_times_sb = self.get_spikes_sniff_basis()
        trial_spikes_times_sb = []
        for start in trial_starts_sb:
            single_trial_spike_times_sb = spike_times_sb[(spike_times_sb > start - pre_window_sniffs) & (spike_times_sb < start + post_window_sniffs)]
            trial_spikes_times_sb.append(single_trial_spike_times_sb)
        return trial_spikes_times_sb

    def get_waveforms(self, data, pre_spike_time=30, post_spike_time=60, single_chan=False):
        spike_times = self.get_spike_times()
        spikes = []
        for i in tqdm(spike_times):
            if single_chan:
                spike = data[int(i-pre_spike_time):int(i+post_spike_time)]
                spike = spike - spike[0]
            else:
                spike = data[:, int(i-pre_spike_time):int(i+post_spike_time)]
                spike = np.subtract(spike.T, spike[:, 0])
                spike = spike.T
            spikes.append(spike)
        return np.array(spikes)
