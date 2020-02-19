import numpy as np


class Spiking():
    '''
    Superclass for any object that contains spikes

    Hold spike times, and recording directory of spiking object
    '''
    def __init__(self, times, recording_dir):
        self.times = times
        self.recording_dir = recording_dir
        self.amplitudes = None

    def get_spike_times(self):
        return self.times

    def get_recording_dir(self):
        return self.recording_dir

    def set_amplitude(self, data, max_chan, *, amplitude_type='minmax', pre_spike_window=1, post_spike_window=2, fs=30000, conversion_factor=0.195):
        '''
        Set the amplitude of the spiking object - is being moved to RecordingBase to be more general, do not use!
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

    def get_amplitudes(self):
        return self.amplitudes

class Cluster(Spiking):
    '''
    Holds information about Kilosorted clusters, a subclass of Spiking

    Information includes cluster number, spiking times, the recording dir, cluster label and templates
    '''

    def __init__(self, cluster_num, times, recording_dir, label, template_ind, template):
        Spiking.__init__(self, times, recording_dir)
        self.cluster_num = cluster_num
        self.label = label
        self.template_ind = template_ind
        self.template = template
        self.max_chan = self._find_max_chan()

    def _find_max_chan(self):
        '''
        Find the maximum template channel - used for amplitude calculations
        '''
        maxes = [max(abs(i)) for i in self.template.T]
        max_chan = self.channel_map[np.argmax(maxes)]
        return max_chan[0]

    def get_max_chan(self):
        return self.max_chan

    def set_amplitude(self, data):
        Spiking.set_amplitude(self, data, self.get_max_chan())

class ThresholdCrossings(Spiking):
    '''
    Represent detected threshold crossings on a single channel

    Contains the spike times, recording directory, the channel number in the orignal recording, and the threshold used
    '''

    def __init__(self, times, recording_dir, channel_num, threshold):
        Spiking.__init__(self, times, recording_dir)
        self.channel_num = channel_num
        self.threshold = threshold

    def get_channel_num(self):
        return self.channel_num

    def set_amplitude(self, data):
        Spiking.set_amplitude(self, data, self.get_channel_num())
