from unit_recording import Unit_Recording
import os
import numpy as np
import openephys as oe
import pickle
import matplotlib.pyplot as plt

class PC_Recording(Unit_Recording):

    def __init__(self, home_dir, channel_count, trialbank_loc, *, trial_length=2, trig_binsize=0.05, trig_numbins=11, trig_chan='100_ADC2.continuous', **kwargs):

        Unit_Recording.__init__(self, home_dir, channel_count, trial_length, trig_chan=trig_chan,**kwargs)
        self.trialbank_loc = trialbank_loc
        self.trig_chan = trig_chan
        self.trig_binsize=trig_binsize
        self.trig_numbins = trig_numbins

        if os.path.isfile(os.path.join(self.home_dir, 'trial_names.npy')):
            self.trial_names = np.load(os.path.join(self.home_dir, 'trial_names.npy'))
            self.trial_bin_reps = np.load(os.path.join(self.home_dir, 'trial_bin_reps.npy'))
        else:
            self.extract_trial_names()
        
    
    def extract_trial_names(self):
        print('Did not find trial names, running extraction')
        trialbank = pickle.Unpickler(open(self.trialbank_loc, 'rb')).load()
        trial_names = [i[-1] for i in trialbank]
        trial_nums = [i[1][-1]['value_to_binarise'] for i in trialbank]
        trig_trace = oe.loadContinuous2(os.path.join(self.home_dir, self.trig_chan))['data']
        trial_snips = []
        trial_start = 0
        bin_width = self.fs*self.trig_binsize
        all_trial_names = []

        for trial_start in self.trial_starts:
            #print(int(trial_start-bin_width), int(trial_start+bin_width*(self.trig_numbins+1)))
            trial_snip = trig_trace[int(trial_start-bin_width):int(trial_start+bin_width*(self.trig_numbins+1))]
            #plt.plot(trial_snip)
            trial_snip[trial_snip > 1] = 1
            trial_snip[trial_snip < 1] = 0
            bin_trial_snip = []
            for i in np.linspace(bin_width, bin_width*(self.trig_numbins+1), self.trig_numbins, endpoint=False):
                #print(i)
                #print(np.mean(trial_snip[int(i):int(i+1500)]))
                if np.mean(trial_snip[int(i):int(i+bin_width)])> 0.5:
                    bin_trial_snip.append(1)
                else:
                    bin_trial_snip.append(0)
            trial_snips.append(bin_trial_snip)
            #print(bin_trial_snip)
            binary_num = int('0b'+''.join([str(i) for i in bin_trial_snip]), 2)
            #print(binary_num)
            all_trial_names.append(trial_names[np.where(np.array(trial_nums) == binary_num)[0][0]])

        self.trial_names = all_trial_names
        np.save(os.path.join(self.home_dir, 'trial_names.npy'), all_trial_names)
        self.trial_bin_reps = trial_snips
        np.save(os.path.join(self.home_dir, 'trial_bin_reps.npy'), trial_snips)
