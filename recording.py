import numpy as np


class Recording():

    def __init__(self, home_dir, channel_count, fs):
        self.home_dir = home_dir
        self.channel_count = channel_count
        self.fs = fs

    def get_home_dir(self):
        return self.home_dir

    def get_channel_count(self):
        return self.channel_count

    def get_fs(self):
        return self.fs
