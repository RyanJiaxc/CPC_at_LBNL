from torch.utils import data
import h5py


class Dataset(data.Dataset):
    def __init__(self, raw_file, list_file, audio_window):
        self.raw_file = raw_file
        self.audio_window = audio_window
        self.samples = []

        with open(list_file) as f:
            temp = f.readlines()
        temp = [x.strip() for x in temp]

        self.h5f = h5py.File(self.raw_file, 'r')
        # count = 0
        for i in temp:
            length = self.h5f[i].shape[0]
            if length >= audio_window:  # and count > 5100:
                self.samples.append(i)
            # count += 1

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample_id = self.samples[index]
        return self.h5f[sample_id][:]


class ValDataset(data.Dataset):
    def __init__(self, raw_file, list_file, audio_window):
        self.raw_file = raw_file
        self.audio_window = audio_window
        self.samples = []

        with open(list_file) as f:
            temp = f.readlines()
        temp = [x.strip() for x in temp]

        self.h5f = h5py.File(self.raw_file, 'r')
        count = 0
        for i in temp:
            length = self.h5f[i].shape[0]
            if length >= audio_window and count > 1200:
                self.samples.append(i)
            count += 1

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample_id = self.samples[index]
        return self.h5f[sample_id][:]