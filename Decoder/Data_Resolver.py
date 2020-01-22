import h5py
import numpy as np


audio_window = 2000
train_raw = '/data1/ryan/dataset/training_new.h5'
train_list = '/data1/ryan/dataset/training_new.txt'
validation_raw = '/data1/ryan/dataset/validation_new.h5'
validation_list = '/data1/ryan/dataset/validation_new.txt'


train_h5f = h5py.File(train_raw, 'r')
samples = []

with open(train_list) as f:
    temp = f.readlines()
temp = [x.strip() for x in temp]

for i in temp:
    length = train_h5f[i].shape[0]
    if length >= audio_window:  # and count > 5100:
        samples.append(i)

train_file = []
for sample_id in samples:
    max_val = np.max(train_h5f[sample_id][:])
    train_file.append(train_h5f[sample_id][:] / max_val)

train_file = np.array(train_file)
np.save('/data1/ryan/train_file.npy', train_file)
