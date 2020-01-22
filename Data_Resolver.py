import h5py
import numpy as np


# Initialize files
audio_window = 2000

train_raw_q001_top = '/data1/ryan/dataset/training_q001_top.h5'
train_raw_q001_bot = '/data1/ryan/dataset/training_q001_bot.h5'
train_raw_q003_top = '/data1/ryan/dataset/training_q003_top.h5'
train_raw_q003_bot = '/data1/ryan/dataset/training_q003_bot.h5'
train_raw_q076_top = '/data1/ryan/dataset/training_q076_top.h5'
train_raw_q076_bot = '/data1/ryan/dataset/training_q076_bot.h5'
train_raw_q103_top = '/data1/ryan/dataset/training_q103_top.h5'
train_raw_q103_bot = '/data1/ryan/dataset/training_q103_bot.h5'

q001_t_top = np.load('/data1/ryan/dataset/time_q001_top.npy')
q001_t_bot = np.load('/data1/ryan/dataset/time_q001_bot.npy')
q003_t_top = np.load('/data1/ryan/dataset/time_q003_top.npy')
q003_t_bot = np.load('/data1/ryan/dataset/time_q003_bot.npy')
q076_t_top = np.load('/data1/ryan/dataset/time_q076_top.npy')
q076_t_bot = np.load('/data1/ryan/dataset/time_q076_bot.npy')
q103_t_top = np.load('/data1/ryan/dataset/time_q103_top.npy')
q103_t_bot = np.load('/data1/ryan/dataset/time_q103_bot.npy')

name_to_use_q001 = 'Q001'
name_to_use_q003 = 'Q003'
name_to_use_q076 = 'Q076'
name_to_use_q103 = 'Q103'

train_q001_top_h5f = h5py.File(train_raw_q001_top, 'r')
train_q001_bot_h5f = h5py.File(train_raw_q001_bot, 'r')
train_q003_top_h5f = h5py.File(train_raw_q003_top, 'r')
train_q003_bot_h5f = h5py.File(train_raw_q003_bot, 'r')
train_q076_top_h5f = h5py.File(train_raw_q076_top, 'r')
train_q076_bot_h5f = h5py.File(train_raw_q076_bot, 'r')
train_q103_top_h5f = h5py.File(train_raw_q103_top, 'r')
train_q103_bot_h5f = h5py.File(train_raw_q103_bot, 'r')


# Q001
samples_top = []
samples_bot = []
temp_top = [name_to_use_q001 + '_' + str(t) for t in q001_t_top]
temp_bot = [name_to_use_q001 + '_' + str(t) for t in q001_t_bot]

for i in temp_top:
    length = train_q001_top_h5f[i].shape[0]
    if length >= audio_window:  # and count > 5100:
        samples_top.append(i)
for i in temp_bot:
    length = train_q001_bot_h5f[i].shape[0]
    if length >= audio_window:  # and count > 5100:
        samples_bot.append(i)

train_file_top = []
train_file_bot = []
for sample_id in samples_top:
    train_file_top.append(train_q001_top_h5f[sample_id][:])
for sample_id in samples_bot:
    train_file_bot.append(train_q001_bot_h5f[sample_id][:])

train_file_top = np.array(train_file_top)
train_file_bot = np.array(train_file_bot)
np.save('/data1/ryan/train-files/train_q001_top.npy', train_file_top)
np.save('/data1/ryan/train-files/train_q001_bot.npy', train_file_bot)


# Q003
samples_top = []
samples_bot = []
temp_top = [name_to_use_q003 + '_' + str(t) for t in q003_t_top]
temp_bot = [name_to_use_q003 + '_' + str(t) for t in q003_t_bot]

for i in temp_top:
    length = train_q003_top_h5f[i].shape[0]
    if length >= audio_window:  # and count > 5100:
        samples_top.append(i)
for i in temp_bot:
    length = train_q003_bot_h5f[i].shape[0]
    if length >= audio_window:  # and count > 5100:
        samples_bot.append(i)

train_file_top = []
train_file_bot = []
for sample_id in samples_top:
    train_file_top.append(train_q003_top_h5f[sample_id][:])
for sample_id in samples_bot:
    train_file_bot.append(train_q003_bot_h5f[sample_id][:])

train_file_top = np.array(train_file_top)
train_file_bot = np.array(train_file_bot)
np.save('/data1/ryan/train-files/train_q003_top.npy', train_file_top)
np.save('/data1/ryan/train-files/train_q003_bot.npy', train_file_bot)


# Q076
samples_top = []
samples_bot = []
temp_top = [name_to_use_q076 + '_' + str(t) for t in q076_t_top]
temp_bot = [name_to_use_q076 + '_' + str(t) for t in q076_t_bot]

for i in temp_top:
    length = train_q076_top_h5f[i].shape[0]
    if length >= audio_window:  # and count > 5100:
        samples_top.append(i)
for i in temp_bot:
    length = train_q076_bot_h5f[i].shape[0]
    if length >= audio_window:  # and count > 5100:
        samples_bot.append(i)

train_file_top = []
train_file_bot = []
for sample_id in samples_top:
    train_file_top.append(train_q076_top_h5f[sample_id][:])
for sample_id in samples_bot:
    train_file_bot.append(train_q076_bot_h5f[sample_id][:])

train_file_top = np.array(train_file_top)
train_file_bot = np.array(train_file_bot)
np.save('/data1/ryan/train-files/train_q076_top.npy', train_file_top)
np.save('/data1/ryan/train-files/train_q076_bot.npy', train_file_bot)


# Q103
samples_top = []
samples_bot = []
temp_top = [name_to_use_q103 + '_' + str(t) for t in q103_t_top]
temp_bot = [name_to_use_q103 + '_' + str(t) for t in q103_t_bot]

for i in temp_top:
    length = train_q103_top_h5f[i].shape[0]
    if length >= audio_window:  # and count > 5100:
        samples_top.append(i)
for i in temp_bot:
    length = train_q103_bot_h5f[i].shape[0]
    if length >= audio_window:  # and count > 5100:
        samples_bot.append(i)

train_file_top = []
train_file_bot = []
for sample_id in samples_top:
    train_file_top.append(train_q103_top_h5f[sample_id][:])
for sample_id in samples_bot:
    train_file_bot.append(train_q103_bot_h5f[sample_id][:])

train_file_top = np.array(train_file_top)
train_file_bot = np.array(train_file_bot)
np.save('/data1/ryan/train-files/train_q103_top.npy', train_file_top)
np.save('/data1/ryan/train-files/train_q103_bot.npy', train_file_bot)
