import torch
from torch.utils import data

from dataset import Dataset
from model import CDCK2, Transposed


batch = 1
timestep = 12
audio_window = 2000
params = {'num_workers': 0, 'pin_memory': False}

encoder_path = '/data1/ryan/snapshot/cdc/cdc-2019-08-20_16_01_46-model_best.pth'
decoder_path = '/data1/ryan/decoder/snapshot/cdc/cdc-2019-08-22_13_18_49-model_best.pth'
encoder = CDCK2(timestep, batch, audio_window).to('cuda')
encoder.load_state_dict(torch.load(encoder_path)['state_dict'])
encoder.eval()
decoder = Transposed().to('cuda')
decoder.load_state_dict(torch.load(decoder_path)['state_dict'])
decoder.eval()

train_raw = '/data1/ryan/dataset/training_bot_ch2.h5'
train_list = '/data1/ryan/dataset/training_bot_ch2.txt'

training_set = Dataset(train_raw, train_list, audio_window)
train_loader = data.DataLoader(training_set, batch_size=1, shuffle=True, **params)


