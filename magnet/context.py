import torch
from torch.utils import data

from src.data_reader.dataset import RealMagnetDataset
from src.model.model import CDCK2


train_raw = '/data1/ryan/dataset/training_bot_ch2.h5'
train_list = '/data1/ryan/dataset/training_bot_ch2.txt'
batch = 64
timestep = 12
audio_window = 2000

model = CDCK2(timestep, batch, audio_window).to('cuda')
model.load_state_dict(torch.load('/data1/ryan/snapshot/cdc/cdc-2019-08-30_14_08_28-model_best.pth')['state_dict'])
model.eval()


training_set = RealMagnetDataset(train_raw, train_list, audio_window)
params = {'num_workers': 0,
          'pin_memory': False}

final_loader = data.DataLoader(training_set, batch_size=100, shuffle=False, **params)
i = 0
for batch_idx, ex in enumerate(final_loader):
    ex = ex.float().unsqueeze(1).to('cuda')  # add channel dimension
    hidden = model.init_hidden(len(ex), use_gpu=True)
    model.predict(ex, hidden, i)
    i += 1
