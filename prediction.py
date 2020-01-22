import numpy as np

import torch
from torch.utils import data

from Decoder.model import Transposed, CDCK2
from Decoder.dataset import Dataset


batch = 1
timestep = 12
audio_window = 2000
params = {'num_workers': 0, 'pin_memory': False}

encoder_path = '/data1/ryan/snapshot/cdc/cdc-2019-08-20_16_01_46-model_best.pth'
decoder_path_1 = '/data1/ryan/decoder/snapshot/cdc/cdc-2019-08-30_01_53_42-model_best.pth'
decoder_path_2 = '/data1/ryan/decoder/snapshot/cdc/cdc-2019-08-30_01_54_09-model_best.pth'

encoder = CDCK2(timestep, batch, audio_window).to("cuda")
encoder.load_state_dict(torch.load(encoder_path)['state_dict'])
encoder.eval()

decoder1 = Transposed().to("cuda")
decoder1.load_state_dict(torch.load(decoder_path_1)['state_dict'])
decoder1.eval()

decoder2 = Transposed().to("cuda")
decoder2.load_state_dict(torch.load(decoder_path_2)['state_dict'])
decoder2.eval()

train_raw = '/data1/ryan/dataset/training_bot_ch2.h5'
train_list = '/data1/ryan/dataset/training_bot_ch2.txt'

training_set = Dataset(train_raw, train_list, audio_window)
train_loader = data.DataLoader(training_set, batch_size=batch, shuffle=True, **params)


for batch_idx, ex in enumerate(train_loader):
    ex = ex.float().unsqueeze(1).to('cuda')  # add channel dimension
    hidden = encoder.init_hidden(len(ex), use_gpu=True)
    pred1 = encoder.predict(ex, hidden, 1)
    pred2 = encoder.predict(ex, hidden, 2)

    pred1 = pred1.transpose(0, 1)
    pred1 = pred1.transpose(1, 2)
    pred2 = pred2.transpose(0, 1)
    pred2 = pred2.transpose(1, 2)

    result1 = decoder1.decoder(pred1).view(-1).detach().cpu().numpy()
    print('OK!')
    break
    result2 = decoder2.decoder(pred2).view(-1).detach().cpu().numpy()

    result1 = np.append(np.zeros(600), result1)
    result2 = np.append(np.zeros(1000), result2)
    ex = ex.detach.cpu().numpy()

    np.save('/data1/ryan/result1.npy', result1)
    np.save('/data1/ryan/result2.npy', result2)
    np.save('/data1/ryan/ex.npy', ex)

    break
