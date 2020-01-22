import numpy as np
import torch
import torch.nn as nn


# Model class for transposed convolution decoder
class Transposed(nn.Module):
    def __init__(self):
        super(Transposed, self).__init__()

        self.decoder = nn.Sequential(  # upsampling factor= 50
            nn.ConvTranspose1d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(512, 512, kernel_size=8, stride=5, padding=2, output_padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(512, 1, kernel_size=10, stride=5, padding=3, output_padding=1),
            nn.Tanh()
            # nn.Linear(600, 600)
        )
        self.softmax = nn.Softmax()
        self.lsoftmax = nn.LogSoftmax()

        def _weights_init(m):
            # if isinstance(m, nn.Linear):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if isinstance(m, nn.ConvTranspose1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.apply(_weights_init)

    def forward(self, x, encoder, device):
        mse = nn.MSELoss()

        # index = torch.randint(int(x.size()[2] - 600), size=(1,)).long()
        # index_ratio = 18

        real_input = x[:, :, 1000:1600]

        z = encoder.encoder(real_input)
        result = self.decoder(z).to(device)
        loss = mse(real_input, result)

        return loss


class CDCK2(nn.Module):
    def __init__(self, timestep, batch_size, seq_len):

        super(CDCK2, self).__init__()

        self.batch_size = batch_size
        self.seq_len = seq_len
        self.timestep = timestep
        self.encoder = nn.Sequential(  # downsampling factor = 50
            nn.Conv1d(1, 512, kernel_size=10, stride=5, padding=3, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, kernel_size=8, stride=5, padding=2, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True)
        )
        self.gru = nn.GRU(512, 256, num_layers=1, bidirectional=False, batch_first=True)
        self.Wk = nn.ModuleList([nn.Linear(256, 512) for _ in range(timestep)])
        self.softmax = nn.Softmax()
        self.lsoftmax = nn.LogSoftmax()

        def _weights_init(m):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # initialize gru
        for layer_p in self.gru._all_weights:
            for p in layer_p:
                if 'weight' in p:
                    nn.init.kaiming_normal_(self.gru.__getattr__(p), mode='fan_out', nonlinearity='relu')

        self.apply(_weights_init)

    def init_hidden(self, batch_size, use_gpu=True):
        if use_gpu: return torch.zeros(1, batch_size, 256).cuda()
        else: return torch.zeros(1, batch_size, 256)

    def forward(self, x, hidden):
        batch = x.size()[0]
        t_samples = torch.randint(int(self.seq_len/50-self.timestep), size=(1,)).long()  # randomly pick time stamps
        # input sequence is N*C*L, e.g. 8*1*20480
        z = self.encoder(x)
        # encoded sequence is N*C*L, e.g. 8*512*128
        # reshape to N*L*C for GRU, e.g. 8*128*512
        z = z.transpose(1, 2)
        nce = 0 # average over timestep and batch
        encode_samples = torch.empty((self.timestep, batch, 512)).float()  # e.g. size 12*8*512
        for i in np.arange(1, self.timestep+1):
            encode_samples[i-1] = z[:, t_samples+i, :].view(batch, 512)  # z_tk e.g. size 8*512
        forward_seq = z[:, :t_samples+1, :]  # e.g. size 8*100*512
        output, hidden = self.gru(forward_seq, hidden)  # output size e.g. 8*100*256
        c_t = output[:, t_samples, :].view(batch, 256)  # c_t e.g. size 8*256
        pred = torch.empty((self.timestep, batch, 512)).float()  # e.g. size 12*8*512
        for i in np.arange(0, self.timestep):
            linear = self.Wk[i]
            pred[i] = linear(c_t)  # Wk*c_t e.g. size 8*512
        for i in np.arange(0, self.timestep):
            total = torch.mm(encode_samples[i], torch.transpose(pred[i],0,1))  # e.g. size 8*8
            correct = torch.sum(torch.eq(torch.argmax(self.softmax(total), dim=0), torch.arange(0, batch)))  # correct is a tensor
            nce += torch.sum(torch.diag(self.lsoftmax(total)))  # nce is a tensor
        nce /= -1.*batch*self.timestep
        accuracy = 1.*correct.item()/batch

        return accuracy, nce, hidden

    def predict(self, x, hidden, number):
        # input sequence is N*C*L, e.g. 8*1*20480
        batch = x.size()[0]

        if number == 1:
            x = x[:, :, :600]
        else:
            x = x[:, :, :1000]

        z = self.encoder(x)
        # encoded sequence is N*C*L, e.g. 8*512*128
        # reshape to N*L*C for GRU, e.g. 8*128*512
        z = z.transpose(1, 2)
        output, hidden = self.gru(z, hidden)  # output size e.g. 8*128*256

        result = output[:, -1, :].view(batch, 256)
        pred = torch.empty((self.timestep, batch, 512)).float().to('cuda')
        for i in np.arange(0, self.timestep):
            linear = self.Wk[i]
            pred[i] = linear(result)  # Wk*c_t e.g. size 8*512

        # np.save('/data1/ryan/clusters/context_training_50_' + str(number), result)
        # print(result.shape)

        return pred  # return every frame
