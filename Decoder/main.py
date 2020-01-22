# Utilities
import time
from timeit import default_timer as timer

# Libraries
import numpy as np

# Torch
import torch
from torch.utils import data
import torch.optim as optim

# Custrom Imports
from logger import setup_logs
from dataset import Dataset, ValDataset
from training import train, validation, snapshot
from model import Transposed, CDCK2


encoder_path = '/data1/ryan/snapshot/cdc/cdc-2019-09-17_15_09_39-model_best.pth'
logging_dir = '/data1/ryan/decoder/snapshot/cdc/'
run_name = "cdc" + time.strftime("-%Y-%m-%d_%H_%M_%S")
print(run_name)

timestep = 12
batch = 64
audio_window = 2000
n_warmup_steps = 1000
epochs = 60
log_interval = 10

train_raw = '/data1/ryan/dataset/training_new.h5'
train_list = '/data1/ryan/dataset/training_new.txt'
validation_raw = '/data1/ryan/dataset/validation_new.h5'
validation_list = '/data1/ryan/dataset/validation_new.txt'


class ScheduledOptim(object):
    """A simple wrapper class for learning rate scheduling"""

    def __init__(self, optimizer, n_warmup_steps):
        self.optimizer = optimizer
        self.d_model = 128
        self.n_warmup_steps = n_warmup_steps
        self.n_current_steps = 0
        self.delta = 1

    def state_dict(self):
        self.optimizer.state_dict()

    def step(self):
        """Step by the inner optimizer"""
        self.optimizer.step()

    def zero_grad(self):
        """Zero out the gradients by the inner optimizer"""
        self.optimizer.zero_grad()

    def increase_delta(self):
        self.delta *= 2

    def update_learning_rate(self):
        """Learning rate scheduling per step"""

        self.n_current_steps += self.delta
        new_lr = np.power(self.d_model, -0.5) * np.min([
            np.power(self.n_current_steps, -0.5),
            np.power(self.n_warmup_steps, -1.5) * self.n_current_steps])

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
        return new_lr


def main():
    global_timer = timer()  # global timer
    logger = setup_logs(logging_dir, run_name)  # setup logs
    device = torch.device("cuda")
    model = Transposed().to(device)
    params = {'num_workers': 0, 'pin_memory': False}

    encoder = CDCK2(timestep, batch, audio_window).to(device)
    encoder.load_state_dict(torch.load(encoder_path)['state_dict'])
    encoder.eval()
    for param in encoder.encoder.parameters():
        param.requires_grad = False

    logger.info('===> loading train, validation and eval dataset')
    training_set = Dataset(train_raw, train_list, audio_window)
    validation_set = Dataset(validation_raw, validation_list, audio_window)
    train_loader = data.DataLoader(training_set, batch_size=batch, shuffle=True, **params)
    validation_loader = data.DataLoader(validation_set, batch_size=batch, shuffle=True, **params)

    optimizer = ScheduledOptim(
        optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            betas=(0.9, 0.98), eps=1e-09, weight_decay=1e-4, amsgrad=True),
        n_warmup_steps)

    model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info('### Model summary below###\n {}\n'.format(str(model)))
    logger.info('===> Model total parameter: {}\n'.format(model_params))

    # Start training
    best_loss = np.inf
    best_epoch = -1

    for epoch in range(1, epochs + 1):
        epoch_timer = timer()
        train(log_interval, model, device, train_loader, optimizer, epoch, encoder)
        val_acc, val_loss = validation(model, device, validation_loader, encoder)

        # Save
        if val_loss < best_loss:
            best_loss = min(val_loss, best_loss)
            snapshot(logging_dir, run_name, {
                'epoch': epoch + 1,
                'validation_acc': val_acc,
                'state_dict': model.state_dict(),
                'validation_loss': val_loss,
                'optimizer': optimizer.state_dict(),
            })
            best_epoch = epoch + 1
        elif epoch - best_epoch > 2:
            optimizer.increase_delta()
            best_epoch = epoch + 1

        end_epoch_timer = timer()
        logger.info("#### End epoch {}/{}, elapsed time: {}".format(epoch, epochs, end_epoch_timer - epoch_timer))

    # End
    end_global_timer = timer()
    logger.info("################## Success #########################")
    logger.info("Total elapsed time: %s" % (end_global_timer - global_timer))


if __name__ == '__main__':
    main()
