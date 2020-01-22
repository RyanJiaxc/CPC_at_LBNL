import torch
import logging
import os


logger = logging.getLogger("cdc")


def train(log_interval, model, device, train_loader, optimizer, epoch, encoder):
    model.train()
    for batch_idx, data in enumerate(train_loader):
        data = data.float().unsqueeze(1).to(device)  # add channel dimension
        optimizer.zero_grad()
        loss = model(data, encoder, device)

        loss.backward()
        optimizer.step()
        lr = optimizer.update_learning_rate()
        if batch_idx % log_interval == 0:
            logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tlr:{:.5f}\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), lr, loss.item()))


def validation(model, device, data_loader, encoder):
    logger.info("Starting Validation")
    model.eval()
    total_loss = 0
    total_acc = 0

    with torch.no_grad():
        for data in data_loader:
            data = data.float().unsqueeze(1).to(device)  # add channel dimension
            loss = model(data, encoder, device)
            total_loss += len(data) * loss

    total_loss /= len(data_loader.dataset)  # average loss

    logger.info('===> Validation set: Average loss: {:.4f}\n'.format(
                total_loss))

    return total_acc, total_loss


def snapshot(dir_path, run_name, state):
    snapshot_file = os.path.join(dir_path, run_name + '-model_best.pth')

    torch.save(state, snapshot_file)
    logger.info("Snapshot saved to {}\n".format(snapshot_file))
