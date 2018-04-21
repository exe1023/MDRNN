import argparse
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from data import mnist
from module import MDRNN

parser = argparse.ArgumentParser(description='MultiDimentional RNN')
parser.add_argument('--rnn_type', type=str, default='gru', help='rnn cell type [gru]')
parser.add_argument('--layer_norm', action='store_true', help='whether to use layernorm [False]')
parser.add_argument('--epoch', type=int, default=10, help='training epoch [10]')
parser.add_argument('--batch_size', type=int, default=32, help='batch size [32]')
parser.add_argument('--log_freq', type=int, default=1, help='print log frequency [5]')
args = parser.parse_args()

use_cuda = torch.cuda.is_available()

if __name__ == '__main__':
    train_loader, test_loader = mnist(args.batch_size)
    mdrnn = MDRNN()
    optimizer = optim.Adam(mdrnn.parameters())
    # training
    if use_cuda:
        mdrnn = mdrnn.cuda()
    mdrnn.train()
    for epoch in range(args.epoch):
        for batch_idx, (data, target) in enumerate(train_loader):
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            optimizer.zero_grad()
            output = mdrnn(data.squeeze(1))
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % args.log_freq == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.data[0]))
