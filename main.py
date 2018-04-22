import argparse
import torch
import torch.nn as nn
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
    mdrnn = MDRNN(layer_norm=args.layer_norm,
                  axis=1, 
                  rnn_type=args.rnn_type)
    optimizer = optim.Adam(mdrnn.parameters(), lr=1e-3)
    # training
    if use_cuda:
        mdrnn = mdrnn.cuda()
    for epoch in range(args.epoch):
        mdrnn.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            
            output = mdrnn(data.squeeze(1))
            loss = F.nll_loss(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if batch_idx % args.log_freq == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.data[0]))
        
        mdrnn.eval()
        test_loss = 0
        correct = 0
        for data, target in test_loader:
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data, volatile=True), Variable(target)
            
            output = mdrnn(data.squeeze(1))
            test_loss += F.nll_loss(output, target, size_average=False).data[0] # sum up batch loss
            
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()

            test_loss /= len(test_loader.dataset)
            print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                test_loss, correct, len(test_loader.dataset),
                100. * correct / len(test_loader.dataset)))
