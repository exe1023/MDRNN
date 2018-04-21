import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.autograd import Variable

use_cuda = torch.cuda.is_available()

def mnist(batch_size):
    '''
    mnist data loader
    copy-paste from pytorch/examples/mnist/main.py
    '''
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
                   datasets.MNIST('./data', train=True, download=True,
                        transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
                        batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
                  datasets.MNIST('./data', train=False, 
                        transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                        ])),
                        batch_size=batch_size, shuffle=True, **kwargs)
    return train_loader, test_loader


if __name__ == '__main__':
    train, test = mnist(32)
    for data, target in train:
        print(data, data.shape)
        print(target, target.shape)
