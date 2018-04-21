import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from rnn import MDGRU

use_cuda = torch.cuda.is_available()

class RNN(nn.Module):
    '''
    Wrapped RNN
    '''
    def __init__(self, 
                 input_size, 
                 hidden_size,
                 layer_norm=False, 
                 dropout=0,
                 rnn_type='gru'):
        super(RNN, self).__init__()
        # these two paramters are now fixed (not implemented),
        # but we may need them?
        self.bidirectional = False
        self.num_layers = 1

        self.hidden_size = hidden_size
        self.rnn_type = rnn_type
        if rnn_type == 'gru':
            self.rnn = MDGRU(input_size, hidden_size, layer_norm)
        elif rnn_type =='lstm':
            self.rnn = nn.LSTM(input_size, 
                               hidden_size, 
                               dropout=dropout)
        else:
            print('Unexpected rnn type')
            exit()
    
    def forward(self, input, h, h2):
        output, hidden = self.rnn(input, h, h2)
        return output, hidden

    def init_hidden(self, batch_size):
        bidirectional = 2 if self.bidirectional else 1
        h = Variable(torch.zeros(bidirectional * self.num_layers, batch_size, self.hidden_size))

        if self.rnn_type == 'gru':
            return h.cuda() if use_cuda else h
        else:
            c = Variable(torch.zeros(bidirectional * self.num_layers, batch_size, self.hidden_size))
            return (h.cuda(), c.cuda()) if use_cuda else (h, c)

class MDRNN(nn.Module):
    def __init__(self, 
                 input_size=1,
                 hidden_size=25,
                 output_size=10,
                 layer_norm=False,
                 axis=4):
        super(MDRNN, self).__init__()
        rnns = []
        for _ in range(axis):
            rnns.append(RNN(input_size, hidden_size, layer_norm=layer_norm))
        self.rnns = nn.ModuleList(rnns)
        self.output = nn.Linear(hidden_size * len(self.rnns), output_size)
    
    def forward(self, input):
        '''
        Args:
            input: (batch, n, n)
        '''
        batch = input.size(0)
        n = input.size(1)

        final_hidden = None
        # 2d case, we need general case?
        x_ori, x_steps = [0, 0, n-1, n-1], [1, 1, -1, -1]
        y_ori, y_steps = [0, n-1, 0, n-1], [1, -1, 1, -1]
        for axis_idx, rnn in enumerate(self.rnns):
            last_row = []
            for i in range(y_ori[axis_idx], n-y_ori[axis_idx], y_steps[axis_idx]):
                row = []
                last_h = None
                for idx, j in enumerate(range(x_ori[axis_idx], n-x_ori[axis_idx], x_steps[axis_idx])):
                    if len(last_row) == 0:
                        h = rnn.init_hidden(batch)
                    else:
                        h = last_row[idx]

                    if last_h is None:
                        h2 = rnn.init_hidden(batch)
                    else:
                        h2 = last_h
                    _, last_h = rnn(input[:, i, j].unsqueeze(0).unsqueeze(2), h, h2)
                    row.append(last_h)
                last_row = row
            
            if final_hidden is None:
                final_hidden = row[-1].squeeze(0)
            else:
                final_hidden = torch.cat((final_hidden, row[-1].squeeze(0)), 1)
        return  F.log_softmax(self.output(final_hidden), dim=1)
        

if __name__ == '__main__':
    pass
