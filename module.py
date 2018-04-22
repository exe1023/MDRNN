import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from rnn import MDGRU, MDLSTM

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
            self.rnn = MDGRU(input_size, 
                             hidden_size, 
                             layer_norm=layer_norm)
        elif rnn_type =='lstm':
            self.rnn = MDLSTM(input_size, 
                              hidden_size, 
                              dropout=dropout,
                              layer_norm=layer_norm)
        else:
            print('Unexpected rnn type')
            exit()
    
    def forward(self, input, h, h2=None):
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
                 input_size=16,
                 hidden_size=128,
                 output_size=10,
                 layer_norm=False,
                 axis=4,
                 rnn_type='gru'
                 ):
        super(MDRNN, self).__init__()
        self.rnn_type = rnn_type
        rnns = []
        for _ in range(axis):
            rnns.append(RNN(input_size, 
                            hidden_size,
                            layer_norm=layer_norm,
                            rnn_type=rnn_type))
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
        grid = 4
        x_ori, x_stop, x_steps = [0, 0, n-1, n-1], [n-(grid-1), n-(grid-1), (grid-1), (grid-1)], [1, 1, -1, -1]
        y_ori, y_stop, y_steps = [0, n-1, 0, n-1], [n-(grid-1), (grid-1), n-(grid-1), (grid-1)], [1, -1, 1, -1]
        for axis_idx, rnn in enumerate(self.rnns):
            last_row = []
            for i in range(y_ori[axis_idx], y_stop[axis_idx], y_steps[axis_idx]):
                row = []
                last_h = None
                for idx, j in enumerate(range(x_ori[axis_idx], x_stop[axis_idx], x_steps[axis_idx])):
                    # handle hidden from last row
                    if len(last_row) == 0:
                        h = rnn.init_hidden(batch)
                    else:
                        h = last_row[idx]
                    # handle hidden from last column
                    if last_h is None:
                        h2 = rnn.init_hidden(batch)
                    else:
                        h2 = last_h

                    # handle input grid
                    if y_steps[axis_idx] > 0:
                        i_start, i_end = i, i + grid
                    else:
                        i_start, i_end = i - grid, i
                    if x_steps[axis_idx] > 0:
                        j_start, j_end = j, j + grid
                    else:
                        j_start, j_end = j - grid, j

                    input_step = input[:, i_start:i_end, j_start:j_end].contiguous()
                    _, last_h = rnn(input_step.view(batch, -1).unsqueeze(0), h, h2)
                    row.append(last_h)
                last_row = row
            
            if self.rnn_type == 'lstm':
                output_hidden = row[-1][0]
            else:
                output_hidden = row[-1]

            if final_hidden is None:
                final_hidden = output_hidden.squeeze(0)
            else:
                final_hidden = torch.cat((final_hidden, output_hidden.squeeze(0)), 1)
        return  F.log_softmax(self.output(final_hidden), dim=1)
        

if __name__ == '__main__':
    pass
