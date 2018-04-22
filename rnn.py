import torch
import torch.nn as nn
import math

class MDGRUCell(nn.GRUCell):
    def __init__(self, input_size, hidden_size, layer_norm=False, bias=True):
        super(MDGRUCell, self).__init__(input_size, hidden_size, bias)
        self.layer_norm = layer_norm
        
        # two forget gates
        self.weight_ih = nn.Parameter(torch.Tensor(5 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.Tensor(3 * hidden_size, hidden_size))
        self.weight_hh2 = nn.Parameter(torch.Tensor(3 * hidden_size, hidden_size))
        if bias:
           self.bias_ih = nn.Parameter(torch.Tensor(5 * hidden_size))
           self.bias_hh = nn.Parameter(torch.Tensor(3 * hidden_size)) 
           self.bias_hh2 = nn.Parameter(torch.Tensor(3 * hidden_size))
        
        if self.layer_norm:
            self.gamma_ih = nn.Parameter(torch.ones(5 * self.hidden_size))
            self.gamma_hh = nn.Parameter(torch.ones(3 * self.hidden_size))
            self.gamma_hh2 = nn.Parameter(torch.ones(3 * self.hidden_size))
            self.eps = 0
        self.reset_parameters()
    
    def _layer_norm_x(self, x, g, b):
        mean = x.mean(1).unsqueeze(1).expand_as(x)
        std = x.std(1).unsqueeze(1).expand_as(x)
        return g.expand_as(x) * ((x - mean) / (std + self.eps)) + b.expand_as(x)

    def _layer_norm_h(self, x, g, b):
        mean = x.mean(1).unsqueeze(1).expand_as(x)
        return g.expand_as(x) * (x - mean) + b.expand_as(x)

    def forward(self, x, h, h2):
        if self.layer_norm:
            ih_rz = self._layer_norm_x(
                torch.mm(x, self.weight_ih[:4*self.hidden_size, :].transpose(0, 1)),
                self.gamma_ih[:4*self.hidden_size],
                self.bias_ih[:4*self.hidden_size]
                )

            hh_rz = self._layer_norm_h(
                torch.mm(h, self.weight_hh[:2*self.hidden_size, :].transpose(0, 1)),
                self.gamma_hh[:2*self.hidden_size],
                self.bias_hh[:2*self.hidden_size]
                )
            
            hh_rz2 = self._layer_norm_h(
                torch.mm(h2, self.weight_hh[:2*self.hidden_size, :].transpose(0, 1)),
                self.gamma_hh2[:2*self.hidden_size],
                self.bias_hh2[2*self.hidden_size]
                )
        else:
            #print(x.shape, self.weight_hh.shape)
            ih_rz = torch.mm(x, self.weight_ih[:4*self.hidden_size, :].transpose(0, 1))
            hh_rz = torch.mm(h, self.weight_hh[:2*self.hidden_size, :].transpose(0, 1))
            hh_rz2 = torch.mm(h2, self.weight_hh2[:2*self.hidden_size, :].transpose(0, 1))
            if self.bias:
                ih_rz = ih_rz + self.bias_ih[:4*self.hidden_size].expand_as(ih_rz)
                hh_rz = hh_rz + self.bias_hh[:2*self.hidden_size].expand_as(hh_rz) 
                hh_rz2 = hh_rz2 + self.bias_hh2[:2*self.hidden_size].expand_as(hh_rz2)
       
        rz = torch.sigmoid(ih_rz[:, :self.hidden_size*2] + hh_rz)
        r = rz[:, :self.hidden_size]
        z = rz[:, self.hidden_size:self.hidden_size*2]
        
        rz2 = torch.sigmoid(ih_rz[:, self.hidden_size*2:self.hidden_size*4] + hh_rz2)
        r2 = rz2[:, :self.hidden_size]
        z2 = rz2[:, self.hidden_size:self.hidden_size*2]

        if self.layer_norm:
            ih_n = self._layer_norm_x(
                torch.mm(x, self.weight_ih[4*self.hidden_size:, :].transpose(0, 1)),
                self.gamma_ih[4*self.hidden_size:],
                self.bias_ih[4*self.hidden_size:]
                )

            hh_n = self._layer_norm_h(
                torch.mm(h, self.weight_hh[2*self.hidden_size:, :].transpose(0, 1)),
                self.gamma_hh[2*self.hidden_size:],
                self.bias_hh[2*self.hidden_size:]
                )
        
            hh_n2 = self._layer_norm_h(
                torch.mm(h2, self.weight_hh2[2*self.hidden_size:, :].transpose(0, 1)),
                self.gamma_hh2[2*self.hidden_size:],
                self.bias_hh2[2*self.hidden_size:]
                )
        else:
            ih_n = torch.mm(x, self.weight_ih[4*self.hidden_size:, :].transpose(0, 1))
            hh_n = torch.mm(h, self.weight_hh[2*self.hidden_size:, :].transpose(0, 1))
            hh_n2 = torch.mm(h2, self.weight_hh2[2*self.hidden_size:, :].transpose(0, 1))
            if self.bias:
                ih_n = ih_n + self.bias_ih[4*self.hidden_size:].expand_as(ih_n)
                hh_n = hh_n + self.bias_hh[2*self.hidden_size:].expand_as(hh_n) 
                hh_n2 = hh_n2 + self.bias_hh2[2*self.hidden_size:].expand_as(hh_n2)
       
        n = torch.tanh(ih_n + r * hh_n + r2 * hh_n2)
        h = (1 - z - z2) * n + (z + z2) * h
        return h

class MDGRU(nn.Module):
    def __init__(self, input_size, hidden_size, layer_norm=False, bias=True):
        super(MDGRU, self).__init__()
        self.cell = MDGRUCell(input_size, hidden_size, layer_norm, bias)
        self.weight_ih_l0 = self.cell.weight_ih
        self.weight_hh_l0 = self.cell.weight_hh
        self.bias_ih_l0 = self.cell.bias_ih
        self.bias_hh_l0 = self.cell.bias_hh

    def forward(self, xs, h, h2):
        h = h.squeeze(0)
        h2 = h2.squeeze(0)
        ys = []
        for i in range(xs.size(0)):
            x = xs.narrow(0, i, 1).squeeze(0)
            h = self.cell(x, h, h2)
            ys.append(h.unsqueeze(0))
        y = torch.cat(ys, 0)
        h = h.unsqueeze(0)
        return y, h

'''
class LSTM(nn.Module):

    """
    An implementation of Hochreiter & Schmidhuber:
    'Long-Short Term Memory'
    http://www.bioinf.jku.at/publications/older/2604.pdf
    Special args:
    
    dropout_method: one of
            * pytorch: default dropout implementation
            * gal: uses GalLSTM's dropout
            * moon: uses MoonLSTM's dropout
            * semeniuta: uses SemeniutaLSTM's dropout
    """

    def __init__(self, input_size, hidden_size, bias=True, dropout=0.0, dropout_method='pytorch'):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.dropout = dropout
        self.i2h = nn.Linear(input_size, 4 * hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 4 * hidden_size, bias=bias)
        self.reset_parameters()
        assert(dropout_method.lower() in ['pytorch', 'gal', 'moon', 'semeniuta'])
        self.dropout_method = dropout_method

    def sample_mask(self):
        keep = 1.0 - self.dropout
        self.mask = V(th.bernoulli(T(1, self.hidden_size).fill_(keep)))

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

class LayerNormLSTM(LSTM):

    """
    Layer Normalization LSTM, based on Ba & al.:
    'Layer Normalization'
    https://arxiv.org/pdf/1607.06450.pdf
    Special args:
        ln_preact: whether to Layer Normalize the pre-activations.
        learnable: whether the LN alpha & gamma should be used.
    """

    def __init__(self, input_size, hidden_size, bias=True, dropout=0.0, 
                 dropout_method='pytorch', ln_preact=True, learnable=True):
        super(LayerNormLSTM, self).__init__(input_size=input_size, 
                                            hidden_size=hidden_size, 
                                            bias=bias,
                                            dropout=dropout,
                                            dropout_method=dropout_method)
        if ln_preact:
            self.ln_i2h = LayerNorm(4*hidden_size, learnable=learnable)
            self.ln_h2h = LayerNorm(4*hidden_size, learnable=learnable)
        self.ln_preact = ln_preact
        self.ln_cell = LayerNorm(hidden_size, learnable=learnable)

    def forward(self, x, hidden):
        do_dropout = self.training and self.dropout > 0.0
        h, c = hidden
        h = h.view(h.size(1), -1)
        c = c.view(c.size(1), -1)
        x = x.view(x.size(1), -1)

        # Linear mappings
        i2h = self.i2h(x)
        h2h = self.h2h(h)
        if self.ln_preact:
            i2h = self.ln_i2h(i2h)
            h2h = self.ln_h2h(h2h)
        preact = i2h + h2h

        # activations
        gates = preact[:, :3 * self.hidden_size].sigmoid()
        g_t = preact[:, 3 * self.hidden_size:].tanh()
        i_t = gates[:, :self.hidden_size] 
        f_t = gates[:, self.hidden_size:2 * self.hidden_size]
        o_t = gates[:, -self.hidden_size:]

        # cell computations
        if do_dropout and self.dropout_method == 'semeniuta':
            g_t = F.dropout(g_t, p=self.dropout, training=self.training)

        c_t = th.mul(c, f_t) + th.mul(i_t, g_t)

        if do_dropout and self.dropout_method == 'moon':
                c_t.data.set_(th.mul(c_t, self.mask).data)
                c_t.data *= 1.0/(1.0 - self.dropout)

        c_t = self.ln_cell(c_t)
        h_t = th.mul(o_t, c_t.tanh())

        # Reshape for compatibility
        if do_dropout:
            if self.dropout_method == 'pytorch':
                F.dropout(h_t, p=self.dropout, training=self.training, inplace=True)
            if self.dropout_method == 'gal':
                    h_t.data.set_(th.mul(h_t, self.mask).data)
                    h_t.data *= 1.0/(1.0 - self.dropout)

        h_t = h_t.view(1, h_t.size(0), -1)
        c_t = c_t.view(1, c_t.size(0), -1)
        return h_t, (h_t, c_t)
'''