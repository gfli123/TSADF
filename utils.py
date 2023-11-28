import torch
import math
from torch import nn
from torch.nn import Conv2d, Conv3d, Sequential, ReLU
import torch.nn.functional as F


class Res_Block2D(nn.Module):
    '''Residual block w/o BN
        ---Conv-ReLU-Conv-+-
         |________________|
    '''
    def __init__(self, nf, act=ReLU()):
        super(Res_Block2D, self).__init__()
        self.nf = nf
        self.act = act
        self.conv1 = Conv2d(self.nf, self.nf, (3, 3), stride=(1,), padding=(1,))
        self.conv1.apply(initialize_weights)
        self.conv2 = Conv2d(self.nf, self.nf, (3, 3), stride=(1,), padding=(1,))
        self.conv2.apply(initialize_weights)

    def forward(self, inputs):
        identity = inputs
        x = self.act(self.conv1(inputs))
        x = self.conv2(x)
        return identity + x


class Res_Block2D_Add(nn.Module):
    '''Residual block w/o BN, Adjust the dimension and carry out residual learning
     ---Conv---Conv-ReLU-Conv-+-
             |________________|
    '''
    def __init__(self, nf, hf, n_b, act=ReLU()):
        super(Res_Block2D_Add, self).__init__()
        self.nf = nf
        self.hf = hf
        self.n_b = n_b
        self.act = act
        self.conv = Conv2d(self.nf, self.hf, (3, 3), stride=(1,), padding=(1,))
        self.res = Sequential(*[Res_Block2D(self.hf) for _ in range(self.n_b)])

    def forward(self, inputs):
        out = self.act(self.conv(inputs))
        out = self.res(out)
        return out


class UPSAMPLE(nn.Module):
    def __init__(self, nf, act):
        super(UPSAMPLE, self).__init__()
        self.nf = nf
        self.act = act
        self.upconv1 = Conv2d(self.nf, self.nf * 4, (3, 3), stride=(1, 1), padding=(1, 1))
        self.upconv2 = Conv2d(self.nf, self.nf * 4, (3, 3), stride=(1, 1), padding=(1, 1))
        self.last = Conv2d(self.nf, 3, (3, 3), stride=(1, 1), padding=(1, 1))

    def forward(self, inputs):
        x = self.act(self.upconv1(inputs))
        x = F.pixel_shuffle(x, 2)
        x = self.act(self.upconv2(x))
        x = F.pixel_shuffle(x, 2)
        x = self.act(self.last(x))
        return x


def initialize_weights(m):
    if isinstance(m, Conv2d):
        nn.init.xavier_normal_(m.weight, gain=1.)
        nn.init.zeros_(m.bias)
    if isinstance(m, Conv3d):
        nn.init.xavier_normal_(m.weight, gain=1.)
        nn.init.zeros_(m.bias)

