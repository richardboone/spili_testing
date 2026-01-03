import torch.nn as nn

from functools import partial
from spikingjelly.clock_driven.neuron import (
    MultiStepLIFNode,
    MultiStepParametricLIFNode,
)

class Decoder(nn.Module):
    def __init__(self,in_channels,out_channels,T):
        super().__init__()

        self.decoder_bn = nn.BatchNorm1d(out_channels)
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.decoder_linear = nn.Linear(in_channels,out_channels,bias=False)
        self.T = T
        self.decoder_lif = MultiStepLIFNode(tau = 2.0,detach_reset = True,backend = 'torch')

    def forward(self,x):   
        T_B ,N, C = x.shape
        B = T_B // self.T
        x = self.decoder_linear(x) 
        x = self.decoder_bn(x.transpose(-1,-2)).reshape(self.T,B,self.out_channels,-1).contiguous()
        x = self.decoder_lif(x).flatten(0,1).transpose(-1,-2).contiguous()
        return x