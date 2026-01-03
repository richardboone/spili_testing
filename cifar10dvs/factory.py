from torch import Tensor
import torch
import torch.nn as nn
from torch.nn.common_types import _size_2_t
from torch.nn.modules.utils import _pair
from torch.nn import functional as F
from typing import List
from spikingjelly.clock_driven.neuron import MultiStepLIFNode,MultiStepParametricLIFNode
from functools import partial
from timm.models.layers import trunc_normal_   #timm==0.5.4

 
class Betascheduler:
    def __init__(self,initial,final,total_epochs):
        self.initial = initial
        self.final = final
        self.total_epochs = total_epochs
        self.currrent_epochs = 0
        self.beta = initial

    def step(self):

        self.beta += (self.final - self.initial)*(1/self.total_epochs)
        self.currrent_epochs += 1

    def get(self):
        return self.beta


