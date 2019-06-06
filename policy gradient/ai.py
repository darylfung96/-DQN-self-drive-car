import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Network(nn.Module):
    def __init__(self, input_shape, num_output):
        super(Network, self).__init__()
        self.input_shape = input_shape
        self.num_output = num_output
        
        self.layer1 = nn.Linear(self.input_shape, 16)
        self.output = nn.Linear(16, num_output)
        
    def forward(self, inp):
        inp = inp.astype(np.float32)
        inp_var = Variable(torch.from_numpy(inp))
        layer1 = F.relu(self.layer1(inp_var))
        output = F.softmax(self.output(layer1), dim=1)
        return output
