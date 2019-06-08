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


class ConvNetwork(nn.Module):
    def __init__(self, input_shape, num_output):
        super(ConvNetwork, self).__init__()

        self.input_shape = input_shape
        self.num_output = num_output

        self.conv_output = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=(8, 8), stride=4),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.Conv2d(32, 64, kernel_size=(8, 8), stride=2),
            nn.BatchNorm2d(64),
            nn.ELU()
        )

        self.linear_output = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ELU(),
            nn.Linear(512, num_output),
            nn.Softmax(dim=-1)
        )

    def forward(self, inp):
        inp_var = Variable(torch.from_numpy(inp))

        conv_output = self.conv_output(inp_var)
        conv_output = torch.reshape(conv_output, [inp_var.shape[0], -1])

        output_prob = self.linear_output(conv_output)
        return output_prob