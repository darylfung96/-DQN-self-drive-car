import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.autograd import Variable


class ActorCritic(nn.Module):
    def __init__(self, num_outputs):
        super(ActorCritic, self).__init__()
        self.num_outputs = num_outputs
        self.conv_layers = nn.Sequential(nn.Conv2d(4, 32, (8, 8), stride=2),
                                         nn.ELU(),
                                         nn.Conv2d(32, 32, (8, 8), stride=4),
                                         nn.ELU(),
                                         nn.Conv2d(32, 32, (8, 8), stride=4),
                                         nn.ELU()
                                         )
        self.linear_layers = nn.Sequential(nn.Linear(32, 64),
                                           nn.ELU()
                                           )

        self.value_output = nn.Linear(64, 1)
        self.actor_output = nn.Linear(64, num_outputs)

    def forward(self, inp):
        inp_var = Variable(torch.from_numpy(inp))

        conv_output = self.conv_layers(inp_var)
        conv_output = torch.reshape(conv_output, shape=[inp_var.shape[0], -1])
        linear_output = self.linear_layers(conv_output)

        value_output = self.value_output(linear_output)
        actor_output = self.actor_output(linear_output)

        actor_prob = F.softmax(actor_output, dim=1)
        dist = Categorical(actor_prob)

        return dist, value_output
