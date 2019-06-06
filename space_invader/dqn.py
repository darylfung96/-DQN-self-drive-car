import torch
import torch.nn as nn
from torch.autograd import Variable


class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.conv_layers = nn.Sequential(
                        nn.Conv2d(input_size, 32, kernel_size=(8, 8), stride=[4, 4]),
                        nn.ELU(),
                        nn.Conv2d(32, 64, (4, 4), [2, 2]),
                        nn.ELU(),
                        nn.Conv2d(64, 64, (3, 3), [2, 2])
                    )

        self.fc_layers = nn.Sequential(
            nn.Linear(1280, 512),
            nn.ELU()
        )

        self.advantage_output = nn.Linear(512, self.output_size)
        self.value_output = nn.Linear(512, 1)

    def forward(self, inp):
        inp_var = Variable(torch.from_numpy(inp))

        conv_output = self.conv_layers(inp_var)

        # reshape

        # fc layers
        conv_output = conv_output.reshape((conv_output.shape[0], -1))
        last_feature_output = self.fc_layers(conv_output)

        advantage_output = self.advantage_output(last_feature_output)
        value_output = self.value_output(last_feature_output)

        q_value_output = value_output + advantage_output - advantage_output.mean(1, keepdim=True)

        return q_value_output
