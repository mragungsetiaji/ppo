import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

class FeedForwardNN(nn.Module):

    def __init__(self, in_dim, out_dim):
        super(FeedForwardNN, self).__init__()
        self.layer1 = nn.Linear(in_dim, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, out_dim)

    def forward(self, observation):
        x = F.relu(self.layer1(observation))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        return x