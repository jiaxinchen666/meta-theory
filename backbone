import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from torch.nn.utils.weight_norm import WeightNorm
import math
import torch.nn.functional as F

class Regression_meta(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Regression_meta, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim=output_dim
        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.output_dim)

        self.act = nn.LeakyReLU()
    def forward(self,x):
        return self.act(self.fc2(self.act(self.fc1(x))))

class Regression(nn.Module):
    def __init__(self, hidden_dim, output_dim):
        super(Regression, self).__init__()
        self.hidden_dim = hidden_dim
        self.output_dim=output_dim
        self.fc1 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.act=nn.LeakyReLU()
        self.fc2=nn.Linear(self.hidden_dim,output_dim)
    def forward(self,x):
        #return self.fc2(x)
        return self.fc2(self.act(self.fc1(x)))
