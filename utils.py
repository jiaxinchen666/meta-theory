import os
import pprint
import shutil
import time
import math
import numpy
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
def set_gpu(x):
    os.environ['CUDA_VISIBLE_DEVICES'] = x
    print('using gpu:', x)


def ensure_path(path):
    if os.path.exists(path):
        # if input('{} exists, remove? ([y]/n)'.format(path)) != 'n':
        print("removing exist path")
        shutil.rmtree(path)
        os.makedirs(path)

    else:
        os.makedirs(path)


class Averager():

    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v


class Timer():

    def __init__(self):
        self.o = time.time()

    def measure(self, p=1):
        x = (time.time() - self.o) / p
        x = int(x)
        if x >= 3600:
            return '{:.1f}h'.format(x / 3600)
        if x >= 60:
            return '{}m'.format(round(x / 60))
        return '{}s'.format(x)


_utils_pp = pprint.PrettyPrinter()


def pprint(x):
    _utils_pp.pprint(x)

def l2_loss(pred, label):
    return ((pred - label) ** 2).sum() / len(pred) / 2


class sample(nn.Module):
    def __init__(self, mean, std):
        super(sample, self).__init__()
        self.mean = mean
        self.std = std
        return

    def forward(self, logits):
        B = torch.normal(mean=torch.tensor(self.mean), std=torch.tensor(self.std)).cuda()
        return logits * torch.abs(B)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

