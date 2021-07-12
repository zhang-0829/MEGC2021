# *_*coding:utf-8 *_*
import torch
import torch.nn as nn


class AvgPooling(nn.Module):
    def __init__(self):
        super(AvgPooling, self).__init__()

    def forward(self, x):
        return torch.mean(x, dim=1)


class MaxPooling(nn.Module):
    def __init__(self):
        super(MaxPooling, self).__init__()

    def forward(self, x):
        return torch.max(x, dim=1)[0]