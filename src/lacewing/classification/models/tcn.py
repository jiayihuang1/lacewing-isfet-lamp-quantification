"""TCN backbone (Bai et al. 2018, paper 20).

Dilated causal 1D convolutions + residual blocks. For offline TTP
prediction we do not require causality strictly — we use the standard
TCN as an inductive-bias-carrying alternative to GRU/attention.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


class Chomp1d(nn.Module):
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, : -self.chomp_size].contiguous() if self.chomp_size > 0 else x


class TemporalBlock(nn.Module):
    def __init__(self, n_in: int, n_out: int, kernel: int, stride: int, dilation: int, padding: int, dropout: float):
        super().__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_in, n_out, kernel, stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)
        self.conv2 = weight_norm(nn.Conv1d(n_out, n_out, kernel, stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(n_in, n_out, 1) if n_in != n_out else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.drop1(self.relu1(self.chomp1(self.conv1(x))))
        out = self.drop2(self.relu2(self.chomp2(self.conv2(out))))
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs: int, num_channels: list[int], kernel_size: int, dropout: float):
        super().__init__()
        layers = []
        prev = num_inputs
        for i, ch in enumerate(num_channels):
            dil = 2 ** i
            pad = (kernel_size - 1) * dil
            layers.append(TemporalBlock(prev, ch, kernel_size, 1, dil, pad, dropout))
            prev = ch
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class TCN(nn.Module):
    """Whole-trace classification/regression wrapper: TCN + global-pool + linear head."""
    def __init__(self, input_length: int, n_classes: int, num_channels: list[int] = None, kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        if num_channels is None:
            num_channels = [32, 32, 64, 64]
        self.tcn = TemporalConvNet(num_inputs=1, num_channels=num_channels, kernel_size=kernel_size, dropout=dropout)
        self.head = nn.Linear(num_channels[-1], n_classes)

    def forward(self, x):
        # x: (B, 1, T)
        h = self.tcn(x)             # (B, C, T)
        h = h.mean(dim=-1)          # global-average pool → (B, C)
        return self.head(h)


class TCNPerTimestep(nn.Module):
    """Per-timestep TCN with 1x1-conv output head."""
    def __init__(self, input_size: int = 1, num_channels: list[int] = None, kernel_size: int = 3, dropout: float = 0.1, n_classes: int = 4):
        super().__init__()
        if num_channels is None:
            num_channels = [32, 32, 64, 64]
        self.tcn = TemporalConvNet(num_inputs=input_size, num_channels=num_channels, kernel_size=kernel_size, dropout=dropout)
        self.head = nn.Conv1d(num_channels[-1], n_classes, 1)

    def forward(self, x):
        h = self.tcn(x)
        return self.head(h)  # (B, n_classes, T)
