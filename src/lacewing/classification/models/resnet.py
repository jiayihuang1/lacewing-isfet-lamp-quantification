"""Paper 3 Model 4: ResNet1D (Wang et al. 2017).

  3 residual blocks, each:
    Conv1D(k=8, BN, relu) -> Conv1D(k=5, BN, relu) -> Conv1D(k=3, BN)
    + skip(1x1 conv if channel mismatch) -> relu
  Filter widths: 64 -> 128 -> 128
  Global average pool -> Dense(1, sigmoid)
"""
from __future__ import annotations

import torch
from torch import nn


class _Block(nn.Module):
    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.conv1 = nn.Conv1d(in_c, out_c, 8, padding="same")
        self.bn1 = nn.BatchNorm1d(out_c)
        self.conv2 = nn.Conv1d(out_c, out_c, 5, padding="same")
        self.bn2 = nn.BatchNorm1d(out_c)
        self.conv3 = nn.Conv1d(out_c, out_c, 3, padding="same")
        self.bn3 = nn.BatchNorm1d(out_c)
        self.shortcut = (nn.Conv1d(in_c, out_c, 1)
                         if in_c != out_c else nn.Identity())
        self.shortcut_bn = (nn.BatchNorm1d(out_c)
                            if in_c != out_c else nn.Identity())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.bn1(self.conv1(x)))
        h = torch.relu(self.bn2(self.conv2(h)))
        h = self.bn3(self.conv3(h))
        s = self.shortcut_bn(self.shortcut(x))
        return torch.relu(h + s)


class ResNet1D(nn.Module):
    def __init__(self):
        super().__init__()
        self.b1 = _Block(1, 64)
        self.b2 = _Block(64, 128)
        self.b3 = _Block(128, 128)
        self.head = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.b3(self.b2(self.b1(x)))
        h = h.mean(dim=-1)
        return self.head(h).squeeze(-1)


def build(input_shape) -> nn.Module:
    return ResNet1D()
