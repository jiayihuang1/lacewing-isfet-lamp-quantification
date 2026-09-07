"""Paper 3 Model 5: InceptionTime (Fawaz et al. 2020).

Single network (the paper uses a 5-net ensemble called InceptionTime;
Tripathi'23 reports a single instance, so we do the same).
  6 inception modules, bottleneck=32, kernel sizes {10, 20, 40},
  residual every 3 modules, GAP -> Dense(1, sigmoid).
"""
from __future__ import annotations

import torch
from torch import nn


class _InceptionModule(nn.Module):
    def __init__(self, in_c: int, n_filters: int = 32, bottleneck: int = 32,
                 kernel_sizes: tuple[int, ...] = (10, 20, 40)):
        super().__init__()
        self.use_bottleneck = in_c > 1
        if self.use_bottleneck:
            self.bottleneck = nn.Conv1d(in_c, bottleneck, 1, bias=False)
            conv_in = bottleneck
        else:
            self.bottleneck = nn.Identity()
            conv_in = in_c
        self.convs = nn.ModuleList([
            nn.Conv1d(conv_in, n_filters, k, padding="same", bias=False)
            for k in kernel_sizes
        ])
        self.maxpool = nn.MaxPool1d(3, stride=1, padding=1)
        self.conv_pool = nn.Conv1d(in_c, n_filters, 1, bias=False)
        self.bn = nn.BatchNorm1d(n_filters * (len(kernel_sizes) + 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = self.bottleneck(x)
        outs = [c(b) for c in self.convs]
        outs.append(self.conv_pool(self.maxpool(x)))
        h = torch.cat(outs, dim=1)
        return torch.relu(self.bn(h))


class _Shortcut(nn.Module):
    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.conv = nn.Conv1d(in_c, out_c, 1, bias=False)
        self.bn = nn.BatchNorm1d(out_c)

    def forward(self, x: torch.Tensor, residual: torch.Tensor
                ) -> torch.Tensor:
        return torch.relu(x + self.bn(self.conv(residual)))


class InceptionTime(nn.Module):
    def __init__(self, n_modules: int = 6, n_filters: int = 32):
        super().__init__()
        self.modules_list = nn.ModuleList()
        self.shortcuts = nn.ModuleList()
        out_c = 4 * n_filters    # 3 kernels + 1 maxpool branch
        in_c = 1
        residual_in = 1
        for i in range(n_modules):
            self.modules_list.append(_InceptionModule(in_c, n_filters))
            in_c = out_c
            if (i + 1) % 3 == 0:
                self.shortcuts.append(_Shortcut(residual_in, out_c))
                residual_in = out_c
        self.head = nn.Linear(out_c, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        sc_idx = 0
        for i, m in enumerate(self.modules_list):
            x = m(x)
            if (i + 1) % 3 == 0:
                x = self.shortcuts[sc_idx](x, residual)
                residual = x
                sc_idx += 1
        x = x.mean(dim=-1)
        return self.head(x).squeeze(-1)


def build(input_shape) -> nn.Module:
    return InceptionTime()
