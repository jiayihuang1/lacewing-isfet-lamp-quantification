"""1D U-Net for time-series segmentation.

Adapted from Joung et al. 2024 (paper 19) §3.6 — 5-block encoder,
4-block decoder, full-scale skip connections. Kernel size 9 with padding 4
matches the paper's ECG-lead configuration.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class _ConvBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int, kernel: int = 9):
        super().__init__()
        pad = kernel // 2
        self.net = nn.Sequential(
            nn.Conv1d(in_c, out_c, kernel, padding=pad),
            nn.BatchNorm1d(out_c),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv1d(out_c, out_c, kernel, padding=pad),
            nn.BatchNorm1d(out_c),
            nn.LeakyReLU(0.01, inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet1D(nn.Module):
    def __init__(self, in_channels: int = 1, n_classes: int = 4, base_channels: int = 8):
        super().__init__()
        c1, c2, c3, c4, c5 = (
            base_channels,
            base_channels * 2,
            base_channels * 4,
            base_channels * 8,
            base_channels * 16,
        )

        # Encoder
        self.enc1 = _ConvBlock(in_channels, c1)
        self.enc2 = _ConvBlock(c1, c2)
        self.enc3 = _ConvBlock(c2, c3)
        self.enc4 = _ConvBlock(c3, c4)
        self.enc5 = _ConvBlock(c4, c5)
        self.pool = nn.MaxPool1d(2)

        # Decoder
        self.up4 = nn.Upsample(scale_factor=2, mode="linear", align_corners=False)
        self.dec4 = _ConvBlock(c5 + c4, c4)
        self.up3 = nn.Upsample(scale_factor=2, mode="linear", align_corners=False)
        self.dec3 = _ConvBlock(c4 + c3, c3)
        self.up2 = nn.Upsample(scale_factor=2, mode="linear", align_corners=False)
        self.dec2 = _ConvBlock(c3 + c2, c2)
        self.up1 = nn.Upsample(scale_factor=2, mode="linear", align_corners=False)
        self.dec1 = _ConvBlock(c2 + c1, c1)

        # Final classifier — 1x1 conv to n_classes.
        self.classifier = nn.Conv1d(c1, n_classes, kernel_size=1)

    def _pad_to_match(self, up: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """Pad `up` to match skip's length exactly (odd-length input handling)."""
        diff = skip.size(-1) - up.size(-1)
        if diff > 0:
            up = nn.functional.pad(up, (0, diff))
        elif diff < 0:
            up = up[..., :skip.size(-1)]
        return up

    def forward(self, x):
        e1 = self.enc1(x)                       # (B, c1, T)
        e2 = self.enc2(self.pool(e1))           # (B, c2, T/2)
        e3 = self.enc3(self.pool(e2))           # (B, c3, T/4)
        e4 = self.enc4(self.pool(e3))           # (B, c4, T/8)
        e5 = self.enc5(self.pool(e4))           # (B, c5, T/16)

        d4 = self._pad_to_match(self.up4(e5), e4)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))
        d3 = self._pad_to_match(self.up3(d4), e3)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = self._pad_to_match(self.up2(d3), e2)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self._pad_to_match(self.up1(d2), e1)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.classifier(d1)  # (B, n_classes, T)
