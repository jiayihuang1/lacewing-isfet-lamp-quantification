"""Paper 3 Model 6: 1D Convolutional Autoencoder (anomaly-style classifier).

Trained on positives only; at test time, classify a pixel as positive
iff reconstruction MSE is below a threshold.

Architecture:
  Encoder: Conv1D(32, k=3) -> MaxPool(2) -> Conv1D(16, k=3) -> MaxPool(2)
  Decoder: ConvT(16, k=3, stride=2) -> ConvT(1, k=3, stride=2)

Output is trimmed/padded to match input length, so decoder out shape == (1, T).

Forward returns the reconstructed signal; the training loop computes MSE
between forward(x) and x. The threshold is fitted on a held-out positive
validation set in train.py.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class AutoEncoder1D(nn.Module):
    def __init__(self, input_len: int):
        super().__init__()
        self.input_len = input_len
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 16, 3, padding=1), nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(16, 16, 3, stride=2, padding=1,
                               output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(16, 1, 3, stride=2, padding=1,
                               output_padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        recon = self.decoder(z)
        # Pad/crop to input length
        if recon.shape[-1] > self.input_len:
            recon = recon[..., :self.input_len]
        elif recon.shape[-1] < self.input_len:
            recon = F.pad(recon, (0, self.input_len - recon.shape[-1]))
        return recon


def build(input_shape) -> nn.Module:
    return AutoEncoder1D(int(input_shape[-1]))
