"""1D U-Net forward-pass shape test."""
import torch

from lacewing.classification.models.unet_1d import UNet1D


def test_unet_forward_shape():
    m = UNet1D(in_channels=1, n_classes=4)
    x = torch.randn(2, 1, 450)
    out = m(x)
    assert out.shape == (2, 4, 450), f"expected (2,4,450), got {out.shape}"


def test_unet_param_count_in_range():
    m = UNet1D(in_channels=1, n_classes=4)
    n = sum(p.numel() for p in m.parameters())
    assert 100_000 < n < 2_000_000, f"expected 100k-2M params, got {n}"
