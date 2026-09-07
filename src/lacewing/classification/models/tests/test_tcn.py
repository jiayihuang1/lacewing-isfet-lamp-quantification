import torch
from lacewing.classification.models.tcn import TCN


def test_tcn_forward_shape():
    m = TCN(input_length=450, n_classes=1, num_channels=[32, 32, 64, 64])
    x = torch.randn(2, 1, 450)
    assert m(x).shape == (2, 1)
