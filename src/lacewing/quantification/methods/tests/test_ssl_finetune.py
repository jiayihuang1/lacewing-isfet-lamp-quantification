"""SSL fine-tuning tests."""
import pytest
import torch

from lacewing.quantification.methods.ssl_finetune import build_finetune_model


BACKBONES = ["unet", "gru", "cnn_gru_par"]
DOWNSTREAMS = ["fa", "fb"]


@pytest.mark.parametrize("backbone", BACKBONES)
@pytest.mark.parametrize("downstream", DOWNSTREAMS)
def test_fa_output_shape(backbone: str, downstream: str) -> None:
    model = build_finetune_model(
        objective="masked", backbone=backbone,
        downstream=downstream, freeze=False,
        pretrained_ckpt=None,
    )
    if downstream == "fa":
        x = torch.randn(2, 1, 450)
        y = model(x)
        assert y.shape == (2, 450)   # per-timestep density
    else:  # fb
        x = torch.randn(2, 1, 120)   # F-B window length
        y = model(x)
        assert y.shape == (2,)       # per-window scalar logit


@pytest.mark.parametrize("backbone", BACKBONES)
def test_frozen_encoder_has_no_grads(backbone: str) -> None:
    model = build_finetune_model(
        objective="masked", backbone=backbone,
        downstream="fa", freeze=True,
        pretrained_ckpt=None,
    )
    x = torch.randn(2, 1, 450)
    y = model(x)
    y.sum().backward()
    for name, param in model.named_parameters():
        if "encoder" in name:
            assert param.grad is None or (param.grad == 0).all(), f"{name} got gradients"
