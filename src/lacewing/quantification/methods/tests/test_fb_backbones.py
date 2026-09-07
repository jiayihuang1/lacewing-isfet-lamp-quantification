"""F-B backbone factory — shape + variable input_len tests."""
import pytest
import torch

from lacewing.quantification.methods._fb_backbones import (
    FB_BACKBONES,
    build_fb_backbone,
)


BACKBONE_IDS = ["ann", "bigru", "cnn_gru_par", "gru", "tcn", "transformer_patch", "unet"]


@pytest.mark.parametrize("name", BACKBONE_IDS)
@pytest.mark.parametrize("input_len", [60, 120])
def test_backbone_produces_scalar_per_input(name: str, input_len: int) -> None:
    model = build_fb_backbone(name, input_len=input_len)
    x = torch.randn(4, 1, input_len)
    logits = model(x)
    assert logits.shape == (4,), f"{name}@W={input_len}: got {logits.shape}, expected (4,)"


def test_unknown_backbone_raises() -> None:
    with pytest.raises(ValueError, match="Unknown backbone"):
        build_fb_backbone("nonexistent", input_len=60)


def test_registry_has_all_seven_backbones() -> None:
    assert set(FB_BACKBONES.keys()) == set(BACKBONE_IDS)


def test_unet_forward_features_shape_and_grad() -> None:
    """`forward_features` returns pooled encoder features and preserves gradient flow."""
    from lacewing.quantification.methods._fb_backbones import build_fb_backbone

    backbone = build_fb_backbone("unet", input_len=120)
    x = torch.randn(4, 1, 120, requires_grad=True)
    feats = backbone.forward_features(x)
    assert feats.shape == (4, 1), f"expected (4, 1); got {feats.shape}"
    # Gradients must flow through pooled features
    loss = feats.sum()
    loss.backward()
    assert x.grad is not None


def test_unet_forward_unchanged() -> None:
    """`forward` must still return a scalar-per-window logit shape (B,)."""
    from lacewing.quantification.methods._fb_backbones import build_fb_backbone

    torch.manual_seed(0)
    backbone = build_fb_backbone("unet", input_len=120)
    x = torch.randn(4, 1, 120)
    out = backbone(x)
    assert out.shape == (4,), f"expected (4,); got {out.shape}"
