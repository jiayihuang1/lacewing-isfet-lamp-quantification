"""F-D backbone-swap tests."""
import pytest
import torch

from lacewing.quantification.methods.unet_segmentation_backbones import (
    build_fd_backbone,
)


BACKBONES = ["unet", "gru", "cnn_gru_par"]


@pytest.mark.parametrize("backbone", BACKBONES)
def test_output_shape_4class(backbone: str) -> None:
    model = build_fd_backbone(backbone, n_classes=4)
    x = torch.randn(2, 1, 450)
    y = model(x)
    assert y.shape == (2, 4, 450), f"{backbone} got {y.shape}"


@pytest.mark.parametrize("backbone", BACKBONES)
def test_argmax_produces_valid_class_ids(backbone: str) -> None:
    model = build_fd_backbone(backbone, n_classes=4)
    x = torch.randn(1, 1, 450)
    with torch.no_grad():
        y = model(x)
        cls = y.argmax(dim=1)
    assert cls.min() >= 0 and cls.max() < 4


def test_unet_matches_existing_build_model() -> None:
    from lacewing.quantification.methods.unet_segmentation import build_model
    a = build_fd_backbone("unet", n_classes=4)
    b = build_model(base_channels=8)
    a_params = sum(p.numel() for p in a.parameters())
    b_params = sum(p.numel() for p in b.parameters())
    assert a_params == b_params, f"unet from build_fd_backbone={a_params}, from build_model={b_params}"


def test_unknown_backbone_raises() -> None:
    with pytest.raises(ValueError, match="unknown"):
        build_fd_backbone("mystery_backbone", n_classes=4)
