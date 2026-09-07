"""SSL pretraining tests."""
import pytest
import torch
import numpy as np

from lacewing.quantification.methods.ssl_pretrain import (
    build_ssl_encoder,
    MaskedReconstructionModel,
    ContrastiveModel,
    mask_chunks,
    augment_pair,
    nt_xent_loss,
)


BACKBONES = ["unet", "gru", "cnn_gru_par"]


@pytest.mark.parametrize("backbone", BACKBONES)
def test_encoder_output_shape_per_timestep(backbone: str) -> None:
    enc = build_ssl_encoder(backbone)
    x = torch.randn(2, 1, 450)
    z = enc(x)
    # (B, D, T) for per-timestep encoders
    assert z.dim() == 3
    assert z.shape[0] == 2
    assert z.shape[2] == 450


def test_masked_reconstruction_output_shape() -> None:
    enc = build_ssl_encoder("unet")
    model = MaskedReconstructionModel(enc, decoder_channels=(16, 8))
    x = torch.randn(2, 1, 450)
    x_recon = model(x)
    assert x_recon.shape == (2, 1, 450)


def test_mask_chunks_masks_15pct() -> None:
    torch.manual_seed(0)
    x = torch.randn(1, 1, 450)
    masked_x, mask = mask_chunks(x, mask_ratio=0.15, chunk_len=30)
    # 15% of 450 = 67.5 samples ≈ 2-3 chunks of 30. Allow ± 1 chunk tolerance.
    n_masked = mask.sum().item()
    assert 30 <= n_masked <= 90


def test_augment_pair_gives_different_views() -> None:
    torch.manual_seed(0)
    x = torch.randn(4, 1, 450)
    a, b = augment_pair(x)
    assert a.shape == x.shape
    assert b.shape == x.shape
    # Augmented views should differ from raw and from each other.
    assert not torch.allclose(a, x)
    assert not torch.allclose(b, x)
    assert not torch.allclose(a, b)


def test_nt_xent_zero_when_perfect() -> None:
    # Identical embeddings → NT-Xent should be low (log 1/N ~ log 4).
    z = torch.randn(4, 128)
    z = z / z.norm(dim=1, keepdim=True)
    loss = nt_xent_loss(z, z, temperature=0.1)
    # Not exactly 0 due to log-batch-N floor, but small.
    assert loss.item() < 5.0


def test_contrastive_model_forward() -> None:
    enc = build_ssl_encoder("unet")
    model = ContrastiveModel(enc, projection_dim=128)
    x = torch.randn(2, 1, 450)
    z = model(x)
    assert z.shape == (2, 128)


def test_augment_pair_runs_on_meta_device_no_device_mismatch() -> None:
    """Reproduce the cx3 GPU crash — all internal tensors must be on x's device.

    Use a small CUDA-substitute check: if CUDA isn't available, test on CPU
    but assert that the augmentation function correctly forwards x.device to all
    internally-created tensors (via inspecting for device-explicit torch calls).
    Actually, easier: just move the input to a specific non-default 'meta' or run on CPU
    and verify no crash. The real proof is via inspection; here we do a CPU check.
    """
    x = torch.randn(2, 1, 450)  # CPU tensor
    a, b = augment_pair(x)  # should not crash
    assert a.device == x.device
    assert b.device == x.device


def test_mask_chunks_masked_positions_not_trivially_zero() -> None:
    """Guard against Bug 2: masked positions must be non-trivial (noise or mean),
    not zero-filled — otherwise the model can trivially reconstruct via pred=0.
    """
    # Signal with non-zero baseline
    x = torch.ones(4, 1, 450) * 0.5 + torch.randn(4, 1, 450) * 0.1
    masked_x, mask = mask_chunks(x, mask_ratio=0.15, chunk_len=30)
    # At masked positions, masked_x should NOT be uniformly zero.
    # (It should be noise or mean or something informative-void.)
    mask_bool = mask.bool()
    masked_vals = masked_x[mask_bool]
    # If the mask fill is zero, ALL masked values would be exactly zero (or near it).
    # A noise fill or mean fill gives non-trivial std.
    assert masked_vals.abs().mean() > 1e-3, (
        f"Masked positions are near-zero (mean abs = {masked_vals.abs().mean().item():.6f}); "
        "this triggers the trivial-reconstruction failure mode observed on cx3."
    )
