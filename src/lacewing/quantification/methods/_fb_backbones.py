"""F-B backbone factory: scalar per-window classification adapters.

F-B framing: input `(B, 1, W)` (single window) → scalar logit `(B,)`.
This is the classification-tier task shape — reuse lacewing.classification.models.*.

Five of the seven backbones already output `(B,)` scalars in the
classification codebase.  `bigru` and `unet` need small adapters because
their classification-tier variants output per-timestep logits.
`tcn` uses the `TCN` (global-pool) wrapper which outputs `(B, 1)` and
just needs a squeeze.
"""
from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Small adapters for bigru / unet (which output (B, T) or (B, n_cls, T))
# ---------------------------------------------------------------------------

class _BiGRUWindowClassifier(nn.Module):
    """BiGRU per-timestep → global mean pool → linear scalar."""

    def __init__(self, input_len: int, hidden: int = 64, layers: int = 2, dropout: float = 0.1):
        super().__init__()
        from lacewing.classification.models.bigru import BiGRUPerTimestep
        self.rnn = BiGRUPerTimestep(
            input_size=1, hidden=hidden, layers=layers, dropout=dropout
        )
        # rnn outputs (B, T); pool over T then project to scalar
        self.head = nn.Linear(1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.rnn(x)                  # (B, T)
        h = h.mean(dim=-1, keepdim=True) # (B, 1)
        return self.head(h).squeeze(-1)  # (B,)


class _UNetWindowClassifier(nn.Module):
    """1D U-Net → global mean pool over T → squeeze → linear scalar."""

    def __init__(self, input_len: int, base_channels: int = 8):
        super().__init__()
        from lacewing.classification.models.unet_1d import UNet1D
        self.unet = UNet1D(in_channels=1, n_classes=1, base_channels=base_channels)
        self.head = nn.Linear(1, 1)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return pooled encoder features (B, 1) — the input to `self.head`.

        Exposed for A1-shared-encoder in joint_cls_quant, which attaches a
        separate `Linear(1, 1)` cls head to the SAME pooled features that
        `self.head` (the quant head) consumes.
        """
        h = self.unet(x)                 # (B, 1, W)
        return h.mean(dim=-1)            # (B, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(x)).squeeze(-1)  # (B,)


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def _build_ann(input_len: int) -> nn.Module:
    from lacewing.classification.models.ann import ANN
    return ANN(input_len=input_len)


def _build_bigru(input_len: int) -> nn.Module:
    return _BiGRUWindowClassifier(input_len=input_len)


def _build_gru(input_len: int) -> nn.Module:
    from lacewing.classification.models.gru import GRUClassifier
    return GRUClassifier(input_len=input_len)


def _build_cnn_gru_par(input_len: int) -> nn.Module:
    from lacewing.classification.models.cnn_gru_parallel import CNNGRUParallel
    return CNNGRUParallel(input_len=input_len)


def _build_transformer_patch(input_len: int) -> nn.Module:
    from lacewing.classification.models.transformer_patch import PatchTransformer
    # patch_size = 15 samples (1s at 15 Hz); input_len must be divisible.
    if input_len % 15 != 0:
        raise ValueError(
            f"transformer_patch requires input_len % 15 == 0; got input_len={input_len}"
        )
    return PatchTransformer(input_len=input_len, patch_size=15)


def _build_unet(input_len: int) -> nn.Module:
    return _UNetWindowClassifier(input_len=input_len, base_channels=8)


def _build_tcn(input_len: int) -> nn.Module:
    from lacewing.classification.models.tcn import TCN
    # TCN outputs (B, n_classes); use n_classes=1 then squeeze inside a wrapper.
    base = TCN(input_length=input_len, n_classes=1)

    class _TCNScalar(nn.Module):
        def __init__(self, tcn: nn.Module):
            super().__init__()
            self.tcn = tcn

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.tcn(x).squeeze(-1)  # (B, 1) → (B,)

    return _TCNScalar(base)


# ---------------------------------------------------------------------------
# Public registry + factory
# ---------------------------------------------------------------------------

FB_BACKBONES: dict[str, Callable[[int], nn.Module]] = {
    "ann":               _build_ann,
    "bigru":             _build_bigru,
    "gru":               _build_gru,
    "cnn_gru_par":       _build_cnn_gru_par,
    "transformer_patch": _build_transformer_patch,
    "unet":              _build_unet,
    "tcn":               _build_tcn,
}


def build_fb_backbone(name: str, input_len: int) -> nn.Module:
    """Build an F-B classifier backbone: (B, 1, input_len) -> (B,) logits."""
    if name not in FB_BACKBONES:
        raise ValueError(
            f"Unknown backbone {name!r}. Valid: {sorted(FB_BACKBONES)}"
        )
    return FB_BACKBONES[name](input_len)
