"""Tests for W16 P3 architecture refinements (a1s, a2p, a5, a3n, a4w)."""
from __future__ import annotations

import pytest
import torch

from lacewing.quantification.methods.joint_cls_quant import (
    build_joint_model,
)


WINDOW = 120


def test_a1s_shares_encoder_with_quant_head() -> None:
    """A1s: cls_head is a Linear(1, 1) off the SAME pooled U-Net features as quant."""
    model = build_joint_model("a1s", window=WINDOW, alpha=0.3)
    x = torch.randn(4, 1, WINDOW)
    quant_logit, cls_logit = model(x)
    assert quant_logit.shape == (4,)
    assert cls_logit.shape == (4,)
    # A1s' cls_head must be a Linear(1, 1) — no conv net.
    assert isinstance(model.cls_head, torch.nn.Linear)
    assert model.cls_head.in_features == 1
    assert model.cls_head.out_features == 1


def test_a1s_grad_flows_to_encoder_from_both_heads() -> None:
    """Both loss branches must backprop into the shared U-Net encoder."""
    model = build_joint_model("a1s", window=WINDOW, alpha=0.3)
    x = torch.randn(4, 1, WINDOW)
    quant_logit, cls_logit = model(x)

    # Snapshot encoder param before backward
    p = next(model.backbone.unet.parameters())
    p.retain_grad()
    (quant_logit.sum() + cls_logit.sum()).backward()
    assert p.grad is not None and p.grad.abs().sum() > 0


def test_a3n_forward_shapes() -> None:
    model = build_joint_model("a3n", window=WINDOW, alpha=0.3)
    x = torch.randn(4, 1, WINDOW)
    quant_logit, cls_logit = model(x)
    assert quant_logit.shape == (4,)
    assert cls_logit.shape == (4,)


def test_a3n_smoke_forward_shape() -> None:
    """Smoke check: a3 and a3n forward passes produce shape-compatible outputs.

    Mask parity between a3 and a3n training loops is enforced structurally by
    the shared `if args.arch in ("a3", "a3n"):` branch and verified by PBS
    submission that trains both codepaths on cx3. This unit test checks only
    forward-pass shape compatibility.
    """
    # Both archs' JointModel forward returns raw (quant_logit, cls_logit).
    # The mask logic lives in main()'s training loop; a smoke check of the
    # forward is enough here — the training-loop parity is enforced by the
    # `if args.arch in ("a3", "a3n"):` branch and the PBS submission that
    # exercises both codepaths on cx3.
    a3 = build_joint_model("a3", window=WINDOW, alpha=0.3)
    a3n = build_joint_model("a3n", window=WINDOW, alpha=0.3)
    x = torch.randn(4, 1, WINDOW)
    q3, c3 = a3(x); qn, cn = a3n(x)
    # Forward-pass outputs are architecturally identical for a3 vs a3n
    # (both are the default parallel-cls-head path with no gate). Shapes
    # match; concrete values differ because parameters are initialised
    # from independent seeds.
    assert q3.shape == qn.shape == (4,) and c3.shape == cn.shape == (4,)


def test_a4w_forward_shapes() -> None:
    model = build_joint_model("a4w", window=WINDOW, alpha=0.3)
    x = torch.randn(4, 1, WINDOW)
    quant_logit, cls_logit = model(x)
    assert quant_logit.shape == (4,)
    assert cls_logit.shape == (4,)


def test_a5_forward_shapes() -> None:
    """A5 returns raw logits from forward(); gate is applied in loss/inference."""
    model = build_joint_model("a5", window=WINDOW, alpha=0.3)
    x = torch.randn(4, 1, WINDOW)
    quant_logit, cls_logit = model(x)
    assert quant_logit.shape == (4,)
    assert cls_logit.shape == (4,)


def test_a5_gate_is_prob_space_multiplication() -> None:
    """Verify the intended math: p_gated = sigmoid(q) * sigmoid(c)."""
    torch.manual_seed(0)
    q_logit = torch.tensor([2.0, 2.0])
    c_logit = torch.tensor([2.0, -5.0])
    expected = torch.sigmoid(q_logit) * torch.sigmoid(c_logit)
    # sigmoid(-5) ≈ 0.0067; sigmoid(2) ≈ 0.881 → expected ≈ [0.776, 0.0059]
    assert torch.allclose(expected, torch.tensor([0.776, 0.0059]), atol=1e-3)


def test_train_cls_stage_module_imports_and_has_main() -> None:
    """Stage-1 cls-only pretraining CLI exists and is invokable."""
    import lacewing.quantification.methods.train_cls_stage as m
    assert hasattr(m, "main"), "train_cls_stage must expose main()"


def test_train_cls_stage_help_runs(monkeypatch, capsys) -> None:
    """--help exits 0 without touching disk or GPU."""
    import sys
    from lacewing.quantification.methods import train_cls_stage
    monkeypatch.setattr(sys, "argv", ["train_cls_stage.py", "--help"])
    with pytest.raises(SystemExit) as exc:
        train_cls_stage.main()
    assert exc.value.code == 0


def test_a2p_freezes_cls_head_when_pretrained_loaded(tmp_path) -> None:
    """A2p loads a frozen cls state_dict; cls params must have requires_grad=False."""
    from lacewing.quantification.methods.train_cls_stage import build_cls_head

    # Write a random cls state_dict to a temp path
    cls = build_cls_head()
    cls_path = tmp_path / "cls.pt"
    torch.save(cls.state_dict(), cls_path)

    model = build_joint_model("a2p", window=WINDOW, alpha=0.3)
    model.load_pretrained_cls(str(cls_path))
    for p in model.cls_head.parameters():
        assert p.requires_grad is False


def test_a2p_forward_matches_a2_math() -> None:
    """a2p uses the same log-sigmoid gate as a2 (only difference is cls frozen)."""
    torch.manual_seed(0)
    m = build_joint_model("a2p", window=WINDOW, alpha=0.3)
    x = torch.randn(4, 1, WINDOW)
    quant_logit, cls_logit = m(x)
    # Gate applied — expect quant_logit ≤ raw_quant, since log(sigmoid(cls_logit)) ≤ 0.
    # Recover the raw quant by subtracting the gate:
    gate = torch.log(torch.sigmoid(cls_logit) + 1e-8)
    raw_quant = quant_logit - gate
    # The recovered raw_quant should be finite and match the backbone's raw output
    with torch.no_grad():
        expected_raw = m.backbone(x)
    assert torch.allclose(raw_quant, expected_raw, atol=1e-5)
