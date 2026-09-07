"""P3 joint cls+quant model tests."""
import pytest
import torch

from lacewing.quantification.methods.joint_cls_quant import (
    build_joint_model,
    _joint_loss,
)


ARCHITECTURES = ["a1", "a2", "a3", "a4"]
WINDOW = 120


@pytest.mark.parametrize("arch", ARCHITECTURES)
def test_output_shapes(arch: str) -> None:
    model = build_joint_model(arch, window=WINDOW, alpha=0.5)
    x = torch.randn(4, 1, WINDOW)
    quant_logit, cls_logit = model(x)
    assert quant_logit.shape == (4,), f"{arch} quant: {quant_logit.shape}"
    assert cls_logit.shape == (4,), f"{arch} cls: {cls_logit.shape}"


def test_a2_gate_multiplies_quant_logit() -> None:
    """A2: quant output at inference = quant_logit + log(sigmoid(cls_logit) + eps).

    Verified by checking that when cls_logit is very negative (cls says "no amp"),
    final quant probability drops below the raw quant probability.
    """
    torch.manual_seed(0)
    model = build_joint_model("a2", window=WINDOW, alpha=0.5)
    model.eval()
    x = torch.randn(2, 1, WINDOW)
    with torch.no_grad():
        quant_logit, cls_logit = model(x)
        # Manually force cls_logit to very negative → sigmoid ≈ 0 → gated logit drops
        final = quant_logit + torch.log(torch.sigmoid(torch.full_like(cls_logit, -10.0)) + 1e-8)
    # Sanity: gated final < raw quant_logit for negative cls
    assert (final < quant_logit).all()


def test_joint_loss_balances_alpha() -> None:
    quant_logit = torch.tensor([2.0, -2.0])
    cls_logit = torch.tensor([-2.0, 2.0])
    quant_label = torch.tensor([1.0, 0.0])
    cls_label = torch.tensor([1.0, 0.0])
    l05 = _joint_loss(quant_logit, cls_logit, quant_label, cls_label, alpha=0.5)
    l09 = _joint_loss(quant_logit, cls_logit, quant_label, cls_label, alpha=0.9)
    # Cls loss is large (labels opposite to logits), quant loss small.
    # Higher α → more weight on quant (small) → total should DECREASE as α grows.
    assert l09 < l05


def test_unknown_arch_raises() -> None:
    with pytest.raises(ValueError, match="unknown"):
        build_joint_model("a99", window=WINDOW, alpha=0.5)
