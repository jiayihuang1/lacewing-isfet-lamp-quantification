"""New F-A loss functions: focal-MSE and KL divergence."""
import pytest
import torch

from lacewing.quantification.methods.pdf_regression import (
    FocalMSE,
    KLDivergenceLoss,
)


def test_focal_mse_zero_when_perfect() -> None:
    pred = torch.tensor([[0.0, 0.5, 1.0]])
    target = torch.tensor([[0.0, 0.5, 1.0]])
    loss = FocalMSE(gamma=2.0)(pred, target)
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)


def test_focal_mse_upweights_hard_examples() -> None:
    """A big residual gets amplified by |err|^gamma compared to a small one."""
    pred_easy = torch.tensor([[0.0, 0.5]])
    tgt_easy = torch.tensor([[0.05, 0.5]])   # error 0.05 on first only

    pred_hard = torch.tensor([[0.0, 0.5]])
    tgt_hard = torch.tensor([[0.5, 0.5]])    # error 0.5 on first only

    fm = FocalMSE(gamma=2.0)
    l_easy = fm(pred_easy, tgt_easy)
    l_hard = fm(pred_hard, tgt_hard)
    # Ratio = (0.5^2 * 0.5^2) / (0.05^2 * 0.05^2) = 10000
    assert l_hard / l_easy > 100.0


def test_focal_mse_gradient_flows() -> None:
    pred = torch.tensor([[0.2, 0.7, 0.1]], requires_grad=True)
    target = torch.tensor([[0.0, 1.0, 0.0]])
    loss = FocalMSE(gamma=2.0)(pred, target)
    loss.backward()
    assert pred.grad is not None
    assert pred.grad.abs().sum() > 0


def test_kl_divergence_zero_when_perfect() -> None:
    logits = torch.tensor([[-10.0, 5.0, -10.0]])   # peaked at index 1
    target = torch.softmax(logits, dim=-1)          # same distribution after softmax
    # After both are softmax'd inside KL, distributions match -> KL = 0.
    loss = KLDivergenceLoss()(target, target)
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-5)


def test_kl_divergence_positive_when_wrong() -> None:
    pred = torch.tensor([[0.0, 1.0, 0.0]])
    target = torch.tensor([[1.0, 0.0, 0.0]])
    loss = KLDivergenceLoss()(pred, target)
    assert loss > 0.0


def test_kl_divergence_gradient_flows() -> None:
    pred = torch.tensor([[0.2, 0.7, 0.1]], requires_grad=True)
    target = torch.tensor([[0.0, 1.0, 0.0]])
    loss = KLDivergenceLoss()(pred, target)
    loss.backward()
    assert pred.grad is not None
    assert pred.grad.abs().sum() > 0
