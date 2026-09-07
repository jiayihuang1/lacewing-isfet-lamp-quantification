"""CLI-level tests for W16 SSL pretrain flags (--tau, --epochs)."""
import pytest
import sys

from lacewing.quantification.methods import ssl_pretrain


def test_ssl_pretrain_help_runs(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["ssl_pretrain.py", "--help"])
    with pytest.raises(SystemExit) as exc:
        ssl_pretrain.main()
    assert exc.value.code == 0


def test_output_dir_naming_defaults_unchanged() -> None:
    """Default epochs=50 + tau=0.1 must produce the legacy dir name."""
    name = ssl_pretrain.output_dir_name(
        objective="contrastive", backbone="unet", epochs=50, tau=0.1,
    )
    assert name == "p4_ssl_contrastive_unet"


def test_output_dir_naming_appends_epochs_and_tau_suffixes() -> None:
    name = ssl_pretrain.output_dir_name(
        objective="contrastive", backbone="unet", epochs=150, tau=0.3,
    )
    assert name == "p4_ssl_contrastive_unet_epochs150_tau0.3"


def test_output_dir_naming_epochs_only() -> None:
    name = ssl_pretrain.output_dir_name(
        objective="masked", backbone="unet", epochs=100, tau=0.1,
    )
    assert name == "p4_ssl_masked_unet_epochs100"
