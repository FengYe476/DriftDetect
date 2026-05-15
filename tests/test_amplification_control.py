from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
pytestmark = pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch is not installed")

if TORCH_AVAILABLE:
    import torch
    from torch import nn
else:
    torch = types.SimpleNamespace(Tensor=object)
    nn = types.SimpleNamespace(Module=object)


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

if TORCH_AVAILABLE:
    from src.regularization.amplification_control import amplification_control_loss
else:
    amplification_control_loss = None


class ScaleDynamics(nn.Module):
    def __init__(self, scale: float) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(float(scale)))

    def img_step(
        self,
        state: dict[str, torch.Tensor],
        action: torch.Tensor,
        sample: bool = False,
    ) -> dict[str, torch.Tensor]:
        del action, sample
        return {
            "stoch": state["stoch"],
            "deter": self.scale * state["deter"],
        }


class GatedDynamics(nn.Module):
    def __init__(self, stoch_flat_dim: int, action_dim: int, scale: float = 1.5) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(float(scale)))
        self.action_gate = nn.Linear(action_dim, 1, bias=False)
        self.stoch_gate = nn.Linear(stoch_flat_dim, 1, bias=False)
        nn.init.constant_(self.action_gate.weight, 0.1)
        nn.init.constant_(self.stoch_gate.weight, 0.01)

    def img_step(
        self,
        state: dict[str, torch.Tensor],
        action: torch.Tensor,
        sample: bool = False,
    ) -> dict[str, torch.Tensor]:
        del sample
        stoch = state["stoch"].reshape(state["stoch"].shape[0], -1)
        gate = self.scale + self.action_gate(action) + self.stoch_gate(stoch)
        return {
            "stoch": state["stoch"],
            "deter": gate * state["deter"],
        }


def make_inputs(
    *,
    batch: int = 4,
    deter_dim: int = 16,
    stoch_dim: int = 8,
    action_dim: int = 4,
    requires_grad: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    post_stoch = torch.randn(batch, stoch_dim, requires_grad=requires_grad)
    post_deter = torch.randn(batch, deter_dim, requires_grad=requires_grad)
    action = torch.randn(batch, action_dim, requires_grad=requires_grad)
    return post_stoch, post_deter, action


def test_ac_shape_and_gradient() -> None:
    """Verify AC loss computes correctly and sends gradients to transition params only."""
    torch.manual_seed(0)
    B, D_deter, D_stoch, D_act = 4, 16, 8, 4
    post_stoch, post_deter, action = make_inputs(
        batch=B,
        deter_dim=D_deter,
        stoch_dim=D_stoch,
        action_dim=D_act,
        requires_grad=True,
    )
    dynamics = GatedDynamics(D_stoch, D_act, scale=1.8)

    loss, metrics = amplification_control_loss(
        dynamics,
        post_stoch,
        post_deter,
        action,
        n_dirs=2,
        delta_scale=0.01,
        rho=0.5,
    )

    assert loss.shape == torch.Size([])
    assert torch.isfinite(loss)
    assert torch.equal(metrics["ac_loss"], loss.detach())
    assert loss.item() > 0.0

    loss.backward()
    assert dynamics.scale.grad is not None
    assert torch.any(dynamics.scale.grad != 0)
    assert dynamics.action_gate.weight.grad is not None
    assert torch.any(dynamics.action_gate.weight.grad != 0)
    assert dynamics.stoch_gate.weight.grad is not None
    assert torch.any(dynamics.stoch_gate.weight.grad != 0)
    assert post_deter.grad is None
    assert post_stoch.grad is None
    assert action.grad is None


def test_ac_guard_zero_and_positive() -> None:
    torch.manual_seed(1)
    post_stoch, post_deter, action = make_inputs()

    low_gain_loss, _ = amplification_control_loss(
        ScaleDynamics(0.5),
        post_stoch,
        post_deter,
        action,
        rho=1.0,
    )
    high_gain_loss, _ = amplification_control_loss(
        ScaleDynamics(2.0),
        post_stoch,
        post_deter,
        action,
        rho=1.0,
    )

    assert torch.equal(low_gain_loss, post_deter.new_tensor(0.0))
    assert high_gain_loss.item() > 0.0


def test_ac_gain_computation() -> None:
    torch.manual_seed(2)
    post_stoch, post_deter, action = make_inputs()

    _, identity_metrics = amplification_control_loss(
        ScaleDynamics(1.0),
        post_stoch,
        post_deter,
        action,
        n_dirs=3,
        rho=10.0,
    )
    _, double_metrics = amplification_control_loss(
        ScaleDynamics(2.0),
        post_stoch,
        post_deter,
        action,
        n_dirs=3,
        rho=10.0,
    )

    assert torch.allclose(identity_metrics["ac_gain_h1_mean"], torch.tensor(1.0), atol=1e-5)
    assert torch.allclose(double_metrics["ac_gain_h1_mean"], torch.tensor(2.0), atol=1e-5)


def test_ac_multi_horizon() -> None:
    """Verify multi-step gain computation."""
    torch.manual_seed(3)
    B, D_act = 4, 4
    post_stoch, post_deter, _ = make_inputs(batch=B, action_dim=D_act)
    actions = torch.randn(B, 2, D_act)

    loss, metrics = amplification_control_loss(
        ScaleDynamics(2.0),
        post_stoch,
        post_deter,
        actions,
        n_dirs=2,
        rho=0.0,
        horizons=(1, 2),
        horizon_weights="sqrt_inv",
    )

    assert loss.item() > 0.0
    assert metrics["ac_gain_h2_mean"] > metrics["ac_gain_h1_mean"]
    assert torch.allclose(metrics["ac_gain_h1_mean"], torch.tensor(2.0), atol=1e-5)
    assert torch.allclose(metrics["ac_gain_h2_mean"], torch.tensor(4.0), atol=1e-5)
    assert metrics["ac_weight_h1"] > metrics["ac_weight_h2"]


def test_ac_multi_horizon_requires_action_sequence() -> None:
    torch.manual_seed(4)
    post_stoch, post_deter, action = make_inputs()

    with pytest.raises(ValueError, match="Multi-step horizons require action"):
        amplification_control_loss(
            ScaleDynamics(1.0),
            post_stoch,
            post_deter,
            action,
            horizons=(1, 2),
        )


def test_ac_rejects_unknown_horizon_weights() -> None:
    torch.manual_seed(5)
    post_stoch, post_deter, action = make_inputs()

    with pytest.raises(ValueError, match="Unsupported horizon_weights"):
        amplification_control_loss(
            ScaleDynamics(1.0),
            post_stoch,
            post_deter,
            action,
            horizon_weights="linear",
        )
