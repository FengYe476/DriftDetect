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
    from src.regularization.sp_dpr import sp_dpr_loss
else:
    sp_dpr_loss = None


class LinearDynamics(nn.Module):
    def __init__(self, stoch_dim: int, deter_dim: int, action_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(stoch_dim + deter_dim + action_dim, deter_dim)

    def img_step(
        self,
        state: dict[str, torch.Tensor],
        action: torch.Tensor,
        sample: bool = False,
    ) -> dict[str, torch.Tensor]:
        del sample
        stoch = state["stoch"].reshape(state["stoch"].shape[0], -1)
        inputs = torch.cat([stoch, state["deter"], action], dim=-1)
        return {"stoch": state["stoch"], "deter": torch.tanh(self.linear(inputs))}


class OffsetDynamics(nn.Module):
    def __init__(self, offset: torch.Tensor) -> None:
        super().__init__()
        self.offset = nn.Parameter(offset.clone().detach())

    def img_step(
        self,
        state: dict[str, torch.Tensor],
        action: torch.Tensor,
        sample: bool = False,
    ) -> dict[str, torch.Tensor]:
        del action, sample
        return {"stoch": state["stoch"], "deter": state["deter"] + self.offset}


class RecordingOffsetDynamics(OffsetDynamics):
    def __init__(self, offset: torch.Tensor) -> None:
        super().__init__(offset)
        self.grad_enabled: list[bool] = []
        self.batch_sizes: list[int] = []

    def img_step(
        self,
        state: dict[str, torch.Tensor],
        action: torch.Tensor,
        sample: bool = False,
    ) -> dict[str, torch.Tensor]:
        self.grad_enabled.append(torch.is_grad_enabled())
        self.batch_sizes.append(int(state["deter"].shape[0]))
        return super().img_step(state, action, sample)


def make_post(
    *,
    batch: int = 4,
    time: int = 6,
    stoch_dim: int = 5,
    deter_dim: int = 7,
    action_dim: int = 3,
    requires_grad: bool = False,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    post = {
        "stoch": torch.randn(batch, time, stoch_dim, requires_grad=requires_grad),
        "deter": torch.randn(batch, time, deter_dim, requires_grad=requires_grad),
    }
    actions = torch.randn(batch, time, action_dim, requires_grad=requires_grad)
    return post, actions


def basis_columns(deter_dim: int, columns: tuple[int, ...]) -> torch.Tensor:
    eye = torch.eye(deter_dim)
    return eye[:, list(columns)]


def test_sp_dpr_basic() -> None:
    """Verify SP-DPR loss computes and backprops correctly."""

    torch.manual_seed(0)
    batch, time, stoch_dim, deter_dim, action_dim = 4, 6, 5, 7, 3
    post, actions = make_post(
        batch=batch,
        time=time,
        stoch_dim=stoch_dim,
        deter_dim=deter_dim,
        action_dim=action_dim,
        requires_grad=True,
    )
    dynamics = LinearDynamics(stoch_dim, deter_dim, action_dim)
    U_basis = torch.eye(deter_dim)

    loss, metrics = sp_dpr_loss(
        dynamics,
        post,
        actions,
        U_basis,
        n_starts=2,
        target_horizons=(1, 2),
    )
    loss.backward()

    assert loss.shape == torch.Size([])
    assert torch.isfinite(loss)
    assert torch.equal(metrics["dpr_loss"], loss.detach())
    assert dynamics.linear.weight.grad is not None
    assert torch.any(dynamics.linear.weight.grad != 0)
    assert post["deter"].grad is None
    assert post["stoch"].grad is None
    assert actions.grad is None


def test_sp_dpr_projection() -> None:
    """Verify loss only penalizes U-space error, not orthogonal complement."""

    torch.manual_seed(1)
    batch, time, deter_dim = 3, 4, 4
    post = {
        "stoch": torch.zeros(batch, time, 2),
        "deter": torch.zeros(batch, time, deter_dim),
    }
    actions = torch.zeros(batch, time, 1)
    U_basis = basis_columns(deter_dim, (0,))

    loss_u, _ = sp_dpr_loss(
        OffsetDynamics(torch.tensor([1.0, 0.0, 0.0, 0.0])),
        post,
        actions,
        U_basis,
        n_starts=1,
        target_horizons=(1,),
    )
    loss_orth, _ = sp_dpr_loss(
        OffsetDynamics(torch.tensor([0.0, 1.0, 0.0, 0.0])),
        post,
        actions,
        U_basis,
        n_starts=1,
        target_horizons=(1,),
    )

    assert loss_u.item() > 0.0
    assert torch.equal(loss_orth, loss_orth.new_tensor(0.0))


def test_sp_dpr_no_grad_rollout() -> None:
    """Verify rollout is no_grad and only recovery img_step calls have grad."""

    torch.manual_seed(2)
    batch, time, deter_dim = 3, 5, 4
    post = {
        "stoch": torch.zeros(batch, time, 2),
        "deter": torch.zeros(batch, time, deter_dim),
    }
    actions = torch.zeros(batch, time, 1)
    dynamics = RecordingOffsetDynamics(torch.tensor([1.0, 0.0, 0.0, 0.0]))

    loss, _ = sp_dpr_loss(
        dynamics,
        post,
        actions,
        basis_columns(deter_dim, (0,)),
        n_starts=1,
        target_horizons=(1, 2),
    )
    loss.backward()

    assert dynamics.grad_enabled == [False, False, True, True]
    assert dynamics.offset.grad is not None
    assert torch.any(dynamics.offset.grad != 0)


def test_sp_dpr_short_sequence_returns_exact_zero() -> None:
    torch.manual_seed(3)
    post, actions = make_post(time=3, deter_dim=5)

    loss, metrics = sp_dpr_loss(
        LinearDynamics(stoch_dim=5, deter_dim=5, action_dim=3),
        post,
        actions,
        torch.eye(5),
        target_horizons=(2,),
    )

    assert torch.equal(loss, post["deter"].new_tensor(0.0))
    assert torch.equal(metrics["dpr_loss"], post["deter"].new_tensor(0.0))
    assert torch.equal(metrics["dpr_loss_h2"], post["deter"].new_tensor(0.0))
    assert torch.equal(metrics["dpr_drift_U_norm"], post["deter"].new_tensor(0.0))
    assert torch.equal(metrics["dpr_drift_orth_norm"], post["deter"].new_tensor(0.0))


def test_sp_dpr_validation_errors() -> None:
    torch.manual_seed(4)
    post, actions = make_post(deter_dim=5)
    dynamics = LinearDynamics(stoch_dim=5, deter_dim=5, action_dim=3)

    with pytest.raises(ValueError, match="leading dimension"):
        sp_dpr_loss(dynamics, post, actions, torch.eye(4))

    with pytest.raises(ValueError, match="Unsupported horizon_weights"):
        sp_dpr_loss(dynamics, post, actions, torch.eye(5), horizon_weights="linear")


def test_sp_dpr_batch_subsample() -> None:
    torch.manual_seed(5)
    batch, time, deter_dim = 8, 4, 4
    post = {
        "stoch": torch.zeros(batch, time, 2),
        "deter": torch.zeros(batch, time, deter_dim),
    }
    actions = torch.zeros(batch, time, 1)
    dynamics = RecordingOffsetDynamics(torch.tensor([1.0, 0.0, 0.0, 0.0]))

    _loss, metrics = sp_dpr_loss(
        dynamics,
        post,
        actions,
        basis_columns(deter_dim, (0,)),
        n_starts=1,
        target_horizons=(1,),
        batch_subsample=0.25,
    )

    assert torch.allclose(metrics["dpr_batch_subsample"], torch.tensor(0.25))
    assert dynamics.batch_sizes == [2, 2]
