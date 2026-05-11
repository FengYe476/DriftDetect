from __future__ import annotations

import sys
import types
import importlib.util
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
    from src.regularization.jacobian_reg import jacobian_reg_loss
else:
    jacobian_reg_loss = None


class FakeRSSM(nn.Module):
    def __init__(self, stoch_flat_dim: int, deter_dim: int, action_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(stoch_flat_dim + deter_dim + action_dim, deter_dim)

    def img_step(
        self,
        state: dict[str, torch.Tensor],
        action: torch.Tensor,
        sample: bool = False,
    ) -> dict[str, torch.Tensor]:
        del sample
        stoch = state["stoch"].reshape(state["stoch"].shape[0], -1)
        inputs = torch.cat([stoch, state["deter"], action], dim=-1)
        deter = torch.tanh(self.linear(inputs))
        return {"stoch": state["stoch"], "deter": deter}


def make_inputs(
    *,
    batch: int = 4,
    time: int = 6,
    stoch_shape: tuple[int, ...] = (5,),
    deter_dim: int = 7,
    action_dim: int = 3,
) -> tuple[FakeRSSM, torch.Tensor, torch.Tensor, torch.Tensor]:
    stoch_flat_dim = 1
    for dim in stoch_shape:
        stoch_flat_dim *= dim
    rssm = FakeRSSM(stoch_flat_dim, deter_dim, action_dim)
    post_stoch = torch.randn(batch, time, *stoch_shape)
    post_deter = torch.randn(batch, time, deter_dim)
    actions = torch.randn(batch, time, action_dim)
    return rssm, post_stoch, post_deter, actions


def test_lambda_zero_weighted_loss_is_exact_zero() -> None:
    torch.manual_seed(0)
    rssm, post_stoch, post_deter, actions = make_inputs()

    loss, _ = jacobian_reg_loss(rssm, post_stoch, post_deter, actions)
    weighted = 0.0 * loss

    assert torch.equal(weighted, loss.new_tensor(0.0))


def test_gradients_flow_to_rssm_parameters() -> None:
    torch.manual_seed(1)
    rssm, post_stoch, post_deter, actions = make_inputs()

    loss, metrics = jacobian_reg_loss(
        rssm,
        post_stoch,
        post_deter,
        actions,
        n_starts=3,
        n_projections=2,
    )
    loss.backward()

    assert loss.ndim == 0
    assert metrics["jacobian_norm_estimate"].ndim == 0
    assert rssm.linear.weight.grad is not None
    assert torch.any(rssm.linear.weight.grad != 0)


@pytest.mark.parametrize(
    ("batch", "time", "stoch_shape"),
    [
        (1, 2, (3,)),
        (3, 5, (4,)),
        (2, 4, (3, 2)),
    ],
)
def test_shape_variants_return_scalar_loss_and_metrics(
    batch: int,
    time: int,
    stoch_shape: tuple[int, ...],
) -> None:
    torch.manual_seed(batch * 100 + time)
    rssm, post_stoch, post_deter, actions = make_inputs(
        batch=batch,
        time=time,
        stoch_shape=stoch_shape,
    )

    loss, metrics = jacobian_reg_loss(
        rssm,
        post_stoch,
        post_deter,
        actions,
        n_starts=2,
        n_projections=1,
    )

    assert loss.shape == torch.Size([])
    assert metrics["jacobian_norm_estimate"].shape == torch.Size([])
    assert metrics["jacobian_reg_loss"].shape == torch.Size([])
    assert loss.item() >= 0.0


def test_short_sequence_returns_exact_zero() -> None:
    torch.manual_seed(2)
    rssm, post_stoch, post_deter, actions = make_inputs(time=1)

    loss, metrics = jacobian_reg_loss(rssm, post_stoch, post_deter, actions)

    assert torch.equal(loss, post_deter.new_tensor(0.0))
    assert torch.equal(metrics["jacobian_norm_estimate"], post_deter.new_tensor(0.0))
    assert torch.equal(metrics["jacobian_reg_loss"], post_deter.new_tensor(0.0))
