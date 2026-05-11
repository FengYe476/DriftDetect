"""Hutchinson Jacobian regularization for DreamerV3 RSSM transitions."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor


def jacobian_reg_loss(
    rssm: Any,
    post_stoch: Tensor,
    post_deter: Tensor,
    actions: Tensor,
    *,
    n_starts: int = 8,
    n_projections: int = 1,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Estimate ``||d deter_{t+1} / d deter_t||_F^2`` for an RSSM transition.

    Inputs are batch-major tensors: ``post_stoch`` and ``post_deter`` have shape
    ``(batch, time, ...)`` and ``actions`` has shape ``(batch, time, act_dim)``.
    The DreamerV3 RSSM API is the state-dict convention used by SHARP:
    ``rssm.img_step(state, action, sample=False)``.

    Backward-mode Hutchinson samples output-space vectors ``v`` and computes
    ``J^T v`` via ``torch.autograd.grad``. Since ``E[||J^T v||^2] = ||J||_F^2``
    for standard normal ``v``, this is the desired Frobenius estimator.
    """

    if n_starts <= 0:
        raise ValueError("n_starts must be positive.")
    if n_projections <= 0:
        raise ValueError("n_projections must be positive.")
    if post_deter.ndim != 3:
        raise ValueError(f"post_deter must have shape (B, T, D), got {post_deter.shape}.")
    if post_stoch.ndim < 3:
        raise ValueError(
            f"post_stoch must have at least shape (B, T, ...), got {post_stoch.shape}."
        )
    if actions.ndim != 3:
        raise ValueError(f"actions must have shape (B, T, A), got {actions.shape}.")

    batch_size, time_steps, _ = post_deter.shape
    if post_stoch.shape[:2] != (batch_size, time_steps):
        raise ValueError(
            "post_stoch and post_deter must share batch/time dimensions: "
            f"{post_stoch.shape[:2]} vs {(batch_size, time_steps)}."
        )
    if actions.shape[:2] != (batch_size, time_steps):
        raise ValueError(
            "actions and post_deter must share batch/time dimensions: "
            f"{actions.shape[:2]} vs {(batch_size, time_steps)}."
        )

    if time_steps < 2:
        zero = post_deter.new_tensor(0.0)
        return zero, {
            "jacobian_norm_estimate": zero,
            "jacobian_reg_loss": zero,
        }

    starts = torch.randint(
        0,
        time_steps - 1,
        (int(n_starts),),
        device=post_deter.device,
    )

    estimates: list[Tensor] = []
    for start in starts:
        t = int(start.item())
        deter_t = post_deter[:, t].detach().requires_grad_(True)
        state = {
            "stoch": post_stoch[:, t].detach(),
            "deter": deter_t,
        }
        action = actions[:, t + 1].detach()
        next_state = rssm.img_step(state, action, sample=False)
        next_deter = next_state["deter"]

        for _ in range(int(n_projections)):
            projection = torch.randn_like(next_deter)
            scalar = (next_deter * projection).sum()
            (grad_deter,) = torch.autograd.grad(
                scalar,
                deter_t,
                create_graph=True,
                retain_graph=True,
            )
            estimates.append(grad_deter.float().pow(2).sum(dim=-1).mean())

    loss = torch.stack(estimates).mean()
    metrics = {
        "jacobian_norm_estimate": loss.detach(),
        "jacobian_reg_loss": loss.detach(),
    }
    return loss, metrics
