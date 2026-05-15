"""Guarded amplification-control loss for RSSM transitions."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor


def amplification_control_loss(
    dynamics: Any,
    post_stoch: Tensor,
    post_deter: Tensor,
    action: Tensor,
    *,
    n_dirs: int = 1,
    delta_scale: float = 0.01,
    rho: float = 1.0,
    horizons: tuple[int, ...] = (1,),
    horizon_weights: str = "uniform",
) -> tuple[Tensor, dict[str, Tensor]]:
    """Compute guarded amplification control loss.

    Inputs are batch-major tensors. ``post_stoch`` has shape ``(B, ...)``,
    ``post_deter`` has shape ``(B, D)``, and ``action`` is either a single
    action ``(B, A)`` for one-step control or an action sequence ``(B, H, A)``
    for multi-step horizons.
    """

    if n_dirs <= 0:
        raise ValueError("n_dirs must be positive.")
    if delta_scale <= 0:
        raise ValueError("delta_scale must be positive.")
    if post_deter.ndim != 2:
        raise ValueError(f"post_deter must have shape (B, D), got {post_deter.shape}.")
    if post_stoch.ndim < 2:
        raise ValueError(f"post_stoch must have shape (B, ...), got {post_stoch.shape}.")

    batch_size = post_deter.shape[0]
    if post_stoch.shape[0] != batch_size:
        raise ValueError(
            "post_stoch and post_deter must share batch dimension: "
            f"{post_stoch.shape[0]} vs {batch_size}."
        )
    if action.shape[0] != batch_size:
        raise ValueError(
            "action and post_deter must share batch dimension: "
            f"{action.shape[0]} vs {batch_size}."
        )

    horizons = _normalize_horizons(horizons)
    max_horizon = max(horizons)
    actions = _prepare_actions(action, max_horizon).detach()
    weights = _horizon_weights(
        horizons,
        horizon_weights,
        device=post_deter.device,
        dtype=post_deter.dtype,
    )

    post_stoch = post_stoch.detach()
    post_deter = post_deter.detach()

    gains_by_horizon: dict[int, list[Tensor]] = {horizon: [] for horizon in horizons}
    for _ in range(int(n_dirs)):
        delta = _sample_delta(post_deter, float(delta_scale))
        delta_norm = delta.float().norm(dim=-1).clamp_min(1e-12)

        base_state = {"stoch": post_stoch, "deter": post_deter}
        pert_state = {"stoch": post_stoch, "deter": post_deter + delta}

        for step in range(max_horizon):
            step_action = actions[:, step]
            base_state = dynamics.img_step(base_state, step_action, sample=False)
            pert_state = dynamics.img_step(pert_state, step_action, sample=False)

            horizon = step + 1
            if horizon in gains_by_horizon:
                diff = pert_state["deter"] - base_state["deter"]
                gain = diff.float().norm(dim=-1) / delta_norm
                gains_by_horizon[horizon].append(gain)

    total_loss = post_deter.new_tensor(0.0)
    metrics: dict[str, Tensor] = {}
    for idx, horizon in enumerate(horizons):
        gains = torch.cat(gains_by_horizon[horizon], dim=0)
        loss_h = torch.relu(gains - float(rho)).pow(2).mean()
        total_loss = total_loss + weights[idx] * loss_h

        gains_detached = gains.detach()
        metrics[f"ac_gain_h{horizon}_mean"] = gains_detached.mean()
        metrics[f"ac_gain_h{horizon}_p90"] = gains_detached.quantile(0.90)
        metrics[f"ac_gain_h{horizon}_max"] = gains_detached.max()
        metrics[f"ac_loss_h{horizon}"] = loss_h.detach()
        metrics[f"ac_weight_h{horizon}"] = weights[idx].detach()

    metrics["ac_loss"] = total_loss.detach()
    metrics["ac_rho"] = post_deter.new_tensor(float(rho))
    metrics["ac_delta_scale"] = post_deter.new_tensor(float(delta_scale))
    metrics["ac_n_dirs"] = post_deter.new_tensor(float(n_dirs))
    return total_loss, metrics


def _normalize_horizons(horizons: tuple[int, ...]) -> tuple[int, ...]:
    if isinstance(horizons, int):
        horizons = (horizons,)
    normalized = tuple(int(horizon) for horizon in horizons)
    if not normalized:
        raise ValueError("horizons must not be empty.")
    if any(horizon <= 0 for horizon in normalized):
        raise ValueError(f"horizons must be positive, got {normalized}.")
    return normalized


def _prepare_actions(action: Tensor, max_horizon: int) -> Tensor:
    if action.ndim == 2:
        if max_horizon > 1:
            raise ValueError(
                "Multi-step horizons require action with shape (B, H, A), "
                f"got {action.shape} for max horizon {max_horizon}."
            )
        return action.unsqueeze(1)
    if action.ndim == 3:
        if action.shape[1] < max_horizon:
            raise ValueError(
                "Action sequence is shorter than max horizon: "
                f"{action.shape[1]} < {max_horizon}."
            )
        return action
    raise ValueError(f"action must have shape (B, A) or (B, H, A), got {action.shape}.")


def _horizon_weights(
    horizons: tuple[int, ...],
    mode: str,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    if mode == "uniform":
        weights = torch.ones(len(horizons), device=device, dtype=dtype)
    elif mode == "sqrt_inv":
        weights = torch.tensor(
            [1.0 / math.sqrt(float(horizon)) for horizon in horizons],
            device=device,
            dtype=dtype,
        )
    else:
        raise ValueError(f"Unsupported horizon_weights: {mode!r}.")
    return weights / weights.sum()


def _sample_delta(post_deter: Tensor, delta_scale: float) -> Tensor:
    delta = torch.randn_like(post_deter)
    delta_norm = delta.float().norm(dim=-1, keepdim=True).clamp_min(1e-12)
    unit_delta = delta / delta_norm.to(dtype=delta.dtype)
    return unit_delta * float(delta_scale)
