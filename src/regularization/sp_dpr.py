"""Subspace-Projected Deter Prior Recovery loss for DreamerV3 RSSMs."""

from __future__ import annotations

import math
from typing import Any, Mapping

import torch
from torch import Tensor


def sp_dpr_loss(
    dynamics: Any,
    post: dict,
    actions: Tensor,
    U_basis: Tensor,
    *,
    n_starts: int = 4,
    target_horizons: tuple[int, ...] = (1, 2, 4, 8, 15),
    horizon_weights: str = "uniform",
    batch_subsample: float = 1.0,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Compute SP-DPR recovery loss.

    ``post`` and ``actions`` are batch-major tensors. ``post["deter"]`` must
    have shape ``(B, T, D)``, ``actions`` must have shape ``(B, T, A)``, and
    ``U_basis`` must have shape ``(D, R)`` with orthonormal columns.

    The future prior states are generated under ``torch.no_grad()``. Each
    selected future state is detached and used for exactly one gradient-enabled
    ``img_step`` whose deterministic error is penalized only in ``U_basis``.
    """

    if n_starts <= 0:
        raise ValueError("n_starts must be positive.")
    if not 0.0 < batch_subsample <= 1.0:
        raise ValueError("batch_subsample must be in (0, 1].")
    if "deter" not in post:
        raise ValueError("post must contain key 'deter'.")
    if "stoch" not in post:
        raise ValueError("post must contain key 'stoch'.")

    post_deter = post["deter"]
    if post_deter.ndim != 3:
        raise ValueError(f"post['deter'] must have shape (B, T, D), got {post_deter.shape}.")
    if actions.ndim != 3:
        raise ValueError(f"actions must have shape (B, T, A), got {actions.shape}.")
    if U_basis.ndim != 2:
        raise ValueError(f"U_basis must have shape (D, R), got {U_basis.shape}.")

    batch_size, time_steps, deter_dim = post_deter.shape
    if actions.shape[:2] != (batch_size, time_steps):
        raise ValueError(
            "actions and post['deter'] must share batch/time dimensions: "
            f"{actions.shape[:2]} vs {(batch_size, time_steps)}."
        )
    if U_basis.shape[0] != deter_dim:
        raise ValueError(
            f"U_basis leading dimension must match deter dim {deter_dim}, got {U_basis.shape}."
        )
    if U_basis.shape[1] <= 0:
        raise ValueError(f"U_basis must contain at least one basis vector, got {U_basis.shape}.")
    for key, value in post.items():
        if hasattr(value, "shape") and value.shape[:2] != (batch_size, time_steps):
            raise ValueError(
                f"post[{key!r}] must share batch/time dimensions, got {value.shape[:2]} "
                f"vs {(batch_size, time_steps)}."
            )

    horizons = _normalize_horizons(target_horizons)
    weights = _horizon_weights(
        horizons,
        horizon_weights,
        device=post_deter.device,
        dtype=post_deter.dtype,
    )
    U_basis = U_basis.detach().to(device=post_deter.device, dtype=post_deter.dtype)
    max_horizon = max(horizons)

    if time_steps < max_horizon + 2:
        return _zero_loss_and_metrics(post_deter, horizons, batch_subsample)

    if batch_subsample < 1.0:
        n_keep = max(1, int(round(batch_size * float(batch_subsample))))
        indices = torch.randperm(batch_size, device=post_deter.device)[:n_keep]
        post_work = {key: value[indices] for key, value in post.items()}
        actions_work = actions[indices]
    else:
        post_work = post
        actions_work = actions

    losses_by_horizon: dict[int, list[Tensor]] = {horizon: [] for horizon in horizons}
    drift_u_norms: list[Tensor] = []
    drift_orth_norms: list[Tensor] = []
    start_losses: list[Tensor] = []

    starts = torch.randint(
        0,
        time_steps - max_horizon - 1,
        (int(n_starts),),
        device=post_deter.device,
    )

    for start in starts:
        t_start = int(start.item())
        with torch.no_grad():
            state = _state_at(post_work, t_start)
            futures: dict[int, dict[str, Tensor]] = {}
            for step in range(max_horizon):
                action_step = actions_work[:, t_start + step].detach()
                state = dynamics.img_step(state, action_step, sample=False)
                state = _detach_state(state)
                horizon = step + 1
                if horizon in losses_by_horizon:
                    futures[horizon] = state

        weighted_start_loss = post_deter.new_tensor(0.0)
        with torch.enable_grad():
            for idx, horizon in enumerate(horizons):
                future_state = _detach_state(futures[horizon])
                action_h = actions_work[:, t_start + horizon].detach()
                pred_next = dynamics.img_step(future_state, action_h, sample=False)
                target_deter = post_work["deter"][:, t_start + horizon + 1].detach()

                error = pred_next["deter"] - target_deter
                error_u = error @ U_basis
                loss_h = error_u.float().pow(2).mean()
                losses_by_horizon[horizon].append(loss_h)
                weighted_start_loss = weighted_start_loss + weights[idx] * loss_h

                posterior_h = post_work["deter"][:, t_start + horizon].detach()
                drift = future_state["deter"] - posterior_h
                drift_u = drift @ U_basis
                drift_proj = drift_u @ U_basis.transpose(0, 1)
                drift_u_norms.append(drift_u.float().norm(dim=-1).mean())
                drift_orth_norms.append((drift - drift_proj).float().norm(dim=-1).mean())

        start_losses.append(weighted_start_loss)

    total_loss = torch.stack(start_losses).mean()
    metrics: dict[str, Tensor] = {
        "dpr_loss": total_loss.detach(),
        "dpr_drift_U_norm": torch.stack(drift_u_norms).mean().detach(),
        "dpr_drift_orth_norm": torch.stack(drift_orth_norms).mean().detach(),
        "dpr_n_starts": post_deter.new_tensor(float(n_starts)),
        "dpr_batch_subsample": post_deter.new_tensor(float(batch_subsample)),
    }
    for idx, horizon in enumerate(horizons):
        metrics[f"dpr_loss_h{horizon}"] = torch.stack(losses_by_horizon[horizon]).mean().detach()
        metrics[f"dpr_weight_h{horizon}"] = weights[idx].detach()
    return total_loss, metrics


def _state_at(post: Mapping[str, Tensor], t: int) -> dict[str, Tensor]:
    return {key: value[:, t].detach() for key, value in post.items()}


def _detach_state(state: Mapping[str, Tensor]) -> dict[str, Tensor]:
    return {key: value.detach() for key, value in state.items()}


def _normalize_horizons(horizons: tuple[int, ...]) -> tuple[int, ...]:
    if isinstance(horizons, int):
        horizons = (horizons,)
    normalized = tuple(sorted({int(horizon) for horizon in horizons}))
    if not normalized:
        raise ValueError("target_horizons must not be empty.")
    if any(horizon <= 0 for horizon in normalized):
        raise ValueError(f"target_horizons must be positive, got {normalized}.")
    return normalized


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


def _zero_loss_and_metrics(
    post_deter: Tensor,
    horizons: tuple[int, ...],
    batch_subsample: float,
) -> tuple[Tensor, dict[str, Tensor]]:
    zero = post_deter.new_tensor(0.0)
    metrics: dict[str, Tensor] = {
        "dpr_loss": zero,
        "dpr_drift_U_norm": zero,
        "dpr_drift_orth_norm": zero,
        "dpr_n_starts": zero,
        "dpr_batch_subsample": post_deter.new_tensor(float(batch_subsample)),
    }
    for horizon in horizons:
        metrics[f"dpr_loss_h{horizon}"] = zero
        metrics[f"dpr_weight_h{horizon}"] = zero
    return zero, metrics
