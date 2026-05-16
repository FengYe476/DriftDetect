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


def amplification_control_loss_lite(
    dynamics: Any,
    post_stoch: Tensor,
    post_deter: Tensor,
    actions: Tensor,
    *,
    n_dirs: int = 1,
    delta_scale: float = 0.01,
    rho: float = 1.0,
    future_steps: tuple[int, ...] = (0, 1, 2, 4, 8),
    horizon_weights: str = "uniform",
    batch_subsample: float = 1.0,
    loss_type: str = "log1p",
) -> tuple[Tensor, dict[str, Tensor]]:
    """Compute local future-state amplification control without chain gradients.

    The future states are generated with ``torch.no_grad()``. Each selected
    future state is then detached and checked with a one-step guarded gain loss,
    so gradients reach transition parameters through only one ``img_step``.
    """

    if n_dirs <= 0:
        raise ValueError("n_dirs must be positive.")
    if delta_scale <= 0:
        raise ValueError("delta_scale must be positive.")
    if not 0.0 < batch_subsample <= 1.0:
        raise ValueError("batch_subsample must be in (0, 1].")
    if post_deter.ndim != 2:
        raise ValueError(f"post_deter must have shape (B, D), got {post_deter.shape}.")
    if post_stoch.ndim < 2:
        raise ValueError(f"post_stoch must have shape (B, ...), got {post_stoch.shape}.")
    if actions.ndim != 3:
        raise ValueError(f"actions must have shape (B, H, A), got {actions.shape}.")

    batch_size = post_deter.shape[0]
    if post_stoch.shape[0] != batch_size:
        raise ValueError(
            "post_stoch and post_deter must share batch dimension: "
            f"{post_stoch.shape[0]} vs {batch_size}."
        )
    if actions.shape[0] != batch_size:
        raise ValueError(
            "actions and post_deter must share batch dimension: "
            f"{actions.shape[0]} vs {batch_size}."
        )
    if actions.shape[1] < 1:
        raise ValueError("actions must contain at least one time step.")

    future_steps = _normalize_future_steps(future_steps)
    max_future = max(future_steps)
    _validate_loss_type(loss_type)

    if batch_subsample < 1.0:
        n_keep = max(1, int(round(batch_size * float(batch_subsample))))
        indices = torch.randperm(batch_size, device=post_deter.device)[:n_keep]
        post_stoch = post_stoch[indices]
        post_deter = post_deter[indices]
        actions = actions[indices]

    post_stoch = post_stoch.detach()
    post_deter = post_deter.detach()
    actions = actions.detach()
    weights = _future_step_weights(
        future_steps,
        horizon_weights,
        device=post_deter.device,
        dtype=post_deter.dtype,
    )

    with torch.no_grad():
        futures: list[dict[str, Tensor]] = [{"stoch": post_stoch, "deter": post_deter}]
        state = futures[0]
        for step in range(max_future):
            step_action = actions[:, min(step, actions.shape[1] - 1)]
            state = dynamics.img_step(state, step_action, sample=False)
            futures.append({
                "stoch": state["stoch"].detach(),
                "deter": state["deter"].detach(),
            })

    gains_by_future: dict[int, list[Tensor]] = {step: [] for step in future_steps}
    with torch.enable_grad():
        for future_step in future_steps:
            future_state = futures[future_step]
            stoch_h = future_state["stoch"].detach()
            deter_h = future_state["deter"].detach()
            action_h = actions[:, min(future_step, actions.shape[1] - 1)].detach()

            for _ in range(int(n_dirs)):
                delta = _sample_delta(deter_h, float(delta_scale))
                delta_norm = delta.float().norm(dim=-1).clamp_min(1e-12)
                base_next = dynamics.img_step(
                    {"stoch": stoch_h, "deter": deter_h},
                    action_h,
                    sample=False,
                )
                pert_next = dynamics.img_step(
                    {"stoch": stoch_h, "deter": deter_h + delta},
                    action_h,
                    sample=False,
                )
                diff = pert_next["deter"] - base_next["deter"]
                gain = diff.float().norm(dim=-1) / delta_norm
                gains_by_future[future_step].append(gain)

    total_loss = post_deter.new_tensor(0.0)
    metrics: dict[str, Tensor] = {}
    for idx, future_step in enumerate(future_steps):
        gains = torch.cat(gains_by_future[future_step], dim=0)
        loss_h = _guarded_gain_loss(gains, float(rho), loss_type).mean()
        total_loss = total_loss + weights[idx] * loss_h

        gains_detached = gains.detach()
        metrics[f"ac_gain_f{future_step}_mean"] = gains_detached.mean()
        metrics[f"ac_gain_f{future_step}_p90"] = gains_detached.quantile(0.90)
        metrics[f"ac_gain_f{future_step}_max"] = gains_detached.max()
        metrics[f"ac_loss_f{future_step}"] = loss_h.detach()
        metrics[f"ac_weight_f{future_step}"] = weights[idx].detach()

    metrics["ac_loss"] = total_loss.detach()
    metrics["ac_rho"] = post_deter.new_tensor(float(rho))
    metrics["ac_delta_scale"] = post_deter.new_tensor(float(delta_scale))
    metrics["ac_n_dirs"] = post_deter.new_tensor(float(n_dirs))
    metrics["ac_batch_subsample"] = post_deter.new_tensor(float(batch_subsample))
    metrics["ac_loss_type_log1p"] = post_deter.new_tensor(float(loss_type == "log1p"))
    metrics["ac_loss_type_squared"] = post_deter.new_tensor(float(loss_type == "squared"))
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


def _normalize_future_steps(future_steps: tuple[int, ...]) -> tuple[int, ...]:
    if isinstance(future_steps, int):
        future_steps = (future_steps,)
    normalized = tuple(int(step) for step in future_steps)
    if not normalized:
        raise ValueError("future_steps must not be empty.")
    if any(step < 0 for step in normalized):
        raise ValueError(f"future_steps must be nonnegative, got {normalized}.")
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


def _future_step_weights(
    future_steps: tuple[int, ...],
    mode: str,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    if mode == "uniform":
        weights = torch.ones(len(future_steps), device=device, dtype=dtype)
    elif mode == "sqrt_inv":
        weights = torch.tensor(
            [1.0 / math.sqrt(float(step + 1)) for step in future_steps],
            device=device,
            dtype=dtype,
        )
    else:
        raise ValueError(f"Unsupported horizon_weights: {mode!r}.")
    return weights / weights.sum()


def _validate_loss_type(loss_type: str) -> None:
    if loss_type not in {"log1p", "squared"}:
        raise ValueError(f"Unsupported loss_type: {loss_type!r}.")


def _guarded_gain_loss(gain: Tensor, rho: float, loss_type: str) -> Tensor:
    excess = torch.relu(gain - float(rho))
    if loss_type == "log1p":
        return torch.log1p(excess)
    if loss_type == "squared":
        return excess.pow(2)
    raise ValueError(f"Unsupported loss_type: {loss_type!r}.")


def _sample_delta(post_deter: Tensor, delta_scale: float) -> Tensor:
    delta = torch.randn_like(post_deter)
    delta_norm = delta.float().norm(dim=-1, keepdim=True).clamp_min(1e-12)
    unit_delta = delta / delta_norm.to(dtype=delta.dtype)
    return unit_delta * float(delta_scale)
