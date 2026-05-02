"""SMAD Anchor Loss.

Penalizes slow-subspace divergence between imagined and posterior deter states
at specified horizon points during training.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def activation_schedule(step: int | float, s0: int | float, s1: int | float) -> float:
    """Return the step-based activation value gamma(s)."""

    if s1 <= s0:
        raise ValueError(f"s1 must be greater than s0, got s0={s0}, s1={s1}.")
    step = float(step)
    s0 = float(s0)
    s1 = float(s1)
    if step < s0:
        return 0.0
    if step >= s1:
        return 1.0
    return float((step - s0) / (s1 - s0))


def compute_anchor_loss(
    imag_deter: torch.Tensor,
    post_deter: torch.Tensor,
    start_indices: torch.Tensor,
    U: torch.Tensor,
    anchor_points: list[int] | tuple[int, ...],
    beta: float,
    gamma: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor | int]]:
    """Compute the SMAD anchor loss.

    For each anchor point ``k``, compares ``imag_deter[k]`` against
    ``post_deter[start_indices + k]`` in the fixed drift subspace. Posterior
    projections are stop-gradient targets.
    """

    _validate_inputs(imag_deter, post_deter, start_indices, U)
    zero = imag_deter.new_zeros(())
    if float(beta) == 0.0 or float(gamma) == 0.0:
        return zero, _metrics(zero, zero, 0)

    H, B, _ = imag_deter.shape
    T = post_deter.shape[0]
    U_fixed = U.to(device=imag_deter.device, dtype=imag_deter.dtype).detach()
    start_indices = start_indices.to(device=imag_deter.device, dtype=torch.long)
    batch_indices = torch.arange(B, device=imag_deter.device)

    losses = []
    n_valid = 0
    for anchor in anchor_points:
        k = int(anchor)
        if k < 0 or k >= H:
            continue
        post_indices = start_indices + k
        valid = (post_indices >= 0) & (post_indices < T)
        if not torch.any(valid):
            continue

        imag_k = imag_deter[k, valid]
        post_k = post_deter[post_indices[valid], batch_indices[valid]]
        imag_proj = imag_k @ U_fixed
        post_proj = (post_k @ U_fixed).detach()
        squared_norm = F.mse_loss(imag_proj, post_proj, reduction="none").sum(dim=-1)
        losses.append(squared_norm)
        n_valid += int(valid.sum().item())

    if not losses:
        return zero, _metrics(zero, zero, 0)

    raw_loss = torch.cat(losses, dim=0).mean()
    scaled_loss = float(beta) * float(gamma) * raw_loss
    return scaled_loss, _metrics(raw_loss.detach(), scaled_loss.detach(), n_valid)


def _metrics(
    raw_loss: torch.Tensor,
    scaled_loss: torch.Tensor,
    n_valid: int,
) -> dict[str, torch.Tensor | int]:
    return {
        "anchor_loss_raw": raw_loss,
        "anchor_loss_scaled": scaled_loss,
        "anchor_n_valid": int(n_valid),
    }


def _validate_inputs(
    imag_deter: torch.Tensor,
    post_deter: torch.Tensor,
    start_indices: torch.Tensor,
    U: torch.Tensor,
) -> None:
    if imag_deter.ndim != 3:
        raise ValueError(f"imag_deter must have shape (H, B, D), got {imag_deter.shape}.")
    if post_deter.ndim != 3:
        raise ValueError(f"post_deter must have shape (T, B, D), got {post_deter.shape}.")
    if imag_deter.shape[1] != post_deter.shape[1]:
        raise ValueError(
            "imag_deter and post_deter must have matching batch dimension, got "
            f"{imag_deter.shape[1]} and {post_deter.shape[1]}."
        )
    if imag_deter.shape[2] != post_deter.shape[2]:
        raise ValueError(
            "imag_deter and post_deter must have matching deter dimension, got "
            f"{imag_deter.shape[2]} and {post_deter.shape[2]}."
        )
    if start_indices.ndim != 1 or start_indices.shape[0] != imag_deter.shape[1]:
        raise ValueError(
            "start_indices must have shape (B,), got "
            f"{start_indices.shape} for B={imag_deter.shape[1]}."
        )
    if U.ndim != 2 or U.shape[0] != imag_deter.shape[2]:
        raise ValueError(
            f"U must have shape (D, r) with D={imag_deter.shape[2]}, got {U.shape}."
        )
