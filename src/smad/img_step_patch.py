"""Runtime monkey-patch for applying SMAD damping inside RSSM.img_step."""

from __future__ import annotations

import types
from collections.abc import Callable

import torch


def patch_img_step(rssm, P: torch.Tensor, eta: float) -> Callable[[], None]:
    """Monkey-patch ``rssm.img_step`` to apply SMAD damping.

    Args:
        rssm: RSSM instance from ``external/dreamerv3-torch/networks.py``.
        P: ``(512, 512)`` projection matrix ``U @ U.T`` on the target device.
        eta: Damping strength. ``0.0`` preserves identity behavior.

    Returns:
        Callable that restores the original ``img_step`` method.
    """

    eta = float(eta)
    _validate_projector(rssm, P)
    original_img_step = rssm.img_step

    def img_step_damped(self, prev_state, prev_action, sample=True):
        if eta == 0.0:
            return original_img_step(prev_state, prev_action, sample=sample)

        # We only need the original GRU-produced deter here. Using sample=False
        # avoids consuming a stale stochastic sample from the pre-damped deter.
        prior = original_img_step(prev_state, prev_action, sample=False)
        prev_deter = prev_state["deter"]
        deter = prior["deter"]
        P_local = P
        if P_local.device != deter.device or P_local.dtype != deter.dtype:
            P_local = P_local.to(device=deter.device, dtype=deter.dtype)

        delta = deter - prev_deter
        deter = deter - eta * (delta @ P_local.T)

        x = self._img_out_layers(deter)
        stats = self._suff_stats_layer("ims", x)
        if sample:
            stoch = self.get_dist(stats).sample()
        else:
            stoch = self.get_dist(stats).mode()
        return {"stoch": stoch, "deter": deter, **stats}

    rssm.img_step = types.MethodType(img_step_damped, rssm)

    def restore_fn() -> None:
        rssm.img_step = original_img_step

    return restore_fn


def _validate_projector(rssm, P: torch.Tensor) -> None:
    if not isinstance(P, torch.Tensor):
        raise TypeError(f"P must be a torch.Tensor, got {type(P).__name__}.")
    if P.ndim != 2 or P.shape[0] != P.shape[1]:
        raise ValueError(f"P must be a square matrix, got shape {tuple(P.shape)}.")
    deter_dim = getattr(rssm, "_deter", None)
    if deter_dim is not None and P.shape != (deter_dim, deter_dim):
        raise ValueError(
            f"P shape must match RSSM deter dim {(deter_dim, deter_dim)}, "
            f"got {tuple(P.shape)}."
        )
