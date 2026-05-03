"""Runtime monkey-patch for applying SMAD damping inside RSSM.img_step."""

from __future__ import annotations

import types
from collections.abc import Callable

import numpy as np
import torch


class MutableImgStepPatch:
    """Runtime SMAD patch with an updateable projection matrix."""

    def __init__(self, rssm, P: torch.Tensor, eta: float) -> None:
        self.rssm = rssm
        self.eta = float(eta)
        _validate_projector(rssm, P)
        self.P = P.detach().clone()
        self._original_img_step = rssm.img_step
        self._install()

    def update_projector(self, P: torch.Tensor) -> None:
        """Replace the active projector in place."""

        _validate_projector(self.rssm, P)
        self.P = P.detach().clone()

    def update_U(self, U: np.ndarray | torch.Tensor) -> None:
        """Replace the active projector using a ``(D, r)`` basis matrix."""

        U_tensor = torch.as_tensor(U, dtype=torch.float32, device=self.P.device)
        if U_tensor.ndim != 2:
            raise ValueError(f"U must have shape (D, r), got {tuple(U_tensor.shape)}.")
        deter_dim = getattr(self.rssm, "_deter", None)
        if deter_dim is not None and U_tensor.shape[0] != deter_dim:
            raise ValueError(
                f"U leading dimension must match RSSM deter dim {deter_dim}, "
                f"got {U_tensor.shape[0]}."
            )
        self.update_projector(U_tensor @ U_tensor.T)

    def restore(self) -> None:
        """Restore the original ``img_step`` method."""

        self.rssm.img_step = self._original_img_step

    def __call__(self) -> None:
        self.restore()

    def _install(self) -> None:
        def img_step_damped(rssm_self, prev_state, prev_action, sample=True):
            if self.eta == 0.0:
                return self._original_img_step(prev_state, prev_action, sample=sample)

            # We only need the original GRU-produced deter here. Using sample=False
            # avoids consuming a stale stochastic sample from the pre-damped deter.
            prior = self._original_img_step(prev_state, prev_action, sample=False)
            prev_deter = prev_state["deter"]
            deter = prior["deter"]
            P_local = self.P
            if P_local.device != deter.device or P_local.dtype != deter.dtype:
                P_local = P_local.to(device=deter.device, dtype=deter.dtype)

            delta = deter - prev_deter
            deter = deter - self.eta * (delta @ P_local.T)

            x = rssm_self._img_out_layers(deter)
            stats = rssm_self._suff_stats_layer("ims", x)
            if sample:
                stoch = rssm_self.get_dist(stats).sample()
            else:
                stoch = rssm_self.get_dist(stats).mode()
            return {"stoch": stoch, "deter": deter, **stats}

        self.rssm.img_step = types.MethodType(img_step_damped, self.rssm)


def patch_img_step(rssm, P: torch.Tensor, eta: float) -> Callable[[], None]:
    """Monkey-patch ``rssm.img_step`` to apply SMAD damping.

    Args:
        rssm: RSSM instance from ``external/dreamerv3-torch/networks.py``.
        P: ``(512, 512)`` projection matrix ``U @ U.T`` on the target device.
        eta: Damping strength. ``0.0`` preserves identity behavior.

    Returns:
        Callable that restores the original ``img_step`` method.
    """

    return MutableImgStepPatch(rssm, P, eta)


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
