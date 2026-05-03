"""Adaptive re-estimation of the SMAD drift basis."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from src.smad.U_estimation import DETER_DIM, DETER_START, estimate_U_from_drift


class AdaptiveReEstimator:
    """Estimate ``U_drift`` from fresh rollouts of the current model state."""

    def __init__(
        self,
        model: Any,
        env_name: str,
        rank: int,
        n_rollouts: int = 10,
        horizon: int = 200,
        device: str | torch.device | None = None,
        imagination_start: int = 50,
    ) -> None:
        if rank <= 0:
            raise ValueError(f"rank must be positive, got {rank}.")
        if n_rollouts <= 0:
            raise ValueError(f"n_rollouts must be positive, got {n_rollouts}.")
        if horizon <= 0:
            raise ValueError(f"horizon must be positive, got {horizon}.")
        if imagination_start < 0:
            raise ValueError(
                f"imagination_start must be non-negative, got {imagination_start}."
            )

        self.model = model
        self.env_name = env_name
        self.rank = int(rank)
        self.n_rollouts = int(n_rollouts)
        self.horizon = int(horizon)
        self.device = device
        self.imagination_start = int(imagination_start)

    def estimate(self) -> np.ndarray:
        """Collect fresh rollouts and return a rank-r drift basis."""

        prior_training = getattr(self.model, "training", None)
        if hasattr(self.model, "eval"):
            self.model.eval()

        drift_chunks = []
        try:
            with torch.no_grad():
                for seed in range(self.n_rollouts):
                    rollout = self.model.extract_rollout(
                        seed=seed,
                        total_steps=self.imagination_start + self.horizon,
                        imagination_start=self.imagination_start,
                        horizon=self.horizon,
                        include_latent=True,
                    )
                    true_latent = _require_rollout_array(
                        rollout, "true_latent", seed=seed
                    )
                    imagined_latent = _require_rollout_array(
                        rollout, "imagined_latent", seed=seed
                    )
                    _validate_latent(true_latent, "true_latent", seed)
                    _validate_latent(imagined_latent, "imagined_latent", seed)

                    true_deter = true_latent[:, DETER_START:].astype(
                        np.float64,
                        copy=False,
                    )
                    imagined_deter = imagined_latent[:, DETER_START:].astype(
                        np.float64,
                        copy=False,
                    )
                    drift_chunks.append(imagined_deter - true_deter)
        finally:
            _restore_training_mode(self.model, prior_training)

        drift_pool = np.concatenate(drift_chunks, axis=0)
        return estimate_U_from_drift(drift_pool, r=self.rank)


def _require_rollout_array(
    rollout: dict[str, Any],
    key: str,
    seed: int,
) -> np.ndarray:
    if key not in rollout:
        raise KeyError(f"rollout seed {seed} does not contain {key!r}.")
    return np.asarray(rollout[key])


def _validate_latent(array: np.ndarray, key: str, seed: int) -> None:
    if array.ndim != 2 or array.shape[1] < DETER_START + DETER_DIM:
        raise ValueError(
            f"rollout seed {seed}:{key} expected shape (T, >=1536), "
            f"got {array.shape}."
        )


def _restore_training_mode(model: Any, prior_training: Any) -> None:
    if prior_training is None:
        return
    if bool(prior_training):
        if hasattr(model, "train"):
            try:
                model.train(True)
            except TypeError:
                model.train()
    elif hasattr(model, "eval"):
        model.eval()
