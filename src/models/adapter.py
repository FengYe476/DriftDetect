"""Architecture-agnostic world model adapter interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class WorldModelAdapter(ABC):
    """
    Architecture-agnostic interface for world model rollout extraction.
    Decouples the diagnostic pipeline from specific implementations
    (DreamerV3 RSSM, DreamerV4 Transformer, etc.)
    """

    @abstractmethod
    def load_checkpoint(self, path: str) -> None:
        """Load model weights from checkpoint file."""
        ...

    @abstractmethod
    def reset(self, seed: int = 0) -> np.ndarray:
        """Reset environment and return initial observation."""
        ...

    @abstractmethod
    def encode(self, obs: np.ndarray) -> dict:
        """Encode observation into latent state."""
        ...

    @abstractmethod
    def imagine(
        self,
        init_obs: np.ndarray,
        actions: np.ndarray,
        horizon: int,
    ) -> np.ndarray:
        """
        Run open-loop imagination from initial observation.

        Args:
            init_obs: Initial observation to condition on (T, obs_dim)
            actions: Action sequence to imagine (H, act_dim)
            horizon: Number of imagination steps H

        Returns:
            imagined_obs: Imagined observations (H, obs_dim)
        """
        ...

    @abstractmethod
    def collect_true_rollout(
        self,
        seed: int,
        total_steps: int,
        imagination_start: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Collect real environment rollout.

        Returns:
            true_obs: (T, obs_dim)
            actions: (T, act_dim)
            rewards: (T,)
        """
        ...

    def extract_rollout(
        self,
        seed: int,
        total_steps: int,
        imagination_start: int,
        horizon: int,
    ) -> dict:
        """
        Full pipeline: collect true rollout + run imagination.
        Returns dict with true_obs, imagined_obs, actions, rewards.
        This method is NOT abstract - subclasses get it for free.
        """
        true_obs, actions, rewards = self.collect_true_rollout(
            seed=seed,
            total_steps=total_steps,
            imagination_start=imagination_start,
        )
        init_obs = true_obs[:imagination_start]
        imagine_actions = actions[imagination_start : imagination_start + horizon]
        imagined_obs = self.imagine(
            init_obs=init_obs,
            actions=imagine_actions,
            horizon=horizon,
        )
        return {
            "true_obs": true_obs[imagination_start : imagination_start + horizon],
            "imagined_obs": imagined_obs,
            "actions": imagine_actions,
            "rewards": rewards[imagination_start : imagination_start + horizon],
        }
