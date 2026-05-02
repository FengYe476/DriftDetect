"""Dual-oscillator environment for SMAD toy validation.

State: [x_slow, x_fast] where:
  x_slow = A_slow * sin(2*pi*f_slow*t + phi_slow) + noise
  x_fast = A_fast * sin(2*pi*f_fast*t + phi_fast) + noise

Default: f_slow=0.5 Hz (period=2s), f_fast=5 Hz (period=0.2s)
Action: 1-dimensional, affects both oscillators with different coupling
Observation: [x_slow, x_fast, x_slow_dot, x_fast_dot] (4-dim)
Reward: -||obs||^2 (keep state near zero)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Space:
    """Small stand-in for gym-style spaces used by the toy scripts."""

    shape: tuple[int, ...]


class DualOscillatorEnv:
    """Simple controlled two-frequency oscillator."""

    observation_space = Space((4,))
    action_space = Space((1,))

    def __init__(
        self,
        f_slow: float = 0.5,
        f_fast: float = 5.0,
        A_slow: float = 1.0,
        A_fast: float = 0.5,
        dt: float = 0.01,
        noise_std: float = 0.01,
        episode_length: int = 200,
        coupling_slow: float = 0.20,
        coupling_fast: float = 0.05,
        action_clip: float = 1.0,
        seed: int | None = None,
    ) -> None:
        self.f_slow = float(f_slow)
        self.f_fast = float(f_fast)
        self.A_slow = float(A_slow)
        self.A_fast = float(A_fast)
        self.dt = float(dt)
        self.noise_std = float(noise_std)
        self.episode_length = int(episode_length)
        self.coupling_slow = float(coupling_slow)
        self.coupling_fast = float(coupling_fast)
        self.action_clip = float(action_clip)
        self.rng = np.random.default_rng(seed)

        self.t = 0.0
        self.step_count = 0
        self.phi_slow = 0.0
        self.phi_fast = 0.0
        self.x_slow = 0.0
        self.x_fast = 0.0
        self.dx_slow = 0.0
        self.dx_fast = 0.0

    def reset(self) -> np.ndarray:
        """Reset with random oscillator phases and return the first observation."""

        self.t = 0.0
        self.step_count = 0
        self.phi_slow = float(self.rng.uniform(0.0, 2.0 * math.pi))
        self.phi_fast = float(self.rng.uniform(0.0, 2.0 * math.pi))
        self.x_slow = self.A_slow * math.sin(self.phi_slow)
        self.x_fast = self.A_fast * math.sin(self.phi_fast)
        self.dx_slow, self.dx_fast = self._base_derivatives(action=0.0)
        return self._obs()

    def step(self, action: np.ndarray | list[float] | float) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        """Advance one timestep."""

        action_scalar = float(np.asarray(action, dtype=np.float64).reshape(-1)[0])
        action_scalar = float(np.clip(action_scalar, -self.action_clip, self.action_clip))

        base_slow, base_fast = self._base_derivatives(action=0.0)
        noise_slow = float(self.rng.normal(0.0, self.noise_std))
        noise_fast = float(self.rng.normal(0.0, self.noise_std))
        self.dx_slow = base_slow + self.coupling_slow * action_scalar + noise_slow
        self.dx_fast = base_fast + self.coupling_fast * action_scalar + noise_fast

        self.x_slow += self.dt * self.dx_slow
        self.x_fast += self.dt * self.dx_fast
        self.t += self.dt
        self.step_count += 1

        obs = self._obs()
        reward = -float(np.sum(obs**2))
        done = self.step_count >= self.episode_length
        info = {
            "t": self.t,
            "step": self.step_count,
            "x_slow": self.x_slow,
            "x_fast": self.x_fast,
            "dx_slow": self.dx_slow,
            "dx_fast": self.dx_fast,
        }
        return obs, reward, done, info

    def _base_derivatives(self, action: float) -> tuple[float, float]:
        omega_slow = 2.0 * math.pi * self.f_slow
        omega_fast = 2.0 * math.pi * self.f_fast
        dx_slow = omega_slow * self.A_slow * math.cos(omega_slow * self.t + self.phi_slow)
        dx_fast = omega_fast * self.A_fast * math.cos(omega_fast * self.t + self.phi_fast)
        dx_slow += self.coupling_slow * action
        dx_fast += self.coupling_fast * action
        return dx_slow, dx_fast

    def _obs(self) -> np.ndarray:
        return np.asarray(
            [self.x_slow, self.x_fast, self.dx_slow, self.dx_fast],
            dtype=np.float32,
        )
