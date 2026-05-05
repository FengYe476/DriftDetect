"""Residual Correction Head for imagination drift.

A lightweight MLP that predicts per-step drift correction during RSSM
imagination. The correction is subtracted from the deterministic state after
each ``img_step``, reducing drift accumulation while keeping the correction
path trainable inside the model distribution.

The head is step-aware: it receives a sinusoidal step embedding so it can learn
that early imagination steps need little correction while later steps may need
more.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch
import torch.nn as nn


class StepEmbedding(nn.Module):
    """Sinusoidal step embedding, similar to positional encoding."""

    def __init__(self, embed_dim: int = 32, max_steps: int = 200) -> None:
        super().__init__()
        if embed_dim <= 0:
            raise ValueError(f"embed_dim must be positive, got {embed_dim}.")
        if max_steps <= 1:
            raise ValueError(f"max_steps must be greater than 1, got {max_steps}.")

        self.embed_dim = int(embed_dim)
        self.max_steps = int(max_steps)
        half_dim = (self.embed_dim + 1) // 2
        exponent = torch.arange(half_dim, dtype=torch.float32)
        exponent = exponent / max(half_dim - 1, 1)
        frequencies = torch.exp(-math.log(float(self.max_steps)) * exponent)
        self.register_buffer("frequencies", frequencies, persistent=False)

    def forward(self, step: int | torch.Tensor) -> torch.Tensor:
        """Return sinusoidal embedding for one step or a batch of steps."""

        if torch.is_tensor(step):
            steps = step.to(device=self.frequencies.device, dtype=self.frequencies.dtype)
        else:
            steps = torch.tensor(
                step,
                device=self.frequencies.device,
                dtype=self.frequencies.dtype,
            )

        angles = steps.unsqueeze(-1) * self.frequencies
        embedding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        return embedding[..., : self.embed_dim]


class CorrectionHead(nn.Module):
    """Predict drift correction from current deter state plus step embedding."""

    def __init__(
        self,
        deter_dim: int = 512,
        step_embed_dim: int = 32,
        hidden_dim: int = 256,
        max_steps: int = 200,
        init_scale: float = 0.01,
    ) -> None:
        super().__init__()
        if deter_dim <= 0:
            raise ValueError(f"deter_dim must be positive, got {deter_dim}.")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}.")
        if init_scale < 0.0:
            raise ValueError(f"init_scale must be non-negative, got {init_scale}.")

        self.deter_dim = int(deter_dim)
        self.step_embedding = StepEmbedding(
            embed_dim=step_embed_dim,
            max_steps=max_steps,
        )
        self.net = nn.Sequential(
            nn.Linear(self.deter_dim + step_embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim, eps=1e-3),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim, eps=1e-3),
            nn.SiLU(),
        )
        self.output = nn.Linear(hidden_dim, self.deter_dim)
        nn.init.uniform_(self.output.weight, -float(init_scale), float(init_scale))
        nn.init.uniform_(self.output.bias, -float(init_scale), float(init_scale))
        self.gate = nn.Parameter(torch.tensor(0.0))

    def forward(self, h_deter: torch.Tensor, step: int | torch.Tensor) -> torch.Tensor:
        """Predict correction for the deterministic state.

        Args:
            h_deter: Current deterministic state with shape ``(batch, deter_dim)``.
            step: Imagination step index, either scalar or shape ``(batch,)``.

        Returns:
            Correction tensor with shape ``(batch, deter_dim)``.
        """

        if h_deter.ndim != 2:
            raise ValueError(
                f"h_deter must have shape (batch, deter_dim), got {tuple(h_deter.shape)}."
            )
        if h_deter.shape[-1] != self.deter_dim:
            raise ValueError(
                f"h_deter last dimension must be {self.deter_dim}, "
                f"got {h_deter.shape[-1]}."
            )

        step_embed = self.step_embedding(step).to(
            device=h_deter.device,
            dtype=h_deter.dtype,
        )
        if step_embed.ndim == 1:
            step_embed = step_embed.expand(h_deter.shape[0], -1)
        elif step_embed.ndim == 2 and step_embed.shape[0] == h_deter.shape[0]:
            pass
        else:
            raise ValueError(
                "step must be scalar or have shape (batch,), got embedding shape "
                f"{tuple(step_embed.shape)} for batch={h_deter.shape[0]}."
            )

        features = torch.cat([h_deter, step_embed], dim=-1)
        raw_correction = self.output(self.net(features))
        return self.gate * raw_correction

    @property
    def gate_value(self) -> float:
        """Current scalar gate value for monitoring."""

        return float(self.gate.detach().cpu().item())


def apply_correction(
    state: Mapping[str, torch.Tensor],
    correction_head: CorrectionHead,
    step: int | torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Apply correction to an RSSM state dict.

    Creates a shallow copy of ``state`` and replaces only ``deter`` with the
    corrected deterministic state. The stochastic state and distribution stats
    are intentionally left unchanged for this standalone v1 module.
    """

    if "deter" not in state:
        raise ValueError("state must contain a 'deter' tensor.")

    corrected = dict(state)
    deter = state["deter"]
    corrected["deter"] = deter - correction_head(deter, step)
    return corrected
