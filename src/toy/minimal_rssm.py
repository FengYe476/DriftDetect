"""Minimal RSSM for toy experiments.

Stripped-down version of DreamerV3's RSSM with:
- GRU-based deterministic state (deter_dim)
- Gaussian stochastic state (stoch_dim), not discrete categorical
- MLP encoder, decoder, reward predictor
- img_step for open-loop imagination
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


State = dict[str, torch.Tensor]


class MinimalRSSM(nn.Module):
    """A compact Gaussian RSSM for the dual-oscillator toy task."""

    def __init__(
        self,
        obs_dim: int = 4,
        action_dim: int = 1,
        deter_dim: int = 64,
        stoch_dim: int = 8,
        hidden_dim: int = 64,
        lr: float = 3e-4,
        device: str | torch.device | None = None,
    ) -> None:
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.deter_dim = int(deter_dim)
        self.stoch_dim = int(stoch_dim)
        self.hidden_dim = int(hidden_dim)
        self.device = torch.device(device or "cpu")

        self.encoder = mlp(self.obs_dim, 2 * self.stoch_dim, self.hidden_dim)
        self.decoder = mlp(self.stoch_dim + self.deter_dim, self.obs_dim, self.hidden_dim)
        self.reward_head = mlp(self.stoch_dim + self.deter_dim, 1, self.hidden_dim)
        self.img_in = nn.Sequential(
            nn.Linear(self.stoch_dim + self.action_dim, self.hidden_dim),
            nn.SiLU(),
        )
        self.gru = nn.GRUCell(self.hidden_dim, self.deter_dim)
        self.prior = mlp(self.deter_dim, 2 * self.stoch_dim, self.hidden_dim)

        self.to(self.device)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=float(lr))

    def initial(self, batch_size: int, device: torch.device | None = None) -> State:
        device = device or self.device
        zeros_stoch = torch.zeros(batch_size, self.stoch_dim, device=device)
        zeros_deter = torch.zeros(batch_size, self.deter_dim, device=device)
        zeros_logvar = torch.zeros(batch_size, self.stoch_dim, device=device)
        return {
            "stoch": zeros_stoch,
            "deter": zeros_deter,
            "mean": zeros_stoch,
            "logvar": zeros_logvar,
        }

    def encode(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, logvar = self.encoder(obs).chunk(2, dim=-1)
        return mean, clamp_logvar(logvar)

    def prior_stats(self, deter: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, logvar = self.prior(deter).chunk(2, dim=-1)
        return mean, clamp_logvar(logvar)

    def get_feat(self, state: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return torch.cat([state["stoch"], state["deter"]], dim=-1)

    def decode_state(self, state: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return self.decoder(self.get_feat(state))

    def predict_reward(self, state: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return self.reward_head(self.get_feat(state)).squeeze(-1)

    def img_step(
        self,
        prev_state: Mapping[str, torch.Tensor],
        action: torch.Tensor,
        sample: bool = True,
    ) -> State:
        action = ensure_last_dim(action, self.action_dim)
        x = torch.cat([prev_state["stoch"], action], dim=-1)
        x = self.img_in(x)
        deter = self.gru(x, prev_state["deter"])
        mean, logvar = self.prior_stats(deter)
        stoch = sample_gaussian(mean, logvar, sample=sample)
        return {"stoch": stoch, "deter": deter, "mean": mean, "logvar": logvar}

    def obs_step(
        self,
        prev_state: Mapping[str, torch.Tensor],
        action: torch.Tensor,
        obs: torch.Tensor,
        sample: bool = True,
    ) -> tuple[State, State]:
        prior = self.img_step(prev_state, action, sample=sample)
        mean, logvar = self.encode(obs)
        stoch = sample_gaussian(mean, logvar, sample=sample)
        post = {
            "stoch": stoch,
            "deter": prior["deter"],
            "mean": mean,
            "logvar": logvar,
        }
        return post, prior

    def observe(
        self,
        obs_seq: torch.Tensor,
        action_seq: torch.Tensor,
        sample: bool = True,
    ) -> tuple[State, State]:
        obs_seq = ensure_batch_time(obs_seq, self.obs_dim, self.device)
        action_seq = ensure_batch_time(action_seq, self.action_dim, self.device)
        batch_size, horizon = obs_seq.shape[:2]
        state = self.initial(batch_size, device=obs_seq.device)
        posts = []
        priors = []
        for t in range(horizon):
            state, prior = self.obs_step(
                state,
                action_seq[:, t],
                obs_seq[:, t],
                sample=sample,
            )
            posts.append(state)
            priors.append(prior)
        return stack_states(posts, dim=1), stack_states(priors, dim=1)

    def imagine(
        self,
        start_state: Mapping[str, torch.Tensor],
        actions: torch.Tensor,
        sample: bool = True,
    ) -> State:
        actions = ensure_batch_time(actions, self.action_dim, self.device)
        state = {key: value.to(self.device) for key, value in start_state.items()}
        states = []
        for t in range(actions.shape[1]):
            state = self.img_step(state, actions[:, t], sample=sample)
            states.append(state)
        return stack_states(states, dim=1)

    def train_step(
        self,
        obs_seq: torch.Tensor,
        action_seq: torch.Tensor,
        reward_seq: torch.Tensor | None = None,
        free_bits: float = 0.0,
        kl_scale: float = 1.0,
        sample: bool = True,
    ) -> dict[str, float]:
        self.train()
        obs_seq = ensure_batch_time(obs_seq, self.obs_dim, self.device)
        action_seq = ensure_batch_time(action_seq, self.action_dim, self.device)
        post, prior = self.observe(obs_seq, action_seq, sample=sample)
        pred_obs = self.decode_state(post)

        recon_loss = F.mse_loss(pred_obs, obs_seq)
        kl_values = gaussian_kl(
            post["mean"],
            post["logvar"],
            prior["mean"],
            prior["logvar"],
        )
        if free_bits > 0:
            kl_values = torch.clamp(kl_values, min=float(free_bits))
        kl_loss = kl_values.mean()

        reward_loss = torch.zeros((), device=self.device)
        if reward_seq is not None:
            reward_seq = torch.as_tensor(reward_seq, device=self.device, dtype=torch.float32)
            if reward_seq.ndim == 3 and reward_seq.shape[-1] == 1:
                reward_seq = reward_seq.squeeze(-1)
            reward_loss = F.mse_loss(self.predict_reward(post), reward_seq)

        loss = recon_loss + float(kl_scale) * kl_loss + reward_loss
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=100.0)
        self.optimizer.step()

        return {
            "loss": float(loss.detach().cpu()),
            "recon_loss": float(recon_loss.detach().cpu()),
            "kl_loss": float(kl_loss.detach().cpu()),
            "reward_loss": float(reward_loss.detach().cpu()),
            "grad_norm": float(torch.as_tensor(grad_norm).detach().cpu()),
        }


def mlp(input_dim: int, output_dim: int, hidden_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim, output_dim),
    )


def clamp_logvar(logvar: torch.Tensor) -> torch.Tensor:
    return torch.clamp(logvar, min=-6.0, max=2.0)


def sample_gaussian(
    mean: torch.Tensor,
    logvar: torch.Tensor,
    sample: bool,
) -> torch.Tensor:
    if not sample:
        return mean
    std = torch.exp(0.5 * logvar)
    return mean + torch.randn_like(std) * std


def gaussian_kl(
    q_mean: torch.Tensor,
    q_logvar: torch.Tensor,
    p_mean: torch.Tensor,
    p_logvar: torch.Tensor,
) -> torch.Tensor:
    q_var = torch.exp(q_logvar)
    p_var = torch.exp(p_logvar)
    kl = 0.5 * (
        p_logvar
        - q_logvar
        + (q_var + (q_mean - p_mean) ** 2) / torch.clamp(p_var, min=1e-8)
        - 1.0
    )
    return kl.sum(dim=-1)


def ensure_last_dim(tensor: torch.Tensor, dim: int) -> torch.Tensor:
    if tensor.ndim == 1:
        tensor = tensor[:, None]
    if tensor.shape[-1] != dim:
        raise ValueError(f"Expected last dimension {dim}, got {tuple(tensor.shape)}.")
    return tensor


def ensure_batch_time(
    array: torch.Tensor | Any,
    dim: int,
    device: torch.device,
) -> torch.Tensor:
    tensor = torch.as_tensor(array, device=device, dtype=torch.float32)
    if tensor.ndim == 2:
        tensor = tensor[None]
    if tensor.ndim != 3 or tensor.shape[-1] != dim:
        raise ValueError(f"Expected shape (B, T, {dim}), got {tuple(tensor.shape)}.")
    return tensor


def stack_states(states: list[State], dim: int) -> State:
    if not states:
        raise ValueError("states must be non-empty.")
    return {
        key: torch.stack([state[key] for state in states], dim=dim)
        for key in states[0]
    }
