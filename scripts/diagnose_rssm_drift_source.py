#!/usr/bin/env python3
"""Diagnose which RSSM component carries open-loop drift.

All diagnostic computations are forward-only and run under torch.no_grad().
The script follows the corrected DreamerV3 tensor convention:

    post["deter"] shape is [B, T, D]
    state_t = {k: v[:, t].detach() for k, v in post.items()}
    action_t = actions[:, t].detach()
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DREAMERV3_ROOT = PROJECT_ROOT / "external" / "dreamerv3-torch"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "tables" / "rssm_drift_source_diagnostics.json"

if "DRIFTDETECT_MUJOCO_GL" not in os.environ and sys.platform == "darwin":
    os.environ["DRIFTDETECT_MUJOCO_GL"] = "disable"
if "DRIFTDETECT_MUJOCO_GL" in os.environ:
    os.environ["MUJOCO_GL"] = os.environ["DRIFTDETECT_MUJOCO_GL"]


class NullLogger:
    """Minimal logger surface required by NM512's Dreamer constructor."""

    step = 0

    def scalar(self, *_args: Any, **_kwargs: Any) -> None:
        pass

    def video(self, *_args: Any, **_kwargs: Any) -> None:
        pass

    def write(self, *_args: Any, **_kwargs: Any) -> None:
        pass


def main() -> None:
    args = parse_args()
    for path in (DREAMERV3_ROOT, PROJECT_ROOT):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    import dreamer
    import torch

    checkpoint = resolve_path(args.checkpoint)
    config_path = resolve_path(args.config)
    output_path = resolve_path(args.output)
    replay_dir = resolve_path(args.replay_dir) if args.replay_dir else checkpoint.parent / "train_eps"

    config = load_run_config(
        dreamer_module=dreamer,
        config_path=config_path,
        checkpoint=checkpoint,
        replay_dir=replay_dir,
        device=args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"),
    )
    dreamer.tools.set_seed_everywhere(config.seed)

    milestones = normalize_milestones(args.milestones, args.horizon)
    print_header(checkpoint, config_path, replay_dir, output_path, config, args, milestones)

    agent, env = build_agent(dreamer, torch, config, checkpoint)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    try:
        diagnostics, collection_meta = collect_rssm_drift_source(
            dreamer_module=dreamer,
            torch_module=torch,
            world_model=agent._wm,
            config=config,
            replay_dir=replay_dir,
            horizon=args.horizon,
            reset_interval=args.reset_interval,
            n_samples=args.n_samples,
            milestones=milestones,
            alpha=args.alpha,
            eps=args.eps,
            dataset_limit=args.dataset_limit,
        )
    finally:
        try:
            env.close()
        except Exception:
            pass

    results = {
        "metadata": {
            "checkpoint": str(checkpoint),
            "config": str(config_path),
            "replay_dir": str(replay_dir),
            "output": str(output_path),
            "device": str(config.device),
            "task": str(config.task),
            "seed": int(config.seed),
            "diagnostic_seed": int(args.seed),
            "horizon": int(args.horizon),
            "reset_interval": int(args.reset_interval),
            "n_samples_requested": int(args.n_samples),
            "alpha": float(args.alpha),
            "eps": float(args.eps),
            "milestones": list(milestones),
            **collection_meta,
        },
        **diagnostics,
    }

    print_results(results)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(json_sanitize(results), indent=2, sort_keys=True) + "\n")
    print(f"\nSaved RSSM drift source diagnostics to {display_path(output_path)}")


def collect_rssm_drift_source(
    *,
    dreamer_module: Any,
    torch_module: Any,
    world_model: Any,
    config: SimpleNamespace,
    replay_dir: Path,
    horizon: int,
    reset_interval: int,
    n_samples: int,
    milestones: Sequence[int],
    alpha: float,
    eps: float,
    dataset_limit: int | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    episodes = dreamer_module.tools.load_episodes(
        replay_dir,
        limit=dataset_limit or config.dataset_size,
    )
    if not episodes:
        raise FileNotFoundError(f"No replay episodes found in {replay_dir}")

    dataset = dreamer_module.make_dataset(episodes, config)
    records = {
        "deter_stoch_decomposition": init_record_map(milestones),
        "reset_experiment": {mode: [] for mode in RESET_MODES},
        "directional_gain_attribution": init_record_map(milestones),
        "sensitivity_decomposition": init_record_map(milestones),
        "decoder_attribution": init_record_map(milestones),
    }
    sample_meta: list[dict[str, int]] = []
    batches_read = 0
    decoder_available_any = False

    with torch_module.no_grad():
        while len(sample_meta) < n_samples:
            raw_data = next(dataset)
            batches_read += 1
            data = world_model.preprocess(raw_data)
            embed = world_model.encoder(data)
            post, _prior = world_model.dynamics.observe(
                embed,
                data["action"],
                data["is_first"],
            )
            actions = data["action"]
            assert_batch_major(post, actions)
            B, T, D = post["deter"].shape
            if T <= horizon:
                raise ValueError(
                    f"Batch length T={T} is too short for horizon={horizon}. "
                    "Need T > horizon so action[:, t+horizon] exists."
                )

            starts_this_batch = min(n_samples - len(sample_meta), max(1, T - horizon))
            starts = torch_module.randint(
                0,
                T - horizon,
                (starts_this_batch,),
                device=post["deter"].device,
            )
            for start in starts:
                t_start = int(start.item())
                sample = analyze_start(
                    world_model=world_model,
                    post=post,
                    actions=actions,
                    data=data,
                    t_start=t_start,
                    horizon=horizon,
                    reset_interval=reset_interval,
                    milestones=milestones,
                    alpha=alpha,
                    eps=eps,
                    torch_module=torch_module,
                )
                append_record_map(records["deter_stoch_decomposition"], sample["deter_stoch_decomposition"])
                append_reset_records(records["reset_experiment"], sample["reset_experiment"])
                append_record_map(records["directional_gain_attribution"], sample["directional_gain_attribution"])
                append_record_map(records["sensitivity_decomposition"], sample["sensitivity_decomposition"])
                append_record_map(records["decoder_attribution"], sample["decoder_attribution"])
                decoder_available_any = decoder_available_any or sample["decoder_available"]
                sample_meta.append(
                    {
                        "sample_index": len(sample_meta),
                        "batch_index": batches_read - 1,
                        "t_start": t_start,
                        "batch_size": int(B),
                        "time_length": int(T),
                        "deter_dim": int(D),
                        "stoch_flat_dim": int(flatten_stoch(post["stoch"][:, t_start]).shape[-1]),
                    }
                )
                print(
                    f"Analyzed sample {len(sample_meta)}/{n_samples}: "
                    f"batch={batches_read - 1}, t_start={t_start}, B={B}, T={T}",
                    flush=True,
                )
                if len(sample_meta) >= n_samples:
                    break

    diagnostics = {
        "deter_stoch_decomposition": aggregate_record_map(records["deter_stoch_decomposition"]),
        "reset_experiment": aggregate_reset_records(records["reset_experiment"]),
        "directional_gain_attribution": aggregate_record_map(records["directional_gain_attribution"]),
        "sensitivity_decomposition": aggregate_record_map(records["sensitivity_decomposition"]),
        "decoder_attribution": aggregate_record_map(records["decoder_attribution"]),
    }
    collection_meta = {
        "n_samples_collected": len(sample_meta),
        "batches_read": batches_read,
        "sample_starts": sample_meta,
        "decoder_available": bool(decoder_available_any),
        "shape_convention": "post['deter'] [B,T,D], post['stoch'] [B,T,...]",
    }
    return diagnostics, collection_meta


def analyze_start(
    *,
    world_model: Any,
    post: Mapping[str, Any],
    actions: Any,
    data: Mapping[str, Any],
    t_start: int,
    horizon: int,
    reset_interval: int,
    milestones: Sequence[int],
    alpha: float,
    eps: float,
    torch_module: Any,
) -> dict[str, Any]:
    dynamics = world_model.dynamics
    post_states = [state_at(post, t_start + h, detach=True) for h in range(horizon + 1)]
    prior_states = open_loop_states(
        dynamics=dynamics,
        initial_state=post_states[0],
        actions=actions,
        t_start=t_start,
        horizon=horizon,
    )
    reset_states = {
        mode: reset_rollout_states(
            dynamics=dynamics,
            post_states=post_states,
            actions=actions,
            t_start=t_start,
            horizon=horizon,
            reset_interval=reset_interval,
            mode=mode,
        )
        for mode in RESET_MODES
    }

    decoder_available = decoder_can_compare(world_model, data, post_states[0], t_start)
    return {
        "deter_stoch_decomposition": diagnostic_deter_stoch(prior_states, post_states, milestones, eps),
        "reset_experiment": diagnostic_reset_experiment(
            world_model=world_model,
            reset_states=reset_states,
            post_states=post_states,
            data=data,
            t_start=t_start,
            horizon=horizon,
        ),
        "directional_gain_attribution": diagnostic_directional_gain(
            dynamics=dynamics,
            prior_states=prior_states,
            post_states=post_states,
            actions=actions,
            t_start=t_start,
            milestones=milestones,
            alpha=alpha,
            eps=eps,
            torch_module=torch_module,
        ),
        "sensitivity_decomposition": diagnostic_sensitivity(
            dynamics=dynamics,
            prior_states=prior_states,
            post_states=post_states,
            actions=actions,
            t_start=t_start,
            horizon=horizon,
            milestones=milestones,
            eps=eps,
        ),
        "decoder_attribution": diagnostic_decoder_attribution(
            world_model=world_model,
            prior_states=prior_states,
            post_states=post_states,
            data=data,
            t_start=t_start,
            milestones=milestones,
        ),
        "decoder_available": decoder_available,
    }


def diagnostic_deter_stoch(
    prior_states: Sequence[Mapping[str, Any]],
    post_states: Sequence[Mapping[str, Any]],
    milestones: Sequence[int],
    eps: float,
) -> dict[int, dict[str, float]]:
    records = {}
    for h in milestones:
        delta_deter = prior_states[h]["deter"] - post_states[h]["deter"]
        delta_stoch = flatten_stoch(prior_states[h]["stoch"]) - flatten_stoch(post_states[h]["stoch"])
        deter_norm = norm_mean(delta_deter)
        stoch_norm = norm_mean(delta_stoch)
        records[int(h)] = {
            "delta_deter_norm": deter_norm,
            "delta_stoch_norm": stoch_norm,
            "ratio_deter_over_total": deter_norm / max(deter_norm + stoch_norm, eps),
        }
    return records


def diagnostic_reset_experiment(
    *,
    world_model: Any,
    reset_states: Mapping[str, Sequence[Mapping[str, Any]]],
    post_states: Sequence[Mapping[str, Any]],
    data: Mapping[str, Any],
    t_start: int,
    horizon: int,
) -> dict[str, dict[str, float]]:
    eval_start = min(25, horizon)
    eval_stop = min(horizon, 50)
    if eval_start > eval_stop:
        eval_start, eval_stop = (1, horizon) if horizon >= 1 else (0, 0)
    eval_indices = list(range(eval_start, eval_stop + 1))
    h15 = min(15, horizon)
    h50 = 50 if horizon >= 50 else None

    records = {}
    for mode, states in reset_states.items():
        raw_values = []
        for h in eval_indices:
            drift = states[h]["deter"] - post_states[h]["deter"]
            raw_values.append(drift.pow(2).sum(dim=-1).mean())
        raw_j = stack_scalars(raw_values).mean()
        obs15, feat15 = decoder_pair_or_feature_metric(world_model, states[h15], post_states[h15])
        obs50 = math.nan
        feat50 = math.nan
        if h50 is not None:
            obs50, feat50 = decoder_pair_or_feature_metric(world_model, states[h50], post_states[h50])
        records[mode] = {
            "raw_j": scalar(raw_j),
            "obs_mse_h15": obs15,
            "obs_mse_h50": obs50,
            "feature_dist_h15": feat15,
            "feature_dist_h50": feat50,
        }
    return records


def diagnostic_directional_gain(
    *,
    dynamics: Any,
    prior_states: Sequence[Mapping[str, Any]],
    post_states: Sequence[Mapping[str, Any]],
    actions: Any,
    t_start: int,
    milestones: Sequence[int],
    alpha: float,
    eps: float,
    torch_module: Any,
) -> dict[int, dict[str, float]]:
    records = {}
    for h in milestones:
        state = post_states[h]
        action = actions[:, t_start + h].detach()
        delta_deter = prior_states[h]["deter"] - post_states[h]["deter"]
        delta_stoch_flat = flatten_stoch(prior_states[h]["stoch"]) - flatten_stoch(post_states[h]["stoch"])

        u_deter, valid_deter = unit_vectors(delta_deter, eps)
        u_stoch_flat, valid_stoch = unit_vectors(delta_stoch_flat, eps)
        u_stoch = unflatten_stoch(u_stoch_flat, state["stoch"])
        u_random, _valid_random = unit_vectors(torch_module.randn_like(state["deter"]), eps)

        gain_random = gain_from_perturbation(
            dynamics=dynamics,
            state=state,
            action=action,
            deter_delta=alpha * u_random,
            stoch_delta=None,
            eps=eps,
        )
        gain_deter = gain_from_perturbation(
            dynamics=dynamics,
            state=state,
            action=action,
            deter_delta=alpha * u_deter,
            stoch_delta=None,
            eps=eps,
        )
        gain_stoch = gain_from_perturbation(
            dynamics=dynamics,
            state=state,
            action=action,
            deter_delta=None,
            stoch_delta=alpha * u_stoch,
            eps=eps,
        )
        records[int(h)] = {
            "gain_random": masked_mean(gain_random, None),
            "gain_deter_dir": masked_mean(gain_deter, valid_deter),
            "gain_stoch_dir": masked_mean(gain_stoch, valid_stoch),
            "deter_valid_fraction": valid_fraction(valid_deter),
            "stoch_valid_fraction": valid_fraction(valid_stoch),
        }
    return records


def diagnostic_sensitivity(
    *,
    dynamics: Any,
    prior_states: Sequence[Mapping[str, Any]],
    post_states: Sequence[Mapping[str, Any]],
    actions: Any,
    t_start: int,
    horizon: int,
    milestones: Sequence[int],
    eps: float,
) -> dict[int, dict[str, float]]:
    records = {}
    for h in milestones:
        if h >= horizon:
            records[int(h)] = {
                "sens_deter_norm": math.nan,
                "cos_deter": math.nan,
                "sens_stoch_norm": math.nan,
                "cos_stoch": math.nan,
            }
            continue
        action_h = actions[:, t_start + h].detach()
        next_from_prior = dynamics.img_step(prior_states[h], action_h, sample=False)
        next_from_post = dynamics.img_step(post_states[h], action_h, sample=False)

        sens_deter = next_from_prior["deter"] - next_from_post["deter"]
        sens_stoch = flatten_stoch(next_from_prior["stoch"]) - flatten_stoch(next_from_post["stoch"])
        delta_deter_next = prior_states[h + 1]["deter"] - post_states[h + 1]["deter"]
        delta_stoch_next = flatten_stoch(prior_states[h + 1]["stoch"]) - flatten_stoch(post_states[h + 1]["stoch"])
        records[int(h)] = {
            "sens_deter_norm": norm_mean(sens_deter),
            "cos_deter": cosine_mean(delta_deter_next, sens_deter, eps),
            "sens_stoch_norm": norm_mean(sens_stoch),
            "cos_stoch": cosine_mean(delta_stoch_next, sens_stoch, eps),
        }
    return records


def diagnostic_decoder_attribution(
    *,
    world_model: Any,
    prior_states: Sequence[Mapping[str, Any]],
    post_states: Sequence[Mapping[str, Any]],
    data: Mapping[str, Any],
    t_start: int,
    milestones: Sequence[int],
) -> dict[int, dict[str, float]]:
    records = {}
    for h in milestones:
        post_state = post_states[h]
        prior_state = prior_states[h]
        deter_reset = mixed_state(deter_source=post_state, stoch_source=prior_state)
        stoch_reset = mixed_state(deter_source=prior_state, stoch_source=post_state)
        t_abs = t_start + h

        obs_prior, feat_prior = decoder_or_feature_metric(world_model, prior_state, post_state, data, t_abs)
        obs_deter_reset, feat_deter_reset = decoder_or_feature_metric(world_model, deter_reset, post_state, data, t_abs)
        obs_stoch_reset, feat_stoch_reset = decoder_or_feature_metric(world_model, stoch_reset, post_state, data, t_abs)
        obs_post, feat_post = decoder_or_feature_metric(world_model, post_state, post_state, data, t_abs)

        records[int(h)] = {
            "obs_full_prior": obs_prior,
            "obs_deter_reset": obs_deter_reset,
            "obs_stoch_reset": obs_stoch_reset,
            "obs_post": obs_post,
            "feature_full_prior_dist": feat_prior,
            "feature_deter_reset_dist": feat_deter_reset,
            "feature_stoch_reset_dist": feat_stoch_reset,
            "feature_post_norm": feat_post,
        }
    return records


def open_loop_states(
    *,
    dynamics: Any,
    initial_state: Mapping[str, Any],
    actions: Any,
    t_start: int,
    horizon: int,
) -> list[dict[str, Any]]:
    state = clone_state(initial_state)
    states = [clone_state(state)]
    for h in range(horizon):
        action_h = actions[:, t_start + h].detach()
        state = detach_state(dynamics.img_step(state, action_h, sample=False))
        states.append(state)
    return states


def reset_rollout_states(
    *,
    dynamics: Any,
    post_states: Sequence[Mapping[str, Any]],
    actions: Any,
    t_start: int,
    horizon: int,
    reset_interval: int,
    mode: str,
) -> list[dict[str, Any]]:
    state = clone_state(post_states[0])
    states = []
    for h in range(horizon + 1):
        if h > 0 and h % reset_interval == 0:
            if mode in {"reset_deter", "reset_both"}:
                state["deter"] = post_states[h]["deter"].detach()
            if mode in {"reset_stoch", "reset_both"}:
                state["stoch"] = post_states[h]["stoch"].detach()
        states.append(clone_state(state))
        if h < horizon:
            action_h = actions[:, t_start + h].detach()
            state = detach_state(dynamics.img_step(state, action_h, sample=False))
    return states


def gain_from_perturbation(
    *,
    dynamics: Any,
    state: Mapping[str, Any],
    action: Any,
    deter_delta: Any | None,
    stoch_delta: Any | None,
    eps: float,
) -> Any:
    base_state = detach_state(state)
    pert_state = clone_state(base_state)
    if deter_delta is not None:
        pert_state["deter"] = base_state["deter"] + deter_delta.detach()
        denom = deter_delta.detach().norm(dim=-1)
    elif stoch_delta is not None:
        pert_state["stoch"] = base_state["stoch"] + stoch_delta.detach()
        denom = flatten_stoch(stoch_delta.detach()).norm(dim=-1)
    else:
        raise ValueError("Either deter_delta or stoch_delta is required.")
    base_next = dynamics.img_step(base_state, action.detach(), sample=False)
    pert_next = dynamics.img_step(pert_state, action.detach(), sample=False)
    return (pert_next["deter"] - base_next["deter"]).norm(dim=-1) / denom.clamp_min(eps)


def decoder_or_feature_metric(
    world_model: Any,
    state: Mapping[str, Any],
    post_state: Mapping[str, Any],
    data: Mapping[str, Any],
    t_abs: int,
) -> tuple[float, float]:
    feat = world_model.dynamics.get_feat(state)
    post_feat = world_model.dynamics.get_feat(post_state)
    feature_dist = scalar((feat - post_feat).pow(2).sum(dim=-1).mean())
    obs_mse = decode_obs_mse(world_model, state, data, t_abs)
    if math.isnan(obs_mse) and state is post_state:
        feature_dist = scalar(feat.norm(dim=-1).mean())
    return obs_mse, feature_dist


def decoder_pair_or_feature_metric(
    world_model: Any,
    state: Mapping[str, Any],
    post_state: Mapping[str, Any],
) -> tuple[float, float]:
    feat = world_model.dynamics.get_feat(state)
    post_feat = world_model.dynamics.get_feat(post_state)
    feature_dist = scalar((feat - post_feat).pow(2).sum(dim=-1).mean())
    obs_mse = decode_pair_mse(world_model, state, post_state)
    return obs_mse, feature_dist


def decode_obs_mse(world_model: Any, state: Mapping[str, Any], data: Mapping[str, Any], t_abs: int) -> float:
    try:
        decoded = decode_flat(world_model, state)
        if decoded is None:
            return math.nan
        pred_parts, target_parts = [], []
        for name, mode in decoded.items():
            if name not in data:
                continue
            target = data[name][:, t_abs]
            if mode.shape[0] != target.shape[0]:
                continue
            pred_parts.append(mode.reshape(mode.shape[0], -1))
            target_parts.append(target.reshape(target.shape[0], -1))
        if not pred_parts:
            return math.nan
        pred_flat = __import__("torch").cat(pred_parts, dim=-1)
        target_flat = __import__("torch").cat(target_parts, dim=-1)
        if pred_flat.shape != target_flat.shape:
            return math.nan
        return scalar((pred_flat - target_flat).pow(2).mean())
    except Exception:
        return math.nan


def decode_pair_mse(world_model: Any, state: Mapping[str, Any], post_state: Mapping[str, Any]) -> float:
    try:
        decoded = decode_flat(world_model, state)
        decoded_post = decode_flat(world_model, post_state)
        if decoded is None or decoded_post is None:
            return math.nan
        pred_parts, target_parts = [], []
        for name, mode in decoded.items():
            if name not in decoded_post:
                continue
            target = decoded_post[name]
            if mode.shape != target.shape:
                continue
            pred_parts.append(mode.reshape(mode.shape[0], -1))
            target_parts.append(target.reshape(target.shape[0], -1))
        if not pred_parts:
            return math.nan
        pred_flat = __import__("torch").cat(pred_parts, dim=-1)
        target_flat = __import__("torch").cat(target_parts, dim=-1)
        return scalar((pred_flat - target_flat).pow(2).mean())
    except Exception:
        return math.nan


def decode_flat(world_model: Any, state: Mapping[str, Any]) -> dict[str, Any] | None:
    decoder = world_model.heads["decoder"]
    feat = world_model.dynamics.get_feat(state)
    decoded = decoder(feat)
    if not isinstance(decoded, Mapping):
        return None
    modes = {}
    for name, pred in decoded.items():
        modes[name] = pred.mode() if hasattr(pred, "mode") else pred
    return modes


def decoder_can_compare(world_model: Any, data: Mapping[str, Any], state: Mapping[str, Any], t_abs: int) -> bool:
    return math.isfinite(decode_obs_mse(world_model, state, data, t_abs))


def state_at(post: Mapping[str, Any], t: int, *, detach: bool) -> dict[str, Any]:
    state = {key: value[:, t] for key, value in post.items()}
    return detach_state(state) if detach else state


def detach_state(state: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value.detach() for key, value in state.items()}


def clone_state(state: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value.detach().clone() for key, value in state.items()}


def mixed_state(*, deter_source: Mapping[str, Any], stoch_source: Mapping[str, Any]) -> dict[str, Any]:
    state = clone_state(deter_source)
    state["stoch"] = stoch_source["stoch"].detach()
    state["deter"] = deter_source["deter"].detach()
    return state


def flatten_stoch(stoch: Any) -> Any:
    return stoch.reshape(stoch.shape[0], -1)


def unflatten_stoch(flat: Any, template: Any) -> Any:
    return flat.reshape_as(template)


def unit_vectors(x: Any, eps: float) -> tuple[Any, Any]:
    norm = x.norm(dim=-1, keepdim=True)
    valid = norm.squeeze(-1) > eps
    return x / norm.clamp_min(eps), valid


def norm_mean(x: Any) -> float:
    return scalar(x.norm(dim=-1).mean())


def cosine_mean(x: Any, y: Any, eps: float) -> float:
    x_norm = x.norm(dim=-1)
    y_norm = y.norm(dim=-1)
    valid = (x_norm > eps) & (y_norm > eps)
    if not bool(valid.any().detach().cpu().item()):
        return math.nan
    cos = (x * y).sum(dim=-1) / (x_norm * y_norm).clamp_min(eps)
    return scalar(cos[valid].mean())


def masked_mean(values: Any, mask: Any | None) -> float:
    if mask is None:
        return scalar(values.mean())
    if not bool(mask.any().detach().cpu().item()):
        return math.nan
    return scalar(values[mask].mean())


def valid_fraction(mask: Any) -> float:
    return scalar(mask.float().mean())


def stack_scalars(values: Sequence[Any]) -> Any:
    if not values:
        raise ValueError("Cannot stack an empty scalar sequence.")
    out = values[0].new_zeros((len(values),))
    for idx, value in enumerate(values):
        out[idx] = value
    return out


def assert_batch_major(post: Mapping[str, Any], actions: Any) -> None:
    post_deter = post["deter"]
    assert post_deter.ndim == 3, f"Expected post['deter'] [B,T,D], got {post_deter.shape}"
    assert actions.ndim == 3, f"Expected actions [B,T,A], got {actions.shape}"
    # post["deter"] shape: [B, T, D] where B=batch, T=time, D=latent_dim
    B, T, _D = post_deter.shape
    assert B == actions.shape[0], f"Batch mismatch: post {B} vs action {actions.shape[0]}"
    assert T == actions.shape[1], f"Time mismatch: post {T} vs action {actions.shape[1]}"


def init_record_map(milestones: Sequence[int]) -> dict[int, list[dict[str, float]]]:
    return {int(h): [] for h in milestones}


def append_record_map(
    accum: dict[int, list[dict[str, float]]],
    sample: Mapping[int, dict[str, float]],
) -> None:
    for horizon, record in sample.items():
        accum[int(horizon)].append(record)


def append_reset_records(
    accum: dict[str, list[dict[str, float]]],
    sample: Mapping[str, dict[str, float]],
) -> None:
    for mode, record in sample.items():
        accum[mode].append(record)


def aggregate_record_map(records: Mapping[int, Sequence[Mapping[str, float]]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for horizon, horizon_records in records.items():
        result[str(horizon)] = aggregate_records(horizon_records)
    return result


def aggregate_reset_records(records: Mapping[str, Sequence[Mapping[str, float]]]) -> dict[str, Any]:
    return {mode: aggregate_records(mode_records) for mode, mode_records in records.items()}


def aggregate_records(records: Sequence[Mapping[str, float]]) -> dict[str, Any]:
    keys = sorted({key for record in records for key in record})
    aggregated: dict[str, Any] = {"n": len(records)}
    for key in keys:
        values = np.asarray([record.get(key, math.nan) for record in records], dtype=np.float64)
        finite = np.isfinite(values)
        if finite.any():
            aggregated[key] = {
                "mean": float(values[finite].mean()),
                "std": float(values[finite].std(ddof=0)),
                "min": float(values[finite].min()),
                "max": float(values[finite].max()),
                "n_finite": int(finite.sum()),
            }
        else:
            aggregated[key] = {
                "mean": math.nan,
                "std": math.nan,
                "min": math.nan,
                "max": math.nan,
                "n_finite": 0,
            }
    return aggregated


def load_run_config(
    *,
    dreamer_module: Any,
    config_path: Path,
    checkpoint: Path,
    replay_dir: Path,
    device: str,
) -> SimpleNamespace:
    loader = dreamer_module.yaml.YAML(typ="safe", pure=True)
    run_config = loader.load(config_path.read_text())
    dreamer_configs = loader.load((DREAMERV3_ROOT / "configs.yaml").read_text())

    legacy_dreamerv3 = run_config.get("dreamerv3", {}) or {}
    config_names = run_config.get("dreamer_configs")
    if config_names is None and legacy_dreamerv3:
        config_names = legacy_dreamerv3.get("configs")
    if config_names is None:
        config_names = ["dmc_proprio"]
    if isinstance(config_names, str):
        config_names = [part.strip() for part in config_names.split(",") if part.strip()]

    values: dict[str, Any] = {}
    for name in ["defaults", *list(config_names)]:
        recursive_update(values, dreamer_configs[name])
    if run_config.get("dreamer"):
        recursive_update(values, run_config["dreamer"])
    if legacy_dreamerv3:
        recursive_update(
            values,
            {
                key: value
                for key, value in legacy_dreamerv3.items()
                if key not in {"configs", "task", "steps", "seed"}
            },
        )
    recursive_update(
        values,
        {
            key: run_config[key]
            for key in (
                "batch_size",
                "batch_length",
                "envs",
                "compile",
                "eval_every",
                "log_every",
                "dataset_size",
                "action_repeat",
                "time_limit",
            )
            if key in run_config
        },
    )

    task = run_config.get("task", legacy_dreamerv3.get("task"))
    if task is None:
        raise KeyError("Config must define task or dreamerv3.task.")
    if not task.startswith("dmc_"):
        task = "dmc_" + task

    values["task"] = task
    values["seed"] = int(run_config.get("seed", legacy_dreamerv3.get("seed", 0)))
    values["device"] = device
    values["compile"] = False
    values["logdir"] = str(checkpoint.parent)
    values["traindir"] = replay_dir
    values["evaldir"] = checkpoint.parent / "eval_eps"
    return SimpleNamespace(**values)


def build_agent(dreamer_module: Any, torch_module: Any, config: SimpleNamespace, checkpoint: Path):
    env = dreamer_module.make_env(config, "eval", 0)
    config.num_actions = (
        env.action_space.n if hasattr(env.action_space, "n") else env.action_space.shape[0]
    )
    agent = dreamer_module.Dreamer(
        env.observation_space,
        env.action_space,
        config,
        logger=NullLogger(),
        dataset=None,
    ).to(config.device)

    ckpt = load_checkpoint(torch_module, checkpoint, config.device)
    state_dict = ckpt.get("agent_state_dict", ckpt)
    agent.load_state_dict(normalize_checkpoint_state_dict(state_dict))
    agent.requires_grad_(requires_grad=False)
    agent.eval()
    return agent, env


def print_results(results: Mapping[str, Any]) -> None:
    meta = results["metadata"]
    print("\nRSSM drift source sample set")
    print_table(
        ["field", "value"],
        [
            ["samples", str(meta["n_samples_collected"])],
            ["batches_read", str(meta["batches_read"])],
            ["horizon", str(meta["horizon"])],
            ["reset_interval", str(meta["reset_interval"])],
            ["alpha", fmt(meta["alpha"])],
            ["decoder_available", str(meta["decoder_available"])],
            ["shape", meta["shape_convention"]],
        ],
    )

    print("\nDiagnostic 1: Deter vs Stoch Drift Decomposition")
    print_table(
        ["h", "||delta_deter||", "||delta_stoch||", "deter/(deter+stoch)"],
        horizon_rows(
            results["deter_stoch_decomposition"],
            ["delta_deter_norm", "delta_stoch_norm", "ratio_deter_over_total"],
        ),
    )

    print("\nDiagnostic 2: Reset Experiment")
    reset_section = results["reset_experiment"]
    print_table(
        ["mode", "RawJ", "Obs@15", "Obs@50", "Feat@15", "Feat@50"],
        [
            [
                mode,
                fmt(reset_section[mode]["raw_j"]["mean"]),
                fmt(reset_section[mode]["obs_mse_h15"]["mean"]),
                fmt(reset_section[mode]["obs_mse_h50"]["mean"]),
                fmt(reset_section[mode]["feature_dist_h15"]["mean"]),
                fmt(reset_section[mode]["feature_dist_h50"]["mean"]),
            ]
            for mode in RESET_MODES
        ],
    )

    print("\nDiagnostic 3: Directional Gain Attribution")
    print_table(
        ["h", "gain_random", "gain_deter_dir", "gain_stoch_dir"],
        horizon_rows(
            results["directional_gain_attribution"],
            ["gain_random", "gain_deter_dir", "gain_stoch_dir"],
        ),
    )

    print("\nDiagnostic 4: Sensitivity Decomposition")
    print_table(
        ["h", "||sens_deter||", "cos_deter", "||sens_stoch||", "cos_stoch"],
        horizon_rows(
            results["sensitivity_decomposition"],
            ["sens_deter_norm", "cos_deter", "sens_stoch_norm", "cos_stoch"],
        ),
    )

    decoder_available = bool(meta["decoder_available"])
    print("\nDiagnostic 5: Decoder Attribution")
    if decoder_available:
        print_table(
            ["h", "obs_full_prior", "obs_deter_reset", "obs_stoch_reset", "obs_post"],
            horizon_rows(
                results["decoder_attribution"],
                ["obs_full_prior", "obs_deter_reset", "obs_stoch_reset", "obs_post"],
            ),
        )
    else:
        print_table(
            ["h", "feat_prior_dist", "feat_deter_reset", "feat_stoch_reset", "feat_post_norm"],
            horizon_rows(
                results["decoder_attribution"],
                [
                    "feature_full_prior_dist",
                    "feature_deter_reset_dist",
                    "feature_stoch_reset_dist",
                    "feature_post_norm",
                ],
            ),
        )


def horizon_rows(section: Mapping[str, Any], keys: Sequence[str]) -> list[list[str]]:
    rows = []
    for horizon in sorted(section, key=lambda value: int(value)):
        row = [horizon]
        stats = section[horizon]
        for key in keys:
            row.append(fmt(stats[key]["mean"]))
        rows.append(row)
    return rows


def print_header(
    checkpoint: Path,
    config_path: Path,
    replay_dir: Path,
    output_path: Path,
    config: SimpleNamespace,
    args: argparse.Namespace,
    milestones: Sequence[int],
) -> None:
    print("=" * 78)
    print("DreamerV3 RSSM Drift Source Diagnostics")
    print("=" * 78)
    print(f"Task: {config.task}, seed: {config.seed}, device: {config.device}")
    print(f"Checkpoint: {display_path(checkpoint)}")
    print(f"Config: {display_path(config_path)}")
    print(f"Replay: {display_path(replay_dir)}")
    print(
        f"Horizon: {args.horizon}, reset_interval: {args.reset_interval}, "
        f"samples: {args.n_samples}, alpha: {args.alpha}"
    )
    print(f"Milestones: {list(milestones)}")
    print(f"Output: {display_path(output_path)}")
    print("=" * 78, flush=True)


def print_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> None:
    rows = [[str(cell) for cell in row] for row in rows]
    widths = [
        max(len(str(header)), *(len(row[i]) for row in rows)) if rows else len(str(header))
        for i, header in enumerate(headers)
    ]
    fmt_row = "  ".join(f"{{:<{width}}}" for width in widths)
    print(fmt_row.format(*headers))
    print(fmt_row.format(*["-" * width for width in widths]))
    for row in rows:
        print(fmt_row.format(*row))


def normalize_milestones(milestones: Sequence[int], horizon: int) -> tuple[int, ...]:
    kept = sorted({int(h) for h in milestones if 0 <= int(h) <= horizon})
    if not kept:
        raise ValueError("At least one milestone must fall within [0, horizon].")
    return tuple(kept)


def recursive_update(base: dict[str, Any], update: Mapping[str, Any]) -> None:
    for key, value in update.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), dict):
            recursive_update(base[key], value)
        else:
            base[key] = copy.deepcopy(value)


def normalize_checkpoint_state_dict(state_dict: Mapping[str, Any]) -> dict[str, Any]:
    return {key.replace("._orig_mod.", "."): value for key, value in state_dict.items()}


def load_checkpoint(torch_module: Any, checkpoint: Path, device: str) -> Mapping[str, Any]:
    try:
        return torch_module.load(checkpoint, map_location=device, weights_only=False)
    except TypeError:
        return torch_module.load(checkpoint, map_location=device)


def scalar(x: Any) -> float:
    if hasattr(x, "detach"):
        return float(x.detach().cpu().item())
    return float(x)


def fmt(value: Any) -> str:
    value = float(value)
    if math.isnan(value):
        return "nan"
    if value == 0.0:
        return "0"
    abs_value = abs(value)
    if abs_value < 1e-3 or abs_value >= 1e4:
        return f"{value:.4e}"
    return f"{value:.6f}"


def json_sanitize(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): json_sanitize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_sanitize(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, np.generic):
        return json_sanitize(value.item())
    return value


def resolve_path(path: str | Path) -> Path:
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = PROJECT_ROOT / resolved
    return resolved


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose RSSM source of open-loop drift.")
    parser.add_argument("--checkpoint", required=True, help="Path to DreamerV3 latest.pt.")
    parser.add_argument("--config", required=True, help="Training config for model architecture.")
    parser.add_argument("--horizon", type=int, default=50, help="Open-loop horizon H.")
    parser.add_argument("--reset_interval", type=int, default=10, help="Component reset interval K.")
    parser.add_argument("--n_samples", type=int, default=10, help="Number of sampled start times.")
    parser.add_argument("--alpha", type=float, default=0.01, help="Perturbation magnitude.")
    parser.add_argument("--device", default=None, help="Torch device, e.g. cuda:0 or cpu.")
    parser.add_argument(
        "--replay_dir",
        default=None,
        help="Replay directory. Defaults to CHECKPOINT_PARENT/train_eps.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="JSON output path.",
    )
    parser.add_argument(
        "--dataset_limit",
        type=int,
        default=None,
        help="Optional replay step limit passed to tools.load_episodes.",
    )
    parser.add_argument(
        "--milestones",
        type=int,
        nargs="+",
        default=[0, 1, 2, 4, 8, 15, 30, 50],
        help="Horizons to report; values outside [0,H] are ignored.",
    )
    parser.add_argument("--eps", type=float, default=1e-8, help="Near-zero norm threshold.")
    parser.add_argument("--seed", type=int, default=0, help="Diagnostic random seed.")
    args = parser.parse_args()
    if args.horizon <= 0:
        parser.error("--horizon must be positive.")
    if args.reset_interval <= 0:
        parser.error("--reset_interval must be positive.")
    if args.n_samples <= 0:
        parser.error("--n_samples must be positive.")
    if args.alpha <= 0:
        parser.error("--alpha must be positive.")
    if args.eps <= 0:
        parser.error("--eps must be positive.")
    return args


RESET_MODES = ("no_reset", "reset_deter", "reset_stoch", "reset_both")


if __name__ == "__main__":
    main()
