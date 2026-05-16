#!/usr/bin/env python3
"""Diagnose DreamerV3 residual dynamics in the open-loop prior regime.

This script compares teacher-forced posterior transitions against open-loop
prior rollouts. It keeps the corrected DreamerV3 tensor convention throughout:

    post["deter"] shape is [B, T, D]
    state_t = {k: v[:, t].detach() for k, v in post.items()}
    action_t = actions[:, t].detach()

Rollouts and decomposition terms are computed under torch.no_grad(). One-step
gain probes are isolated helpers and do not backpropagate or update weights.
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
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "tables" / "prior_regime_diagnostics.json"

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
        diagnostics, collection_meta = collect_prior_regime_diagnostics(
            dreamer_module=dreamer,
            torch_module=torch,
            agent=agent,
            config=config,
            replay_dir=replay_dir,
            horizon=args.horizon,
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
    print(f"\nSaved prior-regime diagnostics to {display_path(output_path)}")


def collect_prior_regime_diagnostics(
    *,
    dreamer_module: Any,
    torch_module: Any,
    agent: Any,
    config: SimpleNamespace,
    replay_dir: Path,
    horizon: int,
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
        "drift_decomposition": init_record_map(milestones),
        "error_aligned_gain": init_record_map(milestones),
        "transported_direction_gain": init_record_map(milestones),
    }
    sample_meta: list[dict[str, int]] = []
    batches_read = 0

    while len(sample_meta) < n_samples:
        raw_data = next(dataset)
        batches_read += 1
        with torch_module.no_grad():
            data = agent._wm.preprocess(raw_data)
            embed = agent._wm.encoder(data)
            post, _prior = agent._wm.dynamics.observe(
                embed,
                data["action"],
                data["is_first"],
            )
        actions = data["action"]
        post_deter = post["deter"]
        assert post_deter.ndim == 3, f"Expected post['deter'] [B,T,D], got {post_deter.shape}"
        assert actions.ndim == 3, f"Expected actions [B,T,A], got {actions.shape}"

        # post["deter"] shape: [B, T, D] where B=batch, T=time, D=latent_dim
        B, T, D = post_deter.shape
        assert B == actions.shape[0], f"Batch mismatch: post {B} vs action {actions.shape[0]}"
        assert T == actions.shape[1], f"Time mismatch: post {T} vs action {actions.shape[1]}"
        if T <= horizon:
            raise ValueError(
                f"Batch length T={T} is too short for horizon={horizon}. "
                "Need T > horizon so action[:, t+horizon] exists for gain probes."
            )

        starts_this_batch = min(n_samples - len(sample_meta), max(1, T - horizon))
        starts = torch_module.randint(
            0,
            T - horizon,
            (starts_this_batch,),
            device=post_deter.device,
        )
        for start in starts:
            t_start = int(start.item())
            sample = analyze_start(
                dynamics=agent._wm.dynamics,
                post=post,
                actions=actions,
                t_start=t_start,
                horizon=horizon,
                milestones=milestones,
                alpha=alpha,
                eps=eps,
                torch_module=torch_module,
            )
            append_records(records, sample)
            sample_meta.append(
                {
                    "sample_index": len(sample_meta),
                    "batch_index": batches_read - 1,
                    "t_start": t_start,
                    "batch_size": int(B),
                    "time_length": int(T),
                    "latent_dim": int(D),
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
        name: aggregate_record_map(record_map)
        for name, record_map in records.items()
    }
    collection_meta = {
        "n_samples_collected": len(sample_meta),
        "batches_read": batches_read,
        "sample_starts": sample_meta,
        "shape_convention": "post['deter'] [B,T,D], sampled states [B,D]",
    }
    return diagnostics, collection_meta


def analyze_start(
    *,
    dynamics: Any,
    post: Mapping[str, Any],
    actions: Any,
    t_start: int,
    horizon: int,
    milestones: Sequence[int],
    alpha: float,
    eps: float,
    torch_module: Any,
) -> dict[str, dict[int, dict[str, float]]]:
    with torch_module.no_grad():
        prior_states = open_loop_prior_states(
            dynamics=dynamics,
            post=post,
            actions=actions,
            t_start=t_start,
            horizon=horizon,
        )
        post_states = [
            state_at(post, t_start + h, detach=True)
            for h in range(horizon + 1)
        ]
        decomp = diagnostic_drift_decomposition(
            dynamics=dynamics,
            post_states=post_states,
            prior_states=prior_states,
            actions=actions,
            t_start=t_start,
            horizon=horizon,
            milestones=milestones,
            eps=eps,
            torch_module=torch_module,
        )
        transport_states = transported_rollout_states(
            dynamics=dynamics,
            post=post,
            actions=actions,
            t_start=t_start,
            horizon=horizon,
            alpha=alpha,
            eps=eps,
            torch_module=torch_module,
        )

    error_gain = diagnostic_error_aligned_gain(
        dynamics=dynamics,
        post_states=post_states,
        prior_states=prior_states,
        actions=actions,
        t_start=t_start,
        milestones=milestones,
        alpha=alpha,
        eps=eps,
        torch_module=torch_module,
    )
    transport_gain = diagnostic_transported_direction_gain(
        dynamics=dynamics,
        transport_states=transport_states,
        actions=actions,
        t_start=t_start,
        milestones=milestones,
        alpha=alpha,
        eps=eps,
        torch_module=torch_module,
    )
    return {
        "drift_decomposition": decomp,
        "error_aligned_gain": error_gain,
        "transported_direction_gain": transport_gain,
    }


def open_loop_prior_states(
    *,
    dynamics: Any,
    post: Mapping[str, Any],
    actions: Any,
    t_start: int,
    horizon: int,
) -> list[dict[str, Any]]:
    state = state_at(post, t_start, detach=True)
    states = [clone_state(state)]
    for h in range(horizon):
        action_h = actions[:, t_start + h].detach()
        state = dynamics.img_step(state, action_h, sample=False)
        state = detach_state(state)
        states.append(state)
    return states


def diagnostic_drift_decomposition(
    *,
    dynamics: Any,
    post_states: Sequence[Mapping[str, Any]],
    prior_states: Sequence[Mapping[str, Any]],
    actions: Any,
    t_start: int,
    horizon: int,
    milestones: Sequence[int],
    eps: float,
    torch_module: Any,
) -> dict[int, dict[str, float]]:
    del torch_module
    deltas = [prior_states[h]["deter"] - post_states[h]["deter"] for h in range(horizon + 1)]
    teacher_terms = []
    sensitivity_terms = []
    identity_errors = []
    cosines = []
    for h in range(horizon):
        action_h = actions[:, t_start + h].detach()
        post_next_pred = dynamics.img_step(post_states[h], action_h, sample=False)
        prior_next_pred = dynamics.img_step(prior_states[h], action_h, sample=False)
        teacher = post_next_pred["deter"] - post_states[h + 1]["deter"]
        sensitivity = prior_next_pred["deter"] - post_next_pred["deter"]
        identity_error = deltas[h + 1] - (teacher + sensitivity)
        teacher_terms.append(teacher)
        sensitivity_terms.append(sensitivity)
        identity_errors.append(identity_error)
        cosines.append(cosine_mean(deltas[h + 1], sensitivity, eps))

    teacher_norms = stack_norms(teacher_terms)
    sensitivity_norms = stack_norms(sensitivity_terms)
    records: dict[int, dict[str, float]] = {}
    for h in milestones:
        h = int(h)
        transition_valid = h < horizon
        transition_count = min(h, horizon)
        records[h] = {
            "delta_norm": norm_mean(deltas[h]),
            "teacher_norm": norm_mean(teacher_terms[h]) if transition_valid else math.nan,
            "sensitivity_norm": norm_mean(sensitivity_terms[h]) if transition_valid else math.nan,
            "cos_delta_next_sensitivity": cosines[h] if transition_valid else math.nan,
            "decomposition_error_norm": norm_mean(identity_errors[h]) if transition_valid else math.nan,
            "cum_teacher_norm_to_h": cumulative_norm_mean(teacher_norms, transition_count),
            "cum_sensitivity_norm_to_h": cumulative_norm_mean(sensitivity_norms, transition_count),
        }
    return records


def diagnostic_error_aligned_gain(
    *,
    dynamics: Any,
    post_states: Sequence[Mapping[str, Any]],
    prior_states: Sequence[Mapping[str, Any]],
    actions: Any,
    t_start: int,
    milestones: Sequence[int],
    alpha: float,
    eps: float,
    torch_module: Any,
) -> dict[int, dict[str, float]]:
    records: dict[int, dict[str, float]] = {}
    for h in milestones:
        h = int(h)
        action_h = actions[:, t_start + h].detach()
        delta_h = prior_states[h]["deter"] - post_states[h]["deter"]
        direction, valid = unit_vectors(delta_h, eps)

        random_direction, _ = unit_vectors(torch_module.randn_like(direction), eps)
        random_gain = one_step_gain(
            dynamics=dynamics,
            state=post_states[h],
            action=action_h,
            direction=random_direction,
            alpha=alpha,
            eps=eps,
            torch_module=torch_module,
        )
        error_gain_post = one_step_gain(
            dynamics=dynamics,
            state=post_states[h],
            action=action_h,
            direction=direction,
            alpha=alpha,
            eps=eps,
            torch_module=torch_module,
        )
        error_gain_prior = one_step_gain(
            dynamics=dynamics,
            state=prior_states[h],
            action=action_h,
            direction=direction,
            alpha=alpha,
            eps=eps,
            torch_module=torch_module,
        )
        records[h] = {
            "random_gain": masked_mean(random_gain, None),
            "error_aligned_gain_post": masked_mean(error_gain_post, valid),
            "error_aligned_gain_prior": masked_mean(error_gain_prior, valid),
            "drift_direction_valid_fraction": valid_fraction(valid),
            "delta_norm": norm_mean(delta_h),
        }
    return records


def transported_rollout_states(
    *,
    dynamics: Any,
    post: Mapping[str, Any],
    actions: Any,
    t_start: int,
    horizon: int,
    alpha: float,
    eps: float,
    torch_module: Any,
) -> dict[str, list[dict[str, Any]]]:
    base_state = state_at(post, t_start, detach=True)
    direction, _valid = unit_vectors(torch_module.randn_like(base_state["deter"]), eps)
    pert_state = clone_state(base_state)
    pert_state["deter"] = base_state["deter"] + alpha * direction

    base_states = [clone_state(base_state)]
    pert_states = [clone_state(pert_state)]
    for h in range(horizon):
        action_h = actions[:, t_start + h].detach()
        base_state = detach_state(dynamics.img_step(base_state, action_h, sample=False))
        pert_state = detach_state(dynamics.img_step(pert_state, action_h, sample=False))
        base_states.append(base_state)
        pert_states.append(pert_state)
    return {"base": base_states, "perturbed": pert_states}


def diagnostic_transported_direction_gain(
    *,
    dynamics: Any,
    transport_states: Mapping[str, Sequence[Mapping[str, Any]]],
    actions: Any,
    t_start: int,
    milestones: Sequence[int],
    alpha: float,
    eps: float,
    torch_module: Any,
) -> dict[int, dict[str, float]]:
    records: dict[int, dict[str, float]] = {}
    base_states = transport_states["base"]
    pert_states = transport_states["perturbed"]
    for h in milestones:
        h = int(h)
        action_h = actions[:, t_start + h].detach()
        transported_delta = pert_states[h]["deter"] - base_states[h]["deter"]
        transported_direction, valid = unit_vectors(transported_delta, eps)
        random_direction, _ = unit_vectors(torch_module.randn_like(transported_direction), eps)

        transport_gain = one_step_gain(
            dynamics=dynamics,
            state=base_states[h],
            action=action_h,
            direction=transported_direction,
            alpha=alpha,
            eps=eps,
            torch_module=torch_module,
        )
        random_gain = one_step_gain(
            dynamics=dynamics,
            state=base_states[h],
            action=action_h,
            direction=random_direction,
            alpha=alpha,
            eps=eps,
            torch_module=torch_module,
        )
        records[h] = {
            "gain_transported": masked_mean(transport_gain, valid),
            "gain_random": masked_mean(random_gain, None),
            "transported_valid_fraction": valid_fraction(valid),
            "transported_delta_norm": norm_mean(transported_delta),
        }
    return records


def one_step_gain(
    *,
    dynamics: Any,
    state: Mapping[str, Any],
    action: Any,
    direction: Any,
    alpha: float,
    eps: float,
    torch_module: Any,
) -> Any:
    base_state = detach_state(state)
    pert_state = clone_state(base_state)
    perturb = alpha * direction.detach()
    pert_state["deter"] = base_state["deter"] + perturb

    # Gain probes are one-step computations. They are not open-loop rollout
    # chains and do not call backward; enable_grad keeps this block distinct
    # from the no_grad rollout/decomposition sections.
    with torch_module.enable_grad():
        base_next = dynamics.img_step(base_state, action.detach(), sample=False)
        pert_next = dynamics.img_step(pert_state, action.detach(), sample=False)
        numerator = (pert_next["deter"] - base_next["deter"]).norm(dim=-1)
        denominator = perturb.norm(dim=-1).clamp_min(eps)
        gain = numerator / denominator
    return gain.detach()


def state_at(post: Mapping[str, Any], t: int, *, detach: bool) -> dict[str, Any]:
    state = {key: value[:, t] for key, value in post.items()}
    return detach_state(state) if detach else state


def detach_state(state: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value.detach() for key, value in state.items()}


def clone_state(state: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value.detach().clone() for key, value in state.items()}


def unit_vectors(x: Any, eps: float) -> tuple[Any, Any]:
    norm = x.norm(dim=-1, keepdim=True)
    valid = norm.squeeze(-1) > eps
    return x / norm.clamp_min(eps), valid


def norm_mean(x: Any) -> float:
    return scalar(x.norm(dim=-1).mean())


def stack_norms(xs: Sequence[Any]) -> Any:
    if not xs:
        raise ValueError("Cannot stack empty norm sequence.")
    out = xs[0].new_zeros((len(xs), xs[0].shape[0]))
    for idx, x in enumerate(xs):
        out[idx] = x.norm(dim=-1)
    return out


def cumulative_norm_mean(norms: Any, count: int) -> float:
    if count <= 0:
        return 0.0
    return scalar(norms[:count].sum(dim=0).mean())


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


def init_record_map(milestones: Sequence[int]) -> dict[int, list[dict[str, float]]]:
    return {int(h): [] for h in milestones}


def append_records(
    accum: Mapping[str, dict[int, list[dict[str, float]]]],
    sample: Mapping[str, Mapping[int, dict[str, float]]],
) -> None:
    for name, by_horizon in sample.items():
        for horizon, record in by_horizon.items():
            accum[name][int(horizon)].append(record)


def aggregate_record_map(records: Mapping[int, Sequence[Mapping[str, float]]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for horizon, horizon_records in records.items():
        keys = sorted({key for record in horizon_records for key in record})
        aggregated: dict[str, Any] = {"n": len(horizon_records)}
        for key in keys:
            values = np.asarray([record.get(key, math.nan) for record in horizon_records], dtype=np.float64)
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
        result[str(horizon)] = aggregated
    return result


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
    print("\nPrior-regime sample set")
    print_table(
        ["field", "value"],
        [
            ["samples", str(meta["n_samples_collected"])],
            ["batches_read", str(meta["batches_read"])],
            ["horizon", str(meta["horizon"])],
            ["alpha", fmt(meta["alpha"])],
            ["shape", meta["shape_convention"]],
        ],
    )

    print("\nDiagnostic 1: Drift Decomposition")
    print_table(
        [
            "h",
            "||delta_h||",
            "||teacher_h||",
            "||sensitivity_h||",
            "cos(delta_{h+1}, sens_h)",
            "cum ||teacher||",
            "cum ||sensitivity||",
            "identity err",
        ],
        horizon_rows(
            results["drift_decomposition"],
            [
                "delta_norm",
                "teacher_norm",
                "sensitivity_norm",
                "cos_delta_next_sensitivity",
                "cum_teacher_norm_to_h",
                "cum_sensitivity_norm_to_h",
                "decomposition_error_norm",
            ],
        ),
    )

    print("\nDiagnostic 2: Error-Aligned Gain")
    print_table(
        [
            "h",
            "random_gain",
            "gain_error_post",
            "gain_error_prior",
            "valid_frac",
            "||delta_h||",
        ],
        horizon_rows(
            results["error_aligned_gain"],
            [
                "random_gain",
                "error_aligned_gain_post",
                "error_aligned_gain_prior",
                "drift_direction_valid_fraction",
                "delta_norm",
            ],
        ),
    )

    print("\nDiagnostic 3: Transported-Direction Gain")
    print_table(
        ["h", "gain_random", "gain_transported", "valid_frac", "||transported_delta||"],
        horizon_rows(
            results["transported_direction_gain"],
            [
                "gain_random",
                "gain_transported",
                "transported_valid_fraction",
                "transported_delta_norm",
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
    print("DreamerV3 Prior-Regime Residual Diagnostics")
    print("=" * 78)
    print(f"Task: {config.task}, seed: {config.seed}, device: {config.device}")
    print(f"Checkpoint: {display_path(checkpoint)}")
    print(f"Config: {display_path(config_path)}")
    print(f"Replay: {display_path(replay_dir)}")
    print(f"Horizon: {args.horizon}, samples: {args.n_samples}, alpha: {args.alpha}")
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
    parser = argparse.ArgumentParser(
        description="Diagnose open-loop prior-regime residual mechanisms."
    )
    parser.add_argument("--checkpoint", required=True, help="Path to DreamerV3 latest.pt.")
    parser.add_argument("--config", required=True, help="Training config for model architecture.")
    parser.add_argument("--horizon", type=int, default=50, help="Open-loop horizon H.")
    parser.add_argument("--n_samples", type=int, default=10, help="Number of sampled start times.")
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
    parser.add_argument("--alpha", type=float, default=0.01, help="Perturbation magnitude.")
    parser.add_argument("--eps", type=float, default=1e-8, help="Near-zero norm threshold.")
    parser.add_argument("--seed", type=int, default=0, help="Diagnostic random seed.")
    args = parser.parse_args()
    if args.horizon <= 0:
        parser.error("--horizon must be positive.")
    if args.n_samples <= 0:
        parser.error("--n_samples must be positive.")
    if args.alpha <= 0:
        parser.error("--alpha must be positive.")
    if args.eps <= 0:
        parser.error("--eps must be positive.")
    return args


if __name__ == "__main__":
    main()
