#!/usr/bin/env python3
"""Diagnose which semantic subspaces carry DreamerV3 deter drift.

This script is intentionally diagnostic-only: all posterior extraction,
open-loop rollouts, patch interventions, and one-step gain probes run under
``torch.no_grad()``.

Correct DreamerV3 tensor convention:

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
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "tables" / "deter_semantic_drift.json"
DEFAULT_MILESTONES = (1, 2, 4, 8, 15, 30, 50)
PROBE_NAMES = ("velocity", "position", "phase")
PATCH_MODES = ("none", "velocity", "position", "phase", "all_probed")

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
    ensure_sklearn_available()
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
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    milestones = normalize_milestones(args.milestones, args.horizon)
    print_header(checkpoint, config_path, replay_dir, output_path, config, args, milestones)

    agent, env = build_agent(dreamer, torch, config, checkpoint)
    try:
        probe_data, probe_collection_meta = collect_probe_dataset(
            dreamer_module=dreamer,
            torch_module=torch,
            world_model=agent._wm,
            config=config,
            replay_dir=replay_dir,
            num_batches=args.probe_batches,
            dataset_limit=args.dataset_limit,
        )
        probes, phase_meta = fit_semantic_probes(
            probe_data=probe_data,
            alpha=args.ridge_alpha,
            svd_tol=args.svd_tol,
            seed=args.seed,
        )
        all_probed_basis = union_basis([probes[name]["basis"] for name in PROBE_NAMES], args.svd_tol)
        torch_bases = {
            name: torch.as_tensor(
                probes[name]["basis"],
                device=agent._wm.dynamics._device,
                dtype=next(agent._wm.parameters()).dtype,
            )
            for name in PROBE_NAMES
        }
        torch_bases["all_probed"] = torch.as_tensor(
            all_probed_basis,
            device=agent._wm.dynamics._device,
            dtype=next(agent._wm.parameters()).dtype,
        )

        diagnostics, rollout_collection_meta = collect_deter_semantic_diagnostics(
            dreamer_module=dreamer,
            torch_module=torch,
            world_model=agent._wm,
            config=config,
            replay_dir=replay_dir,
            probes=probes,
            all_probed_basis=all_probed_basis,
            torch_bases=torch_bases,
            horizon=args.horizon,
            n_samples=args.n_samples,
            milestones=milestones,
            alpha=args.alpha,
            eps=args.eps,
            dataset_limit=args.dataset_limit,
            max_batches=args.max_rollout_batches,
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
            "probe_batches_requested": int(args.probe_batches),
            "ridge_alpha": float(args.ridge_alpha),
            "gain_alpha": float(args.alpha),
            "eps": float(args.eps),
            "milestones": list(milestones),
            "shape_convention": "post['deter'] [B,T,D], sampled states [B,D]",
            **probe_collection_meta,
            **rollout_collection_meta,
            **phase_meta,
        },
        "linear_probes": {
            name: probe_summary_for_json(probes[name])
            for name in PROBE_NAMES
        },
        **diagnostics,
    }

    print_results(results)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(json_sanitize(results), indent=2, sort_keys=True) + "\n")
    print(f"\nSaved deter semantic drift diagnostics to {display_path(output_path)}")


def collect_probe_dataset(
    *,
    dreamer_module: Any,
    torch_module: Any,
    world_model: Any,
    config: SimpleNamespace,
    replay_dir: Path,
    num_batches: int,
    dataset_limit: int | None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    episodes = dreamer_module.tools.load_episodes(
        replay_dir,
        limit=dataset_limit or config.dataset_size,
    )
    if not episodes:
        raise FileNotFoundError(f"No replay episodes found in {replay_dir}")

    dataset = dreamer_module.make_dataset(episodes, config)
    deter_batches: list[np.ndarray] = []
    position_batches: list[np.ndarray] = []
    velocity_batches: list[np.ndarray] = []
    batch_shapes: list[dict[str, int]] = []

    with torch_module.no_grad():
        for batch_idx in range(num_batches):
            raw_data = next(dataset)
            data = world_model.preprocess(raw_data)
            require_physical_keys(data)
            embed = world_model.encoder(data)
            post, _prior = world_model.dynamics.observe(
                embed,
                data["action"],
                data["is_first"],
            )
            actions = data["action"]
            assert_batch_major(post, actions)
            post_deter = post["deter"]
            B, T, D = post_deter.shape
            assert B == actions.shape[0], f"Batch mismatch: post {B} vs action {actions.shape[0]}"
            check_physical_shape(data, B, T)

            deter_batches.append(post_deter.detach().cpu().numpy().reshape(B * T, D))
            position_batches.append(data["position"].detach().cpu().numpy().reshape(B * T, -1))
            velocity_batches.append(data["velocity"].detach().cpu().numpy().reshape(B * T, -1))
            batch_shapes.append(
                {
                    "batch_index": batch_idx,
                    "batch_size": int(B),
                    "time_length": int(T),
                    "deter_dim": int(D),
                }
            )
            print(
                f"Collected probe batch {batch_idx + 1}/{num_batches}: "
                f"B={B}, T={T}, D={D}",
                flush=True,
            )

    X = np.concatenate(deter_batches, axis=0).astype(np.float64, copy=False)
    position = np.concatenate(position_batches, axis=0).astype(np.float64, copy=False)
    velocity = np.concatenate(velocity_batches, axis=0).astype(np.float64, copy=False)
    return (
        {
            "deter": X,
            "position": position,
            "velocity": velocity,
        },
        {
            "probe_batches_collected": int(num_batches),
            "probe_num_samples": int(X.shape[0]),
            "probe_deter_dim": int(X.shape[1]),
            "probe_position_dim": int(position.shape[1]),
            "probe_velocity_dim": int(velocity.shape[1]),
            "probe_batch_shapes": batch_shapes,
        },
    )


def fit_semantic_probes(
    *,
    probe_data: Mapping[str, np.ndarray],
    alpha: float,
    svd_tol: float,
    seed: int,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    try:
        from sklearn.decomposition import PCA
        from sklearn.linear_model import Ridge
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "diagnose_deter_semantic_drift.py requires scikit-learn. "
            "Install sklearn in the active environment to run the Ridge probes."
        ) from exc

    X = np.asarray(probe_data["deter"], dtype=np.float64)
    position = np.asarray(probe_data["position"], dtype=np.float64)
    velocity = np.asarray(probe_data["velocity"], dtype=np.float64)
    physical = np.concatenate([position, velocity], axis=-1)

    pca = PCA(n_components=2, random_state=seed)
    pcs = pca.fit_transform(physical)
    phase = np.arctan2(pcs[:, 1], pcs[:, 0])
    phase_target = np.stack([np.sin(phase), np.cos(phase)], axis=-1)

    targets = {
        "velocity": velocity,
        "position": position,
        "phase": phase_target,
    }
    probes: dict[str, dict[str, Any]] = {}
    for name, target in targets.items():
        model = Ridge(alpha=alpha)
        model.fit(X, target)
        coef = np.asarray(model.coef_, dtype=np.float64)
        if coef.ndim == 1:
            coef = coef[None, :]
        basis, singular_values, rank = row_space_basis(coef, svd_tol)
        probes[name] = {
            "name": name,
            "model": model,
            "coef": coef,
            "intercept": np.asarray(model.intercept_, dtype=np.float64),
            "r2": float(model.score(X, target)),
            "target_dim": int(target.shape[1]),
            "basis": basis,
            "rank": int(rank),
            "singular_values": singular_values,
        }
        print(
            f"Fit {name} probe: target_dim={target.shape[1]}, "
            f"R2={probes[name]['r2']:.6f}, rank={rank}",
            flush=True,
        )

    phase_meta = {
        "phase_pca_explained_variance_ratio": [
            float(value) for value in pca.explained_variance_ratio_.tolist()
        ],
        "phase_definition": "phase = atan2(pc2, pc1), target = [sin(phase), cos(phase)]",
    }
    return probes, phase_meta


def ensure_sklearn_available() -> None:
    try:
        from sklearn.decomposition import PCA  # noqa: F401
        from sklearn.linear_model import Ridge  # noqa: F401
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "diagnose_deter_semantic_drift.py requires scikit-learn for "
            "sklearn.linear_model.Ridge probes and PCA phase targets."
        ) from exc


def collect_deter_semantic_diagnostics(
    *,
    dreamer_module: Any,
    torch_module: Any,
    world_model: Any,
    config: SimpleNamespace,
    replay_dir: Path,
    probes: Mapping[str, Mapping[str, Any]],
    all_probed_basis: np.ndarray,
    torch_bases: Mapping[str, Any],
    horizon: int,
    n_samples: int,
    milestones: Sequence[int],
    alpha: float,
    eps: float,
    dataset_limit: int | None,
    max_batches: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    episodes = dreamer_module.tools.load_episodes(
        replay_dir,
        limit=dataset_limit or config.dataset_size,
    )
    if not episodes:
        raise FileNotFoundError(f"No replay episodes found in {replay_dir}")

    dataset = dreamer_module.make_dataset(episodes, config)
    projection_records = init_record_map(milestones)
    gain_records = init_record_map(milestones)
    patch_records = {mode: [] for mode in PATCH_MODES}
    sample_meta: list[dict[str, int | str]] = []
    batches_read = 0
    skipped_short_batches = 0

    with torch_module.no_grad():
        while len(sample_meta) < n_samples:
            if batches_read >= max_batches:
                raise RuntimeError(
                    f"Collected {len(sample_meta)} valid samples after {batches_read} batches; "
                    f"needed {n_samples}. Increase --max_rollout_batches or lower --horizon."
                )
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

            # post["deter"] shape: [B, T, D] where B=batch, T=time, D=latent_dim
            B, T, D = post["deter"].shape
            assert B == actions.shape[0], f"Batch mismatch: post {B} vs action {actions.shape[0]}"
            if T <= horizon:
                skipped_short_batches += 1
                print(
                    f"Skipping short rollout batch {batches_read}: T={T} <= horizon={horizon}",
                    flush=True,
                )
                continue

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
                    dynamics=world_model.dynamics,
                    world_model=world_model,
                    post=post,
                    actions=actions,
                    t_start=t_start,
                    horizon=horizon,
                    milestones=milestones,
                    probes=probes,
                    all_probed_basis=all_probed_basis,
                    torch_bases=torch_bases,
                    alpha=alpha,
                    eps=eps,
                    torch_module=torch_module,
                )
                append_record_map(projection_records, sample["drift_projection"])
                append_record_map(gain_records, sample["per_subspace_gain"])
                append_patch_records(patch_records, sample["patch_attribution"])
                sample_meta.append(
                    {
                        "sample_index": len(sample_meta),
                        "batch_index": batches_read - 1,
                        "t_start": t_start,
                        "batch_size": int(B),
                        "time_length": int(T),
                        "deter_dim": int(D),
                    }
                )
                print(
                    f"Analyzed semantic sample {len(sample_meta)}/{n_samples}: "
                    f"batch={batches_read - 1}, t_start={t_start}, B={B}, T={T}",
                    flush=True,
                )
                if len(sample_meta) >= n_samples:
                    break

    diagnostics = {
        "drift_projection": aggregate_record_map(projection_records),
        "patch_attribution": aggregate_patch_records(patch_records),
        "per_subspace_gain": aggregate_record_map(gain_records),
    }
    diagnostics["per_subspace_gain_overall"] = aggregate_over_horizons(diagnostics["per_subspace_gain"])
    collection_meta = {
        "n_samples_collected": len(sample_meta),
        "rollout_batches_read": batches_read,
        "skipped_short_rollout_batches": skipped_short_batches,
        "sample_starts": sample_meta,
        "subspace_ranks": {
            name: int(probes[name]["rank"])
            for name in PROBE_NAMES
        }
        | {"all_probed": int(torch_bases["all_probed"].shape[0])},
    }
    return diagnostics, collection_meta


def analyze_start(
    *,
    dynamics: Any,
    world_model: Any,
    post: Mapping[str, Any],
    actions: Any,
    t_start: int,
    horizon: int,
    milestones: Sequence[int],
    probes: Mapping[str, Mapping[str, Any]],
    all_probed_basis: np.ndarray,
    torch_bases: Mapping[str, Any],
    alpha: float,
    eps: float,
    torch_module: Any,
) -> dict[str, Any]:
    post_states = [state_at(post, t_start + h, detach=True) for h in range(horizon + 1)]
    prior_states = open_loop_states(
        dynamics=dynamics,
        initial_state=post_states[0],
        actions=actions,
        t_start=t_start,
        horizon=horizon,
    )
    patch_states_by_mode = {
        mode: patched_rollout_states(
            dynamics=dynamics,
            post_states=post_states,
            actions=actions,
            t_start=t_start,
            horizon=horizon,
            basis=None if mode == "none" else torch_bases[mode],
        )
        for mode in PATCH_MODES
    }
    return {
        "drift_projection": diagnostic_drift_projection(
            prior_states=prior_states,
            post_states=post_states,
            probes=probes,
            all_probed_basis=all_probed_basis,
            milestones=milestones,
            eps=eps,
        ),
        "patch_attribution": diagnostic_patch_attribution(
            world_model=world_model,
            patch_states_by_mode=patch_states_by_mode,
            post_states=post_states,
            horizon=horizon,
        ),
        "per_subspace_gain": diagnostic_per_subspace_gain(
            dynamics=dynamics,
            post_states=post_states,
            prior_states=prior_states,
            actions=actions,
            t_start=t_start,
            milestones=milestones,
            torch_bases=torch_bases,
            alpha=alpha,
            eps=eps,
            torch_module=torch_module,
        ),
    }


def diagnostic_drift_projection(
    *,
    prior_states: Sequence[Mapping[str, Any]],
    post_states: Sequence[Mapping[str, Any]],
    probes: Mapping[str, Mapping[str, Any]],
    all_probed_basis: np.ndarray,
    milestones: Sequence[int],
    eps: float,
) -> dict[int, dict[str, float]]:
    records: dict[int, dict[str, float]] = {}
    for h in milestones:
        delta = (prior_states[h]["deter"] - post_states[h]["deter"]).detach().cpu().numpy()
        record = {"delta_deter_norm": float(np.linalg.norm(delta, axis=-1).mean())}
        for name in PROBE_NAMES:
            record[f"{name}_frac"] = projection_energy_fraction(
                delta,
                probes[name]["basis"],
                eps=eps,
            )
        record["all_probed_frac"] = projection_energy_fraction(
            delta,
            all_probed_basis,
            eps=eps,
        )
        records[int(h)] = record
    return records


def diagnostic_patch_attribution(
    *,
    world_model: Any,
    patch_states_by_mode: Mapping[str, Sequence[Mapping[str, Any]]],
    post_states: Sequence[Mapping[str, Any]],
    horizon: int,
) -> dict[str, dict[str, float]]:
    eval_indices = raw_j_indices(horizon)
    h15 = min(15, horizon)
    h30 = 30 if horizon >= 30 else None
    h50 = 50 if horizon >= 50 else None

    records: dict[str, dict[str, float]] = {}
    for mode, states in patch_states_by_mode.items():
        raw_values = []
        for h in eval_indices:
            drift = states[h]["deter"] - post_states[h]["deter"]
            raw_values.append(drift.pow(2).sum(dim=-1).mean())
        raw_j = stack_scalars(raw_values).mean()
        obs15, feat15 = decoder_pair_or_feature_metric(world_model, states[h15], post_states[h15])
        obs30, feat30 = (math.nan, math.nan)
        obs50, feat50 = (math.nan, math.nan)
        if h30 is not None:
            obs30, feat30 = decoder_pair_or_feature_metric(world_model, states[h30], post_states[h30])
        if h50 is not None:
            obs50, feat50 = decoder_pair_or_feature_metric(world_model, states[h50], post_states[h50])
        records[mode] = {
            "raw_j": scalar(raw_j),
            "obs_mse_h15": obs15,
            "obs_mse_h30": obs30,
            "obs_mse_h50": obs50,
            "feature_dist_h15": feat15,
            "feature_dist_h30": feat30,
            "feature_dist_h50": feat50,
        }
    return records


def diagnostic_per_subspace_gain(
    *,
    dynamics: Any,
    post_states: Sequence[Mapping[str, Any]],
    prior_states: Sequence[Mapping[str, Any]],
    actions: Any,
    t_start: int,
    milestones: Sequence[int],
    torch_bases: Mapping[str, Any],
    alpha: float,
    eps: float,
    torch_module: Any,
) -> dict[int, dict[str, float]]:
    records: dict[int, dict[str, float]] = {}
    for h in milestones:
        state = post_states[h]
        action_h = actions[:, t_start + h].detach()
        random_direction, _random_valid = unit_vectors(torch_module.randn_like(state["deter"]), eps)
        gain_random = one_step_deter_gain(
            dynamics=dynamics,
            state=state,
            action=action_h,
            direction=random_direction,
            alpha=alpha,
            eps=eps,
        )

        record = {
            "gain_random": masked_mean(gain_random, None),
        }
        for name in PROBE_NAMES:
            direction, valid = random_direction_in_subspace(
                basis=torch_bases[name],
                batch_size=state["deter"].shape[0],
                eps=eps,
                torch_module=torch_module,
            )
            gain = one_step_deter_gain(
                dynamics=dynamics,
                state=state,
                action=action_h,
                direction=direction,
                alpha=alpha,
                eps=eps,
            )
            record[f"gain_{name}"] = masked_mean(gain, valid)
            record[f"{name}_valid_fraction"] = valid_fraction(valid)

        delta_h = prior_states[h]["deter"] - post_states[h]["deter"]
        drift_direction, drift_valid = unit_vectors(delta_h, eps)
        drift_gain = one_step_deter_gain(
            dynamics=dynamics,
            state=state,
            action=action_h,
            direction=drift_direction,
            alpha=alpha,
            eps=eps,
        )
        record["gain_drift_direction"] = masked_mean(drift_gain, drift_valid)
        record["drift_direction_valid_fraction"] = valid_fraction(drift_valid)
        record["delta_deter_norm"] = norm_mean(delta_h)
        records[int(h)] = record
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
        states.append(clone_state(state))
    return states


def patched_rollout_states(
    *,
    dynamics: Any,
    post_states: Sequence[Mapping[str, Any]],
    actions: Any,
    t_start: int,
    horizon: int,
    basis: Any | None,
) -> list[dict[str, Any]]:
    state = clone_state(post_states[0])
    states = [clone_state(state)]
    for h in range(horizon):
        action_h = actions[:, t_start + h].detach()
        state = detach_state(dynamics.img_step(state, action_h, sample=False))
        if basis is not None and int(basis.shape[0]) > 0:
            state["deter"] = patch_deter_subspace(
                prior_deter=state["deter"],
                post_deter=post_states[h + 1]["deter"],
                basis=basis,
            )
        states.append(clone_state(state))
    return states


def patch_deter_subspace(*, prior_deter: Any, post_deter: Any, basis: Any) -> Any:
    prior_proj = project_torch(prior_deter, basis)
    post_proj = project_torch(post_deter, basis)
    return prior_deter - prior_proj + post_proj


def project_torch(x: Any, basis: Any) -> Any:
    if int(basis.shape[0]) == 0:
        return x.new_zeros(x.shape)
    return (x @ basis.transpose(0, 1)) @ basis


def one_step_deter_gain(
    *,
    dynamics: Any,
    state: Mapping[str, Any],
    action: Any,
    direction: Any,
    alpha: float,
    eps: float,
) -> Any:
    base_state = clone_state(state)
    pert_state = clone_state(state)
    perturb = alpha * direction.detach()
    pert_state["deter"] = base_state["deter"] + perturb
    base_next = dynamics.img_step(base_state, action.detach(), sample=False)
    pert_next = dynamics.img_step(pert_state, action.detach(), sample=False)
    numerator = (pert_next["deter"] - base_next["deter"]).norm(dim=-1)
    denominator = perturb.norm(dim=-1).clamp_min(eps)
    return numerator / denominator


def random_direction_in_subspace(
    *,
    basis: Any,
    batch_size: int,
    eps: float,
    torch_module: Any,
) -> tuple[Any, Any]:
    if int(basis.shape[0]) == 0:
        zeros = basis.new_zeros((batch_size, int(basis.shape[1])))
        return zeros, zeros.new_zeros((batch_size,), dtype=torch_module.bool)
    coeff = torch_module.randn(
        (batch_size, int(basis.shape[0])),
        device=basis.device,
        dtype=basis.dtype,
    )
    direction = coeff @ basis
    return unit_vectors(direction, eps)


def projection_energy_fraction(delta: np.ndarray, basis: np.ndarray, *, eps: float) -> float:
    delta = np.asarray(delta, dtype=np.float64)
    basis = np.asarray(basis, dtype=np.float64)
    per_row_energy = np.sum(delta * delta, axis=-1)
    valid = per_row_energy > eps
    if not np.any(valid):
        return math.nan
    if basis.size == 0 or basis.shape[0] == 0:
        return 0.0
    projected = delta[valid] @ basis.T
    numerator = float(np.sum(projected * projected))
    denominator = float(np.sum(per_row_energy[valid]))
    if denominator <= eps:
        return math.nan
    frac = numerator / denominator
    return float(min(max(frac, 0.0), 1.0))


def row_space_basis(coef: np.ndarray, svd_tol: float) -> tuple[np.ndarray, np.ndarray, int]:
    coef = np.asarray(coef, dtype=np.float64)
    if coef.ndim != 2:
        raise ValueError(f"Expected 2D coefficient matrix, got shape {coef.shape}")
    if coef.shape[0] == 0:
        return np.zeros((0, coef.shape[1]), dtype=np.float64), np.zeros((0,), dtype=np.float64), 0
    _u, singular_values, vh = np.linalg.svd(coef, full_matrices=False)
    if singular_values.size == 0:
        rank = 0
    else:
        rank = int(np.sum(singular_values > svd_tol * max(float(singular_values[0]), 1.0)))
    return vh[:rank].astype(np.float64, copy=False), singular_values.astype(np.float64, copy=False), rank


def union_basis(bases: Sequence[np.ndarray], svd_tol: float) -> np.ndarray:
    nonempty = [np.asarray(basis, dtype=np.float64) for basis in bases if np.asarray(basis).size > 0]
    if not nonempty:
        return np.zeros((0, 0), dtype=np.float64)
    dim = nonempty[0].shape[1]
    if any(basis.ndim != 2 or basis.shape[1] != dim for basis in nonempty):
        raise ValueError("All bases must be 2D and share deter dimension.")
    stacked = np.concatenate(nonempty, axis=0)
    basis, _singular_values, _rank = row_space_basis(stacked, svd_tol)
    return basis


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
        torch_module = __import__("torch")
        pred_flat = torch_module.cat(pred_parts, dim=-1)
        target_flat = torch_module.cat(target_parts, dim=-1)
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


def raw_j_indices(horizon: int) -> list[int]:
    eval_start = min(25, horizon)
    eval_stop = min(50, horizon)
    if eval_start > eval_stop:
        eval_start, eval_stop = (1, horizon) if horizon >= 1 else (0, 0)
    return list(range(eval_start, eval_stop + 1))


def require_physical_keys(data: Mapping[str, Any]) -> None:
    missing = [key for key in ("position", "velocity") if key not in data]
    if missing:
        raise KeyError(f"Replay batch is missing required physical keys: {missing}")


def check_physical_shape(data: Mapping[str, Any], batch_size: int, time_steps: int) -> None:
    for key, expected_dim in (("position", 8), ("velocity", 9)):
        value = data[key]
        if value.ndim != 3:
            raise ValueError(f"Expected data['{key}'] [B,T,D], got {value.shape}")
        if value.shape[0] != batch_size or value.shape[1] != time_steps:
            raise ValueError(
                f"data['{key}'] must share post batch/time dimensions, got {value.shape} "
                f"vs B={batch_size}, T={time_steps}"
            )
        if value.shape[-1] != expected_dim:
            print(
                f"Warning: expected {key} dim {expected_dim}, got {value.shape[-1]}",
                flush=True,
            )


def assert_batch_major(post: Mapping[str, Any], actions: Any) -> None:
    post_deter = post["deter"]
    assert post_deter.ndim == 3, f"Expected post['deter'] [B,T,D], got {post_deter.shape}"
    assert actions.ndim == 3, f"Expected actions [B,T,A], got {actions.shape}"
    B, T, D = post_deter.shape
    del D
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


def append_patch_records(
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


def aggregate_patch_records(records: Mapping[str, Sequence[Mapping[str, float]]]) -> dict[str, Any]:
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


def aggregate_over_horizons(section: Mapping[str, Any]) -> dict[str, Any]:
    rows = []
    for stats in section.values():
        row = {}
        for key, value in stats.items():
            if isinstance(value, Mapping) and "mean" in value:
                row[key] = float(value["mean"])
        if row:
            rows.append(row)
    return aggregate_records(rows)


def probe_summary_for_json(probe: Mapping[str, Any]) -> dict[str, Any]:
    coef = np.asarray(probe["coef"], dtype=np.float64)
    singular_values = np.asarray(probe["singular_values"], dtype=np.float64)
    basis = np.asarray(probe["basis"], dtype=np.float64)
    return {
        "target_dim": int(probe["target_dim"]),
        "r2": float(probe["r2"]),
        "coef_shape": list(coef.shape),
        "coef_fro_norm": float(np.linalg.norm(coef)),
        "basis_shape": list(basis.shape),
        "rank": int(probe["rank"]),
        "singular_values": [float(value) for value in singular_values.tolist()],
        "intercept_norm": float(np.linalg.norm(np.asarray(probe["intercept"], dtype=np.float64))),
    }


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
    print("\nDeter semantic sample set")
    print_table(
        ["field", "value"],
        [
            ["probe samples", str(meta["probe_num_samples"])],
            ["probe batches", str(meta["probe_batches_collected"])],
            ["rollout samples", str(meta["n_samples_collected"])],
            ["rollout batches", str(meta["rollout_batches_read"])],
            ["horizon", str(meta["horizon"])],
            ["shape", meta["shape_convention"]],
        ],
    )

    print("\nAnalysis 1: Linear Probes")
    probe_rows = []
    for name in PROBE_NAMES:
        probe = results["linear_probes"][name]
        probe_rows.append(
            [
                name,
                str(probe["target_dim"]),
                str(probe["rank"]),
                fmt(probe["r2"]),
                fmt(probe["coef_fro_norm"]),
            ]
        )
    print_table(["probe", "target_dim", "rank", "R^2", "||W||_F"], probe_rows)

    print("\nAnalysis 2: Drift Projection Fractions")
    print_table(
        ["h", "velocity", "position", "phase", "all_probed", "||delta||"],
        horizon_rows(
            results["drift_projection"],
            [
                "velocity_frac",
                "position_frac",
                "phase_frac",
                "all_probed_frac",
                "delta_deter_norm",
            ],
            percent_keys={"velocity_frac", "position_frac", "phase_frac", "all_probed_frac"},
        ),
    )

    print("\nAnalysis 3: Patch Attribution")
    patch = results["patch_attribution"]
    print_table(
        ["mode", "RawJ", "Obs@15", "Obs@50", "Feat@15", "Feat@50"],
        [
            [
                mode,
                fmt(patch[mode]["raw_j"]["mean"]),
                fmt(patch[mode]["obs_mse_h15"]["mean"]),
                fmt(patch[mode]["obs_mse_h50"]["mean"]),
                fmt(patch[mode]["feature_dist_h15"]["mean"]),
                fmt(patch[mode]["feature_dist_h50"]["mean"]),
            ]
            for mode in PATCH_MODES
        ],
    )

    print("\nAnalysis 4: Per-Subspace One-Step Gain")
    print_table(
        ["h", "random", "velocity", "position", "phase", "drift_dir", "||delta||"],
        horizon_rows(
            results["per_subspace_gain"],
            [
                "gain_random",
                "gain_velocity",
                "gain_position",
                "gain_phase",
                "gain_drift_direction",
                "delta_deter_norm",
            ],
        ),
    )
    overall = results["per_subspace_gain_overall"]
    print_table(
        ["metric", "mean"],
        [
            ["gain_random", fmt(overall["gain_random"]["mean"])],
            ["gain_velocity", fmt(overall["gain_velocity"]["mean"])],
            ["gain_position", fmt(overall["gain_position"]["mean"])],
            ["gain_phase", fmt(overall["gain_phase"]["mean"])],
            ["gain_drift_direction", fmt(overall["gain_drift_direction"]["mean"])],
        ],
    )


def horizon_rows(
    section: Mapping[str, Any],
    keys: Sequence[str],
    *,
    percent_keys: set[str] | None = None,
) -> list[list[str]]:
    percent_keys = percent_keys or set()
    rows = []
    for horizon in sorted(section, key=lambda value: int(value)):
        row = [horizon]
        stats = section[horizon]
        for key in keys:
            value = stats[key]["mean"]
            row.append(pct(value) if key in percent_keys else fmt(value))
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
    print("DreamerV3 Deter Semantic Drift Diagnostics")
    print("=" * 78)
    print(f"Task: {config.task}, seed: {config.seed}, device: {config.device}")
    print(f"Checkpoint: {display_path(checkpoint)}")
    print(f"Config: {display_path(config_path)}")
    print(f"Replay: {display_path(replay_dir)}")
    print(
        f"Horizon: {args.horizon}, samples: {args.n_samples}, "
        f"probe_batches: {args.probe_batches}, ridge_alpha: {args.ridge_alpha}"
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
    if value is None:
        return "nan"
    value = float(value)
    if math.isnan(value):
        return "nan"
    if value == 0.0:
        return "0"
    abs_value = abs(value)
    if abs_value < 1e-3 or abs_value >= 1e4:
        return f"{value:.4e}"
    return f"{value:.6f}"


def pct(value: Any) -> str:
    if value is None:
        return "nan"
    value = float(value)
    if math.isnan(value):
        return "nan"
    return f"{100.0 * value:.2f}%"


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
        description="Diagnose semantic subspaces of DreamerV3 deterministic RSSM drift."
    )
    parser.add_argument("--checkpoint", required=True, help="Path to DreamerV3 latest.pt.")
    parser.add_argument("--config", required=True, help="Training config for model architecture.")
    parser.add_argument("--horizon", type=int, default=50, help="Open-loop horizon H.")
    parser.add_argument("--n_samples", type=int, default=5, help="Number of sampled start times.")
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
        default=list(DEFAULT_MILESTONES),
        help="Horizons to report; values outside [0,H] are ignored.",
    )
    parser.add_argument(
        "--probe_batches",
        type=int,
        default=2,
        help="Replay batches used to fit semantic probes. Must be at least 2.",
    )
    parser.add_argument("--ridge_alpha", type=float, default=1.0, help="sklearn Ridge alpha.")
    parser.add_argument("--alpha", type=float, default=0.01, help="Gain perturbation magnitude.")
    parser.add_argument("--eps", type=float, default=1e-8, help="Near-zero norm threshold.")
    parser.add_argument("--svd_tol", type=float, default=1e-8, help="Relative SVD rank tolerance.")
    parser.add_argument("--seed", type=int, default=0, help="Diagnostic random seed.")
    parser.add_argument(
        "--max_rollout_batches",
        type=int,
        default=100,
        help="Maximum replay batches to scan while looking for valid rollout starts.",
    )
    args = parser.parse_args()
    if args.horizon <= 0:
        parser.error("--horizon must be positive.")
    if args.n_samples <= 0:
        parser.error("--n_samples must be positive.")
    if args.probe_batches < 2:
        parser.error("--probe_batches must be at least 2.")
    if args.ridge_alpha <= 0:
        parser.error("--ridge_alpha must be positive.")
    if args.alpha <= 0:
        parser.error("--alpha must be positive.")
    if args.eps <= 0:
        parser.error("--eps must be positive.")
    if args.svd_tol <= 0:
        parser.error("--svd_tol must be positive.")
    if args.max_rollout_batches <= 0:
        parser.error("--max_rollout_batches must be positive.")
    return args


if __name__ == "__main__":
    main()
