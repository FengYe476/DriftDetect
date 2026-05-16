#!/usr/bin/env python3
"""DreamerV3 + SP-DPR training.

Adds Subspace-Projected Deter Prior Recovery to the world-model loss:

    L_total = L_DreamerV3 + capped beta_dpr * L_SP_DPR

SP-DPR generates off-manifold prior futures with no multi-step gradients, then
trains one gradient-enabled RSSM transition to recover posterior deter targets
inside a fixed low-rank drift subspace.
"""

from __future__ import annotations

import argparse
import functools
import pathlib
import sys
import types
from typing import Any

import numpy as np


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
DREAMERV3_ROOT = PROJECT_ROOT / "external" / "dreamerv3-torch"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    args = parse_args()
    for path in (DREAMERV3_ROOT, PROJECT_ROOT):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    import dreamer
    import torch

    loader = dreamer.yaml.YAML(typ="safe", pure=True)
    run_config = loader.load(pathlib.Path(args.config).read_text())
    configs = loader.load((DREAMERV3_ROOT / "configs.yaml").read_text())
    config_names = list(run_config.get("dreamer_configs", ["dmc_proprio"]))
    defaults: dict[str, Any] = {}
    for name in ["defaults", *config_names]:
        recursive_update(defaults, configs[name])
    if run_config.get("dreamer"):
        recursive_update(defaults, run_config["dreamer"])

    defaults["task"] = run_config["task"]
    if not defaults["task"].startswith("dmc_"):
        defaults["task"] = "dmc_" + defaults["task"]
    defaults["steps"] = int(run_config["total_steps"])
    defaults["seed"] = int(run_config["seed"])
    defaults["logdir"] = str(PROJECT_ROOT / "results" / args.run_name)
    defaults["device"] = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    config = argparse.Namespace(**defaults)
    tools = dreamer.tools
    tools.set_seed_everywhere(config.seed)

    sp_cfg = run_config.get("sp_dpr", {})
    basis_path = resolve_project_path(str(sp_cfg["basis_path"]))
    beta_dpr = float(sp_cfg.get("beta_dpr", 1.0))
    target_horizons = tuple(int(h) for h in sp_cfg.get("target_horizons", [1, 2, 4, 8, 15]))
    horizon_weights = str(sp_cfg.get("horizon_weights", "uniform"))
    n_starts = int(sp_cfg.get("n_starts", 4))
    batch_subsample = float(sp_cfg.get("batch_subsample", 1.0))
    apply_every = int(sp_cfg.get("apply_every", 1))
    loss_cap = float(sp_cfg.get("loss_cap", 0.0))
    warmup_env_steps = int(sp_cfg.get("warmup_env_steps", 50000))
    if apply_every <= 0:
        raise ValueError("sp_dpr.apply_every must be positive.")
    if not target_horizons:
        raise ValueError("sp_dpr.target_horizons must not be empty.")

    U_basis = load_basis(
        basis_path=basis_path,
        deter_dim=int(config.dyn_deter),
        device=config.device,
        torch_module=torch,
    )

    logdir = pathlib.Path(config.logdir).expanduser()
    config.traindir = config.traindir or logdir / "train_eps"
    config.evaldir = config.evaldir or logdir / "eval_eps"
    config.steps //= config.action_repeat
    config.eval_every //= config.action_repeat
    config.log_every //= config.action_repeat
    config.time_limit //= config.action_repeat

    logdir.mkdir(parents=True, exist_ok=True)
    pathlib.Path(config.traindir).mkdir(parents=True, exist_ok=True)
    pathlib.Path(config.evaldir).mkdir(parents=True, exist_ok=True)

    print_banner(
        config=config,
        basis_path=basis_path,
        basis_dim=int(U_basis.shape[1]),
        target_horizons=target_horizons,
        beta_dpr=beta_dpr,
        horizon_weights=horizon_weights,
        n_starts=n_starts,
        batch_subsample=batch_subsample,
        apply_every=apply_every,
        loss_cap=loss_cap,
        warmup_env_steps=warmup_env_steps,
    )

    step = dreamer.count_steps(config.traindir)
    logger = tools.Logger(logdir, config.action_repeat * step)

    train_eps = tools.load_episodes(config.traindir, limit=config.dataset_size)
    eval_eps = tools.load_episodes(config.evaldir, limit=1)
    train_envs = [dreamer.Damy(dreamer.make_env(config, "train", i)) for i in range(config.envs)]
    eval_envs = [dreamer.Damy(dreamer.make_env(config, "eval", i)) for i in range(config.envs)]
    acts = train_envs[0].action_space
    config.num_actions = acts.n if hasattr(acts, "n") else acts.shape[0]

    prefill = max(0, config.prefill - dreamer.count_steps(config.traindir))
    if hasattr(acts, "discrete"):
        random_actor = tools.OneHotDist(torch.zeros(config.num_actions).repeat(config.envs, 1))
    else:
        random_actor = dreamer.torchd.independent.Independent(
            dreamer.torchd.uniform.Uniform(
                torch.tensor(acts.low).repeat(config.envs, 1),
                torch.tensor(acts.high).repeat(config.envs, 1),
            ),
            1,
        )

    def random_agent(_obs, _done, _state):
        action = random_actor.sample()
        return {"action": action, "logprob": random_actor.log_prob(action)}, None

    state = tools.simulate(
        random_agent,
        train_envs,
        train_eps,
        config.traindir,
        logger,
        limit=config.dataset_size,
        steps=prefill,
    )
    logger.step += prefill * config.action_repeat

    train_dataset = dreamer.make_dataset(train_eps, config)
    agent = dreamer.Dreamer(
        train_envs[0].observation_space,
        train_envs[0].action_space,
        config,
        logger,
        train_dataset,
    ).to(config.device)
    agent.requires_grad_(requires_grad=False)

    dpr_update_count = {"count": 0}

    def sp_dpr_aware_train(world_model, data):
        env_step = agent._step * config.action_repeat
        active = env_step >= warmup_env_steps
        apply_dpr = active and (dpr_update_count["count"] % apply_every == 0)
        result = world_model_train_with_sp_dpr(
            world_model,
            data,
            U_basis=U_basis,
            beta_dpr=beta_dpr if apply_dpr else 0.0,
            target_horizons=target_horizons,
            horizon_weights=horizon_weights,
            n_starts=n_starts,
            batch_subsample=batch_subsample,
            loss_cap=loss_cap,
            tools_module=tools,
            torch_module=torch,
        )
        if active:
            dpr_update_count["count"] += 1
        return result

    agent._wm._train = types.MethodType(sp_dpr_aware_train, agent._wm)

    latest_path = logdir / "latest.pt"
    if latest_path.exists():
        ckpt = torch.load(latest_path, map_location=config.device, weights_only=False)
        agent.load_state_dict(ckpt["agent_state_dict"])
        tools.recursively_load_optim_state_dict(agent, ckpt["optims_state_dict"])
        agent._should_pretrain._once = False

    next_eval = agent._step
    final_step = config.steps + config.eval_every

    try:
        while agent._step < final_step:
            logger.write()
            if config.eval_episode_num > 0 and agent._step >= next_eval:
                eval_policy = functools.partial(agent, training=False)
                tools.simulate(
                    eval_policy,
                    eval_envs,
                    eval_eps,
                    config.evaldir,
                    logger,
                    is_eval=True,
                    episodes=config.eval_episode_num,
                )
                next_eval += config.eval_every

            state = tools.simulate(
                agent,
                train_envs,
                train_eps,
                config.traindir,
                logger,
                limit=config.dataset_size,
                steps=config.eval_every,
                state=state,
            )

            torch.save(
                {
                    "agent_state_dict": agent.state_dict(),
                    "optims_state_dict": tools.recursively_collect_optim_state_dict(agent),
                },
                latest_path,
            )
            print(f"Saved checkpoint at step {config.action_repeat * agent._step}.", flush=True)
    finally:
        for env in train_envs + eval_envs:
            try:
                env.close()
            except Exception:
                pass


def world_model_train_with_sp_dpr(
    world_model,
    data,
    *,
    U_basis,
    beta_dpr: float,
    target_horizons: tuple[int, ...],
    horizon_weights: str,
    n_starts: int,
    batch_subsample: float,
    loss_cap: float,
    tools_module,
    torch_module,
):
    """Mirror DreamerV3 WorldModel._train and add SP-DPR loss."""

    data = world_model.preprocess(data)
    dpr_metrics = zero_dpr_metrics(target_horizons, batch_subsample)

    with tools_module.RequiresGrad(world_model):
        with torch_module.cuda.amp.autocast(world_model._use_amp):
            embed = world_model.encoder(data)
            post, prior = world_model.dynamics.observe(
                embed, data["action"], data["is_first"]
            )
            kl_free = world_model._config.kl_free
            dyn_scale = world_model._config.dyn_scale
            rep_scale = world_model._config.rep_scale
            kl_loss, kl_value, dyn_loss, rep_loss = world_model.dynamics.kl_loss(
                post, prior, kl_free, dyn_scale, rep_scale
            )
            preds = {}
            for name, head in world_model.heads.items():
                grad_head = name in world_model._config.grad_heads
                feat = world_model.dynamics.get_feat(post)
                feat = feat if grad_head else feat.detach()
                pred = head(feat)
                if type(pred) is dict:
                    preds.update(pred)
                else:
                    preds[name] = pred
            losses = {}
            for name, pred in preds.items():
                loss = -pred.log_prob(data[name])
                losses[name] = loss
            scaled = {
                key: value * world_model._scales.get(key, 1.0)
                for key, value in losses.items()
            }
            model_loss = sum(scaled.values()) + kl_loss
            total_model_loss = torch_module.mean(model_loss)

            if beta_dpr > 0.0:
                from src.regularization.sp_dpr import sp_dpr_loss

                dpr_loss, raw_dpr_metrics = sp_dpr_loss(
                    world_model.dynamics,
                    post,
                    data["action"],
                    U_basis,
                    n_starts=n_starts,
                    target_horizons=target_horizons,
                    horizon_weights=horizon_weights,
                    batch_subsample=batch_subsample,
                )
                weighted_uncapped = float(beta_dpr) * dpr_loss
                if loss_cap > 0.0:
                    cap_value = float(loss_cap) * total_model_loss.detach()
                    weighted_dpr_loss = torch_module.minimum(weighted_uncapped, cap_value)
                    dpr_capped = float(
                        (weighted_uncapped.detach() > cap_value.detach()).cpu().item()
                    )
                else:
                    cap_value = total_model_loss.new_tensor(0.0)
                    weighted_dpr_loss = weighted_uncapped
                    dpr_capped = 0.0
                total_model_loss = total_model_loss + weighted_dpr_loss
                dpr_metrics.update({
                    key: tensor_to_float(value)
                    for key, value in raw_dpr_metrics.items()
                })
                dpr_metrics["dpr_active"] = 1.0
                dpr_metrics["dpr_beta"] = float(beta_dpr)
                dpr_metrics["dpr_weighted_loss"] = tensor_to_float(weighted_dpr_loss)
                dpr_metrics["dpr_weighted_uncapped_loss"] = tensor_to_float(weighted_uncapped)
                dpr_metrics["dpr_loss_cap_value"] = tensor_to_float(cap_value)
                dpr_metrics["dpr_capped"] = dpr_capped

        metrics = world_model._model_opt(total_model_loss, world_model.parameters())

    metrics.update({
        f"{name}_loss": torch_to_np(torch_module.mean(loss))
        for name, loss in losses.items()
    })
    metrics["kl_free"] = kl_free
    metrics["dyn_scale"] = dyn_scale
    metrics["rep_scale"] = rep_scale
    metrics["dyn_loss"] = torch_to_np(torch_module.mean(dyn_loss))
    metrics["rep_loss"] = torch_to_np(torch_module.mean(rep_loss))
    metrics["kl"] = torch_to_np(torch_module.mean(kl_value))

    with torch_module.cuda.amp.autocast(world_model._use_amp):
        metrics["prior_ent"] = torch_to_np(
            torch_module.mean(world_model.dynamics.get_dist(prior).entropy())
        )
        metrics["post_ent"] = torch_to_np(
            torch_module.mean(world_model.dynamics.get_dist(post).entropy())
        )
        context = dict(
            embed=embed,
            feat=world_model.dynamics.get_feat(post),
            kl=kl_value,
            postent=world_model.dynamics.get_dist(post).entropy(),
        )

    metrics.update(dpr_metrics)
    post = {key: value.detach() for key, value in post.items()}
    return post, context, metrics


def load_basis(*, basis_path: pathlib.Path, deter_dim: int, device: str, torch_module):
    if not basis_path.exists():
        raise FileNotFoundError(f"SP-DPR basis not found: {basis_path}")
    basis_np = np.load(basis_path)
    if basis_np.ndim != 2:
        raise ValueError(f"SP-DPR basis must be 2D, got {basis_np.shape}.")
    if basis_np.shape[0] != deter_dim:
        raise ValueError(
            f"SP-DPR basis leading dimension {basis_np.shape[0]} "
            f"does not match config dyn_deter {deter_dim}."
        )
    if basis_np.shape[1] <= 0:
        raise ValueError(f"SP-DPR basis must contain at least one vector, got {basis_np.shape}.")
    gram = basis_np.T @ basis_np
    orth_err = float(np.max(np.abs(gram - np.eye(basis_np.shape[1]))))
    if orth_err > 1e-4:
        raise ValueError(f"SP-DPR basis columns are not orthonormal: max error {orth_err}.")
    return torch_module.as_tensor(basis_np, dtype=torch_module.float32, device=device)


def zero_dpr_metrics(target_horizons: tuple[int, ...], batch_subsample: float) -> dict[str, float]:
    metrics = {
        "dpr_active": 0.0,
        "dpr_beta": 0.0,
        "dpr_loss": 0.0,
        "dpr_weighted_loss": 0.0,
        "dpr_weighted_uncapped_loss": 0.0,
        "dpr_loss_cap_value": 0.0,
        "dpr_capped": 0.0,
        "dpr_drift_U_norm": 0.0,
        "dpr_drift_orth_norm": 0.0,
        "dpr_n_starts": 0.0,
        "dpr_batch_subsample": float(batch_subsample),
    }
    for horizon in target_horizons:
        metrics[f"dpr_loss_h{int(horizon)}"] = 0.0
        metrics[f"dpr_weight_h{int(horizon)}"] = 0.0
    return metrics


def print_banner(
    *,
    config: argparse.Namespace,
    basis_path: pathlib.Path,
    basis_dim: int,
    target_horizons: tuple[int, ...],
    beta_dpr: float,
    horizon_weights: str,
    n_starts: int,
    batch_subsample: float,
    apply_every: int,
    loss_cap: float,
    warmup_env_steps: int,
) -> None:
    print("=" * 70)
    print("DreamerV3 + SP-DPR (Subspace-Projected Deter Prior Recovery)")
    print("=" * 70)
    print(f"Task: {config.task}, Seed: {config.seed}")
    print(f"basis: {basis_path.name} ({basis_dim} dims)")
    print(f"horizons={list(target_horizons)}, beta={beta_dpr}")
    print(f"horizon_weights={horizon_weights}, n_starts={n_starts}")
    print(f"batch_subsample={batch_subsample}, apply_every={apply_every}")
    print(f"loss_cap={loss_cap}, warmup={warmup_env_steps}")
    print("Loss: L_DreamerV3 + capped beta_dpr * SP-DPR")
    print("=" * 70, flush=True)


def tensor_to_float(value) -> float:
    return float(value.detach().cpu().item())


def torch_to_np(value):
    return value.detach().cpu().numpy()


def recursive_update(base, update) -> None:
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            recursive_update(base[key], value)
        else:
            base[key] = value


def resolve_project_path(path: str) -> pathlib.Path:
    value = pathlib.Path(path).expanduser()
    if not value.is_absolute():
        value = PROJECT_ROOT / value
    return value.resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "sp_dpr_v1_cheetah.yaml"))
    parser.add_argument("--run_name", default="sp_dpr_v1_cheetah")
    parser.add_argument("--device", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    main()
