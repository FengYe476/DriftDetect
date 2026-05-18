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
import math
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
    basis_path_value = str(sp_cfg["basis_path"])
    auto_basis = basis_path_value == "auto"
    basis_path = None if auto_basis else resolve_project_path(basis_path_value)
    beta_dpr = float(sp_cfg.get("beta_dpr", 1.0))
    target_horizons = tuple(int(h) for h in sp_cfg.get("target_horizons", [1, 2, 4, 8, 15]))
    horizon_weights = str(sp_cfg.get("horizon_weights", "uniform"))
    n_starts = int(sp_cfg.get("n_starts", 4))
    batch_subsample = float(sp_cfg.get("batch_subsample", 1.0))
    apply_every = int(sp_cfg.get("apply_every", 1))
    loss_cap = float(sp_cfg.get("loss_cap", 0.0))
    warmup_env_steps = int(sp_cfg.get("warmup_env_steps", 50000))
    auto_k_method = str(sp_cfg.get("auto_k_method", "variance"))
    auto_k_threshold = float(sp_cfg.get("auto_k_threshold", 0.85))
    auto_k_min = int(sp_cfg.get("auto_k_min", 8))
    auto_k_max = int(sp_cfg.get("auto_k_max", 64))
    auto_horizon = int(sp_cfg.get("auto_horizon", 50))
    auto_n_batches = int(sp_cfg.get("auto_n_batches", 3))
    auto_n_rollouts = int(sp_cfg.get("auto_n_rollouts", 5))
    basis_bank = config_bool(sp_cfg.get("basis_bank", False))
    escape_check_step = int(sp_cfg.get("escape_check_step", 100000))
    escape_marginal_threshold = float(sp_cfg.get("escape_marginal_threshold", 0.01))
    escape_k_max_multiplier = float(sp_cfg.get("escape_k_max_multiplier", 2.0))
    escape_k_max_absolute = int(sp_cfg.get("escape_k_max_absolute", 128))
    escape_variance_threshold = float(sp_cfg.get("escape_variance_threshold", 0.80))
    loss_type = str(sp_cfg.get("loss_type", "mse")).lower()
    huber_delta = float(sp_cfg.get("huber_delta", 1.0))
    if apply_every <= 0:
        raise ValueError("sp_dpr.apply_every must be positive.")
    if not target_horizons:
        raise ValueError("sp_dpr.target_horizons must not be empty.")
    validate_loss_config(loss_type=loss_type, huber_delta=huber_delta)
    if basis_bank:
        validate_basis_bank_config(
            escape_check_step=escape_check_step,
            escape_marginal_threshold=escape_marginal_threshold,
            escape_k_max_multiplier=escape_k_max_multiplier,
            escape_k_max_absolute=escape_k_max_absolute,
            escape_variance_threshold=escape_variance_threshold,
        )
    if auto_basis or basis_bank:
        validate_auto_basis_config(
            auto_k_method=auto_k_method,
            auto_k_threshold=auto_k_threshold,
            auto_k_min=auto_k_min,
            auto_k_max=auto_k_max,
            auto_horizon=auto_horizon,
            auto_n_batches=auto_n_batches,
            auto_n_rollouts=auto_n_rollouts,
        )

    U_basis = None
    if not auto_basis:
        assert basis_path is not None
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
        auto_basis=auto_basis,
        basis_dim=None if U_basis is None else int(U_basis.shape[1]),
        target_horizons=target_horizons,
        beta_dpr=beta_dpr,
        horizon_weights=horizon_weights,
        n_starts=n_starts,
        batch_subsample=batch_subsample,
        apply_every=apply_every,
        loss_cap=loss_cap,
        warmup_env_steps=warmup_env_steps,
        basis_bank=basis_bank,
        escape_check_step=escape_check_step,
        escape_k_max_multiplier=escape_k_max_multiplier,
        escape_k_max_absolute=escape_k_max_absolute,
        loss_type=loss_type,
        huber_delta=huber_delta,
        auto_k_threshold=auto_k_threshold,
        auto_k_min=auto_k_min,
        auto_k_max=auto_k_max,
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
    basis_state = {
        "basis": U_basis,
        "k0": None if U_basis is None else int(U_basis.shape[1]),
        "kmax": None,
        "escape_checked": False,
    }
    if basis_bank and basis_state["k0"] is not None:
        basis_state["kmax"] = compute_bank_kmax(
            int(basis_state["k0"]),
            escape_k_max_multiplier=escape_k_max_multiplier,
            escape_k_max_absolute=escape_k_max_absolute,
        )

    def sp_dpr_aware_train(world_model, data):
        env_step = agent._step * config.action_repeat
        active = env_step >= warmup_env_steps
        if auto_basis and active and basis_state["basis"] is None:
            basis_state["basis"] = calibrate_drift_basis(
                world_model=world_model,
                train_dataset=train_dataset,
                deter_dim=int(config.dyn_deter),
                device=config.device,
                logdir=logdir,
                auto_k_method=auto_k_method,
                auto_k_threshold=auto_k_threshold,
                auto_k_min=auto_k_min,
                auto_k_max=auto_k_max,
                auto_horizon=auto_horizon,
                auto_n_batches=auto_n_batches,
                auto_n_rollouts=auto_n_rollouts,
                torch_module=torch,
            )
            basis_state["k0"] = int(basis_state["basis"].shape[1])
            if basis_bank:
                basis_state["kmax"] = compute_bank_kmax(
                    int(basis_state["k0"]),
                    escape_k_max_multiplier=escape_k_max_multiplier,
                    escape_k_max_absolute=escape_k_max_absolute,
                )
            print(
                f"SP-DPR activated: basis {tuple(basis_state['basis'].shape)}, "
                f"k={basis_state['basis'].shape[1]}",
                flush=True,
            )
            if basis_bank:
                print(
                    f"A-BB-SP-DPR bank initialized: k0={basis_state['k0']}, "
                    f"Kmax={basis_state['kmax']}",
                    flush=True,
                )

        escape_checked_this_step = 0.0
        orth_fraction_this_step = 0.0
        if (
            basis_bank
            and active
            and basis_state["basis"] is not None
            and not bool(basis_state["escape_checked"])
            and env_step >= escape_check_step
        ):
            basis_state["basis"], escape_info = check_escape(
                world_model=world_model,
                train_dataset=train_dataset,
                U_bank=basis_state["basis"],
                deter_dim=int(config.dyn_deter),
                device=config.device,
                logdir=logdir,
                k0=int(basis_state["k0"]),
                kmax=int(basis_state["kmax"]),
                auto_horizon=auto_horizon,
                auto_n_batches=auto_n_batches,
                auto_n_rollouts=auto_n_rollouts,
                escape_marginal_threshold=escape_marginal_threshold,
                escape_variance_threshold=escape_variance_threshold,
                torch_module=torch,
            )
            basis_state["escape_checked"] = True
            escape_checked_this_step = 1.0
            orth_fraction_this_step = float(escape_info["orth_fraction"])

        apply_dpr = (
            active
            and basis_state["basis"] is not None
            and (dpr_update_count["count"] % apply_every == 0)
        )
        result = world_model_train_with_sp_dpr(
            world_model,
            data,
            U_basis=basis_state["basis"],
            beta_dpr=beta_dpr if apply_dpr else 0.0,
            target_horizons=target_horizons,
            horizon_weights=horizon_weights,
            n_starts=n_starts,
            batch_subsample=batch_subsample,
            loss_cap=loss_cap,
            loss_type=loss_type,
            huber_delta=huber_delta,
            bank_dims=0 if basis_state["basis"] is None else int(basis_state["basis"].shape[1]),
            escape_checked=escape_checked_this_step,
            orth_fraction=orth_fraction_this_step,
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
    loss_type: str,
    huber_delta: float,
    bank_dims: int,
    escape_checked: float,
    orth_fraction: float,
    tools_module,
    torch_module,
):
    """Mirror DreamerV3 WorldModel._train and add SP-DPR loss."""

    data = world_model.preprocess(data)
    dpr_metrics = zero_dpr_metrics(target_horizons, batch_subsample)
    dpr_metrics["dpr_bank_dims"] = float(bank_dims)
    dpr_metrics["dpr_escape_checked"] = float(escape_checked)
    dpr_metrics["dpr_orth_fraction"] = float(orth_fraction)

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
                if U_basis is None:
                    raise RuntimeError("SP-DPR is active but U_basis has not been calibrated or loaded.")
                if loss_type == "mse":
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
                elif loss_type == "huber":
                    dpr_loss, raw_dpr_metrics = sp_dpr_huber_loss(
                        world_model.dynamics,
                        post,
                        data["action"],
                        U_basis,
                        n_starts=n_starts,
                        target_horizons=target_horizons,
                        horizon_weights=horizon_weights,
                        batch_subsample=batch_subsample,
                        huber_delta=huber_delta,
                        torch_module=torch_module,
                    )
                else:
                    raise ValueError(f"Unsupported sp_dpr.loss_type: {loss_type!r}.")
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
                dpr_metrics["dpr_bank_dims"] = float(bank_dims)
                dpr_metrics["dpr_escape_checked"] = float(escape_checked)
                dpr_metrics["dpr_orth_fraction"] = float(orth_fraction)

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


def sp_dpr_huber_loss(
    dynamics,
    post: dict,
    actions,
    U_basis,
    *,
    n_starts: int,
    target_horizons: tuple[int, ...],
    horizon_weights: str,
    batch_subsample: float,
    huber_delta: float,
    torch_module,
):
    """SP-DPR loss variant with Huber penalty, local to this training script."""

    if n_starts <= 0:
        raise ValueError("n_starts must be positive.")
    if not 0.0 < batch_subsample <= 1.0:
        raise ValueError("batch_subsample must be in (0, 1].")
    if huber_delta <= 0.0:
        raise ValueError("huber_delta must be positive.")
    if "deter" not in post:
        raise ValueError("post must contain key 'deter'.")
    if "stoch" not in post:
        raise ValueError("post must contain key 'stoch'.")

    post_deter = post["deter"]
    if post_deter.ndim != 3:
        raise ValueError(f"post['deter'] must have shape (B, T, D), got {post_deter.shape}.")
    if actions.ndim != 3:
        raise ValueError(f"actions must have shape (B, T, A), got {actions.shape}.")
    if U_basis.ndim != 2:
        raise ValueError(f"U_basis must have shape (D, R), got {U_basis.shape}.")

    batch_size, time_steps, deter_dim = post_deter.shape
    if actions.shape[:2] != (batch_size, time_steps):
        raise ValueError(
            "actions and post['deter'] must share batch/time dimensions: "
            f"{actions.shape[:2]} vs {(batch_size, time_steps)}."
        )
    if U_basis.shape[0] != deter_dim:
        raise ValueError(
            f"U_basis leading dimension must match deter dim {deter_dim}, got {U_basis.shape}."
        )
    if U_basis.shape[1] <= 0:
        raise ValueError(f"U_basis must contain at least one basis vector, got {U_basis.shape}.")
    for key, value in post.items():
        if hasattr(value, "shape") and value.shape[:2] != (batch_size, time_steps):
            raise ValueError(
                f"post[{key!r}] must share batch/time dimensions, got {value.shape[:2]} "
                f"vs {(batch_size, time_steps)}."
            )

    horizons = normalize_horizons(target_horizons)
    weights = make_horizon_weights(
        horizons,
        horizon_weights,
        device=post_deter.device,
        dtype=post_deter.dtype,
        torch_module=torch_module,
    )
    U_basis = U_basis.detach().to(device=post_deter.device, dtype=post_deter.dtype)
    max_horizon = max(horizons)

    if time_steps < max_horizon + 2:
        return zero_sp_dpr_tensor_metrics(post_deter, horizons, batch_subsample)

    if batch_subsample < 1.0:
        n_keep = max(1, int(round(batch_size * float(batch_subsample))))
        indices = torch_module.randperm(batch_size, device=post_deter.device)[:n_keep]
        post_work = {key: value[indices] for key, value in post.items()}
        actions_work = actions[indices]
    else:
        post_work = post
        actions_work = actions

    losses_by_horizon = {horizon: [] for horizon in horizons}
    drift_u_norms = []
    drift_orth_norms = []
    start_losses = []

    starts = torch_module.randint(
        0,
        time_steps - max_horizon - 1,
        (int(n_starts),),
        device=post_deter.device,
    )

    for start in starts:
        t_start = int(start.item())
        with torch_module.no_grad():
            state = state_at_time(post_work, t_start)
            futures = {}
            for step in range(max_horizon):
                action_step = actions_work[:, t_start + step].detach()
                state = dynamics.img_step(state, action_step, sample=False)
                state = detach_state(state)
                horizon = step + 1
                if horizon in losses_by_horizon:
                    futures[horizon] = state

        weighted_start_loss = post_deter.new_tensor(0.0)
        with torch_module.enable_grad():
            for idx, horizon in enumerate(horizons):
                future_state = detach_state(futures[horizon])
                action_h = actions_work[:, t_start + horizon].detach()
                pred_next = dynamics.img_step(future_state, action_h, sample=False)
                target_deter = post_work["deter"][:, t_start + horizon + 1].detach()

                error = pred_next["deter"] - target_deter
                error_u = error @ U_basis
                error_u_float = error_u.float()
                loss_h = torch_module.nn.functional.huber_loss(
                    error_u_float,
                    torch_module.zeros_like(error_u_float),
                    reduction="mean",
                    delta=float(huber_delta),
                )
                losses_by_horizon[horizon].append(loss_h)
                weighted_start_loss = weighted_start_loss + weights[idx] * loss_h

                posterior_h = post_work["deter"][:, t_start + horizon].detach()
                drift = future_state["deter"] - posterior_h
                drift_u = drift @ U_basis
                drift_proj = drift_u @ U_basis.transpose(0, 1)
                drift_u_norms.append(drift_u.float().norm(dim=-1).mean())
                drift_orth_norms.append((drift - drift_proj).float().norm(dim=-1).mean())

        start_losses.append(weighted_start_loss)

    total_loss = torch_module.stack(start_losses).mean()
    metrics = {
        "dpr_loss": total_loss.detach(),
        "dpr_drift_U_norm": torch_module.stack(drift_u_norms).mean().detach(),
        "dpr_drift_orth_norm": torch_module.stack(drift_orth_norms).mean().detach(),
        "dpr_n_starts": post_deter.new_tensor(float(n_starts)),
        "dpr_batch_subsample": post_deter.new_tensor(float(batch_subsample)),
    }
    for idx, horizon in enumerate(horizons):
        metrics[f"dpr_loss_h{horizon}"] = (
            torch_module.stack(losses_by_horizon[horizon]).mean().detach()
        )
        metrics[f"dpr_weight_h{horizon}"] = weights[idx].detach()
    return total_loss, metrics


def state_at_time(post: dict, t: int) -> dict:
    return {key: value[:, t].detach() for key, value in post.items()}


def detach_state(state: dict) -> dict:
    return {key: value.detach() for key, value in state.items()}


def normalize_horizons(horizons: tuple[int, ...]) -> tuple[int, ...]:
    if isinstance(horizons, int):
        horizons = (horizons,)
    normalized = tuple(sorted({int(horizon) for horizon in horizons}))
    if not normalized:
        raise ValueError("target_horizons must not be empty.")
    if any(horizon <= 0 for horizon in normalized):
        raise ValueError(f"target_horizons must be positive, got {normalized}.")
    return normalized


def make_horizon_weights(
    horizons: tuple[int, ...],
    mode: str,
    *,
    device,
    dtype,
    torch_module,
):
    if mode == "uniform":
        weights = torch_module.ones(len(horizons), device=device, dtype=dtype)
    elif mode == "sqrt_inv":
        weights = torch_module.tensor(
            [1.0 / math.sqrt(float(horizon)) for horizon in horizons],
            device=device,
            dtype=dtype,
        )
    else:
        raise ValueError(f"Unsupported horizon_weights: {mode!r}.")
    return weights / weights.sum()


def zero_sp_dpr_tensor_metrics(post_deter, horizons: tuple[int, ...], batch_subsample: float):
    zero = post_deter.new_tensor(0.0)
    metrics = {
        "dpr_loss": zero,
        "dpr_drift_U_norm": zero,
        "dpr_drift_orth_norm": zero,
        "dpr_n_starts": zero,
        "dpr_batch_subsample": post_deter.new_tensor(float(batch_subsample)),
    }
    for horizon in horizons:
        metrics[f"dpr_loss_h{horizon}"] = zero
        metrics[f"dpr_weight_h{horizon}"] = zero
    return zero, metrics


def calibrate_drift_basis(
    *,
    world_model,
    train_dataset,
    deter_dim: int,
    device: str,
    logdir: pathlib.Path,
    auto_k_method: str,
    auto_k_threshold: float,
    auto_k_min: int,
    auto_k_max: int,
    auto_horizon: int,
    auto_n_batches: int,
    auto_n_rollouts: int,
    torch_module,
):
    if auto_k_method != "variance":
        raise ValueError(f"Unsupported sp_dpr.auto_k_method: {auto_k_method!r}.")

    try:
        from sklearn.decomposition import PCA
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "A-SP-DPR auto calibration requires scikit-learn for PCA."
        ) from exc

    deltas = collect_drift_vectors(
        world_model=world_model,
        train_dataset=train_dataset,
        deter_dim=deter_dim,
        horizon=auto_horizon,
        n_batches=auto_n_batches,
        n_rollouts=auto_n_rollouts,
        torch_module=torch_module,
        label="[SP-DPR Calibration]",
    )

    max_k = min(int(auto_k_max), deltas.shape[0], deltas.shape[1])
    if max_k <= 0:
        raise ValueError(f"Cannot calibrate basis with max_k={max_k}, deltas shape={deltas.shape}.")
    pca = PCA(n_components=max_k).fit(deltas)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    chosen_k = int(np.searchsorted(cumvar, float(auto_k_threshold))) + 1
    chosen_k = max(int(auto_k_min), min(int(auto_k_max), chosen_k))
    chosen_k = min(max_k, chosen_k)

    U = pca.components_[:chosen_k]
    Q, _r = np.linalg.qr(U.T)
    Q = Q[:, :chosen_k].astype(np.float32, copy=False)
    validate_basis_array(Q, expected_dim=deter_dim)

    basis_path = logdir / "drift_basis_auto.npy"
    np.save(basis_path, Q)

    variance_explained = float(cumvar[chosen_k - 1])
    drift_erank = effective_rank(pca.singular_values_)
    print(
        f"[SP-DPR Calibration] drift samples={deltas.shape[0]}, "
        f"auto_k={chosen_k}, variance_explained={variance_explained * 100.0:.1f}%, "
        f"drift_erank={drift_erank:.1f}",
        flush=True,
    )
    print(f"[SP-DPR Calibration] saved basis to {basis_path}", flush=True)
    return torch_module.as_tensor(Q, dtype=torch_module.float32, device=device)


def collect_drift_vectors(
    *,
    world_model,
    train_dataset,
    deter_dim: int,
    horizon: int,
    n_batches: int,
    n_rollouts: int,
    torch_module,
    label: str,
) -> np.ndarray:
    """Collect no-grad open-loop deter drift vectors with batch-major indexing."""

    all_deltas: list[np.ndarray] = []
    was_training = world_model.training
    world_model.eval()
    try:
        with torch_module.no_grad():
            for batch_i in range(int(n_batches)):
                raw_data = next(train_dataset)
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
                del B
                if D != deter_dim:
                    raise ValueError(f"Expected deter dim {deter_dim}, got {D}.")
                if T <= horizon:
                    print(
                        f"{label} skipping short batch {batch_i + 1}: "
                        f"T={T} <= horizon={horizon}",
                        flush=True,
                    )
                    continue

                for _ in range(int(n_rollouts)):
                    t_start = int(
                        torch_module.randint(
                            0,
                            T - horizon,
                            (1,),
                            device=post["deter"].device,
                        ).item()
                    )
                    state = state_at_time(post, t_start)
                    for h in range(int(horizon)):
                        action = actions[:, t_start + h].detach()
                        state = world_model.dynamics.img_step(state, action, sample=False)
                        state = detach_state(state)
                    delta = state["deter"] - post["deter"][:, t_start + horizon].detach()
                    all_deltas.append(delta.detach().cpu().numpy())
    finally:
        if was_training:
            world_model.train()

    if not all_deltas:
        raise RuntimeError(
            f"{label} collected no drift vectors. "
            "Lower sp_dpr.auto_horizon or check replay batch length."
        )

    deltas = np.concatenate(all_deltas, axis=0).astype(np.float64, copy=False)
    if deltas.ndim != 2 or deltas.shape[1] != deter_dim:
        raise ValueError(f"Expected drift deltas shape (N, {deter_dim}), got {deltas.shape}.")
    return deltas


def check_escape(
    *,
    world_model,
    train_dataset,
    U_bank,
    deter_dim: int,
    device: str,
    logdir: pathlib.Path,
    k0: int,
    kmax: int,
    auto_horizon: int,
    auto_n_batches: int,
    auto_n_rollouts: int,
    escape_marginal_threshold: float,
    escape_variance_threshold: float,
    torch_module,
):
    """Check for orthogonal drift escape and expand the basis bank if useful."""

    try:
        from sklearn.decomposition import PCA
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "A-BB-SP-DPR escape checking requires scikit-learn for PCA."
        ) from exc

    deltas = collect_drift_vectors(
        world_model=world_model,
        train_dataset=train_dataset,
        deter_dim=deter_dim,
        horizon=auto_horizon,
        n_batches=auto_n_batches,
        n_rollouts=auto_n_rollouts,
        torch_module=torch_module,
        label="[Escape Check]",
    )

    U_np = U_bank.detach().cpu().numpy().astype(np.float64, copy=False)
    validate_basis_array(U_np, expected_dim=deter_dim)
    current_dims = int(U_np.shape[1])
    available = int(kmax) - current_dims

    projected = (deltas @ U_np) @ U_np.T
    delta_orth = deltas - projected
    total_energy = float(np.mean(np.sum(deltas * deltas, axis=1)))
    orth_energy = float(np.mean(np.sum(delta_orth * delta_orth, axis=1)))
    total_norm = math.sqrt(max(total_energy, 0.0))
    orth_norm = math.sqrt(max(orth_energy, 0.0))
    orth_fraction = orth_norm / (total_norm + 1e-8)

    info = {
        "orth_fraction": orth_fraction,
        "marginal_gain": 0.0,
        "added_dims": 0,
        "current_dims": current_dims,
        "kmax": int(kmax),
    }

    if available <= 0:
        print(
            f"[Escape Check] bank full: current={current_dims}d, "
            f"k0={k0}d, Kmax={kmax}d, orth_frac={orth_fraction:.3f}",
            flush=True,
        )
        return U_bank, info
    if orth_energy <= 1e-12 or total_energy <= 1e-12:
        print(
            f"[Escape Check] no significant orthogonal drift: "
            f"orth_frac={orth_fraction:.3f}, bank={current_dims}d",
            flush=True,
        )
        return U_bank, info

    max_components = min(available, delta_orth.shape[0], delta_orth.shape[1])
    if max_components <= 0:
        print(
            f"[Escape Check] no PCA capacity: available={available}, "
            f"samples={delta_orth.shape[0]}, dim={delta_orth.shape[1]}",
            flush=True,
        )
        return U_bank, info

    pca_esc = PCA(n_components=max_components).fit(delta_orth)
    explained = np.nan_to_num(pca_esc.explained_variance_ratio_, nan=0.0)
    cumvar = np.cumsum(explained)
    if cumvar.size == 0 or cumvar[-1] <= 0.0:
        k_esc = 1
    else:
        k_esc = int(np.searchsorted(cumvar, float(escape_variance_threshold))) + 1
    k_esc = max(1, min(int(k_esc), int(available), int(max_components)))

    Q_esc_rows = pca_esc.components_[:k_esc]
    Q_esc = Q_esc_rows.T
    candidate_scores = delta_orth @ Q_esc
    candidate_energy = float(np.mean(np.sum(candidate_scores * candidate_scores, axis=1)))
    marginal_gain = candidate_energy / (total_energy + 1e-8)
    info["marginal_gain"] = marginal_gain

    if marginal_gain < float(escape_marginal_threshold):
        print(
            f"[Escape Check] no expansion: orth_frac={orth_fraction:.3f}, "
            f"marginal={marginal_gain:.4f} < threshold={escape_marginal_threshold:.4f}, "
            f"bank={current_dims}d",
            flush=True,
        )
        return U_bank, info

    combined = np.vstack([U_np.T, Q_esc_rows])
    Q_expanded, _r = np.linalg.qr(combined.T)
    new_dims = current_dims + k_esc
    Q_expanded = Q_expanded[:, :new_dims].astype(np.float32, copy=False)
    validate_basis_array(Q_expanded, expected_dim=deter_dim)

    basis_path = logdir / "drift_basis_bank.npy"
    np.save(basis_path, Q_expanded)
    info["added_dims"] = k_esc
    info["current_dims"] = new_dims
    print(
        f"[Escape Check] orth_frac={orth_fraction:.3f}, "
        f"marginal={marginal_gain:.4f}, added {k_esc}d, "
        f"bank: {current_dims}d -> {new_dims}d, saved {basis_path}",
        flush=True,
    )

    return torch_module.as_tensor(Q_expanded, dtype=torch_module.float32, device=device), info


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
    validate_basis_array(basis_np, expected_dim=deter_dim)
    return torch_module.as_tensor(basis_np, dtype=torch_module.float32, device=device)


def validate_basis_array(basis_np: np.ndarray, *, expected_dim: int) -> None:
    if basis_np.ndim != 2:
        raise ValueError(f"SP-DPR basis must be 2D, got {basis_np.shape}.")
    if basis_np.shape[0] != expected_dim:
        raise ValueError(
            f"SP-DPR basis leading dimension {basis_np.shape[0]} "
            f"does not match config dyn_deter {expected_dim}."
        )
    if basis_np.shape[1] <= 0:
        raise ValueError(f"SP-DPR basis must contain at least one vector, got {basis_np.shape}.")
    gram = basis_np.T @ basis_np
    orth_err = float(np.max(np.abs(gram - np.eye(basis_np.shape[1]))))
    if orth_err > 1e-4:
        raise ValueError(f"SP-DPR basis columns are not orthonormal: max error {orth_err}.")


def assert_batch_major(post, actions) -> None:
    post_deter = post["deter"]
    assert post_deter.ndim == 3, f"Expected post['deter'] [B,T,D], got {post_deter.shape}"
    assert actions.ndim == 3, f"Expected actions [B,T,A], got {actions.shape}"
    B, T, D = post_deter.shape
    del D
    assert B == actions.shape[0], f"Batch mismatch: post {B} vs action {actions.shape[0]}"
    assert T == actions.shape[1], f"Time mismatch: post {T} vs action {actions.shape[1]}"


def effective_rank(singular_values: np.ndarray) -> float:
    values = np.asarray(singular_values, dtype=np.float64)
    values = values[np.isfinite(values) & (values > 0)]
    if values.size == 0:
        return 0.0
    probs = values / values.sum()
    entropy = -float(np.sum(probs * np.log(probs + 1e-12)))
    return float(np.exp(entropy))


def validate_auto_basis_config(
    *,
    auto_k_method: str,
    auto_k_threshold: float,
    auto_k_min: int,
    auto_k_max: int,
    auto_horizon: int,
    auto_n_batches: int,
    auto_n_rollouts: int,
) -> None:
    if auto_k_method != "variance":
        raise ValueError(f"Unsupported sp_dpr.auto_k_method: {auto_k_method!r}.")
    if not 0.0 < auto_k_threshold <= 1.0:
        raise ValueError("sp_dpr.auto_k_threshold must be in (0, 1].")
    if auto_k_min <= 0:
        raise ValueError("sp_dpr.auto_k_min must be positive.")
    if auto_k_max < auto_k_min:
        raise ValueError("sp_dpr.auto_k_max must be >= auto_k_min.")
    if auto_horizon <= 0:
        raise ValueError("sp_dpr.auto_horizon must be positive.")
    if auto_n_batches <= 0:
        raise ValueError("sp_dpr.auto_n_batches must be positive.")
    if auto_n_rollouts <= 0:
        raise ValueError("sp_dpr.auto_n_rollouts must be positive.")


def validate_basis_bank_config(
    *,
    escape_check_step: int,
    escape_marginal_threshold: float,
    escape_k_max_multiplier: float,
    escape_k_max_absolute: int,
    escape_variance_threshold: float,
) -> None:
    if escape_check_step < 0:
        raise ValueError("sp_dpr.escape_check_step must be non-negative.")
    if escape_marginal_threshold < 0.0:
        raise ValueError("sp_dpr.escape_marginal_threshold must be non-negative.")
    if escape_k_max_multiplier < 1.0:
        raise ValueError("sp_dpr.escape_k_max_multiplier must be >= 1.0.")
    if escape_k_max_absolute <= 0:
        raise ValueError("sp_dpr.escape_k_max_absolute must be positive.")
    if not 0.0 < escape_variance_threshold <= 1.0:
        raise ValueError("sp_dpr.escape_variance_threshold must be in (0, 1].")


def validate_loss_config(*, loss_type: str, huber_delta: float) -> None:
    if loss_type not in {"mse", "huber"}:
        raise ValueError("sp_dpr.loss_type must be 'mse' or 'huber'.")
    if huber_delta <= 0.0:
        raise ValueError("sp_dpr.huber_delta must be positive.")


def compute_bank_kmax(
    k0: int,
    *,
    escape_k_max_multiplier: float,
    escape_k_max_absolute: int,
) -> int:
    if k0 <= 0:
        raise ValueError(f"k0 must be positive, got {k0}.")
    return int(min(int(escape_k_max_absolute), math.ceil(float(escape_k_max_multiplier) * k0)))


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
        "dpr_bank_dims": 0.0,
        "dpr_escape_checked": 0.0,
        "dpr_orth_fraction": 0.0,
    }
    for horizon in target_horizons:
        metrics[f"dpr_loss_h{int(horizon)}"] = 0.0
        metrics[f"dpr_weight_h{int(horizon)}"] = 0.0
    return metrics


def print_banner(
    *,
    config: argparse.Namespace,
    basis_path: pathlib.Path | None,
    auto_basis: bool,
    basis_dim: int | None,
    target_horizons: tuple[int, ...],
    beta_dpr: float,
    horizon_weights: str,
    n_starts: int,
    batch_subsample: float,
    apply_every: int,
    loss_cap: float,
    warmup_env_steps: int,
    basis_bank: bool,
    escape_check_step: int,
    escape_k_max_multiplier: float,
    escape_k_max_absolute: int,
    loss_type: str,
    huber_delta: float,
    auto_k_threshold: float,
    auto_k_min: int,
    auto_k_max: int,
) -> None:
    print("=" * 70)
    if basis_bank:
        print("DreamerV3 + A-BB-SP-DPR (Bounded Basis-Bank SP-DPR)")
    elif auto_basis:
        print("DreamerV3 + A-SP-DPR (Adaptive Subspace-Projected Deter Prior Recovery)")
    else:
        print("DreamerV3 + SP-DPR (Subspace-Projected Deter Prior Recovery)")
    print("=" * 70)
    print(f"Task: {config.task}, Seed: {config.seed}")
    if auto_basis:
        if basis_bank:
            print("basis: auto (k0 pending; Kmax set after warmup calibration)")
        else:
            print("basis: auto (calibrated at warmup end)")
        print(
            f"auto_k: variance >= {auto_k_threshold * 100.0:.0f}%, "
            f"k in [{auto_k_min}, {auto_k_max}]"
        )
    else:
        assert basis_path is not None and basis_dim is not None
        if basis_bank:
            kmax = compute_bank_kmax(
                basis_dim,
                escape_k_max_multiplier=escape_k_max_multiplier,
                escape_k_max_absolute=escape_k_max_absolute,
            )
            print(f"basis: {basis_path.name} (k0={basis_dim}, Kmax={kmax})")
        else:
            print(f"basis: {basis_path.name} ({basis_dim} dims)")
    if basis_bank:
        print(f"basis_bank: enabled, escape_check at {escape_check_step}")
    if loss_type == "huber":
        print(f"loss_type: huber (delta={huber_delta})")
    else:
        print("loss_type: mse")
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


def config_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
        raise ValueError(f"Cannot parse boolean config value: {value!r}.")
    return bool(value)


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
