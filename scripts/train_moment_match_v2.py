#!/usr/bin/env python3
"""DreamerV3 + Moment Matching Loss.

Closes both escape routes that mean-only true_bias_loss left open:
  1. Direction redistribution → mean term catches it
  2. Frequency redistribution → no time dim, can't escape  
  3. Variance increase → variance term catches it (NEW)

Loss: L_mm = ||E[epsilon]||^2 + ||Var[epsilon] - Var_target||^2

where Var_target is the natural posterior variance (computed from real data).
We don't want zero variance (some noise is inherent), we want the variance
to match what posterior trajectory naturally has.

Integrated into model_loss BEFORE backward (single optimizer step).
"""

from __future__ import annotations
import argparse, functools, pathlib, sys, types
from typing import Mapping, Any

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
DREAMERV3_ROOT = PROJECT_ROOT / "external" / "dreamerv3-torch"


def main():
    args = parse_args()
    for p in (DREAMERV3_ROOT, PROJECT_ROOT):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    
    import dreamer
    import torch
    
    loader = dreamer.yaml.YAML(typ="safe", pure=True)
    run_config = loader.load(pathlib.Path(args.config).read_text())
    configs = loader.load((DREAMERV3_ROOT / "configs.yaml").read_text())
    config_names = list(run_config.get("dreamer_configs", ["dmc_proprio"]))
    defaults = {}
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
    
    mm_cfg = run_config.get("moment_match", {})
    beta_mean = float(mm_cfg.get("beta_mean", 1.0))
    beta_var = float(mm_cfg.get("beta_var", 0.1))
    n_starts = int(mm_cfg.get("n_starts", 8))
    warmup_env_steps = int(mm_cfg.get("warmup_env_steps", 50000))
    normalize_mode = args.normalize_mode or str(mm_cfg.get("normalize_mode", "raw"))
    if normalize_mode not in {"raw", "per_dim"}:
        raise ValueError(f"Unsupported normalize_mode: {normalize_mode!r}")
    variance_target = (
        float(args.variance_target)
        if args.variance_target is not None
        else float(mm_cfg.get("variance_target", 1.0))
    )
    diagnostic_only = bool(mm_cfg.get("diagnostic_only", False)) or bool(args.diagnostic_only)
    trace_floor = (
        float(args.trace_floor)
        if args.trace_floor is not None
        else float(mm_cfg.get("trace_floor", 0.0))
    )
    beta_trace = (
        float(args.beta_trace)
        if args.beta_trace is not None
        else float(mm_cfg.get("beta_trace", 0.1))
    )
    jacobian_lambda = (
        float(args.jacobian_lambda)
        if args.jacobian_lambda is not None
        else float(mm_cfg.get("jacobian_lambda", 0.0))
    )
    jacobian_n_projections = int(mm_cfg.get("jacobian_n_projections", 1))
    
    print("=" * 70)
    print("DreamerV3 + SHARP-v2 (transition-only)")
    print("=" * 70)
    print(f"Task: {config.task}, Seed: {config.seed}")
    print(f"beta_mean={beta_mean}, beta_var={beta_var}")
    print(f"n_starts={n_starts}, warmup_env_steps={warmup_env_steps}")
    print(f"normalize_mode={normalize_mode}, variance_target={variance_target}")
    print(f"trace_floor={trace_floor}, beta_trace={beta_trace}")
    print(
        f"jacobian_lambda={jacobian_lambda}, "
        f"jacobian_n_projections={jacobian_n_projections}"
    )
    if diagnostic_only:
        print("MODE: DIAGNOSTIC ONLY (no SHARP gradients)")
    if normalize_mode == "raw":
        print("Loss: ||E[eps]||^2 + ||Var[eps] - 0||^2")
    else:
        print("Loss: ||E[eps/sigma]||^2 + ||Var[eps/sigma] - tau||^2")
    print("Closes mean (true_bias) and variance escape routes")
    print("=" * 70, flush=True)
    
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
            ), 1)
    
    def random_agent(_o, _d, _s):
        a = random_actor.sample()
        return {"action": a, "logprob": random_actor.log_prob(a)}, None
    
    state = tools.simulate(random_agent, train_envs, train_eps, config.traindir,
                           logger, limit=config.dataset_size, steps=prefill)
    logger.step += prefill * config.action_repeat
    
    train_dataset = dreamer.make_dataset(train_eps, config)
    eval_dataset = dreamer.make_dataset(eval_eps, config)
    agent = dreamer.Dreamer(
        train_envs[0].observation_space, train_envs[0].action_space,
        config, logger, train_dataset).to(config.device)
    agent.requires_grad_(requires_grad=False)
    
    def moment_match_aware_train(world_model, data):
        env_step = agent._step * config.action_repeat
        active = env_step >= warmup_env_steps
        return world_model_train_with_moment_match(
            world_model, data,
            beta_mean=beta_mean if active else 0.0,
            beta_var=beta_var if active else 0.0,
            n_starts=n_starts,
            normalize_mode=normalize_mode,
            variance_target=variance_target,
            diagnostic_only=diagnostic_only,
            trace_floor=trace_floor,
            beta_trace=beta_trace,
            jacobian_lambda=jacobian_lambda if active else 0.0,
            jacobian_n_projections=jacobian_n_projections,
            tools_module=tools,
            torch_module=torch,
        )
    
    agent._wm._train = types.MethodType(moment_match_aware_train, agent._wm)
    
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
                tools.simulate(eval_policy, eval_envs, eval_eps, config.evaldir,
                              logger, is_eval=True, episodes=config.eval_episode_num)
                next_eval += config.eval_every
            
            state = tools.simulate(agent, train_envs, train_eps, config.traindir,
                                  logger, limit=config.dataset_size,
                                  steps=config.eval_every, state=state)
            
            torch.save({
                "agent_state_dict": agent.state_dict(),
                "optims_state_dict": tools.recursively_collect_optim_state_dict(agent),
            }, latest_path)
            print(f"Saved checkpoint at step {config.action_repeat * agent._step}.", flush=True)
    finally:
        for env in train_envs + eval_envs:
            try:
                env.close()
            except Exception:
                pass


def world_model_train_with_moment_match(
    world_model, data, *,
    beta_mean: float, beta_var: float, n_starts: int,
    normalize_mode: str, variance_target: float, diagnostic_only: bool,
    trace_floor: float, beta_trace: float,
    jacobian_lambda: float, jacobian_n_projections: int,
    tools_module, torch_module,
):
    """Mirror DreamerV3 WorldModel._train and add moment matching loss."""
    
    data = world_model.preprocess(data)
    mm_metrics = zero_moment_match_metrics()
    mm_metrics["mm_active"] = 0.0
    
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
            
            # ===== Moment Matching Loss =====
            if diagnostic_only or beta_mean > 0 or beta_var > 0:
                mm_loss, mm_metrics = compute_moment_match_loss(
                    world_model=world_model,
                    post=post,
                    actions=data["action"],
                    n_starts=n_starts,
                    beta_mean=beta_mean,
                    beta_var=beta_var,
                    normalize_mode=normalize_mode,
                    variance_target=variance_target,
                    diagnostic_only=diagnostic_only,
                    torch_module=torch_module,
                )
                total_model_loss = total_model_loss + mm_loss
                mm_metrics["mm_active"] = 1.0
            # ================================

            # ===== Trace Floor Loss =====
            if not diagnostic_only and trace_floor > 0:
                post_deter_live = post["deter"]
                if post_deter_live.ndim == 3:
                    current_trace = post_deter_live.var(dim=1).sum(dim=-1).mean()
                    trace_loss = torch_module.relu(trace_floor - current_trace).pow(2).mean()
                    weighted_trace_loss = float(beta_trace) * trace_loss
                    total_model_loss = total_model_loss + weighted_trace_loss
                    mm_metrics.update({
                        "mm_trace_loss": float(trace_loss.detach().cpu().item()),
                        "mm_trace_weighted_loss": float(
                            weighted_trace_loss.detach().cpu().item()
                        ),
                        "mm_current_trace": float(current_trace.detach().cpu().item()),
                        "mm_trace_floor": float(trace_floor),
                        "mm_beta_trace": float(beta_trace),
                    })
            # ============================

            # ===== Jacobian Regularization =====
            if not diagnostic_only and jacobian_lambda > 0:
                from src.regularization.jacobian_reg import jacobian_reg_loss

                jac_loss, _ = jacobian_reg_loss(
                    world_model.dynamics,
                    post["stoch"].transpose(0, 1),
                    post["deter"].transpose(0, 1),
                    data["action"].transpose(0, 1),
                    n_starts=n_starts,
                    n_projections=jacobian_n_projections,
                )
                weighted_jac_loss = float(jacobian_lambda) * jac_loss
                total_model_loss = total_model_loss + weighted_jac_loss
                mm_metrics.update({
                    "mm_jacobian_loss": float(jac_loss.detach().cpu().item()),
                    "mm_jacobian_weighted_loss": float(
                        weighted_jac_loss.detach().cpu().item()
                    ),
                    "mm_jacobian_lambda": float(jacobian_lambda),
                    "mm_jacobian_n_projections": float(jacobian_n_projections),
                })
            # ===================================
        
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
            embed=embed, feat=world_model.dynamics.get_feat(post),
            kl=kl_value, postent=world_model.dynamics.get_dist(post).entropy(),
        )
    
    metrics.update(mm_metrics)
    post = {k: v.detach() for k, v in post.items()}
    return post, context, metrics


def zero_moment_match_metrics():
    return {
        "mm_mean_loss": 0.0,
        "mm_var_loss": 0.0,
        "mm_noise_var": 0.0,
        "mm_norm_scale": 0.0,
        "mm_sigma_mean": 0.0,
        "mm_sigma_min": 0.0,
        "mm_sigma_p05": 0.0,
        "mm_sigma_p50": 0.0,
        "mm_sigma_p95": 0.0,
        "mm_sigma_max": 0.0,
        "mm_sigma_below_01": 0,
        "mm_eps_raw_mean": 0.0,
        "mm_eps_raw_std": 0.0,
        "mm_eps_raw_max": 0.0,
        "mm_eps_norm_mean": 0.0,
        "mm_eps_norm_std": 0.0,
        "mm_eps_norm_max": 0.0,
        "mm_trace_loss": 0.0,
        "mm_trace_weighted_loss": 0.0,
        "mm_current_trace": 0.0,
        "mm_trace_floor": 0.0,
        "mm_beta_trace": 0.0,
        "mm_jacobian_loss": 0.0,
        "mm_jacobian_weighted_loss": 0.0,
        "mm_jacobian_lambda": 0.0,
        "mm_jacobian_n_projections": 0.0,
    }


def compute_moment_match_loss(
    *,
    world_model,
    post,
    actions,
    n_starts,
    beta_mean,
    beta_var,
    normalize_mode,
    variance_target,
    torch_module,
    diagnostic_only=False,
):
    """L = beta_mean * ||E[eps]||^2 + beta_var * ||Var[eps]||^2"""

    if diagnostic_only:
        post_deter = post["deter"]
        with torch_module.no_grad():
            _, metrics = _compute_moment_match_loss_impl(
                world_model=world_model,
                post=post,
                actions=actions,
                n_starts=n_starts,
                beta_mean=beta_mean,
                beta_var=beta_var,
                normalize_mode=normalize_mode,
                variance_target=variance_target,
                torch_module=torch_module,
            )
        return post_deter.new_tensor(0.0), metrics

    return _compute_moment_match_loss_impl(
        world_model=world_model,
        post=post,
        actions=actions,
        n_starts=n_starts,
        beta_mean=beta_mean,
        beta_var=beta_var,
        normalize_mode=normalize_mode,
        variance_target=variance_target,
        torch_module=torch_module,
    )


def _compute_moment_match_loss_impl(
    *,
    world_model,
    post,
    actions,
    n_starts,
    beta_mean,
    beta_var,
    normalize_mode,
    variance_target,
    torch_module,
):
    """L = beta_mean * ||E[eps]||^2 + beta_var * ||Var[eps]||^2"""

    post_deter = post["deter"]
    if post_deter.ndim != 3:
        return torch_module.tensor(0.0, device=post_deter.device), zero_moment_match_metrics()
    
    T, B, D = post_deter.shape
    if T < 2:
        return torch_module.tensor(0.0, device=post_deter.device), zero_moment_match_metrics()
    if normalize_mode not in {"raw", "per_dim"}:
        raise ValueError(f"Unsupported normalize_mode: {normalize_mode!r}")
    
    if not isinstance(actions, torch_module.Tensor):
        actions = torch_module.tensor(actions, device=post_deter.device)
    
    starts = torch_module.randint(0, T - 1, (n_starts,))
    
    epsilons = []
    raw_epsilons = []
    norm_epsilons = []
    sigmas = []
    for start in starts:
        t = int(start)
        # SHARP-v2 transition-only boundary:
        # v1 already detached the deterministic target post["deter"][t + 1],
        # but fed live posterior inputs post["stoch"][t], post["deter"][t],
        # and any other posterior state keys into img_step(). Detach every
        # posterior-derived state tensor here so SHARP gradients update only
        # dynamics.img_step() transition parameters. Detach replay actions as a
        # no-op guard; they are not posterior-derived and normally require no grad.
        state = {k: v[t].detach() for k, v in post.items()}
        action = actions[t] if actions.ndim == 3 else actions[:, t]
        action = action.detach()
        
        next_state = world_model.dynamics.img_step(state, action, sample=False)
        target = post_deter[t + 1].detach()
        epsilon_raw = next_state["deter"] - target
        epsilon = epsilon_raw
        if normalize_mode == "per_dim":
            sigma = target.std(dim=0).clamp(min=1e-6).detach()
            if not getattr(compute_moment_match_loss, "_printed_sigma_debug", False):
                sigma_debug = sigma.float()
                print(
                    "DEBUG sigma stats: "
                    f"mean={sigma_debug.mean().item():.6f} "
                    f"min={sigma_debug.min().item():.6f} "
                    f"max={sigma_debug.max().item():.6f}",
                    flush=True,
                )
                compute_moment_match_loss._printed_sigma_debug = True
            epsilon = epsilon / sigma
            sigmas.append(sigma)
            raw_epsilons.append(epsilon_raw)
            norm_epsilons.append(epsilon)
        epsilons.append(epsilon)
    
    all_epsilon = torch_module.cat(epsilons, dim=0)  # (n_starts*B, D)
    
    # Mean term: ||E[eps]||^2
    bias_estimate = all_epsilon.mean(dim=0)
    
    # Variance term: raw mode preserves Run A's zero target exactly; per-dim
    # mode uses a calibrated nonzero target in normalized residual units.
    eps_var = all_epsilon.var(dim=0)  # (D,)
    if normalize_mode == "raw":
        # .sum() matches Run A raw SHARP; Run B per_dim mode was tested with
        # .mean() to control loss scale.
        L_mean = bias_estimate.pow(2).sum()
        L_var = eps_var.pow(2).sum()
        norm_scale = all_epsilon.new_tensor(0.0)
    else:
        L_mean = bias_estimate.pow(2).mean()
        L_var = (eps_var - variance_target).pow(2).mean()
        sigma_all = torch_module.stack(sigmas, dim=0)
        sigma_flat = sigma_all.float().reshape(-1)
        raw_epsilon_all = torch_module.cat(raw_epsilons, dim=0)
        norm_epsilon_all = torch_module.cat(norm_epsilons, dim=0)
        raw_epsilon_diag = raw_epsilon_all.detach()
        norm_epsilon_diag = norm_epsilon_all.detach()
        norm_scale = sigma_flat.mean()
    
    # Combined
    L_total = beta_mean * L_mean + beta_var * L_var
    
    metrics = zero_moment_match_metrics()
    metrics.update({
        "mm_mean_loss": float(L_mean.detach().cpu().item()),
        "mm_var_loss": float(L_var.detach().cpu().item()),
        "mm_noise_var": float(eps_var.sum().detach().cpu().item()),
        "mm_norm_scale": float(norm_scale.detach().cpu().item()),
    })
    if normalize_mode == "per_dim":
        metrics.update({
            "mm_sigma_mean": float(sigma_flat.mean().detach().cpu().item()),
            "mm_sigma_min": float(sigma_flat.min().detach().cpu().item()),
            "mm_sigma_p05": float(sigma_flat.quantile(0.05).detach().cpu().item()),
            "mm_sigma_p50": float(sigma_flat.quantile(0.50).detach().cpu().item()),
            "mm_sigma_p95": float(sigma_flat.quantile(0.95).detach().cpu().item()),
            "mm_sigma_max": float(sigma_flat.max().detach().cpu().item()),
            "mm_sigma_below_01": int((sigma_flat < 0.1).sum().detach().cpu().item()),
            "mm_eps_raw_mean": float(raw_epsilon_diag.abs().mean().cpu().item()),
            "mm_eps_raw_std": float(raw_epsilon_diag.std().cpu().item()),
            "mm_eps_raw_max": float(raw_epsilon_diag.abs().max().cpu().item()),
            "mm_eps_norm_mean": float(norm_epsilon_diag.abs().mean().cpu().item()),
            "mm_eps_norm_std": float(norm_epsilon_diag.std().cpu().item()),
            "mm_eps_norm_max": float(norm_epsilon_diag.abs().max().cpu().item()),
        })
    return L_total, metrics


def torch_to_np(x):
    return x.detach().cpu().numpy()


def recursive_update(base, update):
    for k, v in update.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            recursive_update(base[k], v)
        else:
            base[k] = v


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "month8_moment_match.yaml"))
    p.add_argument("--run_name", default="month8_moment_match")
    p.add_argument("--device", default=None)
    p.add_argument("--normalize_mode", choices=("raw", "per_dim"), default=None)
    p.add_argument("--variance_target", type=float, default=None)
    p.add_argument("--diagnostic_only", action="store_true")
    p.add_argument("--trace_floor", type=float, default=None)
    p.add_argument("--beta_trace", type=float, default=None)
    p.add_argument("--jacobian_lambda", type=float, default=None)
    return p.parse_args()


if __name__ == "__main__":
    main()
