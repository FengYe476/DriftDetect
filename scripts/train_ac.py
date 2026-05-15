#!/usr/bin/env python3
"""DreamerV3 + Amplification Control (AC-1).

Adds a guarded transition amplification penalty to the world-model loss:

    L_total = L_DreamerV3 + beta_ac * E_starts[L_AC]

AC perturbs only the deterministic RSSM state and penalizes transition gain
only when it exceeds a calibrated threshold rho.
"""

from __future__ import annotations

import argparse
import functools
import pathlib
import sys
import types
from typing import Any


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
DREAMERV3_ROOT = PROJECT_ROOT / "external" / "dreamerv3-torch"


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

    ac_cfg = run_config.get("amplification_control", run_config.get("moment_match", {}))
    beta_ac = float(ac_cfg.get("beta_ac", 1.0))
    horizons = tuple(int(h) for h in ac_cfg.get("horizons", [1]))
    horizon_weights = str(ac_cfg.get("horizon_weights", "uniform"))
    n_starts = int(ac_cfg.get("n_starts", 4))
    n_dirs = int(ac_cfg.get("n_dirs", 1))
    delta_scale = float(ac_cfg.get("delta_scale", 0.01))
    rho = float(ac_cfg.get("rho", 1.0))
    warmup_env_steps = int(ac_cfg.get("warmup_env_steps", 50000))

    print("=" * 70)
    print("DreamerV3 + Amplification Control (AC-1)")
    print("=" * 70)
    print(f"Task: {config.task}, Seed: {config.seed}")
    print(f"horizons={list(horizons)}, rho={rho:.3f}, beta_ac={beta_ac}, n_dirs={n_dirs}")
    print(f"n_starts={n_starts}, delta_scale={delta_scale}")
    print(f"horizon_weights={horizon_weights}, warmup_env_steps={warmup_env_steps}")
    print("Loss: L_DreamerV3 + beta_ac * ReLU(gain - rho)^2")
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

    def ac_aware_train(world_model, data):
        env_step = agent._step * config.action_repeat
        active = env_step >= warmup_env_steps
        return world_model_train_with_ac(
            world_model,
            data,
            beta_ac=beta_ac if active else 0.0,
            horizons=horizons,
            horizon_weights=horizon_weights,
            n_starts=n_starts,
            n_dirs=n_dirs,
            delta_scale=delta_scale,
            rho=rho,
            tools_module=tools,
            torch_module=torch,
        )

    agent._wm._train = types.MethodType(ac_aware_train, agent._wm)

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


def world_model_train_with_ac(
    world_model,
    data,
    *,
    beta_ac: float,
    horizons: tuple[int, ...],
    horizon_weights: str,
    n_starts: int,
    n_dirs: int,
    delta_scale: float,
    rho: float,
    tools_module,
    torch_module,
):
    """Mirror DreamerV3 WorldModel._train and add AC loss."""

    data = world_model.preprocess(data)
    ac_metrics = zero_ac_metrics(horizons)

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

            if beta_ac > 0:
                ac_loss, ac_metrics = compute_ac_loss(
                    world_model=world_model,
                    post=post,
                    actions=data["action"],
                    horizons=horizons,
                    horizon_weights=horizon_weights,
                    n_starts=n_starts,
                    n_dirs=n_dirs,
                    delta_scale=delta_scale,
                    rho=rho,
                    torch_module=torch_module,
                )
                weighted_ac_loss = float(beta_ac) * ac_loss
                total_model_loss = total_model_loss + weighted_ac_loss
                ac_metrics["ac_loss"] = tensor_to_float(ac_loss)
                ac_metrics["ac_weighted_loss"] = tensor_to_float(weighted_ac_loss)
                ac_metrics["ac_active"] = 1.0
                ac_metrics["ac_beta"] = float(beta_ac)

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

    metrics.update(ac_metrics)
    post = {key: value.detach() for key, value in post.items()}
    return post, context, metrics


def compute_ac_loss(
    *,
    world_model,
    post,
    actions,
    horizons: tuple[int, ...],
    horizon_weights: str,
    n_starts: int,
    n_dirs: int,
    delta_scale: float,
    rho: float,
    torch_module,
):
    """Sample posterior time steps and average AC loss across starts."""

    from src.regularization.amplification_control import amplification_control_loss

    post_deter = post["deter"]
    if not isinstance(actions, torch_module.Tensor):
        actions = torch_module.tensor(actions, device=post_deter.device)

    assert post_deter.ndim == 3, f"Expected 3D tensor, got {post_deter.ndim}D"
    assert actions.ndim == 3, f"Expected 3D action tensor, got {actions.ndim}D"

    # post["deter"] shape: [B, T, D] where B=batch, T=time, D=latent_dim.
    B, T, D = post_deter.shape
    del D
    assert B == actions.shape[0], f"Batch mismatch: post {B} vs action {actions.shape[0]}"
    assert T == actions.shape[1], f"Time mismatch: post {T} vs action {actions.shape[1]}"

    horizons = tuple(int(horizon) for horizon in horizons)
    max_horizon = max(horizons)
    if T < max_horizon:
        zero = post_deter.new_tensor(0.0)
        return zero, zero_ac_metrics(horizons)

    starts = torch_module.randint(
        0,
        T - max_horizon + 1,
        (int(n_starts),),
        device=post_deter.device,
    )

    losses = []
    metric_values: dict[str, list] = {}
    for start in starts:
        t = int(start.item())
        stoch = post["stoch"][:, t].detach()
        deter = post_deter[:, t].detach()
        actions_seq = actions[:, t : t + max_horizon].detach()

        ac_loss, ac_metrics = amplification_control_loss(
            world_model.dynamics,
            stoch,
            deter,
            actions_seq,
            n_dirs=n_dirs,
            delta_scale=delta_scale,
            rho=rho,
            horizons=horizons,
            horizon_weights=horizon_weights,
        )
        losses.append(ac_loss)
        for key, value in ac_metrics.items():
            if key == "ac_loss":
                continue
            metric_values.setdefault(key, []).append(value)

    total_loss = torch_module.stack(losses).mean()
    metrics = zero_ac_metrics(horizons)
    metrics.update({
        key: tensor_to_float(torch_module.stack(values).mean())
        for key, values in metric_values.items()
    })
    metrics["ac_n_starts"] = float(n_starts)
    return total_loss, metrics


def zero_ac_metrics(horizons: tuple[int, ...]) -> dict[str, float]:
    metrics = {
        "ac_loss": 0.0,
        "ac_weighted_loss": 0.0,
        "ac_active": 0.0,
        "ac_beta": 0.0,
        "ac_n_starts": 0.0,
    }
    for horizon in horizons:
        metrics[f"ac_gain_h{horizon}_mean"] = 0.0
        metrics[f"ac_gain_h{horizon}_p90"] = 0.0
        metrics[f"ac_gain_h{horizon}_max"] = 0.0
        metrics[f"ac_loss_h{horizon}"] = 0.0
        metrics[f"ac_weight_h{horizon}"] = 0.0
    return metrics


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "ac1_cheetah.yaml"))
    parser.add_argument("--run_name", default="ac1_cheetah")
    parser.add_argument("--device", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    main()
