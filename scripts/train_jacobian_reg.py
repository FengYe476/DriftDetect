#!/usr/bin/env python3
"""DreamerV3 + Jacobian regularization baseline.

Adds a Hutchinson estimate of the deterministic RSSM transition Jacobian norm
to the world-model loss:

    L_total = L_DreamerV3 + lambda_jac * ||d deter_{t+1} / d deter_t||_F^2

The integration mirrors scripts/train_moment_match.py: patch WorldModel._train
and apply a single optimizer step over the combined model loss.
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
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.regularization.jacobian_reg import jacobian_reg_loss  # noqa: E402


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

    jac_cfg = run_config.get("jacobian_reg", {})
    lambda_jac = float(cli_or_config(args.lambda_jac, jac_cfg.get("lambda_jac", 0.01)))
    n_jac_starts = int(cli_or_config(args.n_jac_starts, jac_cfg.get("n_starts", 8)))
    n_jac_projections = int(
        cli_or_config(args.n_jac_projections, jac_cfg.get("n_projections", 1))
    )
    warmup_env_steps = int(
        cli_or_config(args.warmup_env_steps, jac_cfg.get("warmup_env_steps", 50000))
    )

    print("=" * 70)
    print("DreamerV3 + Jacobian Regularization")
    print("=" * 70)
    print(f"Task: {config.task}, Seed: {config.seed}")
    print(f"lambda_jac={lambda_jac}")
    print(f"n_jac_starts={n_jac_starts}, n_jac_projections={n_jac_projections}")
    print(f"warmup_env_steps={warmup_env_steps}")
    print("Loss: L_DreamerV3 + lambda_jac * ||d deter_{t+1}/d deter_t||_F^2")
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

    def jacobian_reg_aware_train(world_model, data):
        env_step = agent._step * config.action_repeat
        active_lambda = lambda_jac if env_step >= warmup_env_steps else 0.0
        return world_model_train_with_jacobian_reg(
            world_model,
            data,
            lambda_jac=active_lambda,
            n_jac_starts=n_jac_starts,
            n_jac_projections=n_jac_projections,
            tools_module=tools,
            torch_module=torch,
        )

    agent._wm._train = types.MethodType(jacobian_reg_aware_train, agent._wm)

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


def world_model_train_with_jacobian_reg(
    world_model,
    data,
    *,
    lambda_jac: float,
    n_jac_starts: int,
    n_jac_projections: int,
    tools_module,
    torch_module,
):
    """Mirror DreamerV3 WorldModel._train and add Jacobian regularization."""

    data = world_model.preprocess(data)
    jac_metrics = zero_jacobian_metrics()

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

            if lambda_jac > 0.0:
                raw_jac_loss, raw_jac_metrics = jacobian_reg_loss(
                    world_model.dynamics,
                    post["stoch"],
                    post["deter"],
                    data["action"],
                    n_starts=n_jac_starts,
                    n_projections=n_jac_projections,
                )
                weighted_jac_loss = float(lambda_jac) * raw_jac_loss
                total_model_loss = total_model_loss + weighted_jac_loss
                jac_metrics = {
                    "jacobian_reg_active": 1.0,
                    "jacobian_norm_estimate": tensor_to_float(
                        raw_jac_metrics["jacobian_norm_estimate"]
                    ),
                    "jacobian_reg_loss": tensor_to_float(raw_jac_loss),
                    "jacobian_reg_weighted_loss": tensor_to_float(weighted_jac_loss),
                    "jacobian_reg_lambda": float(lambda_jac),
                    "jacobian_reg_n_starts": float(n_jac_starts),
                    "jacobian_reg_n_projections": float(n_jac_projections),
                }

        metrics = world_model._model_opt(total_model_loss, world_model.parameters())

    metrics.update(
        {
            f"{name}_loss": torch_to_np(torch_module.mean(loss))
            for name, loss in losses.items()
        }
    )
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

    metrics.update(jac_metrics)
    post = {key: value.detach() for key, value in post.items()}
    return post, context, metrics


def zero_jacobian_metrics() -> dict[str, float]:
    return {
        "jacobian_reg_active": 0.0,
        "jacobian_norm_estimate": 0.0,
        "jacobian_reg_loss": 0.0,
        "jacobian_reg_weighted_loss": 0.0,
        "jacobian_reg_lambda": 0.0,
        "jacobian_reg_n_starts": 0.0,
        "jacobian_reg_n_projections": 0.0,
    }


def tensor_to_float(value) -> float:
    return float(value.detach().cpu().item())


def torch_to_np(value):
    return value.detach().cpu().numpy()


def cli_or_config(cli_value, config_value):
    return config_value if cli_value is None else cli_value


def recursive_update(base, update) -> None:
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            recursive_update(base[key], value)
        else:
            base[key] = value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "jacobian_reg_cheetah.yaml"),
    )
    parser.add_argument("--run_name", default="jacobian_reg_cheetah")
    parser.add_argument("--device", default=None)
    parser.add_argument("--lambda_jac", type=float, default=None)
    parser.add_argument("--n_jac_starts", type=int, default=None)
    parser.add_argument("--n_jac_projections", type=int, default=None)
    parser.add_argument("--warmup_env_steps", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    main()
