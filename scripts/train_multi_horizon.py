#!/usr/bin/env python3
"""DreamerV3 + Multi-horizon Loss: directly penalize multi-step drift L2 norm.

Adds an auxiliary loss to WorldModel training:
  L_multi = mean over k in [1, H] of ||img_step^k(p_t) - p_{t+k}||^2

This globally penalizes drift in all directions and all frequencies,
forcing RSSM to learn transition functions with less systematic bias.
"""

from __future__ import annotations
import argparse, functools, pathlib, sys, types
from typing import Any

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
    
    mh_cfg = run_config.get("multi_horizon", {})
    H = int(mh_cfg.get("horizon", 5))
    n_starts = int(mh_cfg.get("n_starts", 4))
    beta = float(mh_cfg.get("beta", 0.1))
    warmup_steps = int(mh_cfg.get("warmup_steps", 50000))
    
    print("=" * 70)
    print("DreamerV3 + Multi-Horizon Loss")
    print("=" * 70)
    print(f"Task: {config.task}, Seed: {config.seed}")
    print(f"H={H}, n_starts={n_starts}, beta={beta}, warmup_steps={warmup_steps}")
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
    
    # Patch WorldModel._train to add multi-horizon loss
    original_wm_train = agent._wm._train
    
    def wm_train_with_multi_horizon(wm_self, data):
        # Standard training
        post, context, metrics = original_wm_train(data)
        
        # Multi-horizon drift loss
        if agent._step < warmup_steps:
            metrics["mh_loss"] = 0.0
            metrics["mh_active"] = 0.0
            return post, context, metrics
        
        mh_metrics = compute_multi_horizon_loss(
            wm=agent._wm,
            post=post,
            actions=data["action"],
            H=H,
            n_starts=n_starts,
            beta=beta,
            device=config.device,
        )
        metrics.update(mh_metrics)
        return post, context, metrics
    
    agent._wm._train = types.MethodType(wm_train_with_multi_horizon, agent._wm)
    
    # Load checkpoint
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


def compute_multi_horizon_loss(*, wm, post, actions, H, n_starts, beta, device):
    """Compute multi-horizon drift loss with detach at each step (no chain backprop)."""
    import torch
    
    post_deter = post["deter"]  # (T, B, D)
    if post_deter.ndim != 3:
        return {"mh_loss": 0.0, "mh_active": 0.0}
    
    T, B, D = post_deter.shape
    if T <= H + 1:
        return {"mh_loss": 0.0, "mh_active": 0.0}
    
    if not isinstance(actions, torch.Tensor):
        actions = torch.tensor(actions, device=device)
    
    max_start = T - H - 1
    starts = torch.randint(0, max_start, (n_starts,))
    
    drift_losses = []
    
    # Enable gradients for world model dynamics during this loss
    # (img_step uses dynamics parameters which we want to update)
    wm.dynamics.requires_grad_(True)
    
    for start in starts:
        state = {k: v[start].detach() for k, v in post.items()}
        
        for h in range(H):
            t = int(start) + h
            if t + 1 >= T:
                break
            
            action = actions[t] if actions.ndim == 3 else actions[:, t]
            
            # Single-step img_step with gradients
            next_state = wm.dynamics.img_step(state, action, sample=False)
            target = post_deter[t + 1].detach()
            drift = next_state["deter"] - target
            drift_losses.append(drift.pow(2).mean())
            
            # Detach: no chain backprop
            state = {k: v.detach() for k, v in next_state.items()}
    
    if not drift_losses:
        wm.dynamics.requires_grad_(False)
        return {"mh_loss": 0.0, "mh_active": 0.0}
    
    L_multi = torch.stack(drift_losses).mean()
    weighted = beta * L_multi
    
    # Backward through dynamics only
    wm._model_opt._opt.zero_grad()
    weighted.backward()
    torch.nn.utils.clip_grad_norm_(wm.dynamics.parameters(), max_norm=100.0)
    wm._model_opt._opt.step()
    wm._model_opt._opt.zero_grad()
    
    wm.dynamics.requires_grad_(False)
    
    return {
        "mh_loss": float(L_multi.detach().cpu().item()),
        "mh_active": 1.0,
    }


def recursive_update(base, update):
    for k, v in update.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            recursive_update(base[k], v)
        else:
            base[k] = v


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "month8_multi_horizon.yaml"))
    p.add_argument("--run_name", default="month8_multi_horizon")
    p.add_argument("--device", default=None)
    return p.parse_args()


if __name__ == "__main__":
    main()
