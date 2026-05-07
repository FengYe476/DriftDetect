#!/usr/bin/env python3
"""DreamerV3 + DriftHead v2: non-recurrent correction in actor imagination.

Changes from v1:
- DriftHead still trained independently (no gradient through actor)
- Actor imagination uses corrected feat (non-recurrent)
- img_step always uses original state (no OOD)
"""

from __future__ import annotations
import argparse, functools, pathlib, sys, types, json, math
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
    dreamer_overrides = run_config.get("dreamer", {})
    if dreamer_overrides:
        recursive_update(defaults, dreamer_overrides)
    
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
    
    print("=" * 70)
    print("DreamerV3 + DriftHead v2 (Non-Recurrent Actor Correction)")
    print("=" * 70)
    print(f"Task: {config.task}")
    print(f"Steps: {config.steps} (env steps: {config.steps * config.action_repeat})")
    print(f"Device: {config.device}")
    print(f"Seed: {config.seed}")
    print("=" * 70, flush=True)
    
    step = dreamer.count_steps(config.traindir)
    logger = tools.Logger(logdir, config.action_repeat * step)
    
    print("Create envs.", flush=True)
    train_eps = tools.load_episodes(config.traindir, limit=config.dataset_size)
    eval_eps = tools.load_episodes(config.evaldir, limit=1)
    make = lambda mode, env_id: dreamer.make_env(config, mode, env_id)
    train_envs = [dreamer.Damy(make("train", i)) for i in range(config.envs)]
    eval_envs = [dreamer.Damy(make("eval", i)) for i in range(config.envs)]
    acts = train_envs[0].action_space
    config.num_actions = acts.n if hasattr(acts, "n") else acts.shape[0]
    
    # Prefill
    prefill = max(0, config.prefill - dreamer.count_steps(config.traindir))
    print(f"Prefill dataset ({prefill} steps).", flush=True)
    if hasattr(acts, "discrete"):
        random_actor = tools.OneHotDist(torch.zeros(config.num_actions).repeat(config.envs, 1))
    else:
        random_actor = dreamer.torchd.independent.Independent(
            dreamer.torchd.uniform.Uniform(
                torch.tensor(acts.low).repeat(config.envs, 1),
                torch.tensor(acts.high).repeat(config.envs, 1),
            ), 1,
        )
    def random_agent(_o, _d, _s):
        a = random_actor.sample()
        return {"action": a, "logprob": random_actor.log_prob(a)}, None
    
    state = tools.simulate(random_agent, train_envs, train_eps, config.traindir,
                           logger, limit=config.dataset_size, steps=prefill)
    logger.step += prefill * config.action_repeat
    
    print("Simulate agent.", flush=True)
    train_dataset = dreamer.make_dataset(train_eps, config)
    eval_dataset = dreamer.make_dataset(eval_eps, config)
    agent = dreamer.Dreamer(
        train_envs[0].observation_space, train_envs[0].action_space,
        config, logger, train_dataset,
    ).to(config.device)
    agent.requires_grad_(requires_grad=False)
    
    # Create DriftHead
    from src.smad.correction_head import CorrectionHead
    dh_cfg = run_config.get("drift_head", {})
    drift_head = CorrectionHead(
        deter_dim=int(dh_cfg.get("deter_dim", 512)),
        step_embed_dim=int(dh_cfg.get("step_embed_dim", 32)),
        hidden_dim=int(dh_cfg.get("hidden_dim", 256)),
        max_steps=int(dh_cfg.get("max_steps", 200)),
        init_scale=float(dh_cfg.get("init_scale", 0.01)),
    ).to(config.device)
    
    drift_head_lr = float(dh_cfg.get("lr", 1e-4))
    drift_head_optimizer = torch.optim.Adam(drift_head.parameters(), lr=drift_head_lr)
    train_horizon = int(dh_cfg.get("train_horizon", 15))
    
    with torch.no_grad():
        drift_head.gate.fill_(1.0)
    
    print(f"DriftHead: params={sum(p.numel() for p in drift_head.parameters())}, "
          f"lr={drift_head_lr}, train_horizon={train_horizon}", flush=True)
    
    # === Patch 1: DriftHead training in WorldModel._train ===
    original_wm_train = agent._wm._train
    
    def wm_train_with_drift_head(wm_self, data):
        post, context, metrics = original_wm_train(data)
        drift_metrics = train_drift_head_step(
            dynamics=agent._wm.dynamics,
            post=post,
            actions=data["action"],
            drift_head=drift_head,
            optimizer=drift_head_optimizer,
            horizon=train_horizon,
            device=config.device,
        )
        metrics.update(drift_metrics)
        return post, context, metrics
    
    agent._wm._train = types.MethodType(wm_train_with_drift_head, agent._wm)
    
    # === Patch 2: Non-recurrent correction in actor imagination ===
    # Patch img_step to also provide corrected feat for actor
    behavior = agent._task_behavior
    original_img_step = agent._wm.dynamics.img_step
    original_get_feat = agent._wm.dynamics.get_feat
    behavior._dh_imagination_active = False
    behavior._dh_step_counter = 0
    behavior._dh_correction_stats = {"mean": 0.0, "max": 0.0}
    
    def scaling(step, ramp_start=3, decay_start=15, decay_end=200):
        if step < ramp_start:
            return step / ramp_start
        elif step <= decay_start:
            return 1.0
        elif step <= decay_end:
            return 1.0 - (step - decay_start) / (decay_end - decay_start)
        else:
            return 0.0
    
    def get_feat_with_correction(state):
        """Non-recurrent: compute feat from corrected deter, but don't modify state."""
        feat = original_get_feat(state)
        if not behavior._dh_imagination_active:
            return feat
        
        step = behavior._dh_step_counter
        s = scaling(step)
        if s > 0:
            with torch.no_grad():
                correction = drift_head(state["deter"], step=step)
                corrected_deter = state["deter"] - s * correction
                # Build corrected state just for feat computation
                corrected_state = dict(state)
                corrected_state["deter"] = corrected_deter
                feat = original_get_feat(corrected_state)
                
                behavior._dh_correction_stats["mean"] = float(correction.abs().mean().cpu())
                behavior._dh_correction_stats["max"] = float(correction.abs().max().cpu())
        
        behavior._dh_step_counter += 1
        return feat
    
    agent._wm.dynamics.get_feat = get_feat_with_correction
    
    # Patch _imagine to activate/deactivate correction
    original_imagine = behavior._imagine
    
    def imagine_with_dh(self, start, policy, horizon):
        self._dh_imagination_active = True
        self._dh_step_counter = 0
        try:
            result = original_imagine(start, policy, horizon)
        finally:
            self._dh_imagination_active = False
        return result
    
    behavior._imagine = types.MethodType(imagine_with_dh, behavior)
    
    # Patch _train to log correction stats
    original_behavior_train = behavior._train
    
    def behavior_train_with_metrics(self, start, objective):
        imag_feat, imag_state, imag_action, weights, metrics = original_behavior_train(start, objective)
        metrics["dh_correction_mean"] = behavior._dh_correction_stats["mean"]
        metrics["dh_correction_max"] = behavior._dh_correction_stats["max"]
        metrics["dh_gate"] = drift_head.gate_value
        return imag_feat, imag_state, imag_action, weights, metrics
    
    behavior._train = types.MethodType(behavior_train_with_metrics, behavior)
    
    # Load checkpoint
    latest_path = logdir / "latest.pt"
    if latest_path.exists():
        ckpt = torch.load(latest_path, map_location=config.device, weights_only=False)
        agent.load_state_dict(ckpt["agent_state_dict"])
        if "drift_head_state_dict" in ckpt:
            drift_head.load_state_dict(ckpt["drift_head_state_dict"])
        if "drift_head_optimizer_state_dict" in ckpt:
            drift_head_optimizer.load_state_dict(ckpt["drift_head_optimizer_state_dict"])
        tools.recursively_load_optim_state_dict(agent, ckpt["optims_state_dict"])
        agent._should_pretrain._once = False
    
    # Training loop
    next_eval = agent._step
    final_step = config.steps + config.eval_every
    
    try:
        while agent._step < final_step:
            logger.write()
            if config.eval_episode_num > 0 and agent._step >= next_eval:
                print("Start evaluation.", flush=True)
                eval_policy = functools.partial(agent, training=False)
                tools.simulate(eval_policy, eval_envs, eval_eps, config.evaldir,
                              logger, is_eval=True, episodes=config.eval_episode_num)
                next_eval += config.eval_every
            
            print("Start training.", flush=True)
            state = tools.simulate(agent, train_envs, train_eps, config.traindir,
                                  logger, limit=config.dataset_size,
                                  steps=config.eval_every, state=state)
            
            torch.save({
                "agent_state_dict": agent.state_dict(),
                "optims_state_dict": tools.recursively_collect_optim_state_dict(agent),
                "drift_head_state_dict": drift_head.state_dict(),
                "drift_head_optimizer_state_dict": drift_head_optimizer.state_dict(),
            }, latest_path)
            print(f"Saved checkpoint at step {config.action_repeat * agent._step}.", flush=True)
    finally:
        for env in train_envs + eval_envs:
            try:
                env.close()
            except Exception:
                pass


def train_drift_head_step(*, dynamics, post, actions, drift_head, optimizer, horizon, device):
    import torch
    
    post_deter = post["deter"].detach()
    if post_deter.ndim != 3:
        return {"drift_head_loss": 0.0, "drift_head_gate": drift_head.gate_value}
    
    T, B, D = post_deter.shape
    if T <= horizon + 1:
        return {"drift_head_loss": 0.0, "drift_head_gate": drift_head.gate_value}
    
    actions_raw = actions
    if not isinstance(actions_raw, torch.Tensor):
        actions_raw = torch.tensor(actions_raw, device=device)
    actions_det = actions_raw.detach()
    
    max_start = T - horizon - 1
    if max_start <= 0:
        return {"drift_head_loss": 0.0, "drift_head_gate": drift_head.gate_value}
    
    start_idx = torch.randint(0, max_start, (1,)).item()
    state = {k: v[start_idx].detach() for k, v in post.items()}
    
    total_loss = torch.tensor(0.0, device=device)
    n_steps = 0
    
    drift_head.requires_grad_(True)
    
    for k in range(horizon):
        t = start_idx + k + 1
        if t >= T:
            break
        
        action = actions_det[t - 1] if actions_det.ndim == 3 else actions_det[:, t - 1]
        
        with torch.no_grad():
            next_state = dynamics.img_step(state, action, sample=False)
        
        imagined_deter = next_state["deter"].detach()
        posterior_deter = post_deter[t]
        true_drift = imagined_deter - posterior_deter
        
        predicted_correction = drift_head(imagined_deter, step=k)
        step_loss = torch.nn.functional.mse_loss(predicted_correction, true_drift.detach())
        total_loss = total_loss + step_loss
        n_steps += 1
        
        state = {key: val.detach() for key, val in next_state.items()}
    
    if n_steps > 0:
        avg_loss = total_loss / n_steps
        optimizer.zero_grad()
        avg_loss.backward()
        torch.nn.utils.clip_grad_norm_(drift_head.parameters(), max_norm=100.0)
        optimizer.step()
        drift_head.requires_grad_(False)
        return {
            "drift_head_loss": float(avg_loss.detach().cpu().item()),
            "drift_head_gate": drift_head.gate_value,
            "drift_head_n_steps": n_steps,
        }
    
    drift_head.requires_grad_(False)
    return {"drift_head_loss": 0.0, "drift_head_gate": drift_head.gate_value}


def recursive_update(base, update):
    for k, v in update.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            recursive_update(base[k], v)
        else:
            base[k] = v


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "month8_drift_head.yaml"))
    p.add_argument("--run_name", default="month8_drift_head_v2")
    p.add_argument("--device", default=None)
    return p.parse_args()


if __name__ == "__main__":
    main()
