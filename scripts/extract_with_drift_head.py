#!/usr/bin/env python3
"""Extract rollouts with DriftHead inference correction."""
import argparse, sys, pathlib, numpy as np, torch, os

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
DREAMERV3_ROOT = PROJECT_ROOT / "external" / "dreamerv3-torch"
sys.path.insert(0, str(DREAMERV3_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

if "DRIFTDETECT_MUJOCO_GL" not in os.environ and "MUJOCO_GL" not in os.environ:
    os.environ["MUJOCO_GL"] = "osmesa"

from src.smad.correction_head import CorrectionHead

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--drift_head_checkpoint", required=True)
    p.add_argument("--task", default="dmc_cheetah_run")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    import dreamer
    from ruamel.yaml import YAML
    
    loader = YAML(typ="safe", pure=True)
    configs = loader.load((DREAMERV3_ROOT / "configs.yaml").read_text())
    
    defaults = {}
    def recursive_update(base, update):
        for k, v in update.items():
            if isinstance(v, dict) and isinstance(base.get(k), dict):
                recursive_update(base[k], v)
            else:
                base[k] = v
    
    for name in ["defaults", "dmc_proprio"]:
        recursive_update(defaults, configs[name])
    
    defaults["task"] = args.task
    defaults["seed"] = args.seed
    defaults["device"] = "cuda:0" if torch.cuda.is_available() else "cpu"
    defaults["logdir"] = "/tmp/dh_extract"
    config = argparse.Namespace(**defaults)
    config.compile = False
    
    tools = dreamer.tools
    tools.set_seed_everywhere(config.seed)
    
    env = dreamer.make_env(config, "eval", 0)
    env = dreamer.Damy(env)
    acts = env.action_space
    config.num_actions = acts.n if hasattr(acts, "n") else acts.shape[0]
    
    agent = dreamer.Dreamer(
        env.observation_space, env.action_space, config,
        type('L', (), {'scalar':lambda *a,**k:None, 'video':lambda *a,**k:None, 
                       'write':lambda *a,**k:None, 'step':0})(),
        dataset=None,
    ).to(config.device)
    agent.requires_grad_(requires_grad=False)
    
    ckpt = torch.load(args.checkpoint, map_location=config.device, weights_only=False)
    agent.load_state_dict(ckpt["agent_state_dict"])
    
    drift_head = CorrectionHead(deter_dim=512).to(config.device)
    dh_ckpt = torch.load(args.drift_head_checkpoint, map_location=config.device, weights_only=False)
    drift_head.load_state_dict(dh_ckpt["drift_head_state_dict"])
    drift_head.eval()
    drift_head.requires_grad_(False)
    print(f"DriftHead gate: {drift_head.gate_value:.4f}", flush=True)
    
    obs_keys = sorted([k for k in env.observation_space.spaces if k not in ("is_first", "is_terminal", "image", "reward")])
    decoder = agent._wm.heads["decoder"]
    dynamics = agent._wm.dynamics
    
    obs = env.reset()
    if callable(obs):
        obs = obs()
    agent_state = None
    done = np.array([False])
    
    true_obs_list, imagined_obs_list = [], []
    actions_list, rewards_list = [], []
    true_latent_list, imagined_latent_list = [], []
    
    flatten_obs = lambda o: np.concatenate([np.array(o[k]).flatten() for k in obs_keys if k in o and k not in ("is_first", "is_terminal", "image", "reward")])
    
    imag_latent = None
    latent = None
    imagination_start = 50
    horizon = 200
    total_steps = imagination_start + horizon
    corr_step = 0
    
    for step in range(total_steps):
        obs_t = {k: torch.tensor(np.array(v)[None], device=config.device, dtype=torch.float32) 
                 for k, v in obs.items()}
        done_t = torch.tensor(done[None], device=config.device, dtype=torch.float32)
        
        with torch.no_grad():
            action_out, agent_state = agent(obs_t, done_t, agent_state, training=False)
        
        latent, _ = agent_state
        env_action = {k: v.cpu().numpy()[0] for k, v in action_out.items()}
        action_tensor = action_out["action"]
        
        in_window = imagination_start <= step < imagination_start + horizon
        
        if step == imagination_start:
            imag_latent = {k: v.detach().clone() for k, v in latent.items()}
            corr_step = 0
        
        if in_window:
            with torch.no_grad():
                raw_next = dynamics.img_step(imag_latent, action_tensor, sample=False)
                correction = drift_head(raw_next["deter"], step=corr_step)
                corrected = dict(raw_next)
                corrected["deter"] = raw_next["deter"] - correction
                imag_latent = corrected
                corr_step += 1
                
                feat = dynamics.get_feat(imag_latent)
                decoded = decoder(feat)
                imag_obs = np.concatenate([decoded[k].mode().cpu().numpy().flatten() for k in obs_keys])
                
                true_feat = dynamics.get_feat(latent)
                true_latent_list.append(true_feat.cpu().numpy().squeeze())
                imagined_latent_list.append(feat.cpu().numpy().squeeze())
        
        step_result = env.step(env_action)
        if callable(step_result):
            step_result = step_result()
        next_obs, reward, done_bool, _ = step_result
        
        if in_window:
            true_obs_list.append(flatten_obs(next_obs))
            imagined_obs_list.append(imag_obs)
            actions_list.append(env_action["action"])
            rewards_list.append(float(reward))
        
        obs = next_obs
        done = np.array([done_bool])
    
    
    out_path = pathlib.Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path,
        true_obs=np.array(true_obs_list, dtype=np.float32),
        imagined_obs=np.array(imagined_obs_list, dtype=np.float32),
        actions=np.array(actions_list, dtype=np.float32),
        rewards=np.array(rewards_list, dtype=np.float32),
        true_latent=np.array(true_latent_list, dtype=np.float32),
        imagined_latent=np.array(imagined_latent_list, dtype=np.float32),
    )
    print(f"Saved: {out_path} ({len(true_obs_list)} steps)", flush=True)
    try:
        env.close()
    except Exception:
        pass

if __name__ == "__main__":
    main()
