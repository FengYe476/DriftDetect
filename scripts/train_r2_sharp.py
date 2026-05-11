#!/usr/bin/env python3
"""R2-Dreamer + SHARP training entrypoint.

This script keeps R2-Dreamer's trainer, optimizer, and replay buffer intact,
but replaces Dreamer._cal_grad with a local copy that adds SHARP before the
single backward pass.
"""

from __future__ import annotations

import argparse
import atexit
import os
import pathlib
import sys
from types import SimpleNamespace
from typing import Any


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
R2_ROOT = PROJECT_ROOT / "external" / "r2dreamer"


def add_import_paths() -> None:
    for path in (R2_ROOT, PROJECT_ROOT):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


def patch_r2_dmc_size() -> None:
    """Coerce Hydra list sizes to tuples before R2's DMC env stores them."""

    try:
        import envs.dmc as dmc
    except Exception:
        return

    if getattr(dmc.DeepMindControl, "_driftdetect_size_patch", False):
        return

    original_init = dmc.DeepMindControl.__init__

    def patched_init(self, name, action_repeat=1, size=(64, 64), camera=None, seed=0):
        return original_init(
            self,
            name,
            action_repeat=action_repeat,
            size=tuple(size),
            camera=camera,
            seed=seed,
        )

    dmc.DeepMindControl.__init__ = patched_init
    dmc.DeepMindControl._driftdetect_size_patch = True


def compute_r2_sharp_loss(
    agent: Any,
    post_stoch,
    post_deter,
    actions,
    *,
    beta_mean: float,
    beta_var: float,
    n_starts: int,
):
    """SHARP on R2's batch-major deterministic posterior state.

    R2 replay aligns data["action"][:, t] as the action that produced
    posterior state t, so the open-loop transition post_t -> post_{t+1}
    uses data["action"][:, t + 1].
    """

    import torch

    if post_deter.ndim != 3:
        zero = post_deter.new_zeros(())
        return zero, {
            "sharp_mean_loss": zero,
            "sharp_var_loss": zero,
            "sharp_noise_var": zero,
        }

    batch, time, _dim = post_deter.shape
    del batch
    if time < 2:
        zero = post_deter.new_zeros(())
        return zero, {
            "sharp_mean_loss": zero,
            "sharp_var_loss": zero,
            "sharp_noise_var": zero,
        }

    starts = torch.randint(
        0,
        time - 1,
        (max(1, int(n_starts)),),
        device=post_deter.device,
    )

    epsilons = []
    for start in starts.detach().cpu().tolist():
        t = int(start)
        _next_stoch, next_deter = agent.rssm.img_step(
            post_stoch[:, t],
            post_deter[:, t],
            actions[:, t + 1],
        )
        epsilon = next_deter.float() - post_deter[:, t + 1].detach().float()
        epsilons.append(epsilon)

    all_epsilon = torch.cat(epsilons, dim=0)
    bias = all_epsilon.mean(dim=0)
    mean_loss = bias.pow(2).sum()
    eps_var = all_epsilon.var(dim=0, unbiased=False)
    var_loss = eps_var.pow(2).sum()
    total = beta_mean * mean_loss + beta_var * var_loss
    return total, {
        "sharp_mean_loss": mean_loss.detach(),
        "sharp_var_loss": var_loss.detach(),
        "sharp_noise_var": eps_var.sum().detach(),
    }


def r2_cal_grad_with_sharp(self, data, initial):
    """R2 Dreamer._cal_grad with SHARP inserted before backward()."""

    import torch
    import tools
    from tools import to_f32

    losses = {}
    metrics = {}
    batch, time = data.shape

    embed = self.encoder(data)
    post_stoch, post_deter, post_logit = self.rssm.observe(
        embed,
        data["action"],
        initial,
        data["is_first"],
    )
    _, prior_logit = self.rssm.prior(post_deter)
    dyn_loss, rep_loss = self.rssm.kl_loss(post_logit, prior_logit, self.kl_free)
    losses["dyn"] = torch.mean(dyn_loss)
    losses["rep"] = torch.mean(rep_loss)

    feat = self.rssm.get_feat(post_stoch, post_deter)
    if self.rep_loss == "dreamer":
        recon_losses = {
            key: torch.mean(-dist.log_prob(data[key]))
            for key, dist in self.decoder(post_stoch, post_deter).items()
        }
        losses.update(recon_losses)
    elif self.rep_loss == "r2dreamer":
        x1 = self.prj(feat[:, :].reshape(batch * time, -1))
        x2 = embed.reshape(batch * time, -1).detach()
        x1_norm = (x1 - x1.mean(0)) / (x1.std(0) + 1e-8)
        x2_norm = (x2 - x2.mean(0)) / (x2.std(0) + 1e-8)
        c = torch.mm(x1_norm.T, x2_norm) / (batch * time)
        invariance_loss = (torch.diagonal(c) - 1.0).pow(2).sum()
        off_diag_mask = ~torch.eye(x1.shape[-1], dtype=torch.bool, device=x1.device)
        redundancy_loss = c[off_diag_mask].pow(2).sum()
        losses["barlow"] = invariance_loss + self.barlow_lambd * redundancy_loss
    elif self.rep_loss == "infonce":
        x1 = self.prj(feat[:, :].reshape(batch * time, -1))
        x2 = embed.reshape(batch * time, -1).detach()
        logits = torch.matmul(x1, x2.T)
        norm_logits = logits - torch.max(logits, 1)[0][:, None]
        labels = torch.arange(norm_logits.shape[0]).long().to(self.device)
        losses["infonce"] = torch.nn.functional.cross_entropy(norm_logits, labels)
    elif self.rep_loss == "dreamerpro":
        with torch.no_grad():
            data_aug = self.augment_data(data)
            initial_aug = (
                torch.cat([initial[0], initial[0]], dim=0),
                torch.cat([initial[1], initial[1]], dim=0),
            )
            ema_proj = self.ema_proj(data_aug)

        embed_aug = self.encoder(data_aug)
        post_stoch_aug, post_deter_aug, _ = self.rssm.observe(
            embed_aug,
            data_aug["action"],
            initial_aug,
            data_aug["is_first"],
        )
        proto_losses = self.proto_loss(post_stoch_aug, post_deter_aug, embed_aug, ema_proj)
        losses.update(proto_losses)
    else:
        raise NotImplementedError(self.rep_loss)

    losses["rew"] = torch.mean(-self.reward(feat).log_prob(to_f32(data["reward"])))
    cont = 1.0 - to_f32(data["is_terminal"])
    losses["con"] = torch.mean(-self.cont(feat).log_prob(cont))
    metrics["dyn_entropy"] = torch.mean(self.rssm.get_dist(prior_logit).entropy())
    metrics["rep_entropy"] = torch.mean(self.rssm.get_dist(post_logit).entropy())

    sharp_cfg = getattr(self, "_sharp_cfg", None)
    zero = post_deter.new_zeros(())
    sharp_loss = zero
    sharp_metrics = {
        "sharp_mean_loss": zero,
        "sharp_var_loss": zero,
        "sharp_noise_var": zero,
    }
    sharp_active = 0.0
    if sharp_cfg is not None:
        env_step = int(getattr(self, "_sharp_env_step", 0))
        if (
            env_step >= sharp_cfg.warmup_env_steps
            and (sharp_cfg.beta_mean > 0.0 or sharp_cfg.beta_var > 0.0)
        ):
            sharp_loss, sharp_metrics = compute_r2_sharp_loss(
                self,
                post_stoch,
                post_deter,
                data["action"],
                beta_mean=sharp_cfg.beta_mean,
                beta_var=sharp_cfg.beta_var,
                n_starts=sharp_cfg.n_starts,
            )
            sharp_active = 1.0
        metrics["sharp_active"] = sharp_active
        metrics["sharp_env_step"] = float(env_step)
        metrics.update(sharp_metrics)

    start = (
        post_stoch.reshape(-1, *post_stoch.shape[2:]).detach(),
        post_deter.reshape(-1, *post_deter.shape[2:]).detach(),
    )
    imag_feat, imag_action = self._imagine(start, self.imag_horizon + 1)
    imag_feat, imag_action = imag_feat.detach(), imag_action.detach()

    imag_reward = self._frozen_reward(imag_feat).mode()
    imag_cont = self._frozen_cont(imag_feat).mean
    imag_value = self._frozen_value(imag_feat).mode()
    imag_slow_value = self._frozen_slow_value(imag_feat).mode()
    disc = 1 - 1 / self.horizon
    weight = torch.cumprod(imag_cont * disc, dim=1)
    last = torch.zeros_like(imag_cont)
    term = 1 - imag_cont
    ret = self._lambda_return(last, term, imag_reward, imag_value, imag_value, disc, self.lamb)
    ret_offset, ret_scale = self.return_ema(ret)
    adv = (ret - imag_value[:, :-1]) / ret_scale

    policy = self.actor(imag_feat)
    logpi = policy.log_prob(imag_action)[:, :-1].unsqueeze(-1)
    entropy = policy.entropy()[:, :-1].unsqueeze(-1)
    losses["policy"] = torch.mean(
        weight[:, :-1].detach() * -(logpi * adv.detach() + self.act_entropy * entropy)
    )

    imag_value_dist = self.value(imag_feat)
    tar_padded = torch.cat([ret, 0 * ret[:, -1:]], 1)
    losses["value"] = torch.mean(
        weight[:, :-1].detach()
        * (
            -imag_value_dist.log_prob(tar_padded.detach())
            - imag_value_dist.log_prob(imag_slow_value.detach())
        )[:, :-1].unsqueeze(-1)
    )
    ret_normed = (ret - ret_offset) / ret_scale
    metrics["ret"] = torch.mean(ret_normed)
    metrics["ret_005"] = self.return_ema.ema_vals[0]
    metrics["ret_095"] = self.return_ema.ema_vals[1]
    metrics["adv"] = torch.mean(adv)
    metrics["adv_std"] = torch.std(adv)
    metrics["con"] = torch.mean(imag_cont)
    metrics["rew"] = torch.mean(imag_reward)
    metrics["val"] = torch.mean(imag_value)
    metrics["tar"] = torch.mean(ret)
    metrics["slowval"] = torch.mean(imag_slow_value)
    metrics["weight"] = torch.mean(weight)
    metrics["action_entropy"] = torch.mean(entropy)
    metrics.update(tools.tensorstats(imag_action, "action"))

    last, term, reward = (
        to_f32(data["is_last"]),
        to_f32(data["is_terminal"]),
        to_f32(data["reward"]),
    )
    feat = self.rssm.get_feat(post_stoch, post_deter)
    boot = ret[:, 0].reshape(batch, time, 1)
    value = self._frozen_value(feat).mode()
    slow_value = self._frozen_slow_value(feat).mode()
    disc = 1 - 1 / self.horizon
    weight = 1.0 - last
    ret = self._lambda_return(last, term, reward, value, boot, disc, self.lamb)
    ret_padded = torch.cat([ret, 0 * ret[:, -1:]], 1)

    value_dist = self.value(feat)
    losses["repval"] = torch.mean(
        weight[:, :-1]
        * (
            -value_dist.log_prob(ret_padded.detach())
            - value_dist.log_prob(slow_value.detach())
        )[:, :-1].unsqueeze(-1)
    )
    metrics.update(tools.tensorstats(ret, "ret_replay"))
    metrics.update(tools.tensorstats(value, "value_replay"))
    metrics.update(tools.tensorstats(slow_value, "slow_value_replay"))

    total_loss = sum([v * self._loss_scales[k] for k, v in losses.items()]) + sharp_loss
    self._scaler.scale(total_loss).backward()

    metrics.update({f"loss/{name}": loss for name, loss in losses.items()})
    metrics["loss/sharp"] = sharp_loss.detach()
    metrics["opt/loss"] = total_loss
    return (post_stoch, post_deter), metrics


def install_r2_sharp(agent: Any, args: argparse.Namespace, action_repeat: int) -> None:
    agent._sharp_cfg = SimpleNamespace(
        beta_mean=0.0 if args.disable_sharp else float(args.beta_mean),
        beta_var=0.0 if args.disable_sharp else float(args.beta_var),
        n_starts=int(args.n_starts),
        warmup_env_steps=int(args.warmup_env_steps),
    )
    agent._sharp_env_step = 0
    original_update = agent.update

    def update_with_env_step(replay_buffer):
        agent._sharp_env_step = int(replay_buffer.count() * action_repeat)
        return original_update(replay_buffer)

    agent.update = update_with_env_step


def load_r2_config(args: argparse.Namespace):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    logdir = PROJECT_ROOT / "results" / args.run_name
    with initialize_config_dir(version_base=None, config_dir=str(R2_ROOT / "configs")):
        config = compose(config_name="configs", overrides=["env=dmc_proprio", "model=size12M"])

    config.logdir = str(logdir)
    config.seed = int(args.seed)
    config.device = str(args.device)
    config.deterministic_run = bool(args.deterministic)

    config.batch_size = int(args.batch_size)
    config.batch_length = int(args.batch_length)

    config.env.task = str(args.task)
    config.env.steps = int(args.total_steps)
    config.env.env_num = int(args.envs)
    config.env.eval_episode_num = int(args.eval_episode_num)
    config.env.train_ratio = int(args.train_ratio)
    config.env.action_repeat = int(args.action_repeat)
    config.env.time_limit = int(args.time_limit)

    config.buffer.batch_size = int(args.batch_size)
    config.buffer.batch_length = int(args.batch_length)
    config.buffer.max_size = int(args.buffer_max_size)
    config.buffer.device = str(args.device)
    config.buffer.storage_device = str(args.storage_device or args.device)

    config.trainer.steps = int(args.total_steps)
    config.trainer.eval_every = int(args.eval_every)
    config.trainer.eval_episode_num = int(args.eval_episode_num)
    config.trainer.batch_size = int(args.batch_size)
    config.trainer.batch_length = int(args.batch_length)
    config.trainer.train_ratio = int(args.train_ratio)
    config.trainer.update_log_every = int(args.log_every)
    config.trainer.action_repeat = int(args.action_repeat)
    config.trainer.video_pred_log = False

    config.model.rep_loss = "r2dreamer"
    config.model.compile = False
    config.model.device = str(args.device)
    config.model.rssm.device = str(args.device)

    OmegaConf.resolve(config)
    return config


def save_checkpoint(agent: Any, logdir: pathlib.Path, *, step: int) -> pathlib.Path:
    import torch
    import tools

    logdir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = logdir / "latest.pt"
    torch.save(
        {
            "agent_state_dict": agent.state_dict(),
            "optims_state_dict": tools.recursively_collect_optim_state_dict(agent),
            "step": int(step),
            "sharp": vars(getattr(agent, "_sharp_cfg", SimpleNamespace())),
        },
        checkpoint_path,
    )
    print(f"Saved checkpoint: {checkpoint_path}", flush=True)
    return checkpoint_path


def verify_checkpoint(path: str | pathlib.Path) -> bool:
    import torch

    checkpoint_path = pathlib.Path(path).expanduser()
    size_mb = 0.0
    if checkpoint_path.exists():
        size_mb = checkpoint_path.stat().st_size / (1024 * 1024)

    try:
        if not checkpoint_path.is_file() or checkpoint_path.stat().st_size <= 0:
            raise ValueError("checkpoint file is missing or empty")

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        if not isinstance(checkpoint, dict):
            raise ValueError("checkpoint must be a dict")

        required_keys = ("agent_state_dict", "optims_state_dict", "step")
        missing = [key for key in required_keys if key not in checkpoint]
        if missing:
            raise ValueError(f"missing required keys: {missing}")

        agent_state_dict = checkpoint["agent_state_dict"]
        if not isinstance(agent_state_dict, dict) or len(agent_state_dict) == 0:
            raise ValueError("agent_state_dict must be a non-empty dict")

        step = checkpoint["step"]
        if isinstance(step, bool) or not isinstance(step, int) or step <= 0:
            raise ValueError(f"step must be a positive integer, got {step!r}")

        print(
            f"CHECKPOINT VERIFY PASS: {checkpoint_path} ({size_mb:.1f} MB), "
            f"agent_state_dict: {len(agent_state_dict)} keys, step={step}",
            flush=True,
        )
        return True
    except Exception as exc:
        print(
            f"CHECKPOINT VERIFY FAIL: {checkpoint_path} ({size_mb:.1f} MB): {exc}",
            flush=True,
        )
        return False


def close_env_group(env_group: Any) -> None:
    for env in getattr(env_group, "envs", []):
        try:
            env.close()
        except Exception:
            pass


def main() -> None:
    args = parse_args()
    if not R2_ROOT.exists():
        raise FileNotFoundError(
            f"Missing R2-Dreamer repo at {R2_ROOT}. Clone NM512/r2dreamer first."
        )

    os.environ.setdefault("MUJOCO_GL", "osmesa")
    os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
    add_import_paths()

    import torch
    import tools
    from buffer import Buffer
    from dreamer import Dreamer
    from envs import make_envs
    from trainer import OnlineTrainer

    Dreamer._cal_grad = r2_cal_grad_with_sharp
    patch_r2_dmc_size()

    config = load_r2_config(args)
    tools.set_seed_everywhere(config.seed)
    if config.deterministic_run:
        tools.enable_deterministic_run()

    logdir = pathlib.Path(config.logdir).expanduser()
    logdir.mkdir(parents=True, exist_ok=True)
    console_f = tools.setup_console_log(logdir, filename="console.log")
    atexit.register(lambda: console_f.close())

    print("=" * 70)
    print("R2-Dreamer + SHARP")
    print("=" * 70)
    print(f"R2 repo: {R2_ROOT}")
    print(f"Task: {config.env.task}, seed={config.seed}, steps={config.trainer.steps}")
    print(f"Device: {config.device}, envs={config.env.env_num}")
    print(
        "SHARP: "
        f"enabled={not args.disable_sharp}, "
        f"beta_mean={0.0 if args.disable_sharp else args.beta_mean}, "
        f"beta_var={0.0 if args.disable_sharp else args.beta_var}, "
        f"n_starts={args.n_starts}, "
        f"warmup_env_steps={args.warmup_env_steps}"
    )
    print("=" * 70, flush=True)

    logger = tools.Logger(logdir)
    logger.log_hydra_config(config)
    replay_buffer = Buffer(config.buffer)

    train_envs = eval_envs = None
    agent = None
    try:
        print("Create envs.", flush=True)
        train_envs, eval_envs, obs_space, act_space = make_envs(config.env)

        print("Create agent.", flush=True)
        agent = Dreamer(config.model, obs_space, act_space).to(config.device)
        install_r2_sharp(agent, args, action_repeat=int(config.trainer.action_repeat))

        trainer = OnlineTrainer(config.trainer, replay_buffer, logger, logdir, train_envs, eval_envs)
        trainer.begin(agent)
    finally:
        if agent is not None:
            step = int(replay_buffer.count() * int(config.trainer.action_repeat))
            checkpoint_path = save_checkpoint(agent, logdir, step=step)
            if not verify_checkpoint(checkpoint_path):
                raise RuntimeError(f"Saved checkpoint failed verification: {checkpoint_path}")
        if train_envs is not None:
            close_env_group(train_envs)
        if eval_envs is not None:
            close_env_group(eval_envs)

    print("Training completed.", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", default="r2_sharp_cheetah")
    parser.add_argument("--task", default="dmc_cheetah_run")
    parser.add_argument("--total_steps", type=int, default=500000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--storage_device", default=None)
    parser.add_argument("--envs", type=int, default=16)
    parser.add_argument("--eval_episode_num", type=int, default=10)
    parser.add_argument("--action_repeat", type=int, default=2)
    parser.add_argument("--time_limit", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--batch_length", type=int, default=64)
    parser.add_argument("--buffer_max_size", type=int, default=500000)
    parser.add_argument("--train_ratio", type=int, default=512)
    parser.add_argument("--eval_every", type=int, default=10000)
    parser.add_argument("--log_every", type=int, default=5000)
    parser.add_argument("--beta_mean", type=float, default=1.0)
    parser.add_argument("--beta_var", type=float, default=0.1)
    parser.add_argument("--n_starts", type=int, default=8)
    parser.add_argument("--warmup_env_steps", type=int, default=50000)
    parser.add_argument("--disable_sharp", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
