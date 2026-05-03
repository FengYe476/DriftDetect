#!/usr/bin/env python3
"""Adaptive-SMAD DreamerV3 training launcher.

This launcher mirrors ``scripts/train_smad_phase2.py`` and NM512's
``dreamer.main(config)`` loop, but owns the training loop so Adaptive-SMAD can
refresh ``U_drift`` during training and persist scheduler state.
"""

from __future__ import annotations

import argparse
import copy
import functools
import json
import pathlib
import sys
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
DREAMERV3_ROOT = PROJECT_ROOT / "external" / "dreamerv3-torch"
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "adaptive_smad_v1.yaml"
DEFAULT_RUN_NAME = "adaptive_smad_v1"
REEST_LOG_PATH = PROJECT_ROOT / "results" / "tables" / "adaptive_smad_v1_reest_log.json"


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    add_import_paths()

    import dreamer  # type: ignore
    import torch

    run_config = load_yaml(args.config, dreamer)
    config, config_names = build_dreamer_config(
        dreamer,
        torch,
        run_config,
        run_name=args.run_name,
        device_override=args.device,
        dry_run=args.dry_run,
    )
    smad_cfg = require_mapping(run_config, "smad")
    adaptive_cfg = require_mapping(run_config, "adaptive")
    schedule_cfg = require_mapping(run_config, "schedule")
    checkpoint_cfg = require_mapping(run_config, "checkpoint")
    monitor_cfg = dict(run_config.get("monitor", {}))

    validate_smad_config(smad_cfg)
    initial_U_path, initial_U, projector_cpu = load_initial_basis(
        smad_cfg["initial_basis_path"],
        rank=int(smad_cfg["rank"]),
        torch_module=torch,
    )
    if args.dry_run:
        dry_run_setup(
            dreamer=dreamer,
            torch_module=torch,
            config=config,
            config_names=config_names,
            smad_cfg=smad_cfg,
            adaptive_cfg=adaptive_cfg,
            schedule_cfg=schedule_cfg,
            checkpoint_cfg=checkpoint_cfg,
            initial_U_path=initial_U_path,
            initial_U=initial_U,
            projector_cpu=projector_cpu,
        )
        return

    run_training(
        dreamer=dreamer,
        torch_module=torch,
        config=config,
        config_names=config_names,
        smad_cfg=smad_cfg,
        adaptive_cfg=adaptive_cfg,
        schedule_cfg=schedule_cfg,
        checkpoint_cfg=checkpoint_cfg,
        monitor_cfg=monitor_cfg,
        initial_U_path=initial_U_path,
        initial_U=initial_U,
        projector_cpu=projector_cpu,
    )


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch Adaptive-SMAD training.")
    parser.add_argument("--config", type=pathlib.Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run_name", type=str, default=DEFAULT_RUN_NAME)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    args.config = resolve_project_path(args.config)
    validate_run_name(args.run_name)
    return args


def add_import_paths() -> None:
    for path in (DREAMERV3_ROOT, PROJECT_ROOT):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def load_yaml(path: pathlib.Path, dreamer_module) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Adaptive-SMAD config not found: {path}")
    loader = dreamer_module.yaml.YAML(typ="safe", pure=True)
    data = loader.load(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping.")
    return data


def build_dreamer_config(
    dreamer_module,
    torch_module,
    run_config: dict[str, Any],
    *,
    run_name: str,
    device_override: str | None,
    dry_run: bool,
) -> tuple[argparse.Namespace, list[str]]:
    configs_path = DREAMERV3_ROOT / "configs.yaml"
    loader = dreamer_module.yaml.YAML(typ="safe", pure=True)
    configs = loader.load(configs_path.read_text())
    config_names = list(run_config.get("dreamer_configs", ["dmc_proprio"]))
    defaults: dict[str, Any] = {}
    for name in ["defaults", *config_names]:
        if name not in configs:
            raise KeyError(f"DreamerV3 config preset {name!r} not found.")
        recursive_update(defaults, configs[name])

    dreamer_overrides = run_config.get("dreamer", {})
    if dreamer_overrides:
        if not isinstance(dreamer_overrides, dict):
            raise ValueError("dreamer config override must be a mapping.")
        recursive_update(defaults, dreamer_overrides)

    defaults["task"] = normalize_task(str(run_config["task"]))
    defaults["steps"] = int(run_config["total_steps"])
    defaults["seed"] = int(run_config["seed"])
    defaults["logdir"] = str(PROJECT_ROOT / "results" / run_name)
    defaults["device"] = device_override or run_config.get("device") or select_device(torch_module)
    if dry_run:
        defaults["compile"] = False
        defaults["envs"] = 1
        defaults["parallel"] = False
        defaults["eval_episode_num"] = 0
        defaults["video_pred_log"] = False

    return argparse.Namespace(**defaults), config_names


def run_training(
    *,
    dreamer,
    torch_module,
    config: argparse.Namespace,
    config_names: list[str],
    smad_cfg: Mapping[str, Any],
    adaptive_cfg: Mapping[str, Any],
    schedule_cfg: Mapping[str, Any],
    checkpoint_cfg: Mapping[str, Any],
    monitor_cfg: Mapping[str, Any],
    initial_U_path: pathlib.Path,
    initial_U: np.ndarray,
    projector_cpu,
) -> None:
    tools = dreamer.tools
    tools.set_seed_everywhere(config.seed)
    if config.deterministic_run:
        tools.enable_deterministic_run()

    logdir = pathlib.Path(config.logdir).expanduser()
    config.traindir = config.traindir or logdir / "train_eps"
    config.evaldir = config.evaldir or logdir / "eval_eps"
    config.steps //= config.action_repeat
    config.eval_every //= config.action_repeat
    config.log_every //= config.action_repeat
    config.time_limit //= config.action_repeat
    save_freq_steps = max(1, int(checkpoint_cfg["save_freq"]) // config.action_repeat)

    print_banner(
        title="Adaptive-SMAD DreamerV3 Training",
        config=config,
        config_names=config_names,
        smad_cfg=smad_cfg,
        adaptive_cfg=adaptive_cfg,
        schedule_cfg=schedule_cfg,
        checkpoint_cfg=checkpoint_cfg,
        initial_U_path=initial_U_path,
        dry_run=False,
    )

    logdir.mkdir(parents=True, exist_ok=True)
    pathlib.Path(config.traindir).mkdir(parents=True, exist_ok=True)
    pathlib.Path(config.evaldir).mkdir(parents=True, exist_ok=True)
    checkpoint_dir = logdir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    step = dreamer.count_steps(config.traindir)
    logger = tools.Logger(logdir, config.action_repeat * step)

    print("Create envs.", flush=True)
    train_eps = tools.load_episodes(
        config.offline_traindir.format(**vars(config))
        if config.offline_traindir
        else config.traindir,
        limit=config.dataset_size,
    )
    eval_eps = tools.load_episodes(
        config.offline_evaldir.format(**vars(config))
        if config.offline_evaldir
        else config.evaldir,
        limit=1,
    )
    make = lambda mode, env_id: dreamer.make_env(config, mode, env_id)
    train_envs = [make("train", i) for i in range(config.envs)]
    eval_envs = [make("eval", i) for i in range(config.envs)]
    if config.parallel:
        train_envs = [dreamer.Parallel(env, "process") for env in train_envs]
        eval_envs = [dreamer.Parallel(env, "process") for env in eval_envs]
    else:
        train_envs = [dreamer.Damy(env) for env in train_envs]
        eval_envs = [dreamer.Damy(env) for env in eval_envs]
    acts = train_envs[0].action_space
    print("Action Space", acts, flush=True)
    config.num_actions = acts.n if hasattr(acts, "n") else acts.shape[0]

    state = None
    if not config.offline_traindir:
        state = prefill_dataset(
            dreamer=dreamer,
            torch_module=torch_module,
            config=config,
            train_envs=train_envs,
            train_eps=train_eps,
            logger=logger,
            acts=acts,
        )

    print("Simulate agent.", flush=True)
    train_dataset = dreamer.make_dataset(train_eps, config)
    eval_dataset = dreamer.make_dataset(eval_eps, config)
    agent = dreamer.Dreamer(
        train_envs[0].observation_space,
        train_envs[0].action_space,
        config,
        logger,
        train_dataset,
    ).to(config.device)
    agent.requires_grad_(requires_grad=False)

    latest_path = logdir / "latest.pt"
    checkpoint = None
    if latest_path.exists():
        checkpoint = load_training_checkpoint(
            torch_module,
            latest_path,
            map_location=config.device,
        )
        agent.load_state_dict(checkpoint["agent_state_dict"])
        tools.recursively_load_optim_state_dict(agent, checkpoint["optims_state_dict"])
        agent._should_pretrain._once = False

    patch = create_mutable_patch(agent, projector_cpu, eta=float(smad_cfg["eta"]))
    scheduler = create_scheduler(
        agent=agent,
        config=config,
        patch=patch,
        initial_U=initial_U,
        adaptive_cfg=adaptive_cfg,
        schedule_cfg=schedule_cfg,
        make_env_fn=dreamer.make_env,
    )
    reest_log: list[dict[str, Any]] = []
    if checkpoint is not None:
        scheduler_state = checkpoint.get("adaptive_smad_scheduler_state")
        if scheduler_state is not None:
            scheduler.load_state_dict(scheduler_state)
        reest_log = list(checkpoint.get("adaptive_smad_reest_log", []))

    adaptive_agent = AdaptiveTrainCallable(
        agent=agent,
        scheduler=scheduler,
        logger=logger,
        reest_log=reest_log,
        monitor_cfg=monitor_cfg,
    )

    next_eval_step = agent._step
    next_save_step = agent._step + save_freq_steps
    final_step = config.steps + config.eval_every
    try:
        while agent._step < final_step:
            logger.write()
            if config.eval_episode_num > 0 and agent._step >= next_eval_step:
                print("Start evaluation.", flush=True)
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
                if config.video_pred_log:
                    video_pred = agent._wm.video_pred(next(eval_dataset))
                    logger.video("eval_openl", dreamer.to_np(video_pred))
                next_eval_step += config.eval_every

            print("Start training.", flush=True)
            chunk_steps = min(
                max(1, next_eval_step - agent._step),
                max(1, next_save_step - agent._step),
                max(1, final_step - agent._step),
            )
            state = tools.simulate(
                adaptive_agent,
                train_envs,
                train_eps,
                config.traindir,
                logger,
                limit=config.dataset_size,
                steps=chunk_steps,
                state=state,
            )
            if agent._step >= next_save_step or agent._step >= config.steps:
                save_training_checkpoint(
                    torch_module=torch_module,
                    agent=agent,
                    tools=tools,
                    scheduler=scheduler,
                    reest_log=reest_log,
                    latest_path=latest_path,
                    checkpoint_dir=checkpoint_dir,
                    env_step=config.action_repeat * agent._step,
                )
                while next_save_step <= agent._step:
                    next_save_step += save_freq_steps
    finally:
        save_reest_log(reest_log, REEST_LOG_PATH)
        for env in train_envs + eval_envs:
            try:
                env.close()
            except Exception:
                pass


def dry_run_setup(
    *,
    dreamer,
    torch_module,
    config: argparse.Namespace,
    config_names: list[str],
    smad_cfg: Mapping[str, Any],
    adaptive_cfg: Mapping[str, Any],
    schedule_cfg: Mapping[str, Any],
    checkpoint_cfg: Mapping[str, Any],
    initial_U_path: pathlib.Path,
    initial_U: np.ndarray,
    projector_cpu,
) -> None:
    print_banner(
        title="Adaptive-SMAD Dry Run",
        config=config,
        config_names=config_names,
        smad_cfg=smad_cfg,
        adaptive_cfg=adaptive_cfg,
        schedule_cfg=schedule_cfg,
        checkpoint_cfg=checkpoint_cfg,
        initial_U_path=initial_U_path,
        dry_run=True,
    )
    tools = dreamer.tools
    tools.set_seed_everywhere(config.seed)
    env = dreamer.make_env(config, "eval", 0)
    try:
        config.num_actions = (
            env.action_space.n if hasattr(env.action_space, "n") else env.action_space.shape[0]
        )
        agent = dreamer.Dreamer(
            env.observation_space,
            env.action_space,
            config,
            NullLogger(),
            dataset=None,
        ).to(config.device)
        agent.requires_grad_(requires_grad=False)
        patch = create_mutable_patch(agent, projector_cpu, eta=float(smad_cfg["eta"]))
        scheduler = create_scheduler(
            agent=agent,
            config=config,
            patch=patch,
            initial_U=initial_U,
            adaptive_cfg=adaptive_cfg,
            schedule_cfg=schedule_cfg,
            make_env_fn=dreamer.make_env,
        )
        result = scheduler.maybe_update(0)
        print(f"Dry-run scheduler.maybe_update(0): {result}", flush=True)
        print("Dry run complete.", flush=True)
    finally:
        try:
            env.close()
        except Exception:
            pass


class AdaptiveTrainCallable:
    """Callable passed to DreamerV3's simulate loop."""

    def __init__(
        self,
        *,
        agent,
        scheduler,
        logger,
        reest_log: list[dict[str, Any]],
        monitor_cfg: Mapping[str, Any],
    ) -> None:
        self.agent = agent
        self.scheduler = scheduler
        self.logger = logger
        self.reest_log = reest_log
        self.monitor_cfg = monitor_cfg

    def __call__(self, obs, reset, state=None):
        policy_output, state = self.agent(obs, reset, state, training=True)
        env_step = int(self.agent._config.action_repeat * self.agent._step)
        metrics = self.scheduler.maybe_update(env_step)
        if metrics is not None:
            self.reest_log.append(dict(metrics))
            if self.monitor_cfg.get("log_overlap_prev", True):
                self.logger.scalar("adaptive_smad/overlap_prev", metrics["overlap_prev"])
            if self.monitor_cfg.get("log_overlap_baseline", True):
                self.logger.scalar(
                    "adaptive_smad/overlap_baseline",
                    metrics["overlap_baseline"],
                )
            if self.monitor_cfg.get("log_reest_time", True):
                self.logger.scalar("adaptive_smad/reest_time", metrics["time_seconds"])
            print(
                f"[Step {metrics['step']}] Re-estimated U_drift: "
                f"overlap_prev={metrics['overlap_prev']:.4f}, "
                f"overlap_baseline={metrics['overlap_baseline']:.4f}, "
                f"time={metrics['time_seconds']:.1f}s",
                flush=True,
            )
        return policy_output, state


class LiveDreamerRolloutAdapter:
    """Mockable rollout protocol backed by the live DreamerV3 agent."""

    def __init__(self, agent, base_config: argparse.Namespace, make_env_fn) -> None:
        self.agent = agent
        self.base_config = base_config
        self.make_env_fn = make_env_fn
        self.training = bool(agent.training)

    def eval(self) -> None:
        self.training = False
        self.agent.eval()

    def train(self, mode: bool = True) -> None:
        self.training = bool(mode)
        self.agent.train(mode)

    def extract_rollout(
        self,
        *,
        seed: int,
        total_steps: int,
        imagination_start: int,
        horizon: int,
        include_latent: bool,
    ) -> dict[str, np.ndarray]:
        if not include_latent:
            raise ValueError("Adaptive-SMAD re-estimation requires include_latent=True.")
        if total_steps < imagination_start + horizon:
            raise ValueError("total_steps must cover imagination_start + horizon.")

        import torch

        env_config = copy.copy(self.base_config)
        env_config.seed = int(self.base_config.seed) + int(seed)
        env = self.make_env_fn(env_config, "eval", 0)
        true_latent = []
        imagined_latent = []
        actions = []
        rewards = []
        obs = env.reset()
        prev_latent = None
        prev_action = None
        imag_latent = None
        try:
            for step in range(total_steps):
                with torch.no_grad():
                    latent, feat = self._posterior_latent_from_obs(
                        obs,
                        prev_latent,
                        prev_action,
                    )
                    action_tensor = self._policy_action_from_feat(feat)

                in_window = imagination_start <= step < imagination_start + horizon
                if step == imagination_start:
                    imag_latent = clone_latent(latent)

                if in_window:
                    if imag_latent is None:
                        raise RuntimeError("Imagination latent was not initialized.")
                    with torch.no_grad():
                        imag_latent = self.agent._wm.dynamics.img_step(
                            imag_latent,
                            action_tensor,
                            sample=False,
                        )
                        imag_feat = self.agent._wm.dynamics.get_feat(imag_latent)
                    imagined_latent.append(
                        imag_feat[0].detach().cpu().numpy().astype(np.float32)
                    )

                env_action, action_tensor = action_for_env({"action": action_tensor})
                next_obs, reward, done_bool, _info = env.step(env_action)
                actions.append(env_action["action"])
                rewards.append(float(reward))

                if in_window:
                    with torch.no_grad():
                        post_latent, post_feat = self._posterior_latent_from_obs(
                            next_obs,
                            latent,
                            action_tensor,
                        )
                    true_latent.append(
                        post_feat[0].detach().cpu().numpy().astype(np.float32)
                    )

                if done_bool and step + 1 < total_steps:
                    obs = env.reset()
                    prev_latent = None
                    prev_action = None
                else:
                    obs = next_obs
                    prev_latent = latent
                    prev_action = action_tensor
        finally:
            try:
                env.close()
            except Exception:
                pass

        if len(true_latent) != horizon or len(imagined_latent) != horizon:
            raise RuntimeError(
                f"Expected {horizon} latent steps, got true={len(true_latent)} "
                f"imagined={len(imagined_latent)}."
            )
        return {
            "true_latent": np.asarray(true_latent, dtype=np.float32),
            "imagined_latent": np.asarray(imagined_latent, dtype=np.float32),
            "actions": np.asarray(actions, dtype=np.float32),
            "rewards": np.asarray(rewards, dtype=np.float32),
        }

    def _posterior_latent_from_obs(self, obs, prev_latent, prev_action):
        processed = self.agent._wm.preprocess(batch_obs(obs))
        embed = self.agent._wm.encoder(processed)
        latent, _ = self.agent._wm.dynamics.obs_step(
            prev_latent,
            prev_action,
            embed,
            processed["is_first"],
            sample=False,
        )
        feat = self.agent._wm.dynamics.get_feat(latent)
        return clone_latent(latent), feat

    def _policy_action_from_feat(self, feat):
        import torch

        actor = self.agent._task_behavior.actor(feat)
        action = actor.mode().detach()
        if self.agent._config.actor["dist"] == "onehot_gumble":
            action = torch.one_hot(
                torch.argmax(action, dim=-1),
                self.agent._config.num_actions,
            )
        return action


class NullLogger:
    step = 0

    def scalar(self, *_args: Any, **_kwargs: Any) -> None:
        pass

    def video(self, *_args: Any, **_kwargs: Any) -> None:
        pass

    def write(self, *_args: Any, **_kwargs: Any) -> None:
        pass


def create_scheduler(
    *,
    agent,
    config: argparse.Namespace,
    patch,
    initial_U: np.ndarray,
    adaptive_cfg: Mapping[str, Any],
    schedule_cfg: Mapping[str, Any],
    make_env_fn,
):
    from src.smad.adaptive_smad import AdaptiveReEstimator, AdaptiveSMADScheduler

    rollout_adapter = LiveDreamerRolloutAdapter(
        agent=agent,
        base_config=config,
        make_env_fn=make_env_fn,
    )
    re_estimator = AdaptiveReEstimator(
        rollout_adapter,
        env_name=config.task,
        rank=int(initial_U.shape[1]),
        n_rollouts=int(adaptive_cfg["n_rollouts"]),
        horizon=int(adaptive_cfg["horizon"]),
        device=config.device,
        imagination_start=int(adaptive_cfg["imagination_start"]),
    )
    return AdaptiveSMADScheduler(
        re_estimator=re_estimator,
        img_step_patch=patch,
        re_est_freq=int(adaptive_cfg["re_est_freq"]),
        activation_s0=int(schedule_cfg["s0"]),
        activation_s1=int(schedule_cfg["s1"]),
        initial_U=initial_U,
    )


def create_mutable_patch(agent, projector_cpu, eta: float):
    from src.smad.img_step_patch import MutableImgStepPatch

    return MutableImgStepPatch(agent._wm.dynamics, projector_cpu, eta=eta)


def prefill_dataset(
    *,
    dreamer,
    torch_module,
    config: argparse.Namespace,
    train_envs,
    train_eps,
    logger,
    acts,
):
    tools = dreamer.tools
    if config.offline_traindir:
        return None
    prefill = max(0, config.prefill - dreamer.count_steps(config.traindir))
    print(f"Prefill dataset ({prefill} steps).", flush=True)
    if hasattr(acts, "discrete"):
        random_actor = tools.OneHotDist(torch_module.zeros(config.num_actions).repeat(config.envs, 1))
    else:
        random_actor = dreamer.torchd.independent.Independent(
            dreamer.torchd.uniform.Uniform(
                torch_module.tensor(acts.low).repeat(config.envs, 1),
                torch_module.tensor(acts.high).repeat(config.envs, 1),
            ),
            1,
        )

    def random_agent(_obs, _done, _state):
        action = random_actor.sample()
        logprob = random_actor.log_prob(action)
        return {"action": action, "logprob": logprob}, None

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
    print(f"Logger: ({logger.step} steps).", flush=True)
    return state


def save_training_checkpoint(
    *,
    torch_module,
    agent,
    tools,
    scheduler,
    reest_log: list[dict[str, Any]],
    latest_path: pathlib.Path,
    checkpoint_dir: pathlib.Path,
    env_step: int,
) -> None:
    items_to_save = {
        "agent_state_dict": agent.state_dict(),
        "optims_state_dict": tools.recursively_collect_optim_state_dict(agent),
        "adaptive_smad_scheduler_state": scheduler.state_dict(),
        "adaptive_smad_reest_log": reest_log,
    }
    torch_module.save(items_to_save, latest_path)
    torch_module.save(items_to_save, checkpoint_dir / f"step_{env_step}.pt")
    print(f"Saved checkpoint at step {env_step}.", flush=True)


def load_training_checkpoint(torch_module, path: pathlib.Path, *, map_location):
    try:
        return torch_module.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch_module.load(path, map_location=map_location)


def save_reest_log(reest_log: list[dict[str, Any]], path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(reest_log), indent=2) + "\n")
    print(f"Saved re-estimation log: {relative_path(path)}", flush=True)


def load_initial_basis(
    path: str,
    *,
    rank: int,
    torch_module,
) -> tuple[pathlib.Path, np.ndarray, Any]:
    from src.smad.U_estimation import load_U

    resolved = resolve_project_path(pathlib.Path(path))
    U = load_U(resolved)
    if rank > U.shape[1]:
        raise ValueError(f"Requested rank={rank}, but {resolved} only has {U.shape[1]}.")
    U = U[:, :rank].copy()
    projector = torch_module.as_tensor(U @ U.T, dtype=torch_module.float32)
    return resolved, U, projector


def validate_smad_config(smad_cfg: Mapping[str, Any]) -> None:
    if float(smad_cfg["eta"]) < 0.0:
        raise ValueError("smad.eta must be non-negative.")
    if int(smad_cfg["rank"]) <= 0:
        raise ValueError("smad.rank must be positive.")
    if smad_cfg.get("basis_type") != "U_drift":
        raise ValueError("Adaptive-SMAD v1 expects smad.basis_type: U_drift.")


def print_banner(
    *,
    title: str,
    config: argparse.Namespace,
    config_names: list[str],
    smad_cfg: Mapping[str, Any],
    adaptive_cfg: Mapping[str, Any],
    schedule_cfg: Mapping[str, Any],
    checkpoint_cfg: Mapping[str, Any],
    initial_U_path: pathlib.Path,
    dry_run: bool,
) -> None:
    line = "=" * 78
    print(line, flush=True)
    print(title, flush=True)
    print(line, flush=True)
    print(f"Mode:            {'dry-run' if dry_run else 'training'}", flush=True)
    print(f"Logdir:          {config.logdir}", flush=True)
    print(f"Dreamer configs: {', '.join(config_names)}", flush=True)
    print(f"Task:            {config.task}", flush=True)
    print(f"Steps:           {config.steps}", flush=True)
    print(f"Seed:            {config.seed}", flush=True)
    print(f"Device:          {config.device}", flush=True)
    print(f"SMAD eta:        {smad_cfg['eta']}", flush=True)
    print(f"SMAD rank:       {smad_cfg['rank']}", flush=True)
    print(f"Initial U:       {initial_U_path}", flush=True)
    print(f"Re-est freq:     {adaptive_cfg['re_est_freq']}", flush=True)
    print(f"Rollouts/update: {adaptive_cfg['n_rollouts']}", flush=True)
    print(f"Imagination:     start={adaptive_cfg['imagination_start']} horizon={adaptive_cfg['horizon']}", flush=True)
    print(f"Activation:      s0={schedule_cfg['s0']} s1={schedule_cfg['s1']}", flush=True)
    print(f"Checkpoint freq: {checkpoint_cfg['save_freq']}", flush=True)
    print(line, flush=True)


def normalize_task(task: str) -> str:
    known_prefixes = ("dmc_", "atari_", "dmlab_", "memorymaze_", "crafter_", "minecraft_")
    if task.startswith(known_prefixes):
        return task
    return f"dmc_{task}"


def select_device(torch_module) -> str:
    if torch_module.cuda.is_available():
        return "cuda:0"
    if hasattr(torch_module.backends, "mps") and torch_module.backends.mps.is_available():
        return "mps"
    return "cpu"


def batch_obs(obs: Mapping[str, Any]) -> dict[str, np.ndarray]:
    return {
        key: np.stack([np.asarray(value)])
        for key, value in obs.items()
        if not key.startswith("log_")
    }


def clone_latent(latent: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value.detach().clone() for key, value in latent.items()}


def action_for_env(policy_output: Mapping[str, Any]) -> tuple[dict[str, np.ndarray], Any]:
    action = policy_output["action"].detach()
    action_np = action[0].cpu().numpy().astype(np.float32)
    return {"action": action_np}, action


def recursive_update(base: dict, update: dict) -> None:
    for key, value in update.items():
        if isinstance(value, dict) and key in base:
            recursive_update(base[key], value)
        else:
            base[key] = value


def require_mapping(config: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = config.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"config.{key} must be a mapping.")
    return value


def validate_run_name(run_name: str) -> None:
    path = pathlib.Path(run_name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Invalid --run_name {run_name!r}.")
    if not run_name.strip():
        raise ValueError("--run_name cannot be empty.")


def resolve_project_path(path: pathlib.Path) -> pathlib.Path:
    value = path.expanduser()
    if not value.is_absolute():
        value = PROJECT_ROOT / value
    return value.resolve()


def relative_path(path: pathlib.Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pathlib.Path):
        return relative_path(value)
    return value


if __name__ == "__main__":
    main()
