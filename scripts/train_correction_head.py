#!/usr/bin/env python3
"""DreamerV3 training launcher with Residual Correction Head.

This launcher owns the standard DreamerV3 training loop so it can attach a
``CorrectionHead`` to actor imagination without modifying
``external/dreamerv3-torch``. The world-model one-step training path is left
unchanged; correction is applied only inside ``ImagBehavior._imagine``.
"""

from __future__ import annotations

import argparse
import functools
import json
import pathlib
import sys
import types
from collections.abc import Mapping, Sequence
from typing import Any


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
DREAMERV3_ROOT = PROJECT_ROOT / "external" / "dreamerv3-torch"
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "month8_correction.yaml"
DEFAULT_RUN_NAME = "month8_correction"


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
    correction_cfg = normalize_correction_config(run_config.get("correction", {}))
    if args.dry_run:
        dry_run_setup(
            dreamer=dreamer,
            torch_module=torch,
            config=config,
            config_names=config_names,
            correction_cfg=correction_cfg,
        )
        return

    run_training(
        dreamer=dreamer,
        torch_module=torch,
        config=config,
        config_names=config_names,
        correction_cfg=correction_cfg,
    )


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch DreamerV3 with CorrectionHead.")
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
        raise FileNotFoundError(f"Correction config not found: {path}")
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
    defaults["device"] = (
        device_override or run_config.get("device") or select_device(torch_module)
    )
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
    correction_cfg: Mapping[str, Any],
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

    print_banner(
        title="DreamerV3 + Residual Correction Head Training",
        config=config,
        config_names=config_names,
        correction_cfg=correction_cfg,
        dry_run=False,
    )

    logdir.mkdir(parents=True, exist_ok=True)
    pathlib.Path(config.traindir).mkdir(parents=True, exist_ok=True)
    pathlib.Path(config.evaldir).mkdir(parents=True, exist_ok=True)

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
    correction_head = install_correction_head(
        agent=agent,
        correction_cfg=correction_cfg,
        tools_module=tools,
        torch_module=torch_module,
    )

    latest_path = logdir / "latest.pt"
    if latest_path.exists():
        checkpoint = load_training_checkpoint(
            torch_module,
            latest_path,
            map_location=config.device,
        )
        agent.load_state_dict(checkpoint["agent_state_dict"])
        if correction_head is not None and "correction_head_state_dict" in checkpoint:
            correction_head.load_state_dict(checkpoint["correction_head_state_dict"])
        tools.recursively_load_optim_state_dict(agent, checkpoint["optims_state_dict"])
        agent._should_pretrain._once = False

    next_eval_step = agent._step
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
            save_training_checkpoint(
                torch_module=torch_module,
                agent=agent,
                tools=tools,
                correction_head=correction_head,
                latest_path=latest_path,
                env_step=config.action_repeat * agent._step,
            )
    finally:
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
    correction_cfg: Mapping[str, Any],
) -> None:
    print_banner(
        title="DreamerV3 + Residual Correction Head Dry Run",
        config=config,
        config_names=config_names,
        correction_cfg=correction_cfg,
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
        correction_head = install_correction_head(
            agent=agent,
            correction_cfg=correction_cfg,
            tools_module=tools,
            torch_module=torch_module,
        )
        if correction_head is None:
            print("Dry run complete: correction disabled.", flush=True)
        else:
            print(
                "Dry run complete: "
                f"gate={correction_head.gate_value:.6f}, "
                f"params={count_parameters(correction_head)}",
                flush=True,
            )
    finally:
        try:
            env.close()
        except Exception:
            pass


def install_correction_head(
    *,
    agent,
    correction_cfg: Mapping[str, Any],
    tools_module,
    torch_module,
):
    if not bool(correction_cfg["enabled"]):
        return None
    if module_is_compiled(agent._task_behavior):
        raise RuntimeError(
            "CorrectionHead requires patching agent._task_behavior._imagine, but "
            "agent._task_behavior appears to be torch-compiled. Set "
            "dreamer.compile: false for correction-enabled runs."
        )

    from src.smad.correction_head import CorrectionHead

    correction_head = CorrectionHead(
        deter_dim=int(correction_cfg["deter_dim"]),
        step_embed_dim=int(correction_cfg["step_embed_dim"]),
        hidden_dim=int(correction_cfg["hidden_dim"]),
        max_steps=int(correction_cfg["max_steps"]),
        init_scale=float(correction_cfg["init_scale"]),
    ).to(agent._config.device)
    behavior = agent._task_behavior
    behavior.correction_head = correction_head
    behavior._correction_last_mean_abs = 0.0
    behavior._correction_last_max_abs = 0.0

    # Register correction_head as a submodule of actor so state_dict works.
    behavior.actor.correction_head = correction_head
    # Add correction_head parameters to the existing actor optimizer so they
    # actually get updated during training. Simply registering as a submodule
    # is not enough because the Adam optimizer was already constructed.
    behavior._actor_opt._opt.add_param_group({
        "params": list(correction_head.parameters()),
    })
    total_params = sum(sum(p.numel() for p in pg["params"]) for pg in behavior._actor_opt._opt.param_groups)
    print(f"Actor optimizer now has {total_params} parameters (including correction_head).", flush=True)

    patch_imagination_loop(behavior, torch_module=torch_module)
    patch_train_metrics(behavior, tools_module=tools_module)
    print(
        "Installed CorrectionHead: "
        f"params={count_parameters(correction_head)}, "
        f"deter_dim={correction_cfg['deter_dim']}, "
        f"step_embed_dim={correction_cfg['step_embed_dim']}, "
        f"hidden_dim={correction_cfg['hidden_dim']}",
        flush=True,
    )
    return correction_head


def patch_imagination_loop(behavior, *, torch_module) -> None:
    """Patch img_step with a flag-gated correction, keeping static_scan intact."""
    original_img_step = behavior._world_model.dynamics.img_step
    behavior._correction_active = False
    behavior._correction_step_counter = 0
    behavior._correction_last_mean_abs = 0.0
    behavior._correction_last_max_abs = 0.0

    def img_step_with_correction(state, action, sample=False):
        succ = original_img_step(state, action, sample=sample)
        if not behavior._correction_active:
            return succ
        step = behavior._correction_step_counter
        correction = behavior.correction_head(succ["deter"], step)
        corrected = dict(succ)
        corrected["deter"] = succ["deter"] - correction
        correction_abs = correction.detach().abs()
        behavior._correction_last_mean_abs = float(correction_abs.mean().cpu().item())
        behavior._correction_last_max_abs = float(correction_abs.max().cpu().item())
        behavior._correction_step_counter += 1
        return corrected

    behavior._world_model.dynamics.img_step = img_step_with_correction

    # Patch _imagine to enable correction only during actor imagination
    original_imagine = behavior._imagine

    def imagine_with_correction_flag(self, start, policy, horizon):
        self._correction_active = True
        self._correction_step_counter = 0
        try:
            result = original_imagine(start, policy, horizon)
        finally:
            self._correction_active = False
        return result

    behavior._imagine = types.MethodType(imagine_with_correction_flag, behavior)


def patch_train_metrics(behavior, *, tools_module) -> None:
    original_train = behavior._train

    def train_with_correction_metrics(self, start, objective):
        with tools_module.RequiresGrad(self.correction_head):
            imag_feat, imag_state, imag_action, weights, metrics = original_train(
                start,
                objective,
            )
        metrics["correction_gate"] = self.correction_head.gate_value
        metrics["correction_mean_abs"] = float(self._correction_last_mean_abs)
        metrics["correction_max_abs"] = float(self._correction_last_max_abs)
        return imag_feat, imag_state, imag_action, weights, metrics

    behavior._train = types.MethodType(train_with_correction_metrics, behavior)


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
    correction_head,
    latest_path: pathlib.Path,
    env_step: int,
) -> None:
    items_to_save = {
        "agent_state_dict": agent.state_dict(),
        "optims_state_dict": tools.recursively_collect_optim_state_dict(agent),
    }
    if correction_head is not None:
        items_to_save["correction_head_state_dict"] = correction_head.state_dict()
    torch_module.save(items_to_save, latest_path)
    print(f"Saved checkpoint at step {env_step}.", flush=True)


def load_training_checkpoint(torch_module, path: pathlib.Path, *, map_location):
    try:
        return torch_module.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch_module.load(path, map_location=map_location)


class NullLogger:
    step = 0

    def scalar(self, *_args: Any, **_kwargs: Any) -> None:
        pass

    def video(self, *_args: Any, **_kwargs: Any) -> None:
        pass

    def write(self, *_args: Any, **_kwargs: Any) -> None:
        pass


def normalize_correction_config(raw_cfg: Any) -> dict[str, Any]:
    if raw_cfg is None or raw_cfg is False:
        raw_cfg = {}
    if not isinstance(raw_cfg, Mapping):
        raise ValueError("correction config must be a mapping when provided.")
    cfg = {
        "enabled": bool(raw_cfg.get("enabled", False)),
        "deter_dim": int(raw_cfg.get("deter_dim", 512)),
        "step_embed_dim": int(raw_cfg.get("step_embed_dim", 32)),
        "hidden_dim": int(raw_cfg.get("hidden_dim", 256)),
        "max_steps": int(raw_cfg.get("max_steps", 200)),
        "init_scale": float(raw_cfg.get("init_scale", 0.01)),
    }
    for key in ("deter_dim", "step_embed_dim", "hidden_dim", "max_steps"):
        if cfg[key] <= 0:
            raise ValueError(f"correction.{key} must be positive, got {cfg[key]}.")
    if cfg["init_scale"] < 0.0:
        raise ValueError("correction.init_scale must be non-negative.")
    return cfg


def module_is_compiled(module: Any) -> bool:
    if hasattr(module, "_orig_mod"):
        return True
    class_name = type(module).__name__.lower()
    module_name = type(module).__module__.lower()
    return "optimizedmodule" in class_name or "torch._dynamo" in module_name


def count_parameters(module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def print_banner(
    *,
    title: str,
    config: argparse.Namespace,
    config_names: list[str],
    correction_cfg: Mapping[str, Any],
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
    print(f"Batch size:      {config.batch_size}", flush=True)
    print(f"Envs:            {config.envs}", flush=True)
    print(f"Seed:            {config.seed}", flush=True)
    print(f"Device:          {config.device}", flush=True)
    print(f"Compile:         {config.compile}", flush=True)
    print(f"Imag horizon:    {config.imag_horizon}", flush=True)
    print(
        "Correction:      "
        f"enabled={correction_cfg['enabled']} "
        f"deter_dim={correction_cfg['deter_dim']} "
        f"step_embed_dim={correction_cfg['step_embed_dim']} "
        f"hidden_dim={correction_cfg['hidden_dim']} "
        f"max_steps={correction_cfg['max_steps']} "
        f"init_scale={correction_cfg['init_scale']}",
        flush=True,
    )
    print(line, flush=True)


def recursive_update(base: dict, update: dict) -> None:
    for key, value in update.items():
        if isinstance(value, dict) and key in base:
            recursive_update(base[key], value)
        else:
            base[key] = value


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


if __name__ == "__main__":
    main()
