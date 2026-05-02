#!/usr/bin/env python3
"""SMAD Phase 2 DreamerV3 training launcher.

This wrapper applies runtime SMAD damping to NM512's DreamerV3 RSSM and then
hands off to the standard ``dreamer.main(config)`` training loop. It supports
both the identity baseline (``eta=0``) and the damping intervention
(``eta>0``) without modifying files under ``external/dreamerv3-torch``.
"""

from __future__ import annotations

import argparse
import functools
import pathlib
import sys
from collections.abc import Sequence


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
DREAMERV3_ROOT = PROJECT_ROOT / "external" / "dreamerv3-torch"
DEFAULT_SEED = 42


def main(argv: Sequence[str] | None = None) -> None:
    wrapper_args, dreamer_args = parse_wrapper_args(argv)
    validate_run_name(wrapper_args.run_name)

    add_import_paths()
    import dreamer  # type: ignore
    import networks  # type: ignore
    import torch

    config, config_names = parse_dreamer_config(dreamer, dreamer_args)
    if not dreamer_arg_present(dreamer_args, "--seed"):
        config.seed = DEFAULT_SEED
    config.logdir = str(PROJECT_ROOT / "results" / "smad_phase2" / wrapper_args.run_name)

    projector_cpu = None
    resolved_u_path = None
    if wrapper_args.smad_eta > 0.0:
        resolved_u_path, projector_cpu = load_projector(
            wrapper_args.smad_U_path,
            wrapper_args.smad_r,
            torch,
        )

    patch_rssm_img_step(
        networks,
        eta=wrapper_args.smad_eta,
        projector_cpu=projector_cpu,
    )
    print_banner(wrapper_args, config, config_names, resolved_u_path)
    dreamer.main(config)


def parse_wrapper_args(
    argv: Sequence[str] | None,
) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description=(
            "Launch DreamerV3 training with optional SMAD damping. Unknown "
            "arguments are forwarded to DreamerV3's config parser."
        )
    )
    parser.add_argument("--smad_eta", type=float, default=0.0)
    parser.add_argument("--smad_r", type=int, default=10)
    parser.add_argument("--smad_U_path", type=str, default=None)
    parser.add_argument("--run_name", type=str, required=True)
    args, remaining = parser.parse_known_args(argv)

    if args.smad_eta < 0.0:
        raise ValueError(f"--smad_eta must be non-negative, got {args.smad_eta}.")
    if args.smad_r <= 0:
        raise ValueError(f"--smad_r must be positive, got {args.smad_r}.")
    if args.smad_eta > 0.0 and not args.smad_U_path:
        raise ValueError("--smad_U_path is required when --smad_eta > 0.")
    return args, remaining


def add_import_paths() -> None:
    for path in (DREAMERV3_ROOT, PROJECT_ROOT):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def parse_dreamer_config(
    dreamer_module,
    dreamer_args: Sequence[str],
) -> tuple[argparse.Namespace, list[str]]:
    configs_parser = argparse.ArgumentParser(add_help=False)
    configs_parser.add_argument("--configs", nargs="+")
    config_args, remaining = configs_parser.parse_known_args(dreamer_args)

    configs_path = DREAMERV3_ROOT / "configs.yaml"
    yaml_loader = dreamer_module.yaml.YAML(typ="safe", pure=True)
    configs = yaml_loader.load(configs_path.read_text())

    defaults = {}
    config_names = ["defaults", *(config_args.configs or [])]
    for name in config_names:
        if name not in configs:
            raise KeyError(f"DreamerV3 config preset {name!r} not found.")
        recursive_update(defaults, configs[name])

    parser = argparse.ArgumentParser(
        description="DreamerV3 passthrough config parser",
    )
    for key, value in sorted(defaults.items(), key=lambda item: item[0]):
        arg_type = dreamer_module.tools.args_type(value)
        parser.add_argument(f"--{key}", type=arg_type, default=arg_type(value))
    return parser.parse_args(remaining), config_names[1:]


def recursive_update(base: dict, update: dict) -> None:
    for key, value in update.items():
        if isinstance(value, dict) and key in base:
            recursive_update(base[key], value)
        else:
            base[key] = value


def load_projector(
    u_path: str | None,
    rank: int,
    torch_module,
) -> tuple[pathlib.Path, "torch.Tensor"]:
    from src.smad.U_estimation import load_U

    if u_path is None:
        raise ValueError("u_path cannot be None when loading a SMAD projector.")
    resolved = resolve_project_path(u_path)
    U = load_U(resolved)
    if rank > U.shape[1]:
        raise ValueError(
            f"Requested --smad_r={rank}, but {resolved} only has rank {U.shape[1]}."
        )
    U = U[:, :rank]
    projector = torch_module.as_tensor(U @ U.T, dtype=torch_module.float32)
    return resolved, projector


def patch_rssm_img_step(networks_module, eta: float, projector_cpu) -> None:
    original_img_step = networks_module.RSSM.img_step
    projector_cache = {}

    def get_projector(deter):
        if projector_cpu is None:
            raise RuntimeError("SMAD projector was not loaded.")
        key = (str(deter.device), str(deter.dtype))
        if key not in projector_cache:
            projector_cache[key] = projector_cpu.to(
                device=deter.device,
                dtype=deter.dtype,
            )
        return projector_cache[key]

    @functools.wraps(original_img_step)
    def patched_img_step(self, prev_state, prev_action, sample=True):
        if eta == 0.0:
            return original_img_step(self, prev_state, prev_action, sample=sample)

        # Compute the undamped deterministic transition without consuming a
        # stochastic sample that would be discarded after damping.
        prior = original_img_step(self, prev_state, prev_action, sample=False)
        prev_deter = prev_state["deter"]
        deter = prior["deter"]
        P = get_projector(deter)
        if P.shape != (deter.shape[-1], deter.shape[-1]):
            raise RuntimeError(
                f"SMAD projector shape {tuple(P.shape)} does not match "
                f"deter dimension {deter.shape[-1]}."
            )

        delta = deter - prev_deter
        deter = deter - eta * (delta @ P.T)

        x = self._img_out_layers(deter)
        stats = self._suff_stats_layer("ims", x)
        if sample:
            stoch = self.get_dist(stats).sample()
        else:
            stoch = self.get_dist(stats).mode()
        return {"stoch": stoch, "deter": deter, **stats}

    networks_module.RSSM.img_step = patched_img_step


def print_banner(
    wrapper_args: argparse.Namespace,
    config: argparse.Namespace,
    config_names: Sequence[str],
    resolved_u_path: pathlib.Path | None,
) -> None:
    line = "=" * 78
    print(line, flush=True)
    print("SMAD Phase 2 DreamerV3 Training", flush=True)
    print(line, flush=True)
    print(f"Run name:        {wrapper_args.run_name}", flush=True)
    print(f"Logdir:          {config.logdir}", flush=True)
    print(f"Dreamer configs: {', '.join(config_names) if config_names else 'defaults'}", flush=True)
    print(f"Task:            {config.task}", flush=True)
    print(f"Steps:           {config.steps}", flush=True)
    print(f"Batch size:      {config.batch_size}", flush=True)
    print(f"Batch length:    {config.batch_length}", flush=True)
    print(f"Envs:            {config.envs}", flush=True)
    print(f"Seed:            {config.seed}", flush=True)
    print(f"Device:          {config.device}", flush=True)
    print(f"SMAD eta:        {wrapper_args.smad_eta}", flush=True)
    print(f"SMAD rank:       {wrapper_args.smad_r}", flush=True)
    print(f"SMAD U path:     {resolved_u_path if resolved_u_path else 'not loaded'}", flush=True)
    print("Anchor loss:     disabled for damping-only Phase 2", flush=True)
    print(line, flush=True)


def validate_run_name(run_name: str) -> None:
    path = pathlib.Path(run_name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Invalid --run_name {run_name!r}.")
    if not run_name.strip():
        raise ValueError("--run_name cannot be empty.")


def resolve_project_path(path: str) -> pathlib.Path:
    value = pathlib.Path(path).expanduser()
    if not value.is_absolute():
        value = PROJECT_ROOT / value
    return value.resolve()


def dreamer_arg_present(args: Sequence[str], name: str) -> bool:
    return any(arg == name or arg.startswith(f"{name}=") for arg in args)


if __name__ == "__main__":
    main()
