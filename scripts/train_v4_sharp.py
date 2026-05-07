#!/usr/bin/env python3
"""DreamerV4 Dynamics + SHARP fine-tune.

Loads pretrained V4 dynamics checkpoint and fine-tunes with SHARP loss added
to the original flow matching loss.

SHARP on V4: penalize mean + variance of denoising prediction error.
L_SHARP = β_mean * ||E[z1_hat - z1]||² + β_var * ||Var[z1_hat - z1]||²
"""

import argparse
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DREAMER4_ROOT = PROJECT_ROOT / "external" / "dreamer4"
DREAMER4_PKG = DREAMER4_ROOT / "dreamer4"

for p in (DREAMER4_PKG, DREAMER4_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from model import (
    Encoder, Decoder, Tokenizer, Dynamics,
    temporal_patchify, pack_bottleneck_to_spatial,
)
from sharded_frame_dataset import ShardedFrameDataset
from task_set import TASK_SET


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_tokenizer(tok_path, device):
    ckpt = torch.load(tok_path, map_location=device, weights_only=False)
    tok_args = ckpt.get("args", {})
    
    H = int(tok_args.get("H", 128))
    W = int(tok_args.get("W", 128))
    C = int(tok_args.get("C", 3))
    patch = int(tok_args.get("patch", 4))
    Np = (H // patch) * (W // patch)
    Dp = patch * patch * C
    
    enc = Encoder(
        patch_dim=Dp,
        n_patches=Np,
        d_model=int(tok_args.get("d_model", 256)),
        n_latents=int(tok_args.get("n_latents", 16)),
        n_heads=int(tok_args.get("n_heads", 4)),
        depth=int(tok_args.get("depth", 8)),
        d_bottleneck=int(tok_args.get("d_bottleneck", 32)),
        dropout=float(tok_args.get("dropout", 0.05)),
        mlp_ratio=float(tok_args.get("mlp_ratio", 4.0)),
    ).to(device)
    
    state = ckpt["model"]
    enc_state = {k.replace("encoder.", ""): v for k, v in state.items() if k.startswith("encoder.")}
    enc.load_state_dict(enc_state, strict=False)
    enc.eval()
    enc.requires_grad_(False)
    
    return enc, tok_args


def load_dynamics(dyn_path, device, tok_args):
    ckpt = torch.load(dyn_path, map_location=device, weights_only=False)
    dyn_args = ckpt.get("args", {})
    
    d_bottleneck = int(tok_args.get("d_bottleneck", 32))
    packing_factor = int(dyn_args.get("packing_factor", 2))
    n_latents = int(tok_args.get("n_latents", 16))
    n_spatial = n_latents // packing_factor
    d_spatial = d_bottleneck * packing_factor
    
    dyn = Dynamics(
        d_model=int(dyn_args.get("d_model_dyn", 512)),
        d_bottleneck=d_bottleneck,
        d_spatial=d_spatial,
        n_spatial=n_spatial,
        n_register=int(dyn_args.get("n_register", 4)),
        n_agent=int(dyn_args.get("n_agent", 1)),
        n_heads=int(dyn_args.get("n_heads", 4)),
        depth=int(dyn_args.get("dyn_depth", 8)),
        k_max=int(dyn_args.get("k_max", 8)),
        dropout=float(dyn_args.get("dropout", 0.0)),
        mlp_ratio=float(dyn_args.get("mlp_ratio", 4.0)),
        time_every=int(dyn_args.get("time_every", 1)),
        space_mode=str(dyn_args.get("space_mode", "wm_agent_isolated")),
        scale_pos_embeds=bool(dyn_args.get("scale_pos_embeds", True)),
    ).to(device)
    
    dyn.load_state_dict(ckpt["dynamics"])
    
    return dyn, dyn_args, ckpt


def compute_sharp_loss(z1_hat, z1_clean, beta_mean, beta_var):
    """SHARP loss on V4: penalize mean + variance of denoising error."""
    # z1_hat: (B, T, Sz, Dz), z1_clean: (B, T, Sz, Dz)
    eps = (z1_hat.float() - z1_clean.float())  # (B, T, Sz, Dz)
    
    # Flatten spatial dims for statistics
    eps_flat = eps.reshape(eps.shape[0], -1)  # (B, T*Sz*Dz)
    
    # Mean across batch
    bias = eps_flat.mean(dim=0)  # (T*Sz*Dz,)
    L_mean = bias.pow(2).mean()
    
    # Variance across batch
    L_var = eps_flat.var(dim=0).mean()
    
    L_sharp = beta_mean * L_mean + beta_var * L_var
    
    return L_sharp, {
        "sharp_mean_loss": float(L_mean.detach().cpu()),
        "sharp_var_loss": float(L_var.detach().cpu()),
    }


def main():
    args = parse_args()
    device = torch.device(args.device)
    seed_everything(args.seed)
    
    print("=" * 70)
    print("DreamerV4 Dynamics + SHARP Fine-tune")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"beta_mean={args.beta_mean}, beta_var={args.beta_var}")
    print(f"warmup_steps={args.warmup_steps}, max_steps={args.max_steps}")
    print("=" * 70, flush=True)
    
    # Load tokenizer (frozen)
    print("Loading tokenizer...", flush=True)
    enc, tok_args = load_tokenizer(args.tokenizer_ckpt, device)
    
    H = int(tok_args.get("H", 128))
    W = int(tok_args.get("W", 128))
    C = int(tok_args.get("C", 3))
    patch = int(tok_args.get("patch", 4))
    n_latents = int(tok_args.get("n_latents", 16))
    d_bottleneck = int(tok_args.get("d_bottleneck", 32))
    
    # Load dynamics (trainable)
    print("Loading dynamics...", flush=True)
    dyn, dyn_args, dyn_ckpt = load_dynamics(args.dynamics_ckpt, device, tok_args)
    
    packing_factor = int(dyn_args.get("packing_factor", 2))
    n_spatial = n_latents // packing_factor
    k_max = int(dyn_args.get("k_max", 8))
    
    start_step = dyn_ckpt.get("step", 0)
    print(f"Resumed dynamics from step {start_step}", flush=True)
    
    n_params = sum(p.numel() for p in dyn.parameters())
    print(f"Dynamics params: {n_params:,}", flush=True)
    
    # Optimizer
    opt = torch.optim.AdamW(dyn.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler("cuda", enabled=args.use_amp)
    
    # Dataset
    print("Loading dataset...", flush=True)
    dataset = ShardedFrameDataset(
        outdirs=args.frame_dirs,
        seq_len=args.seq_len,
        tasks=TASK_SET,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    
    # Training loop
    logdir = Path(args.logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    
    step = 0
    dyn.train()
    
    from train_dynamics import dynamics_pretrain_loss, _emax_from_kmax
    
    print(f"Starting fine-tune for {args.max_steps} steps...", flush=True)
    
    for epoch in range(9999):
        for batch in loader:
            if step >= args.max_steps:
                break
            
            # ShardedFrameDataset returns raw tensor (B,T,C,H,W)
            if isinstance(batch, dict):
                frames = batch["frames"].to(device, non_blocking=True)
            else:
                frames = batch.to(device, non_blocking=True)
            actions = None
            act_mask = None
            
            if frames.dtype == torch.uint8:
                frames = frames.float() / 255.0
            
            # Frozen tokenizer
            with torch.no_grad():
                patches = temporal_patchify(frames, patch)
                z_btLd, _ = enc(patches)
                z1 = pack_bottleneck_to_spatial(z_btLd, n_spatial=n_spatial, k=packing_factor)
            
            B = z1.shape[0]
            B_self = int(round(0.25 * B))
            B_self = max(0, min(B - 1, B_self))
            
            with autocast(device_type="cuda", enabled=args.use_amp):
                # Original flow matching loss
                loss_flow, aux = dynamics_pretrain_loss(
                    dyn,
                    z1=z1,
                    actions=actions,
                    act_mask=act_mask,
                    k_max=k_max,
                    B_self=B_self,
                    step=start_step + step,
                    bootstrap_start=dyn_args.get("bootstrap_start", 5000),
                    agent_tokens=None,
                )
                
                # SHARP loss (after warmup)
                sharp_metrics = {"sharp_mean_loss": 0.0, "sharp_var_loss": 0.0}
                if step >= args.warmup_steps:
                    # Get clean prediction for SHARP
                    emax = _emax_from_kmax(k_max)
                    step_idx = torch.full((B, z1.shape[1]), emax, device=device, dtype=torch.long)
                    # At finest level (sigma ~= 1), prediction should match z1
                    sigma_idx = torch.full((B, z1.shape[1]), k_max, device=device, dtype=torch.long)
                    
                    z1_hat, _ = dyn(actions, step_idx, sigma_idx, z1, act_mask=act_mask)
                    L_sharp, sharp_metrics = compute_sharp_loss(
                        z1_hat, z1, args.beta_mean, args.beta_var
                    )
                    loss = loss_flow + L_sharp
                else:
                    loss = loss_flow
            
            opt.zero_grad()
            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(dyn.parameters(), max_norm=args.grad_clip)
            scaler.step(opt)
            scaler.update()
            
            if step % args.log_every == 0:
                print(
                    f"[{step:06d}] loss={float(loss):.4f} "
                    f"flow={float(loss_flow):.4f} "
                    f"sharp_mean={sharp_metrics['sharp_mean_loss']:.4f} "
                    f"sharp_var={sharp_metrics['sharp_var_loss']:.4f} "
                    f"flow_mse={float(aux['flow_mse']):.6f}",
                    flush=True,
                )
            
            if step > 0 and step % args.save_every == 0:
                ckpt_path = logdir / "latest.pt"
                torch.save({
                    "step": start_step + step,
                    "dynamics": dyn.state_dict(),
                    "opt": opt.state_dict(),
                    "scaler": scaler.state_dict(),
                    "args": vars(args),
                }, ckpt_path)
                print(f"Saved checkpoint at step {step}", flush=True)
            
            step += 1
        
        if step >= args.max_steps:
            break
    
    # Final save
    ckpt_path = logdir / "latest.pt"
    torch.save({
        "step": start_step + step,
        "dynamics": dyn.state_dict(),
        "opt": opt.state_dict(),
        "scaler": scaler.state_dict(),
        "args": vars(args),
    }, ckpt_path)
    print(f"Done. Final checkpoint at step {step}", flush=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tokenizer_ckpt", type=str, 
                    default=str(PROJECT_ROOT / "external/dreamer4/checkpoints/tokenizer.pt"))
    p.add_argument("--dynamics_ckpt", type=str,
                    default=str(PROJECT_ROOT / "external/dreamer4/checkpoints/dynamics.pt"))
    p.add_argument("--frame_dirs", type=str, nargs="+",
                    default=[str(PROJECT_ROOT / "external/dreamer4/data/expert-shards")])
    p.add_argument("--logdir", type=str,
                    default=str(PROJECT_ROOT / "results/v4_sharp"))
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--seq_len", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--max_steps", type=int, default=5000)
    p.add_argument("--warmup_steps", type=int, default=500)
    p.add_argument("--beta_mean", type=float, default=1.0)
    p.add_argument("--beta_var", type=float, default=0.1)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--use_amp", action="store_true")
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--save_every", type=int, default=1000)
    return p.parse_args()


if __name__ == "__main__":
    main()
