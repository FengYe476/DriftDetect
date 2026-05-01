"""Tests for DreamerV4Adapter. Requires CUDA and V4 checkpoints."""

from __future__ import annotations

import importlib.util
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - local non-V4 env.
    torch = None


pytestmark = pytest.mark.skipif(
    torch is None or not torch.cuda.is_available(),
    reason="V4 requires CUDA",
)

HF_REPO_ID = "nicklashansen/dreamer4"
HF_REVISION = "10e04fce900745e266ab68925b6779831551f157"
REPO_ROOT = Path(__file__).resolve().parents[1]


def load_adapter_class():
    path = REPO_ROOT / "src" / "models" / "dreamerv4_adapter.py"
    spec = importlib.util.spec_from_file_location("_driftdetect_dreamerv4_adapter", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load DreamerV4Adapter from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.DreamerV4Adapter


DreamerV4Adapter = (
    load_adapter_class() if torch is not None and torch.cuda.is_available() else Any
)


@pytest.fixture(scope="session")
def v4_adapter() -> DreamerV4Adapter:
    checkpoint_dir = os.environ.get("DREAMERV4_CHECKPOINT_DIR")
    if checkpoint_dir is None:
        from huggingface_hub import hf_hub_download

        repo_id = os.environ.get("DREAMERV4_HF_REPO_ID", HF_REPO_ID)
        revision = os.environ.get("DREAMERV4_HF_REVISION", HF_REVISION)
        tok = hf_hub_download(repo_id=repo_id, filename="tokenizer.pt", revision=revision)
        dyn = hf_hub_download(repo_id=repo_id, filename="dynamics.pt", revision=revision)
        assert Path(tok).parent == Path(dyn).parent
        checkpoint_dir = str(Path(tok).parent)

    adapter = DreamerV4Adapter()
    adapter.load_checkpoint(checkpoint_dir)
    return adapter


def random_frames(length: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random((length, 3, 128, 128), dtype=np.float32)


def zero_actions(total: int) -> tuple[np.ndarray, np.ndarray]:
    actions = np.zeros((total, 16), dtype=np.float32)
    mask = np.ones((total, 16), dtype=np.float32)
    mask[0] = 0.0
    return actions, mask


class TestV4AdapterSmoke:
    """Minimal smoke tests - load checkpoint, encode one frame, imagine one step, decode."""

    def test_checkpoint_loads(self, v4_adapter: DreamerV4Adapter) -> None:
        """Checkpoint loads without error, config attributes are set."""
        assert v4_adapter.tokenizer is not None
        assert v4_adapter.dynamics is not None
        assert v4_adapter.k_max == 8
        assert v4_adapter.n_latents == 16
        assert v4_adapter.d_bottleneck == 32
        assert v4_adapter.n_spatial == 8
        assert v4_adapter.d_spatial == 64

    def test_encode_single_frame(self, v4_adapter: DreamerV4Adapter) -> None:
        """Encode a random (1, 3, 128, 128) frame, check output shapes."""
        encoded = v4_adapter.encode(random_frames(1, seed=1))
        assert encoded["z_btLd"].shape == (1, 16, 32)
        assert encoded["z_packed"].shape == (1, 8, 64)
        assert encoded["flat"].shape == (1, 512)
        assert np.isfinite(encoded["flat"]).all()

    def test_imagine_one_step(self, v4_adapter: DreamerV4Adapter) -> None:
        """Imagine 1 step from 1 context frame, check latent shape is (1, 512)."""
        actions, mask = zero_actions(total=2)
        imagined = v4_adapter.imagine(random_frames(1, seed=2), actions, mask, horizon=1)
        assert imagined["imagined_latent"].shape == (1, 512)
        assert imagined["imagined_packed"].shape == (1, 8, 64)
        assert np.isfinite(imagined["imagined_latent"]).all()

    def test_decode_roundtrip(self, v4_adapter: DreamerV4Adapter) -> None:
        """Encode then decode a frame, check output is (1, 3, 128, 128) in [0, 1]."""
        encoded = v4_adapter.encode(random_frames(1, seed=3))
        decoded = v4_adapter.decode(encoded["z_packed"])
        assert decoded.shape == (1, 3, 128, 128)
        assert decoded.dtype == np.float32
        assert float(decoded.min()) >= 0.0
        assert float(decoded.max()) <= 1.0

    def test_no_nans(self, v4_adapter: DreamerV4Adapter) -> None:
        """All outputs (latent, decoded frame) are finite, no NaN/Inf."""
        encoded = v4_adapter.encode(random_frames(1, seed=4))
        decoded = v4_adapter.decode(encoded["z_packed"])
        assert np.isfinite(encoded["z_btLd"]).all()
        assert np.isfinite(encoded["z_packed"]).all()
        assert np.isfinite(encoded["flat"]).all()
        assert np.isfinite(decoded).all()


class TestV4AdapterShapes:
    """Shape tests for longer sequences."""

    def test_imagine_horizon_10(self, v4_adapter: DreamerV4Adapter) -> None:
        """Imagine 10 steps from 8 context frames, check shapes."""
        horizon = 10
        context = 8
        actions, mask = zero_actions(total=context + horizon)
        imagined = v4_adapter.imagine(
            random_frames(context, seed=5),
            actions,
            mask,
            horizon=horizon,
        )
        assert imagined["imagined_latent"].shape == (horizon, 512)
        assert imagined["imagined_packed"].shape == (horizon, 8, 64)
        assert np.isfinite(imagined["imagined_latent"]).all()

    def test_imagine_horizon_200(self, v4_adapter: DreamerV4Adapter) -> None:
        """Imagine 200 steps from 8 context frames. This is the target horizon."""
        horizon = 200
        context = 8
        actions, mask = zero_actions(total=context + horizon)

        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        imagined = v4_adapter.imagine(
            random_frames(context, seed=6),
            actions,
            mask,
            horizon=horizon,
        )
        elapsed = time.perf_counter() - start
        peak_gb = torch.cuda.max_memory_allocated() / (1024**3)
        print(f"V4 horizon={horizon} elapsed={elapsed:.2f}s peak_vram={peak_gb:.2f}GB")

        assert imagined["imagined_latent"].shape == (horizon, 512)
        assert imagined["imagined_packed"].shape == (horizon, 8, 64)
        assert np.isfinite(imagined["imagined_latent"]).all()
        assert np.isfinite(imagined["imagined_packed"]).all()
