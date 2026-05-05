from __future__ import annotations

import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.smad.correction_head import (  # noqa: E402
    CorrectionHead,
    apply_correction,
)


DETER_DIM = 512


def test_output_shape() -> None:
    torch.manual_seed(42)
    head = CorrectionHead(deter_dim=DETER_DIM)
    h_deter = torch.randn(4, DETER_DIM)

    correction = head(h_deter, step=7)

    assert correction.shape == (4, DETER_DIM)


def test_zero_init() -> None:
    torch.manual_seed(42)
    head = CorrectionHead(deter_dim=DETER_DIM)
    h_deter = torch.randn(4, DETER_DIM)

    correction = head(h_deter, step=25)

    assert head.gate_value == 0.0
    assert torch.equal(correction, torch.zeros_like(correction))


def test_step_sensitivity() -> None:
    torch.manual_seed(42)
    head = CorrectionHead(deter_dim=64, step_embed_dim=16, hidden_dim=32)
    h_deter = torch.randn(3, 64)
    with torch.no_grad():
        head.gate.fill_(1.0)
        for parameter in head.parameters():
            if parameter.ndim > 0:
                parameter.add_(0.01 * torch.randn_like(parameter))

    correction_step_1 = head(h_deter, step=1)
    correction_step_50 = head(h_deter, step=50)

    assert not torch.allclose(correction_step_1, correction_step_50)


def test_apply_correction() -> None:
    torch.manual_seed(42)
    head = CorrectionHead(deter_dim=64, step_embed_dim=16, hidden_dim=32)
    with torch.no_grad():
        head.gate.fill_(1.0)
    deter = torch.randn(5, 64)
    stoch = torch.randn(5, 32, 32)
    logit = torch.randn(5, 32, 32)
    state = {"deter": deter, "stoch": stoch, "logit": logit}

    corrected = apply_correction(state, head, step=10)

    assert corrected is not state
    assert corrected["stoch"] is stoch
    assert corrected["logit"] is logit
    assert not torch.allclose(corrected["deter"], deter)
    assert torch.equal(state["deter"], deter)
