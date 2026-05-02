from __future__ import annotations

import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.smad.anchor_loss import activation_schedule, compute_anchor_loss  # noqa: E402


DETER_DIM = 512
RANK = 10


def make_U() -> torch.Tensor:
    torch.manual_seed(0)
    q, _ = torch.linalg.qr(torch.randn(DETER_DIM, RANK), mode="reduced")
    return q


def test_zero_beta_or_gamma_returns_exact_zero() -> None:
    torch.manual_seed(1)
    imag = torch.randn(6, 3, DETER_DIM, requires_grad=True)
    post = torch.randn(10, 3, DETER_DIM, requires_grad=True)
    starts = torch.tensor([0, 1, 2])
    U = make_U()

    loss_beta, metrics_beta = compute_anchor_loss(
        imag, post, starts, U, [1, 3, 5], beta=0.0, gamma=1.0
    )
    loss_gamma, metrics_gamma = compute_anchor_loss(
        imag, post, starts, U, [1, 3, 5], beta=1.0, gamma=0.0
    )

    assert torch.equal(loss_beta, torch.tensor(0.0))
    assert torch.equal(loss_gamma, torch.tensor(0.0))
    assert metrics_beta["anchor_n_valid"] == 0
    assert metrics_gamma["anchor_n_valid"] == 0


def test_identity_at_anchor_points_returns_zero() -> None:
    torch.manual_seed(2)
    H, T, B = 6, 12, 4
    post = torch.randn(T, B, DETER_DIM)
    imag = torch.randn(H, B, DETER_DIM)
    starts = torch.tensor([0, 1, 2, 3])
    anchors = [1, 3, 5]
    for k in anchors:
        for b in range(B):
            imag[k, b] = post[starts[b] + k, b]

    loss, metrics = compute_anchor_loss(
        imag, post, starts, make_U(), anchors, beta=1.0, gamma=1.0
    )

    assert torch.equal(loss, torch.tensor(0.0))
    assert metrics["anchor_n_valid"] == len(anchors) * B


def test_random_inputs_produce_positive_loss() -> None:
    torch.manual_seed(3)
    imag = torch.randn(6, 3, DETER_DIM)
    post = torch.randn(10, 3, DETER_DIM)
    starts = torch.tensor([0, 1, 2])

    loss, metrics = compute_anchor_loss(
        imag, post, starts, make_U(), [1, 3, 5], beta=0.5, gamma=0.8
    )

    assert loss.item() > 0.0
    assert metrics["anchor_loss_raw"].item() > 0.0
    assert metrics["anchor_loss_scaled"].item() > 0.0


def test_gradients_flow_to_imag_only() -> None:
    torch.manual_seed(4)
    imag = torch.randn(6, 3, DETER_DIM, requires_grad=True)
    post = torch.randn(10, 3, DETER_DIM, requires_grad=True)
    starts = torch.tensor([0, 1, 2])

    loss, _ = compute_anchor_loss(
        imag, post, starts, make_U(), [1, 3, 5], beta=1.0, gamma=1.0
    )
    loss.backward()

    assert imag.grad is not None
    assert torch.any(imag.grad != 0)
    assert post.grad is None


def test_out_of_bounds_post_indices_are_masked() -> None:
    torch.manual_seed(5)
    H, T, B = 4, 5, 4
    imag = torch.randn(H, B, DETER_DIM)
    post = torch.randn(T, B, DETER_DIM)
    starts = torch.tensor([0, 3, 4, 10])

    loss, metrics = compute_anchor_loss(
        imag, post, starts, make_U(), [1, 2, 3], beta=1.0, gamma=1.0
    )

    assert loss.item() >= 0.0
    assert metrics["anchor_n_valid"] == 4


def test_no_valid_anchors_returns_zero() -> None:
    torch.manual_seed(6)
    imag = torch.randn(4, 2, DETER_DIM)
    post = torch.randn(5, 2, DETER_DIM)
    starts = torch.tensor([4, 5])

    loss, metrics = compute_anchor_loss(
        imag, post, starts, make_U(), [1, 2, 10], beta=1.0, gamma=1.0
    )

    assert torch.equal(loss, torch.tensor(0.0))
    assert metrics["anchor_n_valid"] == 0


def test_activation_schedule() -> None:
    assert activation_schedule(5, s0=10, s1=20) == 0.0
    assert activation_schedule(15, s0=10, s1=20) == 0.5
    assert activation_schedule(25, s0=10, s1=20) == 1.0
