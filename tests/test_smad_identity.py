from __future__ import annotations

import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_DREAMERV3 = REPO_ROOT / "external" / "dreamerv3-torch"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(EXTERNAL_DREAMERV3))

import networks  # noqa: E402

from src.smad.img_step_patch import patch_img_step  # noqa: E402


BATCH_SIZE = 4
ACTION_DIM = 6
HORIZON = 5
DETER_DIM = 512


def make_rssm() -> networks.RSSM:
    return networks.RSSM(
        stoch=32,
        deter=DETER_DIM,
        hidden=512,
        rec_depth=1,
        discrete=32,
        act="SiLU",
        norm=True,
        mean_act="none",
        std_act="softplus",
        min_std=0.1,
        unimix_ratio=0.01,
        initial="learned",
        num_actions=ACTION_DIM,
        embed=1024,
        device="cpu",
    )


def make_projector(rank: int = 10) -> torch.Tensor:
    matrix = torch.randn(DETER_DIM, rank)
    q, _ = torch.linalg.qr(matrix, mode="reduced")
    return q @ q.T


def clone_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.clone() for key, value in state.items()}


def run_img_steps(
    rssm: networks.RSSM,
    initial_state: dict[str, torch.Tensor],
    actions: torch.Tensor,
    *,
    sample_seed: int,
) -> list[dict[str, torch.Tensor]]:
    torch.manual_seed(sample_seed)
    state = clone_state(initial_state)
    outputs = []
    with torch.no_grad():
        for action in actions:
            state = rssm.img_step(state, action)
            outputs.append(clone_state(state))
    return outputs


def assert_outputs_equal(
    actual: list[dict[str, torch.Tensor]],
    expected: list[dict[str, torch.Tensor]],
) -> None:
    assert len(actual) == len(expected)
    for step, (actual_state, expected_state) in enumerate(zip(actual, expected)):
        for key in ("deter", "stoch"):
            if not torch.equal(actual_state[key], expected_state[key]):
                raise AssertionError(f"{key} differs at step {step}")


def any_deter_differs(
    actual: list[dict[str, torch.Tensor]],
    expected: list[dict[str, torch.Tensor]],
) -> bool:
    return any(
        not torch.equal(actual_state["deter"], expected_state["deter"])
        for actual_state, expected_state in zip(actual, expected)
    )


def make_fixture() -> tuple[networks.RSSM, dict[str, torch.Tensor], torch.Tensor]:
    torch.manual_seed(42)
    rssm = make_rssm()
    initial_state = rssm.initial(BATCH_SIZE)
    actions = torch.randn(HORIZON, BATCH_SIZE, ACTION_DIM)
    return rssm, initial_state, actions


def test_eta_zero_identity_and_restore() -> None:
    rssm, initial_state, actions = make_fixture()
    P = make_projector()

    original = run_img_steps(rssm, initial_state, actions, sample_seed=123)
    restore_fn = patch_img_step(rssm, P, eta=0.0)
    patched = run_img_steps(rssm, initial_state, actions, sample_seed=123)
    assert_outputs_equal(patched, original)

    restore_fn()
    restored = run_img_steps(rssm, initial_state, actions, sample_seed=123)
    assert_outputs_equal(restored, original)
    print("IDENTITY TEST PASSED")


def test_eta_positive_changes_deter() -> None:
    rssm, initial_state, actions = make_fixture()
    P = make_projector()

    original = run_img_steps(rssm, initial_state, actions, sample_seed=123)
    restore_fn = patch_img_step(rssm, P, eta=0.3)
    damped = run_img_steps(rssm, initial_state, actions, sample_seed=123)
    restore_fn()

    if not any_deter_differs(damped, original):
        raise AssertionError("eta=0.3 damping did not change any deter values")
    print("NON-IDENTITY TEST PASSED")


if __name__ == "__main__":
    test_eta_zero_identity_and_restore()
    test_eta_positive_changes_deter()
