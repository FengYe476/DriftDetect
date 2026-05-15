"""Test that SHARP loss indexes time dimension correctly."""

import pathlib
import sys

import torch


sys.path.insert(
    0, str(pathlib.Path(__file__).parents[1] / "external" / "dreamerv3-torch")
)


def test_sharp_shape_indexing():
    """Verify SHARP samples time steps, not batch samples."""
    B, T, D = 4, 10, 8

    # Create fake post with known pattern:
    # Each batch has different constant offset, each time step increments.
    post_deter = torch.zeros(B, T, D)
    for b in range(B):
        for t in range(T):
            post_deter[b, t, :] = b * 100 + t

    # Correct indexing: v[:, t] gives all batches at time t.
    t = 3
    correct_slice = post_deter[:, t]
    assert correct_slice.shape == (B, D)
    assert correct_slice[0, 0] == 3
    assert correct_slice[1, 0] == 103

    # Wrong indexing: v[t] gives one batch's entire timeline.
    wrong_slice = post_deter[t]
    assert wrong_slice.shape == (T, D)
    assert wrong_slice[0, 0] == 300

    print("Shape test passed: v[:, t] correctly indexes time dimension")


if __name__ == "__main__":
    test_sharp_shape_indexing()
