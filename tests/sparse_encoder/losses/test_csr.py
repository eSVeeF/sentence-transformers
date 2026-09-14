from __future__ import annotations

import pytest
import torch

from sentence_transformers.sparse_encoder.losses.csr import normalized_mean_squared_error


@pytest.mark.parametrize(
    "original_input",
    [
        torch.tensor([[1.0, 2.0, 3.0]]),
        torch.ones(2, 3),
    ],
    ids=["batch_size_one", "constant_batch"],
)
def test_normalized_mean_squared_error_is_finite_for_zero_variance_batches(
    original_input: torch.Tensor,
) -> None:
    # A singleton or constant batch has zero variance, so the normalization denominator
    # needs the same finite-loss safeguard used during CSR auxiliary reconstruction.
    reconstruction = original_input + 0.5

    loss = normalized_mean_squared_error(reconstruction, original_input)

    assert torch.isfinite(loss)


@pytest.mark.parametrize(
    "original_input",
    [
        torch.tensor([[1.0, 2.0, 3.0]]),
        torch.ones(2, 3),
    ],
    ids=["batch_size_one", "constant_batch"],
)
def test_normalized_mean_squared_error_is_zero_for_perfect_reconstruction(
    original_input: torch.Tensor,
) -> None:
    # Clamping the denominator must prevent non-finite values without changing this baseline.
    loss = normalized_mean_squared_error(original_input, original_input)

    assert loss == 0