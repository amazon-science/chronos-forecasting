# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sampling a short window must not retain a copy of the complete history."""

import numpy as np
import pytest
import torch

from chronos.chronos2.dataset import Chronos2Dataset, DatasetMode


@pytest.mark.parametrize("mode", list(DatasetMode))
@pytest.mark.parametrize("history_length", [32, 10000])
def test_sampled_window_has_bounded_storage_and_independent_values(mode, history_length, monkeypatch):
    prediction_length, context_length = 8, 64
    past = torch.arange(4 * history_length, dtype=torch.float32).reshape(4, history_length)
    past[0, 3] = torch.nan
    future = torch.full((4, prediction_length), torch.nan)
    future[-1] = torch.arange(prediction_length)
    prepared = {
        "context": past,
        "future_covariates": future,
        "n_targets": 2,
        "n_covariates": 2,
        "n_future_covariates": 1,
    }
    past_before, future_before = past.clone(), future.clone()
    dataset = Chronos2Dataset(
        [prepared], context_length, prediction_length, batch_size=4, output_patch_size=8, mode=mode
    )
    if mode == DatasetMode.TRAIN:
        slice_idx = min(history_length - prediction_length, 96)
        monkeypatch.setattr(np.random, "randint", lambda *args, **kwargs: slice_idx)
    elif mode == DatasetMode.VALIDATION:
        slice_idx = history_length - prediction_length
    else:
        slice_idx = history_length

    context, target, covariates, n_targets = dataset._construct_slice(0)

    assert n_targets == 2
    torch.testing.assert_close(
        context, past[:, max(0, slice_idx - context_length) : slice_idx], rtol=0, atol=0, equal_nan=True
    )
    # Account for the underlying allocation, not just the view's visible shape.
    assert context.untyped_storage().nbytes() == context.numel() * context.element_size()
    if mode == DatasetMode.TEST:
        assert target is None
        torch.testing.assert_close(covariates, future, rtol=0, atol=0, equal_nan=True)
    else:
        torch.testing.assert_close(target[:2], past[:2, slice_idx : slice_idx + prediction_length], rtol=0, atol=0)
        assert torch.isnan(target[2:]).all()
        assert torch.isnan(covariates[:-1]).all()
        torch.testing.assert_close(covariates[-1], past[-1, slice_idx : slice_idx + prediction_length], rtol=0, atol=0)
        target.fill_(-999)
    context.fill_(-999)
    covariates.fill_(-999)
    torch.testing.assert_close(past, past_before, rtol=0, atol=0, equal_nan=True)
    torch.testing.assert_close(future, future_before, rtol=0, atol=0, equal_nan=True)
