# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from chronos.utils import interpolate_quantiles, left_pad_and_stack_1D


@pytest.mark.parametrize(
    "tensors",
    [
        [
            torch.tensor([2.0, 3.0], dtype=dtype),
            torch.tensor([4.0, 5.0, 6.0], dtype=dtype),
            torch.tensor([7.0, 8.0, 9.0, 10.0], dtype=dtype),
        ]
        for dtype in [torch.int, torch.float16, torch.float32]
    ],
)
def test_pad_and_stack(tensors: list):
    stacked_and_padded = left_pad_and_stack_1D(tensors)

    assert stacked_and_padded.dtype == torch.float32
    assert stacked_and_padded.shape == (len(tensors), max(len(t) for t in tensors))

    ref = torch.concat(tensors).to(dtype=stacked_and_padded.dtype)

    assert torch.sum(torch.nan_to_num(stacked_and_padded, nan=0)) == torch.sum(ref)


@pytest.mark.parametrize(
    "query_quantiles, orig_quantiles, orig_values, expected_values",
    [
        (
            [0.01, 0.1, 0.15, 0.2, 0.8, 0.87, 0.9, 0.99],
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            torch.arange(1, 10, dtype=torch.float32),
            torch.tensor([1.0, 1.0, 1.5, 2.0, 8.0, 8.7, 9.0, 9.0]),
        ),
        (
            torch.tensor([0.01, 0.1, 0.15, 0.2, 0.5, 0.8, 0.87, 0.9, 0.999]),
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.85, 0.9],
            torch.arange(1, 10, dtype=torch.float32),
            torch.tensor([1.0, 1.0, 1.5, 2.0, 5.0, 23 / 3, 8.4, 9.0, 9.0]),
        ),
        (
            torch.tensor([0.01, 0.1, 0.2, 0.5, 0.9, 0.97]),
            torch.tensor([0.05, 0.25, 0.5, 0.8, 0.95]),
            torch.tensor(
                [
                    [10.0, 20.0, 30.0, 40.0, 50.0],
                    [110.0, 125.0, 150.0, 180.0, 210.0],
                ]
            ),
            torch.tensor(
                [
                    [10.0, 12.5, 17.5, 30.0, 140 / 3, 50.0],
                    [110.0, 113.75, 121.25, 150.0, 200.0, 210.0],
                ]
            ),
        ),
    ],
)
def test_interpolate_quantiles(query_quantiles, orig_quantiles, orig_values, expected_values):
    output_values = interpolate_quantiles(query_quantiles, orig_quantiles, orig_values)
    assert output_values.dtype == torch.float32
    assert torch.allclose(output_values, expected_values)
