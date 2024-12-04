# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from chronos.utils import left_pad_and_stack_1D


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
