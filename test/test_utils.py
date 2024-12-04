import pytest
import torch

from chronos.utils import left_pad_and_stack_1D


@pytest.mark.parametrize("tensors", [
    list(map(torch.tensor, [[1, 2, 3], [5, 6]])),
    list(map(torch.tensor, [[2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0, 10.0]])),
])
def test_pad_and_stack(tensors: list):
    stacked_and_padded = left_pad_and_stack_1D(tensors)
    assert stacked_and_padded.shape == (len(tensors), max(len(t) for t in tensors))

    ref = torch.concat(tensors).to(dtype=stacked_and_padded.dtype)

    assert torch.sum(stacked_and_padded) == torch.sum(ref)
    assert torch.nanmean(stacked_and_padded) == torch.nanmean(ref)
