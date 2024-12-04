from typing import Optional, Tuple

import torch


def validate_tensor(
    a: torch.Tensor, shape: Tuple[int, ...], dtype: Optional[torch.dtype] = None
) -> None:
    assert isinstance(a, torch.Tensor)
    assert a.shape == shape

    if dtype is not None:
        assert a.dtype == dtype
