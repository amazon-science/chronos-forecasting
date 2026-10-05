import pytest
import torch

from chronos.chronos_bolt import InstanceNorm


@pytest.mark.parametrize("magnitude", [1.0, 5e19])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_instance_norm_large_values(magnitude, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    x = torch.tensor(
        [[1, 2, 3, torch.nan], [2, 2, torch.nan, torch.nan], [torch.nan] * 4], device=device
    ) * magnitude
    norm = InstanceNorm()
    normalized, (loc, scale) = norm(x)

    reference = x.double()
    expected_loc = reference.nanmean(dim=-1, keepdim=True).nan_to_num(nan=0.0)
    expected_scale = (reference - expected_loc).square().nanmean(dim=-1, keepdim=True).sqrt().nan_to_num(nan=1.0)
    expected_scale = torch.where(expected_scale == 0, norm.eps, expected_scale)

    torch.testing.assert_close(loc, expected_loc.float())
    torch.testing.assert_close(scale, expected_scale.float())
    torch.testing.assert_close(normalized, ((reference - expected_loc) / expected_scale).float(), equal_nan=True)
    torch.testing.assert_close(norm.inverse(normalized, (loc, scale)), x, equal_nan=True)
