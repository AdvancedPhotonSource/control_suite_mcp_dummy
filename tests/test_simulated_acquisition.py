"""Tests for cameraman pixel sampling and zero padding."""

import numpy as np
import pytest

from control_suite_mcp_dummy.simulated_acquisition import SimulatedInstrument
from control_suite_mcp_dummy.worker import load_image


def test_default_image_is_cameraman() -> None:
    image = load_image(None)

    assert image.shape == (512, 512)
    assert image.dtype == np.float32
    assert image[0, 0] == 200
    assert image[256, 256] == 14
    assert image[511, 511] == 149


@pytest.mark.parametrize(
    ("y_center", "x_center", "size_y", "size_x", "scan_step", "rows", "columns"),
    [
        (256, 256, 512, 512, 1, slice(None), slice(None)),
        (120, 230, 8, 12, 1, slice(116, 124), slice(224, 236)),
        (120, 230, 8, 12, 2, slice(116, 124, 2), slice(224, 236, 2)),
    ],
)
def test_acquisition_uses_pixel_indices(
    tmp_path, y_center, x_center, size_y, size_x, scan_step, rows, columns
) -> None:
    image = load_image(None)
    instrument = SimulatedInstrument(whole_image=image, output_dir=tmp_path)

    result = instrument.acquire_image(y_center, x_center, size_y, size_x, scan_step)

    np.testing.assert_array_equal(np.load(result["array_path"]), image[rows, columns])
    assert result["psize"] == scan_step


@pytest.mark.parametrize("center", [0, 512, -10, 522])
@pytest.mark.parametrize(
    "effects",
    [{}, {"blur": 1}, {"gaussian_psf_sigma": 1, "poisson_noise_scale": 100, "scan_jitter": 0}],
)
def test_acquisition_zero_pads_outside_source(tmp_path, center, effects) -> None:
    image = load_image(None)
    instrument = SimulatedInstrument(whole_image=image, output_dir=tmp_path, **effects)

    result = instrument.acquire_image(center, center, 4, 4)
    acquired = np.load(result["array_path"])

    indices = np.arange(center - 2, center + 2)
    valid = (indices >= 0) & (indices < 512)
    inside = valid[:, None] & valid[None, :]
    np.testing.assert_array_equal(acquired[~inside], 0)
    if not effects:
        expected = np.zeros((4, 4))
        expected[np.ix_(valid, valid)] = image[np.ix_(indices[valid], indices[valid])]
        np.testing.assert_array_equal(acquired, expected)
