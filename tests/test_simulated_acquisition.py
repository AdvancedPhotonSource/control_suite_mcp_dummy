"""Tests for cameraman pixel sampling and zero padding."""

import numpy as np
import pytest

from control_suite_mcp_dummy.simulated_acquisition import SimulatedInstrument
from control_suite_mcp_dummy import simulated_acquisition
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


def test_anisotropic_pixels_and_fractional_region(tmp_path) -> None:
    image = np.arange(10000, dtype=np.float32).reshape(100, 100)
    instrument = SimulatedInstrument(
        whole_image=image, output_dir=tmp_path, pixel_size_y=0.2, pixel_size_x=0.1
    )

    result = instrument.acquire_image(10.25, 5.25, 0.5, 0.5, scan_step=0.05)

    rows = 50 + np.arange(10) * 0.25
    columns = 50 + np.arange(10) * 0.5
    expected = rows[:, None] * 100 + columns[None, :]
    np.testing.assert_allclose(np.load(result["array_path"]), expected)
    assert result["psize"] == 0.05


@pytest.mark.parametrize("angle", [0, 45, 90])
def test_line_scan_uses_configured_pixel_sizes(tmp_path, monkeypatch, angle) -> None:
    image = np.arange(10000, dtype=np.float32).reshape(100, 100)
    instrument = SimulatedInstrument(
        whole_image=image, output_dir=tmp_path, pixel_size_y=0.2, pixel_size_x=0.1
    )
    samples = {}

    def capture_fit(x, y, **kwargs):
        samples["distances"] = x
        samples["values"] = y
        return (np.nan,) * 7

    monkeypatch.setattr(simulated_acquisition, "fit_gaussian_1d", capture_fit)
    instrument.acquire_line_scan(5.5, 10.5, 1.0, 0.1, angle=angle)

    distance_from_center = samples["distances"] - 0.5
    rows = (10.5 + distance_from_center * np.sin(np.radians(angle))) / 0.2
    columns = (5.5 + distance_from_center * np.cos(np.radians(angle))) / 0.1
    np.testing.assert_allclose(samples["values"], rows * 100 + columns)


@pytest.mark.parametrize("name", ["pixel_size_x", "pixel_size_y"])
@pytest.mark.parametrize("value", [0, -1, np.nan, np.inf])
def test_pixel_sizes_must_be_positive_and_finite(tmp_path, name, value) -> None:
    with pytest.raises(ValueError, match=f"{name} must be finite and positive"):
        SimulatedInstrument(whole_image=np.ones((2, 2)), output_dir=tmp_path, **{name: value})


@pytest.mark.parametrize("effects", [{}, {"blur": 1}, {"gaussian_psf_sigma": 1}])
def test_scaled_source_bounds_and_psf(tmp_path, effects) -> None:
    image = np.arange(1, 13, dtype=np.float32).reshape(3, 4)
    unscaled = SimulatedInstrument(whole_image=image, output_dir=tmp_path, **effects)
    scaled = SimulatedInstrument(
        whole_image=image, output_dir=tmp_path, pixel_size_y=0.2, pixel_size_x=0.1, **effects
    )
    rows = np.array([-1, 0, 1, 2, 3, 1, 1, 1, 1])
    columns = np.array([1, 1, 1, 1, 1, -1, 0, 3, 4])

    values = scaled.sample(rows * 0.2, columns * 0.1)

    np.testing.assert_allclose(values, unscaled.sample(rows, columns))
    np.testing.assert_array_equal(values[[0, 4, 5, 8]], 0)


@pytest.mark.parametrize("show_axes", [True, False])
@pytest.mark.parametrize("scale", [1, 0.001])
def test_image_axis_labels_use_acquisition_positions(tmp_path, monkeypatch, show_axes, scale):
    instrument = SimulatedInstrument(
        whole_image=np.ones((100, 100)),
        output_dir=tmp_path,
        pixel_size_x=0.1 * scale,
        pixel_size_y=0.2 * scale,
    )
    if not show_axes:
        instrument.set_config("add_axis_ticks", False)
    subplots = simulated_acquisition.plt.subplots
    captured = {}

    def capture_axes(*args, **kwargs):
        fig, ax = subplots(*args, **kwargs)
        captured["axes"] = ax
        return fig, ax

    monkeypatch.setattr(simulated_acquisition.plt, "subplots", capture_axes)
    instrument.acquire_image(10.5 * scale, 5.5 * scale, scale, scale, 0.1 * scale)

    ax = captured["axes"]
    if show_axes:
        x_labels = [float(label.get_text()) for label in ax.get_xticklabels()]
        y_labels = [float(label.get_text()) for label in ax.get_yticklabels()]
        np.testing.assert_allclose(x_labels, (5 + np.arange(10) * 0.1) * scale)
        np.testing.assert_allclose(y_labels, (10 + np.arange(10) * 0.1) * scale)
        assert ax.get_xlabel() == "x"
        assert ax.get_ylabel() == "y"
    else:
        assert len(ax.get_xticks()) == 0
        assert len(ax.get_yticks()) == 0
