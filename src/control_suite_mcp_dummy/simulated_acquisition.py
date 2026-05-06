"""Simulated image-acquisition instrument implementation.

This module intentionally has no EAA dependency. It recreates the core behavior
of ``eaa_imaging.tool.imaging.acquisition.SimulatedAcquireImage`` for a process
that can own instrument-control state independently of the MCP server.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi
from scipy import optimize
from scipy.interpolate import RegularGridInterpolator

logger = logging.getLogger(__name__)


def get_timestamp() -> str:
    """Return a filesystem-friendly timestamp."""
    return datetime.now().strftime("%Y%m%d%H%M%S%f")


def gaussian_1d(x: np.ndarray, a: float, mu: float, sigma: float, c: float) -> np.ndarray:
    """Evaluate a one-dimensional Gaussian with offset."""
    return a * np.exp(-0.5 * ((x - mu) / sigma) ** 2) + c


def fit_gaussian_1d(
    x: np.ndarray,
    y: np.ndarray,
    y_threshold: float = 0.0,
) -> tuple[float, float, float, float, float, float, float]:
    """Fit a one-dimensional Gaussian curve.

    Parameters
    ----------
    x, y
        Data to fit.
    y_threshold
        Fractional threshold above ``min(y)`` used to select fit points. A value
        of zero uses all points.

    Returns
    -------
    tuple
        ``a, mu, sigma, c, normalized_residual, x_min, x_max``. Failed fits
        return NaN fit parameters.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 4 or y.size < 4:
        return (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)

    fit_x = x
    fit_y = y
    if y_threshold > 0:
        threshold = y.min() + y_threshold * (y.max() - y.min())
        mask = y >= threshold
        if np.count_nonzero(mask) >= 4:
            fit_x = x[mask]
            fit_y = y[mask]

    amplitude = float(fit_y.max() - fit_y.min())
    center = float(fit_x[np.argmax(fit_y)])
    sigma = float(max((fit_x.max() - fit_x.min()) / 6, np.finfo(float).eps))
    offset = float(fit_y.min())
    try:
        params, _ = optimize.curve_fit(
            gaussian_1d,
            fit_x,
            fit_y,
            p0=(amplitude, center, sigma, offset),
            maxfev=10000,
        )
        a, mu, sigma, c = (float(value) for value in params)
        residual = fit_y - gaussian_1d(fit_x, a, mu, sigma, c)
        denom = float(np.linalg.norm(fit_y)) or 1.0
        normalized_residual = float(np.linalg.norm(residual) / denom)
        return a, mu, sigma, c, normalized_residual, float(fit_x.min()), float(fit_x.max())
    except Exception:
        logger.exception("Gaussian fit failed")
        return (np.nan, np.nan, np.nan, np.nan, np.nan, float(fit_x.min()), float(fit_x.max()))


def create_default_image(shape: tuple[int, int] = (512, 512)) -> np.ndarray:
    """Create a deterministic synthetic image for the dummy instrument."""
    y, x = np.indices(shape, dtype=float)
    image = (
        40 * np.exp(-((x - 155) ** 2 + (y - 160) ** 2) / (2 * 42**2))
        + 95 * np.exp(-((x - 330) ** 2 + (y - 245) ** 2) / (2 * 55**2))
        + 65 * np.exp(-((x - 245) ** 2 + (y - 375) ** 2) / (2 * 35**2))
        + 12 * np.sin(x / 18)
        + 8 * np.cos(y / 23)
    )
    image -= image.min()
    return image.astype(np.float32)


@dataclass
class SimulatedAcquireImageInstrument:
    """Stateful simulated image-acquisition instrument.

    Parameters
    ----------
    whole_image
        Source image sampled by acquisition and line-scan commands.
    output_dir
        Directory where generated plots are written.
    add_axis_ticks
        Add physical coordinate ticks to acquired image plots.
    n_ticks
        Number of axis ticks when ``add_axis_ticks`` is enabled.
    add_grid_lines
        Add grid lines to acquired image plots.
    invert_yaxis
        Invert the y-axis on image plots.
    line_scan_gaussian_fit_y_threshold
        Fractional threshold for Gaussian line-scan fitting.
    plot_image_in_log_scale
        Plot acquired images in ``log10(image + 1)`` scale.
    line_scan_return_gaussian_fit
        Include Gaussian fit metadata in line-scan responses.
    poisson_noise_scale
        Optional Poisson noise scale.
    gaussian_psf_sigma
        Optional Gaussian point-spread-function sigma.
    scan_jitter
        Optional uniform coordinate jitter.
    """

    whole_image: np.ndarray
    output_dir: Path
    add_axis_ticks: bool = False
    n_ticks: int = 10
    add_grid_lines: bool = False
    invert_yaxis: bool = False
    line_scan_gaussian_fit_y_threshold: float = 0.0
    plot_image_in_log_scale: bool = False
    line_scan_return_gaussian_fit: bool = False
    poisson_noise_scale: float | None = None
    gaussian_psf_sigma: float | None = None
    scan_jitter: float | None = None
    blur: float | None = None
    offset: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0]))

    def __post_init__(self) -> None:
        """Build interpolators and initialize state buffers."""
        self.output_dir = self.output_dir.expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.interpolator = RegularGridInterpolator(
            (np.arange(self.whole_image.shape[0]), np.arange(self.whole_image.shape[1])),
            self.whole_image,
            method="linear",
            bounds_error=False,
            fill_value=0.0,
        )
        self.image_0: np.ndarray | None = None
        self.image_km1: np.ndarray | None = None
        self.image_k: np.ndarray | None = None
        self.psize_0: float | None = None
        self.psize_km1: float | None = None
        self.psize_k: float | None = None
        self.image_acquisition_call_history: list[dict[str, Any]] = []
        self.line_scan_call_history: list[dict[str, Any]] = []

    @property
    def counter_acquire_image(self) -> int:
        """Return the number of image acquisitions."""
        return len(self.image_acquisition_call_history)

    def set_blur(self, blur: float | None) -> dict[str, Any]:
        """Set Gaussian blur applied to sampled data."""
        self.blur = blur
        return {"blur": self.blur}

    def set_offset(self, y_offset: float, x_offset: float) -> dict[str, Any]:
        """Set simulated drift offset in ``(y, x)`` order."""
        self.offset = np.array([y_offset, x_offset], dtype=float)
        return {"offset": self.offset.tolist()}

    def get_state(self) -> dict[str, Any]:
        """Return a JSON-serializable instrument state summary."""
        return {
            "image_shape": list(self.whole_image.shape),
            "output_dir": str(self.output_dir),
            "blur": self.blur,
            "offset": self.offset.tolist(),
            "counter_acquire_image": self.counter_acquire_image,
            "image_acquisition_call_history": self.image_acquisition_call_history,
            "line_scan_call_history": self.line_scan_call_history,
        }

    def update_image_buffers(self, new_image: np.ndarray, psize: float = 1.0) -> None:
        """Update first, previous, and current image buffers."""
        if self.image_0 is None:
            self.image_0 = new_image
            self.psize_0 = psize
        self.image_km1 = self.image_k
        self.psize_km1 = self.psize_k
        self.image_k = new_image
        self.psize_k = psize

    def sample(self, yy_flat: np.ndarray, xx_flat: np.ndarray, shape=None) -> np.ndarray:
        """Sample the source image and apply optional simulated physics effects."""

        def apply_blur(arr_flat: np.ndarray) -> np.ndarray:
            if shape is not None:
                return ndi.gaussian_filter(arr_flat.reshape(shape), self.blur, mode="nearest").ravel()
            return ndi.gaussian_filter(arr_flat, self.blur, mode="nearest")

        use_advanced = (
            self.poisson_noise_scale is not None
            or self.gaussian_psf_sigma is not None
            or self.scan_jitter is not None
        )
        if not use_advanced:
            arr_flat = self.interpolator(np.column_stack([yy_flat, xx_flat]))
            if self.blur is not None and self.blur > 0:
                arr_flat = apply_blur(arr_flat)
            return arr_flat

        if self.scan_jitter is not None:
            yy_flat = yy_flat + np.random.uniform(-self.scan_jitter, self.scan_jitter, yy_flat.shape)
            xx_flat = xx_flat + np.random.uniform(-self.scan_jitter, self.scan_jitter, xx_flat.shape)

        if self.gaussian_psf_sigma is not None:
            effective_sigma = self.gaussian_psf_sigma
            if self.blur is not None and self.blur > 0:
                effective_sigma *= 1 + self.blur
            half_n = max(1, int(np.ceil(2 * effective_sigma)))
            offsets = np.arange(-half_n, half_n + 1)
            dy_grid, dx_grid = np.meshgrid(offsets, offsets, indexing="ij")
            dy_flat = dy_grid.ravel()
            dx_flat = dx_grid.ravel()
            weights = np.exp(-0.5 * (dy_flat**2 + dx_flat**2) / effective_sigma**2)
            weights /= weights.sum()
            arr_flat = np.zeros(yy_flat.size)
            for index, weight in enumerate(weights):
                points = np.column_stack([yy_flat + dy_flat[index], xx_flat + dx_flat[index]])
                arr_flat += weight * self.interpolator(points)
        else:
            arr_flat = self.interpolator(np.column_stack([yy_flat, xx_flat]))
            if self.blur is not None and self.blur > 0:
                arr_flat = apply_blur(arr_flat)

        if self.poisson_noise_scale is not None:
            max_val = arr_flat.max()
            if max_val > 0:
                scaling_factor = self.poisson_noise_scale / max_val
                arr_flat = (
                    np.random.poisson(np.clip(arr_flat * scaling_factor, 0, None)).astype(float)
                    / scaling_factor
                )
        return arr_flat

    def acquire_image(
        self,
        y_center: float,
        x_center: float,
        size_y: int,
        size_x: int,
        scan_step: float = 1.0,
    ) -> dict[str, str]:
        """Acquire and plot an image sampled from the whole image."""
        loc_y = y_center - size_y / 2
        loc_x = x_center - size_x / 2
        self.image_acquisition_call_history.append(
            {
                "x_center": x_center,
                "y_center": y_center,
                "size_x": size_x,
                "size_y": size_y,
                "psize_x": scan_step,
                "psize_y": scan_step,
            }
        )
        y = np.arange(loc_y, loc_y + size_y, scan_step)
        x = np.arange(loc_x, loc_x + size_x, scan_step)
        yy, xx = np.meshgrid(y + self.offset[0], x + self.offset[1], indexing="ij")
        arr_shape = (len(y), len(x))
        arr = self.sample(yy.ravel(), xx.ravel(), shape=arr_shape).reshape(arr_shape)
        self.update_image_buffers(arr, psize=scan_step)

        image_to_plot = arr if not self.plot_image_in_log_scale else np.log10(arr + 1)
        fig, ax = plt.subplots()
        ax.imshow(image_to_plot, cmap="inferno", origin="upper")
        if self.add_axis_ticks:
            x_ticks = np.linspace(0, max(len(x) - 1, 0), min(self.n_ticks, len(x)), dtype=int)
            y_ticks = np.linspace(0, max(len(y) - 1, 0), min(self.n_ticks, len(y)), dtype=int)
            ax.set_xticks(x_ticks)
            ax.set_xticklabels([f"{x[idx]:.1f}" for idx in x_ticks])
            ax.set_yticks(y_ticks)
            ax.set_yticklabels([f"{y[idx]:.1f}" for idx in y_ticks])
        else:
            ax.set_xticks([])
            ax.set_yticks([])
        if self.add_grid_lines:
            ax.grid(True)
        if self.invert_yaxis:
            ax.invert_yaxis()
        fig.tight_layout()

        filename = f"image_{y_center}_{x_center}_{size_y}_{size_x}_{get_timestamp()}.png"
        path = self.output_dir / filename
        fig.savefig(path, bbox_inches="tight", pad_inches=0)
        plt.close(fig)
        return {"img_path": str(path)}

    def acquire_line_scan(
        self,
        x_center: float,
        y_center: float,
        length: float,
        scan_step: float,
        angle: float = 0.0,
    ) -> dict[str, Any]:
        """Acquire and plot a line scan sampled from the whole image."""
        angle_rad = np.radians(angle)
        half = length / 2
        start_x = x_center - half * np.cos(angle_rad)
        end_x = x_center + half * np.cos(angle_rad)
        start_y = y_center - half * np.sin(angle_rad)
        end_y = y_center + half * np.sin(angle_rad)
        self.line_scan_call_history.append(
            {
                "step": scan_step,
                "x_center": x_center,
                "y_center": y_center,
                "length": length,
                "angle": angle,
            }
        )

        pt_start = np.array([start_y, start_x])
        pt_end = np.array([end_y, end_x])
        distance_total = np.linalg.norm(pt_end - pt_start)
        distances = np.arange(0, distance_total, scan_step)
        points = pt_start + distances[:, None] * (pt_end - pt_start) / distance_total
        points = points + self.offset
        arr = self.sample(points[:, 0], points[:, 1])

        a, mu, sigma, c, residual, fit_x_min, fit_x_max = fit_gaussian_1d(
            distances,
            arr,
            y_threshold=self.line_scan_gaussian_fit_y_threshold,
        )
        if np.any(np.isnan([a, mu, sigma, c])):
            gaussian_values = None
            fwhm = np.nan
        else:
            gaussian_values = gaussian_1d(distances, a, mu, sigma, c)
            fwhm = 2.35 * abs(sigma)

        show_scan_line = self.image_k is not None and len(self.image_acquisition_call_history) > 0
        if show_scan_line:
            fig, axes = plt.subplots(1, 2, squeeze=True, figsize=(10, 4))
            line_ax = axes[0]
            image_ax = axes[1]
        else:
            fig, line_ax = plt.subplots(1, 1, squeeze=True)
            image_ax = None

        line_ax.plot(distances, arr, label="data")
        if gaussian_values is not None:
            line_ax.plot(distances, gaussian_values, linestyle="--", label="Gaussian fit")
        line_ax.text(
            0.05,
            0.95,
            f"FWHM = {fwhm:.2f}",
            transform=line_ax.transAxes,
            verticalalignment="top",
            horizontalalignment="left",
        )
        line_ax.legend()
        line_ax.set_xlabel("distance")
        line_ax.set_ylabel("value")
        line_ax.set_title("Line scan")
        line_ax.grid(True)

        if image_ax is not None and self.image_k is not None:
            image_info = self.image_acquisition_call_history[-1]
            image_to_plot = self.image_k if not self.plot_image_in_log_scale else np.log10(self.image_k + 1)
            image_x_min = image_info["x_center"] - image_info["size_x"] / 2
            image_x_max = image_info["x_center"] + image_info["size_x"] / 2
            image_y_min = image_info["y_center"] - image_info["size_y"] / 2
            image_y_max = image_info["y_center"] + image_info["size_y"] / 2
            image_ax.imshow(
                image_to_plot,
                cmap="inferno",
                origin="upper",
                extent=[image_x_min, image_x_max, image_y_max, image_y_min],
            )
            image_ax.plot([start_x, end_x], [start_y, end_y], color="red", linewidth=2)
            image_ax.set_xlabel("x")
            image_ax.set_ylabel("y")
            image_ax.set_title("Line scan position")
            if self.invert_yaxis:
                image_ax.invert_yaxis()

        fig.tight_layout()
        path = self.output_dir / f"line_scan_{y_center}_{x_center}_{length}_{angle}_{scan_step}_{get_timestamp()}.png"
        fig.savefig(path)
        plt.close(fig)

        result: dict[str, Any] = {"img_path": str(path)}
        if self.line_scan_return_gaussian_fit:
            result.update(
                {
                    "fwhm": fwhm,
                    "a": a,
                    "mu": mu,
                    "sigma": sigma,
                    "c": c,
                    "normalized_residual": residual,
                    "x_min": fit_x_min,
                    "x_max": fit_x_max,
                }
            )
        return result
