"""ZMQ instrument worker for the dummy control suite."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable
import argparse
import json
import logging

import numpy as np
from PIL import Image
import tifffile
import yaml
import zmq

from control_suite_mcp_dummy.protocol import error_response, ok_response
from control_suite_mcp_dummy.simulated_acquisition import (
    SimulatedInstrument,
    create_default_image,
)

logger = logging.getLogger(__name__)


class InstrumentWorker:
    """Blocking ZMQ worker that owns the simulated instrument instance.

    Parameters
    ----------
    instrument
        Simulated instrument object. This object lives only in the worker
        process.
    """

    def __init__(self, instrument: SimulatedInstrument) -> None:
        self.instrument = instrument
        self.handlers: dict[str, Callable[..., Any]] = {
            "health": self.health,
            "get_state": self.instrument.get_state,
            "set_blur": self.instrument.set_blur,
            "set_offset": self.instrument.set_offset,
            "set_config": self.instrument.set_config,
            "set_attribute": self.instrument.set_attribute,
            "set_parameters": self.instrument.set_parameters,
            "acquire_image": self.instrument.acquire_image,
            "acquire_line_scan": self.instrument.acquire_line_scan,
        }

    def health(self) -> dict[str, str]:
        """Return worker health."""
        return {"status": "ok"}

    def handle_message(self, message: dict[str, Any]) -> dict[str, Any]:
        """Execute one command message and return a response envelope."""
        command_id = message.get("id")
        method = message.get("method")
        params = message.get("params", {})
        if not isinstance(method, str) or method not in self.handlers:
            return error_response(command_id, f"Unknown method: {method}")
        if not isinstance(params, dict):
            return error_response(command_id, "Command params must be a JSON object.")
        try:
            result = self.handlers[method](**params)
            return ok_response(command_id, result)
        except Exception as exc:
            logger.exception("Worker command failed: %s", method)
            return error_response(command_id, str(exc))

    def serve(self, bind: str) -> None:
        """Serve worker commands forever on a ZMQ REP socket."""
        context = zmq.Context.instance()
        socket = context.socket(zmq.REP)
        socket.bind(bind)
        logger.info("Instrument worker listening on %s", bind)
        try:
            while True:
                raw_message = socket.recv_string()
                try:
                    message = json.loads(raw_message)
                    if not isinstance(message, dict):
                        response = error_response(None, "Command must be a JSON object.")
                    else:
                        response = self.handle_message(message)
                except json.JSONDecodeError as exc:
                    response = error_response(None, f"Invalid JSON command: {exc}")
                socket.send_string(json.dumps(response, allow_nan=True))
        finally:
            socket.close(linger=0)


def load_image(path: str | None) -> np.ndarray:
    """Load a float32 grayscale source, using cameraman when no path is given.

    RGB and RGBA images use luminance weights; alpha is ignored.
    """
    if path is None:
        return create_default_image()
    image_path = Path(path).expanduser()
    suffix = image_path.suffix.lower()
    if suffix == ".npy":
        image = np.load(image_path)
    elif suffix in {".tif", ".tiff"}:
        image = tifffile.imread(image_path)
    elif suffix in {".png", ".jpg", ".jpeg"}:
        with Image.open(image_path) as source:
            if source.mode not in {"1", "L", "I", "I;16", "F", "RGB", "RGBA"}:
                source = source.convert("RGB")
            image = np.asarray(source)
    else:
        raise ValueError("Supported source images: .npy, .tif, .tiff, .png, .jpg, .jpeg.")
    image = np.asarray(image, dtype=np.float32)
    if image.ndim == 3 and image.shape[-1] in {3, 4}:
        image = image[..., :3] @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    elif image.ndim == 3 and image.shape[-1] == 1:
        image = image[..., 0]
    if image.ndim != 2 or 0 in image.shape:
        raise ValueError("Source image must be nonempty grayscale, RGB, or RGBA.")
    return image


def load_config(path: str | None) -> dict[str, Any]:
    """Load an optional YAML worker configuration file.

    Parameters
    ----------
    path
        Path to a YAML configuration file.

    Returns
    -------
    dict[str, Any]
        Parsed configuration dictionary.
    """
    if path is None:
        return {}
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError("Worker config YAML must contain a mapping at the top level.")
    return config


def resolve_config_value(
    config: dict[str, Any],
    section_name: str,
    key: str,
    default: Any,
) -> Any:
    """Resolve a setting from a config section or top-level fallback."""
    section = config.get(section_name, {})
    if section is not None and not isinstance(section, dict):
        raise ValueError(f"Config section {section_name!r} must be a mapping.")
    if isinstance(section, dict) and key in section:
        return section[key]
    return config.get(key, default)


def resolve_parameter_ranges(value: Any) -> list[tuple[float, ...]]:
    """Convert configured parameter ranges into tuple rows."""
    if value is None:
        return [(-1.0,), (1.0,)]
    if not isinstance(value, list | tuple) or len(value) != 2:
        raise ValueError("parameter_ranges must contain lower and upper bound rows.")
    return [tuple(float(item) for item in row) for row in value]


def build_instrument(args: argparse.Namespace) -> SimulatedInstrument:
    """Build the simulated instrument from CLI arguments."""
    config = load_config(args.config)
    parameter_config = config.get("parameter_setting", {})
    if parameter_config is not None and not isinstance(parameter_config, dict):
        raise ValueError("Config section 'parameter_setting' must be a mapping.")

    return SimulatedInstrument(
        whole_image=load_image(resolve_config_value(config, "acquisition", "image", args.image)),
        output_dir=Path(resolve_config_value(config, "worker", "output_dir", args.output_dir)),
        pixel_size_x=resolve_config_value(config, "acquisition", "pixel_size_x", 1.0),
        pixel_size_y=resolve_config_value(config, "acquisition", "pixel_size_y", 1.0),
        add_axis_ticks=resolve_config_value(
            config,
            "acquisition",
            "add_axis_ticks",
            args.add_axis_ticks,
        ),
        n_ticks=resolve_config_value(config, "acquisition", "n_ticks", args.n_ticks),
        add_grid_lines=resolve_config_value(
            config,
            "acquisition",
            "add_grid_lines",
            args.add_grid_lines,
        ),
        invert_yaxis=resolve_config_value(
            config,
            "acquisition",
            "invert_yaxis",
            args.invert_yaxis,
        ),
        line_scan_gaussian_fit_y_threshold=resolve_config_value(
            config,
            "acquisition",
            "line_scan_gaussian_fit_y_threshold",
            args.line_scan_gaussian_fit_y_threshold,
        ),
        plot_image_in_log_scale=resolve_config_value(
            config,
            "acquisition",
            "plot_image_in_log_scale",
            args.plot_image_in_log_scale,
        ),
        line_scan_return_gaussian_fit=resolve_config_value(
            config,
            "acquisition",
            "line_scan_return_gaussian_fit",
            args.line_scan_return_gaussian_fit,
        ),
        poisson_noise_scale=resolve_config_value(
            config,
            "acquisition",
            "poisson_noise_scale",
            args.poisson_noise_scale,
        ),
        gaussian_psf_sigma=resolve_config_value(
            config,
            "acquisition",
            "gaussian_psf_sigma",
            args.gaussian_psf_sigma,
        ),
        scan_jitter=resolve_config_value(config, "acquisition", "scan_jitter", args.scan_jitter),
        add_line_scan_candidates_to_image=resolve_config_value(
            config,
            "acquisition",
            "add_line_scan_candidates_to_image",
            False,
        ),
        parameter_names=parameter_config.get("parameter_names", ["focus"]),
        true_parameters=parameter_config.get("true_parameters", [0.0]),
        parameter_ranges=resolve_parameter_ranges(parameter_config.get("parameter_ranges")),
        drift_factor=parameter_config.get("drift_factor", 0.0),
        blur_factor=parameter_config.get("blur_factor", 0.0),
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the worker CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bind", default="tcp://127.0.0.1:5555")
    parser.add_argument("--config", default=None, help="Optional YAML worker configuration.")
    parser.add_argument("--image", default=None, help="Optional NumPy, TIFF, PNG, or JPEG image.")
    parser.add_argument("--output-dir", default=".tmp/control_suite_mcp_dummy")
    parser.add_argument(
        "--add-axis-ticks",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show acquisition coordinate ticks on images (default: enabled).",
    )
    parser.add_argument("--n-ticks", type=int, default=10)
    parser.add_argument("--add-grid-lines", action="store_true")
    parser.add_argument("--invert-yaxis", action="store_true")
    parser.add_argument("--line-scan-gaussian-fit-y-threshold", type=float, default=0.0)
    parser.add_argument("--plot-image-in-log-scale", action="store_true")
    parser.add_argument("--line-scan-return-gaussian-fit", action="store_true")
    parser.add_argument("--poisson-noise-scale", type=float, default=None)
    parser.add_argument("--gaussian-psf-sigma", type=float, default=None)
    parser.add_argument("--scan-jitter", type=float, default=None)
    return parser


def main() -> None:
    """Run the instrument worker process."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = build_parser().parse_args()
    worker = InstrumentWorker(build_instrument(args))
    worker.serve(args.bind)


if __name__ == "__main__":
    main()
