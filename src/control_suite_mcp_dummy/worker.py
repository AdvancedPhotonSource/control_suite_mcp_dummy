"""ZMQ instrument worker for the dummy control suite."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable
import argparse
import json
import logging

import numpy as np
import zmq

from control_suite_mcp_dummy.protocol import error_response, ok_response
from control_suite_mcp_dummy.simulated_acquisition import (
    SimulatedAcquireImageInstrument,
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

    def __init__(self, instrument: SimulatedAcquireImageInstrument) -> None:
        self.instrument = instrument
        self.handlers: dict[str, Callable[..., Any]] = {
            "health": self.health,
            "get_state": self.instrument.get_state,
            "set_blur": self.instrument.set_blur,
            "set_offset": self.instrument.set_offset,
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
    """Load a source image from ``.npy`` or create the built-in synthetic image."""
    if path is None:
        return create_default_image()
    image_path = Path(path)
    if image_path.suffix != ".npy":
        raise ValueError("Only .npy source images are supported by this dummy worker.")
    return np.load(image_path)


def build_instrument(args: argparse.Namespace) -> SimulatedAcquireImageInstrument:
    """Build the simulated instrument from CLI arguments."""
    return SimulatedAcquireImageInstrument(
        whole_image=load_image(args.image),
        output_dir=Path(args.output_dir),
        add_axis_ticks=args.add_axis_ticks,
        n_ticks=args.n_ticks,
        add_grid_lines=args.add_grid_lines,
        invert_yaxis=args.invert_yaxis,
        line_scan_gaussian_fit_y_threshold=args.line_scan_gaussian_fit_y_threshold,
        plot_image_in_log_scale=args.plot_image_in_log_scale,
        line_scan_return_gaussian_fit=args.line_scan_return_gaussian_fit,
        poisson_noise_scale=args.poisson_noise_scale,
        gaussian_psf_sigma=args.gaussian_psf_sigma,
        scan_jitter=args.scan_jitter,
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the worker CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bind", default="tcp://127.0.0.1:5555")
    parser.add_argument("--image", default=None, help="Optional .npy source image.")
    parser.add_argument("--output-dir", default=".tmp/control_suite_mcp_dummy")
    parser.add_argument("--add-axis-ticks", action="store_true")
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

