"""Integration tests for the ZMQ worker boundary."""

from __future__ import annotations

from pathlib import Path
from argparse import Namespace
import multiprocessing as mp
import time

import numpy as np
import tifffile

from control_suite_mcp_dummy.simulated_acquisition import (
    SimulatedInstrument,
    create_default_image,
)
from control_suite_mcp_dummy.worker import InstrumentWorker
from control_suite_mcp_dummy.worker import build_instrument
from control_suite_mcp_dummy.worker import load_image
from control_suite_mcp_dummy.zmq_client import WorkerClient


def run_worker(endpoint: str, output_dir: str) -> None:
    """Run a worker process for tests."""
    instrument = SimulatedInstrument(
        whole_image=create_default_image(),
        output_dir=Path(output_dir),
    )
    InstrumentWorker(instrument).serve(endpoint)


def test_worker_client_acquires_image_and_line_scan(tmp_path: Path) -> None:
    """A ZMQ client can drive the worker and receive image payloads."""
    endpoint = "tcp://127.0.0.1:5567"
    process = mp.Process(target=run_worker, args=(endpoint, str(tmp_path)), daemon=True)
    process.start()
    client = WorkerClient(endpoint, timeout_ms=1000)

    try:
        deadline = time.time() + 5
        while True:
            try:
                assert client.call("health") == {"status": "ok"}
                break
            except TimeoutError:
                if time.time() > deadline:
                    raise
                time.sleep(0.1)

        image_result = client.call(
            "acquire_image",
            {"y_center": 228, "x_center": 228, "size_y": 64, "size_x": 64, "scan_step": 1},
        )
        image_path = Path(image_result["img_path"])
        assert image_path.is_absolute()
        assert image_path.exists()
        array_path = Path(image_result["array_path"])
        assert array_path.is_absolute()
        assert array_path.exists()
        assert image_result["psize"] == 1

        config_result = client.call(
            "set_attribute",
            {"name": "line_scan_return_gaussian_fit", "value": True},
        )
        assert config_result == {"name": "line_scan_return_gaussian_fit", "value": True}
        line_result = client.call(
            "acquire_line_scan",
            {"x_center": 228, "y_center": 228, "length": 32, "scan_step": 1.0, "angle": 0},
        )
        line_path = Path(line_result["img_path"])
        assert line_path.is_absolute()
        assert line_path.exists()
        assert "fwhm" in line_result

        parameter_result = client.call("set_parameters", {"parameters": [0.25]})
        assert parameter_result == "Set simulated parameters: focus=0.25"

        state = client.call("get_state")
        assert Path(state["output_dir"]).is_absolute()
        assert state["counter_acquire_image"] == 1
        assert state["parameters"] == [0.25]
        assert state["parameter_history"] == [[0.25]]
        assert state["current_image_info"]["array_path"] == str(array_path)
    finally:
        process.terminate()
        process.join(timeout=5)


def test_worker_builds_instrument_from_yaml_config(tmp_path: Path) -> None:
    """YAML config can reproduce simulated acquisition and parameter settings."""
    config_path = tmp_path / "config.yaml"
    output_dir = tmp_path / "out"
    config_path.write_text(
        """
worker:
  output_dir: {output_dir}
acquisition:
  add_axis_ticks: true
  add_grid_lines: false
  invert_yaxis: false
  add_line_scan_candidates_to_image: false
  plot_image_in_log_scale: false
  poisson_noise_scale: 100
  scan_jitter: 0
  gaussian_psf_sigma: 1
parameter_setting:
  parameter_names: [z]
  true_parameters: [3.0]
  parameter_ranges:
    - [0.0]
    - [10.0]
  drift_factor: 30
  blur_factor: 5
""".format(output_dir=output_dir),
        encoding="utf-8",
    )
    args = Namespace(
        config=str(config_path),
        image=None,
        output_dir=".tmp/out",
        add_axis_ticks=False,
        n_ticks=10,
        add_grid_lines=False,
        invert_yaxis=False,
        line_scan_gaussian_fit_y_threshold=0.0,
        plot_image_in_log_scale=False,
        line_scan_return_gaussian_fit=False,
        poisson_noise_scale=None,
        gaussian_psf_sigma=None,
        scan_jitter=None,
    )

    instrument = build_instrument(args)

    assert instrument.output_dir == output_dir.resolve()
    assert instrument.add_axis_ticks is True
    assert instrument.poisson_noise_scale == 100
    assert instrument.scan_jitter == 0
    assert instrument.gaussian_psf_sigma == 1
    assert instrument.parameter_names == ["z"]
    assert instrument.true_parameters == [3.0]
    assert instrument.parameter_ranges == [(0.0,), (10.0,)]
    assert instrument.drift_factor == 30
    assert instrument.blur_factor == 5

    assert instrument.set_parameters([3.0]) == "Set simulated parameters: z=3.0"
    assert instrument.blur == 0.0
    assert instrument.offset.tolist() == [0.0, 0.0]
    assert instrument.set_parameters([5.0]) == "Set simulated parameters: z=5.0"
    assert instrument.blur == 1.0
    assert instrument.offset.tolist() == [-6.0, -6.0]


def test_load_image_supports_tiff(tmp_path: Path) -> None:
    """Worker image loader accepts TIFF files."""
    image = np.arange(25, dtype=np.float32).reshape(5, 5)
    image_path = tmp_path / "image.tiff"
    tifffile.imwrite(image_path, image)

    loaded = load_image(str(image_path))

    assert loaded.dtype == np.float32
    np.testing.assert_array_equal(loaded, image)
