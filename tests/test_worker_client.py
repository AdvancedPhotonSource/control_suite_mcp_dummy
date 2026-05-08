"""Integration tests for the ZMQ worker boundary."""

from __future__ import annotations

from pathlib import Path
import multiprocessing as mp
import time

from control_suite_mcp_dummy.simulated_acquisition import (
    SimulatedInstrument,
    create_default_image,
)
from control_suite_mcp_dummy.worker import InstrumentWorker
from control_suite_mcp_dummy.zmq_client import WorkerClient


def run_worker(endpoint: str, output_dir: str) -> None:
    """Run a worker process for tests."""
    instrument = SimulatedInstrument(
        whole_image=create_default_image(),
        output_dir=Path(output_dir),
        line_scan_return_gaussian_fit=True,
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
    finally:
        process.terminate()
        process.join(timeout=5)
