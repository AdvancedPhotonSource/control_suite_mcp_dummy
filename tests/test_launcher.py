"""Tests for launcher command construction."""

from __future__ import annotations

from argparse import Namespace

from control_suite_mcp_dummy.launcher import build_mcp_command, build_worker_command


def test_launcher_builds_separate_process_commands(monkeypatch) -> None:
    """Launcher commands target separate worker and MCP executables."""
    monkeypatch.setattr(
        "control_suite_mcp_dummy.launcher.resolve_executable",
        lambda name: f"/venv/bin/{name}",
    )
    args = Namespace(
        worker_endpoint="tcp://127.0.0.1:5555",
        output_dir=".tmp/out",
        image=None,
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
        request_timeout_ms=30000,
        mcp_host="127.0.0.1",
        mcp_port=8050,
        mcp_path="/mcp",
    )

    worker_command = build_worker_command(args)
    mcp_command = build_mcp_command(args)

    assert worker_command[0] == "/venv/bin/control-suite-worker"
    assert mcp_command[0] == "/venv/bin/control-suite-mcp"
    assert "--bind" in worker_command
    assert "--worker" in mcp_command

