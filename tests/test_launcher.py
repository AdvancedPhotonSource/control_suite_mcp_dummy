"""Tests for launcher command construction."""

from __future__ import annotations

from argparse import Namespace

import pytest

from control_suite_mcp_dummy.launcher import build_mcp_command, build_parser, build_worker_command
from control_suite_mcp_dummy.mcp_server import build_parser as build_mcp_parser
from control_suite_mcp_dummy.mcp_server import resolve_include_images
from control_suite_mcp_dummy.worker import build_parser as build_worker_parser


def test_launcher_builds_separate_process_commands(monkeypatch) -> None:
    """Launcher commands target separate worker and MCP executables."""
    monkeypatch.setattr(
        "control_suite_mcp_dummy.launcher.resolve_executable",
        lambda name: f"/venv/bin/{name}",
    )
    args = Namespace(
        worker_endpoint="tcp://127.0.0.1:5555",
        config=None,
        include_images=None,
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


def test_launcher_forwards_config_path(monkeypatch) -> None:
    """Launcher forwards YAML config paths to both processes."""
    monkeypatch.setattr(
        "control_suite_mcp_dummy.launcher.resolve_executable",
        lambda name: f"/venv/bin/{name}",
    )
    args = Namespace(
        worker_endpoint="tcp://127.0.0.1:5555",
        config="configs/simulated_parameter_tuning.yaml",
        include_images=None,
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

    assert "--config" in worker_command
    assert "configs/simulated_parameter_tuning.yaml" in worker_command
    mcp_command = build_mcp_command(args)
    assert "--config" in mcp_command
    assert "configs/simulated_parameter_tuning.yaml" in mcp_command


@pytest.mark.parametrize(
    ("flags", "expected"),
    [([], False), (["--include-images"], True), (["--no-include-images"], False)],
)
def test_launcher_forwards_image_settings(monkeypatch, tmp_path, flags, expected) -> None:
    """The frontend receives the config and any explicit image-content override."""
    monkeypatch.setattr(
        "control_suite_mcp_dummy.launcher.resolve_executable", lambda name: name
    )
    config = tmp_path / "config.yaml"
    config.write_text("mcp:\n  include_images: false\n", encoding="utf-8")
    args = build_parser().parse_args(["--config", str(config), *flags])

    command = build_mcp_command(args)
    frontend_args = build_mcp_parser().parse_args(command[1:])

    assert frontend_args.include_images is args.include_images
    assert resolve_include_images(frontend_args.config, frontend_args.include_images) is expected
    assert "--include-images" not in build_worker_command(args)
    assert "--no-include-images" not in build_worker_command(args)


@pytest.mark.parametrize(
    ("flags", "expected"),
    [([], True), (["--add-axis-ticks"], True), (["--no-add-axis-ticks"], False)],
)
def test_launcher_forwards_axis_display(monkeypatch, flags, expected) -> None:
    monkeypatch.setattr(
        "control_suite_mcp_dummy.launcher.resolve_executable", lambda name: name
    )
    args = build_parser().parse_args(flags)

    worker_args = build_worker_parser().parse_args(build_worker_command(args)[1:])

    assert worker_args.add_axis_ticks is expected
