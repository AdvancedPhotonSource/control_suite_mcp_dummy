"""Tests for image content and frontend configuration over the MCP boundary."""

from __future__ import annotations

from pathlib import Path
import asyncio
import base64
import json
from unittest.mock import Mock

from fastmcp import Client
import pytest

from control_suite_mcp_dummy import mcp_server
from control_suite_mcp_dummy.simulated_acquisition import SimulatedInstrument, create_default_image


@pytest.mark.parametrize("include_images", [True, False])
@pytest.mark.parametrize(
    ("tool_name", "params"),
    [
        (
            "acquire_image",
            {"y_center": 228, "x_center": 228, "size_y": 32, "size_x": 32},
        ),
        (
            "acquire_image",
            {"y_center": 5.5, "x_center": 5.5, "size_y": 0.5, "size_x": 0.5, "scan_step": 0.1},
        ),
        (
            "acquire_line_scan",
            {"x_center": 228, "y_center": 228, "length": 32, "scan_step": 1.0},
        ),
    ],
)
def test_acquisition_content(monkeypatch, tmp_path, include_images, tool_name, params) -> None:
    """MCP clients receive unchanged JSON plus the exact PNG bytes when enabled."""
    instrument = SimulatedInstrument(
        whole_image=create_default_image(),
        output_dir=tmp_path,
        line_scan_return_gaussian_fit=True,
    )
    payloads = []
    image_bytes = []

    def call_worker(self, method, params=None):
        payload = getattr(instrument, method)(**(params or {}))
        payloads.append(payload)
        path = Path(payload["img_path"])
        image_bytes.append(path.read_bytes())
        if not include_images:
            path.unlink()
        return payload

    monkeypatch.setattr(mcp_server.WorkerClient, "call", call_worker)
    options = {} if include_images else {"include_images": False}
    server = mcp_server.create_mcp("tcp://unused", **options)

    async def exercise():
        async with Client(server) as client:
            return await client.call_tool(tool_name, params)

    result = asyncio.run(exercise())

    assert not result.is_error
    assert result.structured_content == payloads[0]
    assert json.loads(result.content[0].text) == payloads[0]
    if include_images:
        assert [block.type for block in result.content] == ["text", "image"]
        assert result.content[1].mimeType == "image/png"
        assert base64.b64decode(result.content[1].data) == image_bytes[0]
    else:
        assert [block.type for block in result.content] == ["text"]


def test_non_image_tool_content(monkeypatch) -> None:
    """Ordinary tools retain their JSON-only responses with images enabled."""
    payload = {"status": "ok"}
    monkeypatch.setattr(mcp_server.WorkerClient, "call", lambda *args: payload)

    async def exercise():
        async with Client(mcp_server.create_mcp("tcp://unused")) as client:
            return await client.call_tool("health")

    result = asyncio.run(exercise())
    assert [block.type for block in result.content] == ["text"]
    assert json.loads(result.content[0].text) == payload
    assert result.structured_content == payload


@pytest.mark.parametrize(
    ("config_text", "flags", "expected"),
    [
        (None, [], True),
        ("", [], True),
        ("worker: {}", [], True),
        ("mcp: {include_images: true}", [], True),
        ("mcp: {include_images: false}", [], False),
        (None, ["--no-include-images"], False),
        ("mcp: {include_images: true}", ["--no-include-images"], False),
        ("mcp: {include_images: false}", ["--include-images"], True),
    ],
)
def test_main_image_settings(monkeypatch, tmp_path, config_text, flags, expected) -> None:
    """Frontend startup applies the default, YAML setting, and CLI precedence."""
    argv = ["control-suite-mcp", *flags]
    if config_text is not None:
        config = tmp_path / "config.yaml"
        config.write_text(config_text, encoding="utf-8")
        argv.extend(["--config", str(config)])
    monkeypatch.setattr("sys.argv", argv)
    create_mcp = Mock()
    monkeypatch.setattr(mcp_server, "create_mcp", create_mcp)

    mcp_server.main()

    create_mcp.assert_called_once_with(
        "tcp://127.0.0.1:5555", timeout_ms=30_000, include_images=expected
    )
    create_mcp.return_value.run.assert_called_once()


@pytest.mark.parametrize("config_text", ["[]", "mcp: []", "mcp: {include_images: 'false'}"])
def test_invalid_image_config(tmp_path, config_text) -> None:
    """Malformed settings fail clearly instead of silently enabling images."""
    config = tmp_path / "config.yaml"
    config.write_text(config_text, encoding="utf-8")
    with pytest.raises(ValueError):
        mcp_server.resolve_include_images(str(config), None)
