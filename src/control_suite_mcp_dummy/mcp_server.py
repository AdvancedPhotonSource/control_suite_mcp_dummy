"""FastMCP HTTP server that forwards simulated acquisition tools to ZMQ."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any
import argparse
import asyncio
import logging

from fastmcp import FastMCP
from fastmcp.tools.tool import ToolResult
from fastmcp.utilities.types import Image
import yaml

from control_suite_mcp_dummy.zmq_client import WorkerClient

logger = logging.getLogger(__name__)


def create_mcp(
    worker_endpoint: str, timeout_ms: int = 30_000, include_images: bool = True
) -> FastMCP:
    """Create the FastMCP server.

    Parameters
    ----------
    worker_endpoint
        ZMQ endpoint used to reach the instrument worker.
    timeout_ms
        Worker request timeout in milliseconds.
    include_images
        Include PNG image content alongside acquisition JSON payloads.

    Returns
    -------
    FastMCP
        Configured MCP server.
    """
    mcp = FastMCP("Control Suite MCP Dummy")
    client = WorkerClient(worker_endpoint, timeout_ms=timeout_ms)

    async def call_worker(method: str, params: dict[str, Any] | None = None) -> Any:
        return await asyncio.to_thread(client.call, method, params)

    async def call_acquisition(method: str, params: dict[str, Any]) -> ToolResult:
        payload = await call_worker(method, params)
        result = ToolResult(content=payload, structured_content=payload)
        if include_images:
            image = Image(path=payload["img_path"])
            result.content.append(await asyncio.to_thread(image.to_image_content))
        return result

    @mcp.tool()
    async def health() -> dict[str, str]:
        """Check whether the instrument worker is reachable."""
        return await call_worker("health")

    @mcp.tool()
    async def get_state() -> dict[str, Any]:
        """Return simulated instrument state."""
        return await call_worker("get_state")

    @mcp.tool()
    async def set_blur(
        blur: Annotated[float | None, "Gaussian blur sigma applied to sampled data."],
    ) -> dict[str, Any]:
        """Set the simulated image blur."""
        return await call_worker("set_blur", {"blur": blur})

    @mcp.tool()
    async def set_offset(
        y_offset: Annotated[float, "Simulated y drift offset in acquisition units."],
        x_offset: Annotated[float, "Simulated x drift offset in acquisition units."],
    ) -> dict[str, Any]:
        """Set simulated drift offset in y, x order."""
        return await call_worker("set_offset", {"y_offset": y_offset, "x_offset": x_offset})

    @mcp.tool()
    async def set_config(
        name: Annotated[str, "Writable simulated instrument configuration name."],
        value: Annotated[Any, "JSON-serializable value to assign."],
    ) -> dict[str, Any]:
        """Set a writable simulated instrument configuration value."""
        return await call_worker("set_config", {"name": name, "value": value})

    @mcp.tool()
    async def set_attribute(
        name: Annotated[str, "Writable simulated instrument configuration name."],
        value: Annotated[Any, "JSON-serializable value to assign."],
    ) -> dict[str, Any]:
        """Alias for ``set_config`` used by the EAA MCP acquisition proxy."""
        return await call_worker("set_attribute", {"name": name, "value": value})

    @mcp.tool()
    async def set_parameters(
        parameters: Annotated[
            list[float],
            "Simulated tuning parameter values to set.",
        ],
    ) -> str:
        """Set simulated instrument tuning parameters."""
        return await call_worker("set_parameters", {"parameters": parameters})

    @mcp.tool(output_schema={"type": "object", "additionalProperties": True})
    async def acquire_image(
        y_center: Annotated[float, "The center y-coordinate in acquisition units."],
        x_center: Annotated[float, "The center x-coordinate in acquisition units."],
        size_y: Annotated[float, "The height of the acquisition region in acquisition units."],
        size_x: Annotated[float, "The width of the acquisition region in acquisition units."],
        scan_step: Annotated[
            float,
            "The step size in acquisition units between sampled points in both y and x.",
        ] = 1.0,
    ) -> ToolResult:
        """Acquire an image, returning img_path, array_path, and psize in JSON.

        The source image and its pixel_size_x/y are configured on the worker.
        Source pixel indices are x/pixel_size_x and y/pixel_size_y. Samples outside
        the source are zero. Each axis samples from center-size/2 to center+size/2
        (exclusive). Defaults are a 512x512 cameraman image and unit pixel sizes.
        PNG image content is also included when enabled in the server settings.
        """
        return await call_acquisition(
            "acquire_image",
            {
                "y_center": y_center,
                "x_center": x_center,
                "size_y": size_y,
                "size_x": size_x,
                "scan_step": scan_step,
            },
        )

    @mcp.tool(output_schema={"type": "object", "additionalProperties": True})
    async def acquire_line_scan(
        x_center: Annotated[float, "The center x-coordinate in acquisition units."],
        y_center: Annotated[float, "The center y-coordinate in acquisition units."],
        length: Annotated[float, "The length of the line scan in acquisition units."],
        scan_step: Annotated[float, "The step size of the line scan in acquisition units."],
        angle: Annotated[
            float,
            "The angle of the line scan in degrees. 0 is horizontal.",
        ] = 0.0,
    ) -> ToolResult:
        """Acquire a line scan, returning img_path and optional fit metadata in JSON.

        Coordinates use the source pixel sizes configured on the worker, as in
        acquire_image. Samples outside the source are zero.
        PNG plot content is also included when enabled in the server settings.
        """
        return await call_acquisition(
            "acquire_line_scan",
            {
                "x_center": x_center,
                "y_center": y_center,
                "length": length,
                "scan_step": scan_step,
                "angle": angle,
            },
        )

    return mcp


def resolve_include_images(config_path: str | None, override: bool | None) -> bool:
    """Resolve image content settings with CLI overrides taking precedence."""
    if override is not None:
        return override
    if config_path is None:
        return True
    with Path(config_path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if config is None:
        return True
    if not isinstance(config, dict):
        raise ValueError("MCP config YAML must contain a mapping at the top level.")
    section = config.get("mcp", {})
    if not isinstance(section, dict):
        raise ValueError("Config section 'mcp' must be a mapping.")
    include_images = section.get("include_images", True)
    if not isinstance(include_images, bool):
        raise ValueError("mcp.include_images must be a boolean.")
    return include_images


def build_parser() -> argparse.ArgumentParser:
    """Build the MCP server CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="Optional YAML configuration.")
    parser.add_argument(
        "--include-images",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Include image content alongside JSON (default: enabled; overrides YAML).",
    )
    parser.add_argument("--worker", default="tcp://127.0.0.1:5555")
    parser.add_argument("--timeout-ms", type=int, default=30_000)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--path", default="/mcp")
    return parser


def main() -> None:
    """Run the FastMCP HTTP server."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = build_parser().parse_args()
    logger.info("Starting MCP server, forwarding to worker at %s", args.worker)
    mcp = create_mcp(
        args.worker,
        timeout_ms=args.timeout_ms,
        include_images=resolve_include_images(args.config, args.include_images),
    )
    mcp.run(transport="http", host=args.host, port=args.port, path=args.path)


if __name__ == "__main__":
    main()
