"""FastMCP HTTP server that forwards simulated acquisition tools to ZMQ."""

from __future__ import annotations

from typing import Annotated, Any
import argparse
import asyncio
import logging

from fastmcp import FastMCP

from control_suite_mcp_dummy.zmq_client import WorkerClient

logger = logging.getLogger(__name__)


def create_mcp(worker_endpoint: str, timeout_ms: int = 30_000) -> FastMCP:
    """Create the FastMCP server.

    Parameters
    ----------
    worker_endpoint
        ZMQ endpoint used to reach the instrument worker.
    timeout_ms
        Worker request timeout in milliseconds.

    Returns
    -------
    FastMCP
        Configured MCP server.
    """
    mcp = FastMCP("Control Suite MCP Dummy")
    client = WorkerClient(worker_endpoint, timeout_ms=timeout_ms)

    async def call_worker(method: str, params: dict[str, Any] | None = None) -> Any:
        return await asyncio.to_thread(client.call, method, params)

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
        y_offset: Annotated[float, "Simulated y drift offset."],
        x_offset: Annotated[float, "Simulated x drift offset."],
    ) -> dict[str, Any]:
        """Set simulated drift offset in y, x order."""
        return await call_worker("set_offset", {"y_offset": y_offset, "x_offset": x_offset})

    @mcp.tool()
    async def set_parameters(
        parameters: Annotated[
            list[float],
            "Simulated tuning parameter values to set.",
        ],
    ) -> str:
        """Set simulated instrument tuning parameters."""
        return await call_worker("set_parameters", {"parameters": parameters})

    @mcp.tool()
    async def acquire_image(
        y_center: Annotated[float, "The y-coordinate of the center of the image to acquire."],
        x_center: Annotated[float, "The x-coordinate of the center of the image to acquire."],
        size_y: Annotated[int, "The height of the image to acquire."],
        size_x: Annotated[int, "The width of the image to acquire."],
        scan_step: Annotated[
            float,
            "The step size between sampled points in both y and x directions.",
        ] = 1.0,
    ) -> dict[str, str]:
        """Acquire an image from the simulated instrument."""
        return await call_worker(
            "acquire_image",
            {
                "y_center": y_center,
                "x_center": x_center,
                "size_y": size_y,
                "size_x": size_x,
                "scan_step": scan_step,
            },
        )

    @mcp.tool()
    async def acquire_line_scan(
        x_center: Annotated[float, "The x-coordinate of the center of the line scan."],
        y_center: Annotated[float, "The y-coordinate of the center of the line scan."],
        length: Annotated[float, "The length of the line scan."],
        scan_step: Annotated[float, "The step size of the line scan."],
        angle: Annotated[
            float,
            "The angle of the line scan in degrees. 0 is horizontal.",
        ] = 0.0,
    ) -> dict[str, Any]:
        """Acquire a line scan from the simulated instrument."""
        return await call_worker(
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


def build_parser() -> argparse.ArgumentParser:
    """Build the MCP server CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
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
    mcp = create_mcp(args.worker, timeout_ms=args.timeout_ms)
    mcp.run(transport="http", host=args.host, port=args.port, path=args.path)


if __name__ == "__main__":
    main()
