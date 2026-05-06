# Control Suite MCP Dummy

This project demonstrates the target deployment pattern:

```text
MCP client -> MCP over HTTP -> FastMCP server -> ZMQ -> instrument worker
```

The instrument worker recreates a simulated image-acquisition instrument. It
owns the instrument state and executes commands through a blocking ZMQ `REP`
loop. The MCP server exposes FastMCP tools and forwards each call to the worker
over a local ZMQ request.

## Setup

```bash
uv sync
```

## Run

Start both processes with one supervisor command:

```bash
uv run control-suite --worker-endpoint tcp://127.0.0.1:5555 --mcp-host 0.0.0.0 --mcp-port 8050 --mcp-path /mcp
```

The launcher starts the worker first, waits for a ZMQ health response, then
starts the MCP server. The worker and MCP server remain separate OS processes.

You can also start each process manually.

Start the worker:

```bash
uv run control-suite-worker --bind tcp://127.0.0.1:5555
```

Start the MCP HTTP server in another terminal:

```bash
uv run control-suite-mcp --worker tcp://127.0.0.1:5555 --host 0.0.0.0 --port 8050 --path /mcp
```

Any MCP client that supports HTTP transport can connect to:

```text
http://127.0.0.1:8050/mcp
```

## Client Configuration

Many MCP clients use a JSON configuration with an `mcpServers` table. For this
server, the client entry should point to the HTTP MCP endpoint:

```json
{
  "mcpServers": {
    "control_suite_dummy": {
      "url": "http://127.0.0.1:8050/mcp",
      "transport": "http"
    }
  }
}
```

Codex uses TOML in `~/.codex/config.toml`. Add this entry:

```toml
[mcp_servers.control-suite-dummy]
url = "http://127.0.0.1:8050/mcp"
```

The equivalent Codex CLI command is:

```bash
codex mcp add control-suite-dummy --url http://127.0.0.1:8050/mcp
```

## Tools

- `acquire_image(y_center, x_center, size_y, size_x, scan_step=1.0)`
- `acquire_line_scan(x_center, y_center, length, scan_step, angle=0.0)`
- `set_blur(blur)`
- `set_offset(y_offset, x_offset)`
- `get_state()`
- `health()`
