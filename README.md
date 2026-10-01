# Control Suite MCP Dummy

This project demonstrates the target deployment pattern:

```text
MCP client -> MCP over HTTP -> FastMCP server -> ZMQ -> instrument worker
```

The instrument worker recreates a simulated image-acquisition instrument. It
owns the instrument state and executes commands through a blocking ZMQ `REP`
loop. The MCP server exposes FastMCP tools and forwards each call to the worker
over a local ZMQ request.

The two-process design is intentional. FastMCP and many instrument-control
libraries may each depend on their own event loop, callback thread, or runtime
lifecycle. Running the MCP HTTP server and the instrument controller in
separate OS processes prevents those async runtimes from sharing a Python event
loop or thread. ZMQ is used as a narrow local command boundary between the two
processes.

In this pattern:

- The FastMCP server owns MCP/HTTP request handling.
- The instrument worker owns the control library, instrument state, and command
  execution.
- ZMQ carries JSON-serializable commands and responses between them.
- The worker can keep a simple blocking command loop or adapt internally to the
  event loop required by a real instrument library.
- The launcher starts both processes for convenience, but it does not merge
  their runtimes into one process.

## Setup

```bash
uv sync
```

## Run

Start both processes with one supervisor command:

```bash
uv run control-suite --worker-endpoint tcp://127.0.0.1:5555 --mcp-host 0.0.0.0 --mcp-port 8050 --mcp-path /mcp
```

To configure the simulated acquisition and parameter-setting behavior from
YAML, pass a config file to the launcher:

```bash
uv run control-suite --config configs/simulated_parameter_tuning.yaml
```

The YAML `acquisition.image` setting accepts `.npy`, `.tif`, `.tiff`, `.png`,
`.jpg`, and `.jpeg` images. Paths are relative to the working directory.
If omitted or `null`, the worker uses the bundled 512×512 cameraman image.
RGB/RGBA images are converted to grayscale using
`0.299 R + 0.587 G + 0.114 B`; alpha is ignored. Intensities retain their original
scale and are stored as float32.

Set the source pixel spacing independently for each axis:

```yaml
acquisition:
  image: data/my_image.png
  pixel_size_x: 0.1
  pixel_size_y: 0.1
  add_axis_ticks: true
```

Both pixel sizes default to 1 and must be finite and positive. Image and line-scan
positions, region sizes, scan steps, drift offsets, and jitter use acquisition
coordinate units. The source row is `y / pixel_size_y` and the source column is
`x / pixel_size_x`; fractional indices use linear interpolation. Gaussian PSF
sigma remains in source pixels. Samples outside the source pixel centers are
zero, including when simulation effects are enabled.

Image axis ticks are enabled by default. Set `acquisition.add_axis_ticks: false`
to hide them. Tick labels show acquisition positions in the configured real units;
for example, pixel index 50 is labeled 5 when the pixel size is 0.1.

For a 100×100 image with pixel sizes of 0.1, this samples coordinates 5–6 on both
axes (upper endpoint excluded), corresponding to source rows and columns 50–59:

```python
acquire_image(y_center=5.5, x_center=5.5, size_y=1.0, size_x=1.0, scan_step=0.1)
```

The returned `psize` is the acquisition scan step. With the default cameraman
image and unit pixel sizes, acquire the full image with
`acquire_image(y_center=256, x_center=256, size_y=512, size_x=512, scan_step=1)`.

Acquisition tools return PNG image content alongside their existing JSON payloads
by default. The JSON still includes artifact paths and acquisition metadata.
The MCP frontend must be able to read the worker's PNG files.

For text-only responses, set `mcp.include_images: false` in YAML or use:

```bash
uv run control-suite --config configs/simulated_parameter_tuning.yaml --no-include-images
```

Both `control-suite` and `control-suite-mcp` accept `--config`,
`--include-images`, and `--no-include-images`. The image-content CLI flags
override the YAML setting; `--include-images` re-enables images when YAML
disables them.

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
- `set_config(name, value)`
- `set_attribute(name, value)`
- `set_parameters(parameters)`
- `get_state()`
- `health()`
