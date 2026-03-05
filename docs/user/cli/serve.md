# izwi serve

Start the Izwi inference server.

---

## Synopsis

```bash
izwi serve [OPTIONS]
```

---

## Description

Launches the HTTP API server that powers all Izwi functionality. The server provides:

- REST API endpoints (OpenAI-compatible)
- Web UI (unless disabled)
- Model management
- Real-time inference

---

## Options

| Option | Description | Default |
|--------|-------------|---------|
| `--mode <MODE>` | Startup mode: `server`, `desktop`, `web` | `server` |
| `-H, --host <HOST>` | Host to bind to | `0.0.0.0` |
| `-p, --port <PORT>` | Port to listen on | `8080` |
| `-m, --models-dir <PATH>` | Models directory | Platform default |
| `--max-batch-size <N>` | Maximum batch size | `8` |
| `--backend <BACKEND>` | Backend preference: `auto`, `cpu`, `metal`, `cuda` | `auto` |
| `-t, --threads <N>` | Number of CPU threads | Auto |
| `--max-concurrent <N>` | Max concurrent requests | `100` |
| `--timeout <SECONDS>` | Request timeout | `300` |
| `--log-level <LEVEL>` | Log level | `warn` |
| `--cors` | Enable CORS for all origins | — |
| `--no-ui` | Disable the web UI | — |

---

## Modes

### Server Mode (Default)

Starts only the HTTP server:

```bash
izwi serve
izwi serve --mode server
```

Access at `http://localhost:8080`

### Desktop Mode

Starts the server and opens the native desktop application:

```bash
izwi serve --mode desktop
```

### Web Mode

Starts the server and opens the web UI in your default browser:

```bash
izwi serve --mode web
```

---

## Examples

### Basic server

```bash
izwi serve
```

### Custom port

```bash
izwi serve --port 9000
```

### With Metal backend (macOS)

```bash
izwi serve --backend metal
```

### Custom models directory

```bash
izwi serve --models-dir /path/to/models
```

### Production settings

```bash
izwi serve \
  --host 0.0.0.0 \
  --port 8080 \
  --max-concurrent 200 \
  --timeout 600 \
  --log-level info
```

### Development mode

```bash
izwi serve --cors --log-level debug
```

---

## Environment Variables

| Variable | Equivalent Option |
|----------|-------------------|
| `IZWI_HOST` | `--host` |
| `IZWI_PORT` | `--port` |
| `IZWI_MODELS_DIR` | `--models-dir` |
| `IZWI_BACKEND` | `--backend` |
| `IZWI_MAX_BATCH_SIZE` | `--max-batch-size` |
| `IZWI_NUM_THREADS` | `--threads` |
| `IZWI_MAX_CONCURRENT` | `--max-concurrent` |
| `IZWI_TIMEOUT` | `--timeout` |
| `RUST_LOG` | `--log-level` |
| `IZWI_SERVE_MODE` | `--mode` |

---

## Graceful Shutdown

Press `Ctrl+C` to gracefully shut down the server. Active requests will complete before shutdown.

---

## See Also

- [`izwi status`](./status.md) — Check server health
- [`izwi config`](./config.md) — Manage configuration
