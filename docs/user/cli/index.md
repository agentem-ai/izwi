# CLI Reference

The `izwi` command-line interface provides complete control over Izwi's audio inference capabilities.

---

## Installation

The CLI is installed automatically with Izwi. Verify installation:

```bash
izwi --version
```

---

## Global Options

These options work with all commands:

| Option | Description |
|--------|-------------|
| `--server <URL>` | Server URL (default: `http://localhost:8080`) |
| `--config <PATH>` | Configuration file path |
| `--output-format <FORMAT>` | Output format: `table`, `json`, `plain`, `yaml` |
| `--quiet` | Suppress all output except results |
| `--verbose` | Enable verbose output |
| `--no-color` | Disable colored output |
| `--help` | Show help information |
| `--version` | Show version |

---

## Commands

### Server

| Command | Description |
|---------|-------------|
| [`serve`](./serve.md) | Start the inference server |
| [`status`](./status.md) | Show server health and status |

### Models

| Command | Description |
|---------|-------------|
| [`list`](./list.md) | List available models |
| [`pull`](./pull.md) | Download a model |
| [`rm`](./rm.md) | Remove a downloaded model |
| [`models`](./models.md) | Model management subcommands |

### Inference

| Command | Description |
|---------|-------------|
| [`tts`](./tts.md) | Text-to-speech generation |
| [`transcribe`](./transcribe.md) | Speech-to-text transcription |
| [`chat`](./chat.md) | Interactive chat |

### Utilities

| Command | Description |
|---------|-------------|
| [`bench`](./bench.md) | Run benchmarks |
| [`config`](./config.md) | Manage configuration |
| [`completions`](./completions.md) | Generate shell completions |
| [`version`](./version.md) | Show version information |

---

## Quick Examples

### Start the server

```bash
izwi serve
izwi serve --mode desktop
izwi serve --port 9000
```

### Download and use models

```bash
izwi list
izwi pull qwen3-tts-0.6b-base
izwi list --local
```

### Generate speech

```bash
izwi tts "Hello world" --output hello.wav
izwi tts "Hello world" --play
```

### Transcribe audio

```bash
izwi transcribe audio.wav
izwi transcribe audio.wav --format json
```

### Interactive chat

```bash
izwi chat
izwi chat --system "You are a helpful assistant"
```

---

## Getting Help

Get help for any command:

```bash
izwi --help
izwi serve --help
izwi tts --help
```

---

## Environment Variables

| Variable | Description |
|----------|-------------|
| `IZWI_HOST` | Server host |
| `IZWI_PORT` | Server port |
| `IZWI_MODELS_DIR` | Models directory |
| `IZWI_USE_METAL` | Enable Metal acceleration |
| `IZWI_MAX_BATCH_SIZE` | Maximum batch size |
| `IZWI_MAX_CONCURRENT` | Maximum concurrent requests |
| `IZWI_TIMEOUT` | Request timeout (seconds) |
| `RUST_LOG` | Log level |
| `NO_COLOR` | Disable colored output |

---

## See Also

- [Getting Started](../getting-started.md)
- [Features](../features/index.md)
- [Troubleshooting](../troubleshooting.md)
