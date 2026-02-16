# izwi status

Show server health and status.

---

## Synopsis

```bash
izwi status [OPTIONS]
```

---

## Description

Displays the current state of the Izwi server, including health, loaded models, and resource usage.

---

## Options

| Option | Description | Default |
|--------|-------------|---------|
| `-d, --detailed` | Show detailed metrics | — |
| `-w, --watch <SECONDS>` | Continuous updates | — |

---

## Examples

### Basic status

```bash
izwi status
```

### Detailed metrics

```bash
izwi status --detailed
```

### Watch mode

```bash
izwi status --watch 2
```

Updates every 2 seconds. Press `Ctrl+C` to stop.

---

## Output

The status command shows:

- **Server health** — Running, stopped, or error
- **Loaded models** — Currently loaded models
- **Memory usage** — RAM and VRAM usage
- **GPU status** — Metal/CUDA availability
- **Active requests** — Current request count

---

## See Also

- [`izwi serve`](./serve.md) — Start the server
- [`izwi models`](./models.md) — Model management
