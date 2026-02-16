# izwi config

Manage configuration.

---

## Synopsis

```bash
izwi config <COMMAND>
```

---

## Subcommands

| Command | Description |
|---------|-------------|
| `show` | Show current configuration |
| `set` | Set a configuration value |
| `get` | Get a configuration value |
| `edit` | Edit in default editor |
| `reset` | Reset to defaults |
| `path` | Show config file path |

---

## izwi config show

Display the current configuration.

```bash
izwi config show
```

---

## izwi config set

Set a configuration value.

```bash
izwi config set <KEY> <VALUE>
```

### Examples

```bash
izwi config set server.host 0.0.0.0
izwi config set server.port 9000
izwi config set models.dir /path/to/models
```

---

## izwi config get

Get a specific configuration value.

```bash
izwi config get <KEY>
```

### Examples

```bash
izwi config get server.port
izwi config get models.dir
```

---

## izwi config edit

Open the configuration file in your default editor.

```bash
izwi config edit
```

Uses `$EDITOR` environment variable, or falls back to system default.

---

## izwi config reset

Reset configuration to defaults.

```bash
izwi config reset
izwi config reset --yes
```

### Options

| Option | Description |
|--------|-------------|
| `-y, --yes` | Reset without confirmation |

---

## izwi config path

Show the configuration file path.

```bash
izwi config path
```

---

## Configuration File

The configuration file is TOML format:

```toml
[server]
host = "0.0.0.0"
port = 8080

[models]
dir = "/path/to/models"

[inference]
max_batch_size = 8
use_metal = true
```

### File Locations

| Platform | Path |
|----------|------|
| **macOS** | `~/Library/Application Support/izwi/config.toml` |
| **Linux** | `~/.config/izwi/config.toml` |
| **Windows** | `%APPDATA%\izwi\config.toml` |

---

## See Also

- [`izwi serve`](./serve.md) — Server options
