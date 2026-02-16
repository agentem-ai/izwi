# izwi models

Model management commands.

---

## Synopsis

```bash
izwi models <COMMAND>
```

---

## Subcommands

| Command | Description |
|---------|-------------|
| `list` | List available models |
| `info` | Show model information |
| `load` | Load a model into memory |
| `unload` | Unload a model from memory |
| `progress` | Show download progress |

---

## izwi models list

List available models.

```bash
izwi models list
izwi models list --local
izwi models list --detailed
```

Same as [`izwi list`](./list.md).

---

## izwi models info

Show detailed information about a model.

```bash
izwi models info <MODEL>
izwi models info qwen3-tts-0.6b-base
izwi models info qwen3-tts-0.6b-base --json
```

### Options

| Option | Description |
|--------|-------------|
| `--json` | Output raw JSON |

---

## izwi models load

Load a model into memory for inference.

```bash
izwi models load <MODEL>
izwi models load qwen3-tts-0.6b-base
izwi models load qwen3-tts-0.6b-base --wait
```

### Options

| Option | Description |
|--------|-------------|
| `-w, --wait` | Wait for model to be fully loaded |

---

## izwi models unload

Unload a model from memory.

```bash
izwi models unload <MODEL>
izwi models unload qwen3-tts-0.6b-base
izwi models unload all --yes
```

### Arguments

| Argument | Description |
|----------|-------------|
| `<MODEL>` | Model variant to unload, or `all` |

### Options

| Option | Description |
|--------|-------------|
| `-y, --yes` | Unload without confirmation |

---

## izwi models progress

Show download progress for active downloads.

```bash
izwi models progress
izwi models progress qwen3-tts-0.6b-base
```

---

## See Also

- [`izwi list`](./list.md)
- [`izwi pull`](./pull.md)
- [`izwi rm`](./rm.md)
