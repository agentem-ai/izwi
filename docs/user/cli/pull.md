# izwi pull

Download a model from Hugging Face.

---

## Synopsis

```bash
izwi pull <MODEL> [OPTIONS]
```

---

## Description

Downloads a model from the Hugging Face Hub and caches it locally. Supports resuming interrupted downloads.

---

## Arguments

| Argument | Description |
|----------|-------------|
| `<MODEL>` | Model variant to download |

---

## Options

| Option | Description | Default |
|--------|-------------|---------|
| `-f, --force` | Force re-download even if exists | — |
| `-y, --yes` | Download without confirmation | — |

---

## Examples

### Download a model

```bash
izwi pull qwen3-tts-0.6b-base
```

### Skip confirmation

```bash
izwi pull qwen3-tts-0.6b-base --yes
```

### Force re-download

```bash
izwi pull qwen3-tts-0.6b-base --force
```

---

## Available Models

Run `izwi list` to see all available models.

Common models:

| Model | Type | Size |
|-------|------|------|
| `qwen3-tts-0.6b-base` | TTS | ~1.2 GB |
| `qwen3-tts-0.6b-customvoice` | TTS (cloning) | ~1.2 GB |
| `qwen3-tts-0.6b-voicedesign` | TTS (design) | ~1.2 GB |
| `qwen3-asr-0.6b` | ASR | ~1.2 GB |
| `qwen3-chat-0.6b-4bit` | Chat | ~400 MB |

---

## Resume Downloads

If a download is interrupted, run the same command again. The download will resume from where it left off.

---

## See Also

- [`izwi list`](./list.md) — List models
- [`izwi rm`](./rm.md) — Remove models
- [Models Guide](../models/index.md) — Model documentation
