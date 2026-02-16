# Models

Izwi uses AI models for text-to-speech, speech recognition, and chat. This guide explains how to find, download, and manage models.

---

## Available Models

Izwi supports several model families optimized for different tasks:

### Text-to-Speech (TTS)

| Model | Size | Description |
|-------|------|-------------|
| `qwen3-tts-0.6b-base` | ~1.2 GB | Fast, general-purpose TTS |
| `qwen3-tts-0.6b-customvoice` | ~1.2 GB | TTS with voice cloning support |
| `qwen3-tts-0.6b-voicedesign` | ~1.2 GB | TTS with voice design from descriptions |
| `qwen3-tts-1.7b-base` | ~3.4 GB | Higher quality TTS |
| `qwen3-tts-1.7b-customvoice` | ~3.4 GB | Higher quality with voice cloning |

### Speech Recognition (ASR)

| Model | Size | Description |
|-------|------|-------------|
| `qwen3-asr-0.6b` | ~1.2 GB | Fast speech-to-text |
| `qwen3-asr-1.7b` | ~3.4 GB | Higher accuracy transcription |

### Chat

| Model | Size | Description |
|-------|------|-------------|
| `qwen3-chat-0.6b-4bit` | ~400 MB | Compact chat model |

### Specialized

| Model | Size | Description |
|-------|------|-------------|
| `qwen3-forced-aligner` | ~600 MB | Word-level audio alignment |

---

## Downloading Models

### Via CLI

```bash
# List all available models
izwi list

# Download a model
izwi pull qwen3-tts-0.6b-base

# Download with progress
izwi pull qwen3-asr-0.6b
```

### Via Web UI

1. Open `http://localhost:8080`
2. Go to **Models** in the sidebar
3. Click **Download** on any model

---

## Managing Models

### View Downloaded Models

```bash
izwi list --local
```

### Get Model Information

```bash
izwi models info qwen3-tts-0.6b-base
```

### Load a Model into Memory

```bash
izwi models load qwen3-tts-0.6b-base
```

### Unload a Model

```bash
izwi models unload qwen3-tts-0.6b-base
```

### Delete a Model

```bash
izwi rm qwen3-tts-0.6b-base
```

---

## Model Storage

Models are stored in your system's application data directory:

| Platform | Location |
|----------|----------|
| **macOS** | `~/Library/Application Support/izwi/models/` |
| **Linux** | `~/.local/share/izwi/models/` |
| **Windows** | `%APPDATA%\izwi\models\` |

### Custom Model Directory

Set a custom location:

```bash
# Via CLI flag
izwi serve --models-dir /path/to/models

# Via environment variable
export IZWI_MODELS_DIR=/path/to/models
izwi serve
```

---

## Manual Downloads

Some models require manual download from Hugging Face due to licensing:

- [Manual Download: Gemma 3 1B](./manual-gemma-3-1b-download.md)
- [Manual Download Guide](./manual-download.md)

---

## Model Status

Models can be in several states:

| Status | Description |
|--------|-------------|
| **not_downloaded** | Model available but not on disk |
| **downloading** | Currently downloading |
| **downloaded** | On disk but not loaded |
| **loading** | Being loaded into memory |
| **ready** | Loaded and ready for inference |

Check status:

```bash
izwi status --detailed
```

---

## Quantized Models

Some models offer quantized variants for reduced memory usage:

- **4-bit** — Smallest size, some quality loss
- **8-bit** — Balanced size and quality
- **Full** — Original quality, largest size

Quantized models have suffixes like `-4bit` or `-q4`.

---

## Next Steps

- [Manual Model Downloads](./manual-download.md)
- [CLI Reference](../cli/index.md)
- [Troubleshooting](../troubleshooting.md)
