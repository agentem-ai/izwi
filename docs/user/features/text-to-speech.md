# Text-to-Speech

Generate natural, human-like speech from text using state-of-the-art TTS models.

---

## Overview

Izwi's text-to-speech converts written text into spoken audio. Features include:

- **Natural voices** — High-quality, expressive speech
- **Multiple formats** — WAV, MP3, OGG, FLAC, AAC
- **Speed control** — Adjust playback speed
- **Streaming** — Real-time audio generation
- **Local processing** — No cloud, complete privacy

---

## Getting Started

### Download a TTS Model

```bash
izwi pull qwen3-tts-0.6b-base
```

### Kokoro-82M Prerequisite (`espeak-ng`)

If you plan to use `Kokoro-82M`, install `espeak-ng` on your system first.
Izwi uses it for Kokoro phonemization and will return an error if it is missing.

- macOS: see [macOS Installation](../installation/macos.md#optional-install-espeak-ng-for-kokoro-82m)
- Linux: see [Linux Installation](../installation/linux.md#optional-install-espeak-ng-for-kokoro-82m)
- Windows: see [Windows Installation](../installation/windows.md#optional-install-espeak-ng-for-kokoro-82m)

### Generate Speech

**Command line:**

```bash
izwi tts "Hello, welcome to Izwi!" --output hello.wav
```

**With playback:**

```bash
izwi tts "Hello, welcome to Izwi!" --play
```

---

## Using the CLI

### Basic Usage

```bash
izwi tts "Your text here" --output output.wav
```

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `--model`, `-m` | TTS model to use | `qwen3-tts-0.6b-base` |
| `--output`, `-o` | Output file path | stdout |
| `--format`, `-f` | Audio format | `wav` |
| `--speed`, `-r` | Speech speed (0.5-2.0) | `1.0` |
| `--speaker`, `-s` | Voice/speaker ID | `default` |
| `--temperature`, `-t` | Sampling temperature | `0.7` |
| `--play`, `-p` | Play audio after generation | — |
| `--stream` | Stream output in real-time | — |

### Examples

**Different formats:**

```bash
izwi tts "Hello world" --format mp3 --output hello.mp3
izwi tts "Hello world" --format ogg --output hello.ogg
```

**Adjust speed:**

```bash
# Slower (0.5x - 1.0x)
izwi tts "Speaking slowly" --speed 0.75 --output slow.wav

# Faster (1.0x - 2.0x)
izwi tts "Speaking quickly" --speed 1.5 --output fast.wav
```

**Read from stdin:**

```bash
echo "Text from pipe" | izwi tts - --output piped.wav
cat article.txt | izwi tts - --output article.wav
```

**Streaming output:**

```bash
izwi tts "Long text for streaming" --stream --play
```

---

## Using the Web UI

1. Navigate to **Text to Speech** in the sidebar
2. Enter your text in the input field
3. Select a voice (if available)
4. Click **Generate**
5. Play or download the audio

### Features

- **Live preview** — Hear audio as it generates
- **Download** — Save audio files locally
- **History** — Access recent generations

---

## Using the API

### Endpoint

```
POST /v1/audio/speech
```

### Request

```json
{
  "model": "qwen3-tts-0.6b-base",
  "input": "Hello, this is a test.",
  "voice": "default",
  "speed": 1.0,
  "response_format": "wav"
}
```

### Response

Binary audio data with appropriate `Content-Type` header.

### Example (curl)

```bash
curl -X POST http://localhost:8080/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen3-tts-0.6b-base", "input": "Hello world"}' \
  --output speech.wav
```

---

## Available Models

| Model | Size | Quality | Speed |
|-------|------|---------|-------|
| `Kokoro-82M` | ~0.4 GB | Good | Fast |
| `qwen3-tts-0.6b-base` | 1.2 GB | Good | Fast |
| `qwen3-tts-1.7b-base` | 3.4 GB | Better | Medium |

For voice cloning, use `customvoice` variants. For voice design, use `voicedesign` variants.
`Kokoro-82M` requires `espeak-ng` to be installed separately.

---

## Audio Formats

| Format | Extension | Notes |
|--------|-----------|-------|
| WAV | `.wav` | Uncompressed, highest quality |
| MP3 | `.mp3` | Compressed, widely compatible |
| OGG | `.ogg` | Open format, good compression |
| FLAC | `.flac` | Lossless compression |
| AAC | `.aac` | High efficiency compression |

---

## Tips

1. **Punctuation matters** — Use proper punctuation for natural pauses
2. **Break long text** — Split very long text into paragraphs
3. **Test different speeds** — Find the right pace for your use case
4. **Use appropriate models** — Larger models = better quality but slower

---

## See Also

- [Voice Cloning](./voice-cloning.md) — Clone custom voices
- [Voice Design](./voice-design.md) — Create voices from descriptions
- [CLI Reference](../cli/index.md) — Full command documentation
