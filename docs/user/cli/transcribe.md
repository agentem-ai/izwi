# izwi transcribe

Convert audio to text.

---

## Synopsis

```bash
izwi transcribe <FILE> [OPTIONS]
```

---

## Description

Transcribes audio files to text using automatic speech recognition (ASR). Supports multiple audio formats and output options.

---

## Arguments

| Argument | Description |
|----------|-------------|
| `<FILE>` | Audio file to transcribe |

---

## Options

| Option | Description | Default |
|--------|-------------|---------|
| `-m, --model <MODEL>` | ASR model to use | `qwen3-asr-0.6b` |
| `-l, --language <LANG>` | Language hint (e.g., `en`, `es`) | Auto-detect |
| `-f, --format <FORMAT>` | Output format: `text`, `json`, `verbose_json` | `text` |
| `-o, --output <PATH>` | Output file (default: stdout) | — |
| `--word-timestamps` | Include word-level timestamps | — |

---

## Examples

### Basic transcription

```bash
izwi transcribe audio.wav
```

### Save to file

```bash
izwi transcribe audio.wav --output transcript.txt
```

### JSON output

```bash
izwi transcribe audio.wav --format json
```

### With timestamps

```bash
izwi transcribe audio.wav --format verbose_json --word-timestamps
```

### Specify language

```bash
izwi transcribe audio.wav --language en
izwi transcribe audio.wav --language es
```

### Use larger model

```bash
izwi transcribe audio.wav --model qwen3-asr-1.7b
```

---

## Output Formats

### Text

Plain text transcript:

```
Hello, this is a transcription test.
```

### JSON

```json
{
  "text": "Hello, this is a transcription test."
}
```

### Verbose JSON

```json
{
  "text": "Hello, this is a transcription test.",
  "language": "en",
  "duration": 3.5,
  "words": [
    {"word": "Hello", "start": 0.0, "end": 0.5},
    {"word": "this", "start": 0.6, "end": 0.8}
  ]
}
```

---

## Supported Audio Formats

- WAV (`.wav`)
- MP3 (`.mp3`)
- M4A (`.m4a`)
- FLAC (`.flac`)
- OGG (`.ogg`)
- WebM (`.webm`)

---

## Models

| Model | Size | Speed | Accuracy |
|-------|------|-------|----------|
| `qwen3-asr-0.6b` | 1.2 GB | Fast | Good |
| `qwen3-asr-1.7b` | 3.4 GB | Medium | Better |

---

## See Also

- [Transcription Guide](../features/transcription.md)
- [Diarization Guide](../features/diarization.md)
