# izwi tts

Generate speech from text.

---

## Synopsis

```bash
izwi tts <TEXT> [OPTIONS]
```

---

## Description

Converts text to speech using a TTS model. Supports multiple output formats, voice selection, and real-time streaming.

### Kokoro-82M Prerequisite (`espeak-ng`)

`Kokoro-82M` requires `espeak-ng` to be installed on the host system (used for phonemization).

- Install instructions:
  - [macOS](../installation/macos.md#optional-install-espeak-ng-for-kokoro-82m)
  - [Linux](../installation/linux.md#optional-install-espeak-ng-for-kokoro-82m)
  - [Windows](../installation/windows.md#optional-install-espeak-ng-for-kokoro-82m)

---

## Arguments

| Argument | Description |
|----------|-------------|
| `<TEXT>` | Text to synthesize (use `-` to read from stdin) |

---

## Options

| Option | Description | Default |
|--------|-------------|---------|
| `-m, --model <MODEL>` | TTS model to use | `qwen3-tts-0.6b-base` |
| `-s, --speaker <VOICE>` | Voice/speaker (name or audio path) | `default` |
| `-o, --output <PATH>` | Output file path | stdout |
| `-f, --format <FORMAT>` | Audio format: `wav`, `mp3`, `ogg`, `flac`, `aac` | `wav` |
| `-r, --speed <SPEED>` | Speech speed (0.5-2.0) | `1.0` |
| `-t, --temperature <TEMP>` | Sampling temperature | `0.7` |
| `--stream` | Stream output in real-time | — |
| `-p, --play` | Play audio after generation | — |

---

## Examples

### Basic usage

```bash
izwi tts "Hello, world!" --output hello.wav
```

### Kokoro-82M

```bash
izwi tts "Hello my name is Bella" \
  --model Kokoro-82M \
  --speaker af_bella \
  --output kokoro.wav
```

### Play immediately

```bash
izwi tts "Hello, world!" --play
```

### Different format

```bash
izwi tts "Hello, world!" --format mp3 --output hello.mp3
```

### Adjust speed

```bash
# Slower
izwi tts "Speaking slowly" --speed 0.75 --output slow.wav

# Faster
izwi tts "Speaking quickly" --speed 1.5 --output fast.wav
```

### Read from stdin

```bash
echo "Text from pipe" | izwi tts - --output piped.wav
cat article.txt | izwi tts - --output article.wav
```

### Voice cloning

```bash
izwi tts "Hello in cloned voice" \
  --model qwen3-tts-0.6b-customvoice \
  --speaker /path/to/reference.wav \
  --output cloned.wav
```

### Voice design

```bash
izwi tts "Hello in designed voice" \
  --model qwen3-tts-0.6b-voicedesign \
  --speaker "A warm, friendly female voice" \
  --output designed.wav
```

### Streaming with playback

```bash
izwi tts "Long text for streaming" --stream --play
```

---

## Audio Formats

| Format | Extension | Notes |
|--------|-----------|-------|
| `wav` | `.wav` | Uncompressed, highest quality |
| `mp3` | `.mp3` | Compressed, widely compatible |
| `ogg` | `.ogg` | Open format, good compression |
| `flac` | `.flac` | Lossless compression |
| `aac` | `.aac` | High efficiency compression |

---

## Models

| Model | Type | Description |
|-------|------|-------------|
| `Kokoro-82M` | Standard | Lightweight TTS (requires `espeak-ng`) |
| `qwen3-tts-0.6b-base` | Standard | General-purpose TTS |
| `qwen3-tts-0.6b-customvoice` | Cloning | Voice cloning support |
| `qwen3-tts-0.6b-voicedesign` | Design | Voice from descriptions |
| `qwen3-tts-1.7b-*` | Larger | Higher quality variants |

---

## See Also

- [Text-to-Speech Guide](../features/text-to-speech.md)
- [Voice Cloning Guide](../features/voice-cloning.md)
- [Voice Design Guide](../features/voice-design.md)
