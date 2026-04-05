# izwi diarize

Speaker diarization — identify and separate multiple speakers in audio.

---

## Synopsis

```bash
izwi diarize <FILE> [OPTIONS]
```

---

## Description

Analyzes audio to identify different speakers and when they spoke. Optionally includes transcription with speaker labels.

---

## Arguments

| Argument | Description |
|----------|-------------|
| `<FILE>` | Audio file to analyze |

---

## Options

| Option | Description | Default |
|--------|-------------|---------|
| `-m, --model <MODEL>` | Diarization model | `sortformer-4spk` |
| `-n, --num-speakers <N>` | Expected number of speakers | Auto-detect |
| `-f, --format <FORMAT>` | Output format: `text`, `json`, `verbose_json` | `text` |
| `-o, --output <PATH>` | Output file (default: stdout) | — |
| `--transcribe` | Include transcription with speaker labels | — |
| `--asr-model <MODEL>` | ASR model for transcription | `parakeet-tdt-0.6b-v3` |

---

## Examples

### Basic diarization

```bash
izwi diarize meeting.wav
```

### With known speaker count

```bash
izwi diarize meeting.wav --num-speakers 3
```

### With transcription

```bash
izwi diarize meeting.wav --transcribe
```

### JSON output

```bash
izwi diarize meeting.wav --format json --output diarization.json
```

### Full pipeline with custom models

```bash
izwi diarize interview.wav \
  --transcribe \
  --asr-model Qwen3-ASR-1.7B-GGUF \
  --format verbose_json \
  --output interview_transcript.json
```

---

## Output Formats

### Text

```
[00:00 - 00:05] Speaker 1: Welcome to the meeting.
[00:05 - 00:12] Speaker 2: Thanks for having me.
[00:12 - 00:20] Speaker 1: Let's start with the agenda.
```

### JSON

```json
{
  "segments": [
    {"speaker": "Speaker 1", "start": 0.0, "end": 5.2},
    {"speaker": "Speaker 2", "start": 5.5, "end": 12.1}
  ],
  "num_speakers": 2
}
```

### Verbose JSON (with transcription)

```json
{
  "segments": [
    {
      "speaker": "Speaker 1",
      "start": 0.0,
      "end": 5.2,
      "text": "Welcome to the meeting."
    },
    {
      "speaker": "Speaker 2", 
      "start": 5.5,
      "end": 12.1,
      "text": "Thanks for having me."
    }
  ],
  "num_speakers": 2,
  "duration": 120.5
}
```

---

## Available Models

| Model | Description |
|-------|-------------|
| `sortformer-4spk` | Alias for `diar_streaming_sortformer_4spk-v2.1` (default) |
| `diar_streaming_sortformer_4spk-v2.1` | Canonical Sortformer model ID |

---

## See Also

- [Diarization Guide](../features/diarization.md)
- [`izwi transcribe`](./transcribe.md) — Single-speaker transcription
