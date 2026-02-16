# Diarization

Identify and separate multiple speakers in audio recordings with speaker diarization.

---

## Overview

Speaker diarization answers the question "who spoke when?" It segments audio by speaker, making it invaluable for:

- **Meeting transcripts** — Attribute statements to participants
- **Interviews** — Separate interviewer and interviewee
- **Podcasts** — Identify hosts and guests
- **Call recordings** — Distinguish callers

---

## Getting Started

### Download an ASR Model

Diarization uses ASR models with speaker detection:

```bash
izwi pull qwen3-asr-0.6b
```

### Start the Server

```bash
izwi serve
```

---

## Using the Web UI

1. Navigate to **Diarization** in the sidebar
2. Upload an audio file with multiple speakers
3. Click **Analyze**
4. View the speaker-segmented transcript

### Output

The diarization view shows:
- **Speaker labels** — Speaker 1, Speaker 2, etc.
- **Timestamps** — When each speaker talks
- **Transcript** — What each speaker said

Example output:

```
[00:00 - 00:05] Speaker 1: Welcome to the meeting.
[00:05 - 00:12] Speaker 2: Thanks for having me.
[00:12 - 00:20] Speaker 1: Let's start with the agenda.
```

---

## Using the API

### Endpoint

```
POST /v1/audio/diarize
```

### Request (multipart/form-data)

| Field | Type | Description |
|-------|------|-------------|
| `file` | File | Audio file to analyze |
| `model` | String | Model name |
| `num_speakers` | Integer | Expected speakers (optional) |

### Example (curl)

```bash
curl -X POST http://localhost:8080/v1/audio/diarize \
  -F "file=@meeting.wav" \
  -F "model=qwen3-asr-0.6b"
```

### Response

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

## Configuration

### Number of Speakers

If you know how many speakers are in the audio, specify it for better accuracy:

```bash
# Via API
curl -X POST http://localhost:8080/v1/audio/diarize \
  -F "file=@meeting.wav" \
  -F "num_speakers=3"
```

### Speaker Labels

By default, speakers are labeled "Speaker 1", "Speaker 2", etc. You can rename them in the UI after processing.

---

## Tips for Best Results

1. **Quality audio** — Clear recordings with minimal background noise
2. **Distinct voices** — Works best when speakers have different voice characteristics
3. **Minimal overlap** — Speakers talking over each other reduces accuracy
4. **Specify speaker count** — If known, helps the algorithm
5. **Longer segments** — Short utterances are harder to attribute

---

## Limitations

- **Similar voices** — May confuse speakers with very similar voices
- **Overlapping speech** — Simultaneous talking is challenging
- **Background noise** — Reduces speaker detection accuracy
- **Very short clips** — Need enough audio to identify speaker patterns

---

## Use Cases

### Meeting Minutes

Upload a meeting recording to get a transcript with speaker attribution:

1. Record your meeting
2. Upload to Diarization
3. Export the speaker-labeled transcript
4. Edit speaker names as needed

### Interview Transcription

Perfect for journalist interviews or research:

1. Record the interview
2. Process with diarization
3. Get clean Q&A format output

### Podcast Production

Identify speakers for editing and show notes:

1. Upload raw podcast audio
2. See who spoke when
3. Use timestamps for editing

---

## See Also

- [Transcription](./transcription.md) — Single-speaker transcription
- [Voice Mode](./voice.md) — Real-time conversations
- [CLI Reference](../cli/index.md) — Command documentation
