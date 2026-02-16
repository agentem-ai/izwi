# Chat

Have text-based conversations with AI models running locally on your device.

---

## Overview

Izwi's chat feature provides:

- **Local AI** — Models run entirely on your device
- **Privacy** — No data sent to external servers
- **Multiple models** — Choose from available chat models
- **Context memory** — Maintains conversation history
- **System prompts** — Customize AI behavior

---

## Getting Started

### Download a Chat Model

```bash
izwi pull qwen3-chat-0.6b-4bit
```

### Start Chatting

**Command line:**

```bash
izwi chat
```

**Web UI:**

```
http://localhost:8080/chat
```

---

## Using the CLI

### Interactive Mode

Start an interactive chat session:

```bash
izwi chat
```

Type your messages and press Enter. Type `exit` or `quit` to end.

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `--model`, `-m` | Chat model to use | `qwen3-chat-0.6b-4bit` |
| `--system`, `-s` | System prompt | — |
| `--voice`, `-v` | Voice for spoken responses | — |

### Examples

**With custom system prompt:**

```bash
izwi chat --system "You are a helpful coding assistant."
```

**With specific model:**

```bash
izwi chat --model qwen3-chat-0.6b-4bit
```

**With voice responses:**

```bash
izwi chat --voice default
```

---

## Using the Web UI

1. Navigate to **Chat** in the sidebar
2. Type your message in the input field
3. Press Enter or click Send
4. View the AI response

### Features

- **Conversation history** — Scroll through past messages
- **Clear chat** — Start a fresh conversation
- **Model selection** — Switch between loaded models
- **Copy responses** — One-click copy

---

## Using the API

### Endpoint

```
POST /v1/chat/completions
```

### Request

```json
{
  "model": "qwen3-chat-0.6b-4bit",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"}
  ],
  "temperature": 0.7,
  "max_tokens": 1000
}
```

### Response

```json
{
  "id": "chat-123",
  "object": "chat.completion",
  "created": 1234567890,
  "model": "qwen3-chat-0.6b-4bit",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Hello! How can I help you today?"
      },
      "finish_reason": "stop"
    }
  ]
}
```

### Streaming

For streaming responses, add `"stream": true`:

```json
{
  "model": "qwen3-chat-0.6b-4bit",
  "messages": [...],
  "stream": true
}
```

### Example (curl)

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-chat-0.6b-4bit",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

---

## System Prompts

Customize AI behavior with system prompts:

**Coding assistant:**
```
You are an expert programmer. Provide clear, well-commented code examples.
```

**Writing helper:**
```
You are a professional editor. Help improve writing clarity and style.
```

**Concise responder:**
```
You are a helpful assistant. Keep responses brief and to the point.
```

---

## Available Models

| Model | Size | Speed | Notes |
|-------|------|-------|-------|
| `qwen3-chat-0.6b-4bit` | ~400 MB | Fast | Compact, efficient |

More models coming soon.

---

## Tips

1. **Be specific** — Clear questions get better answers
2. **Use system prompts** — Guide the AI's behavior
3. **Break down complex tasks** — Ask step by step
4. **Provide context** — Include relevant background

---

## See Also

- [Voice Mode](./voice.md) — Spoken conversations
- [Models](../models/index.md) — Download more models
- [CLI Reference](../cli/index.md) — Full command documentation
