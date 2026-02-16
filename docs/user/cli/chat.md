# izwi chat

Interactive chat with AI models.

---

## Synopsis

```bash
izwi chat [OPTIONS]
```

---

## Description

Starts an interactive chat session with a loaded chat model. Type messages and receive AI responses in real-time.

---

## Options

| Option | Description | Default |
|--------|-------------|---------|
| `-m, --model <MODEL>` | Chat model to use | `qwen3-0.6b-4bit` |
| `-s, --system <PROMPT>` | System prompt | — |
| `-v, --voice <VOICE>` | Voice for spoken responses | — |

---

## Examples

### Start chat

```bash
izwi chat
```

### With system prompt

```bash
izwi chat --system "You are a helpful coding assistant"
```

### With specific model

```bash
izwi chat --model qwen3-chat-0.6b-4bit
```

### With voice responses

```bash
izwi chat --voice default
```

---

## Interactive Commands

During a chat session:

| Command | Action |
|---------|--------|
| Type message + Enter | Send message |
| `exit` or `quit` | End session |
| `clear` | Clear conversation |
| `Ctrl+C` | Exit immediately |

---

## See Also

- [Chat Guide](../features/chat.md)
- [Voice Mode](../features/voice.md)
