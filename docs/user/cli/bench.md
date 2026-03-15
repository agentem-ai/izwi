# izwi bench

Run performance benchmarks.

---

## Synopsis

```bash
izwi bench <COMMAND>
```

---

## Subcommands

| Command | Description |
|---------|-------------|
| `chat` | Benchmark chat inference |
| `tts` | Benchmark TTS inference |
| `asr` | Benchmark ASR inference |
| `throughput` | Benchmark system throughput |

---

## izwi bench chat

Benchmark chat performance, including time-to-first-token.

```bash
izwi bench chat [OPTIONS]
```

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `-m, --model <MODEL>` | Model to benchmark | `Qwen3.5-4B` |
| `-i, --iterations <N>` | Number of requests | `10` |
| `-p, --prompt <TEXT>` | User prompt to send | Default benchmark prompt |
| `--system <TEXT>` | Optional system prompt | — |
| `--max-tokens <N>` | Maximum completion tokens | `128` |
| `-c, --concurrent <N>` | Concurrent requests | `1` |
| `--warmup` | Enable warmup iteration | — |

### Example

```bash
izwi bench chat --model Qwen3.5-4B --iterations 8 --max-tokens 160 --warmup
```

---

## izwi bench tts

Benchmark text-to-speech performance.

```bash
izwi bench tts [OPTIONS]
```

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `-m, --model <MODEL>` | Model to benchmark | `qwen3-tts-0.6b-base` |
| `-i, --iterations <N>` | Number of iterations | `10` |
| `-t, --text <TEXT>` | Text to synthesize | Default test text |
| `--warmup` | Enable warmup iteration | — |

### Example

```bash
izwi bench tts --model qwen3-tts-0.6b-base --iterations 20 --warmup
```

---

## izwi bench asr

Benchmark speech recognition performance.

```bash
izwi bench asr [OPTIONS]
```

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `-m, --model <MODEL>` | Model to benchmark | `qwen3-asr-0.6b` |
| `-i, --iterations <N>` | Number of iterations | `10` |
| `-f, --file <PATH>` | Audio file to use | Built-in test audio |
| `--warmup` | Enable warmup iteration | — |

### Example

```bash
izwi bench asr --model qwen3-asr-0.6b --file test.wav --iterations 20
```

---

## izwi bench throughput

Benchmark overall system throughput.

```bash
izwi bench throughput [OPTIONS]
```

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `-d, --duration <SECONDS>` | Test duration | `30` |
| `-c, --concurrent <N>` | Concurrent requests | `1` |

### Example

```bash
izwi bench throughput --duration 60 --concurrent 4
```

---

## Output

Benchmarks report:

- **TTFT** — Time to first streamed token for chat benchmarks
- **Latency** — Average, min, max, p50, p95, p99
- **Throughput** — Requests per second
- **Tokens/second** — For TTS benchmarks
- **Real-time factor** — Audio duration vs processing time

---

## See Also

- [`izwi status`](./status.md) — System status
