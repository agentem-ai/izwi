# Izwi - Qwen3-TTS Inference Engine for Apple Silicon

A high-performance, Rust-based text-to-speech inference engine optimized for Qwen3-TTS models on Apple Silicon (M1+) using MLX.

![Izwi Screenshot](images/screenshot.png)

## Features

- **Apple Silicon Optimized**: Built on MLX for unified memory and Metal GPU acceleration
- **Streaming Audio**: Ultra-low-latency streaming with ~97ms first-packet emission
- **Model Management**: Download and manage Qwen3-TTS models directly from the UI
- **Modern Web UI**: Beautiful React-based interface for testing TTS
- **REST API**: OpenAI-compatible endpoints for easy integration
- **Voice Cloning**: Support for reference audio-based voice cloning (CustomVoice models)

## Quick Start (Recommended: CLI)

The `izwi` CLI is the easiest way to start testing and using this repository.

```bash
# Build CLI
cargo build -p izwi-cli

# Start server
./target/debug/izwi serve
```

In a second terminal:

```bash
# Check status and local models
./target/debug/izwi status
./target/debug/izwi list --local

# ASR smoke test
./target/debug/izwi transcribe data/test.wav --model qwen3-asr-0.6b --format text

# TTS smoke test
./target/debug/izwi tts "hello from izwi cli" --model qwen3-tts-0.6b-base --output /tmp/hello.wav
```

CLI documentation: `crates/izwi-cli/README.md`

## Supported Models

### Text-to-Speech (TTS)

| Model | Size | What You Can Do |
|-------|------|----------------|
| **Qwen3-TTS-12Hz-0.6B-Base** | ~1.2GB | Generate speech with 9 built-in voices (fast, lightweight) |
| **Qwen3-TTS-12Hz-0.6B-CustomVoice** | ~1.2GB | Clone any voice using a reference audio sample |
| **Qwen3-TTS-12Hz-1.7B-Base** | ~3.4GB | Generate higher quality speech with 9 built-in voices |
| **Qwen3-TTS-12Hz-1.7B-CustomVoice** | ~3.4GB | Clone any voice with better quality (requires reference audio) |
| **Qwen3-TTS-12Hz-1.7B-VoiceDesign** | ~3.4GB | Design custom voices using text descriptions (e.g., "deep male voice with British accent") |

### Speech-to-Text (ASR)

| Model | Size | What You Can Do |
|-------|------|----------------|
| **Qwen3-ASR-0.6B** | ~1.2GB | Transcribe audio to text (fast, lightweight) |
| **Qwen3-ASR-1.7B** | ~3.4GB | Transcribe audio to text with higher accuracy |

## Requirements

### Native Development
- macOS 12+ with Apple Silicon (M1/M2/M3) or Linux with CUDA
- **Rust 1.83+** (required for tokenizers dependency)
- Node.js 18+ (for UI development)

### Docker (Recommended)
- Docker 24+ with Compose V2
- NVIDIA Container Toolkit (for GPU support on Linux)

### Upgrading Rust

```bash
rustup update stable
# Or install if not present:
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

## Quick Start (Docker)

### Production Deployment

```bash
# CPU version
docker compose up -d

# CUDA/GPU version (Linux only)
docker compose --profile cuda up -d

# View logs
docker compose logs -f
```

The server will be available at `http://localhost:8080`

### Development with Docker

```bash
# Start development environment
./scripts/dev.sh up

# Open shell in container
./scripts/dev.sh shell

# Inside the container, run:
cargo watch -x run          # Backend with hot reload
cd ui && npm run dev --host # Frontend dev server
```

## Quick Start (Native)

### 1. Build the Rust Server

```bash
# Build in release mode
cargo build --release
```

### 2. Build the Web UI

```bash
cd ui
npm install
npm run build
cd ..
```

### 3. Run the Server

```bash
# Run the server
./target/release/izwi-server
```

The server will start at `http://localhost:8080`

### 4. Open the UI

Navigate to `http://localhost:8080` in your browser.

## Development (Native)

### Run in Development Mode

**Terminal 1 - Rust Server:**
```bash
cargo run -p izwi-server
```

**Terminal 2 - UI Dev Server:**
```bash
cd ui
npm run dev
```

The UI will be available at `http://localhost:5173` with hot reload.

## API Reference

### List Models

```bash
GET /v1/models
```

### Download Model

```bash
POST /v1/admin/models/{variant}/download
```

### Load Model

```bash
POST /v1/admin/models/{variant}/load
```

### Generate Speech

```bash
POST /v1/audio/speech
Content-Type: application/json

{
  "model": "qwen3-tts-0.6b-base",
  "input": "Hello, world!",
  "voice": "default",
  "temperature": 0.7,
  "speed": 1.0,
  "response_format": "wav"
}
```

### Transcribe Audio

```bash
POST /v1/audio/transcriptions
Content-Type: application/json

{
  "audio_base64": "<base64-encoded-audio>",
  "model": "qwen3-asr-0.6b",
  "language": "auto"
}
```

## License

Apache 2.0

## Acknowledgments

- [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) by Alibaba
- [MLX](https://github.com/ml-explore/mlx) by Apple
- [HuggingFace Hub](https://huggingface.co/) for model hosting
