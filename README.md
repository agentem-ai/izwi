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
- **Python 3.11+** with uv package manager
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

### 1. Install Python Dependencies (uv)

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv venv
uv pip install -e .
```

### 2. Build the Rust Server

```bash
# Build in release mode
cargo build --release
```

### 3. Build the Web UI

```bash
cd ui
npm install
npm run build
cd ..
```

### 4. Run the Server

```bash
# Activate Python environment first
source .venv/bin/activate

# Run the server
./target/release/izwi
```

The server will start at `http://localhost:8080`

### 5. Open the UI

Navigate to `http://localhost:8080` in your browser.

## Development (Native)

### Run in Development Mode

**Terminal 1 - Rust Server:**
```bash
source .venv/bin/activate
cargo run
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
GET /api/v1/models
```

### Download Model

```bash
POST /api/v1/models/{variant}/download
```

### Load Model

```bash
POST /api/v1/models/{variant}/load
```

### Generate Speech

```bash
POST /api/v1/tts/generate
Content-Type: application/json

{
  "text": "Hello, world!",
  "speaker": "default",
  "temperature": 0.7,
  "speed": 1.0,
  "format": "wav"
}
```

### Transcribe Audio

```bash
POST /api/v1/asr/transcribe
Content-Type: application/json

{
  "audio_base64": "<base64-encoded-audio>",
  "model_id": "Qwen/Qwen3-ASR-0.6B",
  "language": "auto"
}
```

## License

Apache 2.0

## Acknowledgments

- [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) by Alibaba
- [MLX](https://github.com/ml-explore/mlx) by Apple
- [HuggingFace Hub](https://huggingface.co/) for model hosting
