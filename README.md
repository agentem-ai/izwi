# Izwi - Qwen3-TTS Inference Engine for Apple Silicon

A high-performance, Rust-based text-to-speech inference engine optimized for Qwen3-TTS models on Apple Silicon (M1+) using MLX.

## Features

- **Apple Silicon Optimized**: Built on MLX for unified memory and Metal GPU acceleration
- **Streaming Audio**: Ultra-low-latency streaming with ~97ms first-packet emission
- **Model Management**: Download and manage Qwen3-TTS models directly from the UI
- **Modern Web UI**: Beautiful React-based interface for testing TTS
- **REST API**: OpenAI-compatible endpoints for easy integration
- **Voice Cloning**: Support for reference audio-based voice cloning (CustomVoice models)

## Supported Models

| Model | Size | Description |
|-------|------|-------------|
| Qwen3-TTS-12Hz-0.6B-Base | ~1.2GB | Fast, lightweight base model |
| Qwen3-TTS-12Hz-0.6B-CustomVoice | ~1.2GB | Voice cloning with 0.6B model |
| Qwen3-TTS-12Hz-1.7B-Base | ~3.4GB | Higher quality base model |
| Qwen3-TTS-12Hz-1.7B-CustomVoice | ~3.4GB | Voice cloning with 1.7B model |
| Qwen3-TTS-12Hz-1.7B-VoiceDesign | ~3.4GB | Voice design with descriptions |

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

### Stream Speech

```bash
POST /api/v1/tts/stream
Content-Type: application/json

{
  "text": "Hello, world!",
  "format": "wav"
}
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Web UI (React)                          │
└─────────────────────────────────────────────────────────────┘
                            │
┌─────────────────────────────────────────────────────────────┐
│                   API Server (Axum)                          │
│  - REST endpoints                                            │
│  - Streaming audio responses                                 │
│  - Model management                                          │
└─────────────────────────────────────────────────────────────┘
                            │
┌─────────────────────────────────────────────────────────────┐
│                 Inference Engine (Rust)                      │
│  - Qwen3-TTS model loading                                   │
│  - Text tokenization                                         │
│  - Audio token generation                                    │
│  - Audio codec decoding                                      │
└─────────────────────────────────────────────────────────────┘
                            │
┌─────────────────────────────────────────────────────────────┐
│                    MLX Backend                               │
│  - Metal GPU acceleration                                    │
│  - Unified memory                                            │
│  - Optimized matrix operations                               │
└─────────────────────────────────────────────────────────────┘
```

## Configuration

Create a `config.toml` file in the project root:

```toml
[engine]
models_dir = "~/.izwi/models"
max_batch_size = 8
max_sequence_length = 4096
chunk_size = 128
use_metal = true

[server]
host = "0.0.0.0"
port = 8080
cors_enabled = true
```

## Performance Targets

- **First chunk latency**: < 100ms
- **Streaming RTF**: < 0.5 (faster than real-time)
- **Memory usage**: 2-6GB depending on model size

## Docker Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Docker Container                          │
│  ┌─────────────────────────────────────────────────────────┐│
│  │                  Izwi Server (Rust)                     ││
│  │  - REST API endpoints                                   ││
│  │  - Model management                                     ││
│  │  - Static file serving (UI)                             ││
│  └──────────────────────────┬──────────────────────────────┘│
│                             │ Unix Socket                    │
│  ┌──────────────────────────▼──────────────────────────────┐│
│  │               Python TTS Daemon                          ││
│  │  - Qwen3-TTS inference                                  ││
│  │  - Model caching                                        ││
│  │  - GPU acceleration (CUDA/MPS)                          ││
│  └─────────────────────────────────────────────────────────┘│
│                                                              │
│  Volume: /app/models (HuggingFace cache)                    │
└─────────────────────────────────────────────────────────────┘
```

## Project Structure

```
izwi-audio/
├── crates/
│   ├── izwi-core/        # Core inference engine
│   └── izwi-server/      # Axum web server
├── scripts/
│   ├── tts_daemon.py     # Python TTS daemon
│   ├── tts_inference.py  # Direct inference script
│   └── dev.sh            # Development helper
├── ui/                   # React frontend
├── pyproject.toml        # Python dependencies (uv)
├── Cargo.toml            # Rust dependencies
├── Dockerfile            # Production multi-stage build
├── Dockerfile.dev        # Development container
├── docker-compose.yml    # Production orchestration
└── docker-compose.dev.yml # Development orchestration
```

## License

Apache 2.0

## Acknowledgments

- [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) by Alibaba
- [MLX](https://github.com/ml-explore/mlx) by Apple
- [HuggingFace Hub](https://huggingface.co/) for model hosting
