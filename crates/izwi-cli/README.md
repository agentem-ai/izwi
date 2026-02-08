# Izwi CLI Documentation

## Overview

The Izwi CLI is a world-class command-line interface for the Izwi audio inference engine. Inspired by vLLM, SGlang, and Ollama, it provides intuitive commands for managing models, generating speech, transcribing audio, and running benchmarks.

## Installation

### Quick Install (macOS/Linux)

```bash
curl -sSL https://raw.githubusercontent.com/agentem/izwi-audio/main/scripts/install-cli.sh | bash
```

### Build from Source

```bash
git clone https://github.com/agentem/izwi-audio
cd izwi-audio
cargo build --release --bin izwi
```

### Manual Installation

1. Download the latest release for your platform
2. Extract the binary to a directory in your PATH
3. Run `izwi --help` to verify installation

**Note:** If using the install script directly, make it executable first:
```bash
chmod +x ./scripts/install-cli.sh
./scripts/install-cli.sh
```

## Quick Start

```bash
# Start the server
izwi serve

# Download a model
izwi pull qwen3-tts-0.6b-base

# Generate speech
izwi tts "Hello, world!"

# Transcribe audio
izwi transcribe audio.wav
```

## Commands

### Server Management

```bash
# Start server with default settings
izwi serve

# Start with custom configuration
izwi serve --host 0.0.0.0 --port 8080 --metal

# Start in development mode
izwi serve --dev
```

### Model Management

```bash
# List available models
izwi list
izwi list --local
izwi list --detailed

# Download a model
izwi pull qwen3-tts-0.6b-base
izwi pull qwen3-tts-1.7b-customvoice --force

# Show model information
izwi models info qwen3-tts-0.6b-base

# Load/unload models
izwi models load qwen3-tts-0.6b-base
izwi models unload qwen3-tts-0.6b-base

# Remove a model
izwi rm qwen3-tts-0.6b-base

# View download progress
izwi models progress
```

### Text-to-Speech

```bash
# Basic usage
izwi tts "Hello, world!"

# With options
izwi tts "Hello, world!" \
  --model qwen3-tts-1.7b-base \
  --speaker female \
  --output hello.wav \
  --speed 1.2 \
  --temperature 0.8

# Read from stdin
echo "Hello, world!" | izwi tts -

# Stream output
izwi tts "Long text here..." --stream

# Play immediately (if supported)
izwi tts "Hello!" --play
```

### Speech-to-Text

```bash
# Basic transcription
izwi transcribe audio.wav

# With options
izwi transcribe audio.wav \
  --model qwen3-asr-1.7b \
  --language en \
  --format json \
  --output transcript.json

# With word-level timestamps
izwi transcribe audio.wav --word-timestamps
```

### Interactive Chat

```bash
# Start chat mode
izwi chat --model qwen3-tts-1.7b-base

# With system prompt
izwi chat --system "You are a helpful assistant"
```

### Benchmarking

```bash
# Benchmark TTS
izwi bench tts --model qwen3-tts-0.6b-base --iterations 10

# Benchmark ASR
izwi bench asr --file audio.wav --iterations 10

# Benchmark throughput
izwi bench throughput --duration 30 --concurrent 4
```

### System Status

```bash
# Show status
izwi status

# Detailed status
izwi status --detailed

# Watch mode
izwi status --watch 5
```

### Configuration

```bash
# Show configuration
izwi config show

# Set configuration values
izwi config set server.host "0.0.0.0"
izwi config set server.port 8080

# Get configuration value
izwi config get server.host

# Edit configuration in editor
izwi config edit

# Show config file path
izwi config path

# Reset configuration
izwi config reset
```

### Shell Completions

```bash
# Bash
izwi completions bash > ~/.izwi-completion.bash
echo "source ~/.izwi-completion.bash" >> ~/.bashrc

# Zsh
izwi completions zsh > ~/.zsh/completions/_izwi

# Fish
izwi completions fish > ~/.config/fish/completions/izwi.fish
```

## Global Options

```
-c, --config <PATH>     Configuration file path
-s, --server <URL>      Server URL (default: http://localhost:8080)
-f, --format <FORMAT>   Output format: table, json, plain, yaml
-q, --quiet            Suppress all output except results
-v, --verbose          Enable verbose output
    --no-color         Disable colored output
```

## Environment Variables

```bash
# Server configuration
IZWI_HOST=0.0.0.0
IZWI_PORT=8080
IZWI_MODELS_DIR=/path/to/models
IZWI_USE_METAL=1
IZWI_NUM_THREADS=8
IZWI_MAX_BATCH_SIZE=8
IZWI_MAX_CONCURRENT=100
IZWI_TIMEOUT=300

# Logging
RUST_LOG=info

# Disable colors
NO_COLOR=1
```

## Configuration File

The configuration file is located at `~/.config/izwi/config.toml`:

```toml
# Izwi Configuration

[server]
host = "localhost"
port = 8080

[models]
dir = "~/.local/share/izwi/models"

[defaults]
model = "qwen3-tts-0.6b-base"
speaker = "default"
format = "wav"
```

## Model Variants

### Text-to-Speech
- `qwen3-tts-0.6b-base` - Fast, lightweight TTS with 9 built-in voices
- `qwen3-tts-0.6b-customvoice` - Voice cloning with reference audio
- `qwen3-tts-1.7b-base` - Higher quality TTS
- `qwen3-tts-1.7b-customvoice` - Higher quality voice cloning
- `qwen3-tts-1.7b-voicedesign` - Design voices from text descriptions

### Speech-to-Text
- `qwen3-asr-0.6b` - Fast, lightweight ASR
- `qwen3-asr-1.7b` - Higher accuracy ASR

## Tips

1. **Use quiet mode for scripts**: `izwi -q tts "Hello" -o out.wav`
2. **JSON output for automation**: `izwi -f json list`
3. **Stream for long texts**: `izwi tts "Long text..." --stream`
4. **Benchmark before production**: `izwi bench tts -i 50`

## Troubleshooting

### Connection refused
- Ensure the server is running: `izwi serve`
- Check server URL: `izwi -s http://localhost:8080 list`

### Model not found
- Download the model: `izwi pull <model>`
- List available models: `izwi list`

### Out of memory
- Use smaller models (0.6B instead of 1.7B)
- Reduce batch size: `izwi serve --max-batch-size 4`
- Enable Metal on macOS: `izwi serve --metal`

## License

Apache 2.0 - See LICENSE file for details.
