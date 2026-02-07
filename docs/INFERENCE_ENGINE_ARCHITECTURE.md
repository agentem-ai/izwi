# Izwi Audio Inference Architecture

**Version:** 0.2.0  
**Last Updated:** February 2026

## Overview
This repository now uses a runtime-centric structure that mirrors patterns used by modern inference engines (vLLM, TGI, llama.cpp):

- A clear **runtime orchestration layer** (`runtime/`) for request lifecycle.
- A dedicated **model catalog layer** (`catalog/`) for variant metadata/capabilities/parsing.
- A **backend routing layer** (`backends/`) that decides execution backend per model.
- A **model family namespace** (`families/`) for native architecture implementations.
- A **codec namespace** (`codecs/`) for audio encode/decode surfaces.
- Backward-compatible facades for existing imports (`inference/`, `model/`, `models/`).

This keeps current behavior intact while giving a clean foundation for native MLX execution, new model families, and additional codecs.

## Design Goals
1. Preserve all current API behavior and runtime outputs.
2. Reduce coupling between routing/orchestration and model-specific implementation details.
3. Centralize model capability metadata and parsing logic.
4. Make backend expansion (especially MLX) an additive change.
5. Keep legacy public module paths functional.

## Module Layout

```text
crates/izwi-core/src/
├── runtime/                   # Canonical request lifecycle orchestration
│   ├── mod.rs
│   ├── service.rs             # InferenceEngine struct + base lifecycle methods
│   ├── model_router.rs        # load/unload + backend selection integration
│   ├── tts.rs                 # TTS generation + streaming methods
│   ├── asr.rs                 # ASR / Voxtral / forced alignment methods
│   ├── chat.rs                # Chat generation methods
│   ├── audio_io.rs            # shared base64/WAV/decode/preprocessing helpers
│   └── types.rs               # GenerationRequest/Result/AudioChunk/etc.
├── catalog/                   # Model metadata, capabilities, parsing
│   ├── mod.rs
│   └── variant.rs
├── backends/
│   └── mod.rs                 # ExecutionBackend + BackendRouter
├── families/
│   └── mod.rs                 # family namespace wrappers over native models
├── codecs/
│   └── mod.rs                 # codec namespace wrappers over audio module
├── inference/                 # compatibility facade to runtime
├── model/                     # existing model manager/download implementation
├── models/                    # existing native family implementations
└── engine/                    # existing vLLM-style experimental engine core
```

## Runtime Flow

### 1) API Layer (izwi-server)
- Parses requests.
- Uses shared `InferenceEngine` runtime.
- Uses centralized catalog parsers for model IDs.

### 2) Runtime Layer (`runtime/`)
- `service.rs`: owns engine state (device profile, model manager, registry, codec, backend router).
- `model_router.rs`: resolves loading path by model variant and selected backend.
- Task handlers:
  - `tts.rs`: synchronous + streaming TTS.
  - `asr.rs`: ASR, Voxtral transcription, forced alignment.
  - `chat.rs`: chat completions + streaming deltas.
- `audio_io.rs`: shared decode/preprocessing utilities used by multiple tasks.

### 3) Backend Routing (`backends/`)
- `BackendRouter` computes a `BackendPlan` from model capabilities.
- Current active backend remains native Candle path.
- MLX selection is scaffolded behind configuration (`IZWI_ENABLE_MLX_RUNTIME`), with explicit non-implemented guard.

### 4) Catalog (`catalog/`)
- Canonical model identifier parsing (`parse_model_variant`, `parse_tts_model_variant`, `parse_chat_model_variant`, `resolve_asr_model_variant`).
- Model capability classification:
  - family
  - primary task
  - backend hint

This avoids duplicated parsing logic in each API endpoint.

## Compatibility Guarantees
The refactor is non-breaking by design:

- `crate::inference::*` remains available via compatibility re-exports.
- Existing `InferenceEngine` type name is preserved.
- Existing `model` and `models` modules remain intact.
- API endpoint paths and request/response formats remain unchanged.

## Extension Points

### Add Native MLX Execution
1. Implement MLX backend runtime execution path.
2. Extend `backends::BackendRouter` selection to route compatible variants to MLX.
3. Keep fallback to Candle path for unsupported variants.
4. Add runtime-level integration tests for per-model backend dispatch.

### Add New Model Families
1. Add family implementation under `models/`.
2. Export under `families/` namespace.
3. Add `ModelVariant` entries + capability mapping in `catalog/variant.rs`.
4. Add load/unload route in `runtime/model_router.rs`.
5. Add task handlers (if new task type) under `runtime/`.

### Add Additional Audio Codecs
1. Implement codec components under `audio/`.
2. Re-export in `codecs/` namespace.
3. Add selection/config mapping in runtime API handling.
4. Add roundtrip and streaming tests for new format.

## Why This Aligns With Modern Inference Engines
- **Separated orchestration and execution concerns** (runtime vs backend/family).
- **Centralized model registry/catalog responsibilities**.
- **Pluggable backend routing path** for hardware/runtime specialization.
- **Compatibility layer strategy** to avoid migration breaks while evolving internals.

## Validation Status
- Workspace compiles successfully (`cargo check --workspace`).
- Existing behavior paths are preserved through compatibility facades.
