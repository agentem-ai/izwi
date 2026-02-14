# Izwi Audio Inference Architecture

**Version:** 0.3.0  
**Last Updated:** February 2026

## Overview
This repository now uses a runtime-centric structure that mirrors patterns used by modern inference engines (vLLM, TGI, llama.cpp):

- A clear **runtime orchestration layer** (`runtime/`) for request lifecycle.
- A dedicated **model catalog layer** (`catalog/`) for variant metadata/capabilities/parsing.
- A **backend routing layer** (`backends/`) that decides execution backend per model.
- A **model family namespace** (`families/`) as a stable API surface over native implementations.
- A split native model layout under `models/`: **shared infra** (`shared/`) and **family implementations** (`architectures/`).
- A **codec namespace** (`codecs/`) for audio encode/decode surfaces.
- A strict canonical module surface with deprecated compatibility shims removed.

This keeps current behavior intact while giving a clean foundation for native MLX execution, new model families, and additional codecs.

## Design Goals
1. Keep runtime functionality while aggressively removing deprecated structure.
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
├── model/                     # existing model manager/download implementation
├── models/
│   ├── mod.rs                 # model namespace + compatibility re-exports
│   ├── registry.rs            # native model instance registry/load-once cache
│   ├── shared/                # shared infra (device/weights/attention/memory/chat types)
│   └── architectures/         # concrete model families (qwen3, gemma3, lfm2, etc.)
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

## API Surface

### OpenAI-Compatible Runtime Endpoints
- `POST /v1/responses`
- `GET /v1/responses/:response_id`
- `DELETE /v1/responses/:response_id`
- `POST /v1/responses/:response_id/cancel`
- `GET /v1/responses/:response_id/input_items`
- `POST /v1/audio/speech`
- `POST /v1/audio/transcriptions`
- `POST /v1/chat/completions`
- `GET /v1/models`
- `GET /v1/models/:model`

### Admin Endpoints (UI/Local Ops)
- `GET /v1/admin/models`
- `GET /v1/admin/models/:variant`
- `POST /v1/admin/models/:variant/download`
- `GET /v1/admin/models/:variant/download/progress`
- `POST /v1/admin/models/:variant/download/cancel`
- `POST /v1/admin/models/:variant/load`
- `POST /v1/admin/models/:variant/unload`
- `DELETE /v1/admin/models/:variant`

The previous `/api/v1/tts/*`, `/api/v1/asr/*`, and shimmed compatibility paths were removed.

## Extension Points

### Add Native MLX Execution
1. Implement MLX backend runtime execution path.
2. Extend `backends::BackendRouter` selection to route compatible variants to MLX.
3. Keep fallback to Candle path for unsupported variants.
4. Add runtime-level integration tests for per-model backend dispatch.

### Add New Model Families
1. Add family implementation under `models/architectures/<family>/`.
2. Reuse/add shared primitives under `models/shared/` when cross-family.
3. Export the family via `models/architectures/mod.rs`, then expose it in `models/mod.rs` (including compatibility alias if needed).
4. Wire load/caching in `models/registry.rs`.
5. Add `ModelVariant` entries + capability mapping in `catalog/variant.rs`.
6. Add load/unload routing in `runtime/model_router.rs` and task handlers in `runtime/` (if needed).

### Add Additional Audio Codecs
1. Implement codec components under `audio/`.
2. Re-export in `codecs/` namespace.
3. Add selection/config mapping in runtime API handling.
4. Add roundtrip and streaming tests for new format.

## Why This Aligns With Modern Inference Engines
- **Separated orchestration and execution concerns** (runtime vs backend/family).
- **Centralized model registry/catalog responsibilities**.
- **Pluggable backend routing path** for hardware/runtime specialization.
- **Clean contract strategy** where canonical paths are explicit and deprecated layers are removed.

## Validation Status
- Workspace compiles successfully (`cargo check --workspace`).
- Runtime and tests pass without legacy `inference` compatibility shims.
