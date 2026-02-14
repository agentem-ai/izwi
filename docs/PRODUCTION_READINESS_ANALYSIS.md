# Izwi Engine vs SGLang Docs: Production Readiness Analysis

Last updated: 2026-02-14

## Scope and Method
This document is a code-accurate reassessment of izwi's engine against current SGLang documentation.

- Izwi evidence comes from current source code under `crates/izwi-core` and `crates/izwi-server`.
- SGLang baseline comes from official docs pages (links in References).
- Goal: capture the real current state, not intended architecture.

## Executive Summary
Izwi now has a unified runtime path and a single core engine loop with real incremental streaming in key paths, but it is still a single-node, single-process engine tuned for local use rather than production-scale serving.

What is strong today:
- One runtime source of truth via `RuntimeService` and core `engine::Engine`.
- Centralized engine step driver in runtime.
- Continuous batching scheduler with chunked prefill, adaptive token budget, and preemption logic.
- Incremental core-loop streaming for:
  - Chat (Qwen3)
  - ASR (Qwen3)
  - Speech-to-speech (LFM2)

What is still behind SGLang production baseline:
- No multi-node/multi-process serving architecture.
- No PD disaggregation, HiCache tiering, or quantized KV cache.
- No production metrics/tracing endpoints equivalent to SGLang docs.
- TTS still lacks true per-step decode state in the scheduler/executor loop.

## Verified Izwi Engine State (Code)

### 1) Unified runtime and engine flow
- Server routes go through `RuntimeService` (`crates/izwi-server/src/state.rs`, `crates/izwi-server/src/api/openai/**`).
- `RuntimeService` owns `core_engine: Arc<Engine>` and drives requests via `run_request` / `run_streaming_request`.
- Central background step driver calls `engine.step()` and resolves completion waiters:
  - `crates/izwi-core/src/runtime/service.rs:223`
  - `crates/izwi-core/src/runtime/service.rs:305`
  - `crates/izwi-core/src/runtime/service.rs:326`

### 2) Core loop and scheduler
- `EngineCore::step()` performs schedule -> execute -> process with decode/prefill split:
  - `crates/izwi-core/src/engine/core.rs:187`
- Scheduler supports adaptive batching, chunked prefill, preemption, and prefix-hash based reuse:
  - `crates/izwi-core/src/engine/scheduler.rs:353`
  - `crates/izwi-core/src/engine/scheduler.rs:523`
  - `crates/izwi-core/src/engine/scheduler.rs:848`

### 3) KV cache and paged attention
- Engine-side KV manager exists with block allocation, residency, COW split, and soft-limit tuning:
  - `crates/izwi-core/src/engine/kv_cache.rs:121`
  - `crates/izwi-core/src/engine/kv_cache.rs:474`
  - `crates/izwi-core/src/engine/kv_cache.rs:579`
- Model-side paged attention helpers exist (`append_to_pages`, `paged_decode_attention`):
  - `crates/izwi-core/src/models/shared/attention/paged.rs:22`
  - `crates/izwi-core/src/models/shared/attention/paged.rs:77`

### 4) True incremental streaming currently implemented
- Chat incremental decode state (Qwen3):
  - `crates/izwi-core/src/models/architectures/qwen3/chat.rs:195`
  - `crates/izwi-core/src/engine/executor.rs:777`
- ASR incremental decode state (Qwen3):
  - `crates/izwi-core/src/models/architectures/qwen3/asr/mod.rs:259`
  - `crates/izwi-core/src/engine/executor.rs:529`
- S2S incremental decode state (LFM2):
  - `crates/izwi-core/src/models/architectures/lfm2/audio/mod.rs:392`
  - `crates/izwi-core/src/engine/executor.rs:1020`

### 5) Important partials / limitations in current behavior
- TTS paths are still monolithic executor calls (streamed chunks may be emitted during call, but not stepped statefully by scheduler fairness):
  - `crates/izwi-core/src/engine/executor.rs:907`
- ASR incremental state is Qwen3-only; other ASR backends use callback-based monolithic calls in executor pass:
  - `crates/izwi-core/src/models/registry.rs:74`
  - `crates/izwi-core/src/engine/executor.rs:649`
- Preemption currently requeues request scheduling state; resume semantics are not equivalent to production-grade disaggregated resumable decoding:
  - `crates/izwi-core/src/engine/scheduler.rs:885`

## Capability Matrix: Izwi vs SGLang Docs

| Capability | Izwi current state | SGLang docs baseline | Gap |
|---|---|---|---|
| Unified engine entrypoint | Implemented (`RuntimeService` + core `Engine`) | Implemented | Low |
| Continuous batching scheduler | Implemented (adaptive token budget, chunked prefill, preemption) | Implemented | Medium (resume semantics) |
| Incremental streaming from core loop | Partial (Chat Qwen3, ASR Qwen3, S2S LFM2) | Broad production support | Medium |
| TTS step-state incremental decode | Not fully implemented | Production-grade streaming architectures | High |
| Prefix cache | Basic prefix-hash + shared block reuse | Radix/advanced cache strategies | High |
| KV cache quantization | Not implemented | Documented quantized KV cache support | High |
| PD disaggregation | Not implemented | Documented feature | High |
| HiCache / tiered cache | Not implemented | Documented feature | High |
| Multi-node / large-scale parallel serving | Not implemented | Documented serving at scale (DP/EP/Multi-node) | High |
| Attention backend pluggability | Limited (Candle + local paged helpers) | Multiple attention backend controls in server args/docs | Medium |
| Production metrics endpointing | No dedicated production metrics endpointing stack | Documented production metrics guidance | High |
| Production request tracing | Basic HTTP trace layer only | Documented production request tracing guidance | High |
| Fault isolation (process-level) | Single-process runtime | Production systems commonly isolate workers/processes | High |

## Industry-Standard Gaps Remaining (Prioritized)

### P0 (must close for production confidence)
1. True scheduler-integrated incremental TTS decode state.
2. Strong preemption/resume semantics that preserve decode progress consistently across scheduler and executor state.
3. Production observability surface: exportable metrics, latency phase breakdown (queue/prefill/decode), request-level tracing and correlation IDs.
4. Worker/process isolation strategy so one model/runtime failure does not take down the whole serving stack.

### P1 (major performance and scale enablers)
5. KV cache architecture convergence: tighter coupling between engine KV manager and actual model-layer KV memory lifecycle.
6. Advanced prefix cache strategy (radix/tree match, better reuse granularity than fixed short prefix hash).
7. Overlap scheduling improvements for CPU/GPU work overlap and lower tail latency under concurrency.
8. Quantized KV cache support for memory-bound workloads.

### P2 (distributed production platform)
9. PD disaggregation support.
10. HiCache/tiered cache support.
11. Multi-node parallel serving (data/model/expert parallel coordination where relevant).

## Practical Roadmap (Updated)

### Milestone 1: Engine Correctness + Streaming Completeness
- Implement decode-stateful TTS execution in executor core loop.
- Add preemption-safe request state contracts (scheduler + executor coherence tests).
- Add regression tests for interruption/preemption/requeue across chat/asr/tts/s2s.

### Milestone 2: Production Telemetry
- Add metrics endpoint and structured engine telemetry export.
- Add request correlation IDs and tracing spans across API -> runtime -> engine step -> model exec.
- Publish TTFT/TPOT/queue-wait/KV-hit metrics.

### Milestone 3: Cache and Scale Features
- Upgrade prefix cache strategy.
- Add quantized KV cache path.
- Plan PD disaggregation and cache tiering architecture (HiCache-equivalent strategy for izwi workloads).

## Performance Outlook After Remaining Work
Expected improvements if P0+P1 are completed (estimate ranges, benchmark required):
- Streaming TTS first-chunk latency: 20-50% better under load.
- Mixed workload tail latency (p95/p99): 1.5-3x better from stronger resume + overlap behavior.
- Concurrent capacity before memory pressure: materially higher with better cache reuse + KV quantization.

## References

### SGLang Docs
- https://docs.sglang.ai/
- https://docs.sglang.ai/advanced_features/pd_disaggregation.html
- https://docs.sglang.ai/advanced_features/hicache_best_practices.html
- https://docs.sglang.ai/advanced_features/quantized_kv_cache.html
- https://docs.sglang.ai/references/launch_server.html
- https://docs.sglang.ai/references/production_metrics.html
- https://docs.sglang.ai/references/production_request_tracing.html

### Izwi Code Evidence
- `crates/izwi-core/src/runtime/service.rs`
- `crates/izwi-core/src/engine/core.rs`
- `crates/izwi-core/src/engine/executor.rs`
- `crates/izwi-core/src/engine/scheduler.rs`
- `crates/izwi-core/src/engine/kv_cache.rs`
- `crates/izwi-core/src/models/shared/attention/paged.rs`
- `crates/izwi-core/src/models/architectures/qwen3/chat.rs`
- `crates/izwi-core/src/models/architectures/qwen3/asr/mod.rs`
- `crates/izwi-core/src/models/architectures/lfm2/audio/mod.rs`
