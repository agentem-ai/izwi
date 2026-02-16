# Izwi Inference Engine Deep Dive

**Version:** 0.3.0  
**Last Updated:** February 2026  
**Author:** Architecture Analysis

---

## Executive Summary

The Izwi inference engine implements a **vLLM-inspired architecture** for audio model inference on Apple Silicon and CPU devices. It features:

- **Paged KV-cache memory management** with block-level allocation
- **Continuous batching** with adaptive scheduling
- **Chunked prefill** for long prompts
- **Priority-based preemption** for latency-sensitive workloads
- **Streaming output** with real-time audio generation
- **Multi-model support** (TTS, ASR, Chat, Speech-to-Speech)

This document provides a comprehensive analysis of the architecture, implementation status, and optimization opportunities.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Entry Points & Request Flow](#2-entry-points--request-flow)
3. [Scheduler Design](#3-scheduler-design)
4. [Prefill & Decode Pipeline](#4-prefill--decode-pipeline)
5. [KV Cache Implementation](#5-kv-cache-implementation)
6. [Attention Mechanisms](#6-attention-mechanisms)
7. [Model Executor](#7-model-executor)
8. [Metal/Apple Silicon Optimizations](#8-metalapple-silicon-optimizations)
9. [Unimplemented Features](#9-unimplemented-features)
10. [Optimization Opportunities](#10-optimization-opportunities)
11. [Recommendations](#11-recommendations)

---

## 1. Architecture Overview

### High-Level Component Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              Engine                                      │
│  ┌──────────────────┐  ┌─────────────┐  ┌────────────────────────────┐  │
│  │  RequestProcessor │  │  Scheduler  │  │       EngineCore           │  │
│  │  - Validation     │──│  - FCFS     │──│  ┌──────────────────────┐  │  │
│  │  - Tokenization   │  │  - Priority │  │  │   UnifiedExecutor    │  │  │
│  │  - Preprocessing  │  │  - Adaptive │  │  │   (NativeExecutor)   │  │  │
│  └──────────────────┘  └─────────────┘  │  └──────────────────────┘  │  │
│                                          │  ┌──────────────────────┐  │  │
│  ┌──────────────────┐                    │  │   KVCacheManager     │  │  │
│  │  OutputProcessor │◄───────────────────│  │   (Paged Attention)  │  │  │
│  │  - Streaming     │                    │  └──────────────────────┘  │  │
│  │  - Formatting    │                    └────────────────────────────┘  │
│  └──────────────────┘                                                    │
└─────────────────────────────────────────────────────────────────────────┘
```

### Core Components

| Component | File | Responsibility |
|-----------|------|----------------|
| `Engine` | `engine/mod.rs` | Top-level API, orchestrates all components |
| `EngineCore` | `engine/core.rs` | Central inference loop coordinator |
| `Scheduler` | `engine/scheduler.rs` | Request scheduling with FCFS/Priority policies |
| `KVCacheManager` | `engine/kv_cache.rs` | Paged KV-cache memory management |
| `UnifiedExecutor` | `engine/executor.rs` | Model forward pass execution |
| `RequestProcessor` | `engine/request.rs` | Request validation and preprocessing |
| `OutputProcessor` | `engine/output.rs` | Output formatting and streaming |
| `ModelRegistry` | `models/registry.rs` | Model loading and caching |

---

## 2. Entry Points & Request Flow

### Primary Entry Points

#### 1. `Engine::generate()` - Synchronous Generation
```rust
// engine/mod.rs:122-147
pub async fn generate(&self, request: EngineCoreRequest) -> Result<EngineOutput> {
    let request_id = self.add_request(request).await?;
    loop {
        let outputs = self.step().await?;
        for output in outputs {
            if output.request_id == request_id && output.is_finished {
                return Ok(output);
            }
        }
    }
}
```

#### 2. `Engine::generate_streaming()` - Streaming Generation
```rust
// engine/mod.rs:152-167
pub async fn generate_streaming(
    &self,
    request: EngineCoreRequest,
) -> Result<(RequestId, mpsc::UnboundedReceiver<StreamingOutput>)>
```

#### 3. `Engine::run()` - Continuous Processing Loop
```rust
// engine/mod.rs:194-219
pub async fn run(&self) -> Result<()> {
    while self.running.load(Ordering::SeqCst) {
        if has_work {
            self.step().await?;
        } else {
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
    }
}
```

### Request Types

The engine supports four task types defined in `EngineCoreRequest`:

| Task Type | Constructor | Input | Output |
|-----------|-------------|-------|--------|
| `TTS` | `EngineCoreRequest::tts(text)` | Text string | Audio samples |
| `ASR` | `EngineCoreRequest::asr(audio_b64)` | Base64 audio | Transcription text |
| `Chat` | `EngineCoreRequest::chat(messages)` | Chat messages | Generated text |
| `SpeechToSpeech` | `EngineCoreRequest::speech_to_speech(audio_b64)` | Base64 audio | Text + Audio |

### Request Lifecycle

```
┌─────────────┐    ┌─────────────────┐    ┌───────────┐    ┌──────────┐
│  add_request │───▶│ RequestProcessor │───▶│ Scheduler │───▶│ Waiting  │
└─────────────┘    │   (validate)     │    │ (add)     │    │  Queue   │
                   └─────────────────┘    └───────────┘    └────┬─────┘
                                                                │
┌─────────────┐    ┌─────────────────┐    ┌───────────┐    ┌────▼─────┐
│   Output    │◄───│  OutputProcessor │◄───│  Executor │◄───│ Running  │
│  (stream)   │    │   (format)       │    │ (forward) │    │  Queue   │
└─────────────┘    └─────────────────┘    └───────────┘    └──────────┘
```

---

## 3. Scheduler Design

### Scheduling Policies

The scheduler (`engine/scheduler.rs`) supports two primary policies:

#### 1. FCFS (First-Come, First-Served)
```rust
SchedulingPolicy::FCFS  // Default
```
- Simple queue-based ordering
- Fair but may cause head-of-line blocking

#### 2. Priority-Based
```rust
SchedulingPolicy::Priority
```
- Requests ordered by `Priority` enum (Low, Normal, High, Critical)
- Supports priority aging to prevent starvation

### Adaptive Batching

When `enable_adaptive_batching` is true, the scheduler dynamically adjusts:

1. **Token Budget**: Adjusts `dynamic_tokens_per_step` based on latency feedback
2. **Prefill/Decode Split**: Reserves budget for prefill when TTFT is high
3. **Priority Aging**: Boosts old requests to prevent starvation

```rust
// scheduler.rs:922-934
fn adaptive_waiting_score(&self, request_id: &RequestId) -> f64 {
    let base_priority = metadata.priority as i32 as f64;
    let age_ms = metadata.arrival_time.elapsed().as_millis() as f64;
    let age_boost = age_ms / self.config.priority_aging_ms.max(1) as f64;
    let prompt_bonus = 1.0 / (1.0 + (metadata.total_prompt_tokens as f64 / threshold));
    base_priority + age_boost + (prompt_bonus * 0.2)
}
```

### Preemption Support

The scheduler implements **priority-based preemption** for memory pressure scenarios:

```rust
// scheduler.rs:1019-1090
fn try_preempt_for_blocks(
    &mut self,
    blocks_needed: usize,
    requesting_priority: Priority,
    kv_cache: &mut KVCacheManager,
) -> Vec<RequestId>
```

**Key behaviors:**
- Only preempts requests with **lower priority** than the requester
- Preempted requests are **paused, not aborted** (preserves decode state)
- Requests can resume when resources become available

### VAD-Triggered Preemption

The scheduler supports Voice Activity Detection (VAD) preemption for real-time audio:

```rust
pub struct VadPreemptionEvent {
    pub timestamp: Instant,
    pub speech_probability: f32,
    pub requests_to_preempt: Vec<RequestId>,
}
```

This enables **barge-in** functionality where user speech interrupts AI output.

---

## 4. Prefill & Decode Pipeline

### Two-Phase Execution Model

The engine separates inference into two distinct phases:

#### Phase 1: Prefill
- Processes the **entire prompt** in parallel
- Populates the KV cache with prompt context
- High compute intensity, memory bandwidth bound

#### Phase 2: Decode
- Generates tokens **one at a time** (autoregressive)
- Each step uses cached KV from previous tokens
- Lower compute, latency sensitive

### Chunked Prefill

For long prompts, the engine supports **chunked prefill** to avoid blocking:

```rust
// scheduler.rs:968-994
fn effective_prefill_chunk_threshold(
    &self,
    kv_utilization: f64,
    has_decode_demand: bool,
) -> usize {
    let base = self.config.chunked_prefill_threshold.max(32);
    let mut threshold = base;

    // Under memory pressure, shrink prefill chunks
    if kv_utilization > 0.95 {
        threshold = threshold.min((base / 4).max(32));
    } else if kv_utilization > 0.85 {
        threshold = threshold.min((base / 2).max(64));
    }

    // If decode is active, avoid over-investing in prefill
    if has_decode_demand {
        threshold = threshold.min((base / 2).max(64));
    }
    
    threshold.max(32)
}
```

**Benefits:**
- Prevents long prompts from starving decode requests
- Maintains low latency for concurrent requests
- Adapts to memory pressure

### Execution Flow in EngineCore::step()

```rust
// core.rs:233-437
pub async fn step(&mut self) -> Result<Vec<EngineOutput>> {
    // Phase 1: Schedule
    let schedule_result = self.scheduler.schedule(self.kv_cache.inner_mut());
    
    // Phase 2: Execute (decode first, then prefill)
    let (decode_outputs, decode_elapsed) = run_decode.await?;
    let (prefill_outputs, prefill_elapsed) = run_prefill.await?;
    
    // Phase 3: Process outputs
    for exec_output in executor_outputs {
        // Update scheduler state
        // Process streaming output
        // Check completion conditions
    }
}
```

### Metal Sequential Execution

On Metal devices, prefill and decode run **sequentially** to reduce contention:

```rust
// core.rs:314-337
if self.config.use_metal && !decode_request_refs.is_empty() && !prefill_request_refs.is_empty() {
    // Sequential execution for Metal
    let (decode_outputs, decode_elapsed) = run_decode.await?;
    let (prefill_outputs, prefill_elapsed) = run_prefill.await?;
} else {
    // Parallel execution for CPU
    let (decode_result, prefill_result) = tokio::join!(run_decode, run_prefill);
}
```

---

## 5. KV Cache Implementation

### Paged Attention Design

The KV cache (`engine/kv_cache.rs`) implements **vLLM-style paged attention**:

```rust
pub struct KVCacheConfig {
    pub num_layers: usize,      // 24 default
    pub num_heads: usize,       // 16 default
    pub head_dim: usize,        // 64 default
    pub block_size: usize,      // 16 tokens per block
    pub max_blocks: usize,      // 1024 default
    pub dtype_bytes: usize,     // 2 for F16, 4 for F32
}
```

### Block Allocator

```rust
pub struct BlockAllocator {
    blocks: Vec<KVBlock>,
    free_list: VecDeque<BlockId>,  // LIFO for cache locality
    num_allocated: usize,
    soft_max_blocks: usize,        // Adaptive limit
}
```

**Key features:**
- **LIFO allocation**: Recently freed blocks reused first for cache locality
- **Soft limit**: Adaptive cap that adjusts based on churn
- **Reference counting**: Enables copy-on-write for shared prefixes

### Shared Prefix Caching

The KV cache supports **prefix caching** for common prompts:

```rust
// kv_cache.rs:350-425
pub fn allocate_with_prefix(
    &mut self,
    request_id: &RequestId,
    num_blocks: usize,
    prefix_hash: Option<u64>,
) -> Vec<BlockId>
```

**Advanced prefix matching** at block granularity:

```rust
// kv_cache.rs:436-513
pub fn allocate_with_prefix_tokens(
    &mut self,
    request_id: &RequestId,
    num_blocks: usize,
    prompt_tokens: &[u32],
) -> Vec<BlockId>
```

This enables partial prefix reuse when prompts share common beginnings.

### Copy-on-Write (CoW)

When a shared block needs modification:

```rust
// kv_cache.rs:696-745
pub fn ensure_writable_block(
    &mut self,
    request_id: &RequestId,
    logical_block_idx: usize,
) -> Option<BlockId>
```

The block is copied before modification, preserving the original for other requests.

### Streaming KV Cache

For continuous audio prefill, a specialized streaming cache exists:

```rust
pub struct StreamingKVCacheManager {
    config: StreamingKVCacheConfig,
    allocator: BlockAllocator,
    sequences: HashMap<String, StreamingSequence>,
    // Sliding window with automatic eviction
}
```

**Features:**
- Sliding window context management
- Automatic eviction of old tokens
- Designed for real-time audio streaming

### KV Cache Telemetry

```rust
pub struct KVCacheTelemetry {
    pub total_allocations: u64,
    pub total_frees: u64,
    pub shared_prefix_hits: u64,
    pub copy_on_write_splits: u64,
    pub last_churn_ratio: f64,
    pub soft_max_blocks: usize,
}
```

---

## 6. Attention Mechanisms

### Paged Decode Attention

For single-token decode with paged KV cache:

```rust
// models/shared/attention/paged.rs:215-310
pub fn paged_decode_attention(
    q: &Tensor,           // [batch, 1, heads, head_dim]
    k_pages: &[KvPage],   // Paged K tensors
    v_pages: &[KvPage],   // Paged V tensors
    num_heads: usize,
    head_dim: usize,
) -> Result<Tensor>
```

**Implementation details:**
- Uses **online softmax** (FlashAttention-style) to avoid materializing full attention matrix
- Processes pages incrementally with running max/sum accumulators
- Supports both dense and quantized (Int8) pages

### KV Cache Quantization

```rust
pub enum KvCacheQuantization {
    None,   // Native dtype
    Int8,   // Per-page symmetric quantization
}
```

Quantization reduces memory by ~4x with acceptable accuracy loss:

```rust
// paged.rs:108-132
fn quantize_tensor_int8(tensor: &Tensor) -> Result<(Tensor, f32, DType)> {
    let max_abs = tensor.abs()?.max_all()?;
    let scale = (max_abs / 127.0).max(1e-8);
    // Quantize to U8 with offset 128
}
```

### Batched Attention

For multi-sequence processing:

```rust
// models/shared/attention/batched.rs:65-129
pub fn batched_scaled_dot_product_attention(
    input: &BatchedAttentionInput,
    config: &BatchedAttentionConfig,
) -> Result<Tensor>
```

**Features:**
- Supports variable-length sequences with padding
- Optional Flash Attention via `candle_nn::ops::sdpa`
- Automatic padding mask generation

### Flash Attention Integration

```rust
// batched.rs:92-102
if config.use_flash_attention {
    let scale = 1.0f32 / config.scale as f32;
    if let Ok(sdpa_out) = candle_nn::ops::sdpa(&q, &k, &v, mask, false, scale, 1.0) {
        return Ok(output);
    }
}
```

Controlled via environment variable `IZWI_USE_FLASH_ATTENTION`.

---

## 7. Model Executor

### Executor Architecture

```rust
pub trait ModelExecutor: Send + Sync {
    fn execute_prefill(&self, requests: &[&EngineCoreRequest], scheduled: &[ScheduledRequest]) -> Result<Vec<ExecutorOutput>>;
    fn execute_decode(&self, requests: &[&EngineCoreRequest], scheduled: &[ScheduledRequest]) -> Result<Vec<ExecutorOutput>>;
    fn initialize(&mut self) -> Result<()>;
    fn shutdown(&mut self) -> Result<()>;
    fn cleanup_request(&self, request_id: &str);
}
```

### NativeExecutor

The primary executor implementation handles all task types:

```rust
pub struct NativeExecutor {
    config: WorkerConfig,
    loaded_tts_model: Option<Arc<Qwen3TtsModel>>,
    chat_decode_states: Mutex<HashMap<String, ActiveChatDecode>>,
    asr_decode_states: Mutex<HashMap<String, ActiveAsrDecode>>,
    qwen_tts_decode_states: Mutex<HashMap<String, ActiveQwenTtsDecode>>,
    lfm2_tts_decode_states: Mutex<HashMap<String, ActiveLfm2TtsDecode>>,
    speech_to_speech_decode_states: Mutex<HashMap<String, ActiveSpeechToSpeechDecode>>,
}
```

### Decode State Management

Each task type maintains incremental decode state:

```rust
struct ActiveQwenTtsDecode {
    variant: Option<ModelVariant>,
    state: QwenTtsDecodeState,
    prompt_accounted: bool,
    last_frames_generated: usize,
    stream_sequence: usize,
    audio_samples_accum: Vec<f32>,
}
```

This enables:
- **Preemption/resume**: State preserved when request is paused
- **Streaming**: Incremental output without recomputation
- **Progress tracking**: Accurate token counts for scheduling

### Parallel Execution

For CPU execution with multiple requests:

```rust
// executor.rs:1365-1417
fn execute_requests_parallel(
    &self,
    requests: &[&EngineCoreRequest],
    scheduled: &[ScheduledRequest],
) -> Result<Vec<ExecutorOutput>> {
    thread::scope(|scope| {
        for chunk in partitions {
            scope.spawn(move || {
                // Execute requests in parallel threads
            });
        }
    });
}
```

**Note:** Metal execution is kept **serial** to avoid command queue contention.

---

## 8. Metal/Apple Silicon Optimizations

### Metal KV Cache Manager

```rust
// engine/metal_kv_cache.rs
pub struct MetalKVCacheManager {
    pub inner: KVCacheManager,
    pub config: MetalKVCacheConfig,
    device_profile: DeviceProfile,
    memory_pressure: MemoryPressure,
}
```

### Memory Pressure Handling

```rust
pub enum MemoryPressure {
    Normal,    // Full operation
    Warning,   // Reduce soft limit by 10%
    Critical,  // Aggressive compaction, reduce by 30%
}
```

Automatic responses to pressure:
- Reduce soft block limit
- Compact shared prefixes
- Reduce recommended batch size

### Unified Memory Awareness

```rust
// Metal-specific config defaults
MetalKVCacheConfig {
    dtype_bytes: 4,  // F32 instead of F16 for Metal
    enable_unified_memory: true,
    memory_pressure_threshold: 0.80,
}
```

### Optimal Block Sizing

```rust
// metal_kv_cache.rs:280-289
pub fn optimal_block_size(seq_len: usize, _num_layers: usize) -> usize {
    if seq_len <= 128 { 16 }
    else if seq_len <= 1024 { 32 }
    else { 64 }
}
```

---

## 9. Unimplemented Features

### 9.1 Voice Activity Detection (VAD)

**Location:** `engine/signal_frontend.rs:113`

```rust
/// TODO: Integrate Silero VAD for production use.
pub struct VoiceActivityDetector {
    // Currently uses simple energy-based VAD
}
```

**Current state:** Simple RMS energy-based detection  
**Needed:** Silero VAD model integration for production accuracy

### 9.2 Audio Codec Decoder

**Location:** `audio/codec.rs`

```rust
// Uses placeholder decoder when weights not found
info!("No codec weights found, using placeholder decoder");
```

**Current state:** Placeholder sine wave generation  
**Needed:** Full ConvNet decoder implementation

### 9.3 Speculative Decoding

**Status:** Not implemented

Speculative decoding could significantly improve throughput by:
- Using a smaller draft model to propose tokens
- Verifying multiple tokens in parallel with the main model

### 9.4 Continuous Batching with Dynamic Sequence Insertion

**Status:** Partially implemented

The scheduler supports adding requests dynamically, but true continuous batching with mid-batch insertion is not fully optimized.

### 9.5 Tensor Parallelism

**Status:** Not implemented

For larger models, tensor parallelism across multiple Metal devices would be beneficial.

### 9.6 Quantized Model Support

**Status:** Partial

- KV cache quantization: ✅ Implemented (Int8)
- Weight quantization: ⚠️ GGUF loading exists but not all models
- Activation quantization: ❌ Not implemented

---

## 10. Optimization Opportunities

### 10.1 High Priority

#### A. Flash Attention Kernel Optimization
**Current:** Using `candle_nn::ops::sdpa` when available  
**Opportunity:** Custom Metal compute shaders for fused attention

**Expected impact:** 2-3x decode speedup

#### B. KV Cache Memory Layout
**Current:** Standard tensor layout  
**Opportunity:** Contiguous memory layout optimized for Metal access patterns

**Expected impact:** 20-30% memory bandwidth improvement

#### C. Prefix Caching Hit Rate
**Current:** Hash-based prefix matching  
**Opportunity:** Trie-based prefix index for faster lookups

**Expected impact:** Better cache utilization for similar prompts

### 10.2 Medium Priority

#### D. Batch Size Auto-Tuning
**Current:** Fixed max batch size  
**Opportunity:** Dynamic batch sizing based on memory pressure and latency targets

#### E. Request Coalescing
**Current:** Individual request processing  
**Opportunity:** Coalesce similar requests (same prompt prefix) for shared computation

#### F. Async Model Loading
**Current:** Blocking model load  
**Opportunity:** Background loading with progress streaming

### 10.3 Lower Priority

#### G. Multi-Device Support
**Current:** Single device execution  
**Opportunity:** Pipeline or tensor parallelism across devices

#### H. Request Priority Learning
**Current:** Static priority assignment  
**Opportunity:** ML-based priority prediction based on request characteristics

#### I. Adaptive Chunked Prefill
**Current:** Fixed chunk threshold with pressure adjustments  
**Opportunity:** Per-request optimal chunk size based on model and hardware

---

## 11. Recommendations

### Immediate Actions (1-2 weeks)

1. **Integrate Silero VAD** for production-quality voice activity detection
2. **Implement proper audio codec decoder** to replace placeholder
3. **Add comprehensive benchmarking** for prefill/decode latency

### Short-term (1-2 months)

4. **Optimize Metal attention kernels** with custom compute shaders
5. **Implement speculative decoding** for supported models
6. **Add request coalescing** for shared prefix computation

### Medium-term (3-6 months)

7. **Full weight quantization support** (INT4/INT8)
8. **Multi-device tensor parallelism**
9. **Advanced prefix caching** with trie-based index

### Architecture Improvements

10. **Separate prefill and decode executors** for better resource isolation
11. **Add request priority queue persistence** for crash recovery
12. **Implement request deduplication** for identical concurrent requests

---

## Appendix A: Configuration Reference

### EngineCoreConfig

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_batch_size` | 8 | Maximum concurrent requests |
| `max_seq_len` | 4096 | Maximum sequence length |
| `max_tokens_per_step` | 512 | Token budget per step |
| `block_size` | 16 | KV cache block size |
| `max_blocks` | 1024 | Maximum KV cache blocks |
| `enable_chunked_prefill` | true | Enable chunked prefill |
| `chunked_prefill_threshold` | 256 | Chunk size threshold |
| `enable_preemption` | true | Enable priority preemption |
| `enable_adaptive_batching` | true | Enable adaptive scheduling |
| `target_ttft_ms` | 250.0 | Target time-to-first-token |
| `target_decode_tpot_ms` | 40.0 | Target time-per-output-token |

### Environment Variables

| Variable | Description |
|----------|-------------|
| `IZWI_KV_PAGE_SIZE` | Override default KV page size |
| `IZWI_KV_CACHE_DTYPE` | KV cache dtype (float16, int8) |
| `IZWI_USE_FLASH_ATTENTION` | Enable Flash Attention |

---

## Appendix B: File Reference

### Core Engine Files

| File | Lines | Description |
|------|-------|-------------|
| `engine/mod.rs` | 285 | Engine facade and public API |
| `engine/core.rs` | 659 | EngineCore orchestration |
| `engine/scheduler.rs` | 1376 | Request scheduling |
| `engine/kv_cache.rs` | 1404 | Paged KV cache |
| `engine/executor.rs` | 1750 | Model execution |
| `engine/metal_kv_cache.rs` | 380 | Metal optimizations |
| `engine/request.rs` | 608 | Request types |
| `engine/signal_frontend.rs` | 616 | Audio signal processing |

### Model Files

| File | Description |
|------|-------------|
| `models/registry.rs` | Model loading and caching |
| `models/shared/attention/paged.rs` | Paged attention implementation |
| `models/shared/attention/batched.rs` | Batched attention |
| `models/architectures/qwen3/` | Qwen3 model family |
| `models/architectures/lfm2/` | LFM2 audio model |

---

*Document generated from codebase analysis. For implementation details, refer to the source files.*
