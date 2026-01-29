# Izwi Audio Inference Engine Architecture

**Version:** 0.1.0  
**Last Updated:** January 2026  
**Target Audience:** Engineers onboarding to the izwi-audio project

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture Principles](#architecture-principles)
3. [System Architecture](#system-architecture)
4. [Core Components](#core-components)
5. [Data Flow](#data-flow)
6. [Request Lifecycle](#request-lifecycle)
7. [Memory Management](#memory-management)
8. [Performance & Metrics](#performance--metrics)
9. [Configuration](#configuration)
10. [API Reference](#api-reference)
11. [Examples](#examples)
12. [Troubleshooting](#troubleshooting)

---

## Overview

The Izwi Audio Inference Engine is a production-ready, high-throughput audio generation system designed for TTS (Text-to-Speech), ASR (Automatic Speech Recognition), and audio-to-audio chat tasks. It follows architectural patterns from vLLM, adapted for audio models running on Apple Silicon (M1/M2/M3) and CPU environments.

### Key Features

- **High Throughput**: Continuous batching and efficient scheduling
- **Memory Efficient**: Paged KV-cache with block-based allocation
- **Flexible Scheduling**: FCFS and priority-based policies
- **Streaming Support**: Real-time audio chunk delivery
- **Production Ready**: Comprehensive metrics, tracing, and error handling
- **Extensible**: Plugin architecture for multiple model backends

### Supported Models

- **LFM2-Audio** (LiquidAI): Multi-modal audio model for TTS, ASR, and chat
- **Qwen3-TTS** (Alibaba): High-quality text-to-speech with voice cloning

---

## Architecture Principles

The engine is built on these core principles:

1. **Separation of Concerns**: Each component has a single, well-defined responsibility
2. **Step-based Execution**: Discrete processing steps (schedule → execute → process)
3. **Resource Management**: Explicit memory allocation and tracking
4. **Async-First**: Built on Tokio for efficient concurrency
5. **Observability**: Comprehensive metrics and tracing throughout

### Design Inspiration

The architecture draws from **vLLM** (UC Berkeley), specifically:
- Paged attention for memory efficiency
- Continuous batching for throughput
- Token budget management
- Request scheduling with preemption support

---

## System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Engine                                   │
│  ┌──────────────┐  ┌───────────┐  ┌──────────────────────────┐ │
│  │   Request    │  │           │  │      Engine Core          │ │
│  │  Processor   │──│ Scheduler │──│  ┌────────────────────┐  │ │
│  │              │  │           │  │  │  Model Executor    │  │ │
│  └──────────────┘  └───────────┘  │  │  (Python Bridge)   │  │ │
│                                    │  └────────────────────┘  │ │
│  ┌──────────────┐                 │  ┌────────────────────┐  │ │
│  │   Output     │◄────────────────│  │  KV Cache Manager  │  │ │
│  │  Processor   │                 │  └────────────────────┘  │ │
│  └──────────────┘                 └──────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### Module Structure

```
crates/izwi-core/src/engine/
├── mod.rs              # Engine - main interface
├── core.rs             # EngineCore - orchestrator
├── scheduler.rs        # Scheduler - request scheduling
├── kv_cache.rs         # KVCacheManager - memory management
├── request.rs          # RequestProcessor - input validation
├── output.rs           # OutputProcessor - result formatting
├── executor.rs         # ModelExecutor - inference backend
├── types.rs            # Core type definitions
├── config.rs           # Configuration types
└── metrics.rs          # Metrics & benchmarking
```

---

## Core Components

### 1. Engine

**Location:** `engine/mod.rs`  
**Responsibility:** Primary user-facing interface

The `Engine` struct is the main entry point for all inference operations. It provides:

- **Synchronous generation**: `generate(request) -> EngineOutput`
- **Streaming generation**: `generate_streaming(request) -> (RequestId, Receiver<StreamingOutput>)`
- **Continuous execution**: `run()` - processes requests in a loop
- **Request management**: `add_request()`, `abort_request()`
- **Metrics access**: `metrics()`, `pending_requests()`, `running_requests()`

**Key Methods:**

```rust
pub async fn generate(&self, request: EngineCoreRequest) -> Result<EngineOutput>
pub async fn generate_streaming(&self, request: EngineCoreRequest) 
    -> Result<(RequestId, mpsc::Receiver<StreamingOutput>)>
pub async fn step(&self) -> Result<Vec<EngineOutput>>
pub async fn run(&self) -> Result<()>
```

### 2. EngineCore

**Location:** `engine/core.rs`  
**Responsibility:** Central orchestrator for the inference loop

The `EngineCore` coordinates all components and implements the step-based execution model:

**Step Phases:**
1. **Schedule**: Select requests to process (via Scheduler)
2. **Execute**: Run forward pass (via ModelExecutor)
3. **Process**: Format outputs and update state (via OutputProcessor)

**State Management:**
- Tracks active requests by ID
- Records request start times for latency measurement
- Manages sequence ID allocation
- Coordinates initialization and shutdown

**Key Methods:**

```rust
pub async fn step(&mut self) -> Result<Vec<EngineOutput>>
pub fn add_request(&mut self, request: EngineCoreRequest) -> Result<()>
pub async fn initialize(&mut self) -> Result<()>
pub async fn shutdown(&mut self) -> Result<()>
```

### 3. Scheduler

**Location:** `engine/scheduler.rs`  
**Responsibility:** Request queue management and scheduling decisions

The Scheduler determines which requests to process in each step based on:

**Scheduling Policies:**
- **FCFS** (First-Come-First-Served): Default, fair ordering
- **Priority**: Higher priority requests processed first

**Queue Management:**
- **Waiting Queue**: New requests awaiting resources
- **Running Queue**: Requests currently being processed

**Resource Constraints:**
- **Batch Size**: Maximum concurrent requests (default: 8)
- **Token Budget**: Maximum tokens per step (default: 512)
- **KV Cache**: Available memory blocks

**Advanced Features:**
- **Chunked Prefill**: Split long prompts across multiple steps
- **Preemption**: Pause low-priority requests when resources are scarce

**Key Methods:**

```rust
pub fn schedule(&mut self, kv_cache: &mut KVCacheManager) -> ScheduleResult
pub fn add_request(&mut self, request: &EngineCoreRequest)
pub fn finish_request(&mut self, request_id: &RequestId, kv_cache: &mut KVCacheManager)
pub fn abort_request(&mut self, request_id: &RequestId, kv_cache: &mut KVCacheManager) -> bool
```

**ScheduleResult Structure:**

```rust
pub struct ScheduleResult {
    pub decode_requests: Vec<ScheduledRequest>,   // Continuing requests
    pub prefill_requests: Vec<ScheduledRequest>,  // New requests
    pub preempted_requests: Vec<RequestId>,       // Paused requests
    pub total_tokens: usize,                      // Token budget used
    pub blocks_allocated: usize,                  // KV cache blocks allocated
}
```

### 4. KVCacheManager

**Location:** `engine/kv_cache.rs`  
**Responsibility:** Paged attention-style memory management

Implements block-based memory allocation inspired by virtual memory paging:

**Architecture:**
- **Block Size**: 16 tokens per block (configurable)
- **Block Allocator**: Free list with LIFO allocation for cache locality
- **Block Table**: Maps request IDs to physical block IDs

**Memory Layout:**

```
Request A: [Block 0] → [Block 1] → [Block 2]
Request B: [Block 5] → [Block 6]
Free List: [Block 3, Block 4, Block 7, ...]
```

**Benefits:**
- Non-contiguous allocation (no fragmentation)
- Efficient memory reuse
- Copy-on-write support (future: prefix caching)

**Key Methods:**

```rust
pub fn allocate(&mut self, request_id: &RequestId, num_blocks: usize) -> Vec<BlockId>
pub fn extend(&mut self, request_id: &RequestId, additional_blocks: usize) -> Vec<BlockId>
pub fn free(&mut self, request_id: &RequestId)
pub fn stats(&self) -> KVCacheStats
```

### 5. RequestProcessor

**Location:** `engine/request.rs`  
**Responsibility:** Input validation and preprocessing

Validates and normalizes incoming requests:

**Validation:**
- Task-specific requirements (TTS needs text, ASR needs audio)
- Parameter bounds (temperature, top_p, max_tokens)
- Model compatibility

**Preprocessing:**
- Tokenization estimation
- Parameter clamping
- Default value application

**Request Types:**

```rust
pub struct EngineCoreRequest {
    pub id: RequestId,
    pub task_type: TaskType,           // TTS, ASR, AudioChat
    pub model_type: ModelType,         // LFM2Audio, Qwen3TTS
    pub text: Option<String>,
    pub audio_input: Option<String>,   // Base64 encoded
    pub params: GenerationParams,
    pub priority: Priority,
    pub streaming: bool,
    // ... additional fields
}
```

### 6. OutputProcessor

**Location:** `engine/output.rs`  
**Responsibility:** Result formatting and streaming

Converts raw executor outputs to user-facing results:

**Capabilities:**
- Format audio samples and metadata
- Calculate statistics (RTF, tokens/sec)
- Manage streaming sessions
- Chunk audio for streaming delivery

**Streaming:**
- Configurable chunk size (default: 4800 samples = 200ms @ 24kHz)
- Real-time statistics per chunk
- Automatic session cleanup

**Key Methods:**

```rust
pub fn process(&mut self, executor_output: ExecutorOutput, 
               sequence_id: SequenceId, generation_time: Duration) -> EngineOutput
pub async fn add_streaming_samples(&mut self, request_id: &RequestId, 
                                    samples: Vec<f32>) -> bool
pub async fn finish_streaming(&mut self, request_id: &RequestId, 
                               text: Option<String>) -> Option<StreamingStats>
```

### 7. ModelExecutor

**Location:** `engine/executor.rs`  
**Responsibility:** Abstract interface for model inference

Trait-based design allowing multiple backends:

**Current Implementation:**
- **PythonExecutor**: Bridges to Python daemons (lfm2_daemon.py, tts_daemon.py)

**Future Implementations:**
- Native Rust inference (MLX-rs, candle, etc.)
- Remote inference (HTTP, gRPC)

**Execution Flow:**

```rust
pub trait ModelExecutor {
    fn execute(&self, requests: &[&EngineCoreRequest], 
               scheduled: &[ScheduledRequest]) -> Result<Vec<ExecutorOutput>>;
    fn is_ready(&self) -> bool;
    fn initialize(&mut self) -> Result<()>;
    fn shutdown(&mut self) -> Result<()>;
}
```

**PythonExecutor Details:**
- Manages daemon lifecycle (start/stop)
- Unix socket communication
- Automatic retry with fallback
- Supports batching (future enhancement)

### 8. MetricsCollector

**Location:** `engine/metrics.rs`  
**Responsibility:** Performance tracking and benchmarking

Comprehensive metrics collection:

**Tracked Metrics:**
- Latency (avg, p50, p90, p99)
- Real-Time Factor (RTF)
- Throughput (tokens/sec, requests/sec)
- Audio duration generated
- Processing time

**Storage:**
- Rolling window of samples (default: 1000)
- Atomic counters for totals
- Instant-based timing

**Key Methods:**

```rust
pub async fn record_request(&self, latency: Duration, tokens_generated: u64, 
                             audio_duration: Duration)
pub async fn snapshot(&self) -> MetricsSnapshot
pub async fn reset(&self)
```

---

## Data Flow

### Request Flow Diagram

```
┌─────────────┐
│   Client    │
└──────┬──────┘
       │ 1. EngineCoreRequest
       ▼
┌─────────────────────┐
│ RequestProcessor    │ ◄── Validate input
│ - Validate          │     Normalize params
│ - Tokenize estimate │     Set defaults
└──────┬──────────────┘
       │ 2. Validated Request
       ▼
┌─────────────────────┐
│   EngineCore        │
│ - Track request     │
│ - Record start time │
└──────┬──────────────┘
       │ 3. Add to scheduler
       ▼
┌─────────────────────┐
│    Scheduler        │
│ - Waiting Queue     │ ◄── FCFS or Priority
│ - Running Queue     │     Token budget
└──────┬──────────────┘     KV cache check
       │
       │ ═══ STEP LOOP ═══
       │
       │ 4. schedule()
       ▼
┌─────────────────────┐
│  ScheduleResult     │ ◄── Prefill requests
│ - Prefill: [...]    │     Decode requests
│ - Decode: [...]     │     Resource allocation
└──────┬──────────────┘
       │ 5. Execute scheduled
       ▼
┌─────────────────────┐
│  ModelExecutor      │ ◄── Python daemon
│ - Forward pass      │     Model inference
│ - Audio generation  │     Token sampling
└──────┬──────────────┘
       │ 6. ExecutorOutput
       ▼
┌─────────────────────┐
│  OutputProcessor    │ ◄── Format results
│ - Format audio      │     Calculate stats
│ - Calculate RTF     │     Stream chunks
└──────┬──────────────┘
       │ 7. EngineOutput
       ▼
┌─────────────────────┐
│   Client            │
│ - Audio samples     │
│ - Metadata          │
│ - Statistics        │
└─────────────────────┘
```

### Memory Flow (KV Cache)

```
Request Added
     │
     ▼
┌─────────────────────────────────────┐
│ Scheduler.schedule()                │
│ - Calculate blocks needed           │
│ - Check KVCacheManager.can_allocate│
└──────────┬──────────────────────────┘
           │
           ▼ YES
┌─────────────────────────────────────┐
│ KVCacheManager.allocate()           │
│ - Pop blocks from free list         │
│ - Map request_id → block_ids        │
│ - Update block table                │
└──────────┬──────────────────────────┘
           │
           ▼
┌─────────────────────────────────────┐
│ ScheduledRequest                    │
│ - request_id                        │
│ - block_ids: [0, 1, 2]             │
│ - num_tokens                        │
└──────────┬──────────────────────────┘
           │
           ▼ (During generation)
┌─────────────────────────────────────┐
│ KVCacheManager.extend()             │
│ - Allocate more blocks if needed    │
└──────────┬──────────────────────────┘
           │
           ▼ (Request finished)
┌─────────────────────────────────────┐
│ KVCacheManager.free()               │
│ - Return blocks to free list        │
│ - Remove from block table           │
└─────────────────────────────────────┘
```

---

## Request Lifecycle

### State Transitions

```
┌─────────┐
│ Created │
└────┬────┘
     │ add_request()
     ▼
┌─────────┐
│ Waiting │ ◄─────────────────┐
└────┬────┘                   │
     │ schedule()             │ (chunked prefill)
     ▼                        │
┌─────────┐                   │
│ Running │ ──────────────────┘
│ (Prefill)│
└────┬────┘
     │ prefill complete
     ▼
┌─────────┐
│ Running │
│ (Decode)│
└────┬────┘
     │ stop condition met
     ▼
┌──────────┐
│ Finished │
└──────────┘

Abort: Any state → Aborted
```

### Detailed Lifecycle

1. **Creation**
   - Client creates `EngineCoreRequest`
   - Sets task type, text/audio, parameters

2. **Validation** (RequestProcessor)
   - Check required fields
   - Validate parameters
   - Estimate token count

3. **Waiting** (Scheduler)
   - Added to waiting queue
   - Ordered by policy (FCFS/Priority)
   - Waits for resources

4. **Scheduling** (Scheduler.schedule)
   - Check batch size limit
   - Check token budget
   - Allocate KV cache blocks
   - Move to running queue

5. **Prefill** (First execution)
   - Process prompt tokens
   - Build KV cache
   - May be chunked if prompt is long

6. **Decode** (Subsequent executions)
   - Generate one token per step
   - Update KV cache
   - Check stop conditions

7. **Completion**
   - Free KV cache blocks
   - Remove from running queue
   - Return final output

8. **Cleanup**
   - Remove from tracking maps
   - Record metrics
   - Notify streaming clients

---

## Memory Management

### KV Cache Architecture

The KV cache uses a **paged attention** design:

**Block Structure:**
```
Block 0: [K₀, K₁, ..., K₁₅, V₀, V₁, ..., V₁₅]
         └─── 16 tokens ───┘  └─── 16 tokens ───┘
```

**Memory Calculation:**
```
block_memory = 2 (K+V) × block_size × num_heads × head_dim × dtype_bytes × num_layers
             = 2 × 16 × 16 × 64 × 2 × 24
             = 1,572,864 bytes per block
             ≈ 1.5 MB per block
```

**Default Configuration:**
- Max blocks: 1024
- Total memory: ~1.5 GB
- Supports ~16,384 tokens across all requests

### Block Allocation Strategy

**LIFO (Last-In-First-Out):**
- Recently freed blocks reused first
- Better cache locality
- Reduces memory fragmentation

**Allocation Example:**

```rust
// Request A needs 3 blocks
let blocks_a = kv_cache.allocate("req_a", 3);
// blocks_a = [0, 1, 2]

// Request B needs 2 blocks
let blocks_b = kv_cache.allocate("req_b", 2);
// blocks_b = [3, 4]

// Request A finishes
kv_cache.free("req_a");
// Free list: [2, 1, 0, 5, 6, ...]

// Request C needs 1 block
let blocks_c = kv_cache.allocate("req_c", 1);
// blocks_c = [2]  (LIFO - most recently freed)
```

### Memory Pressure Handling

When KV cache is full:

1. **Check preemption enabled**
2. **Select victim** (lowest priority running request)
3. **Save state** (future: swap to disk)
4. **Free blocks**
5. **Schedule new request**

---

## Performance & Metrics

### Key Performance Indicators

**Real-Time Factor (RTF):**
```
RTF = generation_time / audio_duration

RTF < 1.0 = Faster than real-time ✓
RTF = 1.0 = Real-time
RTF > 1.0 = Slower than real-time ✗
```

**Throughput:**
```
Tokens/sec = total_tokens_generated / total_time
Requests/sec = total_requests / total_time
```

**Latency Percentiles:**
- **p50**: Median latency (typical user experience)
- **p90**: 90th percentile (most users)
- **p99**: 99th percentile (worst case for most)

### Metrics Collection

**Automatic Recording:**
```rust
// Metrics are automatically recorded on request completion
let timer = RequestTimer::start(metrics.clone());
// ... process request ...
timer.stop(tokens_generated, audio_duration).await;
```

**Accessing Metrics:**
```rust
let snapshot = engine.metrics().await;
println!("Average RTF: {:.3}", snapshot.avg_rtf);
println!("p99 Latency: {:.1}ms", snapshot.p99_latency_ms);
println!("Throughput: {:.1} tokens/sec", snapshot.avg_tokens_per_sec);
```

### Benchmarking

```rust
use izwi_core::engine::metrics::BenchmarkResult;

let start = Instant::now();
// Run benchmark...
let duration = start.elapsed();
let snapshot = metrics.snapshot().await;

let result = BenchmarkResult::new(
    "TTS Benchmark",
    num_requests,
    duration,
    snapshot,
);

println!("{}", result.summary());
```

---

## Configuration

### EngineCoreConfig

**Location:** `engine/config.rs`

```rust
pub struct EngineCoreConfig {
    // Model settings
    pub model_type: ModelType,              // LFM2Audio or Qwen3TTS
    pub models_dir: PathBuf,                // Model storage directory
    
    // Scheduling
    pub max_batch_size: usize,              // Default: 8
    pub max_seq_len: usize,                 // Default: 4096
    pub max_tokens_per_step: usize,         // Default: 512
    pub scheduling_policy: SchedulingPolicy, // FCFS or Priority
    
    // KV Cache
    pub block_size: usize,                  // Default: 16 tokens
    pub max_blocks: usize,                  // Default: 1024
    
    // Chunked Prefill
    pub enable_chunked_prefill: bool,       // Default: true
    pub chunked_prefill_threshold: usize,   // Default: 256 tokens
    
    // Audio
    pub sample_rate: u32,                   // Default: 24000 Hz
    pub num_codebooks: usize,               // Default: 8
    pub streaming_chunk_size: usize,        // Default: 4800 samples
    
    // Hardware
    pub use_metal: bool,                    // Default: true on macOS
    pub num_threads: usize,                 // Default: min(8, num_cpus)
    
    // Advanced
    pub enable_preemption: bool,            // Default: true
    pub daemon_config: DaemonConfig,
}
```

### Preset Configurations

```rust
// For LFM2-Audio
let config = EngineCoreConfig::for_lfm2();

// For Qwen3-TTS
let config = EngineCoreConfig::for_qwen3_tts();

// Custom
let config = EngineCoreConfig {
    max_batch_size: 16,
    scheduling_policy: SchedulingPolicy::Priority,
    ..Default::default()
};
```

---

## API Reference

### Creating an Engine

```rust
use izwi_core::engine::{Engine, EngineCoreConfig};

let config = EngineCoreConfig::for_lfm2();
let engine = Engine::new(config)?;
```

### Synchronous Generation

```rust
use izwi_core::engine::EngineCoreRequest;

let request = EngineCoreRequest::tts("Hello, world!")
    .with_voice("us_female")
    .with_params(GenerationParams {
        temperature: 0.7,
        max_tokens: 2048,
        ..Default::default()
    });

let output = engine.generate(request).await?;

println!("Generated {} samples", output.audio.samples.len());
println!("RTF: {:.3}", output.rtf());
```

### Streaming Generation

```rust
let request = EngineCoreRequest::tts("This is a longer text that will be streamed.")
    .with_streaming(true);

let (request_id, mut rx) = engine.generate_streaming(request).await?;

while let Some(chunk) = rx.recv().await {
    println!("Received chunk {}: {} samples", 
             chunk.sequence, chunk.samples.len());
    
    if chunk.is_final {
        println!("Final chunk received");
        break;
    }
}
```

### Continuous Execution

```rust
// Start engine in background
let engine = Arc::new(engine);
let engine_clone = engine.clone();

tokio::spawn(async move {
    engine_clone.run().await.unwrap();
});

// Add requests
engine.add_request(request1).await?;
engine.add_request(request2).await?;

// Stop engine
engine.stop();
```

### ASR (Automatic Speech Recognition)

```rust
let audio_base64 = "..."; // Base64 encoded WAV

let request = EngineCoreRequest::asr(audio_base64)
    .with_params(GenerationParams {
        max_tokens: 1024,
        ..Default::default()
    });

let output = engine.generate(request).await?;
println!("Transcription: {}", output.text.unwrap());
```

### Audio Chat

```rust
let request = EngineCoreRequest::audio_chat(
    Some(audio_base64),  // Optional audio input
    Some("What's the weather?".to_string())  // Optional text input
)
.with_voice("us_female");

let output = engine.generate(request).await?;
println!("Response text: {}", output.text.unwrap());
// output.audio contains the audio response
```

---

## Examples

### Example 1: Basic TTS

```rust
use izwi_core::engine::{Engine, EngineCoreConfig, EngineCoreRequest};

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize engine
    let config = EngineCoreConfig::for_lfm2();
    let engine = Engine::new(config)?;
    
    // Create request
    let request = EngineCoreRequest::tts("Hello from Izwi!")
        .with_voice("us_female");
    
    // Generate
    let output = engine.generate(request).await?;
    
    // Save to file
    save_wav("output.wav", &output.audio.samples, output.audio.sample_rate)?;
    
    println!("Generated audio in {:.2}s (RTF: {:.3})", 
             output.generation_time.as_secs_f32(),
             output.rtf());
    
    Ok(())
}
```

### Example 2: Batch Processing

```rust
use futures::future::join_all;

#[tokio::main]
async fn main() -> Result<()> {
    let config = EngineCoreConfig {
        max_batch_size: 16,
        ..EngineCoreConfig::for_lfm2()
    };
    let engine = Arc::new(Engine::new(config)?);
    
    // Start engine
    let engine_clone = engine.clone();
    tokio::spawn(async move {
        engine_clone.run().await.unwrap();
    });
    
    // Submit multiple requests
    let texts = vec![
        "First sentence.",
        "Second sentence.",
        "Third sentence.",
    ];
    
    let mut handles = vec![];
    for text in texts {
        let engine = engine.clone();
        let handle = tokio::spawn(async move {
            let request = EngineCoreRequest::tts(text);
            engine.add_request(request).await
        });
        handles.push(handle);
    }
    
    // Wait for all
    join_all(handles).await;
    
    // Wait for processing
    while engine.pending_requests().await > 0 || 
          engine.running_requests().await > 0 {
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    
    engine.stop();
    Ok(())
}
```

### Example 3: Priority Scheduling

```rust
use izwi_core::engine::types::Priority;

let config = EngineCoreConfig {
    scheduling_policy: SchedulingPolicy::Priority,
    ..Default::default()
};
let engine = Engine::new(config)?;

// High priority request (processed first)
let urgent = EngineCoreRequest::tts("Urgent message!")
    .with_priority(Priority::High);

// Normal priority
let normal = EngineCoreRequest::tts("Normal message.")
    .with_priority(Priority::Normal);

engine.add_request(normal).await?;
engine.add_request(urgent).await?;  // Will be processed first
```

### Example 4: Metrics Monitoring

```rust
use std::time::Duration;

#[tokio::main]
async fn main() -> Result<()> {
    let engine = Engine::new(EngineCoreConfig::for_lfm2())?;
    
    // Start monitoring task
    let engine_clone = engine.clone();
    tokio::spawn(async move {
        loop {
            tokio::time::sleep(Duration::from_secs(5)).await;
            
            let metrics = engine_clone.metrics().await;
            println!("=== Metrics ===");
            println!("Requests: {}", metrics.total_requests);
            println!("Avg RTF: {:.3}", metrics.avg_rtf);
            println!("p99 Latency: {:.1}ms", metrics.p99_latency_ms);
            println!("Throughput: {:.1} tokens/sec", metrics.avg_tokens_per_sec);
        }
    });
    
    // ... process requests ...
    
    Ok(())
}
```

---

## Troubleshooting

### Common Issues

#### 1. "KV cache full" warnings

**Symptom:** Requests are not being scheduled

**Solutions:**
- Increase `max_blocks` in config
- Reduce `max_batch_size`
- Enable preemption
- Reduce `max_seq_len`

```rust
let config = EngineCoreConfig {
    max_blocks: 2048,  // Double the default
    enable_preemption: true,
    ..Default::default()
};
```

#### 2. High RTF (> 1.0)

**Symptom:** Generation slower than real-time

**Solutions:**
- Reduce batch size
- Reduce max_tokens_per_step
- Check CPU/memory usage
- Verify daemon is using MPS (Apple Silicon)

```rust
let config = EngineCoreConfig {
    max_batch_size: 4,
    max_tokens_per_step: 256,
    ..Default::default()
};
```

#### 3. Python daemon not starting

**Symptom:** "Daemon not available" errors

**Solutions:**
- Check Python dependencies installed
- Verify socket path permissions
- Check daemon logs in stderr
- Manually start daemon for debugging

```bash
# Start daemon manually
python scripts/lfm2_daemon.py

# Check socket
ls -la /tmp/izwi_lfm2_daemon.sock
```

#### 4. Out of memory

**Symptom:** Process killed or allocation errors

**Solutions:**
- Reduce KV cache size
- Reduce batch size
- Monitor with `kv_cache_stats()`

```rust
let stats = engine_core.kv_cache_stats();
println!("Memory used: {} MB", stats.memory_used_bytes / 1_000_000);
println!("Utilization: {:.1}%", stats.utilization() * 100.0);
```

### Debugging Tips

**Enable detailed logging:**
```bash
RUST_LOG=izwi_core=debug cargo run
```

**Check scheduler state:**
```rust
println!("Waiting: {}", engine.pending_requests().await);
println!("Running: {}", engine.running_requests().await);
```

**Monitor step execution:**
```rust
loop {
    let outputs = engine.step().await?;
    println!("Step completed: {} outputs", outputs.len());
}
```

**Profile with metrics:**
```rust
let snapshot = metrics_collector.snapshot().await;
println!("{:#?}", snapshot);
```

---

## Appendix

### Glossary

- **RTF (Real-Time Factor)**: Ratio of generation time to audio duration
- **Prefill**: Initial processing of prompt tokens
- **Decode**: Autoregressive token generation
- **KV Cache**: Key-Value cache for attention mechanism
- **Block**: Fixed-size memory unit for KV cache
- **Token Budget**: Maximum tokens processable per step
- **Chunked Prefill**: Splitting long prompts across multiple steps
- **Preemption**: Pausing low-priority requests for high-priority ones

### References

- [vLLM Paper](https://arxiv.org/abs/2309.06180)
- [vLLM Blog: Anatomy of vLLM](https://blog.vllm.ai/2025/09/05/anatomy-of-vllm.html)
- [Paged Attention](https://arxiv.org/abs/2309.06180)
- [LFM2-Audio](https://huggingface.co/LiquidAI/LFM2-Audio-1.5B)
- [Qwen3-TTS](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice)

### Contributing

When extending the engine:

1. **Maintain separation of concerns**: Each component should have a single responsibility
2. **Add tests**: Unit tests for new components, integration tests for flows
3. **Update metrics**: Add relevant performance counters
4. **Document**: Update this architecture doc and inline comments
5. **Benchmark**: Measure performance impact

### Future Enhancements

- [ ] Native Rust inference (MLX-rs, Candle)
- [ ] Prefix caching for common prompts
- [ ] Speculative decoding
- [ ] Multi-GPU support
- [ ] Request batching in Python daemon
- [ ] Disk-based KV cache swapping
- [ ] Dynamic batching
- [ ] Model parallelism

---

**Document Version:** 1.0  
**Last Updated:** January 28, 2026  
**Maintained by:** Izwi Audio Team
