# Izwi Audio vs Mini-SGLang: Production Readiness Analysis

## Executive Summary

After analyzing both repositories, **izwi-audio** has a solid foundation with modern architecture patterns, but several critical gaps prevent it from being production-ready compared to **mini-sglang**. This document outlines the current state, key gaps, and a strategic roadmap for production readiness.

---

## 1. Architecture Comparison

### Mini-SGLang Architecture
```
┌─────────────────────────────────────────────────────────────┐
│                    Mini-SGLang System                        │
├─────────────────────────────────────────────────────────────┤
│  API Server (FastAPI)                                       │
│       ↓                                                     │
│  Tokenizer Worker → ZeroMQ → Scheduler Worker (per GPU)    │
│       ↓                          ↓                          │
│  Detokenizer Worker ← NCCL ← Engine (CUDA kernels)         │
└─────────────────────────────────────────────────────────────┘
```

**Key Design Principles:**
- **Multi-process distributed architecture**: Separate processes for API, tokenization, scheduling, and compute
- **ZeroMQ for control**: Lightweight message passing between components
- **NCCL for tensor parallelism**: Efficient GPU-to-GPU communication
- **Modular backends**: FlashAttention, FlashInfer, RadixCache pluggable
- **~5,000 lines of Python**: Compact, readable, type-annotated codebase

### Izwi-Audio Architecture
```
┌─────────────────────────────────────────────────────────────┐
│                    Izwi-Audio System                         │
├─────────────────────────────────────────────────────────────┤
│  REST API Server (Axum)                                     │
│       ↓                                                     │
│  InferenceEngine (Runtime Service)                          │
│       ↓                                                     │
│  ModelRouter → BackendRouter → ModelRegistry               │
│       ↓                                                     │
│  Qwen3TtsModel / Qwen3AsrModel (Candle/MLX)               │
└─────────────────────────────────────────────────────────────┘
```

**Current Strengths:**
- Modern Rust codebase with strong type safety
- Runtime-centric architecture (runtime/, catalog/, backends/)
- vLLM-style scheduler with continuous batching
- Paged KV cache with prefix caching
- Supports TTS, ASR, and Chat tasks
- Metal GPU support for Apple Silicon
- Good separation of concerns (API, runtime, models)

---

## 2. Critical Gaps for Production Readiness

### 2.1 Multi-Process Architecture (HIGH PRIORITY)

**Mini-SGLang**: Uses separate processes for:
- API Server (1 process)
- Tokenizer Worker (1 process)
- Scheduler Workers (N processes, one per GPU/TP rank)
- Detokenizer Worker (1 process)

**Izwi-Audio**: Single-process monolithic architecture

**Impact:**
- GIL contention in Python not applicable, but Rust async can still saturate
- No process isolation - crash in model inference kills entire server
- Cannot scale across multiple Metal GPUs
- Tokenization blocks inference thread
- No way to restart components independently

**Recommendation:** Implement multi-process architecture:
```rust
// Proposed architecture
┌─────────────────────────────────────────────────────────┐
│ API Server Process (Axum)                               │
│    ↓ IPC (gRPC/Unix sockets)                            │
│ Tokenizer Process                                       │
│    ↓ IPC                                                │
│ Scheduler Process (per device)                          │
│    ↓                                                    │
│ Engine Process (Candle/MLX execution)                   │
└─────────────────────────────────────────────────────────┘
```

### 2.2 Tensor Parallelism (HIGH PRIORITY)

**Mini-SGLang**: Full tensor parallelism support with NCCL for multi-GPU

**Izwi-Audio**: Limited to single-device execution

**Current State:**
- Device selection exists but no cross-device communication
- No sharding of model weights across multiple Metal GPUs
- Model loads entirely on one device

**Recommendation:**
1. Implement tensor parallelism for multi-GPU setups (Metal MPS)
2. Weight sharding across devices
3. All-reduce operations for distributed attention
4. This is lower priority since Apple Silicon typically has unified memory on single SoC

### 2.3 KV Cache Management (MEDIUM-HIGH PRIORITY)

**Mini-SGLang**: Advanced KV cache features:
- RadixCache for prefix reuse
- PagedAttention-style block management
- NaiveCacheManager for simple use cases

**Izwi-Audio**: Good foundation but gaps:
```rust
// Current: Paged KV cache exists
pub struct KVCacheManager {
    block_allocator: BlockAllocator,
    block_tables: HashMap<RequestId, Vec<BlockId>>,
    prefix_cache: HashMap<u64, Vec<BlockId>>, // Basic prefix caching
}
```

**Gaps:**
1. No RadixCache (tree-based prefix matching)
2. Limited prefix caching (only 128 token prefix hash)
3. No automatic prefix detection from request batches
4. No overlapping generation support (speculative decoding)

**Recommendation:**
```rust
// Enhanced RadixCache
pub struct RadixCache {
    root: RadixNode,
    block_manager: BlockManager,
    eviction_policy: EvictionPolicy,
}

struct RadixNode {
    tokens: Vec<u32>,
    block_ids: Vec<BlockId>,
    children: HashMap<u32, RadixNode>,
    refcount: usize,
}
```

### 2.4 Attention Backends (MEDIUM PRIORITY)

**Mini-SGLang**: Multiple optimized backends:
- FlashAttention 3
- FlashInfer
- Custom CUDA kernels via TVM FFI

**Izwi-Audio**: Limited to Candle's native attention

**Current:**
- Uses Candle's standard attention implementation
- No custom Metal kernels
- No FlashAttention-style fused kernels

**Recommendation for CPU/Metal:**
1. Implement Metal Performance Shaders (MPS) attention kernels
2. Use MLX's optimized attention primitives where possible
3. Implement custom Metal compute shaders for:
   - Fused QKV projection + attention
   - Fused attention + output projection
   - Memory-efficient attention variants

### 2.5 Batching and Scheduling (MEDIUM PRIORITY)

**Mini-SGLang**:
- Continuous batching
- Chunked prefill (split long prompts)
- Overlap scheduling (hide CPU overhead)

**Izwi-Audio**:
- Good scheduler with FCFS and Priority modes
- Adaptive batching based on latency targets
- Chunked prefill partially implemented
- VAD preemption for audio apps (unique feature!)

**Gaps:**
1. No overlap scheduling - CPU operations block GPU
2. Limited chunked prefill integration
3. No in-flight batching optimization

### 2.6 Quantization and Optimization (MEDIUM PRIORITY)

**Mini-SGLang**:
- FP4, FP8, INT4, AWQ, GPTQ support
- Dynamic quantization

**Izwi-Audio**:
- Supports 4-bit and 8-bit models
- Uses Candle's quantization

**Gaps:**
1. No GPTQ/AWQ optimized kernels
2. No dynamic quantization at runtime
3. Limited Metal-optimized quantized operations

### 2.7 Monitoring and Observability (HIGH PRIORITY)

**Mini-SGLang**:
- Comprehensive metrics collection
- Benchmarking utilities
- Performance profiling

**Izwi-Audio**:
- Basic metrics exists (`metrics.rs`)
- Limited telemetry

**Gaps:**
1. No Prometheus/OpenTelemetry integration
2. Limited request tracing
3. No latency breakdown (tokenize/prefill/decode)
4. Missing key metrics:
   - KV cache hit rate
   - Batch size distribution
   - GPU/Metal utilization
   - Queue depth over time

### 2.8 Error Handling and Resilience (HIGH PRIORITY)

**Mini-SGLang**:
- Process-level isolation means crashes don't bring down system
- Request-level error boundaries

**Izwi-Audio**:
- Good Rust error handling with `thiserror`
- But: single process means any panic kills server

**Gaps:**
1. No request isolation (one bad request can corrupt state)
2. No automatic recovery mechanisms
3. Limited graceful degradation
4. No circuit breaker patterns

---

## 3. What's Already Production-Ready

### Strengths of Current Implementation

1. **Modern Rust Architecture**
   - Type-safe, memory-safe
   - Async/await throughout
   - Good crate organization

2. **Solid Scheduling Foundation**
   - Continuous batching implemented
   - Adaptive scheduling with latency targets
   - Priority-based and FCFS policies
   - VAD preemption (unique audio-first feature)

3. **Model Management**
   - Model registry with variant support
   - Download and caching from HuggingFace
   - Multi-format support (safetensors, GGUF)

4. **Audio-Specific Features**
   - Streaming audio generation (~97ms first packet)
   - Voice cloning support
   - Voice design capability
   - ASR/TTS/Chat in one engine

5. **API Compatibility**
   - OpenAI-compatible endpoints
   - REST API with proper error codes
   - CLI tool for management

---

## 4. Strategic Implementation Plan

### Phase 1: Foundation (Weeks 1-3)
**Goal:** Implement multi-process architecture and observability

#### Week 1: Multi-Process Architecture
- [ ] Implement process manager (`ProcessManager`)
- [ ] Create IPC layer using Unix domain sockets + gRPC
- [ ] Separate API server into standalone process
- [ ] Move tokenization to dedicated process
- [ ] Implement health checks between processes

```rust
// New crate: izwi-daemon
pub struct ProcessManager {
    api_server: ChildProcess,
    tokenizer_worker: ChildProcess,
    scheduler_workers: Vec<ChildProcess>,
}
```

#### Week 2: Observability Stack
- [ ] Prometheus metrics export
- [ ] OpenTelemetry tracing integration
- [ ] Structured logging with JSON output
- [ ] Request tracing across process boundaries
- [ ] Dashboard for key metrics

```rust
// Metrics to track
pub struct EngineMetrics {
    // Request metrics
    pub requests_total: Counter,
    pub request_duration: Histogram,
    pub queue_wait_time: Histogram,
    
    // Inference metrics
    pub time_to_first_token: Histogram,
    pub decode_latency: Histogram,
    pub prefill_tokens: Counter,
    pub decode_tokens: Counter,
    
    // KV cache metrics
    pub kv_cache_hit_rate: Gauge,
    pub kv_cache_blocks_used: Gauge,
    pub prefix_cache_hits: Counter,
    
    // System metrics
    pub metal_memory_used: Gauge,
    pub batch_size: Histogram,
}
```

#### Week 3: Resilience and Error Handling
- [ ] Process supervision with auto-restart
- [ ] Request-level timeouts and cancellation
- [ ] Circuit breaker for model inference
- [ ] Graceful degradation modes
- [ ] Request isolation (sandboxing)

### Phase 2: Performance (Weeks 4-6)
**Goal:** Advanced scheduling and KV cache optimization

#### Week 4: RadixCache Implementation
- [ ] Tree-based prefix cache
- [ ] Automatic prefix detection
- [ ] Reference counting for shared blocks
- [ ] LRU eviction policy

#### Week 5: Overlap Scheduling
- [ ] Async tokenization pipeline
- [ ] Decode stream overlapping
- [ ] Prefetch next batch while current runs
- [ ] CPU/GPU overlap optimization

#### Week 6: Metal Optimization
- [ ] Custom Metal kernels for attention
- [ ] MPS-optimized matrix operations
- [ ] Memory pool management
- [ ] Unified memory optimization

### Phase 3: Scale (Weeks 7-9)
**Goal:** Multi-device and distributed inference

#### Week 7: Tensor Parallelism
- [ ] Model sharding across Metal devices
- [ ] All-reduce communication primitives
- [ ] Pipeline parallelism support

#### Week 8: Advanced Batching
- [ ] Speculative decoding
- [ ] Prompt caching with embeddings
- [ ] Dynamic batch size adjustment
- [ ] Heterogeneous batch support

#### Week 9: Production Hardening
- [ ] Load testing and benchmarking
- [ ] Resource limits and quotas
- [ ] Rate limiting
- [ ] Authentication/authorization
- [ ] Request logging and audit

---

## 5. Priority Recommendations

### Immediate (Do First)
1. ✅ **Multi-process architecture** - Critical for stability
2. ✅ **Observability** - Can't run what you can't measure
3. ✅ **Error isolation** - Prevent cascade failures

### Short-term (Next Month)
4. ✅ **RadixCache** - Major throughput improvement
5. ✅ **Metal kernels** - Performance optimization
6. ✅ **Overlap scheduling** - Hide latency

### Medium-term (Next Quarter)
7. **Tensor parallelism** - Multi-device support
8. **Speculative decoding** - Latency reduction
9. **Advanced quantization** - Memory efficiency

---

## 6. Code Structure Recommendations

### New Crate Layout
```
crates/
├── izwi-core/              # Core types and traits
├── izwi-runtime/           # Single-process runtime (current)
├── izwi-daemon/            # Multi-process orchestration
├── izwi-scheduler/         # Distributed scheduler
├── izwi-tokenizer/         # Tokenization service
├── izwi-server/            # API server
├── izwi-cli/               # CLI tool
├── izwi-backends/
│   ├── candle-backend/     # Candle execution
│   ├── mlx-backend/        # MLX execution
│   └── metal-kernels/      # Custom Metal shaders
└── izwi-metrics/           # Observability
```

### Key Traits to Implement
```rust
// Process isolation trait
pub trait Worker: Send + Sync {
    async fn start(&self) -> Result<WorkerHandle>;
    async fn health_check(&self) -> HealthStatus;
    async fn shutdown(&self) -> Result<()>;
}

// Backend abstraction
pub trait InferenceBackend: Send + Sync {
    async fn load_model(&self, model: ModelVariant) -> Result<()>;
    async fn execute(&self, batch: Batch) -> Result<BatchOutput>;
    fn memory_usage(&self) -> MemoryStats;
}

// Cache backend trait
pub trait CacheBackend: Send + Sync {
    fn get(&self, tokens: &[u32]) -> Option<Vec<BlockId>>;
    fn insert(&self, tokens: &[u32], blocks: Vec<BlockId>);
    fn evict(&self, num_blocks: usize) -> Vec<BlockId>;
}
```

---

## 7. MLX-Specific Recommendations

Since izwi-audio targets Apple Silicon with MLX:

### Advantages of MLX
1. Unified memory (CPU/GPU share address space)
2. Lazy evaluation for graph optimization
3. Native Swift/Objective-C++ integration
4. Optimized for Apple Silicon

### Implementation Strategy
```rust
// Use mlx-rs bindings when mature
// Until then, create FFI layer
pub struct MlxBackend {
    device: MlxDevice,
    graph: MlxGraph,
}

impl InferenceBackend for MlxBackend {
    async fn execute(&self, batch: Batch) -> Result<BatchOutput> {
        // Lazy graph construction
        // Unified memory means no copies
        // Compile and execute
    }
}
```

### Metal Kernel Priorities
1. **FlashAttention-style fused attention** - Most impactful
2. **Quantized matmul** - For 4-bit inference
3. **RMSNorm fusion** - Reduce kernel launches
4. **RoPE fusion** - Position embedding

---

## 8. Testing Strategy

### Unit Tests
- Model loading and inference
- Scheduler logic
- Cache management
- Tokenizer correctness

### Integration Tests
- End-to-end TTS/ASR pipeline
- Multi-process communication
- Error recovery scenarios

### Load Tests
- Concurrent request handling
- Memory pressure scenarios
- Long-running stability

### Benchmarks
- Compare with mini-sglang on Qwen3 models
- Throughput (tokens/sec)
- Latency percentiles (p50, p95, p99)
- Memory efficiency

---

## 9. Success Metrics

### Performance Targets (vs Mini-SGLang)
- **Latency**: Within 20% of mini-sglang on equivalent hardware
- **Throughput**: 80%+ of mini-sglang throughput
- **Memory**: Efficient unified memory usage on Metal
- **Reliability**: 99.9% uptime with process supervision

### Production Readiness Checklist
- [ ] Multi-process architecture stable
- [ ] Comprehensive metrics and alerting
- [ ] Automatic recovery from failures
- [ ] Resource limits and isolation
- [ ] Security hardening
- [ ] Documentation and runbooks
- [ ] Load tested to 10x expected traffic
- [ ] graceful degradation tested

---

## 10. Conclusion

**Izwi-audio** has excellent bones with its modern Rust architecture and audio-specific optimizations. The main gap is **infrastructure hardening** rather than core inference capabilities.

**Key takeaways:**
1. Multi-process architecture is the #1 priority for production
2. Current scheduler is quite sophisticated (better than mini-sglang in some ways)
3. Metal/MLX optimization will differentiate from CUDA-focused alternatives
4. Audio-first features (VAD preemption, streaming) are unique advantages
5. Strong type safety and Rust async provide good foundation

**Estimated timeline to production-ready:** 6-9 weeks with focused effort on infrastructure (multi-process, observability, resilience), followed by ongoing performance optimization.

---

## References

- [Mini-SGLang Repository](https://github.com/sgl-project/mini-sglang)
- [Mini-SGLang Architecture Doc](https://github.com/sgl-project/mini-sglang/blob/main/docs/structures.md)
- [SGLang Blog Post](https://lmsys.org/blog/2025-12-17-minisgl/)
- [vLLM Paper](https://arxiv.org/abs/2309.06180)
- [MLX Documentation](https://ml-explore.github.io/mlx/build/html/)
