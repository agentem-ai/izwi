//! Engine configuration types.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

use super::scheduler::SchedulingPolicy;
use super::types::ModelType;

/// Configuration for the engine core.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineCoreConfig {
    /// Model type to use
    #[serde(default)]
    pub model_type: ModelType,

    /// Directory containing models
    #[serde(default = "default_models_dir")]
    pub models_dir: PathBuf,

    /// Maximum batch size for inference
    #[serde(default = "default_max_batch_size")]
    pub max_batch_size: usize,

    /// Maximum sequence length (tokens)
    #[serde(default = "default_max_seq_len")]
    pub max_seq_len: usize,

    /// Maximum number of tokens per step (token budget)
    #[serde(default = "default_max_tokens_per_step")]
    pub max_tokens_per_step: usize,

    /// Block size for KV cache paged attention
    #[serde(default = "default_block_size")]
    pub block_size: usize,

    /// Maximum number of KV cache blocks
    #[serde(default = "default_max_blocks")]
    pub max_blocks: usize,

    /// Scheduling policy
    #[serde(default)]
    pub scheduling_policy: SchedulingPolicy,

    /// Enable chunked prefill for long prompts
    #[serde(default = "default_chunked_prefill")]
    pub enable_chunked_prefill: bool,

    /// Threshold for chunked prefill (tokens)
    #[serde(default = "default_chunked_prefill_threshold")]
    pub chunked_prefill_threshold: usize,

    /// Output sample rate (Hz)
    #[serde(default = "default_sample_rate")]
    pub sample_rate: u32,

    /// Number of audio codebooks
    #[serde(default = "default_num_codebooks")]
    pub num_codebooks: usize,

    /// Chunk size for streaming output (samples)
    #[serde(default = "default_streaming_chunk_size")]
    pub streaming_chunk_size: usize,

    /// Enable Metal/MPS acceleration (macOS)
    #[serde(default = "default_use_metal")]
    pub use_metal: bool,

    /// Number of CPU threads
    #[serde(default = "default_num_threads")]
    pub num_threads: usize,

    /// Enable request preemption when KV cache is full
    #[serde(default = "default_enable_preemption")]
    pub enable_preemption: bool,

    /// Enable adaptive scheduling heuristics driven by runtime latency feedback.
    #[serde(default = "default_enable_adaptive_batching")]
    pub enable_adaptive_batching: bool,

    /// Minimum token budget per scheduler step when adaptive batching is enabled.
    #[serde(default = "default_min_tokens_per_step")]
    pub min_tokens_per_step: usize,

    /// Target time-to-first-token for adaptive scheduling.
    #[serde(default = "default_target_ttft_ms")]
    pub target_ttft_ms: f64,

    /// Target time-per-output-token for adaptive scheduling.
    #[serde(default = "default_target_decode_tpot_ms")]
    pub target_decode_tpot_ms: f64,

    /// Waiting-time interval used for priority aging in adaptive scheduling.
    #[serde(default = "default_priority_aging_ms")]
    pub priority_aging_ms: u64,
}

fn default_models_dir() -> PathBuf {
    dirs::data_local_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("izwi")
        .join("models")
}

fn default_max_batch_size() -> usize {
    8
}
fn default_max_seq_len() -> usize {
    4096
}
fn default_max_tokens_per_step() -> usize {
    512
}
fn default_block_size() -> usize {
    16
}
fn default_max_blocks() -> usize {
    1024
}
fn default_chunked_prefill() -> bool {
    true
}
fn default_chunked_prefill_threshold() -> usize {
    256
}
fn default_sample_rate() -> u32 {
    24000
}
fn default_num_codebooks() -> usize {
    8
}
fn default_streaming_chunk_size() -> usize {
    4800
} // 200ms at 24kHz
fn default_use_metal() -> bool {
    cfg!(target_os = "macos")
}
fn default_num_threads() -> usize {
    std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(4)
        .min(8)
}
fn default_enable_preemption() -> bool {
    true
}
fn default_enable_adaptive_batching() -> bool {
    true
}
fn default_min_tokens_per_step() -> usize {
    128
}
fn default_target_ttft_ms() -> f64 {
    250.0
}
fn default_target_decode_tpot_ms() -> f64 {
    40.0
}
fn default_priority_aging_ms() -> u64 {
    1_000
}

impl Default for EngineCoreConfig {
    fn default() -> Self {
        Self {
            model_type: ModelType::default(),
            models_dir: default_models_dir(),
            max_batch_size: default_max_batch_size(),
            max_seq_len: default_max_seq_len(),
            max_tokens_per_step: default_max_tokens_per_step(),
            block_size: default_block_size(),
            max_blocks: default_max_blocks(),
            scheduling_policy: SchedulingPolicy::default(),
            enable_chunked_prefill: default_chunked_prefill(),
            chunked_prefill_threshold: default_chunked_prefill_threshold(),
            sample_rate: default_sample_rate(),
            num_codebooks: default_num_codebooks(),
            streaming_chunk_size: default_streaming_chunk_size(),
            use_metal: default_use_metal(),
            num_threads: default_num_threads(),
            enable_preemption: default_enable_preemption(),
            enable_adaptive_batching: default_enable_adaptive_batching(),
            min_tokens_per_step: default_min_tokens_per_step(),
            target_ttft_ms: default_target_ttft_ms(),
            target_decode_tpot_ms: default_target_decode_tpot_ms(),
            priority_aging_ms: default_priority_aging_ms(),
        }
    }
}

impl EngineCoreConfig {
    /// Create config for Qwen3-TTS model
    pub fn for_qwen3_tts() -> Self {
        Self {
            model_type: ModelType::Qwen3TTS,
            sample_rate: 24000,
            num_codebooks: 8,
            ..Default::default()
        }
    }

    /// Calculate memory required for KV cache
    pub fn kv_cache_memory_bytes(&self) -> usize {
        // Approximate: 2 (K+V) * block_size * hidden_dim * num_layers * dtype_size
        // Using typical values for audio models
        let hidden_dim = 1024;
        let num_layers = 24;
        let dtype_bytes = 2; // float16

        self.max_blocks * self.block_size * hidden_dim * num_layers * 2 * dtype_bytes
    }
}
