//! Configuration for Voxtral Realtime model.

use serde::Deserialize;

use crate::models::qwen3::Qwen3Config;

/// Main Voxtral configuration
#[derive(Debug, Clone, Deserialize)]
pub struct VoxtralConfig {
    pub model_type: String,
    pub text_config: MistralConfig,
    pub audio_config: AudioEncoderConfig,
    pub downsample_factor: usize,
    pub block_pool_size: usize,
    pub num_delay_tokens: usize,
    #[serde(default)]
    pub is_causal: bool,
    #[serde(default = "default_transcription_delay_ms")]
    pub transcription_delay_ms: f32,
    #[serde(default = "default_streaming_look_ahead_ms")]
    pub streaming_look_ahead_ms: f32,
    #[serde(default = "default_streaming_look_back_ms")]
    pub streaming_look_back_ms: f32,
    #[serde(default = "default_frame_rate")]
    pub frame_rate: f32,
}

fn default_transcription_delay_ms() -> f32 {
    400.0
}

fn default_streaming_look_ahead_ms() -> f32 {
    80.0
}

fn default_streaming_look_back_ms() -> f32 {
    160.0
}

fn default_frame_rate() -> f32 {
    12.5
}

/// Mistral text model configuration
#[derive(Debug, Clone, Deserialize)]
pub struct MistralConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f32,
    pub tie_word_embeddings: bool,
    pub sliding_window: usize,
    #[serde(default)]
    pub use_sliding_window: bool,
}

/// Whisper-based audio encoder configuration
#[derive(Debug, Clone, Deserialize)]
pub struct AudioEncoderConfig {
    pub d_model: usize,
    pub encoder_layers: usize,
    pub encoder_attention_heads: usize,
    pub encoder_ffn_dim: usize,
    pub num_mel_bins: usize,
    pub max_source_positions: usize,
    pub window_size: usize,
    pub hop_length: usize,
    pub sampling_rate: usize,
    #[serde(default)]
    pub is_causal: bool,
    pub conv1_kernel_size: usize,
    pub conv1_stride: usize,
    pub conv2_kernel_size: usize,
    pub conv2_stride: usize,
    #[serde(default = "default_global_log_mel_max")]
    pub global_log_mel_max: Option<f32>,
}

fn default_global_log_mel_max() -> Option<f32> {
    None
}

impl From<MistralConfig> for Qwen3Config {
    fn from(cfg: MistralConfig) -> Self {
        Self {
            hidden_size: cfg.hidden_size,
            intermediate_size: cfg.intermediate_size,
            num_attention_heads: cfg.num_attention_heads,
            num_hidden_layers: cfg.num_hidden_layers,
            num_key_value_heads: cfg.num_key_value_heads,
            head_dim: None,
            rms_norm_eps: cfg.rms_norm_eps,
            rope_theta: cfg.rope_theta as f64,
            vocab_size: cfg.vocab_size,
            rope_scaling: None,
        }
    }
}
