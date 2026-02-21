//! Native model implementations and registry.
//!
//! Canonical module layout:
//! - `architectures::*` for model-family implementations
//! - `shared::*` for reusable runtime/model utilities
//! - `registry` for loaded native model handles

pub mod architectures;
pub mod registry;
pub mod shared;

pub use registry::ModelRegistry;
pub use shared::device::{DeviceProfile, DeviceSelector};
pub use shared::weights::gguf::{
    is_gguf_file, load_model_weights, var_builder_from_gguf, GgufLoader, GgufModelInfo,
};

/// Temporary compatibility exports for legacy flat module paths.
#[doc(hidden)]
pub mod compat {
    pub use super::architectures::gemma3::chat as gemma3_chat;
    pub use super::architectures::lfm2::audio as lfm2_audio;
    pub use super::architectures::parakeet::asr as parakeet_asr;
    pub use super::architectures::qwen3::asr as qwen3_asr;
    pub use super::architectures::qwen3::chat as qwen3_chat;
    pub use super::architectures::qwen3::core as qwen3;
    pub use super::architectures::qwen3::tts as qwen3_tts;
    pub use super::architectures::sortformer::diarization as sortformer_diarization;
    pub use super::architectures::voxtral::lm as voxtral_lm;
    pub use super::architectures::voxtral::realtime as voxtral;
    pub use super::shared::attention::batched as batched_attention;
    pub use super::shared::chat as chat_types;
    pub use super::shared::device;
    pub use super::shared::memory::metal as metal_memory;
    pub use super::shared::weights::gguf as gguf_loader;
    pub use super::shared::weights::mlx as mlx_compat;
}
