//! Native model implementations and registry.

pub mod architectures;
pub mod registry;
pub mod shared;

// Backward-compatible aliases for the legacy flat module layout.
pub use architectures::gemma3::chat as gemma3_chat;
pub use architectures::lfm2::audio as lfm2_audio;
pub use architectures::parakeet::asr as parakeet_asr;
pub use architectures::qwen3::asr as qwen3_asr;
pub use architectures::qwen3::chat as qwen3_chat;
pub use architectures::qwen3::core as qwen3;
pub use architectures::qwen3::tts as qwen3_tts;
pub use architectures::voxtral::lm as voxtral_lm;
pub use architectures::voxtral::realtime as voxtral;
pub use shared::attention::batched as batched_attention;
pub use shared::chat as chat_types;
pub use shared::device;
pub use shared::memory::metal as metal_memory;
pub use shared::weights::gguf as gguf_loader;
pub use shared::weights::mlx as mlx_compat;

pub use registry::ModelRegistry;
pub use shared::device::{DeviceProfile, DeviceSelector};
pub use shared::weights::gguf::{
    is_gguf_file, load_model_weights, var_builder_from_gguf, GgufLoader, GgufModelInfo,
};
