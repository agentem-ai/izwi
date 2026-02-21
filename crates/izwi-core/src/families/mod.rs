//! Compatibility namespace for legacy model-family imports.

pub use crate::models::architectures::gemma3::chat as gemma3_chat;
pub use crate::models::architectures::lfm2::audio as lfm2_audio;
pub use crate::models::architectures::parakeet::asr as parakeet_asr;
pub use crate::models::architectures::qwen3::asr as qwen3_asr;
pub use crate::models::architectures::qwen3::chat as qwen3_chat;
pub use crate::models::architectures::qwen3::core as qwen3;
pub use crate::models::architectures::qwen3::tts as qwen3_tts;
pub use crate::models::architectures::voxtral::lm as voxtral_lm;
pub use crate::models::architectures::voxtral::realtime as voxtral;
pub use crate::models::shared::chat as chat_types;
pub use crate::models::shared::device::{DeviceProfile, DeviceSelector};
pub use crate::models::shared::weights::gguf::{
    is_gguf_file, load_model_weights, var_builder_from_gguf, GgufLoader, GgufModelInfo,
};
pub use crate::models::shared::weights::mlx as mlx_compat;
pub use crate::models::ModelRegistry;
