//! Native model implementations and registry.

pub mod batched_attention;
pub mod chat_types;
pub mod device;
pub mod gemma3_chat;
pub mod gguf_loader;
pub mod metal_memory;
pub mod mlx_compat;
pub mod qwen3;
pub mod qwen3_asr;
pub mod qwen3_chat;
pub mod qwen3_tts;
pub mod registry;
pub mod voxtral;
pub mod voxtral_lm;

pub use device::{DeviceProfile, DeviceSelector};
pub use gguf_loader::{
    is_gguf_file, load_model_weights, var_builder_from_gguf, GgufLoader, GgufModelInfo,
};
pub use registry::ModelRegistry;
