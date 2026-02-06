//! Native model implementations and registry.

pub mod device;
pub mod qwen3;
pub mod qwen3_asr;
pub mod qwen3_tts;
pub mod registry;
pub mod voxtral;
pub mod voxtral_lm;

pub use device::{DeviceProfile, DeviceSelector};
pub use registry::ModelRegistry;
