//! Native model implementations and registry.

pub mod device;
pub mod qwen3;
pub mod qwen3_asr;
pub mod registry;

pub use device::{DeviceProfile, DeviceSelector};
pub use registry::ModelRegistry;
