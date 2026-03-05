//! Backend routing, device probing, and execution policy primitives.

pub mod capabilities;
pub mod device;
pub mod model_io;
pub mod policy;
pub mod router;
pub mod types;

pub use capabilities::BackendCapabilities;
pub use device::{DeviceCapabilities, DeviceKind, DeviceProfile, DeviceSelector};
pub use model_io::{
    auto_gguf_mmap_for_backend, gguf_mmap_enabled, gguf_mmap_mode_from_env,
    resolve_gguf_mmap_mode, GgufMmapMode,
};
pub use policy::{can_parallelize_requests, default_dtype_for_device, kv_dtype_bytes};
pub use router::{BackendPlan, BackendRouter};
pub use types::{BackendKind, BackendPreference, BackendSelectionSource, ExecutionBackend};
