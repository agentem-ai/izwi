//! Device selection for native inference.

use candle_core::{DType, Device};
use tracing::info;

use crate::error::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceKind {
    Cuda,
    Metal,
    Cpu,
}

impl DeviceKind {
    pub fn is_cpu(&self) -> bool {
        matches!(self, DeviceKind::Cpu)
    }
}

#[derive(Debug, Clone)]
pub struct DeviceProfile {
    pub device: Device,
    pub kind: DeviceKind,
}

impl DeviceProfile {
    pub fn select_dtype(&self, requested: Option<&str>) -> DType {
        match requested.unwrap_or("") {
            "bfloat16" | "bf16" => match self.kind {
                DeviceKind::Cpu => DType::F32,
                DeviceKind::Metal => DType::F16,
                DeviceKind::Cuda => DType::BF16,
            },
            "float16" | "f16" => match self.kind {
                DeviceKind::Cpu => DType::F32,
                _ => DType::F16,
            },
            "float32" | "f32" => DType::F32,
            _ => match self.kind {
                DeviceKind::Cpu => DType::F32,
                DeviceKind::Metal => DType::F16,
                DeviceKind::Cuda => DType::BF16,
            },
        }
    }
}

pub struct DeviceSelector;

impl DeviceSelector {
    pub fn detect() -> Result<DeviceProfile> {
        if cfg!(target_os = "macos") {
            if let Ok(device) = Device::metal_if_available(0) {
                info!("Using Metal device for inference");
                return Ok(DeviceProfile {
                    device,
                    kind: DeviceKind::Metal,
                });
            }
        } else if let Ok(device) = Device::cuda_if_available(0) {
            info!("Using CUDA device for inference");
            return Ok(DeviceProfile {
                device,
                kind: DeviceKind::Cuda,
            });
        }

        if let Ok(device) = Device::metal_if_available(0) {
            info!("Using Metal device for inference");
            return Ok(DeviceProfile {
                device,
                kind: DeviceKind::Metal,
            });
        }

        info!("Falling back to CPU for inference");
        Ok(DeviceProfile {
            device: Device::Cpu,
            kind: DeviceKind::Cpu,
        })
    }

    pub fn detect_with_preference(preference: Option<&str>) -> Result<DeviceProfile> {
        match preference.unwrap_or("") {
            "cuda" => {
                if cfg!(target_os = "macos") {
                    return Self::detect();
                }
                Device::cuda_if_available(0)
                    .map(|device| DeviceProfile {
                        device,
                        kind: DeviceKind::Cuda,
                    })
                    .or_else(|_| Self::detect())
            }
            "metal" | "mps" => Device::metal_if_available(0)
                .map(|device| DeviceProfile {
                    device,
                    kind: DeviceKind::Metal,
                })
                .or_else(|_| Self::detect()),
            "cpu" => Ok(DeviceProfile {
                device: Device::Cpu,
                kind: DeviceKind::Cpu,
            }),
            _ => Self::detect(),
        }
    }
}
