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
    fn try_metal() -> Option<DeviceProfile> {
        let device = std::panic::catch_unwind(|| Device::metal_if_available(0))
            .ok()?
            .ok()?;
        if device.is_metal() {
            Some(DeviceProfile {
                device,
                kind: DeviceKind::Metal,
            })
        } else {
            None
        }
    }

    fn try_cuda() -> Option<DeviceProfile> {
        let device = std::panic::catch_unwind(|| Device::cuda_if_available(0))
            .ok()?
            .ok()?;
        if device.is_cuda() {
            Some(DeviceProfile {
                device,
                kind: DeviceKind::Cuda,
            })
        } else {
            None
        }
    }

    pub fn detect() -> Result<DeviceProfile> {
        if cfg!(target_os = "macos") {
            if let Some(profile) = Self::try_metal() {
                info!("Using Metal device for inference");
                return Ok(profile);
            }
        } else if let Some(profile) = Self::try_cuda() {
            info!("Using CUDA device for inference");
            return Ok(profile);
        }

        if let Some(profile) = Self::try_metal() {
            info!("Using Metal device for inference");
            return Ok(profile);
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
                if let Some(profile) = Self::try_cuda() {
                    Ok(profile)
                } else {
                    Self::detect()
                }
            }
            "metal" | "mps" => {
                if let Some(profile) = Self::try_metal() {
                    Ok(profile)
                } else {
                    Self::detect()
                }
            }
            "cpu" => Ok(DeviceProfile {
                device: Device::Cpu,
                kind: DeviceKind::Cpu,
            }),
            _ => Self::detect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_with_cpu_preference_returns_cpu() {
        let profile = DeviceSelector::detect_with_preference(Some("cpu")).unwrap();
        assert_eq!(profile.kind, DeviceKind::Cpu);
        assert!(profile.device.is_cpu());
    }

    #[test]
    fn test_detect_kind_matches_device() {
        let profile = DeviceSelector::detect().unwrap();
        match profile.kind {
            DeviceKind::Cpu => assert!(profile.device.is_cpu()),
            DeviceKind::Metal => assert!(profile.device.is_metal()),
            DeviceKind::Cuda => assert!(profile.device.is_cuda()),
        }
    }
}
