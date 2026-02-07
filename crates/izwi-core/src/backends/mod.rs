//! Backend routing primitives.
//!
//! Mirrors modern inference engine patterns where request/runtime layers are
//! decoupled from execution backends (native Candle, MLX, etc).

use crate::catalog::{InferenceBackendHint, ModelVariant};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionBackend {
    CandleNative,
    MlxNative,
}

#[derive(Debug, Clone)]
pub struct BackendPlan {
    pub backend: ExecutionBackend,
    pub reason: String,
}

#[derive(Debug, Clone)]
pub struct BackendRouter {
    mlx_runtime_enabled: bool,
    default_backend: ExecutionBackend,
}

impl Default for BackendRouter {
    fn default() -> Self {
        Self {
            mlx_runtime_enabled: false,
            default_backend: ExecutionBackend::CandleNative,
        }
    }
}

impl BackendRouter {
    pub fn from_env() -> Self {
        let enabled = std::env::var("IZWI_ENABLE_MLX_RUNTIME")
            .ok()
            .map(|raw| {
                let normalized = raw.trim().to_ascii_lowercase();
                matches!(normalized.as_str(), "1" | "true" | "yes" | "on")
            })
            .unwrap_or(false);

        Self {
            mlx_runtime_enabled: enabled,
            ..Self::default()
        }
    }

    pub fn select(&self, variant: ModelVariant) -> BackendPlan {
        match (variant.backend_hint(), self.mlx_runtime_enabled) {
            (InferenceBackendHint::MlxCandidate, true) => BackendPlan {
                backend: ExecutionBackend::MlxNative,
                reason: format!(
                    "{} is an mlx-community artifact and MLX runtime is enabled",
                    variant.dir_name()
                ),
            },
            (InferenceBackendHint::MlxCandidate, false) => BackendPlan {
                backend: self.default_backend,
                reason: format!(
                    "{} is MLX-compatible, but MLX runtime is disabled; using native backend",
                    variant.dir_name()
                ),
            },
            (InferenceBackendHint::CandleNative, _) => BackendPlan {
                backend: self.default_backend,
                reason: format!(
                    "{} targets the native Candle execution path",
                    variant.dir_name()
                ),
            },
        }
    }
}
