//! Backend routing primitives.
//!
//! Mirrors modern inference engine patterns where request/runtime layers are
//! decoupled from execution backends (native Candle, MLX, etc).

use crate::catalog::{InferenceBackendHint, ModelVariant};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionBackend {
    CandleNative,
    CandleMetal,
    MlxNative,
}

#[derive(Debug, Clone)]
pub struct BackendPlan {
    pub backend: ExecutionBackend,
    pub reason: String,
}

#[derive(Debug, Clone)]
pub struct BackendRouter {
    default_backend: ExecutionBackend,
}

impl Default for BackendRouter {
    fn default() -> Self {
        Self {
            default_backend: ExecutionBackend::CandleNative,
        }
    }
}

impl BackendRouter {
    pub fn from_env_with_default(default_backend: ExecutionBackend) -> Self {
        Self { default_backend }
    }

    pub fn from_env() -> Self {
        Self::from_env_with_default(ExecutionBackend::CandleNative)
    }

    pub fn select(&self, variant: ModelVariant) -> BackendPlan {
        let default_desc = match self.default_backend {
            ExecutionBackend::CandleMetal => "Metal backend",
            ExecutionBackend::CandleNative => "native backend",
            ExecutionBackend::MlxNative => "MLX runtime",
        };

        match variant.backend_hint() {
            InferenceBackendHint::MlxCandidate => BackendPlan {
                backend: self.default_backend,
                reason: format!(
                    "{} is MLX-compatible; using {}",
                    variant.dir_name(),
                    default_desc
                ),
            },
            InferenceBackendHint::CandleNative => BackendPlan {
                backend: self.default_backend,
                reason: format!(
                    "{} targets the native Candle execution path ({})",
                    variant.dir_name(),
                    default_desc
                ),
            },
        }
    }
}
