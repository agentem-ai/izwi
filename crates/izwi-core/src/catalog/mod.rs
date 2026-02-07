//! Model catalog surface aligned with inference-engine style organization.
//!
//! This module is the canonical place for model metadata, capabilities,
//! and identifier parsing. Legacy `crate::model` paths are still available
//! for backward compatibility.

mod variant;

pub use crate::model::{
    DownloadProgress, ModelDownloader, ModelInfo, ModelManager, ModelStatus, ModelVariant,
    ModelWeights,
};
pub use variant::{
    parse_chat_model_variant, parse_model_variant, parse_tts_model_variant,
    resolve_asr_model_variant, InferenceBackendHint, ModelFamily, ModelTask,
    ParseModelVariantError,
};
