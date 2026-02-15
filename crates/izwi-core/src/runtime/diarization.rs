//! Diarization runtime methods.

use crate::catalog::resolve_diarization_model_variant;
use crate::error::{Error, Result};
use crate::runtime::audio_io::{base64_decode, decode_wav_bytes};
use crate::runtime::service::RuntimeService;
use crate::runtime::types::{DiarizationConfig, DiarizationResult};

impl RuntimeService {
    /// Run speaker diarization over a single audio input.
    pub async fn diarize(
        &self,
        audio_base64: &str,
        model_id: Option<&str>,
        config: &DiarizationConfig,
    ) -> Result<DiarizationResult> {
        let variant = resolve_diarization_model_variant(model_id);
        self.load_model(variant).await?;

        let model = self
            .model_registry
            .get_diarization(variant)
            .await
            .ok_or_else(|| Error::ModelNotFound(variant.to_string()))?;

        let (samples, sample_rate) = decode_wav_bytes(&base64_decode(audio_base64)?)?;
        model.diarize(&samples, sample_rate, config)
    }
}
