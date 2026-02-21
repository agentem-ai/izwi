use crate::error::Result;
use crate::model::ModelVariant;
use crate::runtime::service::RuntimeService;

impl RuntimeService {
    /// Unload a model from memory.
    pub async fn unload_model(&self, variant: ModelVariant) -> Result<()> {
        if variant.is_asr() || variant.is_forced_aligner() {
            self.model_registry.unload_asr(variant).await;
        } else if variant.is_diarization() {
            self.model_registry.unload_diarization(variant).await;
        } else if variant.is_chat() {
            self.model_registry.unload_chat(variant).await;
        } else if variant.is_lfm2() {
            self.model_registry.unload_lfm2(variant).await;
            self.clear_active_tts_variant().await;
        } else if variant.is_voxtral() {
            self.model_registry.unload_voxtral(variant).await;
        } else if variant.is_tts() {
            let mut model_guard = self.tts_model.write().await;
            *model_guard = None;
            drop(model_guard);

            self.clear_active_tts_variant().await;
        } else if variant.is_tokenizer() {
            let mut tokenizer_guard = self.tokenizer.write().await;
            *tokenizer_guard = None;
        }

        self.model_manager.unload_model(variant).await
    }
}
