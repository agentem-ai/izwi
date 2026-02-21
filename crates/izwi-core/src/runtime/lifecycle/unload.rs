use crate::error::Result;
use crate::model::ModelVariant;
use crate::runtime::lifecycle::families::{resolve_runtime_model_family, RuntimeModelFamily};
use crate::runtime::service::RuntimeService;

impl RuntimeService {
    /// Unload a model from memory.
    pub async fn unload_model(&self, variant: ModelVariant) -> Result<()> {
        match resolve_runtime_model_family(variant) {
            RuntimeModelFamily::Asr => {
                self.model_registry.unload_asr(variant).await;
            }
            RuntimeModelFamily::Diarization => {
                self.model_registry.unload_diarization(variant).await;
            }
            RuntimeModelFamily::Chat => {
                self.model_registry.unload_chat(variant).await;
            }
            RuntimeModelFamily::Lfm2 => {
                self.model_registry.unload_lfm2(variant).await;
                self.clear_active_tts_variant().await;
            }
            RuntimeModelFamily::Voxtral => {
                self.model_registry.unload_voxtral(variant).await;
            }
            RuntimeModelFamily::Tts => {
                let mut model_guard = self.tts_model.write().await;
                *model_guard = None;
                drop(model_guard);
                self.clear_active_tts_variant().await;
            }
            RuntimeModelFamily::Auxiliary => {
                if variant.is_tokenizer() {
                    let mut tokenizer_guard = self.tokenizer.write().await;
                    *tokenizer_guard = None;
                }
            }
        }

        self.model_manager.unload_model(variant).await
    }
}
