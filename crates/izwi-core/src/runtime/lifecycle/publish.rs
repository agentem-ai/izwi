use crate::error::Result;
use crate::runtime::lifecycle::instantiate::InstantiatedModelLoad;
use crate::runtime::service::RuntimeService;

impl RuntimeService {
    pub(super) async fn publish_loaded_model(
        &self,
        instantiated: InstantiatedModelLoad,
    ) -> Result<()> {
        match instantiated {
            InstantiatedModelLoad::Asr { variant }
            | InstantiatedModelLoad::Diarization { variant }
            | InstantiatedModelLoad::Chat { variant }
            | InstantiatedModelLoad::Voxtral { variant }
            | InstantiatedModelLoad::TtsAlreadyLoaded { variant } => {
                self.model_manager.mark_loaded(variant).await;
            }
            InstantiatedModelLoad::Lfm2 {
                variant,
                model_path,
            } => {
                // LFM2 owns active TTS routing and does not use the legacy Qwen slot.
                let mut tts_guard = self.tts_model.write().await;
                *tts_guard = None;
                drop(tts_guard);

                self.set_active_tts_variant(variant, model_path).await;
                self.model_manager.mark_loaded(variant).await;
            }
            InstantiatedModelLoad::TtsLoaded {
                variant,
                model_path,
                model,
            } => {
                let mut model_guard = self.tts_model.write().await;
                *model_guard = Some(model);
                drop(model_guard);

                self.set_active_tts_variant(variant, model_path).await;
                self.model_manager.mark_loaded(variant).await;
            }
            InstantiatedModelLoad::Auxiliary {
                variant,
                model_path,
                tokenizer,
            } => {
                if let Some(tokenizer) = tokenizer {
                    let mut guard = self.tokenizer.write().await;
                    *guard = Some(tokenizer);
                }

                if variant.is_tokenizer() {
                    let mut codec_guard = self.codec.write().await;
                    codec_guard.load_weights(&model_path)?;
                }
            }
        }

        Ok(())
    }
}
