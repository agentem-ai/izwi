use tracing::info;

use crate::error::Result;
use crate::model::ModelVariant;
use crate::models::architectures::qwen3::tts::Qwen3TtsModel;
use crate::runtime::lifecycle::phases::AcquiredModelLoad;
use crate::runtime::service::RuntimeService;
use crate::tokenizer::Tokenizer;

pub(super) enum InstantiatedModelLoad {
    Asr {
        variant: ModelVariant,
    },
    Diarization {
        variant: ModelVariant,
    },
    Chat {
        variant: ModelVariant,
    },
    Lfm2 {
        variant: ModelVariant,
        model_path: std::path::PathBuf,
    },
    Voxtral {
        variant: ModelVariant,
    },
    TtsLoaded {
        variant: ModelVariant,
        model_path: std::path::PathBuf,
        model: Qwen3TtsModel,
    },
    TtsAlreadyLoaded {
        variant: ModelVariant,
    },
    Auxiliary {
        variant: ModelVariant,
        model_path: std::path::PathBuf,
        tokenizer: Option<Tokenizer>,
    },
}

impl RuntimeService {
    pub(super) async fn instantiate_model(
        &self,
        acquired: AcquiredModelLoad,
    ) -> Result<InstantiatedModelLoad> {
        let variant = acquired.variant;

        if variant.is_asr() || variant.is_forced_aligner() {
            self.model_registry
                .load_asr(variant, &acquired.model_path)
                .await?;
            return Ok(InstantiatedModelLoad::Asr { variant });
        }

        if variant.is_diarization() {
            self.model_registry
                .load_diarization(variant, &acquired.model_path)
                .await?;
            return Ok(InstantiatedModelLoad::Diarization { variant });
        }

        if variant.is_chat() {
            self.model_registry
                .load_chat(variant, &acquired.model_path)
                .await?;
            return Ok(InstantiatedModelLoad::Chat { variant });
        }

        if variant.is_lfm2() {
            self.model_registry
                .load_lfm2(variant, &acquired.model_path)
                .await?;
            return Ok(InstantiatedModelLoad::Lfm2 {
                variant,
                model_path: acquired.model_path,
            });
        }

        if variant.is_voxtral() {
            self.model_registry
                .load_voxtral(variant, &acquired.model_path)
                .await?;
            return Ok(InstantiatedModelLoad::Voxtral { variant });
        }

        if variant.is_tts() {
            if self.is_tts_model_already_loaded(&acquired.model_path).await {
                return Ok(InstantiatedModelLoad::TtsAlreadyLoaded { variant });
            }

            info!("Loading native TTS model from {:?}", acquired.model_path);
            let model = Qwen3TtsModel::load(
                &acquired.model_path,
                self.device.clone(),
                self.config.kv_page_size,
                &self.config.kv_cache_dtype,
            )?;

            return Ok(InstantiatedModelLoad::TtsLoaded {
                variant,
                model_path: acquired.model_path,
                model,
            });
        }

        let tokenizer = match Tokenizer::from_path(&acquired.model_path) {
            Ok(tokenizer) => Some(tokenizer),
            Err(err) => {
                tracing::warn!("Failed to load tokenizer from model directory: {}", err);
                None
            }
        };

        Ok(InstantiatedModelLoad::Auxiliary {
            variant,
            model_path: acquired.model_path,
            tokenizer,
        })
    }

    async fn is_tts_model_already_loaded(&self, model_path: &std::path::Path) -> bool {
        let loaded_path = self.loaded_model_path.read().await;
        let tts_model = self.tts_model.read().await;

        tts_model.is_some()
            && loaded_path
                .as_ref()
                .map(|p| p.as_path() == model_path)
                .unwrap_or(false)
    }
}
