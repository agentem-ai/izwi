use tracing::info;

use crate::error::{Error, Result};
use crate::model::ModelVariant;
use crate::models::architectures::qwen3::tts::Qwen3TtsModel;
use crate::runtime::lifecycle::families::{resolve_runtime_model_family, RuntimeModelFamily};
use crate::runtime::lifecycle::phases::AcquiredModelLoad;
use crate::runtime::service::RuntimeService;
use crate::tokenizer::Tokenizer;

type TtsLoaderFn =
    fn(&std::path::Path, crate::models::DeviceProfile, usize, &str) -> Result<Qwen3TtsModel>;

struct TtsLoaderRegistration {
    name: &'static str,
    matcher: fn(ModelVariant) -> bool,
    loader: TtsLoaderFn,
}

fn is_qwen_tts_variant(variant: ModelVariant) -> bool {
    variant.is_tts()
}

fn load_qwen_tts_model(
    model_dir: &std::path::Path,
    device: crate::models::DeviceProfile,
    kv_page_size: usize,
    kv_cache_dtype: &str,
) -> Result<Qwen3TtsModel> {
    Qwen3TtsModel::load(model_dir, device, kv_page_size, kv_cache_dtype)
}

const TTS_LOADER_REGISTRY: &[TtsLoaderRegistration] = &[TtsLoaderRegistration {
    name: "qwen3_tts",
    matcher: is_qwen_tts_variant,
    loader: load_qwen_tts_model,
}];

fn resolve_tts_loader_registration(
    variant: ModelVariant,
) -> Option<&'static TtsLoaderRegistration> {
    TTS_LOADER_REGISTRY
        .iter()
        .find(|registration| (registration.matcher)(variant))
}

pub(super) enum InstantiatedPayload {
    None,
    TtsModel(Qwen3TtsModel),
    Tokenizer(Option<Tokenizer>),
}

pub(super) struct InstantiatedModelLoad {
    pub family: RuntimeModelFamily,
    pub variant: ModelVariant,
    pub model_path: std::path::PathBuf,
    pub payload: InstantiatedPayload,
}

impl RuntimeService {
    pub(super) async fn instantiate_model(
        &self,
        acquired: AcquiredModelLoad,
    ) -> Result<InstantiatedModelLoad> {
        let AcquiredModelLoad {
            variant,
            model_path,
        } = acquired;
        let family = resolve_runtime_model_family(variant);

        let payload = match family {
            RuntimeModelFamily::Asr => {
                self.model_registry.load_asr(variant, &model_path).await?;
                InstantiatedPayload::None
            }
            RuntimeModelFamily::Diarization => {
                self.model_registry
                    .load_diarization(variant, &model_path)
                    .await?;
                InstantiatedPayload::None
            }
            RuntimeModelFamily::Chat => {
                self.model_registry.load_chat(variant, &model_path).await?;
                InstantiatedPayload::None
            }
            RuntimeModelFamily::Lfm2 => {
                self.model_registry.load_lfm2(variant, &model_path).await?;
                InstantiatedPayload::None
            }
            RuntimeModelFamily::Voxtral => {
                self.model_registry
                    .load_voxtral(variant, &model_path)
                    .await?;
                InstantiatedPayload::None
            }
            RuntimeModelFamily::Tts => {
                let registration = resolve_tts_loader_registration(variant).ok_or_else(|| {
                    Error::InvalidInput(format!("Unsupported TTS model variant: {variant}"))
                })?;
                if self.is_tts_model_already_loaded(&model_path).await {
                    InstantiatedPayload::None
                } else {
                    info!(
                        "Loading native TTS model {variant} ({}) from {:?}",
                        registration.name, model_path
                    );
                    let model = (registration.loader)(
                        &model_path,
                        self.device.clone(),
                        self.config.kv_page_size,
                        &self.config.kv_cache_dtype,
                    )?;
                    InstantiatedPayload::TtsModel(model)
                }
            }
            RuntimeModelFamily::Auxiliary => {
                let tokenizer = match Tokenizer::from_path(&model_path) {
                    Ok(tokenizer) => Some(tokenizer),
                    Err(err) => {
                        tracing::warn!("Failed to load tokenizer from model directory: {}", err);
                        None
                    }
                };
                InstantiatedPayload::Tokenizer(tokenizer)
            }
        };

        Ok(InstantiatedModelLoad {
            family,
            variant,
            model_path,
            payload,
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
