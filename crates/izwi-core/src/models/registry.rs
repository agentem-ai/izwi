//! Model registry to ensure models are loaded once and shared across the app.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::{OnceCell, RwLock};
use tracing::info;

use crate::error::{Error, Result};
use crate::model::ModelVariant;

use super::chat_types::ChatMessage;
use super::device::DeviceProfile;
use super::gemma3_chat::Gemma3ChatModel;
use super::qwen3_asr::Qwen3AsrModel;
use super::qwen3_chat::{ChatGenerationOutput, Qwen3ChatModel};
use super::voxtral::VoxtralRealtimeModel;

pub enum NativeChatModel {
    Qwen3(Qwen3ChatModel),
    Gemma3(Gemma3ChatModel),
}

impl NativeChatModel {
    pub fn generate(
        &self,
        messages: &[ChatMessage],
        max_new_tokens: usize,
    ) -> Result<ChatGenerationOutput> {
        match self {
            Self::Qwen3(model) => model.generate(messages, max_new_tokens),
            Self::Gemma3(model) => {
                let output = model.generate(messages, max_new_tokens)?;
                Ok(ChatGenerationOutput {
                    text: output.text,
                    tokens_generated: output.tokens_generated,
                })
            }
        }
    }

    pub fn generate_with_callback(
        &self,
        messages: &[ChatMessage],
        max_new_tokens: usize,
        on_delta: &mut dyn FnMut(&str),
    ) -> Result<ChatGenerationOutput> {
        match self {
            Self::Qwen3(model) => model.generate_with_callback(messages, max_new_tokens, on_delta),
            Self::Gemma3(model) => {
                let output = model.generate_with_callback(messages, max_new_tokens, on_delta)?;
                Ok(ChatGenerationOutput {
                    text: output.text,
                    tokens_generated: output.tokens_generated,
                })
            }
        }
    }
}

#[derive(Clone)]
pub struct ModelRegistry {
    models_dir: PathBuf,
    device: DeviceProfile,
    asr_models: Arc<RwLock<HashMap<ModelVariant, Arc<OnceCell<Arc<Qwen3AsrModel>>>>>>,
    chat_models: Arc<RwLock<HashMap<ModelVariant, Arc<OnceCell<Arc<NativeChatModel>>>>>>,
    voxtral_models: Arc<RwLock<HashMap<ModelVariant, Arc<OnceCell<Arc<VoxtralRealtimeModel>>>>>>,
}

impl ModelRegistry {
    pub fn new(models_dir: PathBuf, device: DeviceProfile) -> Self {
        Self {
            models_dir,
            device,
            asr_models: Arc::new(RwLock::new(HashMap::new())),
            chat_models: Arc::new(RwLock::new(HashMap::new())),
            voxtral_models: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub fn device(&self) -> &DeviceProfile {
        &self.device
    }

    pub fn models_dir(&self) -> &Path {
        &self.models_dir
    }

    pub async fn load_asr(
        &self,
        variant: ModelVariant,
        model_dir: &Path,
    ) -> Result<Arc<Qwen3AsrModel>> {
        if !variant.is_asr() && !variant.is_forced_aligner() {
            return Err(Error::InvalidInput(format!(
                "Model variant {variant} is not an ASR or ForcedAligner model"
            )));
        }

        let cell = {
            let mut guard = self.asr_models.write().await;
            guard
                .entry(variant)
                .or_insert_with(|| Arc::new(OnceCell::new()))
                .clone()
        };

        info!("Loading native ASR/ForcedAligner model {variant} from {model_dir:?}");

        let model = cell
            .get_or_try_init({
                let model_dir = model_dir.to_path_buf();
                let device = self.device.clone();
                move || async move {
                    tokio::task::spawn_blocking(move || Qwen3AsrModel::load(&model_dir, device))
                        .await
                        .map_err(|e| Error::ModelLoadError(e.to_string()))?
                        .map(Arc::new)
                }
            })
            .await?;

        Ok(model.clone())
    }

    pub async fn load_chat(
        &self,
        variant: ModelVariant,
        model_dir: &Path,
    ) -> Result<Arc<NativeChatModel>> {
        if !variant.is_chat() {
            return Err(Error::InvalidInput(format!(
                "Model variant {variant} is not a chat model"
            )));
        }

        let cell = {
            let mut guard = self.chat_models.write().await;
            guard
                .entry(variant)
                .or_insert_with(|| Arc::new(OnceCell::new()))
                .clone()
        };

        info!("Loading native chat model {variant} from {model_dir:?}");

        let model = cell
            .get_or_try_init({
                let model_dir = model_dir.to_path_buf();
                let device = self.device.clone();
                move || async move {
                    tokio::task::spawn_blocking(move || {
                        let model = match variant {
                            ModelVariant::Qwen306B4Bit => {
                                NativeChatModel::Qwen3(Qwen3ChatModel::load(&model_dir, device)?)
                            }
                            ModelVariant::Gemma31BIt | ModelVariant::Gemma34BIt => {
                                NativeChatModel::Gemma3(Gemma3ChatModel::load(
                                    &model_dir, variant, device,
                                )?)
                            }
                            _ => {
                                return Err(Error::InvalidInput(format!(
                                    "Unsupported chat model variant: {variant}"
                                )));
                            }
                        };

                        Ok::<NativeChatModel, Error>(model)
                    })
                    .await
                    .map_err(|e| Error::ModelLoadError(e.to_string()))?
                    .map(Arc::new)
                }
            })
            .await?;

        Ok(model.clone())
    }

    pub async fn load_voxtral(
        &self,
        variant: ModelVariant,
        model_dir: &Path,
    ) -> Result<Arc<VoxtralRealtimeModel>> {
        if !variant.is_voxtral() {
            return Err(Error::InvalidInput(format!(
                "Model variant {variant} is not a Voxtral model"
            )));
        }

        let cell = {
            let mut guard = self.voxtral_models.write().await;
            guard
                .entry(variant)
                .or_insert_with(|| Arc::new(OnceCell::new()))
                .clone()
        };

        info!("Loading native Voxtral model {variant} from {model_dir:?}");

        let model = cell
            .get_or_try_init({
                let model_dir = model_dir.to_path_buf();
                let device = self.device.clone();
                move || async move {
                    tokio::task::spawn_blocking(move || {
                        VoxtralRealtimeModel::load(&model_dir, device)
                    })
                    .await
                    .map_err(|e| Error::ModelLoadError(e.to_string()))?
                    .map(Arc::new)
                }
            })
            .await?;

        Ok(model.clone())
    }

    pub async fn get_asr(&self, variant: ModelVariant) -> Option<Arc<Qwen3AsrModel>> {
        let guard = self.asr_models.read().await;
        guard.get(&variant).and_then(|cell| cell.get().cloned())
    }

    pub async fn get_chat(&self, variant: ModelVariant) -> Option<Arc<NativeChatModel>> {
        let guard = self.chat_models.read().await;
        guard.get(&variant).and_then(|cell| cell.get().cloned())
    }

    pub async fn get_voxtral(&self, variant: ModelVariant) -> Option<Arc<VoxtralRealtimeModel>> {
        let guard = self.voxtral_models.read().await;
        guard.get(&variant).and_then(|cell| cell.get().cloned())
    }

    pub async fn unload_asr(&self, variant: ModelVariant) {
        let mut guard = self.asr_models.write().await;
        guard.remove(&variant);
    }

    pub async fn unload_chat(&self, variant: ModelVariant) {
        let mut guard = self.chat_models.write().await;
        guard.remove(&variant);
    }

    pub async fn unload_voxtral(&self, variant: ModelVariant) {
        let mut guard = self.voxtral_models.write().await;
        guard.remove(&variant);
    }
}
