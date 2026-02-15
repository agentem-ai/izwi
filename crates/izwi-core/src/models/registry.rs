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
use super::lfm2_audio::Lfm2AudioModel;
use super::parakeet_asr::ParakeetAsrModel;
use super::qwen3_asr::{
    AsrDecodeState as Qwen3AsrDecodeState, AsrDecodeStep as Qwen3AsrDecodeStep, Qwen3AsrModel,
};
use super::qwen3_chat::{
    ChatDecodeState as Qwen3ChatDecodeState, ChatGenerationOutput, Qwen3ChatModel,
};
use super::sortformer_diarization::SortformerDiarizerModel;
use super::voxtral::VoxtralRealtimeModel;
use crate::runtime::{DiarizationConfig, DiarizationResult};

pub enum NativeAsrModel {
    Qwen3(Qwen3AsrModel),
    Parakeet(ParakeetAsrModel),
}

pub enum NativeAsrDecodeState {
    Qwen3(Qwen3AsrDecodeState),
}

pub enum NativeDiarizationModel {
    Sortformer(SortformerDiarizerModel),
}

#[derive(Debug, Clone)]
pub struct NativeAsrDecodeStep {
    pub delta: String,
    pub text: String,
    pub tokens_generated: usize,
    pub finished: bool,
}

impl NativeAsrModel {
    pub fn transcribe_with_callback(
        &self,
        audio: &[f32],
        sample_rate: u32,
        language: Option<&str>,
        on_delta: &mut dyn FnMut(&str),
    ) -> Result<String> {
        match self {
            Self::Qwen3(model) => {
                model.transcribe_with_callback(audio, sample_rate, language, on_delta)
            }
            Self::Parakeet(model) => {
                model.transcribe_with_callback(audio, sample_rate, language, on_delta)
            }
        }
    }

    pub fn force_align(
        &self,
        audio: &[f32],
        sample_rate: u32,
        reference_text: &str,
    ) -> Result<Vec<(String, u32, u32)>> {
        match self {
            Self::Qwen3(model) => model.force_align(audio, sample_rate, reference_text),
            Self::Parakeet(_) => Err(Error::InvalidInput(
                "Forced alignment is only available for Qwen3-ForcedAligner models".to_string(),
            )),
        }
    }

    pub fn supports_incremental_decode(&self) -> bool {
        matches!(self, Self::Qwen3(_))
    }

    pub fn start_decode_state(
        &self,
        audio: &[f32],
        sample_rate: u32,
        language: Option<&str>,
        max_new_tokens: usize,
    ) -> Result<NativeAsrDecodeState> {
        match self {
            Self::Qwen3(model) => Ok(NativeAsrDecodeState::Qwen3(model.start_decode(
                audio,
                sample_rate,
                language,
                max_new_tokens,
            )?)),
            Self::Parakeet(_) => Err(Error::InvalidInput(
                "Incremental decode state is not available for this ASR model".to_string(),
            )),
        }
    }

    pub fn decode_step(&self, state: &mut NativeAsrDecodeState) -> Result<NativeAsrDecodeStep> {
        match (self, state) {
            (Self::Qwen3(model), NativeAsrDecodeState::Qwen3(state)) => {
                let step: Qwen3AsrDecodeStep = model.decode_step(state)?;
                Ok(NativeAsrDecodeStep {
                    delta: step.delta,
                    text: step.text,
                    tokens_generated: step.tokens_generated,
                    finished: step.finished,
                })
            }
            _ => Err(Error::InvalidInput(
                "ASR decode state does not match loaded ASR model".to_string(),
            )),
        }
    }
}

impl NativeDiarizationModel {
    pub fn diarize(
        &self,
        audio: &[f32],
        sample_rate: u32,
        config: &DiarizationConfig,
    ) -> Result<DiarizationResult> {
        match self {
            Self::Sortformer(model) => model.diarize(audio, sample_rate, config),
        }
    }
}

pub enum NativeChatModel {
    Qwen3(Qwen3ChatModel),
    Gemma3(Gemma3ChatModel),
}

pub enum NativeChatDecodeState {
    Qwen3(Qwen3ChatDecodeState),
}

#[derive(Debug, Clone)]
pub struct NativeChatDecodeStep {
    pub delta: String,
    pub text: String,
    pub tokens_generated: usize,
    pub finished: bool,
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

    pub fn supports_incremental_decode(&self) -> bool {
        matches!(self, Self::Qwen3(_))
    }

    pub fn start_decode_state(
        &self,
        messages: &[ChatMessage],
        max_new_tokens: usize,
    ) -> Result<NativeChatDecodeState> {
        match self {
            Self::Qwen3(model) => Ok(NativeChatDecodeState::Qwen3(
                model.start_decode(messages, max_new_tokens)?,
            )),
            Self::Gemma3(_) => Err(Error::InvalidInput(
                "Incremental decode state is not available for this chat model".to_string(),
            )),
        }
    }

    pub fn decode_step(&self, state: &mut NativeChatDecodeState) -> Result<NativeChatDecodeStep> {
        match (self, state) {
            (Self::Qwen3(model), NativeChatDecodeState::Qwen3(state)) => {
                let step = model.decode_step(state)?;
                Ok(NativeChatDecodeStep {
                    delta: step.delta,
                    text: step.text,
                    tokens_generated: step.tokens_generated,
                    finished: step.finished,
                })
            }
            _ => Err(Error::InvalidInput(
                "Chat decode state does not match loaded chat model".to_string(),
            )),
        }
    }
}

#[derive(Clone)]
pub struct ModelRegistry {
    models_dir: PathBuf,
    device: DeviceProfile,
    asr_models: Arc<RwLock<HashMap<ModelVariant, Arc<OnceCell<Arc<NativeAsrModel>>>>>>,
    diarization_models:
        Arc<RwLock<HashMap<ModelVariant, Arc<OnceCell<Arc<NativeDiarizationModel>>>>>>,
    chat_models: Arc<RwLock<HashMap<ModelVariant, Arc<OnceCell<Arc<NativeChatModel>>>>>>,
    voxtral_models: Arc<RwLock<HashMap<ModelVariant, Arc<OnceCell<Arc<VoxtralRealtimeModel>>>>>>,
    lfm2_models: Arc<RwLock<HashMap<ModelVariant, Arc<OnceCell<Arc<Lfm2AudioModel>>>>>>,
}

impl ModelRegistry {
    pub fn new(models_dir: PathBuf, device: DeviceProfile) -> Self {
        Self {
            models_dir,
            device,
            asr_models: Arc::new(RwLock::new(HashMap::new())),
            diarization_models: Arc::new(RwLock::new(HashMap::new())),
            chat_models: Arc::new(RwLock::new(HashMap::new())),
            voxtral_models: Arc::new(RwLock::new(HashMap::new())),
            lfm2_models: Arc::new(RwLock::new(HashMap::new())),
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
    ) -> Result<Arc<NativeAsrModel>> {
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
                    tokio::task::spawn_blocking(move || {
                        let model = if variant.is_parakeet() {
                            NativeAsrModel::Parakeet(ParakeetAsrModel::load(&model_dir, variant)?)
                        } else {
                            NativeAsrModel::Qwen3(Qwen3AsrModel::load(&model_dir, device)?)
                        };
                        Ok::<NativeAsrModel, Error>(model)
                    })
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

    pub async fn load_diarization(
        &self,
        variant: ModelVariant,
        model_dir: &Path,
    ) -> Result<Arc<NativeDiarizationModel>> {
        if !variant.is_diarization() {
            return Err(Error::InvalidInput(format!(
                "Model variant {variant} is not a diarization model"
            )));
        }

        let cell = {
            let mut guard = self.diarization_models.write().await;
            guard
                .entry(variant)
                .or_insert_with(|| Arc::new(OnceCell::new()))
                .clone()
        };

        info!("Loading native diarization model {variant} from {model_dir:?}");

        let model = cell
            .get_or_try_init({
                let model_dir = model_dir.to_path_buf();
                move || async move {
                    tokio::task::spawn_blocking(move || {
                        let model = match variant {
                            ModelVariant::DiarStreamingSortformer4SpkV21 => {
                                NativeDiarizationModel::Sortformer(SortformerDiarizerModel::load(
                                    &model_dir, variant,
                                )?)
                            }
                            _ => {
                                return Err(Error::InvalidInput(format!(
                                    "Unsupported diarization model variant: {variant}"
                                )));
                            }
                        };
                        Ok::<NativeDiarizationModel, Error>(model)
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

    pub async fn load_lfm2(
        &self,
        variant: ModelVariant,
        model_dir: &Path,
    ) -> Result<Arc<Lfm2AudioModel>> {
        if !variant.is_lfm2() {
            return Err(Error::InvalidInput(format!(
                "Model variant {variant} is not an LFM2 model"
            )));
        }

        let cell = {
            let mut guard = self.lfm2_models.write().await;
            guard
                .entry(variant)
                .or_insert_with(|| Arc::new(OnceCell::new()))
                .clone()
        };

        info!("Loading LFM2 model {variant} from {model_dir:?}");

        let model = cell
            .get_or_try_init({
                let model_dir = model_dir.to_path_buf();
                let device = self.device.clone();
                move || async move {
                    tokio::task::spawn_blocking(move || Lfm2AudioModel::load(&model_dir, device))
                        .await
                        .map_err(|e| Error::ModelLoadError(e.to_string()))?
                        .map(Arc::new)
                }
            })
            .await?;

        Ok(model.clone())
    }

    pub async fn get_asr(&self, variant: ModelVariant) -> Option<Arc<NativeAsrModel>> {
        let guard = self.asr_models.read().await;
        guard.get(&variant).and_then(|cell| cell.get().cloned())
    }

    pub fn try_get_asr(&self, variant: ModelVariant) -> Option<Arc<NativeAsrModel>> {
        let guard = self.asr_models.try_read().ok()?;
        guard.get(&variant).and_then(|cell| cell.get().cloned())
    }

    pub async fn get_diarization(
        &self,
        variant: ModelVariant,
    ) -> Option<Arc<NativeDiarizationModel>> {
        let guard = self.diarization_models.read().await;
        guard.get(&variant).and_then(|cell| cell.get().cloned())
    }

    pub fn try_get_diarization(
        &self,
        variant: ModelVariant,
    ) -> Option<Arc<NativeDiarizationModel>> {
        let guard = self.diarization_models.try_read().ok()?;
        guard.get(&variant).and_then(|cell| cell.get().cloned())
    }

    pub async fn get_chat(&self, variant: ModelVariant) -> Option<Arc<NativeChatModel>> {
        let guard = self.chat_models.read().await;
        guard.get(&variant).and_then(|cell| cell.get().cloned())
    }

    pub fn try_get_chat(&self, variant: ModelVariant) -> Option<Arc<NativeChatModel>> {
        let guard = self.chat_models.try_read().ok()?;
        guard.get(&variant).and_then(|cell| cell.get().cloned())
    }

    pub async fn get_voxtral(&self, variant: ModelVariant) -> Option<Arc<VoxtralRealtimeModel>> {
        let guard = self.voxtral_models.read().await;
        guard.get(&variant).and_then(|cell| cell.get().cloned())
    }

    pub fn try_get_voxtral(&self, variant: ModelVariant) -> Option<Arc<VoxtralRealtimeModel>> {
        let guard = self.voxtral_models.try_read().ok()?;
        guard.get(&variant).and_then(|cell| cell.get().cloned())
    }

    pub async fn get_lfm2(&self, variant: ModelVariant) -> Option<Arc<Lfm2AudioModel>> {
        let guard = self.lfm2_models.read().await;
        guard.get(&variant).and_then(|cell| cell.get().cloned())
    }

    pub fn try_get_lfm2(&self, variant: ModelVariant) -> Option<Arc<Lfm2AudioModel>> {
        let guard = self.lfm2_models.try_read().ok()?;
        guard.get(&variant).and_then(|cell| cell.get().cloned())
    }

    pub async fn unload_asr(&self, variant: ModelVariant) {
        let mut guard = self.asr_models.write().await;
        guard.remove(&variant);
    }

    pub async fn unload_diarization(&self, variant: ModelVariant) {
        let mut guard = self.diarization_models.write().await;
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

    pub async fn unload_lfm2(&self, variant: ModelVariant) {
        let mut guard = self.lfm2_models.write().await;
        guard.remove(&variant);
    }
}
