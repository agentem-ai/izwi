//! Model executor - handles forward pass execution.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::thread;
use tokio::sync::{mpsc, RwLock};
use tracing::{debug, error, info};

use super::config::EngineCoreConfig;
use super::output::StreamingOutput;
use super::request::EngineCoreRequest;
use super::scheduler::ScheduledRequest;
use super::types::{AudioOutput, ModelType, TaskType};
use crate::error::{Error, Result};
use crate::model::ModelVariant;
use crate::models::chat_types::ChatMessage;
use crate::models::lfm2_audio::{
    lfm2_tts_voice_prompt, SpeechToSpeechDecodeState, TtsDecodeState, LFM2_DEFAULT_S2S_PROMPT,
};
use crate::models::qwen3_tts::{
    Qwen3TtsModel, SpeakerReference, TtsDecodeState as QwenTtsDecodeState, TtsGenerationParams,
    TtsStreamingConfig,
};
use crate::models::registry::{NativeAsrDecodeState, NativeChatDecodeState};
use crate::models::DeviceSelector;
use crate::models::ModelRegistry;

fn panic_payload_to_string(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(msg) = payload.downcast_ref::<&str>() {
        return (*msg).to_string();
    }
    if let Some(msg) = payload.downcast_ref::<String>() {
        return msg.clone();
    }
    "unknown panic payload".to_string()
}

/// Configuration for the model executor.
#[derive(Clone)]
pub struct WorkerConfig {
    /// Model type
    pub model_type: ModelType,
    /// Path to models directory
    pub models_dir: PathBuf,
    /// Device to use (cpu, mps, cuda)
    pub device: String,
    /// Data type (float32, float16, bfloat16)
    pub dtype: String,
    /// KV cache storage dtype hint (e.g. float16, int8).
    pub kv_cache_dtype: String,
    /// Number of threads
    pub num_threads: usize,
    /// Decode-time KV cache page size.
    pub kv_page_size: usize,
    /// Optional shared model handle provided by higher-level runtime.
    pub shared_tts_model: Option<Arc<RwLock<Option<Qwen3TtsModel>>>>,
    /// Optional shared model registry for non-TTS tasks (ASR/Chat/LFM2).
    pub model_registry: Option<Arc<ModelRegistry>>,
}

impl std::fmt::Debug for WorkerConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WorkerConfig")
            .field("model_type", &self.model_type)
            .field("models_dir", &self.models_dir)
            .field("device", &self.device)
            .field("dtype", &self.dtype)
            .field("kv_cache_dtype", &self.kv_cache_dtype)
            .field("num_threads", &self.num_threads)
            .field("kv_page_size", &self.kv_page_size)
            .field(
                "shared_tts_model",
                &self.shared_tts_model.as_ref().map(|_| "<shared>"),
            )
            .field(
                "model_registry",
                &self.model_registry.as_ref().map(|_| "<shared>"),
            )
            .finish()
    }
}

impl Default for WorkerConfig {
    fn default() -> Self {
        Self {
            model_type: ModelType::Qwen3TTS,
            models_dir: dirs::data_local_dir()
                .unwrap_or_else(|| PathBuf::from("."))
                .join("izwi")
                .join("models"),
            device: if cfg!(target_os = "macos") {
                "mps".to_string()
            } else {
                "cpu".to_string()
            },
            dtype: "float32".to_string(),
            kv_cache_dtype: "float16".to_string(),
            num_threads: 4,
            kv_page_size: 64,
            shared_tts_model: None,
            model_registry: None,
        }
    }
}

impl From<&EngineCoreConfig> for WorkerConfig {
    fn from(config: &EngineCoreConfig) -> Self {
        Self {
            model_type: config.model_type,
            models_dir: config.models_dir.clone(),
            device: if config.use_metal {
                "mps".to_string()
            } else {
                "cpu".to_string()
            },
            dtype: "float32".to_string(),
            kv_cache_dtype: config.kv_cache_dtype.clone(),
            num_threads: config.num_threads,
            kv_page_size: config.block_size.max(1),
            shared_tts_model: None,
            model_registry: None,
        }
    }
}

/// Output from the executor after a forward pass.
#[derive(Debug, Clone)]
pub struct ExecutorOutput {
    /// Request ID
    pub request_id: String,
    /// Generated audio samples
    pub audio: Option<AudioOutput>,
    /// Generated text (for ASR/chat)
    pub text: Option<String>,
    /// Number of tokens processed
    pub tokens_processed: usize,
    /// Number of tokens generated
    pub tokens_generated: usize,
    /// Whether generation is complete
    pub finished: bool,
    /// Error if any
    pub error: Option<String>,
}

impl ExecutorOutput {
    pub fn error(request_id: String, error: impl Into<String>) -> Self {
        Self {
            request_id,
            audio: None,
            text: None,
            tokens_processed: 0,
            tokens_generated: 0,
            finished: true,
            error: Some(error.into()),
        }
    }
}

/// Model executor trait - abstracts the model inference backend.
pub trait ModelExecutor: Send + Sync {
    /// Execute prefill pass for newly admitted or in-progress prefill requests.
    fn execute_prefill(
        &self,
        requests: &[&EngineCoreRequest],
        scheduled: &[ScheduledRequest],
    ) -> Result<Vec<ExecutorOutput>>;

    /// Execute decode pass for running requests.
    fn execute_decode(
        &self,
        requests: &[&EngineCoreRequest],
        scheduled: &[ScheduledRequest],
    ) -> Result<Vec<ExecutorOutput>>;

    /// Execute forward pass for scheduled requests.
    /// Compatibility helper that executes decode and prefill paths.
    fn execute(
        &self,
        requests: &[&EngineCoreRequest],
        scheduled: &[ScheduledRequest],
    ) -> Result<Vec<ExecutorOutput>> {
        let mut decode = Vec::new();
        let mut prefill = Vec::new();
        for req in scheduled {
            if req.is_prefill {
                prefill.push(req.clone());
            } else {
                decode.push(req.clone());
            }
        }

        let mut outputs = Vec::new();
        if !decode.is_empty() {
            outputs.extend(self.execute_decode(requests, &decode)?);
        }
        if !prefill.is_empty() {
            outputs.extend(self.execute_prefill(requests, &prefill)?);
        }
        Ok(outputs)
    }

    /// Check if the executor is ready.
    fn is_ready(&self) -> bool;

    /// Initialize the executor (load models, etc.)
    fn initialize(&mut self) -> Result<()>;

    /// Shutdown the executor.
    fn shutdown(&mut self) -> Result<()>;

    /// Cleanup transient per-request state held by the executor backend.
    fn cleanup_request(&self, _request_id: &str) {}
}

pub struct NativeExecutor {
    config: WorkerConfig,
    initialized: bool,
    loaded_tts_model: Option<Arc<Qwen3TtsModel>>,
    chat_decode_states: Mutex<HashMap<String, ActiveChatDecode>>,
    asr_decode_states: Mutex<HashMap<String, ActiveAsrDecode>>,
    qwen_tts_decode_states: Mutex<HashMap<String, ActiveQwenTtsDecode>>,
    lfm2_tts_decode_states: Mutex<HashMap<String, ActiveLfm2TtsDecode>>,
    speech_to_speech_decode_states: Mutex<HashMap<String, ActiveSpeechToSpeechDecode>>,
}

struct ActiveChatDecode {
    variant: ModelVariant,
    state: NativeChatDecodeState,
    prompt_accounted: bool,
    last_tokens_generated: usize,
    stream_sequence: usize,
}

struct ActiveAsrDecode {
    variant: ModelVariant,
    state: NativeAsrDecodeState,
    prompt_accounted: bool,
    last_tokens_generated: usize,
    stream_sequence: usize,
    input_sample_rate: u32,
    input_sample_count: usize,
}

struct ActiveSpeechToSpeechDecode {
    variant: ModelVariant,
    state: SpeechToSpeechDecodeState,
    prompt_accounted: bool,
    last_tokens_generated: usize,
    stream_sequence: usize,
    audio_samples_accum: Vec<f32>,
}

struct ActiveQwenTtsDecode {
    variant: Option<ModelVariant>,
    state: QwenTtsDecodeState,
    prompt_accounted: bool,
    last_frames_generated: usize,
    stream_sequence: usize,
    audio_samples_accum: Vec<f32>,
}

struct ActiveLfm2TtsDecode {
    variant: ModelVariant,
    state: TtsDecodeState,
    prompt_accounted: bool,
    last_tokens_generated: usize,
    stream_sequence: usize,
    audio_samples_accum: Vec<f32>,
}

impl NativeExecutor {
    /// Create a new native executor.
    pub fn new(config: WorkerConfig) -> Self {
        Self {
            config,
            initialized: false,
            loaded_tts_model: None,
            chat_decode_states: Mutex::new(HashMap::new()),
            asr_decode_states: Mutex::new(HashMap::new()),
            qwen_tts_decode_states: Mutex::new(HashMap::new()),
            lfm2_tts_decode_states: Mutex::new(HashMap::new()),
            speech_to_speech_decode_states: Mutex::new(HashMap::new()),
        }
    }

    fn with_model<T>(&self, f: impl FnOnce(&Qwen3TtsModel) -> Result<T>) -> Result<T> {
        if let Some(shared_model) = &self.config.shared_tts_model {
            let guard = shared_model.try_read().map_err(|_| {
                Error::InferenceError("Shared TTS model is busy (try again)".to_string())
            })?;
            let model = guard
                .as_ref()
                .ok_or_else(|| Error::InferenceError("No TTS model loaded".to_string()))?;
            return f(model);
        }

        let model = self
            .loaded_tts_model
            .as_deref()
            .ok_or_else(|| Error::InferenceError("Executor model not initialized".to_string()))?;
        f(model)
    }

    fn with_registry<T>(&self, f: impl FnOnce(&ModelRegistry) -> Result<T>) -> Result<T> {
        let registry =
            self.config.model_registry.as_ref().ok_or_else(|| {
                Error::InferenceError("Model registry is not configured".to_string())
            })?;
        f(registry)
    }

    fn run_blocking<T>(f: impl FnOnce() -> Result<T>) -> Result<T> {
        let unwind_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f));

        match unwind_result {
            Ok(result) => result,
            Err(payload) => {
                let message = panic_payload_to_string(payload.as_ref());
                error!("Model execution panicked: {message}");
                Err(Error::InferenceError(format!(
                    "Model execution panicked: {message}"
                )))
            }
        }
    }

    fn stream_sender(
        request: &EngineCoreRequest,
    ) -> Option<mpsc::UnboundedSender<StreamingOutput>> {
        if request.streaming {
            request.streaming_tx.clone()
        } else {
            None
        }
    }

    fn stream_text(
        tx: &mpsc::UnboundedSender<StreamingOutput>,
        request_id: &str,
        sequence: &mut usize,
        text: String,
    ) -> Result<()> {
        tx.send(StreamingOutput {
            request_id: request_id.to_string(),
            sequence: *sequence,
            samples: Vec::new(),
            sample_rate: 0,
            is_final: false,
            text: Some(text),
            stats: None,
        })
        .map_err(|_| Error::InferenceError("Streaming output channel closed".to_string()))?;
        *sequence += 1;
        Ok(())
    }

    fn stream_audio(
        tx: &mpsc::UnboundedSender<StreamingOutput>,
        request_id: &str,
        sequence: &mut usize,
        samples: Vec<f32>,
        sample_rate: u32,
        is_final: bool,
    ) -> Result<()> {
        tx.send(StreamingOutput {
            request_id: request_id.to_string(),
            sequence: *sequence,
            samples,
            sample_rate,
            is_final,
            text: None,
            stats: None,
        })
        .map_err(|_| Error::InferenceError("Streaming output channel closed".to_string()))?;
        *sequence += 1;
        Ok(())
    }

    fn stream_final_marker(
        tx: &mpsc::UnboundedSender<StreamingOutput>,
        request_id: &str,
        sequence: &mut usize,
    ) -> Result<()> {
        Self::stream_audio(tx, request_id, sequence, Vec::new(), 0, true)
    }

    fn find_request<'a>(
        requests: &'a [&EngineCoreRequest],
        scheduled: &ScheduledRequest,
    ) -> Option<&'a EngineCoreRequest> {
        requests
            .iter()
            .copied()
            .find(|r| r.id == scheduled.request_id)
    }

    fn to_tts_params(request: &EngineCoreRequest) -> TtsGenerationParams {
        TtsGenerationParams {
            temperature: request.params.temperature.max(0.0),
            top_p: request.params.top_p.clamp(0.0, 1.0),
            top_k: if request.params.top_k == 0 {
                50
            } else {
                request.params.top_k
            },
            repetition_penalty: request.params.repetition_penalty.max(1.0),
            max_frames: if request.params.max_tokens == 0 {
                512
            } else {
                request.params.max_tokens.clamp(16, 8192)
            },
        }
    }

    fn reference_from_request(request: &EngineCoreRequest) -> Result<Option<SpeakerReference>> {
        if request.reference_audio.is_none() && request.reference_text.is_none() {
            return Ok(None);
        }

        let ref_audio = request.reference_audio.as_deref().ok_or_else(|| {
            Error::InvalidInput(
                "reference_audio and reference_text must both be provided".to_string(),
            )
        })?;
        let ref_text = request.reference_text.as_deref().ok_or_else(|| {
            Error::InvalidInput(
                "reference_audio and reference_text must both be provided".to_string(),
            )
        })?;
        if ref_text.trim().is_empty() {
            return Err(Error::InvalidInput(
                "reference_text cannot be empty".to_string(),
            ));
        }

        let (audio_samples, sample_rate) = decode_audio_base64_with_rate(ref_audio)?;
        Ok(Some(SpeakerReference {
            audio_samples,
            text: ref_text.to_string(),
            sample_rate,
        }))
    }

    fn qwen_tts_request(
        &self,
        request: &EngineCoreRequest,
        scheduled: &ScheduledRequest,
    ) -> Result<ExecutorOutput> {
        let stream_tx = Self::stream_sender(request);
        let variant = request.model_variant;
        let params = Self::to_tts_params(request);
        let language = request.language.as_deref();

        self.with_model(|model| {
            let mut active_state = {
                let mut guard = self.qwen_tts_decode_states.lock().map_err(|_| {
                    Error::InferenceError("Qwen TTS decode state mutex poisoned".to_string())
                })?;
                guard.remove(&request.id)
            };

            if active_state
                .as_ref()
                .map(|state| state.variant != variant)
                .unwrap_or(false)
            {
                active_state = None;
            }

            let mut active_state = if let Some(state) = active_state {
                state
            } else {
                let text = request
                    .text
                    .as_deref()
                    .ok_or_else(|| Error::InvalidInput("TTS request missing text".to_string()))?;
                let available_speakers = model.available_speakers();
                let requested_speaker = request
                    .params
                    .speaker
                    .as_deref()
                    .or(request.params.voice.as_deref())
                    .filter(|s| !s.trim().is_empty());
                let reference = Self::reference_from_request(request)?;

                let decode_state = if let Some(reference) = reference {
                    Self::run_blocking(|| {
                        model.start_decode_with_voice_clone_params(
                            text,
                            &reference,
                            language,
                            &params,
                            TtsStreamingConfig::default(),
                        )
                    })?
                } else if available_speakers.is_empty() {
                    Self::run_blocking(|| {
                        model.start_decode_with_text_params(
                            text,
                            language,
                            request.voice_description.as_deref(),
                            &params,
                            TtsStreamingConfig::default(),
                        )
                    })?
                } else {
                    let speaker_to_use =
                        requested_speaker.unwrap_or_else(|| available_speakers[0].as_str());
                    Self::run_blocking(|| {
                        model.start_decode_with_speaker_params(
                            text,
                            speaker_to_use,
                            language,
                            request.voice_description.as_deref(),
                            &params,
                            TtsStreamingConfig::default(),
                        )
                    })?
                };

                ActiveQwenTtsDecode {
                    variant,
                    state: decode_state,
                    prompt_accounted: false,
                    last_frames_generated: 0,
                    stream_sequence: 0,
                    audio_samples_accum: Vec::new(),
                }
            };

            let step = Self::run_blocking(|| model.tts_decode_step(&mut active_state.state))?;
            let step_tokens_generated = step
                .frames_generated
                .saturating_sub(active_state.last_frames_generated);
            active_state.last_frames_generated = step.frames_generated;

            let mut tokens_processed = scheduled.num_tokens.max(1);
            if !active_state.prompt_accounted {
                active_state.prompt_accounted = true;
                tokens_processed = tokens_processed.saturating_add(request.num_prompt_tokens());
            }

            if !step.samples.is_empty() {
                active_state
                    .audio_samples_accum
                    .extend_from_slice(&step.samples);
                if let Some(tx) = stream_tx.as_ref() {
                    Self::stream_audio(
                        tx,
                        &request.id,
                        &mut active_state.stream_sequence,
                        step.samples.clone(),
                        24_000,
                        false,
                    )?;
                }
            }

            if step.finished {
                if let Some(tx) = stream_tx.as_ref() {
                    Self::stream_final_marker(tx, &request.id, &mut active_state.stream_sequence)?;
                }
            }

            let finished_samples = if step.finished {
                active_state.audio_samples_accum.clone()
            } else {
                Vec::new()
            };

            if !step.finished {
                let mut guard = self.qwen_tts_decode_states.lock().map_err(|_| {
                    Error::InferenceError("Qwen TTS decode state mutex poisoned".to_string())
                })?;
                guard.insert(request.id.clone(), active_state);
            }

            Ok(ExecutorOutput {
                request_id: request.id.clone(),
                audio: Some(AudioOutput::new(finished_samples, 24_000)),
                text: None,
                tokens_processed,
                tokens_generated: step_tokens_generated,
                finished: step.finished,
                error: None,
            })
        })
    }

    fn resolve_variant(request: &EngineCoreRequest) -> Result<ModelVariant> {
        request.model_variant.ok_or_else(|| {
            Error::InvalidInput(format!(
                "Request {} is missing model variant routing information",
                request.id
            ))
        })
    }

    fn transcribe_request(
        &self,
        request: &EngineCoreRequest,
        scheduled: &ScheduledRequest,
    ) -> Result<ExecutorOutput> {
        let variant = Self::resolve_variant(request)?;
        let language = request.language.as_deref();
        let stream_tx = Self::stream_sender(request);

        if let Some(tx) = stream_tx.as_ref() {
            if !variant.is_voxtral() && !variant.is_lfm2() {
                let model = self.with_registry(|registry| {
                    registry.try_get_asr(variant).ok_or_else(|| {
                        Error::ModelNotFound(format!("ASR model {variant} is not loaded"))
                    })
                })?;

                if model.supports_incremental_decode() {
                    let mut active_state = {
                        let mut guard = self.asr_decode_states.lock().map_err(|_| {
                            Error::InferenceError("ASR decode state mutex poisoned".to_string())
                        })?;
                        guard.remove(&request.id)
                    };

                    if active_state
                        .as_ref()
                        .map(|state| state.variant != variant)
                        .unwrap_or(false)
                    {
                        active_state = None;
                    }

                    let mut active_state = if let Some(state) = active_state {
                        state
                    } else {
                        let audio_b64 = request.audio_input.as_deref().ok_or_else(|| {
                            Error::InvalidInput("ASR request missing audio input".to_string())
                        })?;
                        let (samples, sample_rate) = decode_audio_base64_with_rate(audio_b64)?;
                        let max_new_tokens = request.params.max_tokens.clamp(1, 1024);
                        let decode_state = Self::run_blocking(|| {
                            model.start_decode_state(
                                &samples,
                                sample_rate,
                                language,
                                max_new_tokens,
                            )
                        })?;
                        ActiveAsrDecode {
                            variant,
                            state: decode_state,
                            prompt_accounted: false,
                            last_tokens_generated: 0,
                            stream_sequence: 0,
                            input_sample_rate: sample_rate,
                            input_sample_count: samples.len(),
                        }
                    };

                    let step = Self::run_blocking(|| model.decode_step(&mut active_state.state))?;
                    let step_tokens_generated = step
                        .tokens_generated
                        .saturating_sub(active_state.last_tokens_generated);
                    active_state.last_tokens_generated = step.tokens_generated;

                    let mut tokens_processed = scheduled.num_tokens.max(1);
                    if !active_state.prompt_accounted {
                        active_state.prompt_accounted = true;
                        tokens_processed =
                            tokens_processed.saturating_add(request.num_prompt_tokens());
                    }

                    if !step.delta.is_empty() {
                        Self::stream_text(
                            tx,
                            &request.id,
                            &mut active_state.stream_sequence,
                            step.delta.clone(),
                        )?;
                    }
                    if step.finished {
                        Self::stream_final_marker(
                            tx,
                            &request.id,
                            &mut active_state.stream_sequence,
                        )?;
                    }

                    let input_sample_rate = active_state.input_sample_rate;
                    let input_sample_count = active_state.input_sample_count;

                    if !step.finished {
                        let mut guard = self.asr_decode_states.lock().map_err(|_| {
                            Error::InferenceError("ASR decode state mutex poisoned".to_string())
                        })?;
                        guard.insert(request.id.clone(), active_state);
                    }

                    return Ok(ExecutorOutput {
                        request_id: request.id.clone(),
                        audio: Some(AudioOutput {
                            samples: Vec::new(),
                            sample_rate: input_sample_rate,
                            duration_secs: if input_sample_rate > 0 {
                                input_sample_count as f32 / input_sample_rate as f32
                            } else {
                                0.0
                            },
                        }),
                        text: Some(step.text),
                        tokens_processed,
                        tokens_generated: step_tokens_generated,
                        finished: step.finished,
                        error: None,
                    });
                }
            }
        }

        let audio_b64 = request
            .audio_input
            .as_deref()
            .ok_or_else(|| Error::InvalidInput("ASR request missing audio input".to_string()))?;
        let (samples, sample_rate) = decode_audio_base64_with_rate(audio_b64)?;
        let samples_len = samples.len();

        let text = Self::run_blocking(|| {
            let mut sequence = 0usize;
            if variant.is_voxtral() {
                let model = self.with_registry(|registry| {
                    registry.try_get_voxtral(variant).ok_or_else(|| {
                        Error::ModelNotFound(format!(
                            "Voxtral model {variant} is not loaded in registry"
                        ))
                    })
                })?;
                if let Some(tx) = stream_tx.as_ref() {
                    let mut stream_err: Option<Error> = None;
                    let mut emit = |delta: &str| {
                        if stream_err.is_none() {
                            if let Err(err) =
                                Self::stream_text(tx, &request.id, &mut sequence, delta.to_string())
                            {
                                stream_err = Some(err);
                            }
                        }
                    };
                    let text = model.transcribe_with_callback(
                        &samples,
                        sample_rate,
                        language,
                        &mut emit,
                    )?;
                    if let Some(err) = stream_err {
                        return Err(err);
                    }
                    Self::stream_final_marker(tx, &request.id, &mut sequence)?;
                    return Ok(text);
                }
                return model.transcribe(&samples, sample_rate, language);
            }

            if variant.is_lfm2() {
                let model = self.with_registry(|registry| {
                    registry.try_get_lfm2(variant).ok_or_else(|| {
                        Error::ModelNotFound(format!("LFM2 model {variant} is not loaded"))
                    })
                })?;
                if let Some(tx) = stream_tx.as_ref() {
                    let mut stream_err: Option<Error> = None;
                    let mut emit = |delta: &str| {
                        if stream_err.is_none() {
                            if let Err(err) =
                                Self::stream_text(tx, &request.id, &mut sequence, delta.to_string())
                            {
                                stream_err = Some(err);
                            }
                        }
                    };
                    let text = model.transcribe_with_callback(
                        &samples,
                        sample_rate,
                        language,
                        &mut emit,
                    )?;
                    if let Some(err) = stream_err {
                        return Err(err);
                    }
                    Self::stream_final_marker(tx, &request.id, &mut sequence)?;
                    return Ok(text);
                }
                let mut sink = |_delta: &str| {};
                return model.transcribe_with_callback(&samples, sample_rate, language, &mut sink);
            }

            let model = self.with_registry(|registry| {
                registry.try_get_asr(variant).ok_or_else(|| {
                    Error::ModelNotFound(format!("ASR model {variant} is not loaded"))
                })
            })?;
            if let Some(tx) = stream_tx.as_ref() {
                let mut stream_err: Option<Error> = None;
                let mut emit = |delta: &str| {
                    if stream_err.is_none() {
                        if let Err(err) =
                            Self::stream_text(tx, &request.id, &mut sequence, delta.to_string())
                        {
                            stream_err = Some(err);
                        }
                    }
                };
                let text =
                    model.transcribe_with_callback(&samples, sample_rate, language, &mut emit)?;
                if let Some(err) = stream_err {
                    return Err(err);
                }
                Self::stream_final_marker(tx, &request.id, &mut sequence)?;
                return Ok(text);
            }
            let mut sink = |_delta: &str| {};
            model.transcribe_with_callback(&samples, sample_rate, language, &mut sink)
        })?;

        Ok(ExecutorOutput {
            request_id: request.id.clone(),
            audio: Some(AudioOutput {
                samples: Vec::new(),
                sample_rate,
                duration_secs: samples_len as f32 / sample_rate as f32,
            }),
            text: Some(text),
            tokens_processed: request.num_prompt_tokens(),
            tokens_generated: (samples_len / 256).max(1),
            finished: true,
            error: None,
        })
    }

    fn chat_messages(request: &EngineCoreRequest) -> Result<&[ChatMessage]> {
        request
            .chat_messages
            .as_deref()
            .ok_or_else(|| Error::InvalidInput("Chat request missing messages".to_string()))
    }

    fn chat_request(
        &self,
        request: &EngineCoreRequest,
        scheduled: &ScheduledRequest,
    ) -> Result<ExecutorOutput> {
        let variant = Self::resolve_variant(request)?;
        let messages = Self::chat_messages(request)?;
        let max_new_tokens = request.params.max_tokens.max(1);
        let stream_tx = Self::stream_sender(request);

        let model = self.with_registry(|registry| {
            registry
                .try_get_chat(variant)
                .ok_or_else(|| Error::ModelNotFound(format!("Chat model {variant} is not loaded")))
        })?;

        // Fallback path for chat backends that do not expose incremental decode state.
        if !model.supports_incremental_decode() {
            let output = Self::run_blocking(|| {
                if let Some(tx) = stream_tx.as_ref() {
                    let mut sequence = 0usize;
                    let mut stream_err: Option<Error> = None;
                    let mut emit = |delta: &str| {
                        if stream_err.is_none() {
                            if let Err(err) =
                                Self::stream_text(tx, &request.id, &mut sequence, delta.to_string())
                            {
                                stream_err = Some(err);
                            }
                        }
                    };
                    let output =
                        model.generate_with_callback(messages, max_new_tokens, &mut emit)?;
                    if let Some(err) = stream_err {
                        return Err(err);
                    }
                    Self::stream_final_marker(tx, &request.id, &mut sequence)?;
                    Ok(output)
                } else {
                    model.generate(messages, max_new_tokens)
                }
            })?;

            return Ok(ExecutorOutput {
                request_id: request.id.clone(),
                audio: Some(AudioOutput::empty(24_000)),
                text: Some(output.text),
                tokens_processed: request.num_prompt_tokens(),
                tokens_generated: output.tokens_generated.max(1),
                finished: true,
                error: None,
            });
        }

        let mut active_state = {
            let mut guard = self.chat_decode_states.lock().map_err(|_| {
                Error::InferenceError("Chat decode state mutex poisoned".to_string())
            })?;
            if scheduled.is_prefill {
                // Prefill scheduling can happen after preemption; reset stale state.
                guard.remove(&request.id)
            } else {
                guard.remove(&request.id)
            }
        };

        if active_state
            .as_ref()
            .map(|state| state.variant != variant)
            .unwrap_or(false)
        {
            active_state = None;
        }

        let mut active_state = if let Some(state) = active_state {
            state
        } else {
            let decode_state =
                Self::run_blocking(|| model.start_decode_state(messages, max_new_tokens))?;
            ActiveChatDecode {
                variant,
                state: decode_state,
                prompt_accounted: false,
                last_tokens_generated: 0,
                stream_sequence: 0,
            }
        };

        let step = Self::run_blocking(|| model.decode_step(&mut active_state.state))?;
        let step_tokens_generated = step
            .tokens_generated
            .saturating_sub(active_state.last_tokens_generated);
        active_state.last_tokens_generated = step.tokens_generated;

        let mut tokens_processed = scheduled.num_tokens.max(1);
        if !active_state.prompt_accounted {
            active_state.prompt_accounted = true;
            tokens_processed = tokens_processed.saturating_add(request.num_prompt_tokens());
        }

        if let Some(tx) = stream_tx.as_ref() {
            if !step.delta.is_empty() {
                Self::stream_text(
                    tx,
                    &request.id,
                    &mut active_state.stream_sequence,
                    step.delta.clone(),
                )?;
            }
            if step.finished {
                Self::stream_final_marker(tx, &request.id, &mut active_state.stream_sequence)?;
            }
        }

        if !step.finished {
            let mut guard = self.chat_decode_states.lock().map_err(|_| {
                Error::InferenceError("Chat decode state mutex poisoned".to_string())
            })?;
            guard.insert(request.id.clone(), active_state);
        }

        Ok(ExecutorOutput {
            request_id: request.id.clone(),
            audio: Some(AudioOutput::empty(24_000)),
            text: Some(step.text),
            tokens_processed,
            tokens_generated: step_tokens_generated,
            finished: step.finished,
            error: None,
        })
    }

    fn lfm2_tts_request(
        &self,
        request: &EngineCoreRequest,
        scheduled: &ScheduledRequest,
    ) -> Result<ExecutorOutput> {
        let variant = Self::resolve_variant(request)?;
        let text = request
            .text
            .as_deref()
            .ok_or_else(|| Error::InvalidInput("LFM2 TTS request missing text".to_string()))?;
        if request.reference_audio.is_some() || request.reference_text.is_some() {
            return Err(Error::InvalidInput(
                "LFM2 does not support reference-audio voice cloning".to_string(),
            ));
        }

        let speaker = request
            .params
            .speaker
            .as_deref()
            .or(request.params.voice.as_deref());
        let voice_instruction = request
            .voice_description
            .clone()
            .unwrap_or_else(|| lfm2_tts_voice_prompt(speaker).to_string());

        let using_generic_defaults =
            request.params.top_k == 0 && (request.params.temperature - 0.7).abs() < f32::EPSILON;
        let temperature = if using_generic_defaults {
            0.8
        } else {
            request.params.temperature
        };
        let top_k = if request.params.top_k > 0 {
            request.params.top_k
        } else {
            64
        };
        let max_new_tokens = request.params.max_tokens.max(256);
        let stream_tx = Self::stream_sender(request);

        let model = self.with_registry(|registry| {
            registry
                .try_get_lfm2(variant)
                .ok_or_else(|| Error::ModelNotFound(format!("LFM2 model {variant} is not loaded")))
        })?;

        let mut active_state = {
            let mut guard = self.lfm2_tts_decode_states.lock().map_err(|_| {
                Error::InferenceError("LFM2 TTS decode state mutex poisoned".to_string())
            })?;
            guard.remove(&request.id)
        };

        if active_state
            .as_ref()
            .map(|state| state.variant != variant)
            .unwrap_or(false)
        {
            active_state = None;
        }

        let mut active_state = if let Some(state) = active_state {
            state
        } else {
            let decode_state = Self::run_blocking(|| {
                model.start_tts_decode(
                    text,
                    &voice_instruction,
                    Some(temperature),
                    Some(top_k),
                    max_new_tokens,
                )
            })?;
            ActiveLfm2TtsDecode {
                variant,
                state: decode_state,
                prompt_accounted: false,
                last_tokens_generated: 0,
                stream_sequence: 0,
                audio_samples_accum: Vec::new(),
            }
        };

        let step = Self::run_blocking(|| model.tts_decode_step(&mut active_state.state))?;
        let step_tokens_generated = step
            .tokens_generated
            .saturating_sub(active_state.last_tokens_generated);
        active_state.last_tokens_generated = step.tokens_generated;

        let mut tokens_processed = scheduled.num_tokens.max(1);
        if !active_state.prompt_accounted {
            active_state.prompt_accounted = true;
            tokens_processed = tokens_processed.saturating_add(request.num_prompt_tokens());
        }

        if let Some(tx) = stream_tx.as_ref() {
            if !step.delta.is_empty() {
                Self::stream_text(
                    tx,
                    &request.id,
                    &mut active_state.stream_sequence,
                    step.delta.clone(),
                )?;
            }
        }

        if let Some(frame) = step.audio_frame.as_ref() {
            let chunk_samples = Self::run_blocking(|| model.decode_audio_frame(frame))?;
            if !chunk_samples.is_empty() {
                active_state
                    .audio_samples_accum
                    .extend_from_slice(&chunk_samples);
                if let Some(tx) = stream_tx.as_ref() {
                    Self::stream_audio(
                        tx,
                        &request.id,
                        &mut active_state.stream_sequence,
                        chunk_samples,
                        24_000,
                        false,
                    )?;
                }
            }
        }

        if step.finished {
            if let Some(tx) = stream_tx.as_ref() {
                Self::stream_final_marker(tx, &request.id, &mut active_state.stream_sequence)?;
            }
        }

        let finished_samples = if step.finished {
            active_state.audio_samples_accum.clone()
        } else {
            Vec::new()
        };

        if !step.finished {
            let mut guard = self.lfm2_tts_decode_states.lock().map_err(|_| {
                Error::InferenceError("LFM2 TTS decode state mutex poisoned".to_string())
            })?;
            guard.insert(request.id.clone(), active_state);
        }

        Ok(ExecutorOutput {
            request_id: request.id.clone(),
            audio: Some(AudioOutput::new(finished_samples, 24_000)),
            text: None,
            tokens_processed,
            tokens_generated: step_tokens_generated,
            finished: step.finished,
            error: None,
        })
    }

    fn speech_to_speech_request(
        &self,
        request: &EngineCoreRequest,
        scheduled: &ScheduledRequest,
    ) -> Result<ExecutorOutput> {
        let variant = Self::resolve_variant(request)?;
        let system_prompt = request
            .system_prompt
            .as_deref()
            .unwrap_or(LFM2_DEFAULT_S2S_PROMPT);
        let resolved_temperature = request.params.audio_temperature.unwrap_or(1.0);
        let resolved_top_k = request.params.audio_top_k.unwrap_or_else(|| {
            if request.params.top_k > 0 {
                request.params.top_k
            } else {
                4
            }
        });
        let max_new_tokens = request.params.max_tokens.max(1);
        let stream_tx = Self::stream_sender(request);

        if let Some(tx) = stream_tx.as_ref() {
            let model = self.with_registry(|registry| {
                registry.try_get_lfm2(variant).ok_or_else(|| {
                    Error::ModelNotFound(format!("LFM2 model {variant} is not loaded"))
                })
            })?;

            let mut active_state = {
                let mut guard = self.speech_to_speech_decode_states.lock().map_err(|_| {
                    Error::InferenceError(
                        "Speech-to-speech decode state mutex poisoned".to_string(),
                    )
                })?;
                guard.remove(&request.id)
            };

            if active_state
                .as_ref()
                .map(|state| state.variant != variant)
                .unwrap_or(false)
            {
                active_state = None;
            }

            let mut active_state = if let Some(state) = active_state {
                state
            } else {
                let audio_b64 = request.audio_input.as_deref().ok_or_else(|| {
                    Error::InvalidInput("Speech-to-speech request missing audio input".to_string())
                })?;
                let (samples, sample_rate) = decode_audio_base64_with_rate(audio_b64)?;
                let decode_state = Self::run_blocking(|| {
                    model.start_speech_to_speech_decode(
                        &samples,
                        sample_rate,
                        Some(system_prompt),
                        Some(resolved_temperature),
                        Some(resolved_top_k),
                        max_new_tokens,
                    )
                })?;
                ActiveSpeechToSpeechDecode {
                    variant,
                    state: decode_state,
                    prompt_accounted: false,
                    last_tokens_generated: 0,
                    stream_sequence: 0,
                    audio_samples_accum: Vec::new(),
                }
            };

            let step =
                Self::run_blocking(|| model.speech_to_speech_decode_step(&mut active_state.state))?;
            let step_tokens_generated = step
                .tokens_generated
                .saturating_sub(active_state.last_tokens_generated);
            active_state.last_tokens_generated = step.tokens_generated;

            let mut tokens_processed = scheduled.num_tokens.max(1);
            if !active_state.prompt_accounted {
                active_state.prompt_accounted = true;
                tokens_processed = tokens_processed.saturating_add(request.num_prompt_tokens());
            }

            if !step.delta.is_empty() {
                Self::stream_text(
                    tx,
                    &request.id,
                    &mut active_state.stream_sequence,
                    step.delta.clone(),
                )?;
            }

            if let Some(frame) = step.audio_frame.as_ref() {
                let chunk_samples = Self::run_blocking(|| model.decode_audio_frame(frame))?;
                if !chunk_samples.is_empty() {
                    active_state
                        .audio_samples_accum
                        .extend_from_slice(&chunk_samples);
                    Self::stream_audio(
                        tx,
                        &request.id,
                        &mut active_state.stream_sequence,
                        chunk_samples,
                        24_000,
                        false,
                    )?;
                }
            }

            if step.finished {
                Self::stream_final_marker(tx, &request.id, &mut active_state.stream_sequence)?;
            }

            let finished_samples = if step.finished {
                active_state.audio_samples_accum.clone()
            } else {
                Vec::new()
            };

            if !step.finished {
                let mut guard = self.speech_to_speech_decode_states.lock().map_err(|_| {
                    Error::InferenceError(
                        "Speech-to-speech decode state mutex poisoned".to_string(),
                    )
                })?;
                guard.insert(request.id.clone(), active_state);
            }

            return Ok(ExecutorOutput {
                request_id: request.id.clone(),
                audio: Some(AudioOutput::new(finished_samples, 24_000)),
                text: Some(step.text),
                tokens_processed,
                tokens_generated: step_tokens_generated,
                finished: step.finished,
                error: None,
            });
        }

        let audio_b64 = request.audio_input.as_deref().ok_or_else(|| {
            Error::InvalidInput("Speech-to-speech request missing audio input".to_string())
        })?;
        let (samples, sample_rate) = decode_audio_base64_with_rate(audio_b64)?;
        let model = self.with_registry(|registry| {
            registry
                .try_get_lfm2(variant)
                .ok_or_else(|| Error::ModelNotFound(format!("LFM2 model {variant} is not loaded")))
        })?;
        let (text, output_samples) = Self::run_blocking(|| {
            let mut sink = |_delta: &str| {};
            model.speech_to_speech_with_callback(
                &samples,
                sample_rate,
                Some(system_prompt),
                Some(resolved_temperature),
                Some(resolved_top_k),
                max_new_tokens,
                &mut sink,
            )
        })?;

        Ok(ExecutorOutput {
            request_id: request.id.clone(),
            audio: Some(AudioOutput::new(output_samples.clone(), 24_000)),
            text: Some(text),
            tokens_processed: request.num_prompt_tokens(),
            tokens_generated: (output_samples.len() / 256).max(1),
            finished: true,
            error: None,
        })
    }

    fn execute_single_request(
        &self,
        requests: &[&EngineCoreRequest],
        scheduled_req: &ScheduledRequest,
    ) -> ExecutorOutput {
        let Some(request) = Self::find_request(requests, scheduled_req) else {
            return ExecutorOutput::error(
                scheduled_req.request_id.clone(),
                "Scheduled request not found in batch",
            );
        };

        let result =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| match request.task_type {
                TaskType::TTS => {
                    let variant = request.model_variant;
                    if variant.map(|v| v.is_lfm2()).unwrap_or(false) {
                        self.lfm2_tts_request(request, scheduled_req)
                    } else {
                        self.qwen_tts_request(request, scheduled_req)
                    }
                }
                TaskType::ASR => self.transcribe_request(request, scheduled_req),
                TaskType::Chat => self.chat_request(request, scheduled_req),
                TaskType::SpeechToSpeech => self.speech_to_speech_request(request, scheduled_req),
            }));

        let result = match result {
            Ok(result) => result,
            Err(payload) => {
                let message = panic_payload_to_string(payload.as_ref());
                error!(
                    request_id = %request.id,
                    task = ?request.task_type,
                    "Executor request handling panicked: {message}"
                );
                Err(Error::InferenceError(format!(
                    "Executor request handling panicked: {message}"
                )))
            }
        };

        match result {
            Ok(output) => output,
            Err(err) => ExecutorOutput::error(request.id.clone(), err.to_string()),
        }
    }

    fn can_parallelize_requests(&self, scheduled_len: usize) -> bool {
        if scheduled_len <= 1 || self.config.num_threads <= 1 {
            return false;
        }
        // Keep Metal execution serialized to avoid command-queue contention.
        !self.config.device.eq_ignore_ascii_case("mps")
    }

    fn execute_requests_parallel(
        &self,
        requests: &[&EngineCoreRequest],
        scheduled: &[ScheduledRequest],
    ) -> Result<Vec<ExecutorOutput>> {
        let worker_count = self.config.num_threads.min(scheduled.len()).max(1);
        let mut partitions: Vec<Vec<(usize, ScheduledRequest)>> = vec![Vec::new(); worker_count];
        for (idx, item) in scheduled.iter().enumerate() {
            partitions[idx % worker_count].push((idx, item.clone()));
        }

        let (tx, rx) = std::sync::mpsc::channel::<Vec<(usize, ExecutorOutput)>>();
        thread::scope(|scope| {
            for chunk in partitions {
                if chunk.is_empty() {
                    continue;
                }
                let tx = tx.clone();
                scope.spawn(move || {
                    let mut local = Vec::with_capacity(chunk.len());
                    for (idx, scheduled_req) in chunk {
                        let output = self.execute_single_request(requests, &scheduled_req);
                        local.push((idx, output));
                    }
                    let _ = tx.send(local);
                });
            }
        });
        drop(tx);

        let mut ordered: Vec<Option<ExecutorOutput>> = vec![None; scheduled.len()];
        while let Ok(batch_outputs) = rx.recv() {
            for (idx, output) in batch_outputs {
                if idx < ordered.len() {
                    ordered[idx] = Some(output);
                }
            }
        }

        let outputs = ordered
            .into_iter()
            .enumerate()
            .map(|(idx, output)| {
                output.unwrap_or_else(|| {
                    ExecutorOutput::error(
                        scheduled[idx].request_id.clone(),
                        "Parallel executor worker failed to produce output",
                    )
                })
            })
            .collect();
        Ok(outputs)
    }

    fn execute_requests(
        &self,
        requests: &[&EngineCoreRequest],
        scheduled: &[ScheduledRequest],
    ) -> Result<Vec<ExecutorOutput>> {
        if self.can_parallelize_requests(scheduled.len()) {
            return self.execute_requests_parallel(requests, scheduled);
        }

        let outputs = scheduled
            .iter()
            .map(|scheduled_req| self.execute_single_request(requests, scheduled_req))
            .collect();
        Ok(outputs)
    }
}

impl ModelExecutor for NativeExecutor {
    fn execute_prefill(
        &self,
        requests: &[&EngineCoreRequest],
        scheduled: &[ScheduledRequest],
    ) -> Result<Vec<ExecutorOutput>> {
        if !self.initialized {
            return Err(Error::InferenceError("Executor not initialized".into()));
        }
        self.execute_requests(requests, scheduled)
    }

    fn execute_decode(
        &self,
        requests: &[&EngineCoreRequest],
        scheduled: &[ScheduledRequest],
    ) -> Result<Vec<ExecutorOutput>> {
        if !self.initialized {
            return Err(Error::InferenceError("Executor not initialized".into()));
        }
        self.execute_requests(requests, scheduled)
    }

    fn execute(
        &self,
        requests: &[&EngineCoreRequest],
        scheduled: &[ScheduledRequest],
    ) -> Result<Vec<ExecutorOutput>> {
        if !self.initialized {
            return Err(Error::InferenceError("Executor not initialized".into()));
        }
        self.execute_requests(requests, scheduled)
    }

    fn is_ready(&self) -> bool {
        self.initialized
    }

    fn initialize(&mut self) -> Result<()> {
        info!("Initializing native executor");
        if self.config.shared_tts_model.is_none() {
            let device = if self.config.device.eq_ignore_ascii_case("mps") {
                DeviceSelector::detect_with_preference(Some("metal"))?
            } else {
                DeviceSelector::detect_with_preference(Some("cpu"))?
            };
            let model = Qwen3TtsModel::load(
                &self.config.models_dir,
                device,
                self.config.kv_page_size.max(1),
                &self.config.kv_cache_dtype,
            )?;
            self.loaded_tts_model = Some(Arc::new(model));
            debug!(
                "Native executor loaded TTS model from {:?}",
                self.config.models_dir
            );
        } else {
            debug!("Native executor will use shared TTS model handle");
        }
        self.initialized = true;
        Ok(())
    }

    fn shutdown(&mut self) -> Result<()> {
        info!("Shutting down native executor");
        self.initialized = false;
        self.loaded_tts_model = None;
        if let Ok(mut guard) = self.chat_decode_states.lock() {
            guard.clear();
        }
        if let Ok(mut guard) = self.asr_decode_states.lock() {
            guard.clear();
        }
        if let Ok(mut guard) = self.qwen_tts_decode_states.lock() {
            guard.clear();
        }
        if let Ok(mut guard) = self.lfm2_tts_decode_states.lock() {
            guard.clear();
        }
        if let Ok(mut guard) = self.speech_to_speech_decode_states.lock() {
            guard.clear();
        }
        Ok(())
    }

    fn cleanup_request(&self, request_id: &str) {
        if let Ok(mut guard) = self.chat_decode_states.lock() {
            guard.remove(request_id);
        }
        if let Ok(mut guard) = self.asr_decode_states.lock() {
            guard.remove(request_id);
        }
        if let Ok(mut guard) = self.qwen_tts_decode_states.lock() {
            guard.remove(request_id);
        }
        if let Ok(mut guard) = self.lfm2_tts_decode_states.lock() {
            guard.remove(request_id);
        }
        if let Ok(mut guard) = self.speech_to_speech_decode_states.lock() {
            guard.remove(request_id);
        }
    }
}

/// Unified executor that wraps a model executor implementation.
pub struct UnifiedExecutor {
    inner: Arc<RwLock<Box<dyn ModelExecutor>>>,
}

impl UnifiedExecutor {
    /// Create a new unified executor with native backend.
    pub fn new_native(config: WorkerConfig) -> Self {
        Self {
            inner: Arc::new(RwLock::new(Box::new(NativeExecutor::new(config)))),
        }
    }

    #[cfg(test)]
    pub(crate) fn new_for_test(executor: Box<dyn ModelExecutor>) -> Self {
        Self {
            inner: Arc::new(RwLock::new(executor)),
        }
    }

    /// Execute requests.
    pub async fn execute(
        &self,
        requests: &[&EngineCoreRequest],
        scheduled: &[ScheduledRequest],
    ) -> Result<Vec<ExecutorOutput>> {
        let executor = self.inner.read().await;
        executor.execute(requests, scheduled)
    }

    /// Execute prefill requests.
    pub async fn execute_prefill(
        &self,
        requests: &[&EngineCoreRequest],
        scheduled: &[ScheduledRequest],
    ) -> Result<Vec<ExecutorOutput>> {
        let executor = self.inner.read().await;
        executor.execute_prefill(requests, scheduled)
    }

    /// Execute decode requests.
    pub async fn execute_decode(
        &self,
        requests: &[&EngineCoreRequest],
        scheduled: &[ScheduledRequest],
    ) -> Result<Vec<ExecutorOutput>> {
        let executor = self.inner.read().await;
        executor.execute_decode(requests, scheduled)
    }

    /// Check if ready.
    pub async fn is_ready(&self) -> bool {
        let executor = self.inner.read().await;
        executor.is_ready()
    }

    /// Initialize.
    pub async fn initialize(&self) -> Result<()> {
        let mut executor = self.inner.write().await;
        executor.initialize()
    }

    /// Shutdown.
    pub async fn shutdown(&self) -> Result<()> {
        let mut executor = self.inner.write().await;
        executor.shutdown()
    }

    /// Cleanup transient backend state for a completed/aborted request.
    pub async fn cleanup_request(&self, request_id: &str) {
        let executor = self.inner.read().await;
        executor.cleanup_request(request_id);
    }
}

/// Decode base64-encoded audio to samples.
pub fn decode_audio_base64(audio_b64: &str, _sample_rate: u32) -> Result<Vec<f32>> {
    let (samples, _) = decode_audio_base64_with_rate(audio_b64)?;
    Ok(samples)
}

fn decode_audio_base64_with_rate(audio_b64: &str) -> Result<(Vec<f32>, u32)> {
    use base64::Engine;
    use std::io::Cursor;

    let payload = if audio_b64.starts_with("data:") {
        audio_b64
            .split_once(',')
            .map(|(_, b64)| b64)
            .unwrap_or(audio_b64)
    } else {
        audio_b64
    };
    let normalized: String = payload.chars().filter(|c| !c.is_whitespace()).collect();

    let wav_bytes = base64::engine::general_purpose::STANDARD
        .decode(normalized.as_bytes())
        .map_err(|e| Error::InferenceError(format!("Failed to decode base64 audio: {}", e)))?;

    let cursor = Cursor::new(wav_bytes);
    let mut reader = hound::WavReader::new(cursor)
        .map_err(|e| Error::InferenceError(format!("Failed to parse WAV: {}", e)))?;

    let spec = reader.spec();
    let sample_rate = spec.sample_rate;
    let channels = spec.channels.max(1) as usize;

    let mut samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Int => {
            let bits = spec.bits_per_sample.max(1) as u32;
            let max_val = if bits > 1 {
                ((1i64 << (bits - 1)) - 1) as f32
            } else {
                1.0
            };
            reader
                .samples::<i32>()
                .filter_map(|s| s.ok())
                .map(|s| (s as f32 / max_val).clamp(-1.0, 1.0))
                .collect()
        }
        hound::SampleFormat::Float => reader.samples::<f32>().filter_map(|s| s.ok()).collect(),
    };

    if channels > 1 {
        let mut mono = Vec::with_capacity(samples.len() / channels + 1);
        for frame in samples.chunks(channels) {
            if frame.is_empty() {
                continue;
            }
            let sum: f32 = frame.iter().copied().sum();
            mono.push(sum / frame.len() as f32);
        }
        samples = mono;
    }

    for sample in &mut samples {
        if !sample.is_finite() {
            *sample = 0.0;
        } else {
            *sample = sample.clamp(-1.0, 1.0);
        }
    }

    Ok((samples, sample_rate))
}

#[cfg(test)]
mod tests {
    use super::*;
    use base64::Engine;

    #[test]
    fn test_worker_config_default() {
        let config = WorkerConfig::default();
        assert_eq!(config.model_type, ModelType::Qwen3TTS);
    }

    #[test]
    fn test_run_blocking_converts_panic_to_error() {
        let result = NativeExecutor::run_blocking(|| -> Result<()> {
            panic!("executor panic sentinel");
        });

        let Err(Error::InferenceError(message)) = result else {
            panic!("expected inference error from panic");
        };
        assert!(message.contains("executor panic sentinel"));
    }

    #[test]
    fn test_run_blocking_is_safe_inside_current_thread_runtime() {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("failed to build runtime");

        let result =
            runtime.block_on(async { NativeExecutor::run_blocking(|| Ok::<_, Error>(())) });
        assert!(result.is_ok());
    }

    #[test]
    fn test_stream_audio_send_is_safe_inside_current_thread_runtime() {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("failed to build runtime");

        let result = runtime.block_on(async {
            let (tx, mut rx) = mpsc::unbounded_channel();
            let mut sequence = 0usize;
            NativeExecutor::stream_audio(
                &tx,
                "req-1",
                &mut sequence,
                vec![0.1, -0.1],
                24_000,
                false,
            )?;
            let chunk = rx
                .recv()
                .await
                .ok_or_else(|| Error::InferenceError("missing streamed chunk".to_string()))?;
            if chunk.request_id != "req-1" || chunk.sequence != 0 || chunk.samples.len() != 2 {
                return Err(Error::InferenceError(
                    "unexpected streamed chunk payload".to_string(),
                ));
            }
            Ok::<(), Error>(())
        });
        assert!(result.is_ok());
    }

    #[test]
    fn test_stream_audio_send_returns_error_when_channel_closed() {
        let (tx, rx) = mpsc::unbounded_channel::<StreamingOutput>();
        drop(rx);

        let mut sequence = 0usize;
        let result = NativeExecutor::stream_audio(
            &tx,
            "req-closed",
            &mut sequence,
            vec![0.2],
            24_000,
            false,
        );
        let Err(Error::InferenceError(message)) = result else {
            panic!("expected inference error when streaming channel is closed");
        };
        assert!(message.contains("Streaming output channel closed"));
    }

    #[test]
    fn decode_audio_base64_with_rate_downmixes_stereo_wav() {
        let spec = hound::WavSpec {
            channels: 2,
            sample_rate: 16_000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };

        let mut wav_bytes = Vec::new();
        {
            let cursor = std::io::Cursor::new(&mut wav_bytes);
            let mut writer = hound::WavWriter::new(cursor, spec).expect("writer");
            // 2 stereo frames: [L,R]=[0.25,0.75] then [0.5,-0.5]
            writer.write_sample((0.25f32 * 32767.0) as i16).unwrap();
            writer.write_sample((0.75f32 * 32767.0) as i16).unwrap();
            writer.write_sample((0.5f32 * 32767.0) as i16).unwrap();
            writer.write_sample((-0.5f32 * 32767.0) as i16).unwrap();
            writer.finalize().unwrap();
        }

        let b64 = base64::engine::general_purpose::STANDARD.encode(&wav_bytes);
        let (samples, sample_rate) =
            decode_audio_base64_with_rate(&b64).expect("decode should succeed");

        assert_eq!(sample_rate, 16_000);
        assert_eq!(samples.len(), 2);
        // After downmixing, expected mono values are averages: 0.5 and 0.0.
        assert!(
            (samples[0] - 0.5).abs() < 0.02,
            "first sample was {}",
            samples[0]
        );
        assert!(samples[1].abs() < 0.02, "second sample was {}", samples[1]);
    }
}
