//! Main inference engine for Qwen3-TTS and Qwen3-ASR using native Rust models.

use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::{broadcast, mpsc, RwLock};
use tracing::{debug, info, warn};

use crate::audio::{AudioChunkBuffer, AudioCodec, AudioEncoder, StreamingConfig};
use crate::config::EngineConfig;
use crate::error::{Error, Result};
use crate::inference::generation::{
    AudioChunk, GenerationConfig, GenerationRequest, GenerationResult,
};
use crate::inference::kv_cache::{KVCache, KVCacheConfig};
use crate::model::{ModelInfo, ModelManager, ModelVariant};
use crate::models::qwen3_chat::{ChatMessage, Qwen3ChatModel};
use crate::models::qwen3_tts::{
    Qwen3TtsModel, SpeakerReference, TtsGenerationParams, TtsTokenizer,
};
use crate::models::voxtral::VoxtralRealtimeModel;
use crate::models::{DeviceProfile, DeviceSelector, ModelRegistry};
use crate::tokenizer::Tokenizer;

#[derive(Debug, Clone)]
pub struct AsrTranscription {
    pub text: String,
    pub language: Option<String>,
    pub duration_secs: f32,
}

#[derive(Debug, Clone)]
pub struct ChatGeneration {
    pub text: String,
    pub tokens_generated: usize,
    pub generation_time_ms: f64,
}

/// Main TTS inference engine with interior mutability for concurrent access
pub struct InferenceEngine {
    config: EngineConfig,
    model_manager: Arc<ModelManager>,
    model_registry: Arc<ModelRegistry>,
    tokenizer: RwLock<Option<Tokenizer>>,
    codec: RwLock<AudioCodec>,
    _kv_cache: KVCache,
    streaming_config: StreamingConfig,
    /// Currently loaded native TTS model
    tts_model: Arc<RwLock<Option<Qwen3TtsModel>>>,
    /// Currently loaded model path
    loaded_model_path: RwLock<Option<PathBuf>>,
    /// Device profile for inference
    device: DeviceProfile,
}

impl InferenceEngine {
    /// Create a new inference engine
    pub fn new(config: EngineConfig) -> Result<Self> {
        let model_manager = Arc::new(ModelManager::new(config.clone())?);
        let device = DeviceSelector::detect()?;
        let model_registry = Arc::new(ModelRegistry::new(
            config.models_dir.clone(),
            device.clone(),
        ));
        let codec = AudioCodec::new();
        let kv_cache = KVCache::new(KVCacheConfig::default());

        Ok(Self {
            config,
            model_manager,
            model_registry,
            tokenizer: RwLock::new(None),
            codec: RwLock::new(codec),
            _kv_cache: kv_cache,
            streaming_config: StreamingConfig::default(),
            tts_model: Arc::new(RwLock::new(None)),
            loaded_model_path: RwLock::new(None),
            device,
        })
    }

    /// Get reference to model manager
    pub fn model_manager(&self) -> &Arc<ModelManager> {
        &self.model_manager
    }

    /// List available models
    pub async fn list_models(&self) -> Vec<ModelInfo> {
        self.model_manager.list_models().await
    }

    /// Download a model
    pub async fn download_model(&self, variant: ModelVariant) -> Result<()> {
        self.model_manager.download_model(variant).await?;
        Ok(())
    }

    /// Spawn a non-blocking background download
    pub async fn spawn_download(
        &self,
        variant: ModelVariant,
    ) -> Result<broadcast::Receiver<crate::model::download::DownloadProgress>> {
        let progress_rx = self.model_manager.spawn_download(variant).await?;
        Ok(progress_rx)
    }

    /// Check if a download is active
    pub async fn is_download_active(&self, variant: ModelVariant) -> bool {
        self.model_manager.is_download_active(variant).await
    }

    /// Unload a model from memory
    pub async fn unload_model(&self, variant: ModelVariant) -> Result<()> {
        if variant.is_asr() || variant.is_forced_aligner() {
            self.model_registry.unload_asr(variant).await;
        } else if variant.is_chat() {
            self.model_registry.unload_chat(variant).await;
        } else if variant.is_voxtral() {
            self.model_registry.unload_voxtral(variant).await;
        }
        self.model_manager.unload_model(variant).await
    }

    /// Load a model for inference (now with interior mutability)
    pub async fn load_model(&self, variant: ModelVariant) -> Result<()> {
        // Ensure model is downloaded
        if !self.model_manager.is_ready(variant).await {
            let info = self.model_manager.get_model_info(variant).await;
            if info.map(|i| i.local_path.is_none()).unwrap_or(true) {
                return Err(Error::ModelNotFound(format!(
                    "Model {} not downloaded. Please download it first.",
                    variant
                )));
            }
        }

        let model_path = self
            .model_manager
            .get_model_info(variant)
            .await
            .and_then(|i| i.local_path)
            .ok_or_else(|| Error::ModelNotFound(variant.to_string()))?;

        if variant.is_asr() || variant.is_forced_aligner() {
            self.model_registry.load_asr(variant, &model_path).await?;
            self.model_manager.mark_loaded(variant).await;
            return Ok(());
        }

        if variant.is_chat() {
            self.model_registry.load_chat(variant, &model_path).await?;
            self.model_manager.mark_loaded(variant).await;
            return Ok(());
        }

        if variant.is_voxtral() {
            self.model_registry
                .load_voxtral(variant, &model_path)
                .await?;
            self.model_manager.mark_loaded(variant).await;
            return Ok(());
        }

        if variant.is_tts() {
            // Load native TTS model
            info!("Loading native TTS model from {:?}", model_path);
            let tts_model = Qwen3TtsModel::load(&model_path, self.device.clone())?;

            let mut model_guard = self.tts_model.write().await;
            *model_guard = Some(tts_model);
            let mut path_guard = self.loaded_model_path.write().await;
            *path_guard = Some(model_path.clone());

            info!("Native TTS model loaded successfully");
            self.model_manager.mark_loaded(variant).await;
            return Ok(());
        }

        // Load tokenizer from model directory
        match Tokenizer::from_path(&model_path) {
            Ok(tokenizer) => {
                info!("Loaded tokenizer from {:?}", model_path);
                let mut guard = self.tokenizer.write().await;
                *guard = Some(tokenizer);
            }
            Err(e) => {
                warn!("Failed to load tokenizer: {}", e);
            }
        }

        // Load codec if this is a tokenizer model
        if variant.is_tokenizer() {
            let mut codec_guard = self.codec.write().await;
            codec_guard.load_weights(&model_path)?;
        }

        Ok(())
    }

    /// Generate audio from text (non-streaming) using native model - now with spawn_blocking
    pub async fn generate(&self, request: GenerationRequest) -> Result<GenerationResult> {
        let start_time = std::time::Instant::now();

        // Clone necessary data for spawn_blocking
        let tts_model = self.tts_model.clone();
        let text = request.text.clone();
        let speaker = request.config.speaker.clone();
        let runtime_gen_config = request.config.clone();
        let voice_description = request.voice_description.clone();
        let ref_audio = request.reference_audio.clone();
        let ref_text = request.reference_text.clone();
        let request_id = request.id.clone();

        // Run CPU-intensive generation in spawn_blocking
        let samples = tokio::task::spawn_blocking(move || {
            let rt = tokio::runtime::Handle::try_current();
            let model_guard = rt
                .as_ref()
                .map(|r| r.block_on(async { tts_model.read().await }))
                .unwrap_or_else(|_| {
                    // Fallback: this shouldn't happen in normal async context
                    panic!("No async runtime available");
                });

            let model = model_guard
                .as_ref()
                .ok_or_else(|| Error::InferenceError("No TTS model loaded".to_string()))?;

            if ref_audio.is_some() && ref_text.is_some() {
                // Voice cloning mode
                let ref_audio = ref_audio.unwrap_or_default();
                let ref_bytes = base64_decode(&ref_audio).map_err(|e| {
                    Error::InferenceError(format!("Failed to decode reference audio: {}", e))
                })?;

                let (ref_samples, sample_rate) = decode_wav_bytes(&ref_bytes)?;
                let ref_samples = preprocess_reference_audio(ref_samples, sample_rate);
                if ref_samples.is_empty() {
                    return Err(Error::InvalidInput(
                        "Reference audio is silent or invalid after preprocessing".to_string(),
                    ));
                }

                let speaker_ref = SpeakerReference {
                    audio_samples: ref_samples,
                    text: ref_text.unwrap_or_default(),
                    sample_rate,
                };

                model.generate_with_voice_clone(&text, &speaker_ref, Some("Auto"))
            } else {
                // Preset speaker mode
                let speaker = speaker.as_deref().unwrap_or("Vivian");
                let params = TtsGenerationParams::from_generation_config(&runtime_gen_config);
                model.generate_with_speaker_params(
                    &text,
                    speaker,
                    Some("Auto"),
                    voice_description.as_deref(),
                    &params,
                )
            }
        })
        .await
        .map_err(|e| Error::InferenceError(format!("Generation task failed: {}", e)))??;

        let total_time_ms = start_time.elapsed().as_secs_f32() * 1000.0;
        let num_samples = samples.len();

        info!(
            "Generated {} samples in {:.1}ms",
            num_samples, total_time_ms
        );

        Ok(GenerationResult {
            request_id,
            samples,
            sample_rate: 24000,
            total_tokens: num_samples / 256,
            total_time_ms,
        })
    }

    /// Generate audio with streaming output
    pub async fn generate_streaming(
        &self,
        request: GenerationRequest,
        chunk_tx: mpsc::Sender<AudioChunk>,
    ) -> Result<()> {
        // Streaming uses the same non-streaming implementation but sends chunks
        let result = self.generate(request.clone()).await?;

        let chunk_size = 1024;
        let mut sequence = 0;

        for chunk_samples in result.samples.chunks(chunk_size) {
            let chunk = if chunk_samples.len() < chunk_size && sequence > 0 {
                AudioChunk::final_chunk(request.id.clone(), sequence, chunk_samples.to_vec())
            } else {
                AudioChunk::new(request.id.clone(), sequence, chunk_samples.to_vec())
            };

            sequence += 1;

            if chunk_tx.send(chunk).await.is_err() {
                warn!("Streaming channel closed");
                return Ok(());
            }
        }

        info!("Streaming generation complete");
        Ok(())
    }

    /// Generate audio tokens from input tokens
    #[allow(dead_code)]
    async fn generate_audio_tokens(
        &self,
        _input_tokens: &[u32],
        config: &GenerationConfig,
    ) -> Result<Vec<Vec<u32>>> {
        // Placeholder: Generate dummy tokens
        // In real implementation, this runs the transformer forward pass
        let codec = self.codec.read().await;
        let num_codebooks = codec.config().num_codebooks;
        let num_tokens = config.max_tokens.min(256);

        let mut audio_tokens = Vec::with_capacity(num_codebooks);
        for _ in 0..num_codebooks {
            let tokens: Vec<u32> = (0..num_tokens)
                .map(|i| ((i * 17 + 42) % 4096) as u32)
                .collect();
            audio_tokens.push(tokens);
        }

        Ok(audio_tokens)
    }

    /// Generate next audio token (for streaming)
    async fn generate_next_token(
        &self,
        _input_tokens: &[u32],
        _audio_tokens: &[Vec<u32>],
        _config: &GenerationConfig,
    ) -> Result<Vec<u32>> {
        // Placeholder: Generate single token per codebook
        // In real implementation, this runs incremental inference
        let codec = self.codec.read().await;
        let num_codebooks = codec.config().num_codebooks;
        let tokens: Vec<u32> = (0..num_codebooks)
            .map(|_i| (rand_u32() % 4096) as u32)
            .collect();

        // Simulate generation time
        tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;

        Ok(tokens)
    }

    /// Check if generation should end
    fn is_end_of_audio(&self, audio_tokens: &[Vec<u32>]) -> bool {
        // Check for end token or maximum length
        if audio_tokens.is_empty() || audio_tokens[0].is_empty() {
            return false;
        }

        let len = audio_tokens[0].len();
        len >= self.config.max_sequence_length
    }

    /// Get engine configuration
    pub fn config(&self) -> &EngineConfig {
        &self.config
    }

    /// Get codec sample rate (acquires read lock)
    pub async fn sample_rate(&self) -> u32 {
        self.codec.read().await.sample_rate()
    }

    /// Create audio encoder (acquires read lock)
    pub async fn audio_encoder(&self) -> AudioEncoder {
        let codec = self.codec.read().await;
        AudioEncoder::new(codec.sample_rate(), 1)
    }

    /// Get available speakers for the loaded model
    pub async fn available_speakers(&self) -> Result<Vec<String>> {
        let tts_model = self.tts_model.read().await;
        let model = tts_model
            .as_ref()
            .ok_or_else(|| Error::InferenceError("No TTS model loaded".to_string()))?;

        Ok(model.available_speakers().into_iter().cloned().collect())
    }

    /// Stop all daemons (no-op for native implementation)
    pub fn stop_all_daemons(&self) -> Result<()> {
        Ok(())
    }

    /// Preload a model (no-op for native implementation)
    pub fn preload_model(&self, _model_path: &str) -> Result<()> {
        Ok(())
    }

    async fn get_or_load_chat_model(&self, variant: ModelVariant) -> Result<Arc<Qwen3ChatModel>> {
        if !variant.is_chat() {
            return Err(Error::InvalidInput(format!(
                "Model {variant} is not a chat model"
            )));
        }

        if let Some(model) = self.model_registry.get_chat(variant).await {
            return Ok(model);
        }

        let path = self
            .model_manager
            .get_model_info(variant)
            .await
            .and_then(|i| i.local_path)
            .ok_or_else(|| Error::ModelNotFound(variant.to_string()))?;

        let model = self.model_registry.load_chat(variant, &path).await?;
        self.model_manager.mark_loaded(variant).await;
        Ok(model)
    }

    pub async fn chat_generate(
        &self,
        variant: ModelVariant,
        messages: Vec<ChatMessage>,
        max_new_tokens: usize,
    ) -> Result<ChatGeneration> {
        let model = self.get_or_load_chat_model(variant).await?;
        let started = std::time::Instant::now();

        let output = tokio::task::spawn_blocking(move || model.generate(&messages, max_new_tokens))
            .await
            .map_err(|e| Error::InferenceError(format!("Chat generation task failed: {}", e)))??;

        Ok(ChatGeneration {
            text: output.text,
            tokens_generated: output.tokens_generated,
            generation_time_ms: started.elapsed().as_secs_f64() * 1000.0,
        })
    }

    pub async fn chat_generate_streaming<F>(
        &self,
        variant: ModelVariant,
        messages: Vec<ChatMessage>,
        max_new_tokens: usize,
        on_delta: F,
    ) -> Result<ChatGeneration>
    where
        F: FnMut(String) + Send + 'static,
    {
        let model = self.get_or_load_chat_model(variant).await?;
        let started = std::time::Instant::now();

        let output = tokio::task::spawn_blocking(move || {
            let mut callback = on_delta;
            let mut emit = |delta: &str| callback(delta.to_string());
            model.generate_with_callback(&messages, max_new_tokens, &mut emit)
        })
        .await
        .map_err(|e| Error::InferenceError(format!("Chat generation task failed: {}", e)))??;

        Ok(ChatGeneration {
            text: output.text,
            tokens_generated: output.tokens_generated,
            generation_time_ms: started.elapsed().as_secs_f64() * 1000.0,
        })
    }

    /// Transcribe audio with Voxtral Realtime (native).
    pub async fn voxtral_transcribe(
        &self,
        audio_base64: &str,
        language: Option<&str>,
    ) -> Result<AsrTranscription> {
        let variant = ModelVariant::VoxtralMini4BRealtime2602;

        let model = if let Some(model) = self.model_registry.get_voxtral(variant).await {
            model
        } else {
            let path = self
                .model_manager
                .get_model_info(variant)
                .await
                .and_then(|i| i.local_path)
                .ok_or_else(|| Error::ModelNotFound(variant.to_string()))?;
            self.model_registry.load_voxtral(variant, &path).await?
        };

        let (samples, sample_rate) = decode_wav_bytes(&base64_decode(audio_base64)?)?;
        let samples_len = samples.len();

        // Run transcription in spawn_blocking for CPU-intensive work
        let text = tokio::task::spawn_blocking({
            let model = model.clone();
            let language = language.map(|s| s.to_string());
            move || model.transcribe(&samples, sample_rate, language.as_deref())
        })
        .await
        .map_err(|e| {
            Error::InferenceError(format!("Voxtral transcription task failed: {}", e))
        })??;

        Ok(AsrTranscription {
            text,
            language: language.map(|s| s.to_string()),
            duration_secs: samples_len as f32 / sample_rate as f32,
        })
    }

    // ============ Qwen3-ASR Methods ============

    /// Transcribe audio with Qwen3-ASR (native).
    pub async fn asr_transcribe(
        &self,
        audio_base64: &str,
        model_id: Option<&str>,
        language: Option<&str>,
    ) -> Result<AsrTranscription> {
        let variant = match model_id {
            Some(id) if id.contains("1.7") => ModelVariant::Qwen3Asr17B,
            Some(_) => ModelVariant::Qwen3Asr06B,
            None => ModelVariant::Qwen3Asr06B,
        };

        let model = if let Some(model) = self.model_registry.get_asr(variant).await {
            model
        } else {
            let path = self
                .model_manager
                .get_model_info(variant)
                .await
                .and_then(|i| i.local_path)
                .ok_or_else(|| Error::ModelNotFound(variant.to_string()))?;
            self.model_registry.load_asr(variant, &path).await?
        };

        let (samples, sample_rate) = decode_wav_bytes(&base64_decode(audio_base64)?)?;
        let text = model.transcribe(&samples, sample_rate, language)?;

        Ok(AsrTranscription {
            text,
            language: language.map(|s| s.to_string()),
            duration_secs: samples.len() as f32 / sample_rate as f32,
        })
    }

    /// Force alignment: align reference text with audio timestamps
    pub async fn force_align(
        &self,
        audio_base64: &str,
        reference_text: &str,
    ) -> Result<Vec<(String, u32, u32)>> {
        let variant = ModelVariant::Qwen3ForcedAligner06B;

        let model = if let Some(model) = self.model_registry.get_asr(variant).await {
            model
        } else {
            let path = self
                .model_manager
                .get_model_info(variant)
                .await
                .and_then(|i| i.local_path)
                .ok_or_else(|| Error::ModelNotFound(variant.to_string()))?;
            self.model_registry.load_asr(variant, &path).await?
        };

        let (samples, sample_rate) = decode_wav_bytes(&base64_decode(audio_base64)?)?;
        let alignments = model.force_align(&samples, sample_rate, reference_text)?;

        Ok(alignments)
    }
}

fn rand_u32() -> u32 {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .subsec_nanos();
    nanos.wrapping_mul(1103515245).wrapping_add(12345)
}

fn base64_decode(data: &str) -> Result<Vec<u8>> {
    use base64::Engine;
    let payload = if data.starts_with("data:") {
        data.split_once(',').map(|(_, b64)| b64).unwrap_or(data)
    } else {
        data
    };

    let normalized: String = payload.chars().filter(|c| !c.is_whitespace()).collect();
    base64::engine::general_purpose::STANDARD
        .decode(normalized.as_bytes())
        .map_err(|e| Error::InferenceError(format!("Base64 decode error: {}", e)))
}

fn decode_wav_bytes(wav_bytes: &[u8]) -> Result<(Vec<f32>, u32)> {
    use std::io::Cursor;

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

fn preprocess_reference_audio(mut samples: Vec<f32>, sample_rate: u32) -> Vec<f32> {
    if samples.is_empty() || sample_rate == 0 {
        return Vec::new();
    }

    let original_len = samples.len();

    for sample in &mut samples {
        if !sample.is_finite() {
            *sample = 0.0;
        }
    }

    // Remove DC bias.
    let mean = samples.iter().copied().sum::<f32>() / samples.len() as f32;
    for sample in &mut samples {
        *sample -= mean;
    }

    let initial_peak = samples.iter().fold(0.0f32, |p, &s| p.max(s.abs()));
    if initial_peak < 1e-5 {
        return Vec::new();
    }

    // Trim leading/trailing silence while keeping short context margins.
    let silence_threshold = (initial_peak * 0.04).max(0.0025);
    let first_idx = samples.iter().position(|s| s.abs() >= silence_threshold);
    let last_idx = samples.iter().rposition(|s| s.abs() >= silence_threshold);
    if let (Some(first), Some(last)) = (first_idx, last_idx) {
        let margin = ((sample_rate as f32) * 0.12) as usize;
        let start = first.saturating_sub(margin);
        let end = (last + margin + 1).min(samples.len());
        samples = samples[start..end].to_vec();
    }

    // Bound reference length to avoid conditioning on long silence/noise tails.
    let max_seconds = 12usize;
    let max_len = sample_rate as usize * max_seconds;
    if samples.len() > max_len && max_len > 0 {
        let window = (sample_rate as usize * 6).clamp(sample_rate as usize, samples.len());
        let best_start = highest_energy_window_start(&samples, window);
        let start = best_start.min(samples.len() - max_len);
        samples = samples[start..start + max_len].to_vec();
    }

    // Normalize into a practical loudness band so encoder sees stable dynamics.
    let mut peak = samples.iter().fold(0.0f32, |p, &s| p.max(s.abs()));
    if peak > 0.95 {
        let scale = 0.95 / peak;
        for sample in &mut samples {
            *sample *= scale;
        }
    }

    let rms = (samples
        .iter()
        .map(|&s| (s as f64) * (s as f64))
        .sum::<f64>()
        / samples.len() as f64)
        .sqrt() as f32;
    let min_rms = 0.035f32;
    if rms > 1e-6 && rms < min_rms {
        let gain = (min_rms / rms).min(6.0);
        for sample in &mut samples {
            *sample *= gain;
        }
    }

    // Final hard limit.
    peak = samples.iter().fold(0.0f32, |p, &s| p.max(s.abs()));
    if peak > 0.95 {
        let scale = 0.95 / peak;
        for sample in &mut samples {
            *sample *= scale;
        }
    }

    debug!(
        "Reference preprocessing: {} -> {} samples @ {} Hz",
        original_len,
        samples.len(),
        sample_rate
    );

    samples
}

fn highest_energy_window_start(samples: &[f32], window: usize) -> usize {
    if samples.is_empty() || window == 0 || samples.len() <= window {
        return 0;
    }

    let mut prefix = Vec::with_capacity(samples.len() + 1);
    prefix.push(0.0f64);
    for &sample in samples {
        let e = (sample as f64) * (sample as f64);
        let next = prefix.last().copied().unwrap_or(0.0) + e;
        prefix.push(next);
    }

    let mut best_start = 0usize;
    let mut best_energy = f64::NEG_INFINITY;
    for start in 0..=samples.len() - window {
        let end = start + window;
        let energy = prefix[end] - prefix[start];
        if energy > best_energy {
            best_energy = energy;
            best_start = start;
        }
    }

    best_start
}
