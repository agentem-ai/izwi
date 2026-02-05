//! Main inference engine for Qwen3-TTS and Qwen3-ASR using native Rust models.

use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::{mpsc, RwLock};
use tracing::{info, warn};

use crate::audio::{AudioChunkBuffer, AudioCodec, AudioEncoder, StreamingConfig};
use crate::config::EngineConfig;
use crate::error::{Error, Result};
use crate::inference::generation::{
    AudioChunk, GenerationConfig, GenerationRequest, GenerationResult,
};
use crate::inference::kv_cache::{KVCache, KVCacheConfig};
use crate::model::{ModelInfo, ModelManager, ModelVariant};
use crate::models::qwen3_tts::{Qwen3TtsModel, SpeakerReference, TtsTokenizer};
use crate::models::{DeviceProfile, DeviceSelector, ModelRegistry};
use crate::tokenizer::Tokenizer;

#[derive(Debug, Clone)]
pub struct AsrTranscription {
    pub text: String,
    pub language: Option<String>,
    pub duration_secs: f32,
}

/// Main TTS inference engine
pub struct InferenceEngine {
    config: EngineConfig,
    model_manager: Arc<ModelManager>,
    model_registry: Arc<ModelRegistry>,
    tokenizer: Option<Tokenizer>,
    codec: AudioCodec,
    _kv_cache: KVCache,
    streaming_config: StreamingConfig,
    /// Currently loaded native TTS model
    tts_model: Arc<RwLock<Option<Qwen3TtsModel>>>,
    /// Currently loaded model path
    loaded_model_path: Option<PathBuf>,
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
            tokenizer: None,
            codec,
            _kv_cache: kv_cache,
            streaming_config: StreamingConfig::default(),
            tts_model: Arc::new(RwLock::new(None)),
            loaded_model_path: None,
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

    /// Unload a model from memory
    pub async fn unload_model(&self, variant: ModelVariant) -> Result<()> {
        if variant.is_asr() || variant.is_forced_aligner() || variant.is_voxtral() {
            self.model_registry.unload_asr(variant).await;
        }
        self.model_manager.unload_model(variant).await
    }

    /// Load a model for inference
    pub async fn load_model(&mut self, variant: ModelVariant) -> Result<()> {
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

        if variant.is_asr() || variant.is_forced_aligner() || variant.is_voxtral() {
            self.model_registry.load_asr(variant, &model_path).await?;
            self.model_manager.mark_loaded(variant).await;
            return Ok(());
        }

        if variant.is_tts() {
            // Load native TTS model
            info!("Loading native TTS model from {:?}", model_path);
            let tts_model = Qwen3TtsModel::load(&model_path, self.device.clone())?;

            let mut model_guard = self.tts_model.write().await;
            *model_guard = Some(tts_model);
            self.loaded_model_path = Some(model_path.clone());

            info!("Native TTS model loaded successfully");
            self.model_manager.mark_loaded(variant).await;
            return Ok(());
        }

        // Load tokenizer from model directory
        match Tokenizer::from_path(&model_path) {
            Ok(tokenizer) => {
                info!("Loaded tokenizer from {:?}", model_path);
                self.tokenizer = Some(tokenizer);
            }
            Err(e) => {
                warn!("Failed to load tokenizer: {}", e);
            }
        }

        // Load codec if this is a tokenizer model
        if variant.is_tokenizer() {
            self.codec.load_weights(&model_path)?;
        }

        Ok(())
    }

    /// Generate audio from text (non-streaming) using native model
    pub async fn generate(&self, request: GenerationRequest) -> Result<GenerationResult> {
        let start_time = std::time::Instant::now();

        let tts_model = self.tts_model.read().await;
        let model = tts_model
            .as_ref()
            .ok_or_else(|| Error::InferenceError("No TTS model loaded".to_string()))?;

        info!("Generating TTS for: {}", request.text);

        // Generate using native model
        let samples = if request.reference_audio.is_some() && request.reference_text.is_some() {
            // Voice cloning mode
            let ref_audio = request.reference_audio.unwrap_or_default();
            let ref_bytes = base64_decode(&ref_audio).map_err(|e| {
                Error::InferenceError(format!("Failed to decode reference audio: {}", e))
            })?;

            let (ref_samples, sample_rate) = decode_wav_bytes(&ref_bytes)?;

            let speaker_ref = SpeakerReference {
                audio_samples: ref_samples,
                text: request.reference_text.unwrap_or_default(),
                sample_rate,
            };

            model.generate_with_voice_clone(&request.text, &speaker_ref, Some("Auto"))?
        } else {
            // Preset speaker mode
            let speaker = request.config.speaker.as_deref().unwrap_or("Vivian");
            model.generate_with_speaker(
                &request.text,
                speaker,
                Some("Auto"),
                request.voice_description.as_deref(),
            )?
        };

        let total_time_ms = start_time.elapsed().as_secs_f32() * 1000.0;
        let num_samples = samples.len();

        info!(
            "Generated {} samples in {:.1}ms",
            num_samples, total_time_ms
        );

        Ok(GenerationResult {
            request_id: request.id,
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
        let num_codebooks = self.codec.config().num_codebooks;
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
        let num_codebooks = self.codec.config().num_codebooks;
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

    /// Get codec sample rate
    pub fn sample_rate(&self) -> u32 {
        self.codec.sample_rate()
    }

    /// Create audio encoder
    pub fn audio_encoder(&self) -> AudioEncoder {
        AudioEncoder::new(self.codec.sample_rate(), 1)
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
    base64::engine::general_purpose::STANDARD
        .decode(data)
        .map_err(|e| Error::InferenceError(format!("Base64 decode error: {}", e)))
}

fn decode_wav_bytes(wav_bytes: &[u8]) -> Result<(Vec<f32>, u32)> {
    use std::io::Cursor;

    let cursor = Cursor::new(wav_bytes);
    let mut reader = hound::WavReader::new(cursor)
        .map_err(|e| Error::InferenceError(format!("Failed to parse WAV: {}", e)))?;

    let spec = reader.spec();
    let sample_rate = spec.sample_rate;

    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Int => {
            let max_val = (1 << (spec.bits_per_sample - 1)) as f32;
            reader
                .samples::<i32>()
                .filter_map(|s| s.ok())
                .map(|s| s as f32 / max_val)
                .collect()
        }
        hound::SampleFormat::Float => reader.samples::<f32>().filter_map(|s| s.ok()).collect(),
    };

    Ok((samples, sample_rate))
}
