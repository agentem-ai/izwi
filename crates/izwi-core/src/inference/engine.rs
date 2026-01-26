//! Main inference engine for Qwen3-TTS

use std::sync::Arc;
use tokio::sync::mpsc;
use tracing::{info, warn};

use crate::audio::{AudioChunkBuffer, AudioCodec, AudioEncoder, StreamingConfig};
use crate::config::EngineConfig;
use crate::error::{Error, Result};
use crate::inference::generation::{
    AudioChunk, GenerationConfig, GenerationRequest, GenerationResult,
};
use crate::inference::kv_cache::{KVCache, KVCacheConfig};
use crate::inference::python_bridge::PythonBridge;
use crate::model::{ModelInfo, ModelManager, ModelVariant};
use crate::tokenizer::Tokenizer;

/// Main TTS inference engine
pub struct InferenceEngine {
    config: EngineConfig,
    model_manager: Arc<ModelManager>,
    tokenizer: Option<Tokenizer>,
    codec: AudioCodec,
    _kv_cache: KVCache,
    streaming_config: StreamingConfig,
    python_bridge: PythonBridge,
    loaded_model_path: Option<std::path::PathBuf>,
}

impl InferenceEngine {
    /// Create a new inference engine
    pub fn new(config: EngineConfig) -> Result<Self> {
        let model_manager = Arc::new(ModelManager::new(config.clone())?);
        let codec = AudioCodec::new();
        let kv_cache = KVCache::new(KVCacheConfig::default());

        Ok(Self {
            config,
            model_manager,
            tokenizer: None,
            codec,
            _kv_cache: kv_cache,
            streaming_config: StreamingConfig::default(),
            python_bridge: PythonBridge::new(),
            loaded_model_path: None,
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

        // Load the model weights
        let weights = self.model_manager.load_model(variant).await?;
        info!(
            "Loaded model: {} ({} bytes)",
            variant,
            weights.memory_bytes()
        );

        // Load tokenizer from model directory (optional - may not exist for all models)
        if let Some(path) = self
            .model_manager
            .get_model_info(variant)
            .await
            .and_then(|i| i.local_path)
        {
            match Tokenizer::from_path(&path) {
                Ok(tokenizer) => {
                    info!("Loaded tokenizer from {:?}", path);
                    self.tokenizer = Some(tokenizer);
                }
                Err(e) => {
                    warn!("Failed to load tokenizer: {}. TTS generation may not work until tokenizer files are available.", e);
                }
            }
        }

        // Load codec if this is a tokenizer model, or load from separate tokenizer
        if variant.is_tokenizer() {
            if let Some(path) = self
                .model_manager
                .get_model_info(variant)
                .await
                .and_then(|i| i.local_path)
            {
                self.codec.load_weights(&path)?;
            }
        }

        // Store model path for Python bridge
        if let Some(path) = self
            .model_manager
            .get_model_info(variant)
            .await
            .and_then(|i| i.local_path)
        {
            self.loaded_model_path = Some(path);
        }

        Ok(())
    }

    /// Generate audio from text (non-streaming)
    pub async fn generate(&self, request: GenerationRequest) -> Result<GenerationResult> {
        let start_time = std::time::Instant::now();

        // Get model path
        let model_path = self
            .loaded_model_path
            .as_ref()
            .ok_or_else(|| Error::InferenceError("No model loaded".to_string()))?;

        info!("Generating TTS for: {}", request.text);

        // Use Python bridge for actual inference
        let (samples, sample_rate) = self.python_bridge.generate_with_clone(
            model_path,
            &request.text,
            request.config.speaker.as_deref(),
            Some("Auto"), // language
            None,         // instruct
            request.reference_audio,
            request.reference_text,
        )?;

        let total_time_ms = start_time.elapsed().as_secs_f32() * 1000.0;
        let num_samples = samples.len();

        info!(
            "Generated {} samples in {:.1}ms",
            num_samples, total_time_ms
        );

        Ok(GenerationResult {
            request_id: request.id,
            samples,
            sample_rate,
            total_tokens: num_samples / 256, // approximate
            total_time_ms,
        })
    }

    /// Generate audio with streaming output
    pub async fn generate_streaming(
        &self,
        request: GenerationRequest,
        chunk_tx: mpsc::Sender<AudioChunk>,
    ) -> Result<()> {
        let tokenizer = self
            .tokenizer
            .as_ref()
            .ok_or_else(|| Error::InferenceError("No tokenizer loaded".to_string()))?;

        // Tokenize input text
        let prompt = tokenizer.format_tts_prompt(&request.text, request.config.speaker.as_deref());
        let input_tokens = tokenizer.encode(&prompt)?;

        info!(
            "Starting streaming generation for {} input tokens",
            input_tokens.len()
        );

        // Create streaming buffer
        let mut buffer =
            AudioChunkBuffer::new(self.streaming_config.clone(), self.codec.sample_rate());

        let mut sequence = 0;
        let mut audio_tokens: Vec<Vec<u32>> = vec![Vec::new(); self.codec.config().num_codebooks];

        // Generate tokens incrementally
        for _step in 0..request.config.max_tokens {
            // Generate next audio token(s)
            let next_tokens = self
                .generate_next_token(&input_tokens, &audio_tokens, &request.config)
                .await?;

            // Add to token buffer
            for (codebook, token) in next_tokens.iter().enumerate() {
                if codebook < audio_tokens.len() {
                    audio_tokens[codebook].push(*token);
                }
            }
            buffer.push_tokens(next_tokens);

            // Check for end of generation
            if self.is_end_of_audio(&audio_tokens) {
                break;
            }

            // Decode and stream when buffer is ready
            if buffer.ready_to_stream() {
                let chunk_tokens: Vec<Vec<u32>> =
                    audio_tokens.iter().map(|cb| cb.clone()).collect();

                let samples = self.codec.decode(&chunk_tokens)?;
                buffer.push_samples(&samples);

                while let Some(chunk_samples) = buffer.take_chunk() {
                    let chunk = AudioChunk::new(request.id.clone(), sequence, chunk_samples);
                    sequence += 1;

                    if chunk_tx.send(chunk).await.is_err() {
                        warn!("Streaming channel closed");
                        return Ok(());
                    }
                }
            }
        }

        // Send remaining samples
        let remaining = buffer.take_remaining();
        if !remaining.is_empty() {
            let chunk = AudioChunk::final_chunk(request.id.clone(), sequence, remaining);
            let _ = chunk_tx.send(chunk).await;
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

    /// Ensure the TTS daemon is running
    pub fn ensure_daemon_running(&self) -> Result<()> {
        self.python_bridge.ensure_daemon_running()
    }

    /// Stop the TTS daemon
    pub fn stop_daemon(&self) -> Result<()> {
        self.python_bridge.stop_daemon()
    }

    /// Get daemon status
    pub fn get_daemon_status(&self) -> Result<super::python_bridge::PythonTTSResponse> {
        self.python_bridge.get_status()
    }

    /// Preload a model into the daemon cache
    pub fn preload_model(&self, model_path: &str) -> Result<()> {
        self.python_bridge
            .preload_model(std::path::Path::new(model_path))
    }
}

// Simple pseudo-random number generator for placeholder
fn rand_u32() -> u32 {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .subsec_nanos();
    nanos.wrapping_mul(1103515245).wrapping_add(12345)
}
