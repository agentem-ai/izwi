//! Text-to-speech runtime methods.

use tokio::sync::mpsc;
use tracing::{debug, info, warn};

use crate::error::{Error, Result};
use crate::models::qwen3_tts::{SpeakerReference, TtsGenerationParams};
use crate::runtime::audio_io::{base64_decode, decode_wav_bytes, preprocess_reference_audio};
use crate::runtime::service::InferenceEngine;
use crate::runtime::types::{AudioChunk, GenerationConfig, GenerationRequest, GenerationResult};

impl InferenceEngine {
    /// Generate audio from text using the loaded native TTS model.
    pub async fn generate(&self, request: GenerationRequest) -> Result<GenerationResult> {
        if self.config.max_batch_size > 1
            && request.reference_audio.is_none()
            && request.reference_text.is_none()
        {
            if let Some(batcher) = &self.tts_batcher {
                return batcher.submit(request).await;
            }
        }

        let start_time = std::time::Instant::now();

        let tts_model = self.tts_model.clone();
        let text = request.text.clone();
        let speaker = request.config.speaker.clone();
        let runtime_gen_config = request.config.clone();
        let voice_description = request.voice_description.clone();
        let ref_audio = request.reference_audio.clone();
        let ref_text = request.reference_text.clone();
        let request_id = request.id.clone();

        let samples = tokio::task::spawn_blocking(move || {
            let rt = tokio::runtime::Handle::try_current();
            let model_guard = rt
                .as_ref()
                .map(|r| r.block_on(async { tts_model.read().await }))
                .unwrap_or_else(|_| panic!("No async runtime available"));

            let model = model_guard
                .as_ref()
                .ok_or_else(|| Error::InferenceError("No TTS model loaded".to_string()))?;

            if ref_audio.is_some() && ref_text.is_some() {
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
                let params = TtsGenerationParams::from_generation_config(&runtime_gen_config);
                let available_speakers = model.available_speakers();
                let requested_speaker = speaker.as_deref().filter(|s| !s.trim().is_empty());

                if available_speakers.is_empty() {
                    if let Some(req_speaker) = requested_speaker {
                        debug!(
                            "Model has no preset speakers; ignoring requested speaker '{}'",
                            req_speaker
                        );
                    }
                    model.generate_with_text_params(
                        &text,
                        Some("Auto"),
                        voice_description.as_deref(),
                        &params,
                    )
                } else {
                    let speaker_to_use =
                        requested_speaker.unwrap_or_else(|| available_speakers[0].as_str());
                    model.generate_with_speaker_params(
                        &text,
                        speaker_to_use,
                        Some("Auto"),
                        voice_description.as_deref(),
                        &params,
                    )
                }
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

    /// Generate audio with streaming output.
    pub async fn generate_streaming(
        &self,
        request: GenerationRequest,
        chunk_tx: mpsc::Sender<AudioChunk>,
    ) -> Result<()> {
        let result = self.generate(request.clone()).await?;

        let chunk_size = 1024;
        let mut sequence = 0usize;

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

    /// Generate audio tokens from input tokens.
    #[allow(dead_code)]
    async fn generate_audio_tokens(
        &self,
        _input_tokens: &[u32],
        config: &GenerationConfig,
    ) -> Result<Vec<Vec<u32>>> {
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

    /// Generate next audio token (for streaming).
    #[allow(dead_code)]
    async fn generate_next_token(
        &self,
        _input_tokens: &[u32],
        _audio_tokens: &[Vec<u32>],
        _config: &GenerationConfig,
    ) -> Result<Vec<u32>> {
        let codec = self.codec.read().await;
        let num_codebooks = codec.config().num_codebooks;
        let tokens: Vec<u32> = (0..num_codebooks)
            .map(|_| (rand_u32() % 4096) as u32)
            .collect();

        tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;

        Ok(tokens)
    }

    /// Check if generation should end.
    #[allow(dead_code)]
    fn is_end_of_audio(&self, audio_tokens: &[Vec<u32>]) -> bool {
        if audio_tokens.is_empty() || audio_tokens[0].is_empty() {
            return false;
        }

        let len = audio_tokens[0].len();
        len >= self.config.max_sequence_length
    }
}

fn rand_u32() -> u32 {
    use std::time::{SystemTime, UNIX_EPOCH};

    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .subsec_nanos();

    nanos.wrapping_mul(1103515245).wrapping_add(12345)
}
