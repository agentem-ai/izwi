//! Text-to-speech runtime methods.

use tokio::sync::mpsc;
use tracing::{debug, info, warn};

use crate::error::{Error, Result};
use crate::models::qwen3_tts::{SpeakerReference, TtsGenerationParams, TtsStreamingConfig};
use crate::runtime::audio_io::{base64_decode, decode_wav_bytes, preprocess_reference_audio};
use crate::runtime::service::InferenceEngine;
use crate::runtime::types::{AudioChunk, GenerationConfig, GenerationRequest, GenerationResult};

impl InferenceEngine {
    /// Generate audio from text using the loaded native TTS model.
    pub async fn generate(&self, request: GenerationRequest) -> Result<GenerationResult> {
        if matches!(
            *self.loaded_tts_variant.read().await,
            Some(crate::model::ModelVariant::Lfm2Audio15B)
        ) {
            return self.lfm2_tts_generate(request).await;
        }

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
        let language = request.language.clone();
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

            if ref_audio.is_some() || ref_text.is_some() {
                let ref_audio = ref_audio.ok_or_else(|| {
                    Error::InvalidInput(
                        "reference_audio and reference_text must both be provided".to_string(),
                    )
                })?;
                let ref_text = ref_text.unwrap_or_default();
                if ref_text.trim().is_empty() {
                    return Err(Error::InvalidInput(
                        "reference_text cannot be empty for voice cloning".to_string(),
                    ));
                }

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
                    text: ref_text,
                    sample_rate,
                };

                model.generate_with_voice_clone(&text, &speaker_ref, language.as_deref())
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
                        language.as_deref(),
                        voice_description.as_deref(),
                        &params,
                    )
                } else {
                    let speaker_to_use =
                        requested_speaker.unwrap_or_else(|| available_speakers[0].as_str());
                    model.generate_with_speaker_params(
                        &text,
                        speaker_to_use,
                        language.as_deref(),
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
        if matches!(
            *self.loaded_tts_variant.read().await,
            Some(crate::model::ModelVariant::Lfm2Audio15B)
        ) {
            return self.lfm2_tts_generate_streaming(request, chunk_tx).await;
        }

        let tts_model = self.tts_model.clone();
        let text = request.text.clone();
        let speaker = request.config.speaker.clone();
        let runtime_gen_config = request.config.clone();
        let language = request.language.clone();
        let voice_description = request.voice_description.clone();
        let ref_audio = request.reference_audio.clone();
        let ref_text = request.reference_text.clone();
        let request_id = request.id.clone();
        let chunk_hint = self.config.chunk_size.max(1);

        tokio::task::spawn_blocking(move || {
            let rt = tokio::runtime::Handle::try_current();
            let model_guard = rt
                .as_ref()
                .map(|r| r.block_on(async { tts_model.read().await }))
                .unwrap_or_else(|_| panic!("No async runtime available"));

            let model = model_guard
                .as_ref()
                .ok_or_else(|| Error::InferenceError("No TTS model loaded".to_string()))?;

            let params = TtsGenerationParams::from_generation_config(&runtime_gen_config);
            let stream_config = TtsStreamingConfig {
                min_frames_before_stream: chunk_hint.clamp(2, 24),
                decode_interval_frames: chunk_hint.clamp(2, 24),
                decode_lookahead_frames: 2,
            };

            let mut sequence = 0usize;
            let mut emit_chunk = |samples: Vec<f32>| -> Result<()> {
                if samples.is_empty() {
                    return Ok(());
                }
                let chunk = AudioChunk::new(request_id.clone(), sequence, samples);
                sequence += 1;
                chunk_tx.blocking_send(chunk).map_err(|_| {
                    Error::InferenceError("Streaming output channel closed".to_string())
                })?;
                Ok(())
            };

            if ref_audio.is_some() || ref_text.is_some() {
                let ref_audio = ref_audio.ok_or_else(|| {
                    Error::InvalidInput(
                        "reference_audio and reference_text must both be provided".to_string(),
                    )
                })?;
                let ref_text = ref_text.unwrap_or_default();
                if ref_text.trim().is_empty() {
                    return Err(Error::InvalidInput(
                        "reference_text cannot be empty for voice cloning".to_string(),
                    ));
                }

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
                    text: ref_text,
                    sample_rate,
                };

                model.generate_with_voice_clone_streaming(
                    &text,
                    &speaker_ref,
                    language.as_deref(),
                    &params,
                    stream_config,
                    &mut emit_chunk,
                )?;
            } else {
                let available_speakers = model.available_speakers();
                let requested_speaker = speaker.as_deref().filter(|s| !s.trim().is_empty());

                if available_speakers.is_empty() {
                    if let Some(req_speaker) = requested_speaker {
                        debug!(
                            "Model has no preset speakers; ignoring requested speaker '{}'",
                            req_speaker
                        );
                    }
                    model.generate_with_text_params_streaming(
                        &text,
                        language.as_deref(),
                        voice_description.as_deref(),
                        &params,
                        stream_config,
                        &mut emit_chunk,
                    )?;
                } else {
                    let speaker_to_use =
                        requested_speaker.unwrap_or_else(|| available_speakers[0].as_str());
                    model.generate_with_speaker_params_streaming(
                        &text,
                        speaker_to_use,
                        language.as_deref(),
                        voice_description.as_deref(),
                        &params,
                        stream_config,
                        &mut emit_chunk,
                    )?;
                }
            }

            Ok(())
        })
        .await
        .map_err(|e| Error::InferenceError(format!("Streaming generation task failed: {}", e)))?
        .or_else(|err| {
            if matches!(err, Error::InferenceError(ref msg) if msg.contains("channel closed")) {
                warn!("Streaming channel closed");
                Ok(())
            } else {
                Err(err)
            }
        })?;

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
