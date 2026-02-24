//! Kokoro TTS runtime helpers (isolated from generic runtime routing).

use std::sync::Arc;

use tokio::sync::mpsc;
use tracing::info;

use crate::error::{Error, Result};
use crate::model::ModelVariant;
use crate::models::architectures::kokoro::{KokoroSynthesisResult, KokoroTtsModel};
use crate::runtime::service::RuntimeService;
use crate::runtime::types::{AudioChunk, GenerationRequest, GenerationResult};

impl RuntimeService {
    fn default_kokoro_variant() -> ModelVariant {
        ModelVariant::Kokoro82M
    }

    async fn resolve_active_kokoro_variant(&self) -> ModelVariant {
        if let Some(variant) = *self.loaded_tts_variant.read().await {
            if matches!(variant.family(), crate::catalog::ModelFamily::KokoroTts) {
                return variant;
            }
        }
        Self::default_kokoro_variant()
    }

    pub async fn kokoro_tts_generate(
        &self,
        request: GenerationRequest,
    ) -> Result<GenerationResult> {
        let variant = self.resolve_active_kokoro_variant().await;
        self.load_model(variant).await?;
        let model = self
            .model_registry
            .get_kokoro(variant)
            .await
            .ok_or_else(|| Error::InferenceError("Kokoro model not loaded".to_string()))?;

        let opts = &request.config.options;
        let speaker = opts.speaker.as_deref().or(opts.voice.as_deref());
        let result = match model.generate(
            &request.text,
            speaker,
            request.language.as_deref(),
            opts.speed,
        ) {
            Ok(result) => result,
            Err(err) if is_kokoro_voice_pack_limit_error(&err) => {
                info!(
                    "Kokoro phoneme limit hit for request; retrying with adaptive chunking fallback"
                );
                generate_chunked_kokoro(
                    model.clone(),
                    &request.text,
                    speaker,
                    request.language.as_deref(),
                    opts.speed,
                )?
            }
            Err(err) => return Err(err),
        };

        Ok(GenerationResult {
            request_id: request.id,
            samples: result.samples,
            sample_rate: result.sample_rate,
            total_tokens: result.tokens_generated,
            total_time_ms: 0.0,
        })
    }

    pub async fn kokoro_tts_generate_streaming(
        &self,
        request: GenerationRequest,
        chunk_tx: mpsc::Sender<AudioChunk>,
    ) -> Result<()> {
        let request_id = request.id.clone();
        let result = self.kokoro_tts_generate(request).await?;
        let chunk = AudioChunk::final_chunk(request_id, 0, result.samples);
        chunk_tx
            .send(chunk)
            .await
            .map_err(|_| Error::InferenceError("Streaming output channel closed".to_string()))?;
        Ok(())
    }
}

fn is_kokoro_voice_pack_limit_error(err: &Error) -> bool {
    match err {
        Error::InvalidInput(msg) => {
            msg.contains("Kokoro phoneme sequence length") && msg.contains("voice-pack limit (510)")
        }
        _ => false,
    }
}

fn generate_chunked_kokoro(
    model: Arc<KokoroTtsModel>,
    text: &str,
    speaker: Option<&str>,
    language: Option<&str>,
    speed: f32,
) -> Result<KokoroSynthesisResult> {
    let chunks = plan_kokoro_text_chunks(model.as_ref(), text, speaker, language, speed)?;
    if chunks.is_empty() {
        return Err(Error::InvalidInput(
            "Kokoro adaptive chunking produced no chunks".to_string(),
        ));
    }

    let mut combined_samples = Vec::new();
    let mut combined_phonemes = String::new();
    let mut total_tokens = 0usize;
    let mut sample_rate: Option<u32> = None;

    for (idx, chunk_text) in chunks.iter().enumerate() {
        let chunk = model.generate(chunk_text, speaker, language, speed)?;
        let current_sample_rate = chunk.sample_rate;
        match sample_rate {
            Some(expected) if expected != current_sample_rate => {
                return Err(Error::InferenceError(format!(
                    "Kokoro chunked synthesis sample rate mismatch: expected {}, got {}",
                    expected, current_sample_rate
                )));
            }
            None => sample_rate = Some(current_sample_rate),
            _ => {}
        }

        if !combined_phonemes.is_empty() {
            combined_phonemes.push(' ');
        }
        combined_phonemes.push_str(chunk.phonemes.trim());
        total_tokens += chunk.tokens_generated;
        combined_samples.extend(chunk.samples);

        if idx + 1 < chunks.len() {
            let pause_samples = ((current_sample_rate as f32) * 0.04).round() as usize;
            if pause_samples > 0 {
                combined_samples.resize(combined_samples.len() + pause_samples, 0.0);
            }
        }
    }

    let sample_rate = sample_rate.ok_or_else(|| {
        Error::InferenceError("Kokoro adaptive chunking failed to synthesize audio".to_string())
    })?;
    info!(
        chunks = chunks.len(),
        total_tokens,
        sample_rate,
        total_samples = combined_samples.len(),
        "Kokoro adaptive chunking completed"
    );
    Ok(KokoroSynthesisResult {
        samples: combined_samples,
        sample_rate,
        tokens_generated: total_tokens,
        phonemes: combined_phonemes,
    })
}

fn plan_kokoro_text_chunks(
    model: &KokoroTtsModel,
    text: &str,
    speaker: Option<&str>,
    language: Option<&str>,
    speed: f32,
) -> Result<Vec<String>> {
    let mut chunks = Vec::new();
    let mut remaining = text;
    let mut iterations = 0usize;

    loop {
        remaining = remaining.trim_start();
        if remaining.is_empty() {
            break;
        }
        iterations += 1;
        if iterations > 1024 {
            return Err(Error::InferenceError(
                "Kokoro adaptive chunking exceeded maximum chunk iterations".to_string(),
            ));
        }

        match model.prepare_request(remaining, speaker, language, speed) {
            Ok(_) => {
                chunks.push(remaining.trim_end().to_string());
                break;
            }
            Err(err) if is_kokoro_voice_pack_limit_error(&err) => {
                let max_fit_chars =
                    find_max_fitting_prefix_chars(model, remaining, speaker, language, speed)?;
                let mut split_chars = pick_readable_split_point(remaining, max_fit_chars);
                if split_chars == 0 || split_chars > max_fit_chars {
                    split_chars = max_fit_chars;
                }

                let (candidate_head, candidate_tail) = split_at_char_index(remaining, split_chars);
                let candidate_head = candidate_head.trim_end();
                let (head, tail) = if candidate_head.is_empty() {
                    let (fallback_head, fallback_tail) =
                        split_at_char_index(remaining, max_fit_chars);
                    (fallback_head.trim_end(), fallback_tail)
                } else {
                    (candidate_head, candidate_tail)
                };

                if head.is_empty() {
                    return Err(Error::InvalidInput(
                        "Kokoro adaptive chunking could not produce a non-empty chunk".to_string(),
                    ));
                }

                chunks.push(head.to_string());
                remaining = tail;
            }
            Err(err) => return Err(err),
        }
    }

    Ok(chunks)
}

fn find_max_fitting_prefix_chars(
    model: &KokoroTtsModel,
    text: &str,
    speaker: Option<&str>,
    language: Option<&str>,
    speed: f32,
) -> Result<usize> {
    let total_chars = text.chars().count();
    if total_chars == 0 {
        return Err(Error::InvalidInput(
            "Kokoro adaptive chunking received empty text".to_string(),
        ));
    }

    let mut lo = 1usize;
    let mut hi = total_chars;
    let mut best = 0usize;

    while lo <= hi {
        let mid = lo + ((hi - lo) / 2);
        let (prefix, _) = split_at_char_index(text, mid);
        let prefix = prefix.trim_end();
        if prefix.is_empty() {
            lo = mid.saturating_add(1);
            continue;
        }

        match model.prepare_request(prefix, speaker, language, speed) {
            Ok(_) => {
                best = prefix.chars().count();
                lo = mid.saturating_add(1);
            }
            Err(err) if is_kokoro_voice_pack_limit_error(&err) => {
                if mid == 0 {
                    break;
                }
                hi = mid - 1;
            }
            Err(Error::InvalidInput(msg))
                if msg.contains("Kokoro phonemizer produced no phonemes") =>
            {
                if mid == 0 {
                    break;
                }
                hi = mid - 1;
            }
            Err(err) => return Err(err),
        }
    }

    if best > 0 {
        return Ok(best);
    }

    for n in (1..=total_chars).rev() {
        let (prefix, _) = split_at_char_index(text, n);
        let prefix = prefix.trim_end();
        if prefix.is_empty() {
            continue;
        }
        match model.prepare_request(prefix, speaker, language, speed) {
            Ok(_) => return Ok(prefix.chars().count()),
            Err(err) if is_kokoro_voice_pack_limit_error(&err) => continue,
            Err(Error::InvalidInput(msg))
                if msg.contains("Kokoro phonemizer produced no phonemes") =>
            {
                continue;
            }
            Err(err) => return Err(err),
        }
    }

    Err(Error::InvalidInput(
        "Kokoro adaptive chunking could not find a chunk within the voice-pack phoneme limit"
            .to_string(),
    ))
}

fn pick_readable_split_point(text: &str, max_chars: usize) -> usize {
    if max_chars == 0 {
        return 0;
    }

    let mut last_sentence_break = None;
    let mut last_clause_break = None;
    let mut last_whitespace = None;

    for (idx, ch) in text.chars().enumerate() {
        let pos = idx + 1;
        if pos > max_chars {
            break;
        }
        if ch.is_whitespace() {
            last_whitespace = Some(pos);
            continue;
        }
        if matches!(ch, '.' | '!' | '?' | '\n') {
            last_sentence_break = Some(pos);
        } else if matches!(ch, ';' | ':' | ',') {
            last_clause_break = Some(pos);
        }
    }

    let preferred_min = (max_chars * 2) / 3;
    for candidate in [last_sentence_break, last_clause_break, last_whitespace] {
        if let Some(pos) = candidate {
            if pos >= preferred_min && pos <= max_chars {
                return pos;
            }
        }
    }
    for candidate in [last_sentence_break, last_clause_break, last_whitespace] {
        if let Some(pos) = candidate {
            if pos > 0 && pos <= max_chars {
                return pos;
            }
        }
    }
    max_chars
}

fn split_at_char_index(s: &str, n: usize) -> (&str, &str) {
    if n == 0 {
        return ("", s);
    }
    let byte_idx = s
        .char_indices()
        .nth(n)
        .map(|(idx, _)| idx)
        .unwrap_or(s.len());
    s.split_at(byte_idx)
}
