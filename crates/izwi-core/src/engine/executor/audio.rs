use izwi_asr_toolkit::{plan_audio_chunks, AsrLongFormConfig, AudioChunk, TranscriptAssembler};
use tokio::sync::mpsc;
use tracing::debug;

use crate::engine::EngineCoreRequest;
use crate::error::{Error, Result};
use crate::runtime::audio_io::{base64_decode, decode_audio_bytes};

use super::super::output::StreamingOutput;
use super::NativeExecutor;

const DEFAULT_STREAM_SHORT_TARGET_CHUNK_SECS: f32 = 2.4;
const DEFAULT_STREAM_SHORT_MAX_CHUNK_SECS: f32 = 3.2;
const DEFAULT_STREAM_SHORT_OVERLAP_SECS: f32 = 0.45;
const DEFAULT_STREAM_SHORT_MIN_CHUNK_SECS: f32 = 1.0;
const DEFAULT_STREAM_SHORT_SILENCE_SEARCH_SECS: f32 = 0.75;
const DEFAULT_STREAM_LONG_TARGET_CHUNK_SECS: f32 = 6.0;
const DEFAULT_STREAM_LONG_MAX_CHUNK_SECS: f32 = 8.0;
const DEFAULT_STREAM_LONG_OVERLAP_SECS: f32 = 1.0;
const DEFAULT_STREAM_LONG_MIN_CHUNK_SECS: f32 = 2.0;
const DEFAULT_STREAM_LONG_SILENCE_SEARCH_SECS: f32 = 1.5;
const STREAM_SHORT_AUDIO_SECS_THRESHOLD: f32 = 12.0;

impl NativeExecutor {
    pub(super) fn env_f32(key: &str) -> Option<f32> {
        std::env::var(key)
            .ok()
            .and_then(|raw| raw.trim().parse::<f32>().ok())
            .filter(|v| v.is_finite() && *v > 0.0)
    }

    pub(super) fn asr_long_form_config() -> AsrLongFormConfig {
        let mut cfg = AsrLongFormConfig::default();
        if let Some(v) = Self::env_f32("IZWI_ASR_CHUNK_TARGET_SECS") {
            cfg.target_chunk_secs = v;
        }
        if let Some(v) = Self::env_f32("IZWI_ASR_CHUNK_MAX_SECS") {
            cfg.hard_max_chunk_secs = v;
        }
        if let Some(v) = Self::env_f32("IZWI_ASR_CHUNK_OVERLAP_SECS") {
            cfg.overlap_secs = v;
        }
        cfg
    }

    pub(super) fn asr_streaming_low_latency_config(audio_secs: f32) -> AsrLongFormConfig {
        let mut cfg = Self::asr_long_form_config();
        if audio_secs <= STREAM_SHORT_AUDIO_SECS_THRESHOLD {
            cfg.target_chunk_secs = DEFAULT_STREAM_SHORT_TARGET_CHUNK_SECS;
            cfg.hard_max_chunk_secs = DEFAULT_STREAM_SHORT_MAX_CHUNK_SECS;
            cfg.overlap_secs = DEFAULT_STREAM_SHORT_OVERLAP_SECS;
            cfg.min_chunk_secs = DEFAULT_STREAM_SHORT_MIN_CHUNK_SECS;
            cfg.silence_search_secs = DEFAULT_STREAM_SHORT_SILENCE_SEARCH_SECS;
        } else {
            cfg.target_chunk_secs = cfg.target_chunk_secs.min(DEFAULT_STREAM_LONG_TARGET_CHUNK_SECS);
            cfg.hard_max_chunk_secs =
                cfg.hard_max_chunk_secs.min(DEFAULT_STREAM_LONG_MAX_CHUNK_SECS);
            cfg.overlap_secs = cfg.overlap_secs.min(DEFAULT_STREAM_LONG_OVERLAP_SECS);
            cfg.min_chunk_secs = cfg.min_chunk_secs.min(DEFAULT_STREAM_LONG_MIN_CHUNK_SECS);
            cfg.silence_search_secs = cfg
                .silence_search_secs
                .min(DEFAULT_STREAM_LONG_SILENCE_SEARCH_SECS);
        }

        if let Some(v) = Self::env_f32("IZWI_ASR_STREAM_CHUNK_TARGET_SECS") {
            cfg.target_chunk_secs = v;
        }
        if let Some(v) = Self::env_f32("IZWI_ASR_STREAM_CHUNK_MAX_SECS") {
            cfg.hard_max_chunk_secs = v;
        }
        if let Some(v) = Self::env_f32("IZWI_ASR_STREAM_CHUNK_OVERLAP_SECS") {
            cfg.overlap_secs = v;
        }
        if let Some(v) = Self::env_f32("IZWI_ASR_STREAM_CHUNK_MIN_SECS") {
            cfg.min_chunk_secs = v;
        }
        if let Some(v) = Self::env_f32("IZWI_ASR_STREAM_CHUNK_SILENCE_SEARCH_SECS") {
            cfg.silence_search_secs = v;
        }

        if cfg.hard_max_chunk_secs < cfg.min_chunk_secs {
            cfg.hard_max_chunk_secs = cfg.min_chunk_secs;
        }
        if cfg.target_chunk_secs < cfg.min_chunk_secs {
            cfg.target_chunk_secs = cfg.min_chunk_secs;
        }
        if cfg.target_chunk_secs > cfg.hard_max_chunk_secs {
            cfg.target_chunk_secs = cfg.hard_max_chunk_secs;
        }
        if cfg.overlap_secs > cfg.target_chunk_secs * 0.45 {
            cfg.overlap_secs = cfg.target_chunk_secs * 0.45;
        }
        if cfg.silence_search_secs > cfg.target_chunk_secs * 0.5 {
            cfg.silence_search_secs = cfg.target_chunk_secs * 0.5;
        }

        cfg
    }

    pub(super) fn asr_chunk_plan(
        samples: &[f32],
        sample_rate: u32,
        model_max_chunk_secs: Option<f32>,
        streaming_low_latency: bool,
    ) -> (AsrLongFormConfig, Vec<AudioChunk>) {
        let audio_secs = if sample_rate > 0 {
            samples.len() as f32 / sample_rate as f32
        } else {
            0.0
        };
        let cfg = if streaming_low_latency {
            Self::asr_streaming_low_latency_config(audio_secs)
        } else {
            Self::asr_long_form_config()
        };
        let tuned_limit = model_max_chunk_secs
            .filter(|v| v.is_finite() && *v > 0.0)
            .map(|v| (v * 0.95).max(cfg.min_chunk_secs.max(1.0)));
        let chunks = plan_audio_chunks(samples, sample_rate, &cfg, tuned_limit);
        (cfg, chunks)
    }

    pub(super) fn transcribe_with_chunk_plan<F>(
        request_id: &str,
        stream_tx: Option<&mpsc::Sender<StreamingOutput>>,
        sequence: &mut usize,
        samples: &[f32],
        sample_rate: u32,
        chunk_plan: &[AudioChunk],
        chunk_cfg: &AsrLongFormConfig,
        mut transcribe_chunk: F,
    ) -> Result<String>
    where
        F: FnMut(&[f32], u32) -> Result<String>,
    {
        if chunk_plan.is_empty() {
            return Err(Error::InvalidInput(
                "ASR chunk planner produced no chunks".to_string(),
            ));
        }

        debug!(
            "ASR long-form chunking enabled for request {}: {} chunks (~{:.1}s audio)",
            request_id,
            chunk_plan.len(),
            samples.len() as f32 / sample_rate.max(1) as f32
        );

        let mut assembler = TranscriptAssembler::new(chunk_cfg.clone());
        for chunk in chunk_plan {
            if chunk.end_sample <= chunk.start_sample || chunk.end_sample > samples.len() {
                continue;
            }
            let chunk_audio = &samples[chunk.start_sample..chunk.end_sample];
            let chunk_text = transcribe_chunk(chunk_audio, sample_rate)?;
            let delta = assembler.push_chunk_text(&chunk_text);
            if !delta.is_empty() {
                if let Some(tx) = stream_tx {
                    Self::stream_text_per_character(tx, request_id, sequence, &delta)?;
                }
            }
        }

        if let Some(tx) = stream_tx {
            Self::stream_final_marker(tx, request_id, sequence)?;
        }

        Ok(assembler.finish())
    }

    pub(super) fn next_audio_delta(all_samples: &[f32], emitted_samples: &mut usize) -> Vec<f32> {
        let start = (*emitted_samples).min(all_samples.len());
        let delta = all_samples[start..].to_vec();
        *emitted_samples = all_samples.len();
        delta
    }

    pub(super) fn next_audio_delta_stable(
        all_samples: &[f32],
        emitted_samples: &mut usize,
        holdback_samples: usize,
        is_final: bool,
    ) -> Vec<f32> {
        let stable_end = if is_final {
            all_samples.len()
        } else {
            all_samples.len().saturating_sub(holdback_samples)
        };
        let start = (*emitted_samples).min(stable_end);
        let delta = all_samples[start..stable_end].to_vec();
        *emitted_samples = stable_end;
        delta
    }
}

pub(super) fn decode_audio_base64_with_rate(audio_b64: &str) -> Result<(Vec<f32>, u32)> {
    let audio_bytes = base64_decode(audio_b64)?;
    decode_audio_bytes(&audio_bytes)
}

pub(super) fn decode_request_audio_with_rate(
    request: &EngineCoreRequest,
) -> Result<(Vec<f32>, u32)> {
    if let Some(audio_bytes) = request.audio_bytes.as_deref() {
        return decode_audio_bytes(audio_bytes);
    }

    let audio_b64 = request
        .audio_input
        .as_deref()
        .ok_or_else(|| Error::InvalidInput("Request missing audio input".to_string()))?;
    decode_audio_base64_with_rate(audio_b64)
}

#[cfg(test)]
mod tests {
    use super::NativeExecutor;

    #[test]
    fn next_audio_delta_emits_only_new_tail_samples() {
        let mut emitted = 0usize;
        let all1 = vec![0.1f32, 0.2, 0.3];
        let delta1 = NativeExecutor::next_audio_delta(&all1, &mut emitted);
        assert_eq!(delta1, all1);
        assert_eq!(emitted, 3);

        let all2 = vec![0.1f32, 0.2, 0.3, 0.4, 0.5];
        let delta2 = NativeExecutor::next_audio_delta(&all2, &mut emitted);
        assert_eq!(delta2, vec![0.4, 0.5]);
        assert_eq!(emitted, 5);
    }

    #[test]
    fn next_audio_delta_handles_shorter_redecode_safely() {
        let mut emitted = 5usize;
        let all = vec![1.0f32, 2.0];
        let delta = NativeExecutor::next_audio_delta(&all, &mut emitted);
        assert!(delta.is_empty());
        assert_eq!(emitted, 2);
    }

    #[test]
    fn next_audio_delta_stable_holds_back_tail_until_final() {
        let mut emitted = 0usize;
        let all = vec![0.1f32, 0.2, 0.3, 0.4, 0.5];
        let delta = NativeExecutor::next_audio_delta_stable(&all, &mut emitted, 2, false);
        assert_eq!(delta, vec![0.1, 0.2, 0.3]);
        assert_eq!(emitted, 3);

        let delta_final = NativeExecutor::next_audio_delta_stable(&all, &mut emitted, 2, true);
        assert_eq!(delta_final, vec![0.4, 0.5]);
        assert_eq!(emitted, 5);
    }

    #[test]
    fn next_audio_delta_stable_emits_nothing_when_window_is_unstable() {
        let mut emitted = 0usize;
        let all = vec![0.1f32, 0.2, 0.3];
        let delta = NativeExecutor::next_audio_delta_stable(&all, &mut emitted, 8, false);
        assert!(delta.is_empty());
        assert_eq!(emitted, 0);
    }

    #[test]
    fn streaming_low_latency_chunk_plan_splits_short_audio() {
        let sr = 16_000u32;
        let samples = vec![0.0f32; (sr as usize) * 4];
        let (_cfg, chunks) = NativeExecutor::asr_chunk_plan(&samples, sr, Some(30.0), true);
        assert!(chunks.len() > 1, "expected multiple chunks, got {}", chunks.len());
    }

    #[test]
    fn standard_chunk_plan_keeps_short_audio_single_chunk() {
        let sr = 16_000u32;
        let samples = vec![0.0f32; (sr as usize) * 4];
        let (_cfg, chunks) = NativeExecutor::asr_chunk_plan(&samples, sr, Some(30.0), false);
        assert_eq!(chunks.len(), 1);
    }
}
