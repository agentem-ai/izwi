//! Diarization runtime methods.

use crate::catalog::{
    parse_chat_model_variant, resolve_asr_model_variant, resolve_diarization_model_variant,
};
use crate::error::{Error, Result};
use crate::runtime::audio_io::{base64_decode, decode_wav_bytes};
use crate::runtime::service::RuntimeService;
use crate::runtime::types::{
    DiarizationConfig, DiarizationResult, DiarizationSegment, DiarizationTranscriptResult,
    DiarizationUtterance, DiarizationWord,
};
use crate::{models::chat_types::ChatMessage, models::chat_types::ChatRole, ModelVariant};
use tracing::warn;

const UNKNOWN_SPEAKER: &str = "UNKNOWN";
const MAX_UTTERANCE_GAP_SECS: f32 = 0.9;

impl RuntimeService {
    /// Run speaker diarization over a single audio input.
    pub async fn diarize(
        &self,
        audio_base64: &str,
        model_id: Option<&str>,
        config: &DiarizationConfig,
    ) -> Result<DiarizationResult> {
        let (samples, sample_rate) = decode_wav_bytes(&base64_decode(audio_base64)?)?;
        let variant = resolve_diarization_model_variant(model_id);
        self.load_model(variant).await?;

        let model = self
            .model_registry
            .get_diarization(variant)
            .await
            .ok_or_else(|| Error::ModelNotFound(variant.to_string()))?;

        model.diarize(&samples, sample_rate, config)
    }

    /// Run diarization and produce speaker-attributed transcript artifacts.
    pub async fn diarize_with_transcript(
        &self,
        audio_base64: &str,
        diarization_model_id: Option<&str>,
        asr_model_id: Option<&str>,
        aligner_model_id: Option<&str>,
        llm_model_id: Option<&str>,
        config: &DiarizationConfig,
        enable_llm_refinement: bool,
    ) -> Result<DiarizationTranscriptResult> {
        let diarization = self
            .diarize(audio_base64, diarization_model_id, config)
            .await?;

        let asr_variant = resolve_asr_model_variant(asr_model_id);
        let asr = self
            .asr_transcribe(audio_base64, Some(asr_variant.dir_name()), None)
            .await?;
        let asr_text = asr.text.trim().to_string();

        let alignments = if asr_text.is_empty() {
            Vec::new()
        } else {
            match self
                .force_align_with_model(audio_base64, &asr_text, aligner_model_id)
                .await
            {
                Ok(aligned) if !aligned.is_empty() => aligned,
                Ok(_) => fallback_word_timings(&asr_text, diarization.duration_secs),
                Err(err) => {
                    warn!("Forced alignment failed, falling back to uniform word timing: {err}");
                    fallback_word_timings(&asr_text, diarization.duration_secs)
                }
            }
        };

        let (words, overlap_assigned_words, unattributed_words) =
            attribute_words_to_speakers(&alignments, &diarization.segments);
        let utterances = build_utterances(&words);
        let raw_transcript = if utterances.is_empty() {
            asr_text.clone()
        } else {
            format_utterance_transcript(&utterances)
        };

        let mut transcript = raw_transcript.clone();
        let mut llm_refined = false;
        if enable_llm_refinement && !raw_transcript.trim().is_empty() {
            let llm_variant = resolve_chat_variant(llm_model_id)?;
            match self
                .polish_diarized_transcript(llm_variant, &raw_transcript)
                .await
            {
                Ok(polished) if !polished.trim().is_empty() => {
                    transcript = polished.trim().to_string();
                    llm_refined = true;
                }
                Ok(_) => {}
                Err(err) => {
                    warn!("Transcript refinement failed, returning raw speaker transcript: {err}");
                }
            }
        }

        let alignment_coverage = if words.is_empty() {
            0.0
        } else {
            overlap_assigned_words as f32 / words.len() as f32
        };

        Ok(DiarizationTranscriptResult {
            segments: diarization.segments,
            words,
            utterances,
            asr_text,
            raw_transcript,
            transcript,
            duration_secs: diarization.duration_secs,
            speaker_count: diarization.speaker_count,
            alignment_coverage,
            unattributed_words,
            llm_refined,
        })
    }

    async fn polish_diarized_transcript(
        &self,
        llm_variant: ModelVariant,
        raw_transcript: &str,
    ) -> Result<String> {
        let messages = vec![
            ChatMessage {
                role: ChatRole::System,
                content: "You are a transcript editor. Improve punctuation and readability only. Do not invent content, do not change speaker labels, and do not change timestamps.".to_string(),
            },
            ChatMessage {
                role: ChatRole::User,
                content: format!(
                    "Rewrite the following transcript with minimal edits. Keep one line per utterance and preserve each leading speaker label + timestamp prefix exactly as-is.\n\n{}",
                    raw_transcript
                ),
            },
        ];

        let generation = self.chat_generate(llm_variant, messages, 1024).await?;
        Ok(generation.text)
    }
}

fn resolve_chat_variant(model_id: Option<&str>) -> Result<ModelVariant> {
    parse_chat_model_variant(model_id).map_err(|err| Error::InvalidInput(err.to_string()))
}

fn fallback_word_timings(text: &str, duration_secs: f32) -> Vec<(String, u32, u32)> {
    let words = extract_words(text);
    if words.is_empty() {
        return Vec::new();
    }

    let max_duration_ms = ((duration_secs.max(0.0)) * 1000.0).round() as u32;
    let duration_ms = if max_duration_ms > 0 {
        max_duration_ms
    } else {
        (words.len() as u32).saturating_mul(300).max(1)
    };
    let step = (duration_ms as f32 / words.len() as f32).max(1.0);

    words
        .into_iter()
        .enumerate()
        .map(|(idx, word)| {
            let start = ((idx as f32) * step).round() as u32;
            let mut end = (((idx + 1) as f32) * step).round() as u32;
            if end <= start {
                end = start.saturating_add(1);
            }
            (word, start, end)
        })
        .collect()
}

fn extract_words(text: &str) -> Vec<String> {
    text.split_whitespace()
        .map(|word| {
            word.trim_matches(|ch: char| !ch.is_alphanumeric() && ch != '\'' && ch != '-')
                .to_string()
        })
        .filter(|word| !word.is_empty())
        .collect()
}

fn attribute_words_to_speakers(
    alignments: &[(String, u32, u32)],
    segments: &[DiarizationSegment],
) -> (Vec<DiarizationWord>, usize, usize) {
    let mut words = Vec::new();
    let mut overlap_assigned_words = 0usize;
    let mut unattributed_words = 0usize;

    for (word, start_ms, end_ms) in alignments {
        let cleaned = word.trim();
        if cleaned.is_empty() {
            continue;
        }

        let start_secs = (*start_ms as f32 / 1000.0).max(0.0);
        let mut end_secs = (*end_ms as f32 / 1000.0).max(start_secs + 0.001);
        if !end_secs.is_finite() {
            end_secs = start_secs + 0.001;
        }

        let (speaker, speaker_confidence, overlaps_segment) =
            assign_speaker_for_span(start_secs, end_secs, segments);
        if overlaps_segment {
            overlap_assigned_words += 1;
        } else {
            unattributed_words += 1;
        }

        words.push(DiarizationWord {
            word: cleaned.to_string(),
            speaker,
            start_secs,
            end_secs,
            speaker_confidence,
            overlaps_segment,
        });
    }

    words.sort_by(|a, b| a.start_secs.total_cmp(&b.start_secs));
    (words, overlap_assigned_words, unattributed_words)
}

fn assign_speaker_for_span(
    start_secs: f32,
    end_secs: f32,
    segments: &[DiarizationSegment],
) -> (String, Option<f32>, bool) {
    if segments.is_empty() {
        return (UNKNOWN_SPEAKER.to_string(), None, false);
    }

    let mut best_overlap = 0.0f32;
    let mut best_segment: Option<&DiarizationSegment> = None;
    for segment in segments {
        let overlap = interval_overlap(start_secs, end_secs, segment.start_secs, segment.end_secs);
        if overlap > best_overlap {
            best_overlap = overlap;
            best_segment = Some(segment);
        }
    }

    if let Some(segment) = best_segment.filter(|_| best_overlap > 0.0) {
        return (segment.speaker.clone(), segment.confidence, true);
    }

    let midpoint = (start_secs + end_secs) * 0.5;
    let nearest = segments
        .iter()
        .min_by(|left, right| {
            span_distance(midpoint, left.start_secs, left.end_secs).total_cmp(&span_distance(
                midpoint,
                right.start_secs,
                right.end_secs,
            ))
        })
        .expect("segments checked non-empty");

    (nearest.speaker.clone(), nearest.confidence, false)
}

fn span_distance(point: f32, start: f32, end: f32) -> f32 {
    if point < start {
        start - point
    } else if point > end {
        point - end
    } else {
        0.0
    }
}

fn interval_overlap(a_start: f32, a_end: f32, b_start: f32, b_end: f32) -> f32 {
    (a_end.min(b_end) - a_start.max(b_start)).max(0.0)
}

fn build_utterances(words: &[DiarizationWord]) -> Vec<DiarizationUtterance> {
    if words.is_empty() {
        return Vec::new();
    }

    let mut utterances = Vec::new();
    let mut current = DiarizationUtterance {
        speaker: words[0].speaker.clone(),
        start_secs: words[0].start_secs,
        end_secs: words[0].end_secs,
        text: words[0].word.clone(),
        word_start: 0,
        word_end: 0,
    };

    for (idx, word) in words.iter().enumerate().skip(1) {
        let gap = (word.start_secs - current.end_secs).max(0.0);
        let same_speaker = word.speaker == current.speaker;

        if same_speaker && gap <= MAX_UTTERANCE_GAP_SECS {
            append_token(&mut current.text, &word.word);
            current.end_secs = current.end_secs.max(word.end_secs);
            current.word_end = idx;
            continue;
        }

        utterances.push(current);
        current = DiarizationUtterance {
            speaker: word.speaker.clone(),
            start_secs: word.start_secs,
            end_secs: word.end_secs,
            text: word.word.clone(),
            word_start: idx,
            word_end: idx,
        };
    }

    utterances.push(current);
    utterances
}

fn append_token(target: &mut String, token: &str) {
    if target.is_empty() {
        target.push_str(token);
        return;
    }

    let punct_only = token
        .chars()
        .all(|ch| !ch.is_alphanumeric() && ch != '\'' && ch != '-');
    if punct_only {
        target.push_str(token);
    } else {
        target.push(' ');
        target.push_str(token);
    }
}

fn format_utterance_transcript(utterances: &[DiarizationUtterance]) -> String {
    utterances
        .iter()
        .map(|utterance| {
            format!(
                "{} [{:.2}s - {:.2}s]: {}",
                utterance.speaker, utterance.start_secs, utterance.end_secs, utterance.text
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fallback_word_timings_generates_monotonic_ranges() {
        let timings = fallback_word_timings("hello world from test", 2.0);
        assert_eq!(timings.len(), 4);
        for (idx, (_, start, end)) in timings.iter().enumerate() {
            assert!(end > start, "word {} should have positive duration", idx);
            if idx > 0 {
                assert!(*start >= timings[idx - 1].1);
            }
        }
    }

    #[test]
    fn attribution_prefers_overlap_then_nearest() {
        let segments = vec![
            DiarizationSegment {
                speaker: "SPEAKER_00".to_string(),
                start_secs: 0.0,
                end_secs: 1.0,
                confidence: Some(0.9),
            },
            DiarizationSegment {
                speaker: "SPEAKER_01".to_string(),
                start_secs: 2.0,
                end_secs: 3.0,
                confidence: Some(0.8),
            },
        ];

        let aligned = vec![
            ("hello".to_string(), 100, 400),
            ("there".to_string(), 1200, 1300),
            ("friend".to_string(), 2400, 2800),
        ];

        let (words, overlap_count, unattributed) = attribute_words_to_speakers(&aligned, &segments);
        assert_eq!(words.len(), 3);
        assert_eq!(words[0].speaker, "SPEAKER_00");
        assert_eq!(words[2].speaker, "SPEAKER_01");
        assert_eq!(overlap_count, 2);
        assert_eq!(unattributed, 1);
    }

    #[test]
    fn build_utterances_merges_small_gaps_for_same_speaker() {
        let words = vec![
            DiarizationWord {
                word: "hello".to_string(),
                speaker: "SPEAKER_00".to_string(),
                start_secs: 0.0,
                end_secs: 0.4,
                speaker_confidence: None,
                overlaps_segment: true,
            },
            DiarizationWord {
                word: "world".to_string(),
                speaker: "SPEAKER_00".to_string(),
                start_secs: 0.5,
                end_secs: 0.8,
                speaker_confidence: None,
                overlaps_segment: true,
            },
            DiarizationWord {
                word: "next".to_string(),
                speaker: "SPEAKER_01".to_string(),
                start_secs: 1.2,
                end_secs: 1.6,
                speaker_confidence: None,
                overlaps_segment: true,
            },
        ];

        let utterances = build_utterances(&words);
        assert_eq!(utterances.len(), 2);
        assert_eq!(utterances[0].speaker, "SPEAKER_00");
        assert!(utterances[0].text.contains("hello world"));
        assert_eq!(utterances[1].speaker, "SPEAKER_01");
    }
}
