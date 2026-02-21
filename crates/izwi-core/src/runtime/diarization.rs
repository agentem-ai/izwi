//! Diarization runtime methods.

use crate::catalog::{
    resolve_asr_model_variant, resolve_diarization_llm_variant, resolve_diarization_model_variant,
};
use crate::error::{Error, Result};
use crate::models::shared::chat::{ChatMessage, ChatRole};
use crate::runtime::audio_io::{base64_decode, decode_audio_bytes};
use crate::runtime::service::RuntimeService;
use crate::runtime::types::{
    DiarizationConfig, DiarizationResult, DiarizationSegment, DiarizationTranscriptResult,
    DiarizationUtterance, DiarizationWord,
};
use crate::ModelVariant;
use std::collections::HashMap;
use tracing::warn;

const UNKNOWN_SPEAKER: &str = "UNKNOWN";
const MAX_UTTERANCE_GAP_SECS: f32 = 0.9;
const MIN_ALIGNMENT_COVERAGE: f32 = 0.25;
const ALIGNMENT_COLLAPSE_TAIL_MS: u32 = 250;

impl RuntimeService {
    /// Run speaker diarization over a single audio input.
    pub async fn diarize(
        &self,
        audio_base64: &str,
        model_id: Option<&str>,
        config: &DiarizationConfig,
    ) -> Result<DiarizationResult> {
        let (samples, sample_rate) = decode_audio_bytes(&base64_decode(audio_base64)?)?;
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
        let asr_words = extract_words(&asr_text);

        let mut alignments = if asr_text.is_empty() {
            Vec::new()
        } else {
            match self
                .force_align_with_model(audio_base64, &asr_text, aligner_model_id)
                .await
            {
                Ok(aligned) if !aligned.is_empty() => aligned,
                Ok(_) => fallback_word_timings_with_segments(
                    &asr_words,
                    &diarization.segments,
                    diarization.duration_secs,
                ),
                Err(err) => {
                    warn!(
                        "Forced alignment failed, falling back to diarization-guided word timing: {err}"
                    );
                    fallback_word_timings_with_segments(
                        &asr_words,
                        &diarization.segments,
                        diarization.duration_secs,
                    )
                }
            }
        };

        if alignment_is_suspicious(&alignments, asr_words.len(), diarization.duration_secs) {
            warn!(
                "Forced aligner output looked invalid, using diarization-guided fallback timings"
            );
            alignments = fallback_word_timings_with_segments(
                &asr_words,
                &diarization.segments,
                diarization.duration_secs,
            );
        }

        let (mut words, mut overlap_assigned_words, mut unattributed_words) =
            attribute_words_to_speakers(&alignments, &diarization.segments);

        if !words.is_empty() {
            let coverage = overlap_assigned_words as f32 / words.len() as f32;
            if coverage < MIN_ALIGNMENT_COVERAGE {
                warn!(
                    "Alignment coverage too low ({:.1}%), retrying with diarization-guided fallback timings",
                    coverage * 100.0
                );
                alignments = fallback_word_timings_with_segments(
                    &asr_words,
                    &diarization.segments,
                    diarization.duration_secs,
                );
                let (fallback_words, fallback_overlap_assigned_words, fallback_unattributed_words) =
                    attribute_words_to_speakers(&alignments, &diarization.segments);
                words = fallback_words;
                overlap_assigned_words = fallback_overlap_assigned_words;
                unattributed_words = fallback_unattributed_words;
            }
        }

        let utterances = build_utterances(&words);
        let raw_transcript = if utterances.is_empty() {
            asr_text.clone()
        } else {
            format_utterance_transcript(&utterances)
        };

        let raw_transcript_trimmed = raw_transcript.trim();
        let mut transcript = raw_transcript.clone();
        let mut llm_refined = false;
        if enable_llm_refinement && !raw_transcript_trimmed.is_empty() {
            let llm_variant = resolve_chat_variant(llm_model_id)?;
            match self
                .polish_diarized_transcript(llm_variant, &raw_transcript)
                .await
            {
                Ok(polished) if !polished.trim().is_empty() => {
                    let polished_trimmed = polished.trim();
                    transcript = polished_trimmed.to_string();
                    llm_refined = polished_trimmed != raw_transcript_trimmed;
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
                content: "You are a diarized transcript editor. Return only final transcript lines, with no analysis or hidden reasoning. Never output tags (including <think>), markdown, code fences, or commentary. Keep speaker labels and timestamps exactly unchanged. Keep line count and line order exactly unchanged. Do not invent, repeat, or omit spoken content. Only improve punctuation and readability of spoken text after the colon on each line."
                    .to_string(),
            },
            ChatMessage {
                role: ChatRole::User,
                content: format!(
                    "Rewrite this diarized transcript with minimal edits.\nRules:\n- Keep exactly one output line per input line.\n- Preserve each leading speaker label + timestamp prefix exactly as-is.\n- Edit only the spoken text after the colon.\n- Do not add, remove, or merge lines.\n- Do not invent new words, drop spoken words, or repeat content not present in the line.\n- Output only the final transcript lines.\n\n{}",
                    raw_transcript
                ),
            },
        ];

        let generation = self.chat_generate(llm_variant, messages, 1024).await?;
        Ok(sanitize_refined_transcript(
            &generation.text,
            raw_transcript,
        ))
    }
}

fn resolve_chat_variant(model_id: Option<&str>) -> Result<ModelVariant> {
    resolve_diarization_llm_variant(model_id).map_err(|err| Error::InvalidInput(err.to_string()))
}

fn fallback_word_timings(text: &str, duration_secs: f32) -> Vec<(String, u32, u32)> {
    let words = extract_words(text);
    fallback_word_timings_from_words(&words, duration_secs)
}

fn fallback_word_timings_from_words(
    words: &[String],
    duration_secs: f32,
) -> Vec<(String, u32, u32)> {
    if words.is_empty() {
        return Vec::new();
    }

    let max_duration_ms = secs_to_ms(duration_secs);
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
            (word.clone(), start, end)
        })
        .collect()
}

fn fallback_word_timings_with_segments(
    words: &[String],
    segments: &[DiarizationSegment],
    duration_secs: f32,
) -> Vec<(String, u32, u32)> {
    if words.is_empty() {
        return Vec::new();
    }

    let mut windows = segments
        .iter()
        .filter_map(|segment| {
            let start = secs_to_ms(segment.start_secs);
            let mut end = secs_to_ms(segment.end_secs);
            if end <= start {
                end = start.saturating_add(1);
            }
            (end > start).then_some((start, end))
        })
        .collect::<Vec<_>>();
    windows.sort_by_key(|(start, _)| *start);

    if windows.is_empty() {
        return fallback_word_timings_from_words(words, duration_secs);
    }

    let durations = windows
        .iter()
        .map(|(start, end)| end.saturating_sub(*start).max(1))
        .collect::<Vec<_>>();
    let total_duration_ms = durations.iter().copied().sum::<u32>();
    if total_duration_ms == 0 {
        return fallback_word_timings_from_words(words, duration_secs);
    }

    let word_count = words.len();
    let mut allocations = vec![0usize; windows.len()];
    let mut assigned = 0usize;
    let mut remainders = Vec::with_capacity(windows.len());

    for (idx, duration_ms) in durations.iter().copied().enumerate() {
        let target = word_count as f32 * duration_ms as f32 / total_duration_ms as f32;
        let base = target.floor() as usize;
        allocations[idx] = base;
        assigned += base;
        remainders.push((target - base as f32, idx));
    }

    let mut remaining = word_count.saturating_sub(assigned);
    remainders.sort_by(|left, right| right.0.total_cmp(&left.0).then(left.1.cmp(&right.1)));
    for (_, idx) in remainders {
        if remaining == 0 {
            break;
        }
        allocations[idx] += 1;
        remaining -= 1;
    }

    let allocated_total = allocations.iter().sum::<usize>();
    if allocated_total < word_count {
        if let Some(last) = allocations.last_mut() {
            *last += word_count - allocated_total;
        }
    } else if allocated_total > word_count {
        let mut excess = allocated_total - word_count;
        for allocation in allocations.iter_mut().rev() {
            if excess == 0 {
                break;
            }
            let delta = (*allocation).min(excess);
            *allocation -= delta;
            excess -= delta;
        }
    }

    let mut alignments = Vec::with_capacity(word_count);
    let mut word_idx = 0usize;
    for ((segment_start, segment_end), allocation) in windows.into_iter().zip(allocations) {
        if allocation == 0 {
            continue;
        }
        let segment_span = segment_end.saturating_sub(segment_start).max(1);
        let step = segment_span as f32 / allocation as f32;

        for local_idx in 0..allocation {
            if word_idx >= word_count {
                break;
            }
            let start = segment_start.saturating_add((local_idx as f32 * step).floor() as u32);
            let mut end = if local_idx + 1 == allocation {
                segment_end
            } else {
                segment_start.saturating_add(((local_idx + 1) as f32 * step).floor() as u32)
            };
            if end <= start {
                end = start.saturating_add(1);
            }
            alignments.push((words[word_idx].clone(), start, end));
            word_idx += 1;
        }
    }

    if word_idx < word_count {
        let remaining_words = &words[word_idx..];
        let mut carry = fallback_word_timings_from_words(remaining_words, duration_secs);
        alignments.append(&mut carry);
    }

    alignments
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

fn alignment_is_suspicious(
    alignments: &[(String, u32, u32)],
    expected_word_count: usize,
    duration_secs: f32,
) -> bool {
    if expected_word_count == 0 {
        return false;
    }
    if alignments.is_empty() {
        return true;
    }
    if alignments.len() < expected_word_count.saturating_div(2).max(1) {
        return true;
    }

    let duration_ms = secs_to_ms(duration_secs).max(1);
    let tail_start = duration_ms.saturating_sub(ALIGNMENT_COLLAPSE_TAIL_MS);

    let mut min_start = u32::MAX;
    let mut max_end = 0u32;
    let mut tiny_spans = 0usize;
    let mut tail_heavy = 0usize;

    for (_, start, end) in alignments {
        min_start = min_start.min((*start).min(duration_ms));
        max_end = max_end.max((*end).min(duration_ms));
        if *end <= start.saturating_add(1) {
            tiny_spans += 1;
        }
        if *start >= tail_start {
            tail_heavy += 1;
        }
    }

    let span = max_end.saturating_sub(min_start);
    let len = alignments.len();
    len >= 8
        && (tiny_spans * 10 >= len * 8
            || tail_heavy * 10 >= len * 8
            || span <= (duration_ms / 20).max(1))
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

fn sanitize_refined_transcript(candidate: &str, fallback: &str) -> String {
    let fallback_trimmed = fallback.trim();
    let stripped = strip_tagged_sections(candidate, "<think>", "</think>")
        .replace("```text", "")
        .replace("```", "");

    let candidate_lines = stripped
        .lines()
        .filter_map(extract_utterance_line)
        .collect::<Vec<_>>();
    let fallback_lines = fallback_trimmed
        .lines()
        .filter_map(extract_utterance_line)
        .collect::<Vec<_>>();

    if !candidate_lines.is_empty() {
        if fallback_lines.is_empty() {
            return candidate_lines.join("\n");
        }
        // Accept refined output only when it preserves line count and line headers.
        let structurally_consistent = candidate_lines.len() == fallback_lines.len()
            && candidate_lines.iter().zip(fallback_lines.iter()).all(
                |(candidate_line, fallback_line)| {
                    utterance_prefix(candidate_line) == utterance_prefix(fallback_line)
                        && utterance_text_similarity_ok(candidate_line, fallback_line)
                },
            );
        if structurally_consistent {
            return candidate_lines.join("\n");
        }
    }

    if !fallback_trimmed.is_empty() {
        return fallback_trimmed.to_string();
    }

    stripped.trim().to_string()
}

fn utterance_prefix(line: &str) -> Option<&str> {
    let mut trimmed = line.trim();
    if trimmed.is_empty() {
        return None;
    }
    if let Some(stripped) = trimmed.strip_prefix("- ") {
        trimmed = stripped.trim();
    } else if let Some(stripped) = trimmed.strip_prefix("* ") {
        trimmed = stripped.trim();
    }
    let header_end = trimmed.find(':')?;
    Some(trimmed[..header_end].trim())
}

fn utterance_text_similarity_ok(candidate_line: &str, fallback_line: &str) -> bool {
    let Some((_, candidate_text)) = candidate_line.split_once(':') else {
        return false;
    };
    let Some((_, fallback_text)) = fallback_line.split_once(':') else {
        return false;
    };

    let candidate_words = extract_words(candidate_text)
        .into_iter()
        .map(|word| word.to_ascii_lowercase())
        .collect::<Vec<_>>();
    let fallback_words = extract_words(fallback_text)
        .into_iter()
        .map(|word| word.to_ascii_lowercase())
        .collect::<Vec<_>>();

    if fallback_words.is_empty() {
        return true;
    }
    if candidate_words.is_empty() {
        return false;
    }

    let (recall, precision) = bag_word_overlap(&fallback_words, &candidate_words);
    recall >= 0.75 && precision >= 0.6
}

fn bag_word_overlap(reference_words: &[String], candidate_words: &[String]) -> (f32, f32) {
    if reference_words.is_empty() || candidate_words.is_empty() {
        return (0.0, 0.0);
    }

    let mut counts = HashMap::<&str, usize>::new();
    for word in reference_words {
        *counts.entry(word.as_str()).or_insert(0) += 1;
    }

    let mut common = 0usize;
    for word in candidate_words {
        if let Some(remaining) = counts.get_mut(word.as_str()) {
            if *remaining > 0 {
                *remaining -= 1;
                common += 1;
            }
        }
    }

    let recall = common as f32 / reference_words.len() as f32;
    let precision = common as f32 / candidate_words.len() as f32;
    (recall, precision)
}

fn strip_tagged_sections(input: &str, start_tag: &str, end_tag: &str) -> String {
    let mut output = input.to_string();
    let start_tag = start_tag.to_ascii_lowercase();
    let end_tag = end_tag.to_ascii_lowercase();
    let start_len = start_tag.len();
    let end_len = end_tag.len();

    loop {
        let lowered = output.to_ascii_lowercase();
        let Some(start_idx) = lowered.find(&start_tag) else {
            break;
        };
        let search_from = start_idx.saturating_add(start_len);
        if let Some(end_rel) = lowered[search_from..].find(&end_tag) {
            let end_idx = search_from + end_rel + end_len;
            output.replace_range(start_idx..end_idx, "");
        } else {
            output.replace_range(start_idx..output.len(), "");
            break;
        }
    }

    output
}

fn extract_utterance_line(line: &str) -> Option<String> {
    let mut candidate = line.trim();
    if candidate.is_empty() {
        return None;
    }

    if let Some(stripped) = candidate.strip_prefix("- ") {
        candidate = stripped.trim();
    } else if let Some(stripped) = candidate.strip_prefix("* ") {
        candidate = stripped.trim();
    } else {
        let numeric_end = candidate
            .bytes()
            .take_while(|byte| byte.is_ascii_digit())
            .count();
        if numeric_end > 0 && candidate[numeric_end..].starts_with(". ") {
            candidate = candidate[numeric_end + 2..].trim();
        }
    }

    if is_utterance_line(candidate) {
        Some(candidate.to_string())
    } else {
        None
    }
}

fn is_utterance_line(line: &str) -> bool {
    let trimmed = line.trim();
    let Some(header_end) = trimmed.find("]:") else {
        return false;
    };
    let header = &trimmed[..=header_end];
    let Some(bracket_start) = header.rfind('[') else {
        return false;
    };
    if header[..bracket_start].trim().is_empty() {
        return false;
    }
    let time_range = &header[bracket_start + 1..header.len() - 1];
    let Some((start, end)) = time_range.split_once(" - ") else {
        return false;
    };
    is_seconds_token(start) && is_seconds_token(end)
}

fn is_seconds_token(token: &str) -> bool {
    let Some(value) = token.trim().strip_suffix('s') else {
        return false;
    };
    value
        .parse::<f32>()
        .map(|parsed| parsed.is_finite() && parsed >= 0.0)
        .unwrap_or(false)
}

fn secs_to_ms(value: f32) -> u32 {
    if !value.is_finite() || value <= 0.0 {
        0
    } else {
        (value * 1000.0).round() as u32
    }
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

    #[test]
    fn sanitize_refined_transcript_removes_thinking_and_keeps_lines() {
        let fallback = "SPEAKER_00 [0.00s - 1.00s]: hello there";
        let candidate = r#"
<think>
internal reasoning
</think>
Here is the refined transcript:
- SPEAKER_00 [0.00s - 1.00s]: Hello there.
"#;

        let sanitized = sanitize_refined_transcript(candidate, fallback);
        assert_eq!(sanitized, "SPEAKER_00 [0.00s - 1.00s]: Hello there.");
    }

    #[test]
    fn sanitize_refined_transcript_falls_back_when_no_utterance_lines() {
        let fallback = "SPEAKER_00 [0.00s - 1.00s]: hello there";
        let candidate = "Here is the rewrite with cleaner punctuation.";

        let sanitized = sanitize_refined_transcript(candidate, fallback);
        assert_eq!(sanitized, fallback);
    }

    #[test]
    fn sanitize_refined_transcript_falls_back_on_line_count_mismatch() {
        let fallback = "SPEAKER_00 [0.00s - 1.00s]: hello there";
        let candidate =
            "SPEAKER_00 [0.00s - 1.00s]: Hello there.\nSPEAKER_01 [1.00s - 2.00s]: Extra line";

        let sanitized = sanitize_refined_transcript(candidate, fallback);
        assert_eq!(sanitized, fallback);
    }

    #[test]
    fn sanitize_refined_transcript_falls_back_on_low_similarity() {
        let fallback = "SPEAKER_00 [0.00s - 1.00s]: hello there from class";
        let candidate =
            "SPEAKER_00 [0.00s - 1.00s]: completely new invented content that was never spoken";

        let sanitized = sanitize_refined_transcript(candidate, fallback);
        assert_eq!(sanitized, fallback);
    }

    #[test]
    fn fallback_word_timings_with_segments_places_words_in_windows() {
        let words = vec![
            "one".to_string(),
            "two".to_string(),
            "three".to_string(),
            "four".to_string(),
        ];
        let segments = vec![
            DiarizationSegment {
                speaker: "SPEAKER_00".to_string(),
                start_secs: 0.0,
                end_secs: 1.0,
                confidence: None,
            },
            DiarizationSegment {
                speaker: "SPEAKER_01".to_string(),
                start_secs: 2.0,
                end_secs: 3.0,
                confidence: None,
            },
        ];

        let alignments = fallback_word_timings_with_segments(&words, &segments, 3.0);
        assert_eq!(alignments.len(), 4);
        assert!(alignments[0].1 < 1_000);
        assert!(alignments[1].1 < 1_000);
        assert!(alignments[2].1 >= 2_000);
        assert!(alignments[3].1 >= 2_000);
    }

    #[test]
    fn alignment_is_suspicious_detects_tail_collapse() {
        let alignments = (0..20)
            .map(|idx| (format!("w{idx}"), 27_302u32, 27_303u32))
            .collect::<Vec<_>>();
        assert!(alignment_is_suspicious(&alignments, 20, 27.303));
    }
}
