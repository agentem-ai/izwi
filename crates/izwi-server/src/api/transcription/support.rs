use crate::transcription_store::{
    TranscriptionSegmentRecord, TranscriptionWordRecord,
};

const MAX_SEGMENT_WORDS: usize = 18;
const MAX_SEGMENT_CHARS: usize = 140;
const MAX_SEGMENT_DURATION_SECS: f32 = 7.0;
const HARD_GAP_SECS: f32 = 0.85;

fn normalize_word(raw: &str) -> String {
    raw.trim().to_string()
}

fn ends_sentence(word: &str) -> bool {
    matches!(word.chars().last(), Some('.' | '!' | '?'))
}

fn join_words(words: &[TranscriptionWordRecord]) -> String {
    let mut out = String::new();
    for word in words {
        let token = word.word.trim();
        if token.is_empty() {
            continue;
        }

        let should_prefix_space = !out.is_empty()
            && !matches!(
                token.chars().next(),
                Some(',' | '.' | '!' | '?' | ':' | ';' | ')' | ']' | '}')
            )
            && !out.ends_with('(')
            && !out.ends_with('[')
            && !out.ends_with('{')
            && !out.ends_with('\'')
            && !out.ends_with('"');

        if should_prefix_space {
            out.push(' ');
        }

        out.push_str(token);
    }
    out.trim().to_string()
}

fn fallback_segments(text: &str, duration_secs: f32) -> Vec<TranscriptionSegmentRecord> {
    let normalized = text.trim();
    if normalized.is_empty() {
        return Vec::new();
    }

    vec![TranscriptionSegmentRecord {
        start_secs: 0.0,
        end_secs: duration_secs.max(0.1),
        text: normalized.to_string(),
        word_start: 0,
        word_end: 0,
    }]
}

pub(crate) fn build_words(
    aligned_words: &[(String, u32, u32)],
) -> Vec<TranscriptionWordRecord> {
    aligned_words
        .iter()
        .filter_map(|(word, start_ms, end_ms)| {
            let normalized = normalize_word(word);
            if normalized.is_empty() || end_ms <= start_ms {
                return None;
            }

            Some(TranscriptionWordRecord {
                word: normalized,
                start_secs: *start_ms as f32 / 1000.0,
                end_secs: *end_ms as f32 / 1000.0,
            })
        })
        .collect()
}

pub(crate) fn build_segments(
    words: &[TranscriptionWordRecord],
    fallback_text: &str,
    duration_secs: f32,
) -> Vec<TranscriptionSegmentRecord> {
    if words.is_empty() {
        return fallback_segments(fallback_text, duration_secs);
    }

    let mut segments = Vec::new();
    let mut start_index = 0usize;
    let mut char_count = 0usize;

    for (index, word) in words.iter().enumerate() {
        char_count += word.word.chars().count() + usize::from(index > start_index);
        let current_len = index + 1 - start_index;
        let current_duration = word.end_secs - words[start_index].start_secs;
        let next_gap = words
            .get(index + 1)
            .map(|next| (next.start_secs - word.end_secs).max(0.0))
            .unwrap_or_default();

        let should_flush = index + 1 == words.len()
            || (ends_sentence(&word.word) && current_len >= 4)
            || (next_gap >= HARD_GAP_SECS && current_len >= 2)
            || char_count >= MAX_SEGMENT_CHARS
            || current_len >= MAX_SEGMENT_WORDS
            || current_duration >= MAX_SEGMENT_DURATION_SECS;

        if !should_flush {
            continue;
        }

        let slice = &words[start_index..=index];
        segments.push(TranscriptionSegmentRecord {
            start_secs: slice.first().map(|item| item.start_secs).unwrap_or_default(),
            end_secs: slice.last().map(|item| item.end_secs).unwrap_or_default(),
            text: join_words(slice),
            word_start: start_index,
            word_end: index + 1,
        });

        start_index = index + 1;
        char_count = 0;
    }

    if segments.is_empty() {
        fallback_segments(fallback_text, duration_secs)
    } else {
        segments
    }
}

pub(crate) fn build_transcript_text(
    segments: &[TranscriptionSegmentRecord],
    fallback_text: &str,
) -> String {
    let text = segments
        .iter()
        .map(|segment| segment.text.trim())
        .filter(|segment| !segment.is_empty())
        .collect::<Vec<_>>()
        .join("\n\n");

    if text.is_empty() {
        fallback_text.trim().to_string()
    } else {
        text
    }
}

fn format_srt_timestamp(total_seconds: f32) -> String {
    let clamped = total_seconds.max(0.0);
    let hours = (clamped / 3600.0).floor() as u32;
    let minutes = ((clamped % 3600.0) / 60.0).floor() as u32;
    let seconds = (clamped % 60.0).floor() as u32;
    let milliseconds = ((clamped - clamped.floor()) * 1000.0).round() as u32;

    format!(
        "{:02}:{:02}:{:02},{:03}",
        hours, minutes, seconds, milliseconds
    )
}

fn format_vtt_timestamp(total_seconds: f32) -> String {
    format_srt_timestamp(total_seconds).replace(',', ".")
}

pub(crate) fn format_srt(
    segments: &[TranscriptionSegmentRecord],
    fallback_text: &str,
    duration_secs: f32,
) -> String {
    let cues = if segments.is_empty() {
        fallback_segments(fallback_text, duration_secs)
    } else {
        segments.to_vec()
    };

    cues.iter()
        .enumerate()
        .map(|(index, segment)| {
            format!(
                "{}\n{} --> {}\n{}",
                index + 1,
                format_srt_timestamp(segment.start_secs),
                format_srt_timestamp(segment.end_secs.max(segment.start_secs + 0.1)),
                segment.text.trim()
            )
        })
        .collect::<Vec<_>>()
        .join("\n\n")
}

pub(crate) fn format_vtt(
    segments: &[TranscriptionSegmentRecord],
    fallback_text: &str,
    duration_secs: f32,
) -> String {
    let cues = if segments.is_empty() {
        fallback_segments(fallback_text, duration_secs)
    } else {
        segments.to_vec()
    };

    let lines = cues
        .iter()
        .map(|segment| {
            format!(
                "{} --> {}\n{}",
                format_vtt_timestamp(segment.start_secs),
                format_vtt_timestamp(segment.end_secs.max(segment.start_secs + 0.1)),
                segment.text.trim()
            )
        })
        .collect::<Vec<_>>()
        .join("\n\n");

    format!("WEBVTT\n\n{}", lines)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builds_segments_from_aligned_words() {
        let words = build_words(&[
            ("Hello".to_string(), 0, 320),
            ("world.".to_string(), 330, 710),
            ("A".to_string(), 1600, 1720),
            ("second".to_string(), 1730, 1910),
            ("sentence.".to_string(), 1920, 2280),
        ]);

        let segments = build_segments(&words, "Hello world. A second sentence.", 2.4);

        assert_eq!(segments.len(), 2);
        assert_eq!(segments[0].text, "Hello world.");
        assert_eq!(segments[1].text, "A second sentence.");
    }

    #[test]
    fn formats_subtitles_from_segments() {
        let segments = vec![TranscriptionSegmentRecord {
            start_secs: 0.0,
            end_secs: 1.4,
            text: "Hello world.".to_string(),
            word_start: 0,
            word_end: 2,
        }];

        assert!(format_srt(&segments, "", 1.4).contains("00:00:00,000 --> 00:00:01,400"));
        assert!(format_vtt(&segments, "", 1.4).starts_with("WEBVTT"));
    }
}
