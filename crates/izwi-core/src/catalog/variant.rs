//! Model variant capability helpers and parser utilities.

use std::fmt;

use crate::model::ModelVariant;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelFamily {
    Qwen3Tts,
    Qwen3Asr,
    ParakeetAsr,
    Qwen3Chat,
    Gemma3Chat,
    Qwen3ForcedAligner,
    Voxtral,
    Lfm2Audio,
    Tokenizer,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelTask {
    Tts,
    Asr,
    Chat,
    ForcedAlign,
    AudioChat,
    Tokenizer,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InferenceBackendHint {
    CandleNative,
    MlxCandidate,
}

#[derive(Debug, Clone)]
pub struct ParseModelVariantError {
    input: String,
}

impl ParseModelVariantError {
    fn new(input: impl Into<String>) -> Self {
        Self {
            input: input.into(),
        }
    }
}

impl fmt::Display for ParseModelVariantError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Unsupported model identifier: {}",
            self.input.trim().if_empty("<empty>")
        )
    }
}

impl std::error::Error for ParseModelVariantError {}

trait EmptyFallback {
    fn if_empty(self, fallback: &str) -> String;
}

impl EmptyFallback for &str {
    fn if_empty(self, fallback: &str) -> String {
        if self.trim().is_empty() {
            fallback.to_string()
        } else {
            self.to_string()
        }
    }
}

impl ModelVariant {
    pub fn family(&self) -> ModelFamily {
        use ModelVariant::*;

        match self {
            Qwen3Tts12Hz06BBase
            | Qwen3Tts12Hz06BBase4Bit
            | Qwen3Tts12Hz06BBase8Bit
            | Qwen3Tts12Hz06BBaseBf16
            | Qwen3Tts12Hz06BCustomVoice
            | Qwen3Tts12Hz06BCustomVoice4Bit
            | Qwen3Tts12Hz06BCustomVoice8Bit
            | Qwen3Tts12Hz06BCustomVoiceBf16
            | Qwen3Tts12Hz17BBase
            | Qwen3Tts12Hz17BCustomVoice
            | Qwen3Tts12Hz17BVoiceDesign
            | Qwen3Tts12Hz17BVoiceDesign4Bit
            | Qwen3Tts12Hz17BVoiceDesign8Bit
            | Qwen3Tts12Hz17BVoiceDesignBf16 => ModelFamily::Qwen3Tts,
            Qwen3TtsTokenizer12Hz => ModelFamily::Tokenizer,
            Lfm2Audio15B => ModelFamily::Lfm2Audio,
            Qwen3Asr06B | Qwen3Asr06B4Bit | Qwen3Asr06B8Bit | Qwen3Asr06BBf16 | Qwen3Asr17B
            | Qwen3Asr17B4Bit | Qwen3Asr17B8Bit | Qwen3Asr17BBf16 => ModelFamily::Qwen3Asr,
            ParakeetTdt06BV2 | ParakeetTdt06BV3 => ModelFamily::ParakeetAsr,
            Qwen306B4Bit => ModelFamily::Qwen3Chat,
            Gemma31BIt | Gemma34BIt => ModelFamily::Gemma3Chat,
            Qwen3ForcedAligner06B => ModelFamily::Qwen3ForcedAligner,
            VoxtralMini4BRealtime2602 => ModelFamily::Voxtral,
        }
    }

    pub fn primary_task(&self) -> ModelTask {
        match self.family() {
            ModelFamily::Qwen3Tts => ModelTask::Tts,
            ModelFamily::Qwen3Asr | ModelFamily::ParakeetAsr => ModelTask::Asr,
            ModelFamily::Qwen3Chat | ModelFamily::Gemma3Chat => ModelTask::Chat,
            ModelFamily::Qwen3ForcedAligner => ModelTask::ForcedAlign,
            ModelFamily::Voxtral => ModelTask::AudioChat,
            ModelFamily::Lfm2Audio => ModelTask::AudioChat,
            ModelFamily::Tokenizer => ModelTask::Tokenizer,
        }
    }

    pub fn backend_hint(&self) -> InferenceBackendHint {
        if self.repo_id().starts_with("mlx-community/") {
            InferenceBackendHint::MlxCandidate
        } else {
            InferenceBackendHint::CandleNative
        }
    }
}

pub fn parse_model_variant(input: &str) -> Result<ModelVariant, ParseModelVariantError> {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return Err(ParseModelVariantError::new(input));
    }

    let normalized = normalize_identifier(trimmed);

    if let Some(found) = ModelVariant::all()
        .iter()
        .copied()
        .find(|variant| matches_variant_alias(*variant, trimmed, &normalized))
    {
        return Ok(found);
    }

    resolve_by_heuristic(&normalized).ok_or_else(|| ParseModelVariantError::new(input))
}

pub fn parse_tts_model_variant(input: &str) -> Result<ModelVariant, ParseModelVariantError> {
    let variant = parse_model_variant(input)?;
    if variant.is_tts() || variant.is_lfm2() {
        Ok(variant)
    } else {
        Err(ParseModelVariantError::new(input))
    }
}

pub fn parse_chat_model_variant(
    input: Option<&str>,
) -> Result<ModelVariant, ParseModelVariantError> {
    match input.unwrap_or("Qwen3-0.6B-4bit") {
        id => {
            let variant = parse_model_variant(id)?;
            if variant.is_chat() {
                Ok(variant)
            } else {
                Err(ParseModelVariantError::new(id))
            }
        }
    }
}

pub fn resolve_asr_model_variant(input: Option<&str>) -> ModelVariant {
    use ModelVariant::*;

    let Some(raw) = input else {
        return Qwen3Asr06B;
    };

    match parse_model_variant(raw) {
        Ok(variant) if variant.is_asr() || variant.is_voxtral() || variant.is_lfm2() => variant,
        Ok(_) => Qwen3Asr06B,
        Err(_) => {
            let normalized = normalize_identifier(raw);
            if normalized.contains("voxtral") {
                VoxtralMini4BRealtime2602
            } else if normalized.contains("lfm2") && normalized.contains("audio") {
                Lfm2Audio15B
            } else if normalized.contains("parakeet") {
                if normalized.contains("v3") {
                    ParakeetTdt06BV3
                } else {
                    ParakeetTdt06BV2
                }
            } else if normalized.contains("17") {
                Qwen3Asr17B
            } else {
                Qwen3Asr06B
            }
        }
    }
}

fn resolve_by_heuristic(normalized: &str) -> Option<ModelVariant> {
    use ModelVariant::*;

    if normalized.contains("voxtral") {
        return Some(VoxtralMini4BRealtime2602);
    }

    if normalized.contains("parakeet") && normalized.contains("tdt") {
        if normalized.contains("v3") {
            return Some(ParakeetTdt06BV3);
        }
        return Some(ParakeetTdt06BV2);
    }

    if normalized.contains("forcedaligner") {
        return Some(Qwen3ForcedAligner06B);
    }

    if normalized.contains("qwen3") && normalized.contains("asr") {
        let is_17b = normalized.contains("17b") || normalized.contains("17");
        let q4 = normalized.contains("4bit") || normalized.contains("int4");
        let q8 = normalized.contains("8bit") || normalized.contains("int8");
        let bf16 = normalized.contains("bf16") || normalized.contains("bfloat16");

        return Some(match (is_17b, q4, q8, bf16) {
            (true, true, _, _) => Qwen3Asr17B4Bit,
            (true, _, true, _) => Qwen3Asr17B8Bit,
            (true, _, _, true) => Qwen3Asr17BBf16,
            (true, _, _, _) => Qwen3Asr17B,
            (false, true, _, _) => Qwen3Asr06B4Bit,
            (false, _, true, _) => Qwen3Asr06B8Bit,
            (false, _, _, true) => Qwen3Asr06BBf16,
            (false, _, _, _) => Qwen3Asr06B,
        });
    }

    if normalized.contains("qwen3") && normalized.contains("tts") {
        let is_17b = normalized.contains("17b") || normalized.contains("17");
        let q4 = normalized.contains("4bit") || normalized.contains("int4");
        let q8 = normalized.contains("8bit") || normalized.contains("int8");
        let bf16 = normalized.contains("bf16") || normalized.contains("bfloat16");

        if normalized.contains("tokenizer") {
            return Some(Qwen3TtsTokenizer12Hz);
        }

        if is_17b && normalized.contains("voicedesign") {
            return Some(match (q4, q8, bf16) {
                (true, _, _) => Qwen3Tts12Hz17BVoiceDesign4Bit,
                (_, true, _) => Qwen3Tts12Hz17BVoiceDesign8Bit,
                (_, _, true) => Qwen3Tts12Hz17BVoiceDesignBf16,
                _ => Qwen3Tts12Hz17BVoiceDesign,
            });
        }

        if is_17b && normalized.contains("customvoice") {
            return Some(Qwen3Tts12Hz17BCustomVoice);
        }

        if is_17b {
            return Some(Qwen3Tts12Hz17BBase);
        }

        if normalized.contains("customvoice") {
            return Some(match (q4, q8, bf16) {
                (true, _, _) => Qwen3Tts12Hz06BCustomVoice4Bit,
                (_, true, _) => Qwen3Tts12Hz06BCustomVoice8Bit,
                (_, _, true) => Qwen3Tts12Hz06BCustomVoiceBf16,
                _ => Qwen3Tts12Hz06BCustomVoice,
            });
        }

        return Some(match (q4, q8, bf16) {
            (true, _, _) => Qwen3Tts12Hz06BBase4Bit,
            (_, true, _) => Qwen3Tts12Hz06BBase8Bit,
            (_, _, true) => Qwen3Tts12Hz06BBaseBf16,
            _ => Qwen3Tts12Hz06BBase,
        });
    }

    if normalized.contains("qwen3") && normalized.contains("06b") && normalized.contains("4bit") {
        return Some(Qwen306B4Bit);
    }

    if normalized.contains("gemma3") || (normalized.contains("gemma") && normalized.contains("it"))
    {
        if normalized.contains("1b") {
            return Some(Gemma31BIt);
        }
        if normalized.contains("4b") {
            return Some(Gemma34BIt);
        }
    }

    if normalized.contains("lfm2") && normalized.contains("audio") {
        return Some(Lfm2Audio15B);
    }

    None
}

fn matches_variant_alias(variant: ModelVariant, raw: &str, normalized: &str) -> bool {
    let repo = variant.repo_id();
    let repo_tail = repo.rsplit('/').next().unwrap_or(repo);

    let aliases = [
        variant.dir_name(),
        variant.repo_id(),
        repo_tail,
        variant.display_name(),
    ];

    if aliases
        .iter()
        .any(|alias| normalize_identifier(alias) == normalized)
    {
        return true;
    }

    let maybe_compact = raw
        .trim()
        .replace('_', "-")
        .replace(' ', "-")
        .to_ascii_lowercase();

    variant.dir_name().eq_ignore_ascii_case(&maybe_compact)
        || variant.repo_id().eq_ignore_ascii_case(&maybe_compact)
}

fn normalize_identifier(input: &str) -> String {
    input
        .chars()
        .filter(|ch| ch.is_ascii_alphanumeric())
        .map(|ch| ch.to_ascii_lowercase())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_by_repo_tail() {
        let parsed = parse_model_variant("Qwen3-ASR-0.6B").unwrap();
        assert_eq!(parsed, ModelVariant::Qwen3Asr06B);
    }

    #[test]
    fn parse_by_display_name() {
        let parsed = parse_model_variant("Qwen3-TTS 0.6B Base 4-bit").unwrap();
        assert_eq!(parsed, ModelVariant::Qwen3Tts12Hz06BBase4Bit);
    }

    #[test]
    fn parse_tts_rejects_non_tts() {
        assert!(parse_tts_model_variant("Qwen3-ASR-0.6B").is_err());
    }

    #[test]
    fn resolve_asr_fallback_defaults_to_06b() {
        let resolved = resolve_asr_model_variant(Some("not-a-real-model"));
        assert_eq!(resolved, ModelVariant::Qwen3Asr06B);
    }

    #[test]
    fn parse_tts_accepts_lfm2_audio() {
        let parsed = parse_tts_model_variant("LFM2-Audio-1.5B").unwrap();
        assert_eq!(parsed, ModelVariant::Lfm2Audio15B);
    }

    #[test]
    fn resolve_asr_accepts_lfm2_audio() {
        let resolved = resolve_asr_model_variant(Some("LFM2-Audio-1.5B"));
        assert_eq!(resolved, ModelVariant::Lfm2Audio15B);
    }

    #[test]
    fn parse_gemma_by_repo_tail() {
        let parsed = parse_model_variant("gemma-3-4b-it").unwrap();
        assert_eq!(parsed, ModelVariant::Gemma34BIt);
    }

    #[test]
    fn parse_chat_accepts_gemma() {
        let parsed = parse_chat_model_variant(Some("google/gemma-3-1b-it")).unwrap();
        assert_eq!(parsed, ModelVariant::Gemma31BIt);
    }

    #[test]
    fn parse_parakeet_by_repo_tail() {
        let parsed = parse_model_variant("parakeet-tdt-0.6b-v3").unwrap();
        assert_eq!(parsed, ModelVariant::ParakeetTdt06BV3);
    }

    #[test]
    fn resolve_asr_accepts_parakeet() {
        let resolved = resolve_asr_model_variant(Some("nvidia/parakeet-tdt-0.6b-v2"));
        assert_eq!(resolved, ModelVariant::ParakeetTdt06BV2);
    }
}
