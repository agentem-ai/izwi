//! Native Rust LFM2-Audio model metadata and runtime policy.
//!
//! This module intentionally keeps LFM2-specific behavior isolated from
//! Qwen/Voxtral/Parakeet model files while remaining fully Rust-native.

use std::path::{Path, PathBuf};

use tracing::info;

use crate::catalog::parse_model_variant;
use crate::error::{Error, Result};
use crate::model::ModelVariant;

pub const LFM2_DEFAULT_S2S_PROMPT: &str = "Respond with interleaved text and audio.";

const TTS_US_MALE_PROMPT: &str = "Perform TTS. Use the US male voice.";
const TTS_US_FEMALE_PROMPT: &str = "Perform TTS. Use the US female voice.";
const TTS_UK_MALE_PROMPT: &str = "Perform TTS. Use the UK male voice.";
const TTS_UK_FEMALE_PROMPT: &str = "Perform TTS. Use the UK female voice.";

const ASR_FALLBACK_ENV: &str = "IZWI_LFM2_NATIVE_ASR_VARIANT";
const TTS_FALLBACK_ENV: &str = "IZWI_LFM2_NATIVE_TTS_VARIANT";
const CHAT_FALLBACK_ENV: &str = "IZWI_LFM2_NATIVE_CHAT_VARIANT";

#[derive(Debug)]
pub struct Lfm2AudioModel {
    model_dir: PathBuf,
    asr_fallback_variant: ModelVariant,
    tts_fallback_variant: ModelVariant,
    chat_fallback_variant: ModelVariant,
}

impl Lfm2AudioModel {
    pub fn load(model_dir: &Path) -> Result<Self> {
        validate_model_dir(model_dir)?;

        let asr_fallback_variant = parse_env_variant(
            ASR_FALLBACK_ENV,
            ModelVariant::Qwen3Asr06B,
            |variant| variant.is_asr() || variant.is_voxtral(),
            "an ASR-capable variant (Qwen3-ASR/Parakeet/Voxtral)",
        )?;

        let tts_fallback_variant = parse_env_variant(
            TTS_FALLBACK_ENV,
            ModelVariant::Qwen3Tts12Hz06BBase,
            |variant| variant.is_tts(),
            "a Qwen3-TTS variant",
        )?;

        let chat_fallback_variant = parse_env_variant(
            CHAT_FALLBACK_ENV,
            ModelVariant::Qwen306B4Bit,
            |variant| variant.is_chat(),
            "a chat-capable variant",
        )?;

        info!(
            "Loaded native LFM2 model metadata from {:?} (ASR fallback: {}, TTS fallback: {}, Chat fallback: {})",
            model_dir,
            asr_fallback_variant,
            tts_fallback_variant,
            chat_fallback_variant
        );

        Ok(Self {
            model_dir: model_dir.to_path_buf(),
            asr_fallback_variant,
            tts_fallback_variant,
            chat_fallback_variant,
        })
    }

    pub fn model_dir(&self) -> &Path {
        &self.model_dir
    }

    pub fn asr_fallback_variant(&self) -> ModelVariant {
        self.asr_fallback_variant
    }

    pub fn tts_fallback_variant(&self) -> ModelVariant {
        self.tts_fallback_variant
    }

    pub fn chat_fallback_variant(&self) -> ModelVariant {
        self.chat_fallback_variant
    }

    pub fn available_voices(&self) -> Vec<String> {
        vec![
            "US Male".to_string(),
            "US Female".to_string(),
            "UK Male".to_string(),
            "UK Female".to_string(),
        ]
    }
}

pub fn lfm2_tts_voice_prompt(speaker: Option<&str>) -> &'static str {
    let normalized = speaker
        .unwrap_or("")
        .chars()
        .filter(|ch| ch.is_ascii_alphanumeric())
        .map(|ch| ch.to_ascii_lowercase())
        .collect::<String>();

    if normalized.contains("ukmale")
        || normalized == "dylan"
        || normalized == "unclefu"
        || normalized == "ukm"
    {
        return TTS_UK_MALE_PROMPT;
    }

    if normalized.contains("ukfemale") || normalized == "vivian" {
        return TTS_UK_FEMALE_PROMPT;
    }

    if normalized.contains("usmale")
        || normalized == "ryan"
        || normalized == "aiden"
        || normalized == "eric"
        || normalized.contains("male")
    {
        return TTS_US_MALE_PROMPT;
    }

    if normalized.contains("usfemale")
        || normalized == "serena"
        || normalized == "sohee"
        || normalized == "onoanna"
        || normalized == "anna"
    {
        return TTS_US_FEMALE_PROMPT;
    }

    TTS_US_FEMALE_PROMPT
}

fn parse_env_variant(
    key: &str,
    default: ModelVariant,
    validator: fn(ModelVariant) -> bool,
    expected_label: &str,
) -> Result<ModelVariant> {
    let Some(raw) = std::env::var(key)
        .ok()
        .filter(|value| !value.trim().is_empty())
    else {
        return Ok(default);
    };

    let parsed = parse_model_variant(raw.as_str())
        .map_err(|err| Error::ModelLoadError(format!("Invalid {}='{}': {}", key, raw, err)))?;

    if !validator(parsed) {
        return Err(Error::ModelLoadError(format!(
            "Invalid {}='{}': expected {}, got {}",
            key, raw, expected_label, parsed
        )));
    }

    Ok(parsed)
}

fn validate_model_dir(model_dir: &Path) -> Result<()> {
    let required_files = ["config.json", "model.safetensors", "tokenizer.json"];
    for file in &required_files {
        let path = model_dir.join(file);
        if !path.exists() {
            return Err(Error::ModelLoadError(format!(
                "LFM2 model is missing required file: {}",
                path.display()
            )));
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    #[test]
    fn maps_known_speakers_to_expected_prompts() {
        assert_eq!(lfm2_tts_voice_prompt(Some("Ryan")), TTS_US_MALE_PROMPT);
        assert_eq!(lfm2_tts_voice_prompt(Some("Serena")), TTS_US_FEMALE_PROMPT);
        assert_eq!(lfm2_tts_voice_prompt(Some("Dylan")), TTS_UK_MALE_PROMPT);
        assert_eq!(lfm2_tts_voice_prompt(Some("Vivian")), TTS_UK_FEMALE_PROMPT);
    }

    #[test]
    fn defaults_to_us_female_prompt() {
        assert_eq!(lfm2_tts_voice_prompt(None), TTS_US_FEMALE_PROMPT);
        assert_eq!(lfm2_tts_voice_prompt(Some("")), TTS_US_FEMALE_PROMPT);
    }

    #[test]
    fn loads_model_metadata_with_defaults() {
        let _guard = crate::env_test_lock().lock().expect("env lock poisoned");

        unsafe {
            std::env::remove_var(ASR_FALLBACK_ENV);
            std::env::remove_var(TTS_FALLBACK_ENV);
            std::env::remove_var(CHAT_FALLBACK_ENV);
        }

        let root = std::env::temp_dir().join(format!("izwi-lfm2-native-{}", Uuid::new_v4()));
        let model_dir = root.join("LFM2-Audio-1.5B");
        std::fs::create_dir_all(&model_dir).expect("failed to create model dir");

        std::fs::write(model_dir.join("config.json"), "{}").expect("config write failed");
        std::fs::write(model_dir.join("model.safetensors"), b"mock").expect("weights write failed");
        std::fs::write(model_dir.join("tokenizer.json"), "{}").expect("tokenizer write failed");

        let model = Lfm2AudioModel::load(&model_dir).expect("LFM2 model should load");

        assert_eq!(model.asr_fallback_variant(), ModelVariant::Qwen3Asr06B);
        assert_eq!(
            model.tts_fallback_variant(),
            ModelVariant::Qwen3Tts12Hz06BBase
        );
        assert_eq!(model.chat_fallback_variant(), ModelVariant::Qwen306B4Bit);

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn applies_env_fallback_overrides() {
        let _guard = crate::env_test_lock().lock().expect("env lock poisoned");

        let root = std::env::temp_dir().join(format!("izwi-lfm2-native-env-{}", Uuid::new_v4()));
        let model_dir = root.join("LFM2-Audio-1.5B");
        std::fs::create_dir_all(&model_dir).expect("failed to create model dir");

        std::fs::write(model_dir.join("config.json"), "{}").expect("config write failed");
        std::fs::write(model_dir.join("model.safetensors"), b"mock").expect("weights write failed");
        std::fs::write(model_dir.join("tokenizer.json"), "{}").expect("tokenizer write failed");

        unsafe {
            std::env::set_var(ASR_FALLBACK_ENV, "Parakeet-TDT-0.6B-v2");
            std::env::set_var(TTS_FALLBACK_ENV, "Qwen3-TTS-12Hz-1.7B-Base");
            std::env::set_var(CHAT_FALLBACK_ENV, "Gemma-3-1b-it");
        }

        let model = Lfm2AudioModel::load(&model_dir).expect("LFM2 model should load");
        assert_eq!(model.asr_fallback_variant(), ModelVariant::ParakeetTdt06BV2);
        assert_eq!(
            model.tts_fallback_variant(),
            ModelVariant::Qwen3Tts12Hz17BBase
        );
        assert_eq!(model.chat_fallback_variant(), ModelVariant::Gemma31BIt);

        unsafe {
            std::env::remove_var(ASR_FALLBACK_ENV);
            std::env::remove_var(TTS_FALLBACK_ENV);
            std::env::remove_var(CHAT_FALLBACK_ENV);
        }

        let _ = std::fs::remove_dir_all(&root);
    }
}
