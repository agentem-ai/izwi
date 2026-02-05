//! Tokenizer for Qwen3-TTS models.
//!
//! Handles both text tokenization (using Qwen tokenizer) and codec token processing
//! for the multi-codebook RVQ speech representation.

use crate::error::{Error, Result};
use crate::models::qwen3_tts::config::TalkerConfig;
use serde::Deserialize;
use std::collections::HashMap;
use std::path::Path;

/// Special token IDs for TTS
#[derive(Debug, Clone)]
pub struct TtsSpecialTokens {
    pub assistant_token_id: u32,
    pub im_end_token_id: u32,
    pub im_start_token_id: u32,
    pub tts_bos_token_id: u32,
    pub tts_eos_token_id: u32,
    pub tts_pad_token_id: u32,
    pub codec_bos_id: u32,
    pub codec_eos_token_id: u32,
    pub codec_think_id: u32,
    pub codec_nothink_id: u32,
    pub codec_pad_id: u32,
    pub codec_think_bos_id: u32,
    pub codec_think_eos_id: u32,
}

impl TtsSpecialTokens {
    /// Create from main config and talker config
    pub fn from_configs(
        main_config: &super::config::Qwen3TtsConfig,
        talker_config: &TalkerConfig,
    ) -> Self {
        Self {
            assistant_token_id: main_config.assistant_token_id,
            im_end_token_id: main_config.im_end_token_id,
            im_start_token_id: main_config.im_start_token_id,
            tts_bos_token_id: main_config.tts_bos_token_id,
            tts_eos_token_id: main_config.tts_eos_token_id,
            tts_pad_token_id: main_config.tts_pad_token_id,
            codec_bos_id: talker_config.codec_bos_id,
            codec_eos_token_id: talker_config.codec_eos_token_id,
            codec_think_id: talker_config.codec_think_id,
            codec_nothink_id: talker_config.codec_nothink_id,
            codec_pad_id: talker_config.codec_pad_id,
            codec_think_bos_id: talker_config.codec_think_bos_id,
            codec_think_eos_id: talker_config.codec_think_eos_id,
        }
    }
}

/// TTS Tokenizer that wraps the base Qwen text tokenizer
pub struct TtsTokenizer {
    /// Base text tokenizer (Qwen)
    text_tokenizer: tokenizers::Tokenizer,
    /// Special tokens
    specials: TtsSpecialTokens,
    /// Speaker ID mapping
    speaker_ids: HashMap<String, u32>,
    /// Language ID mapping
    language_ids: HashMap<String, u32>,
    /// Number of code groups (codebooks) in RVQ
    num_code_groups: usize,
    /// Vocab size for text tokens
    text_vocab_size: usize,
    /// Vocab size for codec tokens per codebook
    codec_vocab_size: usize,
}

/// Tokenized input for TTS generation
#[derive(Debug, Clone)]
pub struct TtsTokenizedInput {
    /// Text token IDs
    pub text_ids: Vec<u32>,
    /// Speaker ID (if using preset speaker)
    pub speaker_id: Option<u32>,
    /// Language ID
    pub language_id: u32,
    /// Whether to use thinking mode
    pub use_thinking: bool,
}

/// Speaker reference for voice cloning
#[derive(Debug, Clone)]
pub struct SpeakerReference {
    /// Reference audio samples
    pub audio_samples: Vec<f32>,
    /// Reference text (transcription of the audio)
    pub text: String,
    /// Sample rate of the audio
    pub sample_rate: u32,
}

impl TtsTokenizer {
    /// Load tokenizer from model directory
    pub fn load(
        model_dir: &Path,
        specials: TtsSpecialTokens,
        talker_config: &TalkerConfig,
    ) -> Result<Self> {
        // Load base Qwen text tokenizer
        // Qwen3-TTS models use vocab.json + merges.txt (BPE format), not tokenizer.json
        let tokenizer_json_path = model_dir.join("tokenizer.json");
        let vocab_path = model_dir.join("vocab.json");
        let merges_path = model_dir.join("merges.txt");

        let text_tokenizer = if tokenizer_json_path.exists() {
            // Use tokenizer.json if available
            tokenizers::Tokenizer::from_file(&tokenizer_json_path)
                .map_err(|e| Error::ModelError(format!("Failed to load tokenizer: {}", e)))?
        } else if vocab_path.exists() && merges_path.exists() {
            // Build tokenizer from vocab.json + merges.txt (Qwen BPE format)
            Self::build_bpe_tokenizer(&vocab_path, &merges_path)?
        } else {
            return Err(Error::ModelError(format!(
                "No tokenizer found in {:?}. Expected tokenizer.json or vocab.json + merges.txt",
                model_dir
            )));
        };

        Ok(Self {
            text_tokenizer,
            specials,
            speaker_ids: talker_config.spk_id.clone(),
            language_ids: talker_config.codec_language_id.clone(),
            num_code_groups: talker_config.num_code_groups,
            text_vocab_size: talker_config.text_vocab_size,
            codec_vocab_size: talker_config.code_predictor_config.vocab_size,
        })
    }

    /// Build a BPE tokenizer from vocab.json and merges.txt files
    fn build_bpe_tokenizer(vocab_path: &Path, merges_path: &Path) -> Result<tokenizers::Tokenizer> {
        use tokenizers::models::bpe::BPE;

        let vocab_str = vocab_path
            .to_str()
            .ok_or_else(|| Error::ModelError("Invalid vocab path".to_string()))?;
        let merges_str = merges_path
            .to_str()
            .ok_or_else(|| Error::ModelError("Invalid merges path".to_string()))?;

        // Load vocab and merges using BPE builder
        let bpe = BPE::from_file(vocab_str, merges_str)
            .build()
            .map_err(|e| Error::ModelError(format!("Failed to build BPE tokenizer: {}", e)))?;

        let tokenizer = tokenizers::Tokenizer::new(bpe);
        Ok(tokenizer)
    }

    /// Encode text for TTS generation
    pub fn encode_text(&self, text: &str, language: Option<&str>) -> Result<Vec<u32>> {
        let encoding = self
            .text_tokenizer
            .encode(text, true)
            .map_err(|e| Error::ModelError(format!("Tokenization failed: {}", e)))?;

        Ok(encoding.get_ids().to_vec())
    }

    /// Decode text tokens back to string
    pub fn decode_text(&self, tokens: &[u32]) -> Result<String> {
        // Filter out special tokens above text_vocab_size
        let text_tokens: Vec<u32> = tokens
            .iter()
            .filter(|&&t| t < self.text_vocab_size as u32)
            .copied()
            .collect();

        self.text_tokenizer
            .decode(&text_tokens, true)
            .map_err(|e| Error::ModelError(format!("Decoding failed: {}", e)))
    }

    /// Get speaker ID by name
    pub fn get_speaker_id(&self, speaker_name: &str) -> Option<u32> {
        // Case-insensitive lookup
        let lower_name = speaker_name.to_lowercase();
        self.speaker_ids
            .iter()
            .find(|(k, _)| k.to_lowercase() == lower_name)
            .map(|(_, &v)| v)
    }

    /// Get language ID by name
    pub fn get_language_id(&self, language: &str) -> u32 {
        let lower_lang = language.to_lowercase();
        self.language_ids
            .get(&lower_lang)
            .copied()
            .unwrap_or_else(|| {
                // Default to English if not found
                self.language_ids.get("english").copied().unwrap_or(2050)
            })
    }

    /// Build the full input sequence for TTS generation
    /// Format: [IM_START] assistant [text_tokens] [IM_END] [TTS_BOS] [speaker_id] [language_id] [thinking_token] [CODEC_BOS]
    pub fn build_input_sequence(
        &self,
        text: &str,
        speaker: Option<&str>,
        language: Option<&str>,
        use_thinking: bool,
    ) -> Result<Vec<u32>> {
        let mut sequence = Vec::new();

        // Add IM_START
        sequence.push(self.specials.im_start_token_id);

        // Add assistant role marker (text token)
        sequence.push(self.specials.assistant_token_id);

        // Encode and add text
        let text_ids = self.encode_text(text, language)?;
        sequence.extend(text_ids);

        // Add IM_END
        sequence.push(self.specials.im_end_token_id);

        // Add TTS_BOS to signal start of audio generation
        sequence.push(self.specials.tts_bos_token_id);

        // Add speaker ID if specified
        if let Some(speaker_name) = speaker {
            if let Some(speaker_id) = self.get_speaker_id(speaker_name) {
                // Speaker IDs are offset in the combined vocab
                let speaker_token = self.text_vocab_size as u32 + speaker_id;
                sequence.push(speaker_token);
            }
        }

        // Add language ID
        let lang_id = language
            .map(|l| self.get_language_id(l))
            .unwrap_or_else(|| self.get_language_id("english"));
        let lang_token = self.text_vocab_size as u32 + lang_id;
        sequence.push(lang_token);

        // Add thinking/nothinking token
        if use_thinking {
            sequence.push(self.text_vocab_size as u32 + self.specials.codec_think_id);
        } else {
            sequence.push(self.text_vocab_size as u32 + self.specials.codec_nothink_id);
        }

        // Add CODEC_BOS to start audio tokens
        let codec_bos_token = self.text_vocab_size as u32 + self.specials.codec_bos_id;
        sequence.push(codec_bos_token);

        Ok(sequence)
    }

    /// Build voice clone input sequence
    /// This includes reference audio tokens encoded by the speech tokenizer
    pub fn build_voice_clone_sequence(
        &self,
        text: &str,
        ref_codec_tokens: &[Vec<u32>], // [num_code_groups, seq_len]
        language: Option<&str>,
        use_thinking: bool,
    ) -> Result<Vec<u32>> {
        let mut sequence = Vec::new();

        // Add IM_START
        sequence.push(self.specials.im_start_token_id);

        // Add assistant role marker
        sequence.push(self.specials.assistant_token_id);

        // Encode and add text
        let text_ids = self.encode_text(text, language)?;
        sequence.extend(text_ids);

        // Add IM_END
        sequence.push(self.specials.im_end_token_id);

        // Add TTS_BOS
        sequence.push(self.specials.tts_bos_token_id);

        // Add reference audio tokens if provided
        // Reference tokens are interleaved: [c0_t0, c1_t0, ..., cN_t0, c0_t1, c1_t1, ...]
        if !ref_codec_tokens.is_empty() && !ref_codec_tokens[0].is_empty() {
            let seq_len = ref_codec_tokens[0].len();
            for t in 0..seq_len {
                for (group_idx, group_tokens) in ref_codec_tokens.iter().enumerate() {
                    if t < group_tokens.len() {
                        // Offset token by text_vocab_size + group-specific offset
                        let token = self.text_vocab_size as u32
                            + group_tokens[t]
                            + (group_idx as u32 * self.codec_vocab_size as u32);
                        sequence.push(token);
                    }
                }
            }
        }

        // Add language ID
        let lang_id = language
            .map(|l| self.get_language_id(l))
            .unwrap_or_else(|| self.get_language_id("english"));
        let lang_token = self.text_vocab_size as u32 + lang_id;
        sequence.push(lang_token);

        // Add thinking/nothinking token
        if use_thinking {
            sequence.push(self.text_vocab_size as u32 + self.specials.codec_think_id);
        } else {
            sequence.push(self.text_vocab_size as u32 + self.specials.codec_nothink_id);
        }

        // Add CODEC_BOS
        let codec_bos_token = self.text_vocab_size as u32 + self.specials.codec_bos_id;
        sequence.push(codec_bos_token);

        Ok(sequence)
    }

    /// Decode codec tokens from model output
    /// Returns [num_code_groups, seq_len] token arrays
    pub fn decode_codec_tokens(&self, tokens: &[u32]) -> Vec<Vec<u32>> {
        let mut code_groups: Vec<Vec<u32>> = vec![Vec::new(); self.num_code_groups];

        for &token in tokens {
            // Skip non-codec tokens
            if token < self.text_vocab_size as u32 {
                continue;
            }

            let codec_token = token - self.text_vocab_size as u32;

            // Check if it's a special codec token
            if codec_token == self.specials.codec_eos_token_id {
                break; // End of sequence
            }
            if codec_token == self.specials.codec_pad_id {
                continue; // Skip padding
            }

            // Determine which codebook this belongs to
            let group_idx = (codec_token as usize) / self.codec_vocab_size;
            let group_token = (codec_token as usize) % self.codec_vocab_size;

            if group_idx < self.num_code_groups {
                code_groups[group_idx].push(group_token as u32);
            }
        }

        code_groups
    }

    /// Get number of code groups
    pub fn num_code_groups(&self) -> usize {
        self.num_code_groups
    }

    /// Get codec vocab size
    pub fn codec_vocab_size(&self) -> usize {
        self.codec_vocab_size
    }

    /// Get text vocab size
    pub fn text_vocab_size(&self) -> usize {
        self.text_vocab_size
    }

    /// Get special tokens
    pub fn specials(&self) -> &TtsSpecialTokens {
        &self.specials
    }

    /// List available speakers
    pub fn available_speakers(&self) -> Vec<&String> {
        self.speaker_ids.keys().collect()
    }

    /// List available languages
    pub fn available_languages(&self) -> Vec<&String> {
        self.language_ids.keys().collect()
    }
}

/// Build ChatML-style conversation format for TTS
pub fn build_chatml_prompt(text: &str, system_prompt: Option<&str>) -> String {
    if let Some(system) = system_prompt {
        format!(
            "<|im_start|>system\n{}<|im_end|>\n<|im_start|>assistant\n{}<|im_end|>",
            system, text
        )
    } else {
        format!("<|im_start|>assistant\n{}<|im_end|>", text)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_special_tokens() {
        let specials = TtsSpecialTokens {
            assistant_token_id: 77091,
            im_end_token_id: 151645,
            im_start_token_id: 151644,
            tts_bos_token_id: 151672,
            tts_eos_token_id: 151673,
            tts_pad_token_id: 151671,
            codec_bos_id: 2149,
            codec_eos_token_id: 2150,
            codec_think_id: 2154,
            codec_nothink_id: 2155,
            codec_pad_id: 2148,
            codec_think_bos_id: 2156,
            codec_think_eos_id: 2157,
        };

        assert_eq!(specials.codec_bos_id, 2149);
        assert_eq!(specials.codec_eos_token_id, 2150);
    }

    #[test]
    fn test_chatml_format() {
        let prompt = build_chatml_prompt("Hello world", Some("You are a helpful assistant."));
        assert!(prompt.contains("<|im_start|>system"));
        assert!(prompt.contains("<|im_start|>assistant"));
        assert!(prompt.contains("Hello world"));
    }
}
