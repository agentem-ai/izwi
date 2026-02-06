//! Tokenizer wrapper for Qwen3-ASR.

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use serde::Deserialize;

use crate::error::{Error, Result};
use crate::tokenizer::Tokenizer;

#[derive(Debug, Clone)]
pub struct SpecialTokenIds {
    pub im_start: u32,
    pub im_end: u32,
    pub audio_start: u32,
    pub audio_end: u32,
    pub audio_token: u32,
    pub eos: u32,
    pub eos_alt: Option<u32>,
    pub pad: u32,
}

#[derive(Debug, Deserialize)]
struct TokenizerConfig {
    #[serde(default)]
    added_tokens_decoder: HashMap<String, AddedToken>,
    #[serde(default)]
    eos_token: Option<String>,
    #[serde(default)]
    pad_token: Option<String>,
}

#[derive(Debug, Deserialize)]
struct AddedToken {
    content: String,
}

pub struct AsrTokenizer {
    inner: Tokenizer,
    vocab_size: usize,
    specials: SpecialTokenIds,
}

impl AsrTokenizer {
    pub fn load(model_dir: &Path, expected_vocab_size: usize) -> Result<Self> {
        let inner = Tokenizer::from_path_with_expected_vocab(model_dir, Some(expected_vocab_size))?;
        let vocab_size = inner.vocab_size();

        let config_path = model_dir.join("tokenizer_config.json");
        let config_str = fs::read_to_string(config_path)?;
        let config: TokenizerConfig = serde_json::from_str(&config_str)?;

        let mut id_for = |token: &str| -> Option<u32> {
            config.added_tokens_decoder.iter().find_map(|(id, entry)| {
                if entry.content == token {
                    id.parse().ok()
                } else {
                    None
                }
            })
        };

        let im_start = id_for("<|im_start|>")
            .ok_or_else(|| Error::TokenizationError("Missing <|im_start|> token id".to_string()))?;
        let im_end = id_for("<|im_end|>")
            .ok_or_else(|| Error::TokenizationError("Missing <|im_end|> token id".to_string()))?;
        let audio_start = id_for("<|audio_start|>").ok_or_else(|| {
            Error::TokenizationError("Missing <|audio_start|> token id".to_string())
        })?;
        let audio_end = id_for("<|audio_end|>").ok_or_else(|| {
            Error::TokenizationError("Missing <|audio_end|> token id".to_string())
        })?;
        let audio_token = id_for("<|audio_pad|>").ok_or_else(|| {
            Error::TokenizationError("Missing <|audio_pad|> token id".to_string())
        })?;

        let eos = config
            .eos_token
            .as_deref()
            .and_then(&mut id_for)
            .unwrap_or(im_end);
        let eos_alt = id_for("<|endoftext|>");
        let pad = config
            .pad_token
            .as_deref()
            .and_then(&mut id_for)
            .unwrap_or(eos);

        Ok(Self {
            inner,
            vocab_size,
            specials: SpecialTokenIds {
                im_start,
                im_end,
                audio_start,
                audio_end,
                audio_token,
                eos,
                eos_alt,
                pad,
            },
        })
    }

    pub fn encode_text(&self, text: &str) -> Result<Vec<u32>> {
        self.inner.encode(text)
    }

    pub fn decode_text(&self, ids: &[u32]) -> Result<String> {
        let filtered: Vec<u32> = ids
            .iter()
            .copied()
            .filter(|id| (*id as usize) < self.vocab_size)
            .collect();
        self.inner.decode(&filtered)
    }

    pub fn specials(&self) -> &SpecialTokenIds {
        &self.specials
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }
}
