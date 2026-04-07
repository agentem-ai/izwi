//! LFM2/LFM2.5 GGUF text-chat model loader and generation.

use std::borrow::Cow;
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::sync::{Arc, Mutex};

use candle_core::quantized::gguf_file;
use candle_core::{DType, IndexOp, Tensor, D};
use candle_transformers::models::quantized_lfm2::ModelWeights as QuantizedLfm2Model;
use serde::Deserialize;
use tracing::info;

use crate::backends::DeviceProfile;
use crate::backends::{open_gguf_reader, BackendKind};
use crate::error::{Error, Result};
use crate::model::ModelVariant;
use crate::models::shared::chat::{ChatMessage, ChatRole};
use crate::tokenizer::Tokenizer;

#[derive(Debug, Clone)]
pub struct ChatGenerationOutput {
    pub text: String,
    pub tokens_generated: usize,
}

#[derive(Debug, Clone)]
struct SpecialTokenIds {
    bos: Option<u32>,
    im_start: u32,
    im_end: u32,
    eos: u32,
    eos_alt: Option<u32>,
}

#[derive(Debug, Deserialize)]
struct TokenizerConfig {
    #[serde(default)]
    added_tokens_decoder: HashMap<String, AddedToken>,
    #[serde(default)]
    bos_token: Option<String>,
    #[serde(default)]
    eos_token: Option<String>,
}

#[derive(Debug, Deserialize)]
struct AddedToken {
    content: String,
}

struct ChatTokenizer {
    inner: Tokenizer,
    vocab_size: usize,
    specials: SpecialTokenIds,
    decode_piece_cache: Mutex<Vec<Option<Arc<str>>>>,
}

struct PromptScaffoldTokens {
    system_header: Vec<u32>,
    user_header: Vec<u32>,
    assistant_header: Vec<u32>,
    newline: Vec<u32>,
}

impl PromptScaffoldTokens {
    fn load(tokenizer: &ChatTokenizer) -> Result<Self> {
        Ok(Self {
            system_header: tokenizer.encode_text("system\n")?,
            user_header: tokenizer.encode_text("user\n")?,
            assistant_header: tokenizer.encode_text("assistant\n")?,
            newline: tokenizer.encode_text("\n")?,
        })
    }

    fn role_header(&self, role: &ChatRole) -> &[u32] {
        match role {
            ChatRole::System => &self.system_header,
            ChatRole::User => &self.user_header,
            ChatRole::Assistant => &self.assistant_header,
        }
    }
}

impl ChatTokenizer {
    fn load(model_dir: &Path) -> Result<Self> {
        let inner = Tokenizer::from_path(model_dir)?;
        let vocab_size = inner.vocab_size();

        let config_path = model_dir.join("tokenizer_config.json");
        let config_str = fs::read_to_string(config_path)?;
        let config: TokenizerConfig = serde_json::from_str(&config_str)?;

        let id_for = |token: &str| -> Option<u32> {
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
        let bos = config
            .bos_token
            .as_deref()
            .and_then(id_for)
            .or_else(|| id_for("<|startoftext|>"));
        let eos = config
            .eos_token
            .as_deref()
            .and_then(id_for)
            .unwrap_or(im_end);
        let eos_alt = id_for("<|endoftext|>");

        Ok(Self {
            inner,
            vocab_size,
            specials: SpecialTokenIds {
                bos,
                im_start,
                im_end,
                eos,
                eos_alt,
            },
            decode_piece_cache: Mutex::new(vec![None; vocab_size]),
        })
    }

    fn encode_text(&self, text: &str) -> Result<Vec<u32>> {
        self.inner.encode(text)
    }

    fn decode_text(&self, ids: &[u32]) -> Result<String> {
        let filtered: Vec<u32> = ids
            .iter()
            .copied()
            .filter(|id| (*id as usize) < self.vocab_size)
            .collect();
        self.inner.decode(&filtered)
    }

    fn decode_token_piece(&self, token_id: u32) -> Result<Arc<str>> {
        let idx = token_id as usize;
        if idx >= self.vocab_size {
            return Ok(Arc::from(""));
        }
        if let Ok(mut cache) = self.decode_piece_cache.lock() {
            if let Some(piece) = cache.get(idx).and_then(|slot| slot.as_ref()) {
                return Ok(piece.clone());
            }
            let piece = Arc::<str>::from(self.decode_text(&[token_id])?);
            if let Some(slot) = cache.get_mut(idx) {
                *slot = Some(piece.clone());
            }
            return Ok(piece);
        }

        Ok(Arc::<str>::from(self.decode_text(&[token_id])?))
    }
}

pub struct Lfm2ChatModel {
    device: DeviceProfile,
    tokenizer: ChatTokenizer,
    prompt_scaffold: PromptScaffoldTokens,
    text_model: Mutex<QuantizedLfm2Model>,
}

impl Lfm2ChatModel {
    pub fn load(model_dir: &Path, variant: ModelVariant, device: DeviceProfile) -> Result<Self> {
        let gguf_name = match variant {
            ModelVariant::Lfm2512BInstructGguf => "LFM2.5-1.2B-Instruct-Q4_K_M.gguf",
            ModelVariant::Lfm2512BThinkingGguf => "LFM2.5-1.2B-Thinking-Q4_K_M.gguf",
            _ => {
                return Err(Error::ModelLoadError(format!(
                    "Unsupported LFM2 GGUF chat variant: {variant}"
                )));
            }
        };
        let gguf_path = model_dir.join(gguf_name);
        if !gguf_path.exists() {
            return Err(Error::ModelLoadError(format!(
                "GGUF checkpoint not found: {}",
                gguf_path.display()
            )));
        }

        let tokenizer = ChatTokenizer::load(model_dir)?;
        let mut reader = open_gguf_reader(&gguf_path, BackendKind::from(device.kind))?;
        let content = gguf_file::Content::read(&mut reader)
            .map_err(|e| Error::ModelLoadError(format!("Failed to parse GGUF header: {e}")))?;
        let text_model = QuantizedLfm2Model::from_gguf(content, &mut reader, &device.device)
            .map_err(|e| Error::ModelLoadError(format!("Failed to load LFM2 GGUF model: {e}")))?;
        let prompt_scaffold = PromptScaffoldTokens::load(&tokenizer)?;

        info!(
            "Loaded LFM2 GGUF chat model on {:?} from {}",
            device.kind,
            gguf_path.display()
        );

        Ok(Self {
            device,
            tokenizer,
            prompt_scaffold,
            text_model: Mutex::new(text_model),
        })
    }

    pub fn supports_incremental_decode(&self) -> bool {
        false
    }

    pub fn generate(
        &self,
        messages: &[ChatMessage],
        max_new_tokens: usize,
    ) -> Result<ChatGenerationOutput> {
        let mut no_op = |_delta: &str| {};
        self.generate_with_callback(messages, max_new_tokens, &mut no_op)
    }

    pub fn generate_with_callback(
        &self,
        messages: &[ChatMessage],
        max_new_tokens: usize,
        on_delta: &mut dyn FnMut(&str),
    ) -> Result<ChatGenerationOutput> {
        let prompt_ids = self.build_prompt(messages)?;
        let mut model = self
            .text_model
            .lock()
            .map_err(|_| Error::InferenceError("LFM2 GGUF model mutex poisoned".to_string()))?;

        let prompt_len = prompt_ids.len();
        let input_ids =
            Tensor::from_slice(prompt_ids.as_slice(), (1, prompt_len), &self.device.device)?;
        let mut logits = model
            .forward(&input_ids, 0)
            .map_err(|e| Error::InferenceError(format!("LFM2 GGUF forward failed: {e}")))?;
        let mut position = prompt_len;

        let mut generated_ids = Vec::new();
        let mut assembled = String::new();
        let max_new_tokens = max_new_tokens.max(1);

        while generated_ids.len() < max_new_tokens {
            let next = argmax(&logits)?;
            if next == self.tokenizer.specials.im_end
                || next == self.tokenizer.specials.eos
                || self.tokenizer.specials.eos_alt == Some(next)
            {
                break;
            }

            let delta = self.tokenizer.decode_token_piece(next)?;
            generated_ids.push(next);
            if !delta.is_empty() {
                on_delta(delta.as_ref());
            }
            assembled.push_str(delta.as_ref());

            if has_token_repetition_loop(&generated_ids) {
                break;
            }

            let next_tensor = Tensor::from_slice(&[next], (1, 1), &self.device.device)?;
            logits = model
                .forward(&next_tensor, position)
                .map_err(|e| Error::InferenceError(format!("LFM2 GGUF decode failed: {e}")))?;
            position += 1;
        }

        Ok(ChatGenerationOutput {
            text: assembled.trim().to_string(),
            tokens_generated: generated_ids.len(),
        })
    }

    pub fn prompt_token_ids(&self, messages: &[ChatMessage]) -> Result<Vec<u32>> {
        self.build_prompt(messages)
    }

    fn build_prompt(&self, messages: &[ChatMessage]) -> Result<Vec<u32>> {
        if messages.is_empty() {
            return Err(Error::InvalidInput(
                "Chat request must include at least one message".to_string(),
            ));
        }

        let prepend_default_system = !matches!(
            messages.first().map(|message| &message.role),
            Some(ChatRole::System)
        );

        let mut ids = Vec::new();
        if let Some(bos) = self.tokenizer.specials.bos {
            ids.push(bos);
        }

        let last_assistant_index = messages
            .iter()
            .rposition(|message| matches!(message.role, ChatRole::Assistant))
            .map(|index| index + usize::from(prepend_default_system));

        let mut prompt_index = 0usize;
        if prepend_default_system {
            self.append_prompt_message(
                &mut ids,
                prompt_index,
                ChatRole::System,
                "You are a helpful assistant.",
                last_assistant_index,
            )?;
            prompt_index += 1;
        }

        for message in messages {
            self.append_prompt_message(
                &mut ids,
                prompt_index,
                message.role.clone(),
                message.content.as_str(),
                last_assistant_index,
            )?;
            prompt_index += 1;
        }

        ids.push(self.tokenizer.specials.im_start);
        ids.extend_from_slice(&self.prompt_scaffold.assistant_header);

        Ok(ids)
    }

    fn append_prompt_message(
        &self,
        ids: &mut Vec<u32>,
        prompt_index: usize,
        role: ChatRole,
        content: &str,
        last_assistant_index: Option<usize>,
    ) -> Result<()> {
        let trimmed = content.trim();
        if trimmed.is_empty() {
            return Ok(());
        }

        let normalized =
            if matches!(role, ChatRole::Assistant) && Some(prompt_index) != last_assistant_index {
                strip_past_assistant_thinking(trimmed)
            } else {
                Cow::Borrowed(trimmed)
            };
        if normalized.is_empty() {
            return Ok(());
        }

        ids.push(self.tokenizer.specials.im_start);
        ids.extend_from_slice(self.prompt_scaffold.role_header(&role));
        ids.extend(self.tokenizer.encode_text(normalized.as_ref())?);
        ids.push(self.tokenizer.specials.im_end);
        ids.extend_from_slice(&self.prompt_scaffold.newline);
        Ok(())
    }
}

fn argmax(logits: &Tensor) -> Result<u32> {
    let logits = match logits.rank() {
        1 => logits.clone(),
        2 => {
            let (batch, _vocab) = logits.dims2()?;
            if batch != 1 {
                return Err(Error::InferenceError(format!(
                    "Unexpected batched logits for argmax: expected batch=1, got {batch}"
                )));
            }
            logits.i(0)?
        }
        rank => {
            return Err(Error::InferenceError(format!(
                "Unexpected logits rank for argmax: {rank}"
            )));
        }
    };
    let idx = logits.argmax(D::Minus1)?;
    let idx = if idx.rank() == 0 {
        idx
    } else {
        idx.squeeze(0)?
    };
    idx.to_dtype(DType::U32)?
        .to_scalar::<u32>()
        .map_err(Error::from)
}

fn strip_past_assistant_thinking(input: &str) -> Cow<'_, str> {
    if let Some((_reasoning, tail)) = input.rsplit_once("</think>") {
        Cow::Owned(tail.trim().to_string())
    } else {
        Cow::Borrowed(input.trim())
    }
}

fn has_suffix_repeat(ids: &[u32], span: usize, repeats: usize) -> bool {
    if span == 0 || repeats < 2 || ids.len() < span * repeats {
        return false;
    }
    let tail_start = ids.len() - span;
    let tail = &ids[tail_start..];
    (2..=repeats).all(|rep| {
        let start = ids.len() - (span * rep);
        &ids[start..start + span] == tail
    })
}

fn has_token_repetition_loop(ids: &[u32]) -> bool {
    // Catch common degenerate loops from greedy decode where the same token span
    // is emitted repeatedly (frequent in tiny reasoning models).
    if ids.len() < 48 {
        return false;
    }
    const PATTERNS: &[(usize, usize)] = &[(24, 3), (16, 3), (12, 3), (8, 4), (6, 5)];
    PATTERNS
        .iter()
        .any(|(span, repeats)| has_suffix_repeat(ids, *span, *repeats))
}

#[cfg(test)]
mod tests {
    use super::{has_token_repetition_loop, strip_past_assistant_thinking};

    #[test]
    fn strip_past_assistant_thinking_keeps_only_tail_after_close_tag() {
        let input = "<think>reasoning</think>\nFinal answer";
        assert_eq!(strip_past_assistant_thinking(input), "Final answer");
    }

    #[test]
    fn strip_past_assistant_thinking_keeps_unclosed_content() {
        let input = "<think>still reasoning";
        assert_eq!(
            strip_past_assistant_thinking(input),
            "<think>still reasoning"
        );
    }

    #[test]
    fn detects_token_repetition_loop() {
        let mut ids = Vec::new();
        let phrase = vec![1, 2, 3, 4, 5, 6, 7, 8];
        for _ in 0..5 {
            ids.extend(phrase.iter().copied());
        }
        ids.splice(0..0, vec![42; 16]);
        assert!(has_token_repetition_loop(&ids));
    }

    #[test]
    fn does_not_flag_short_sequences_as_loop() {
        let ids: Vec<u32> = (1..30).collect();
        assert!(!has_token_repetition_loop(&ids));
    }
}
