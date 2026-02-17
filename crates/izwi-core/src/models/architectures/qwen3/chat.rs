//! Native Qwen3 text-chat model loader and generation.

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use candle_core::{DType, IndexOp, Tensor};
use candle_nn::VarBuilder;
use serde::Deserialize;
use serde_json::Value;
use tracing::info;

use crate::error::{Error, Result};
use crate::models::chat_types::{ChatMessage, ChatRole};
use crate::models::device::DeviceProfile;
use crate::models::qwen3::{Qwen3Cache, Qwen3Config, Qwen3Model};
use crate::tokenizer::Tokenizer;

#[derive(Debug, Clone)]
pub struct ChatGenerationOutput {
    pub text: String,
    pub tokens_generated: usize,
}

pub struct ChatDecodeState {
    cache: Qwen3Cache,
    embeds: Tensor,
    pos: usize,
    generated_ids: Vec<u32>,
    assembled: String,
    max_new_tokens: usize,
    finished: bool,
}

#[derive(Debug, Clone)]
pub struct ChatDecodeStep {
    pub delta: String,
    pub text: String,
    pub tokens_generated: usize,
    pub finished: bool,
}

#[derive(Debug, Clone)]
struct SpecialTokenIds {
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
}

impl ChatTokenizer {
    fn load(model_dir: &Path, expected_vocab_size: usize) -> Result<Self> {
        let inner = Tokenizer::from_path_with_expected_vocab(model_dir, Some(expected_vocab_size))?;
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
                im_start,
                im_end,
                eos,
                eos_alt,
            },
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
}

pub struct Qwen3ChatModel {
    device: DeviceProfile,
    tokenizer: ChatTokenizer,
    text_model: Qwen3Model,
}

impl Qwen3ChatModel {
    pub fn load(model_dir: &Path, device: DeviceProfile) -> Result<Self> {
        let config_path = model_dir.join("config.json");
        let config_str = fs::read_to_string(config_path)?;
        let config = parse_qwen3_config(&config_str)?;

        let tokenizer = ChatTokenizer::load(model_dir, config.vocab_size)?;
        let dtype_override = std::env::var("IZWI_CHAT_DTYPE")
            .ok()
            .or_else(|| std::env::var("IZWI_QWEN_DTYPE").ok());
        let dtype = match dtype_override.as_deref().map(str::trim) {
            Some(raw) if !raw.is_empty() => device.select_dtype(Some(raw)),
            _ if device.kind.is_metal() => {
                // Keep chat memory/latency practical on Apple Silicon.
                DType::F16
            }
            _ => device.select_dtype(None),
        };

        let index_path = model_dir.join("model.safetensors.index.json");
        let vb = if index_path.exists() {
            let index_data = fs::read_to_string(&index_path)?;
            let index: Value = serde_json::from_str(&index_data)?;
            let weight_map = index
                .get("weight_map")
                .and_then(|m| m.as_object())
                .ok_or_else(|| {
                    Error::InvalidInput("Invalid model.safetensors.index.json format".to_string())
                })?;

            let mut shard_files: Vec<String> = weight_map
                .values()
                .filter_map(|v| v.as_str().map(String::from))
                .collect();
            shard_files.sort();
            shard_files.dedup();

            let shard_paths: Vec<std::path::PathBuf> =
                shard_files.iter().map(|f| model_dir.join(f)).collect();
            unsafe { VarBuilder::from_mmaped_safetensors(&shard_paths, dtype, &device.device)? }
        } else {
            let weights_path = model_dir.join("model.safetensors");
            unsafe { VarBuilder::from_mmaped_safetensors(&[weights_path], dtype, &device.device)? }
        };

        let text_model = Qwen3Model::load(config, vb)?;

        info!(
            "Loaded Qwen3 chat model on {:?} with dtype {:?}",
            device.kind, dtype
        );

        Ok(Self {
            device,
            tokenizer,
            text_model,
        })
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
        let mut state = self.start_decode(messages, max_new_tokens)?;
        loop {
            let step = self.decode_step(&mut state)?;
            if !step.delta.is_empty() {
                for ch in step.delta.chars() {
                    let mut buf = [0u8; 4];
                    on_delta(ch.encode_utf8(&mut buf));
                }
            }
            if step.finished {
                return Ok(ChatGenerationOutput {
                    text: step.text,
                    tokens_generated: step.tokens_generated,
                });
            }
        }
    }

    pub fn start_decode(
        &self,
        messages: &[ChatMessage],
        max_new_tokens: usize,
    ) -> Result<ChatDecodeState> {
        let prompt_ids = self.build_prompt(messages)?;
        let input_ids = Tensor::from_vec(
            prompt_ids.clone(),
            (1, prompt_ids.len()),
            &self.device.device,
        )?;

        let mut cache = Qwen3Cache::new(self.text_model.num_layers());
        let embeds = self.text_model.forward(&input_ids, 0, Some(&mut cache))?;
        let pos = embeds.dim(1)?;

        Ok(ChatDecodeState {
            cache,
            embeds,
            pos,
            generated_ids: Vec::new(),
            assembled: String::new(),
            max_new_tokens: max_new_tokens.max(1),
            finished: false,
        })
    }

    pub fn decode_step(&self, state: &mut ChatDecodeState) -> Result<ChatDecodeStep> {
        if state.finished || state.generated_ids.len() >= state.max_new_tokens {
            state.finished = true;
            return Ok(ChatDecodeStep {
                delta: String::new(),
                text: state.assembled.trim().to_string(),
                tokens_generated: state.generated_ids.len(),
                finished: true,
            });
        }

        let logits = state.embeds.i((0, state.embeds.dim(1)? - 1))?;
        let next = argmax(&logits)?;

        if next == self.tokenizer.specials.im_end
            || next == self.tokenizer.specials.eos
            || self.tokenizer.specials.eos_alt == Some(next)
        {
            state.finished = true;
            return Ok(ChatDecodeStep {
                delta: String::new(),
                text: state.assembled.trim().to_string(),
                tokens_generated: state.generated_ids.len(),
                finished: true,
            });
        }

        state.generated_ids.push(next);
        let decoded = self.tokenizer.decode_text(&state.generated_ids)?;
        let delta = text_delta(&state.assembled, &decoded);
        state.assembled = decoded;

        let next_tensor = Tensor::from_vec(vec![next], (1, 1), &self.device.device)?;
        state.embeds = self
            .text_model
            .forward(&next_tensor, state.pos, Some(&mut state.cache))?;
        state.pos += 1;

        if state.generated_ids.len() >= state.max_new_tokens {
            state.finished = true;
        }

        Ok(ChatDecodeStep {
            delta,
            text: state.assembled.trim().to_string(),
            tokens_generated: state.generated_ids.len(),
            finished: state.finished,
        })
    }

    fn build_prompt(&self, messages: &[ChatMessage]) -> Result<Vec<u32>> {
        if messages.is_empty() {
            return Err(Error::InvalidInput(
                "Chat request must include at least one message".to_string(),
            ));
        }

        let mut prompt_messages = messages.to_vec();
        if !matches!(
            prompt_messages.first().map(|m| &m.role),
            Some(ChatRole::System)
        ) {
            prompt_messages.insert(
                0,
                ChatMessage {
                    role: ChatRole::System,
                    content: "You are a helpful assistant.".to_string(),
                },
            );
        }

        let mut ids = Vec::new();
        for message in &prompt_messages {
            let content = if matches!(message.role, ChatRole::Assistant) {
                strip_think_blocks(message.content.trim())
            } else {
                message.content.trim().to_string()
            };

            if content.is_empty() {
                continue;
            }

            ids.push(self.tokenizer.specials.im_start);
            ids.extend(
                self.tokenizer
                    .encode_text(&format!("{}\n", message.role.as_prompt_role()))?,
            );
            ids.extend(self.tokenizer.encode_text(&content)?);
            ids.push(self.tokenizer.specials.im_end);
            ids.extend(self.tokenizer.encode_text("\n")?);
        }

        ids.push(self.tokenizer.specials.im_start);
        ids.extend(self.tokenizer.encode_text("assistant\n")?);

        Ok(ids)
    }
}

fn strip_think_blocks(input: &str) -> String {
    let mut output = input.to_string();
    let open = "<think>";
    let close = "</think>";

    loop {
        let Some(start) = output.find(open) else {
            break;
        };

        let search_from = start + open.len();
        if let Some(end_rel) = output[search_from..].find(close) {
            let end = search_from + end_rel + close.len();
            output.replace_range(start..end, "");
            continue;
        }

        output.truncate(start);
        break;
    }

    output.trim().to_string()
}

fn parse_qwen3_config(config_str: &str) -> Result<Qwen3Config> {
    let value: Value = serde_json::from_str(config_str)?;
    if let Some(text_config) = value.get("text_config") {
        serde_json::from_value(text_config.clone()).map_err(Error::from)
    } else {
        serde_json::from_value(value).map_err(Error::from)
    }
}

fn argmax(logits: &Tensor) -> Result<u32> {
    let values = logits.to_dtype(DType::F32)?.to_vec1::<f32>()?;
    let (idx, _) = values
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .ok_or_else(|| Error::InferenceError("Empty logits".to_string()))?;
    Ok(idx as u32)
}

fn text_delta(previous: &str, current: &str) -> String {
    if let Some(delta) = current.strip_prefix(previous) {
        return delta.to_string();
    }
    let common = previous
        .chars()
        .zip(current.chars())
        .take_while(|(a, b)| a == b)
        .count();
    current.chars().skip(common).collect()
}
