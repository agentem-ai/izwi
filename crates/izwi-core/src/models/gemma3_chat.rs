//! Native Gemma 3 text-chat model loader and generation.

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;
use std::sync::Mutex;

use candle_core::{DType, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::gemma3::{Config as Gemma3Config, Model as Gemma3Model};
use serde_json::Value;
use tracing::info;

use crate::error::{Error, Result};
use crate::model::ModelVariant;
use crate::models::chat_types::{ChatMessage, ChatRole};
use crate::models::device::DeviceProfile;
use crate::tokenizer::Tokenizer;

#[derive(Debug, Clone)]
pub struct ChatGenerationOutput {
    pub text: String,
    pub tokens_generated: usize,
}

#[derive(Debug, Clone)]
struct SpecialTokenIds {
    bos: Option<u32>,
    eos: u32,
    start_of_turn: u32,
    end_of_turn: u32,
}

struct GemmaTokenizer {
    inner: Tokenizer,
    vocab_size: usize,
    specials: SpecialTokenIds,
}

impl GemmaTokenizer {
    fn load(model_dir: &Path) -> Result<Self> {
        let inner = Tokenizer::from_path(model_dir)?;
        let vocab_size = inner.vocab_size();

        let token_id = |token: &str| -> Result<u32> {
            inner.token_to_id(token).ok_or_else(|| {
                Error::TokenizationError(format!("Missing Gemma special token: {token}"))
            })
        };

        let start_of_turn = token_id("<start_of_turn>")?;
        let end_of_turn = token_id("<end_of_turn>")?;
        let eos = inner.token_to_id("<eos>").unwrap_or(end_of_turn);
        let bos = inner.token_to_id("<bos>");

        Ok(Self {
            inner,
            vocab_size,
            specials: SpecialTokenIds {
                bos,
                eos,
                start_of_turn,
                end_of_turn,
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

struct GemmaDefaults {
    hidden_size: usize,
    intermediate_size: usize,
    num_attention_heads: usize,
    num_hidden_layers: usize,
    num_key_value_heads: usize,
    head_dim: usize,
}

fn defaults_for_variant(variant: ModelVariant) -> GemmaDefaults {
    match variant {
        ModelVariant::Gemma31BIt => GemmaDefaults {
            hidden_size: 1152,
            intermediate_size: 6912,
            num_attention_heads: 4,
            num_hidden_layers: 26,
            num_key_value_heads: 1,
            head_dim: 256,
        },
        ModelVariant::Gemma34BIt => GemmaDefaults {
            hidden_size: 2560,
            intermediate_size: 10240,
            num_attention_heads: 8,
            num_hidden_layers: 34,
            num_key_value_heads: 4,
            head_dim: 256,
        },
        _ => GemmaDefaults {
            hidden_size: 2560,
            intermediate_size: 10240,
            num_attention_heads: 8,
            num_hidden_layers: 34,
            num_key_value_heads: 4,
            head_dim: 256,
        },
    }
}

fn parse_gemma3_config(
    config_str: &str,
    variant: ModelVariant,
    tokenizer_vocab_size: usize,
) -> Result<Gemma3Config> {
    let root_value: Value = serde_json::from_str(config_str)?;
    let source = root_value.get("text_config").cloned().unwrap_or(root_value);

    let mut object: BTreeMap<String, Value> = source
        .as_object()
        .ok_or_else(|| Error::InvalidInput("Invalid Gemma config.json format".to_string()))?
        .iter()
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();

    let defaults = defaults_for_variant(variant);

    let mut set_default = |key: &str, value: Value| {
        if !object.contains_key(key) {
            object.insert(key.to_string(), value);
        }
    };

    set_default("attention_bias", Value::Bool(false));
    set_default(
        "hidden_activation",
        Value::String("gelu_pytorch_tanh".to_string()),
    );
    set_default("hidden_size", Value::from(defaults.hidden_size as u64));
    set_default(
        "intermediate_size",
        Value::from(defaults.intermediate_size as u64),
    );
    set_default(
        "num_attention_heads",
        Value::from(defaults.num_attention_heads as u64),
    );
    set_default(
        "num_hidden_layers",
        Value::from(defaults.num_hidden_layers as u64),
    );
    set_default(
        "num_key_value_heads",
        Value::from(defaults.num_key_value_heads as u64),
    );
    set_default("head_dim", Value::from(defaults.head_dim as u64));
    set_default("rms_norm_eps", Value::from(1e-6f64));
    set_default("rope_theta", Value::from(1_000_000f64));
    set_default("rope_local_base_freq", Value::from(10_000f64));
    set_default(
        "query_pre_attn_scalar",
        Value::from(defaults.head_dim as u64),
    );
    set_default("sliding_window", Value::from(1024u64));
    set_default("sliding_window_pattern", Value::from(6u64));
    set_default("max_position_embeddings", Value::from(131_072u64));
    set_default("vocab_size", Value::from(tokenizer_vocab_size as u64));

    let config =
        serde_json::from_value::<Gemma3Config>(Value::Object(object.into_iter().collect()))
            .map_err(Error::from)?;

    Ok(config)
}

pub struct Gemma3ChatModel {
    variant: ModelVariant,
    device: DeviceProfile,
    tokenizer: GemmaTokenizer,
    text_model: Mutex<Gemma3Model>,
}

impl Gemma3ChatModel {
    pub fn load(model_dir: &Path, variant: ModelVariant, device: DeviceProfile) -> Result<Self> {
        let tokenizer = GemmaTokenizer::load(model_dir)?;

        let config_path = model_dir.join("config.json");
        let config_str = fs::read_to_string(config_path)?;
        let config = parse_gemma3_config(&config_str, variant, tokenizer.vocab_size)?;
        let dtype = device.select_dtype(None);

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

        let text_model = Gemma3Model::new(false, &config, vb).map_err(Error::from)?;

        info!(
            "Loaded Gemma chat model {} on {:?}",
            variant.dir_name(),
            device.kind
        );

        Ok(Self {
            variant,
            device,
            tokenizer,
            text_model: Mutex::new(text_model),
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
        let prompt_ids = self.build_prompt(messages)?;
        let mut input_ids = Tensor::from_vec(
            prompt_ids.clone(),
            (1, prompt_ids.len()),
            &self.device.device,
        )?;
        let mut seqlen_offset = 0usize;

        let mut generated_ids = Vec::new();
        let mut assembled = String::new();

        let mut model = self
            .text_model
            .lock()
            .map_err(|_| Error::InferenceError("Gemma model mutex poisoned".to_string()))?;
        model.clear_kv_cache();

        for _ in 0..max_new_tokens {
            let logits = model
                .forward(&input_ids, seqlen_offset)
                .map_err(Error::from)?;
            let next = select_next_token(&logits)?;

            if next == self.tokenizer.specials.end_of_turn || next == self.tokenizer.specials.eos {
                break;
            }

            generated_ids.push(next);

            let decoded = self.tokenizer.decode_text(&generated_ids)?;
            let delta = text_delta(&assembled, &decoded);
            for ch in delta.chars() {
                let mut buf = [0u8; 4];
                on_delta(ch.encode_utf8(&mut buf));
            }
            assembled = decoded;

            seqlen_offset += input_ids.dim(1)?;
            input_ids = Tensor::from_vec(vec![next], (1, 1), &self.device.device)?;
        }

        Ok(ChatGenerationOutput {
            text: assembled.trim().to_string(),
            tokens_generated: generated_ids.len(),
        })
    }

    fn build_prompt(&self, messages: &[ChatMessage]) -> Result<Vec<u32>> {
        if messages.is_empty() {
            return Err(Error::InvalidInput(
                "Chat request must include at least one message".to_string(),
            ));
        }

        let mut system_parts = Vec::new();
        let mut turns: Vec<(ChatRole, String)> = Vec::new();

        for message in messages {
            let content = if matches!(message.role, ChatRole::Assistant) {
                strip_think_blocks(message.content.trim())
            } else {
                message.content.trim().to_string()
            };
            if content.is_empty() {
                continue;
            }

            if matches!(message.role, ChatRole::System) {
                system_parts.push(content);
            } else {
                turns.push((message.role.clone(), content));
            }
        }

        let system = if system_parts.is_empty() {
            "You are a helpful assistant.".to_string()
        } else {
            system_parts.join("\n\n")
        };

        if let Some((role, first_content)) = turns.first_mut() {
            if matches!(role, ChatRole::User) {
                *first_content = format!("{system}\n\n{first_content}");
            } else {
                turns.insert(0, (ChatRole::User, system));
            }
        } else {
            turns.push((ChatRole::User, system));
        }

        let mut ids = Vec::new();
        if let Some(bos) = self.tokenizer.specials.bos {
            ids.push(bos);
        }

        for (role, content) in &turns {
            let role_name = match role {
                ChatRole::Assistant => "model",
                ChatRole::User | ChatRole::System => "user",
            };

            ids.push(self.tokenizer.specials.start_of_turn);
            ids.extend(self.tokenizer.encode_text(role_name)?);
            ids.extend(self.tokenizer.encode_text("\n")?);
            ids.extend(self.tokenizer.encode_text(content)?);
            ids.push(self.tokenizer.specials.end_of_turn);
            ids.extend(self.tokenizer.encode_text("\n")?);
        }

        ids.push(self.tokenizer.specials.start_of_turn);
        ids.extend(self.tokenizer.encode_text("model")?);
        ids.extend(self.tokenizer.encode_text("\n")?);

        Ok(ids)
    }

    pub fn variant(&self) -> ModelVariant {
        self.variant
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

fn argmax(logits: &Tensor) -> Result<u32> {
    let values = logits.to_dtype(DType::F32)?.to_vec1::<f32>()?;
    let (idx, _) = values
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .ok_or_else(|| Error::InferenceError("Empty logits".to_string()))?;
    Ok(idx as u32)
}

fn select_next_token(logits: &Tensor) -> Result<u32> {
    match logits.rank() {
        // [vocab]
        1 => argmax(logits),
        // [seq, vocab]
        2 => {
            let seq_len = logits.dim(0)?;
            argmax(&logits.i(seq_len.saturating_sub(1))?)
        }
        // [batch, seq, vocab]
        3 => {
            let seq_len = logits.dim(1)?;
            argmax(&logits.i((0, seq_len.saturating_sub(1)))?)
        }
        _ => Err(Error::InferenceError(format!(
            "Unexpected Gemma logits rank: {} with dims {:?}",
            logits.rank(),
            logits.dims()
        ))),
    }
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
