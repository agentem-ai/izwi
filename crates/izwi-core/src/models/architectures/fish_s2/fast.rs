//! Fast codebook decoder for Fish S2 DualAR generation.

use candle_core::{DType, IndexOp, Tensor};
use candle_nn::{ops, Embedding, Linear, Module, RmsNorm, VarBuilder};

pub use super::sampling::FishS2Sampler;
use super::sampling::FishS2SamplingDistribution;
use crate::error::{Error, Result};
use crate::models::architectures::fish_s2::config::FishS2Config;
use crate::models::architectures::fish_s2::rotary::FishS2RotaryCache;
use crate::models::shared::attention::physical::{PhysicalPagedKvCache, PreparedPhysicalPagedStep};

#[derive(Debug, Clone, PartialEq)]
pub struct FishS2FastConfig {
    pub input_hidden_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub codebook_size: usize,
    pub num_codebooks: usize,
    pub semantic_start_token_id: u32,
    pub semantic_end_token_id: u32,
    pub rope_theta: f64,
    pub rms_norm_eps: f64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FishS2GeneratedFrame {
    pub semantic_token_id: u32,
    pub codebooks: Vec<u32>,
}

pub struct FishS2FastDecoder {
    cfg: FishS2FastConfig,
    project_in: FishS2FastProjectIn,
    embeddings: Embedding,
    layers: Vec<FishS2FastLayer>,
    norm: RmsNorm,
    output: Linear,
}

enum FishS2FastProjectIn {
    Identity,
    Linear(Linear),
}

struct FishS2FastLayer {
    input_layernorm: RmsNorm,
    self_attn: FishS2FastAttention,
    post_attention_layernorm: RmsNorm,
    mlp: FishS2FastMlp,
}

struct FishS2FastAttention {
    qkv_proj: Linear,
    o_proj: Linear,
    q_norm: Option<RmsNorm>,
    k_norm: Option<RmsNorm>,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rotary: FishS2RotaryCache,
}

struct FishS2FastMlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl FishS2FastConfig {
    pub fn from_config(config: &FishS2Config) -> Result<Self> {
        let audio = &config.audio_decoder_config;
        let text = &config.text_config;
        let head_dim = audio
            .head_dim
            .unwrap_or_else(|| audio.hidden_size / audio.num_attention_heads);
        let intermediate_size = audio
            .intermediate_size
            .unwrap_or_else(|| audio.hidden_size * 3);
        Ok(Self {
            input_hidden_size: text.hidden_size,
            hidden_size: audio.hidden_size,
            intermediate_size,
            num_hidden_layers: audio.num_hidden_layers,
            num_attention_heads: audio.num_attention_heads,
            num_key_value_heads: audio.num_key_value_heads,
            head_dim,
            codebook_size: config.codebook_size,
            num_codebooks: config.num_codebooks,
            semantic_start_token_id: config.semantic_start_token_id,
            semantic_end_token_id: config.semantic_end_token_id,
            rope_theta: audio.rope_theta.unwrap_or(1_000_000.0),
            rms_norm_eps: audio.rms_norm_eps.unwrap_or(1e-6),
        })
    }

    fn q_size(&self) -> usize {
        self.num_attention_heads * self.head_dim
    }

    fn kv_size(&self) -> usize {
        self.num_key_value_heads * self.head_dim
    }
}

impl FishS2FastDecoder {
    pub fn load(cfg: FishS2FastConfig, vb: VarBuilder) -> Result<Self> {
        let rotary = FishS2RotaryCache::new(
            cfg.num_codebooks,
            cfg.head_dim,
            cfg.rope_theta,
            DType::BF16,
            vb.device(),
        )?;
        let project_in = if cfg.input_hidden_size == cfg.hidden_size {
            FishS2FastProjectIn::Identity
        } else {
            let prefix = if vb.contains_tensor("fast_project_in.weight") {
                "fast_project_in"
            } else if vb.contains_tensor("project_in.weight") {
                "project_in"
            } else {
                "fast_project_in"
            };
            FishS2FastProjectIn::Linear(candle_nn::linear(
                cfg.input_hidden_size,
                cfg.hidden_size,
                vb.pp(prefix),
            )?)
        };
        let embeddings =
            candle_nn::embedding(cfg.codebook_size, cfg.hidden_size, vb.pp("fast_embeddings"))?;
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for idx in 0..cfg.num_hidden_layers {
            layers.push(FishS2FastLayer::load(
                &cfg,
                &rotary,
                vb.pp(format!("fast_layers.{idx}")),
            )?);
        }
        let norm = load_rms_norm_alias(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            &vb,
            &["fast_norm", "norm"],
        )?;
        let output =
            candle_nn::linear_no_bias(cfg.hidden_size, cfg.codebook_size, vb.pp("fast_output"))?;
        Ok(Self {
            cfg,
            project_in,
            embeddings,
            layers,
            norm,
            output,
        })
    }

    pub fn config(&self) -> &FishS2FastConfig {
        &self.cfg
    }

    /// Persistent RoPE table bytes, shared by every fast layer.
    pub fn rotary_cache_bytes(&self) -> u64 {
        self.layers
            .first()
            .map(|layer| layer.self_attn.rotary.storage_bytes())
            .unwrap_or(0)
    }

    pub fn project_slow_hidden(&self, hidden: &Tensor) -> Result<Tensor> {
        match &self.project_in {
            FishS2FastProjectIn::Identity => Ok(hidden.clone()),
            FishS2FastProjectIn::Linear(linear) => linear.forward(hidden).map_err(Error::from),
        }
    }

    pub fn codebook_embedding(&self, code: u32) -> Result<Tensor> {
        if code as usize >= self.cfg.codebook_size {
            return Err(Error::InvalidInput(format!(
                "Fish S2 fast code {code} exceeds codebook size {}",
                self.cfg.codebook_size
            )));
        }
        let ids = Tensor::from_vec(vec![code], (1, 1), self.embeddings.embeddings().device())?;
        self.embeddings.forward(&ids).map_err(Error::from)
    }

    pub fn forward_step(
        &self,
        x: &Tensor,
        input_pos: usize,
        cache: &mut PhysicalPagedKvCache,
    ) -> Result<Tensor> {
        self.forward_step_inner(x, input_pos, cache, true)
    }

    fn forward_step_inner(
        &self,
        x: &Tensor,
        input_pos: usize,
        cache: &mut PhysicalPagedKvCache,
        project_logits: bool,
    ) -> Result<Tensor> {
        let (batch_size, sequence_len, hidden_size) = x.dims3()?;
        if batch_size != 1 || sequence_len != 1 || hidden_size != self.cfg.hidden_size {
            return Err(Error::InvalidInput(format!(
                "Fish S2 fast physical paging expects [1,1,{}], got {:?}",
                self.cfg.hidden_size,
                x.dims()
            )));
        }
        if input_pos >= self.cfg.num_codebooks || input_pos >= cache.capacity_tokens() {
            return Err(Error::InvalidInput(format!(
                "Fish S2 fast codebook position {input_pos} exceeds model/cache capacity {}/{}",
                self.cfg.num_codebooks,
                cache.capacity_tokens()
            )));
        }
        cache.validate_model(
            self.cfg.num_hidden_layers,
            self.cfg.num_key_value_heads,
            self.cfg.head_dim,
        )?;
        let mut prepared = cache.prepare_append(input_pos, sequence_len)?;
        let mut hidden = x.clone();
        for (idx, layer) in self.layers.iter().enumerate() {
            hidden = layer.forward(&hidden, input_pos, cache, &mut prepared, idx)?;
        }
        let logits = if project_logits {
            self.output.forward(&self.norm.forward(&hidden)?)?
        } else {
            hidden
        };
        cache.commit_prepared(prepared)?;
        Ok(logits)
    }

    fn forward_step_batch(
        &self,
        x: &Tensor,
        input_pos: usize,
        caches: &mut [&mut PhysicalPagedKvCache],
        project_logits: bool,
    ) -> Result<Tensor> {
        let (rows, tokens, hidden) = x.dims3()?;
        if rows == 0
            || rows != caches.len()
            || tokens != 1
            || hidden != self.cfg.hidden_size
            || input_pos >= self.cfg.num_codebooks
        {
            return Err(Error::InvalidInput(
                "Fish Fast AR batch shape or clock is invalid".into(),
            ));
        }
        for cache in caches.iter() {
            cache.validate_model(
                self.cfg.num_hidden_layers,
                self.cfg.num_key_value_heads,
                self.cfg.head_dim,
            )?;
            if cache.context_len() != input_pos {
                return Err(Error::InvalidInput(
                    "Fish Fast AR batch clocks differ".into(),
                ));
            }
        }
        let mut batch = super::batch::FishPhysicalBatch::new(
            caches,
            &vec![1; rows],
            self.cfg.num_hidden_layers,
        )?;
        let execution = (|| {
            let mut hidden = x.clone();
            for (index, layer) in self.layers.iter().enumerate() {
                let norm = layer.input_layernorm.forward(&hidden)?;
                let attn = layer
                    .self_attn
                    .forward_batch(&norm, input_pos, caches[0], &mut batch, index)?;
                hidden = hidden.broadcast_add(&attn)?;
                hidden = hidden.broadcast_add(
                    &layer
                        .mlp
                        .forward(&layer.post_attention_layernorm.forward(&hidden)?)?,
                )?;
            }
            if project_logits {
                self.output
                    .forward(&self.norm.forward(&hidden)?)
                    .map_err(Error::from)
            } else {
                Ok(hidden)
            }
        })();
        batch.finish(caches, execution)
    }

    /// Depth steps remain sequential; every dense projection and attention step
    /// within a depth operates on the complete batch of independent requests.
    pub(crate) fn generate_frames_batch(
        &self,
        semantic_tokens: &[u32],
        slow_hidden: &Tensor,
        samplers: &mut [&mut FishS2Sampler],
        caches: &mut [&mut PhysicalPagedKvCache],
    ) -> Result<Vec<FishS2GeneratedFrame>> {
        let rows = semantic_tokens.len();
        if rows == 0
            || rows != caches.len()
            || rows != samplers.len()
            || slow_hidden.dims() != [rows, 1, self.cfg.input_hidden_size]
        {
            return Err(Error::InvalidInput(
                "Fish Fast AR frame batch rows do not match".into(),
            ));
        }
        let semantic_codes = semantic_tokens
            .iter()
            .map(|&token| semantic_code_from_token_id_from_fast_config(&self.cfg, token))
            .collect::<Result<Vec<_>>>()?;
        for cache in caches.iter_mut() {
            cache.validate_model(
                self.cfg.num_hidden_layers,
                self.cfg.num_key_value_heads,
                self.cfg.head_dim,
            )?;
            if cache.capacity_tokens() < self.cfg.num_codebooks {
                return Err(Error::InvalidInput(
                    "Fish Fast AR batch cache has insufficient capacity".into(),
                ));
            }
            cache.reset_invocation()?;
        }
        let hidden = self.project_slow_hidden(slow_hidden)?;
        self.forward_step_batch(&hidden, 0, caches, false)?;
        let mut frames = semantic_tokens
            .iter()
            .zip(&semantic_codes)
            .map(|(&semantic_token_id, &code)| FishS2GeneratedFrame {
                semantic_token_id,
                codebooks: vec![code],
            })
            .collect::<Vec<_>>();
        let ids = Tensor::from_slice(&semantic_codes, (rows, 1), slow_hidden.device())?;
        let mut current = self.embeddings.forward(&ids)?;
        for depth in 1..self.cfg.num_codebooks {
            let logits = self.forward_step_batch(&current, depth, caches, true)?;
            let mut codes = Vec::with_capacity(rows);
            for (row, (sampler, frame)) in samplers.iter_mut().zip(&mut frames).enumerate() {
                let code = sample_logits(&logits.i((row, 0))?, sampler)?;
                frame.codebooks.push(code);
                codes.push(code);
            }
            if depth + 1 < self.cfg.num_codebooks {
                current = self.embeddings.forward(&Tensor::from_slice(
                    &codes,
                    (rows, 1),
                    slow_hidden.device(),
                )?)?;
            }
        }
        Ok(frames)
    }

    pub fn generate_frame(
        &self,
        semantic_token_id: u32,
        slow_hidden: &Tensor,
        sampler: &mut FishS2Sampler,
        cache: &mut PhysicalPagedKvCache,
    ) -> Result<FishS2GeneratedFrame> {
        let semantic_code =
            semantic_code_from_token_id_from_fast_config(&self.cfg, semantic_token_id)?;
        cache.reset_invocation()?;
        let hidden = self.project_slow_hidden(slow_hidden)?;
        let _ = self.forward_step_inner(&hidden, 0, cache, false)?;

        let mut codebooks = vec![semantic_code];
        let mut current = self.codebook_embedding(semantic_code)?;
        for codebook_idx in 1..self.cfg.num_codebooks {
            let logits = self.forward_step(&current, codebook_idx, cache)?;
            let row = logits.i((0, 0))?;
            let code = sample_logits(&row, sampler)?;
            codebooks.push(code);
            if codebook_idx + 1 < self.cfg.num_codebooks {
                current = self.codebook_embedding(code)?;
            }
        }
        Ok(FishS2GeneratedFrame {
            semantic_token_id,
            codebooks,
        })
    }
}

impl FishS2FastLayer {
    fn load(cfg: &FishS2FastConfig, rotary: &FishS2RotaryCache, vb: VarBuilder) -> Result<Self> {
        let input_layernorm = load_rms_norm_alias(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            &vb,
            &["input_layernorm", "attention_norm"],
        )?;
        let self_attn = FishS2FastAttention::load(cfg, rotary, vb.pp("self_attn"))?;
        let post_attention_layernorm = load_rms_norm_alias(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            &vb,
            &["post_attention_layernorm", "ffn_norm"],
        )?;
        let mlp = FishS2FastMlp::load(cfg, vb.pp("mlp"))?;
        Ok(Self {
            input_layernorm,
            self_attn,
            post_attention_layernorm,
            mlp,
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        input_pos: usize,
        cache: &PhysicalPagedKvCache,
        prepared: &mut PreparedPhysicalPagedStep,
        layer_idx: usize,
    ) -> Result<Tensor> {
        let normed = self.input_layernorm.forward(x)?;
        let attn = self
            .self_attn
            .forward(&normed, input_pos, cache, prepared, layer_idx)?;
        let x = x.broadcast_add(&attn)?;
        let normed = self.post_attention_layernorm.forward(&x)?;
        let mlp = self.mlp.forward(&normed)?;
        x.broadcast_add(&mlp).map_err(Error::from)
    }
}

impl FishS2FastAttention {
    fn load(cfg: &FishS2FastConfig, rotary: &FishS2RotaryCache, vb: VarBuilder) -> Result<Self> {
        let total = cfg.q_size() + 2 * cfg.kv_size();
        if vb.contains_tensor("q_norm.weight") || vb.contains_tensor("k_norm.weight") {
            return Err(Error::ModelLoadError(
                "Fish S2 fast attention does not use Q/K normalization".into(),
            ));
        }
        Ok(Self {
            qkv_proj: candle_nn::linear_no_bias(cfg.hidden_size, total, vb.pp("qkv_proj"))?,
            o_proj: candle_nn::linear_no_bias(cfg.q_size(), cfg.hidden_size, vb.pp("o_proj"))?,
            q_norm: if vb.contains_tensor("q_norm.weight") {
                Some(candle_nn::rms_norm(
                    cfg.head_dim,
                    cfg.rms_norm_eps,
                    vb.pp("q_norm"),
                )?)
            } else {
                None
            },
            k_norm: if vb.contains_tensor("k_norm.weight") {
                Some(candle_nn::rms_norm(
                    cfg.head_dim,
                    cfg.rms_norm_eps,
                    vb.pp("k_norm"),
                )?)
            } else {
                None
            },
            num_heads: cfg.num_attention_heads,
            num_kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
            rotary: rotary.clone(),
        })
    }

    fn forward_batch(
        &self,
        x: &Tensor,
        position: usize,
        cache: &PhysicalPagedKvCache,
        batch: &mut super::batch::FishPhysicalBatch,
        layer: usize,
    ) -> Result<Tensor> {
        let rows = x.dim(0)?;
        let qsize = self.num_heads * self.head_dim;
        let kvsize = self.num_kv_heads * self.head_dim;
        let qkv = self.qkv_proj.forward(x)?;
        let q = self
            .rotary
            .apply(
                &qkv.narrow(2, 0, qsize)?
                    .reshape((rows, 1, self.num_heads, self.head_dim))?,
                position,
            )?
            .reshape((rows, self.num_heads, self.head_dim))?
            .contiguous()?;
        let k = self
            .rotary
            .apply(
                &qkv.narrow(2, qsize, kvsize)?.reshape((
                    rows,
                    1,
                    self.num_kv_heads,
                    self.head_dim,
                ))?,
                position,
            )?
            .reshape((rows, self.num_kv_heads, self.head_dim))?
            .contiguous()?;
        let v = qkv
            .narrow(2, qsize + kvsize, kvsize)?
            .reshape((rows, self.num_kv_heads, self.head_dim))?
            .contiguous()?;
        let output = batch.attend(
            cache,
            layer,
            &q,
            &k,
            &v,
            1.0 / (self.head_dim as f32).sqrt(),
        )?;
        self.o_proj
            .forward(&output.reshape((rows, 1, qsize))?)
            .map_err(Error::from)
    }

    fn forward(
        &self,
        x: &Tensor,
        input_pos: usize,
        cache: &PhysicalPagedKvCache,
        prepared: &mut PreparedPhysicalPagedStep,
        layer_idx: usize,
    ) -> Result<Tensor> {
        let bsz = x.dim(0)?;
        let seq_len = x.dim(1)?;
        if bsz != 1 {
            return Err(Error::InvalidInput(
                "Fish S2 fast physical paged attention expects one sequence".into(),
            ));
        }
        let q_size = self.num_heads * self.head_dim;
        let kv_size = self.num_kv_heads * self.head_dim;
        let qkv = self.qkv_proj.forward(x)?;
        let q = qkv
            .narrow(2, 0, q_size)?
            .reshape((bsz, seq_len, self.num_heads, self.head_dim))?;
        let k = qkv.narrow(2, q_size, kv_size)?.reshape((
            bsz,
            seq_len,
            self.num_kv_heads,
            self.head_dim,
        ))?;
        let v = qkv.narrow(2, q_size + kv_size, kv_size)?.reshape((
            bsz,
            seq_len,
            self.num_kv_heads,
            self.head_dim,
        ))?;
        let q = match &self.q_norm {
            Some(norm) => norm.forward(&q)?,
            None => q,
        };
        let k = match &self.k_norm {
            Some(norm) => norm.forward(&k)?,
            None => k,
        };

        let q = self.rotary.apply(&q, input_pos)?.squeeze(0)?;
        let k = self.rotary.apply(&k, input_pos)?.squeeze(0)?;
        let v = v.squeeze(0)?;
        let scale = 1.0f32 / (self.head_dim as f32).sqrt();
        let out = cache.write_and_attend(layer_idx, prepared, &q, &k, &v, scale)?;
        let out = out.reshape((bsz, seq_len, self.num_heads * self.head_dim))?;
        self.o_proj.forward(&out).map_err(Error::from)
    }
}

impl FishS2FastMlp {
    fn load(cfg: &FishS2FastConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            gate_proj: candle_nn::linear_no_bias(
                cfg.hidden_size,
                cfg.intermediate_size,
                vb.pp("gate_proj"),
            )?,
            up_proj: candle_nn::linear_no_bias(
                cfg.hidden_size,
                cfg.intermediate_size,
                vb.pp("up_proj"),
            )?,
            down_proj: candle_nn::linear_no_bias(
                cfg.intermediate_size,
                cfg.hidden_size,
                vb.pp("down_proj"),
            )?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = self.gate_proj.forward(x)?;
        let up = self.up_proj.forward(x)?;
        let hidden = ops::silu(&gate)?.broadcast_mul(&up)?;
        self.down_proj.forward(&hidden).map_err(Error::from)
    }
}

pub(crate) fn sample_logits(row: &Tensor, sampler: &mut FishS2Sampler) -> Result<u32> {
    super::sampling::validate_policy(sampler.temperature, sampler.top_p)?;
    let distribution =
        FishS2SamplingDistribution::from_logits(row, None, sampler.top_k, sampler.top_p, "fast")?;
    sampler.sample(&distribution)
}

fn semantic_code_from_token_id_from_fast_config(
    cfg: &FishS2FastConfig,
    token_id: u32,
) -> Result<u32> {
    if !(cfg.semantic_start_token_id..=cfg.semantic_end_token_id).contains(&token_id) {
        return Err(Error::InvalidInput(format!(
            "Fish S2 token {token_id} is outside the semantic range"
        )));
    }
    Ok(token_id - cfg.semantic_start_token_id)
}

fn load_rms_norm_alias(dim: usize, eps: f64, vb: &VarBuilder, aliases: &[&str]) -> Result<RmsNorm> {
    for alias in aliases {
        if vb.contains_tensor(&format!("{alias}.weight")) {
            return candle_nn::rms_norm(dim, eps, vb.pp(*alias)).map_err(Error::from);
        }
    }
    candle_nn::rms_norm(dim, eps, vb.pp(aliases[0])).map_err(Error::from)
}

#[cfg(test)]
pub(super) fn tiny_for_batch_tests() -> FishS2FastDecoder {
    let mut cfg = tests::tiny_cfg();
    cfg.input_hidden_size = 4;
    cfg.num_codebooks = 2;
    tests::tiny_decoder_config(&candle_core::Device::Cpu, cfg)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Shape};
    use std::collections::HashMap;

    pub(super) fn tiny_cfg() -> FishS2FastConfig {
        FishS2FastConfig {
            input_hidden_size: 3,
            hidden_size: 4,
            intermediate_size: 8,
            num_hidden_layers: 1,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            head_dim: 2,
            codebook_size: 8,
            num_codebooks: 3,
            semantic_start_token_id: 20,
            semantic_end_token_id: 27,
            rope_theta: 10_000.0,
            rms_norm_eps: 1e-6,
        }
    }

    fn tensor(device: &Device, shape: impl Into<Shape>, value: f32) -> Tensor {
        Tensor::full(value, shape, device).unwrap()
    }

    fn tiny_decoder(device: &Device) -> FishS2FastDecoder {
        tiny_decoder_config(device, tiny_cfg())
    }

    pub(super) fn tiny_decoder_config(device: &Device, cfg: FishS2FastConfig) -> FishS2FastDecoder {
        let mut tensors = HashMap::new();
        tensors.insert(
            "fast_project_in.weight".to_string(),
            tensor(device, (cfg.hidden_size, cfg.input_hidden_size), 0.01),
        );
        tensors.insert(
            "fast_project_in.bias".to_string(),
            tensor(device, (cfg.hidden_size,), 0.0),
        );
        tensors.insert(
            "fast_embeddings.weight".to_string(),
            tensor(device, (cfg.codebook_size, cfg.hidden_size), 0.02),
        );
        tensors.insert(
            "fast_norm.weight".to_string(),
            tensor(device, (cfg.hidden_size,), 1.0),
        );
        tensors.insert(
            "fast_output.weight".to_string(),
            tensor(device, (cfg.codebook_size, cfg.hidden_size), 0.03),
        );
        tensors.insert(
            "fast_layers.0.input_layernorm.weight".to_string(),
            tensor(device, (cfg.hidden_size,), 1.0),
        );
        tensors.insert(
            "fast_layers.0.post_attention_layernorm.weight".to_string(),
            tensor(device, (cfg.hidden_size,), 1.0),
        );
        tensors.insert(
            "fast_layers.0.self_attn.qkv_proj.weight".to_string(),
            tensor(
                device,
                (cfg.q_size() + 2 * cfg.kv_size(), cfg.hidden_size),
                0.01,
            ),
        );
        tensors.insert(
            "fast_layers.0.self_attn.o_proj.weight".to_string(),
            tensor(device, (cfg.hidden_size, cfg.q_size()), 0.01),
        );
        tensors.insert(
            "fast_layers.0.mlp.gate_proj.weight".to_string(),
            tensor(device, (cfg.intermediate_size, cfg.hidden_size), 0.01),
        );
        tensors.insert(
            "fast_layers.0.mlp.up_proj.weight".to_string(),
            tensor(device, (cfg.intermediate_size, cfg.hidden_size), 0.01),
        );
        tensors.insert(
            "fast_layers.0.mlp.down_proj.weight".to_string(),
            tensor(device, (cfg.hidden_size, cfg.intermediate_size), 0.01),
        );
        let vb = VarBuilder::from_tensors(tensors, DType::F32, device);
        FishS2FastDecoder::load(cfg, vb).unwrap()
    }

    #[test]
    fn sampler_argmax_is_deterministic_at_zero_temperature() {
        let device = Device::Cpu;
        let row = Tensor::from_vec(vec![0.1f32, 2.0, 1.0], (3,), &device)
            .unwrap()
            .to_dtype(DType::F16)
            .unwrap();
        let mut sampler = FishS2Sampler::new(0.0, 1.0, 7);
        assert_eq!(sample_logits(&row, &mut sampler).unwrap(), 1);
    }

    #[test]
    fn sampler_rejects_empty_greedy_logits_before_readback() {
        let device = Device::Cpu;
        let row = Tensor::from_vec(Vec::<f32>::new(), (0,), &device).unwrap();
        let mut sampler = FishS2Sampler::new(0.0, 1.0, 7);
        let err = sample_logits(&row, &mut sampler).expect_err("empty logits should fail");

        assert!(format!("{err}").contains("Fish S2 fast sampler received empty logits"));
    }

    #[test]
    fn native_fast_batch_matches_independent_sampling_and_reordered_rows() {
        let decoder = tiny_decoder(&Device::Cpu);
        let cfg = decoder.config();
        let cache = || {
            super::super::physical::test_physical_caches(
                901,
                cfg.num_hidden_layers,
                cfg.num_key_value_heads,
                cfg.head_dim,
                cfg.num_codebooks,
                7,
            )
        };
        let mut batch_caches = cache();
        let mut scalar_caches = cache();
        let mut samplers = (0..7)
            .map(|row| FishS2Sampler::new(0.7, 0.9, row + 17))
            .collect::<Vec<_>>();
        let mut scalar_samplers = samplers.clone();
        for iteration in 0..3 {
            let ids = (0..7)
                .map(|row| 20 + (row + iteration) % 8)
                .collect::<Vec<_>>();
            let hidden = Tensor::from_vec(
                (0..21).map(|i| (i as f32 * 0.31).sin()).collect::<Vec<_>>(),
                (7, 1, 3),
                &Device::Cpu,
            )
            .unwrap();
            let expected = (0..7)
                .map(|row| {
                    decoder
                        .generate_frame(
                            ids[row],
                            &hidden.narrow(0, row, 1).unwrap(),
                            &mut scalar_samplers[row],
                            &mut scalar_caches[row],
                        )
                        .unwrap()
                })
                .collect::<Vec<_>>();
            let actual = decoder
                .generate_frames_batch(
                    &ids,
                    &hidden,
                    &mut samplers.iter_mut().collect::<Vec<_>>(),
                    &mut batch_caches.iter_mut().collect::<Vec<_>>(),
                )
                .unwrap();
            assert_eq!(actual, expected);
            for cache in &mut batch_caches {
                assert_eq!(cache.context_len(), cfg.num_codebooks);
                assert_eq!(cache.take_completed_writes().len(), cfg.num_codebooks);
            }
            samplers.reverse();
            scalar_samplers.reverse();
            batch_caches.reverse();
            scalar_caches.reverse();
        }
    }

    #[test]
    fn fast_decoder_generates_full_codebook_frame() {
        let device = Device::Cpu;
        let decoder = tiny_decoder(&device);
        let slow_hidden = Tensor::full(0.5f32, (1, 1, 3), &device).unwrap();
        let mut sampler = FishS2Sampler::new(0.0, 1.0, 11);
        let cfg = decoder.config();
        let mut cache = super::super::physical::test_physical_cache(
            202,
            cfg.num_hidden_layers,
            cfg.num_key_value_heads,
            cfg.head_dim,
            cfg.num_codebooks,
        );
        let frame = decoder
            .generate_frame(22, &slow_hidden, &mut sampler, &mut cache)
            .expect("frame");
        assert_eq!(frame.semantic_token_id, 22);
        assert_eq!(frame.codebooks.len(), 3);
        assert_eq!(frame.codebooks[0], 2);
        assert!(frame.codebooks[1] < 8);
        assert!(frame.codebooks[2] < 8);
        assert_eq!(cache.context_len(), cfg.num_codebooks);

        let second = decoder
            .generate_frame(23, &slow_hidden, &mut sampler, &mut cache)
            .expect("second frame");
        assert_eq!(second.codebooks.len(), 3);
        assert_eq!(cache.context_len(), cfg.num_codebooks);
    }
}
