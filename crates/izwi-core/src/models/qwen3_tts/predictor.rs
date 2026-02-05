//! Code Predictor for multi-codebook RVQ token generation.
//!
//! The code predictor generates the residual codebook tokens after the talker
//! has produced the first (semantic) codebook. It uses a smaller transformer
//! for efficient multi-token prediction.

use candle_core::{DType, Device, Tensor, D};
use candle_nn::{ops, Embedding, Linear, Module, RmsNorm, VarBuilder};

use crate::error::{Error, Result};
use crate::models::qwen3_tts::config::CodePredictorConfig;

/// KV Cache for the code predictor
pub struct CodePredictorCache {
    k: Vec<Option<Tensor>>,
    v: Vec<Option<Tensor>>,
}

impl CodePredictorCache {
    /// Create a new cache
    pub fn new(num_layers: usize) -> Self {
        Self {
            k: vec![None; num_layers],
            v: vec![None; num_layers],
        }
    }

    /// Append new k, v to cache
    fn append(&mut self, layer: usize, k: Tensor, v: Tensor) -> Result<(Tensor, Tensor)> {
        let k = if let Some(prev) = &self.k[layer] {
            Tensor::cat(&[prev, &k], 1)?
        } else {
            k
        };
        let v = if let Some(prev) = &self.v[layer] {
            Tensor::cat(&[prev, &v], 1)?
        } else {
            v
        };
        self.k[layer] = Some(k.clone());
        self.v[layer] = Some(v.clone());
        Ok((k, v))
    }

    /// Clear the cache
    pub fn clear(&mut self) {
        for i in 0..self.k.len() {
            self.k[i] = None;
            self.v[i] = None;
        }
    }
}

/// Code Predictor model
pub struct CodePredictor {
    codec_embeddings: Vec<Embedding>,
    layers: Vec<Layer>,
    norm: RmsNorm,
    lm_heads: Vec<Linear>,
    device: Device,
    cfg: CodePredictorConfig,
    num_code_groups: usize,
}

impl CodePredictor {
    /// Load the code predictor from VarBuilder
    pub fn load(cfg: CodePredictorConfig, vb: VarBuilder, num_code_groups: usize) -> Result<Self> {
        // Load codec embeddings (one per codebook, but weights only have 15)
        // The model has embeddings 0-14 (15 total), not 16
        let num_codec_embeddings = num_code_groups.min(15);
        let mut codec_embeddings = Vec::with_capacity(num_codec_embeddings);
        for idx in 0..num_codec_embeddings {
            let embed = candle_nn::embedding(
                cfg.vocab_size,
                cfg.hidden_size,
                vb.pp(format!("model.codec_embedding.{idx}")),
            )?;
            codec_embeddings.push(embed);
        }

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for idx in 0..cfg.num_hidden_layers {
            let layer = Layer::load(&cfg, vb.pp(format!("model.layers.{idx}")))?;
            layers.push(layer);
        }

        let norm = candle_nn::rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("model.norm"))?;

        // Load output heads (one per code group, but weights only have 15)
        let num_lm_heads = num_code_groups.min(15);
        let mut lm_heads = Vec::with_capacity(num_lm_heads);
        for idx in 0..num_lm_heads {
            let head = candle_nn::linear_no_bias(
                cfg.hidden_size,
                cfg.vocab_size,
                vb.pp(format!("lm_head.{idx}")),
            )?;
            lm_heads.push(head);
        }

        Ok(Self {
            codec_embeddings,
            layers,
            norm,
            lm_heads,
            device: vb.device().clone(),
            cfg,
            num_code_groups,
        })
    }

    /// Get the device
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Get number of layers
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Get number of code groups
    pub fn num_code_groups(&self) -> usize {
        self.num_code_groups
    }

    /// Forward pass to predict all code groups from first codebook
    pub fn forward(
        &self,
        first_codebook: &Tensor,
        start_pos: usize,
        cache: Option<&mut CodePredictorCache>,
    ) -> Result<Vec<Tensor>> {
        // Embed the first codebook tokens using the first codec embedding
        let mut x = self.codec_embeddings[0].forward(first_codebook)?;

        // Pass through transformer layers
        let mut cache_ref = cache;
        for (idx, layer) in self.layers.iter().enumerate() {
            x = layer.forward(&x, start_pos, cache_ref.as_deref_mut(), idx)?;
        }

        // Final normalization
        let x = self.norm.forward(&x)?;

        // Generate logits for each code group
        let mut outputs = Vec::with_capacity(self.num_code_groups);
        for head in &self.lm_heads {
            let logits = head.forward(&x)?;
            outputs.push(logits);
        }

        Ok(outputs)
    }
}

/// Transformer layer for code predictor
struct Layer {
    input_layernorm: RmsNorm,
    self_attn: Attention,
    post_attention_layernorm: RmsNorm,
    mlp: Mlp,
}

impl Layer {
    fn load(cfg: &CodePredictorConfig, vb: VarBuilder) -> Result<Self> {
        let input_layernorm =
            candle_nn::rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        let self_attn = Attention::load(cfg, vb.pp("self_attn"))?;
        let post_attention_layernorm = candle_nn::rms_norm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;
        let mlp = Mlp::load(cfg, vb.pp("mlp"))?;

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
        start_pos: usize,
        cache: Option<&mut CodePredictorCache>,
        layer_idx: usize,
    ) -> Result<Tensor> {
        // Self-attention with residual
        let normed = self.input_layernorm.forward(x)?;
        let attn_out = self
            .self_attn
            .forward(&normed, start_pos, cache, layer_idx)?;
        let x = x.broadcast_add(&attn_out)?;

        // MLP with residual
        let normed = self.post_attention_layernorm.forward(&x)?;
        let mlp_out = self.mlp.forward(&normed)?;
        x.broadcast_add(&mlp_out).map_err(Error::from)
    }
}

/// Multi-head attention for code predictor
struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rope_theta: f64,
}

impl Attention {
    fn load(cfg: &CodePredictorConfig, vb: VarBuilder) -> Result<Self> {
        let head_dim = cfg.head_dim();

        let q_proj = candle_nn::linear_no_bias(
            cfg.hidden_size,
            cfg.num_attention_heads * head_dim,
            vb.pp("q_proj"),
        )?;
        let k_proj = candle_nn::linear_no_bias(
            cfg.hidden_size,
            cfg.num_key_value_heads * head_dim,
            vb.pp("k_proj"),
        )?;
        let v_proj = candle_nn::linear_no_bias(
            cfg.hidden_size,
            cfg.num_key_value_heads * head_dim,
            vb.pp("v_proj"),
        )?;
        let o_proj = candle_nn::linear_no_bias(
            cfg.num_attention_heads * head_dim,
            cfg.hidden_size,
            vb.pp("o_proj"),
        )?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads: cfg.num_attention_heads,
            num_kv_heads: cfg.num_key_value_heads,
            head_dim,
            rope_theta: cfg.rope_theta,
        })
    }

    fn apply_rope(&self, x: Tensor, start_pos: usize) -> Result<Tensor> {
        let bsz = x.dim(0)?;
        let seq_len = x.dim(1)?;
        let heads = x.dim(2)?;
        let half_dim = self.head_dim / 2;

        let (cos, sin) = build_rope_cache(
            seq_len,
            self.head_dim,
            start_pos,
            self.rope_theta,
            x.device(),
            x.dtype(),
        )?;

        let x = x.reshape((bsz, seq_len, heads, half_dim, 2))?;
        let x1 = x.narrow(4, 0, 1)?.squeeze(4)?;
        let x2 = x.narrow(4, 1, 1)?.squeeze(4)?;

        let cos = cos.unsqueeze(0)?.unsqueeze(2)?;
        let sin = sin.unsqueeze(0)?.unsqueeze(2)?;

        let rot1 = x1.broadcast_mul(&cos)?;
        let rot1 = rot1.broadcast_sub(&x2.broadcast_mul(&sin)?)?;
        let rot2 = x1.broadcast_mul(&sin)?;
        let rot2 = rot2.broadcast_add(&x2.broadcast_mul(&cos)?)?;

        let rot1 = rot1.unsqueeze(4)?;
        let rot2 = rot2.unsqueeze(4)?;
        let out = Tensor::cat(&[rot1, rot2], 4)?;
        out.reshape((bsz, seq_len, heads, self.head_dim))
            .map_err(Error::from)
    }

    fn forward(
        &self,
        x: &Tensor,
        start_pos: usize,
        cache: Option<&mut CodePredictorCache>,
        layer_idx: usize,
    ) -> Result<Tensor> {
        let bsz = x.dim(0)?;
        let seq_len = x.dim(1)?;

        let mut q =
            self.q_proj
                .forward(x)?
                .reshape((bsz, seq_len, self.num_heads, self.head_dim))?;
        let mut k =
            self.k_proj
                .forward(x)?
                .reshape((bsz, seq_len, self.num_kv_heads, self.head_dim))?;
        let v =
            self.v_proj
                .forward(x)?
                .reshape((bsz, seq_len, self.num_kv_heads, self.head_dim))?;

        q = self.apply_rope(q, start_pos)?;
        k = self.apply_rope(k, start_pos)?;

        let (k, v) = if let Some(cache) = cache {
            cache.append(layer_idx, k, v)?
        } else {
            (k, v)
        };

        let k = repeat_kv(&k, self.num_heads, self.num_kv_heads)?;
        let v = repeat_kv(&v, self.num_heads, self.num_kv_heads)?;

        let q = q.transpose(1, 2)?;
        let k = k.transpose(1, 2)?;
        let v = v.transpose(1, 2)?;

        let total_len = k.dim(2)?;

        let q = q.reshape((bsz * self.num_heads, seq_len, self.head_dim))?;
        let k = k.reshape((bsz * self.num_heads, total_len, self.head_dim))?;
        let v = v.reshape((bsz * self.num_heads, total_len, self.head_dim))?;

        let mut att = q.matmul(&k.transpose(1, 2)?)?;
        let scale = (self.head_dim as f64).sqrt();
        let scale_t =
            Tensor::from_vec(vec![scale as f32], (1,), att.device())?.to_dtype(att.dtype())?;
        att = att.broadcast_div(&scale_t)?;

        if seq_len > 1 || start_pos == 0 {
            let mask = causal_mask(seq_len, total_len, start_pos, att.device(), att.dtype())?;
            att = att.broadcast_add(&mask)?;
        }

        let att = ops::softmax(&att, D::Minus1)?;
        let out = att.matmul(&v)?;

        let out = out.reshape((bsz, self.num_heads, seq_len, self.head_dim))?;
        let out = out
            .transpose(1, 2)?
            .reshape((bsz, seq_len, self.num_heads * self.head_dim))?;

        self.o_proj.forward(&out).map_err(Error::from)
    }
}

/// SwiGLU MLP
struct Mlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl Mlp {
    fn load(cfg: &CodePredictorConfig, vb: VarBuilder) -> Result<Self> {
        let gate_proj =
            candle_nn::linear_no_bias(cfg.hidden_size, cfg.intermediate_size, vb.pp("gate_proj"))?;
        let up_proj =
            candle_nn::linear_no_bias(cfg.hidden_size, cfg.intermediate_size, vb.pp("up_proj"))?;
        let down_proj =
            candle_nn::linear_no_bias(cfg.intermediate_size, cfg.hidden_size, vb.pp("down_proj"))?;

        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = self.gate_proj.forward(x)?;
        let up = self.up_proj.forward(x)?;
        let act = ops::silu(&gate)?;
        let hidden = act.broadcast_mul(&up)?;
        self.down_proj.forward(&hidden).map_err(Error::from)
    }
}

/// Repeat KV for Grouped Query Attention
fn repeat_kv(x: &Tensor, num_heads: usize, num_kv_heads: usize) -> Result<Tensor> {
    if num_heads == num_kv_heads {
        return Ok(x.clone());
    }
    let repeats = num_heads / num_kv_heads;
    let mut parts = Vec::with_capacity(repeats);
    for _ in 0..repeats {
        parts.push(x.clone());
    }
    Tensor::cat(&parts, 2).map_err(Error::from)
}

/// Build RoPE cache
fn build_rope_cache(
    seq_len: usize,
    head_dim: usize,
    start_pos: usize,
    rope_theta: f64,
    device: &Device,
    dtype: DType,
) -> Result<(Tensor, Tensor)> {
    let half_dim = head_dim / 2;
    let mut inv_freq = Vec::with_capacity(half_dim);
    for i in 0..half_dim {
        let power = (2.0 * i as f64) / head_dim as f64;
        inv_freq.push((1.0 / rope_theta.powf(power)) as f32);
    }

    let mut angles = Vec::with_capacity(seq_len * half_dim);
    for pos in start_pos..start_pos + seq_len {
        for &inv in inv_freq.iter() {
            angles.push(pos as f32 * inv);
        }
    }

    let angles = Tensor::from_vec(angles, (seq_len, half_dim), device)?;
    let cos = angles.cos()?.to_dtype(dtype)?;
    let sin = angles.sin()?.to_dtype(dtype)?;
    Ok((cos, sin))
}

/// Create causal attention mask
fn causal_mask(
    seq_len: usize,
    total_len: usize,
    start_pos: usize,
    device: &Device,
    dtype: DType,
) -> Result<Tensor> {
    let mut data = vec![0f32; seq_len * total_len];
    for i in 0..seq_len {
        let limit = start_pos + i;
        for j in 0..total_len {
            if j > limit {
                data[i * total_len + j] = -1e4;
            }
        }
    }
    Tensor::from_vec(data, (1, seq_len, total_len), device)?
        .to_dtype(dtype)
        .map_err(Error::from)
}
