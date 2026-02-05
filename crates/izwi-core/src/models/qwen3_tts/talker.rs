//! Qwen3-TTS Talker model implementation.
//!
//! The talker is the main LLM component that generates speech tokens from text input.
//! It uses a Qwen3 architecture with MRoPE (Multi-modal Rotary Position Embeddings)
//! to handle both text and audio modalities.

use candle_core::{DType, Device, Tensor, D};
use candle_nn::{ops, Embedding, Linear, Module, RmsNorm, VarBuilder};

use crate::error::{Error, Result};
use crate::models::qwen3_tts::config::TalkerConfig;

/// KV Cache for the talker model
pub struct TalkerCache {
    k: Vec<Option<Tensor>>,
    v: Vec<Option<Tensor>>,
}

impl TalkerCache {
    /// Create a new cache for the specified number of layers
    pub fn new(num_layers: usize) -> Self {
        Self {
            k: vec![None; num_layers],
            v: vec![None; num_layers],
        }
    }

    /// Append new k, v to cache and return full sequence
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

/// Multi-head attention with optional Q/K normalization
struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: Option<RmsNorm>,
    k_norm: Option<RmsNorm>,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rope_theta: f64,
    use_mrope: bool,
    mrope_section: Vec<usize>,
}

impl Attention {
    fn load(cfg: &TalkerConfig, vb: VarBuilder) -> Result<Self> {
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

        // Q/K normalization (optional, for Qwen3)
        let q_norm = candle_nn::rms_norm(head_dim, cfg.rms_norm_eps, vb.pp("q_norm")).ok();
        let k_norm = candle_nn::rms_norm(head_dim, cfg.rms_norm_eps, vb.pp("k_norm")).ok();

        let use_mrope = cfg.uses_mrope();
        let mrope_section = cfg.mrope_section();

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            num_heads: cfg.num_attention_heads,
            num_kv_heads: cfg.num_key_value_heads,
            head_dim,
            rope_theta: cfg.rope_theta,
            use_mrope,
            mrope_section,
        })
    }

    fn apply_qk_norm(
        &self,
        x: Tensor,
        norm: &Option<RmsNorm>,
        heads: usize,
        seq_len: usize,
    ) -> Result<Tensor> {
        if let Some(norm) = norm {
            let bsz = x.dim(0)?;
            let reshaped = x.reshape((bsz * seq_len * heads, self.head_dim))?;
            let normed = norm.forward(&reshaped)?;
            normed
                .reshape((bsz, seq_len, heads, self.head_dim))
                .map_err(Error::from)
        } else {
            Ok(x)
        }
    }

    fn apply_rope(
        &self,
        x: Tensor,
        start_pos: usize,
        position_ids: Option<&Tensor>,
    ) -> Result<Tensor> {
        let bsz = x.dim(0)?;
        let seq_len = x.dim(1)?;
        let heads = x.dim(2)?;
        let half_dim = self.head_dim / 2;

        let (cos, sin) = if self.use_mrope {
            let position_ids = if let Some(position_ids) = position_ids {
                position_ids.clone()
            } else {
                // Default: create standard position IDs replicated for 3 dims
                let mut data = Vec::with_capacity(3 * seq_len);
                let base = start_pos as i64;
                for _ in 0..3 {
                    for idx in 0..seq_len {
                        data.push(base + idx as i64);
                    }
                }
                Tensor::from_vec(data, (3, seq_len), x.device())?
            };
            build_mrope_cache(
                seq_len,
                self.head_dim,
                self.rope_theta,
                x.device(),
                x.dtype(),
                &position_ids,
                &self.mrope_section,
            )?
        } else {
            build_rope_cache(
                seq_len,
                self.head_dim,
                start_pos,
                self.rope_theta,
                x.device(),
                x.dtype(),
            )?
        };

        // Apply rotary embeddings
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
        position_ids: Option<&Tensor>,
        cache: Option<&mut TalkerCache>,
        layer_idx: usize,
    ) -> Result<Tensor> {
        let bsz = x.dim(0)?;
        let seq_len = x.dim(1)?;

        // Project to Q, K, V
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

        // Apply Q/K normalization if present
        q = self.apply_qk_norm(q, &self.q_norm, self.num_heads, seq_len)?;
        k = self.apply_qk_norm(k, &self.k_norm, self.num_kv_heads, seq_len)?;

        // Apply RoPE
        q = self.apply_rope(q, start_pos, position_ids)?;
        k = self.apply_rope(k, start_pos, position_ids)?;

        // Update cache and get full K, V
        let (k, v) = if let Some(cache) = cache {
            cache.append(layer_idx, k, v)?
        } else {
            (k, v)
        };

        // Repeat K/V for GQA
        let k = repeat_kv(&k, self.num_heads, self.num_kv_heads)?;
        let v = repeat_kv(&v, self.num_heads, self.num_kv_heads)?;

        // Transpose for attention
        let q = q.transpose(1, 2)?;
        let k = k.transpose(1, 2)?;
        let v = v.transpose(1, 2)?;

        let total_len = k.dim(2)?;

        // Reshape for batch matrix multiply
        let q = q.reshape((bsz * self.num_heads, seq_len, self.head_dim))?;
        let k = k.reshape((bsz * self.num_heads, total_len, self.head_dim))?;
        let v = v.reshape((bsz * self.num_heads, total_len, self.head_dim))?;

        // Compute attention
        let mut att = q.matmul(&k.transpose(1, 2)?)?;
        let scale = (self.head_dim as f64).sqrt();
        let scale_t =
            Tensor::from_vec(vec![scale as f32], (1,), att.device())?.to_dtype(att.dtype())?;
        att = att.broadcast_div(&scale_t)?;

        // Apply causal mask
        if seq_len > 1 || start_pos == 0 {
            let mask = causal_mask(seq_len, total_len, start_pos, att.device(), att.dtype())?;
            att = att.broadcast_add(&mask)?;
        }

        // Softmax and apply to values
        let att = ops::softmax(&att, D::Minus1)?;
        let out = att.matmul(&v)?;

        // Reshape back
        let out = out.reshape((bsz, self.num_heads, seq_len, self.head_dim))?;
        let out = out
            .transpose(1, 2)?
            .reshape((bsz, seq_len, self.num_heads * self.head_dim))?;

        // Output projection
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
    fn load(cfg: &TalkerConfig, vb: VarBuilder) -> Result<Self> {
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

/// Transformer layer
struct Layer {
    input_layernorm: RmsNorm,
    self_attn: Attention,
    post_attention_layernorm: RmsNorm,
    mlp: Mlp,
}

impl Layer {
    fn load(cfg: &TalkerConfig, vb: VarBuilder) -> Result<Self> {
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
        position_ids: Option<&Tensor>,
        cache: Option<&mut TalkerCache>,
        layer_idx: usize,
    ) -> Result<Tensor> {
        // Self-attention with residual
        let normed = self.input_layernorm.forward(x)?;
        let attn_out =
            self.self_attn
                .forward(&normed, start_pos, position_ids, cache, layer_idx)?;
        let x = x.broadcast_add(&attn_out)?;

        // MLP with residual
        let normed = self.post_attention_layernorm.forward(&x)?;
        let mlp_out = self.mlp.forward(&normed)?;
        x.broadcast_add(&mlp_out).map_err(Error::from)
    }
}

/// Text projection MLP to project text embeddings to model hidden size
struct TextProjection {
    linear_fc1: Linear,
    linear_fc2: Linear,
}

impl TextProjection {
    fn load(text_hidden_size: usize, hidden_size: usize, vb: VarBuilder) -> Result<Self> {
        let linear_fc1 =
            candle_nn::linear(text_hidden_size, text_hidden_size, vb.pp("linear_fc1"))?;
        let linear_fc2 = candle_nn::linear(text_hidden_size, hidden_size, vb.pp("linear_fc2"))?;
        Ok(Self {
            linear_fc1,
            linear_fc2,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.linear_fc1.forward(x)?;
        let x = ops::silu(&x)?;
        self.linear_fc2.forward(&x).map_err(Error::from)
    }
}

/// Qwen3-TTS Talker model
pub struct TalkerModel {
    text_embedding: Embedding,
    text_projection: TextProjection,
    codec_embedding: Embedding,
    layers: Vec<Layer>,
    norm: RmsNorm,
    lm_head: Linear,
    device: Device,
    cfg: TalkerConfig,
    use_mrope: bool,
}

impl TalkerModel {
    /// Load the talker model from VarBuilder
    pub fn load(cfg: TalkerConfig, vb: VarBuilder) -> Result<Self> {
        let text_embedding = candle_nn::embedding(
            cfg.text_vocab_size,
            cfg.text_hidden_size,
            vb.pp("model.text_embedding"),
        )?;
        let text_projection = TextProjection::load(
            cfg.text_hidden_size,
            cfg.hidden_size,
            vb.pp("text_projection"),
        )?;
        let codec_embedding = candle_nn::embedding(
            cfg.vocab_size,
            cfg.hidden_size,
            vb.pp("model.codec_embedding"),
        )?;
        let lm_head =
            candle_nn::linear_no_bias(cfg.hidden_size, cfg.vocab_size, vb.pp("codec_head"))?;

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for idx in 0..cfg.num_hidden_layers {
            let layer = Layer::load(&cfg, vb.pp(format!("model.layers.{idx}")))?;
            layers.push(layer);
        }

        let norm = candle_nn::rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("model.norm"))?;
        let use_mrope = cfg.uses_mrope();

        Ok(Self {
            text_embedding,
            text_projection,
            codec_embedding,
            layers,
            norm,
            lm_head,
            device: vb.device().clone(),
            cfg,
            use_mrope,
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

    /// Forward pass starting from token IDs
    pub fn forward(
        &self,
        input_ids: &Tensor,
        start_pos: usize,
        cache: Option<&mut TalkerCache>,
    ) -> Result<Tensor> {
        let embeds = self.embeddings(input_ids)?;
        self.forward_with_embeds(&embeds, start_pos, cache, None)
    }

    /// Get embeddings for token IDs
    /// Uses text_embedding for text tokens and codec_embedding for codec tokens
    pub fn embeddings(&self, input_ids: &Tensor) -> Result<Tensor> {
        // For now, use text_embedding for all tokens and project to hidden_size
        // TODO: Properly handle mixed text/codec tokens based on vocab ranges
        let text_embeds = self.text_embedding.forward(input_ids)?;
        self.text_projection.forward(&text_embeds)
    }

    /// Forward pass with pre-computed embeddings
    pub fn forward_with_embeds(
        &self,
        embeds: &Tensor,
        start_pos: usize,
        mut cache: Option<&mut TalkerCache>,
        position_ids: Option<&Tensor>,
    ) -> Result<Tensor> {
        let mut x = embeds.clone();
        for (idx, layer) in self.layers.iter().enumerate() {
            let cache_ref = cache.as_deref_mut();
            x = layer.forward(&x, start_pos, position_ids, cache_ref, idx)?;
        }
        let x = self.norm.forward(&x)?;
        self.lm_head.forward(&x).map_err(Error::from)
    }

    /// Check if using MRoPE
    pub fn uses_mrope(&self) -> bool {
        self.use_mrope
    }
}

/// Repeat key/value heads for Grouped Query Attention
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

/// Build standard RoPE cache
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

/// Build MRoPE cache for multi-modal position encoding
fn build_mrope_cache(
    seq_len: usize,
    head_dim: usize,
    rope_theta: f64,
    device: &Device,
    dtype: DType,
    position_ids: &Tensor,
    mrope_section: &[usize],
) -> Result<(Tensor, Tensor)> {
    let half_dim = head_dim / 2;

    // Validate mrope_section
    if mrope_section.is_empty() || mrope_section.iter().sum::<usize>() != half_dim {
        return build_rope_cache(seq_len, head_dim, 0, rope_theta, device, dtype);
    }

    let positions = position_ids.to_vec2::<i64>()?;
    if positions.len() != 3 {
        return build_rope_cache(seq_len, head_dim, 0, rope_theta, device, dtype);
    }

    // Build inverse frequencies
    let mut inv_freq = Vec::with_capacity(half_dim);
    for i in 0..half_dim {
        let power = (2.0 * i as f64) / head_dim as f64;
        inv_freq.push((1.0 / rope_theta.powf(power)) as f32);
    }

    // Build cos/sin for each of the 3 axes
    let mut cos_axes = Vec::with_capacity(3);
    let mut sin_axes = Vec::with_capacity(3);

    for axis in 0..3 {
        let mut angles = Vec::with_capacity(seq_len * half_dim);
        for &pos in positions[axis].iter() {
            let p = pos as f32;
            for &inv in inv_freq.iter() {
                angles.push(p * inv);
            }
        }
        let angles = Tensor::from_vec(angles, (seq_len, half_dim), device)?;
        cos_axes.push(angles.cos()?.to_dtype(dtype)?);
        sin_axes.push(angles.sin()?.to_dtype(dtype)?);
    }

    // Concatenate sections from different axes
    let mut cos_parts = Vec::with_capacity(3);
    let mut sin_parts = Vec::with_capacity(3);
    let mut start = 0usize;

    for (axis, section) in mrope_section.iter().enumerate() {
        let cos = cos_axes[axis].narrow(1, start, *section)?;
        let sin = sin_axes[axis].narrow(1, start, *section)?;
        cos_parts.push(cos);
        sin_parts.push(sin);
        start += *section;
    }

    let cos = Tensor::cat(&cos_parts, 1)?;
    let sin = Tensor::cat(&sin_parts, 1)?;
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
