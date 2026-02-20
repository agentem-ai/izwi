use candle_core::{DType, Result as CandleResult, Tensor};
use candle_nn::{Conv1d, Conv1dConfig, Linear, Module, RmsNorm, VarBuilder};

use crate::error::{Error, Result};
use crate::models::shared::attention::flash::{
    flash_attention_requested, try_fused_self_attention,
};

use super::config::LfmConfig;

#[derive(Debug, Clone)]
pub struct LfmCache {
    layers: Vec<LayerCache>,
}

#[derive(Debug, Clone)]
enum LayerCache {
    Attention {
        k: Option<Tensor>,
        v: Option<Tensor>,
    },
    ShortConv {
        state: Option<Tensor>,
    },
}

impl LfmCache {
    pub fn new(cfg: &LfmConfig) -> Self {
        let layers = cfg
            .layer_types
            .iter()
            .map(|kind| {
                if kind == "full_attention" {
                    LayerCache::Attention { k: None, v: None }
                } else {
                    LayerCache::ShortConv { state: None }
                }
            })
            .collect();
        Self { layers }
    }

    pub fn reset(&mut self) {
        for layer in &mut self.layers {
            match layer {
                LayerCache::Attention { k, v } => {
                    *k = None;
                    *v = None;
                }
                LayerCache::ShortConv { state } => {
                    *state = None;
                }
            }
        }
    }
}

struct Mlp {
    w1: Linear,
    w2: Linear,
    w3: Linear,
}

impl Mlp {
    fn load(vb: VarBuilder) -> Result<Self> {
        let w1 = vb
            .pp("feed_forward.w1")
            .get_unchecked_dtype("weight", vb.dtype())?;
        let w2 = vb
            .pp("feed_forward.w2")
            .get_unchecked_dtype("weight", vb.dtype())?;
        let w3 = vb
            .pp("feed_forward.w3")
            .get_unchecked_dtype("weight", vb.dtype())?;

        let w1 = Linear::new(w1, None);
        let w2 = Linear::new(w2, None);
        let w3 = Linear::new(w3, None);
        Ok(Self { w1, w2, w3 })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let w1 = self.w1.forward(xs)?;
        let w3 = self.w3.forward(xs)?;
        let gate = candle_nn::ops::silu(&w1)?;
        Ok(self.w2.forward(&(gate * w3)?)?)
    }
}

struct AttentionLayer {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
    cos: Tensor,
    sin: Tensor,
    neg_inf: Tensor,
}

impl AttentionLayer {
    fn load(cfg: &LfmConfig, layer_idx: usize, vb: VarBuilder) -> Result<Self> {
        let hidden = cfg.hidden_size;
        let n_head = cfg.num_attention_heads;
        let n_kv_head = cfg.num_key_value_heads;
        let head_dim = hidden / n_head;

        let q_proj = candle_nn::linear_no_bias(hidden, hidden, vb.pp("self_attn.q_proj"))?;
        let k_proj =
            candle_nn::linear_no_bias(hidden, n_kv_head * head_dim, vb.pp("self_attn.k_proj"))?;
        let v_proj =
            candle_nn::linear_no_bias(hidden, n_kv_head * head_dim, vb.pp("self_attn.v_proj"))?;
        let out_proj = candle_nn::linear_no_bias(hidden, hidden, vb.pp("self_attn.out_proj"))?;

        let q_norm = candle_nn::rms_norm(head_dim, cfg.norm_eps, vb.pp("self_attn.q_layernorm"))?;
        let k_norm = candle_nn::rms_norm(head_dim, cfg.norm_eps, vb.pp("self_attn.k_layernorm"))?;

        let (cos, sin) = precompute_freqs_cis(
            head_dim,
            cfg.rope_theta as f32,
            cfg.max_position_embeddings,
            vb.device(),
        )?;
        let neg_inf = Tensor::new(f32::NEG_INFINITY, vb.device())?;

        let _ = layer_idx;
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            out_proj,
            q_norm,
            k_norm,
            n_head,
            n_kv_head,
            head_dim,
            cos,
            sin,
            neg_inf,
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        cache: &mut LayerCache,
        index_pos: usize,
        mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (b, seq_len, hidden) = xs.dims3()?;

        let q = self.q_proj.forward(xs)?;
        let k = self.k_proj.forward(xs)?;
        let v = self.v_proj.forward(xs)?;

        let q = q
            .reshape((b, seq_len, self.n_head, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b, seq_len, self.n_kv_head, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b, seq_len, self.n_kv_head, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        let q = apply_rms_head_norm(&self.q_norm, &q)?;
        let k = apply_rms_head_norm(&self.k_norm, &k)?;

        let q = self.apply_rotary_emb(&q, index_pos)?;
        let k = self.apply_rotary_emb(&k, index_pos)?;

        let (k, v) = match cache {
            LayerCache::Attention { k: kc, v: vc } => {
                if index_pos == 0 || kc.is_none() || vc.is_none() {
                    *kc = Some(k.clone());
                    *vc = Some(v.clone());
                    (k, v)
                } else {
                    let merged_k = Tensor::cat(&[kc.as_ref().unwrap(), &k], 2)?;
                    let merged_v = Tensor::cat(&[vc.as_ref().unwrap(), &v], 2)?;
                    *kc = Some(merged_k.clone());
                    *vc = Some(merged_v.clone());
                    (merged_k, merged_v)
                }
            }
            LayerCache::ShortConv { .. } => {
                return Err(Error::InferenceError(
                    "Invalid LFM cache type for attention layer".to_string(),
                ));
            }
        };

        let k = repeat_kv(&k, self.n_head / self.n_kv_head)?;
        let v = repeat_kv(&v, self.n_head / self.n_kv_head)?;
        let total_len = k.dim(2)?;

        // Fused kernels are valid for regular prefill (no cache offset) and for
        // single-token decode where causal masking is implicit in the cache.
        if flash_attention_requested()
            && mask.is_none()
            && ((index_pos == 0 && total_len == seq_len) || seq_len == 1)
        {
            let causal = index_pos == 0 && seq_len > 1;
            if let Some(fused_out) =
                try_fused_self_attention(&q, &k, &v, None, self.head_dim, causal)?
            {
                let y = fused_out.transpose(1, 2)?.reshape((b, seq_len, hidden))?;
                return self.out_proj.forward(&y).map_err(Error::from);
            }
        }

        let att = (q.matmul(&k.t()?)? / (self.head_dim as f64).sqrt())?;
        let att = match mask {
            None => att,
            Some(mask) => {
                let mask = mask.broadcast_as(att.shape())?;
                masked_fill(&att, &mask, &self.neg_inf)?
            }
        };
        let att = candle_nn::ops::softmax_last_dim(&att)?;
        let y = att.matmul(&v.contiguous()?)?;

        let y = y.transpose(1, 2)?.reshape((b, seq_len, hidden))?;
        self.out_proj.forward(&y).map_err(Error::from)
    }

    fn apply_rotary_emb(&self, x: &Tensor, index_pos: usize) -> Result<Tensor> {
        let (_b, _h, seq_len, _d) = x.dims4()?;
        let cos = self.cos.narrow(0, index_pos, seq_len)?;
        let sin = self.sin.narrow(0, index_pos, seq_len)?;
        candle_nn::rotary_emb::rope(&x.contiguous()?, &cos, &sin).map_err(Error::from)
    }
}

struct ShortConvLayer {
    in_proj: Linear,
    out_proj: Linear,
    conv: Tensor,
    l_cache: usize,
}

impl ShortConvLayer {
    fn load(hidden: usize, l_cache: usize, vb: VarBuilder) -> Result<Self> {
        let in_proj = candle_nn::linear_no_bias(hidden, hidden * 3, vb.pp("conv.in_proj"))?;
        let out_proj = candle_nn::linear_no_bias(hidden, hidden, vb.pp("conv.out_proj"))?;
        let conv = vb.get_unchecked_dtype("conv.conv.weight", vb.dtype())?;

        Ok(Self {
            in_proj,
            out_proj,
            conv,
            l_cache,
        })
    }

    fn forward(&self, xs: &Tensor, cache: &mut LayerCache) -> Result<Tensor> {
        let (b, seq_len, hidden) = xs.dims3()?;
        let bcx = self.in_proj.forward(xs)?.transpose(1, 2)?;
        let b_gate = bcx.narrow(1, 0, hidden)?;
        let c_gate = bcx.narrow(1, hidden, hidden)?;
        let x = bcx.narrow(1, 2 * hidden, hidden)?;
        let bx = (b_gate * &x)?.contiguous()?;

        let mut conv_weight = self.conv.clone();
        if conv_weight.dims().len() == 3 {
            conv_weight = conv_weight.squeeze(1)?;
        }
        let conv_weight = conv_weight.contiguous()?;

        let mut conv_out = if seq_len == 1 {
            let mut state = match cache {
                LayerCache::ShortConv { state } => {
                    if let Some(state) = state.clone() {
                        state
                    } else {
                        Tensor::zeros((b, hidden, self.l_cache), bx.dtype(), bx.device())?
                    }
                }
                LayerCache::Attention { .. } => {
                    return Err(Error::InferenceError(
                        "Invalid LFM cache type for conv layer".to_string(),
                    ));
                }
            };

            if self.l_cache > 1 {
                let tail = state.narrow(2, 1, self.l_cache - 1)?;
                state = Tensor::cat(&[tail, bx.clone()], 2)?;
            } else {
                state = bx.clone();
            }

            if let LayerCache::ShortConv { state: slot } = cache {
                *slot = Some(state.clone());
            }

            (state * &conv_weight.unsqueeze(0)?)?
                .sum_keepdim(2)?
                .contiguous()?
        } else {
            let conv = Conv1d::new(
                conv_weight
                    .reshape((hidden, 1, self.l_cache))?
                    .contiguous()?,
                None,
                Conv1dConfig {
                    padding: self.l_cache.saturating_sub(1),
                    groups: hidden,
                    ..Default::default()
                },
            );
            let mut out = conv.forward(&bx.contiguous()?)?;
            out = out.narrow(2, 0, seq_len)?;

            if self.l_cache > 0 {
                let (_, _, cur_len) = bx.dims3()?;
                let start = cur_len.saturating_sub(self.l_cache);
                let mut cache_src = bx.narrow(2, start, cur_len - start)?;
                if cache_src.dims3()?.2 < self.l_cache {
                    let pad = self.l_cache - cache_src.dims3()?.2;
                    let zeros =
                        Tensor::zeros((b, hidden, pad), cache_src.dtype(), cache_src.device())?;
                    cache_src = Tensor::cat(&[zeros, cache_src], 2)?;
                }
                if let LayerCache::ShortConv { state: slot } = cache {
                    *slot = Some(cache_src);
                }
            }

            out
        };

        conv_out = (c_gate * &conv_out)?;
        let conv_out = conv_out.transpose(1, 2)?.contiguous()?;
        self.out_proj.forward(&conv_out).map_err(Error::from)
    }
}

enum LayerKind {
    Attention(AttentionLayer),
    ShortConv(ShortConvLayer),
}

struct LayerWeights {
    operator_norm: RmsNorm,
    ffn_norm: RmsNorm,
    mlp: Mlp,
    kind: LayerKind,
}

pub struct LfmBackbone {
    cfg: LfmConfig,
    embed_tokens: candle_nn::Embedding,
    layers: Vec<LayerWeights>,
    norm: RmsNorm,
}

impl LfmBackbone {
    pub fn load(cfg: LfmConfig, vb: VarBuilder) -> Result<Self> {
        let hidden = cfg.hidden_size;

        let embed_tokens = candle_nn::embedding(cfg.vocab_size, hidden, vb.pp("embed_tokens"))?;
        let norm = candle_nn::rms_norm(hidden, cfg.norm_eps, vb.pp("embedding_norm"))?;

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for layer_idx in 0..cfg.num_hidden_layers {
            let layer_vb = vb.pp(format!("layers.{layer_idx}"));

            let operator_norm =
                candle_nn::rms_norm(hidden, cfg.norm_eps, layer_vb.pp("operator_norm"))?;
            let ffn_norm = candle_nn::rms_norm(hidden, cfg.norm_eps, layer_vb.pp("ffn_norm"))?;
            let mlp = Mlp::load(layer_vb.clone())?;

            let kind =
                if cfg.layer_types.get(layer_idx).map(|s| s.as_str()) == Some("full_attention") {
                    LayerKind::Attention(AttentionLayer::load(&cfg, layer_idx, layer_vb.clone())?)
                } else {
                    LayerKind::ShortConv(ShortConvLayer::load(
                        hidden,
                        cfg.conv_l_cache,
                        layer_vb.clone(),
                    )?)
                };

            layers.push(LayerWeights {
                operator_norm,
                ffn_norm,
                mlp,
                kind,
            });
        }

        Ok(Self {
            cfg,
            embed_tokens,
            layers,
            norm,
        })
    }

    pub fn config(&self) -> &LfmConfig {
        &self.cfg
    }

    pub fn embed_tokens_weight(&self) -> &Tensor {
        self.embed_tokens.embeddings()
    }

    pub fn embed_sequence(&self, tokens: &Tensor) -> Result<Tensor> {
        self.embed_tokens.forward(tokens).map_err(Error::from)
    }

    pub fn embed_tokens(&self, token: u32) -> Result<Tensor> {
        let token =
            Tensor::new(vec![token], self.embed_tokens_weight().device())?.reshape((1, 1))?;
        self.embed_tokens.forward(&token).map_err(Error::from)
    }

    pub fn forward_embeds_cached(
        &self,
        inputs_embeds: &Tensor,
        cache: &mut LfmCache,
    ) -> Result<Tensor> {
        let (_b, seq_len, _d) = inputs_embeds.dims3()?;
        let index_pos = first_attention_kv_len(cache);

        let mask = if seq_len == 1 {
            None
        } else {
            Some(causal_mask(seq_len, inputs_embeds.device())?)
        };

        let mut hidden = inputs_embeds.clone();
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let residual = hidden.clone();
            let normed = layer.operator_norm.forward(&hidden)?;
            hidden = match &layer.kind {
                LayerKind::Attention(attn) => attn.forward(
                    &normed,
                    &mut cache.layers[layer_idx],
                    index_pos,
                    mask.as_ref(),
                )?,
                LayerKind::ShortConv(conv) => {
                    conv.forward(&normed, &mut cache.layers[layer_idx])?
                }
            };
            hidden = (hidden + residual)?;

            let residual = hidden.clone();
            let ff = layer.ffn_norm.forward(&hidden)?;
            let ff = layer.mlp.forward(&ff)?;
            hidden = (ff + residual)?;
        }

        self.norm.forward(&hidden).map_err(Error::from)
    }

    pub fn forward_embeds(
        &self,
        inputs_embeds: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (_b, seq_len, _d) = inputs_embeds.dims3()?;
        let mut cache = LfmCache::new(&self.cfg);

        let default_mask = if seq_len == 1 {
            None
        } else {
            Some(causal_mask(seq_len, inputs_embeds.device())?)
        };
        let mask = attention_mask.or(default_mask.as_ref());

        let mut hidden = inputs_embeds.clone();
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let residual = hidden.clone();
            let normed = layer.operator_norm.forward(&hidden)?;
            hidden = match &layer.kind {
                LayerKind::Attention(attn) => {
                    attn.forward(&normed, &mut cache.layers[layer_idx], 0, mask)?
                }
                LayerKind::ShortConv(conv) => {
                    conv.forward(&normed, &mut cache.layers[layer_idx])?
                }
            };
            hidden = (hidden + residual)?;

            let residual = hidden.clone();
            let ff = layer.ffn_norm.forward(&hidden)?;
            let ff = layer.mlp.forward(&ff)?;
            hidden = (ff + residual)?;
        }

        self.norm.forward(&hidden).map_err(Error::from)
    }
}

fn first_attention_kv_len(cache: &LfmCache) -> usize {
    for layer in &cache.layers {
        if let LayerCache::Attention { k: Some(k), .. } = layer {
            return k.dim(2).unwrap_or(0);
        }
    }
    0
}

fn precompute_freqs_cis(
    head_dim: usize,
    freq_base: f32,
    context_length: usize,
    device: &candle_core::Device,
) -> Result<(Tensor, Tensor)> {
    let theta: Vec<_> = (0..head_dim)
        .step_by(2)
        .map(|i| 1f32 / freq_base.powf(i as f32 / head_dim as f32))
        .collect();
    let theta = Tensor::new(theta.as_slice(), device)?;
    let idx_theta = Tensor::arange(0, context_length as u32, device)?
        .to_dtype(DType::F32)?
        .reshape((context_length, 1))?
        .matmul(&theta.reshape((1, theta.elem_count()))?)?;
    Ok((idx_theta.cos()?, idx_theta.sin()?))
}

fn apply_rms_head_norm(norm: &RmsNorm, x: &Tensor) -> Result<Tensor> {
    let (b, h, t, d) = x.dims4()?;
    let flat = x.reshape((b * h * t, d))?;
    let normed = norm.forward(&flat)?;
    normed.reshape((b, h, t, d)).map_err(Error::from)
}

fn repeat_kv(xs: &Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep == 1 {
        return Ok(xs.clone());
    }
    let (b, n_kv, t, d) = xs.dims4()?;
    xs.unsqueeze(2)?
        .expand((b, n_kv, n_rep, t, d))?
        .reshape((b, n_kv * n_rep, t, d))
        .map_err(Error::from)
}

fn masked_fill(on_false: &Tensor, mask: &Tensor, on_true: &Tensor) -> CandleResult<Tensor> {
    let shape = mask.shape();
    mask.where_cond(&on_true.broadcast_as(shape.dims())?, on_false)
}

fn causal_mask(t: usize, device: &candle_core::Device) -> Result<Tensor> {
    let mask: Vec<u8> = (0..t)
        .flat_map(|i| (0..t).map(move |j| u8::from(j > i)))
        .collect();
    Tensor::from_slice(&mask, (t, t), device).map_err(Error::from)
}
