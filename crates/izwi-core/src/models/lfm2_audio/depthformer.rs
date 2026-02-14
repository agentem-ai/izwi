use candle_core::{DType, IndexOp, Tensor};
use candle_nn::{Embedding, Linear, Module, RmsNorm, VarBuilder};

use crate::error::{Error, Result};

use super::config::Lfm2AudioConfig;

const AUDIO_VOCAB_SIZE: usize = 2049;

#[derive(Debug, Clone)]
pub struct DepthformerCache {
    layers: Vec<Option<(Tensor, Tensor)>>,
}

impl DepthformerCache {
    fn new(n: usize) -> Self {
        Self {
            layers: vec![None; n],
        }
    }
}

struct SharedEmbedding {
    embedding: Embedding,
    embedding_norm: RmsNorm,
    to_logits: Linear,
}

impl SharedEmbedding {
    fn load(vocab_size: usize, dim: usize, vb: VarBuilder) -> Result<Self> {
        let embedding = candle_nn::embedding(vocab_size, dim, vb.pp("embedding"))?;
        let embedding_norm = candle_nn::rms_norm(dim, 1e-5, vb.pp("embedding_norm"))?;
        let to_logits = candle_nn::linear_no_bias(dim, vocab_size, vb.pp("to_logits"))?;

        Ok(Self {
            embedding,
            embedding_norm,
            to_logits,
        })
    }

    fn embed(&self, tokens: &Tensor) -> Result<Tensor> {
        self.embedding.forward(tokens).map_err(Error::from)
    }

    fn logits(&self, embeddings: &Tensor) -> Result<Tensor> {
        let x = self.embedding_norm.forward(embeddings)?;
        self.to_logits.forward(&x).map_err(Error::from)
    }
}

struct Mha {
    qkv_proj: Linear,
    out_proj: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    num_heads: usize,
    gqa_dim: usize,
    head_dim: usize,
    cos: Tensor,
    sin: Tensor,
}

impl Mha {
    fn load(dim: usize, vb: VarBuilder) -> Result<Self> {
        let qkv_proj = candle_nn::linear_no_bias(dim, dim + 2 * 8 * (dim / 32), vb.pp("qkv_proj"))?;
        let out_proj = candle_nn::linear_no_bias(dim, dim, vb.pp("out_proj"))?;
        let head_dim = dim / 32;
        let q_norm = candle_nn::rms_norm(head_dim, 1e-5, vb.pp("bounded_attention.q_layernorm"))?;
        let k_norm = candle_nn::rms_norm(head_dim, 1e-5, vb.pp("bounded_attention.k_layernorm"))?;

        let (cos, sin) = precompute_freqs_cis(head_dim, 1_000_000.0, 4096, vb.device())?;

        Ok(Self {
            qkv_proj,
            out_proj,
            q_norm,
            k_norm,
            num_heads: 32,
            gqa_dim: 8,
            head_dim,
            cos,
            sin,
        })
    }

    fn forward_cached(
        &self,
        x: &Tensor,
        cache: Option<(Tensor, Tensor)>,
    ) -> Result<(Tensor, Option<(Tensor, Tensor)>)> {
        let (b, t, dim) = x.dims3()?;
        let packed = self.qkv_proj.forward(x)?;
        let q = packed.narrow(2, 0, dim)?;
        let k = packed.narrow(2, dim, self.head_dim * self.gqa_dim)?;
        let v = packed.narrow(
            2,
            dim + self.head_dim * self.gqa_dim,
            self.head_dim * self.gqa_dim,
        )?;

        let q = q
            .reshape((b, t, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b, t, self.gqa_dim, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b, t, self.gqa_dim, self.head_dim))?
            .transpose(1, 2)?;

        let q = apply_rms_head_norm(&self.q_norm, &q)?;
        let k = apply_rms_head_norm(&self.k_norm, &k)?;

        let cache_len = cache
            .as_ref()
            .map(|(k, _)| k.dim(2).unwrap_or(0))
            .unwrap_or(0);

        let q = apply_rope(&q, &self.cos, &self.sin, cache_len)?;
        let k = apply_rope(&k, &self.cos, &self.sin, cache_len)?;

        let (k, v) = if let Some((pk, pv)) = cache {
            let k = Tensor::cat(&[&pk, &k], 2)?;
            let v = Tensor::cat(&[&pv, &v], 2)?;
            (k, v)
        } else {
            (k, v)
        };

        let new_cache = Some((k.clone(), v.clone()));

        let k = repeat_kv(&k, self.num_heads / self.gqa_dim)?;
        let v = repeat_kv(&v, self.num_heads / self.gqa_dim)?;

        let q = q.transpose(1, 2)?; // [B,T,H,D]
        let k = k.transpose(1, 2)?;
        let v = v.transpose(1, 2)?;

        let q = q.reshape((b * self.num_heads, t, self.head_dim))?;
        let total_t = k.dim(1)?;
        let k = k.reshape((b * self.num_heads, total_t, self.head_dim))?;
        let v = v.reshape((b * self.num_heads, total_t, self.head_dim))?;

        let mut scores = q.matmul(&k.transpose(1, 2)?)?;
        let scale = Tensor::new(vec![(self.head_dim as f32).sqrt()], scores.device())?
            .reshape((1, 1, 1))?;
        scores = scores.broadcast_div(&scale)?;

        // Causal mask (right aligned for cached decoding).
        if t > 1 || cache_len == 0 {
            let mask = causal_mask(t, total_t, scores.device())?;
            scores = scores.broadcast_add(&mask)?;
        }

        let attn = candle_nn::ops::softmax_last_dim(&scores)?;
        let out = attn.matmul(&v)?;
        let out = out.reshape((b, self.num_heads, t, self.head_dim))?;
        let out = out.transpose(1, 2)?.reshape((b, t, dim))?;

        let out = self.out_proj.forward(&out)?;
        Ok((out, new_cache))
    }
}

struct Glu {
    w1: Linear,
    w2: Linear,
    w3: Linear,
}

impl Glu {
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

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let w1 = self.w1.forward(x)?;
        let w3 = self.w3.forward(x)?;
        self.w2
            .forward(&(candle_nn::ops::silu(&w1)? * w3)?)
            .map_err(Error::from)
    }
}

struct StandardBlock {
    operator_norm: RmsNorm,
    operator: Mha,
    ffn_norm: RmsNorm,
    ff: Glu,
}

impl StandardBlock {
    fn load(dim: usize, vb: VarBuilder) -> Result<Self> {
        let operator_norm = candle_nn::rms_norm(dim, 1e-5, vb.pp("operator_norm"))?;
        let operator = Mha::load(dim, vb.pp("operator"))?;
        let ffn_norm = candle_nn::rms_norm(dim, 1e-5, vb.pp("ffn_norm"))?;
        let ff = Glu::load(vb)?;

        Ok(Self {
            operator_norm,
            operator,
            ffn_norm,
            ff,
        })
    }

    fn forward_cached(
        &self,
        x: &Tensor,
        cache: Option<(Tensor, Tensor)>,
    ) -> Result<(Tensor, Option<(Tensor, Tensor)>)> {
        let norm = self.operator_norm.forward(x)?;
        let (h, cache) = self.operator.forward_cached(&norm, cache)?;
        let h = (h + x)?;
        let ff = self.ff.forward(&self.ffn_norm.forward(&h)?)?;
        Ok(((h + ff)?, cache))
    }
}

pub struct Depthformer {
    pub codebooks: usize,
    pub audio_vocab_size: usize,
    pub codebook_offsets: Tensor,
    dim: usize,
    audio_embedding: SharedEmbedding,
    depth_linear: Linear,
    depth_embeddings: Vec<SharedEmbedding>,
    layers: Vec<StandardBlock>,
}

impl Depthformer {
    pub fn load(cfg: &Lfm2AudioConfig, vb: VarBuilder) -> Result<Self> {
        let codebooks = cfg.codebooks;
        let dim = cfg.depthformer.dim;
        let hidden = cfg.lfm.hidden_size;

        let audio_embedding = SharedEmbedding::load(
            AUDIO_VOCAB_SIZE * codebooks,
            hidden,
            vb.pp("audio_embedding"),
        )?;
        let depth_linear = candle_nn::linear(hidden, dim * codebooks, vb.pp("depth_linear"))?;

        let mut depth_embeddings = Vec::with_capacity(codebooks);
        for i in 0..codebooks {
            depth_embeddings.push(SharedEmbedding::load(
                AUDIO_VOCAB_SIZE,
                dim,
                vb.pp(format!("depth_embeddings.{i}")),
            )?);
        }

        let mut layers = Vec::with_capacity(cfg.depthformer.layers);
        for i in 0..cfg.depthformer.layers {
            layers.push(StandardBlock::load(
                dim,
                vb.pp(format!("depthformer.layers.{i}")),
            )?);
        }

        let offsets: Vec<u32> = (0..codebooks as u32)
            .map(|i| i * AUDIO_VOCAB_SIZE as u32)
            .collect();
        let codebook_offsets = Tensor::from_vec(offsets, codebooks, vb.device())?;

        Ok(Self {
            codebooks,
            audio_vocab_size: AUDIO_VOCAB_SIZE,
            codebook_offsets,
            dim,
            audio_embedding,
            depth_linear,
            depth_embeddings,
            layers,
        })
    }

    pub fn audio_embedding_sum(&self, frame_tokens: &Tensor) -> Result<Tensor> {
        let offsets = self.codebook_offsets.to_dtype(DType::U32)?;
        let tokens = frame_tokens.broadcast_add(&offsets)?;
        let emb = self.audio_embedding.embed(&tokens)?; // [C, H]
        emb.sum(0)?
            .reshape((1, 1, emb.dim(1)?))
            .map_err(Error::from)
    }

    pub fn sample_audio_frame(
        &self,
        embedding: &Tensor,
        temperature: Option<f32>,
        top_k: Option<usize>,
        rng: &mut super::SimpleRng,
    ) -> Result<Vec<u32>> {
        let greedy = temperature.unwrap_or(0.0) <= 0.0 || top_k == Some(1);

        let emb = if embedding.dims().len() == 1 {
            embedding.reshape((1, embedding.dim(0)?))?
        } else {
            embedding.clone()
        };

        let depth_in = self
            .depth_linear
            .forward(&emb)?
            .reshape((self.codebooks, self.dim))?;

        let mut token_embed = Tensor::zeros(self.dim, depth_in.dtype(), depth_in.device())?;

        let mut out = Vec::with_capacity(self.codebooks);
        let mut cache = DepthformerCache::new(self.layers.len());

        for i in 0..self.codebooks {
            let cur_in = depth_in.i(i)?.broadcast_add(&token_embed)?;
            let mut x = cur_in.reshape((1, 1, cur_in.dim(0)?))?;

            for (layer_idx, layer) in self.layers.iter().enumerate() {
                let (y, new_cache) = layer.forward_cached(&x, cache.layers[layer_idx].clone())?;
                cache.layers[layer_idx] = new_cache;
                x = y;
            }

            let logits = self.depth_embeddings[i].logits(&x.squeeze(0)?)?;
            let logits = if logits.dims().len() > 1 {
                logits.squeeze(0)?
            } else {
                logits
            };
            let token = super::sample_token(&logits, greedy, temperature, top_k, rng)?;
            out.push(token);

            let token_t = Tensor::new(vec![token], embedding.device())?;
            token_embed = self.depth_embeddings[i].embed(&token_t)?.squeeze(0)?;
        }

        Ok(out)
    }
}

fn precompute_freqs_cis(
    head_dim: usize,
    theta: f64,
    end: usize,
    device: &candle_core::Device,
) -> Result<(Tensor, Tensor)> {
    let freqs: Vec<f32> = (0..head_dim)
        .step_by(2)
        .map(|i| 1.0f32 / (theta as f32).powf(i as f32 / head_dim as f32))
        .collect();
    let freqs = Tensor::from_vec(freqs, (1, head_dim / 2), device)?;
    let t = Tensor::arange(0u32, end as u32, device)?
        .to_dtype(DType::F32)?
        .reshape((end, 1))?;
    let freqs = t.broadcast_mul(&freqs)?;
    Ok((freqs.cos()?, freqs.sin()?))
}

fn apply_rope(x: &Tensor, cos: &Tensor, sin: &Tensor, start: usize) -> Result<Tensor> {
    let (_b, _h, t, _d) = x.dims4()?;
    let cos = cos.narrow(0, start, t)?;
    let sin = sin.narrow(0, start, t)?;
    // LFM2 Depthformer uses interleaved rotary embeddings (complex-pair layout),
    // which maps to candle's rope_i implementation.
    candle_nn::rotary_emb::rope_i(&x.contiguous()?, &cos, &sin).map_err(Error::from)
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

fn causal_mask(q_len: usize, kv_len: usize, device: &candle_core::Device) -> Result<Tensor> {
    let mut mask = vec![0f32; q_len * kv_len];
    for i in 0..q_len {
        let max_k = kv_len.saturating_sub(q_len - i);
        for j in 0..kv_len {
            if j > max_k {
                mask[i * kv_len + j] = f32::NEG_INFINITY;
            }
        }
    }
    Tensor::from_vec(mask, (1, q_len, kv_len), device).map_err(Error::from)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn apply_rope_matches_interleaved_reference() {
        let device = candle_core::Device::Cpu;
        let x = Tensor::from_vec(
            (0..(2 * 3 * 4 * 8))
                .map(|v| v as f32 / 32.0)
                .collect::<Vec<_>>(),
            (2, 3, 4, 8),
            &device,
        )
        .expect("tensor");
        let (cos, sin) = precompute_freqs_cis(8, 1_000_000.0, 32, &device).expect("rope cache");

        let got = apply_rope(&x, &cos, &sin, 5).expect("apply_rope");
        let ref_cos = cos.narrow(0, 5, 4).expect("narrow cos");
        let ref_sin = sin.narrow(0, 5, 4).expect("narrow sin");
        let expected =
            candle_nn::rotary_emb::rope_i(&x.contiguous().expect("contig"), &ref_cos, &ref_sin)
                .expect("rope_i");

        let got = got
            .flatten_all()
            .expect("flatten got")
            .to_vec1::<f32>()
            .expect("vec got");
        let expected = expected
            .flatten_all()
            .expect("flatten expected")
            .to_vec1::<f32>()
            .expect("vec expected");
        assert_eq!(got.len(), expected.len());
        for (lhs, rhs) in got.iter().zip(expected.iter()) {
            assert!((lhs - rhs).abs() < 1e-5);
        }
    }
}
