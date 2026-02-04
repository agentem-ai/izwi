//! Audio tower for Qwen3-ASR.

use candle_core::{IndexOp, Module, Tensor, D};
use candle_nn::ops;
use candle_nn::{layer_norm, Conv2d, Conv2dConfig, LayerNorm, Linear, VarBuilder};

use crate::error::Result;
use crate::models::qwen3_asr::config::AudioConfig;

/// Compute output length after CNN downsampling (3 conv2d layers with stride=2).
/// Each layer roughly halves the length, giving ~8x total downsampling.
pub fn get_cnn_output_lengths(input_lengths: &[usize]) -> Vec<usize> {
    input_lengths
        .iter()
        .map(|&len| {
            // Apply 3 conv2d layers with stride=2, kernel=3, padding=1
            // Formula: output = (input + 2*padding - kernel) / stride + 1
            // = (input - 1) / 2 + 1 = ceil(input / 2)
            let mut out = len;
            for _ in 0..3 {
                out = (out + 1) / 2; // Equivalent to ceil(out / 2)
            }
            out
        })
        .collect()
}

/// Compute output length after a single conv2d with stride=2, kernel=3, padding=1.
fn conv_output_len(input_len: usize) -> usize {
    (input_len.saturating_sub(1)) / 2 + 1
}

struct SinusoidalPositionEmbedding {
    embedding: Tensor,
}

impl SinusoidalPositionEmbedding {
    fn new(max_len: usize, channels: usize, device: &candle_core::Device) -> Result<Self> {
        let half_channels = channels / 2;
        let log_timescale = (10000f32).ln() / (half_channels as f32 - 1.0);
        let inv_timescales: Vec<f32> = (0..half_channels)
            .map(|i| (-log_timescale * i as f32).exp())
            .collect();

        let mut embedding_data = Vec::with_capacity(max_len * channels);
        for pos in 0..max_len {
            for i in 0..half_channels {
                let timescale = inv_timescales[i];
                embedding_data.push((pos as f32 * timescale).sin());
            }
            for i in 0..half_channels {
                let timescale = inv_timescales[i];
                embedding_data.push((pos as f32 * timescale).cos());
            }
        }

        let embedding = Tensor::from_vec(embedding_data, (max_len, channels), device)?;
        Ok(Self { embedding })
    }

    fn get(&self, seqlen: usize) -> Result<Tensor> {
        Ok(self.embedding.narrow(0, 0, seqlen)?)
    }
}

/// Create attention mask for chunked sequences using cu_seqlens
fn create_chunked_attention_mask(
    seq_len: usize,
    cu_seqlens: &[i64],
    device: &candle_core::Device,
    dtype: candle_core::DType,
) -> Result<Tensor> {
    let min_val = f32::MIN;
    let mut mask = vec![min_val; seq_len * seq_len];

    // For each chunk, allow attention within the chunk
    for i in 1..cu_seqlens.len() {
        let start = cu_seqlens[i - 1] as usize;
        let end = cu_seqlens[i] as usize;
        for row in start..end {
            for col in start..end {
                mask[row * seq_len + col] = 0.0;
            }
        }
    }

    Tensor::from_vec(mask, (seq_len, seq_len), device)?
        .to_dtype(dtype)
        .map_err(|e| crate::error::Error::InferenceError(e.to_string()))
}

struct AudioAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    num_heads: usize,
    head_dim: usize,
}

impl AudioAttention {
    fn load(cfg: &AudioConfig, vb: VarBuilder) -> Result<Self> {
        let head_dim = cfg.d_model / cfg.encoder_attention_heads;
        let q_proj = candle_nn::linear(cfg.d_model, cfg.d_model, vb.pp("q_proj"))?;
        let k_proj = candle_nn::linear(cfg.d_model, cfg.d_model, vb.pp("k_proj"))?;
        let v_proj = candle_nn::linear(cfg.d_model, cfg.d_model, vb.pp("v_proj"))?;
        let out_proj = candle_nn::linear(cfg.d_model, cfg.d_model, vb.pp("out_proj"))?;
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            out_proj,
            num_heads: cfg.encoder_attention_heads,
            head_dim,
        })
    }

    fn forward(&self, x: &Tensor, cu_seqlens: &[i64]) -> Result<Tensor> {
        let seq_len = x.dim(1)?;

        let q = self
            .q_proj
            .forward(x)?
            .reshape((1, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = self
            .k_proj
            .forward(x)?
            .reshape((1, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = self
            .v_proj
            .forward(x)?
            .reshape((1, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;

        let q = q.reshape((self.num_heads, seq_len, self.head_dim))?;
        let k = k.reshape((self.num_heads, seq_len, self.head_dim))?;
        let v = v.reshape((self.num_heads, seq_len, self.head_dim))?;

        let scale = (self.head_dim as f64).powf(-0.5);
        let mut attn = q.matmul(&k.transpose(1, 2)?)?;
        attn = attn.broadcast_mul(
            &Tensor::from_vec(vec![scale as f32], (1, 1, 1), attn.device())?
                .to_dtype(attn.dtype())?,
        )?;

        // Apply chunked attention mask
        let mask = create_chunked_attention_mask(seq_len, cu_seqlens, x.device(), attn.dtype())?;
        attn = attn.broadcast_add(&mask.unsqueeze(0)?)?;

        let attn = ops::softmax(&attn, D::Minus1)?;
        let out = attn.matmul(&v)?;

        let out = out
            .reshape((1, self.num_heads, seq_len, self.head_dim))?
            .transpose(1, 2)?
            .reshape((1, seq_len, self.num_heads * self.head_dim))?;
        self.out_proj
            .forward(&out)
            .map_err(|e| crate::error::Error::InferenceError(e.to_string()))
    }
}

struct AudioEncoderLayer {
    self_attn_layer_norm: LayerNorm,
    self_attn: AudioAttention,
    final_layer_norm: LayerNorm,
    fc1: Linear,
    fc2: Linear,
}

impl AudioEncoderLayer {
    fn load(cfg: &AudioConfig, vb: VarBuilder) -> Result<Self> {
        let self_attn_layer_norm = layer_norm(cfg.d_model, 1e-5, vb.pp("self_attn_layer_norm"))?;
        let self_attn = AudioAttention::load(cfg, vb.pp("self_attn"))?;
        let final_layer_norm = layer_norm(cfg.d_model, 1e-5, vb.pp("final_layer_norm"))?;
        let fc1 = candle_nn::linear(cfg.d_model, cfg.encoder_ffn_dim, vb.pp("fc1"))?;
        let fc2 = candle_nn::linear(cfg.encoder_ffn_dim, cfg.d_model, vb.pp("fc2"))?;
        Ok(Self {
            self_attn_layer_norm,
            self_attn,
            final_layer_norm,
            fc1,
            fc2,
        })
    }

    fn forward(&self, x: &Tensor, cu_seqlens: &[i64]) -> Result<Tensor> {
        let normed = self.self_attn_layer_norm.forward(x)?;
        let attn = self.self_attn.forward(&normed, cu_seqlens)?;
        let x = x.broadcast_add(&attn)?;

        let normed = self.final_layer_norm.forward(&x)?;
        let hidden = self.fc1.forward(&normed)?;
        let hidden = gelu(&hidden)?;
        let hidden = self.fc2.forward(&hidden)?;
        let x = x.broadcast_add(&hidden)?;

        Ok(x)
    }
}

pub struct AudioTower {
    conv2d1: Conv2d,
    conv2d2: Conv2d,
    conv2d3: Conv2d,
    conv_out: Linear,
    layers: Vec<AudioEncoderLayer>,
    ln_post: LayerNorm,
    proj1: Linear,
    proj2: Linear,
    pos_embed: SinusoidalPositionEmbedding,
    cfg: AudioConfig,
}

impl AudioTower {
    pub fn load(cfg: AudioConfig, vb: VarBuilder) -> Result<Self> {
        let conv_cfg = Conv2dConfig {
            stride: 2,
            padding: 1,
            ..Default::default()
        };

        let conv2d1 =
            candle_nn::conv2d(1, cfg.downsample_hidden_size, 3, conv_cfg, vb.pp("conv2d1"))?;
        let conv2d2 = candle_nn::conv2d(
            cfg.downsample_hidden_size,
            cfg.downsample_hidden_size,
            3,
            conv_cfg,
            vb.pp("conv2d2"),
        )?;
        let conv2d3 = candle_nn::conv2d(
            cfg.downsample_hidden_size,
            cfg.downsample_hidden_size,
            3,
            conv_cfg,
            vb.pp("conv2d3"),
        )?;

        let conv_out = candle_nn::linear_no_bias(
            cfg.downsample_hidden_size * (cfg.num_mel_bins / 8),
            cfg.d_model,
            vb.pp("conv_out"),
        )?;

        let mut layers = Vec::with_capacity(cfg.encoder_layers);
        for idx in 0..cfg.encoder_layers {
            layers.push(AudioEncoderLayer::load(
                &cfg,
                vb.pp(format!("layers.{idx}")),
            )?);
        }

        let ln_post = layer_norm(cfg.d_model, 1e-5, vb.pp("ln_post"))?;
        let proj1 = candle_nn::linear(cfg.d_model, cfg.d_model, vb.pp("proj1"))?;
        let proj2 = candle_nn::linear(cfg.d_model, cfg.output_dim, vb.pp("proj2"))?;
        let pos_embed = SinusoidalPositionEmbedding::new(1500, cfg.d_model, vb.device())?;

        Ok(Self {
            conv2d1,
            conv2d2,
            conv2d3,
            conv_out,
            layers,
            ln_post,
            proj1,
            proj2,
            pos_embed,
            cfg,
        })
    }

    pub fn forward(&self, mel: &Tensor, feature_lens: Option<&[usize]>) -> Result<Tensor> {
        let _mel_len = mel.dim(3)?;

        // mel: [b, 1, n_mels, frames]
        let mut x = self.conv2d1.forward(mel)?;
        x = gelu(&x)?;
        x = self.conv2d2.forward(&x)?;
        x = gelu(&x)?;
        x = self.conv2d3.forward(&x)?;
        x = gelu(&x)?;

        let bsz = x.dim(0)?;
        let channels = x.dim(1)?;
        let freq = x.dim(2)?;
        let frames = x.dim(3)?;

        // [b, c, f, t] -> [b, t, c, f]
        let x = x.transpose(1, 3)?.transpose(2, 3)?;
        let x = x.reshape((bsz, frames, channels * freq))?;

        let mut x = self.conv_out.forward(&x)?;

        // Add positional embedding (convert to match x's dtype)
        let pos_emb = self.pos_embed.get(x.dim(1)?)?;
        let pos_emb = pos_emb.unsqueeze(0)?.to_dtype(x.dtype())?;
        x = x.broadcast_add(&pos_emb)?;

        // Create cu_seqlens based on feature_lens
        let cu_seqlens = if let Some(lens) = feature_lens {
            let n_window = self.cfg.n_window.unwrap_or(100);
            let n_window_infer = self.cfg.n_window_infer.unwrap_or(400);
            let window_frames = frames * (n_window_infer / (n_window * 2));

            let mut cu = vec![0i64];
            for &len in lens {
                let num_full = len / window_frames;
                let remainder = len % window_frames;
                for _ in 0..num_full {
                    cu.push(*cu.last().unwrap() + window_frames as i64);
                }
                if remainder > 0 {
                    cu.push(*cu.last().unwrap() + remainder as i64);
                }
            }
            cu
        } else {
            vec![0, x.dim(1)? as i64]
        };

        for layer in &self.layers {
            x = layer.forward(&x, &cu_seqlens)?;
        }

        let x = self.ln_post.forward(&x)?;
        let x = self.proj1.forward(&x)?;
        let x = gelu(&x)?;
        let x = self.proj2.forward(&x)?;
        Ok(x)
    }
}

fn gelu(x: &Tensor) -> Result<Tensor> {
    let coeff = 0.044715f32;
    let sqrt_2_over_pi = (2.0f32 / std::f32::consts::PI).sqrt();
    let dtype = x.dtype();
    let x3 = x.powf(3.0)?;
    let coeff_t = Tensor::from_vec(vec![coeff], (1,), x.device())?.to_dtype(dtype)?;
    let x3 = x3.broadcast_mul(&coeff_t)?;
    let sqrt_t = Tensor::from_vec(vec![sqrt_2_over_pi], (1,), x.device())?.to_dtype(dtype)?;
    let inner = (x + x3)?.broadcast_mul(&sqrt_t)?;
    let tanh = inner.tanh()?;
    let one = Tensor::from_vec(vec![1.0f32], (1,), x.device())?.to_dtype(dtype)?;
    let half = Tensor::from_vec(vec![0.5f32], (1,), x.device())?.to_dtype(dtype)?;
    let out = x.broadcast_mul(&one.broadcast_add(&tanh)?)?;
    let out = out.broadcast_mul(&half)?;
    Ok(out)
}
