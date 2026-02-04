//! Audio tower for Qwen3-ASR.

use candle_core::{Module, Tensor, D};
use candle_nn::{layer_norm, Conv2d, Conv2dConfig, LayerNorm, Linear, VarBuilder};
use candle_nn::ops;

use crate::error::Result;
use crate::models::qwen3_asr::config::AudioConfig;

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

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let bsz = x.dim(0)?;
        let seq_len = x.dim(1)?;

        let q = self
            .q_proj
            .forward(x)?
            .reshape((bsz, seq_len, self.num_heads, self.head_dim))?;
        let k = self
            .k_proj
            .forward(x)?
            .reshape((bsz, seq_len, self.num_heads, self.head_dim))?;
        let v = self
            .v_proj
            .forward(x)?
            .reshape((bsz, seq_len, self.num_heads, self.head_dim))?;

        let q = q.transpose(1, 2)?; // [b, h, s, d]
        let k = k.transpose(1, 2)?;
        let v = v.transpose(1, 2)?;

        let q = q.reshape((bsz * self.num_heads, seq_len, self.head_dim))?;
        let k = k.reshape((bsz * self.num_heads, seq_len, self.head_dim))?;
        let v = v.reshape((bsz * self.num_heads, seq_len, self.head_dim))?;

        let mut att = q.matmul(&k.transpose(1, 2)?)?;
        let scale = (self.head_dim as f64).sqrt();
        let scale_t = Tensor::from_vec(vec![scale as f32], (1,), att.device())?
            .to_dtype(att.dtype())?;
        att = att.broadcast_div(&scale_t)?;
        let att = ops::softmax(&att, D::Minus1)?;
        let out = att.matmul(&v)?;
        let out = out.reshape((bsz, self.num_heads, seq_len, self.head_dim))?;
        let out = out.transpose(1, 2)?.reshape((bsz, seq_len, self.num_heads * self.head_dim))?;
        let out = self.out_proj.forward(&out)?;
        Ok(out)
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

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let normed = self.self_attn_layer_norm.forward(x)?;
        let attn = self.self_attn.forward(&normed)?;
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
    cfg: AudioConfig,
}

impl AudioTower {
    pub fn load(cfg: AudioConfig, vb: VarBuilder) -> Result<Self> {
        let conv_cfg = Conv2dConfig {
            stride: 2,
            padding: 1,
            ..Default::default()
        };

        let conv2d1 = candle_nn::conv2d(1, cfg.downsample_hidden_size, 3, conv_cfg, vb.pp("conv2d1"))?;
        let conv2d2 = candle_nn::conv2d(cfg.downsample_hidden_size, cfg.downsample_hidden_size, 3, conv_cfg, vb.pp("conv2d2"))?;
        let conv2d3 = candle_nn::conv2d(cfg.downsample_hidden_size, cfg.downsample_hidden_size, 3, conv_cfg, vb.pp("conv2d3"))?;

        let conv_out = candle_nn::linear_no_bias(cfg.downsample_hidden_size * (cfg.num_mel_bins / 8), cfg.d_model, vb.pp("conv_out"))?;

        let mut layers = Vec::with_capacity(cfg.encoder_layers);
        for idx in 0..cfg.encoder_layers {
            layers.push(AudioEncoderLayer::load(&cfg, vb.pp(format!("layers.{idx}")))?);
        }

        let ln_post = layer_norm(cfg.d_model, 1e-5, vb.pp("ln_post"))?;
        let proj1 = candle_nn::linear(cfg.d_model, cfg.d_model, vb.pp("proj1"))?;
        let proj2 = candle_nn::linear(cfg.d_model, cfg.output_dim, vb.pp("proj2"))?;

        Ok(Self {
            conv2d1,
            conv2d2,
            conv2d3,
            conv_out,
            layers,
            ln_post,
            proj1,
            proj2,
            cfg,
        })
    }

    pub fn forward(&self, mel: &Tensor) -> Result<Tensor> {
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

        for layer in &self.layers {
            x = layer.forward(&x)?;
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
    let coeff_t = Tensor::from_vec(vec![coeff], (1,), x.device())?
        .to_dtype(dtype)?;
    let x3 = x3.broadcast_mul(&coeff_t)?;
    let sqrt_t = Tensor::from_vec(vec![sqrt_2_over_pi], (1,), x.device())?
        .to_dtype(dtype)?;
    let inner = (x + x3)?.broadcast_mul(&sqrt_t)?;
    let tanh = inner.tanh()?;
    let one = Tensor::from_vec(vec![1.0f32], (1,), x.device())?
        .to_dtype(dtype)?;
    let half = Tensor::from_vec(vec![0.5f32], (1,), x.device())?
        .to_dtype(dtype)?;
    let out = x.broadcast_mul(&one.broadcast_add(&tanh)?)?;
    let out = out.broadcast_mul(&half)?;
    Ok(out)
}
