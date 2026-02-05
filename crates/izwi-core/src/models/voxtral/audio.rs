//! Time embedding for Voxtral Realtime delay conditioning.
//!
//! Sinusoidal embedding for encoding time/delay tokens.

use candle_core::{Result, Tensor, D};
use candle_nn::Module;

/// Sinusoidal Time Embedding for encoding delay tokens
pub struct TimeEmbedding {
    dim: usize,
    theta: f32,
    inv_freq: Tensor,
}

impl TimeEmbedding {
    pub fn new(dim: usize, theta: f32, device: &candle_core::Device) -> Result<Self> {
        let half_dim = dim / 2;
        let log_theta = theta.ln();
        let inv_freq: Vec<f32> = (0..half_dim)
            .map(|i| (-log_theta * i as f32 / (half_dim as f32 - 1.0)).exp())
            .collect();
        let inv_freq = Tensor::from_vec(inv_freq, (half_dim,), device)?;

        Ok(Self {
            dim,
            theta,
            inv_freq,
        })
    }

    /// Forward pass
    /// t: (B,) -> (B, dim) or (B, T) -> (B, T, dim)
    pub fn forward(&self, t: &Tensor) -> Result<Tensor> {
        // Add dimension: (B,) -> (B, 1) or (B, T) -> (B, T, 1)
        let t = t.unsqueeze(D::Minus1)?;

        // Broadcast: (B, 1) x (dim/2,) -> (B, dim/2)
        let emb = t.broadcast_mul(&self.inv_freq)?;

        // cat([cos(emb), sin(emb)], dim=-1)
        let cos_emb = emb.cos()?;
        let sin_emb = emb.sin()?;
        let emb = Tensor::cat(&[&cos_emb, &sin_emb], D::Minus1)?;

        Ok(emb)
    }
}

/// Audio-Language Adapter for projecting audio features to LM dimension
pub struct AudioLanguageAdapter {
    w_in: candle_nn::Linear,
    w_out: candle_nn::Linear,
}

impl AudioLanguageAdapter {
    pub fn load(hidden_size: usize, dim: usize, vb: candle_nn::VarBuilder) -> Result<Self> {
        let w_in = candle_nn::linear_no_bias(hidden_size, dim, vb.pp("w_in"))?;
        let w_out = candle_nn::linear_no_bias(dim, dim, vb.pp("w_out"))?;

        Ok(Self { w_in, w_out })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.w_in.forward(x)?;
        let x = gelu(&x)?;
        self.w_out.forward(&x)
    }
}

/// GELU activation function
fn gelu(x: &Tensor) -> Result<Tensor> {
    let coeff = 0.044715f64;
    let sqrt_2_over_pi = (2.0f64 / std::f64::consts::PI).sqrt();
    let dtype = x.dtype();
    let x_f64 = x.to_dtype(candle_core::DType::F64)?;
    let x3 = x_f64.powf(3.0)?;
    let coeff_t = Tensor::from_vec(vec![coeff], (1,), x.device())?;
    let x3 = x3.broadcast_mul(&coeff_t)?;
    let sqrt_t = Tensor::from_vec(vec![sqrt_2_over_pi], (1,), x.device())?;
    let inner = (&x_f64 + x3)?.broadcast_mul(&sqrt_t)?;
    let tanh = inner.tanh()?;
    let one = Tensor::from_vec(vec![1.0f64], (1,), x.device())?;
    let half = Tensor::from_vec(vec![0.5f64], (1,), x.device())?;
    let out = x_f64.broadcast_mul(&one.broadcast_add(&tanh)?)?;
    let out = out.broadcast_mul(&half)?;
    out.to_dtype(dtype)
}
