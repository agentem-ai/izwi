use candle_core::{IndexOp, Tensor};
use candle_nn::ops;
use candle_nn::{
    batch_norm, layer_norm, Conv1d, Conv1dConfig, Conv2d, Conv2dConfig, LayerNorm, Linear, Module,
    ModuleT, VarBuilder,
};

use crate::error::{Error, Result};
use crate::models::shared::weights::mlx;

use super::config::ConformerConfig;

fn has_any_tensor(vb: &VarBuilder, names: &[&str]) -> bool {
    names.iter().any(|name| vb.contains_tensor(name))
}

pub struct ConformerEncoder {
    cfg: ConformerConfig,
    pre_encode: ConvSubsamplingDw,
    layers: Vec<ConformerLayer>,
}

impl ConformerEncoder {
    pub fn load(cfg: ConformerConfig, vb: VarBuilder) -> Result<Self> {
        let pre_encode = ConvSubsamplingDw::load(&cfg, vb.pp("pre_encode"))?;
        let mut layers = Vec::with_capacity(cfg.n_layers);
        for idx in 0..cfg.n_layers {
            layers.push(ConformerLayer::load(&cfg, vb.pp(format!("layers.{idx}")))?);
        }

        Ok(Self {
            cfg,
            pre_encode,
            layers,
        })
    }

    pub fn output_dim(&self) -> usize {
        self.cfg.d_model
    }

    pub fn encode(&self, features: &Tensor, feature_frames: usize) -> Result<(Tensor, usize)> {
        let (mut x, encoded_len) = self.pre_encode.forward(features, feature_frames)?;
        let pos_len = x.dim(1)?;
        let pos_emb = build_rel_positional_embedding(pos_len, self.cfg.d_model, x.device())?;

        for layer in &self.layers {
            x = layer.forward(&x, &pos_emb)?;
        }

        Ok((x, encoded_len))
    }
}

struct ConvSubsamplingDw {
    cfg: ConformerConfig,
    conv0: Conv2d,
    conv2: Conv2d,
    conv3: Conv2d,
    conv5: Conv2d,
    conv6: Conv2d,
    out: Linear,
}

impl ConvSubsamplingDw {
    fn load(cfg: &ConformerConfig, vb: VarBuilder) -> Result<Self> {
        let stride_cfg = Conv2dConfig {
            stride: 2,
            padding: 1,
            ..Default::default()
        };
        let point_cfg = Conv2dConfig {
            stride: 1,
            padding: 0,
            ..Default::default()
        };

        let channels = cfg.subsampling_conv_channels;

        let conv0 = mlx::load_conv2d(1, channels, 3, stride_cfg, vb.pp("conv.0"))?;

        let mut dw_stride_cfg = stride_cfg;
        dw_stride_cfg.groups = channels;
        let conv2 = mlx::load_conv2d(1, channels, 3, dw_stride_cfg, vb.pp("conv.2"))?;
        let conv3 = mlx::load_conv2d(channels, channels, 1, point_cfg, vb.pp("conv.3"))?;
        let conv5 = mlx::load_conv2d(1, channels, 3, dw_stride_cfg, vb.pp("conv.5"))?;
        let conv6 = mlx::load_conv2d(channels, channels, 1, point_cfg, vb.pp("conv.6"))?;

        let flatten_freq = cfg.feat_in / cfg.subsampling_factor.max(1);
        let out_in = channels * flatten_freq.max(1);
        let out = mlx::load_linear(out_in, cfg.d_model, vb.pp("out"))?;

        Ok(Self {
            cfg: cfg.clone(),
            conv0,
            conv2,
            conv3,
            conv5,
            conv6,
            out,
        })
    }

    fn forward(&self, features: &Tensor, feature_frames: usize) -> Result<(Tensor, usize)> {
        // [B, MELS, T] -> [B, 1, T, MELS]
        let mut x = features.transpose(1, 2)?.unsqueeze(1)?;

        x = self.conv0.forward(&x)?;
        x = x.relu()?;

        x = self.conv2.forward(&x)?;
        x = self.conv3.forward(&x)?;
        x = x.relu()?;

        x = self.conv5.forward(&x)?;
        x = self.conv6.forward(&x)?;
        x = x.relu()?;

        let (b, c, t, f) = x.dims4()?;
        let x = x
            .transpose(1, 2)?
            .reshape((b, t, c * f))?
            .apply(&self.out)?;

        let encoded_len = subsampled_len_3x(feature_frames).min(t);
        Ok((x, encoded_len))
    }
}

fn subsampled_len_3x(mut len: usize) -> usize {
    for _ in 0..3 {
        len = len.div_ceil(2);
    }
    len
}

struct ConformerLayer {
    cfg: ConformerConfig,
    norm_ff1: LayerNorm,
    ff1: FeedForward,
    norm_self_att: LayerNorm,
    self_attn: RelPosSelfAttention,
    norm_conv: LayerNorm,
    conv: ConformerConv,
    norm_ff2: LayerNorm,
    ff2: FeedForward,
    norm_out: LayerNorm,
}

impl ConformerLayer {
    fn load(cfg: &ConformerConfig, vb: VarBuilder) -> Result<Self> {
        let d_model = cfg.d_model;

        let norm_ff1_path = if has_any_tensor(&vb, &["norm_feed_forward1.weight"]) {
            "norm_feed_forward1"
        } else {
            "ff1_norm"
        };
        let ff1_path = if has_any_tensor(
            &vb,
            &[
                "feed_forward1.linear1.weight",
                "feed_forward1.linear1.scales",
            ],
        ) {
            "feed_forward1"
        } else {
            "ff1"
        };
        let norm_self_att_path = if has_any_tensor(&vb, &["norm_self_att.weight"]) {
            "norm_self_att"
        } else {
            "attn_norm"
        };
        let self_attn_path = if has_any_tensor(
            &vb,
            &["self_attn.linear_q.weight", "self_attn.linear_q.scales"],
        ) {
            "self_attn"
        } else {
            "attn"
        };
        let norm_conv_path = if has_any_tensor(&vb, &["norm_conv.weight"]) {
            "norm_conv"
        } else {
            "conv_norm"
        };
        let norm_ff2_path = if has_any_tensor(&vb, &["norm_feed_forward2.weight"]) {
            "norm_feed_forward2"
        } else {
            "ff2_norm"
        };
        let ff2_path = if has_any_tensor(
            &vb,
            &[
                "feed_forward2.linear1.weight",
                "feed_forward2.linear1.scales",
            ],
        ) {
            "feed_forward2"
        } else {
            "ff2"
        };
        let norm_out_path = if has_any_tensor(&vb, &["norm_out.weight"]) {
            "norm_out"
        } else {
            "final_norm"
        };

        let norm_ff1 = layer_norm(d_model, 1e-5, vb.pp(norm_ff1_path))?;
        let ff1 = FeedForward::load(cfg, vb.pp(ff1_path))?;

        let norm_self_att = layer_norm(d_model, 1e-5, vb.pp(norm_self_att_path))?;
        let self_attn = RelPosSelfAttention::load(cfg, vb.pp(self_attn_path))?;

        let norm_conv = layer_norm(d_model, 1e-5, vb.pp(norm_conv_path))?;
        let conv = ConformerConv::load(cfg, vb.pp("conv"))?;

        let norm_ff2 = layer_norm(d_model, 1e-5, vb.pp(norm_ff2_path))?;
        let ff2 = FeedForward::load(cfg, vb.pp(ff2_path))?;

        let norm_out = layer_norm(d_model, 1e-5, vb.pp(norm_out_path))?;

        Ok(Self {
            cfg: cfg.clone(),
            norm_ff1,
            ff1,
            norm_self_att,
            self_attn,
            norm_conv,
            conv,
            norm_ff2,
            ff2,
            norm_out,
        })
    }

    fn forward(&self, x: &Tensor, pos_emb: &Tensor) -> Result<Tensor> {
        let mut residual = x.clone();

        let ff1 = self.ff1.forward(&self.norm_ff1.forward(&residual)?)?;
        residual = residual.broadcast_add(&ff1.affine(0.5, 0.0)?)?;

        let attn = self
            .self_attn
            .forward(&self.norm_self_att.forward(&residual)?, pos_emb)?;
        residual = residual.broadcast_add(&attn)?;

        let conv = self.conv.forward(&self.norm_conv.forward(&residual)?)?;
        residual = residual.broadcast_add(&conv)?;

        let ff2 = self.ff2.forward(&self.norm_ff2.forward(&residual)?)?;
        residual = residual.broadcast_add(&ff2.affine(0.5, 0.0)?)?;

        self.norm_out
            .forward(&residual)
            .map_err(|e| Error::InferenceError(e.to_string()))
    }
}

struct FeedForward {
    linear1: Linear,
    linear2: Linear,
}

impl FeedForward {
    fn load(cfg: &ConformerConfig, vb: VarBuilder) -> Result<Self> {
        let ff_dim = cfg.d_model * cfg.ff_expansion_factor;
        let linear1 = mlx::load_linear(cfg.d_model, ff_dim, vb.pp("linear1"))?;
        let linear2 = mlx::load_linear(ff_dim, cfg.d_model, vb.pp("linear2"))?;
        Ok(Self { linear1, linear2 })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.linear1.forward(x)?;
        let x = swish(&x)?;
        self.linear2
            .forward(&x)
            .map_err(|e| Error::InferenceError(e.to_string()))
    }
}

struct ConformerConv {
    d_model: usize,
    pointwise_conv1: Conv1d,
    depthwise_conv: Conv1d,
    batch_norm: candle_nn::BatchNorm,
    pointwise_conv2: Conv1d,
}

impl ConformerConv {
    fn load(cfg: &ConformerConfig, vb: VarBuilder) -> Result<Self> {
        let d_model = cfg.d_model;

        let pointwise_conv1 = mlx::load_conv1d_no_bias(
            d_model,
            d_model * 2,
            1,
            Conv1dConfig::default(),
            vb.pp("pointwise_conv1"),
        )?;

        let depthwise_conv = mlx::load_conv1d_no_bias(
            d_model,
            d_model,
            cfg.conv_kernel_size,
            Conv1dConfig {
                padding: (cfg.conv_kernel_size - 1) / 2,
                groups: d_model,
                ..Default::default()
            },
            vb.pp("depthwise_conv"),
        )?;

        let batch_norm_path = if has_any_tensor(&vb, &["batch_norm.weight"]) {
            "batch_norm"
        } else {
            "norm"
        };
        let batch_norm = batch_norm(d_model, 1e-5, vb.pp(batch_norm_path))?;

        let pointwise_conv2 = mlx::load_conv1d_no_bias(
            d_model,
            d_model,
            1,
            Conv1dConfig::default(),
            vb.pp("pointwise_conv2"),
        )?;

        Ok(Self {
            d_model,
            pointwise_conv1,
            depthwise_conv,
            batch_norm,
            pointwise_conv2,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut x = x.transpose(1, 2)?;

        x = self.pointwise_conv1.forward(&x)?;
        let x_a = x.i((.., ..self.d_model, ..))?;
        let x_b = x.i((.., self.d_model.., ..))?;
        x = x_a.broadcast_mul(&ops::sigmoid(&x_b)?)?;

        x = self.depthwise_conv.forward(&x)?;
        x = self.batch_norm.forward_t(&x, false)?;
        x = swish(&x)?;
        x = self.pointwise_conv2.forward(&x)?;

        x.transpose(1, 2).map_err(Error::from)
    }
}

struct RelPosSelfAttention {
    d_model: usize,
    n_heads: usize,
    head_dim: usize,
    linear_q: Linear,
    linear_k: Linear,
    linear_v: Linear,
    linear_out: Linear,
    linear_pos: Linear,
    pos_bias_u: Tensor,
    pos_bias_v: Tensor,
}

impl RelPosSelfAttention {
    fn load(cfg: &ConformerConfig, vb: VarBuilder) -> Result<Self> {
        let d_model = cfg.d_model;
        let n_heads = cfg.n_heads;
        let head_dim = d_model / n_heads;

        let linear_q_path = if has_any_tensor(&vb, &["linear_q.weight", "linear_q.scales"]) {
            "linear_q"
        } else {
            "q_proj"
        };
        let linear_k_path = if has_any_tensor(&vb, &["linear_k.weight", "linear_k.scales"]) {
            "linear_k"
        } else {
            "k_proj"
        };
        let linear_v_path = if has_any_tensor(&vb, &["linear_v.weight", "linear_v.scales"]) {
            "linear_v"
        } else {
            "v_proj"
        };
        let linear_out_path = if has_any_tensor(&vb, &["linear_out.weight", "linear_out.scales"]) {
            "linear_out"
        } else {
            "out_proj"
        };
        let linear_pos_path = if has_any_tensor(&vb, &["linear_pos.weight", "linear_pos.scales"]) {
            "linear_pos"
        } else {
            "pos_proj"
        };

        let linear_q = mlx::load_linear(d_model, d_model, vb.pp(linear_q_path))?;
        let linear_k = mlx::load_linear(d_model, d_model, vb.pp(linear_k_path))?;
        let linear_v = mlx::load_linear(d_model, d_model, vb.pp(linear_v_path))?;
        let linear_out = mlx::load_linear(d_model, d_model, vb.pp(linear_out_path))?;
        let linear_pos = mlx::load_linear_no_bias(d_model, d_model, vb.pp(linear_pos_path))?;

        let pos_bias_u = vb.get((n_heads, head_dim), "pos_bias_u")?;
        let pos_bias_v = vb.get((n_heads, head_dim), "pos_bias_v")?;

        Ok(Self {
            d_model,
            n_heads,
            head_dim,
            linear_q,
            linear_k,
            linear_v,
            linear_out,
            linear_pos,
            pos_bias_u,
            pos_bias_v,
        })
    }

    fn forward(&self, x: &Tensor, pos_emb: &Tensor) -> Result<Tensor> {
        let (b, t, _d) = x.dims3()?;

        let q = self
            .linear_q
            .forward(x)?
            .reshape((b, t, self.n_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = self
            .linear_k
            .forward(x)?
            .reshape((b, t, self.n_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = self
            .linear_v
            .forward(x)?
            .reshape((b, t, self.n_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        let p = self
            .linear_pos
            .forward(pos_emb)?
            .reshape((1, 2 * t - 1, self.n_heads, self.head_dim))?
            .transpose(1, 2)?;

        let pos_bias_u = self
            .pos_bias_u
            .reshape((1, self.n_heads, 1, self.head_dim))?;
        let pos_bias_v = self
            .pos_bias_v
            .reshape((1, self.n_heads, 1, self.head_dim))?;

        let q_u = q.broadcast_add(&pos_bias_u)?;
        let q_v = q.broadcast_add(&pos_bias_v)?;

        let k_t = k.transpose(2, 3)?.contiguous()?;
        let p_t = p.transpose(2, 3)?.contiguous()?;
        let matrix_ac = q_u.matmul(&k_t)?;
        let matrix_bd = rel_shift(&q_v.matmul(&p_t)?)?;
        let matrix_bd = matrix_bd.narrow(3, 0, t)?;

        let scores = matrix_ac
            .broadcast_add(&matrix_bd)?
            .affine(1.0 / (self.head_dim as f64).sqrt(), 0.0)?;
        let attn = ops::softmax(&scores, 3)?;

        let out = attn.matmul(&v)?;
        let out = out.transpose(1, 2)?.reshape((b, t, self.d_model))?;

        self.linear_out
            .forward(&out)
            .map_err(|e| Error::InferenceError(e.to_string()))
    }
}

fn rel_shift(x: &Tensor) -> Result<Tensor> {
    let (b, h, qlen, pos_len) = x.dims4()?;
    let x = x.pad_with_zeros(3, 1, 0)?;
    let x = x.reshape((b, h, pos_len + 1, qlen))?;
    let x = x.narrow(2, 1, pos_len)?;
    x.reshape((b, h, qlen, pos_len)).map_err(Error::from)
}

fn swish(x: &Tensor) -> Result<Tensor> {
    x.broadcast_mul(&ops::sigmoid(x)?)
        .map_err(|e| Error::InferenceError(e.to_string()))
}

fn build_rel_positional_embedding(
    seq_len: usize,
    d_model: usize,
    device: &candle_core::Device,
) -> Result<Tensor> {
    let total = seq_len.saturating_mul(2).saturating_sub(1);
    let start = -(seq_len as isize - 1);
    let half = d_model / 2;
    let mut out = vec![0f32; total * d_model];

    for p in 0..total {
        let pos = (start + p as isize) as f32;
        for i in 0..half {
            let exponent = (2.0 * (i / 2) as f32) / d_model as f32;
            let inv = 1.0f32 / 10_000f32.powf(exponent);
            let angle = pos * inv;
            let base = i * 2;
            if base < d_model {
                out[p * d_model + base] = angle.sin();
            }
            if base + 1 < d_model {
                out[p * d_model + base + 1] = angle.cos();
            }
        }
    }

    Tensor::from_vec(out, (total, d_model), device).map_err(Error::from)
}
