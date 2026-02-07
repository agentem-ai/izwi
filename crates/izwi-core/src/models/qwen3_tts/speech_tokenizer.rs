//! Speech Tokenizer Decoder for Qwen3-TTS.
//!
//! Converts multi-codebook RVQ codec tokens to waveform audio.
//! Implementation mirrors the official 12Hz decoder architecture:
//! RVQ projection -> pre-conv -> pre-transformer -> upsample stack -> decoder blocks -> waveform.

use candle_core::{DType, Device, Tensor, D};
use candle_nn::{
    ops, Conv1d, Conv1dConfig, ConvTranspose1d, ConvTranspose1dConfig, LayerNorm, LayerNormConfig,
    Linear, Module, RmsNorm, VarBuilder,
};
use serde::Deserialize;
use tracing::info;

use crate::error::{Error, Result};

/// Speech Tokenizer Configuration
#[derive(Debug, Clone, Deserialize)]
pub struct SpeechTokenizerConfig {
    pub model_type: String,
    pub architectures: Vec<String>,
    pub encoder_valid_num_quantizers: usize,
    pub input_sample_rate: usize,
    pub output_sample_rate: usize,
    pub decode_upsample_rate: usize,
    pub encode_downsample_rate: usize,
    pub decoder_config: DecoderConfig,
    pub encoder_config: Option<serde_json::Value>,
}

/// Decoder configuration
#[derive(Debug, Clone, Deserialize)]
pub struct DecoderConfig {
    pub latent_dim: usize,
    pub codebook_dim: usize,
    pub codebook_size: usize,
    pub decoder_dim: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub hidden_act: String,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub num_quantizers: usize,
    pub num_semantic_quantizers: usize,
    pub semantic_codebook_size: usize,
    pub upsample_rates: Vec<usize>,
    pub upsampling_ratios: Vec<usize>,
    pub vector_quantization_hidden_dimension: usize,
}

/// Causal Conv1d with left padding only.
struct CausalConv1d {
    conv: Conv1d,
    causal_padding: usize,
}

impl CausalConv1d {
    fn load(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        dilation: usize,
        groups: usize,
        with_bias: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let config = Conv1dConfig {
            padding: 0,
            stride: 1,
            dilation,
            groups,
            ..Default::default()
        };
        let conv = if with_bias {
            candle_nn::conv1d(in_channels, out_channels, kernel_size, config, vb)?
        } else {
            candle_nn::conv1d_no_bias(in_channels, out_channels, kernel_size, config, vb)?
        };
        Ok(Self {
            conv,
            causal_padding: dilation * (kernel_size - 1),
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = if self.causal_padding > 0 {
            x.pad_with_zeros(2, self.causal_padding, 0)?
        } else {
            x.clone()
        };
        self.conv.forward(&x).map_err(Error::from)
    }
}

/// Causal transposed conv that trims the right side to preserve exact stride upsampling.
struct CausalTransConv1d {
    conv: ConvTranspose1d,
    right_trim: usize,
}

impl CausalTransConv1d {
    fn load(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let config = ConvTranspose1dConfig {
            padding: 0,
            output_padding: 0,
            stride,
            dilation: 1,
            groups: 1,
        };
        let conv = candle_nn::conv_transpose1d(in_channels, out_channels, kernel_size, config, vb)?;
        Ok(Self {
            conv,
            right_trim: kernel_size.saturating_sub(stride),
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let out = self.conv.forward(x)?;
        if self.right_trim == 0 {
            return Ok(out);
        }
        let out_len = out.dim(2)?;
        let keep = out_len.saturating_sub(self.right_trim);
        out.narrow(2, 0, keep).map_err(Error::from)
    }
}

/// SnakeBeta activation.
struct SnakeBeta {
    alpha: Tensor,
    beta: Tensor,
}

impl SnakeBeta {
    fn load(channels: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            alpha: vb.get((channels,), "alpha")?,
            beta: vb.get((channels,), "beta")?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let alpha = self.alpha.unsqueeze(0)?.unsqueeze(2)?.exp()?;
        let beta = self.beta.unsqueeze(0)?.unsqueeze(2)?.exp()?;
        let sin2 = x.broadcast_mul(&alpha)?.sin()?.sqr()?;
        let inv_beta = beta
            .broadcast_add(&Tensor::new(1e-9f32, x.device())?)?
            .recip()?;
        x.broadcast_add(&sin2.broadcast_mul(&inv_beta)?)
            .map_err(Error::from)
    }
}

/// Residual unit used inside decoder blocks.
struct ResidualUnit {
    act1: SnakeBeta,
    conv1: CausalConv1d,
    act2: SnakeBeta,
    conv2: CausalConv1d,
}

impl ResidualUnit {
    fn load(channels: usize, dilation: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            act1: SnakeBeta::load(channels, vb.pp("act1"))?,
            conv1: CausalConv1d::load(
                channels,
                channels,
                7,
                dilation,
                1,
                true,
                vb.pp("conv1.conv"),
            )?,
            act2: SnakeBeta::load(channels, vb.pp("act2"))?,
            conv2: CausalConv1d::load(channels, channels, 1, 1, 1, true, vb.pp("conv2.conv"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let residual = x.clone();
        let hidden = self.act1.forward(x)?;
        let hidden = self.conv1.forward(&hidden)?;
        let hidden = self.act2.forward(&hidden)?;
        let hidden = self.conv2.forward(&hidden)?;
        residual.broadcast_add(&hidden).map_err(Error::from)
    }
}

/// BigVGAN-style decoder block.
struct DecoderBlock {
    snake: SnakeBeta,
    upsample: CausalTransConv1d,
    res1: ResidualUnit,
    res2: ResidualUnit,
    res3: ResidualUnit,
}

impl DecoderBlock {
    fn load(
        in_channels: usize,
        out_channels: usize,
        upsample_rate: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        Ok(Self {
            snake: SnakeBeta::load(in_channels, vb.pp("block.0"))?,
            upsample: CausalTransConv1d::load(
                in_channels,
                out_channels,
                upsample_rate * 2,
                upsample_rate,
                vb.pp("block.1.conv"),
            )?,
            res1: ResidualUnit::load(out_channels, 1, vb.pp("block.2"))?,
            res2: ResidualUnit::load(out_channels, 3, vb.pp("block.3"))?,
            res3: ResidualUnit::load(out_channels, 9, vb.pp("block.4"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let hidden = self.snake.forward(x)?;
        let hidden = self.upsample.forward(&hidden)?;
        let hidden = self.res1.forward(&hidden)?;
        let hidden = self.res2.forward(&hidden)?;
        self.res3.forward(&hidden)
    }
}

/// ConvNeXt block used after upsample transposed conv.
struct ConvNeXtBlock {
    dwconv: CausalConv1d,
    norm: LayerNorm,
    pwconv1: Linear,
    pwconv2: Linear,
    gamma: Tensor,
}

impl ConvNeXtBlock {
    fn load(dim: usize, vb: VarBuilder) -> Result<Self> {
        let dwconv = CausalConv1d::load(dim, dim, 7, 1, dim, true, vb.pp("dwconv.conv"))?;
        let norm = candle_nn::layer_norm(
            dim,
            LayerNormConfig {
                eps: 1e-6,
                ..Default::default()
            },
            vb.pp("norm"),
        )?;
        let pwconv1 = candle_nn::linear(dim, 4 * dim, vb.pp("pwconv1"))?;
        let pwconv2 = candle_nn::linear(4 * dim, dim, vb.pp("pwconv2"))?;
        let gamma = vb.get((dim,), "gamma")?;
        Ok(Self {
            dwconv,
            norm,
            pwconv1,
            pwconv2,
            gamma,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let residual = x.clone();
        let hidden = self.dwconv.forward(x)?;
        let hidden = hidden.transpose(1, 2)?;
        let hidden = self.norm.forward(&hidden)?;
        let hidden = self.pwconv1.forward(&hidden)?;
        let hidden = hidden.gelu_erf()?;
        let hidden = self.pwconv2.forward(&hidden)?;
        let hidden = hidden.broadcast_mul(&self.gamma)?;
        let hidden = hidden.transpose(1, 2)?;
        residual.broadcast_add(&hidden).map_err(Error::from)
    }
}

/// Upsample stage: transposed conv + ConvNeXt block.
struct UpsampleStage {
    trans_conv: CausalTransConv1d,
    convnext: ConvNeXtBlock,
}

impl UpsampleStage {
    fn load(channels: usize, ratio: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            trans_conv: CausalTransConv1d::load(channels, channels, ratio, ratio, vb.pp("0.conv"))?,
            convnext: ConvNeXtBlock::load(channels, vb.pp("1"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let hidden = self.trans_conv.forward(x)?;
        self.convnext.forward(&hidden)
    }
}

/// Multi-head attention for pre-transformer.
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
    fn load(cfg: &DecoderConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            q_proj: candle_nn::linear_no_bias(
                cfg.hidden_size,
                cfg.num_attention_heads * cfg.head_dim,
                vb.pp("q_proj"),
            )?,
            k_proj: candle_nn::linear_no_bias(
                cfg.hidden_size,
                cfg.num_key_value_heads * cfg.head_dim,
                vb.pp("k_proj"),
            )?,
            v_proj: candle_nn::linear_no_bias(
                cfg.hidden_size,
                cfg.num_key_value_heads * cfg.head_dim,
                vb.pp("v_proj"),
            )?,
            o_proj: candle_nn::linear_no_bias(
                cfg.num_attention_heads * cfg.head_dim,
                cfg.hidden_size,
                vb.pp("o_proj"),
            )?,
            num_heads: cfg.num_attention_heads,
            num_kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
            rope_theta: cfg.rope_theta,
        })
    }

    fn apply_rope(&self, x: Tensor) -> Result<Tensor> {
        let seq_len = x.dim(1)?;
        let half_dim = self.head_dim / 2;
        let (cos, sin) = build_rope_cache(
            seq_len,
            self.head_dim,
            0,
            self.rope_theta,
            x.device(),
            x.dtype(),
        )?;

        let cos = Tensor::cat(&[cos.clone(), cos], 1)?;
        let sin = Tensor::cat(&[sin.clone(), sin], 1)?;
        let cos = cos.unsqueeze(0)?.unsqueeze(2)?;
        let sin = sin.unsqueeze(0)?.unsqueeze(2)?;

        let x1 = x.narrow(3, 0, half_dim)?;
        let x2 = x.narrow(3, half_dim, half_dim)?;
        let neg = Tensor::new(-1.0f32, x.device())?.to_dtype(x.dtype())?;
        let rotated = Tensor::cat(&[x2.broadcast_mul(&neg)?, x1], 3)?;

        x.broadcast_mul(&cos)?
            .broadcast_add(&rotated.broadcast_mul(&sin)?)
            .map_err(Error::from)
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
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

        q = self.apply_rope(q)?;
        k = self.apply_rope(k)?;

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
        let scale =
            Tensor::new((self.head_dim as f32).sqrt(), att.device())?.to_dtype(att.dtype())?;
        att = att.broadcast_div(&scale)?;

        let mask = causal_mask(seq_len, total_len, 0, att.device(), att.dtype())?;
        att = att.broadcast_add(&mask)?;

        let att = ops::softmax(&att, D::Minus1)?;
        let out = att.matmul(&v)?;

        let out = out.reshape((bsz, self.num_heads, seq_len, self.head_dim))?;
        let out = out
            .transpose(1, 2)?
            .reshape((bsz, seq_len, self.num_heads * self.head_dim))?;

        self.o_proj.forward(&out).map_err(Error::from)
    }
}

/// SwiGLU MLP block.
struct Mlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl Mlp {
    fn load(cfg: &DecoderConfig, vb: VarBuilder) -> Result<Self> {
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
        let act = ops::silu(&gate)?;
        let hidden = act.broadcast_mul(&up)?;
        self.down_proj.forward(&hidden).map_err(Error::from)
    }
}

/// Pre-transformer layer with official layer scales.
struct DecoderLayer {
    input_layernorm: RmsNorm,
    self_attn: Attention,
    self_attn_layer_scale: Tensor,
    post_attention_layernorm: RmsNorm,
    mlp: Mlp,
    mlp_layer_scale: Tensor,
}

impl DecoderLayer {
    fn load(cfg: &DecoderConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            input_layernorm: candle_nn::rms_norm(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                vb.pp("input_layernorm"),
            )?,
            self_attn: Attention::load(cfg, vb.pp("self_attn"))?,
            self_attn_layer_scale: vb
                .pp("self_attn_layer_scale")
                .get((cfg.hidden_size,), "scale")?,
            post_attention_layernorm: candle_nn::rms_norm(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                vb.pp("post_attention_layernorm"),
            )?,
            mlp: Mlp::load(cfg, vb.pp("mlp"))?,
            mlp_layer_scale: vb.pp("mlp_layer_scale").get((cfg.hidden_size,), "scale")?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let normed = self.input_layernorm.forward(x)?;
        let attn_out = self.self_attn.forward(&normed)?;
        let attn_out = attn_out.broadcast_mul(&self.self_attn_layer_scale)?;
        let x = x.broadcast_add(&attn_out)?;

        let normed = self.post_attention_layernorm.forward(&x)?;
        let mlp_out = self.mlp.forward(&normed)?;
        let mlp_out = mlp_out.broadcast_mul(&self.mlp_layer_scale)?;
        x.broadcast_add(&mlp_out).map_err(Error::from)
    }
}

/// Normalized RVQ codebook (embedding_sum / cluster_usage).
struct RVQCodebook {
    embeddings: Tensor,
    codebook_size: usize,
    codebook_dim: usize,
}

impl RVQCodebook {
    fn load(
        vb: VarBuilder,
        layer_idx: usize,
        codebook_size: usize,
        codebook_dim: usize,
    ) -> Result<Self> {
        let embedding_sum = vb.get(
            (codebook_size, codebook_dim),
            &format!("vq.layers.{layer_idx}._codebook.embedding_sum"),
        )?;
        let cluster_usage = vb.get(
            (codebook_size,),
            &format!("vq.layers.{layer_idx}._codebook.cluster_usage"),
        )?;
        let cluster_usage = cluster_usage.clamp(1e-7f64, f64::MAX)?;
        let embeddings = embedding_sum.broadcast_div(&cluster_usage.unsqueeze(1)?)?;
        Ok(Self {
            embeddings,
            codebook_size,
            codebook_dim,
        })
    }

    fn lookup(&self, indices: &Tensor) -> Result<Tensor> {
        let original_shape = indices.dims().to_vec();
        let flat_indices = indices.flatten(0, original_shape.len() - 1)?;
        let embeddings = self.embeddings.index_select(&flat_indices, 0)?;
        let mut target_shape = original_shape;
        target_shape.push(self.codebook_dim);
        embeddings
            .reshape(target_shape.as_slice())
            .map_err(Error::from)
    }
}

/// First (semantic) quantizer path.
struct RVQFirstQuantizer {
    codebook: RVQCodebook,
    output_proj: Conv1d,
}

impl RVQFirstQuantizer {
    fn load(
        vb: VarBuilder,
        codebook_size: usize,
        codebook_dim: usize,
        hidden_size: usize,
    ) -> Result<Self> {
        let codebook = RVQCodebook::load(vb.clone(), 0, codebook_size, codebook_dim)?;
        let output_proj = candle_nn::conv1d_no_bias(
            codebook_dim,
            hidden_size,
            1,
            Conv1dConfig {
                padding: 0,
                stride: 1,
                dilation: 1,
                groups: 1,
                ..Default::default()
            },
            vb.pp("output_proj"),
        )?;
        Ok(Self {
            codebook,
            output_proj,
        })
    }

    fn decode(&self, codes: &Tensor) -> Result<Tensor> {
        let flat = codes.flatten_all()?;
        let codebook_size = self.codebook.codebook_size as i64;
        let mut flat_codes: Vec<i64> = flat.to_vec1()?;
        for code in &mut flat_codes {
            *code = code.rem_euclid(codebook_size);
        }
        let mapped = Tensor::from_vec(flat_codes, flat.dims(), codes.device())?;
        let embeddings = self.codebook.lookup(&mapped)?;
        let embeddings =
            embeddings.reshape((codes.dim(0)?, codes.dim(1)?, self.codebook.codebook_dim))?;
        let embeddings = embeddings.transpose(1, 2)?;
        self.output_proj.forward(&embeddings).map_err(Error::from)
    }
}

/// Rest (acoustic) quantizer path over 15 residual codebooks.
struct RVQRestQuantizer {
    codebooks: Vec<RVQCodebook>,
    output_proj: Conv1d,
}

impl RVQRestQuantizer {
    fn load(
        vb: VarBuilder,
        num_codebooks: usize,
        codebook_size: usize,
        codebook_dim: usize,
        hidden_size: usize,
    ) -> Result<Self> {
        let mut codebooks = Vec::with_capacity(num_codebooks);
        for i in 0..num_codebooks {
            codebooks.push(RVQCodebook::load(
                vb.clone(),
                i,
                codebook_size,
                codebook_dim,
            )?);
        }
        let output_proj = candle_nn::conv1d_no_bias(
            codebook_dim,
            hidden_size,
            1,
            Conv1dConfig {
                padding: 0,
                stride: 1,
                dilation: 1,
                groups: 1,
                ..Default::default()
            },
            vb.pp("output_proj"),
        )?;
        Ok(Self {
            codebooks,
            output_proj,
        })
    }

    fn decode(&self, codec_tokens: &[Vec<u32>], seq_len: usize, device: &Device) -> Result<Tensor> {
        let codebook_dim = self.codebooks[0].codebook_dim;
        let mut rest_embed = Tensor::zeros((1, seq_len, codebook_dim), DType::F32, device)?;
        for (idx, codebook) in self.codebooks.iter().enumerate() {
            let group_tokens = codec_tokens.get(idx + 1);
            let mut values = Vec::with_capacity(seq_len);
            for t in 0..seq_len {
                let token = group_tokens.and_then(|g| g.get(t)).copied().unwrap_or(0);
                values.push((token as i64).rem_euclid(codebook.codebook_size as i64));
            }
            let codes = Tensor::from_vec(values, (1, seq_len), device)?;
            let embed = codebook.lookup(&codes)?;
            rest_embed = rest_embed.broadcast_add(&embed)?;
        }
        let rest_embed = rest_embed.transpose(1, 2)?;
        self.output_proj.forward(&rest_embed).map_err(Error::from)
    }
}

/// Speech tokenizer decoder model.
pub struct SpeechTokenizerDecoder {
    rvq_first: RVQFirstQuantizer,
    rvq_rest: RVQRestQuantizer,
    pre_conv: CausalConv1d,
    pre_transformer_input_proj: Linear,
    pre_transformer_layers: Vec<DecoderLayer>,
    pre_transformer_norm: RmsNorm,
    pre_transformer_output_proj: Linear,
    upsample_stages: Vec<UpsampleStage>,
    decoder_init_conv: CausalConv1d,
    decoder_blocks: Vec<DecoderBlock>,
    final_snake: SnakeBeta,
    final_conv: CausalConv1d,
    device: Device,
    decode_upsample_rate: usize,
    config: DecoderConfig,
}

impl SpeechTokenizerDecoder {
    /// Load decoder from speech_tokenizer directory.
    pub fn load(model_dir: &std::path::Path, device: Device, dtype: DType) -> Result<Self> {
        info!("Loading speech tokenizer decoder from {:?}", model_dir);

        let config_str = std::fs::read_to_string(model_dir.join("config.json"))?;
        let speech_cfg: SpeechTokenizerConfig = serde_json::from_str(&config_str)?;
        let decode_upsample_rate = speech_cfg.decode_upsample_rate;
        let decoder_config = speech_cfg.decoder_config;

        let weights_path = model_dir.join("model.safetensors");
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights_path], dtype, &device)? };
        let vb = vb.pp("decoder");

        // Official checkpoints store RVQ embeddings at 256 dims.
        let rvq_codebook_dim = decoder_config.codebook_dim / 2;

        let rvq_first = RVQFirstQuantizer::load(
            vb.pp("quantizer.rvq_first"),
            decoder_config.codebook_size,
            rvq_codebook_dim,
            decoder_config.hidden_size,
        )?;
        let rvq_rest = RVQRestQuantizer::load(
            vb.pp("quantizer.rvq_rest"),
            decoder_config.num_quantizers.saturating_sub(1),
            decoder_config.codebook_size,
            rvq_codebook_dim,
            decoder_config.hidden_size,
        )?;

        let pre_conv = CausalConv1d::load(
            decoder_config.hidden_size,
            decoder_config.latent_dim,
            3,
            1,
            1,
            true,
            vb.pp("pre_conv.conv"),
        )?;
        let pre_transformer_input_proj = candle_nn::linear(
            decoder_config.latent_dim,
            decoder_config.hidden_size,
            vb.pp("pre_transformer.input_proj"),
        )?;

        let mut pre_transformer_layers = Vec::with_capacity(decoder_config.num_hidden_layers);
        for idx in 0..decoder_config.num_hidden_layers {
            pre_transformer_layers.push(DecoderLayer::load(
                &decoder_config,
                vb.pp(format!("pre_transformer.layers.{idx}")),
            )?);
        }
        let pre_transformer_norm = candle_nn::rms_norm(
            decoder_config.hidden_size,
            decoder_config.rms_norm_eps,
            vb.pp("pre_transformer.norm"),
        )?;
        let pre_transformer_output_proj = candle_nn::linear(
            decoder_config.hidden_size,
            decoder_config.latent_dim,
            vb.pp("pre_transformer.output_proj"),
        )?;

        let mut upsample_stages = Vec::with_capacity(decoder_config.upsampling_ratios.len());
        for (idx, &ratio) in decoder_config.upsampling_ratios.iter().enumerate() {
            upsample_stages.push(UpsampleStage::load(
                decoder_config.latent_dim,
                ratio,
                vb.pp(format!("upsample.{idx}")),
            )?);
        }

        let decoder_init_conv = CausalConv1d::load(
            decoder_config.latent_dim,
            decoder_config.decoder_dim,
            7,
            1,
            1,
            true,
            vb.pp("decoder.0.conv"),
        )?;

        let mut decoder_blocks = Vec::with_capacity(decoder_config.upsample_rates.len());
        let mut in_channels = decoder_config.decoder_dim;
        for (idx, &rate) in decoder_config.upsample_rates.iter().enumerate() {
            let out_channels = in_channels / 2;
            decoder_blocks.push(DecoderBlock::load(
                in_channels,
                out_channels,
                rate,
                vb.pp(format!("decoder.{}", idx + 1)),
            )?);
            in_channels = out_channels;
        }

        let final_snake = SnakeBeta::load(in_channels, vb.pp("decoder.5"))?;
        let final_conv =
            CausalConv1d::load(in_channels, 1, 7, 1, 1, true, vb.pp("decoder.6.conv"))?;

        Ok(Self {
            rvq_first,
            rvq_rest,
            pre_conv,
            pre_transformer_input_proj,
            pre_transformer_layers,
            pre_transformer_norm,
            pre_transformer_output_proj,
            upsample_stages,
            decoder_init_conv,
            decoder_blocks,
            final_snake,
            final_conv,
            device,
            decode_upsample_rate,
            config: decoder_config,
        })
    }

    /// Decode codec tokens to audio waveform samples.
    pub fn decode(&self, codec_tokens: &[Vec<u32>]) -> Result<Vec<f32>> {
        if codec_tokens.is_empty() || codec_tokens[0].is_empty() {
            return Ok(Vec::new());
        }

        let seq_len = codec_tokens[0].len();
        let first_codes: Vec<i64> = codec_tokens[0].iter().map(|&x| x as i64).collect();
        let first_tensor = Tensor::from_vec(first_codes, (1, seq_len), &self.device)?;

        // RVQ decode (first + residual codebooks) and project to hidden features.
        let first_proj = self.rvq_first.decode(&first_tensor)?;
        let rest_proj = self.rvq_rest.decode(codec_tokens, seq_len, &self.device)?;
        let quantized = first_proj.broadcast_add(&rest_proj)?;

        // Pre-conv + pre-transformer.
        let hidden = self.pre_conv.forward(&quantized)?;
        let mut hidden = hidden.transpose(1, 2)?;
        hidden = self.pre_transformer_input_proj.forward(&hidden)?;
        for layer in &self.pre_transformer_layers {
            hidden = layer.forward(&hidden)?;
        }
        hidden = self.pre_transformer_norm.forward(&hidden)?;
        hidden = self.pre_transformer_output_proj.forward(&hidden)?;

        // Upsample stack + decoder blocks.
        let mut hidden = hidden.transpose(1, 2)?;
        for stage in &self.upsample_stages {
            hidden = stage.forward(&hidden)?;
        }
        hidden = self.decoder_init_conv.forward(&hidden)?;
        for block in &self.decoder_blocks {
            hidden = block.forward(&hidden)?;
        }
        hidden = self.final_snake.forward(&hidden)?;
        let audio = self.final_conv.forward(&hidden)?;

        // Clamp to audio range and flatten to mono sample vector.
        let audio = audio.clamp(-1.0f32, 1.0f32)?;
        let mut audio_vec = audio.squeeze(0)?.squeeze(0)?.to_vec1::<f32>()?;

        Ok(audio_vec)
    }
}

/// Repeat KV heads for GQA.
fn repeat_kv(x: &Tensor, num_heads: usize, num_kv_heads: usize) -> Result<Tensor> {
    if num_heads == num_kv_heads {
        return Ok(x.clone());
    }
    let repeats = num_heads / num_kv_heads;
    let mut parts = Vec::with_capacity(num_heads);
    for kv_idx in 0..num_kv_heads {
        let head = x.narrow(2, kv_idx, 1)?;
        for _ in 0..repeats {
            parts.push(head.clone());
        }
    }
    Tensor::cat(&parts, 2).map_err(Error::from)
}

/// Build standard RoPE cos/sin cache.
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
        for &inv in &inv_freq {
            angles.push(pos as f32 * inv);
        }
    }

    let angles = Tensor::from_vec(angles, (seq_len, half_dim), device)?;
    let cos = angles.cos()?.to_dtype(dtype)?;
    let sin = angles.sin()?.to_dtype(dtype)?;
    Ok((cos, sin))
}

/// Causal mask for self-attention.
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
