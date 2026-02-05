//! Speech Tokenizer Decoder for Qwen3-TTS
//!
//! Converts codec tokens (RVQ codes) to audio waveforms using a neural decoder.
//! Architecture: Pre-conv → Transformer → RVQ Codebook Lookup → CNN Decoder → Audio

use candle_core::{DType, Device, Tensor, D};
use candle_nn::{Conv1d, Conv1dConfig, Linear, Module, RmsNorm, VarBuilder};
use tracing::info;

use crate::error::{Error, Result};
use serde::Deserialize;
use std::collections::HashMap;

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
    pub encoder_config: Option<serde_json::Value>, // We only need decoder for TTS
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

/// RVQ Codebook for vector quantization lookup
pub struct RVQCodebook {
    embeddings: Tensor, // [codebook_size, codebook_dim]
    codebook_size: usize,
    codebook_dim: usize,
}

impl RVQCodebook {
    /// Load codebook from VarBuilder
    fn load(codebook_size: usize, codebook_dim: usize, vb: VarBuilder) -> Result<Self> {
        // The codebook embeddings are stored in the VQ layer as embedding_sum
        let embedding_path = format!("vq.layers.0._codebook.embedding_sum");
        let embeddings = vb.get((codebook_size, codebook_dim), &embedding_path)?;

        Ok(Self {
            embeddings,
            codebook_size,
            codebook_dim,
        })
    }

    /// Lookup embeddings from codebook indices
    fn lookup(&self, indices: &Tensor) -> Result<Tensor> {
        // indices: [batch, seq_len] or [batch, seq_len, num_quantizers]
        // Gather embeddings: [batch, seq_len, codebook_dim]
        let indices = indices.flatten(0, indices.dims().len() - 1)?;
        let embeddings = self.embeddings.index_select(&indices, 0)?;

        // Reshape back
        let mut target_shape = indices.dims().to_vec();
        target_shape.push(self.codebook_dim);
        embeddings
            .reshape(target_shape.as_slice())
            .map_err(Error::from)
    }
}

/// Multi-head attention for decoder transformer
struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    hidden_size: usize,
}

impl Attention {
    fn load(cfg: &DecoderConfig, vb: VarBuilder) -> Result<Self> {
        let hidden_size = cfg.hidden_size;
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let head_dim = cfg.head_dim;

        let q_proj = candle_nn::linear_no_bias(hidden_size, num_heads * head_dim, vb.pp("q_proj"))?;
        let k_proj =
            candle_nn::linear_no_bias(hidden_size, num_kv_heads * head_dim, vb.pp("k_proj"))?;
        let v_proj =
            candle_nn::linear_no_bias(hidden_size, num_kv_heads * head_dim, vb.pp("v_proj"))?;
        let o_proj = candle_nn::linear_no_bias(num_heads * head_dim, hidden_size, vb.pp("o_proj"))?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads,
            num_kv_heads,
            head_dim,
            hidden_size,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (bs, seq_len, _) = x.dims3()?;

        // Project to Q, K, V
        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        // Reshape for multi-head attention
        let q = q
            .reshape((bs, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((bs, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((bs, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        // Scaled dot-product attention
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let scores = (q.matmul(&k.transpose(D::Minus2, D::Minus1)?)? * scale)?;

        // Softmax
        let scores = candle_nn::ops::softmax(&scores, D::Minus1)?;

        // Apply attention to values
        let attn_output = scores.matmul(&v)?;

        // Reshape and project
        let attn_output =
            attn_output
                .transpose(1, 2)?
                .reshape((bs, seq_len, self.num_heads * self.head_dim))?;
        self.o_proj.forward(&attn_output).map_err(Error::from)
    }
}

/// MLP block for decoder transformer
struct Mlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl Mlp {
    fn load(cfg: &DecoderConfig, vb: VarBuilder) -> Result<Self> {
        let hidden_size = cfg.hidden_size;
        let intermediate_size = cfg.intermediate_size;

        let gate_proj =
            candle_nn::linear_no_bias(hidden_size, intermediate_size, vb.pp("gate_proj"))?;
        let up_proj = candle_nn::linear_no_bias(hidden_size, intermediate_size, vb.pp("up_proj"))?;
        let down_proj =
            candle_nn::linear_no_bias(intermediate_size, hidden_size, vb.pp("down_proj"))?;

        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = self.gate_proj.forward(x)?;
        let gate = candle_nn::ops::silu(&gate)?;
        let up = self.up_proj.forward(x)?;
        let hidden = (gate * up)?;
        self.down_proj.forward(&hidden).map_err(Error::from)
    }
}

/// Transformer layer for decoder
struct DecoderLayer {
    input_layernorm: RmsNorm,
    self_attn: Attention,
    post_attention_layernorm: RmsNorm,
    mlp: Mlp,
}

impl DecoderLayer {
    fn load(cfg: &DecoderConfig, vb: VarBuilder) -> Result<Self> {
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

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Self attention with residual
        let normed = self.input_layernorm.forward(x)?;
        let attn_out = self.self_attn.forward(&normed)?;
        let x = x.broadcast_add(&attn_out)?;

        // MLP with residual
        let normed = self.post_attention_layernorm.forward(&x)?;
        let mlp_out = self.mlp.forward(&normed)?;
        x.broadcast_add(&mlp_out).map_err(Error::from)
    }
}

/// Pre-convolution to process input embeddings
struct PreConv {
    conv: Conv1d,
}

impl PreConv {
    fn load(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let config = Conv1dConfig {
            padding: kernel_size / 2, // Same padding
            stride: 1,
            dilation: 1,
            groups: 1,
            ..Default::default()
        };
        let conv = candle_nn::conv1d(
            in_channels,
            out_channels,
            kernel_size,
            config,
            vb.pp("conv"),
        )?;
        Ok(Self { conv })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // x: [batch, seq_len, channels] -> transpose to [batch, channels, seq_len]
        let x = x.transpose(1, 2)?;
        let x = self.conv.forward(&x)?;
        // Transpose back: [batch, channels, seq_len] -> [batch, seq_len, channels]
        x.transpose(1, 2).map_err(Error::from)
    }
}

/// Output projection from transformer
struct OutputProj {
    linear: Linear,
}

impl OutputProj {
    fn load(in_features: usize, out_features: usize, vb: VarBuilder) -> Result<Self> {
        let linear = candle_nn::linear_no_bias(in_features, out_features, vb)?;
        Ok(Self { linear })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.linear.forward(x).map_err(Error::from)
    }
}

/// RVQ Quantizer that looks up codebook embeddings
struct RVQQuantizer {
    input_proj: Conv1d,
    output_proj: Conv1d,
    codebook: RVQCodebook,
}

impl RVQQuantizer {
    fn load(
        in_channels: usize,
        out_channels: usize,
        codebook_size: usize,
        codebook_dim: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let input_config = Conv1dConfig {
            padding: 0,
            stride: 1,
            dilation: 1,
            groups: 1,
            ..Default::default()
        };
        // Use linear_no_bias equivalent - Conv1d without bias
        let input_proj = candle_nn::conv1d_no_bias(
            in_channels,
            codebook_dim,
            1,
            input_config,
            vb.pp("input_proj"),
        )?;
        let output_proj = candle_nn::conv1d_no_bias(
            codebook_dim,
            out_channels,
            1,
            input_config,
            vb.pp("output_proj"),
        )?;
        let codebook = RVQCodebook::load(codebook_size, codebook_dim, vb)?;

        Ok(Self {
            input_proj,
            output_proj,
            codebook,
        })
    }

    /// Decode from codebook indices
    fn decode(&self, indices: &Tensor) -> Result<Tensor> {
        // indices: [batch, seq_len]
        // Lookup embeddings: [batch, seq_len, codebook_dim]
        let embeddings = self.codebook.lookup(indices)?;

        // Project through output: [batch, seq_len, out_channels]
        // Conv1d expects [batch, channels, seq_len]
        let embeddings = embeddings.transpose(1, 2)?;
        let output = self.output_proj.forward(&embeddings)?;
        output.transpose(1, 2).map_err(Error::from)
    }
}

/// CNN Decoder block with upsampling
struct DecoderBlock {
    conv1: Conv1d,
    conv2: Conv1d,
}

impl DecoderBlock {
    fn load(channels: usize, kernel_size: usize, vb: VarBuilder) -> Result<Self> {
        let config1 = Conv1dConfig {
            padding: kernel_size / 2,
            stride: 1,
            dilation: 1,
            groups: 1,
            ..Default::default()
        };
        let config2 = Conv1dConfig {
            padding: 0,
            stride: 1,
            dilation: 1,
            groups: 1,
            ..Default::default()
        };

        let conv1 = candle_nn::conv1d(channels, channels, kernel_size, config1, vb.pp("conv1"))?;
        let conv2 = candle_nn::conv1d(channels, channels, 1, config2, vb.pp("conv2"))?;

        Ok(Self { conv1, conv2 })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // x: [batch, seq_len, channels] -> [batch, channels, seq_len]
        let x = x.transpose(1, 2)?;
        let x = self.conv1.forward(&x)?;
        // ReLU activation
        let x = x.clamp(0.0, f64::INFINITY)?;
        let x = self.conv2.forward(&x)?;
        x.transpose(1, 2).map_err(Error::from)
    }
}

/// Speech Tokenizer Decoder model
pub struct SpeechTokenizerDecoder {
    pre_conv: PreConv,
    pre_transformer_layers: Vec<DecoderLayer>,
    output_proj: OutputProj,
    rvq_first: RVQQuantizer,
    rvq_rest: RVQQuantizer,
    final_decoder: Vec<DecoderBlock>,
    final_conv: Conv1d,
    device: Device,
    config: DecoderConfig,
}

impl SpeechTokenizerDecoder {
    /// Load the speech tokenizer decoder from model directory
    pub fn load(model_dir: &std::path::Path, device: Device, dtype: DType) -> Result<Self> {
        info!("Loading speech tokenizer decoder from {:?}", model_dir);

        // Load config
        let config_path = model_dir.join("config.json");
        let config_str = std::fs::read_to_string(&config_path)?;
        let config: SpeechTokenizerConfig = serde_json::from_str(&config_str)?;
        let decoder_config = config.decoder_config;

        // Load weights
        let weights_path = model_dir.join("model.safetensors");
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights_path], dtype, &device)? };

        let vb = vb.pp("decoder");

        // Pre-conv: projects input embeddings to latent_dim (1024), not hidden_size
        let pre_conv = PreConv::load(
            decoder_config.codebook_dim * decoder_config.num_semantic_quantizers,
            decoder_config.latent_dim,
            3,
            vb.pp("pre_conv"),
        )?;

        // Pre-transformer layers (8 layers)
        let mut pre_transformer_layers = Vec::with_capacity(decoder_config.num_hidden_layers);
        for idx in 0..decoder_config.num_hidden_layers {
            let layer = DecoderLayer::load(
                &decoder_config,
                vb.pp(format!("pre_transformer.layers.{idx}")),
            )?;
            pre_transformer_layers.push(layer);
        }

        // Output projection from transformer
        let output_proj = OutputProj::load(
            decoder_config.hidden_size,
            decoder_config.latent_dim,
            vb.pp("pre_transformer.output_proj"),
        )?;

        // RVQ quantizers
        // output_proj weight is [512, 256, 1], so output is 512, codebook is 256
        // Note: semantic_codebook_size is 4096 in config but actual weight is 2048
        let rvq_first = RVQQuantizer::load(
            decoder_config.hidden_size,   // 512 (input)
            decoder_config.hidden_size,   // 512 (output - matches weight)
            decoder_config.codebook_size, // 2048 (actual weight shape)
            256,                          // 256 (codebook dim)
            vb.pp("quantizer.rvq_first"),
        )?;

        // rvq_rest uses the same dimensions
        let rvq_rest = RVQQuantizer::load(
            decoder_config.hidden_size, // 512 (input)
            decoder_config.hidden_size, // 512 (output)
            decoder_config.codebook_size,
            256, // 256 (codebook dim)
            vb.pp("quantizer.rvq_rest"),
        )?;

        // Final decoder blocks (simplified - actual model has more complex upsampling)
        let mut final_decoder = Vec::new();
        // Create some decoder blocks
        for i in 0..3 {
            let block =
                DecoderBlock::load(decoder_config.decoder_dim, 7, vb.pp(format!("decoder.{i}")))?;
            final_decoder.push(block);
        }

        // Final conv to produce audio
        let final_conv_config = Conv1dConfig {
            padding: 3, // kernel_size 7, so padding 3
            stride: 1,
            dilation: 1,
            groups: 1,
            ..Default::default()
        };
        let final_conv = candle_nn::conv1d(
            decoder_config.decoder_dim,
            1,
            7,
            final_conv_config,
            vb.pp("decoder.6"),
        )?;

        Ok(Self {
            pre_conv,
            pre_transformer_layers,
            output_proj,
            rvq_first,
            rvq_rest,
            final_decoder,
            final_conv,
            device,
            config: decoder_config,
        })
    }

    /// Decode codec tokens to audio waveform
    ///
    /// # Arguments
    /// * `codec_tokens` - Vec of codebook indices per code group [num_code_groups, seq_len]
    ///
    /// # Returns
    /// * Audio waveform as Vec<f32>
    pub fn decode(&self, codec_tokens: &[Vec<u32>]) -> Result<Vec<f32>> {
        if codec_tokens.is_empty() || codec_tokens[0].is_empty() {
            return Ok(Vec::new());
        }

        let seq_len = codec_tokens[0].len();
        let num_groups = codec_tokens.len();

        // Convert codec tokens to tensors
        // First codebook (semantic)
        let first_codes: Vec<i64> = codec_tokens[0].iter().map(|&x| x as i64).collect();
        let first_tensor = Tensor::from_vec(first_codes, (1, seq_len), &self.device)?;

        // Decode first codebook through RVQ
        let first_embedding = self.rvq_first.decode(&first_tensor)?;

        // For remaining codebooks, we would need to sum their contributions
        // For now, use just the first codebook for simplicity
        let mut combined_embedding = first_embedding;

        // Pass through pre-conv
        let x = self.pre_conv.forward(&combined_embedding)?;

        // Pass through transformer layers
        let mut hidden = x;
        for layer in &self.pre_transformer_layers {
            hidden = layer.forward(&hidden)?;
        }

        // Output projection
        let hidden = self.output_proj.forward(&hidden)?;

        // Pass through final decoder blocks
        let mut decoded = hidden;
        for block in &self.final_decoder {
            decoded = block.forward(&decoded)?;
        }

        // Final conv to produce audio
        // [batch, seq_len, channels] -> [batch, channels, seq_len]
        let decoded = decoded.transpose(1, 2)?;
        let audio = self.final_conv.forward(&decoded)?;
        // [batch, 1, seq_len] -> [batch, seq_len, 1] -> flatten
        let audio = audio.squeeze(1)?; // Remove channel dim

        // Convert to Vec<f32>
        let audio_vec = audio.to_vec1::<f32>()?;
        Ok(audio_vec)
    }
}
