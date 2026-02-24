use candle_core::{D, DType, Tensor};
use candle_nn::{embedding, ops, Embedding, LayerNorm, Linear, Module, VarBuilder};

use crate::error::{Error, Result};

use super::config::KokoroConfig;

#[derive(Debug, Clone)]
pub struct AlbertModelConfig {
    pub vocab_size: usize,
    pub embedding_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    pub layer_norm_eps: f64,
}

impl AlbertModelConfig {
    pub fn from_kokoro(cfg: &KokoroConfig) -> Self {
        Self {
            vocab_size: cfg.n_token,
            embedding_size: 128,
            hidden_size: cfg.plbert.hidden_size,
            num_hidden_layers: cfg.plbert.num_hidden_layers,
            num_attention_heads: cfg.plbert.num_attention_heads,
            intermediate_size: cfg.plbert.intermediate_size,
            max_position_embeddings: cfg.plbert.max_position_embeddings,
            layer_norm_eps: 1e-12,
        }
    }
}

#[derive(Debug)]
pub struct CustomAlbert {
    embeddings: AlbertEmbeddings,
    encoder: AlbertEncoder,
}

impl CustomAlbert {
    pub fn load(cfg: &AlbertModelConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            embeddings: AlbertEmbeddings::load(cfg, vb.pp("embeddings"))?,
            encoder: AlbertEncoder::load(cfg, vb.pp("encoder"))?,
        })
    }

    pub fn forward(&self, input_ids: &Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor> {
        let (b, t) = input_ids.dims2()?;
        let token_type_ids = Tensor::zeros((b, t), DType::U32, input_ids.device())?;
        let x = self.embeddings.forward(input_ids, &token_type_ids)?;
        self.encoder.forward(&x, attention_mask)
    }
}

#[derive(Debug)]
struct AlbertEmbeddings {
    word_embeddings: Embedding,
    position_embeddings: Embedding,
    token_type_embeddings: Embedding,
    layer_norm: LayerNorm,
}

impl AlbertEmbeddings {
    fn load(cfg: &AlbertModelConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            word_embeddings: embedding(
                cfg.vocab_size,
                cfg.embedding_size,
                vb.pp("word_embeddings"),
            )
            .map_err(Error::from)?,
            position_embeddings: embedding(
                cfg.max_position_embeddings,
                cfg.embedding_size,
                vb.pp("position_embeddings"),
            )
            .map_err(Error::from)?,
            token_type_embeddings: embedding(2, cfg.embedding_size, vb.pp("token_type_embeddings"))
                .map_err(Error::from)?,
            layer_norm: candle_nn::layer_norm(
                cfg.embedding_size,
                cfg.layer_norm_eps,
                vb.pp("LayerNorm"),
            )
            .map_err(Error::from)?,
        })
    }

    fn forward(&self, input_ids: &Tensor, token_type_ids: &Tensor) -> Result<Tensor> {
        let (_b, t) = input_ids.dims2()?;
        let w = self.word_embeddings.forward(input_ids).map_err(Error::from)?;
        let tt = self
            .token_type_embeddings
            .forward(token_type_ids)
            .map_err(Error::from)?;
        let pos_ids = Tensor::arange(0u32, t as u32, input_ids.device())
            .map_err(Error::from)?
            .reshape((1, t))
            .map_err(Error::from)?;
        let pos = self
            .position_embeddings
            .forward(&pos_ids)
            .map_err(Error::from)?;
        let x = (&w + &tt).map_err(Error::from)?;
        let x = x.broadcast_add(&pos).map_err(Error::from)?;
        self.layer_norm.forward(&x).map_err(Error::from)
    }
}

#[derive(Debug)]
struct AlbertEncoder {
    embedding_hidden_mapping_in: Linear,
    shared_layer: AlbertLayer,
    num_hidden_layers: usize,
}

impl AlbertEncoder {
    fn load(cfg: &AlbertModelConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            embedding_hidden_mapping_in: candle_nn::linear(
                cfg.embedding_size,
                cfg.hidden_size,
                vb.pp("embedding_hidden_mapping_in"),
            )
            .map_err(Error::from)?,
            shared_layer: AlbertLayer::load(cfg, vb.pp("albert_layer_groups.0.albert_layers.0"))?,
            num_hidden_layers: cfg.num_hidden_layers,
        })
    }

    fn forward(&self, x: &Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor> {
        let mut h = self
            .embedding_hidden_mapping_in
            .forward(x)
            .map_err(Error::from)?;
        for _ in 0..self.num_hidden_layers {
            h = self.shared_layer.forward(&h, attention_mask)?;
        }
        Ok(h)
    }
}

#[derive(Debug)]
struct AlbertLayer {
    attention: AlbertAttention,
    ffn: Linear,
    ffn_output: Linear,
    full_layer_norm: LayerNorm,
}

impl AlbertLayer {
    fn load(cfg: &AlbertModelConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            attention: AlbertAttention::load(cfg, vb.pp("attention"))?,
            ffn: candle_nn::linear(cfg.hidden_size, cfg.intermediate_size, vb.pp("ffn"))
                .map_err(Error::from)?,
            ffn_output: candle_nn::linear(cfg.intermediate_size, cfg.hidden_size, vb.pp("ffn_output"))
                .map_err(Error::from)?,
            full_layer_norm: candle_nn::layer_norm(
                cfg.hidden_size,
                cfg.layer_norm_eps,
                vb.pp("full_layer_layer_norm"),
            )
            .map_err(Error::from)?,
        })
    }

    fn forward(&self, x: &Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor> {
        let h = self.attention.forward(x, attention_mask)?;
        let ff = self.ffn.forward(&h).map_err(Error::from)?;
        let ff = ff.gelu().map_err(Error::from)?;
        let ff = self.ffn_output.forward(&ff).map_err(Error::from)?;
        let y = (&h + ff).map_err(Error::from)?;
        self.full_layer_norm.forward(&y).map_err(Error::from)
    }
}

#[derive(Debug)]
struct AlbertAttention {
    query: Linear,
    key: Linear,
    value: Linear,
    dense: Linear,
    layer_norm: LayerNorm,
    num_heads: usize,
    head_dim: usize,
}

impl AlbertAttention {
    fn load(cfg: &AlbertModelConfig, vb: VarBuilder) -> Result<Self> {
        let num_heads = cfg.num_attention_heads;
        let head_dim = cfg.hidden_size / num_heads;
        Ok(Self {
            query: candle_nn::linear(cfg.hidden_size, cfg.hidden_size, vb.pp("query"))
                .map_err(Error::from)?,
            key: candle_nn::linear(cfg.hidden_size, cfg.hidden_size, vb.pp("key"))
                .map_err(Error::from)?,
            value: candle_nn::linear(cfg.hidden_size, cfg.hidden_size, vb.pp("value"))
                .map_err(Error::from)?,
            dense: candle_nn::linear(cfg.hidden_size, cfg.hidden_size, vb.pp("dense"))
                .map_err(Error::from)?,
            layer_norm: candle_nn::layer_norm(cfg.hidden_size, cfg.layer_norm_eps, vb.pp("LayerNorm"))
                .map_err(Error::from)?,
            num_heads,
            head_dim,
        })
    }

    fn forward(&self, x: &Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor> {
        let (b, t, h) = x.dims3().map_err(Error::from)?;
        let q = self
            .query
            .forward(x)
            .map_err(Error::from)?
            .reshape((b, t, self.num_heads, self.head_dim))
            .map_err(Error::from)?
            .transpose(1, 2)
            .map_err(Error::from)?
            .contiguous()
            .map_err(Error::from)?
            .reshape((b * self.num_heads, t, self.head_dim))
            .map_err(Error::from)?;
        let k = self
            .key
            .forward(x)
            .map_err(Error::from)?
            .reshape((b, t, self.num_heads, self.head_dim))
            .map_err(Error::from)?
            .transpose(1, 2)
            .map_err(Error::from)?
            .contiguous()
            .map_err(Error::from)?
            .reshape((b * self.num_heads, t, self.head_dim))
            .map_err(Error::from)?;
        let v = self
            .value
            .forward(x)
            .map_err(Error::from)?
            .reshape((b, t, self.num_heads, self.head_dim))
            .map_err(Error::from)?
            .transpose(1, 2)
            .map_err(Error::from)?
            .contiguous()
            .map_err(Error::from)?
            .reshape((b * self.num_heads, t, self.head_dim))
            .map_err(Error::from)?;

        let kt = k
            .transpose(1, 2)
            .map_err(Error::from)?
            .contiguous()
            .map_err(Error::from)?;
        let mut scores = q.matmul(&kt).map_err(Error::from)?;
        scores = (scores * (1.0f64 / (self.head_dim as f64).sqrt())).map_err(Error::from)?;

        if let Some(mask) = attention_mask {
            // mask expected [B, T] with 1 for valid, 0 for padded.
            let mask = mask.to_dtype(DType::F32).map_err(Error::from)?;
            let inv = (Tensor::ones(mask.shape(), DType::F32, mask.device()).map_err(Error::from)?
                - &mask)
                .map_err(Error::from)?;
            let inv = inv
                .reshape((b, 1, 1, t))
                .map_err(Error::from)?
                .broadcast_as((b, self.num_heads, t, t))
                .map_err(Error::from)?
                .contiguous()
                .map_err(Error::from)?
                .reshape(scores.shape())
                .map_err(Error::from)?;
            scores = (scores + (inv * -1e4f64).map_err(Error::from)?).map_err(Error::from)?;
        }

        let attn = ops::softmax(&scores, D::Minus1).map_err(Error::from)?;
        let ctx = attn.matmul(&v.contiguous().map_err(Error::from)?).map_err(Error::from)?;
        let ctx = ctx
            .reshape((b, self.num_heads, t, self.head_dim))
            .map_err(Error::from)?
            .transpose(1, 2)
            .map_err(Error::from)?
            .reshape((b, t, h))
            .map_err(Error::from)?;
        let out = self.dense.forward(&ctx).map_err(Error::from)?;
        let y = (out + x).map_err(Error::from)?;
        self.layer_norm.forward(&y).map_err(Error::from)
    }
}
