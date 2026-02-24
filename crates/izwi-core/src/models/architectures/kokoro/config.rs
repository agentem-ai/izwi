use std::collections::HashMap;

use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct KokoroIstftNetConfig {
    pub upsample_kernel_sizes: Vec<usize>,
    pub upsample_rates: Vec<usize>,
    pub gen_istft_hop_size: usize,
    pub gen_istft_n_fft: usize,
    pub resblock_dilation_sizes: Vec<Vec<usize>>,
    pub resblock_kernel_sizes: Vec<usize>,
    pub upsample_initial_channel: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct KokoroPlbertConfig {
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    pub num_hidden_layers: usize,
    #[serde(default)]
    pub dropout: f32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct KokoroConfig {
    pub istftnet: KokoroIstftNetConfig,
    pub dim_in: usize,
    pub dropout: f32,
    pub hidden_dim: usize,
    pub max_conv_dim: usize,
    pub max_dur: usize,
    pub multispeaker: bool,
    pub n_layer: usize,
    pub n_mels: usize,
    pub n_token: usize,
    pub style_dim: usize,
    pub text_encoder_kernel_size: usize,
    pub plbert: KokoroPlbertConfig,
    pub vocab: HashMap<String, u32>,
}

impl KokoroConfig {
    pub const TARGET_SAMPLE_RATE: u32 = 24_000;

    pub fn context_length(&self) -> usize {
        self.plbert.max_position_embeddings
    }
}
