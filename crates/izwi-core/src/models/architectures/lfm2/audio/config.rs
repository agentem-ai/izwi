use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct Lfm2AudioConfig {
    pub architectures: Vec<String>,
    pub codebooks: usize,
    pub tie_audio_embeddings: bool,
    pub semantic_codebook_factor: f32,
    pub codebook_weight: String,
    pub interleaved_n_text: usize,
    pub interleaved_n_audio: usize,
    pub preprocessor: PreprocessorConfig,
    pub encoder: ConformerConfig,
    pub lfm: LfmConfig,
    pub depthformer: DepthformerConfig,
}

#[derive(Debug, Clone, Deserialize)]
pub struct PreprocessorConfig {
    pub sample_rate: usize,
    pub normalize: String,
    pub window_size: f32,
    pub window_stride: f32,
    pub window: String,
    pub features: usize,
    pub n_fft: usize,
    pub log: bool,
    pub frame_splicing: usize,
    pub dither: f32,
    pub pad_to: usize,
    pub pad_value: f32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ConformerConfig {
    pub feat_in: usize,
    pub feat_out: isize,
    pub n_layers: usize,
    pub d_model: usize,
    pub subsampling: String,
    pub subsampling_factor: usize,
    pub subsampling_conv_channels: usize,
    pub causal_downsampling: bool,
    pub reduction: Option<String>,
    pub reduction_position: Option<usize>,
    pub reduction_factor: usize,
    pub ff_expansion_factor: usize,
    pub self_attention_model: String,
    pub n_heads: usize,
    pub att_context_size: Vec<isize>,
    pub xscaling: bool,
    pub untie_biases: bool,
    pub pos_emb_max_len: usize,
    pub conv_kernel_size: usize,
    pub conv_norm_type: String,
    pub conv_context_size: Option<Vec<isize>>,
    pub dropout: f32,
    pub dropout_pre_encoder: f32,
    pub dropout_emb: f32,
    pub dropout_att: f32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct LfmConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub max_position_embeddings: usize,
    pub layer_types: Vec<String>,
    pub rope_theta: f64,
    #[serde(alias = "conv_L_cache")]
    pub conv_l_cache: usize,
    pub norm_eps: f64,
    pub vocab_size: usize,
    pub eos_token_id: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct DepthformerConfig {
    pub layers: usize,
    pub dim: usize,
    pub tie: bool,
}
