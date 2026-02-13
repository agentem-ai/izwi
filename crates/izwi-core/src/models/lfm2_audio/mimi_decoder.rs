use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use candle_core::{DType, Device, Tensor};
use candle_transformers::models::mimi::encodec::Encodec;
use safetensors::tensor::{serialize_to_file, Dtype, TensorView};
use safetensors::SafeTensors;

use crate::error::{Error, Result};

#[derive(Debug)]
struct OwnedTensor {
    name: String,
    dtype: Dtype,
    shape: Vec<usize>,
    data: Vec<u8>,
}

pub struct MimiDecoder {
    device: Device,
    model: Mutex<Encodec>,
}

impl MimiDecoder {
    pub fn load(model_dir: &Path, device: &Device) -> Result<Self> {
        let source = model_dir.join("tokenizer-e351c8d8-checkpoint125.safetensors");
        if !source.exists() {
            return Err(Error::ModelLoadError(format!(
                "Missing LFM2 Mimi tokenizer checkpoint: {}",
                source.display()
            )));
        }

        let converted = ensure_candle_checkpoint(&source)?;
        let model = candle_transformers::models::mimi::encodec::load(
            converted
                .to_str()
                .ok_or_else(|| Error::ModelLoadError("Invalid tokenizer path".to_string()))?,
            Some(8),
            device,
        )
        .map_err(|e| Error::ModelLoadError(format!("Failed to load Mimi decoder: {e}")))?;

        Ok(Self {
            device: device.clone(),
            model: Mutex::new(model),
        })
    }

    pub fn decode_tokens(&self, codebooks: &[Vec<u32>]) -> Result<Vec<f32>> {
        if codebooks.is_empty() || codebooks[0].is_empty() {
            return Ok(Vec::new());
        }

        let n_codebooks = codebooks.len();
        let frames = codebooks[0].len();
        if codebooks.iter().any(|c| c.len() != frames) {
            return Err(Error::InferenceError(
                "Inconsistent LFM2 audio token frame lengths".to_string(),
            ));
        }

        let mut flat = Vec::with_capacity(n_codebooks * frames);
        for row in codebooks {
            flat.extend_from_slice(row);
        }

        let codes =
            Tensor::from_vec(flat, (1, n_codebooks, frames), &self.device)?.to_dtype(DType::U32)?;

        let mut model = self
            .model
            .lock()
            .map_err(|_| Error::InferenceError("Mimi decoder mutex poisoned".to_string()))?;

        let wav = model
            .decode(&codes)
            .map_err(|e| Error::InferenceError(format!("Mimi decode failed: {e}")))?;
        let wav = wav.squeeze(0)?.squeeze(0)?;
        wav.to_vec1::<f32>().map_err(Error::from)
    }
}

fn ensure_candle_checkpoint(source: &Path) -> Result<PathBuf> {
    let preconverted = [
        "tokenizer-e351c8d8-checkpoint125.candle.v4.safetensors",
        "tokenizer-e351c8d8-checkpoint125.candle.v3.safetensors",
        "tokenizer-e351c8d8-checkpoint125.candle.v2.safetensors",
        "tokenizer-e351c8d8-checkpoint125.candle.safetensors",
    ];
    for name in preconverted {
        let candidate = source.with_file_name(name);
        if candidate.exists() {
            return Ok(candidate);
        }
    }

    let target = source.with_file_name("tokenizer-e351c8d8-checkpoint125.candle.v4.safetensors");
    if target.exists() {
        return Ok(target);
    }

    let bytes = std::fs::read(source).map_err(|e| {
        Error::ModelLoadError(format!(
            "Failed to read Mimi tokenizer checkpoint {}: {}",
            source.display(),
            e
        ))
    })?;

    let safetensors = SafeTensors::deserialize(&bytes)
        .map_err(|e| Error::ModelLoadError(format!("Invalid safetensors file: {e}")))?;

    let mut owned: Vec<OwnedTensor> = Vec::new();

    for name in safetensors.names() {
        let tensor = safetensors
            .tensor(name)
            .map_err(|e| Error::ModelLoadError(format!("Failed reading tensor {name}: {e}")))?;
        let dtype = tensor.dtype();
        let shape = tensor.shape().to_vec();
        let data = tensor.data();

        if name.ends_with(".self_attn.in_proj_weight") {
            if shape.len() != 2 || shape[0] % 3 != 0 {
                return Err(Error::ModelLoadError(format!(
                    "Unexpected in_proj_weight shape for {name}: {shape:?}"
                )));
            }
            let rows = shape[0] / 3;
            let cols = shape[1];
            let elem = dtype_size(dtype)?;
            let row_bytes = cols * elem;
            let part_bytes = rows * row_bytes;

            let base = remap_tensor_name(name);
            let q_name = base.replace("self_attn.in_proj_weight", "self_attn.q_proj.weight");
            let k_name = base.replace("self_attn.in_proj_weight", "self_attn.k_proj.weight");
            let v_name = base.replace("self_attn.in_proj_weight", "self_attn.v_proj.weight");

            owned.push(OwnedTensor {
                name: q_name,
                dtype,
                shape: vec![rows, cols],
                data: data[0..part_bytes].to_vec(),
            });
            owned.push(OwnedTensor {
                name: k_name,
                dtype,
                shape: vec![rows, cols],
                data: data[part_bytes..part_bytes * 2].to_vec(),
            });
            owned.push(OwnedTensor {
                name: v_name,
                dtype,
                shape: vec![rows, cols],
                data: data[part_bytes * 2..part_bytes * 3].to_vec(),
            });
            continue;
        }

        owned.push(OwnedTensor {
            name: remap_tensor_name(name),
            dtype,
            shape,
            data: data.to_vec(),
        });
    }

    let mut views = BTreeMap::new();
    for tensor in &owned {
        let view = TensorView::new(tensor.dtype, tensor.shape.clone(), tensor.data.as_slice())
            .map_err(|e| {
                Error::ModelLoadError(format!("Failed to build tensor view {}: {e}", tensor.name))
            })?;
        views.insert(tensor.name.clone(), view);
    }

    serialize_to_file(views, &None, &target).map_err(|e| {
        Error::ModelLoadError(format!("Failed writing converted Mimi checkpoint: {e}"))
    })?;

    Ok(target)
}

fn remap_tensor_name(name: &str) -> String {
    let mut mapped = name
        .replace(".model.", ".layers.")
        .replace(".transformer.layers.", ".layers.")
        .replace(".norm1.", ".input_layernorm.")
        .replace(".norm2.", ".post_attention_layernorm.")
        .replace(".linear1.", ".mlp.fc1.")
        .replace(".linear2.", ".mlp.fc2.")
        .replace(".layer_scale_1.", ".self_attn_layer_scale.")
        .replace(".layer_scale_2.", ".mlp_layer_scale.")
        .replace(".self_attn.out_proj.", ".self_attn.o_proj.")
        .replace(
            "quantizer.rvq_first.",
            "quantizer.semantic_residual_vector_quantizer.",
        )
        .replace(
            "quantizer.rvq_rest.",
            "quantizer.acoustic_residual_vector_quantizer.",
        )
        .replace(".vq.layers.", ".layers.")
        .replace("._codebook.embedding_sum", ".codebook.embed_sum")
        .replace("._codebook.cluster_usage", ".codebook.cluster_usage")
        .replace("._codebook._initialized", ".codebook.initialized");

    while mapped.contains(".convtr.") {
        mapped = mapped.replace(".convtr.", ".conv.");
    }
    while mapped.contains(".conv.conv.") {
        mapped = mapped.replace(".conv.conv.", ".conv.");
    }

    mapped
}

fn dtype_size(dtype: Dtype) -> Result<usize> {
    Ok(match dtype {
        Dtype::BOOL | Dtype::U8 | Dtype::I8 => 1,
        Dtype::U16 | Dtype::I16 | Dtype::BF16 | Dtype::F16 => 2,
        Dtype::U32 | Dtype::I32 | Dtype::F32 => 4,
        Dtype::U64 | Dtype::I64 | Dtype::F64 => 8,
        _ => {
            return Err(Error::ModelLoadError(format!(
                "Unsupported safetensors dtype in Mimi checkpoint: {dtype:?}"
            )))
        }
    })
}
