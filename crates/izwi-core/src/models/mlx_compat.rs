//! Compatibility helpers for MLX-exported safetensors checkpoints.
//!
//! MLX quantized checkpoints store affine-packed weights as:
//! - `weight`  (U8/U32 packed values)
//! - `scales`  (per-group scale)
//! - `biases`  (per-group bias)
//!
//! This module dequantizes those tensors into dense weights so existing Candle
//! layers can be reused without model-specific branching.

use candle_core::{DType, Tensor};
use candle_nn::{Conv2d, Conv2dConfig, Embedding, Linear, VarBuilder};

use crate::error::{Error, Result};

fn tensor2_f32(t: Tensor) -> Result<Vec<Vec<f32>>> {
    let t = t.to_dtype(DType::F32)?;
    t.to_vec2::<f32>().map_err(Error::from)
}

fn quant_bits_from_packing(
    expected_in_dim: usize,
    packed_in_dim: usize,
    word_bits: usize,
) -> Result<usize> {
    if packed_in_dim == 0 || expected_in_dim == 0 {
        return Err(Error::ModelLoadError(
            "Invalid packed shape for quantized weight".to_string(),
        ));
    }
    if expected_in_dim % packed_in_dim != 0 {
        return Err(Error::ModelLoadError(format!(
            "Quantized weight has incompatible input dim: expected_in={expected_in_dim}, packed_in={packed_in_dim}"
        )));
    }
    let pack_factor = expected_in_dim / packed_in_dim;
    if word_bits % pack_factor != 0 {
        return Err(Error::ModelLoadError(format!(
            "Invalid quant packing factor: word_bits={word_bits}, pack_factor={pack_factor}"
        )));
    }
    let bits = word_bits / pack_factor;
    if bits == 0 || bits > 8 {
        return Err(Error::ModelLoadError(format!(
            "Unsupported quantization bit-width inferred from packing: {bits}"
        )));
    }
    Ok(bits)
}

fn dequantize_affine_u32(
    packed: Vec<Vec<u32>>,
    scales: &[Vec<f32>],
    biases: &[Vec<f32>],
    out_dim: usize,
    in_dim: usize,
) -> Result<Vec<f32>> {
    let packed_in = packed
        .first()
        .map(|r| r.len())
        .ok_or_else(|| Error::ModelLoadError("Empty packed quantized weight".to_string()))?;
    let bits = quant_bits_from_packing(in_dim, packed_in, 32)?;
    let pack_factor = in_dim / packed_in;
    let mask = if bits == 32 {
        u32::MAX
    } else {
        (1u32 << bits) - 1
    };

    let groups = scales
        .first()
        .map(|r| r.len())
        .ok_or_else(|| Error::ModelLoadError("Empty quantization scales".to_string()))?;
    if groups == 0 || in_dim % groups != 0 {
        return Err(Error::ModelLoadError(format!(
            "Invalid quantization groups: in_dim={in_dim}, groups={groups}"
        )));
    }
    let group_size = in_dim / groups;

    let mut out = vec![0f32; out_dim * in_dim];
    for row in 0..out_dim {
        for pcol in 0..packed_in {
            let word = packed[row][pcol];
            let base_col = pcol * pack_factor;
            for pi in 0..pack_factor {
                let col = base_col + pi;
                if col >= in_dim {
                    break;
                }
                let q = ((word >> (pi * bits)) & mask) as f32;
                let g = col / group_size;
                out[row * in_dim + col] = q * scales[row][g] + biases[row][g];
            }
        }
    }
    Ok(out)
}

fn dequantize_affine_u8(
    packed: Vec<Vec<u8>>,
    scales: &[Vec<f32>],
    biases: &[Vec<f32>],
    out_dim: usize,
    in_dim: usize,
) -> Result<Vec<f32>> {
    let packed_in = packed
        .first()
        .map(|r| r.len())
        .ok_or_else(|| Error::ModelLoadError("Empty packed quantized weight".to_string()))?;
    let bits = quant_bits_from_packing(in_dim, packed_in, 8)?;
    let pack_factor = in_dim / packed_in;
    let mask = if bits == 8 { 0xFF } else { (1u32 << bits) - 1 };

    let groups = scales
        .first()
        .map(|r| r.len())
        .ok_or_else(|| Error::ModelLoadError("Empty quantization scales".to_string()))?;
    if groups == 0 || in_dim % groups != 0 {
        return Err(Error::ModelLoadError(format!(
            "Invalid quantization groups: in_dim={in_dim}, groups={groups}"
        )));
    }
    let group_size = in_dim / groups;

    let mut out = vec![0f32; out_dim * in_dim];
    for row in 0..out_dim {
        for pcol in 0..packed_in {
            let word = packed[row][pcol] as u32;
            let base_col = pcol * pack_factor;
            for pi in 0..pack_factor {
                let col = base_col + pi;
                if col >= in_dim {
                    break;
                }
                let q = ((word >> (pi * bits)) & mask) as f32;
                let g = col / group_size;
                out[row * in_dim + col] = q * scales[row][g] + biases[row][g];
            }
        }
    }
    Ok(out)
}

pub fn has_mlx_affine_quantized_weight(vb: &VarBuilder) -> bool {
    vb.contains_tensor("weight") && vb.contains_tensor("scales") && vb.contains_tensor("biases")
}

pub fn load_weight(vb: &VarBuilder, out_dim: usize, in_dim: usize) -> Result<Tensor> {
    if !has_mlx_affine_quantized_weight(vb) {
        return vb.get((out_dim, in_dim), "weight").map_err(Error::from);
    }

    let scales = tensor2_f32(vb.get_unchecked_dtype("scales", DType::F32)?)?;
    let biases = tensor2_f32(vb.get_unchecked_dtype("biases", DType::F32)?)?;
    if scales.len() != out_dim || biases.len() != out_dim {
        return Err(Error::ModelLoadError(format!(
            "Quantized tensor rows mismatch: expected out_dim={out_dim}, scales_rows={}, biases_rows={}",
            scales.len(),
            biases.len()
        )));
    }

    let weight = if let Ok(w_u32) = vb.get_unchecked_dtype("weight", DType::U32) {
        let packed = w_u32.to_vec2::<u32>()?;
        dequantize_affine_u32(packed, &scales, &biases, out_dim, in_dim)?
    } else if let Ok(w_u8) = vb.get_unchecked_dtype("weight", DType::U8) {
        let packed = w_u8.to_vec2::<u8>()?;
        dequantize_affine_u8(packed, &scales, &biases, out_dim, in_dim)?
    } else {
        return Err(Error::ModelLoadError(
            "MLX quantized tensor has unsupported packed dtype for `weight` (expected U32 or U8)"
                .to_string(),
        ));
    };

    let t = Tensor::from_vec(weight, (out_dim, in_dim), vb.device())?;
    t.to_dtype(vb.dtype()).map_err(Error::from)
}

pub fn load_linear(in_dim: usize, out_dim: usize, vb: VarBuilder) -> Result<Linear> {
    if has_mlx_affine_quantized_weight(&vb) {
        let ws = load_weight(&vb, out_dim, in_dim)?;
        let bias = if vb.contains_tensor("bias") {
            Some(vb.get(out_dim, "bias")?)
        } else {
            None
        };
        Ok(Linear::new(ws, bias))
    } else {
        candle_nn::linear(in_dim, out_dim, vb).map_err(Error::from)
    }
}

pub fn load_linear_no_bias(in_dim: usize, out_dim: usize, vb: VarBuilder) -> Result<Linear> {
    if has_mlx_affine_quantized_weight(&vb) {
        let ws = load_weight(&vb, out_dim, in_dim)?;
        Ok(Linear::new(ws, None))
    } else {
        candle_nn::linear_no_bias(in_dim, out_dim, vb).map_err(Error::from)
    }
}

pub fn load_embedding(vocab_size: usize, hidden_size: usize, vb: VarBuilder) -> Result<Embedding> {
    if has_mlx_affine_quantized_weight(&vb) {
        let ws = load_weight(&vb, vocab_size, hidden_size)?;
        Ok(Embedding::new(ws, hidden_size))
    } else {
        candle_nn::embedding(vocab_size, hidden_size, vb).map_err(Error::from)
    }
}

pub fn load_conv2d(
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    cfg: Conv2dConfig,
    vb: VarBuilder,
) -> Result<Conv2d> {
    let mut ws = vb.get_unchecked_dtype("weight", vb.dtype())?;

    let dims = ws.dims4()?;
    let expected_oihw = (out_channels, in_channels, kernel_size, kernel_size);
    let expected_ohwi = (out_channels, kernel_size, kernel_size, in_channels);

    if dims == expected_ohwi {
        ws = ws.permute((0, 3, 1, 2))?;
    } else if dims != expected_oihw {
        return Err(Error::ModelLoadError(format!(
            "Conv2d weight shape mismatch: got={dims:?}, expected OIHW={expected_oihw:?} or OHWI={expected_ohwi:?}"
        )));
    }

    let bias = if vb.contains_tensor("bias") {
        Some(vb.get(out_channels, "bias")?)
    } else {
        None
    };
    Ok(Conv2d::new(ws, bias, cfg))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dequantize_u32_4bit_affine() {
        let packed = vec![vec![0x7654_3210]];
        let scales = vec![vec![0.1, 0.2]];
        let biases = vec![vec![-1.0, 0.5]];
        let out = dequantize_affine_u32(packed, &scales, &biases, 1, 8).unwrap();

        let expected = vec![
            -1.0 + 0.0 * 0.1,
            -1.0 + 1.0 * 0.1,
            -1.0 + 2.0 * 0.1,
            -1.0 + 3.0 * 0.1,
            0.5 + 4.0 * 0.2,
            0.5 + 5.0 * 0.2,
            0.5 + 6.0 * 0.2,
            0.5 + 7.0 * 0.2,
        ];
        for (a, b) in out.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn dequantize_u8_8bit_affine() {
        let packed = vec![vec![1u8, 2, 3, 4]];
        let scales = vec![vec![0.5, 2.0]];
        let biases = vec![vec![0.0, -1.0]];
        let out = dequantize_affine_u8(packed, &scales, &biases, 1, 4).unwrap();

        let expected = vec![0.5, 1.0, 5.0, 7.0];
        for (a, b) in out.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }
}
