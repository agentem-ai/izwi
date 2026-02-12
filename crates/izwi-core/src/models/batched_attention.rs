//! Batched attention operations for improved throughput.
//!
//! This module provides batched attention implementations that can process
//! multiple sequences in parallel, significantly improving throughput for
//! concurrent requests.

use candle_core::{DType, IndexOp, Shape, Tensor, D};
use tracing::debug;

use crate::error::{Error, Result};
use crate::models::metal_memory::{metal_pool_for_device, PooledTensor};

/// Batched attention input for multiple sequences
#[derive(Debug)]
pub struct BatchedAttentionInput {
    /// Query tensors [batch_size, seq_len, hidden_size]
    pub queries: Tensor,
    /// Key tensors [batch_size, seq_len, hidden_size]
    pub keys: Tensor,
    /// Value tensors [batch_size, seq_len, hidden_size]
    pub values: Tensor,
    /// Attention mask [batch_size, seq_len, seq_len] or None
    pub attention_mask: Option<Tensor>,
    /// Actual sequence lengths for each batch item (for padding)
    pub seq_lengths: Vec<usize>,
}

/// Configuration for batched attention
#[derive(Debug, Clone)]
pub struct BatchedAttentionConfig {
    /// Number of attention heads
    pub num_heads: usize,
    /// Head dimension
    pub head_dim: usize,
    /// Scaling factor for attention scores (typically sqrt(head_dim))
    pub scale: f64,
    /// Use flash attention if available
    pub use_flash_attention: bool,
}

impl BatchedAttentionConfig {
    pub fn new(num_heads: usize, head_dim: usize) -> Self {
        Self {
            num_heads,
            head_dim,
            scale: (head_dim as f64).sqrt(),
            use_flash_attention: false, // Flash attention not yet available in Candle
        }
    }
}

/// Compute batched scaled dot-product attention
///
/// Processes multiple sequences in parallel for better GPU utilization.
/// This is more efficient than processing sequences one at a time.
pub fn batched_scaled_dot_product_attention(
    input: &BatchedAttentionInput,
    config: &BatchedAttentionConfig,
) -> Result<Tensor> {
    let bsz = input.queries.dim(0)?;
    let seq_len = input.queries.dim(1)?;
    let _hidden_size = input.queries.dim(2)?;

    debug!(
        "Batched attention: batch={}, seq_len={}, heads={}",
        bsz, seq_len, config.num_heads
    );

    // Reshape for multi-head attention: [batch, seq, heads, head_dim]
    let q = input
        .queries
        .reshape((bsz, seq_len, config.num_heads, config.head_dim))?
        .transpose(1, 2)?; // [batch, heads, seq, head_dim]
    let k = input
        .keys
        .reshape((bsz, seq_len, config.num_heads, config.head_dim))?
        .transpose(1, 2)?;
    let v = input
        .values
        .reshape((bsz, seq_len, config.num_heads, config.head_dim))?
        .transpose(1, 2)?;

    // Compute attention scores: [batch, heads, seq, seq]
    let scale_tensor = Tensor::new(&[config.scale as f32], input.queries.device())?
        .to_dtype(input.queries.dtype())?;

    let mut attn = q.matmul(&k.transpose(D::Minus2, D::Minus1)?)?;
    attn = attn.broadcast_div(&scale_tensor)?;

    // Apply attention mask if provided
    if let Some(mask) = &input.attention_mask {
        attn = attn.broadcast_add(mask)?;
    }

    // Softmax over the last dimension (keys)
    let attn = candle_nn::ops::softmax(&attn, D::Minus1)?;

    // Apply attention to values: [batch, heads, seq, head_dim]
    let output = attn.matmul(&v)?;

    // Transpose and reshape back: [batch, seq, hidden_size]
    let output = output
        .transpose(1, 2)?
        .reshape((bsz, seq_len, _hidden_size))?;

    Ok(output)
}

/// Compute attention for a single sequence (convenience wrapper)
pub fn single_sequence_attention(
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    attention_mask: Option<&Tensor>,
    config: &BatchedAttentionConfig,
) -> Result<Tensor> {
    let mask = match attention_mask {
        Some(m) => Some(m.unsqueeze(0).map_err(Error::from)?),
        None => None,
    };

    let input = BatchedAttentionInput {
        queries: query.unsqueeze(0).map_err(Error::from)?,
        keys: key.unsqueeze(0).map_err(Error::from)?,
        values: value.unsqueeze(0).map_err(Error::from)?,
        attention_mask: mask,
        seq_lengths: vec![query.dim(0).map_err(Error::from)?],
    };

    let output = batched_scaled_dot_product_attention(&input, config)?;
    output.squeeze(0).map_err(Error::from)
}

/// Attention batch builder for accumulating multiple sequences
pub struct AttentionBatchBuilder {
    queries: Vec<Tensor>,
    keys: Vec<Tensor>,
    values: Vec<Tensor>,
    masks: Vec<Option<Tensor>>,
    max_seq_len: usize,
    hidden_size: usize,
    dtype: DType,
    device: candle_core::Device,
    memory_pool: Option<std::sync::Arc<crate::models::metal_memory::MetalMemoryPool>>,
}

impl AttentionBatchBuilder {
    /// Create a new batch builder
    pub fn new(hidden_size: usize, dtype: DType, device: candle_core::Device) -> Self {
        let memory_pool = metal_pool_for_device(&device);
        Self {
            queries: Vec::new(),
            keys: Vec::new(),
            values: Vec::new(),
            masks: Vec::new(),
            max_seq_len: 0,
            hidden_size,
            dtype,
            device,
            memory_pool,
        }
    }

    /// Add a sequence to the batch
    pub fn add_sequence(
        &mut self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Option<Tensor>,
    ) -> Result<()> {
        let seq_len = query.dim(0)?;
        self.max_seq_len = self.max_seq_len.max(seq_len);

        self.queries.push(query);
        self.keys.push(key);
        self.values.push(value);
        self.masks.push(mask);

        Ok(())
    }

    /// Build the batched attention input with padding
    pub fn build(mut self) -> Result<BatchedAttentionInput> {
        let batch_size = self.queries.len();
        if batch_size == 0 {
            return Err(Error::InvalidInput(
                "Cannot build empty attention batch".to_string(),
            ));
        }

        // Take ownership of vectors
        let queries = std::mem::take(&mut self.queries);
        let keys = std::mem::take(&mut self.keys);
        let values = std::mem::take(&mut self.values);

        // Pad sequences to max length
        let seq_lengths: Vec<usize> = queries
            .iter()
            .map(|q| q.dim(0).map_err(Error::from))
            .collect::<Result<Vec<_>>>()?;

        let max_seq_len = self.max_seq_len;
        let hidden_size = self.hidden_size;
        let dtype = self.dtype;
        let device = self.device.clone();
        let memory_pool = self.memory_pool.clone();

        let padded_queries = Self::pad_tensors(
            queries,
            max_seq_len,
            hidden_size,
            dtype,
            &device,
            memory_pool.clone(),
        )?;
        let padded_keys = Self::pad_tensors(
            keys,
            max_seq_len,
            hidden_size,
            dtype,
            &device,
            memory_pool.clone(),
        )?;
        let padded_values = Self::pad_tensors(
            values,
            max_seq_len,
            hidden_size,
            dtype,
            &device,
            memory_pool,
        )?;

        // Stack into batch tensors
        let queries = Tensor::stack(&padded_queries, 0).map_err(Error::from)?;
        let keys = Tensor::stack(&padded_keys, 0).map_err(Error::from)?;
        let values = Tensor::stack(&padded_values, 0).map_err(Error::from)?;

        // Build attention mask for padding if needed
        let attention_mask = if seq_lengths.iter().any(|&len| len < max_seq_len) {
            Some(Self::build_padding_mask(
                &seq_lengths,
                max_seq_len,
                &device,
            )?)
        } else {
            None
        };

        Ok(BatchedAttentionInput {
            queries,
            keys,
            values,
            attention_mask,
            seq_lengths,
        })
    }

    fn pad_tensors(
        tensors: Vec<Tensor>,
        target_len: usize,
        hidden_size: usize,
        dtype: DType,
        device: &candle_core::Device,
        memory_pool: Option<std::sync::Arc<crate::models::metal_memory::MetalMemoryPool>>,
    ) -> Result<Vec<Tensor>> {
        tensors
            .into_iter()
            .map(|t| {
                let current_len = t.dim(0).map_err(Error::from)?;
                if current_len < target_len {
                    // Pad with zeros
                    let pad_len = target_len - current_len;
                    let shape = Shape::from(vec![pad_len, hidden_size]);
                    let pooled = if let Some(pool) = memory_pool.clone() {
                        let tensor = pool.acquire(&shape, dtype)?;
                        PooledTensor::new(tensor, Some(pool))
                    } else {
                        let tensor = Tensor::zeros(shape, dtype, device).map_err(Error::from)?;
                        PooledTensor::new(tensor, None)
                    };
                    let padded = Tensor::cat(&[&t, pooled.tensor()], 0).map_err(Error::from)?;
                    Ok(padded)
                } else {
                    Ok(t)
                }
            })
            .collect::<Result<Vec<_>>>()
    }

    fn build_padding_mask(
        seq_lengths: &[usize],
        max_len: usize,
        device: &candle_core::Device,
    ) -> Result<Tensor> {
        let batch_size = seq_lengths.len();
        let mut mask_data = vec![0.0f32; batch_size * max_len * max_len];

        for (b, &seq_len) in seq_lengths.iter().enumerate() {
            for i in 0..max_len {
                for j in seq_len..max_len {
                    // Mask out padded positions
                    mask_data[b * max_len * max_len + i * max_len + j] = f32::NEG_INFINITY;
                }
            }
        }

        Tensor::from_vec(mask_data, (batch_size, max_len, max_len), device).map_err(Error::from)
    }

    /// Returns the current batch size
    pub fn len(&self) -> usize {
        self.queries.len()
    }

    /// Returns true if the batch is empty
    pub fn is_empty(&self) -> bool {
        self.queries.is_empty()
    }
}

/// Optimized attention for Qwen3 models
pub struct Qwen3BatchedAttention {
    config: BatchedAttentionConfig,
}

impl Qwen3BatchedAttention {
    pub fn new(num_heads: usize, head_dim: usize) -> Self {
        Self {
            config: BatchedAttentionConfig::new(num_heads, head_dim),
        }
    }

    /// Process multiple sequences with batched attention
    pub fn forward_batched(
        &self,
        queries: &[Tensor],
        keys: &[Tensor],
        values: &[Tensor],
    ) -> Result<Vec<Tensor>> {
        if queries.len() == 1 {
            // Single sequence - no need to batch
            let output =
                single_sequence_attention(&queries[0], &keys[0], &values[0], None, &self.config)?;
            return Ok(vec![output]);
        }

        // Build batched input
        let mut builder = AttentionBatchBuilder::new(
            queries[0].dim(1)?,
            queries[0].dtype(),
            queries[0].device().clone(),
        );

        for ((q, k), v) in queries.iter().zip(keys.iter()).zip(values.iter()) {
            builder.add_sequence(q.clone(), k.clone(), v.clone(), None)?;
        }

        let batched_input = builder.build()?;
        let batched_output = batched_scaled_dot_product_attention(&batched_input, &self.config)?;

        // Split back into individual sequences
        let mut outputs = Vec::with_capacity(queries.len());
        for (i, seq_len) in batched_input.seq_lengths.iter().enumerate() {
            let output = batched_output.i(i)?.narrow(0, 0, *seq_len)?;
            outputs.push(output);
        }

        Ok(outputs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_single_sequence_attention() {
        let device = Device::Cpu;
        let config = BatchedAttentionConfig::new(8, 64);

        // Create test tensors [seq_len, hidden_size]
        let query = Tensor::randn(0.0f32, 1.0f32, (10, 512), &device).unwrap();
        let key = Tensor::randn(0.0f32, 1.0f32, (10, 512), &device).unwrap();
        let value = Tensor::randn(0.0f32, 1.0f32, (10, 512), &device).unwrap();

        let output = single_sequence_attention(&query, &key, &value, None, &config).unwrap();

        assert_eq!(output.dims(), &[10, 512]);
    }

    #[test]
    fn test_attention_batch_builder() {
        let device = Device::Cpu;
        let hidden_size = 512;

        let mut builder = AttentionBatchBuilder::new(hidden_size, DType::F32, device.clone());

        // Add sequences of different lengths
        let q1 = Tensor::randn(0.0f32, 1.0f32, (5, hidden_size), &device).unwrap();
        let k1 = Tensor::randn(0.0f32, 1.0f32, (5, hidden_size), &device).unwrap();
        let v1 = Tensor::randn(0.0f32, 1.0f32, (5, hidden_size), &device).unwrap();
        builder.add_sequence(q1, k1, v1, None).unwrap();

        let q2 = Tensor::randn(0.0f32, 1.0f32, (8, hidden_size), &device).unwrap();
        let k2 = Tensor::randn(0.0f32, 1.0f32, (8, hidden_size), &device).unwrap();
        let v2 = Tensor::randn(0.0f32, 1.0f32, (8, hidden_size), &device).unwrap();
        builder.add_sequence(q2, k2, v2, None).unwrap();

        let batched = builder.build().unwrap();

        assert_eq!(batched.queries.dim(0).unwrap(), 2); // batch_size
        assert_eq!(batched.queries.dim(1).unwrap(), 8); // max_seq_len
        assert_eq!(batched.seq_lengths, vec![5, 8]);
    }
}
