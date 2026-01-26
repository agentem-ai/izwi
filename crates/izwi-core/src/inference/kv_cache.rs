//! KV Cache management for efficient inference
//!
//! Implements paged attention-style memory management for long-form
//! audio generation without memory explosions.

use std::collections::HashMap;
use tracing::debug;

/// Configuration for KV cache
#[derive(Debug, Clone)]
pub struct KVCacheConfig {
    /// Number of layers in the model
    pub num_layers: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Head dimension
    pub head_dim: usize,
    /// Block size for paged attention
    pub block_size: usize,
    /// Maximum sequence length
    pub max_seq_len: usize,
    /// Data type (affects memory usage)
    pub dtype: KVCacheDtype,
}

#[derive(Debug, Clone, Copy)]
pub enum KVCacheDtype {
    Float32,
    Float16,
    BFloat16,
}

impl KVCacheDtype {
    pub fn size_bytes(&self) -> usize {
        match self {
            Self::Float32 => 4,
            Self::Float16 | Self::BFloat16 => 2,
        }
    }
}

impl Default for KVCacheConfig {
    fn default() -> Self {
        Self {
            num_layers: 28,
            num_heads: 12,
            head_dim: 128,
            block_size: 16,
            max_seq_len: 4096,
            dtype: KVCacheDtype::Float16,
        }
    }
}

/// A block of KV cache memory
#[derive(Clone)]
pub struct KVBlock {
    /// Block ID
    pub id: usize,
    /// Key cache [num_layers, num_heads, block_size, head_dim]
    pub keys: Vec<f32>,
    /// Value cache [num_layers, num_heads, block_size, head_dim]  
    pub values: Vec<f32>,
    /// Number of tokens stored in this block
    pub num_tokens: usize,
}

impl KVBlock {
    fn new(id: usize, config: &KVCacheConfig) -> Self {
        let block_elements =
            config.num_layers * config.num_heads * config.block_size * config.head_dim;
        Self {
            id,
            keys: vec![0.0; block_elements],
            values: vec![0.0; block_elements],
            num_tokens: 0,
        }
    }

    #[allow(dead_code)]
    fn is_full(&self, block_size: usize) -> bool {
        self.num_tokens >= block_size
    }

    fn remaining_capacity(&self, block_size: usize) -> usize {
        block_size.saturating_sub(self.num_tokens)
    }
}

/// Paged KV Cache for efficient memory management
pub struct KVCache {
    config: KVCacheConfig,
    /// All allocated blocks
    blocks: Vec<KVBlock>,
    /// Free block IDs
    free_blocks: Vec<usize>,
    /// Sequence to block mapping
    sequence_blocks: HashMap<String, Vec<usize>>,
    /// Next block ID
    next_block_id: usize,
}

impl KVCache {
    /// Create a new KV cache
    pub fn new(config: KVCacheConfig) -> Self {
        // Pre-allocate some blocks
        let initial_blocks = 64;
        let mut blocks = Vec::with_capacity(initial_blocks);
        let mut free_blocks = Vec::with_capacity(initial_blocks);

        for i in 0..initial_blocks {
            blocks.push(KVBlock::new(i, &config));
            free_blocks.push(i);
        }

        Self {
            config,
            blocks,
            free_blocks,
            sequence_blocks: HashMap::new(),
            next_block_id: initial_blocks,
        }
    }

    /// Allocate blocks for a new sequence
    pub fn allocate_sequence(&mut self, sequence_id: &str, num_tokens: usize) -> Vec<usize> {
        let num_blocks = (num_tokens + self.config.block_size - 1) / self.config.block_size;
        let mut allocated = Vec::with_capacity(num_blocks);

        for _ in 0..num_blocks {
            let block_id = self.allocate_block();
            allocated.push(block_id);
        }

        self.sequence_blocks
            .insert(sequence_id.to_string(), allocated.clone());
        debug!(
            "Allocated {} blocks for sequence {}",
            allocated.len(),
            sequence_id
        );
        allocated
    }

    /// Allocate a single block
    fn allocate_block(&mut self) -> usize {
        if let Some(id) = self.free_blocks.pop() {
            // Reset the block
            self.blocks[id].num_tokens = 0;
            id
        } else {
            // Allocate new block
            let id = self.next_block_id;
            self.next_block_id += 1;
            self.blocks.push(KVBlock::new(id, &self.config));
            id
        }
    }

    /// Extend a sequence with more tokens
    pub fn extend_sequence(&mut self, sequence_id: &str, additional_tokens: usize) {
        // Check if sequence exists and if current blocks have space
        let needs_more_blocks = {
            let blocks = match self.sequence_blocks.get(sequence_id) {
                Some(b) => b,
                None => return,
            };

            // Check if current last block has space
            if let Some(&last_block_id) = blocks.last() {
                let block = &self.blocks[last_block_id];
                let remaining = block.remaining_capacity(self.config.block_size);
                remaining < additional_tokens
            } else {
                true
            }
        };

        if !needs_more_blocks {
            return;
        }

        // Allocate additional blocks
        let additional_blocks =
            (additional_tokens + self.config.block_size - 1) / self.config.block_size;
        let mut new_block_ids = Vec::with_capacity(additional_blocks);

        for _ in 0..additional_blocks {
            let block_id = self.allocate_block();
            new_block_ids.push(block_id);
        }

        // Now add the new blocks to the sequence
        if let Some(blocks) = self.sequence_blocks.get_mut(sequence_id) {
            blocks.extend(new_block_ids);
        }
    }

    /// Free blocks for a sequence
    pub fn free_sequence(&mut self, sequence_id: &str) {
        if let Some(blocks) = self.sequence_blocks.remove(sequence_id) {
            debug!(
                "Freeing {} blocks for sequence {}",
                blocks.len(),
                sequence_id
            );
            self.free_blocks.extend(blocks);
        }
    }

    /// Get blocks for a sequence
    pub fn get_sequence_blocks(&self, sequence_id: &str) -> Option<&[usize]> {
        self.sequence_blocks.get(sequence_id).map(|v| v.as_slice())
    }

    /// Get mutable reference to a block
    pub fn get_block_mut(&mut self, block_id: usize) -> Option<&mut KVBlock> {
        self.blocks.get_mut(block_id)
    }

    /// Update KV cache for a token
    pub fn update(&mut self, sequence_id: &str, _layer: usize, _keys: &[f32], _values: &[f32]) {
        // Get the last block for this sequence
        if let Some(blocks) = self.sequence_blocks.get(sequence_id) {
            if let Some(&block_id) = blocks.last() {
                if let Some(block) = self.blocks.get_mut(block_id) {
                    // In a real implementation, we would copy the KV tensors here
                    block.num_tokens += 1;
                }
            }
        }
    }

    /// Get memory usage in bytes
    pub fn memory_bytes(&self) -> usize {
        let block_size = self.config.num_layers
            * self.config.num_heads
            * self.config.block_size
            * self.config.head_dim
            * self.config.dtype.size_bytes();

        self.blocks.len() * block_size * 2 // *2 for keys and values
    }

    /// Get cache statistics
    pub fn stats(&self) -> KVCacheStats {
        KVCacheStats {
            total_blocks: self.blocks.len(),
            free_blocks: self.free_blocks.len(),
            active_sequences: self.sequence_blocks.len(),
            memory_bytes: self.memory_bytes(),
        }
    }
}

/// KV cache statistics
#[derive(Debug, Clone)]
pub struct KVCacheStats {
    pub total_blocks: usize,
    pub free_blocks: usize,
    pub active_sequences: usize,
    pub memory_bytes: usize,
}
