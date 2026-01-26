//! Model weight loading from safetensors

use safetensors::SafeTensors;
use std::collections::HashMap;
use std::path::Path;
use tracing::{debug, info};

use crate::config::ModelConfig;
use crate::error::Result;

/// Tensor data loaded from safetensors
#[derive(Debug)]
pub struct TensorData {
    pub name: String,
    pub shape: Vec<usize>,
    pub dtype: TensorDtype,
    pub data: Vec<u8>,
}

/// Supported tensor data types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorDtype {
    Float32,
    Float16,
    BFloat16,
    Int32,
    Int64,
    Uint8,
}

impl TensorDtype {
    pub fn from_safetensors(dtype: safetensors::Dtype) -> Self {
        match dtype {
            safetensors::Dtype::F32 => Self::Float32,
            safetensors::Dtype::F16 => Self::Float16,
            safetensors::Dtype::BF16 => Self::BFloat16,
            safetensors::Dtype::I32 => Self::Int32,
            safetensors::Dtype::I64 => Self::Int64,
            safetensors::Dtype::U8 => Self::Uint8,
            _ => Self::Float32, // Default fallback
        }
    }

    pub fn size_bytes(&self) -> usize {
        match self {
            Self::Float32 | Self::Int32 => 4,
            Self::Float16 | Self::BFloat16 => 2,
            Self::Int64 => 8,
            Self::Uint8 => 1,
        }
    }
}

/// Loaded model weights
pub struct ModelWeights {
    pub config: ModelConfig,
    pub tensors: HashMap<String, TensorData>,
}

impl ModelWeights {
    /// Load model weights from a directory
    pub fn load(model_dir: &Path) -> Result<Self> {
        info!("Loading model weights from {:?}", model_dir);

        // Load config
        let config_path = model_dir.join("config.json");
        let config: ModelConfig = if config_path.exists() {
            let config_str = std::fs::read_to_string(&config_path)?;
            serde_json::from_str(&config_str)?
        } else {
            ModelConfig::default()
        };

        debug!("Model config: {:?}", config);

        // Find and load safetensors files
        let mut tensors = HashMap::new();
        let safetensor_files = Self::find_safetensor_files(model_dir)?;

        for file_path in safetensor_files {
            debug!("Loading weights from {:?}", file_path);
            let file_tensors = Self::load_safetensors(&file_path)?;
            tensors.extend(file_tensors);
        }

        info!("Loaded {} tensors", tensors.len());

        Ok(Self { config, tensors })
    }

    /// Find all safetensors files in directory
    fn find_safetensor_files(dir: &Path) -> Result<Vec<std::path::PathBuf>> {
        let mut files = Vec::new();

        for entry in std::fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            if path
                .extension()
                .map(|e| e == "safetensors")
                .unwrap_or(false)
            {
                files.push(path);
            }
        }

        // Sort to ensure consistent loading order
        files.sort();
        Ok(files)
    }

    /// Load tensors from a single safetensors file
    fn load_safetensors(path: &Path) -> Result<HashMap<String, TensorData>> {
        let data = std::fs::read(path)?;
        let tensors = SafeTensors::deserialize(&data)?;

        let mut result = HashMap::new();

        for (name, tensor) in tensors.tensors() {
            let shape: Vec<usize> = tensor.shape().to_vec();
            let dtype = TensorDtype::from_safetensors(tensor.dtype());
            let tensor_data = tensor.data().to_vec();

            result.insert(
                name.to_string(),
                TensorData {
                    name: name.to_string(),
                    shape,
                    dtype,
                    data: tensor_data,
                },
            );
        }

        Ok(result)
    }

    /// Get a tensor by name
    pub fn get(&self, name: &str) -> Option<&TensorData> {
        self.tensors.get(name)
    }

    /// Get tensor names matching a prefix
    pub fn get_by_prefix(&self, prefix: &str) -> Vec<&TensorData> {
        self.tensors
            .iter()
            .filter(|(name, _)| name.starts_with(prefix))
            .map(|(_, tensor)| tensor)
            .collect()
    }

    /// Total memory used by weights
    pub fn memory_bytes(&self) -> usize {
        self.tensors.values().map(|t| t.data.len()).sum()
    }
}
