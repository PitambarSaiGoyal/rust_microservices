//! PyTorch/TorchScript model loading for tch-rs backend
//!
//! This module provides functionality to load PyTorch models exported
//! to TorchScript JIT format for use with the tch-rs inference engine.
//!
//! Part of Phase 2: Model Loading & Format Conversion


use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;
use tch::{CModule, Device};

/// Metadata for a loaded PyTorch model
///
/// This struct contains information about the model that's loaded,
/// which is used for validation and configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    /// Original model name or path (e.g., "meta-llama/Prompt-Guard-86M")
    pub model_name_or_path: String,
    /// Maximum sequence length the model was exported with
    pub max_seq_length: usize,
    /// Number of output labels/classes
    pub num_labels: usize,
    /// Vocabulary size
    pub vocab_size: usize,
    /// Hidden dimension size
    pub hidden_size: usize,
    /// Number of attention heads
    pub num_attention_heads: usize,
    /// Number of transformer layers
    pub num_hidden_layers: usize,
    /// Model architecture type (e.g., "bert", "deberta")
    pub model_type: String,
    /// File size in MB
    pub file_size_mb: f64,
    /// PyTorch version used for export
    pub pytorch_version: String,
    /// Device used during export
    pub device: String,
    /// Whether the model was optimized during export
    pub optimized: bool,
}

impl ModelMetadata {
    /// Load metadata from a JSON file
    ///
    /// The metadata file should be created by the Python export script
    /// (export_to_jit.py) and stored alongside the model file with a .json extension.
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let metadata_path = path.as_ref();
        let contents = std::fs::read_to_string(metadata_path)
            .with_context(|| format!("Failed to read metadata file: {:?}", metadata_path))?;

        let metadata: ModelMetadata = serde_json::from_str(&contents)
            .with_context(|| format!("Failed to parse metadata JSON: {:?}", metadata_path))?;

        Ok(metadata)
    }

    /// Get the expected model file path from metadata file path
    ///
    /// E.g., models/promptguard.json -> models/promptguard.pt
    pub fn model_path_from_metadata<P: AsRef<Path>>(metadata_path: P) -> std::path::PathBuf {
        let path = metadata_path.as_ref();
        path.with_extension("pt")
    }
}

/// Loaded PyTorch model container
///
/// Contains both the TorchScript compiled module and associated metadata.
pub struct TchModel {
    /// The loaded TorchScript module
    pub module: CModule,
    /// Model metadata
    pub metadata: ModelMetadata,
    /// Device the model is loaded on
    pub device: Device,
}

impl TchModel {
    /// Load a PyTorch model from a TorchScript file
    ///
    /// # Arguments
    /// * `model_path` - Path to the .pt TorchScript model file
    /// * `device` - Device to load the model on (CPU, CUDA, MPS)
    ///
    /// # Returns
    /// A `TchModel` containing the loaded module and metadata
    ///
    /// # Example
    /// ```no_run
    /// # use llamafirewall_ml::tch_loader::TchModel;
    /// # use tch::Device;
    /// # fn example() -> anyhow::Result<()> {
    /// let model = TchModel::load("models/promptguard.pt", Device::Cpu)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn load<P: AsRef<Path>>(model_path: P, device: Device) -> Result<Self> {
        let model_path = model_path.as_ref();

        // Verify model file exists
        if !model_path.exists() {
            anyhow::bail!("Model file not found: {:?}", model_path);
        }

        // Load metadata (optional but recommended)
        let metadata_path = model_path.with_extension("json");
        let metadata = if metadata_path.exists() {
            ModelMetadata::from_file(&metadata_path)
                .context("Failed to load model metadata")?
        } else {
            tracing::warn!(
                "Metadata file not found: {:?}. Using default values.",
                metadata_path
            );

            // Create default metadata
            ModelMetadata {
                model_name_or_path: model_path.to_string_lossy().to_string(),
                max_seq_length: 512,
                num_labels: 2,
                vocab_size: 30522, // BERT default
                hidden_size: 768,
                num_attention_heads: 12,
                num_hidden_layers: 12,
                model_type: "unknown".to_string(),
                file_size_mb: 0.0,
                pytorch_version: "unknown".to_string(),
                device: "unknown".to_string(),
                optimized: false,
            }
        };

        tracing::info!(
            "Loading PyTorch model from {:?} on device {:?}",
            model_path,
            device
        );

        // Load the TorchScript module
        let mut module = CModule::load(model_path)
            .with_context(|| format!("Failed to load TorchScript model: {:?}", model_path))?;

        // Set to evaluation mode (disables dropout, batch norm training mode, etc.)
        module.set_eval();

        // Move model to target device (v0.17.0 API: modifies in-place)
        module.to(device, tch::Kind::Float, false);

        tracing::info!(
            "Successfully loaded model: {} (type: {}, labels: {})",
            metadata.model_name_or_path,
            metadata.model_type,
            metadata.num_labels
        );

        Ok(TchModel {
            module,
            metadata,
            device,
        })
    }

    /// Load model and automatically select the best available device
    ///
    /// Selection priority:
    /// 1. CUDA if available
    /// 2. MPS if available (Apple Silicon)
    /// 3. CPU as fallback
    pub fn load_auto<P: AsRef<Path>>(model_path: P) -> Result<Self> {
        let device = Self::select_device();
        tracing::info!("Auto-selected device: {:?}", device);
        Self::load(model_path, device)
    }

    /// Select the best available device
    ///
    /// Priority: CUDA > MPS > CPU
    pub fn select_device() -> Device {
        if tch::Cuda::is_available() {
            tracing::info!("CUDA available, using GPU device 0");
            Device::Cuda(0)
        } else if tch::utils::has_mps() {
            tracing::info!("MPS available, using Apple Silicon GPU");
            Device::Mps
        } else {
            tracing::info!("Using CPU device");
            Device::Cpu
        }
    }

    /// Get information about available devices
    pub fn device_info() -> String {
        let mut info = Vec::new();

        info.push(format!("CPU: Available"));

        if tch::Cuda::is_available() {
            let device_count = tch::Cuda::device_count();
            info.push(format!("CUDA: Available ({} devices)", device_count));

            for i in 0..device_count {
                info.push(format!("  - Device {}", i));
            }
        } else {
            info.push(format!("CUDA: Not available"));
        }

        if tch::utils::has_mps() {
            info.push(format!("MPS: Available (Apple Silicon)"));
        } else {
            info.push(format!("MPS: Not available"));
        }

        info.join("\n")
    }

    /// Warm up the model with a dummy forward pass
    ///
    /// This can help with more accurate benchmarking by ensuring
    /// all initialization is done before timing inference.
    pub fn warmup(&mut self, batch_size: i64, seq_length: i64) -> Result<()> {
        tracing::info!("Warming up model with batch_size={}, seq_length={}", batch_size, seq_length);

        // Create dummy inputs
        let input_ids = tch::Tensor::zeros(&[batch_size, seq_length], (tch::Kind::Int64, self.device));
        let attention_mask = tch::Tensor::ones(&[batch_size, seq_length], (tch::Kind::Int64, self.device));

        // Run inference (no_grad mode for efficiency)
        tch::no_grad(|| {
            let _output = self.module.forward_ts(&[input_ids, attention_mask])
                .context("Warmup forward pass failed")?;
            Ok::<_, anyhow::Error>(())
        })?;

        tracing::info!("Model warmup complete");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metadata_serialization() {
        let metadata = ModelMetadata {
            model_name_or_path: "test-model".to_string(),
            max_seq_length: 512,
            num_labels: 2,
            vocab_size: 30522,
            hidden_size: 768,
            num_attention_heads: 12,
            num_hidden_layers: 12,
            model_type: "bert".to_string(),
            file_size_mb: 330.0,
            pytorch_version: "2.1.0".to_string(),
            device: "cpu".to_string(),
            optimized: true,
        };

        // Test serialization
        let json = serde_json::to_string_pretty(&metadata).unwrap();
        println!("Metadata JSON:\n{}", json);

        // Test deserialization
        let parsed: ModelMetadata = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.model_name_or_path, "test-model");
        assert_eq!(parsed.max_seq_length, 512);
    }

    #[test]
    fn test_device_selection() {
        let device = TchModel::select_device();
        println!("Selected device: {:?}", device);

        // Device should be one of the supported types
        match device {
            Device::Cpu => println!("Using CPU"),
            Device::Cuda(_) => println!("Using CUDA"),
            Device::Mps => println!("Using MPS"),
            _ => println!("Using other device: {:?}", device),
        }
    }

    #[test]
    fn test_device_info() {
        let info = TchModel::device_info();
        println!("Device Information:\n{}", info);

        // Should at least have CPU
        assert!(info.contains("CPU"));
    }

    #[test]
    fn test_model_path_from_metadata() {
        let metadata_path = Path::new("models/promptguard.json");
        let model_path = ModelMetadata::model_path_from_metadata(metadata_path);

        assert_eq!(model_path, Path::new("models/promptguard.pt"));
    }
}
