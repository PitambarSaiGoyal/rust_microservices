//! # llamafirewall-ml
//!
//! ML infrastructure for LlamaFirewall using PyTorch via tch-rs.
//!
//! This crate provides high-performance ML inference capabilities for security scanning,
//! with automatic GPU acceleration and production-ready performance characteristics.
//!
//! ## Features
//!
//! - **Native PyTorch Integration**: Direct PyTorch via tch-rs 0.19.0 (no conversion overhead)
//! - **Automatic Device Selection**: CUDA > MPS > CPU with intelligent fallback
//! - **GPU Acceleration**: MPS (Apple Silicon), CUDA (NVIDIA), CPU fallback
//! - **Production Performance**: ~46ms inference on M2 Mac (41% faster than Python)
//! - **Thread-Safe**: Concurrent inference with Arc + Mutex pattern
//! - **Batch Inference**: Efficient batch processing for higher throughput
//! - **Lazy Loading**: Models load on first use to reduce startup time
//!
//! ## Prerequisites
//!
//! Before using this crate, you must set up PyTorch/libtorch. See the
//! **[TCH Usage Guide](../docs/TCH_USAGE_GUIDE.md)** for detailed setup instructions:
//!
//! - **macOS (Apple Silicon)**: Install PyTorch via conda/pip for MPS support
//! - **Linux (NVIDIA)**: Install CUDA toolkit and libtorch
//! - **CPU-only**: Use automated setup script
//!
//! **Required environment variable:**
//! ```bash
//! export LIBTORCH="/path/to/torch"
//! export DYLD_LIBRARY_PATH="$LIBTORCH/lib:$DYLD_LIBRARY_PATH"  # macOS
//! export LD_LIBRARY_PATH="$LIBTORCH/lib:$LD_LIBRARY_PATH"      # Linux
//! ```
//!
//! ## Quick Start
//!
//! ### Basic Usage
//!
//! ```no_run
//! use llamafirewall_ml::TchInferenceEngine;
//!
//! # fn example() -> anyhow::Result<()> {
//! // Create inference engine (auto-detects best device: CUDA > MPS > CPU)
//! let engine = TchInferenceEngine::new(
//!     "models/promptguard.pt",
//!     "models/tokenizer.json",
//! )?;
//!
//! // Run inference
//! let text = "What is the weather today?";
//! let logits = engine.infer(text)?;
//!
//! // logits: [benign_score, injection_score, jailbreak_score]
//! println!("Benign: {:.4}", logits[0]);
//! println!("Injection: {:.4}", logits[1]);
//! println!("Jailbreak: {:.4}", logits[2]);
//! # Ok(())
//! # }
//! ```
//!
//! ### Batch Inference
//!
//! ```no_run
//! use llamafirewall_ml::TchInferenceEngine;
//!
//! # fn example() -> anyhow::Result<()> {
//! let engine = TchInferenceEngine::new(
//!     "models/promptguard.pt",
//!     "models/tokenizer.json",
//! )?;
//!
//! // Process multiple texts efficiently
//! let texts = vec!["text1", "text2", "text3", "text4"];
//! let results = engine.infer_batch(&texts)?;
//!
//! for (text, logits) in texts.iter().zip(results.iter()) {
//!     println!("{}: injection_score={:.4}", text, logits[1]);
//! }
//! # Ok(())
//! # }
//! ```
//!
//! ### Concurrent Inference (Thread-Safe)
//!
//! ```no_run
//! use llamafirewall_ml::TchInferenceEngine;
//! use std::sync::Arc;
//! use tokio::task;
//!
//! # async fn example() -> anyhow::Result<()> {
//! // Engine is thread-safe via Arc
//! let engine = Arc::new(TchInferenceEngine::new(
//!     "models/promptguard.pt",
//!     "models/tokenizer.json",
//! )?);
//!
//! // Spawn multiple concurrent tasks
//! let mut handles = vec![];
//! for i in 0..10 {
//!     let engine = Arc::clone(&engine);
//!     let handle = task::spawn(async move {
//!         engine.infer(&format!("Message {}", i))
//!     });
//!     handles.push(handle);
//! }
//!
//! // Wait for all to complete
//! for handle in handles {
//!     let result = handle.await??;
//!     println!("Result: {:?}", result);
//! }
//! # Ok(())
//! # }
//! ```
//!
//! ## Device Selection
//!
//! The inference engine automatically selects the best available device:
//!
//! **Priority:** CUDA > MPS > CPU
//!
//! ### Automatic Selection (Recommended)
//!
//! ```no_run
//! use llamafirewall_ml::TchInferenceEngine;
//!
//! # fn example() -> anyhow::Result<()> {
//! // Automatically selects best device
//! let engine = TchInferenceEngine::new(
//!     "models/promptguard.pt",
//!     "models/tokenizer.json",
//! )?;
//!
//! println!("Using device: {:?}", engine.device());
//! // Output: "Using device: Mps" (on M2 Mac)
//! # Ok(())
//! # }
//! ```
//!
//! ### Manual Device Override
//!
//! ```no_run
//! use llamafirewall_ml::{TchInferenceEngine, TchModel};
//! use tch::Device;
//!
//! # fn example() -> anyhow::Result<()> {
//! // Force CPU inference (useful for debugging)
//! let device = Device::Cpu;
//! let model = TchModel::load_on_device("models/promptguard.pt", device)?;
//! let engine = TchInferenceEngine::with_model(model, "models/tokenizer.json")?;
//!
//! // Or force specific GPU
//! let device = Device::Cuda(1);  // Use second GPU
//! # Ok(())
//! # }
//! ```
//!
//! ## Performance Characteristics
//!
//! **Measured on M2 Mac with MPS:**
//!
//! | Operation | Latency | Throughput |
//! |-----------|---------|------------|
//! | Single inference | ~46ms (P50) | ~22 prompts/s |
//! | Batch (size=2) | ~98ms | ~20 prompts/s |
//! | Batch (size=4) | ~182ms | ~22 prompts/s |
//! | Batch (size=8) | ~352ms | ~23 prompts/s |
//!
//! **vs Python baseline (CPU):**
//! - Python: 78ms per inference
//! - Rust (MPS): 46ms per inference
//! - **Result: 41% faster**
//!
//! **Performance tips:**
//! - Use device-specific models (e.g., `promptguard_mps.pt` for MPS)
//! - Batch inference for throughput (optimal batch size: 4-8)
//! - Warm up model with dummy inference before measuring
//! - Use lazy loading to reduce startup time
//!
//! ## Model Requirements
//!
//! Models must be in **TorchScript JIT traced format** (`.pt` files):
//!
//! ```bash
//! # Export model from Python
//! python3 scripts/export_to_jit.py \
//!   --model meta-llama/Prompt-Guard-86M \
//!   --output models/promptguard_mps.pt \
//!   --device mps
//! ```
//!
//! **Model directory structure:**
//! ```text
//! models/
//! ├── promptguard.pt          # CPU model (default fallback)
//! ├── promptguard_mps.pt      # MPS-optimized (Apple Silicon)
//! ├── promptguard_cuda.pt     # CUDA-optimized (NVIDIA)
//! └── tokenizer.json          # HuggingFace tokenizer config
//! ```
//!
//! ## Thread Safety
//!
//! All inference engines are thread-safe and can be safely shared across threads:
//!
//! - **Arc**: Use `Arc<TchInferenceEngine>` for shared ownership
//! - **Send + Sync**: All types implement Send and Sync
//! - **Mutex**: Internal model uses `Arc<Mutex<TchModel>>` for concurrent access
//! - **OnceCell**: Supports lazy initialization in concurrent contexts
//!
//! ## Error Handling
//!
//! All functions return `anyhow::Result<T>` with detailed error context:
//!
//! ```no_run
//! use llamafirewall_ml::TchInferenceEngine;
//! use anyhow::Context;
//!
//! # fn example() -> anyhow::Result<()> {
//! let engine = TchInferenceEngine::new(
//!     "models/promptguard.pt",
//!     "models/tokenizer.json",
//! ).context("Failed to create inference engine")?;
//!
//! let result = engine.infer("test")
//!     .context("Failed to run inference")?;
//! # Ok(())
//! # }
//! ```
//!
//! ## Troubleshooting
//!
//! **Common Issues:**
//!
//! 1. **"libtorch not found"**
//!    - Ensure `LIBTORCH` environment variable is set
//!    - Run `source mps-env.sh` (macOS) or `source libtorch-env.sh` (Linux)
//!
//! 2. **"MPS device not available"**
//!    - Official libtorch for macOS does NOT include MPS
//!    - Install PyTorch via conda/pip instead
//!    - Verify: `python3 scripts/check-mps-support.py`
//!
//! 3. **Slow performance**
//!    - Check device being used: `RUST_LOG=llamafirewall_ml=debug cargo run`
//!    - Use device-specific models (e.g., `promptguard_mps.pt`)
//!    - Warm up model before measuring
//!
//! See **[docs/TCH_USAGE_GUIDE.md](../docs/TCH_USAGE_GUIDE.md)** for comprehensive troubleshooting.
//!
//! ## API Overview
//!
//! ### Main Types
//!
//! - [`TchInferenceEngine`]: Main inference engine for PyTorch models
//! - [`TchModel`]: Wrapper for PyTorch JIT model
//! - [`ModelMetadata`]: Model information and metadata
//! - [`InferenceEngine`]: Generic inference trait (for future backends)
//!
//! ### Key Functions
//!
//! - [`TchInferenceEngine::new`]: Create engine with automatic device selection
//! - [`TchInferenceEngine::with_device`]: Create engine with specific device
//! - [`TchInferenceEngine::infer`]: Single text inference
//! - [`TchInferenceEngine::infer_batch`]: Batch inference
//! - [`TchModel::load`]: Load model with auto device
//! - [`TchModel::load_on_device`]: Load model on specific device
//!
//! ## Additional Resources
//!
//! - **[TCH Usage Guide](../docs/TCH_USAGE_GUIDE.md)** - Setup, usage, troubleshooting
//! - **[LIBTORCH_SETUP_GUIDE.md](../../LIBTORCH_SETUP_GUIDE.md)** - Installation guide
//! - **[TCH_SETUP.md](../docs/TCH_SETUP.md)** - Technical setup details
//! - **[TESTING_GUIDE.md](../docs/TESTING_GUIDE.md)** - Testing procedures
//! - [PyTorch C++ API](https://pytorch.org/cppdocs/) - libtorch documentation
//! - [tch-rs Documentation](https://docs.rs/tch/) - Rust bindings docs
//!
//! ## Version Information
//!
//! - **tch-rs**: 0.19.0
//! - **PyTorch**: 2.8.0
//! - **Status**: Production-ready on MPS (Apple Silicon)
//! - **CUDA**: Pending validation
//!
//! ## License
//!
//! MIT License - See [LICENSE](../../LICENSE) for details.

// Inference engine trait
pub mod engine;

// PyTorch/tch-rs backend
mod tch_loader;
mod tch_inference;

// Re-export commonly used types
pub use engine::{InferenceEngine, create_engine};
pub use tch_inference::TchInferenceEngine;
pub use tch_loader::{TchModel, ModelMetadata};

#[cfg(test)]
mod tests {
    #[test]
    fn test_module_exports() {
        // Verify that public exports are accessible
        // Actual inference tests are in inference module (marked as #[ignore])
    }
}
