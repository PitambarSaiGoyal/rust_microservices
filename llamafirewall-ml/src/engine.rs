//! Inference engine trait for PyTorch-based ML inference
//!
//! This module provides the common interface for text classification using
//! PyTorch models via tch-rs, optimized for GPU acceleration (MPS/CUDA).

use anyhow::Result;

/// Inference engine trait for text classification
///
/// This trait provides a common API for text classification tasks like
/// prompt injection detection using PyTorch models.
///
/// ## Design Goals
/// - **GPU Optimized**: Leverages MPS (Apple Silicon) and CUDA (NVIDIA) acceleration
/// - **Zero-cost Abstraction**: Trait methods compile to direct calls
/// - **Testability**: Easy to mock for testing
///
/// ## Example
/// ```no_run
/// use llamafirewall_ml::{InferenceEngine, TchInferenceEngine};
/// # fn example() -> anyhow::Result<()> {
/// let engine = TchInferenceEngine::new("model.pt", "tokenizer.json")?;
///
/// let logits = engine.infer("What is the weather?")?;
/// println!("Classification: {:?}", logits);
/// # Ok(())
/// # }
/// ```
pub trait InferenceEngine: Send + Sync {
    /// Run inference on a single text input
    ///
    /// # Arguments
    /// * `text` - Input text to classify
    ///
    /// # Returns
    /// Logits for each class. For binary classification (like PromptGuard),
    /// returns `[benign_logit, jailbreak_logit]`.
    ///
    /// # Example
    /// ```no_run
    /// # use llamafirewall_ml::{InferenceEngine, TchInferenceEngine};
    /// # fn example() -> anyhow::Result<()> {
    /// # let engine = TchInferenceEngine::new("model.pt", "tokenizer.json")?;
    /// let logits = engine.infer("Hello world")?;
    /// assert_eq!(logits.len(), 2); // Binary classification
    /// # Ok(())
    /// # }
    /// ```
    fn infer(&self, text: &str) -> Result<Vec<f32>>;

    /// Run inference on a batch of texts (more efficient than sequential infer calls)
    ///
    /// # Arguments
    /// * `texts` - Slice of text inputs to classify
    ///
    /// # Returns
    /// Vector of logit vectors, one per input text
    ///
    /// # Example
    /// ```no_run
    /// # use llamafirewall_ml::{InferenceEngine, TchInferenceEngine};
    /// # fn example() -> anyhow::Result<()> {
    /// # let engine = TchInferenceEngine::new("model.pt", "tokenizer.json")?;
    /// let texts = vec!["Hello", "World"];
    /// let results = engine.infer_batch(&texts)?;
    /// assert_eq!(results.len(), 2);
    /// # Ok(())
    /// # }
    /// ```
    fn infer_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>>;

    /// Get the backend name (always "tch" for PyTorch)
    ///
    /// Useful for logging and monitoring.
    fn backend_name(&self) -> &str;

    /// Get backend-specific information (device, version, etc.)
    ///
    /// Returns a human-readable string with backend details.
    ///
    /// # Example Output
    /// - `"PyTorch 2.8.0 via tch-rs (MPS)"`
    /// - `"PyTorch 2.8.0 via tch-rs (CUDA:0)"`
    /// - `"PyTorch 2.8.0 via tch-rs (CPU)"`
    fn backend_info(&self) -> String {
        format!("{} backend", self.backend_name())
    }

    /// Warm up the model with dummy inference
    ///
    /// Useful for benchmarking to ensure JIT compilation and GPU memory
    /// allocation are done before measurements.
    ///
    /// Default implementation runs a single inference with empty text.
    fn warmup(&self) -> Result<()> {
        let _ = self.infer("warmup")?;
        Ok(())
    }
}

/// Create a PyTorch inference engine
///
/// This is a convenience function that creates a TchInferenceEngine.
///
/// # Arguments
/// * `model_path` - Path to model file (.pt for PyTorch JIT)
/// * `tokenizer_path` - Path to tokenizer.json
///
/// # Returns
/// A boxed inference engine trait object
///
/// # Example
/// ```no_run
/// use llamafirewall_ml::create_engine;
/// # fn example() -> anyhow::Result<()> {
/// let engine = create_engine(
///     "models/promptguard.pt",
///     "models/tokenizer.json"
/// )?;
/// let logits = engine.infer("Hello world")?;
/// # Ok(())
/// # }
/// ```
pub fn create_engine(
    model_path: &str,
    tokenizer_path: &str,
) -> anyhow::Result<Box<dyn InferenceEngine>> {
    use crate::TchInferenceEngine;
    let engine = TchInferenceEngine::new(model_path, tokenizer_path)?;
    Ok(Box::new(engine))
}

