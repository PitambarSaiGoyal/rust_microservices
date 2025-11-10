# Running HuggingFace Models with tch-rs in Rust

A comprehensive guide for Rust developers to load and run HuggingFace models using PyTorch via tch-rs.

**Last Updated:** 2025-11-05
**tch-rs Version:** 0.19.0+
**Target Audience:** Rust developers familiar with ML concepts

---

## Table of Contents

1. [Introduction](#introduction)
2. [Understanding the Stack](#understanding-the-stack)
3. [Prerequisites](#prerequisites)
4. [Quick Start](#quick-start)
5. [Model Export Process](#model-export-process)
6. [Loading Models in Rust](#loading-models-in-rust)
7. [Tokenization](#tokenization)
8. [Running Inference](#running-inference)
9. [Common Patterns](#common-patterns)
10. [Performance Optimization](#performance-optimization)
11. [Troubleshooting](#troubleshooting)
12. [Complete Examples](#complete-examples)

---

## Introduction

### What This Guide Covers

This guide teaches you how to:
- Export HuggingFace models to a format Rust can use (TorchScript JIT)
- Set up tch-rs for PyTorch inference in Rust
- Load and run transformer models efficiently
- Handle tokenization and preprocessing
- Optimize for production use

### What You'll Build

By the end of this guide, you'll be able to run any HuggingFace model in Rust with GPU acceleration, achieving performance comparable to or better than Python.

---

## Understanding the Stack

### The Complete Pipeline

```
┌─────────────────┐
│  HuggingFace    │  Models hosted on huggingface.co
│  Model Hub      │  (BERT, GPT, RoBERTa, etc.)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Python Export  │  Convert to TorchScript JIT format
│  (torch.jit)    │  Using Python + transformers library
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  TorchScript    │  Serialized .pt file
│  JIT Model      │  Device-specific (CPU/CUDA/MPS)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  tch-rs         │  Rust bindings to libtorch
│  (Rust)         │  Zero-cost abstraction over PyTorch C++ API
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Your Rust App  │  Fast, type-safe ML inference
│                 │  Production-ready deployment
└─────────────────┘
```

### Key Components

| Component | Purpose | Language |
|-----------|---------|----------|
| **HuggingFace Transformers** | Source of pretrained models | Python |
| **PyTorch** | ML framework for training and inference | Python/C++ |
| **TorchScript JIT** | Serialization format for PyTorch models | N/A |
| **libtorch** | PyTorch C++ library | C++ |
| **tch-rs** | Rust bindings to libtorch | Rust |

### Why This Approach?

**Advantages:**
- ✅ Native PyTorch performance (no conversion overhead)
- ✅ GPU acceleration (CUDA, MPS, ROCm)
- ✅ Type safety from Rust
- ✅ Memory safety guarantees
- ✅ Production-ready deployment

**Trade-offs:**
- ⚠️ Requires libtorch installation (~2GB)
- ⚠️ Two-step process (export in Python, use in Rust)
- ⚠️ Less flexible than pure Python for experimentation

---

## Prerequisites

### System Requirements

- **Rust:** 1.70+ (2021 edition)
- **Python:** 3.9+ (for model export only)
- **Memory:** 4GB+ RAM (8GB+ recommended)
- **Disk:** 5GB+ free space (for libtorch + models)

### Software Dependencies

#### Python Side (for export only)

```bash
pip install torch transformers tokenizers
```

#### Rust Side (for inference)

Add to your `Cargo.toml`:

```toml
[dependencies]
tch = "0.19"              # PyTorch bindings
anyhow = "1.0"            # Error handling
tokenizers = "0.15"       # HuggingFace tokenizers
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
```

### Installing libtorch

#### Option 1: Use PyTorch from conda/pip (Recommended for macOS)

```bash
# Install PyTorch with MPS support (macOS Apple Silicon)
conda install pytorch torchvision torchaudio -c pytorch

# Set environment variables
export LIBTORCH="$(python3 -c 'import torch; print(torch.__path__[0])')"
export DYLD_LIBRARY_PATH="$LIBTORCH/lib:$DYLD_LIBRARY_PATH"  # macOS
export LD_LIBRARY_PATH="$LIBTORCH/lib:$LD_LIBRARY_PATH"      # Linux
```

#### Option 2: Download Official libtorch (Linux/Windows)

```bash
# For Linux with CUDA 12.1
wget https://download.pytorch.org/libtorch/cu121/libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcu121.zip
unzip libtorch-cxx11-abi-shared-with-deps-2.1.0+cu121.zip
export LIBTORCH=$PWD/libtorch
export LD_LIBRARY_PATH=$LIBTORCH/lib:$LD_LIBRARY_PATH
```

**Verify installation:**

```bash
echo $LIBTORCH
# Should print a valid path

cargo build
# Should compile without "libtorch not found" errors
```

---

## Quick Start

### 5-Minute Example

**Step 1: Export a model (Python)**

```python
# export_model.py
import torch
from transformers import AutoModel, AutoTokenizer

model_name = "bert-base-uncased"

# Load model
model = AutoModel.from_pretrained(model_name)
model.eval()

# Create example input
tokenizer = AutoTokenizer.from_pretrained(model_name)
example_text = "Hello, world!"
inputs = tokenizer(example_text, return_tensors="pt")

# Trace and save
with torch.no_grad():
    traced_model = torch.jit.trace(model, (inputs["input_ids"],))
    traced_model.save("bert_base.pt")

# Save tokenizer
tokenizer.save_pretrained("./")

print("✓ Model exported to bert_base.pt")
```

**Step 2: Load in Rust**

```rust
// main.rs
use tch::{CModule, Device, Tensor};
use anyhow::Result;

fn main() -> Result<()> {
    // Load model
    let device = Device::cuda_if_available();
    let model = CModule::load_on_device("bert_base.pt", device)?;

    // Create input (dummy token IDs)
    let input_ids = Tensor::of_slice(&[101i64, 7592, 1010, 2088, 102])
        .unsqueeze(0)  // Add batch dimension
        .to_device(device);

    // Run inference
    let output = model.forward_ts(&[input_ids])?;
    println!("Output shape: {:?}", output.size());

    Ok(())
}
```

**Step 3: Run**

```bash
cargo run
# Output: Output shape: [1, 5, 768]  # (batch, seq_len, hidden_size)
```

---

## Model Export Process

### Understanding TorchScript

TorchScript is PyTorch's serialization format that allows models to run without Python:

- **Tracing:** Records operations during a forward pass (recommended for inference)
- **Scripting:** Converts Python code to TorchScript (needed for control flow)

### Export Script Template

```python
#!/usr/bin/env python3
"""
Generic HuggingFace model export script for tch-rs
"""
import torch
from transformers import AutoModel, AutoTokenizer
import argparse
from pathlib import Path

def export_model(
    model_name: str,
    output_path: str,
    device: str = "cpu",
    optimize: bool = True,
    sequence_length: int = 512
):
    """Export HuggingFace model to TorchScript JIT format."""

    print(f"📦 Loading model: {model_name}")

    # Load model and tokenizer
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Move to target device
    device_obj = torch.device(device)
    model = model.to(device_obj)
    model.eval()

    print(f"🎯 Target device: {device}")

    # Create example inputs with realistic shapes
    example_text = "This is an example sentence for model tracing."
    inputs = tokenizer(
        example_text,
        return_tensors="pt",
        padding="max_length",
        max_length=sequence_length,
        truncation=True
    ).to(device_obj)

    print(f"📊 Input shape: {inputs['input_ids'].shape}")

    # Trace the model
    print("🔍 Tracing model...")
    with torch.no_grad():
        # For models that return dict/tuple, extract the main output
        outputs = model(**inputs)

        # Trace with proper forward signature
        if hasattr(outputs, 'last_hidden_state'):
            # Standard transformer output
            traced_model = torch.jit.trace(
                model,
                (inputs["input_ids"],),
                strict=False  # Allow some flexibility
            )
        else:
            # Custom output structure
            traced_model = torch.jit.trace(
                model,
                example_inputs=(inputs["input_ids"],)
            )

    # Optimize for inference
    if optimize:
        print("⚡ Optimizing for inference...")
        traced_model = torch.jit.optimize_for_inference(traced_model)

    # Save model
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    traced_model.save(str(output_file))

    print(f"✅ Model saved to: {output_file}")
    print(f"📦 File size: {output_file.stat().st_size / 1024 / 1024:.1f} MB")

    # Save tokenizer
    tokenizer_dir = output_file.parent
    tokenizer.save_pretrained(tokenizer_dir)
    print(f"✅ Tokenizer saved to: {tokenizer_dir}")

    # Verify model loads
    print("🔬 Verifying model loads...")
    loaded_model = torch.jit.load(str(output_file))
    test_output = loaded_model(inputs["input_ids"])
    print(f"✅ Verification successful! Output shape: {test_output[0].shape}")

    return {
        "model_path": str(output_file),
        "tokenizer_path": str(tokenizer_dir),
        "hidden_size": test_output[0].shape[-1],
        "device": device
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export HuggingFace model to TorchScript")
    parser.add_argument("--model", required=True, help="HuggingFace model name or path")
    parser.add_argument("--output", required=True, help="Output .pt file path")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--no-optimize", action="store_true", help="Skip optimization")
    parser.add_argument("--seq-length", type=int, default=512, help="Max sequence length")

    args = parser.parse_args()

    metadata = export_model(
        model_name=args.model,
        output_path=args.output,
        device=args.device,
        optimize=not args.no_optimize,
        sequence_length=args.seq_length
    )

    print("\n📋 Export Summary:")
    for key, value in metadata.items():
        print(f"  {key}: {value}")
```

### Export Examples

```bash
# Classification model (BERT)
python export_model.py \
  --model bert-base-uncased \
  --output models/bert_base_cpu.pt \
  --device cpu

# Sentence embeddings model
python export_model.py \
  --model sentence-transformers/all-MiniLM-L6-v2 \
  --output models/minilm_mps.pt \
  --device mps

# Large model with custom sequence length
python export_model.py \
  --model roberta-large \
  --output models/roberta_large_cuda.pt \
  --device cuda \
  --seq-length 256
```

### Export Best Practices

1. **Device-Specific Exports:**
   - Export separate models for CPU, CUDA, and MPS
   - Device-specific optimizations improve performance

2. **Input Shapes:**
   - Use realistic input shapes during tracing
   - Consider your production use case

3. **Optimization:**
   - Always use `optimize_for_inference()` for production
   - Can improve latency by 10-30%

4. **Verification:**
   - Always verify the exported model loads and produces correct output
   - Compare outputs between Python and exported model

---

## Loading Models in Rust

### Basic Loading

```rust
use tch::{CModule, Device};
use anyhow::Result;

pub struct ModelLoader {
    device: Device,
}

impl ModelLoader {
    pub fn new() -> Self {
        // Auto-detect best device: CUDA > MPS > CPU
        let device = if tch::Cuda::is_available() {
            Device::Cuda(0)
        } else if tch::utils::has_mps() {
            Device::Mps
        } else {
            Device::Cpu
        };

        println!("Using device: {:?}", device);
        Self { device }
    }

    pub fn load_model(&self, path: &str) -> Result<CModule> {
        let model = CModule::load_on_device(path, self.device)?;
        Ok(model)
    }
}
```

### Advanced Loading with Metadata

```rust
use std::path::Path;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct ModelMetadata {
    pub model_name: String,
    pub hidden_size: usize,
    pub max_length: usize,
    pub export_device: String,
    pub export_date: String,
}

pub struct Model {
    module: CModule,
    metadata: ModelMetadata,
    device: Device,
}

impl Model {
    pub fn load(model_path: &str) -> Result<Self> {
        // Load metadata
        let metadata_path = Path::new(model_path)
            .with_extension("json");

        let metadata: ModelMetadata = if metadata_path.exists() {
            let json = std::fs::read_to_string(metadata_path)?;
            serde_json::from_str(&json)?
        } else {
            // Default metadata
            ModelMetadata {
                model_name: "unknown".to_string(),
                hidden_size: 768,
                max_length: 512,
                export_device: "cpu".to_string(),
                export_date: "unknown".to_string(),
            }
        };

        // Auto-detect device
        let device = Device::cuda_if_available();

        // Load model on device
        let module = CModule::load_on_device(model_path, device)?;

        Ok(Self {
            module,
            metadata,
            device,
        })
    }

    pub fn device(&self) -> Device {
        self.device
    }

    pub fn hidden_size(&self) -> usize {
        self.metadata.hidden_size
    }
}
```

### Model Caching and Lazy Loading

```rust
use std::sync::{Arc, Mutex};
use once_cell::sync::OnceCell;

pub struct LazyModel {
    model: Arc<OnceCell<CModule>>,
    model_path: String,
    device: Device,
}

impl LazyModel {
    pub fn new(model_path: String) -> Self {
        Self {
            model: Arc::new(OnceCell::new()),
            model_path,
            device: Device::cuda_if_available(),
        }
    }

    fn get_or_load(&self) -> Result<&CModule> {
        self.model.get_or_try_init(|| {
            println!("Loading model from: {}", self.model_path);
            CModule::load_on_device(&self.model_path, self.device)
        })
    }

    pub fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let model = self.get_or_load()?;
        Ok(model.forward_ts(&[input_ids.shallow_clone()])?)
    }
}
```

---

## Tokenization

### Using rust-tokenizers

```rust
use tokenizers::tokenizer::{Tokenizer, EncodeInput};
use anyhow::Result;

pub struct TokenizerWrapper {
    tokenizer: Tokenizer,
    max_length: usize,
}

impl TokenizerWrapper {
    pub fn from_file(path: &str) -> Result<Self> {
        let tokenizer = Tokenizer::from_file(path)?;
        Ok(Self {
            tokenizer,
            max_length: 512,
        })
    }

    pub fn encode(&self, text: &str) -> Result<Vec<i64>> {
        let encoding = self.tokenizer.encode(text, false)?;
        let ids = encoding.get_ids()
            .iter()
            .map(|&id| id as i64)
            .collect();
        Ok(ids)
    }

    pub fn encode_batch(&self, texts: &[&str]) -> Result<Vec<Vec<i64>>> {
        let inputs: Vec<EncodeInput> = texts
            .iter()
            .map(|&text| text.into())
            .collect();

        let encodings = self.tokenizer.encode_batch(inputs, false)?;

        let ids = encodings
            .iter()
            .map(|enc| {
                enc.get_ids()
                    .iter()
                    .map(|&id| id as i64)
                    .collect()
            })
            .collect();

        Ok(ids)
    }

    pub fn decode(&self, ids: &[i64]) -> Result<String> {
        let ids_u32: Vec<u32> = ids.iter().map(|&id| id as u32).collect();
        let text = self.tokenizer.decode(&ids_u32, true)?;
        Ok(text)
    }
}
```

### Creating Tensors from Token IDs

```rust
use tch::Tensor;

fn create_input_tensor(token_ids: &[i64], device: Device) -> Tensor {
    // Create tensor from token IDs
    let tensor = Tensor::of_slice(token_ids)
        .unsqueeze(0)  // Add batch dimension: [seq_len] -> [1, seq_len]
        .to_device(device);

    tensor
}

fn create_batch_tensor(batch_ids: &[Vec<i64>], device: Device) -> Tensor {
    // Find max length for padding
    let max_len = batch_ids.iter().map(|ids| ids.len()).max().unwrap_or(0);

    // Pad all sequences to max_len
    let mut padded: Vec<i64> = Vec::with_capacity(batch_ids.len() * max_len);
    for ids in batch_ids {
        padded.extend_from_slice(ids);
        // Pad with 0s (or your pad token ID)
        padded.extend(std::iter::repeat(0).take(max_len - ids.len()));
    }

    // Create tensor [batch_size, max_len]
    Tensor::of_slice(&padded)
        .view([batch_ids.len() as i64, max_len as i64])
        .to_device(device)
}
```

---

## Running Inference

### Single Inference

```rust
use tch::{CModule, Device, Tensor, no_grad};
use anyhow::Result;

pub struct InferenceEngine {
    model: CModule,
    tokenizer: TokenizerWrapper,
    device: Device,
}

impl InferenceEngine {
    pub fn new(model_path: &str, tokenizer_path: &str) -> Result<Self> {
        let device = Device::cuda_if_available();
        let model = CModule::load_on_device(model_path, device)?;
        let tokenizer = TokenizerWrapper::from_file(tokenizer_path)?;

        Ok(Self {
            model,
            tokenizer,
            device,
        })
    }

    pub fn infer(&self, text: &str) -> Result<Vec<f32>> {
        // Tokenize
        let token_ids = self.tokenizer.encode(text)?;

        // Create tensor
        let input_ids = Tensor::of_slice(&token_ids)
            .unsqueeze(0)
            .to_device(self.device);

        // Run inference with no_grad
        let output = no_grad(|| {
            self.model.forward_ts(&[input_ids])
        })?;

        // Extract results (assuming classification output)
        let logits: Vec<f32> = output.squeeze().try_into()?;

        Ok(logits)
    }
}
```

### Batch Inference

```rust
impl InferenceEngine {
    pub fn infer_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        // Tokenize all texts
        let batch_ids = self.tokenizer.encode_batch(texts)?;

        // Create padded batch tensor
        let input_tensor = self.create_batch_tensor(&batch_ids);

        // Run inference
        let output = no_grad(|| {
            self.model.forward_ts(&[input_tensor])
        })?;

        // Convert to Vec<Vec<f32>>
        let batch_size = texts.len() as i64;
        let mut results = Vec::with_capacity(batch_size as usize);

        for i in 0..batch_size {
            let logits: Vec<f32> = output.get(i).try_into()?;
            results.push(logits);
        }

        Ok(results)
    }

    fn create_batch_tensor(&self, batch_ids: &[Vec<i64>]) -> Tensor {
        let max_len = batch_ids.iter().map(|ids| ids.len()).max().unwrap_or(0);
        let batch_size = batch_ids.len();

        let mut padded = vec![0i64; batch_size * max_len];
        for (i, ids) in batch_ids.iter().enumerate() {
            let start = i * max_len;
            padded[start..start + ids.len()].copy_from_slice(ids);
        }

        Tensor::of_slice(&padded)
            .view([batch_size as i64, max_len as i64])
            .to_device(self.device)
    }
}
```

---

## Common Patterns

### Pattern 1: Classification Task

```rust
pub struct TextClassifier {
    engine: InferenceEngine,
    labels: Vec<String>,
}

impl TextClassifier {
    pub fn classify(&self, text: &str) -> Result<(String, f32)> {
        let logits = self.engine.infer(text)?;

        // Apply softmax
        let probs = softmax(&logits);

        // Find max probability
        let (max_idx, max_prob) = probs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap();

        let label = self.labels[max_idx].clone();
        Ok((label, *max_prob))
    }
}

fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp: Vec<f32> = logits.iter().map(|x| (x - max).exp()).collect();
    let sum: f32 = exp.iter().sum();
    exp.iter().map(|x| x / sum).collect()
}
```

### Pattern 2: Sentence Embeddings

```rust
pub struct SentenceEncoder {
    engine: InferenceEngine,
}

impl SentenceEncoder {
    pub fn encode(&self, text: &str) -> Result<Vec<f32>> {
        let token_ids = self.engine.tokenizer.encode(text)?;
        let input_ids = Tensor::of_slice(&token_ids)
            .unsqueeze(0)
            .to_device(self.engine.device);

        // Get hidden states
        let output = no_grad(|| {
            self.engine.model.forward_ts(&[input_ids])
        })?;

        // Mean pooling over sequence dimension
        let embeddings = output.mean_dim(&[1], false, tch::Kind::Float);

        let embedding_vec: Vec<f32> = embeddings.squeeze().try_into()?;
        Ok(embedding_vec)
    }

    pub fn similarity(&self, text1: &str, text2: &str) -> Result<f32> {
        let emb1 = self.encode(text1)?;
        let emb2 = self.encode(text2)?;

        Ok(cosine_similarity(&emb1, &emb2))
    }
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (norm_a * norm_b)
}
```

### Pattern 3: Streaming Inference

```rust
use std::sync::mpsc::{channel, Sender, Receiver};
use std::thread;

pub struct StreamingInference {
    sender: Sender<String>,
}

impl StreamingInference {
    pub fn new(model_path: String, tokenizer_path: String) -> Result<Self> {
        let (tx, rx): (Sender<String>, Receiver<String>) = channel();

        thread::spawn(move || {
            let engine = InferenceEngine::new(&model_path, &tokenizer_path)
                .expect("Failed to load model");

            for text in rx.iter() {
                match engine.infer(&text) {
                    Ok(result) => println!("Result: {:?}", result),
                    Err(e) => eprintln!("Error: {}", e),
                }
            }
        });

        Ok(Self { sender: tx })
    }

    pub fn submit(&self, text: String) -> Result<()> {
        self.sender.send(text)?;
        Ok(())
    }
}
```

---

## Performance Optimization

### 1. Model Warmup

```rust
impl InferenceEngine {
    pub fn warmup(&self) -> Result<()> {
        println!("Warming up model...");

        // Run a few dummy inferences
        for _ in 0..3 {
            let _ = self.infer("warmup text")?;
        }

        println!("Warmup complete");
        Ok(())
    }
}
```

### 2. Thread-Safe Concurrent Inference

```rust
use std::sync::{Arc, Mutex};
use parking_lot::Mutex as ParkingLotMutex;

pub struct ConcurrentInferenceEngine {
    model: Arc<ParkingLotMutex<CModule>>,
    tokenizer: Arc<TokenizerWrapper>,
    device: Device,
}

impl ConcurrentInferenceEngine {
    pub fn infer(&self, text: &str) -> Result<Vec<f32>> {
        let token_ids = self.tokenizer.encode(text)?;
        let input_ids = Tensor::of_slice(&token_ids)
            .unsqueeze(0)
            .to_device(self.device);

        // Lock only during forward pass
        let output = {
            let model = self.model.lock();
            no_grad(|| model.forward_ts(&[input_ids]))?
        };

        let logits: Vec<f32> = output.squeeze().try_into()?;
        Ok(logits)
    }
}
```

### 3. Batch Size Tuning

```rust
pub struct OptimalBatchInference {
    engine: InferenceEngine,
    optimal_batch_size: usize,
}

impl OptimalBatchInference {
    pub fn new(engine: InferenceEngine) -> Self {
        // Auto-detect optimal batch size based on device
        let optimal_batch_size = match engine.device {
            Device::Cuda(_) => 32,
            Device::Mps => 8,
            Device::Cpu => 4,
        };

        Self {
            engine,
            optimal_batch_size,
        }
    }

    pub fn infer_many(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let mut results = Vec::with_capacity(texts.len());

        // Process in optimal batches
        for chunk in texts.chunks(self.optimal_batch_size) {
            let batch_results = self.engine.infer_batch(chunk)?;
            results.extend(batch_results);
        }

        Ok(results)
    }
}
```

### 4. Memory Optimization

```rust
impl InferenceEngine {
    pub fn infer_lowmem(&self, text: &str) -> Result<Vec<f32>> {
        // Explicitly scope tensors to free memory quickly
        let output = {
            let token_ids = self.tokenizer.encode(text)?;
            let input_ids = Tensor::of_slice(&token_ids)
                .unsqueeze(0)
                .to_device(self.device);

            no_grad(|| self.model.forward_ts(&[input_ids]))?
        };
        // input_ids dropped here

        let logits: Vec<f32> = output.squeeze().try_into()?;
        Ok(logits)
    }
}
```

---

## Troubleshooting

### Common Issues

#### 1. "libtorch not found"

**Symptom:**
```
error: could not find native static library `torch`
```

**Solution:**
```bash
# Verify LIBTORCH is set
echo $LIBTORCH

# If not set
export LIBTORCH="/path/to/torch"
export DYLD_LIBRARY_PATH="$LIBTORCH/lib:$DYLD_LIBRARY_PATH"  # macOS
export LD_LIBRARY_PATH="$LIBTORCH/lib:$LD_LIBRARY_PATH"      # Linux
```

#### 2. Model Loading Fails

**Symptom:**
```
Error: Failed to load model: invalid model file
```

**Solutions:**
- Verify model was exported correctly (check file is not empty)
- Ensure model was exported for the correct device
- Try re-exporting with `strict=False` in torch.jit.trace()

#### 3. Shape Mismatch Errors

**Symptom:**
```
RuntimeError: shape '[1, 512]' is invalid for input of size 256
```

**Solutions:**
- Check tokenizer output length matches model's expected input
- Ensure padding/truncation is applied correctly
- Verify batch dimension is added (use `.unsqueeze(0)`)

#### 4. Slow Performance

**Diagnostic steps:**
```rust
use std::time::Instant;

let start = Instant::now();
let result = engine.infer(text)?;
println!("Inference took: {:?}", start.elapsed());

// Check device
println!("Using device: {:?}", engine.device);
```

**Common causes:**
- Running on CPU instead of GPU
- Not using `no_grad()` scope
- Model not optimized during export
- Inefficient tokenization

#### 5. Out of Memory (OOM)

**Solutions:**
- Reduce batch size
- Use gradient checkpointing (if available)
- Process data in smaller chunks
- Ensure `no_grad()` is used
- Free tensors explicitly by limiting scope

---

## Complete Examples

### Example 1: BERT Text Classification

```rust
use tch::{CModule, Device, Tensor, no_grad};
use tokenizers::Tokenizer;
use anyhow::Result;

fn main() -> Result<()> {
    // Load model and tokenizer
    let device = Device::cuda_if_available();
    let model = CModule::load_on_device("bert_classifier.pt", device)?;
    let tokenizer = Tokenizer::from_file("tokenizer.json")?;

    // Input text
    let text = "This movie was absolutely fantastic!";

    // Tokenize
    let encoding = tokenizer.encode(text, false)?;
    let input_ids: Vec<i64> = encoding.get_ids().iter().map(|&id| id as i64).collect();

    // Create tensor
    let input_tensor = Tensor::of_slice(&input_ids)
        .unsqueeze(0)
        .to_device(device);

    // Inference
    let logits = no_grad(|| {
        model.forward_ts(&[input_tensor])
    })?;

    // Get prediction
    let probs = logits.softmax(-1, tch::Kind::Float);
    let prediction = probs.argmax(-1, false);

    println!("Prediction: {:?}", i64::from(prediction));
    println!("Probabilities: {:?}", Vec::<f32>::from(probs.squeeze()));

    Ok(())
}
```

### Example 2: Sentence Similarity

```rust
fn main() -> Result<()> {
    let device = Device::cuda_if_available();
    let model = CModule::load_on_device("sentence_transformer.pt", device)?;
    let tokenizer = Tokenizer::from_file("tokenizer.json")?;

    let sentences = [
        "The cat sits on the mat",
        "A feline rests on a rug",
        "Python is a programming language",
    ];

    // Encode all sentences
    let mut embeddings = Vec::new();
    for sentence in &sentences {
        let encoding = tokenizer.encode(sentence, false)?;
        let ids: Vec<i64> = encoding.get_ids().iter().map(|&id| id as i64).collect();
        let input = Tensor::of_slice(&ids).unsqueeze(0).to_device(device);

        let output = no_grad(|| model.forward_ts(&[input]))?;

        // Mean pooling
        let embedding = output.mean_dim(&[1], false, tch::Kind::Float);
        embeddings.push(embedding);
    }

    // Compute similarities
    for i in 0..sentences.len() {
        for j in (i + 1)..sentences.len() {
            let sim = cosine_similarity_tensor(&embeddings[i], &embeddings[j]);
            println!(
                "Similarity('{}', '{}'): {:.4}",
                sentences[i], sentences[j], sim
            );
        }
    }

    Ok(())
}

fn cosine_similarity_tensor(a: &Tensor, b: &Tensor) -> f32 {
    let dot = (a * b).sum(tch::Kind::Float);
    let norm_a = a.norm();
    let norm_b = b.norm();
    f32::from(dot / (norm_a * norm_b))
}
```

### Example 3: Production-Ready Service

```rust
use axum::{
    routing::post,
    Router, Json,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::Mutex;

#[derive(Deserialize)]
struct InferenceRequest {
    text: String,
}

#[derive(Serialize)]
struct InferenceResponse {
    logits: Vec<f32>,
    inference_time_ms: u64,
}

struct AppState {
    engine: Arc<Mutex<InferenceEngine>>,
}

#[tokio::main]
async fn main() {
    // Initialize model
    let engine = InferenceEngine::new("model.pt", "tokenizer.json")
        .expect("Failed to load model");

    // Warmup
    engine.warmup().expect("Warmup failed");

    let state = Arc::new(AppState {
        engine: Arc::new(Mutex::new(engine)),
    });

    // Build router
    let app = Router::new()
        .route("/infer", post(infer_handler))
        .with_state(state);

    // Run server
    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000")
        .await
        .unwrap();

    println!("Server running on http://0.0.0.0:3000");
    axum::serve(listener, app).await.unwrap();
}

async fn infer_handler(
    axum::extract::State(state): axum::extract::State<Arc<AppState>>,
    Json(payload): Json<InferenceRequest>,
) -> Json<InferenceResponse> {
    let start = std::time::Instant::now();

    let engine = state.engine.lock().await;
    let logits = engine.infer(&payload.text)
        .expect("Inference failed");

    let inference_time_ms = start.elapsed().as_millis() as u64;

    Json(InferenceResponse {
        logits,
        inference_time_ms,
    })
}
```

---

## Additional Resources

### Documentation
- [tch-rs GitHub](https://github.com/LaurentMazare/tch-rs)
- [tch-rs API Docs](https://docs.rs/tch/)
- [PyTorch C++ API](https://pytorch.org/cppdocs/)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers/)

### Example Projects
- [tch-rs Examples](https://github.com/LaurentMazare/tch-rs/tree/master/examples)
- [Rust BERT](https://github.com/guillaume-be/rust-bert)

### Community
- [tch-rs Discussions](https://github.com/LaurentMazare/tch-rs/discussions)
- [Rust ML Discord](https://discord.gg/rust-ml)

---

## Summary

You now know how to:
- ✅ Export HuggingFace models to TorchScript format
- ✅ Set up tch-rs and libtorch in Rust projects
- ✅ Load and run models with GPU acceleration
- ✅ Handle tokenization and preprocessing
- ✅ Implement common ML patterns (classification, embeddings)
- ✅ Optimize for production performance
- ✅ Troubleshoot common issues

### Next Steps

1. **Experiment:** Try exporting and running different HuggingFace models
2. **Optimize:** Profile your inference pipeline and apply optimizations
3. **Deploy:** Build a production service using the patterns shown
4. **Contribute:** Share your findings with the tch-rs community

---

**Happy coding! 🦀🔥**
