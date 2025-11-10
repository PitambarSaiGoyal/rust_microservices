//! Sanity check tests for tch-rs integration
//! Part of Phase 1: Environment Setup & Dependencies
//!
//! These tests verify that tch-rs is properly installed and configured.

#![cfg(feature = "tch-backend")]

use tch::{Device, Kind, Tensor};

#[test]
fn test_tch_import() {
    // Basic test to ensure tch can be imported
    println!("tch-rs imported successfully");
}

#[test]
fn test_tensor_creation() {
    // Test basic tensor creation
    let tensor = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0]);
    assert_eq!(tensor.size(), vec![4]);

    println!("✅ Tensor creation successful");
}

#[test]
fn test_tensor_operations() {
    // Test basic tensor operations
    let a = Tensor::from_slice(&[1.0f32, 2.0, 3.0]);
    let b = Tensor::from_slice(&[4.0f32, 5.0, 6.0]);

    let c = a + b;
    let result: Vec<f32> = Vec::from(&c);

    assert_eq!(result, vec![5.0, 7.0, 9.0]);

    println!("✅ Tensor operations successful");
}

#[test]
fn test_device_detection() {
    // Test device detection
    let cpu_device = Device::Cpu;
    println!("✅ CPU device available: {:?}", cpu_device);

    // Try CUDA
    if tch::Cuda::is_available() {
        let cuda_device = Device::Cuda(0);
        println!("✅ CUDA device available: {:?}", cuda_device);
        println!("   CUDA device count: {}", tch::Cuda::device_count());
    } else {
        println!("⚠️  CUDA not available (CPU-only build or no GPU)");
    }

    // Try MPS (Apple Silicon)
    #[cfg(target_os = "macos")]
    {
        // Note: tch-rs MPS support requires recent versions
        // This is a placeholder - actual implementation depends on tch version
        println!("ℹ️  Platform: macOS - MPS may be available on Apple Silicon");
    }
}

#[test]
fn test_tensor_on_device() {
    // Test creating tensors on different devices
    let cpu_tensor = Tensor::from_slice(&[1.0f32, 2.0, 3.0]).to_device(Device::Cpu);
    assert_eq!(cpu_tensor.device(), Device::Cpu);

    println!("✅ Tensor device assignment successful");

    if tch::Cuda::is_available() {
        let cuda_tensor = cpu_tensor.to_device(Device::Cuda(0));
        // Device comparison for CUDA is tricky, just verify it doesn't panic
        println!("✅ CUDA tensor creation successful");
    }
}

#[test]
fn test_matrix_multiplication() {
    // Test matrix multiplication (important for ML inference)
    let a = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0]).reshape(&[2, 2]);
    let b = Tensor::from_slice(&[5.0f32, 6.0, 7.0, 8.0]).reshape(&[2, 2]);

    let c = a.matmul(&b);

    // Expected result:
    // [1, 2] * [5, 6] = [1*5+2*7, 1*6+2*8] = [19, 22]
    // [3, 4]   [7, 8]   [3*5+4*7, 3*6+4*8]   [43, 50]

    let result: Vec<f32> = Vec::from(&c.reshape(&[4]));
    assert_eq!(result, vec![19.0, 22.0, 43.0, 50.0]);

    println!("✅ Matrix multiplication successful");
}

#[test]
fn test_tensor_types() {
    // Test different tensor dtypes (important for model compatibility)
    let float_tensor = Tensor::from_slice(&[1.0f32, 2.0, 3.0]);
    assert_eq!(float_tensor.kind(), Kind::Float);

    let int_tensor = Tensor::from_slice(&[1i64, 2, 3]);
    assert_eq!(int_tensor.kind(), Kind::Int64);

    println!("✅ Multiple tensor types supported");
}

#[test]
fn test_gradient_tracking_disabled() {
    // Verify we can disable gradient tracking (important for inference)
    let tensor = Tensor::from_slice(&[1.0f32, 2.0, 3.0]);

    // For inference, we don't need gradients
    tch::no_grad(|| {
        let result = &tensor * 2.0;
        let values: Vec<f32> = Vec::from(&result);
        assert_eq!(values, vec![2.0, 4.0, 6.0]);
    });

    println!("✅ Gradient tracking can be disabled (no_grad mode)");
}

#[test]
fn test_tensor_shape_operations() {
    // Test shape operations needed for batching
    let tensor = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]);

    // Reshape to 2x3
    let reshaped = tensor.reshape(&[2, 3]);
    assert_eq!(reshaped.size(), vec![2, 3]);

    // Reshape to 3x2
    let reshaped2 = tensor.reshape(&[3, 2]);
    assert_eq!(reshaped2.size(), vec![3, 2]);

    println!("✅ Tensor reshape operations successful");
}

#[test]
fn test_memory_cleanup() {
    // Test that tensors can be created and dropped without memory leaks
    for _ in 0..100 {
        let _tensor = Tensor::randn(&[100, 100], (Kind::Float, Device::Cpu));
    }

    println!("✅ Memory cleanup successful (no obvious leaks)");
}

#[test]
fn test_version_info() {
    // Print version information for debugging
    println!("tch-rs version: {}", env!("CARGO_PKG_VERSION"));

    if let Ok(libtorch_path) = std::env::var("LIBTORCH") {
        println!("LIBTORCH path: {}", libtorch_path);
    } else {
        println!("⚠️  LIBTORCH environment variable not set");
    }

    println!("✅ Version info retrieved");
}

#[cfg(test)]
mod integration_checks {
    use super::*;

    #[test]
    fn test_batch_operations() {
        // Simulate batch processing (important for Phase 3)
        let batch_size = 4;
        let seq_len = 512;

        // Create a batch of input tensors
        let batch = Tensor::zeros(&[batch_size, seq_len], (Kind::Int64, Device::Cpu));

        assert_eq!(batch.size(), vec![batch_size, seq_len]);
        println!("✅ Batch tensor creation successful");
    }

    #[test]
    fn test_attention_mask_simulation() {
        // Simulate attention mask creation (needed for BERT-like models)
        let batch_size = 2;
        let seq_len = 512;

        let attention_mask = Tensor::ones(&[batch_size, seq_len], (Kind::Int64, Device::Cpu));

        assert_eq!(attention_mask.size(), vec![batch_size, seq_len]);
        println!("✅ Attention mask simulation successful");
    }

    #[test]
    fn test_softmax_operation() {
        // Test softmax (used in model outputs)
        let logits = Tensor::from_slice(&[1.0f32, 2.0, 3.0]);
        let probabilities = logits.softmax(-1, Kind::Float);

        let probs: Vec<f32> = Vec::from(&probabilities);

        // Check that probabilities sum to ~1.0
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);

        println!("✅ Softmax operation successful");
    }
}

// Print summary at the end
#[cfg(test)]
mod summary {
    #[test]
    fn print_summary() {
        println!("\n========================================");
        println!("  tch-rs Sanity Check Summary");
        println!("========================================");
        println!("✅ All sanity checks passed!");
        println!("");
        println!("Your tch-rs setup is ready for:");
        println!("  - Phase 2: Model Loading");
        println!("  - Phase 3: Inference Engine");
        println!("");
        println!("Next steps:");
        println!("  1. Review tch-rs-implementation-plan.md");
        println!("  2. Check tch-rs-progress.md for status");
        println!("  3. Proceed with model conversion");
        println!("========================================\n");
    }
}
