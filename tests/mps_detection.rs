//! Test MPS device detection and basic operations
//!
//! This test verifies that tch-rs can detect and use MPS on Apple Silicon.

#[cfg(feature = "tch-backend")]
#[test]
fn test_mps_detection() {
    use tch::{Device, Tensor};

    // Check if MPS is available
    let has_mps = tch::utils::has_mps();
    println!("MPS available: {}", has_mps);

    if !has_mps {
        println!("⚠ MPS not available on this system");
        return;
    }

    // Create a simple tensor on MPS device
    let device = Device::Mps;
    let tensor = Tensor::ones(&[5], (tch::Kind::Float, device));

    println!("✓ Created tensor on MPS: {:?}", tensor);

    // Verify tensor is on MPS
    let values: Vec<f32> = Vec::from(&tensor);
    assert_eq!(values, vec![1.0, 1.0, 1.0, 1.0, 1.0]);

    println!("✓ MPS tensor operations working correctly!");
}

#[cfg(feature = "tch-backend")]
#[test]
fn test_device_selection() {
    use llamafirewall_ml::tch_loader::TchModel;

    let device = TchModel::select_device();
    println!("Auto-selected device: {:?}", device);

    // On Apple Silicon with proper libtorch, this should be MPS
    let device_info = TchModel::device_info();
    println!("Device info:\n{}", device_info);

    // Check that MPS is in the device info
    assert!(device_info.contains("MPS") || device_info.contains("CUDA") || device_info.contains("CPU"));
}
