//! Quick test to verify MPS detection
//!
//! Run with: cargo run --release --features tch-backend --example test_mps

#[cfg(feature = "tch-backend")]
fn main() {
    use tch::{Device, Tensor};
    use llamafirewall_ml::tch_loader::TchModel;

    println!("=== MPS Detection Test ===\n");

    // Check device info
    println!("Device Information:");
    println!("{}", TchModel::device_info());
    println!();

    // Check auto-selected device
    let device = TchModel::select_device();
    println!("Auto-selected device: {:?}", device);
    println!();

    // Try creating a tensor on MPS
    if device == Device::Mps {
        println!("✓ MPS is available and selected!");
        println!("\nTesting MPS tensor operations...");

        let tensor = Tensor::randn(&[100, 100], (tch::Kind::Float, device));
        let result = tensor.matmul(&tensor);

        println!("✓ Matrix multiplication on MPS successful!");
        println!("  Result shape: {:?}", result.size());
    } else {
        println!("⚠ MPS not selected. Device: {:?}", device);
    }
}

#[cfg(not(feature = "tch-backend"))]
fn main() {
    println!("tch-backend feature not enabled");
}
