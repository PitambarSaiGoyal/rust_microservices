fn main() {
    configure_libtorch();
}

fn configure_libtorch() {
    use std::env;
    use std::path::PathBuf;
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=LIBTORCH");
    println!("cargo:rerun-if-env-changed=LIBTORCH_INCLUDE");
    println!("cargo:rerun-if-env-changed=LIBTORCH_LIB");

    // Try to detect libtorch installation
    let libtorch_path = env::var("LIBTORCH")
        .or_else(|_| env::var("LIBTORCH_PATH"))
        .ok()
        .map(PathBuf::from);

    if let Some(path) = libtorch_path {
        println!("cargo:warning=Found LIBTORCH at: {}", path.display());

        // Verify the path exists
        if !path.exists() {
            println!("cargo:warning=LIBTORCH path does not exist: {}", path.display());
            println!("cargo:warning=Please install libtorch and set LIBTORCH environment variable");
            println!("cargo:warning=See rust/docs/TCH_SETUP.md for installation instructions");
        }
    } else {
        // Provide helpful error message if libtorch is not found
        println!("cargo:warning=LIBTORCH environment variable not set");
        println!("cargo:warning=Please install libtorch and set LIBTORCH environment variable");
        println!("cargo:warning=See rust/docs/TCH_SETUP.md for installation instructions");
        println!("cargo:warning=");
        println!("cargo:warning=Quick start:");
        println!("cargo:warning=  1. Download libtorch from https://pytorch.org/");
        println!("cargo:warning=  2. Extract to a location (e.g., ~/libtorch)");
        println!("cargo:warning=  3. Set environment variable:");
        println!("cargo:warning=     export LIBTORCH=~/libtorch");
        println!("cargo:warning=  4. Rebuild: cargo build --release");
    }
}
