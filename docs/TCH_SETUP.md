# tch-rs Setup Guide

This guide walks through setting up the tch-rs PyTorch backend for LlamaFirewall.

## Overview

The tch-rs backend provides native PyTorch inference capabilities, enabling:
- Direct PyTorch model loading (no ONNX conversion needed)
- Access to CUDA, MPS (Apple Silicon), and CPU backends
- Potential for custom CUDA kernels
- Tighter PyTorch ecosystem integration

## Prerequisites

- Rust 1.75 or later
- C++ compiler (clang/gcc)
- CMake 3.10+ (for building tch-rs)
- Python 3.8+ with PyTorch (for model preparation)

## Platform-Specific Requirements

### Linux (CUDA)
- CUDA 11.8 or later
- cuDNN 8.x
- NVIDIA GPU with compute capability 3.5+

### macOS (Apple Silicon)
- macOS 12.0+ (for MPS support)
- M1/M2/M3 processor
- Xcode Command Line Tools

### macOS (Intel)
- macOS 10.15+
- MKL (installed automatically by libtorch)

## Installation

### Step 1: Download libtorch

#### Option A: Download Pre-built Binaries (Recommended)

1. Visit https://pytorch.org/get-started/locally/
2. Select your platform configuration:
   - PyTorch Build: Stable
   - Your OS: Linux/Mac
   - Package: LibTorch
   - Language: C++/Java
   - Compute Platform: CUDA 11.8/CPU

3. Download the appropriate package:

**Linux (CUDA 11.8):**
```bash
wget https://download.pytorch.org/libtorch/cu118/libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcu118.zip
unzip libtorch-cxx11-abi-shared-with-deps-2.1.0+cu118.zip
```

**Linux (CPU):**
```bash
wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcpu.zip
unzip libtorch-cxx11-abi-shared-with-deps-2.1.0+cpu.zip
```

**macOS (CPU/MPS):**
```bash
wget https://download.pytorch.org/libtorch/cpu/libtorch-macos-2.1.0.zip
unzip libtorch-macos-2.1.0.zip
```

4. Move to a permanent location:
```bash
sudo mv libtorch /opt/libtorch
# OR keep in home directory
mv libtorch ~/libtorch
```

#### Option B: Build from Source

See [Building libtorch from source](https://github.com/pytorch/pytorch/blob/master/docs/libtorch.rst)

### Step 2: Set Environment Variables

Add to your shell configuration (~/.bashrc, ~/.zshrc, etc.):

```bash
# Path to libtorch installation
export LIBTORCH=/opt/libtorch
# OR if installed in home directory:
# export LIBTORCH=~/libtorch

# Required for dynamic library loading
export LD_LIBRARY_PATH=$LIBTORCH/lib:$LD_LIBRARY_PATH  # Linux
export DYLD_LIBRARY_PATH=$LIBTORCH/lib:$DYLD_LIBRARY_PATH  # macOS
```

Reload your shell:
```bash
source ~/.bashrc  # or ~/.zshrc
```

### Step 3: Verify Installation

Run the verification script:
```bash
cd rust
./scripts/verify-libtorch.sh
```

Expected output:
```
✅ libtorch found at: /opt/libtorch
✅ libtorch version: 2.1.0
✅ CUDA available: true (11.8)
✅ MPS available: false
✅ CPU backend available: true
```

### Step 4: Build with tch-backend

```bash
# Build with tch backend
cargo build --features tch-backend

# Run tests
cargo test --features tch-backend

# Run sanity check
cargo test --features tch-backend --test tch_sanity_check
```

## Troubleshooting

### Issue: "LIBTORCH environment variable not set"

**Solution:** Ensure you've set the LIBTORCH environment variable and reloaded your shell:
```bash
export LIBTORCH=/opt/libtorch
source ~/.bashrc
```

### Issue: "cannot find -ltorch" or linking errors

**Solution:** Verify LD_LIBRARY_PATH (Linux) or DYLD_LIBRARY_PATH (macOS) includes libtorch:
```bash
# Linux
export LD_LIBRARY_PATH=$LIBTORCH/lib:$LD_LIBRARY_PATH

# macOS
export DYLD_LIBRARY_PATH=$LIBTORCH/lib:$DYLD_LIBRARY_PATH
```

### Issue: "CUDA not available" on Linux with NVIDIA GPU

**Possible causes:**
1. Downloaded CPU version instead of CUDA version
2. CUDA driver not installed
3. CUDA version mismatch

**Solution:**
```bash
# Check CUDA installation
nvidia-smi

# Verify you downloaded the CUDA version of libtorch
ls $LIBTORCH/lib | grep cuda

# If no cuda libraries, re-download CUDA version
```

### Issue: MPS not available on Apple Silicon

**Solution:** MPS requires:
- macOS 12.0+
- Python PyTorch 1.12+ with MPS support
- Matching libtorch version

Check with:
```python
import torch
print(torch.backends.mps.is_available())
```

### Issue: Build takes very long or runs out of memory

**Solution:** Reduce parallel compilation:
```bash
cargo build --features tch-backend -j 2
```

## Version Compatibility

| tch-rs | PyTorch/libtorch | CUDA | Notes |
|--------|------------------|------|-------|
| 0.17.x | 2.1.x - 2.4.x | 11.8+ | Recommended |
| 0.16.x | 2.0.x - 2.1.x | 11.7+ | Older |

Always use matching versions of:
- tch-rs crate version
- libtorch C++ library
- Python PyTorch (for model export)

## Docker Setup

For consistent environments, use Docker:

```dockerfile
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Install libtorch
RUN wget https://download.pytorch.org/libtorch/cu118/libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcu118.zip && \\
    unzip libtorch-cxx11-abi-shared-with-deps-2.1.0+cu118.zip && \\
    mv libtorch /opt/libtorch && \\
    rm libtorch-cxx11-abi-shared-with-deps-2.1.0+cu118.zip

ENV LIBTORCH=/opt/libtorch
ENV LD_LIBRARY_PATH=/opt/libtorch/lib:$LD_LIBRARY_PATH

# Build Rust project with tch-backend
WORKDIR /app
COPY . .
RUN cargo build --release --features tch-backend
```

## Performance Tuning

### CUDA Settings

```bash
# Enable cuDNN benchmarking (finds optimal algorithms)
export CUDNN_BENCHMARK=1

# Set cuDNN deterministic mode (reproducible but slower)
export CUDNN_DETERMINISTIC=0
```

### Thread Configuration

```bash
# Limit OpenMP threads (useful for multi-process setups)
export OMP_NUM_THREADS=4

# Limit PyTorch threads
export MKL_NUM_THREADS=4
```

### Memory Management

```bash
# Set CUDA memory allocator
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

## Next Steps

After successful setup:

1. Run the sanity check tests: `cargo test --test tch_sanity_check`
2. Follow the comprehensive usage guide: `TCH_USAGE_GUIDE.md`
3. Export models from Python: See [Model Management](TCH_USAGE_GUIDE.md#model-management) section
4. Run benchmarks: `source ../mps-env.sh && cargo bench --bench tch_bench`

## Resources

- [tch-rs documentation](https://docs.rs/tch/)
- [PyTorch C++ API](https://pytorch.org/cppdocs/)
- [libtorch installation guide](https://pytorch.org/cppdocs/installing.html)
- [tch-rs examples](https://github.com/LaurentMazare/tch-rs/tree/master/examples)

## Support

For issues specific to:
- **tch-rs:** https://github.com/LaurentMazare/tch-rs/issues
- **libtorch:** https://github.com/pytorch/pytorch/issues
- **LlamaFirewall:** File an issue in the project repository

---

**Last Updated:** 2025-11-05
**Version:** 1.0
