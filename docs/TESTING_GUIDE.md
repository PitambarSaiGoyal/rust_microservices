# Testing Guide for LlamaFirewall

> **⚠️ NOTE:** This guide contains historical ONNX backend references. The ONNX backend has been removed as of 2025-11-05.
> The current implementation uses **PyTorch via tch-rs only**. References to `--features onnx-backend` should be ignored.

This guide covers the comprehensive testing infrastructure for LlamaFirewall's ML backend (PyTorch via tch-rs).

## Table of Contents

1. [Overview](#overview)
2. [Test Categories](#test-categories)
3. [Running Tests](#running-tests)
4. [Test Infrastructure](#test-infrastructure)
5. [Accuracy Validation](#accuracy-validation)
6. [Stress Testing](#stress-testing)
7. [Continuous Integration](#continuous-integration)

---

## Overview

LlamaFirewall uses a multi-tiered testing approach:

- **Unit Tests**: Component-level validation
- **Integration Tests**: Full pipeline testing
- **Stress Tests**: Memory leak detection and sustained load
- **Accuracy Validation**: Cross-backend consistency verification
- **Performance Benchmarks**: Latency and throughput measurement

### Test Files

```
rust/
├── tests/
│   ├── integration_suite.rs    # Integration tests
│   ├── model_loading.rs        # Model loading tests
│   ├── stress_tests.rs         # Stress and reliability tests
│   └── tch_sanity_check.rs     # tch-rs sanity checks
├── benches/
│   └── tch_bench.rs            # PyTorch inference benchmarks
└── scripts/
    ├── run-integration-tests.sh # Test orchestration script
    └── compare-backends.sh      # Backend comparison

scripts/
└── validate_accuracy.py         # Accuracy validation
```

---

## Test Categories

### 1. Unit Tests

Test individual components in isolation.

```bash
# Test PyTorch backend (default)
cargo test --lib

# Test tch-rs backend
cargo test --lib --features tch-backend

# Test all backends
cargo test --lib --all-features
```

### 2. Integration Tests

Test the full pipeline with both backends.

```bash
# Integration tests
cargo test --test integration_suite

# All integration tests
cargo test --test integration_suite

# Cross-backend consistency tests
cargo test --test integration_suite --all-features
```

**Key Test Cases:**
- Empty input handling
- Unicode and emoji support
- Varying text lengths (1 char to 500 chars)
- Batch processing
- Concurrent inference
- Error recovery

### 3. Stress Tests

Long-running tests for reliability and memory leak detection.

```bash
# Run all stress tests (takes ~5 minutes)
cargo test --test stress_tests --all-features -- --ignored --nocapture

# Individual stress tests
cargo test --test stress_tests --features onnx-backend -- --ignored test_onnx_memory_leak
cargo test --test stress_tests --features tch-backend -- --ignored test_tch_sustained_load
```

**Stress Test Types:**
- **Memory Leak Detection**: 1000 iterations with memory tracking
- **Sustained Load**: 60-second concurrent load test
- **Error Recovery**: Edge case handling validation

### 4. Model Loading Tests

Verify model loading and format compatibility.

```bash
# Test ONNX model loading
cargo test --test model_loading --features onnx-backend

# Test tch-rs model loading
cargo test --test model_loading --features tch-backend
```

### 5. Performance Benchmarks

Measure latency and throughput.

```bash
# PyTorch benchmarks
source ../mps-env.sh  # Required for MPS on macOS
cargo bench --bench tch_bench

# tch-rs benchmarks
cargo bench --bench tch_bench --features tch-backend

# Compare backends
./rust/scripts/compare-backends.sh
```

---

## Running Tests

### Quick Test Suite (Recommended for Development)

```bash
# Run fast tests only
./rust/scripts/run-integration-tests.sh --quick

# Test specific backend
./rust/scripts/run-integration-tests.sh --quick --onnx-only
./rust/scripts/run-integration-tests.sh --quick --tch-only
```

### Full Test Suite (CI/CD)

```bash
# Run all tests including stress tests
./rust/scripts/run-integration-tests.sh --full
```

### Accuracy Validation

```bash
# Validate backend consistency (1000 samples)
./rust/scripts/run-integration-tests.sh --accuracy

# Custom accuracy validation
python3 scripts/validate_accuracy.py --samples 10000 --detailed
```

### Stress Testing Only

```bash
# Run stress tests only
./rust/scripts/run-integration-tests.sh --stress
```

---

## Test Infrastructure

### Integration Test Suite (`integration_suite.rs`)

Comprehensive test cases covering:

1. **Pipeline Integration**: Full end-to-end testing
2. **Backend Consistency**: Cross-backend output comparison
3. **Batch Processing**: Dynamic batching validation
4. **Performance Regression**: Latency threshold checks
5. **Error Handling**: Invalid input handling
6. **Concurrent Inference**: Thread-safety validation
7. **Varying Input Lengths**: 0 to 500+ character inputs

**Test Data:**
- Benign prompts (greetings, questions)
- Empty and whitespace inputs
- Unicode text (Japanese, Chinese, Russian, Arabic)
- Emojis and special characters
- Long texts (100+ words)

**Example Test:**
```rust
#[test]
#[cfg(all(feature = "onnx-backend", feature = "tch-backend"))]
fn test_backend_output_consistency() {
    let onnx_engine = OnnxInferenceEngine::new(/* ... */);
    let tch_engine = TchInferenceEngine::new(/* ... */);

    for test_case in TEST_CASES {
        let onnx_result = onnx_engine.infer(test_case.input)?;
        let tch_result = tch_engine.infer(test_case.input)?;

        const TOLERANCE: f32 = 1e-3;
        assert!((onnx_result[0] - tch_result[0]).abs() < TOLERANCE);
        assert!((onnx_result[1] - tch_result[1]).abs() < TOLERANCE);
    }
}
```

---

## Accuracy Validation

### Python Validation Script

The `validate_accuracy.py` script compares ONNX and tch-rs backends across thousands of samples.

**Usage:**
```bash
# Basic validation (1000 samples)
python3 scripts/validate_accuracy.py

# Large-scale validation
python3 scripts/validate_accuracy.py --samples 10000

# Custom prompts from file
python3 scripts/validate_accuracy.py --input prompts.txt --detailed

# Save mismatches to JSON
python3 scripts/validate_accuracy.py --save-mismatches mismatches.json
```

**Metrics Reported:**
- **Match Rate**: Percentage of samples within tolerance
- **Mean Absolute Difference**: Average difference across all outputs
- **Max Absolute Difference**: Largest observed difference
- **P95/P99 Differences**: Percentile-based outlier detection
- **Statistical Bias**: Systematic difference detection

**Pass Criteria:**
- ✅ Match Rate ≥ 99%
- ✅ Mean Diff < 1e-4
- ✅ No statistical bias

---

## Stress Testing

### Memory Leak Detection

Runs 1000+ inference iterations while tracking memory usage.

```bash
cargo test --test stress_tests --features onnx-backend \
    -- --ignored test_onnx_memory_leak_detection
```

**Measured Metrics:**
- Initial memory (MB)
- Peak memory (MB)
- Final memory (MB)
- Growth percentage
- Growth rate (MB/iteration)

**Pass Criteria:**
- Memory growth < 10%

### Sustained Load Testing

60-second concurrent load test with 4 threads.

```bash
cargo test --test stress_tests --features tch-backend \
    -- --ignored test_tch_sustained_load
```

**Measured Metrics:**
- Total requests completed
- Throughput (req/s)
- Duration

**Pass Criteria:**
- ONNX: ≥ 10 req/s
- tch-rs: ≥ 15 req/s (target: 18+)

### Error Recovery

Tests that the engine can recover from edge cases.

```bash
cargo test --test stress_tests --all-features \
    -- --ignored error_recovery
```

**Edge Cases Tested:**
- Empty strings
- Null bytes
- Very long inputs (10,000 chars)
- Invalid UTF-8 sequences

---

## Continuous Integration

### GitHub Actions Workflow (Example)

```yaml
name: Integration Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v3

      - name: Install Rust
        uses: actions-rs/toolchain@v1
        with:
          toolchain: stable

      - name: Install ONNX Runtime
        run: # ... install steps

      - name: Run Quick Tests
        run: ./rust/scripts/run-integration-tests.sh --quick

      - name: Run Accuracy Validation
        run: ./rust/scripts/run-integration-tests.sh --accuracy

  stress-test:
    runs-on: ubuntu-latest

    steps:
      # ... setup steps

      - name: Run Full Test Suite
        run: ./rust/scripts/run-integration-tests.sh --full
```

### Pre-commit Hooks

```bash
# .git/hooks/pre-commit
#!/bin/bash
./rust/scripts/run-integration-tests.sh --quick
```

---

## Troubleshooting

### Issue: Tests Fail with "Model Not Found"

**Solution:** Ensure models are exported:
```bash
# Export ONNX model
python scripts/export_to_jit.py --model meta-llama/Prompt-Guard-86M \
    --output models/promptguard.onnx

# Export PyTorch model
python scripts/export_to_jit.py --model meta-llama/Prompt-Guard-86M \
    --output models/promptguard.pt
```

### Issue: tch-rs Tests Skip

**Solution:** Set LIBTORCH environment variable:
```bash
export LIBTORCH=/path/to/libtorch
export LD_LIBRARY_PATH=$LIBTORCH/lib:$LD_LIBRARY_PATH
```

### Running Benchmarks

Run tch-rs benchmarks with:
```bash
source ../mps-env.sh  # Required for MPS on macOS
cargo bench --bench tch_bench
```

### Issue: Accuracy Validation Fails

**Check:**
1. Models are from the same source
2. Tokenizer is identical for both backends
3. Model export was successful
4. Tolerance is appropriate (default: 1e-4)

---

## Test Metrics Summary

| Test Type | Duration | Pass Criteria |
|-----------|----------|---------------|
| Unit Tests | ~10s | All pass |
| Integration Tests | ~30s | All pass |
| Stress Tests | ~5min | < 10% memory growth, > 10 req/s |
| Accuracy Validation | ~1min | ≥ 99% match, < 1e-4 mean diff |
| Performance Benchmarks | ~5min | P95 < 55ms (tch-rs target) |

---

## Contributing

When adding new features:

1. Add unit tests for new components
2. Add integration tests for new inference paths
3. Update `integration_suite.rs` with edge cases
4. Run full test suite before submitting PR
5. Update this guide with new test procedures

---

**Last Updated:** 2025-11-05
**Maintained By:** LlamaFirewall Team
