# Development Guide

This document provides guidelines for developing and maintaining the llamafirewall-rs library.

## Prerequisites

- Rust 1.75 or later
- Cargo (comes with Rust)
- (Optional) CUDA toolkit for GPU acceleration

## Setup

### Installing Rust

If you haven't already, install Rust using rustup:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

### Installing Development Tools

Install essential Rust development tools:

```bash
# Code formatting
rustup component add rustfmt

# Linting
rustup component add clippy

# Optional but recommended tools
cargo install cargo-watch      # Auto-rebuild on file changes
cargo install cargo-expand     # Expand macros
cargo install cargo-tarpaulin  # Code coverage
cargo install cargo-audit      # Security audits
cargo install flamegraph       # Performance profiling
```

## Development Workflow

### Building the Project

```bash
# Build all crates
cd rust/
cargo build

# Build with optimizations
cargo build --release

# Build specific crate
cargo build -p llamafirewall-core
```

### Running Tests

```bash
# Run all tests
cargo test

# Run tests for specific crate
cargo test -p llamafirewall-core

# Run tests with output
cargo test -- --nocapture

# Run specific test
cargo test test_message_creation

# Run tests with all features
cargo test --all-features
```

### Code Quality Checks

Before each commit, run these checks:

```bash
# Format code
cargo fmt

# Check formatting (CI mode)
cargo fmt --check

# Run linter
cargo clippy -- -D warnings

# Check for unused dependencies
cargo +nightly udeps
```

### Running Benchmarks

```bash
# Run all benchmarks
cargo bench

# Run specific benchmark
cargo bench --bench firewall_benchmark

# Generate flamegraph for profiling
cargo flamegraph --bench firewall_benchmark
```

### Documentation

```bash
# Build documentation
cargo doc --no-deps

# Build and open documentation
cargo doc --no-deps --open

# Check documentation links
cargo doc --no-deps 2>&1 | grep warning
```

## Pre-Commit Checklist

Before committing code, ensure:

- [ ] `cargo fmt` - Code is formatted
- [ ] `cargo clippy -- -D warnings` - No clippy warnings
- [ ] `cargo test --all-features` - All tests pass
- [ ] `cargo doc --no-deps` - Documentation builds
- [ ] Update `CHANGELOG.md` if applicable
- [ ] Update tests for new functionality

## Pre-Release Checklist

Before releasing a new version:

- [ ] `cargo test --all-features` - All tests pass
- [ ] `cargo clippy -- -D warnings` - No warnings
- [ ] `cargo bench` - Benchmarks run successfully
- [ ] `cargo audit` - No security vulnerabilities
- [ ] `cargo doc --no-deps` - Documentation builds
- [ ] Update version in `Cargo.toml`
- [ ] Update `CHANGELOG.md` with release notes
- [ ] Update `README.md` if needed
- [ ] Tag release in git: `git tag -a v0.x.0 -m "Release v0.x.0"`

## Code Style Guidelines

### Rust Style

Follow the [Rust Style Guide](https://doc.rust-lang.org/nightly/style-guide/):

- Use `rustfmt` for automatic formatting
- Maximum line length: 100 characters
- Use descriptive variable names
- Prefer explicit types when clarity is improved
- Document all public APIs

### Documentation

- All public items must have doc comments
- Use examples in doc comments when helpful
- Use `///` for item documentation
- Use `//!` for module-level documentation
- Include code examples that compile (or mark as `ignore` if they don't)

Example:
```rust
/// Scans a message for security threats
///
/// # Arguments
/// * `message` - The message to scan
/// * `trace` - Optional conversation history
///
/// # Returns
/// `ScanResult` indicating whether to allow or block
///
/// # Example
/// ```rust,ignore
/// let result = firewall.scan(&message, None).await;
/// ```
pub async fn scan(&self, message: &Message, trace: Option<&Trace>) -> ScanResult {
    // Implementation
}
```

### Error Handling

- Use `Result<T, E>` for fallible operations
- Use `thiserror` for custom error types
- Fail gracefully - scanners should return `ScanResult::error()` rather than panic
- Log errors using `tracing` macros

### Testing

- Write unit tests for all public APIs
- Use `#[cfg(test)]` modules
- Test both success and failure cases
- Use `#[ignore]` for tests requiring external resources
- Use `proptest` for property-based testing when applicable

## Performance Guidelines

- Profile before optimizing
- Use `criterion` for benchmarks
- Prefer zero-copy operations with slices
- Use `Arc` for shared ownership, avoid `clone()` when possible
- Use SIMD operations when beneficial
- Consider async vs blocking carefully

## Debugging

### Viewing Expanded Macros

```bash
cargo expand --lib
cargo expand -p llamafirewall-core types
```

### Tracing and Logging

Enable trace logging during development:

```bash
RUST_LOG=debug cargo test
RUST_LOG=trace cargo run --example basic
```

### Using lldb/gdb

```bash
# Build with debug symbols
cargo build

# Run with debugger
rust-lldb target/debug/llamafirewall
rust-gdb target/debug/llamafirewall
```

## Continuous Integration

While CI/CD setup is not included in this repository, here's a recommended GitHub Actions workflow:

```yaml
name: CI

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions-rs/toolchain@v1
        with:
          toolchain: stable
      - run: cargo fmt --check
      - run: cargo clippy -- -D warnings
      - run: cargo test --all-features
      - run: cargo bench --no-fail-fast
```

## Common Issues

### Build Failures

1. **Missing dependencies**: Run `cargo clean && cargo build`
2. **Outdated lockfile**: Run `cargo update`
3. **Incompatible features**: Check feature flags in `Cargo.toml`

### Test Failures

1. **Async runtime issues**: Ensure tests use `#[tokio::test]`
2. **Flaky tests**: Look for race conditions, add proper synchronization
3. **Resource cleanup**: Use `tempfile` for temporary files in tests

## Getting Help

- Check the [Rust Book](https://doc.rust-lang.org/book/)
- Read [Rust by Example](https://doc.rust-lang.org/rust-by-example/)
- Visit the [Rust Users Forum](https://users.rust-lang.org/)
- Join the [Rust Discord](https://discord.gg/rust-lang)

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Follow the code style guidelines
4. Add tests for new functionality
5. Run the pre-commit checklist
6. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
