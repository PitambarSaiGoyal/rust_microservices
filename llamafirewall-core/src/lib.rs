//! # llamafirewall-core
//!
//! Core types and traits for the LlamaFirewall Rust implementation.
//!
//! This crate provides the fundamental building blocks for building LLM guardrails:
//! - **Type System**: Message, Trace, ScanResult, and decision types
//! - **Scanner Trait**: Async interface for implementing custom scanners
//! - **Firewall Orchestrator**: Coordinates multiple scanners and aggregates results
//! - **Configuration**: Builder pattern for configuring firewall behavior
//!
//! ## Example
//!
//! ```rust,ignore
//! use llamafirewall_core::{Firewall, Configuration, Message, Role};
//!
//! #[tokio::main]
//! async fn main() {
//!     let config = Configuration::new();
//!     let firewall = Firewall::new(config).unwrap();
//!
//!     let message = Message::new(Role::User, "Hello, world!");
//!     let result = firewall.scan(&message, None).await;
//!
//!     println!("Scan result: {:?}", result);
//! }
//! ```

pub mod config;
pub mod error;
pub mod firewall;
pub mod scanner;
pub mod types;
pub mod utils;

// Re-export commonly used types
pub use config::Configuration;
pub use error::FirewallError;
pub use firewall::Firewall;
pub use scanner::{ScanError, Scanner, ScannerRegistry};
pub use types::{
    Message, Role, ScanDecision, ScanResult, ScanStatus, ScannerType, ToolCall, Trace,
};
