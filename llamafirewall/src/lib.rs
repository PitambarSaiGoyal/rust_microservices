//! # LlamaFirewall
//!
//! High-performance Rust library for LLM guardrails.
//!
//! LlamaFirewall provides a comprehensive set of scanners to protect LLM applications
//! from security threats including:
//! - Prompt injection attacks
//! - Insecure code generation
//! - Credential leakage
//! - Hidden malicious content
//!
//! ## Features
//!
//! - **2-3x Performance**: Faster than Python implementation
//! - **Zero-Copy Architecture**: Efficient trace management
//! - **Parallel Scanning**: Multiple scanners run concurrently
//! - **Type-Safe**: Leverages Rust's type system for compile-time guarantees
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use llamafirewall::{Firewall, Configuration, Message, Role, RegexScanner};
//! use std::sync::Arc;
//!
//! #[tokio::main]
//! async fn main() {
//!     // Create configuration with scanners
//!     let regex_scanner = Arc::new(RegexScanner::new().unwrap());
//!     let config = Configuration::new()
//!         .add_scanner(Role::User, regex_scanner);
//!
//!     // Initialize firewall
//!     let firewall = Firewall::new(config).unwrap();
//!
//!     // Scan a message
//!     let message = Message::new(Role::User, "My API key is sk_test_123456");
//!     let result = firewall.scan(&message, None).await;
//!
//!     println!("Decision: {:?}", result.decision);
//!     println!("Reason: {}", result.reason);
//! }
//! ```
//!
//! ## Architecture
//!
//! The library is organized into several crates:
//!
//! - **llamafirewall-core**: Core types, traits, and orchestration
//! - **llamafirewall-scanners**: Scanner implementations
//! - **llamafirewall-ml**: ML infrastructure (Phase 3+)
//! - **llamafirewall**: Main library that re-exports everything
//!
//! ## Scanners
//!
//! ### Available Now (Phase 0-2)
//! - `RegexScanner`: Pattern-based detection
//! - `HiddenASCIIScanner`: Hidden character detection
//!
//! ### Coming Soon (Phase 3+)
//! - `PromptGuardScanner`: ML-based prompt injection detection
//! - `CodeShieldScanner`: Tree-sitter based code analysis
//!
//! ## Performance
//!
//! Target performance improvements over Python:
//! - Single scan: 163ms → 70ms (2.3x)
//! - Trace replay: 970ms → 420ms (2.3x)
//! - With native CodeShield: 163ms → 52ms (3.1x)

// Re-export core types and traits
pub use llamafirewall_core::{
    Configuration, Firewall, FirewallError, Message, Role, ScanDecision, ScanError, ScanResult,
    ScanStatus, Scanner, ScannerRegistry, ScannerType, ToolCall, Trace,
};

// Re-export scanners
pub use llamafirewall_scanners::{HiddenASCIIScanner, PromptGuardScanner, RegexScanner};

// Re-export ML infrastructure (when available)
// pub use llamafirewall_ml::*;

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Get library version
pub fn version() -> &'static str {
    VERSION
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        assert!(!version().is_empty());
    }

    #[tokio::test]
    async fn test_basic_firewall_usage() {
        // Create a simple configuration
        let config = Configuration::new();
        let firewall = Firewall::new(config).unwrap();

        // Test with a simple message (no scanners configured)
        let message = Message::new(Role::User, "Hello, world!");
        let result = firewall.scan(&message, None).await;

        // Should allow when no scanners are configured
        assert_eq!(result.decision, ScanDecision::Allow);
    }

    #[tokio::test]
    async fn test_with_regex_scanner() {
        let scanner =
            std::sync::Arc::new(RegexScanner::new().unwrap()) as std::sync::Arc<dyn Scanner>;
        let config = Configuration::new().add_scanner(Role::User, scanner);
        let firewall = Firewall::new(config).unwrap();

        let message = Message::new(Role::User, "Normal message");
        let result = firewall.scan(&message, None).await;

        // Placeholder scanner always allows
        assert_eq!(result.decision, ScanDecision::Allow);
    }
}
