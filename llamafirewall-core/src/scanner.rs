//! Scanner trait and related types
//!
//! This module defines the core Scanner trait that all scanners must implement,
//! along with scanner-specific error types and utilities.

use crate::types::{Message, ScanResult};
use async_trait::async_trait;
use std::sync::Arc;

/// Core scanner trait
///
/// All scanners must implement this trait to be used with the Firewall.
/// Scanners are responsible for analyzing messages and returning scan results
/// indicating whether the message should be allowed, blocked, or needs human review.
#[async_trait]
pub trait Scanner: Send + Sync {
    /// Scanner name (for logging and errors)
    fn name(&self) -> &str;

    /// Scan a message with optional conversation history
    ///
    /// # Arguments
    /// * `message` - The message to scan
    /// * `past_trace` - Optional conversation history (zero-copy slice)
    ///
    /// # Returns
    /// ScanResult with decision, reason, and confidence score
    async fn scan(
        &self,
        message: &Message,
        past_trace: Option<&[Arc<Message>]>,
    ) -> Result<ScanResult, ScanError>;

    /// Minimum confidence score to trigger a block (default: 0.5)
    fn block_threshold(&self) -> f64 {
        0.5
    }

    /// Whether this scanner requires full conversation history
    fn requires_full_trace(&self) -> bool {
        false
    }

    /// Scanner-specific configuration validation
    fn validate_config(&self) -> Result<(), ScanError> {
        Ok(())
    }
}

/// Scanner-specific errors
#[derive(Debug, thiserror::Error)]
pub enum ScanError {
    #[error("Model loading failed: {0}")]
    ModelLoadError(String),

    #[error("Inference failed: {0}")]
    InferenceError(String),

    #[error("External API call failed: {0}")]
    ApiError(String),

    #[error("Invalid input: {0}")]
    InvalidInput(String),

    #[error("Configuration error: {0}")]
    ConfigError(String),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
}

/// Type alias for scanner registry
pub type ScannerRegistry = std::collections::HashMap<String, Arc<dyn Scanner>>;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Role;

    // Mock scanner for testing
    struct MockScanner {
        name: String,
        result: ScanResult,
    }

    #[async_trait]
    impl Scanner for MockScanner {
        fn name(&self) -> &str {
            &self.name
        }

        async fn scan(
            &self,
            _message: &Message,
            _past_trace: Option<&[Arc<Message>]>,
        ) -> Result<ScanResult, ScanError> {
            Ok(self.result.clone())
        }
    }

    #[tokio::test]
    async fn test_mock_scanner() {
        let scanner = MockScanner {
            name: "test".to_string(),
            result: ScanResult::allow(),
        };

        let msg = Message::new(Role::User, "test");
        let result = scanner.scan(&msg, None).await.unwrap();
        assert_eq!(result.decision, crate::types::ScanDecision::Allow);
    }

    #[tokio::test]
    async fn test_scanner_validation() {
        let scanner = MockScanner {
            name: "test".to_string(),
            result: ScanResult::allow(),
        };

        assert!(scanner.validate_config().is_ok());
    }

    #[test]
    fn test_scanner_defaults() {
        let scanner = MockScanner {
            name: "test".to_string(),
            result: ScanResult::allow(),
        };

        assert_eq!(scanner.name(), "test");
        assert_eq!(scanner.block_threshold(), 0.5);
        assert!(!scanner.requires_full_trace());
    }
}
