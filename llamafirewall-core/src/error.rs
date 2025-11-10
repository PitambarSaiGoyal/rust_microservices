//! Error types for LlamaFirewall

use thiserror::Error;

/// Top-level firewall errors
#[derive(Debug, Error)]
pub enum FirewallError {
    #[error("Configuration error: {0}")]
    ConfigError(String),

    #[error("Scanner initialization failed: {0}")]
    ScannerInitError(String),

    #[error("Invalid message: {0}")]
    InvalidMessage(String),

    #[error(transparent)]
    ScanError(#[from] crate::scanner::ScanError),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = FirewallError::ConfigError("invalid config".to_string());
        assert_eq!(err.to_string(), "Configuration error: invalid config");

        let err = FirewallError::InvalidMessage("empty message".to_string());
        assert_eq!(err.to_string(), "Invalid message: empty message");
    }
}
