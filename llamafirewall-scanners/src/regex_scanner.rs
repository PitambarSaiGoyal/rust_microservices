//! Regex-based scanner for pattern matching
//!
//! This scanner uses regular expressions to detect sensitive patterns
//! like API keys, secrets, passwords, and other credentials.

use async_trait::async_trait;
use llamafirewall_core::{
    scanner::{ScanError, Scanner},
    types::{Message, ScanDecision, ScanResult, ScanStatus},
};
use regex::{Regex, RegexSet};
use std::sync::Arc;

/// Default regex patterns (superset of Python + Rust security patterns)
const DEFAULT_PATTERNS: &[(&str, &str)] = &[
    // Python implementation patterns - PII and prompt injection
    (
        "Prompt injection",
        r"(?i)(ignore|disregard)\s+(previous|all|former)?\s*(previous|all|former)?\s*(instructions|directives)",
    ),
    (
        "Email address",
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
    ),
    (
        "Phone number",
        r"\b(\+\d{1,2}\s?)?\(?\d{3}\)?[\s.\-]?\d{3}[\s.\-]?\d{4}\b",
    ),
    (
        "Credit card",
        r"\b(?:\d{4}[- ]?){3}\d{4}\b",
    ),
    (
        "Social security number",
        r"\b\d{3}-\d{2}-\d{4}\b",
    ),
    // Rust implementation patterns - API keys and secrets
    (
        "api_key",
        r#"(?i)(api[_\-]?key|apikey)\s*[:=]\s*['\"]?([a-zA-Z0-9_\-]{16,})"#,
    ),
    (
        "secret",
        r#"(?i)(secret|password)\s*[:=]\s*['\"]?([^\s'\"]{8,})"#,
    ),
    ("aws_key", r"(?i)(AKIA[0-9A-Z]{16})"),
    ("private_key", r"-----BEGIN (RSA |EC )?PRIVATE KEY-----"),
    (
        "jwt",
        r"eyJ[A-Za-z0-9_-]+\.eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+",
    ),
];

/// Regex scanner for detecting sensitive patterns
#[derive(Debug)]
pub struct RegexScanner {
    name: String,
    patterns: RegexSet,
    pattern_names: Vec<String>,
    individual_patterns: Vec<Regex>,
    block_threshold: f64,
}

impl RegexScanner {
    /// Create scanner with default patterns
    pub fn new() -> Result<Self, ScanError> {
        Self::with_patterns(
            DEFAULT_PATTERNS
                .iter()
                .map(|&(k, v)| (k.to_string(), v.to_string()))
                .collect(),
        )
    }

    /// Create scanner with custom patterns
    ///
    /// # Arguments
    /// * `patterns` - Vec of (name, pattern) tuples
    ///
    /// # Example
    /// ```
    /// use llamafirewall_scanners::RegexScanner;
    ///
    /// let patterns = vec![
    ///     ("email".to_string(), r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b".to_string()),
    /// ];
    /// let scanner = RegexScanner::with_patterns(patterns).unwrap();
    /// ```
    pub fn with_patterns(patterns: Vec<(String, String)>) -> Result<Self, ScanError> {
        if patterns.is_empty() {
            return Err(ScanError::ConfigError(
                "At least one pattern is required".to_string(),
            ));
        }

        let pattern_strs: Vec<_> = patterns.iter().map(|(_, p)| p.as_str()).collect();
        let pattern_names: Vec<_> = patterns.iter().map(|(n, _)| n.clone()).collect();

        // Create RegexSet for efficient simultaneous matching
        let patterns_set = RegexSet::new(&pattern_strs)
            .map_err(|e| ScanError::ConfigError(format!("Invalid regex pattern: {}", e)))?;

        // Compile individual patterns for detailed matching
        let individual_patterns: Result<Vec<_>, _> =
            pattern_strs.iter().map(|p| Regex::new(p)).collect();

        let individual_patterns = individual_patterns
            .map_err(|e| ScanError::ConfigError(format!("Invalid regex pattern: {}", e)))?;

        Ok(Self {
            name: "regex_scanner".to_string(),
            patterns: patterns_set,
            pattern_names,
            individual_patterns,
            block_threshold: 1.0,
        })
    }

    /// Set block threshold (default: 1.0)
    pub fn with_block_threshold(mut self, threshold: f64) -> Self {
        self.block_threshold = threshold;
        self
    }

    /// Set scanner name (default: "regex_scanner")
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }
}

impl Default for RegexScanner {
    fn default() -> Self {
        Self::new().expect("Default patterns should always compile")
    }
}

#[async_trait]
impl Scanner for RegexScanner {
    fn name(&self) -> &str {
        &self.name
    }

    async fn scan(
        &self,
        message: &Message,
        _past_trace: Option<&[Arc<Message>]>,
    ) -> Result<ScanResult, ScanError> {
        let text = &message.content;

        // Fast path: Check all patterns simultaneously with RegexSet
        let matches = self.patterns.matches(text);

        if !matches.matched_any() {
            return Ok(ScanResult {
                decision: ScanDecision::Allow,
                reason: "No regex patterns matched".to_string(),
                score: 0.0,
                status: ScanStatus::Success,
            });
        }

        // Find first match and extract details
        let idx = matches
            .iter()
            .next()
            .expect("matched_any() returned true but no matches found");
        let pattern_name = &self.pattern_names[idx];

        // Format reason to match Python: "Regex match: {pattern_name} - score: 1.0"
        Ok(ScanResult::block(
            format!("Regex match: {} - score: 1.0", pattern_name),
            1.0,
        ))
    }

    fn block_threshold(&self) -> f64 {
        self.block_threshold
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use llamafirewall_core::types::{Role, ScanDecision};

    #[tokio::test]
    async fn test_regex_scanner_creation() {
        let scanner = RegexScanner::new().unwrap();
        assert_eq!(scanner.name(), "regex_scanner");
    }

    #[tokio::test]
    async fn test_no_match() {
        let scanner = RegexScanner::new().unwrap();
        let msg = Message::new(Role::User, "Hello world, nothing sensitive here");

        let result = scanner.scan(&msg, None).await.unwrap();
        assert_eq!(result.decision, ScanDecision::Allow);
    }

    #[tokio::test]
    async fn test_api_key_detection() {
        let scanner = RegexScanner::new().unwrap();
        let msg = Message::new(Role::User, "My API_KEY=sk_test_123456789abcdef");

        let result = scanner.scan(&msg, None).await.unwrap();
        assert_eq!(result.decision, ScanDecision::Block);
        assert!(result.reason.contains("api_key"));
    }

    #[tokio::test]
    async fn test_api_key_variations() {
        let scanner = RegexScanner::new().unwrap();

        // Test different API key formats
        let test_cases = vec![
            "api_key: sk_live_1234567890abcdef",
            "apiKey=pk_test_abcdefghijklmnop",
            "API-KEY = 'my_secret_key_12345'",
        ];

        for test_case in test_cases {
            let msg = Message::new(Role::User, test_case);
            let result = scanner.scan(&msg, None).await.unwrap();
            assert_eq!(
                result.decision,
                ScanDecision::Block,
                "Failed to detect: {}",
                test_case
            );
        }
    }

    #[tokio::test]
    async fn test_secret_detection() {
        let scanner = RegexScanner::new().unwrap();
        let msg = Message::new(Role::User, "password=mysecretpassword123");

        let result = scanner.scan(&msg, None).await.unwrap();
        assert_eq!(result.decision, ScanDecision::Block);
        assert!(result.reason.contains("secret"));
    }

    #[tokio::test]
    async fn test_aws_key_detection() {
        let scanner = RegexScanner::new().unwrap();
        let msg = Message::new(Role::User, "AWS key: AKIAIOSFODNN7EXAMPLE");

        let result = scanner.scan(&msg, None).await.unwrap();
        assert_eq!(result.decision, ScanDecision::Block);
        assert!(result.reason.contains("aws_key"));
    }

    #[tokio::test]
    async fn test_private_key_detection() {
        let scanner = RegexScanner::new().unwrap();
        let msg = Message::new(
            Role::User,
            "-----BEGIN RSA PRIVATE KEY-----\nMIIBogIBAAJBALR...",
        );

        let result = scanner.scan(&msg, None).await.unwrap();
        assert_eq!(result.decision, ScanDecision::Block);
        assert!(result.reason.contains("private_key"));
    }

    #[tokio::test]
    async fn test_jwt_detection() {
        let scanner = RegexScanner::new().unwrap();
        let jwt = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U";
        let msg = Message::new(Role::User, format!("Token: {}", jwt));

        let result = scanner.scan(&msg, None).await.unwrap();
        assert_eq!(result.decision, ScanDecision::Block);
        assert!(result.reason.contains("jwt"));
    }

    #[tokio::test]
    async fn test_custom_patterns() {
        let patterns = vec![(
            "email".to_string(),
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b".to_string(),
        )];
        let scanner = RegexScanner::with_patterns(patterns).unwrap();

        let msg = Message::new(Role::User, "Contact me at test@example.com");
        let result = scanner.scan(&msg, None).await.unwrap();
        assert_eq!(result.decision, ScanDecision::Block);
        assert!(result.reason.contains("email"));
    }

    #[tokio::test]
    async fn test_multiple_custom_patterns() {
        let patterns = vec![
            ("phone".to_string(), r"\d{3}-\d{3}-\d{4}".to_string()),
            ("ssn".to_string(), r"\d{3}-\d{2}-\d{4}".to_string()),
        ];
        let scanner = RegexScanner::with_patterns(patterns).unwrap();

        // Test phone number
        let msg = Message::new(Role::User, "Call me at 555-123-4567");
        let result = scanner.scan(&msg, None).await.unwrap();
        assert_eq!(result.decision, ScanDecision::Block);
        assert!(result.reason.contains("phone"));

        // Test SSN
        let msg = Message::new(Role::User, "My SSN is 123-45-6789");
        let result = scanner.scan(&msg, None).await.unwrap();
        assert_eq!(result.decision, ScanDecision::Block);
        assert!(result.reason.contains("ssn"));
    }

    #[tokio::test]
    async fn test_empty_patterns_error() {
        let result = RegexScanner::with_patterns(vec![]);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("At least one pattern is required"));
    }

    #[tokio::test]
    async fn test_invalid_regex_pattern() {
        let patterns = vec![("bad".to_string(), r"[invalid(".to_string())];
        let result = RegexScanner::with_patterns(patterns);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Invalid regex pattern"));
    }

    #[tokio::test]
    async fn test_truncated_match_output() {
        let scanner = RegexScanner::new().unwrap();
        // Create a very long API key
        let long_key = "A".repeat(100);
        let msg = Message::new(Role::User, format!("api_key={}", long_key));

        let result = scanner.scan(&msg, None).await.unwrap();
        assert_eq!(result.decision, ScanDecision::Block);
        // Verify the matched text is truncated
        assert!(result.reason.len() < 100 + "Pattern 'api_key' matched: ".len());
        assert!(result.reason.contains("..."));
    }

    #[tokio::test]
    async fn test_with_name() {
        let scanner = RegexScanner::new().unwrap().with_name("my_custom_scanner");
        assert_eq!(scanner.name(), "my_custom_scanner");
    }

    #[tokio::test]
    async fn test_with_block_threshold() {
        let scanner = RegexScanner::new().unwrap().with_block_threshold(0.8);
        assert_eq!(scanner.block_threshold(), 0.8);
    }

    #[tokio::test]
    async fn test_case_insensitive_matching() {
        let scanner = RegexScanner::new().unwrap();

        // Test that API_KEY, api_key, Api_Key all match
        let test_cases = vec![
            "API_KEY=test1234567890abc",
            "api_key=test1234567890abc",
            "Api_Key=test1234567890abc",
            "aPi_KeY=test1234567890abc",
        ];

        for test_case in test_cases {
            let msg = Message::new(Role::User, test_case);
            let result = scanner.scan(&msg, None).await.unwrap();
            assert_eq!(
                result.decision,
                ScanDecision::Block,
                "Failed case-insensitive match: {}",
                test_case
            );
        }
    }
}
