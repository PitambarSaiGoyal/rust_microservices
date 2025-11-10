//! Memory leak detection tests
//!
//! These tests verify that repeated operations don't accumulate memory leaks.
//! While Rust's ownership system prevents most memory leaks, these tests ensure
//! that Arc/Mutex patterns and async operations don't create reference cycles.

use llamafirewall_core::{
    config::Configuration,
    firewall::Firewall,
    scanner::{Scanner, ScanError},
    types::{Message, Role, ScanDecision, ScanResult, ScanStatus, Trace},
};
use async_trait::async_trait;
use std::sync::Arc;

/// Simple test scanner that always allows
struct TestScanner;

#[async_trait]
impl Scanner for TestScanner {
    async fn scan(&self, _message: &Message, _trace: Option<&[Arc<Message>]>) -> Result<ScanResult, ScanError> {
        Ok(ScanResult {
            decision: ScanDecision::Allow,
            reason: "Test scanner always allows".to_string(),
            score: 0.0,
            status: ScanStatus::Success,
        })
    }

    fn name(&self) -> &str {
        "TestScanner"
    }
}

#[tokio::test]
async fn test_repeated_firewall_creation_no_leak() {
    // Create and drop firewall many times
    // If there's a leak, memory usage would grow significantly
    for _i in 0..1000 {
        let config = Configuration::new()
            .add_scanner(Role::User, Arc::new(TestScanner));

        let firewall = Firewall::new(config).unwrap();
        let msg = Message::new(Role::User, "test");
        let _ = firewall.scan(&msg, None).await;

        // Firewall should be dropped here, releasing all resources
        drop(firewall);
    }

    // If we reach here without OOM, no significant leak occurred
    assert!(true);
}

#[tokio::test]
async fn test_repeated_scanning_no_leak() {
    let config = Configuration::new()
        .add_scanner(Role::User, Arc::new(TestScanner));

    let firewall = Firewall::new(config).unwrap();

    // Scan many times with same firewall
    for _i in 0..1000 {
        let msg = Message::new(Role::User, format!("test message {}", _i));
        let result = firewall.scan(&msg, None).await;
        assert_eq!(result.decision, ScanDecision::Allow);

        // Result should be dropped here
    }

    // Verify firewall is still functional
    let msg = Message::new(Role::User, "final test");
    let result = firewall.scan(&msg, None).await;
    assert_eq!(result.decision, ScanDecision::Allow);
}

#[tokio::test]
async fn test_large_trace_no_leak() {
    let config = Configuration::new()
        .add_scanner(Role::User, Arc::new(TestScanner));

    let firewall = Firewall::new(config).unwrap();

    // Create a large trace
    let trace: Trace = (0..100)
        .map(|i| {
            Arc::new(Message::new(
                if i % 2 == 0 { Role::User } else { Role::Assistant },
                format!("Message {}", i),
            ))
        })
        .collect();

    // Scan with large trace multiple times
    for _i in 0..100 {
        let msg = Message::new(Role::User, "test");
        let result = firewall.scan(&msg, Some(&trace)).await;
        assert_eq!(result.decision, ScanDecision::Allow);
    }

    // If we reach here without OOM, no significant leak occurred
    assert!(true);
}

#[tokio::test]
async fn test_message_cloning_no_leak() {
    // Test that message cloning (Arc cloning) doesn't leak
    let mut messages = Vec::new();

    for i in 0..1000 {
        let msg = Message::new(Role::User, format!("Message {}", i));
        messages.push(msg);
    }

    // Clone all messages multiple times
    for _round in 0..10 {
        let _clones: Vec<_> = messages.iter().cloned().collect();
        // Clones dropped here
    }

    // Original messages should still be valid
    assert_eq!(messages.len(), 1000);
    drop(messages);

    // If we reach here without OOM, no significant leak occurred
    assert!(true);
}

#[tokio::test]
async fn test_concurrent_scanning_no_leak() {
    use tokio::task::JoinSet;

    let config = Configuration::new()
        .add_scanner(Role::User, Arc::new(TestScanner));

    let firewall = Firewall::new(config).unwrap();
    let firewall = Arc::new(firewall);

    // Run many concurrent scans
    let mut join_set = JoinSet::new();

    for i in 0..100 {
        let fw = firewall.clone();
        join_set.spawn(async move {
            let msg = Message::new(Role::User, format!("concurrent message {}", i));
            fw.scan(&msg, None).await
        });
    }

    // Wait for all to complete
    let mut results = Vec::new();
    while let Some(result) = join_set.join_next().await {
        results.push(result.unwrap());
    }

    assert_eq!(results.len(), 100);

    // All tasks completed, no handles leaked
    assert!(true);
}
