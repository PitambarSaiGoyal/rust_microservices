/// Python compatibility tests
///
/// These tests validate that the Rust implementation produces the same results
/// as the Python implementation for common use cases.

use llamafirewall_core::{
    types::{Message, Role, ScanDecision},
    Scanner,
    Configuration, Firewall,
};
use llamafirewall_scanners::{RegexScanner, HiddenASCIIScanner};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[derive(Debug, Deserialize, Serialize)]
struct ScannerTestCase {
    name: String,
    content: String,
    expected_decision: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    expected_min_score: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    expected_max_score: Option<f64>,
    expected_reason_contains: String,
}

#[derive(Debug, Deserialize, Serialize)]
struct FirewallTestCase {
    name: String,
    role: String,
    content: String,
    scanners: Vec<String>,
    expected_decision: String,
}

#[derive(Debug, Deserialize, Serialize)]
struct TestData {
    regex_scanner_tests: Vec<ScannerTestCase>,
    hidden_ascii_scanner_tests: Vec<ScannerTestCase>,
    firewall_integration_tests: Vec<FirewallTestCase>,
}

fn parse_decision(decision_str: &str) -> ScanDecision {
    match decision_str.to_lowercase().as_str() {
        "allow" => ScanDecision::Allow,
        "block" => ScanDecision::Block,
        "human_in_the_loop_required" => ScanDecision::HumanInTheLoopRequired,
        _ => panic!("Unknown decision: {}", decision_str),
    }
}

fn parse_role(role_str: &str) -> Role {
    match role_str.to_uppercase().as_str() {
        "USER" => Role::User,
        "ASSISTANT" => Role::Assistant,
        "TOOL" => Role::Tool,
        "SYSTEM" => Role::System,
        "MEMORY" => Role::Memory,
        _ => panic!("Unknown role: {}", role_str),
    }
}

#[tokio::test]
async fn test_regex_scanner_python_compatibility() {
    let test_data_str = include_str!("python_compat_test_data.json");
    let test_data: TestData = serde_json::from_str(test_data_str).unwrap();

    let scanner = RegexScanner::new().expect("Failed to create RegexScanner");

    for test_case in test_data.regex_scanner_tests {
        let message = Message::new(Role::User, &test_case.content);
        let result = scanner.scan(&message, None).await
            .expect(&format!("Scan failed for test: {}", test_case.name));

        let expected_decision = parse_decision(&test_case.expected_decision);
        assert_eq!(
            result.decision,
            expected_decision,
            "Test '{}' failed: Expected decision {:?}, got {:?}",
            test_case.name,
            expected_decision,
            result.decision
        );

        if let Some(min_score) = test_case.expected_min_score {
            assert!(
                result.score >= min_score,
                "Test '{}' failed: Expected score >= {}, got {}",
                test_case.name,
                min_score,
                result.score
            );
        }

        if let Some(max_score) = test_case.expected_max_score {
            assert!(
                result.score <= max_score,
                "Test '{}' failed: Expected score <= {}, got {}",
                test_case.name,
                max_score,
                result.score
            );
        }

        assert!(
            result.reason.contains(&test_case.expected_reason_contains),
            "Test '{}' failed: Expected reason to contain '{}', got '{}'",
            test_case.name,
            test_case.expected_reason_contains,
            result.reason
        );
    }
}

#[tokio::test]
async fn test_hidden_ascii_scanner_python_compatibility() {
    let test_data_str = include_str!("python_compat_test_data.json");
    let test_data: TestData = serde_json::from_str(test_data_str).unwrap();

    let scanner = HiddenASCIIScanner::new();

    for test_case in test_data.hidden_ascii_scanner_tests {
        let message = Message::new(Role::Tool, &test_case.content);
        let result = scanner.scan(&message, None).await
            .expect(&format!("Scan failed for test: {}", test_case.name));

        let expected_decision = parse_decision(&test_case.expected_decision);
        assert_eq!(
            result.decision,
            expected_decision,
            "Test '{}' failed: Expected decision {:?}, got {:?}",
            test_case.name,
            expected_decision,
            result.decision
        );

        if let Some(min_score) = test_case.expected_min_score {
            assert!(
                result.score >= min_score,
                "Test '{}' failed: Expected score >= {}, got {}",
                test_case.name,
                min_score,
                result.score
            );
        }

        if let Some(max_score) = test_case.expected_max_score {
            assert!(
                result.score <= max_score,
                "Test '{}' failed: Expected score <= {}, got {}",
                test_case.name,
                max_score,
                result.score
            );
        }

        assert!(
            result.reason.to_lowercase().contains(&test_case.expected_reason_contains.to_lowercase()),
            "Test '{}' failed: Expected reason to contain '{}', got '{}'",
            test_case.name,
            test_case.expected_reason_contains,
            result.reason
        );
    }
}

#[tokio::test]
async fn test_firewall_integration_python_compatibility() {
    let test_data_str = include_str!("python_compat_test_data.json");
    let test_data: TestData = serde_json::from_str(test_data_str).unwrap();

    for test_case in test_data.firewall_integration_tests {
        let role = parse_role(&test_case.role);
        let mut config = Configuration::new();

        // Add scanners based on test case
        for scanner_name in &test_case.scanners {
            match scanner_name.as_str() {
                "REGEX" => {
                    let scanner = RegexScanner::new().expect("Failed to create RegexScanner");
                    config = config.add_scanner(role, Arc::new(scanner));
                }
                "HIDDEN_ASCII" => {
                    config = config.add_scanner(role, Arc::new(HiddenASCIIScanner::new()));
                }
                _ => panic!("Unknown scanner: {}", scanner_name),
            }
        }

        let firewall = Firewall::new(config).unwrap();
        let message = Message::new(role, &test_case.content);
        let result = firewall.scan(&message, None).await;

        let expected_decision = parse_decision(&test_case.expected_decision);
        assert_eq!(
            result.decision,
            expected_decision,
            "Test '{}' failed: Expected decision {:?}, got {:?}. Reason: {}",
            test_case.name,
            expected_decision,
            result.decision,
            result.reason
        );
    }
}

/// Test that serialization format matches Python
#[test]
fn test_serialization_format_compatibility() {
    use llamafirewall_core::types::{Role, ScanDecision, ScanStatus, ScanResult};

    // Test Role serialization
    assert_eq!(serde_json::to_string(&Role::User).unwrap(), r#""USER""#);
    assert_eq!(serde_json::to_string(&Role::Assistant).unwrap(), r#""ASSISTANT""#);
    assert_eq!(serde_json::to_string(&Role::Tool).unwrap(), r#""TOOL""#);
    assert_eq!(serde_json::to_string(&Role::System).unwrap(), r#""SYSTEM""#);
    assert_eq!(serde_json::to_string(&Role::Memory).unwrap(), r#""MEMORY""#);

    // Test ScanDecision serialization
    assert_eq!(serde_json::to_string(&ScanDecision::Allow).unwrap(), r#""allow""#);
    assert_eq!(serde_json::to_string(&ScanDecision::Block).unwrap(), r#""block""#);
    assert_eq!(
        serde_json::to_string(&ScanDecision::HumanInTheLoopRequired).unwrap(),
        r#""human_in_the_loop_required""#
    );

    // Test ScanStatus serialization
    assert_eq!(serde_json::to_string(&ScanStatus::Success).unwrap(), r#""success""#);
    assert_eq!(serde_json::to_string(&ScanStatus::Error).unwrap(), r#""error""#);
    assert_eq!(serde_json::to_string(&ScanStatus::Skipped).unwrap(), r#""skipped""#);

    // Test ScanResult serialization
    let result = ScanResult::block("test reason", 0.95);
    let json = serde_json::to_string(&result).unwrap();
    assert!(json.contains(r#""decision":"block""#));
    assert!(json.contains(r#""reason":"test reason""#));
    assert!(json.contains(r#""score":0.95"#));
    assert!(json.contains(r#""status":"success""#));
}

/// Test deserialization from Python format
#[test]
fn test_deserialization_from_python() {
    use llamafirewall_core::types::{Message, Role};

    // Test Message deserialization
    let python_json = r#"{"role":"USER","content":"hello"}"#;
    let message: Message = serde_json::from_str(python_json).unwrap();
    assert_eq!(message.role, Role::User);
    assert_eq!(message.content, "hello");

    // Test with tool_calls
    let python_json_with_tools = r#"{"role":"ASSISTANT","content":"test","tool_calls":[{"id":"1","function":"test","arguments":{}}]}"#;
    let message: Message = serde_json::from_str(python_json_with_tools).unwrap();
    assert_eq!(message.role, Role::Assistant);
    assert!(message.tool_calls.is_some());
}
