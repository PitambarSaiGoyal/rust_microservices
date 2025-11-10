//! Core type definitions for LlamaFirewall
//!
//! This module defines the fundamental types used throughout the library:
//! - Message and conversation traces
//! - Scan results and decisions
//! - Scanner configuration types

use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Message role in conversation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum Role {
    User,
    Assistant,
    Tool,
    System,
    Memory,
}

/// Tool call information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub function: String,
    pub arguments: serde_json::Value,
}

/// A message in the conversation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
}

impl Message {
    /// Create a new message
    pub fn new(role: Role, content: impl Into<String>) -> Self {
        Self {
            role,
            content: content.into(),
            tool_calls: None,
        }
    }

    /// Add tool calls to this message
    pub fn with_tool_calls(mut self, calls: Vec<ToolCall>) -> Self {
        self.tool_calls = Some(calls);
        self
    }
}

/// Conversation trace (zero-copy with Arc)
pub type Trace = Vec<Arc<Message>>;

/// Scan decision
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ScanDecision {
    Allow,
    Block,
    #[serde(rename = "human_in_the_loop_required")]
    HumanInTheLoopRequired,
}

/// Scan status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ScanStatus {
    Success,
    Error,
    Skipped,
}

/// Result of a scan operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScanResult {
    pub decision: ScanDecision,
    pub reason: String,
    pub score: f64,
    pub status: ScanStatus,
}

impl ScanResult {
    /// Create an "allow" result
    pub fn allow() -> Self {
        Self {
            decision: ScanDecision::Allow,
            reason: "default".to_string(),
            score: 0.0,
            status: ScanStatus::Success,
        }
    }

    /// Create a "block" result
    pub fn block(reason: impl Into<String>, score: f64) -> Self {
        Self {
            decision: ScanDecision::Block,
            reason: reason.into(),
            score,
            status: ScanStatus::Success,
        }
    }

    /// Create a "human review required" result
    pub fn human_review(reason: impl Into<String>, score: f64) -> Self {
        Self {
            decision: ScanDecision::HumanInTheLoopRequired,
            reason: reason.into(),
            score,
            status: ScanStatus::Success,
        }
    }

    /// Create an "error" result
    pub fn error(reason: impl Into<String>) -> Self {
        Self {
            decision: ScanDecision::Allow,
            reason: reason.into(),
            score: 0.0,
            status: ScanStatus::Error,
        }
    }
}

/// Scanner type enum (for configuration)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ScannerType {
    PromptGuard,
    CodeShield,
    Regex,
    HiddenAscii,
    // Note: AlignmentCheck and PiiDetection are intentionally excluded
    // See "Architectural Decisions" section in the implementation plan
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_message_creation() {
        let msg = Message::new(Role::User, "Hello");
        assert_eq!(msg.role, Role::User);
        assert_eq!(msg.content, "Hello");
        assert!(msg.tool_calls.is_none());
    }

    #[test]
    fn test_message_with_tool_calls() {
        let tool_call = ToolCall {
            id: "call_123".to_string(),
            function: "get_weather".to_string(),
            arguments: serde_json::json!({"city": "New York"}),
        };

        let msg = Message::new(Role::Assistant, "Let me check the weather")
            .with_tool_calls(vec![tool_call]);

        assert_eq!(msg.role, Role::Assistant);
        assert!(msg.tool_calls.is_some());
        assert_eq!(msg.tool_calls.unwrap().len(), 1);
    }

    #[test]
    fn test_scan_result_constructors() {
        let allow = ScanResult::allow();
        assert_eq!(allow.decision, ScanDecision::Allow);
        assert_eq!(allow.status, ScanStatus::Success);

        let block = ScanResult::block("test", 0.9);
        assert_eq!(block.decision, ScanDecision::Block);
        assert_eq!(block.score, 0.9);
        assert_eq!(block.status, ScanStatus::Success);

        let human = ScanResult::human_review("needs review", 0.7);
        assert_eq!(human.decision, ScanDecision::HumanInTheLoopRequired);
        assert_eq!(human.score, 0.7);

        let error = ScanResult::error("scanner failed");
        assert_eq!(error.decision, ScanDecision::Allow); // Fail open
        assert_eq!(error.status, ScanStatus::Error);
    }

    #[test]
    fn test_serialization_compatibility() {
        // Ensure JSON format matches Python
        let msg = Message::new(Role::User, "test");
        let json = serde_json::to_string(&msg).unwrap();
        assert!(json.contains(r#""role":"USER""#));

        let result = ScanResult::block("blocked", 0.9);
        let json = serde_json::to_string(&result).unwrap();
        assert!(json.contains(r#""decision":"block""#));
        assert!(json.contains(r#""status":"success""#));
    }

    #[test]
    fn test_role_serialization() {
        assert_eq!(serde_json::to_string(&Role::User).unwrap(), r#""USER""#);
        assert_eq!(
            serde_json::to_string(&Role::Assistant).unwrap(),
            r#""ASSISTANT""#
        );
        assert_eq!(serde_json::to_string(&Role::System).unwrap(), r#""SYSTEM""#);
    }

    #[test]
    fn test_scan_decision_serialization() {
        assert_eq!(
            serde_json::to_string(&ScanDecision::Allow).unwrap(),
            r#""allow""#
        );
        assert_eq!(
            serde_json::to_string(&ScanDecision::Block).unwrap(),
            r#""block""#
        );
        assert_eq!(
            serde_json::to_string(&ScanDecision::HumanInTheLoopRequired).unwrap(),
            r#""human_in_the_loop_required""#
        );
    }
}
