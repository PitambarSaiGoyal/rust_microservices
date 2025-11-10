//! Hidden ASCII character scanner
//!
//! This scanner detects hidden ASCII tag characters (U+E0000 to U+E007F)
//! that can be used for steganography or prompt injection attacks.

use async_trait::async_trait;
use llamafirewall_core::{
    scanner::{ScanError, Scanner},
    types::{Message, ScanResult},
};
use std::sync::Arc;

/// Range for hidden ASCII tag characters
const HIDDEN_ASCII_START: u32 = 0xE0000;
const HIDDEN_ASCII_END: u32 = 0xE007F;

/// Hidden ASCII scanner for detecting steganographic text
pub struct HiddenASCIIScanner {
    name: String,
    block_threshold: f64,
}

impl Default for HiddenASCIIScanner {
    fn default() -> Self {
        Self::new()
    }
}

impl HiddenASCIIScanner {
    /// Create a new HiddenASCIIScanner with default settings
    pub fn new() -> Self {
        Self {
            name: "hidden_ascii_scanner".to_string(),
            block_threshold: 1.0,
        }
    }

    /// Set block threshold (default: 1.0)
    pub fn with_block_threshold(mut self, threshold: f64) -> Self {
        self.block_threshold = threshold;
        self
    }

    /// Set scanner name (default: "hidden_ascii_scanner")
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Check if text contains hidden ASCII tag characters
    ///
    /// This is optimized for early termination - stops at first hidden character found
    #[inline]
    fn contains_hidden_ascii(text: &str) -> bool {
        text.chars().any(|c| {
            let codepoint = c as u32;
            (HIDDEN_ASCII_START..=HIDDEN_ASCII_END).contains(&codepoint)
        })
    }

    /// Decode hidden ASCII tag characters to regular ASCII
    ///
    /// Characters in the range U+E0000 to U+E007F are decoded by subtracting
    /// HIDDEN_ASCII_START to get the original ASCII character.
    fn decode_hidden_ascii(text: &str) -> String {
        let mut decoded = String::with_capacity(text.len());

        for ch in text.chars() {
            let codepoint = ch as u32;
            if (HIDDEN_ASCII_START..=HIDDEN_ASCII_END).contains(&codepoint) {
                // Decode tag to ASCII by subtracting the offset
                if let Some(decoded_char) = char::from_u32(codepoint - HIDDEN_ASCII_START) {
                    decoded.push(decoded_char);
                } else {
                    // If decoding fails, keep the original character
                    decoded.push(ch);
                }
            } else {
                decoded.push(ch);
            }
        }

        decoded
    }

    /// Extract only the hidden ASCII characters from text
    #[allow(dead_code)]
    fn extract_hidden_chars(text: &str) -> Vec<char> {
        text.chars()
            .filter(|&c| {
                let codepoint = c as u32;
                (HIDDEN_ASCII_START..=HIDDEN_ASCII_END).contains(&codepoint)
            })
            .collect()
    }

    /// Count hidden ASCII characters in text
    fn count_hidden_chars(text: &str) -> usize {
        text.chars()
            .filter(|&c| {
                let codepoint = c as u32;
                (HIDDEN_ASCII_START..=HIDDEN_ASCII_END).contains(&codepoint)
            })
            .count()
    }
}

#[async_trait]
impl Scanner for HiddenASCIIScanner {
    fn name(&self) -> &str {
        &self.name
    }

    async fn scan(
        &self,
        message: &Message,
        _past_trace: Option<&[Arc<Message>]>,
    ) -> Result<ScanResult, ScanError> {
        // Fast path: check if any hidden characters exist
        if !Self::contains_hidden_ascii(&message.content) {
            return Ok(ScanResult::allow());
        }

        // Decode the hidden characters
        let decoded = Self::decode_hidden_ascii(&message.content);
        let hidden_count = Self::count_hidden_chars(&message.content);

        // Truncate decoded text for readability
        let decoded_preview = if decoded.len() > 100 {
            format!("{}...", &decoded[..97])
        } else {
            decoded
        };

        Ok(ScanResult::block(
            format!(
                "Hidden ASCII detected ({} hidden characters). Decoded: {}",
                hidden_count, decoded_preview
            ),
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
    async fn test_hidden_ascii_scanner_creation() {
        let scanner = HiddenASCIIScanner::new();
        assert_eq!(scanner.name(), "hidden_ascii_scanner");
        assert_eq!(scanner.block_threshold(), 1.0);
    }

    #[tokio::test]
    async fn test_no_hidden_ascii() {
        let scanner = HiddenASCIIScanner::new();
        let msg = Message::new(Role::User, "Normal text without any hidden characters");

        let result = scanner.scan(&msg, None).await.unwrap();
        assert_eq!(result.decision, ScanDecision::Allow);
    }

    #[tokio::test]
    async fn test_hidden_ascii_detection() {
        let scanner = HiddenASCIIScanner::new();

        // Create text with hidden ASCII tag (encode 'A')
        let hidden_char = char::from_u32(HIDDEN_ASCII_START + ('A' as u32)).unwrap();
        let msg = Message::new(Role::User, format!("Text with hidden: {}", hidden_char));

        let result = scanner.scan(&msg, None).await.unwrap();
        assert_eq!(result.decision, ScanDecision::Block);
        assert!(result.reason.contains("Hidden ASCII detected"));
        assert!(result.reason.contains("1 hidden characters"));
    }

    #[tokio::test]
    async fn test_decode_single_character() {
        // Encode 'X'
        let hidden_text = format!(
            "{}",
            char::from_u32(HIDDEN_ASCII_START + ('X' as u32)).unwrap()
        );
        let decoded = HiddenASCIIScanner::decode_hidden_ascii(&hidden_text);
        assert_eq!(decoded, "X");
    }

    #[tokio::test]
    async fn test_decode_word() {
        // Encode "Hi"
        let hidden_text: String = ['H', 'i']
            .iter()
            .map(|&c| char::from_u32(HIDDEN_ASCII_START + (c as u32)).unwrap())
            .collect();

        let decoded = HiddenASCIIScanner::decode_hidden_ascii(&hidden_text);
        assert_eq!(decoded, "Hi");
    }

    #[tokio::test]
    async fn test_decode_mixed_text() {
        // Mix normal and hidden text
        let hidden_a = char::from_u32(HIDDEN_ASCII_START + ('A' as u32)).unwrap();
        let text = format!("Normal{}Hidden", hidden_a);

        let decoded = HiddenASCIIScanner::decode_hidden_ascii(&text);
        assert_eq!(decoded, "NormalAHidden");
    }

    #[tokio::test]
    async fn test_multiple_hidden_chars() {
        let scanner = HiddenASCIIScanner::new();

        // Create text with multiple hidden characters
        let hidden_word: String = ['T', 'e', 's', 't']
            .iter()
            .map(|&c| char::from_u32(HIDDEN_ASCII_START + (c as u32)).unwrap())
            .collect();

        let msg = Message::new(Role::User, format!("Hidden: {}", hidden_word));

        let result = scanner.scan(&msg, None).await.unwrap();
        assert_eq!(result.decision, ScanDecision::Block);
        assert!(result.reason.contains("4 hidden characters"));
        assert!(result.reason.contains("Test"));
    }

    #[tokio::test]
    async fn test_contains_hidden_ascii_true() {
        let hidden_char = char::from_u32(HIDDEN_ASCII_START + ('A' as u32)).unwrap();
        let text = format!("Text {}", hidden_char);
        assert!(HiddenASCIIScanner::contains_hidden_ascii(&text));
    }

    #[tokio::test]
    async fn test_contains_hidden_ascii_false() {
        let text = "Normal text without hidden characters";
        assert!(!HiddenASCIIScanner::contains_hidden_ascii(&text));
    }

    #[tokio::test]
    async fn test_extract_hidden_chars() {
        let hidden_a = char::from_u32(HIDDEN_ASCII_START + ('A' as u32)).unwrap();
        let hidden_b = char::from_u32(HIDDEN_ASCII_START + ('B' as u32)).unwrap();
        let text = format!("Normal{}Text{}", hidden_a, hidden_b);

        let extracted = HiddenASCIIScanner::extract_hidden_chars(&text);
        assert_eq!(extracted.len(), 2);
        assert_eq!(extracted[0], hidden_a);
        assert_eq!(extracted[1], hidden_b);
    }

    #[tokio::test]
    async fn test_count_hidden_chars() {
        let hidden_chars: String = ['H', 'e', 'l', 'l', 'o']
            .iter()
            .map(|&c| char::from_u32(HIDDEN_ASCII_START + (c as u32)).unwrap())
            .collect();

        let text = format!("Before{}After", hidden_chars);
        let count = HiddenASCIIScanner::count_hidden_chars(&text);
        assert_eq!(count, 5);
    }

    #[tokio::test]
    async fn test_truncated_decoded_output() {
        let scanner = HiddenASCIIScanner::new();

        // Create a long hidden message (more than 100 chars)
        let long_message = "A".repeat(150);
        let hidden_text: String = long_message
            .chars()
            .map(|c| char::from_u32(HIDDEN_ASCII_START + (c as u32)).unwrap())
            .collect();

        let msg = Message::new(Role::User, hidden_text);

        let result = scanner.scan(&msg, None).await.unwrap();
        assert_eq!(result.decision, ScanDecision::Block);
        // Verify the decoded text is truncated
        assert!(result.reason.contains("..."));
        assert!(result.reason.len() < 300); // Should be truncated
    }

    #[tokio::test]
    async fn test_with_name() {
        let scanner = HiddenASCIIScanner::new().with_name("custom_scanner");
        assert_eq!(scanner.name(), "custom_scanner");
    }

    #[tokio::test]
    async fn test_with_block_threshold() {
        let scanner = HiddenASCIIScanner::new().with_block_threshold(0.75);
        assert_eq!(scanner.block_threshold(), 0.75);
    }

    #[tokio::test]
    async fn test_boundary_chars() {
        // Test the start and end of the hidden ASCII range
        let start_char = char::from_u32(HIDDEN_ASCII_START).unwrap();
        let end_char = char::from_u32(HIDDEN_ASCII_END).unwrap();

        let text = format!("{}{}", start_char, end_char);
        assert!(HiddenASCIIScanner::contains_hidden_ascii(&text));

        let count = HiddenASCIIScanner::count_hidden_chars(&text);
        assert_eq!(count, 2);
    }

    #[tokio::test]
    async fn test_outside_range() {
        // Test characters just outside the range
        let before_range = char::from_u32(HIDDEN_ASCII_START - 1).unwrap();
        let after_range = char::from_u32(HIDDEN_ASCII_END + 1).unwrap();

        let text = format!("{}{}", before_range, after_range);
        assert!(!HiddenASCIIScanner::contains_hidden_ascii(&text));
    }

    #[tokio::test]
    async fn test_empty_string() {
        let scanner = HiddenASCIIScanner::new();
        let msg = Message::new(Role::User, "");

        let result = scanner.scan(&msg, None).await.unwrap();
        assert_eq!(result.decision, ScanDecision::Allow);
    }

    #[tokio::test]
    async fn test_unicode_text_without_hidden() {
        let scanner = HiddenASCIIScanner::new();
        let msg = Message::new(Role::User, "Hello 世界 🌍 Привет");

        let result = scanner.scan(&msg, None).await.unwrap();
        assert_eq!(result.decision, ScanDecision::Allow);
    }

    #[tokio::test]
    async fn test_special_chars_decoding() {
        // Test decoding special ASCII characters like newline, tab, etc.
        let hidden_newline = char::from_u32(HIDDEN_ASCII_START + ('\n' as u32)).unwrap();
        let hidden_tab = char::from_u32(HIDDEN_ASCII_START + ('\t' as u32)).unwrap();

        let text = format!("{}{}", hidden_newline, hidden_tab);
        let decoded = HiddenASCIIScanner::decode_hidden_ascii(&text);

        assert_eq!(decoded, "\n\t");
    }

    #[tokio::test]
    async fn test_performance_early_termination() {
        // Test that contains_hidden_ascii terminates early
        let hidden_char = char::from_u32(HIDDEN_ASCII_START + ('X' as u32)).unwrap();
        let large_text = format!("{}{}", hidden_char, "A".repeat(10000));

        // Should find hidden char immediately without scanning entire string
        let start = std::time::Instant::now();
        let result = HiddenASCIIScanner::contains_hidden_ascii(&large_text);
        let duration = start.elapsed();

        assert!(result);
        // Should be very fast due to early termination
        assert!(duration.as_micros() < 1000);
    }
}
