use crate::semgrep_cli::{SemgrepCli, SemgrepConfig, SemgrepFinding, Severity};
use async_trait::async_trait;
use llamafirewall_core::{
    scanner::{ScanError, Scanner},
    types::{Message, ScanResult},
};
use std::sync::Arc;
use tokio::sync::OnceCell;
use tracing::{debug, info, warn};

/// Language detection from code content
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Language {
    Python,
    JavaScript,
    TypeScript,
    Java,
    Go,
    Rust,
    C,
    Cpp,
    Ruby,
    Php,
    Kotlin,
    Swift,
    Shell,
    Unknown,
}

impl Language {
    /// Detect language from code heuristics
    pub fn detect(code: &str) -> Self {
        // Simple heuristic-based detection (order matters - check more specific patterns first)

        // Rust: check before JavaScript as both use "let"
        if code.contains("fn ") && (code.contains("let ") || code.contains("let mut ") || code.contains("impl ")) {
            return Self::Rust;
        }

        // Python
        if code.contains("def ") || (code.contains("import ") && code.contains("__init__")) {
            return Self::Python;
        }

        // TypeScript: check before JavaScript
        if code.contains("interface ") && (code.contains(": ") || code.contains("<T>")) {
            return Self::TypeScript;
        }

        // JavaScript
        if code.contains("function ") || code.contains("const ") || (code.contains("let ") && !code.contains("fn ")) {
            return Self::JavaScript;
        }

        // Java
        if code.contains("public class ") || (code.contains("private ") && code.contains("void ")) {
            return Self::Java;
        }

        // Go
        if code.contains("func ") && code.contains("package ") {
            return Self::Go;
        }

        // Swift: check before other languages with "func"
        if code.contains("func ") && (code.contains("var ") || code.contains("let ")) && !code.contains("fn ") {
            return Self::Swift;
        }

        // C
        if code.contains("#include") && code.contains("<stdio.h>") {
            return Self::C;
        }

        // C++
        if code.contains("#include") && code.contains("<iostream>") {
            return Self::Cpp;
        }

        // Ruby
        if code.contains("def ") && code.contains("end") && code.contains("require ") {
            return Self::Ruby;
        }

        // PHP
        if code.contains("<?php") || code.contains("$this->") {
            return Self::Php;
        }

        // Kotlin
        if code.contains("fun ") && code.contains("val ") {
            return Self::Kotlin;
        }

        // Shell
        if code.contains("#!/bin/bash") || code.contains("#!/bin/sh") {
            return Self::Shell;
        }

        Self::Unknown
    }

    /// Convert to string representation
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Python => "python",
            Self::JavaScript => "javascript",
            Self::TypeScript => "typescript",
            Self::Java => "java",
            Self::Go => "go",
            Self::Rust => "rust",
            Self::C => "c",
            Self::Cpp => "cpp",
            Self::Ruby => "ruby",
            Self::Php => "php",
            Self::Kotlin => "kotlin",
            Self::Swift => "swift",
            Self::Shell => "shell",
            Self::Unknown => "unknown",
        }
    }
}

/// CodeShield scanner using Semgrep CLI
pub struct CodeShieldScanner {
    name: String,
    cli: Arc<OnceCell<SemgrepCli>>,
    config: SemgrepConfig,
    block_threshold: f64,
    /// Minimum severity to trigger a block (Error, Warning, or Info)
    min_severity: Severity,
    /// Whether to auto-detect language
    auto_detect_language: bool,
    /// Explicit language override (if None, auto-detection is used)
    language_override: Option<String>,
}

impl CodeShieldScanner {
    /// Create new CodeShield scanner with default configuration
    pub fn new() -> Self {
        Self {
            name: "code_shield".to_string(),
            cli: Arc::new(OnceCell::new()),
            config: SemgrepConfig::default(),
            block_threshold: 1.0,
            min_severity: Severity::Error,
            auto_detect_language: true,
            language_override: None,
        }
    }

    /// Create scanner with custom configuration
    pub fn with_config(mut self, config: SemgrepConfig) -> Self {
        self.config = config;
        self
    }

    /// Set block threshold (0.0 - 1.0)
    pub fn with_block_threshold(mut self, threshold: f64) -> Self {
        self.block_threshold = threshold;
        self
    }

    /// Set minimum severity to trigger blocking
    pub fn with_min_severity(mut self, severity: Severity) -> Self {
        self.min_severity = severity;
        self
    }

    /// Set custom scanner name
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Override language detection
    pub fn with_language(mut self, language: impl Into<String>) -> Self {
        self.language_override = Some(language.into());
        self.auto_detect_language = false;
        self
    }

    /// Lazy-load Semgrep CLI
    async fn get_cli(&self) -> Result<&SemgrepCli, ScanError> {
        self.cli
            .get_or_try_init(|| async {
                info!("Initializing Semgrep CLI for CodeShield scanner");
                let cli = SemgrepCli::new(self.config.clone());

                // Verify Semgrep is installed
                cli.check_installation()
                    .await
                    .map_err(|e| ScanError::ConfigError(format!("Semgrep not found: {}", e)))?;

                Ok(cli)
            })
            .await
    }

    /// Determine language for code scanning
    fn determine_language(&self, code: &str) -> String {
        if let Some(ref lang) = self.language_override {
            return lang.clone();
        }

        if self.auto_detect_language {
            Language::detect(code).as_str().to_string()
        } else {
            "unknown".to_string()
        }
    }

    /// Format findings into human-readable reason
    fn format_findings(&self, findings: &[SemgrepFinding]) -> String {
        if findings.is_empty() {
            return "No security issues found".to_string();
        }

        let mut reasons = Vec::new();

        for (idx, finding) in findings.iter().take(5).enumerate() {
            let severity_str = match finding.extra.severity {
                Severity::Error => "ERROR",
                Severity::Warning => "WARNING",
                Severity::Info => "INFO",
            };

            let cwe_info = if !finding.extra.metadata.cwe.is_empty() {
                format!(" (CWE: {})", finding.extra.metadata.cwe.join(", "))
            } else {
                String::new()
            };

            reasons.push(format!(
                "{}. [{severity_str}] {} at line {}{}: {}",
                idx + 1,
                finding.check_id,
                finding.start.line,
                cwe_info,
                finding.extra.message.lines().next().unwrap_or("")
            ));
        }

        if findings.len() > 5 {
            reasons.push(format!("... and {} more findings", findings.len() - 5));
        }

        reasons.join("; ")
    }

    /// Calculate severity score
    fn calculate_score(&self, findings: &[SemgrepFinding]) -> f64 {
        if findings.is_empty() {
            return 0.0;
        }

        let mut error_count = 0;
        let mut warning_count = 0;
        let mut info_count = 0;

        for finding in findings {
            match finding.extra.severity {
                Severity::Error => error_count += 1,
                Severity::Warning => warning_count += 1,
                Severity::Info => info_count += 1,
            }
        }

        // Weight: Error = 1.0, Warning = 0.6, Info = 0.3
        // Normalize to 0.0-1.0 range (cap at 1.0)
        let weighted_score = (error_count as f64 * 1.0
            + warning_count as f64 * 0.6
            + info_count as f64 * 0.3)
            / 5.0; // Normalize assuming 5 findings = 1.0

        weighted_score.min(1.0)
    }

    /// Filter findings by minimum severity
    fn filter_by_severity(&self, findings: Vec<SemgrepFinding>) -> Vec<SemgrepFinding> {
        findings
            .into_iter()
            .filter(|f| self.should_include_severity(&f.extra.severity))
            .collect()
    }

    /// Check if severity meets minimum threshold
    fn should_include_severity(&self, severity: &Severity) -> bool {
        match (&self.min_severity, severity) {
            (Severity::Error, Severity::Error) => true,
            (Severity::Error, _) => false,
            (Severity::Warning, Severity::Error | Severity::Warning) => true,
            (Severity::Warning, Severity::Info) => false,
            (Severity::Info, _) => true,
        }
    }
}

impl Default for CodeShieldScanner {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Scanner for CodeShieldScanner {
    fn name(&self) -> &str {
        &self.name
    }

    async fn scan(
        &self,
        message: &Message,
        _past_trace: Option<&[Arc<Message>]>,
    ) -> Result<ScanResult, ScanError> {
        let code = &message.content;

        // Skip empty content
        if code.trim().is_empty() {
            return Ok(ScanResult::allow());
        }

        // Get CLI instance
        let cli = self.get_cli().await?;

        // Determine language
        let language = self.determine_language(code);
        debug!("Detected language: {}", language);

        // Scan code with Semgrep
        let results = cli
            .scan_code(code, &language)
            .await
            .map_err(|e| ScanError::InferenceError(format!("Semgrep scan failed: {}", e)))?;

        // Log Semgrep errors (non-fatal)
        for error in &results.errors {
            warn!("Semgrep error: {} ({})", error.message, error.level);
        }

        // Filter findings by severity
        let filtered_findings = self.filter_by_severity(results.results);

        debug!("Semgrep found {} relevant findings", filtered_findings.len());

        // Calculate score
        let score = self.calculate_score(&filtered_findings);

        // Decide based on findings
        if !filtered_findings.is_empty() && score >= self.block_threshold {
            let reason = self.format_findings(&filtered_findings);
            Ok(ScanResult::block(reason, score))
        } else {
            Ok(ScanResult::allow())
        }
    }

    fn block_threshold(&self) -> f64 {
        self.block_threshold
    }

    fn validate_config(&self) -> Result<(), ScanError> {
        // Validation happens during lazy initialization
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use llamafirewall_core::types::{Role, ScanDecision};

    #[test]
    fn test_language_detection() {
        assert_eq!(
            Language::detect("def hello():\n    print('hi')"),
            Language::Python
        );
        assert_eq!(
            Language::detect("function test() { const x = 1; }"),
            Language::JavaScript
        );
        assert_eq!(
            Language::detect("fn main() { let x = 5; }"),
            Language::Rust
        );
    }

    #[test]
    fn test_scanner_builder() {
        let scanner = CodeShieldScanner::new()
            .with_name("custom_codeshield")
            .with_block_threshold(0.8)
            .with_min_severity(Severity::Warning)
            .with_language("python");

        assert_eq!(scanner.name(), "custom_codeshield");
        assert_eq!(scanner.block_threshold, 0.8);
        assert_eq!(scanner.min_severity, Severity::Warning);
    }

    #[tokio::test]
    async fn test_scan_empty_code() {
        let scanner = CodeShieldScanner::new();
        let msg = Message::new(Role::User, "   ");

        let result = scanner.scan(&msg, None).await.unwrap();
        assert_eq!(result.decision, ScanDecision::Allow);
    }

    #[tokio::test]
    #[ignore] // Requires Semgrep installation
    async fn test_scan_safe_python_code() {
        let scanner = CodeShieldScanner::new().with_language("python");
        let code = r#"
def greet(name):
    return f"Hello, {name}!"

print(greet("World"))
"#;
        let msg = Message::new(Role::User, code);

        let result = scanner.scan(&msg, None).await.unwrap();
        // Safe code should pass (depending on ruleset)
        println!("Result: {:?}", result);
    }

    #[test]
    fn test_severity_filtering() {
        let scanner = CodeShieldScanner::new().with_min_severity(Severity::Warning);

        assert!(scanner.should_include_severity(&Severity::Error));
        assert!(scanner.should_include_severity(&Severity::Warning));
        assert!(!scanner.should_include_severity(&Severity::Info));
    }

    #[test]
    fn test_score_calculation() {
        let scanner = CodeShieldScanner::new();

        // No findings
        let findings = vec![];
        assert_eq!(scanner.calculate_score(&findings), 0.0);

        // Test score is in valid range
        // Note: actual findings would need proper SemgrepFinding structs
    }
}
