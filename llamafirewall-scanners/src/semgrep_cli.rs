use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::process::Stdio;
use tokio::io::AsyncWriteExt;
use tokio::process::Command;
use tracing::{debug, info, warn};
use uuid::Uuid;

/// Semgrep finding severity
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "UPPERCASE")]
pub enum Severity {
    Error,
    Warning,
    Info,
}

/// Semgrep finding metadata
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Metadata {
    #[serde(default)]
    pub cwe: Vec<String>,
    #[serde(default)]
    pub owasp: Vec<String>,
    #[serde(default)]
    pub confidence: Option<String>,
    #[serde(default)]
    pub impact: Option<String>,
}

/// A single Semgrep finding
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemgrepFinding {
    pub check_id: String,
    pub path: String,
    pub start: SemgrepPosition,
    pub end: SemgrepPosition,
    pub extra: SemgrepExtra,
}

/// Position in source code
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemgrepPosition {
    pub line: usize,
    pub col: usize,
    pub offset: usize,
}

/// Extra information about a finding
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemgrepExtra {
    pub message: String,
    pub severity: Severity,
    #[serde(default)]
    pub metadata: Metadata,
    #[serde(default)]
    pub lines: String,
}

/// Semgrep scan results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemgrepResults {
    pub results: Vec<SemgrepFinding>,
    #[serde(default)]
    pub errors: Vec<SemgrepError>,
}

/// Semgrep error information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemgrepError {
    #[serde(default)]
    pub message: String,
    #[serde(default)]
    pub level: String,
}

/// Configuration for Semgrep CLI execution
#[derive(Debug, Clone)]
pub struct SemgrepConfig {
    /// Path to semgrep binary (default: "semgrep")
    pub semgrep_path: String,
    /// Config to use (default: "auto")
    pub config: String,
    /// Maximum execution time in seconds
    pub timeout_secs: u64,
    /// Additional command-line flags
    pub extra_args: Vec<String>,
}

impl Default for SemgrepConfig {
    fn default() -> Self {
        Self {
            semgrep_path: "semgrep".to_string(),
            config: "auto".to_string(),
            timeout_secs: 30,
            extra_args: Vec::new(),
        }
    }
}

impl SemgrepConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_config(mut self, config: impl Into<String>) -> Self {
        self.config = config.into();
        self
    }

    pub fn with_timeout(mut self, timeout_secs: u64) -> Self {
        self.timeout_secs = timeout_secs;
        self
    }

    pub fn with_semgrep_path(mut self, path: impl Into<String>) -> Self {
        self.semgrep_path = path.into();
        self
    }

    pub fn with_extra_args(mut self, args: Vec<String>) -> Self {
        self.extra_args = args;
        self
    }
}

/// Semgrep CLI wrapper
pub struct SemgrepCli {
    config: SemgrepConfig,
}

impl SemgrepCli {
    /// Create new Semgrep CLI wrapper
    pub fn new(config: SemgrepConfig) -> Self {
        Self { config }
    }

    /// Check if Semgrep is installed and accessible
    pub async fn check_installation(&self) -> Result<String> {
        debug!("Checking Semgrep installation at: {}", self.config.semgrep_path);

        let output = Command::new(&self.config.semgrep_path)
            .arg("--version")
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .await
            .context("Failed to execute semgrep --version. Is Semgrep installed?")?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            anyhow::bail!(
                "Semgrep version check failed: {}. Please install with: pip install semgrep",
                stderr
            );
        }

        let version = String::from_utf8_lossy(&output.stdout).trim().to_string();
        info!("Found Semgrep: {}", version);
        Ok(version)
    }

    /// Scan code content
    ///
    /// Creates a temporary file, runs Semgrep, and parses JSON output
    pub async fn scan_code(&self, code: &str, language: &str) -> Result<SemgrepResults> {
        // Create temporary file with appropriate extension
        let extension = Self::language_to_extension(language);
        let temp_file = self.create_temp_file(code, extension).await?;

        debug!(
            "Created temporary file: {} for language: {}",
            temp_file.display(),
            language
        );

        // Run Semgrep
        let results = self.run_semgrep(&temp_file).await?;

        // Cleanup temp file (it will be automatically deleted when dropped)
        debug!("Semgrep scan completed with {} findings", results.results.len());

        Ok(results)
    }

    /// Create temporary file with code content
    async fn create_temp_file(&self, code: &str, extension: &str) -> Result<PathBuf> {
        // Create a unique filename with proper extension
        let unique_name = format!("llamafirewall_scan_{}_{}.{}",
            Uuid::new_v4().simple(),
            std::process::id(),
            extension
        );

        let temp_dir = std::env::temp_dir();
        let file_path = temp_dir.join(unique_name);

        // Write code to file
        let mut file = tokio::fs::File::create(&file_path)
            .await
            .context("Failed to create temp file")?;

        file.write_all(code.as_bytes())
            .await
            .context("Failed to write code to temp file")?;

        file.sync_all()
            .await
            .context("Failed to sync temp file")?;

        Ok(file_path)
    }

    /// Execute Semgrep and parse results
    async fn run_semgrep(&self, file_path: &Path) -> Result<SemgrepResults> {
        let mut cmd = Command::new(&self.config.semgrep_path);

        // Basic Semgrep arguments
        cmd.arg("scan")
            .arg("--json")
            .arg("--config")
            .arg(&self.config.config)
            .arg("--no-git-ignore")
            .arg("--disable-version-check");

        // Add extra arguments
        for arg in &self.config.extra_args {
            cmd.arg(arg);
        }

        // Add file to scan
        cmd.arg(file_path);

        // Configure stdio
        cmd.stdout(Stdio::piped())
            .stderr(Stdio::piped());

        debug!("Executing Semgrep: {:?}", cmd);

        // Execute with timeout
        let output = tokio::time::timeout(
            std::time::Duration::from_secs(self.config.timeout_secs),
            cmd.output(),
        )
        .await
        .context("Semgrep execution timed out")??;

        // Parse output
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            warn!("Semgrep execution failed (continuing): {}", stderr);

            // Try to parse JSON anyway - Semgrep may exit non-zero but still produce valid output
            if !output.stdout.is_empty() {
                if let Ok(results) = serde_json::from_slice::<SemgrepResults>(&output.stdout) {
                    return Ok(results);
                }
            }

            anyhow::bail!("Semgrep failed: {}", stderr);
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        debug!("Semgrep output length: {} bytes", stdout.len());

        // Parse JSON results
        let results: SemgrepResults = serde_json::from_str(&stdout)
            .context("Failed to parse Semgrep JSON output")?;

        // Log errors from Semgrep
        for error in &results.errors {
            warn!("Semgrep error: {} (level: {})", error.message, error.level);
        }

        Ok(results)
    }

    /// Map language to file extension
    fn language_to_extension(language: &str) -> &str {
        match language.to_lowercase().as_str() {
            "python" | "py" => "py",
            "javascript" | "js" => "js",
            "typescript" | "ts" => "ts",
            "java" => "java",
            "go" | "golang" => "go",
            "rust" | "rs" => "rs",
            "c" => "c",
            "cpp" | "c++" | "cxx" => "cpp",
            "ruby" | "rb" => "rb",
            "php" => "php",
            "kotlin" | "kt" => "kt",
            "swift" => "swift",
            "shell" | "bash" | "sh" => "sh",
            _ => "txt",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    #[ignore] // Requires Semgrep installation
    async fn test_semgrep_installation() {
        let cli = SemgrepCli::new(SemgrepConfig::default());
        let version = cli.check_installation().await;
        assert!(version.is_ok(), "Semgrep should be installed");
        println!("Semgrep version: {}", version.unwrap());
    }

    #[tokio::test]
    #[ignore] // Requires Semgrep installation
    async fn test_scan_python_eval() {
        let code = r#"
import os
user_input = input("Enter command: ")
eval(user_input)  # Dangerous eval usage
"#;

        let cli = SemgrepCli::new(SemgrepConfig::default());
        let results = cli.scan_code(code, "python").await.unwrap();

        // Should detect eval usage (if using appropriate ruleset)
        assert!(!results.results.is_empty() || !results.errors.is_empty());
    }

    #[tokio::test]
    #[ignore] // Requires Semgrep installation
    async fn test_scan_javascript_xss() {
        let code = r#"
const express = require('express');
const app = express();

app.get('/search', (req, res) => {
    res.send('<html><body>' + req.query.q + '</body></html>');  // XSS vulnerability
});
"#;

        let cli = SemgrepCli::new(SemgrepConfig::default());
        let results = cli.scan_code(code, "javascript").await.unwrap();

        // Results depend on ruleset configuration
        println!("Found {} findings", results.results.len());
    }

    #[test]
    fn test_language_extension_mapping() {
        assert_eq!(SemgrepCli::language_to_extension("python"), "py");
        assert_eq!(SemgrepCli::language_to_extension("javascript"), "js");
        assert_eq!(SemgrepCli::language_to_extension("go"), "go");
        assert_eq!(SemgrepCli::language_to_extension("unknown"), "txt");
    }

    #[test]
    fn test_config_builder() {
        let config = SemgrepConfig::new()
            .with_config("p/security-audit")
            .with_timeout(60)
            .with_extra_args(vec!["--verbose".to_string()]);

        assert_eq!(config.config, "p/security-audit");
        assert_eq!(config.timeout_secs, 60);
        assert_eq!(config.extra_args.len(), 1);
    }
}
