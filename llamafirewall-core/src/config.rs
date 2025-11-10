//! Configuration types for the Firewall

use crate::scanner::Scanner;
use crate::types::Role;
use std::collections::HashMap;
use std::sync::Arc;

/// Firewall configuration
///
/// Configuration defines which scanners should be applied to which message roles.
/// Multiple scanners can be configured for each role and will run in parallel.
pub struct Configuration {
    pub scanners: HashMap<Role, Vec<Arc<dyn Scanner>>>,
}

impl Configuration {
    /// Create new empty configuration
    pub fn new() -> Self {
        Self {
            scanners: HashMap::new(),
        }
    }

    /// Add scanner for a specific role
    ///
    /// # Example
    /// ```rust,ignore
    /// let config = Configuration::new()
    ///     .add_scanner(Role::User, Arc::new(regex_scanner));
    /// ```
    pub fn add_scanner(mut self, role: Role, scanner: Arc<dyn Scanner>) -> Self {
        self.scanners.entry(role).or_default().push(scanner);
        self
    }

    /// Add multiple scanners for a role
    pub fn add_scanners(mut self, role: Role, scanners: Vec<Arc<dyn Scanner>>) -> Self {
        self.scanners.entry(role).or_default().extend(scanners);
        self
    }

    /// Get default configuration (matches Python defaults)
    ///
    /// Note: Returns empty configuration for now. Will be populated
    /// as scanners are implemented in later phases.
    pub fn default_config() -> Self {
        Self::new()
    }
}

impl Default for Configuration {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scanner::{ScanError, Scanner};
    use crate::types::{Message, ScanResult};
    use async_trait::async_trait;

    struct TestScanner {
        name: String,
    }

    #[async_trait]
    impl Scanner for TestScanner {
        fn name(&self) -> &str {
            &self.name
        }

        async fn scan(
            &self,
            _message: &Message,
            _past_trace: Option<&[Arc<Message>]>,
        ) -> Result<ScanResult, ScanError> {
            Ok(ScanResult::allow())
        }
    }

    #[test]
    fn test_config_creation() {
        let config = Configuration::new();
        assert!(config.scanners.is_empty());
    }

    #[test]
    fn test_add_scanner() {
        let scanner = Arc::new(TestScanner {
            name: "test".to_string(),
        }) as Arc<dyn Scanner>;

        let config = Configuration::new().add_scanner(Role::User, scanner);

        assert_eq!(config.scanners.len(), 1);
        assert!(config.scanners.contains_key(&Role::User));
        assert_eq!(config.scanners[&Role::User].len(), 1);
    }

    #[test]
    fn test_add_multiple_scanners() {
        let scanner1 = Arc::new(TestScanner {
            name: "test1".to_string(),
        }) as Arc<dyn Scanner>;

        let scanner2 = Arc::new(TestScanner {
            name: "test2".to_string(),
        }) as Arc<dyn Scanner>;

        let config = Configuration::new()
            .add_scanner(Role::User, scanner1)
            .add_scanner(Role::User, scanner2);

        assert_eq!(config.scanners[&Role::User].len(), 2);
    }

    #[test]
    fn test_add_scanners_batch() {
        let scanners: Vec<Arc<dyn Scanner>> = vec![
            Arc::new(TestScanner {
                name: "test1".to_string(),
            }),
            Arc::new(TestScanner {
                name: "test2".to_string(),
            }),
        ];

        let config = Configuration::new().add_scanners(Role::User, scanners);

        assert_eq!(config.scanners[&Role::User].len(), 2);
    }

    #[test]
    fn test_multiple_roles() {
        let scanner1 = Arc::new(TestScanner {
            name: "test1".to_string(),
        }) as Arc<dyn Scanner>;

        let scanner2 = Arc::new(TestScanner {
            name: "test2".to_string(),
        }) as Arc<dyn Scanner>;

        let config = Configuration::new()
            .add_scanner(Role::User, scanner1)
            .add_scanner(Role::Assistant, scanner2);

        assert_eq!(config.scanners.len(), 2);
        assert!(config.scanners.contains_key(&Role::User));
        assert!(config.scanners.contains_key(&Role::Assistant));
    }
}
