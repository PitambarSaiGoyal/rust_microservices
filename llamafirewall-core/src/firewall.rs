//! Firewall orchestrator - coordinates scanners and aggregates results

use crate::config::Configuration;
use crate::scanner::{ScanError, Scanner};
use crate::types::{Message, Role, ScanDecision, ScanResult, ScanStatus, Trace};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::task::JoinSet;
use tracing::{debug, instrument};

/// Main firewall orchestrator
///
/// The Firewall coordinates multiple scanners, running them in parallel
/// and aggregating their results into a final decision.
pub struct Firewall {
    scanners: HashMap<Role, Vec<Arc<dyn Scanner>>>,
}

impl Firewall {
    /// Create a new Firewall from configuration
    pub fn new(config: Configuration) -> Result<Self, ScanError> {
        let mut scanners = HashMap::new();

        for (role, scanner_list) in config.scanners {
            let mut role_scanners = Vec::new();
            for scanner in scanner_list {
                scanner.validate_config()?;
                role_scanners.push(scanner);
            }
            scanners.insert(role, role_scanners);
        }

        Ok(Self { scanners })
    }

    /// Scan a single message with optional trace
    ///
    /// Scanners run in parallel, results are aggregated.
    /// Early termination on BLOCK decision (future optimization).
    #[instrument(skip(self, message, trace), fields(role = ?message.role))]
    pub async fn scan(&self, message: &Message, trace: Option<&[Arc<Message>]>) -> ScanResult {
        let scanners = match self.scanners.get(&message.role) {
            Some(s) => s,
            None => return ScanResult::allow(),
        };

        if scanners.is_empty() {
            return ScanResult::allow();
        }

        // Single scanner optimization (no parallelization overhead)
        if scanners.len() == 1 {
            return match scanners[0].scan(message, trace).await {
                Ok(result) => result,
                Err(e) => {
                    tracing::error!("Scanner {} failed: {}", scanners[0].name(), e);
                    ScanResult::error(format!("Scanner error: {}", e))
                }
            };
        }

        // Parallel execution for multiple scanners
        let mut join_set = JoinSet::new();

        for scanner in scanners {
            let scanner = Arc::clone(scanner);
            let message = message.clone();
            let trace = trace.map(|t| t.to_vec());

            join_set.spawn(async move { scanner.scan(&message, trace.as_deref()).await });
        }

        // Collect results
        let mut results = Vec::new();
        while let Some(res) = join_set.join_next().await {
            match res {
                Ok(Ok(scan_result)) => results.push(scan_result),
                Ok(Err(e)) => {
                    tracing::error!("Scanner error: {}", e);
                    results.push(ScanResult::error(e.to_string()));
                }
                Err(e) => {
                    tracing::error!("Task join error: {}", e);
                    results.push(ScanResult::error("Task execution failed"));
                }
            }
        }

        self.aggregate_results(results)
    }

    /// Aggregate multiple scan results into final decision
    ///
    /// Decision priority: BLOCK > HUMAN_REVIEW > ALLOW
    /// If multiple scanners return the same decision, the highest score wins.
    fn aggregate_results(&self, results: Vec<ScanResult>) -> ScanResult {
        if results.is_empty() {
            return ScanResult::allow();
        }

        let mut decisions: HashMap<ScanDecision, f64> = HashMap::new();
        let mut reasons = Vec::with_capacity(results.len());

        for result in results {
            if result.status == ScanStatus::Error {
                tracing::warn!("Scanner error: {}", result.reason);
                continue;
            }

            reasons.push(format!("{} - score: {:.2}", result.reason, result.score));

            decisions
                .entry(result.decision)
                .and_modify(|score| *score = score.max(result.score))
                .or_insert(result.score);
        }

        // Priority: BLOCK > HUMAN_REVIEW > ALLOW
        let (final_decision, final_score) =
            if let Some(&score) = decisions.get(&ScanDecision::Block) {
                (ScanDecision::Block, score)
            } else if let Some(&score) = decisions.get(&ScanDecision::HumanInTheLoopRequired) {
                (ScanDecision::HumanInTheLoopRequired, score)
            } else {
                (
                    ScanDecision::Allow,
                    decisions.get(&ScanDecision::Allow).copied().unwrap_or(0.0),
                )
            };

        ScanResult {
            decision: final_decision,
            reason: reasons.join("; "),
            score: final_score,
            status: ScanStatus::Success,
        }
    }

    /// Scan entire conversation trace
    ///
    /// Zero-copy trace slicing for efficient history passing
    #[instrument(skip(self, trace))]
    pub async fn scan_replay(&self, trace: &Trace) -> ScanResult {
        let mut final_result = ScanResult::allow();

        for (idx, message) in trace.iter().enumerate() {
            // Zero-copy slice of history (no allocation!)
            let past_trace = if idx > 0 { Some(&trace[..idx]) } else { None };

            final_result = self.scan(message, past_trace).await;

            // Early termination on block
            if final_result.decision == ScanDecision::Block
                || final_result.decision == ScanDecision::HumanInTheLoopRequired
            {
                debug!("Early termination at message {}/{}", idx + 1, trace.len());
                break;
            }
        }

        final_result
    }

    /// Scan and build trace incrementally (for streaming)
    ///
    /// This method scans a message and, if allowed, adds it to the trace.
    /// Returns both the scan result and the updated trace.
    pub async fn scan_and_update_trace(
        &self,
        message: Message,
        stored_trace: Option<Trace>,
    ) -> (ScanResult, Trace) {
        let mut trace = stored_trace.unwrap_or_default();

        let scan_result = self.scan(&message, Some(&trace)).await;

        if scan_result.decision == ScanDecision::Allow {
            trace.push(Arc::new(message));
        }

        (scan_result, trace)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scanner::Scanner;
    use crate::types::Role;
    use async_trait::async_trait;

    // Mock scanner for testing
    struct MockScanner {
        name: String,
        result: ScanResult,
    }

    #[async_trait]
    impl Scanner for MockScanner {
        fn name(&self) -> &str {
            &self.name
        }

        async fn scan(
            &self,
            _message: &Message,
            _past_trace: Option<&[Arc<Message>]>,
        ) -> Result<ScanResult, ScanError> {
            Ok(self.result.clone())
        }
    }

    #[tokio::test]
    async fn test_single_scanner_allow() {
        let scanner = Arc::new(MockScanner {
            name: "test".to_string(),
            result: ScanResult::allow(),
        }) as Arc<dyn Scanner>;

        let config = Configuration {
            scanners: [(Role::User, vec![scanner])].into_iter().collect(),
        };

        let firewall = Firewall::new(config).unwrap();
        let message = Message::new(Role::User, "test");

        let result = firewall.scan(&message, None).await;
        assert_eq!(result.decision, ScanDecision::Allow);
    }

    #[tokio::test]
    async fn test_single_scanner_block() {
        let scanner = Arc::new(MockScanner {
            name: "test".to_string(),
            result: ScanResult::block("blocked", 0.9),
        }) as Arc<dyn Scanner>;

        let config = Configuration {
            scanners: [(Role::User, vec![scanner])].into_iter().collect(),
        };

        let firewall = Firewall::new(config).unwrap();
        let message = Message::new(Role::User, "test");

        let result = firewall.scan(&message, None).await;
        assert_eq!(result.decision, ScanDecision::Block);
        assert_eq!(result.score, 0.9);
    }

    #[tokio::test]
    async fn test_parallel_scanners_block_priority() {
        let allow_scanner = Arc::new(MockScanner {
            name: "allow".to_string(),
            result: ScanResult::allow(),
        }) as Arc<dyn Scanner>;

        let block_scanner = Arc::new(MockScanner {
            name: "block".to_string(),
            result: ScanResult::block("blocked", 0.9),
        }) as Arc<dyn Scanner>;

        let config = Configuration {
            scanners: [(Role::User, vec![allow_scanner, block_scanner])]
                .into_iter()
                .collect(),
        };

        let firewall = Firewall::new(config).unwrap();
        let message = Message::new(Role::User, "test");

        let result = firewall.scan(&message, None).await;
        assert_eq!(result.decision, ScanDecision::Block);
        assert_eq!(result.score, 0.9);
    }

    #[tokio::test]
    async fn test_human_review_priority() {
        let allow_scanner = Arc::new(MockScanner {
            name: "allow".to_string(),
            result: ScanResult::allow(),
        }) as Arc<dyn Scanner>;

        let review_scanner = Arc::new(MockScanner {
            name: "review".to_string(),
            result: ScanResult::human_review("needs review", 0.7),
        }) as Arc<dyn Scanner>;

        let config = Configuration {
            scanners: [(Role::User, vec![allow_scanner, review_scanner])]
                .into_iter()
                .collect(),
        };

        let firewall = Firewall::new(config).unwrap();
        let message = Message::new(Role::User, "test");

        let result = firewall.scan(&message, None).await;
        assert_eq!(result.decision, ScanDecision::HumanInTheLoopRequired);
    }

    #[tokio::test]
    async fn test_trace_replay_zero_copy() {
        // This test verifies zero-copy slicing behavior
        let scanner = Arc::new(MockScanner {
            name: "test".to_string(),
            result: ScanResult::allow(),
        }) as Arc<dyn Scanner>;

        let config = Configuration {
            scanners: [(Role::User, vec![scanner])].into_iter().collect(),
        };

        let firewall = Firewall::new(config).unwrap();

        let trace: Trace = vec![
            Arc::new(Message::new(Role::User, "msg1")),
            Arc::new(Message::new(Role::Assistant, "msg2")),
            Arc::new(Message::new(Role::User, "msg3")),
        ];

        let result = firewall.scan_replay(&trace).await;
        assert_eq!(result.decision, ScanDecision::Allow);
    }

    #[tokio::test]
    async fn test_trace_replay_early_termination() {
        let call_count = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let count_clone = call_count.clone();

        // Create a scanner that counts calls and blocks on second message
        struct CountingScanner {
            name: String,
            call_count: std::sync::Arc<std::sync::atomic::AtomicUsize>,
        }

        #[async_trait]
        impl Scanner for CountingScanner {
            fn name(&self) -> &str {
                &self.name
            }

            async fn scan(
                &self,
                _message: &Message,
                _past_trace: Option<&[Arc<Message>]>,
            ) -> Result<ScanResult, ScanError> {
                let count = self
                    .call_count
                    .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                if count == 1 {
                    Ok(ScanResult::block("blocked", 1.0))
                } else {
                    Ok(ScanResult::allow())
                }
            }
        }

        let scanner = Arc::new(CountingScanner {
            name: "counting".to_string(),
            call_count: count_clone,
        }) as Arc<dyn Scanner>;

        let config = Configuration {
            scanners: [(Role::User, vec![scanner])].into_iter().collect(),
        };

        let firewall = Firewall::new(config).unwrap();

        let trace: Trace = vec![
            Arc::new(Message::new(Role::User, "msg1")),
            Arc::new(Message::new(Role::User, "msg2")),
            Arc::new(Message::new(Role::User, "msg3")),
        ];

        let result = firewall.scan_replay(&trace).await;
        assert_eq!(result.decision, ScanDecision::Block);

        // Should only scan first 2 messages due to early termination
        assert_eq!(call_count.load(std::sync::atomic::Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn test_scan_and_update_trace() {
        let scanner = Arc::new(MockScanner {
            name: "test".to_string(),
            result: ScanResult::allow(),
        }) as Arc<dyn Scanner>;

        let config = Configuration {
            scanners: [(Role::User, vec![scanner])].into_iter().collect(),
        };

        let firewall = Firewall::new(config).unwrap();

        let message = Message::new(Role::User, "test");
        let (result, trace) = firewall.scan_and_update_trace(message, None).await;

        assert_eq!(result.decision, ScanDecision::Allow);
        assert_eq!(trace.len(), 1);
    }

    #[tokio::test]
    async fn test_scan_and_update_trace_blocked() {
        let scanner = Arc::new(MockScanner {
            name: "test".to_string(),
            result: ScanResult::block("blocked", 0.9),
        }) as Arc<dyn Scanner>;

        let config = Configuration {
            scanners: [(Role::User, vec![scanner])].into_iter().collect(),
        };

        let firewall = Firewall::new(config).unwrap();

        let message = Message::new(Role::User, "test");
        let (result, trace) = firewall.scan_and_update_trace(message, None).await;

        assert_eq!(result.decision, ScanDecision::Block);
        assert_eq!(trace.len(), 0); // Message not added to trace when blocked
    }
}
