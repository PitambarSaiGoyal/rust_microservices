//! Stress Testing Framework for LlamaFirewall ML Backends
//!
//! Tests system reliability under load, memory leak detection,
//! and long-running stability.

#[cfg(test)]
mod stress_tests {
    use std::sync::{Arc, Mutex};
    use std::thread;
    use std::time::{Duration, Instant};

    /// Memory usage tracker
    #[derive(Clone, Default)]
    struct MemoryStats {
        initial: usize,
        peak: usize,
        final_: usize,
        measurements: Vec<usize>,
    }

    impl MemoryStats {
        fn record(&mut self, usage: usize) {
            if self.measurements.is_empty() {
                self.initial = usage;
            }
            self.peak = self.peak.max(usage);
            self.final_ = usage;
            self.measurements.push(usage);
        }

        fn growth_rate(&self) -> f64 {
            if self.measurements.len() < 2 {
                return 0.0;
            }

            // Calculate linear regression slope
            let n = self.measurements.len() as f64;
            let x_sum: f64 = (0..self.measurements.len()).map(|i| i as f64).sum();
            let y_sum: f64 = self.measurements.iter().map(|&y| y as f64).sum();
            let xy_sum: f64 = self
                .measurements
                .iter()
                .enumerate()
                .map(|(i, &y)| i as f64 * y as f64)
                .sum();
            let x2_sum: f64 = (0..self.measurements.len()).map(|i| (i * i) as f64).sum();

            let slope = (n * xy_sum - x_sum * y_sum) / (n * x2_sum - x_sum * x_sum);
            slope
        }

        fn growth_percentage(&self) -> f64 {
            if self.initial == 0 {
                return 0.0;
            }
            100.0 * (self.final_ as f64 - self.initial as f64) / self.initial as f64
        }
    }

    #[cfg(target_os = "macos")]
    fn get_process_memory_mb() -> usize {
        use std::process::Command;

        let output = Command::new("ps")
            .args(["-o", "rss=", "-p", &std::process::id().to_string()])
            .output()
            .expect("Failed to run ps command");

        let rss_kb = String::from_utf8_lossy(&output.stdout)
            .trim()
            .parse::<usize>()
            .unwrap_or(0);

        rss_kb / 1024 // Convert KB to MB
    }

    #[cfg(target_os = "linux")]
    fn get_process_memory_mb() -> usize {
        use std::fs;

        let status = fs::read_to_string("/proc/self/status").unwrap_or_default();
        for line in status.lines() {
            if line.starts_with("VmRSS:") {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 2 {
                    if let Ok(kb) = parts[1].parse::<usize>() {
                        return kb / 1024; // Convert KB to MB
                    }
                }
            }
        }
        0
    }

    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    fn get_process_memory_mb() -> usize {
        0 // Not supported on this platform
    }

    #[test]
    #[ignore] // Run with: cargo test --test stress_tests -- --ignored
    #[cfg(feature = "onnx-backend")]
    fn test_onnx_memory_leak_detection() {
        use llamafirewall_ml::inference::OnnxInferenceEngine;

        let model_path = "models/promptguard.onnx";
        let tokenizer_path = "models/tokenizer.json";

        if !std::path::Path::new(model_path).exists() {
            eprintln!("Skipping test: model file not found");
            return;
        }

        let engine = OnnxInferenceEngine::new(model_path, tokenizer_path)
            .expect("Failed to create ONNX engine");

        let test_prompts = vec![
            "Test prompt 1",
            "Test prompt 2 with more text to make it longer",
            "Short",
            "A".repeat(100).as_str(),
        ];

        let iterations = 1000;
        let sample_interval = 100;
        let mut stats = MemoryStats::default();

        println!("Running {} iterations...", iterations);

        for i in 0..iterations {
            let prompt = test_prompts[i % test_prompts.len()];
            let _ = engine.infer(prompt).expect("Inference failed");

            // Sample memory usage
            if i % sample_interval == 0 {
                let mem_mb = get_process_memory_mb();
                stats.record(mem_mb);
                println!(
                    "Iteration {}/{}: {}MB (growth: {:.2}%)",
                    i,
                    iterations,
                    mem_mb,
                    stats.growth_percentage()
                );
            }
        }

        // Final measurement
        stats.record(get_process_memory_mb());

        println!("\nMemory Statistics:");
        println!("  Initial: {}MB", stats.initial);
        println!("  Peak:    {}MB", stats.peak);
        println!("  Final:   {}MB", stats.final_);
        println!("  Growth:  {:.2}%", stats.growth_percentage());
        println!("  Rate:    {:.2} MB/iteration", stats.growth_rate());

        // Memory leak threshold: < 10% growth
        assert!(
            stats.growth_percentage() < 10.0,
            "Memory leak detected: {:.2}% growth exceeds 10% threshold",
            stats.growth_percentage()
        );
    }

    #[test]
    #[ignore] // Run with: cargo test --test stress_tests -- --ignored
    #[cfg(feature = "tch-backend")]
    fn test_tch_memory_leak_detection() {
        use llamafirewall_ml::tch_inference::TchInferenceEngine;

        let model_path = "models/promptguard.pt";
        let tokenizer_path = "models/tokenizer.json";

        if !std::path::Path::new(model_path).exists() {
            eprintln!("Skipping test: model file not found");
            return;
        }

        let engine = TchInferenceEngine::new(model_path, tokenizer_path)
            .expect("Failed to create tch engine");

        let test_prompts = vec![
            "Test prompt 1",
            "Test prompt 2 with more text to make it longer",
            "Short",
            "A".repeat(100).as_str(),
        ];

        let iterations = 1000;
        let sample_interval = 100;
        let mut stats = MemoryStats::default();

        println!("Running {} iterations...", iterations);

        for i in 0..iterations {
            let prompt = test_prompts[i % test_prompts.len()];
            let _ = engine.infer(prompt).expect("Inference failed");

            // Sample memory usage
            if i % sample_interval == 0 {
                let mem_mb = get_process_memory_mb();
                stats.record(mem_mb);
                println!(
                    "Iteration {}/{}: {}MB (growth: {:.2}%)",
                    i,
                    iterations,
                    mem_mb,
                    stats.growth_percentage()
                );
            }
        }

        // Final measurement
        stats.record(get_process_memory_mb());

        println!("\nMemory Statistics:");
        println!("  Initial: {}MB", stats.initial);
        println!("  Peak:    {}MB", stats.peak);
        println!("  Final:   {}MB", stats.final_);
        println!("  Growth:  {:.2}%", stats.growth_percentage());
        println!("  Rate:    {:.2} MB/iteration", stats.growth_rate());

        // Memory leak threshold: < 10% growth
        assert!(
            stats.growth_percentage() < 10.0,
            "Memory leak detected: {:.2}% growth exceeds 10% threshold",
            stats.growth_percentage()
        );
    }

    #[test]
    #[ignore] // Run with: cargo test --test stress_tests -- --ignored
    #[cfg(feature = "onnx-backend")]
    fn test_onnx_sustained_load() {
        use llamafirewall_ml::inference::OnnxInferenceEngine;

        let model_path = "models/promptguard.onnx";
        let tokenizer_path = "models/tokenizer.json";

        if !std::path::Path::new(model_path).exists() {
            eprintln!("Skipping test: model file not found");
            return;
        }

        let engine = Arc::new(
            OnnxInferenceEngine::new(model_path, tokenizer_path)
                .expect("Failed to create ONNX engine"),
        );

        let test_duration = Duration::from_secs(60); // 1 minute sustained load
        let num_threads = 4;

        println!("Running sustained load test for {:?}...", test_duration);

        let start = Instant::now();
        let total_requests = Arc::new(Mutex::new(0u64));
        let mut handles = vec![];

        for thread_id in 0..num_threads {
            let engine_clone = Arc::clone(&engine);
            let requests_clone = Arc::clone(&total_requests);
            let duration = test_duration.clone();

            let handle = thread::spawn(move || {
                let start = Instant::now();
                let mut local_requests = 0u64;

                while start.elapsed() < duration {
                    let text = format!("Load test request from thread {}", thread_id);
                    match engine_clone.infer(&text) {
                        Ok(_) => local_requests += 1,
                        Err(e) => eprintln!("Inference error: {}", e),
                    }
                }

                let mut total = requests_clone.lock().unwrap();
                *total += local_requests;
                local_requests
            });

            handles.push(handle);
        }

        for handle in handles {
            let local_count = handle.join().expect("Thread panicked");
            println!("Thread completed {} requests", local_count);
        }

        let elapsed = start.elapsed();
        let total = *total_requests.lock().unwrap();
        let rps = total as f64 / elapsed.as_secs_f64();

        println!("\nSustained Load Results:");
        println!("  Duration:        {:?}", elapsed);
        println!("  Total Requests:  {}", total);
        println!("  Throughput:      {:.2} req/s", rps);

        // Should maintain at least 10 req/s under sustained load
        assert!(
            rps >= 10.0,
            "Throughput {:.2} req/s below 10 req/s threshold",
            rps
        );
    }

    #[test]
    #[ignore] // Run with: cargo test --test stress_tests -- --ignored
    #[cfg(feature = "tch-backend")]
    fn test_tch_sustained_load() {
        use llamafirewall_ml::tch_inference::TchInferenceEngine;

        let model_path = "models/promptguard.pt";
        let tokenizer_path = "models/tokenizer.json";

        if !std::path::Path::new(model_path).exists() {
            eprintln!("Skipping test: model file not found");
            return;
        }

        let engine = Arc::new(
            TchInferenceEngine::new(model_path, tokenizer_path)
                .expect("Failed to create tch engine"),
        );

        let test_duration = Duration::from_secs(60); // 1 minute sustained load
        let num_threads = 4;

        println!("Running sustained load test for {:?}...", test_duration);

        let start = Instant::now();
        let total_requests = Arc::new(Mutex::new(0u64));
        let mut handles = vec![];

        for thread_id in 0..num_threads {
            let engine_clone = Arc::clone(&engine);
            let requests_clone = Arc::clone(&total_requests);
            let duration = test_duration.clone();

            let handle = thread::spawn(move || {
                let start = Instant::now();
                let mut local_requests = 0u64;

                while start.elapsed() < duration {
                    let text = format!("Load test request from thread {}", thread_id);
                    match engine_clone.infer(&text) {
                        Ok(_) => local_requests += 1,
                        Err(e) => eprintln!("Inference error: {}", e),
                    }
                }

                let mut total = requests_clone.lock().unwrap();
                *total += local_requests;
                local_requests
            });

            handles.push(handle);
        }

        for handle in handles {
            let local_count = handle.join().expect("Thread panicked");
            println!("Thread completed {} requests", local_count);
        }

        let elapsed = start.elapsed();
        let total = *total_requests.lock().unwrap();
        let rps = total as f64 / elapsed.as_secs_f64();

        println!("\nSustained Load Results:");
        println!("  Duration:        {:?}", elapsed);
        println!("  Total Requests:  {}", total);
        println!("  Throughput:      {:.2} req/s", rps);

        // Target: > 18 req/s
        assert!(
            rps >= 15.0,
            "Throughput {:.2} req/s below 15 req/s threshold",
            rps
        );
    }

    #[test]
    #[ignore] // Run with: cargo test --test stress_tests -- --ignored
    #[cfg(feature = "onnx-backend")]
    fn test_onnx_error_recovery() {
        use llamafirewall_ml::inference::OnnxInferenceEngine;

        let model_path = "models/promptguard.onnx";
        let tokenizer_path = "models/tokenizer.json";

        if !std::path::Path::new(model_path).exists() {
            eprintln!("Skipping test: model file not found");
            return;
        }

        let engine = OnnxInferenceEngine::new(model_path, tokenizer_path)
            .expect("Failed to create ONNX engine");

        // Test recovery from various edge cases
        let edge_cases = vec![
            "",                        // Empty
            "\0\0\0",                  // Null bytes
            "a".repeat(10000).as_str(), // Very long
            "\u{FFFD}",                // Replacement character
        ];

        for (i, case) in edge_cases.iter().enumerate() {
            match engine.infer(case) {
                Ok(_) => println!("✓ Handled edge case {}", i),
                Err(e) => println!("✓ Gracefully errored on case {}: {}", i, e),
            }

            // Engine should still work after edge case
            let normal_result = engine.infer("Normal prompt").expect("Engine broken after edge case");
            assert_eq!(normal_result.len(), 2, "Engine corrupted after edge case {}", i);
        }

        println!("✓ Error recovery test passed");
    }

    #[test]
    #[ignore]
    #[cfg(feature = "tch-backend")]
    fn test_tch_error_recovery() {
        use llamafirewall_ml::tch_inference::TchInferenceEngine;

        let model_path = "models/promptguard.pt";
        let tokenizer_path = "models/tokenizer.json";

        if !std::path::Path::new(model_path).exists() {
            eprintln!("Skipping test: model file not found");
            return;
        }

        let engine = TchInferenceEngine::new(model_path, tokenizer_path)
            .expect("Failed to create tch engine");

        // Test recovery from various edge cases
        let edge_cases = vec![
            "",                        // Empty
            "\0\0\0",                  // Null bytes
            "a".repeat(10000).as_str(), // Very long
            "\u{FFFD}",                // Replacement character
        ];

        for (i, case) in edge_cases.iter().enumerate() {
            match engine.infer(case) {
                Ok(_) => println!("✓ Handled edge case {}", i),
                Err(e) => println!("✓ Gracefully errored on case {}: {}", i, e),
            }

            // Engine should still work after edge case
            let normal_result = engine.infer("Normal prompt").expect("Engine broken after edge case");
            assert_eq!(normal_result.len(), 2, "Engine corrupted after edge case {}", i);
        }

        println!("✓ Error recovery test passed");
    }
}
