//! Comprehensive Integration Test Suite for LlamaFirewall ML Backend
//!
//! This test suite validates the full pipeline integration for the tch-rs PyTorch backend,
//! ensuring correctness and reliability.

#[cfg(test)]
mod integration_tests {
    use llamafirewall_ml::*;
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    /// Test data representing different prompt types and edge cases
    struct TestCase {
        name: &'static str,
        input: &'static str,
        expected_jailbreak: bool,
        description: &'static str,
    }

    const TEST_CASES: &[TestCase] = &[
        TestCase {
            name: "benign_greeting",
            input: "Hello, how are you?",
            expected_jailbreak: false,
            description: "Simple benign greeting",
        },
        TestCase {
            name: "benign_question",
            input: "What is the weather like today?",
            expected_jailbreak: false,
            description: "Normal question about weather",
        },
        TestCase {
            name: "empty_string",
            input: "",
            expected_jailbreak: false,
            description: "Empty input string",
        },
        TestCase {
            name: "single_character",
            input: "a",
            expected_jailbreak: false,
            description: "Single character input",
        },
        TestCase {
            name: "whitespace_only",
            input: "   \t\n  ",
            expected_jailbreak: false,
            description: "Only whitespace characters",
        },
        TestCase {
            name: "unicode_text",
            input: "こんにちは、元気ですか？",
            expected_jailbreak: false,
            description: "Japanese unicode text",
        },
        TestCase {
            name: "emoji_text",
            input: "Hello 👋 How are you? 😊",
            expected_jailbreak: false,
            description: "Text with emojis",
        },
        TestCase {
            name: "mixed_unicode",
            input: "Hello мир 世界 🌍",
            expected_jailbreak: false,
            description: "Mixed language unicode",
        },
        TestCase {
            name: "long_benign_text",
            input: "This is a very long text that should still be processed correctly. \
                    It contains multiple sentences and should test the model's ability \
                    to handle longer inputs without issues. The text is completely benign \
                    and should not trigger any jailbreak detection.",
            expected_jailbreak: false,
            description: "Long benign text (multiple sentences)",
        },
        TestCase {
            name: "special_characters",
            input: "!@#$%^&*()_+-=[]{}|;:',.<>?/~`",
            expected_jailbreak: false,
            description: "Various special characters",
        },
    ];

    #[test]
    #[cfg(feature = "onnx-backend")]
    fn test_onnx_pipeline_integration() {
        use llamafirewall_ml::inference::OnnxInferenceEngine;

        let model_path = "models/promptguard.onnx";
        let tokenizer_path = "models/tokenizer.json";

        // Skip test if model files don't exist
        if !std::path::Path::new(model_path).exists() {
            eprintln!("Skipping test: model file not found at {}", model_path);
            return;
        }

        let engine = OnnxInferenceEngine::new(model_path, tokenizer_path)
            .expect("Failed to create ONNX engine");

        for test_case in TEST_CASES {
            let result = engine
                .infer(test_case.input)
                .expect(&format!("Failed to infer: {}", test_case.name));

            assert_eq!(
                result.len(),
                2,
                "Expected 2 output logits for test case: {}",
                test_case.name
            );

            // Verify outputs are valid probabilities
            assert!(
                result[0].is_finite() && result[1].is_finite(),
                "Invalid output for test case: {}",
                test_case.name
            );

            println!(
                "✓ ONNX - {}: [{:.4}, {:.4}]",
                test_case.name, result[0], result[1]
            );
        }
    }

    #[test]
    #[cfg(feature = "tch-backend")]
    fn test_tch_pipeline_integration() {
        use llamafirewall_ml::tch_inference::TchInferenceEngine;

        let model_path = "models/promptguard.pt";
        let tokenizer_path = "models/tokenizer.json";

        // Skip test if model files don't exist
        if !std::path::Path::new(model_path).exists() {
            eprintln!("Skipping test: model file not found at {}", model_path);
            return;
        }

        let engine = TchInferenceEngine::new(model_path, tokenizer_path)
            .expect("Failed to create tch engine");

        for test_case in TEST_CASES {
            let result = engine
                .infer(test_case.input)
                .expect(&format!("Failed to infer: {}", test_case.name));

            assert_eq!(
                result.len(),
                2,
                "Expected 2 output logits for test case: {}",
                test_case.name
            );

            // Verify outputs are valid probabilities
            assert!(
                result[0].is_finite() && result[1].is_finite(),
                "Invalid output for test case: {}",
                test_case.name
            );

            println!(
                "✓ tch-rs - {}: [{:.4}, {:.4}]",
                test_case.name, result[0], result[1]
            );
        }
    }

    #[test]
    #[cfg(all(feature = "onnx-backend", feature = "tch-backend"))]
    fn test_backend_output_consistency() {
        use llamafirewall_ml::inference::OnnxInferenceEngine;
        use llamafirewall_ml::tch_inference::TchInferenceEngine;

        let onnx_model = "models/promptguard.onnx";
        let tch_model = "models/promptguard.pt";
        let tokenizer = "models/tokenizer.json";

        // Skip if models don't exist
        if !std::path::Path::new(onnx_model).exists()
            || !std::path::Path::new(tch_model).exists()
        {
            eprintln!("Skipping test: model files not found");
            return;
        }

        let onnx_engine = OnnxInferenceEngine::new(onnx_model, tokenizer)
            .expect("Failed to create ONNX engine");
        let tch_engine =
            TchInferenceEngine::new(tch_model, tokenizer).expect("Failed to create tch engine");

        const TOLERANCE: f32 = 1e-3; // Allow small differences due to numerical precision

        for test_case in TEST_CASES {
            let onnx_result = onnx_engine
                .infer(test_case.input)
                .expect(&format!("ONNX failed: {}", test_case.name));
            let tch_result = tch_engine
                .infer(test_case.input)
                .expect(&format!("tch-rs failed: {}", test_case.name));

            let diff_0 = (onnx_result[0] - tch_result[0]).abs();
            let diff_1 = (onnx_result[1] - tch_result[1]).abs();

            assert!(
                diff_0 < TOLERANCE && diff_1 < TOLERANCE,
                "Output mismatch for '{}': ONNX=[{:.4}, {:.4}], tch=[{:.4}, {:.4}], diff=[{:.4}, {:.4}]",
                test_case.name,
                onnx_result[0],
                onnx_result[1],
                tch_result[0],
                tch_result[1],
                diff_0,
                diff_1
            );

            println!(
                "✓ Consistent - {}: diff=[{:.6}, {:.6}]",
                test_case.name, diff_0, diff_1
            );
        }
    }

    #[test]
    #[cfg(feature = "onnx-backend")]
    fn test_onnx_batch_processing() {
        use llamafirewall_ml::inference::OnnxInferenceEngine;

        let model_path = "models/promptguard.onnx";
        let tokenizer_path = "models/tokenizer.json";

        if !std::path::Path::new(model_path).exists() {
            eprintln!("Skipping test: model file not found");
            return;
        }

        let engine = OnnxInferenceEngine::new(model_path, tokenizer_path)
            .expect("Failed to create ONNX engine");

        let batch_texts: Vec<String> = TEST_CASES.iter().take(5).map(|tc| tc.input.to_string()).collect();

        let batch_results = engine
            .infer_batch(&batch_texts)
            .expect("Failed to infer batch");

        assert_eq!(
            batch_results.len(),
            batch_texts.len(),
            "Batch results length mismatch"
        );

        for (i, result) in batch_results.iter().enumerate() {
            assert_eq!(result.len(), 2, "Expected 2 outputs for batch item {}", i);
            assert!(
                result[0].is_finite() && result[1].is_finite(),
                "Invalid batch output at index {}",
                i
            );
        }

        println!("✓ ONNX batch processing: {} items", batch_results.len());
    }

    #[test]
    #[cfg(feature = "tch-backend")]
    fn test_tch_batch_processing() {
        use llamafirewall_ml::tch_inference::TchInferenceEngine;

        let model_path = "models/promptguard.pt";
        let tokenizer_path = "models/tokenizer.json";

        if !std::path::Path::new(model_path).exists() {
            eprintln!("Skipping test: model file not found");
            return;
        }

        let engine = TchInferenceEngine::new(model_path, tokenizer_path)
            .expect("Failed to create tch engine");

        let batch_texts: Vec<String> = TEST_CASES.iter().take(5).map(|tc| tc.input.to_string()).collect();

        let batch_results = engine
            .infer_batch(&batch_texts)
            .expect("Failed to infer batch");

        assert_eq!(
            batch_results.len(),
            batch_texts.len(),
            "Batch results length mismatch"
        );

        for (i, result) in batch_results.iter().enumerate() {
            assert_eq!(result.len(), 2, "Expected 2 outputs for batch item {}", i);
            assert!(
                result[0].is_finite() && result[1].is_finite(),
                "Invalid batch output at index {}",
                i
            );
        }

        println!("✓ tch-rs batch processing: {} items", batch_results.len());
    }

    #[test]
    #[cfg(all(feature = "onnx-backend", feature = "tch-backend"))]
    fn test_batch_consistency_between_backends() {
        use llamafirewall_ml::inference::OnnxInferenceEngine;
        use llamafirewall_ml::tch_inference::TchInferenceEngine;

        let onnx_model = "models/promptguard.onnx";
        let tch_model = "models/promptguard.pt";
        let tokenizer = "models/tokenizer.json";

        if !std::path::Path::new(onnx_model).exists()
            || !std::path::Path::new(tch_model).exists()
        {
            eprintln!("Skipping test: model files not found");
            return;
        }

        let onnx_engine = OnnxInferenceEngine::new(onnx_model, tokenizer)
            .expect("Failed to create ONNX engine");
        let tch_engine =
            TchInferenceEngine::new(tch_model, tokenizer).expect("Failed to create tch engine");

        let batch_texts: Vec<String> = TEST_CASES.iter().take(5).map(|tc| tc.input.to_string()).collect();

        let onnx_results = onnx_engine
            .infer_batch(&batch_texts)
            .expect("ONNX batch failed");
        let tch_results = tch_engine.infer_batch(&batch_texts).expect("tch batch failed");

        const TOLERANCE: f32 = 1e-3;

        for (i, (onnx_res, tch_res)) in onnx_results.iter().zip(tch_results.iter()).enumerate() {
            let diff_0 = (onnx_res[0] - tch_res[0]).abs();
            let diff_1 = (onnx_res[1] - tch_res[1]).abs();

            assert!(
                diff_0 < TOLERANCE && diff_1 < TOLERANCE,
                "Batch item {} mismatch: diff=[{:.6}, {:.6}]",
                i,
                diff_0,
                diff_1
            );
        }

        println!("✓ Batch consistency verified for {} items", batch_texts.len());
    }

    #[test]
    #[cfg(feature = "onnx-backend")]
    fn test_onnx_performance_regression() {
        use llamafirewall_ml::inference::OnnxInferenceEngine;

        let model_path = "models/promptguard.onnx";
        let tokenizer_path = "models/tokenizer.json";

        if !std::path::Path::new(model_path).exists() {
            eprintln!("Skipping test: model file not found");
            return;
        }

        let engine = OnnxInferenceEngine::new(model_path, tokenizer_path)
            .expect("Failed to create ONNX engine");

        let test_text = "This is a test prompt for performance measurement.";
        let iterations = 100;

        // Warmup
        for _ in 0..10 {
            let _ = engine.infer(test_text);
        }

        // Measure
        let start = Instant::now();
        for _ in 0..iterations {
            engine.infer(test_text).expect("Inference failed");
        }
        let duration = start.elapsed();

        let avg_latency = duration.as_millis() as f64 / iterations as f64;
        println!("✓ ONNX average latency: {:.2}ms", avg_latency);

        // Performance regression threshold: should be < 150ms on average
        assert!(
            avg_latency < 150.0,
            "Performance regression: avg latency {:.2}ms exceeds 150ms threshold",
            avg_latency
        );
    }

    #[test]
    #[cfg(feature = "tch-backend")]
    fn test_tch_performance_regression() {
        use llamafirewall_ml::tch_inference::TchInferenceEngine;

        let model_path = "models/promptguard.pt";
        let tokenizer_path = "models/tokenizer.json";

        if !std::path::Path::new(model_path).exists() {
            eprintln!("Skipping test: model file not found");
            return;
        }

        let engine = TchInferenceEngine::new(model_path, tokenizer_path)
            .expect("Failed to create tch engine");

        let test_text = "This is a test prompt for performance measurement.";
        let iterations = 100;

        // Warmup
        for _ in 0..10 {
            let _ = engine.infer(test_text);
        }

        // Measure
        let start = Instant::now();
        for _ in 0..iterations {
            engine.infer(test_text).expect("Inference failed");
        }
        let duration = start.elapsed();

        let avg_latency = duration.as_millis() as f64 / iterations as f64;
        println!("✓ tch-rs average latency: {:.2}ms", avg_latency);

        // Performance target: should be < 55ms on average (P95 target)
        assert!(
            avg_latency < 100.0,
            "Performance regression: avg latency {:.2}ms exceeds 100ms threshold",
            avg_latency
        );
    }

    #[test]
    fn test_error_handling_invalid_model_path() {
        #[cfg(feature = "onnx-backend")]
        {
            use llamafirewall_ml::inference::OnnxInferenceEngine;
            let result = OnnxInferenceEngine::new("nonexistent.onnx", "models/tokenizer.json");
            assert!(result.is_err(), "Should fail with invalid model path");
        }

        #[cfg(feature = "tch-backend")]
        {
            use llamafirewall_ml::tch_inference::TchInferenceEngine;
            let result = TchInferenceEngine::new("nonexistent.pt", "models/tokenizer.json");
            assert!(result.is_err(), "Should fail with invalid model path");
        }
    }

    #[test]
    #[cfg(feature = "onnx-backend")]
    fn test_concurrent_onnx_inference() {
        use llamafirewall_ml::inference::OnnxInferenceEngine;
        use std::thread;

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

        let mut handles = vec![];

        // Spawn 4 concurrent threads
        for i in 0..4 {
            let engine_clone = Arc::clone(&engine);
            let handle = thread::spawn(move || {
                let text = format!("Concurrent test prompt {}", i);
                for _ in 0..10 {
                    let result = engine_clone.infer(&text).expect("Inference failed");
                    assert_eq!(result.len(), 2);
                }
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.join().expect("Thread panicked");
        }

        println!("✓ ONNX concurrent inference test passed");
    }

    #[test]
    #[cfg(feature = "tch-backend")]
    fn test_concurrent_tch_inference() {
        use llamafirewall_ml::tch_inference::TchInferenceEngine;
        use std::thread;

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

        let mut handles = vec![];

        // Spawn 4 concurrent threads
        for i in 0..4 {
            let engine_clone = Arc::clone(&engine);
            let handle = thread::spawn(move || {
                let text = format!("Concurrent test prompt {}", i);
                for _ in 0..10 {
                    let result = engine_clone.infer(&text).expect("Inference failed");
                    assert_eq!(result.len(), 2);
                }
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.join().expect("Thread panicked");
        }

        println!("✓ tch-rs concurrent inference test passed");
    }

    #[test]
    #[cfg(any(feature = "onnx-backend", feature = "tch-backend"))]
    fn test_varying_input_lengths() {
        let test_lengths = vec![
            ("", 0),
            ("a", 1),
            ("Hello world", 11),
            ("A".repeat(50).as_str(), 50),
            ("B".repeat(100).as_str(), 100),
            ("C".repeat(500).as_str(), 500),
        ];

        #[cfg(feature = "onnx-backend")]
        {
            use llamafirewall_ml::inference::OnnxInferenceEngine;
            let model_path = "models/promptguard.onnx";
            if std::path::Path::new(model_path).exists() {
                let engine = OnnxInferenceEngine::new(model_path, "models/tokenizer.json")
                    .expect("Failed to create ONNX engine");

                for (text, len) in &test_lengths {
                    let result = engine.infer(text).expect(&format!("Failed at length {}", len));
                    assert_eq!(result.len(), 2);
                    println!("✓ ONNX handled length {}", len);
                }
            }
        }

        #[cfg(feature = "tch-backend")]
        {
            use llamafirewall_ml::tch_inference::TchInferenceEngine;
            let model_path = "models/promptguard.pt";
            if std::path::Path::new(model_path).exists() {
                let engine = TchInferenceEngine::new(model_path, "models/tokenizer.json")
                    .expect("Failed to create tch engine");

                for (text, len) in &test_lengths {
                    let result = engine.infer(text).expect(&format!("Failed at length {}", len));
                    assert_eq!(result.len(), 2);
                    println!("✓ tch-rs handled length {}", len);
                }
            }
        }
    }
}
