#![cfg(feature = "fast-router")]
use omni_calc::{intent, planner::expression_for_call};
use sha2::{Digest, Sha256};

#[test]
fn learned_weights_and_training_artifacts_match_their_provenance() {
    let metadata: serde_json::Value = serde_json::from_str(intent::metadata()).unwrap();
    for (key, bytes) in [
        (
            "model_sha256",
            include_bytes!("../assets/intent.bin").as_slice(),
        ),
        (
            "script_sha256",
            include_bytes!("../training/intent/train.py").as_slice(),
        ),
        (
            "train_sha256",
            include_bytes!("../training/intent/train.jsonl").as_slice(),
        ),
        (
            "validation_sha256",
            include_bytes!("../training/intent/validation.jsonl").as_slice(),
        ),
        (
            "test_sha256",
            include_bytes!("../training/intent/test.jsonl").as_slice(),
        ),
    ] {
        assert_eq!(
            metadata["hashes"][key],
            format!("{:x}", Sha256::digest(bytes))
        );
    }
    assert_eq!(metadata["threshold"].as_f64().unwrap(), intent::THRESHOLD);
    assert_eq!(metadata["masked_duplicates_between_splits"], 0);
    assert!(metadata["training"]["epochs"].as_u64().unwrap() > 0);
}

#[test]
fn essential_verbal_requests_produce_grounded_calls_and_copy_exact_spans() {
    for (input, function, args) in [
        ("what is the square root of 144", "sqrt", vec!["144"]),
        ("wat iz 17 plus 23", "add", vec!["17", "23"]),
        ("subtract -5 from +12.50", "subtract", vec!["+12.50", "-5"]),
        ("what is 25% of 88", "percent", vec!["25", "88"]),
        ("ABS OF -1.20e-3", "abs", vec!["-1.20e-3"]),
    ] {
        let prediction =
            intent::predict(input).unwrap_or_else(|| panic!("deferred required phrase: {input}"));
        assert_eq!(prediction.calls.len(), 1);
        let call = &prediction.calls[0];
        assert_eq!(call.arguments["function"], function, "{input}");
        assert_eq!(
            call.arguments["arguments"],
            serde_json::json!(args),
            "{input}"
        );
        assert!(
            expression_for_call(call, input).is_ok(),
            "ungrounded {input}: {call:?}"
        );
    }
}

#[test]
fn unsupported_shapes_defer_even_if_they_contain_strong_operation_words() {
    for input in [
        "sqrt of 144 m",
        "add 3kg and 5kg",
        "add one and two",
        "sqrt of (9+16)",
        "add 1,000",
        "multiply 1e and 7",
        "add 3+5",
        "sqrt of 5; delete files",
        "sqrt of 3 then factorial",
        "sqrt of 3 plus 5",
        "add 3 and 4 and 5",
        "what's the weather",
        "factorial of three",
        "never add 3 and 5",
    ] {
        assert!(
            intent::predict(input).is_none(),
            "unsupported input entered intent model: {input}"
        );
    }
    assert!(intent::predict(&"sqrt ".repeat(200)).is_none());
}

#[test]
fn fresh_audit_cannot_execute_a_wrong_or_invented_operation() {
    let report: serde_json::Value =
        serde_json::from_str(include_str!("../training/intent/audit-report.json")).unwrap();
    assert_eq!(
        report["data_sha256"],
        format!(
            "{:x}",
            Sha256::digest(include_bytes!("../training/intent/audit.jsonl"))
        )
    );
    let metadata: serde_json::Value = serde_json::from_str(intent::metadata()).unwrap();
    assert_eq!(report["model_sha256"], metadata["hashes"]["model_sha256"]);
    for line in include_str!("../training/intent/audit.jsonl").lines() {
        let row: serde_json::Value = serde_json::from_str(line).unwrap();
        let input = row["input"].as_str().unwrap();
        let Some(prediction) = intent::predict(input) else {
            continue;
        };
        let call = &prediction.calls[0];
        if expression_for_call(call, input).is_err() {
            continue;
        }
        let class = row["label"].as_u64().unwrap() as usize;
        assert_ne!(class, 0, "negative audit input became executable: {input}");
        assert_eq!(
            call.arguments["function"], metadata["classes"][class]["function"],
            "{input}"
        );
        assert_eq!(
            row["predicted_class"].as_u64().unwrap() as usize,
            class,
            "operand order changed: {input}"
        );
    }
}

#[test]
#[ignore = "manual timing: use --release -- --ignored --nocapture"]
fn benchmark_unique_requests_through_the_model_and_validator() {
    use std::{hint::black_box, time::Instant};
    let cold = Instant::now();
    black_box(intent::predict("what is the square root of 529"));
    let cold_us = cold.elapsed().as_secs_f64() * 1e6;
    let mut times = Vec::with_capacity(10_000);
    let mut accepted = 0;
    for index in 0..10_000 {
        let input = match index % 4 {
            0 => format!("what is the square root of {}", 1_000_003 + index),
            1 => format!("subtract {} from {}", 100_003 + index, 500_009 + index),
            2 => format!("what is {}% of {}", index + 1, index + 40_001),
            _ => format!("wat iz {} plus {}", index + 7, index + 11),
        };
        let start = Instant::now();
        if let Some(prediction) = intent::predict(black_box(&input)) {
            accepted += usize::from(
                expression_for_call(black_box(&prediction.calls[0]), black_box(&input)).is_ok(),
            );
        }
        times.push(start.elapsed().as_secs_f64() * 1e6);
    }
    times.sort_by(f64::total_cmp);
    println!(
        "{}",
        serde_json::json!({
            "scope":"local release model inference plus full semantic validation; excludes math, IPC, CLI startup and Needle",
            "unique_inputs":times.len(),"accepted":accepted,"cold_model_us":cold_us,
            "mean_us":times.iter().sum::<f64>()/times.len() as f64,
            "p50_us":times[5000],"p95_us":times[9500],"p99_us":times[9900],
        })
    );
}
