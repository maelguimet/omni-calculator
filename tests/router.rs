#![cfg(feature = "fast-router")]

use omni_calc::router::{REFERENCE_THRESHOLD, metadata, reference_confidence};
use sha2::{Digest, Sha256};

#[test]
fn embedded_weights_match_training_provenance() {
    let metadata: serde_json::Value = serde_json::from_str(metadata()).unwrap();
    let hash = format!(
        "{:x}",
        Sha256::digest(include_bytes!("../assets/router-v2.bin"))
    );
    assert_eq!(metadata["hashes"]["model_sha256"], hash);
    assert_eq!(metadata["threshold"].as_f64().unwrap(), REFERENCE_THRESHOLD);
    assert_eq!(metadata["split_policy"]["duplicates_between_splits"], 0);
    assert!(metadata["training"]["epochs"].as_u64().unwrap() > 0);
    assert_eq!(
        metadata["hashes"]["script_sha256"],
        format!(
            "{:x}",
            Sha256::digest(include_bytes!("../training/reference-v2/train.py"))
        )
    );
    for (name, bytes) in [
        (
            "train",
            include_bytes!("../training/reference-v2/train.jsonl").as_slice(),
        ),
        (
            "validation",
            include_bytes!("../training/reference-v2/validation.jsonl").as_slice(),
        ),
        (
            "test",
            include_bytes!("../training/reference-v2/test.jsonl").as_slice(),
        ),
    ] {
        assert_eq!(
            metadata["hashes"][format!("{name}_sha256")],
            format!("{:x}", Sha256::digest(bytes))
        );
    }
}

#[test]
fn curated_cases_are_measured_without_requiring_universal_recall() {
    let cases: serde_json::Value =
        serde_json::from_str(include_str!("../training/reference-v2/parity.json")).unwrap();
    for case in cases.as_array().unwrap() {
        let input = case["input"].as_str().unwrap();
        let confidence = reference_confidence(input);
        assert!(confidence.is_finite() && (0.0..=1.0).contains(&confidence));
        if case["label"] == 0 {
            assert!(
                confidence < REFERENCE_THRESHOLD,
                "false input-reference routing: {input:?} {confidence}"
            );
        }
    }
    for input in ["what is 1+1", "what is 1 + 1", "wat iz 1 + 1", "1/0"] {
        assert!(
            reference_confidence(input) >= REFERENCE_THRESHOLD,
            "critical literal request deferred: {input:?}"
        );
    }
}

#[test]
fn oversized_inputs_defer_without_allocating_model_sized_per_input_data() {
    assert_eq!(reference_confidence(&"1".repeat(4097)), 0.0);
}

#[test]
fn fresh_audit_preserves_raw_model_failures_and_validators_reject_them() {
    use omni_calc::{planner::expression_for_call, protocol::ToolCall};

    let report: serde_json::Value =
        serde_json::from_str(include_str!("../training/reference-v2/audit-report.json")).unwrap();
    let model: serde_json::Value = serde_json::from_str(metadata()).unwrap();
    assert_eq!(report["model_sha256"], model["hashes"]["model_sha256"]);
    assert_eq!(
        report["data_sha256"],
        format!(
            "{:x}",
            Sha256::digest(include_bytes!("../training/reference-v2/audit.jsonl"))
        )
    );
    let call = ToolCall {
        name: "evaluate".to_owned(),
        arguments: serde_json::json!({}),
    };
    let mut false_positives = 0;
    let mut negatives = 0;
    for line in include_str!("../training/reference-v2/audit.jsonl").lines() {
        let case: serde_json::Value = serde_json::from_str(line).unwrap();
        if case["label"] != 0 {
            continue;
        }
        negatives += 1;
        let input = case["input"].as_str().unwrap();
        if reference_confidence(input) >= REFERENCE_THRESHOLD {
            false_positives += 1;
            assert!(
                expression_for_call(&call, input).is_err(),
                "model and validator accepted a fresh non-reference input: {input:?}"
            );
        }
    }
    // The report retains the imperfect raw classifier's actual failures.
    assert_eq!(false_positives, report["overall"]["fp"].as_u64().unwrap());
    assert_eq!(
        negatives,
        report["overall"]["fp"].as_u64().unwrap() + report["overall"]["tn"].as_u64().unwrap()
    );
}
