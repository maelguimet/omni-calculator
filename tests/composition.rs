#![cfg(feature = "fast-router")]
use omni_calc::{
    composition,
    model::{LocalModel, Provider},
    planner::expression_for_call,
};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

#[test]
fn artifacts_match_the_training_provenance() {
    let metadata: Value =
        serde_json::from_str(include_str!("../training/composition/metadata.json")).unwrap();
    assert_eq!(metadata["threshold"], composition::THRESHOLD);
    for (name, bytes) in [
        (
            "model",
            include_bytes!("../assets/composition.bin").as_slice(),
        ),
        (
            "script",
            include_bytes!("../training/composition/train.py").as_slice(),
        ),
        (
            "feature_dependency",
            include_bytes!("../training/intent/train.py").as_slice(),
        ),
        (
            "classes.json",
            include_bytes!("../training/composition/classes.json").as_slice(),
        ),
        (
            "vocabulary.txt",
            include_bytes!("../training/composition/vocabulary.txt").as_slice(),
        ),
        (
            "parity.json",
            include_bytes!("../training/composition/parity.json").as_slice(),
        ),
        (
            "train.jsonl",
            include_bytes!("../training/composition/train.jsonl").as_slice(),
        ),
        (
            "validation.jsonl",
            include_bytes!("../training/composition/validation.jsonl").as_slice(),
        ),
        (
            "test.jsonl",
            include_bytes!("../training/composition/test.jsonl").as_slice(),
        ),
    ] {
        assert_eq!(
            metadata["hashes"][name],
            format!("{:x}", Sha256::digest(bytes)),
            "{name}"
        );
    }
}

#[test]
fn held_out_classifications_cannot_execute_a_wrong_structure() {
    let classes: Value =
        serde_json::from_str(include_str!("../training/composition/classes.json")).unwrap();
    let mut accepted = 0;
    for line in include_str!("../training/composition/test.jsonl").lines() {
        let row: Value = serde_json::from_str(line).unwrap();
        let input = row["input"].as_str().unwrap();
        let Some(prediction) = composition::predict(input) else {
            continue;
        };
        let call = &prediction.calls[0];
        if expression_for_call(call, input).is_err() {
            continue;
        }
        accepted += 1;
        let label = row["label"].as_u64().unwrap() as usize;
        assert_ne!(label, 0, "out-of-scope composition executed: {input}");
        assert_eq!(
            call.arguments["function"], classes[label]["function"],
            "{input}"
        );
        // Compare the complete structure with the independently labeled class.
        let numbers = input
            .split(|c: char| c.is_whitespace() || matches!(c, ',' | '?'))
            .filter(|s| s.parse::<f64>().is_ok())
            .collect::<Vec<_>>();
        let expected = classes[label]["arguments"]
            .as_array()
            .unwrap()
            .iter()
            .map(|template| {
                let mut argument = template.as_str().unwrap().to_owned();
                for (i, number) in numbers.iter().enumerate() {
                    argument =
                        argument.replace(&format!("{{{}}}", ['a', 'b', 'c', 'd'][i]), number);
                }
                argument
            })
            .collect::<Vec<_>>();
        assert_eq!(call.arguments["arguments"], json!(expected), "{input}");
    }
    assert!(
        accepted > 1000,
        "coverage unexpectedly collapsed: {accepted}"
    );
}

#[test]
fn common_compositions_do_not_initialize_needle() {
    for (input, expected) in [
        ("add the squares of 6 and 8", 100),
        ("subtract the product of 7 and 4 from 65", 37),
        ("divide 144 by the product of 3 and 4", 12),
        ("what is 9 less than 42", 33),
    ] {
        let mut model = LocalModel::new(None, false);
        let prediction = model.predict(input, None).unwrap();
        assert!(!model.needle_loaded(), "{input}");
        let expression = expression_for_call(&prediction.calls[0], input).unwrap();
        let result = omni_calc::engine::evaluate(
            &expression,
            std::time::Instant::now() + std::time::Duration::from_secs(1),
        )
        .unwrap();
        assert_eq!(result.value, json!(expected), "{input}");
    }
}

#[test]
fn malformed_and_unrelated_inputs_defer_or_fail_grounding() {
    for input in [
        "add the squares of 6 and 8 kg",
        "sum of squares of 1,000 and 2",
        "never take 4 away from 12",
        "translate add the cubes of 4 and 5",
        "add the squares of 6",
        "add the squares of 6 and",
        "add the squares of 6 and 8 then divide by 2",
        "multiply the sum of 3 and 4 by the sum of 5 and 6",
    ] {
        if let Some(prediction) = composition::predict(input) {
            assert!(
                expression_for_call(&prediction.calls[0], input).is_err(),
                "{input}"
            );
        }
    }
}
