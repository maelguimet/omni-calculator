use omni_calc::engine;
use serde_json::json;
use std::time::{Duration, Instant};

#[test]
fn json_consumers_cannot_round_exact_integer_answers() {
    for (expression, expected) in [
        ("2^53-1", json!(9_007_199_254_740_991i64)),
        ("2^53", json!("9007199254740992")),
        ("2^53+1", json!("9007199254740993")),
        ("-(2^53+1)", json!("-9007199254740993")),
        ("2^63-1", json!("9223372036854775807")),
        ("2^64-1", json!("18446744073709551615")),
    ] {
        let result = engine::evaluate(expression, Instant::now() + Duration::from_secs(1)).unwrap();
        assert_eq!(result.value, expected, "{expression}");
        assert!(result.exact);
        assert_eq!(result.kind, "integer");
    }
    // Why a JSON number is insufficient: IEEE-754 clients collapse adjacent values.
    let left: f64 = serde_json::from_str("9007199254740992").unwrap();
    let right: f64 = serde_json::from_str("9007199254740993").unwrap();
    assert_eq!(left, right);
}
