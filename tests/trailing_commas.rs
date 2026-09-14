use omni_calc::engine::evaluate;
use std::time::{Duration, Instant};

#[test]
fn one_final_comma_is_allowed_without_changing_values_or_arity() {
    for (input, expected) in [
        ("mean([6,12,24,])", 14),
        ("gcd(54,24,)", 6),
        ("det([[2,0,],[0,3,],])", 6),
    ] {
        let result = evaluate(input, Instant::now() + Duration::from_secs(1)).unwrap();
        assert_eq!(result.value.as_f64().unwrap(), f64::from(expected));
    }
    for input in [
        "gcd(,)",
        "gcd(54,,24)",
        "gcd(54,24,,)",
        "mean([,])",
        "mean([1,,])",
        "mean([1,,2])",
        "gcd(54,)",
    ] {
        assert!(
            evaluate(input, Instant::now() + Duration::from_secs(1)).is_err(),
            "{input}"
        );
    }
}
