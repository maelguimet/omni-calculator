//! Independent regression checks from the numerical-engine agent's CAS audit.
use omni_calc::{Calculation, error::CalcError, symbolic::try_evaluate};
use std::time::{Duration, Instant};

fn evaluate(input: &str) -> Result<Calculation, CalcError> {
    try_evaluate(input, Instant::now() + Duration::from_millis(500))
        .expect("recognized symbolic operation")
}

fn calc(input: &str) -> Calculation {
    evaluate(input).unwrap_or_else(|error| panic!("{input}: {error}"))
}

fn expression(result: &Calculation) -> &str {
    result
        .value
        .as_str()
        .or_else(|| result.value["expression"].as_str())
        .unwrap()
}

fn numeric(expression: &str, x: f64) -> f64 {
    let mut context = fend_core::Context::new();
    fend_core::evaluate(&format!("x=({x}); ({expression}) to 15 sf"), &mut context)
        .unwrap()
        .get_main_result()
        .trim_start_matches("approx. ")
        .parse()
        .unwrap()
}

#[test]
fn multiplying_maximum_integer_exponent_must_not_panic_or_wrap() {
    let input = "simplify(x^9223372036854775807*x)";
    let outcome = std::panic::catch_unwind(|| evaluate(input));
    assert!(
        outcome.is_ok(),
        "symbolic exponent addition panicked on {input}"
    );
    assert_eq!(outcome.unwrap().unwrap_err().code, "resource_limit");
}

#[test]
fn logarithm_means_the_same_thing_in_numerical_and_symbolic_dialects() {
    // Omni's ordinary log() is base 10, while ln() is natural logarithm.
    let derivative = calc("diff(log(x),x)");
    let actual = numeric(expression(&derivative), 10.0);
    let expected = 1.0 / (10.0 * 10.0_f64.ln());
    assert!(
        (actual - expected).abs() < 1e-12,
        "{} evaluates to {actual}, expected {expected}",
        derivative.display
    );
    let primitive = calc("integrate(log(x),x)");
    let at_10 = numeric(expression(&primitive), 10.0);
    let expected = (10.0 * 10.0_f64.ln() - 10.0) / 10.0_f64.ln();
    assert!((at_10 - expected).abs() < 1e-10);
}

#[test]
fn substitution_preserves_domains_of_the_replacement_expression() {
    let quotient = calc("substitute(x,x,y/y)");
    assert_eq!(expression(&quotient), "1");
    assert!(
        quotient.display.contains("y != 0"),
        "replacement cancellation lost its domain: {}",
        quotient.display
    );
    let logs = calc("substitute(x,x,ln(y)-ln(y))");
    assert_eq!(expression(&logs), "0");
    assert!(
        logs.display.contains("y > 0"),
        "replacement cancellation lost logarithm domain: {}",
        logs.display
    );
}

#[test]
fn real_power_domain_is_checked_before_numeric_base_cancellation() {
    for input in [
        "simplify((-1)^(1/2))",
        "simplify((-1)^(1/2)-(-1)^(1/2))",
        "substitute(x^(1/2),x,-1)",
    ] {
        let error = evaluate(input).expect_err("real symbolic fractional powers must reject a known negative base with an even denominator");
        assert_eq!(error.code, "math_error", "{input}: {error}");
    }
}

#[test]
fn reciprocal_quadratic_integrals_have_the_correct_sign_and_scale() {
    for (source, x) in [
        ("1/(2*x^2+3*x+7)", 0.4),
        ("1/(-2*x^2-3*x-7)", 0.4),
        ("1/(2*x^2+3*x-7)", 0.4),
        ("1/(4*x^2+4*x+1)", 0.4),
        ("1/sqrt(9-4*x^2)", 0.4),
    ] {
        let primitive = calc(&format!("integrate({source},x)"));
        // A numerical finite difference is independent of the symbolic derivative implementation.
        let h = 1e-5;
        let derivative = (numeric(expression(&primitive), x + h)
            - numeric(expression(&primitive), x - h))
            / (2.0 * h);
        let expected = numeric(source, x);
        assert!(
            (derivative - expected).abs() < 1e-8,
            "{source} -> {}: {derivative} != {expected}",
            primitive.display
        );
    }
}

#[test]
fn higher_degree_solve_returns_complete_sets_or_an_explicit_error() {
    let roots = calc("solve((x+2)*(x-3)*(x^2+1),x)");
    let solutions = roots.value["solutions"].as_array().unwrap();
    assert_eq!(solutions.len(), 4);
    for expected in ["-2", "3", "i", "-i"] {
        assert!(solutions.contains(&serde_json::json!(expected)));
    }
    assert_eq!(
        evaluate("solve(x^5-x+1,x)").unwrap_err().code,
        "unsupported_operation"
    );
}
