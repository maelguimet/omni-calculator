use omni_calc::{Calculation, engine::evaluate, error::CalcError};
use std::time::{Duration, Instant};

fn calc(input: &str) -> Result<Calculation, CalcError> {
    evaluate(input, Instant::now() + Duration::from_millis(500))
}

fn number(input: &str) -> f64 {
    calc(input)
        .unwrap_or_else(|e| panic!("{input}: {e}"))
        .value
        .as_f64()
        .expect("numerical answer")
}

#[test]
fn overflow_must_not_masquerade_as_zero_variance() {
    assert!(calc("variance([1e200,-1e200])").is_err());
    if let Ok(result) = calc("stdev([1e200,-1e200])") {
        assert!((result.value.as_f64().unwrap() / 1e200 - 1.0).abs() < 1e-12);
    }
}

#[test]
fn dot_product_preserves_a_small_uncancelled_term() {
    assert_eq!(number("dot([1e16,1,-1e16],[1,1,1])"), 1.0);
}

#[test]
fn root_is_invariant_to_scaling_of_the_function() {
    assert!((number("root(1e-30*(x-1),x,0,4)") - 1.0).abs() < 1e-8);
    assert!(calc("root(1e-30/(x-0.1),x,0,1)").is_err());
}

#[test]
fn divergent_scaled_integral_is_not_a_tiny_finite_answer() {
    // Multiplying a positive second-order pole by a small constant never makes
    // its improper integral converge.
    assert!(calc("integrate(1e-30/(x-0.37)^2,x,0,1)").is_err());
}

#[test]
fn finite_difference_does_not_claim_false_convergence() {
    if let Ok(result) = calc("derivative_at(1e20+x,x,1)") {
        assert!((result.value.as_f64().unwrap() - 1.0).abs() < 1e-5);
    }
    assert!(calc("derivative_at(1e-12*abs(x),x,0)").is_err());
}

#[test]
fn numerical_calculus_enforces_scalar_domains() {
    for input in ["integrate(atan2(0,0),x,0,1)", "integrate(0^0,x,0,1)"] {
        assert!(calc(input).is_err(), "{input} must be undefined");
    }
}

#[test]
fn matrix_norm_and_inverse_are_scale_stable() {
    assert!((number("norm([[1e200]])") / 1e200 - 1.0).abs() < 1e-12);
    let result =
        calc("inverse([[1e200]])").expect("one-by-one nonsingular matrix has condition number 1");
    assert!((result.value[0][0].as_f64().unwrap() / 1e-200 - 1.0).abs() < 1e-12);
}

#[test]
fn exact_integer_operations_do_not_round_fractional_arguments() {
    for input in ["factorial(3.00000000000000001)", "factorial(2+1e-20)"] {
        assert!(calc(input).is_err(), "{input} is not an integer factorial");
    }
}

#[test]
fn result_metadata_respects_units_and_approximation() {
    assert_eq!(calc("1 psi").unwrap().kind, "quantity");
    assert!(!calc("pi to 3 sf").unwrap().exact);
    assert!(!calc("mean([1,2])+1").unwrap().exact);
    assert!(calc("1 m to cm").unwrap().exact);
    assert!(calc("sqrt").is_err());
}

#[test]
fn explicit_function_calls_bind_before_following_powers() {
    assert!((number("sin(2)^2") - 2.0_f64.sin().powi(2)).abs() < 1e-12);
    assert!((number("ln(10)^(-1)") - 1.0 / 10.0_f64.ln()).abs() < 1e-12);
}

#[test]
fn exact_fraction_json_uses_canonical_improper_fractions_of_unbounded_size() {
    for (input, expected) in [
        ("-3^2+3^-2", "-80/9"),
        ("3^2-3^-2", "80/9"),
        ("-1/3", "-1/3"),
        ("1/3", "1/3"),
    ] {
        let result = calc(input).unwrap();
        assert_eq!(result.value, expected, "{input}: {result:?}");
        assert_eq!(result.kind, "rational", "{input}");
        assert!(result.exact, "{input}");
    }
    assert_eq!(calc("-3^2+3^-2").unwrap().display, "-8 8/9");
    let numerator = num_bigint::BigInt::from(10_u8).pow(60) * 7_u8 + 23_u8;
    for (sign, prefix) in [(1_i8, ""), (-1, "-")] {
        let result = calc(&format!("{prefix}((10^60+3)*7+2)/7")).unwrap();
        assert_eq!(result.value, format!("{}/7", &numerator * sign));
        assert_eq!(result.kind, "rational");
        assert!(result.exact);
    }
    let large_denominator = num_bigint::BigInt::from(10_u8).pow(70) + 1_u8;
    let result = calc("1/(10^70+1)").unwrap();
    assert_eq!(result.value, format!("1/{large_denominator}"));
    assert_eq!(result.kind, "rational");
    assert_eq!(calc("1/3 m").unwrap().kind, "quantity");
}

#[test]
fn spelled_out_temperature_units_preserve_affine_offsets() {
    for (input, expected) in [
        ("(86 degrees Fahrenheit) to Celsius", "30 °C"),
        ("86 degree fahrenheit to degrees CELSIUS", "30 °C"),
        ("25 DEGREES CELSIUS to degree Fahrenheit", "77 °F"),
        ("298.15 degree Kelvin to degrees celsius", "25 °C"),
        ("25 degrees Celsius to kelvins", "298.15 K"),
        ("-40 degrees Celsius to degrees Fahrenheit", "-40 °F"),
        ("32 degrees Fahrenheit to degrees Kelvin", "273.15 K"),
        ("1 degree Kelvin to K", "1 K"),
    ] {
        let result = calc(input).unwrap_or_else(|error| panic!("{input}: {error}"));
        assert_eq!(result.display, expected, "{input}");
        assert_eq!(result.kind, "quantity", "{input}");
        assert!(result.exact, "{input}");
    }
    for input in [
        "86 degrees Fahrenheit to meters",
        "25 degree Celsius + 3 seconds",
        "1 degrees Fahrenheitness to Celsius",
        "1 nodegrees Celsius to Fahrenheit",
    ] {
        assert!(
            calc(input).is_err(),
            "invalid combination accepted: {input}"
        );
    }
    assert_eq!(calc("sin(30 degrees)").unwrap().value, "0.5");
}

#[test]
fn floating_function_results_do_not_saturate_or_truncate_at_integer_limits() {
    for (expression, expected) in [
        ("sinh(228)", 228_f64.sinh()),
        ("cosh(-228)", (-228_f64).cosh()),
        ("sinh(1e-30)", 1e-30_f64.sinh()),
        ("sin(1e-30)", 1e-30_f64.sin()),
        ("sinh(228)/cosh(228)", 1.0),
        ("sin(sin(1e-310))", 1e-310),
    ] {
        let result = calc(expression).unwrap();
        let actual = result
            .value
            .as_f64()
            .unwrap_or_else(|| result.value.as_str().unwrap().parse::<f64>().unwrap());
        assert!(
            (actual / expected - 1.0).abs() < 2e-12,
            "{expression}: {actual} vs {expected}"
        );
        assert!(!result.exact, "{expression}");
    }
    // sinh(a+i*pi/2) = i*cosh(a): verify each component independently.
    assert!((number("imag(sinh(50+i*pi/2))") / 50_f64.cosh() - 1.0).abs() < 2e-12);
    for expression in ["sinh(1000)", "cosh(1000)", "sin(sinh(1000))"] {
        assert!(
            calc(expression).is_err(),
            "nonfinite float accepted: {expression}"
        );
    }
}
