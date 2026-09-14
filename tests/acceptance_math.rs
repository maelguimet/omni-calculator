//! Public-engine acceptance checks that combine previously separate subsystems.
use omni_calc::{Calculation, engine::evaluate};
use std::time::{Duration, Instant};

fn calc(input: &str) -> Calculation {
    evaluate(input, Instant::now() + Duration::from_secs(1))
        .unwrap_or_else(|error| panic!("{input}: {error}"))
}

#[test]
fn nested_function_powers_and_signed_exponents_keep_standard_precedence() {
    let result = calc("sqrt(3^2+4^2)^2 + 2^3^2");
    assert_eq!(result.value, 537);
    assert!(result.exact);
    let result = calc("-sin(pi/6)^2 + 2^-3");
    assert_eq!(result.display, "-0.125");
    assert!(result.exact);
}

#[test]
fn kinetic_energy_and_temperature_conversions_preserve_dimensions_and_offsets() {
    // 36 km/h = 10 m/s; kinetic energy = 80 * 10^2 / 2 joules.
    let energy = calc("(1/2 * 80 kg * (36 km/h)^2) to J");
    assert_eq!(energy.display, "4000 J");
    assert_eq!(energy.kind, "quantity");
    assert!(energy.exact);
    let temperature = calc("convert(20 °C,°F)");
    assert_eq!(temperature.display, "68 °F");
    assert!(temperature.exact);
}

#[test]
fn exact_outcomes_survive_large_intermediates_and_tool_composition() {
    for input in ["(2^200+1)-2^200", "ncr(100,50)/(2*ncr(99,49))"] {
        let result = calc(input);
        assert_eq!(result.value, 1, "{input}");
        assert!(result.exact, "{input}");
    }
    let magnitude = calc("conj(3+4i)*(3+4i)");
    assert_eq!(magnitude.value, 25);
    assert!(magnitude.exact);
}

#[test]
fn invalid_domains_fail_through_conversion_and_nested_call_paths() {
    for input in [
        "convert(2 m,s)",
        "sqrt(log(8,1))",
        "integrate(log(x,1),x,1,2)",
    ] {
        let error = evaluate(input, Instant::now() + Duration::from_secs(1)).unwrap_err();
        assert_eq!(error.code, "math_error", "{input}: {error}");
    }
}
