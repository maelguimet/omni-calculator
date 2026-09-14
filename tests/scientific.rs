use omni_calc::{Calculation, engine::evaluate};
use std::time::{Duration, Instant};

fn calc(expression: &str) -> Calculation {
    evaluate(expression, Instant::now() + Duration::from_secs(2))
        .unwrap_or_else(|e| panic!("{expression}: {e}"))
}
fn numeric(expression: &str) -> f64 {
    let result = calc(expression);
    if let Some(value) = result.value.as_f64() {
        return value;
    }
    let text = result.display.trim_start_matches("approx. ");
    if let Some((a, b)) = text.split_once('/') {
        return a.trim().parse::<f64>().unwrap() / b.trim().parse::<f64>().unwrap();
    }
    text.parse()
        .unwrap_or_else(|_| panic!("{expression}: nonnumeric result {result:?}"))
}
fn close(expression: &str, expected: f64, tolerance: f64) {
    let value = numeric(expression);
    assert!(
        (value - expected).abs() <= tolerance * (1.0 + expected.abs()),
        "{expression}: {value} != {expected}"
    );
}
fn rejects(expression: &str) {
    assert!(
        evaluate(expression, Instant::now() + Duration::from_secs(1)).is_err(),
        "unexpectedly accepted {expression}"
    );
}

#[test]
fn exact_arithmetic_precedence_and_big_integers() {
    assert_eq!(calc("1 + 2 * 3^2").display, "19");
    assert_eq!(calc("1/3 + 1/6").display, "0.5");
    assert_eq!(calc("1/3").display, "1/3");
    assert_eq!(calc("0.1+0.2").display, "0.3");
    assert_eq!(calc("30!").display, "265252859812191058636308480000000");
    assert!(calc("30!").exact);
    assert_eq!(calc("2^100").display, "1267650600228229401496703205376");
    assert_eq!(calc("-2^2").display, "-4");
    assert_eq!(calc("5 mod 2").display, "1");
    assert_eq!(calc("255 to hex").display, "ff");
}

#[test]
fn complex_scientific_functions_and_units() {
    assert_eq!(calc("sqrt(-1)").display, "i");
    assert_eq!(calc("(2+3i)*(4-5i)").display, "23 + 2i");
    assert_eq!(calc("conj(2+3i)").display, "2 - 3i");
    close("sin(pi/6)", 0.5, 1e-10);
    close("ln(exp(2))", 2.0, 1e-9);
    close("log(1024,2)", 10.0, 1e-12);
    close("log(100)", 2.0, 1e-12);
    close("sqrt(mean([9,9]))", 3.0, 1e-12);
    assert!(!calc("sqrt(mean([9,9]))").exact);
    assert!(!calc("mean([1,3])+1").exact);
    close("mod(mean([5,5]),2)", 1.0, 1e-12);
    close("sind(30)", 0.5, 1e-12);
    close("cbrt(-8)", -2.0, 1e-12);
    close("sec(pi/3)", 2.0, 1e-12);
    rejects("sec(pi/2)");
    assert!(calc("5 km to m").display.contains("5000"));
    assert!(calc("32 °F to °C").display.starts_with("0"));
    assert!(calc("(10 m/s)*3 s").display.contains("30"));
    rejects("1 m + 2 s");
}

#[test]
fn statistics_use_explicit_population_and_sample_conventions() {
    close("mean([2,4,4,4,5,5,7,9])", 5.0, 1e-13);
    close("variance([2,4,4,4,5,5,7,9])", 4.0, 1e-13);
    close("stdev([2,4,4,4,5,5,7,9])", 2.0, 1e-13);
    close("sample_variance([1,2,3])", 1.0, 1e-13);
    close(
        "variance([1000000001,1000000002,1000000003])",
        2.0 / 3.0,
        1e-13,
    );
    close("mean([1e308,1e308])", 1e308, 1e-13);
    close("rms([1e308,1e308])", 1e308, 1e-13);
    close("sum([10000000000000000,1,-10000000000000000])", 1.0, 1e-13);
    close("median([9,1,5,3])", 4.0, 1e-13);
    close("quantile([1,2,3,4],0.25)", 1.75, 1e-13);
    close("percentile([1,2,3,4],75)", 3.25, 1e-13);
    close("correlation([1,2,3],[2,4,6])", 1.0, 1e-12);
    let regression = calc("linreg([1,2,3],[3,5,7])");
    assert_eq!(regression.value["slope"], 2.0);
    assert_eq!(regression.value["intercept"], 1.0);
    close("mean([sqrt(4),ncr(4,2)]) + det([[1,0],[0,2]])", 6.0, 1e-12);
    rejects("mean([])");
    rejects("sample_variance([1])");
    rejects("quantile([1,2],1.1)");
    rejects("correlation([1,1],[2,3])");
    rejects("mean([1 m,2 m])");
}

#[test]
fn combinatorics_and_special_functions() {
    assert_eq!(calc("ncr(52,5)").display, "2598960");
    assert_eq!(calc("npr(10,3)").display, "720");
    assert_eq!(calc("gcd(84,126,210)").display, "42");
    assert_eq!(calc("lcm(12,18)").display, "36");
    assert_eq!(
        calc("factorial(30)").display,
        "265252859812191058636308480000000"
    );
    close("gamma(6)", 120.0, 1e-12);
    close("erf(0)", 0.0, 1e-12);
    close("beta(2,3)", 1.0 / 12.0, 1e-12);
    rejects("ncr(3,4)");
    rejects("factorial(-1)");
    rejects("gamma(0)");
}

#[test]
fn distributions_match_independent_closed_forms() {
    close(
        "normal_pdf(0)",
        1.0 / (2.0 * std::f64::consts::PI).sqrt(),
        1e-12,
    );
    close("normal_cdf(0)", 0.5, 1e-12);
    close("normal_quantile(0.5,10,2)", 10.0, 1e-12);
    close("binomial_pmf(3,5,0.5)", 10.0 / 32.0, 1e-12);
    close("binomial_cdf(2,5,0.5)", 0.5, 1e-12);
    close("poisson_pmf(0,2)", (-2.0_f64).exp(), 1e-12);
    close("exponential_cdf(2,0.5)", 1.0 - (-1.0_f64).exp(), 1e-12);
    close("gamma_pdf(2,1,2)", 0.5 * (-1.0_f64).exp(), 1e-12);
    close("student_t_cdf(0,5)", 0.5, 1e-12);
    close("chi2_cdf(2,2)", 1.0 - (-1.0_f64).exp(), 1e-12);
    rejects("normal_pdf(0,0,-1)");
    rejects("normal_quantile(1.2)");
    rejects("normal_quantile(0)");
    rejects("binomial_pmf(0.5,4,0.2)");
}

#[test]
fn matrix_operations_obey_inverse_and_solve_residuals() {
    close("det([[1,2],[3,4]])", -2.0, 1e-12);
    let identity = calc("matmul([[4,7],[2,6]],inverse([[4,7],[2,6]]))");
    for r in 0..2 {
        for c in 0..2 {
            assert!(
                (identity.value[r][c].as_f64().unwrap() - if r == c { 1.0 } else { 0.0 }).abs()
                    < 1e-12
            );
        }
    }
    let solution = calc("solve_linear([[3,2],[1,2]],[5,5])");
    let x = solution.value[0].as_f64().unwrap();
    let y = solution.value[1].as_f64().unwrap();
    assert!((3.0 * x + 2.0 * y - 5.0).abs() < 1e-12);
    assert!((x + 2.0 * y - 5.0).abs() < 1e-12);
    close("rank([[1,2],[2,4]])", 1.0, 1e-12);
    assert_eq!(
        calc("matrix_power([[1,1],[1,0]],5)").value,
        serde_json::json!([[8.0, 5.0], [5.0, 3.0]])
    );
    assert_eq!(
        calc("cross([1,0,0],[0,1,0])").value,
        serde_json::json!([0.0, 0.0, 1.0])
    );
    close("norm([3,4])", 5.0, 1e-12);
    let eigen = calc("eigenvalues([[0,-1],[1,0]])");
    assert_eq!(eigen.kind, "complex_vector");
    for value in eigen.value.as_array().unwrap() {
        assert!(value["re"].as_f64().unwrap().abs() < 1e-12);
        assert!((value["im"].as_f64().unwrap().abs() - 1.0).abs() < 1e-12);
    }
    rejects("inverse([[1,2],[2,4]])");
    rejects("det([[1,2,3],[4,5,6]])");
    rejects("matmul([[1,2]],[[3,4]])");
    rejects("det([[1,2],[3]])");
    rejects("normalize([0,0])");
}

#[test]
fn numerical_calculus_matches_analytic_oracles() {
    close("integrate(x^2,x,0,3)", 9.0, 1e-10);
    close("integrate(sin(x),x,0,pi)", 2.0, 1e-10);
    close("integrate(exp(-x^2),x,-3,3)", 1.7724146965190425, 1e-10);
    close("integrate(x^2,x,3,0)", -9.0, 1e-10);
    close("root(x^3-2,x,0,2)", 2.0_f64.cbrt(), 1e-9);
    close("derivative_at(sin(x),x,0)", 1.0, 1e-9);
    close("derivative_at(x^3,x,2)", 12.0, 1e-8);
    close("summation(k^2,k,1,100)", 338350.0, 1e-12);
    close("product_range(k,k,1,10)", 3628800.0, 1e-12);
    close("integrate(2x,x,0,3)", 9.0, 1e-10);
    rejects("root(x^2+1,x,-1,1)");
    rejects("root(1/x,x,-1,1)");
    rejects("integrate(1/x,x,-1,1)");
    rejects("integrate(sqrt(x),x,-1,1)");
    rejects("derivative_at(abs(x),x,0)");
}

#[test]
fn malformed_inputs_and_resource_limits_fail_cleanly() {
    for expression in [
        "",
        "what's the weather",
        "1/0",
        "(1+2",
        "2+3)*(1+2",
        "mean([1,,2])",
        "det([[1],[2,3]])",
        "log(0,10)",
        "tand(90)",
        "normal_pdf(0,0,0)",
        "print(2)",
        "sin",
        "1;2",
        "\\x.x",
        "1\u{1b}",
    ] {
        rejects(expression);
    }
    let deep = format!("{}1{}", "(".repeat(65), ")".repeat(65));
    rejects(&deep);
    rejects(&"1".repeat(4097));
    let error = evaluate("1+1", Instant::now() - Duration::from_secs(1)).unwrap_err();
    assert_eq!(error.code, "resource_limit");
}

#[test]
fn scale_and_cancellation_regressions_cover_related_operations() {
    close("condition([[1e200]])", 1.0, 1e-12);
    close("rank([[1e200,2e200],[2e200,4e200]])", 1.0, 1e-12);
    assert!(
        (calc("solve_linear([[1e200]],[2e200])").value[0]
            .as_f64()
            .unwrap()
            - 2.0)
            .abs()
            < 1e-12
    );
    assert!(
        (calc("eigenvalues([[1e200]])").value[0].as_f64().unwrap() / 1e200 - 1.0).abs() < 1e-12
    );
    assert!(
        (calc("inverse([[1e-200]])").value[0][0].as_f64().unwrap() / 1e200 - 1.0).abs() < 1e-12
    );
    assert_eq!(
        calc("matmul([[1e16,1,-1e16]],[[1],[1],[1]])").value,
        serde_json::json!([[1.0]])
    );
    close("correlation([1e200,-1e200],[1e200,-1e200])", 1.0, 1e-12);
    close("stdev([1e200,-1e200])", 1e200, 1e-12);
    close("derivative_at(1e20+x,x,1)", 1.0, 1e-12);
    close("root(1e-30*(x-1)^3,x,0,4)", 1.0, 1e-9);
    rejects("integrate(exp(-((x-0.1234567)/0.000001)^2),x,0,1)");
    rejects("factorial(3.00000000000000001)");
    rejects("binomial_pmf(0.00000000000000001,5,0.5)");
    rejects("matrix_power([[1]],2.00000000000000001)");
    rejects("summation(x,x,1.00000000000000001,3)");
    rejects("round(1.2345,2.00000000000000001)");
    rejects("root(exp(-1000*x),x,1,2)");
}
