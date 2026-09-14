use omni_calc::{Calculation, symbolic::try_evaluate};
use std::time::{Duration, Instant};

fn calc(input: &str) -> Calculation {
    try_evaluate(input, Instant::now() + Duration::from_secs(2))
        .expect("recognized symbolic operation")
        .unwrap_or_else(|e| panic!("{input}: {e}"))
}

fn expression(result: &Calculation) -> &str {
    result
        .value
        .as_str()
        .or_else(|| result.value.get("expression").and_then(|v| v.as_str()))
        .expect("symbolic expression")
}

fn substitute_value(expr: &str, at: &str) -> String {
    expression(&calc(&format!("substitute({expr},x,{at})"))).to_owned()
}

fn numeric(expr: &str, at: f64) -> f64 {
    let mut context = fend_core::Context::new();
    let source = format!("x = ({at}); ({expr}) to 15 sf");
    let result =
        fend_core::evaluate(&source, &mut context).unwrap_or_else(|e| panic!("{source}: {e}"));
    result
        .get_main_result()
        .trim_start_matches("approx. ")
        .parse()
        .unwrap_or_else(|_| panic!("nonnumeric result: {}", result.get_main_result()))
}

#[test]
fn exact_decimals_big_integers_and_operator_precedence() {
    assert_eq!(expression(&calc("simplify(0.1 + 0.2)")), "3/10");
    assert_eq!(
        expression(&calc("simplify(2^100)")),
        "1267650600228229401496703205376"
    );
    assert_eq!(expression(&calc("simplify(2^3^2)")), "512");
    assert_eq!(expression(&calc("simplify(-2^2)")), "-4");
    assert_eq!(expression(&calc("simplify(2^-3)")), "1/8");
    assert_eq!(expression(&calc("simplify(1e-3+0.009)")), "1/100");
    assert_eq!(expression(&calc("simplify(2x+3x-x)")), "4*x");
    assert_eq!(expression(&calc("simplify(sin(pi/6))")), "1/2");
    assert_eq!(expression(&calc("simplify(cos(3*pi))")), "-1");
    assert_eq!(expression(&calc("simplify(sin(-pi/2))")), "-1");
    assert_eq!(expression(&calc("simplify(tan(pi/4))")), "1");
}

#[test]
fn expansions_match_independent_integer_polynomials() {
    let result = calc("expand((x+2)^5)");
    for x in [-4, -2, -1, 0, 1, 3] {
        assert_eq!(
            substitute_value(expression(&result), &x.to_string()),
            (x as i64 + 2).pow(5).to_string()
        );
    }
    let difference = calc("expand((x+1)*(x-1)-(x^2-1))");
    assert_eq!(expression(&difference), "0");
    assert_eq!(expression(&calc("expand((a+b)*(a-b)-a^2+b^2)")), "0");
}

#[test]
fn factorizations_preserve_polynomial_values() {
    for original in ["x^2-1", "2*x^3-12*x^2+22*x-12", "x^4", "x^2+x+1"] {
        let factored = calc(&format!("factor({original})"));
        for x in [-3, -1, 0, 1, 2, 4] {
            let x = x.to_string();
            assert_eq!(
                substitute_value(expression(&factored), &x),
                substitute_value(original, &x),
                "{original} -> {}",
                expression(&factored)
            );
        }
    }
    assert!(expression(&calc("factor(x^2-1)")).contains('*'));
}

#[test]
fn derivatives_match_closed_form_values() {
    let result = calc("diff(x^5-3*x^2+7,x)");
    for x in [-3, -1, 0, 1, 2, 4] {
        assert_eq!(
            substitute_value(expression(&result), &x.to_string()),
            (5_i64 * (x as i64).pow(4) - 6 * x as i64).to_string()
        );
    }
    assert_eq!(expression(&calc("diff(x^5,x,5)")), "120");
    assert_eq!(expression(&calc("diff(x^5,x,6)")), "0");
    assert_eq!(expression(&calc("diff(y*x^2,x)")), "2*x*y");
}

#[test]
fn chain_and_quotient_derivatives_match_independent_numerics() {
    type Case = (&'static str, fn(f64) -> f64);
    let cases: [Case; 7] = [
        ("sin(x^2)", |x| 2.0 * x * (x * x).cos()),
        ("exp(3*x)", |x| 3.0 * (3.0 * x).exp()),
        ("ln(x^2+1)", |x| 2.0 * x / (x * x + 1.0)),
        ("x/(x+1)", |x| 1.0 / (x + 1.0).powi(2)),
        ("atan(x)", |x| 1.0 / (x * x + 1.0)),
        ("tan(x)", |x| 1.0 / x.cos().powi(2)),
        ("tanh(x)", |x| 1.0 / x.cosh().powi(2)),
    ];
    for (source, expected) in cases {
        let result = calc(&format!("diff({source},x)"));
        for x in [0.25, 0.7, 1.5] {
            let actual = numeric(expression(&result), x);
            assert!(
                (actual - expected(x)).abs() < 1e-10,
                "{source} at {x}: {actual} != {}",
                expected(x)
            );
        }
    }
}

#[test]
fn antiderivatives_differentiate_to_the_integrand_numerically() {
    type Case = (&'static str, fn(f64) -> f64);
    let cases: [Case; 14] = [
        ("3*x^2-4*x+7", |x| 3.0 * x * x - 4.0 * x + 7.0),
        ("sin(2*x+1)", |x| (2.0 * x + 1.0).sin()),
        ("cos(3*x)", |x| (3.0 * x).cos()),
        ("exp(2*x)", |x| (2.0 * x).exp()),
        ("1/x", |x| 1.0 / x),
        ("ln(x)", |x| x.ln()),
        ("sqrt(x)", |x| x.sqrt()),
        ("(x+1)*(x-1)", |x| x * x - 1.0),
        ("x*exp(2*x)", |x| x * (2.0 * x).exp()),
        ("x^2*sin(3*x)", |x| x * x * (3.0 * x).sin()),
        ("1/(x^2+1)", |x| 1.0 / (x * x + 1.0)),
        ("1/(x^2-1)", |x| 1.0 / (x * x - 1.0)),
        ("1/sqrt(x^2+1)", |x| 1.0 / (x * x + 1.0).sqrt()),
        ("2^x", |x| 2.0_f64.powf(x)),
    ];
    for (source, expected) in cases {
        let primitive = calc(&format!("integrate({source},x)"));
        assert_eq!(primitive.kind, "antiderivative");
        let derivative = calc(&format!("diff({},x)", expression(&primitive)));
        for x in [0.3, 0.8, 1.7] {
            let actual = numeric(expression(&derivative), x);
            assert!(
                (actual - expected(x)).abs() < 1e-9,
                "integral {source}, derivative {} at {x}: {actual} != {}",
                expression(&derivative),
                expected(x)
            );
        }
    }
}

#[test]
fn equations_return_all_exact_roots_including_complex() {
    let roots = calc("solve(x^2-5*x+6=0,x)");
    assert_eq!(roots.value["solutions"], serde_json::json!(["2", "3"]));
    let roots = calc("solve(x^2+1,x)");
    let roots = roots.value["solutions"].as_array().unwrap();
    assert!(roots.contains(&serde_json::json!("i")));
    assert!(roots.contains(&serde_json::json!("-i")));
    assert_eq!(
        calc("solve(3*x+2=8,x)").value["solutions"],
        serde_json::json!(["2"])
    );
    assert_eq!(
        calc("solve((x-2)^4,x)").value["solutions"],
        serde_json::json!(["2"])
    );
    assert_eq!(calc("solve(x=x,x)").value["all_values"], true);
    assert_eq!(
        calc("solve(x=x+1,x)").value["solutions"],
        serde_json::json!([])
    );
    let roots = calc("solve(x^2-2,x)");
    for root in roots.value["solutions"].as_array().unwrap() {
        let root = numeric(root.as_str().unwrap(), 0.0);
        assert!((root * root - 2.0).abs() < 1e-12);
    }
}

#[test]
fn domain_conditions_survive_cancellation_and_differentiation() {
    let result = calc("simplify(x/x)");
    assert_eq!(expression(&result), "1");
    assert!(result.display.contains("x != 0"));
    assert!(calc("simplify(x^(1-2)*x)").display.contains("x != 0"));
    assert!(calc("simplify(x^(1/2)/x^(1/2))").display.contains("x >= 0"));
    assert_eq!(expression(&calc("simplify(sqrt(x^2))")), "sqrt(x^2)");
    assert!(calc("diff(abs(x),x)").display.contains("x != 0"));
    assert!(calc("diff(x^x,x)").display.contains("x > 0"));
    assert!(calc("diff(sqrt(x),x)").display.contains("sqrt(x) != 0"));
    for source in [
        "substitute(x/x,x,0)",
        "simplify(1/0)",
        "simplify(0^0)",
        "simplify(ln(-1))",
        "simplify(sqrt(-1))",
        "simplify(asin(2))",
        "simplify(tan(pi/2))",
    ] {
        let error = try_evaluate(source, Instant::now() + Duration::from_secs(1))
            .unwrap()
            .unwrap_err();
        assert_eq!(error.code, "math_error", "{source}: {error}");
    }
}

#[test]
fn unsupported_operations_never_invent_answers() {
    for source in [
        "integrate(exp(-x^2),x)",
        "solve(sin(x)=0,x)",
        "solve((x^2-1)/(x-1)=0,x)",
        "factor(x*y+x)",
        "diff(mystery(x),x)",
    ] {
        let error = try_evaluate(source, Instant::now() + Duration::from_secs(1))
            .unwrap()
            .unwrap_err();
        assert_eq!(error.code, "unsupported_operation", "{source}: {error}");
    }
    assert!(
        try_evaluate(
            "integrate(x,x,0,1)",
            Instant::now() + Duration::from_secs(1)
        )
        .is_none()
    );
    assert!(try_evaluate("1+1", Instant::now() + Duration::from_secs(1)).is_none());
    assert!(
        try_evaluate(
            "derivative_at(x^2,x,2)",
            Instant::now() + Duration::from_secs(1)
        )
        .is_none()
    );
    assert!(
        try_evaluate(
            "solve_linear([[1,2],[3,4]],[5,6])",
            Instant::now() + Duration::from_secs(1)
        )
        .is_none()
    );
}

#[test]
fn malformed_and_adversarial_inputs_fail_boundedly() {
    for source in [
        "simplify()",
        "simplify(1+)",
        "simplify(1..2)",
        "simplify(1)),x(",
        "diff(x)",
        "diff(x,x,-1)",
        "diff(x,pi)",
        "solve(x==1,x)",
        "simplify(💀)",
        "simplify(1,2)",
    ] {
        assert!(
            try_evaluate(source, Instant::now() + Duration::from_secs(1))
                .unwrap()
                .is_err(),
            "{source}"
        );
    }
    for source in [
        "expand((a+b+c+d+e+f)^64)".to_owned(),
        "diff(x,x,100)".to_owned(),
        "simplify(2^1000000)".to_owned(),
        format!("simplify({}x{})", "(".repeat(80), ")".repeat(80)),
        format!("simplify({})", vec!["x"; 500].join("+")),
    ] {
        let error = try_evaluate(&source, Instant::now() + Duration::from_millis(200))
            .unwrap()
            .unwrap_err();
        assert_eq!(error.code, "resource_limit", "{source}: {error}");
    }
    assert_eq!(
        try_evaluate("simplify(1+1)", Instant::now())
            .unwrap()
            .unwrap_err()
            .code,
        "resource_limit"
    );
}

#[test]
fn arbitrary_ascii_input_does_not_panic() {
    let alphabet = b"xy01.23456789+-*/^(),_[]=! abc\t";
    let mut seed = 0x105a_7263_d9b4_e807u64;
    for length in 0..256 {
        let mut input = String::from("simplify(");
        for _ in 0..length % 96 {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            input.push(alphabet[seed as usize % alphabet.len()] as char);
        }
        input.push(')');
        let outcome = std::panic::catch_unwind(|| {
            try_evaluate(&input, Instant::now() + Duration::from_millis(50))
        });
        assert!(outcome.is_ok(), "parser panicked on {input:?}");
    }
}
