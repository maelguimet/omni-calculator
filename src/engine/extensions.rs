use super::*;
use num_bigint::BigUint;
use num_traits::{One, Zero};

pub(super) fn known(name: &str) -> bool {
    statistics::known(name)
        || matrix::known(name)
        || calculus::known(name)
        || matches!(
            name,
            "ncr"
                | "choose"
                | "npr"
                | "permute"
                | "gcd"
                | "lcm"
                | "factorial"
                | "log"
                | "mod"
                | "sqrt"
                | "cbrt"
                | "round"
                | "floor"
                | "ceil"
                | "trunc"
                | "sign"
                | "atan2"
                | "hypot"
                | "sind"
                | "cosd"
                | "tand"
                | "asind"
                | "acosd"
                | "atand"
                | "sec"
                | "csc"
                | "cot"
                | "conj"
                | "conjugate"
                | "convert"
                | "gamma"
                | "lgamma"
                | "beta"
                | "erf"
                | "erfc"
        )
}

pub(super) fn evaluate(
    name: &str,
    args: &[&str],
    deadline: Instant,
    depth: usize,
) -> Option<Result<Calculation, CalcError>> {
    if statistics::known(name) {
        return Some(statistics::evaluate(name, args, deadline, depth));
    }
    if matrix::known(name) {
        return Some(matrix::evaluate(name, args, deadline, depth));
    }
    if calculus::known(name) {
        return Some(calculus::evaluate(name, args, deadline, depth));
    }
    if !known(name) {
        return None;
    }
    Some(simple(name, args, deadline, depth))
}

pub(super) fn natural(input: &str, deadline: Instant, depth: usize) -> Result<u64, CalcError> {
    let value = signed_integer(input, deadline, depth)?;
    u64::try_from(value).map_err(|_| CalcError::math("Expected a nonnegative integer"))
}

pub(super) fn signed_integer(
    input: &str,
    deadline: Instant,
    depth: usize,
) -> Result<i64, CalcError> {
    // Validate integrality before conversion to f64, which would round inputs
    // such as 3.00000000000000001 into an incorrectly accepted integer.
    let (expanded, _) = expand_calls(input, deadline, depth + 1)?;
    let exact = raw_fend(&format!("({expanded}) to fraction"), deadline)?;
    let number = exact.get_main_result().parse::<i64>().map_err(|_| {
        CalcError::math("Expected an exact integer with magnitude no larger than 2^53 - 1")
    })?;
    if number.unsigned_abs() > 9_007_199_254_740_991 {
        return Err(CalcError::math("Integer magnitude exceeds 2^53 - 1"));
    }
    Ok(number)
}

fn integer_result(value: BigUint) -> Result<Calculation, CalcError> {
    let display = value.to_string();
    if display.len() > MAX_OUTPUT {
        return Err(CalcError::limit("Integer result exceeds the output limit"));
    }
    let value = display
        .parse::<u64>()
        .map(Value::from)
        .unwrap_or_else(|_| Value::String(display.clone()));
    Ok(Calculation {
        display,
        value,
        kind: "integer".into(),
        exact: true,
    })
}

fn simple(
    name: &str,
    args: &[&str],
    deadline: Instant,
    depth: usize,
) -> Result<Calculation, CalcError> {
    check_time(deadline)?;
    if matches!(name, "sqrt" | "cbrt") {
        arity(args, 1, name)?;
        let (argument, exact) = expand_calls(args[0], deadline, depth + 1)?;
        let mut result = if let Ok(x) = scalar(&argument, deadline, depth) {
            if x < 0.0 {
                evaluate_fend(
                    &if name == "sqrt" {
                        format!("sqrt(-({argument}))*i")
                    } else {
                        format!("-cbrt(-({argument}))")
                    },
                    deadline,
                )?
            } else {
                evaluate_fend(&format!("{name}({argument})"), deadline)?
            }
        } else {
            evaluate_fend(&format!("{name}({argument})"), deadline)?
        };
        result.exact &= exact;
        return Ok(result);
    }
    if name == "convert" {
        arity(args, 2, name)?;
        let (argument, exact) = expand_calls(args[0], deadline, depth + 1)?;
        let mut result = evaluate_fend(&format!("({argument}) to {}", args[1]), deadline)?;
        result.exact &= exact;
        return Ok(result);
    }
    if matches!(name, "conj" | "conjugate") {
        arity(args, 1, name)?;
        let (argument, exact) = expand_calls(args[0], deadline, depth + 1)?;
        let mut result =
            evaluate_fend(&format!("real({argument}) - i*imag({argument})"), deadline)?;
        result.exact &= exact;
        return Ok(result);
    }
    if name == "log" && args.len() == 1 {
        let (argument, exact) = expand_calls(args[0], deadline, depth + 1)?;
        let mut result = evaluate_fend(&format!("log({argument})"), deadline)?;
        result.exact &= exact;
        return Ok(result);
    }
    if matches!(name, "sec" | "csc" | "cot") {
        arity(args, 1, name)?;
        let (argument, exact) = expand_calls(args[0], deadline, depth + 1)?;
        let function = match name {
            "sec" => "cos",
            "csc" => "sin",
            _ => "tan",
        };
        let mut result = evaluate_fend(&format!("1/{function}({argument})"), deadline)?;
        result.exact &= exact;
        return Ok(result);
    }
    if name == "mod" {
        arity(args, 2, name)?;
        let (a, exact_a) = expand_calls(args[0], deadline, depth + 1)?;
        let (b, exact_b) = expand_calls(args[1], deadline, depth + 1)?;
        let mut result = evaluate_fend(&format!("({a}) mod ({b})"), deadline)?;
        result.exact &= exact_a && exact_b;
        return Ok(result);
    }
    if name == "factorial" {
        arity(args, 1, name)?;
        let n = natural(args[0], deadline, depth)?;
        if n > 5000 {
            return Err(CalcError::limit("Factorial is limited to 5000"));
        }
        let mut result = BigUint::one();
        for n in 2..=n {
            check_time(deadline)?;
            result *= n;
        }
        return integer_result(result);
    }
    if matches!(name, "ncr" | "choose" | "npr" | "permute") {
        arity(args, 2, name)?;
        let n = natural(args[0], deadline, depth)?;
        let mut k = natural(args[1], deadline, depth)?;
        if k > n {
            return Err(CalcError::math("k must not exceed n"));
        }
        let combination = matches!(name, "ncr" | "choose");
        if combination {
            k = k.min(n - k);
        }
        if k > 5000 {
            return Err(CalcError::limit(
                "Combinatorics is limited to 5000 multiplication steps",
            ));
        }
        let mut result = BigUint::one();
        for i in 0..k {
            check_time(deadline)?;
            result *= n - i;
            if combination {
                result /= i + 1;
            }
            if result.bits() > 100_000 {
                return Err(CalcError::limit("Integer result exceeds the output limit"));
            }
        }
        return integer_result(result);
    }
    if matches!(name, "gcd" | "lcm") {
        if args.len() < 2 {
            return Err(CalcError::invalid("gcd/lcm require at least two integers"));
        }
        let mut result = BigUint::from(natural(args[0], deadline, depth)?);
        for arg in &args[1..] {
            let value = BigUint::from(natural(arg, deadline, depth)?);
            let mut a = result.clone();
            let mut b = value.clone();
            while !b.is_zero() {
                check_time(deadline)?;
                let r = a % &b;
                a = b;
                b = r;
            }
            result = if name == "gcd" {
                a
            } else if a.is_zero() {
                BigUint::zero()
            } else {
                result / a * value
            };
        }
        return integer_result(result);
    }
    let expected = if matches!(name, "log" | "mod" | "atan2" | "hypot" | "beta") {
        2
    } else {
        1
    };
    if name == "round" && args.len() == 2 {
    } else {
        arity(args, expected, name)?;
    }
    let x = scalar(args[0], deadline, depth)?;
    let y = if args.len() > 1 {
        scalar(args[1], deadline, depth)?
    } else {
        0.0
    };
    let result = match name {
        "log" => {
            if x <= 0.0 || y <= 0.0 || y == 1.0 {
                return Err(CalcError::math(
                    "Real logarithm requires x > 0 and a positive base other than 1",
                ));
            }
            x.log(y)
        }
        "round" => {
            if args.len() == 2 {
                signed_integer(args[1], deadline, depth)?;
            }
            if y.fract() != 0.0 || !(-308.0..=308.0).contains(&y) {
                return Err(CalcError::math(
                    "Decimal places must be an integer from -308 through 308",
                ));
            }
            if y >= 0.0 {
                let scale = 10_f64.powf(y);
                (x * scale).round() / scale
            } else {
                let scale = 10_f64.powf(-y);
                (x / scale).round() * scale
            }
        }
        "floor" => x.floor(),
        "ceil" => x.ceil(),
        "trunc" => x.trunc(),
        "sign" => {
            if x == 0.0 {
                0.0
            } else {
                x.signum()
            }
        }
        "atan2" => {
            if x == 0.0 && y == 0.0 {
                return Err(CalcError::math("atan2 is undefined at (0,0)"));
            }
            x.atan2(y)
        }
        "hypot" => x.hypot(y),
        "sind" => x.to_radians().sin(),
        "cosd" => x.to_radians().cos(),
        "tand" => {
            if (x.rem_euclid(180.0) - 90.0).abs() < 1e-12 {
                return Err(CalcError::math(
                    "Tangent is undefined at odd multiples of 90 degrees",
                ));
            }
            x.to_radians().tan()
        }
        "asind" => x.asin().to_degrees(),
        "acosd" => x.acos().to_degrees(),
        "atand" => x.atan().to_degrees(),
        "gamma" | "lgamma" => {
            if x <= 0.0 && x.fract() == 0.0 {
                return Err(CalcError::math("Gamma has poles at nonpositive integers"));
            }
            if name == "gamma" {
                statrs::function::gamma::gamma(x)
            } else {
                statrs::function::gamma::ln_gamma(x)
            }
        }
        "beta" => {
            if x <= 0.0 || y <= 0.0 {
                return Err(CalcError::math("Beta requires positive arguments"));
            }
            statrs::function::beta::beta(x, y)
        }
        "erf" => statrs::function::erf::erf(x),
        "erfc" => statrs::function::erf::erfc(x),
        _ => return Err(CalcError::invalid("Unknown function")),
    };
    number(result)
}
