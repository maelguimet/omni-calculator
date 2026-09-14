//! Bounded, deterministic mathematical execution. Models never execute code.

mod calculus;
mod extensions;
mod matrix;
mod statistics;

use crate::{Calculation, error::CalcError};
use num_bigint::{BigInt, Sign};
use num_rational::BigRational;
use serde_json::Value;
use std::time::Instant;

pub(crate) const MAX_ITEMS: usize = 512;
const MAX_EXPRESSION: usize = 4096;
const MAX_OUTPUT: usize = 32768;

struct Deadline(Instant);
impl fend_core::Interrupt for Deadline {
    fn should_interrupt(&self) -> bool {
        Instant::now() >= self.0
    }
}

pub(crate) fn check_time(deadline: Instant) -> Result<(), CalcError> {
    if Instant::now() >= deadline {
        Err(CalcError::limit("Calculation exceeded its time budget"))
    } else {
        Ok(())
    }
}

/// Evaluate the calculator DSL with a caller-supplied, monotonic deadline.
pub fn evaluate(expression: &str, deadline: Instant) -> Result<Calculation, CalcError> {
    validate(expression)?;
    evaluate_inner(expression.trim(), deadline, 0)
}

fn evaluate_inner(
    expression: &str,
    deadline: Instant,
    depth: usize,
) -> Result<Calculation, CalcError> {
    check_time(deadline)?;
    if depth > 32 {
        return Err(CalcError::limit("Too many nested function calls"));
    }
    if let Some(result) = crate::symbolic::try_evaluate(expression, deadline) {
        return result;
    }
    if let Some((name, args)) = call(expression)?
        && let Some(result) = extensions::evaluate(&name, &args, deadline, depth)
    {
        return result;
    }
    // Scalar extension results can also be used inside ordinary arithmetic.
    if expression.starts_with('[') && expression.ends_with(']') {
        if expression[1..].trim_start().starts_with('[') {
            return matrix::result(matrix::parse(expression, deadline, depth)?);
        }
        return vector_result(vector(expression, deadline, depth)?);
    }
    let (expanded, exact) = expand_calls(expression, deadline, depth)?;
    let mut result = evaluate_fend(&expanded, deadline)?;
    result.exact &= exact;
    Ok(result)
}

fn expand_calls(
    expression: &str,
    deadline: Instant,
    depth: usize,
) -> Result<(String, bool), CalcError> {
    if depth > 64 {
        return Err(CalcError::limit("Too many nested function calls"));
    }
    let mut out = String::new();
    let mut exact = true;
    let mut pos = 0;
    let bytes = expression.as_bytes();
    while pos < bytes.len() {
        if bytes[pos].is_ascii_alphabetic() || bytes[pos] == b'_' {
            let start = pos;
            while pos < bytes.len() && (bytes[pos].is_ascii_alphanumeric() || bytes[pos] == b'_') {
                pos += 1;
            }
            let end_name = pos;
            while pos < bytes.len() && bytes[pos].is_ascii_whitespace() {
                pos += 1;
            }
            if pos < bytes.len() && bytes[pos] == b'(' {
                let end = matching_end(expression, pos)?;
                let name = expression[start..end_name].to_ascii_lowercase();
                if extensions::known(&name) {
                    let result = evaluate_inner(&expression[start..=end], deadline, depth + 1)?;
                    exact &= result.exact;
                    if matches!(
                        result.kind.as_str(),
                        "number" | "integer" | "rational" | "complex"
                    ) {
                        out.push('(');
                        out.push_str(
                            result
                                .display
                                .strip_prefix("approx. ")
                                .unwrap_or(&result.display),
                        );
                        out.push(')');
                    } else {
                        return Err(CalcError::invalid(
                            "A vector or matrix result cannot be used as a scalar; use an explicit matrix/vector function",
                        ));
                    }
                } else {
                    // fend's implicit application grammar can consume a power
                    // following a call. Explicit CLI calls bind before powers.
                    out.push('(');
                    out.push_str(&expression[start..=pos]);
                    let (inner, inner_exact) =
                        expand_calls(&expression[pos + 1..end], deadline, depth + 1)?;
                    out.push_str(&inner);
                    exact &= inner_exact;
                    out.push(')');
                    out.push(')');
                }
                pos = end + 1;
            } else {
                out.push_str(&expression[start..pos]);
            }
        } else {
            let ch = expression[pos..].chars().next().unwrap();
            out.push(ch);
            pos += ch.len_utf8();
        }
        if out.len() > MAX_OUTPUT {
            return Err(CalcError::limit("Expanded expression is too large"));
        }
    }
    Ok((out, exact))
}

pub(crate) fn validate(expression: &str) -> Result<(), CalcError> {
    if expression.trim().is_empty() {
        return Err(CalcError::invalid("An expression is required"));
    }
    if expression.len() > MAX_EXPRESSION {
        return Err(CalcError::limit("Expression exceeds 4096 bytes"));
    }
    let mut stack = Vec::new();
    let mut operators = 0;
    let mut literal_len = 0;
    for ch in expression.chars() {
        if ch.is_control() && !matches!(ch, '\n' | '\t' | '\r') {
            return Err(CalcError::invalid("Control characters are not allowed"));
        }
        if matches!(
            ch,
            '\\' | 'λ' | '@' | ';' | ':' | '"' | '\'' | '`' | '{' | '}'
        ) {
            return Err(CalcError::invalid(
                "Only mathematical expressions are allowed; scripts, strings, and attributes are unsupported",
            ));
        }
        if matches!(ch, '(' | '[') {
            stack.push(ch);
            if stack.len() > 64 {
                return Err(CalcError::limit("Expression nesting exceeds 64 levels"));
            }
        } else if matches!(ch, ')' | ']') && stack.pop() != Some(if ch == ')' { '(' } else { '[' })
        {
            return Err(CalcError::invalid("Mismatched brackets"));
        }
        if matches!(ch, '+' | '-' | '*' | '/' | '^' | '!' | '%' | ',') {
            operators += 1;
        }
        if ch.is_ascii_digit() {
            literal_len += 1;
        } else {
            literal_len = 0;
        }
        if literal_len > 1024 {
            return Err(CalcError::limit("A numeric literal exceeds 1024 digits"));
        }
    }
    if !stack.is_empty() {
        return Err(CalcError::invalid("Unclosed brackets"));
    }
    if operators > 1024 {
        return Err(CalcError::limit("Expression contains too many operations"));
    }
    if expression.contains("!!") || expression.contains("=>") {
        return Err(CalcError::invalid(
            "Double factorial and function definitions are unsupported",
        ));
    }
    if expression
        .split(|c: char| !c.is_ascii_alphanumeric() && c != '_')
        .any(|word| {
            matches!(
                word.to_ascii_lowercase().as_str(),
                "print"
                    | "println"
                    | "help"
                    | "version"
                    | "clear"
                    | "roll"
                    | "random"
                    | "string"
                    | "text"
                    | "codepoint"
            )
        })
    {
        return Err(CalcError::invalid(
            "Only mathematical expressions are supported",
        ));
    }
    Ok(())
}

pub(crate) fn split_args(input: &str) -> Result<Vec<&str>, CalcError> {
    let mut depth = 0_i32;
    let mut start = 0;
    let mut args = Vec::new();
    if input.trim().is_empty() {
        return Ok(args);
    }
    for (i, ch) in input.char_indices() {
        match ch {
            '(' | '[' => depth += 1,
            ')' | ']' => depth -= 1,
            ',' if depth == 0 => {
                args.push(input[start..i].trim());
                start = i + 1;
            }
            _ => {}
        }
        if depth < 0 {
            return Err(CalcError::invalid("Mismatched brackets"));
        }
    }
    args.push(input[start..].trim());
    // A final delimiter is unambiguous; interior missing operands remain errors.
    if args.len() > 1 && args.last() == Some(&"") {
        args.pop();
    }
    if depth != 0 || args.iter().any(|a| a.is_empty()) {
        return Err(CalcError::invalid(
            "Missing argument or mismatched brackets",
        ));
    }
    if args.len() > MAX_ITEMS {
        return Err(CalcError::limit("Too many arguments"));
    }
    Ok(args)
}

fn matching_end(input: &str, start: usize) -> Result<usize, CalcError> {
    let mut depth = 0;
    for (offset, ch) in input[start..].char_indices() {
        if ch == '(' {
            depth += 1;
        }
        if ch == ')' {
            depth -= 1;
            if depth == 0 {
                return Ok(start + offset);
            }
        }
    }
    Err(CalcError::invalid("Unclosed function call"))
}

fn call(input: &str) -> Result<Option<(String, Vec<&str>)>, CalcError> {
    let Some(open) = input.find('(') else {
        return Ok(None);
    };
    let name = input[..open].trim();
    if name.is_empty() || !name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
        return Ok(None);
    }
    let close = matching_end(input, open)?;
    if !input[close + 1..].trim().is_empty() {
        return Ok(None);
    }
    Ok(Some((
        name.to_ascii_lowercase(),
        split_args(&input[open + 1..close])?,
    )))
}

fn context() -> fend_core::Context {
    let mut context = fend_core::Context::new();
    context.disable_rng();
    context
}

fn temperature_unit(word: &str) -> Option<&'static str> {
    if word.eq_ignore_ascii_case("celsius") {
        Some("°C")
    } else if word.eq_ignore_ascii_case("fahrenheit") {
        Some("°F")
    } else if word.eq_ignore_ascii_case("kelvin") || word.eq_ignore_ascii_case("kelvins") {
        Some("K")
    } else {
        None
    }
}

fn normalize_temperature_words(input: &str) -> String {
    fn word_char(ch: char) -> bool {
        ch.is_alphanumeric() || ch == '_'
    }
    fn word_end(input: &str, start: usize) -> usize {
        input[start..]
            .char_indices()
            .find_map(|(offset, ch)| (!word_char(ch)).then_some(start + offset))
            .unwrap_or(input.len())
    }

    let mut out = String::with_capacity(input.len());
    let mut position = 0;
    while position < input.len() {
        let ch = input[position..].chars().next().unwrap();
        if !word_char(ch) {
            out.push(ch);
            position += ch.len_utf8();
            continue;
        }
        let end = word_end(input, position);
        let word = &input[position..end];
        if word.eq_ignore_ascii_case("degree") || word.eq_ignore_ascii_case("degrees") {
            let next = input[end..].trim_start();
            let start = input.len() - next.len();
            if start > end {
                let next_end = word_end(input, start);
                if let Some(unit) = temperature_unit(&input[start..next_end]) {
                    // An affine temperature unit must not become angle degrees
                    // multiplied by a temperature scale in fend's unit grammar.
                    out.push_str(unit);
                    position = next_end;
                    continue;
                }
            }
        }
        out.push_str(temperature_unit(word).unwrap_or(word));
        position = end;
    }
    out
}

fn raw_fend(expression: &str, deadline: Instant) -> Result<fend_core::FendResult, CalcError> {
    check_time(deadline)?;
    let expression = normalize_temperature_words(expression);
    let result =
        fend_core::evaluate_with_interrupt(&expression, &mut context(), &Deadline(deadline))
            .map_err(|message| {
                if Instant::now() >= deadline {
                    CalcError::limit("Calculation exceeded its time budget")
                } else {
                    CalcError::math(message)
                }
            })?;
    check_time(deadline)?;
    let display = result.get_main_result();
    if display.len() > MAX_OUTPUT {
        return Err(CalcError::limit("Result exceeds the output size limit"));
    }
    if display.is_empty() || display.chars().any(|c| c.is_control() && c != '\n') {
        return Err(CalcError::invalid(
            "Expression did not produce a mathematical result",
        ));
    }
    Ok(result)
}

/// Convert fend's exact fraction display (including negative mixed numbers) to
/// a canonical improper rational without passing through fixed-width integers.
fn exact_display_rational(input: &str) -> Option<String> {
    fn integer(text: &str) -> Option<BigInt> {
        let unsigned = text.strip_prefix(['-', '+']).unwrap_or(text);
        if unsigned.is_empty() || !unsigned.bytes().all(|b| b.is_ascii_digit()) {
            return None;
        }
        text.parse().ok()
    }

    let terms: Vec<&str> = input.split_whitespace().collect();
    let (whole, fraction) = match terms.as_slice() {
        [fraction] => (None, *fraction),
        [whole, fraction] => (Some(*whole), *fraction),
        _ => return None,
    };
    let (numerator, denominator) = fraction.split_once('/')?;
    let mut numerator = integer(numerator)?;
    let denominator = integer(denominator)?;
    if denominator.sign() == Sign::NoSign {
        return None;
    }
    if let Some(whole) = whole {
        if numerator.sign() == Sign::Minus || denominator.sign() == Sign::Minus {
            return None;
        }
        let signed_whole = integer(whole)?;
        numerator = if whole.starts_with('-') {
            signed_whole * &denominator - numerator
        } else {
            signed_whole * &denominator + numerator
        };
    }
    let rational = BigRational::new(numerator, denominator);
    Some(format!("{}/{}", rational.numer(), rational.denom()))
}

fn evaluate_fend(expression: &str, deadline: Instant) -> Result<Calculation, CalcError> {
    // A user-selected base, precision, or unit is retained; otherwise prefer exact fractions.
    let has_format = expression
        .split_whitespace()
        .any(|w| matches!(w, "to" | "in" | "as"));
    let formatted = if has_format {
        expression.to_owned()
    } else {
        format!("({expression}) to exact")
    };
    let mut result = raw_fend(&formatted, deadline)?;
    if !has_format && result.get_main_result().contains("approx.") {
        result = raw_fend(&format!("({expression}) to 15 sf"), deadline)?;
    }
    let display = result.get_main_result().to_owned();
    let numeric = display.strip_prefix("approx. ").unwrap_or(&display);
    if matches!(numeric, "true" | "false") {
        return Ok(Calculation {
            display: display.clone(),
            value: Value::Bool(numeric == "true"),
            kind: "boolean".into(),
            exact: true,
        });
    }
    if ["nan", "inf", "infinity", "∞", "-∞"].contains(&numeric.to_ascii_lowercase().as_str()) {
        return Err(CalcError::math("Result is not finite"));
    }
    let exact = !display.contains("approx.");
    let rational = exact.then(|| exact_display_rational(numeric)).flatten();
    let value = if let Some(rational) = &rational {
        Value::String(rational.clone())
    } else if let Ok(number) = numeric.parse::<i64>()
        && number.unsigned_abs() <= 9_007_199_254_740_991
    {
        Value::from(number)
    } else if !exact {
        numeric
            .parse::<f64>()
            .ok()
            .filter(|x| x.is_finite())
            .map(Value::from)
            .unwrap_or_else(|| Value::String(numeric.to_owned()))
    } else {
        // JSON consumers commonly parse all numbers as f64. Keep exact integers
        // outside its interoperable range as strings, even when they fit in u64.
        Value::String(numeric.to_owned())
    };
    // fend is also a small language: forbid non-numeric results, e.g. a bare function.
    let has_number = result
        .get_main_result_spans()
        .any(|span| span.kind() == fend_core::SpanKind::Number);
    if !has_number
        && !display.chars().any(|c| c.is_ascii_digit())
        && !matches!(numeric, "i" | "-i" | "pi" | "π" | "tau" | "τ" | "e")
    {
        return Err(CalcError::invalid(
            "Expression did not produce a number or quantity",
        ));
    }
    let kind = if rational.is_some() {
        "rational"
    } else if numeric.parse::<i128>().is_ok()
        || numeric.chars().all(|c| c.is_ascii_digit() || c == '-')
    {
        "integer"
    } else if numeric.parse::<f64>().is_ok() {
        "number"
    } else if numeric.split_whitespace().last().is_some_and(|term| {
        matches!(term, "i" | "-i")
            || term.strip_suffix('i').is_some_and(|coefficient| {
                coefficient.parse::<f64>().is_ok()
                    || coefficient
                        .split_once('/')
                        .is_some_and(|(a, b)| a.parse::<f64>().is_ok() && b.parse::<f64>().is_ok())
            })
    }) {
        "complex"
    } else {
        "quantity"
    };
    Ok(Calculation {
        display,
        value,
        kind: kind.to_owned(),
        exact,
    })
}

pub(crate) fn scalar(input: &str, deadline: Instant, depth: usize) -> Result<f64, CalcError> {
    check_time(deadline)?;
    if depth > 64 {
        return Err(CalcError::limit("Too many nested function calls"));
    }
    if let Ok(value) = input.trim().parse::<f64>() {
        return finite(value);
    }
    if let Some((name, args)) = call(input.trim())?
        && let Some(result) = extensions::evaluate(&name, &args, deadline, depth + 1)
    {
        let result = result?;
        if let Some(value) = result.value.as_f64() {
            return finite(value);
        }
        if let Some(value) = result.value.as_str().and_then(|s| s.parse::<f64>().ok()) {
            return finite(value);
        }
        return Err(CalcError::math(
            "This operation requires dimensionless real numbers",
        ));
    }
    let (expanded, _) = expand_calls(input, deadline, depth + 1)?;
    let result = raw_fend(&format!("({expanded}) to 17 sf"), deadline)?;
    let value = result
        .get_main_result()
        .trim_start_matches("approx. ")
        .parse::<f64>()
        .map_err(|_| CalcError::math("This operation requires dimensionless real numbers"))?;
    finite(value)
}

pub(crate) fn finite(value: f64) -> Result<f64, CalcError> {
    if value.is_finite() {
        Ok(if value == 0.0 { 0.0 } else { value })
    } else {
        Err(CalcError::math(
            "Result is not finite (invalid domain or numeric overflow)",
        ))
    }
}

pub(crate) fn number(value: f64) -> Result<Calculation, CalcError> {
    let value = finite(value)?;
    Ok(Calculation {
        display: value.to_string(),
        value: Value::from(value),
        kind: "number".into(),
        exact: false,
    })
}

pub(crate) fn vector_result(values: Vec<f64>) -> Result<Calculation, CalcError> {
    for value in &values {
        finite(*value)?;
    }
    let display = format!(
        "[{}]",
        values
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join(", ")
    );
    Ok(Calculation {
        display,
        value: serde_json::json!(values),
        kind: "vector".into(),
        exact: false,
    })
}

pub(crate) fn vector(input: &str, deadline: Instant, depth: usize) -> Result<Vec<f64>, CalcError> {
    if let Some(inner) = input
        .trim()
        .strip_prefix('[')
        .and_then(|s| s.strip_suffix(']'))
    {
        let parts = split_args(inner)?;
        if parts.is_empty() {
            return Err(CalcError::math("A nonempty list is required"));
        }
        return parts
            .into_iter()
            .map(|part| scalar(part, deadline, depth + 1))
            .collect();
    }
    let result = evaluate_inner(input, deadline, depth + 1)?;
    result
        .value
        .as_array()
        .ok_or_else(|| CalcError::invalid("Expected a list, for example [1,2,3]"))?
        .iter()
        .map(|v| {
            v.as_f64()
                .ok_or_else(|| CalcError::math("Expected a vector of real numbers"))
        })
        .collect()
}

pub(crate) fn arity(args: &[&str], count: usize, name: &str) -> Result<(), CalcError> {
    if args.len() == count {
        Ok(())
    } else {
        Err(CalcError::invalid(format!(
            "{name} expects {count} arguments, got {}",
            args.len()
        )))
    }
}
