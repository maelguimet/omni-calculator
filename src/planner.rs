use crate::{
    engine,
    error::CalcError,
    model::Provider,
    protocol::{Request, Response, ToolCall},
};
use serde_json::{Map, Value};
use std::time::{Duration, Instant};

mod grounding;

/// Borrow the request after removing the same conversational framing used by
/// literal evaluation. This does not normalize or rewrite its mathematics.
pub(crate) fn normalized_request(input: &str) -> &str {
    grounding::strip_wrapper(input)
}

/// The model interprets language. Only allowlisted, grounded calls reach the engine.
pub fn calculate(provider: &mut impl Provider, request: &Request) -> Response {
    let start = Instant::now();
    let mut response = Response::error(request, CalcError::model("No model response."));
    if let Err(error) = request
        .validate()
        .and_then(|()| grounding::request_scope(&request.input))
    {
        response.error = Some(error);
        return response;
    }
    let deadline = start + Duration::from_millis(request.timeout_ms);
    let mut feedback = None;
    let mut last_failure = None;
    for attempt in 0..=request.retry {
        if Instant::now() >= deadline {
            response.error = Some(CalcError::new(
                "timeout",
                "Calculation exceeded its time limit.",
            ));
            break;
        }
        response.attempts = attempt + 1;
        response.retries = attempt;
        response.expression = None;
        response.tool_call = None;
        response.confidence = None;
        let before = Instant::now();
        let predicted = provider.predict(&request.input, feedback.as_deref());
        response.model = provider.name().to_owned();
        response.timing.model_ms += before.elapsed().as_secs_f64() * 1000.0;
        let outcome = predicted.and_then(|prediction| {
            response.confidence = prediction.confidence;
            if prediction.calls.is_empty() {
                return Err(CalcError::new("unsupported_input", "I could not identify a supported calculation. Include the numbers, expression, or units to calculate."));
            }
            if prediction.calls.len() != 1 { return Err(CalcError::model("Expected one tool call.")); }
            let call = prediction.calls.into_iter().next().unwrap();
            response.tool_call = Some(call.clone());
            let expression = expression_for_call(&call, &request.input)?;
            response.expression = Some(expression.clone());
            let before = Instant::now();
            let result = engine::evaluate(&expression, deadline);
            response.timing.compute_ms += before.elapsed().as_secs_f64() * 1000.0;
            result
        });
        match outcome {
            Ok(result) => {
                response.ok = true;
                response.result = Some(result);
                response.error = None;
                break;
            }
            Err(mut error) => {
                // Mathematical domain errors cannot be fixed by changing the user's math.
                let terminal = matches!(
                    error.code.as_str(),
                    "math_error" | "resource_limit" | "timeout" | "model_unavailable"
                ) || (response.expression.is_some()
                    && error.code == "unsupported_operation")
                    || (response
                        .tool_call
                        .as_ref()
                        .is_some_and(|call| call.name == "evaluate")
                        && response.expression.is_some()
                        && matches!(
                            error.code.as_str(),
                            "invalid_input" | "unsupported_operation"
                        ));
                let signature = (
                    error.code.clone(),
                    error.message.clone(),
                    response.expression.clone(),
                    response.tool_call.clone(),
                );
                // Repeating the same failed call cannot help; retries are an upper bound.
                let repeated = last_failure.as_ref() == Some(&signature);
                error.retryable = !terminal && !repeated && attempt < request.retry;
                feedback = Some(error.message.chars().take(400).collect::<String>());
                response.error = Some(error);
                if terminal || repeated {
                    break;
                }
                last_failure = Some(signature);
            }
        }
    }
    if let Some(error) = &mut response.error {
        error.retryable = false;
    }
    response.timing.total_ms = start.elapsed().as_secs_f64() * 1000.0;
    response
}

fn object(call: &ToolCall, allowed: &[&str]) -> Result<Map<String, Value>, CalcError> {
    let map = call
        .arguments
        .as_object()
        .ok_or_else(|| CalcError::model("Tool arguments must be an object."))?;
    if map.keys().any(|key| !allowed.contains(&key.as_str())) {
        return Err(CalcError::model("Tool call contains an unknown argument."));
    }
    Ok(map.clone())
}
fn string<'a>(map: &'a Map<String, Value>, name: &str) -> Result<&'a str, CalcError> {
    map.get(name)
        .and_then(Value::as_str)
        .filter(|s| !s.trim().is_empty() && s.len() <= 4096)
        .ok_or_else(|| CalcError::model(format!("Missing or invalid {name}.")))
}

pub fn expression_for_call(call: &ToolCall, input: &str) -> Result<String, CalcError> {
    grounding::request_scope(input)?;
    if grounding::negated_calculation(input) {
        return Err(CalcError::new(
            "unsupported_input",
            "The request explicitly says not to perform a calculation.",
        ));
    }
    if !matches!(call.name.as_str(), "evaluate" | "unsupported")
        && literal_expression(input).is_ok()
    {
        return Err(CalcError::model(
            "The request contains a literal mathematical expression. Use evaluate to preserve its operators and parentheses exactly.",
        ));
    }
    let expression = match call.name.as_str() {
        "evaluate" => {
            object(call, &[])?;
            return literal_expression(input);
        }
        "function" => {
            let map = object(call, &["function", "arguments"])?;
            let name = string(&map, "function")?;
            let args = map
                .get("arguments")
                .and_then(Value::as_array)
                .ok_or_else(|| {
                    CalcError::model("Function arguments must be a list of expression strings.")
                })?;
            if args.len() > 128 {
                return Err(CalcError::limit("Too many function arguments."));
            }
            let args: Vec<&str> = args
                .iter()
                .map(|a| {
                    a.as_str()
                        .filter(|s| !s.trim().is_empty() && s.len() <= 4096)
                        .ok_or_else(|| {
                            CalcError::model("Every operand must be a nonempty expression string.")
                        })
                })
                .collect::<Result<_, _>>()?;
            ground_numbers(input, &args.join(","))?;
            grounding::function(input, name, &args)?;
            match name {
                "add" | "subtract" | "multiply" | "divide" | "power" | "percent" => {
                    if args.len() != 2 {
                        return Err(CalcError::model(
                            "This operation requires exactly two operands.",
                        ));
                    }
                    let op = match name {
                        "add" => "+",
                        "subtract" => "-",
                        "multiply" => "*",
                        "divide" => "/",
                        "power" => "^",
                        _ => "*",
                    };
                    if name == "percent" {
                        format!("({})/100*({})", args[0], args[1])
                    } else {
                        format!("({}){op}({})", args[0], args[1])
                    }
                }
                "factorial" => {
                    if args.len() != 1 {
                        return Err(CalcError::model("Factorial requires one operand."));
                    }
                    format!("({})!", args[0])
                }
                "sqrt" | "cbrt" | "sin" | "cos" | "tan" | "asin" | "acos" | "atan" | "sinh"
                | "cosh" | "tanh" | "exp" | "ln" | "abs" | "det" | "inverse" | "transpose"
                | "eigenvalues" => {
                    if args.len() != 1 {
                        return Err(CalcError::model("This operation requires one operand."));
                    }
                    format!("{name}({})", args[0])
                }
                "sum" | "mean" | "median" | "variance" | "stdev" | "min" | "max" => {
                    if args.is_empty() {
                        return Err(CalcError::model("This operation requires data."));
                    }
                    let data = if args.len() == 1 && args[0].trim().starts_with('[') {
                        args[0].to_owned()
                    } else {
                        format!("[{}]", args.join(","))
                    };
                    format!("{name}({data})")
                }
                "log" | "round" | "mod" | "gcd" | "lcm" | "ncr" | "npr" | "quantile" | "matmul"
                | "solve_linear" | "normal_pdf" | "normal_cdf" | "normal_quantile"
                | "binomial_pmf" | "binomial_cdf" | "poisson_pmf" | "poisson_cdf" => {
                    format!("{name}({})", args.join(","))
                }
                _ => return Err(CalcError::model("Unknown calculator function.")),
            }
        }
        "convert" => {
            let map = object(call, &["value", "unit"])?;
            let value = string(&map, "value")?;
            let unit = string(&map, "unit")?;
            if unit
                .chars()
                .any(|c| !(c.is_alphanumeric() || " _/^*.-°µμ".contains(c)))
            {
                return Err(CalcError::model("The destination must be a unit."));
            }
            ground_numbers(input, &format!("{value},{unit}"))?;
            grounding::conversion(input, value, unit)?;
            format!("({value}) to {unit}")
        }
        "algebra" => {
            let map = object(
                call,
                &[
                    "operation",
                    "expression",
                    "variable",
                    "lower",
                    "upper",
                    "order",
                    "value",
                ],
            )?;
            let operation = string(&map, "operation")?;
            let expr = string(&map, "expression")?;
            grounding::algebra_operation(input, operation)?;
            if !grounding::contains_expression(input, expr) {
                return Err(CalcError::model(
                    "Algebra must preserve the expression exactly as written, including parentheses.",
                ));
            }
            let mut args = vec![expr.to_owned()];
            match operation {
                "simplify" | "expand" | "factor" => {
                    if map
                        .keys()
                        .any(|k| !["operation", "expression"].contains(&k.as_str()))
                    {
                        return Err(CalcError::model(
                            "This algebra operation takes only an expression.",
                        ));
                    }
                }
                "diff" | "integrate" | "solve" | "root" | "substitute" => {
                    let var = string(&map, "variable")?;
                    if var.len() > 32
                        || !var.chars().all(|c| c.is_ascii_alphabetic() || c == '_')
                        || !grounding::variable_is_grounded(input, expr, var)
                    {
                        return Err(CalcError::model(
                            "The variable must occur in the original expression.",
                        ));
                    }
                    args.push(var.to_owned());
                    if operation == "substitute" {
                        if map.contains_key("lower")
                            || map.contains_key("upper")
                            || map.contains_key("order")
                        {
                            return Err(CalcError::model(
                                "Substitution takes a variable and replacement value.",
                            ));
                        }
                        args.push(string(&map, "value")?.to_owned());
                    } else if map.contains_key("value") {
                        return Err(CalcError::model(
                            "A replacement value is only used for substitution.",
                        ));
                    }
                    if map.contains_key("lower") || map.contains_key("upper") {
                        if !["integrate", "root"].contains(&operation) {
                            return Err(CalcError::model(
                                "Bounds are supported for integration and root finding.",
                            ));
                        }
                        args.push(string(&map, "lower")?.to_owned());
                        args.push(string(&map, "upper")?.to_owned());
                    }
                    if let Some(order) = map.get("order") {
                        if operation != "diff" {
                            return Err(CalcError::model(
                                "Order is only valid for differentiation.",
                            ));
                        }
                        let n = order
                            .as_u64()
                            .filter(|&n| (1..=10).contains(&n))
                            .ok_or_else(|| CalcError::model("Derivative order must be 1 to 10."))?;
                        args.push(n.to_string());
                    }
                    if operation == "root" && args.len() != 4 {
                        return Err(CalcError::model(
                            "Root finding requires lower and upper bounds.",
                        ));
                    }
                }
                _ => {
                    return Err(CalcError::model(
                        "Unsupported algebra tool operation. Use an explicit calculator expression.",
                    ));
                }
            }
            let mut grounding_arguments = args.join(",");
            if operation == "root" && grounding::equated_to_zero(input, expr) {
                grounding_arguments.push_str(",0");
            }
            ground_numbers(input, &grounding_arguments)?;
            grounding::algebra_arguments(input, expr, operation, &args)?;
            format!("{operation}({})", args.join(","))
        }
        "unsupported" => {
            object(call, &[])?;
            return Err(CalcError::new(
                "unsupported_input",
                "This request is outside the calculator's scope.",
            ));
        }
        _ => return Err(CalcError::model("The model requested an unknown tool.")),
    };
    if expression.len() > 4096 {
        return Err(CalcError::limit("Tool expression exceeds 4096 bytes."));
    }
    Ok(expression)
}

/// A reference tool cannot change a literal's mathematical structure. It gets the
/// complete original expression, with only conversational wrappers removed.
pub fn literal_expression(input: &str) -> Result<String, CalcError> {
    let expr = normalized_request(input);
    if expr.is_empty() {
        return Err(CalcError::invalid("The calculation is missing."));
    }
    let first = expr.chars().next().unwrap();
    let is_call = expr.find('(').is_some_and(|i| {
        let identifier = expr[..i].trim_end();
        let whole_expression_verb = matches!(
            identifier.to_ascii_lowercase().as_str(),
            "simplify"
                | "expand"
                | "factor"
                | "differentiate"
                | "integrate"
                | "solve"
                | "substitute"
        );
        // A spaced verb such as expand (x+1)*(x-1) is not a complete
        // expand(...) DSL call. Complete spaced calls are still literal.
        let partial_call = identifier.len() != i && {
            let mut depth = 0usize;
            let closing = expr[i..].char_indices().find_map(|(position, ch)| {
                match ch {
                    '(' => depth += 1,
                    ')' => depth = depth.saturating_sub(1),
                    _ => {}
                }
                (ch == ')' && depth == 0).then_some(i + position + 1)
            });
            closing.is_some_and(|end| {
                let tail = expr[end..].trim();
                (whole_expression_verb && !tail.is_empty())
                    || (matches!(
                        identifier.to_ascii_lowercase().as_str(),
                        "round" | "rounding" | "exponentiate"
                    ) && tail
                        .split_whitespace()
                        .next()
                        .is_some_and(|word| word.eq_ignore_ascii_case("to")))
            })
        };
        !identifier.is_empty()
            && !partial_call
            && identifier
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '_')
    });
    let constant = ["pi", "e", "i", "tau"].iter().any(|c| {
        expr.strip_prefix(c)
            .is_some_and(|rest| rest.is_empty() || rest.starts_with(|x: char| !x.is_alphabetic()))
    });
    if !(first.is_ascii_digit() || ".+-([√π".contains(first) || is_call || constant) {
        return Err(CalcError::model(
            "There is no complete literal expression to evaluate. Use function, convert, or algebra to interpret the requested operation.",
        ));
    }
    // Verbal operators cannot be handed to the expression engine as units or names.
    let words = expr.to_ascii_lowercase();
    let verbal_division_into = expr.find('(').is_some_and(|i| {
        matches!(
            expr[..i].trim().to_ascii_lowercase().as_str(),
            "divide" | "dividing"
        ) && words.contains(" into ")
    });
    if verbal_division_into {
        return Err(CalcError::model(
            "The request uses division into a quantity. Select divide and preserve the stated divisor and dividend.",
        ));
    }
    for phrase in [
        " plus ",
        " minus ",
        " times ",
        " divided ",
        " by ",
        " multiplied ",
        " and ",
        " power ",
        " squared",
        " cubed",
        "percent of",
        "root of",
        "sine of",
        "cosine of",
        "tangent of",
        "factorial of",
        " factorial",
        " modulo",
        " mod ",
        " choose",
        " permute",
        " percent",
        "% of",
        " per cent",
        " raised",
        " exponent",
        " less ",
        " over ",
        " tims ",
        " plus",
        " minus",
        " after substitut",
        " then ",
        " followed by ",
    ] {
        if words.contains(phrase) {
            return Err(CalcError::model(
                "The request uses an operation in words. Select its function and extract its operands.",
            ));
        }
    }
    if expr
        .chars()
        .any(|c| matches!(c, '`' | '\\' | '"' | '\'' | '\0'))
    {
        return Err(CalcError::invalid(
            "Unsupported character in mathematical expression.",
        ));
    }
    Ok(expr.to_owned())
}

fn ground_numbers(input: &str, output: &str) -> Result<(), CalcError> {
    grounding::numbers(input, output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::Prediction;
    use serde_json::json;
    struct Scripted {
        calls: std::collections::VecDeque<Result<Prediction, CalcError>>,
        feedback: Vec<Option<String>>,
    }
    impl Provider for Scripted {
        fn name(&self) -> &'static str {
            "scripted-test"
        }
        fn predict(&mut self, _: &str, feedback: Option<&str>) -> Result<Prediction, CalcError> {
            self.feedback.push(feedback.map(str::to_owned));
            self.calls.pop_front().unwrap()
        }
    }
    fn prediction(name: &str, args: Value) -> Result<Prediction, CalcError> {
        Ok(Prediction {
            calls: vec![ToolCall {
                name: name.into(),
                arguments: args,
            }],
            confidence: None,
        })
    }
    #[test]
    fn references_preserve_grouping() {
        assert_eq!(literal_expression("wat iz (3+4)*5?").unwrap(), "(3+4)*5");
        assert!(literal_expression("what is the square root of 144").is_err());
        assert!(literal_expression("what's the weather").is_err());
        assert!(
            expression_for_call(
                &ToolCall {
                    name: "calculate".into(),
                    arguments: json!({"expression":"3+4*5"})
                },
                "what is (3+4)*5?"
            )
            .is_err()
        );
    }
    #[test]
    fn retries_correct_a_failed_model() {
        let mut model = Scripted {
            calls: vec![
                Err(CalcError::model("Malformed JSON")),
                prediction("evaluate", json!({})),
            ]
            .into(),
            feedback: vec![],
        };
        let reply = calculate(&mut model, &Request::new("wat iz 1+1"));
        assert!(reply.ok, "{reply:?}");
        assert_eq!(reply.attempts, 2);
        assert_eq!(reply.model, "scripted-test");
        assert_eq!(reply.result.unwrap().display, "2");
        assert!(
            model.feedback[1]
                .as_ref()
                .unwrap()
                .contains("Malformed JSON")
        );
    }
    #[test]
    fn zero_retries_means_one_attempt() {
        let mut model = Scripted {
            calls: vec![Err(CalcError::model("broken"))].into(),
            feedback: vec![],
        };
        let mut request = Request::new("1+1");
        request.retry = 0;
        assert_eq!(calculate(&mut model, &request).attempts, 1);
    }
    #[test]
    fn distinct_invalid_calls_with_identical_errors_can_retry() {
        let mut model = Scripted {
            calls: vec![
                prediction(
                    "function",
                    json!({"function":"add","arguments":["2","3"],"extra":1}),
                ),
                prediction(
                    "function",
                    json!({"function":"add","arguments":["2","3"],"extra":2}),
                ),
                prediction("function", json!({"function":"add","arguments":["2","3"]})),
            ]
            .into(),
            feedback: vec![],
        };
        let mut request = Request::new("add 2 and 3");
        request.retry = 2;
        let reply = calculate(&mut model, &request);
        assert!(reply.ok, "{reply:?}");
        assert_eq!(reply.attempts, 3);
        assert_eq!(reply.retries, 2);
        assert_eq!(reply.result.unwrap().value, json!(5));
        assert_eq!(
            model.feedback[1], model.feedback[2],
            "the first two failures must have the same error text"
        );
    }
    #[test]
    fn identical_invalid_calls_still_stop_early() {
        let invalid = || {
            prediction(
                "function",
                json!({"function":"add","arguments":["2","3"],"extra":1}),
            )
        };
        let mut model = Scripted {
            calls: vec![
                invalid(),
                invalid(),
                prediction("function", json!({"function":"add","arguments":["2","3"]})),
            ]
            .into(),
            feedback: vec![],
        };
        let reply = calculate(&mut model, &Request::new("add 2 and 3"));
        assert!(!reply.ok);
        assert_eq!(reply.attempts, 2);
        assert_eq!(
            model.calls.len(),
            1,
            "identical failed calls cannot consume all retry slots"
        );
    }
    #[test]
    fn immutable_literal_errors_do_not_retry_into_different_math() {
        let mut model = Scripted {
            calls: vec![prediction("evaluate", json!({}))].into(),
            feedback: vec![],
        };
        let reply = calculate(&mut model, &Request::new("integrate(exp(-x^2),x)"));
        assert!(!reply.ok);
        assert_eq!(reply.attempts, 1);
        assert_eq!(reply.error.unwrap().code, "unsupported_operation");
    }
    #[test]
    fn grounded_engine_capability_errors_do_not_retry_into_different_math() {
        let mut model = Scripted {
            calls: vec![
                prediction(
                    "algebra",
                    json!({"operation":"solve","expression":"cos(x)=0","variable":"x"}),
                ),
                prediction(
                    "algebra",
                    json!({"operation":"solve","expression":"x=0","variable":"x"}),
                ),
            ]
            .into(),
            feedback: vec![],
        };
        let reply = calculate(&mut model, &Request::new("solve cos(x)=0 for x"));
        assert!(!reply.ok);
        assert_eq!(reply.attempts, 1);
        assert_eq!(reply.error.unwrap().code, "unsupported_operation");
        assert_eq!(
            model.calls.len(),
            1,
            "capability failure must not request different mathematics"
        );
    }
    #[test]
    fn provider_failure_cannot_retain_an_earlier_attempts_confidence() {
        let mut model = Scripted {
            calls: vec![
                Ok(Prediction {
                    calls: vec![ToolCall {
                        name: "unknown".into(),
                        arguments: json!({}),
                    }],
                    confidence: Some(0.95),
                }),
                Err(CalcError::model("inference failed")),
            ]
            .into(),
            feedback: vec![],
        };
        let mut request = Request::new("add 2 and 3");
        request.retry = 1;
        let reply = calculate(&mut model, &request);
        assert_eq!(reply.attempts, 2);
        assert!(!reply.ok);
        assert_eq!(reply.confidence, None);
        assert_eq!(reply.tool_call, None);
    }
    #[test]
    fn numbers_cannot_be_invented_dropped_or_rounded() {
        assert!(ground_numbers("mean of 1,2,3", "[1,2]").is_err());
        assert!(ground_numbers("add 9007199254740993 and 1", "9007199254740992+1").is_err());
        assert!(ground_numbers("two plus two", "2+2").is_ok());
        assert!(ground_numbers("add one and two", "1+2").is_ok());
        assert!(ground_numbers("add -1 and 2", "1+2").is_err());
        assert!(ground_numbers("add -1 and 2", "-1+2").is_ok());
        assert!(ground_numbers("1e3 plus 0.50", "1000+0.5").is_ok());
    }
    #[test]
    fn domain_error_is_not_repaired_into_different_math() {
        let mut model = Scripted {
            calls: vec![prediction("evaluate", json!({}))].into(),
            feedback: vec![],
        };
        let reply = calculate(&mut model, &Request::new("1/0"));
        assert!(!reply.ok);
        assert_eq!(reply.attempts, 1);
    }
}
