//! Independent checks of model-call grounding and the CLI wire contract.
use omni_calc::{
    error::CalcError,
    model::{Prediction, Provider},
    planner,
    protocol::{Request, ToolCall},
};
use serde_json::{Value, json};
use std::{
    io::Write,
    process::{Command, Stdio},
};

struct Fixed(ToolCall);
impl Provider for Fixed {
    fn predict(&mut self, _: &str, _: Option<&str>) -> Result<Prediction, CalcError> {
        Ok(Prediction {
            calls: vec![self.0.clone()],
            confidence: Some(0.99),
        })
    }
}

fn rejected(input: &str, name: &str, arguments: Value) {
    let mut request = Request::new(input);
    request.retry = 0;
    request.timeout_ms = 500;
    let response = planner::calculate(
        &mut Fixed(ToolCall {
            name: name.into(),
            arguments,
        }),
        &request,
    );
    assert!(
        !response.ok,
        "ungrounded call succeeded: {}",
        serde_json::to_string(&response).unwrap()
    );
}

#[test]
fn calculate_adapter_cannot_invent_an_unrelated_expression() {
    rejected("add 1 and 2", "calculate", json!({"expression":"999"}));
}

#[test]
fn calculate_adapter_cannot_answer_an_out_of_scope_request() {
    rejected(
        "what is the weather",
        "calculate",
        json!({"expression":"1+1"}),
    );
}

#[test]
fn function_call_cannot_rewrite_a_complete_literal_expression() {
    rejected(
        "what is (1+2)*3",
        "function",
        json!({"function":"add","arguments":["1+2","3"]}),
    );
}

#[test]
fn operation_and_noncommutative_operand_order_are_grounded() {
    rejected(
        "what is 20 minus 5",
        "function",
        json!({"function":"add","arguments":["20","5"]}),
    );
}

#[test]
fn division_operand_order_is_grounded() {
    rejected(
        "what is 20 divided by 5",
        "function",
        json!({"function":"divide","arguments":["5","20"]}),
    );
}

#[test]
fn quantity_units_and_constants_cannot_be_dropped_or_invented() {
    rejected(
        "add 5 m and 2 cm",
        "function",
        json!({"function":"add","arguments":["5","2"]}),
    );
}

#[test]
fn function_operands_cannot_invent_constants() {
    rejected(
        "add 2 and 3",
        "function",
        json!({"function":"add","arguments":["2*i","3*i"]}),
    );
}

#[test]
fn conversion_preserves_the_requested_destination() {
    rejected(
        "convert 5 m to cm",
        "convert",
        json!({"value":"5 m","unit":"m"}),
    );
}

#[test]
fn algebra_operation_and_variable_are_grounded() {
    rejected(
        "differentiate x^2 with respect to x",
        "algebra",
        json!({"operation":"solve","expression":"x^2","variable":"x"}),
    );
}

#[test]
fn algebra_variable_must_be_a_variable_token() {
    rejected(
        "differentiate sin(x) with respect to x",
        "algebra",
        json!({"operation":"diff","expression":"sin(x)","variable":"s"}),
    );
}

#[test]
fn whitespace_normalization_cannot_merge_number_or_unit_tokens() {
    rejected("what is 1 2", "calculate", json!({"expression":"12"}));
    rejected("what is 1 m s", "calculate", json!({"expression":"1 ms"}));
}

#[test]
fn inverse_trigonometric_and_percentage_intents_do_not_collapse() {
    rejected(
        "factorial of 5",
        "algebra",
        json!({"operation":"factor","expression":"5"}),
    );
    rejected(
        "inverse sine of 0.5",
        "function",
        json!({"function":"sin","arguments":["0.5"]}),
    );
    rejected(
        "inverse cosine of 0.5",
        "function",
        json!({"function":"cos","arguments":["0.5"]}),
    );
    rejected(
        "percent change from 5 to 10",
        "function",
        json!({"function":"percent","arguments":["5","10"]}),
    );
    rejected(
        "do not add 2 and 3",
        "function",
        json!({"function":"add","arguments":["2","3"]}),
    );
    rejected("don't calculate 2+3", "evaluate", json!({}));
}

#[test]
fn nested_operation_grouping_is_preserved() {
    rejected(
        "factorial of 5",
        "function",
        json!({"function":"factorial","arguments":["5!"]}),
    );
    rejected(
        "inverse of [[1,2],[3,4]]",
        "function",
        json!({"function":"inverse","arguments":["inverse([[1,2],[3,4]])"]}),
    );
    rejected(
        "multiply the difference of 5 and 2 by 3",
        "function",
        json!({"function":"multiply","arguments":["5","(2-3)"]}),
    );
    rejected(
        "multiply the difference of 5 and 2 by 3",
        "function",
        json!({"function":"multiply","arguments":["(2-5)","3"]}),
    );
    rejected(
        "divide 12 by the sum of 2 and 4",
        "function",
        json!({"function":"divide","arguments":["(12+2)","4"]}),
    );
    rejected(
        "square root of the sum of 4 and 9",
        "function",
        json!({"function":"add","arguments":["sqrt(4)","sqrt(9)"]}),
    );
    rejected(
        "sum of square roots of 4 and 9",
        "function",
        json!({"function":"sqrt","arguments":["(4+9)"]}),
    );
}

#[test]
fn wrapper_variants_and_verbal_operator_boundaries_are_recognized() {
    assert_eq!(
        planner::literal_expression("could you evaluate (2+3)*4?").unwrap(),
        "(2+3)*4"
    );
    assert_eq!(planner::literal_expression("wht is 1+1?").unwrap(), "1+1");
    for input in ["7 modulo 3", "10 choose 3", "17 to the exponent 4"] {
        assert!(planner::literal_expression(input).is_err(), "{input}");
    }
}

#[test]
fn spaced_algebra_verbs_do_not_capture_only_the_first_parentheses() {
    for expression in ["expand (x^2+2*x+1)", "integrate (sin(x),x)"] {
        assert_eq!(planner::literal_expression(expression).unwrap(), expression);
    }
    accepted_call(
        "expand (y+23)*(y-9)",
        "algebra",
        json!({"operation":"expand","expression":"(y+23)*(y-9)"}),
    );
    accepted_call(
        "add (96+26) and 5",
        "function",
        json!({"function":"add","arguments":["(96+26)","5"]}),
    );
    rejected_call(
        "expand (y+23)*(y-9)",
        "algebra",
        json!({"operation":"expand","expression":"(y+23)"}),
    );
    accepted_call(
        "round (173e+8+-62) to 6 decimals",
        "function",
        json!({"function":"round","arguments":["(173e+8+-62)","6"]}),
    );
    accepted_call(
        "exponentiate (27443e-9+-83) to 4",
        "function",
        json!({"function":"power","arguments":["(27443e-9+-83)","4"]}),
    );
}

#[test]
fn parenthesized_operands_with_by_are_verbal_binary_requests() {
    for (input, function, operands) in [
        (
            "what's multiply (-276+1) by 8.158 please",
            "multiply",
            ["(-276+1)", "8.158"],
        ),
        (
            "please give me decrease (13+73) by 864.46",
            "subtract",
            ["(13+73)", "864.46"],
        ),
        (
            "please calculate decrease (94.326+640) by 14",
            "subtract",
            ["(94.326+640)", "14"],
        ),
    ] {
        assert!(planner::literal_expression(input).is_err(), "{input}");
        accepted_call(
            input,
            "function",
            json!({"function":function,"arguments":operands}),
        );
        rejected_call(input, "evaluate", json!({}));
        rejected_call(
            input,
            "function",
            json!({"function":"divide","arguments":operands}),
        );
        rejected_call(
            input,
            "function",
            json!({"function":function,"arguments":[operands[0]]}),
        );
    }
    rejected_call(
        "please give me decrease (13+73) by 864.46",
        "function",
        json!({"function":"subtract","arguments":["864.46","(13+73)"]}),
    );
    rejected_call(
        "what's multiply (-276+1) by 8.158 please",
        "function",
        json!({"function":"multiply","arguments":["(-276-1)","8.158"]}),
    );
}

#[test]
fn spaced_numeric_calls_preserve_complete_mathematical_syntax() {
    for (expression, expected) in [("round (1.3)", 1), ("round (1.3)+2", 3)] {
        assert_eq!(planner::literal_expression(expression).unwrap(), expression);
        let response = planner::calculate(
            &mut Fixed(ToolCall {
                name: "evaluate".into(),
                arguments: json!({}),
            }),
            &Request::new(expression),
        );
        assert!(response.ok, "{response:?}");
        assert_eq!(
            response.result.unwrap().value.as_f64(),
            Some(f64::from(expected))
        );
    }
    rejected_call(
        "round (1.3)+2",
        "function",
        json!({"function":"round","arguments":["1.3+2"]}),
    );
    for expression in ["round (1.3)+2+", "round (1.3", "sin (2"] {
        assert_eq!(planner::literal_expression(expression).unwrap(), expression);
        rejected(expression, "evaluate", json!({}));
        rejected_call(
            expression,
            "function",
            json!({"function":"round","arguments":["1.3"]}),
        );
    }
}

#[test]
fn divide_into_reverses_complete_operand_groups() {
    for (input, operands) in [
        ("divide 4 into 68", ["68", "4"]),
        ("please divide -4 into 52 for me", ["52", "-4"]),
        ("divide 7 into 0", ["0", "7"]),
        ("4 divided into 68", ["68", "4"]),
        ("dividing 4 into 68", ["68", "4"]),
        ("divide (2+4) into (60+8)", ["(60+8)", "(2+4)"]),
        ("divide 68 by 4", ["68", "4"]),
        ("68 divided by 4", ["68", "4"]),
        ("68 over 4", ["68", "4"]),
    ] {
        accepted_call(
            input,
            "function",
            json!({"function":"divide","arguments":operands}),
        );
        rejected_call(
            input,
            "function",
            json!({"function":"divide","arguments":[operands[1],operands[0]]}),
        );
    }
    rejected_call(
        "divide (2+4) into (60+8)",
        "function",
        json!({"function":"divide","arguments":["(4+60+8)","2"]}),
    );
    rejected_call(
        "multiply (4 divided into 68) by 2",
        "function",
        json!({"function":"multiply","arguments":["(4/68)","2"]}),
    );
    for (input, function, operands, wrong) in [
        (
            "first divide 4 into 68; after that, multiply the result by 2",
            "multiply",
            ["(68/4)", "2"],
            ["(4/68)", "2"],
        ),
        (
            "first multiply 4 and 2; after that, divide the result into 68",
            "divide",
            ["68", "(4*2)"],
            ["(4*2)", "68"],
        ),
    ] {
        accepted_call(
            input,
            "function",
            json!({"function":function,"arguments":operands}),
        );
        rejected_call(
            input,
            "function",
            json!({"function":function,"arguments":wrong}),
        );
    }
    accepted_call(
        "what is 9 less than 42",
        "function",
        json!({"function":"subtract","arguments":["42","9"]}),
    );
    rejected_call(
        "what is 9 less than 42",
        "function",
        json!({"function":"subtract","arguments":["9","42"]}),
    );
}

#[test]
fn incidental_arithmetic_does_not_change_an_unrelated_requested_task() {
    for input in [
        "write a poem about 7 plus 8",
        "could you tell me a story about 7 plus 8",
        "translate the phrase 7 plus 8 into French",
        "send an email saying 7 plus 8",
        "search the web for 7 plus 8",
        "create a text file named 7 plus 8",
        "read aloud 7 plus 8",
        "write (7+8) without calculating it",
        "For my English class, write a poem about 7 plus 8",
        "calculate 7 plus 8 and write a poem about it",
        "calculate 7 plus 8\nwrite a poem about it",
        "show me how to calculate 7 plus 8",
        "what's the weather, and what is 7 plus 8",
    ] {
        let mut request = Request::new(input);
        request.retry = 3;
        let response = planner::calculate(
            &mut Fixed(ToolCall {
                name: "function".into(),
                arguments: json!({"function":"add","arguments":["7","8"]}),
            }),
            &request,
        );
        assert!(!response.ok, "{input}: {response:?}");
        assert_eq!(response.error.unwrap().code, "unsupported_input", "{input}");
        assert_eq!(
            response.attempts, 0,
            "out-of-scope tasks must not enter model retries: {input}"
        );
        rejected_call(
            input,
            "function",
            json!({"function":"add","arguments":["7","8"]}),
        );
    }
    for input in [
        "tell me the sum of 7 and 8",
        "could you give me 7 plus 8",
        "I need to calculate 7 plus 8 for my homework",
        "For my homework, calculate 7 plus 8",
        "Don't explain it; calculate 7 plus 8",
        "Don’t explain it; calculate 7 plus 8",
        "Don't explain it\ncalculate 7 plus 8",
    ] {
        accepted_call(
            input,
            "function",
            json!({"function":"add","arguments":["7","8"]}),
        );
    }
    rejected_call(
        "do not add 7 and 8",
        "function",
        json!({"function":"add","arguments":["7","8"]}),
    );
    rejected_call(
        "don't differentiate x^2",
        "algebra",
        json!({"operation":"diff","expression":"x^2","variable":"x"}),
    );
}

#[test]
fn validation_feedback_identifies_the_source_constraint_without_rewriting_calls() {
    let error = |input: &str, function: &str, arguments: Value| {
        planner::expression_for_call(
            &ToolCall {
                name: "function".into(),
                arguments: json!({"function":function,"arguments":arguments}),
            },
            input,
        )
        .unwrap_err()
        .message
    };
    let operation = error("calculate 81 over 9", "multiply", json!(["81", "9"]));
    assert!(
        operation.contains("requests divide") && operation.contains("proposed multiply"),
        "{operation}"
    );

    let order = error("what is 7 less than 50", "subtract", json!(["7", "50"]));
    assert!(
        order.contains("subtract") && order.contains("[\"50\",\"7\"]"),
        "{order}"
    );

    let numbers = error("divide -3 into 57", "divide", json!(["57", "3"]));
    assert!(
        numbers.contains("Required quantities") && numbers.contains("-3") && numbers.contains("57"),
        "{numbers}"
    );
    let duplicates = error("mean of 2, 2, 3", "mean", json!(["[2,3]"]));
    assert!(duplicates.contains("[2, 2, 3]"), "{duplicates}");

    for (input, arguments, numerator, denominator) in [
        ("divide 3 into 57", json!(["3", "57"]), "[57]", "[3]"),
        ("divide 81 by 9", json!(["9", "81"]), "[81]", "[9]"),
    ] {
        let groups = error(input, "divide", arguments);
        assert!(
            groups.contains(&format!("numerator quantities: {numerator}"))
                && groups.contains(&format!("denominator quantities: {denominator}")),
            "{groups}"
        );
    }
    let oversized = error(
        &format!("add {} and 1", "9".repeat(400)),
        "add",
        json!(["2", "1"]),
    );
    assert!(oversized.len() <= 400, "feedback must remain bounded");
}

#[test]
fn subtraction_after_a_unit_or_constant_stays_binary() {
    for (input, operands) in [
        (
            "504 meters minus 2 centimeters",
            vec!["504 meters", "2 centimeters"],
        ),
        ("e minus 1", vec!["e", "1"]),
    ] {
        accepted_call(
            input,
            "function",
            json!({"function":"subtract","arguments":operands}),
        );
        rejected_call(
            input,
            "function",
            json!({"function":"subtract","arguments":[operands[1],operands[0]]}),
        );
    }
}

#[test]
fn explicit_two_step_instructions_preserve_both_operations_and_order() {
    for (input, function, operands, wrong) in [
        (
            "first multiply 20 and 582; after that, divide the result by 69339.815",
            "divide",
            vec!["(20*582)", "69339.815"],
            vec!["69339.815", "(20*582)"],
        ),
        (
            "first subtract 9 from 2; after that, subtract 3 from the result",
            "subtract",
            vec!["(2-9)", "3"],
            vec!["(9-2)", "3"],
        ),
        (
            "first subtract 9 from 2; after that, subtract the result from 3",
            "subtract",
            vec!["3", "(2-9)"],
            vec!["(2-9)", "3"],
        ),
    ] {
        accepted_call(
            input,
            "function",
            json!({"function":function,"arguments":operands}),
        );
        rejected_call(
            input,
            "function",
            json!({"function":function,"arguments":wrong}),
        );
    }
    rejected_call(
        "first multiply 20 and 582; after that, divide the result by 69339.815",
        "function",
        json!({"function":"divide","arguments":["(20+582)","69339.815"]}),
    );
}

#[test]
fn two_step_literal_operands_cannot_change_grouping_or_function_boundaries() {
    for prior in ["sin(2)(3)", "sin(2+3)"] {
        rejected_call(
            "first calculate sin(2*3); after that, add the result and 4",
            "function",
            json!({"function":"add","arguments":[prior,"4"]}),
        );
    }
    rejected_call(
        "first calculate sin((2*3)+4); after that, add the result and 5",
        "function",
        json!({"function":"add","arguments":["sin(2+(3*4))","5"]}),
    );
}

#[test]
fn squared_and_cubed_supply_the_exponent_without_inventing_a_source_position() {
    for (input, base, exponent) in [
        ("7 squared", "7", "2"),
        ("cube of 7", "7", "3"),
        ("(-3+5) squared", "(-3+5)", "2"),
    ] {
        accepted_call(
            input,
            "function",
            json!({"function":"power","arguments":[base,exponent]}),
        );
        rejected_call(
            input,
            "function",
            json!({"function":"power","arguments":[exponent,base]}),
        );
    }
}

fn accepted_call(input: &str, name: &str, arguments: Value) {
    let result = planner::expression_for_call(
        &ToolCall {
            name: name.into(),
            arguments,
        },
        input,
    );
    assert!(result.is_ok(), "{input}: {result:?}");
}

fn rejected_call(input: &str, name: &str, arguments: Value) {
    let result = planner::expression_for_call(
        &ToolCall {
            name: name.into(),
            arguments,
        },
        input,
    );
    assert!(result.is_err(), "{input}: ungrounded expression {result:?}");
}

#[test]
fn spaced_function_calls_and_polite_wrappers_preserve_full_literal() {
    for (input, expression) in [
        ("could you evaluate sin (2)?", "sin (2)"),
        ("mean ([1,2]) please", "mean ([1,2])"),
        ("what’s -9 + 2 for me", "-9 + 2"),
        ("please find 1 2 for me", "1 2"),
    ] {
        assert_eq!(planner::literal_expression(input).unwrap(), expression);
    }
    assert!(planner::literal_expression("s in (2)").is_err());
    rejected_call(
        "mean ([1,2])",
        "function",
        json!({"function":"mean", "arguments":["[1,2]"]}),
    );
}

#[test]
fn common_aliases_keep_noncommutative_order() {
    for (input, function, operands) in [
        ("ratio of 9 to 2", "divide", vec!["9", "2"]),
        ("decrease 9 by 2", "subtract", vec!["9", "2"]),
        ("exponentiate 9 to 2", "power", vec!["9", "2"]),
        ("find power of 9 with exponent 2", "power", vec!["9", "2"]),
        ("the second power of 9", "power", vec!["9", "2"]),
        ("what's -9 divided by 2", "divide", vec!["-9", "2"]),
        (
            "what’s negative nine minus two",
            "subtract",
            vec!["-9", "2"],
        ),
    ] {
        accepted_call(
            input,
            "function",
            json!({"function":function, "arguments":operands}),
        );
        rejected_call(
            input,
            "function",
            json!({"function":function, "arguments":[operands[1], operands[0]]}),
        );
    }
    rejected_call(
        "what’s -9 minus 2",
        "function",
        json!({"function":"subtract", "arguments":["9", "2"]}),
    );
    for (input, function, operands, reversed) in [
        (
            "cosine of the ratio of 9 to 2",
            "cos",
            vec!["(9/2)"],
            vec!["(2/9)"],
        ),
        (
            "raise the ratio of 9 to 2 to power 3",
            "power",
            vec!["(9/2)", "3"],
            vec!["(2/9)", "3"],
        ),
    ] {
        accepted_call(
            input,
            "function",
            json!({"function":function,"arguments":operands}),
        );
        rejected_call(
            input,
            "function",
            json!({"function":function,"arguments":reversed}),
        );
    }
}

#[test]
fn extrema_and_distribution_fillers_do_not_remove_real_units() {
    for (input, function) in [
        ("sample smallest value for [3,8]", "min"),
        ("largest value across [3,8]", "max"),
    ] {
        accepted_call(
            input,
            "function",
            json!({"function":function,"arguments":["[3,8]"]}),
        );
        rejected_call(
            input,
            "function",
            json!({"function":if function=="min" {"max"} else {"min"},"arguments":["[3,8]"]}),
        );
    }
    accepted_call(
        "normal pdf at 2 when mean is 3 and sd is 4",
        "function",
        json!({"function":"normal_pdf", "arguments":["2","3","4"]}),
    );
    rejected_call(
        "normal pdf at 2 when mean is 3 and sd is 4",
        "function",
        json!({"function":"normal_pdf", "arguments":["3","2","4"]}),
    );
    rejected_call(
        "minimum of 3 minutes and 8 minutes",
        "function",
        json!({"function":"min", "arguments":["[3,8]"]}),
    );
    rejected_call(
        "add 5 meters and 2 meters",
        "function",
        json!({"function":"add", "arguments":["5","2"]}),
    );
}

#[test]
fn counted_items_and_binomial_annotations_preserve_choose_order() {
    for (input, function) in [
        ("combinations for 3 selections from 8 items", "ncr"),
        ("ordered selections: 3 items from 8", "npr"),
        ("permute 8 elements taking 3", "npr"),
        ("binomial coefficient 8 choose 3 for me", "ncr"),
    ] {
        accepted_call(
            input,
            "function",
            json!({"function":function,"arguments":["8","3"]}),
        );
        rejected_call(
            input,
            "function",
            json!({"function":function,"arguments":["3","8"]}),
        );
    }
}

#[test]
fn as_conversions_keep_destination_after_polite_suffixes() {
    for input in [
        "express 5 kg as pounds",
        "express 5 kg as pounds for me",
        "could you find express 5 kg as pounds please",
    ] {
        accepted_call(input, "convert", json!({"value":"5 kg","unit":"pounds"}));
        rejected_call(input, "convert", json!({"value":"5 kg","unit":"kg"}));
    }
}

#[test]
fn repeated_unary_operands_are_distinct_from_unary_of_a_group() {
    for (input, function, operands) in [
        (
            "the cube root of 27 plus the cube root of 8",
            "add",
            vec!["cbrt(27)", "cbrt(8)"],
        ),
        (
            "the absolute value of 9 minus the absolute value of 2",
            "subtract",
            vec!["abs(9)", "abs(2)"],
        ),
        ("(26+4) minus 7", "subtract", vec!["(26+4)", "7"]),
        (
            "the cube of 8 divided by the cube of 27",
            "divide",
            vec!["8^3", "27^3"],
        ),
        ("cube root of the sum of 27 and 8", "cbrt", vec!["(27+8)"]),
        (
            "absolute value of the difference of 9 and 2",
            "abs",
            vec!["(9-2)"],
        ),
        ("square of the sum of 9 and 2", "power", vec!["(9+2)", "2"]),
    ] {
        accepted_call(
            input,
            "function",
            json!({"function":function,"arguments":operands}),
        );
    }
    rejected_call(
        "cube root of the sum of 27 and 8",
        "function",
        json!({"function":"add","arguments":["cbrt(27)","cbrt(8)"]}),
    );
    rejected_call(
        "absolute value of the difference of 9 and 2",
        "function",
        json!({"function":"subtract","arguments":["abs(9)","abs(2)"]}),
    );
    rejected_call(
        "the cube root of 27 plus the cube root of 8",
        "function",
        json!({"function":"cbrt","arguments":["(27+8)"]}),
    );
    rejected_call(
        "the absolute value of 9 minus the absolute value of 2",
        "function",
        json!({"function":"abs","arguments":["(9-2)"]}),
    );
    rejected_call(
        "the product of the sum of 9 and 2 and 3",
        "function",
        json!({"function":"multiply","arguments":["(9+2)","3"]}),
    );
}

#[test]
fn linear_equation_alias_requires_matrix_and_vector_operands() {
    let input = "solve linear equations with matrix [[1,2],[3,4]] and vector [5,6]";
    accepted_call(
        input,
        "function",
        json!({"function":"solve_linear","arguments":["[[1,2],[3,4]]","[5,6]"]}),
    );
    rejected_call(
        input,
        "function",
        json!({"function":"matmul","arguments":["[[1,2],[3,4]]","[5,6]"]}),
    );
}

#[test]
fn correct_grounded_calls_remain_usable() {
    for (input, args, expected) in [
        (
            "what is 20 divided by 5",
            json!({"function":"divide","arguments":["20","5"]}),
            4,
        ),
        (
            "add 2 and 3",
            json!({"function":"add","arguments":["2","3"]}),
            5,
        ),
    ] {
        let mut request = Request::new(input);
        request.retry = 0;
        let response = planner::calculate(
            &mut Fixed(ToolCall {
                name: "function".into(),
                arguments: args,
            }),
            &request,
        );
        assert!(response.ok, "{}", serde_json::to_string(&response).unwrap());
        assert_eq!(response.result.unwrap().value, json!(expected));
    }
}

#[test]
fn jsonl_recovers_from_bad_records_without_non_json_stdout() {
    let mut child = Command::new(env!("CARGO_BIN_EXE_calc"))
        .args(["--jsonl", "--no-daemon"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    {
        let mut stdin = child.stdin.take().unwrap();
        stdin
            .write_all(b"not json\n{\"id\":2,\"input\":\"\"}\n")
            .unwrap();
        stdin.write_all(&vec![b'x'; 9000]).unwrap();
        stdin
            .write_all(b"\n{\"id\":4,\"input\":\"\",\"retry\":0}\n")
            .unwrap();
    }
    let output = child.wait_with_output().unwrap();
    assert!(output.status.code().is_some_and(|n| n != 0));
    let text = String::from_utf8(output.stdout).unwrap();
    let responses: Vec<Value> = text
        .lines()
        .map(|line| serde_json::from_str(line).expect("one JSON response per line"))
        .collect();
    assert_eq!(responses.len(), 4, "{text}");
    assert!(responses.iter().all(|r| r["ok"] == false));
    assert_eq!(responses[1]["id"], 2);
    assert_eq!(responses[3]["id"], 4);
}
