//! Conservative checks of model-generated arguments against the user's words.
//! This recognizes bounded, common phrasing, not arbitrary natural-language
//! equivalence. The model still chooses every tool and supplies its arguments.

use crate::error::CalcError;
use std::collections::{BTreeMap, BTreeSet};

type Result<T> = std::result::Result<T, CalcError>;
fn reject(message: &str) -> CalcError {
    CalcError::model(message)
}

pub(super) fn negated_calculation(input: &str) -> bool {
    let lower = input.to_ascii_lowercase().replace('’', "'");
    ["do not ", "don't ", "dont ", "never "]
        .iter()
        .flat_map(|prefix| {
            lower
                .match_indices(prefix)
                .map(move |(i, _)| i + prefix.len())
        })
        .any(|start| {
            let tail = lower[start..].trim_start();
            let operation = tail.split_whitespace().next().unwrap_or("");
            matches!(
                operation,
                "calculate"
                    | "calculating"
                    | "compute"
                    | "evaluate"
                    | "calc"
                    | "differentiate"
                    | "integrate"
                    | "solve"
                    | "simplify"
                    | "expand"
                    | "factor"
                    | "substitute"
                    | "convert"
            ) || cues(&lex(tail))
                .first()
                .is_some_and(|(position, _)| *position == 0)
        })
}

/// Mathematical text can be the subject of another task. Check the requested
/// action before the retry loop, so a model cannot turn writing or translation
/// into an unsolicited calculation. This only recognizes clear task framing.
pub(super) fn request_scope(input: &str) -> Result<()> {
    let lower = input.to_ascii_lowercase();
    let mut clauses = Vec::new();
    let mut begin = 0;
    let mut depth = 0usize;
    for (position, ch) in input.char_indices() {
        if position < begin {
            continue;
        }
        match ch {
            '(' | '[' => depth += 1,
            ')' | ']' => depth = depth.saturating_sub(1),
            _ => {}
        }
        if depth != 0 {
            continue;
        }
        let separator = if matches!(ch, ';' | ',' | '\n') {
            Some(1)
        } else {
            [" and ", " then ", " but "]
                .iter()
                .find(|separator| lower[position..].starts_with(**separator))
                .map(|separator| separator.len())
        };
        if let Some(length) = separator {
            clauses.push(&input[begin..position]);
            begin = position + length;
        }
    }
    clauses.push(&input[begin..]);
    for clause in clauses {
        let tokens = lex(strip_wrapper(clause));
        let words: Vec<_> = tokens
            .iter()
            .map(|token| token.text.to_ascii_lowercase())
            .collect();
        let mut beginning = 0;
        while words.get(beginning).is_some_and(|word| {
            matches!(
                word.as_str(),
                "please"
                    | "can"
                    | "could"
                    | "would"
                    | "will"
                    | "you"
                    | "i"
                    | "want"
                    | "need"
                    | "like"
                    | "to"
                    | "me"
                    | "a"
                    | "an"
                    | "the"
                    | "tell"
                    | "show"
                    | "give"
                    | "and"
                    | "then"
                    | "but"
            )
        }) {
            beginning += 1;
        }
        let Some(action) = words.get(beginning) else {
            continue;
        };
        let unrelated = matches!(
            action.as_str(),
            "write"
                | "compose"
                | "draft"
                | "translate"
                | "summarize"
                | "summarise"
                | "explain"
                | "describe"
                | "narrate"
                | "sing"
                | "recite"
                | "draw"
                | "paint"
                | "plot"
                | "recommend"
                | "search"
                | "google"
                | "browse"
                | "fetch"
                | "send"
                | "email"
                | "create"
                | "generate"
                | "make"
                | "read"
                | "play"
                | "open"
                | "delete"
                | "poem"
                | "story"
                | "joke"
                | "song"
                | "lyrics"
                | "essay"
                | "translation"
                | "weather"
                | "forecast"
                | "news"
                | "explanation"
                | "definition"
                | "meaning"
                | "about"
        );
        let asking_how = action == "how"
            && words.get(beginning + 1).is_some_and(|word| {
                matches!(
                    word.as_str(),
                    "to" | "do" | "does" | "should" | "would" | "can"
                )
            });
        if unrelated || asking_how {
            return Err(CalcError::new(
                "unsupported_input",
                "This request asks for a task outside the calculator's scope. Ask explicitly for a calculation.",
            ));
        }
    }
    Ok(())
}

pub(super) fn strip_wrapper(input: &str) -> &str {
    let mut text = input.trim().trim_end_matches('?').trim();
    for _ in 0..3 {
        let lower = text.to_ascii_lowercase();
        if let Some(suffix) = [" please", " for me"]
            .iter()
            .find(|suffix| lower.ends_with(**suffix))
        {
            text = text[..text.len() - suffix.len()].trim();
        } else {
            break;
        }
    }
    for _ in 0..3 {
        let lower = text.to_ascii_lowercase();
        let prefixes = [
            "please ",
            "could you calculate ",
            "could you evaluate ",
            "could you compute ",
            "could you find ",
            "can you calculate ",
            "can you evaluate ",
            "can you compute ",
            "can you find ",
            "i want to calculate ",
            "i need ",
            "work out ",
            "show me ",
            "give me ",
            "tell me ",
            "what is ",
            "what's ",
            "what’s ",
            "whats ",
            "wht is ",
            "whts ",
            "wat iz ",
            "wat is ",
            "wut is ",
            "wot is ",
            "what iz ",
            "how much is ",
            "calculate ",
            "compute ",
            "evaluate ",
            "calc ",
            "find ",
        ];
        if let Some(prefix) = prefixes.iter().find(|p| lower.starts_with(**p)) {
            text = text[prefix.len()..].trim();
        } else {
            break;
        }
    }
    text
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Token {
    text: String,
    kind: u8,
}
const NUMBER: u8 = 0;
const WORD: u8 = 1;
const SYMBOL: u8 = 2;

fn lex(text: &str) -> Vec<Token> {
    let chars: Vec<char> = text.chars().collect();
    let mut i = 0;
    let mut tokens = Vec::new();
    while i < chars.len() {
        let c = chars[i];
        if c.is_whitespace() {
            i += 1;
            continue;
        }
        let start = i;
        if c.is_ascii_digit() || (c == '.' && chars.get(i + 1).is_some_and(char::is_ascii_digit)) {
            i += 1;
            while i < chars.len() && (chars[i].is_ascii_digit() || chars[i] == '.') {
                i += 1;
            }
            if chars.get(i).is_some_and(|c| matches!(c, 'e' | 'E')) {
                let mut end = i + 1;
                if chars.get(end).is_some_and(|c| matches!(c, '+' | '-')) {
                    end += 1;
                }
                let digits = end;
                while chars.get(end).is_some_and(char::is_ascii_digit) {
                    end += 1;
                }
                if end > digits {
                    i = end;
                }
            }
            tokens.push(Token {
                text: chars[start..i].iter().collect(),
                kind: NUMBER,
            });
        } else if c.is_alphabetic() || c == '_' || c == '°' {
            i += 1;
            while i < chars.len()
                && (chars[i].is_alphanumeric() || chars[i] == '_' || chars[i] == '°')
            {
                i += 1;
            }
            tokens.push(Token {
                text: chars[start..i].iter().collect(),
                kind: WORD,
            });
        } else {
            i += 1;
            let text = match c {
                '−' => "-".into(),
                '×' | '·' => "*".into(),
                '÷' => "/".into(),
                '*' if chars.get(i) == Some(&'*') => {
                    i += 1;
                    "^".into()
                }
                _ => c.to_string(),
            };
            tokens.push(Token { text, kind: SYMBOL });
        }
    }
    tokens
}

fn small(word: &str) -> Option<u64> {
    Some(match word {
        "zero" => 0,
        "one" => 1,
        "two" => 2,
        "three" => 3,
        "four" => 4,
        "five" => 5,
        "six" => 6,
        "seven" => 7,
        "eight" => 8,
        "nine" => 9,
        "ten" => 10,
        "eleven" => 11,
        "twelve" => 12,
        "thirteen" => 13,
        "fourteen" => 14,
        "fifteen" => 15,
        "sixteen" => 16,
        "seventeen" => 17,
        "eighteen" => 18,
        "nineteen" => 19,
        "twenty" => 20,
        "thirty" => 30,
        "forty" => 40,
        "fifty" => 50,
        "sixty" => 60,
        "seventy" => 70,
        "eighty" => 80,
        "ninety" => 90,
        _ => return None,
    })
}

fn number_phrase(tokens: &[Token], start: usize) -> Option<(usize, u64)> {
    let first = small(&tokens.get(start)?.text.to_ascii_lowercase())?;
    let mut i = start + 1;
    let mut group = first;
    let mut total = 0u64;
    let mut previous = first;
    let mut remainder = false;
    let mut fresh = false;
    let mut last_scale = u64::MAX;
    while i < tokens.len() && i - start < 24 {
        let word = tokens[i].text.to_ascii_lowercase();
        if word == "hundred" && (1..10).contains(&group) {
            group *= 100;
            remainder = true;
            fresh = false;
            previous = 100;
            i += 1;
            continue;
        }
        if matches!(word.as_str(), "thousand" | "million" | "billion") {
            let scale = match word.as_str() {
                "thousand" => 1000,
                "million" => 1_000_000,
                _ => 1_000_000_000,
            };
            if scale >= last_scale {
                break;
            }
            total = total.checked_add(group.checked_mul(scale)?)?;
            group = 0;
            last_scale = scale;
            fresh = true;
            remainder = false;
            i += 1;
            continue;
        }
        if word == "and"
            && remainder
            && tokens
                .get(i + 1)
                .and_then(|t| small(&t.text.to_ascii_lowercase()))
                .is_some()
        {
            i += 1;
            continue;
        }
        if word == "-"
            && previous >= 20
            && previous % 10 == 0
            && tokens
                .get(i + 1)
                .and_then(|t| small(&t.text.to_ascii_lowercase()))
                .is_some_and(|n| n < 10)
        {
            i += 1;
            continue;
        }
        let Some(n) = small(&word) else {
            break;
        };
        if fresh || remainder || ((20..100).contains(&previous) && previous % 10 == 0 && n < 10) {
            group = group.checked_add(n)?;
            previous = n;
            fresh = false;
            remainder = false;
            i += 1;
        } else {
            break;
        }
    }
    Some((i, total.checked_add(group)?))
}

fn source_tokens(input: &str) -> Vec<Token> {
    let original = lex(strip_wrapper(input));
    let mut out = Vec::new();
    let mut i = 0;
    while i < original.len() {
        let lower = original[i].text.to_ascii_lowercase();
        if lower == "zero"
            && original
                .get(i + 1)
                .is_some_and(|t| t.text.eq_ignore_ascii_case("of"))
        {
            out.push(original[i].clone());
            i += 1;
            continue;
        }
        if original
            .get(i + 1)
            .is_some_and(|t| matches!(t.text.to_ascii_lowercase().as_str(), "power" | "derivative"))
        {
            let ordinal = match lower.as_str() {
                "first" => Some(1),
                "second" => Some(2),
                "third" => Some(3),
                "fifth" => Some(5),
                "eighth" => Some(8),
                "ninth" => Some(9),
                "twelfth" => Some(12),
                _ => lower.strip_suffix("th").and_then(small),
            };
            if let Some(n) = ordinal {
                out.push(Token {
                    text: n.to_string(),
                    kind: NUMBER,
                });
                out.push(Token {
                    text: "th".into(),
                    kind: WORD,
                });
                i += 1;
                continue;
            }
        }
        if matches!(lower.as_str(), "minus" | "negative")
            && original
                .get(i + 1)
                .is_some_and(|t| t.kind == NUMBER || small(&t.text.to_ascii_lowercase()).is_some())
        {
            // A written minus is binary only immediately after a quantity.
            // Retaining it as an operator lets signed-number extraction make
            // the same distinction as for a literal '-'.
            out.push(Token {
                text: "-".into(),
                kind: SYMBOL,
            });
            i += 1;
            continue;
        }
        if original[i].kind == WORD
            && let Some((end, n)) = number_phrase(&original, i)
        {
            out.push(Token {
                text: n.to_string(),
                kind: NUMBER,
            });
            i = end;
            continue;
        }
        out.push(original[i].clone());
        i += 1;
    }
    out
}

fn normalize_number(s: &str) -> String {
    let (mantissa, exponent) = s.split_once(['e', 'E']).unwrap_or((s, "0"));
    let Ok(exponent) = exponent.parse::<i64>() else {
        return format!("literal:{s}");
    };
    let places = mantissa.split_once('.').map_or(0, |(_, f)| f.len() as i64);
    let mut digits = mantissa.replace('.', "").trim_start_matches('0').to_owned();
    if digits.is_empty() {
        return "0".into();
    }
    let Some(mut scale) = exponent.checked_sub(places) else {
        return format!("literal:{s}");
    };
    while digits.ends_with('0') {
        digits.pop();
        let Some(next) = scale.checked_add(1) else {
            return format!("literal:{s}");
        };
        scale = next;
    }
    format!("{digits}e{scale}")
}

fn quantities(tokens: &[Token]) -> BTreeMap<String, usize> {
    let mut out = BTreeMap::new();
    for (i, token) in tokens.iter().enumerate().filter(|(_, t)| t.kind == NUMBER) {
        let mut key = normalize_number(&token.text);
        if i > 0 && tokens[i - 1].text == "-" {
            let unary = !quantity_before(tokens, i - 1);
            if unary && key != "0" {
                key.insert(0, '-');
            }
        }
        *out.entry(key).or_insert(0) += 1;
    }
    out
}

fn quantity_before(tokens: &[Token], position: usize) -> bool {
    position > 0
        && (tokens[position - 1].kind == NUMBER
            || matches!(tokens[position - 1].text.as_str(), ")" | "]")
            || (tokens[position - 1].kind == WORD
                && !language_word(&tokens[position - 1].text.to_ascii_lowercase())
                && (tokens[position - 1].text.len() == 1
                    || (position >= 2 && tokens[position - 2].kind == NUMBER))))
}

fn source_quantities(tokens: &[Token]) -> BTreeMap<String, usize> {
    let mut source = quantities(tokens);
    for (i, t) in tokens.iter().enumerate() {
        let word = t.text.to_ascii_lowercase();
        let implied = match word.as_str() {
            "squared" | "twice" => Some(2),
            "cubed" => Some(3),
            "square"
                if tokens
                    .get(i + 1)
                    .is_some_and(|t| t.text.eq_ignore_ascii_case("of")) =>
            {
                Some(2)
            }
            "cube"
                if tokens
                    .get(i + 1)
                    .is_some_and(|t| t.text.eq_ignore_ascii_case("of")) =>
            {
                Some(3)
            }
            "first"
                if tokens
                    .get(i + 1)
                    .is_some_and(|t| t.text.eq_ignore_ascii_case("derivative")) =>
            {
                Some(1)
            }
            "second"
                if tokens
                    .get(i + 1)
                    .is_some_and(|t| t.text.eq_ignore_ascii_case("derivative")) =>
            {
                Some(2)
            }
            "third"
                if tokens
                    .get(i + 1)
                    .is_some_and(|t| t.text.eq_ignore_ascii_case("derivative")) =>
            {
                Some(3)
            }
            _ => None,
        };
        if let Some(n) = implied {
            *source.entry(normalize_number(&n.to_string())).or_insert(0) += 1;
        }
    }
    source
}

pub(super) fn numbers(input: &str, output: &str) -> Result<()> {
    let expected = source_quantities(&source_tokens(input));
    let proposed = quantities(&lex(output));
    if expected != proposed {
        let mut message = "The tool changed or omitted a signed quantity. Copy every value exactly, without calculating it.".to_owned();
        if let Some(values) = quantity_list(&expected) {
            message.push_str(&format!(
                " Required quantities, preserving counts: {values}."
            ));
        }
        if let Some(values) = quantity_list(&proposed)
            && message.len() + values.len() < 330
        {
            message.push_str(&format!(" Proposed quantities: {values}."));
        }
        return Err(reject(&message));
    }
    Ok(())
}

fn quantity_list(values: &BTreeMap<String, usize>) -> Option<String> {
    if values.values().sum::<usize>() > 12 {
        return None;
    }
    let mut display = Vec::new();
    for (quantity, count) in values {
        let quantity = quantity.strip_suffix("e0").unwrap_or(quantity);
        display.extend(std::iter::repeat_n(quantity, *count));
    }
    let list = format!("[{}]", display.join(", "));
    (list.len() <= 140).then_some(list)
}

fn required_group_quantities(first: &[Token], second: &[Token]) -> Option<String> {
    let first = quantity_list(&source_quantities(first))?;
    let second = quantity_list(&source_quantities(second))?;
    let message = format!(
        " Required numerator quantities: {first}; denominator quantities: {second}. Preserve each complete group and its units."
    );
    (message.len() <= 250).then_some(message)
}

fn normalized(tokens: &[Token]) -> Vec<String> {
    tokens
        .iter()
        .map(|t| {
            if t.kind == NUMBER {
                normalize_number(&t.text)
            } else if t.text == "π" {
                "pi".into()
            } else {
                t.text.clone()
            }
        })
        .collect()
}
fn span(haystack: &[Token], needle: &[Token]) -> Option<(usize, usize)> {
    if needle.is_empty() {
        return None;
    }
    let h = normalized(haystack);
    let n = normalized(needle);
    h.windows(n.len())
        .position(|part| part == n)
        .map(|i| (i, i + n.len()))
}
pub(super) fn contains_expression(input: &str, expr: &str) -> bool {
    span(&source_tokens(input), &lex(expr)).is_some()
}
pub(super) fn equated_to_zero(input: &str, expr: &str) -> bool {
    let source = source_tokens(input);
    let Some((_, end)) = span(&source, &lex(expr)) else {
        return false;
    };
    source.get(end).is_some_and(|t| t.text == "=")
        && source
            .get(end + 1)
            .is_some_and(|t| t.kind == NUMBER && normalize_number(&t.text) == "0")
}

fn language_word(word: &str) -> bool {
    matches!(
        word,
        "out"
            | "express"
            | "a"
            | "an"
            | "the"
            | "of"
            | "and"
            | "or"
            | "by"
            | "from"
            | "to"
            | "in"
            | "into"
            | "for"
            | "at"
            | "with"
            | "when"
            | "is"
            | "are"
            | "as"
            | "on"
            | "over"
            | "under"
            | "between"
            | "through"
            | "than"
            | "then"
            | "after"
            | "before"
            | "all"
            | "each"
            | "these"
            | "that"
            | "this"
            | "those"
            | "it"
            | "its"
            | "them"
            | "me"
            | "you"
            | "your"
            | "please"
            | "what"
            | "how"
            | "much"
            | "many"
            | "give"
            | "take"
            | "find"
            | "compute"
            | "calculate"
            | "evaluate"
            | "get"
            | "need"
            | "want"
            | "tell"
            | "first"
            | "last"
            | "result"
            | "quantity"
            | "values"
            | "value"
            | "sample"
            | "data"
            | "observations"
            | "numbers"
            | "number"
            | "matrix"
            | "matrices"
            | "vector"
            | "equation"
            | "equations"
            | "linear"
            | "system"
            | "expression"
            | "respect"
            | "order"
            | "decimal"
            | "decimals"
            | "places"
            | "digits"
            | "th"
            | "st"
            | "nd"
            | "rd"
            | "base"
            | "taken"
            | "time"
            | "ways"
            | "count"
            | "selecting"
            | "selections"
            | "ordered"
            | "coefficient"
            | "probability"
            | "success"
            | "successes"
            | "trials"
            | "rate"
            | "lambda"
            | "mu"
            | "sigma"
            | "sd"
            | "mean"
            | "standard"
            | "deviation"
            | "normal"
            | "binomial"
            | "poisson"
            | "distribution"
            | "pdf"
            | "cdf"
            | "pmf"
            | "quantile"
            | "rounded"
            | "rounding"
            | "root"
            | "roots"
            | "square"
            | "squared"
            | "cube"
            | "cubed"
            | "cubic"
            | "squre"
            | "inverse"
            | "natural"
            | "hyperbolic"
            | "absolute"
            | "greatest"
            | "highest"
            | "lowest"
            | "least"
            | "common"
            | "divisor"
            | "factor"
            | "multiple"
            | "added"
            | "adding"
            | "multiplied"
            | "divided"
            | "dividing"
            | "raised"
            | "raise"
            | "power"
            | "exponent"
            | "exponentiate"
            | "plus"
            | "minus"
            | "negative"
            | "times"
            | "tims"
            | "less"
            | "percent"
            | "per"
            | "cent"
            | "modulo"
            | "remainder"
            | "choose"
            | "permute"
            | "permutations"
            | "combinations"
            | "increase"
            | "reduce"
            | "decrease"
            | "ratio"
            | "difference"
            | "quotient"
            | "product"
            | "sum"
            | "add"
            | "subtract"
            | "multiply"
            | "divide"
            | "sqrt"
            | "cbrt"
            | "sine"
            | "cosine"
            | "tangent"
            | "arcsine"
            | "arccosine"
            | "arctangent"
            | "sin"
            | "cos"
            | "tan"
            | "asin"
            | "acos"
            | "atan"
            | "sinh"
            | "cosh"
            | "tanh"
            | "exp"
            | "exponential"
            | "ln"
            | "log"
            | "logarithm"
            | "factorial"
            | "abs"
            | "magnitude"
            | "round"
            | "mod"
            | "gcd"
            | "lcm"
            | "ncr"
            | "npr"
            | "average"
            | "median"
            | "variance"
            | "stdev"
            | "minimum"
            | "smallest"
            | "maximum"
            | "largest"
            | "min"
            | "max"
            | "determinant"
            | "det"
            | "transpose"
            | "eigenvalues"
            | "matmul"
            | "solve_linear"
            | "inside"
            | "parentheses"
            | "followed"
    )
}

fn canonical(name: &str) -> &str {
    if name == "sum" { "add" } else { name }
}
fn aliases() -> &'static [(&'static str, &'static str)] {
    &[
        ("greatest common divisor", "gcd"),
        ("highest common factor", "gcd"),
        ("least common multiple", "lcm"),
        ("lowest common multiple", "lcm"),
        ("standard deviation", "stdev"),
        ("smallest value", "min"),
        ("largest value", "max"),
        ("absolute value", "abs"),
        ("natural logarithm", "ln"),
        ("natural log", "ln"),
        ("hyperbolic sine", "sinh"),
        ("hyperbolic cosine", "cosh"),
        ("hyperbolic tangent", "tanh"),
        ("inverse sine", "asin"),
        ("inverse cosine", "acos"),
        ("inverse tangent", "atan"),
        ("square roots", "sqrt"),
        ("square root", "sqrt"),
        ("squre root", "sqrt"),
        ("cube root", "cbrt"),
        ("cubic root", "cbrt"),
        ("square of", "power"),
        ("cube of", "power"),
        ("to the power", "power"),
        ("raised to", "power"),
        ("to the exponent", "power"),
        ("multiplied by", "multiply"),
        ("divided by", "divide"),
        ("divided into", "divide"),
        ("added to", "add"),
        ("per cent", "percent"),
        ("binomial coefficient", "ncr"),
        ("ordered selections", "npr"),
        ("combinations", "ncr"),
        ("permutations", "npr"),
        ("add", "add"),
        ("adding", "add"),
        ("sum", "add"),
        ("plus", "add"),
        ("increase", "add"),
        ("subtract", "subtract"),
        ("difference", "subtract"),
        ("minus", "subtract"),
        ("less", "subtract"),
        ("reduce", "subtract"),
        ("decrease", "subtract"),
        ("multiply", "multiply"),
        ("product", "multiply"),
        ("times", "multiply"),
        ("tims", "multiply"),
        ("twice", "multiply"),
        ("divide", "divide"),
        ("dividing", "divide"),
        ("quotient", "divide"),
        ("ratio", "divide"),
        ("over", "divide"),
        ("power", "power"),
        ("exponent", "power"),
        ("exponentiate", "power"),
        ("squared", "power"),
        ("cubed", "power"),
        ("sqrt", "sqrt"),
        ("cbrt", "cbrt"),
        ("sine", "sin"),
        ("sin", "sin"),
        ("cosine", "cos"),
        ("cos", "cos"),
        ("tangent", "tan"),
        ("tan", "tan"),
        ("arcsine", "asin"),
        ("asin", "asin"),
        ("arccosine", "acos"),
        ("acos", "acos"),
        ("arctangent", "atan"),
        ("atan", "atan"),
        ("sinh", "sinh"),
        ("cosh", "cosh"),
        ("tanh", "tanh"),
        ("exponential", "exp"),
        ("exp", "exp"),
        ("ln", "ln"),
        ("logarithm", "log"),
        ("log", "log"),
        ("factorial", "factorial"),
        ("magnitude", "abs"),
        ("abs", "abs"),
        ("round", "round"),
        ("rounded", "round"),
        ("rounding", "round"),
        ("modulo", "mod"),
        ("mod", "mod"),
        ("remainder", "mod"),
        ("gcd", "gcd"),
        ("lcm", "lcm"),
        ("choose", "ncr"),
        ("ncr", "ncr"),
        ("permute", "npr"),
        ("npr", "npr"),
        ("percent", "percent"),
        ("mean", "mean"),
        ("average", "mean"),
        ("median", "median"),
        ("variance", "variance"),
        ("stdev", "stdev"),
        ("minimum", "min"),
        ("min", "min"),
        ("maximum", "max"),
        ("max", "max"),
        ("quantile", "quantile"),
        ("determinant", "det"),
        ("det", "det"),
        ("inverse", "inverse"),
        ("transpose", "transpose"),
        ("eigenvalues", "eigenvalues"),
    ]
}

fn cues(tokens: &[Token]) -> Vec<(usize, String)> {
    let lower: Vec<_> = tokens.iter().map(|t| t.text.to_ascii_lowercase()).collect();
    let mut out = Vec::new();
    let mut i = 0;
    while i < tokens.len() {
        let matched = aliases()
            .iter()
            .filter_map(|(phrase, name)| {
                let parts: Vec<_> = phrase.split_whitespace().collect();
                (i + parts.len() <= lower.len()
                    && lower[i..i + parts.len()]
                        .iter()
                        .map(String::as_str)
                        .eq(parts.iter().copied()))
                .then_some((parts.len(), *name))
            })
            .max_by_key(|(len, _)| *len);
        if let Some((len, name)) = matched {
            out.push((i, name.into()));
            i += len;
        } else {
            let op = match tokens[i].text.as_str() {
                "+" => Some("add"),
                "*" => Some("multiply"),
                "/" => Some("divide"),
                "^" => Some("power"),
                "√" => Some("sqrt"),
                "%" => Some("percent"),
                "!" => Some("factorial"),
                "-" if quantity_before(tokens, i) => Some("subtract"),
                _ => None,
            };
            if let Some(op) = op {
                out.push((i, op.into()));
            }
            i += 1;
        }
    }
    // In 'binomial coefficient N choose K', the first phrase names the
    // following choose operation. It does not ask for a second nested nCr.
    let combinations: Vec<_> = out
        .iter()
        .enumerate()
        .filter(|(_, (_, name))| name == "ncr")
        .map(|(index, (position, _))| (index, *position))
        .collect();
    if let [(annotation, first), (_, second)] = combinations.as_slice()
        && lower.get(*first).is_some_and(|word| word == "binomial")
        && lower
            .get(*first + 1)
            .is_some_and(|word| word == "coefficient")
        && lower.get(*second).is_some_and(|word| word == "choose")
        && tokens.iter().filter(|token| token.kind == NUMBER).count() == 2
    {
        out.remove(*annotation);
    }
    out
}

fn expected_function(input: &str, tokens: &[Token]) -> Option<String> {
    let lower = input.to_ascii_lowercase();
    for name in [
        "normal_pdf",
        "normal_cdf",
        "normal_quantile",
        "binomial_pmf",
        "binomial_cdf",
        "poisson_pmf",
        "poisson_cdf",
    ] {
        if lower.contains(name) || lower.contains(&name.replace('_', " ")) {
            return Some(name.into());
        }
    }
    if lower.contains("solve_linear")
        || lower.contains("linear system")
        || lower.contains("linear equations")
        || lower.contains("matrix equation")
        || lower.contains("a*x=b")
    {
        return Some("solve_linear".into());
    }
    if (lower.contains("matrix") || lower.contains("matrices"))
        && (lower.contains("multiply") || lower.contains("product"))
    {
        return Some("matmul".into());
    }
    let all = cues(tokens);
    let mut depth = 0usize;
    let depths: Vec<_> = tokens
        .iter()
        .map(|token| {
            if matches!(token.text.as_str(), ")" | "]") {
                depth = depth.saturating_sub(1);
            }
            let current = depth;
            if matches!(token.text.as_str(), "(" | "[") {
                depth += 1;
            }
            current
        })
        .collect();
    let shallowest = all.iter().map(|(position, _)| depths[*position]).min()?;
    let found: Vec<_> = all
        .into_iter()
        .filter(|(position, _)| depths[*position] == shallowest)
        .collect();
    // In sequential descriptions, the final stated operation is the outer call.
    for phrase in ["then", "followed", "after"] {
        if let Some(i) = tokens
            .iter()
            .rposition(|t| t.text.eq_ignore_ascii_case(phrase))
        {
            if phrase == "after"
                && !tokens
                    .get(i + 1)
                    .is_some_and(|token| token.text.eq_ignore_ascii_case("that"))
            {
                continue;
            }
            if let Some((_, name)) = found.iter().find(|(p, _)| *p > i) {
                return Some(name.clone());
            }
            if tokens[i + 1..]
                .iter()
                .any(|t| t.text.eq_ignore_ascii_case("raise"))
            {
                return Some("power".into());
            }
        }
    }
    if lower.contains("and divide") {
        return Some("divide".into());
    }
    if lower.contains("and multiply") {
        return Some("multiply".into());
    }
    if lower.contains(", raised to") {
        return Some("power".into());
    }
    if tokens.iter().any(|t| t.text.eq_ignore_ascii_case("raise")) {
        return Some("power".into());
    }
    let infix = |(position, name): &&(usize, String)| {
        matches!(name.as_str(), "add" | "subtract" | "multiply" | "divide")
            && matches!(
                tokens[*position].text.to_ascii_lowercase().as_str(),
                "+" | "-"
                    | "*"
                    | "/"
                    | "plus"
                    | "minus"
                    | "times"
                    | "tims"
                    | "less"
                    | "over"
                    | "divided"
                    | "multiplied"
            )
    };
    // A postfix power before an infix plus is an operand, not the outer call.
    if found.first().is_some_and(|(_, n)| n == "power")
        && (lower.contains("squared") || lower.contains("square of") || lower.contains("cube of"))
        && let Some((_, op)) = found.iter().find(infix)
    {
        return Some(op.clone());
    }
    if let Some((_, first)) = found.first()
        && matches!(
            first.as_str(),
            "sqrt"
                | "cbrt"
                | "abs"
                | "sin"
                | "cos"
                | "tan"
                | "asin"
                | "acos"
                | "atan"
                | "sinh"
                | "cosh"
                | "tanh"
                | "ln"
                | "log"
                | "exp"
                | "factorial"
        )
        && found.iter().filter(|(_, n)| n == first).count() > 1
        && let Some((_, op)) = found.iter().find(infix)
    {
        return Some(op.clone());
    }
    found.first().map(|(_, name)| name.clone())
}

fn identifiers(tokens: &[Token]) -> BTreeMap<String, usize> {
    let mut out = BTreeMap::new();
    for t in tokens.iter().filter(|t| t.kind == WORD) {
        *out.entry(t.text.clone()).or_insert(0) += 1;
    }
    out
}

fn symbols_and_units(
    input: &[Token],
    output: &[Token],
    allowed: &BTreeSet<String>,
    preserve_units: bool,
) -> Result<()> {
    let original = identifiers(input);
    let generated = identifiers(output);
    for (i, t) in output.iter().enumerate().filter(|(_, t)| t.kind == WORD) {
        let is_function = output.get(i + 1).is_some_and(|n| n.text == "(");
        if !original.contains_key(&t.text)
            && !(is_function && allowed.contains(canonical(&t.text)))
            && !(t.text == "pi" && original.contains_key("π"))
        {
            return Err(reject(
                "The tool invented a symbol, function, or unit. Preserve the source quantities and named operations.",
            ));
        }
    }
    if preserve_units {
        let mut essential = BTreeMap::new();
        for (i, t) in input.iter().enumerate().filter(|(_, t)| t.kind == WORD) {
            let constant = matches!(t.text.as_str(), "pi" | "π" | "e" | "i" | "tau");
            let following_quantity = i > 0
                && (input[i - 1].kind == NUMBER
                    || matches!(input[i - 1].text.as_str(), ")" | "]")
                    || matches!(input[i - 1].text.as_str(), "pi" | "e" | "i"));
            let binding = input.get(i + 1).is_some_and(|t| t.text == "=");
            let counted_items =
                matches!(t.text.to_ascii_lowercase().as_str(), "items" | "elements")
                    && (allowed.contains("ncr") || allowed.contains("npr"));
            if !binding
                && !counted_items
                && (constant
                    || (following_quantity && !language_word(&t.text.to_ascii_lowercase())))
            {
                *essential.entry(t.text.clone()).or_insert(0) += 1;
            }
        }
        for (unit, count) in essential {
            if generated.get(&unit).copied().unwrap_or(0) < count {
                return Err(reject("The tool omitted a source unit or constant."));
            }
        }
    }
    Ok(())
}

fn argument_position(source: &[Token], argument: &str) -> Option<usize> {
    let mut arg = lex(argument);
    if let Some((p, _)) = span(source, &arg) {
        return Some(p);
    }
    while arg.len() > 2
        && arg.first().is_some_and(|t| t.text == "(")
        && arg.last().is_some_and(|t| t.text == ")")
    {
        arg = arg[1..arg.len() - 1].to_vec();
        if let Some((p, _)) = span(source, &arg) {
            return Some(p);
        }
    }
    let leaves: Vec<_> = arg
        .iter()
        .filter(|t| {
            t.kind == NUMBER
                || (t.kind == WORD && matches!(t.text.as_str(), "pi" | "e" | "i" | "tau"))
        })
        .cloned()
        .collect();
    leaves
        .iter()
        .filter_map(|t| span(source, std::slice::from_ref(t)).map(|(p, _)| p))
        .min()
}

fn named_parameter(
    source: &[Token],
    argument: &str,
    labels: &[&str],
    suffixes: &[&str],
) -> Result<()> {
    let argument = normalized(&lex(argument));
    let lower: Vec<_> = source
        .iter()
        .map(|token| token.text.to_ascii_lowercase())
        .collect();
    let mut stated = false;
    let mut matched = false;
    for label in labels {
        let label: Vec<_> = label.split_whitespace().collect();
        for (position, window) in lower.windows(label.len()).enumerate() {
            if !window.iter().map(String::as_str).eq(label.iter().copied()) {
                continue;
            }
            stated = true;
            let mut value = position + label.len();
            while lower
                .get(value)
                .is_some_and(|token| matches!(token.as_str(), "is" | "=" | ":"))
            {
                value += 1;
            }
            if let Some(tokens) = source.get(value..value.saturating_add(argument.len())) {
                matched |= normalized(tokens) == argument;
            }
        }
    }
    for (position, token) in lower.iter().enumerate() {
        if !suffixes.contains(&token.as_str()) {
            continue;
        }
        stated = true;
        if position >= argument.len() {
            matched |= normalized(&source[position - argument.len()..position]) == argument;
        }
    }
    if stated && !matched {
        return Err(reject(
            "A distribution argument does not match its named parameter in the request.",
        ));
    }
    Ok(())
}

fn distribution_parameters(source: &[Token], name: &str, args: &[&str]) -> Result<()> {
    if name.starts_with("normal_") && args.len() == 3 {
        named_parameter(source, args[1], &["mean", "mu"], &[])?;
        named_parameter(source, args[2], &["sd", "sigma", "standard deviation"], &[])?;
    } else if name.starts_with("binomial_") && args.len() == 3 {
        named_parameter(source, args[1], &["n", "out of"], &["trials"])?;
        named_parameter(source, args[2], &["p", "probability"], &[])?;
    } else if name.starts_with("poisson_") && args.len() == 2 {
        named_parameter(source, args[1], &["lambda", "rate", "mean"], &[])?;
    }
    Ok(())
}

fn strip_parentheses(mut tokens: &[Token]) -> &[Token] {
    while tokens.first().is_some_and(|token| token.text == "(") {
        let mut depth = 0usize;
        let closing = tokens.iter().position(|token| {
            match token.text.as_str() {
                "(" => depth += 1,
                ")" => depth = depth.saturating_sub(1),
                _ => {}
            }
            token.text == ")" && depth == 0
        });
        if closing != Some(tokens.len() - 1) {
            break;
        }
        tokens = &tokens[1..tokens.len() - 1];
    }
    tokens
}

fn balanced(tokens: &[Token]) -> bool {
    let mut stack = Vec::new();
    for token in tokens {
        match token.text.as_str() {
            "(" => stack.push(")"),
            "[" => stack.push("]"),
            ")" | "]" if stack.pop() != Some(token.text.as_str()) => return false,
            _ => {}
        }
    }
    stack.is_empty()
}

/// Inspect just the root of a generated operand; the normal guards still
/// validate the operation and both copied operands against the original step.
fn operand_arguments(expression: &str, expected: &str) -> Result<Vec<String>> {
    let expression = lex(expression);
    if !balanced(&expression) {
        return Err(reject(
            "A sequential operand has unbalanced parentheses or brackets.",
        ));
    }
    let tokens = strip_parentheses(&expression);
    let mut depth = 0usize;
    let mut root = None;
    for (position, token) in tokens.iter().enumerate() {
        match token.text.as_str() {
            "(" | "[" => {
                depth += 1;
                continue;
            }
            ")" | "]" => {
                depth = depth.saturating_sub(1);
                continue;
            }
            _ => {}
        }
        if depth != 0 || position == 0 {
            continue;
        }
        let operation = match token.text.as_str() {
            "+" => Some((1, "add")),
            "-" if quantity_before(tokens, position) => Some((1, "subtract")),
            "*" => Some((2, "multiply")),
            "/" => Some((2, "divide")),
            "^" => Some((3, "power")),
            _ => None,
        };
        if let Some((precedence, operation)) = operation
            && root.is_none_or(|(_, current, _)| {
                precedence < current || (precedence == current && precedence != 3)
            })
        {
            root = Some((position, precedence, operation));
        }
    }
    let text = |tokens: &[Token]| {
        tokens
            .iter()
            .map(|token| token.text.as_str())
            .collect::<Vec<_>>()
            .join(" ")
    };
    if let Some((position, _, operation)) = root {
        if operation == canonical(expected) && position + 1 < tokens.len() {
            return Ok(vec![
                text(&tokens[..position]),
                text(&tokens[position + 1..]),
            ]);
        }
    } else if tokens.len() >= 3
        && tokens[0].kind == WORD
        && canonical(&tokens[0].text) == canonical(expected)
        && tokens[1].text == "("
        && tokens.last().is_some_and(|token| token.text == ")")
    {
        let body = &tokens[2..tokens.len() - 1];
        if !balanced(body) {
            return Err(reject(
                "A sequential operand must preserve the complete function call.",
            ));
        }
        let mut parts = Vec::new();
        let mut begin = 0;
        let mut depth = 0usize;
        for (position, token) in body.iter().enumerate() {
            match token.text.as_str() {
                "(" | "[" => depth += 1,
                ")" | "]" => depth = depth.saturating_sub(1),
                "," if depth == 0 => {
                    parts.push(text(&body[begin..position]));
                    begin = position + 1;
                }
                _ => {}
            }
        }
        parts.push(text(&body[begin..]));
        if parts.iter().all(|part| !part.is_empty()) {
            return Ok(parts);
        }
    }
    Err(reject(
        "The operand does not preserve the first step's operation and grouping.",
    ))
}

fn sequential_steps(input: &str, name: &str, args: &[&str]) -> Option<Result<()>> {
    let lower = input.to_ascii_lowercase();
    let (marker, offset) = ["; after that,", "; after that "]
        .iter()
        .find_map(|marker| lower.find(marker).map(|offset| (*marker, offset)))?;
    Some((|| {
        if lower.match_indices("after that").count() != 1 {
            return Err(reject(
                "Use an explicit expression for more than two sequential steps.",
            ));
        }
        let first = &input[..offset];
        let second = input[offset + marker.len()..].trim();
        let first_tokens = source_tokens(first);
        let first_operation = expected_function(first, &first_tokens)
            .ok_or_else(|| reject("Cannot identify the first stated operation."))?;
        let values = source_quantities(&first_tokens);
        let prior = args
            .iter()
            .find(|argument| quantities(&lex(argument)) == values)
            .ok_or_else(|| reject("One operand must preserve the entire first step."))?;
        let first_math = first.trim();
        let first_math = first_math
            .split_once(char::is_whitespace)
            .filter(|(word, _)| word.eq_ignore_ascii_case("first"))
            .map_or(first_math, |(_, expression)| expression.trim());
        if let Ok(expression) = super::literal_expression(first_math)
            && normalized(strip_parentheses(&lex(&expression)))
                != normalized(strip_parentheses(&lex(prior)))
        {
            return Err(reject(
                "The first step's literal expression must retain all its operators and parentheses.",
            ));
        }
        let prior_args = operand_arguments(prior, &first_operation)?;
        function(
            first,
            &first_operation,
            &prior_args.iter().map(String::as_str).collect::<Vec<_>>(),
        )?;
        let lower_second = second.to_ascii_lowercase();
        let (reference, position) = ["the result", "that result", "the answer", "that answer"]
            .iter()
            .find_map(|reference| {
                lower_second
                    .find(reference)
                    .map(|position| (*reference, position))
            })
            .ok_or_else(|| reject("The second step must explicitly refer to the first result."))?;
        let rewritten = format!(
            "{}({}){}",
            &second[..position],
            prior,
            &second[position + reference.len()..]
        );
        if rewritten.len() > 8192 {
            return Err(reject("Sequential expression is too long."));
        }
        numbers(&rewritten, &args.join(","))?;
        function(&rewritten, name, args)
    })())
}

/// In `divide A into B`, A is the divisor. Require both complete operand
/// groups; a nested verbal division needs an explicit expression or step.
fn divided_into(source: &[Token], name: &str, args: &[&str]) -> Result<bool> {
    let verbs: Vec<_> = source
        .iter()
        .enumerate()
        .filter_map(|(position, token)| {
            matches!(
                token.text.to_ascii_lowercase().as_str(),
                "divide" | "divided" | "dividing"
            )
            .then_some(position)
        })
        .collect();
    let separators: Vec<_> = source
        .iter()
        .enumerate()
        .filter_map(|(position, token)| token.text.eq_ignore_ascii_case("into").then_some(position))
        .collect();
    if verbs.is_empty() || separators.is_empty() {
        return Ok(false);
    }
    let ([verb], [separator]) = (verbs.as_slice(), separators.as_slice()) else {
        return Err(reject(
            "Use an explicit expression or separate steps for nested division into a quantity.",
        ));
    };
    if name != "divide" || args.len() != 2 || verb >= separator {
        return Err(reject(
            "Cannot ground the grouping of this divide-into request.",
        ));
    }
    for (group, argument) in [
        (&source[..*separator], args[1]),
        (&source[*separator + 1..], args[0]),
    ] {
        let argument = lex(argument);
        if source_quantities(group) != quantities(&argument)
            || span(group, strip_parentheses(&argument)).is_none()
        {
            let mut message = "Divide A into B requires B as the complete numerator and A as the complete denominator.".to_owned();
            if let Some(order) =
                required_group_quantities(&source[*separator + 1..], &source[..*separator])
            {
                message.push_str(&order);
            }
            return Err(reject(&message));
        }
    }
    Ok(true)
}

pub(super) fn function(input: &str, name: &str, args: &[&str]) -> Result<()> {
    if let Some(result) = sequential_steps(input, name, args) {
        return result;
    }
    if name == "percent" {
        let lower = input.to_ascii_lowercase();
        if !(lower.contains("percent of")
            || lower.contains("% of")
            || lower.contains("per cent of"))
            || lower.contains("change")
            || lower.contains("increase")
            || lower.contains("decrease")
        {
            return Err(reject(
                "The percent tool computes a percent of a quantity; percentage change requires an explicit expression.",
            ));
        }
    }
    let source = source_tokens(input);
    distribution_parameters(&source, name, args)?;
    if name.contains('_') && args.iter().any(|arg| span(&source, &lex(arg)).is_none()) {
        return Err(reject(
            "Matrix and distribution parameters must preserve complete source expressions.",
        ));
    }
    let expected=expected_function(input,&source).ok_or_else(||reject("No matching mathematical operation was stated. Use an explicit expression if the wording is ambiguous."))?;
    if canonical(name) != expected {
        let proposed: String = name.chars().take(64).collect();
        return Err(reject(&format!(
            "The source requests {expected}, but the tool proposed {proposed}. Select {expected} and preserve the original operands.",
        )));
    }
    let reversed_division = divided_into(&source, name, args)?;
    let found = cues(&source);
    let allowed: BTreeSet<_> = found
        .iter()
        .map(|(_, name)| name.clone())
        .chain(std::iter::once(expected))
        .collect();
    let output = lex(&args.join(","));
    symbols_and_units(&source, &output, &allowed, true)?;
    let used = cues(&output);
    for (_, op) in &used {
        if !allowed.contains(op) {
            return Err(reject(
                "An operand introduced an operation absent from the request.",
            ));
        }
    }
    for op in ["add", "subtract", "multiply", "divide", "power"] {
        let applied =
            used.iter().filter(|(_, n)| n == op).count() + usize::from(canonical(name) == op);
        let stated = found.iter().filter(|(_, n)| n == op).count();
        if applied > stated
            && !(op == "power" && input.to_ascii_lowercase().contains("raise"))
            && !(name == "matmul" && op == "multiply")
        {
            return Err(reject("The tool inserted an extra arithmetic operation."));
        }
    }
    grouped_operands(&source, name, args)?;
    // Named unary operations cannot disappear or be duplicated in a composition.
    for op in [
        "sqrt",
        "cbrt",
        "sin",
        "cos",
        "tan",
        "asin",
        "acos",
        "atan",
        "sinh",
        "cosh",
        "tanh",
        "ln",
        "log",
        "exp",
        "abs",
        "mean",
        "factorial",
        "round",
        "mod",
        "gcd",
        "lcm",
        "ncr",
        "npr",
        "percent",
        "det",
        "inverse",
        "transpose",
        "eigenvalues",
        "median",
        "variance",
        "stdev",
        "min",
        "max",
        "quantile",
    ] {
        let mut stated = found.iter().filter(|(_, n)| n == op).count();
        if op == "sqrt" && input.to_ascii_lowercase().contains("square roots") {
            stated = source
                .iter()
                .filter(|t| t.kind == NUMBER)
                .count()
                .max(stated);
        }
        let applied = used.iter().filter(|(_, n)| n == op).count() + usize::from(name == op);
        // Parameter labels such as a normal distribution's mean are not operations.
        if !name.contains('_') && applied != stated {
            return Err(reject(
                "The tool changed a nested function or omitted a stated operation.",
            ));
        }
    }
    if args.len() == 2
        && matches!(
            name,
            "subtract" | "divide" | "power" | "mod" | "log" | "ncr" | "npr" | "round" | "matmul"
        )
    {
        if name == "power"
            && found
                .iter()
                .filter(|(_, operation)| operation == "power")
                .count()
                == 1
        {
            let implicit = found
                .iter()
                .find(|(_, operation)| operation == "power")
                .and_then(|(position, _)| {
                    match source[*position].text.to_ascii_lowercase().as_str() {
                        "square" | "squared" => Some("2"),
                        "cube" | "cubed" => Some("3"),
                        _ => None,
                    }
                });
            if let Some(exponent) = implicit {
                if normalized(strip_parentheses(&lex(args[1]))) == normalized(&lex(exponent)) {
                    return Ok(());
                }
                return Err(reject(
                    "Squared and cubed supply the exponent 2 or 3, respectively.",
                ));
            }
        }
        let left = argument_position(&source, args[0]);
        let right = argument_position(&source, args[1]);
        let lower = input.to_ascii_lowercase();
        let reversed = reversed_division
            || (name == "subtract" && lower.contains("subtract") && lower.contains(" from "))
            || (name == "subtract"
                && source.windows(2).any(|tokens| {
                    tokens[0].text.eq_ignore_ascii_case("less")
                        && tokens[1].text.eq_ignore_ascii_case("than")
                }))
            || (name == "log" && (lower.contains("log base ") || lower.contains("the base ")))
            || (matches!(name, "ncr" | "npr") && lower.contains(" from "))
            || (name == "power"
                && source.windows(4).any(|tokens| {
                    tokens[0].kind == NUMBER
                        && matches!(
                            tokens[1].text.to_ascii_lowercase().as_str(),
                            "th" | "st" | "nd" | "rd"
                        )
                        && tokens[2].text.eq_ignore_ascii_case("power")
                        && tokens[3].text.eq_ignore_ascii_case("of")
                }));
        if let (Some(a), Some(b)) = (left, right) {
            let prior_sum =
                name == "divide" && lower.contains("and divide") && lower.contains("that sum");
            if prior_sum {
                let divide = source
                    .iter()
                    .rposition(|t| t.text.eq_ignore_ascii_case("divide"))
                    .unwrap_or(0);
                if a < divide || b >= divide {
                    return Err(reject(
                        "The numerator must follow divide; the denominator is the previously stated sum.",
                    ));
                }
            } else if a != b && ((reversed && a < b) || (!reversed && a > b)) {
                let required = serde_json::json!([args[1], args[0]]).to_string();
                let order = if required.len() <= 200 {
                    format!("Required argument order from the source: {required}.")
                } else if reversed {
                    "The later source group is the first argument; the earlier source group is the second.".to_owned()
                } else {
                    "The earlier source group is the first argument; the later source group is the second.".to_owned()
                };
                return Err(reject(&format!(
                    "Incorrect operand order for {name}. {order} Preserve the original grouping and units."
                )));
            }
        } else {
            return Err(reject("Cannot locate both operands in the request."));
        }
    }
    Ok(())
}

fn ordered_quantities(tokens: &[Token]) -> Vec<String> {
    tokens
        .iter()
        .enumerate()
        .filter(|(_, t)| t.kind == NUMBER)
        .map(|(i, t)| {
            let mut token = normalize_number(&t.text);
            if i > 0 && tokens[i - 1].text == "-" && !quantity_before(tokens, i - 1) {
                token.insert(0, '-');
            }
            token
        })
        .collect()
}

fn grouped_operands(source: &[Token], name: &str, args: &[&str]) -> Result<()> {
    let words: Vec<_> = source.iter().map(|t| t.text.to_ascii_lowercase()).collect();
    for (noun, operation) in [
        ("difference", "subtract"),
        ("sum", "add"),
        ("quotient", "divide"),
        ("ratio", "divide"),
    ] {
        let Some(start) = words.iter().position(|w| w == noun) else {
            continue;
        };
        if canonical(name) == operation {
            continue;
        }
        let begin = if words.get(start + 1).is_some_and(|w| w == "of") {
            start + 2
        } else {
            start + 1
        };
        let mut ratio_separator = false;
        let end = (begin..words.len())
            .find(|&i| {
                if noun == "ratio" && words[i] == "to" && !ratio_separator {
                    ratio_separator = true;
                    return false;
                }
                matches!(words[i].as_str(), "by" | "to" | "then")
                    || (words[i] == "and"
                        && words
                            .get(i + 1)
                            .is_some_and(|w| matches!(w.as_str(), "divide" | "multiply")))
            })
            .unwrap_or(words.len());
        let group = &source[begin..end];
        if group.iter().filter(|t| t.kind == NUMBER).count() < 2 {
            continue;
        }
        let expected = source_quantities(group);
        let matching = args.iter().find(|arg| {
            quantities(&lex(arg)) == expected
                && cues(&lex(arg)).iter().any(|(_, op)| op == operation)
        });
        let Some(argument) = matching else {
            return Err(reject(
                "The model changed which operands belong to the stated sum, difference, or ratio.",
            ));
        };
        if matches!(operation, "subtract" | "divide")
            && ordered_quantities(group) != ordered_quantities(&lex(argument))
        {
            return Err(reject(
                "The model reversed the operands inside the stated difference or ratio.",
            ));
        }
    }
    // 'divide A by B' also fixes grouping: (A+B)/C is not A/(B+C).
    if name == "divide"
        && !words.windows(2).any(|w| w == ["that", "sum"])
        && let Some(by) = words.iter().rposition(|w| w == "by")
    {
        let left = source_quantities(&source[..by]);
        let right = source_quantities(&source[by + 1..]);
        if args.len() == 2
            && (left != quantities(&lex(args[0])) || right != quantities(&lex(args[1])))
        {
            let mut message =
                "The numerator and denominator must preserve the groups on either side of by."
                    .to_owned();
            if let Some(order) = required_group_quantities(&source[..by], &source[by + 1..]) {
                message.push_str(&order);
            }
            return Err(reject(&message));
        }
    }
    Ok(())
}

pub(super) fn conversion(input: &str, value: &str, unit: &str) -> Result<()> {
    let source = source_tokens(strip_wrapper(input));
    let value_tokens = lex(value);
    let target = lex(unit);
    if span(&source, &value_tokens).is_none() || span(&source, &target).is_none() {
        return Err(reject(
            "Conversion units and quantity must be copied from the request.",
        ));
    }
    let lower: Vec<_> = source.iter().map(|t| t.text.to_ascii_lowercase()).collect();
    let mut destination = None;
    if lower.starts_with(&["how".into(), "many".into()]) {
        if let Some(end) = lower.iter().position(|t| t == "is" || t == "are") {
            destination = Some(&source[2..end]);
        }
    } else if let Some(i) = lower
        .iter()
        .rposition(|t| matches!(t.as_str(), "to" | "into" | "in" | "as"))
    {
        destination = Some(&source[i + 1..]);
    }
    let destination = destination.ok_or_else(|| {
        reject("State the conversion destination explicitly, for example convert 5 m to cm.")
    })?;
    if normalized(destination) != normalized(&target) {
        return Err(reject("The tool changed the requested destination unit."));
    }
    symbols_and_units(
        &source,
        &lex(&format!("{value},{unit}")),
        &BTreeSet::new(),
        true,
    )
}

pub(super) fn algebra_operation(input: &str, operation: &str) -> Result<()> {
    let s = input.to_ascii_lowercase();
    let tokens = lex(&s);
    let has = |words: &[&str]| {
        tokens
            .iter()
            .any(|t| t.kind == WORD && words.contains(&t.text.as_str()))
    };
    let expected = if has(&[
        "substitute",
        "substituting",
        "substitution",
        "replace",
        "put",
    ]) {
        Some("substitute")
    } else if has(&["diff", "differentiate", "differentiation", "derivative"]) || s.contains("d/d")
    {
        Some("diff")
    } else if has(&[
        "integrate",
        "integration",
        "integral",
        "antiderivative",
        "primitive",
    ]) || s.contains("area under")
    {
        Some("integrate")
    } else if (s.contains("root of") && !s.contains("square root") && !s.contains("cube root"))
        || s.contains("zero of")
        || s.contains("solve numerically")
    {
        Some("root")
    } else if has(&["simplify", "simplification"]) {
        Some("simplify")
    } else if has(&["expand", "expansion"]) {
        Some("expand")
    } else if has(&["factor", "factorize", "factorise", "factorization"]) {
        Some("factor")
    } else if has(&["solve", "solutions", "satisfying"]) {
        Some("solve")
    } else {
        None
    };
    if expected != Some(operation) {
        return Err(reject(
            "The algebra tool changed or could not identify the stated operation.",
        ));
    }
    Ok(())
}

pub(super) fn variable_is_grounded(input: &str, expr: &str, var: &str) -> bool {
    let expression = lex(expr);
    let exists = expression.iter().enumerate().any(|(i, t)| {
        t.kind == WORD && t.text == var && !expression.get(i + 1).is_some_and(|n| n.text == "(")
    });
    if !exists {
        return false;
    }
    let source = source_tokens(input);
    for (i, t) in source.iter().enumerate() {
        if t.text.eq_ignore_ascii_case("respect")
            && source
                .get(i + 1)
                .is_some_and(|t| t.text.eq_ignore_ascii_case("to"))
        {
            return source.get(i + 2).is_some_and(|t| t.text == var);
        }
        if matches!(t.text.to_ascii_lowercase().as_str(), "in" | "for" | "all")
            && let Some(candidate) = source
                .get(i + 1)
                .filter(|t| t.kind == WORD && !language_word(&t.text.to_ascii_lowercase()))
        {
            return candidate.text == var;
        }
        if t.text == "/"
            && i > 0
            && source[i - 1].text == "d"
            && let Some(candidate) = source.get(i + 1).and_then(|t| t.text.strip_prefix('d'))
        {
            return candidate == var;
        }
        if matches!(
            t.text.to_ascii_lowercase().as_str(),
            "substitute" | "replace" | "put"
        ) && let Some(candidate) = source.get(i + 1).filter(|t| t.kind == WORD)
        {
            return candidate.text == var;
        }
    }
    true
}

pub(super) fn algebra_arguments(
    input: &str,
    expr: &str,
    operation: &str,
    args: &[String],
) -> Result<()> {
    let source = source_tokens(input);
    let output = lex(&args.join(","));
    symbols_and_units(&source, &output, &BTreeSet::new(), false)?;
    if matches!(operation, "root" | "integrate") && args.len() == 4 {
        let start = span(&source, &lex(expr)).map_or(0, |(_, end)| end);
        let tail = &source[start..];
        let a = span(tail, &lex(&args[2])).map(|(p, _)| p);
        let b = span(tail, &lex(&args[3])).map(|(p, _)| p);
        if !matches!((a,b),(Some(a),Some(b)) if a<=b) {
            return Err(reject(
                "The tool changed the order of the stated integration or root bounds.",
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn number_words_have_separate_values_and_signs() {
        for (source, output) in [
            ("minus five plus two", "-5,2"),
            ("two squared", "2^2"),
            ("add twenty one and two hundred thirty four", "21,234"),
            ("subtract minus five from ten", "10,-5"),
            ("one thousand two hundred and three plus seven", "1203,7"),
        ] {
            assert!(numbers(source, output).is_ok(), "{source}");
        }
        assert!(numbers("minus five plus two", "5,2").is_err());
        assert!(numbers("two squared", "4").is_err());
        assert!(numbers("2 seconds plus 3 seconds", "2,3").is_ok());
    }
    #[test]
    fn token_boundaries_are_preserved() {
        assert!(!contains_expression("1 m s", "1 ms"));
        assert!(!contains_expression("1 2", "12"));
        assert!(!contains_expression("sin(x)", "s"));
    }
    #[test]
    fn source_units_and_correct_composition_are_supported() {
        for (s, f, a) in [
            ("add 5 m and 2 cm", "add", vec!["5 m", "2 cm"]),
            ("square root of the sum of 5 and 4", "sqrt", vec!["(5+4)"]),
            (
                "sum of square roots of 4 and 9",
                "add",
                vec!["sqrt(4)", "sqrt(9)"],
            ),
            (
                "multiply the difference of 5 and 2 by 3",
                "multiply",
                vec!["(5-2)", "3"],
            ),
            ("add 2 squared and 3 cubed", "add", vec!["2^2", "3^3"]),
            ("subtract minus five from ten", "subtract", vec!["10", "-5"]),
        ] {
            assert!(function(s, f, &a).is_ok(), "{s}: {:?}", function(s, f, &a));
        }
    }
}
