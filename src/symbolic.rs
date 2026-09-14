//! A bounded, exact symbolic calculator.
//!
//! This is deliberately a small transparent CAS: unsupported algebra is an error,
//! never a guessed answer. Coefficients are arbitrary-precision rationals, and
//! every recursive or potentially expanding operation shares a work/deadline limit.

use crate::{Calculation, error::CalcError};
use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{One, Signed, ToPrimitive, Zero};
use std::{
    collections::{BTreeMap, BTreeSet},
    time::Instant,
};

type Rat = BigRational;
type Result<T> = std::result::Result<T, CalcError>;

const MAX_INPUT: usize = 4096;
const MAX_DEPTH: usize = 64;
const MAX_NODES: usize = 4096;
const MAX_TERMS: usize = 512;
const MAX_DEGREE: usize = 64;
const MAX_BITS: u64 = 8192;

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Expr {
    Number(Rat),
    Symbol(String),
    Add(Vec<Expr>),
    Mul(Vec<Expr>),
    Pow(Box<Expr>, Box<Expr>),
    Function(String, Box<Expr>),
}

struct Budget {
    deadline: Instant,
    steps: usize,
}
impl Budget {
    fn tick(&mut self) -> Result<()> {
        self.steps += 1;
        if self.steps > 100_000 || Instant::now() >= self.deadline {
            return Err(CalcError::limit(
                "symbolic calculation exceeded its time or work limit",
            ));
        }
        Ok(())
    }
    fn size(&mut self, expression: &Expr) -> Result<()> {
        self.tick()?;
        let mut stack = vec![(expression, 0usize)];
        let mut count = 0;
        while let Some((e, depth)) = stack.pop() {
            count += 1;
            if count > MAX_NODES || depth > MAX_DEPTH {
                return Err(CalcError::limit(
                    "symbolic result is too large or deeply nested",
                ));
            }
            match e {
                Expr::Add(xs) | Expr::Mul(xs) => stack.extend(xs.iter().map(|x| (x, depth + 1))),
                Expr::Pow(a, b) => {
                    stack.push((a, depth + 1));
                    stack.push((b, depth + 1));
                }
                Expr::Function(_, a) => stack.push((a, depth + 1)),
                _ => {}
            }
        }
        Ok(())
    }
}

fn unsupported(message: impl Into<String>) -> CalcError {
    CalcError::new("unsupported_operation", message)
}
fn integer(n: i64) -> Expr {
    Expr::Number(Rat::from_integer(n.into()))
}
fn zero(e: &Expr) -> bool {
    matches!(e, Expr::Number(n) if n.is_zero())
}
fn one(e: &Expr) -> bool {
    matches!(e, Expr::Number(n) if n.is_one())
}
fn rational(n: i64, d: i64) -> Expr {
    Expr::Number(Rat::new(n.into(), d.into()))
}
fn raw_power(a: Expr, b: Expr) -> Expr {
    Expr::Pow(Box::new(a), Box::new(b))
}
fn function(name: &str, arg: Expr) -> Expr {
    Expr::Function(name.into(), Box::new(arg))
}
fn as_integer(e: &Expr) -> Option<i64> {
    match e {
        Expr::Number(n) if n.is_integer() => n.to_integer().to_i64(),
        _ => None,
    }
}
fn checked(n: Rat) -> Result<Rat> {
    if n.numer().bits() > MAX_BITS || n.denom().bits() > MAX_BITS {
        Err(CalcError::limit("exact coefficient exceeds 8192 bits"))
    } else {
        Ok(n)
    }
}

/// Handle the symbolic DSL, or return `None` so the numerical engine can try it.
/// Four-argument integration deliberately belongs to the numerical engine.
pub fn try_evaluate(expression: &str, deadline: Instant) -> Option<Result<Calculation>> {
    let expression = expression.trim();
    let name_end = expression
        .find(|c: char| !c.is_ascii_alphanumeric() && c != '_')
        .unwrap_or(expression.len());
    let name = expression[..name_end].to_ascii_lowercase();
    let operation = match name.as_str() {
        "simplify" => "simplify",
        "expand" => "expand",
        "factor" => "factor",
        "diff" | "differentiate" | "derivative" => "diff",
        "integrate" | "integral" => "integrate",
        "solve" => "solve",
        "substitute" | "subs" => "substitute",
        _ => return None,
    };
    if expression.len() > MAX_INPUT {
        return Some(Err(CalcError::limit("symbolic input exceeds 4096 bytes")));
    }
    let rest = expression[name_end..].trim();
    if !rest.starts_with('(') || !rest.ends_with(')') {
        return Some(Err(CalcError::invalid(format!(
            "use {name}(expression, ...)"
        ))));
    }
    let args = match split_arguments(&rest[1..rest.len() - 1]) {
        Ok(a) => a,
        Err(e) => return Some(Err(e)),
    };
    if operation == "integrate" && args.len() == 4 {
        return None;
    }
    Some(evaluate(operation, &args, deadline))
}

fn split_arguments(s: &str) -> Result<Vec<&str>> {
    let mut depth = 0usize;
    let mut start = 0;
    let mut args = Vec::new();
    for (i, c) in s.char_indices() {
        match c {
            '(' => {
                depth += 1;
                if depth > MAX_DEPTH {
                    return Err(CalcError::limit("symbolic nesting exceeds 64 levels"));
                }
            }
            ')' => {
                depth = depth
                    .checked_sub(1)
                    .ok_or_else(|| CalcError::invalid("unmatched closing parenthesis"))?;
            }
            ',' if depth == 0 => {
                args.push(s[start..i].trim());
                start = i + 1;
            }
            _ => {}
        }
    }
    if depth != 0 {
        return Err(CalcError::invalid("unmatched opening parenthesis"));
    }
    args.push(s[start..].trim());
    if args.iter().any(|a| a.is_empty()) {
        return Err(CalcError::invalid("empty symbolic argument"));
    }
    Ok(args)
}

fn variable(s: &str) -> Result<&str> {
    let valid = !s.is_empty()
        && s.len() <= 32
        && s.as_bytes()[0].is_ascii_alphabetic()
        && s.bytes().all(|c| c.is_ascii_alphanumeric() || c == b'_')
        && !matches!(s, "pi" | "e" | "i" | "C");
    if valid {
        Ok(s)
    } else {
        Err(CalcError::invalid(
            "choose a variable such as x; pi, e, i and C are reserved constants",
        ))
    }
}

fn evaluate(operation: &str, args: &[&str], deadline: Instant) -> Result<Calculation> {
    let expected = match operation {
        "simplify" | "expand" | "factor" => (1, 1),
        "diff" => (2, 3),
        "integrate" | "solve" => (2, 2),
        "substitute" => (3, 3),
        _ => unreachable!(),
    };
    if args.len() < expected.0 || args.len() > expected.1 {
        return Err(CalcError::invalid(format!(
            "{operation} expects {}{} arguments",
            expected.0,
            if expected.0 != expected.1 {
                " or 3"
            } else {
                ""
            }
        )));
    }
    let mut budget = Budget { deadline, steps: 0 };
    budget.tick()?;
    let var = if args.len() >= 2 {
        Some(variable(args[1])?)
    } else {
        None
    };
    if operation == "solve" {
        return solve(args[0], var.unwrap(), &mut budget);
    }
    let parsed = parse(args[0], &mut budget)?;
    let mut constraints = BTreeSet::new();
    collect_constraints(&parsed, &mut constraints, &mut budget)?;
    let expression = simplify(parsed, &mut budget)?;
    let result = match operation {
        "simplify" => expression,
        "expand" => expand(expression, &mut budget)?,
        "factor" => factor(expression, &mut budget)?,
        "substitute" => {
            let replacement = parse(args[2], &mut budget)?;
            // Validate the replacement, but substitute its original tree so
            // y/y and ln(y)-ln(y) retain their domain restrictions.
            simplify(replacement.clone(), &mut budget)?;
            // Reparse the original expression after substitution, so cancellation
            // cannot hide a newly introduced zero denominator.
            let original = parse(args[0], &mut budget)?;
            let replaced = substitute(original, var.unwrap(), &replacement, &mut budget)?;
            constraints.clear();
            collect_constraints(&replaced, &mut constraints, &mut budget)?;
            simplify(replaced, &mut budget)?
        }
        "diff" => {
            let order = if args.len() == 3 {
                args[2].parse::<usize>().map_err(|_| {
                    CalcError::invalid("derivative order must be an integer from 0 to 16")
                })?
            } else {
                1
            };
            if order > 16 {
                return Err(CalcError::limit("derivative order exceeds 16"));
            }
            if order > 0 {
                derivative_constraints(&expression, var.unwrap(), &mut constraints);
            }
            let mut result = expression;
            for _ in 0..order {
                result = simplify(derivative(&result, var.unwrap(), &mut budget)?, &mut budget)?;
                budget.size(&result)?;
            }
            result
        }
        "integrate" => {
            let result = simplify(
                integrate(&expression, var.unwrap(), &mut budget)?,
                &mut budget,
            )?;
            budget.size(&result)?;
            collect_constraints(&result, &mut constraints, &mut budget)?;
            let expr = render(&result);
            let mut result = Calculation::text(format!("{expr} + C"), "antiderivative", true);
            if !constraints.is_empty() {
                result.display.push_str(&format!(
                    "  [where {}]",
                    constraints.iter().cloned().collect::<Vec<_>>().join(", ")
                ));
            }
            result.value = serde_json::json!({"expression":expr,"constant":"C","variable":var.unwrap(),"domain":"real intervals where the input and antiderivative are defined","conditions":constraints.into_iter().collect::<Vec<_>>()});
            return Ok(result);
        }
        _ => unreachable!(),
    };
    budget.size(&result)?;
    collect_constraints(&result, &mut constraints, &mut budget)?;
    let text = render(&result);
    let mut result = Calculation::text(&text, "symbolic", true);
    // Conditions are retained when algebra cancels a denominator or a logarithm.
    if !constraints.is_empty() {
        let conditions: Vec<_> = constraints.into_iter().collect();
        result.display = format!("{text}  [where {}]", conditions.join(", "));
        result.value = serde_json::json!({"expression":text,"conditions":conditions});
    }
    Ok(result)
}

#[derive(Clone, Debug, PartialEq)]
enum Token {
    Number(String),
    Symbol(String),
    Plus,
    Minus,
    Star,
    Slash,
    Caret,
    Open,
    Close,
    Comma,
    End,
}

fn tokenize(s: &str) -> Result<Vec<Token>> {
    if s.len() > MAX_INPUT {
        return Err(CalcError::limit("symbolic expression too long"));
    }
    let bytes = s.as_bytes();
    let mut p = 0;
    let mut out = Vec::new();
    while p < bytes.len() {
        if out.len() > 1024 {
            return Err(CalcError::limit("symbolic token limit exceeded"));
        }
        let c = bytes[p];
        if c.is_ascii_whitespace() {
            p += 1;
            continue;
        }
        if c.is_ascii_digit() || c == b'.' {
            let start = p;
            let mut dots = 0;
            let mut digits = 0;
            while p < bytes.len() && (bytes[p].is_ascii_digit() || bytes[p] == b'.') {
                if bytes[p] == b'.' {
                    dots += 1;
                } else {
                    digits += 1;
                }
                p += 1;
            }
            if dots > 1 || digits == 0 {
                return Err(CalcError::invalid("invalid decimal number"));
            }
            if p < bytes.len() && (bytes[p] == b'e' || bytes[p] == b'E') {
                let mut q = p + 1;
                if q < bytes.len() && matches!(bytes[q], b'+' | b'-') {
                    q += 1;
                }
                let exponent_start = q;
                while q < bytes.len() && bytes[q].is_ascii_digit() {
                    q += 1;
                }
                if q > exponent_start {
                    p = q;
                }
            }
            out.push(Token::Number(s[start..p].into()));
            continue;
        }
        if c.is_ascii_alphabetic() || c == b'_' {
            let start = p;
            p += 1;
            while p < bytes.len() && (bytes[p].is_ascii_alphanumeric() || bytes[p] == b'_') {
                p += 1;
            }
            if p - start > 32 {
                return Err(CalcError::invalid("symbol name exceeds 32 characters"));
            }
            out.push(Token::Symbol(s[start..p].into()));
            continue;
        }
        out.push(match c {
            b'+' => Token::Plus,
            b'-' => Token::Minus,
            b'*' => Token::Star,
            b'/' => Token::Slash,
            b'^' => Token::Caret,
            b'(' => Token::Open,
            b')' => Token::Close,
            b',' => Token::Comma,
            _ => {
                return Err(CalcError::invalid(format!(
                    "unsupported character in symbolic expression at byte {}",
                    p + 1
                )));
            }
        });
        p += 1;
    }
    out.push(Token::End);
    Ok(out)
}

struct Parser<'a> {
    tokens: Vec<Token>,
    pos: usize,
    budget: &'a mut Budget,
}
fn parse(s: &str, budget: &mut Budget) -> Result<Expr> {
    let mut parser = Parser {
        tokens: tokenize(s)?,
        pos: 0,
        budget,
    };
    let result = parser.expression(0, 0)?;
    if parser.tokens[parser.pos] != Token::End {
        return Err(CalcError::invalid(
            "unexpected token after symbolic expression",
        ));
    }
    parser.budget.size(&result)?;
    Ok(result)
}

impl Parser<'_> {
    fn expression(&mut self, minimum: u8, depth: usize) -> Result<Expr> {
        self.budget.tick()?;
        if depth > MAX_DEPTH {
            return Err(CalcError::limit("symbolic nesting exceeds 64 levels"));
        }
        let token = self.tokens[self.pos].clone();
        self.pos += 1;
        let mut lhs = match token {
            Token::Number(n) => Expr::Number(parse_number(&n)?),
            Token::Symbol(s) => {
                if self.tokens[self.pos] == Token::Open && is_function(&s) {
                    self.pos += 1;
                    let argument = self.expression(0, depth + 1)?;
                    if self.tokens[self.pos] != Token::Close {
                        return Err(CalcError::invalid(format!(
                            "{s} expects one argument and a closing parenthesis"
                        )));
                    }
                    self.pos += 1;
                    if s == "log" {
                        // The scalar and numerical calculus dialects use base
                        // ten for log(), and reserve ln() for the natural log.
                        Expr::Mul(vec![
                            function("ln", argument),
                            raw_power(function("ln", integer(10)), integer(-1)),
                        ])
                    } else {
                        function(&s, argument)
                    }
                } else if self.tokens[self.pos] == Token::Open
                    && s.len() > 1
                    && !matches!(s.as_str(), "pi")
                {
                    return Err(unsupported(format!(
                        "unknown symbolic function {s}; write * explicitly for multiplication"
                    )));
                } else {
                    Expr::Symbol(s)
                }
            }
            Token::Minus => Expr::Mul(vec![integer(-1), self.expression(25, depth + 1)?]),
            Token::Plus => self.expression(25, depth + 1)?,
            Token::Open => {
                let expression = self.expression(0, depth + 1)?;
                if self.tokens[self.pos] != Token::Close {
                    return Err(CalcError::invalid("missing closing parenthesis"));
                }
                self.pos += 1;
                expression
            }
            _ => {
                return Err(CalcError::invalid(
                    "expected a number, symbol or parenthesized expression",
                ));
            }
        };
        loop {
            let token = self.tokens[self.pos].clone();
            let (left, right, implicit) = match token {
                Token::Plus | Token::Minus => (10, 11, false),
                Token::Star | Token::Slash => (20, 21, false),
                Token::Caret => (30, 30, false),
                Token::Open | Token::Symbol(_) | Token::Number(_) => (20, 21, true),
                _ => break,
            };
            if left < minimum {
                break;
            }
            if !implicit {
                self.pos += 1;
            }
            let rhs = self.expression(right, depth + 1)?;
            lhs = match token {
                Token::Plus => Expr::Add(vec![lhs, rhs]),
                Token::Minus => Expr::Add(vec![lhs, Expr::Mul(vec![integer(-1), rhs])]),
                Token::Slash => Expr::Mul(vec![lhs, raw_power(rhs, integer(-1))]),
                Token::Caret => raw_power(lhs, rhs),
                _ => Expr::Mul(vec![lhs, rhs]),
            };
            self.budget.size(&lhs)?;
        }
        Ok(lhs)
    }
}

fn parse_number(s: &str) -> Result<Rat> {
    let (mantissa, exponent) = if let Some((a, b)) = s.split_once(['e', 'E']) {
        (
            a,
            b.parse::<i32>()
                .map_err(|_| CalcError::limit("decimal exponent too large"))?,
        )
    } else {
        (s, 0)
    };
    if exponent.unsigned_abs() > 1024 || mantissa.len() > 512 {
        return Err(CalcError::limit("exact number is too large"));
    }
    let decimals = mantissa
        .split_once('.')
        .map(|(_, d)| d.len() as i32)
        .unwrap_or(0);
    let digits = mantissa.replace('.', "");
    let n = BigInt::parse_bytes(digits.as_bytes(), 10)
        .ok_or_else(|| CalcError::invalid("invalid number"))?;
    let scale = exponent - decimals;
    checked(if scale >= 0 {
        Rat::from_integer(n * BigInt::from(10).pow(scale as u32))
    } else {
        Rat::new(n, BigInt::from(10).pow((-scale) as u32))
    })
}

fn is_function(s: &str) -> bool {
    matches!(
        s,
        "sin"
            | "cos"
            | "tan"
            | "asin"
            | "acos"
            | "atan"
            | "sinh"
            | "cosh"
            | "tanh"
            | "asinh"
            | "acosh"
            | "atanh"
            | "ln"
            | "log"
            | "exp"
            | "sqrt"
            | "abs"
            | "sign"
    )
}

fn simplify(expression: Expr, budget: &mut Budget) -> Result<Expr> {
    budget.tick()?;
    match expression {
        Expr::Number(n) => Ok(Expr::Number(checked(n)?)),
        Expr::Symbol(s) => Ok(Expr::Symbol(s)),
        Expr::Add(items) => {
            let mut terms = BTreeMap::<Expr, Rat>::new();
            let mut flat = Vec::new();
            for e in items {
                match simplify(e, budget)? {
                    Expr::Add(xs) => flat.extend(xs),
                    other => flat.push(other),
                }
            }
            for e in flat {
                let (coefficient, base) = coefficient(e);
                let combined =
                    checked(terms.remove(&base).unwrap_or_else(Rat::zero) + coefficient)?;
                if !combined.is_zero() {
                    terms.insert(base, combined);
                }
            }
            let mut out = Vec::new();
            for (base, coefficient) in terms {
                out.push(with_coefficient(coefficient, base));
            }
            Ok(match out.len() {
                0 => integer(0),
                1 => out.remove(0),
                _ => Expr::Add(out),
            })
        }
        Expr::Mul(items) => {
            let mut scalar = Rat::one();
            let mut flat = Vec::new();
            for e in items {
                match simplify(e, budget)? {
                    Expr::Mul(xs) => flat.extend(xs),
                    other => flat.push(other),
                }
            }
            let mut powers = BTreeMap::<Expr, i64>::new();
            let mut others = Vec::new();
            for e in flat {
                match e {
                    Expr::Number(n) => scalar = checked(scalar * n)?,
                    Expr::Pow(a, b) if as_integer(&b).is_some() => {
                        let p = as_integer(&b).unwrap();
                        let exponent = powers.entry(*a).or_default();
                        *exponent = exponent
                            .checked_add(p)
                            .ok_or_else(|| CalcError::limit("symbolic exponent too large"))?;
                    }
                    e => {
                        let exponent = powers.entry(e).or_default();
                        *exponent = exponent
                            .checked_add(1)
                            .ok_or_else(|| CalcError::limit("symbolic exponent too large"))?;
                    }
                }
            }
            if scalar.is_zero() {
                return Ok(integer(0));
            }
            for (base, power) in powers {
                if power != 0 {
                    others.push(if power == 1 {
                        base
                    } else {
                        simplify(raw_power(base, integer(power)), budget)?
                    });
                }
            }
            others.sort();
            if !scalar.is_one() || others.is_empty() {
                others.insert(0, Expr::Number(scalar));
            }
            Ok(if others.len() == 1 {
                others.remove(0)
            } else {
                Expr::Mul(others)
            })
        }
        Expr::Pow(base, exponent) => {
            let base = simplify(*base, budget)?;
            let exponent = simplify(*exponent, budget)?;
            if let (Expr::Number(a), Expr::Number(n)) = (&base, &exponent)
                && a.is_negative()
                && !n.is_integer()
                && (n.denom() % 2u32).is_zero()
            {
                return Err(CalcError::math(
                    "real fractional powers with an even denominator require a nonnegative base",
                ));
            }
            if zero(&base) && matches!(&exponent,Expr::Number(n) if !n.is_positive()) {
                return Err(CalcError::math(
                    "zero cannot be raised to a zero or negative power",
                ));
            }
            if one(&exponent) {
                return Ok(base);
            }
            if zero(&exponent) {
                return Ok(integer(1));
            }
            if one(&base) {
                return Ok(integer(1));
            }
            if let (Expr::Number(a), Some(p)) = (&base, as_integer(&exponent)) {
                if p.unsigned_abs() > 4096
                    || a.numer()
                        .bits()
                        .max(a.denom().bits())
                        .saturating_mul(p.unsigned_abs())
                        > MAX_BITS * 2
                {
                    return Err(CalcError::limit("exact power is too large"));
                }
                return Ok(Expr::Number(checked(a.pow(p as i32))?));
            }
            if let (Expr::Symbol(s), Some(p)) = (&base, as_integer(&exponent))
                && s == "i"
            {
                return Ok(match p.rem_euclid(4) {
                    0 => integer(1),
                    1 => base,
                    2 => integer(-1),
                    _ => Expr::Mul(vec![integer(-1), base]),
                });
            }
            // Integer outer powers are branch-safe. Fractional powers are left
            // untouched: sqrt(x^2) is not x on the real/complex domain.
            if let Expr::Pow(inner, p) = &base
                && let (Some(a), Some(b)) = (as_integer(p), as_integer(&exponent))
            {
                let power = a
                    .checked_mul(b)
                    .ok_or_else(|| CalcError::limit("symbolic exponent too large"))?;
                return simplify(raw_power(*inner.clone(), integer(power)), budget);
            }
            if as_integer(&exponent).is_some()
                && let Expr::Mul(factors) = base
            {
                return simplify(
                    Expr::Mul(
                        factors
                            .into_iter()
                            .map(|x| raw_power(x, exponent.clone()))
                            .collect(),
                    ),
                    budget,
                );
            }
            Ok(raw_power(base, exponent))
        }
        Expr::Function(name, arg) => {
            let arg = simplify(*arg, budget)?;
            if let Expr::Number(n) = &arg {
                if name == "abs" {
                    return Ok(Expr::Number(n.abs()));
                }
                if name == "sign" {
                    return Ok(integer(if n.is_zero() {
                        0
                    } else if n.is_positive() {
                        1
                    } else {
                        -1
                    }));
                }
                if name == "ln" && !n.is_positive() {
                    return Err(CalcError::math(
                        "real logarithm requires a positive argument",
                    ));
                }
                if name == "sqrt" {
                    if n.is_negative() {
                        return Err(CalcError::math(
                            "symbolic sqrt uses the real domain; use the numerical complex engine for negative arguments",
                        ));
                    }
                    if let Some(sqrt) = exact_sqrt(n) {
                        return Ok(Expr::Number(sqrt));
                    }
                }
                if matches!(name.as_str(), "asin" | "acos") && n.abs() > Rat::one() {
                    return Err(CalcError::math(
                        "real inverse trigonometric argument must be between -1 and 1",
                    ));
                }
                if name == "acosh" && n < &Rat::one() {
                    return Err(CalcError::math("real acosh argument must be at least 1"));
                }
                if name == "atanh" && n.abs() >= Rat::one() {
                    return Err(CalcError::math(
                        "real atanh argument must be strictly between -1 and 1",
                    ));
                }
                if n.is_zero() {
                    match name.as_str() {
                        "sin" | "tan" | "asin" | "atan" | "sinh" | "tanh" | "asinh" | "atanh" => {
                            return Ok(integer(0));
                        }
                        "cos" | "cosh" | "exp" => return Ok(integer(1)),
                        _ => {}
                    }
                }
                if n.is_one() && matches!(name.as_str(), "ln" | "acosh") {
                    return Ok(integer(0));
                }
            }
            if name == "ln" && arg == Expr::Symbol("e".into()) {
                return Ok(integer(1));
            }
            if matches!(name.as_str(), "sin" | "cos" | "tan") {
                let (scale, base) = coefficient(arg.clone());
                if base == Expr::Symbol("pi".into()) {
                    let sine = exact_sin_pi(&scale, budget)?;
                    let cosine = exact_sin_pi(&(&scale + Rat::new(1.into(), 2.into())), budget)?;
                    match name.as_str() {
                        "sin" => {
                            if let Some(s) = sine {
                                return Ok(s);
                            }
                        }
                        "cos" => {
                            if let Some(c) = cosine {
                                return Ok(c);
                            }
                        }
                        "tan" => {
                            if cosine.as_ref().is_some_and(zero) {
                                return Err(CalcError::math(
                                    "tangent is undefined at an odd multiple of pi/2",
                                ));
                            }
                            if let (Some(s), Some(c)) = (sine, cosine) {
                                return simplify(
                                    Expr::Mul(vec![s, raw_power(c, integer(-1))]),
                                    budget,
                                );
                            }
                        }
                        _ => {}
                    }
                }
            }
            Ok(function(&name, arg))
        }
    }
}

fn coefficient(e: Expr) -> (Rat, Expr) {
    match e {
        Expr::Number(n) => (n, integer(1)),
        Expr::Mul(mut xs) if matches!(xs.first(), Some(Expr::Number(_))) => {
            let Expr::Number(n) = xs.remove(0) else {
                unreachable!()
            };
            (
                n,
                if xs.len() == 1 {
                    xs.remove(0)
                } else {
                    Expr::Mul(xs)
                },
            )
        }
        e => (Rat::one(), e),
    }
}
fn with_coefficient(c: Rat, base: Expr) -> Expr {
    if one(&base) {
        Expr::Number(c)
    } else if c.is_one() {
        base
    } else if let Expr::Mul(mut xs) = base {
        xs.insert(0, Expr::Number(c));
        Expr::Mul(xs)
    } else {
        Expr::Mul(vec![Expr::Number(c), base])
    }
}

fn exact_sqrt(n: &Rat) -> Option<Rat> {
    if n.is_negative() {
        return None;
    }
    let a = n.numer().sqrt();
    let b = n.denom().sqrt();
    if &a * &a == *n.numer() && &b * &b == *n.denom() {
        Some(Rat::new(a, b))
    } else {
        None
    }
}

fn exact_sin_pi(scale: &Rat, budget: &mut Budget) -> Result<Option<Expr>> {
    let period = scale.denom() * 2;
    let numerator = ((scale.numer() % &period) + &period) % &period;
    let mut angle = Rat::new(numerator, scale.denom().clone());
    let sign = if angle >= Rat::one() {
        angle -= Rat::one();
        -1
    } else {
        1
    };
    if angle > Rat::new(1.into(), 2.into()) {
        angle = Rat::one() - angle;
    }
    let result = if angle.is_zero() {
        integer(0)
    } else if angle == Rat::new(1.into(), 6.into()) {
        rational(1, 2)
    } else if angle == Rat::new(1.into(), 4.into()) {
        Expr::Mul(vec![rational(1, 2), function("sqrt", integer(2))])
    } else if angle == Rat::new(1.into(), 3.into()) {
        Expr::Mul(vec![rational(1, 2), function("sqrt", integer(3))])
    } else if angle == Rat::new(1.into(), 2.into()) {
        integer(1)
    } else {
        return Ok(None);
    };
    Ok(Some(simplify(
        Expr::Mul(vec![integer(sign), result]),
        budget,
    )?))
}

fn collect_constraints(e: &Expr, out: &mut BTreeSet<String>, budget: &mut Budget) -> Result<()> {
    budget.tick()?;
    match e {
        Expr::Add(xs) | Expr::Mul(xs) => {
            for x in xs {
                collect_constraints(x, out, budget)?;
            }
        }
        Expr::Pow(a, b) => {
            collect_constraints(a, out, budget)?;
            collect_constraints(b, out, budget)?;
            // Normalize children before checking domains, but retain their own
            // restrictions before cancellation. For example x^(1-2)*x is 1
            // only for x != 0, even though the exponent was not a literal -1.
            let a = simplify(*a.clone(), budget)?;
            let b = simplify(*b.clone(), budget)?;
            if matches!(&b,Expr::Number(n) if !n.is_positive())
                && !matches!(&a,Expr::Number(n) if !n.is_zero())
            {
                out.insert(format!("{} != 0", render(&a)));
            }
            if !matches!(&a, Expr::Number(_)) && matches!(&b,Expr::Number(n) if !n.is_integer()) {
                out.insert(format!("{} >= 0", render(&a)));
            }
            if !matches!(&a, Expr::Number(_)) && !matches!(&b, Expr::Number(_)) {
                out.insert(format!("{} > 0", render(&a)));
            }
        }
        Expr::Function(f, a) => {
            collect_constraints(a, out, budget)?;
            let a = simplify(*a.clone(), budget)?;
            if !matches!(&a, Expr::Number(_)) {
                match f.as_str() {
                    "ln" | "log" => {
                        out.insert(format!("{} > 0", render(&a)));
                    }
                    "sqrt" => {
                        out.insert(format!("{} >= 0", render(&a)));
                    }
                    "asin" | "acos" => {
                        out.insert(format!("-1 <= {} <= 1", render(&a)));
                    }
                    "acosh" => {
                        out.insert(format!("{} >= 1", render(&a)));
                    }
                    "atanh" => {
                        out.insert(format!("-1 < {} < 1", render(&a)));
                    }
                    "tan" => {
                        out.insert(format!("cos({}) != 0", render(&a)));
                    }
                    _ => {}
                }
            }
        }
        _ => {}
    }
    Ok(())
}

fn derivative_constraints(e: &Expr, var: &str, out: &mut BTreeSet<String>) {
    match e {
        Expr::Function(f, a) => {
            if f == "abs" && contains(a, var) {
                out.insert(format!("{} != 0", render(a)));
            }
            derivative_constraints(a, var, out);
        }
        Expr::Add(xs) | Expr::Mul(xs) => {
            for x in xs {
                derivative_constraints(x, var, out);
            }
        }
        Expr::Pow(a, b) => {
            if contains(b, var) {
                out.insert(format!("{} > 0", render(a)));
            }
            derivative_constraints(a, var, out);
            derivative_constraints(b, var, out);
        }
        _ => {}
    }
}

fn render(e: &Expr) -> String {
    render_at(e, 0)
}
fn render_at(e: &Expr, parent: u8) -> String {
    let (text, precedence) = match e {
        Expr::Number(n) => (
            if n.is_integer() {
                n.to_integer().to_string()
            } else {
                format!("{}/{}", n.numer(), n.denom())
            },
            if n.is_negative() {
                25
            } else if n.is_integer() {
                40
            } else {
                20
            },
        ),
        Expr::Symbol(s) => (s.clone(), 40),
        Expr::Add(xs) => {
            let mut text = String::new();
            for (i, x) in xs.iter().enumerate() {
                let (c, base) = coefficient(x.clone());
                let negative = c.is_negative();
                let term = with_coefficient(c.abs(), base);
                if i == 0 {
                    if negative {
                        text.push('-');
                    }
                } else {
                    text.push_str(if negative { " - " } else { " + " });
                }
                text.push_str(&render_at(&term, 10));
            }
            (text, 10)
        }
        Expr::Mul(xs) => {
            let mut pieces = Vec::new();
            let mut negative = false;
            for (i, x) in xs.iter().enumerate() {
                if i == 0 && *x == integer(-1) {
                    negative = true;
                } else {
                    pieces.push(render_at(x, 20));
                }
            }
            (
                format!("{}{}", if negative { "-" } else { "" }, pieces.join("*")),
                20,
            )
        }
        Expr::Pow(a, b) => {
            // Fend treats sin(x)^2 as sin(x^2), so make a powered function
            // result unambiguous for both the symbolic and numerical dialects.
            let base = if matches!(&**a, Expr::Function(_, _)) {
                format!("({})", render(a))
            } else {
                render_at(a, 31)
            };
            (format!("{base}^{}", render_at(b, 30)), 30)
        }
        Expr::Function(f, a) => (format!("{f}({})", render(a)), 40),
    };
    if precedence < parent {
        format!("({text})")
    } else {
        text
    }
}

fn contains(e: &Expr, var: &str) -> bool {
    match e {
        Expr::Symbol(s) => s == var,
        Expr::Add(xs) | Expr::Mul(xs) => xs.iter().any(|x| contains(x, var)),
        Expr::Pow(a, b) => contains(a, var) || contains(b, var),
        Expr::Function(_, a) => contains(a, var),
        _ => false,
    }
}

fn derivative(e: &Expr, var: &str, budget: &mut Budget) -> Result<Expr> {
    budget.tick()?;
    if !contains(e, var) {
        return Ok(integer(0));
    }
    Ok(match e {
        Expr::Number(_) => integer(0),
        Expr::Symbol(_) => integer(1),
        Expr::Add(xs) => Expr::Add(
            xs.iter()
                .map(|x| derivative(x, var, budget))
                .collect::<Result<_>>()?,
        ),
        Expr::Mul(xs) => {
            let mut terms = Vec::new();
            if xs.len() * xs.len() > MAX_NODES {
                return Err(CalcError::limit("product rule result is too large"));
            }
            for (i, x) in xs.iter().enumerate() {
                let d = derivative(x, var, budget)?;
                if !zero(&d) {
                    let mut factors = xs.clone();
                    factors[i] = d;
                    terms.push(Expr::Mul(factors));
                }
            }
            Expr::Add(terms)
        }
        Expr::Pow(a, b) => {
            let da = derivative(a, var, budget)?;
            if !contains(b, var) {
                Expr::Mul(vec![
                    *b.clone(),
                    raw_power(*a.clone(), Expr::Add(vec![*b.clone(), integer(-1)])),
                    da,
                ])
            } else {
                // Real variable powers use a positive base; the caller exposes
                // that domain condition in its output.
                Expr::Mul(vec![
                    e.clone(),
                    Expr::Add(vec![
                        Expr::Mul(vec![
                            derivative(b, var, budget)?,
                            function("ln", *a.clone()),
                        ]),
                        Expr::Mul(vec![*b.clone(), da, raw_power(*a.clone(), integer(-1))]),
                    ]),
                ])
            }
        }
        Expr::Function(f, a) => {
            let a = *a.clone();
            let outer = match f.as_str() {
                "sin" => function("cos", a.clone()),
                "cos" => Expr::Mul(vec![integer(-1), function("sin", a.clone())]),
                "tan" => raw_power(function("cos", a.clone()), integer(-2)),
                "ln" => raw_power(a.clone(), integer(-1)),
                "exp" => function("exp", a.clone()),
                "sqrt" => Expr::Mul(vec![
                    rational(1, 2),
                    raw_power(function("sqrt", a.clone()), integer(-1)),
                ]),
                "asin" | "acos" => Expr::Mul(vec![
                    integer(if f == "acos" { -1 } else { 1 }),
                    raw_power(
                        function(
                            "sqrt",
                            Expr::Add(vec![
                                integer(1),
                                Expr::Mul(vec![integer(-1), raw_power(a.clone(), integer(2))]),
                            ]),
                        ),
                        integer(-1),
                    ),
                ]),
                "atan" => raw_power(
                    Expr::Add(vec![integer(1), raw_power(a.clone(), integer(2))]),
                    integer(-1),
                ),
                "sinh" => function("cosh", a.clone()),
                "cosh" => function("sinh", a.clone()),
                "tanh" => raw_power(function("cosh", a.clone()), integer(-2)),
                "asinh" => raw_power(
                    function(
                        "sqrt",
                        Expr::Add(vec![integer(1), raw_power(a.clone(), integer(2))]),
                    ),
                    integer(-1),
                ),
                "acosh" => Expr::Mul(vec![
                    raw_power(
                        function("sqrt", Expr::Add(vec![a.clone(), integer(-1)])),
                        integer(-1),
                    ),
                    raw_power(
                        function("sqrt", Expr::Add(vec![a.clone(), integer(1)])),
                        integer(-1),
                    ),
                ]),
                "atanh" => raw_power(
                    Expr::Add(vec![
                        integer(1),
                        Expr::Mul(vec![integer(-1), raw_power(a.clone(), integer(2))]),
                    ]),
                    integer(-1),
                ),
                "abs" => Expr::Mul(vec![
                    a.clone(),
                    raw_power(function("abs", a.clone()), integer(-1)),
                ]),
                _ => {
                    return Err(unsupported(format!(
                        "symbolic differentiation of {f} is unsupported"
                    )));
                }
            };
            Expr::Mul(vec![outer, derivative(&a, var, budget)?])
        }
    })
}

fn substitute(e: Expr, var: &str, value: &Expr, budget: &mut Budget) -> Result<Expr> {
    budget.tick()?;
    Ok(match e {
        Expr::Symbol(s) if s == var => value.clone(),
        Expr::Add(xs) => Expr::Add(
            xs.into_iter()
                .map(|x| substitute(x, var, value, budget))
                .collect::<Result<_>>()?,
        ),
        Expr::Mul(xs) => Expr::Mul(
            xs.into_iter()
                .map(|x| substitute(x, var, value, budget))
                .collect::<Result<_>>()?,
        ),
        Expr::Pow(a, b) => raw_power(
            substitute(*a, var, value, budget)?,
            substitute(*b, var, value, budget)?,
        ),
        Expr::Function(f, a) => function(&f, substitute(*a, var, value, budget)?),
        x => x,
    })
}

fn expand(e: Expr, budget: &mut Budget) -> Result<Expr> {
    budget.tick()?;
    let out = match e {
        Expr::Add(xs) => Expr::Add(
            xs.into_iter()
                .map(|x| expand(x, budget))
                .collect::<Result<_>>()?,
        ),
        Expr::Mul(xs) => {
            let mut terms = vec![integer(1)];
            for x in xs {
                let x = expand(x, budget)?;
                let additions = if let Expr::Add(xs) = x { xs } else { vec![x] };
                if terms.len().saturating_mul(additions.len()) > MAX_TERMS {
                    return Err(CalcError::limit("expansion exceeds 512 intermediate terms"));
                }
                let mut next = Vec::new();
                for a in &terms {
                    for b in &additions {
                        next.push(simplify(Expr::Mul(vec![a.clone(), b.clone()]), budget)?);
                    }
                }
                let merged = simplify(Expr::Add(next), budget)?;
                terms = if let Expr::Add(xs) = merged {
                    xs
                } else {
                    vec![merged]
                };
            }
            Expr::Add(terms)
        }
        Expr::Pow(a, b) => {
            let base = expand(*a, budget)?;
            if let Some(n) = as_integer(&b) {
                if n >= 0 && matches!(base, Expr::Add(_)) {
                    if n > MAX_DEGREE as i64 {
                        return Err(CalcError::limit("expanded polynomial degree exceeds 64"));
                    }
                    let mut out = integer(1);
                    for _ in 0..n {
                        out = expand(Expr::Mul(vec![out, base.clone()]), budget)?;
                    }
                    out
                } else {
                    raw_power(base, *b)
                }
            } else {
                raw_power(base, *b)
            }
        }
        Expr::Function(f, a) => function(&f, expand(*a, budget)?),
        e => e,
    };
    let out = simplify(out, budget)?;
    budget.size(&out)?;
    Ok(out)
}

fn integrate(e: &Expr, var: &str, budget: &mut Budget) -> Result<Expr> {
    budget.tick()?;
    let x = Expr::Symbol(var.into());
    if !contains(e, var) {
        return Ok(Expr::Mul(vec![e.clone(), x]));
    }
    match e {
        Expr::Symbol(s) if s == var => {
            Ok(Expr::Mul(vec![rational(1, 2), raw_power(x, integer(2))]))
        }
        Expr::Add(xs) => Ok(Expr::Add(
            xs.iter()
                .map(|e| integrate(e, var, budget))
                .collect::<Result<_>>()?,
        )),
        Expr::Mul(xs) => {
            let (constant, varying): (Vec<_>, Vec<_>) =
                xs.iter().cloned().partition(|e| !contains(e, var));
            if !constant.is_empty() {
                let inner = if varying.len() == 1 {
                    varying[0].clone()
                } else {
                    Expr::Mul(varying)
                };
                let mut factors = constant;
                factors.push(integrate(&inner, var, budget)?);
                return Ok(Expr::Mul(factors));
            }
            if let Some(result) = integrate_polynomial_product(xs, var, budget) {
                return result;
            }
            // Polynomial products are expanded first. Reject a non-polynomial
            // product explicitly rather than retrying the same representation.
            if to_polynomial(e, var, budget).is_ok() {
                let expanded = expand(e.clone(), budget)?;
                if expanded != *e {
                    return integrate(&expanded, var, budget);
                }
            }
            Err(unsupported(
                "this product has no supported symbolic integral; use integrate(expr, x, lower, upper) for a definite numerical integral",
            ))
        }
        Expr::Pow(base, power) if matches!(&**power, Expr::Number(_)) => {
            let Expr::Number(n) = &**power else {
                unreachable!()
            };
            if n == &Rat::from_integer((-1).into()) {
                if let Expr::Function(f, a) = &**base
                    && f == "sqrt"
                {
                    return integrate(&raw_power(*a.clone(), rational(-1, 2)), var, budget);
                }
                if let Ok(poly) = to_polynomial(base, var, budget)
                    && poly.len() == 3
                {
                    return integrate_reciprocal_quadratic(&poly, var, budget);
                }
            }
            if n == &Rat::new((-1).into(), 2.into())
                && let Ok(poly) = to_polynomial(base, var, budget)
                && poly.len() == 3
                && poly[1].is_zero()
                && poly[0].is_positive()
            {
                let b = poly[2].clone();
                let scale = sqrt_expr(checked(b.abs() / &poly[0])?, budget)?;
                return Ok(Expr::Mul(vec![
                    raw_power(sqrt_expr(b.abs(), budget)?, integer(-1)),
                    function(
                        if b.is_positive() { "asinh" } else { "asin" },
                        Expr::Mul(vec![x, scale]),
                    ),
                ]));
            }
            let slope = linear_slope(base, var, budget)?;
            if n == &Rat::from_integer((-1).into()) {
                Ok(Expr::Mul(vec![
                    raw_power(slope, integer(-1)),
                    function("ln", function("abs", *base.clone())),
                ]))
            } else {
                let next = checked(n + Rat::one())?;
                Ok(Expr::Mul(vec![
                    Expr::Number(next.recip()),
                    raw_power(slope, integer(-1)),
                    raw_power(*base.clone(), Expr::Number(next)),
                ]))
            }
        }
        Expr::Pow(base, power) if !contains(base, var) => {
            if !matches!(&**base,Expr::Number(n) if n.is_positive())
                && **base != Expr::Symbol("e".into())
            {
                return Err(unsupported(
                    "exponential integration requires a known positive base",
                ));
            }
            let slope = linear_slope(power, var, budget)?;
            Ok(Expr::Mul(vec![
                e.clone(),
                raw_power(slope, integer(-1)),
                raw_power(function("ln", *base.clone()), integer(-1)),
            ]))
        }
        Expr::Function(f, base) => {
            let slope = linear_slope(base, var, budget)?;
            let base = *base.clone();
            let primitive = match f.as_str() {
                "sin" => Expr::Mul(vec![integer(-1), function("cos", base)]),
                "cos" => function("sin", base),
                "tan" => Expr::Mul(vec![
                    integer(-1),
                    function("ln", function("abs", function("cos", base))),
                ]),
                "exp" => function("exp", base),
                "sinh" => function("cosh", base),
                "cosh" => function("sinh", base),
                "tanh" => function("ln", function("cosh", base)),
                "ln" => Expr::Add(vec![
                    Expr::Mul(vec![base.clone(), function("ln", base.clone())]),
                    Expr::Mul(vec![integer(-1), base]),
                ]),
                "sqrt" => Expr::Mul(vec![rational(2, 3), raw_power(base, rational(3, 2))]),
                "abs" => Expr::Mul(vec![rational(1, 2), base.clone(), function("abs", base)]),
                _ => {
                    return Err(unsupported(format!(
                        "symbolic integration of {f} is unsupported"
                    )));
                }
            };
            Ok(Expr::Mul(vec![primitive, raw_power(slope, integer(-1))]))
        }
        _ => Err(unsupported(
            "no supported elementary antiderivative; use four-argument integrate for numerical integration",
        )),
    }
}

fn sqrt_expr(n: Rat, budget: &mut Budget) -> Result<Expr> {
    simplify(function("sqrt", Expr::Number(n)), budget)
}

fn integrate_reciprocal_quadratic(
    poly: &Polynomial,
    var: &str,
    budget: &mut Budget,
) -> Result<Expr> {
    let c = &poly[0];
    let b = &poly[1];
    let a = &poly[2];
    let k = checked(Rat::from_integer(4.into()) * a * c - b * b)?;
    let u = simplify(
        Expr::Add(vec![
            Expr::Mul(vec![
                Expr::Number(checked(Rat::from_integer(2.into()) * a)?),
                Expr::Symbol(var.into()),
            ]),
            Expr::Number(b.clone()),
        ]),
        budget,
    )?;
    if k.is_zero() {
        return Ok(Expr::Mul(vec![integer(-2), raw_power(u, integer(-1))]));
    }
    let s = sqrt_expr(k.abs(), budget)?;
    if k.is_positive() {
        Ok(Expr::Mul(vec![
            integer(2),
            raw_power(s.clone(), integer(-1)),
            function("atan", Expr::Mul(vec![u, raw_power(s, integer(-1))])),
        ]))
    } else {
        let ratio = Expr::Mul(vec![
            Expr::Add(vec![u.clone(), Expr::Mul(vec![integer(-1), s.clone()])]),
            raw_power(Expr::Add(vec![u, s.clone()]), integer(-1)),
        ]);
        Ok(Expr::Mul(vec![
            raw_power(s, integer(-1)),
            function("ln", function("abs", ratio)),
        ]))
    }
}

fn integrate_polynomial_product(
    xs: &[Expr],
    var: &str,
    budget: &mut Budget,
) -> Option<Result<Expr>> {
    let candidates: Vec<_> = xs.iter().enumerate().filter(|(_,e)| matches!(e,Expr::Function(f,_) if matches!(f.as_str(),"sin"|"cos"|"exp"|"sinh"|"cosh"))).collect();
    if candidates.len() != 1 {
        return None;
    }
    let (index, f) = candidates[0];
    let polynomial = Expr::Mul(
        xs.iter()
            .enumerate()
            .filter(|(i, _)| *i != index)
            .map(|(_, e)| e.clone())
            .collect(),
    );
    let poly = match to_polynomial(&polynomial, var, budget) {
        Ok(p) => p,
        Err(e) if e.code == "resource_limit" => return Some(Err(e)),
        Err(_) => return None,
    };
    Some((|| {
        if poly.len() > 17 {
            return Err(CalcError::limit(
                "integration by parts supports polynomial degree up to 16",
            ));
        }
        let mut u = simplify(polynomial, budget)?;
        let mut v = f.clone();
        let mut terms = Vec::new();
        for order in 0..poly.len() {
            budget.tick()?;
            v = simplify(integrate(&v, var, budget)?, budget)?;
            terms.push(Expr::Mul(vec![
                integer(if order % 2 == 0 { 1 } else { -1 }),
                u.clone(),
                v.clone(),
            ]));
            u = simplify(derivative(&u, var, budget)?, budget)?;
            if zero(&u) {
                break;
            }
        }
        let result = simplify(Expr::Add(terms), budget)?;
        budget.size(&result)?;
        Ok(result)
    })())
}

fn linear_slope(e: &Expr, var: &str, budget: &mut Budget) -> Result<Expr> {
    let slope = simplify(derivative(e, var, budget)?, budget)?;
    if zero(&slope) || contains(&slope, var) {
        return Err(unsupported(
            "symbolic integration supports a linear inner argument here",
        ));
    }
    // A symbolic slope might be zero. Only a known nonzero rational is accepted.
    if !matches!(slope, Expr::Number(_)) {
        return Err(unsupported(
            "symbolic integration requires a known nonzero numerical slope",
        ));
    }
    Ok(slope)
}

type Polynomial = Vec<Rat>;
fn trim_poly(p: &mut Polynomial) {
    while p.len() > 1 && p.last().is_some_and(Zero::is_zero) {
        p.pop();
    }
}
fn poly_add(a: Polynomial, b: Polynomial) -> Result<Polynomial> {
    let mut out = vec![Rat::zero(); a.len().max(b.len())];
    for (i, n) in a.into_iter().enumerate() {
        out[i] = n;
    }
    for (i, n) in b.into_iter().enumerate() {
        out[i] = checked(out[i].clone() + n)?;
    }
    trim_poly(&mut out);
    Ok(out)
}
fn poly_mul(a: &Polynomial, b: &Polynomial, budget: &mut Budget) -> Result<Polynomial> {
    if a.len() + b.len() - 2 > MAX_DEGREE {
        return Err(CalcError::limit("polynomial degree exceeds 64"));
    }
    let mut out = vec![Rat::zero(); a.len() + b.len() - 1];
    for (i, x) in a.iter().enumerate() {
        for (j, y) in b.iter().enumerate() {
            budget.tick()?;
            out[i + j] = checked(out[i + j].clone() + x * y)?;
        }
    }
    trim_poly(&mut out);
    Ok(out)
}
fn to_polynomial(e: &Expr, var: &str, budget: &mut Budget) -> Result<Polynomial> {
    budget.tick()?;
    match e {
        Expr::Number(n) => Ok(vec![n.clone()]),
        Expr::Symbol(s) if s == var => Ok(vec![Rat::zero(), Rat::one()]),
        Expr::Add(xs) => {
            let mut out = vec![Rat::zero()];
            for x in xs {
                out = poly_add(out, to_polynomial(x, var, budget)?)?;
            }
            Ok(out)
        }
        Expr::Mul(xs) => {
            let mut out = vec![Rat::one()];
            for x in xs {
                out = poly_mul(&out, &to_polynomial(x, var, budget)?, budget)?;
            }
            Ok(out)
        }
        Expr::Pow(a, b) => {
            let n = as_integer(b)
                .filter(|n| *n >= 0 && *n <= MAX_DEGREE as i64)
                .ok_or_else(|| {
                    unsupported("polynomial requires nonnegative integer powers up to 64")
                })?;
            let base = to_polynomial(a, var, budget)?;
            let mut out = vec![Rat::one()];
            for _ in 0..n {
                out = poly_mul(&out, &base, budget)?;
            }
            Ok(out)
        }
        _ => Err(unsupported(
            "operation supports univariate polynomials with rational coefficients",
        )),
    }
}
fn polynomial_expr(poly: &Polynomial, var: &str, budget: &mut Budget) -> Result<Expr> {
    let mut terms = Vec::new();
    for (degree, n) in poly.iter().enumerate() {
        if !n.is_zero() {
            let base = if degree == 0 {
                integer(1)
            } else if degree == 1 {
                Expr::Symbol(var.into())
            } else {
                raw_power(Expr::Symbol(var.into()), integer(degree as i64))
            };
            terms.push(with_coefficient(n.clone(), base));
        }
    }
    simplify(Expr::Add(terms), budget)
}

fn symbols(e: &Expr, set: &mut BTreeSet<String>) {
    match e {
        Expr::Symbol(s) if !matches!(s.as_str(), "pi" | "e" | "i") => {
            set.insert(s.clone());
        }
        Expr::Add(xs) | Expr::Mul(xs) => {
            for x in xs {
                symbols(x, set);
            }
        }
        Expr::Pow(a, b) => {
            symbols(a, set);
            symbols(b, set);
        }
        Expr::Function(_, a) => symbols(a, set),
        _ => {}
    }
}

fn factor(e: Expr, budget: &mut Budget) -> Result<Expr> {
    let mut vars = BTreeSet::new();
    symbols(&e, &mut vars);
    if vars.is_empty() {
        return Ok(e);
    }
    if vars.len() != 1 {
        return Err(unsupported(
            "factor currently supports one variable and rational coefficients",
        ));
    }
    let var = vars.into_iter().next().unwrap();
    let mut poly = to_polynomial(&e, &var, budget)?;
    if poly.len() <= 2 {
        return polynomial_expr(&poly, &var, budget);
    }
    let mut factors = Vec::new();
    let leading = poly.last().unwrap().clone();
    for n in &mut poly {
        *n = checked(n.clone() / &leading)?;
    }
    if !leading.is_one() {
        factors.push(Expr::Number(leading));
    }
    while poly.len() > 1 {
        let Some(root) = rational_root(&poly, budget)? else {
            break;
        };
        factors.push(simplify(
            Expr::Add(vec![Expr::Symbol(var.clone()), Expr::Number(-root.clone())]),
            budget,
        )?);
        poly = divide_root(&poly, &root, budget)?;
    }
    if poly.len() > 1 {
        factors.push(polynomial_expr(&poly, &var, budget)?);
    }
    if factors.is_empty() {
        Ok(integer(1))
    } else {
        simplify(Expr::Mul(factors), budget)
    }
}

fn gcd(mut a: BigInt, mut b: BigInt) -> BigInt {
    while !b.is_zero() {
        let r = &a % &b;
        a = b;
        b = r;
    }
    a.abs()
}
fn divisors(n: &BigInt, budget: &mut Budget) -> Result<Vec<BigInt>> {
    let n = n.abs();
    let mut out = BTreeSet::new();
    out.insert(BigInt::one());
    out.insert(n.clone());
    // Trial divisors are bounded. Missing a factor is a supported-operation
    // limitation, not evidence that the polynomial is irreducible.
    for d in 2u32..=4096 {
        budget.tick()?;
        let d = BigInt::from(d);
        if &d * &d > n {
            break;
        }
        if (&n % &d).is_zero() {
            out.insert(d.clone());
            out.insert(&n / d);
        }
        if out.len() > 128 {
            return Err(CalcError::limit("too many rational root candidates"));
        }
    }
    Ok(out.into_iter().collect())
}
fn rational_root(poly: &Polynomial, budget: &mut Budget) -> Result<Option<Rat>> {
    if poly[0].is_zero() {
        return Ok(Some(Rat::zero()));
    }
    if poly.len() == 2 {
        return Ok(Some(-&poly[0] / &poly[1]));
    }
    if poly.len() == 3 {
        let d = &poly[1] * &poly[1] - Rat::from_integer(4.into()) * &poly[0] * &poly[2];
        if let Some(s) = exact_sqrt(&d) {
            return Ok(Some(
                (-&poly[1] + s) / (Rat::from_integer(2.into()) * &poly[2]),
            ));
        }
    }
    let mut denominator = BigInt::one();
    for n in poly {
        denominator = &denominator / gcd(denominator.clone(), n.denom().clone()) * n.denom();
        if denominator.bits() > MAX_BITS {
            return Err(CalcError::limit("polynomial common denominator too large"));
        }
    }
    let first = poly[0].numer() * (&denominator / poly[0].denom());
    let last = poly.last().unwrap().numer() * (&denominator / poly.last().unwrap().denom());
    let ps = divisors(&first, budget)?;
    let qs = divisors(&last, budget)?;
    if ps.len().saturating_mul(qs.len()) > 4096 {
        return Err(CalcError::limit("too many rational root candidates"));
    }
    for p in ps {
        for q in &qs {
            for sign in [-1, 1] {
                budget.tick()?;
                let r = Rat::new(&p * sign, q.clone());
                if poly_value(poly, &r, budget)?.is_zero() {
                    return Ok(Some(r));
                }
            }
        }
    }
    Ok(None)
}
fn poly_value(poly: &Polynomial, x: &Rat, budget: &mut Budget) -> Result<Rat> {
    let mut value = Rat::zero();
    for c in poly.iter().rev() {
        budget.tick()?;
        value = checked(value * x + c)?;
    }
    Ok(value)
}
fn divide_root(poly: &Polynomial, root: &Rat, budget: &mut Budget) -> Result<Polynomial> {
    let mut out = vec![Rat::zero(); poly.len() - 1];
    out[poly.len() - 2] = poly.last().unwrap().clone();
    for i in (1..poly.len() - 1).rev() {
        budget.tick()?;
        out[i - 1] = checked(&poly[i] + root * &out[i])?;
    }
    if !checked(&poly[0] + root * &out[0])?.is_zero() {
        return Err(CalcError::math("internal exact polynomial division failed"));
    }
    trim_poly(&mut out);
    Ok(out)
}

fn solve(s: &str, var: &str, budget: &mut Budget) -> Result<Calculation> {
    let equals: Vec<_> = s.match_indices('=').collect();
    if equals.len() > 1 {
        return Err(CalcError::invalid(
            "solve accepts one equation, using a single =",
        ));
    }
    let original = if let Some((i, _)) = equals.first() {
        Expr::Add(vec![
            parse(&s[..*i], budget)?,
            Expr::Mul(vec![integer(-1), parse(&s[i + 1..], budget)?]),
        ])
    } else {
        parse(s, budget)?
    };
    let mut constraints = BTreeSet::new();
    collect_constraints(&original, &mut constraints, budget)?;
    let expression = simplify(original, budget)?;
    // Rational equations need pole filtering, not just numerator roots. Avoid
    // presenting unfiltered solutions until that operation is implemented.
    if !constraints.is_empty() {
        return Err(unsupported(
            "solve supports polynomial equations; rational equations and domain restrictions require explicit handling",
        ));
    }
    let mut polynomial = to_polynomial(&expression, var, budget)?;
    if polynomial.len() == 1 {
        let all = polynomial[0].is_zero();
        let mut c = Calculation::text(
            if all {
                format!("all values of {var}")
            } else {
                "no solutions".into()
            },
            "solutions",
            true,
        );
        c.value =
            serde_json::json!({"variable":var,"solutions":[],"all_values":all,"domain":"complex"});
        return Ok(c);
    }
    let mut roots = Vec::<Expr>::new();
    while polynomial.len() > 3 {
        let root = rational_root(&polynomial,budget)?.ok_or_else(|| unsupported("exact solve supports linear/quadratic equations and higher-degree polynomials reducible by bounded rational-root search"))?;
        roots.push(Expr::Number(root.clone()));
        polynomial = divide_root(&polynomial, &root, budget)?;
    }
    if polynomial.len() == 2 {
        roots.push(Expr::Number(-&polynomial[0] / &polynomial[1]));
    }
    if polynomial.len() == 3 {
        let c = &polynomial[0];
        let b = &polynomial[1];
        let a = &polynomial[2];
        let d = checked(b * b - Rat::from_integer(4.into()) * a * c)?;
        let sqrt = if let Some(s) = exact_sqrt(&d.abs()) {
            Expr::Number(s)
        } else {
            function("sqrt", Expr::Number(d.abs()))
        };
        let radical = if d.is_negative() {
            Expr::Mul(vec![sqrt, Expr::Symbol("i".into())])
        } else {
            sqrt
        };
        for sign in [-1, 1] {
            roots.push(simplify(
                Expr::Mul(vec![
                    Expr::Number((Rat::from_integer(2.into()) * a).recip()),
                    Expr::Add(vec![
                        Expr::Number(-b.clone()),
                        Expr::Mul(vec![integer(sign), radical.clone()]),
                    ]),
                ]),
                budget,
            )?);
        }
    }
    roots.sort();
    roots.dedup();
    let texts: Vec<_> = roots.iter().map(render).collect();
    let mut out = Calculation::text(format!("{var} = {}", texts.join(", ")), "solutions", true);
    out.value =
        serde_json::json!({"variable":var,"solutions":texts,"all_values":false,"domain":"complex"});
    Ok(out)
}
