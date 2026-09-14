use super::*;

pub(super) fn known(name: &str) -> bool {
    matches!(
        name,
        "integrate" | "integral" | "root" | "derivative_at" | "summation" | "product_range"
    )
}

pub(super) fn evaluate(
    name: &str,
    args: &[&str],
    deadline: Instant,
    depth: usize,
) -> Result<Calculation, CalcError> {
    arity(args, if name == "derivative_at" { 3 } else { 4 }, name)?;
    let variable = args[1].trim();
    if variable.is_empty()
        || variable.len() > 32
        || !variable
            .chars()
            .all(|c| c.is_ascii_alphabetic() || c == '_')
    {
        return Err(CalcError::invalid("Use a simple variable name such as x"));
    }
    let expression = Parser::parse(args[0], variable)?;
    let a = scalar(args[2], deadline, depth)?;
    if name == "derivative_at" {
        // Forward differentiation of the parsed function avoids subtracting
        // nearly equal values and has explicit checks at nonsmooth points.
        expression.eval(a)?;
        let derivative = expression.derivative(a)?;
        check_time(deadline)?;
        return number(derivative);
    }
    let b = scalar(args[3], deadline, depth)?;
    if matches!(name, "summation" | "product_range") {
        extensions::signed_integer(args[2], deadline, depth)?;
        extensions::signed_integer(args[3], deadline, depth)?;
        if a.fract() != 0.0
            || b.fract() != 0.0
            || a.abs() > 9_007_199_254_740_991.0
            || b.abs() > 9_007_199_254_740_991.0
            || b < a
        {
            return Err(CalcError::math(
                "Finite sums/products require integer bounds with start <= end",
            ));
        }
        if b - a > 100_000.0 {
            return Err(CalcError::limit(
                "Finite sums/products are limited to 100001 terms",
            ));
        }
        let mut result = if name == "summation" { 0.0 } else { 1.0 };
        let mut correction = 0.0;
        for index in 0..=(b - a) as u64 {
            check_time(deadline)?;
            let value = expression.eval(a + index as f64)?;
            if name == "summation" {
                let adjusted = value - correction;
                let next = result + adjusted;
                correction = (next - result) - adjusted;
                result = next;
            } else {
                result *= value;
            }
            finite(result)?;
        }
        return number(result);
    }
    if name == "root" {
        return number(root(&expression, a, b, deadline)?);
    }
    if a == b {
        return number(0.0);
    }
    let (lo, hi, sign) = if a < b { (a, b, 1.0) } else { (b, a, -1.0) };
    if !((hi - lo).is_finite()) {
        return Err(CalcError::math(
            "Integration interval overflows numeric precision",
        ));
    }
    let mut budget = 65536_usize;
    let mut total = 0.0;
    let mut initial = Vec::new();
    // Seed several intervals to reduce accidental under-sampling of oscillation.
    for i in 0..8 {
        let left = lo + (hi - lo) * (i as f64 / 8.0);
        let right = lo + (hi - lo) * ((i + 1) as f64 / 8.0);
        initial.push((left, right, quadrature(&expression, left, right)?));
    }
    // The absolute tolerance scales with the actual integrand. A fixed 1e-10
    // floor would incorrectly accept tiny, divergent integrals immediately.
    let scale = initial
        .iter()
        .map(|(_, _, (_, _, magnitude))| *magnitude)
        .sum::<f64>();
    finite(scale)?;
    if scale == 0.0 {
        if expression.underflowed_exp(lo, hi)? {
            return Err(CalcError::math(
                "Integrand underflow prevents a reliable estimate; narrow features may have been missed",
            ));
        }
        return number(0.0);
    }
    for (left, right, _) in initial {
        total += adaptive(
            &expression,
            left,
            right,
            scale * 1e-10 / 8.0,
            24,
            &mut budget,
            deadline,
        )?;
    }
    number(sign * total)
}

fn root(f: &Node, mut a: f64, mut b: f64, deadline: Instant) -> Result<f64, CalcError> {
    if a >= b {
        return Err(CalcError::math("Root finding requires lower < upper"));
    }
    let mut fa = f.eval(a)?;
    let fb = f.eval(b)?;
    if fa == 0.0 {
        if f.underflowed_exp(a, a)? {
            return Err(CalcError::math(
                "Function underflow prevents validating a root",
            ));
        }
        return Ok(a);
    }
    if fb == 0.0 {
        if f.underflowed_exp(b, b)? {
            return Err(CalcError::math(
                "Function underflow prevents validating a root",
            ));
        }
        return Ok(b);
    }
    if fa.signum() == fb.signum() {
        return Err(CalcError::math("Root interval must bracket a sign change"));
    }
    let scale = fa.abs().min(fb.abs());
    for _ in 0..256 {
        check_time(deadline)?;
        let middle = a * 0.5 + b * 0.5;
        let fm = f.eval(middle)?;
        if fm == 0.0
            || ((fm.abs() / scale) <= 1e-10 && (b - a).abs() <= 1e-12 * (1.0 + middle.abs()))
        {
            if fm == 0.0 && f.underflowed_exp(middle, middle)? {
                return Err(CalcError::math(
                    "Function underflow prevents validating a root",
                ));
            }
            return Ok(middle);
        }
        if middle == a || middle == b || (b - a).abs() <= 2.0 * f64::EPSILON * (1.0 + middle.abs())
        {
            return Err(CalcError::math(
                "Interval shrank without a small residual; the function may be discontinuous",
            ));
        }
        if fm.signum() == fa.signum() {
            a = middle;
            fa = fm;
        } else {
            b = middle;
        }
    }
    Err(CalcError::math("Root finding did not converge"))
}

// Embedded Gauss 7 / Kronrod 15 quadrature. Nonuniform nodes avoid the most
// common aliasing failures of a single equally spaced Simpson estimate.
fn quadrature(f: &Node, a: f64, b: f64) -> Result<(f64, f64, f64), CalcError> {
    const X: [f64; 8] = [
        0.9914553711208126,
        0.9491079123427585,
        0.8648644233597691,
        0.7415311855993945,
        0.5860872354676911,
        0.4058451513773972,
        0.20778495500789848,
        0.0,
    ];
    const K: [f64; 8] = [
        0.022935322010529225,
        0.06309209262997856,
        0.10479001032225019,
        0.14065325971552592,
        0.1690047266392679,
        0.19035057806478542,
        0.20443294007529889,
        0.20948214108472783,
    ];
    const G: [f64; 4] = [
        0.1294849661688697,
        0.27970539148927667,
        0.3818300505051189,
        0.4179591836734694,
    ];
    let mid = a * 0.5 + b * 0.5;
    let half = (b - a) * 0.5;
    let center = f.eval(mid)?;
    let mut kronrod = center * K[7];
    let mut gauss = center * G[3];
    let mut magnitude = center.abs() * K[7];
    for i in 0..7 {
        let left = f.eval(mid - half * X[i])?;
        let right = f.eval(mid + half * X[i])?;
        let pair = left + right;
        magnitude += K[i] * (left.abs() + right.abs());
        kronrod += K[i] * pair;
        if i % 2 == 1 {
            gauss += G[i / 2] * pair;
        }
    }
    Ok((
        finite(kronrod * half)?,
        finite(((kronrod - gauss) * half).abs())?,
        finite(magnitude * half.abs())?,
    ))
}

fn adaptive(
    f: &Node,
    a: f64,
    b: f64,
    tolerance: f64,
    levels: usize,
    budget: &mut usize,
    deadline: Instant,
) -> Result<f64, CalcError> {
    check_time(deadline)?;
    if *budget < 15 {
        return Err(CalcError::limit(
            "Integration exceeded its evaluation budget",
        ));
    }
    *budget -= 15;
    let (estimate, error, _) = quadrature(f, a, b)?;
    if error <= tolerance.max(1e-10 * estimate.abs()) {
        return Ok(estimate);
    }
    if levels == 0 {
        return Err(CalcError::math(
            "Integration did not converge; split singularities or choose a smoother interval",
        ));
    }
    let mid = a * 0.5 + b * 0.5;
    if mid == a || mid == b {
        return Err(CalcError::math(
            "Integration interval reached floating-point resolution",
        ));
    }
    Ok(
        adaptive(f, a, mid, tolerance / 2.0, levels - 1, budget, deadline)?
            + adaptive(f, mid, b, tolerance / 2.0, levels - 1, budget, deadline)?,
    )
}

#[derive(Debug)]
enum Node {
    Constant(f64),
    Variable,
    Neg(Box<Node>),
    Binary(char, Box<Node>, Box<Node>),
    Function(String, Vec<Node>),
}

impl Node {
    fn eval(&self, x: f64) -> Result<f64, CalcError> {
        let value = match self {
            Self::Constant(n) => *n,
            Self::Variable => x,
            Self::Neg(n) => -n.eval(x)?,
            Self::Binary(op, a, b) => {
                let a = a.eval(x)?;
                let b = b.eval(x)?;
                match op {
                    '+' => a + b,
                    '-' => a - b,
                    '*' => a * b,
                    '/' => a / b,
                    '^' => checked_power(a, b)?,
                    '%' => a % b,
                    _ => unreachable!(),
                }
            }
            Self::Function(name, args) => {
                let a = args[0].eval(x)?;
                let b = if args.len() == 2 {
                    args[1].eval(x)?
                } else {
                    0.0
                };
                match name.as_str() {
                    "sin" => a.sin(),
                    "cos" => a.cos(),
                    "tan" => {
                        if a.cos().abs() < 4.0 * f64::EPSILON {
                            return Err(CalcError::math("Tangent is undefined at this angle"));
                        }
                        a.tan()
                    }
                    "asin" => a.asin(),
                    "acos" => a.acos(),
                    "atan" => a.atan(),
                    "sinh" => a.sinh(),
                    "cosh" => a.cosh(),
                    "tanh" => a.tanh(),
                    "asinh" => a.asinh(),
                    "acosh" => a.acosh(),
                    "atanh" => a.atanh(),
                    "sqrt" => a.sqrt(),
                    "cbrt" => a.cbrt(),
                    "abs" => a.abs(),
                    "exp" => a.exp(),
                    "ln" => a.ln(),
                    "log" | "log10" => {
                        if args.len() == 2 {
                            a.log(b)
                        } else {
                            a.log10()
                        }
                    }
                    "log2" => a.log2(),
                    "floor" => a.floor(),
                    "ceil" => a.ceil(),
                    "round" => a.round(),
                    "trunc" => a.trunc(),
                    "sign" => {
                        if a == 0.0 {
                            0.0
                        } else {
                            a.signum()
                        }
                    }
                    "sind" => a.to_radians().sin(),
                    "cosd" => a.to_radians().cos(),
                    "tand" => {
                        if (a.rem_euclid(180.0) - 90.0).abs() < 1e-12 {
                            return Err(CalcError::math("Tangent is undefined at this angle"));
                        }
                        a.to_radians().tan()
                    }
                    "atan2" => {
                        if a == 0.0 && b == 0.0 {
                            return Err(CalcError::math("atan2 is undefined at (0,0)"));
                        }
                        a.atan2(b)
                    }
                    "hypot" => a.hypot(b),
                    "min" => a.min(b),
                    "max" => a.max(b),
                    "pow" => checked_power(a, b)?,
                    "erf" => statrs::function::erf::erf(a),
                    "erfc" => statrs::function::erf::erfc(a),
                    "gamma" => statrs::function::gamma::gamma(a),
                    _ => {
                        return Err(CalcError::invalid(format!(
                            "Unsupported numerical function {name}"
                        )));
                    }
                }
            }
        };
        finite(value)
    }

    fn depends(&self) -> bool {
        match self {
            Self::Constant(_) => false,
            Self::Variable => true,
            Self::Neg(a) => a.depends(),
            Self::Binary(_, a, b) => a.depends() || b.depends(),
            Self::Function(_, args) => args.iter().any(Self::depends),
        }
    }

    fn derivative(&self, x: f64) -> Result<f64, CalcError> {
        if !self.depends() {
            self.eval(x)?;
            return Ok(0.0);
        }
        let derivative = match self {
            Self::Constant(_) => 0.0,
            Self::Variable => 1.0,
            Self::Neg(a) => -a.derivative(x)?,
            Self::Binary(op, a, b) => {
                let av = a.eval(x)?;
                let bv = b.eval(x)?;
                let da = a.derivative(x)?;
                let db = b.derivative(x)?;
                match op {
                    '+' => da + db,
                    '-' => da - db,
                    '*' => da * bv + av * db,
                    '/' => (da - (av / bv) * db) / bv,
                    '^' => power_derivative(av, bv, da, db, b.depends())?,
                    '%' => {
                        if bv == 0.0 || av % bv == 0.0 {
                            return Err(CalcError::math(
                                "Modulo is not differentiable at this point",
                            ));
                        }
                        da - (av / bv).trunc() * db
                    }
                    _ => unreachable!(),
                }
            }
            Self::Function(name, args) => {
                let a = args[0].eval(x)?;
                let da = args[0].derivative(x)?;
                let b = if args.len() > 1 {
                    args[1].eval(x)?
                } else {
                    0.0
                };
                let db = if args.len() > 1 {
                    args[1].derivative(x)?
                } else {
                    0.0
                };
                match name.as_str() {
                    "sin" => a.cos() * da,
                    "cos" => -a.sin() * da,
                    "tan" => da / a.cos().powi(2),
                    "asin" => da / (1.0 - a * a).sqrt(),
                    "acos" => -da / (1.0 - a * a).sqrt(),
                    "atan" => da / (1.0 + a * a),
                    "sinh" => a.cosh() * da,
                    "cosh" => a.sinh() * da,
                    "tanh" => da / a.cosh().powi(2),
                    "asinh" => da / a.hypot(1.0),
                    "acosh" => da / (a - 1.0).sqrt() / (a + 1.0).sqrt(),
                    "atanh" => da / (1.0 - a * a),
                    "sqrt" => da / (2.0 * a.sqrt()),
                    "cbrt" => da / (3.0 * a.cbrt().powi(2)),
                    "abs" => {
                        if a == 0.0 {
                            return Err(CalcError::math(
                                "Absolute value is not differentiable at zero",
                            ));
                        }
                        a.signum() * da
                    }
                    "exp" => a.exp() * da,
                    "ln" => da / a,
                    "log2" => da / (a * std::f64::consts::LN_2),
                    "log" | "log10" => {
                        if args.len() > 1 {
                            da / (a * b.ln()) - a.ln() * db / (b * b.ln().powi(2))
                        } else {
                            da / (a * std::f64::consts::LN_10)
                        }
                    }
                    "floor" | "ceil" | "trunc" => {
                        if a.fract() == 0.0 {
                            return Err(CalcError::math("Rounding function is discontinuous here"));
                        }
                        0.0
                    }
                    "round" => {
                        if a.abs().fract() == 0.5 {
                            return Err(CalcError::math("Rounding function is discontinuous here"));
                        }
                        0.0
                    }
                    "sign" => {
                        if a == 0.0 {
                            return Err(CalcError::math("Sign is discontinuous at zero"));
                        }
                        0.0
                    }
                    "sind" => a.to_radians().cos() * da * std::f64::consts::PI / 180.0,
                    "cosd" => -a.to_radians().sin() * da * std::f64::consts::PI / 180.0,
                    "tand" => da * std::f64::consts::PI / 180.0 / a.to_radians().cos().powi(2),
                    "atan2" => {
                        let scale = a.abs().max(b.abs());
                        let aa = a / scale;
                        let bb = b / scale;
                        (bb * da - aa * db) / (scale * (aa * aa + bb * bb))
                    }
                    "hypot" => {
                        let magnitude = a.hypot(b);
                        if magnitude == 0.0 {
                            return Err(CalcError::math("Norm is not differentiable at zero"));
                        }
                        (a / magnitude) * da + (b / magnitude) * db
                    }
                    "min" | "max" => {
                        if a == b && da != db {
                            return Err(CalcError::math(
                                "Min/max is not differentiable at this crossing",
                            ));
                        }
                        if (a < b) == (name == "min") { da } else { db }
                    }
                    "pow" => power_derivative(a, b, da, db, args[1].depends())?,
                    "erf" => 2.0 / std::f64::consts::PI.sqrt() * (-a * a).exp() * da,
                    "erfc" => -2.0 / std::f64::consts::PI.sqrt() * (-a * a).exp() * da,
                    "gamma" => {
                        statrs::function::gamma::gamma(a) * statrs::function::gamma::digamma(a) * da
                    }
                    _ => return Err(CalcError::invalid("Unsupported derivative function")),
                }
            }
        };
        finite(derivative)
    }

    fn underflowed_exp(&self, a: f64, b: f64) -> Result<bool, CalcError> {
        match self {
            Self::Function(name, args) => {
                if name == "exp"
                    && [a, a * 0.5 + b * 0.5, b]
                        .iter()
                        .any(|x| args[0].eval(*x).is_ok_and(|y| y.exp() == 0.0))
                {
                    return Ok(true);
                }
                for arg in args {
                    if arg.underflowed_exp(a, b)? {
                        return Ok(true);
                    }
                }
                Ok(false)
            }
            Self::Binary(_, left, right) => {
                Ok(left.underflowed_exp(a, b)? || right.underflowed_exp(a, b)?)
            }
            Self::Neg(inner) => inner.underflowed_exp(a, b),
            _ => Ok(false),
        }
    }
}

fn checked_power(base: f64, exponent: f64) -> Result<f64, CalcError> {
    if base == 0.0 && exponent <= 0.0 {
        return Err(CalcError::math(
            "Zero cannot be raised to a zero or negative power",
        ));
    }
    finite(base.powf(exponent))
}

fn power_derivative(
    base: f64,
    exponent: f64,
    da: f64,
    db: f64,
    varying_exponent: bool,
) -> Result<f64, CalcError> {
    checked_power(base, exponent)?;
    if !varying_exponent {
        if exponent == 0.0 {
            return Ok(0.0);
        }
        if exponent == 1.0 {
            return finite(da);
        }
        return finite(exponent * base.powf(exponent - 1.0) * da);
    }
    if base <= 0.0 {
        return Err(CalcError::math(
            "A varying real power requires a positive base",
        ));
    }
    finite(base.powf(exponent) * (db * base.ln() + exponent * da / base))
}

struct Parser<'a> {
    input: &'a [u8],
    position: usize,
    variable: &'a str,
    nodes: usize,
}
impl<'a> Parser<'a> {
    fn parse(input: &'a str, variable: &'a str) -> Result<Node, CalcError> {
        let mut parser = Self {
            input: input.as_bytes(),
            position: 0,
            variable,
            nodes: 0,
        };
        let node = parser.expression(0)?;
        parser.space();
        if parser.position != parser.input.len() {
            return Err(CalcError::invalid(
                "Unexpected token in numerical expression",
            ));
        }
        Ok(node)
    }
    fn space(&mut self) {
        while self.position < self.input.len() && self.input[self.position].is_ascii_whitespace() {
            self.position += 1;
        }
    }
    fn peek(&mut self) -> Option<u8> {
        self.space();
        self.input.get(self.position).copied()
    }
    fn expression(&mut self, min_bp: u8) -> Result<Node, CalcError> {
        self.nodes += 1;
        if self.nodes > 512 {
            return Err(CalcError::limit("Numerical expression exceeds 512 nodes"));
        }
        let mut left = match self.peek() {
            Some(b'+') => {
                self.position += 1;
                self.expression(5)?
            }
            Some(b'-') => {
                self.position += 1;
                Node::Neg(Box::new(self.expression(5)?))
            }
            Some(b'(') => {
                self.position += 1;
                let n = self.expression(0)?;
                if self.peek() != Some(b')') {
                    return Err(CalcError::invalid("Missing closing parenthesis"));
                }
                self.position += 1;
                n
            }
            Some(c) if c.is_ascii_digit() || c == b'.' => self.numeric()?,
            Some(c) if c.is_ascii_alphabetic() || c == b'_' => self.identifier()?,
            _ => return Err(CalcError::invalid("Expected a real numerical expression")),
        };
        while let Some(ch) = self.peek() {
            let (op, l, r, implicit) = match ch {
                b'+' => ('+', 1, 2, false),
                b'-' => ('-', 1, 2, false),
                b'*' => ('*', 3, 4, false),
                b'/' => ('/', 3, 4, false),
                b'%' => ('%', 3, 4, false),
                b'^' => ('^', 7, 6, false),
                b'(' | b'a'..=b'z' | b'A'..=b'Z' | b'_' => ('*', 3, 4, true),
                _ => break,
            };
            if l < min_bp {
                break;
            }
            if !implicit {
                self.position += 1;
            }
            let right = self.expression(r)?;
            left = Node::Binary(op, Box::new(left), Box::new(right));
        }
        Ok(left)
    }
    fn numeric(&mut self) -> Result<Node, CalcError> {
        let start = self.position;
        while self.position < self.input.len()
            && (self.input[self.position].is_ascii_digit() || self.input[self.position] == b'.')
        {
            self.position += 1;
        }
        if self
            .input
            .get(self.position)
            .is_some_and(|c| matches!(c, b'e' | b'E'))
        {
            let exponent = self.position;
            self.position += 1;
            if self
                .input
                .get(self.position)
                .is_some_and(|c| matches!(c, b'+' | b'-'))
            {
                self.position += 1;
            }
            let digits = self.position;
            while self
                .input
                .get(self.position)
                .is_some_and(u8::is_ascii_digit)
            {
                self.position += 1;
            }
            if self.position == digits {
                self.position = exponent;
            }
        }
        let s = std::str::from_utf8(&self.input[start..self.position]).unwrap();
        let value = s
            .parse::<f64>()
            .map_err(|_| CalcError::invalid("Invalid numerical literal"))?;
        Ok(Node::Constant(finite(value)?))
    }
    fn identifier(&mut self) -> Result<Node, CalcError> {
        let start = self.position;
        while self
            .input
            .get(self.position)
            .is_some_and(|c| c.is_ascii_alphanumeric() || *c == b'_')
        {
            self.position += 1;
        }
        let name = std::str::from_utf8(&self.input[start..self.position]).unwrap();
        if name == self.variable {
            return Ok(Node::Variable);
        }
        if name == "pi" {
            return Ok(Node::Constant(std::f64::consts::PI));
        }
        if name == "tau" {
            return Ok(Node::Constant(std::f64::consts::TAU));
        }
        if name == "e" {
            return Ok(Node::Constant(std::f64::consts::E));
        }
        let binary = matches!(name, "atan2" | "hypot" | "min" | "max" | "pow");
        if !binary
            && !matches!(
                name,
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
                    | "sqrt"
                    | "cbrt"
                    | "abs"
                    | "exp"
                    | "ln"
                    | "log"
                    | "log10"
                    | "log2"
                    | "floor"
                    | "ceil"
                    | "round"
                    | "trunc"
                    | "sign"
                    | "sind"
                    | "cosd"
                    | "tand"
                    | "erf"
                    | "erfc"
                    | "gamma"
            )
        {
            return Err(CalcError::invalid(format!(
                "Unknown variable or numerical function {name}"
            )));
        }
        if self.peek() != Some(b'(') {
            return Err(CalcError::invalid(
                "Use parentheses around numerical function arguments",
            ));
        }
        self.position += 1;
        let mut args = vec![self.expression(0)?];
        if self.peek() == Some(b',') {
            self.position += 1;
            args.push(self.expression(0)?);
        }
        if self.peek() != Some(b')') {
            return Err(CalcError::invalid("Missing closing parenthesis"));
        }
        self.position += 1;
        if (binary && args.len() != 2) || (!binary && name != "log" && args.len() != 1) {
            return Err(CalcError::invalid(format!(
                "Wrong argument count for {name}"
            )));
        }
        Ok(Node::Function(name.to_owned(), args))
    }
}
