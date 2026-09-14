use super::*;
use statrs::distribution::{
    Beta, Binomial, ChiSquared, Continuous, ContinuousCDF, Discrete, DiscreteCDF, Exp,
    FisherSnedecor, Gamma, Normal, Poisson, StudentsT, Uniform, Weibull,
};

pub(super) fn known(name: &str) -> bool {
    matches!(
        name,
        "sum"
            | "product"
            | "mean"
            | "median"
            | "variance"
            | "sample_variance"
            | "stdev"
            | "stddev"
            | "sample_stdev"
            | "min"
            | "max"
            | "quantile"
            | "percentile"
            | "geomean"
            | "harmonic_mean"
            | "rms"
            | "covariance"
            | "correlation"
            | "linreg"
            | "normal_pdf"
            | "normal_cdf"
            | "normal_quantile"
            | "binomial_pmf"
            | "binomial_cdf"
            | "poisson_pmf"
            | "poisson_cdf"
            | "student_t_pdf"
            | "student_t_cdf"
            | "student_t_quantile"
            | "chi2_pdf"
            | "chi2_cdf"
            | "chi2_quantile"
            | "exponential_pdf"
            | "exponential_cdf"
            | "exponential_quantile"
            | "uniform_pdf"
            | "uniform_cdf"
            | "uniform_quantile"
            | "beta_pdf"
            | "beta_cdf"
            | "beta_quantile"
            | "gamma_pdf"
            | "gamma_cdf"
            | "gamma_quantile"
            | "weibull_pdf"
            | "weibull_cdf"
            | "weibull_quantile"
            | "f_pdf"
            | "f_cdf"
            | "f_quantile"
    )
}

pub(super) fn sum(values: &[f64]) -> f64 {
    // Neumaier compensation preserves small terms in sums with mixed scales.
    let (mut sum, mut correction) = (0.0_f64, 0.0_f64);
    for &x in values {
        let t = sum + x;
        correction += if sum.abs() >= x.abs() {
            (sum - t) + x
        } else {
            (x - t) + sum
        };
        sum = t;
    }
    sum + correction
}

struct Moments {
    mean: f64,
    scale: f64,
    centered: Vec<f64>,
    squares: f64,
}

fn moments(values: &[f64]) -> Result<Moments, CalcError> {
    let mean = finite(mean(values))?;
    let deviations: Vec<f64> = values.iter().map(|x| x - mean).collect();
    // Keep centering in the original scale when possible: this retains small
    // variations on a large offset. Scale before squaring, never mask a NaN.
    let (scale, centered) = if deviations.iter().all(|x| x.is_finite()) {
        let scale = deviations.iter().map(|x| x.abs()).fold(0.0_f64, f64::max);
        let centered = if scale == 0.0 {
            vec![0.0; values.len()]
        } else {
            deviations.iter().map(|x| x / scale).collect()
        };
        (scale, centered)
    } else {
        let scale = values.iter().map(|x| x.abs()).fold(0.0_f64, f64::max);
        (
            scale,
            values.iter().map(|x| x / scale - mean / scale).collect(),
        )
    };
    let squares = finite(sum(&centered.iter().map(|x| x * x).collect::<Vec<_>>()))?;
    Ok(Moments {
        mean,
        scale,
        centered,
        squares,
    })
}

fn mean(values: &[f64]) -> f64 {
    // Divide before summing to avoid overflow when the mean itself is finite.
    let n = values.len() as f64;
    sum(&values.iter().map(|x| x / n).collect::<Vec<_>>())
}

fn probability(value: f64) -> Result<f64, CalcError> {
    if (0.0..=1.0).contains(&value) {
        Ok(value)
    } else {
        Err(CalcError::math("Probability must be between 0 and 1"))
    }
}

pub(super) fn evaluate(
    name: &str,
    args: &[&str],
    deadline: Instant,
    depth: usize,
) -> Result<Calculation, CalcError> {
    check_time(deadline)?;
    if name.ends_with("_pdf")
        || name.ends_with("_cdf")
        || name.ends_with("_pmf")
        || name.ends_with("_quantile")
    {
        return distribution(name, args, deadline, depth);
    }
    if matches!(name, "covariance" | "correlation" | "linreg") {
        arity(args, 2, name)?;
        let x = vector(args[0], deadline, depth)?;
        let y = vector(args[1], deadline, depth)?;
        if x.len() != y.len() || x.len() < 2 {
            return Err(CalcError::math(
                "Paired data must have the same length and at least two observations",
            ));
        }
        let mx = moments(&x)?;
        let my = moments(&y)?;
        let cross: Vec<f64> = mx
            .centered
            .iter()
            .zip(&my.centered)
            .map(|(x, y)| x * y)
            .collect();
        let sxy = finite(sum(&cross))?;
        if name == "covariance" {
            return number((sxy / x.len() as f64 * mx.scale) * my.scale);
        }
        if mx.squares == 0.0 || (name == "correlation" && my.squares == 0.0) {
            return Err(CalcError::math(
                "Constant data has undefined correlation or regression slope",
            ));
        }
        if name == "correlation" {
            return number((sxy / mx.squares.sqrt() / my.squares.sqrt()).clamp(-1.0, 1.0));
        }
        let slope = finite((sxy / mx.squares) * (my.scale / mx.scale))?;
        let intercept = finite(my.mean - slope * mx.mean)?;
        let r_squared = if my.squares == 0.0 {
            1.0
        } else {
            finite((sxy / mx.squares.sqrt() / my.squares.sqrt()).powi(2))?.clamp(0.0, 1.0)
        };
        let value = serde_json::json!({"slope": slope, "intercept": intercept, "r_squared": r_squared, "n": x.len()});
        return Ok(Calculation {
            display: format!("y = {slope} * x + {intercept}; R² = {r_squared}"),
            value,
            kind: "regression".into(),
            exact: false,
        });
    }
    let values = if matches!(name, "min" | "max") && args.len() > 1 {
        args.iter()
            .map(|arg| scalar(arg, deadline, depth))
            .collect::<Result<Vec<_>, _>>()?
    } else {
        arity(
            args,
            if matches!(name, "quantile" | "percentile") {
                2
            } else {
                1
            },
            name,
        )?;
        vector(args[0], deadline, depth)?
    };
    if values.is_empty() {
        return Err(CalcError::math("A nonempty list is required"));
    }
    let n = values.len() as f64;
    let result = match name {
        "sum" => sum(&values),
        "product" => values.iter().product(),
        "mean" => mean(&values),
        "min" => values.iter().copied().fold(f64::INFINITY, f64::min),
        "max" => values.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        "median" | "quantile" | "percentile" => {
            let p = if name == "median" {
                0.5
            } else {
                scalar(args[1], deadline, depth)? / if name == "percentile" { 100.0 } else { 1.0 }
            };
            probability(p)?;
            let mut sorted = values;
            sorted.sort_by(f64::total_cmp);
            // R type 7 / NumPy linear quantile: interpolate positions p * (n - 1).
            let position = p * (n - 1.0);
            let low = position.floor() as usize;
            let high = position.ceil() as usize;
            let weight = position.fract();
            sorted[low] * (1.0 - weight) + sorted[high] * weight
        }
        "variance" | "sample_variance" | "stdev" | "stddev" | "sample_stdev" => {
            let sample = name.starts_with("sample_");
            if sample && values.len() < 2 {
                return Err(CalcError::math(
                    "Sample variance requires at least two observations",
                ));
            }
            let moments = moments(&values)?;
            let deviation = finite(
                moments.scale * (moments.squares / (n - if sample { 1.0 } else { 0.0 })).sqrt(),
            )?;
            if matches!(name, "stdev" | "stddev" | "sample_stdev") {
                deviation
            } else {
                deviation * deviation
            }
        }
        "geomean" => {
            if values.iter().any(|x| *x <= 0.0) {
                return Err(CalcError::math(
                    "Geometric mean requires positive observations",
                ));
            }
            (sum(&values.iter().map(|x| x.ln()).collect::<Vec<_>>()) / n).exp()
        }
        "harmonic_mean" => {
            if values.iter().any(|x| *x <= 0.0) {
                return Err(CalcError::math(
                    "Harmonic mean requires positive observations",
                ));
            }
            n / sum(&values.iter().map(|x| 1.0 / x).collect::<Vec<_>>())
        }
        "rms" => values.iter().fold(0.0_f64, |a, x| a.hypot(*x / n.sqrt())),
        _ => return Err(CalcError::invalid("Unknown statistics function")),
    };
    number(result)
}

fn distribution(
    name: &str,
    args: &[&str],
    deadline: Instant,
    depth: usize,
) -> Result<Calculation, CalcError> {
    let (family, operation) = name.rsplit_once('_').unwrap();
    let required = if matches!(
        family,
        "normal" | "uniform" | "beta" | "gamma" | "weibull" | "f" | "binomial"
    ) {
        3
    } else {
        2
    };
    if !(family == "normal" && args.len() == 1) {
        arity(args, required, name)?;
    }
    let values = args
        .iter()
        .map(|arg| scalar(arg, deadline, depth))
        .collect::<Result<Vec<_>, _>>()?;
    let x = values[0];
    let a = *values.get(1).unwrap_or(&0.0);
    let b = *values.get(2).unwrap_or(&1.0);
    if operation == "quantile" {
        probability(x)?;
    }
    macro_rules! continuous {
        ($distribution:expr) => {{
            let d = $distribution.map_err(|e| CalcError::math(e.to_string()))?;
            match operation {
                "pdf" => d.pdf(x),
                "cdf" => d.cdf(x),
                "quantile" => d.inverse_cdf(x),
                _ => unreachable!(),
            }
        }};
    }
    let result = match family {
        "normal" => continuous!(Normal::new(a, b)),
        "student_t" => continuous!(StudentsT::new(0.0, 1.0, a)),
        "chi2" => continuous!(ChiSquared::new(a)),
        "exponential" => continuous!(Exp::new(a)),
        "uniform" => continuous!(Uniform::new(a, b)),
        "beta" => continuous!(Beta::new(a, b)),
        // Calculator convention is shape, scale; statrs expects shape, rate.
        "gamma" => {
            if b <= 0.0 {
                return Err(CalcError::math("Gamma scale must be positive"));
            }
            continuous!(Gamma::new(a, 1.0 / b))
        }
        "weibull" => continuous!(Weibull::new(a, b)),
        "f" => continuous!(FisherSnedecor::new(a, b)),
        "binomial" => {
            let k = extensions::natural(args[0], deadline, depth)?;
            let n = extensions::natural(args[1], deadline, depth)?;
            probability(b)?;
            let d = Binomial::new(b, n).map_err(|e| CalcError::math(e.to_string()))?;
            if operation == "pmf" {
                d.pmf(k)
            } else {
                d.cdf(k)
            }
        }
        "poisson" => {
            let k = extensions::natural(args[0], deadline, depth)?;
            let d = Poisson::new(a).map_err(|e| CalcError::math(e.to_string()))?;
            if operation == "pmf" {
                d.pmf(k)
            } else {
                d.cdf(k)
            }
        }
        _ => return Err(CalcError::invalid("Unknown distribution")),
    };
    number(result)
}
