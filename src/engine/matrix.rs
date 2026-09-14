use super::*;
use nalgebra::{DMatrix, DVector, SVD, Schur};

const MAX_DIMENSION: usize = 32;

pub(super) fn known(name: &str) -> bool {
    matches!(
        name,
        "det"
            | "determinant"
            | "inverse"
            | "inv"
            | "transpose"
            | "trace"
            | "matmul"
            | "matadd"
            | "matsub"
            | "matrix_power"
            | "solve_linear"
            | "least_squares"
            | "eigenvalues"
            | "rank"
            | "singular_values"
            | "condition"
            | "dot"
            | "cross"
            | "norm"
            | "normalize"
            | "outer"
    )
}

pub(super) fn parse(
    input: &str,
    deadline: Instant,
    depth: usize,
) -> Result<DMatrix<f64>, CalcError> {
    let input = input.trim();
    let rows = if let Some(inner) = input.strip_prefix('[').and_then(|s| s.strip_suffix(']')) {
        split_args(inner)?
            .into_iter()
            .map(|row| vector(row, deadline, depth + 1))
            .collect::<Result<Vec<_>, _>>()?
    } else {
        let result = evaluate_inner(input, deadline, depth + 1)?;
        result
            .value
            .as_array()
            .ok_or_else(|| CalcError::invalid("Expected a matrix such as [[1,2],[3,4]]"))?
            .iter()
            .map(|row| {
                row.as_array()
                    .ok_or_else(|| CalcError::invalid("Expected a matrix"))?
                    .iter()
                    .map(|v| {
                        v.as_f64()
                            .ok_or_else(|| CalcError::math("Matrix entries must be real numbers"))
                    })
                    .collect::<Result<Vec<_>, _>>()
            })
            .collect::<Result<Vec<_>, _>>()?
    };
    if rows.is_empty() || rows[0].is_empty() {
        return Err(CalcError::math("A nonempty matrix is required"));
    }
    let cols = rows[0].len();
    if rows.len() > MAX_DIMENSION || cols > MAX_DIMENSION {
        return Err(CalcError::limit(
            "Matrices are limited to 32 rows and 32 columns",
        ));
    }
    if rows.iter().any(|r| r.len() != cols) {
        return Err(CalcError::math("All matrix rows must have the same length"));
    }
    let flattened: Vec<f64> = rows.iter().flatten().copied().collect();
    for x in &flattened {
        finite(*x)?;
    }
    Ok(DMatrix::from_row_slice(rows.len(), cols, &flattened))
}

pub(super) fn result(matrix: DMatrix<f64>) -> Result<Calculation, CalcError> {
    let rows: Vec<Vec<f64>> = (0..matrix.nrows())
        .map(|r| (0..matrix.ncols()).map(|c| matrix[(r, c)]).collect())
        .collect();
    for x in rows.iter().flatten() {
        finite(*x)?;
    }
    let display = format!(
        "[{}]",
        rows.iter()
            .map(|row| format!(
                "[{}]",
                row.iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>()
                    .join(", ")
            ))
            .collect::<Vec<_>>()
            .join(", ")
    );
    Ok(Calculation {
        display,
        value: serde_json::json!(rows),
        kind: "matrix".into(),
        exact: false,
    })
}

fn square(matrix: &DMatrix<f64>) -> Result<(), CalcError> {
    if matrix.is_square() {
        Ok(())
    } else {
        Err(CalcError::math("This operation requires a square matrix"))
    }
}

fn svd(
    matrix: DMatrix<f64>,
    vectors: bool,
) -> Result<SVD<f64, nalgebra::Dyn, nalgebra::Dyn>, CalcError> {
    let decomposition = SVD::try_new(matrix, vectors, vectors, f64::EPSILON, 4096)
        .ok_or_else(|| CalcError::math("Singular value decomposition did not converge"))?;
    for value in decomposition.singular_values.iter() {
        finite(*value)?;
    }
    Ok(decomposition)
}

fn inverse(matrix: &DMatrix<f64>) -> Result<DMatrix<f64>, CalcError> {
    square(matrix)?;
    let scale = matrix.iter().map(|x| x.abs()).fold(0.0_f64, f64::max);
    if scale == 0.0 {
        return Err(CalcError::math("Matrix is singular"));
    }
    let normalized = matrix / scale;
    let inverse = normalized
        .clone()
        .full_piv_lu()
        .try_inverse()
        .ok_or_else(|| CalcError::math("Matrix is singular"))?;
    let condition = norm(&normalized) * norm(&inverse);
    if !condition.is_finite() || condition > 1e14 {
        return Err(CalcError::math(
            "Matrix is too ill-conditioned for a reliable floating-point inverse",
        ));
    }
    Ok(inverse / scale)
}

fn norm(matrix: &DMatrix<f64>) -> f64 {
    matrix.iter().fold(0.0_f64, |n, x| n.hypot(*x))
}

fn matmul(a: &DMatrix<f64>, b: &DMatrix<f64>) -> Result<DMatrix<f64>, CalcError> {
    let mut product = DMatrix::zeros(a.nrows(), b.ncols());
    for r in 0..a.nrows() {
        for c in 0..b.ncols() {
            let terms: Vec<f64> = (0..a.ncols()).map(|k| a[(r, k)] * b[(k, c)]).collect();
            product[(r, c)] = finite(statistics::sum(&terms))?;
        }
    }
    Ok(product)
}

pub(super) fn evaluate(
    name: &str,
    args: &[&str],
    deadline: Instant,
    depth: usize,
) -> Result<Calculation, CalcError> {
    check_time(deadline)?;
    let binary = matches!(
        name,
        "matmul"
            | "matadd"
            | "matsub"
            | "matrix_power"
            | "solve_linear"
            | "least_squares"
            | "dot"
            | "cross"
            | "outer"
    );
    arity(args, if binary { 2 } else { 1 }, name)?;
    if matches!(name, "dot" | "cross" | "norm" | "normalize" | "outer") {
        if name == "norm" && args[0].trim_start().starts_with("[[") {
            return number(norm(&parse(args[0], deadline, depth)?));
        }
        let x = vector(args[0], deadline, depth)?;
        let n = x.iter().fold(0.0_f64, |n, value| n.hypot(*value));
        if name == "norm" {
            return number(n);
        }
        if name == "normalize" {
            if n == 0.0 {
                return Err(CalcError::math("The zero vector cannot be normalized"));
            }
            let scale = x.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
            let scaled_norm = x.iter().fold(0.0_f64, |n, v| n.hypot(*v / scale));
            return vector_result(x.iter().map(|v| (*v / scale) / scaled_norm).collect());
        }
        let y = vector(args[1], deadline, depth)?;
        if name == "outer" {
            if x.len() > MAX_DIMENSION || y.len() > MAX_DIMENSION {
                return Err(CalcError::limit("Outer products are limited to 32 by 32"));
            }
            return result(DVector::from_vec(x) * DVector::from_vec(y).transpose());
        }
        if x.len() != y.len() {
            return Err(CalcError::math("Vectors must have the same length"));
        }
        if name == "dot" {
            return number(statistics::sum(
                &x.iter().zip(&y).map(|(x, y)| x * y).collect::<Vec<_>>(),
            ));
        }
        if x.len() != 3 {
            return Err(CalcError::math("Cross product requires two 3D vectors"));
        }
        return vector_result(vec![
            x[1] * y[2] - x[2] * y[1],
            x[2] * y[0] - x[0] * y[2],
            x[0] * y[1] - x[1] * y[0],
        ]);
    }
    let a = parse(args[0], deadline, depth)?;
    let calculation = match name {
        "det" | "determinant" => {
            square(&a)?;
            number(a.full_piv_lu().determinant())
        }
        "inverse" | "inv" => result(inverse(&a)?),
        "transpose" => result(a.transpose()),
        "trace" => {
            square(&a)?;
            number(statistics::sum(
                &(0..a.nrows()).map(|i| a[(i, i)]).collect::<Vec<_>>(),
            ))
        }
        "matmul" | "matadd" | "matsub" => {
            let b = parse(args[1], deadline, depth)?;
            if name == "matmul" {
                if a.ncols() != b.nrows() {
                    return Err(CalcError::math(
                        "Matrix multiplication requires columns(A) = rows(B)",
                    ));
                }
                result(matmul(&a, &b)?)
            } else {
                if a.shape() != b.shape() {
                    return Err(CalcError::math(
                        "Matrix addition/subtraction requires equal shapes",
                    ));
                }
                result(if name == "matadd" { a + b } else { a - b })
            }
        }
        "matrix_power" => {
            square(&a)?;
            let n = extensions::signed_integer(args[1], deadline, depth)?;
            if n.abs() > 10000 {
                return Err(CalcError::math(
                    "Matrix exponent must be an integer from -10000 to 10000",
                ));
            }
            let mut power = if n < 0 { inverse(&a)? } else { a };
            let mut product = DMatrix::identity(power.nrows(), power.ncols());
            let mut exponent = n.unsigned_abs();
            while exponent > 0 {
                check_time(deadline)?;
                if exponent & 1 != 0 {
                    product = matmul(&product, &power)?;
                }
                exponent >>= 1;
                if exponent > 0 {
                    power = matmul(&power, &power)?;
                }
            }
            result(product)
        }
        "solve_linear" | "least_squares" => {
            let b = DVector::from_vec(vector(args[1], deadline, depth)?);
            if a.nrows() != b.len() {
                return Err(CalcError::math(
                    "The right-hand vector must have one entry per matrix row",
                ));
            }
            if name == "solve_linear" {
                square(&a)?;
            }
            let scale_a = a
                .iter()
                .map(|x| x.abs())
                .fold(0.0_f64, f64::max)
                .max(f64::MIN_POSITIVE);
            let scale_b = b
                .iter()
                .map(|x| x.abs())
                .fold(0.0_f64, f64::max)
                .max(f64::MIN_POSITIVE);
            let normalized_a = &a / scale_a;
            let normalized_b = &b / scale_b;
            let decomposition = svd(normalized_a.clone(), true)?;
            let largest = decomposition.singular_values[0];
            let tolerance = f64::EPSILON * a.nrows().max(a.ncols()) as f64 * largest;
            if name == "solve_linear"
                && decomposition
                    .singular_values
                    .iter()
                    .any(|s| *s <= tolerance)
            {
                return Err(CalcError::math(
                    "Linear system is singular or numerically rank deficient",
                ));
            }
            let answer = decomposition
                .solve(&normalized_b, tolerance)
                .map_err(CalcError::math)?;
            if name == "solve_linear" {
                let residual = (&normalized_a * &answer - &normalized_b).norm();
                if !residual.is_finite()
                    || residual
                        > 1e-10 * (1.0 + normalized_a.norm() * answer.norm() + normalized_b.norm())
                {
                    return Err(CalcError::math("Linear solve failed its residual check"));
                }
            }
            let ratio = scale_b / scale_a;
            vector_result(
                answer
                    .iter()
                    .map(|x| {
                        if ratio.is_finite() && ratio != 0.0 {
                            x * ratio
                        } else {
                            (x * scale_b) / scale_a
                        }
                    })
                    .collect(),
            )
        }
        "eigenvalues" => {
            square(&a)?;
            let scale = a
                .iter()
                .map(|x| x.abs())
                .fold(0.0_f64, f64::max)
                .max(f64::MIN_POSITIVE);
            let decomposition = Schur::try_new(a / scale, f64::EPSILON, 4096)
                .ok_or_else(|| CalcError::math("Eigenvalue calculation did not converge"))?;
            let eigenvalues =
                decomposition.complex_eigenvalues() * nalgebra::Complex::new(scale, 0.0);
            for value in eigenvalues.iter() {
                finite(value.re)?;
                finite(value.im)?;
            }
            if eigenvalues.iter().all(|v| v.im == 0.0) {
                vector_result(eigenvalues.iter().map(|v| v.re).collect())
            } else {
                let values: Vec<Value> = eigenvalues
                    .iter()
                    .map(|v| serde_json::json!({"re":v.re,"im":v.im}))
                    .collect();
                let display = format!(
                    "[{}]",
                    eigenvalues
                        .iter()
                        .map(|v| format!("{} {:+}i", v.re, v.im))
                        .collect::<Vec<_>>()
                        .join(", ")
                );
                Ok(Calculation {
                    display,
                    value: Value::Array(values),
                    kind: "complex_vector".into(),
                    exact: false,
                })
            }
        }
        "rank" | "singular_values" | "condition" => {
            let shape = a.shape();
            let scale = a
                .iter()
                .map(|x| x.abs())
                .fold(0.0_f64, f64::max)
                .max(f64::MIN_POSITIVE);
            let singular = svd(a / scale, false)?.singular_values;
            if name == "singular_values" {
                vector_result(singular.iter().map(|x| x * scale).collect())
            } else if name == "condition" {
                if singular[singular.len() - 1] == 0.0 {
                    return Err(CalcError::math(
                        "Singular matrix has an infinite condition number",
                    ));
                }
                number(singular[0] / singular[singular.len() - 1])
            } else {
                let threshold = f64::EPSILON * shape.0.max(shape.1) as f64 * singular[0];
                number(singular.iter().filter(|x| **x > threshold).count() as f64)
            }
        }
        _ => Err(CalcError::invalid("Unknown matrix operation")),
    }?;
    check_time(deadline)?;
    Ok(calculation)
}
