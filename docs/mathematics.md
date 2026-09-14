# Mathematical reference

All examples below are calculator expressions. Put them in quotes when invoking
`calc` from a shell. Invalid domains, incompatible units, nonfinite results,
unreliable convergence, and exceeded limits produce errors.

| Area | Examples and functions |
| --- | --- |
| Exact arithmetic | `0.1+0.2`, `1/3`, `2^100`, `30!`; `^` is exponentiation, right associative, and binds more tightly than unary minus |
| Complex numbers | `sqrt(-1)`, `(2+3i)*(4-5i)`, `conj(2+3i)` |
| Units and bases | `5 km to m`, `32 °F to °C`, `(10 m/s)*3 s`, `255 to hex` |
| Trigonometry | `sin`, `cos`, `tan`, inverse and hyperbolic variants; `sind`, `cosd`, `tand` and `asind`, `acosd`, `atand`; `sec`, `csc`, `cot` |
| Roots and logs | `sqrt`, real `cbrt`, `exp`, `ln`, `log(x)` (base 10), `log(x,base)` |
| Rounding and special functions | `floor(x)`, `ceil(x)`, `trunc(x)`, `round(x[,places])`, `abs(x)`, `sign(x)`, `hypot(x,y)`, `atan2(y,x)`, `gamma(x)`, `lgamma(x)`, `beta(a,b)`, `erf(x)`, `erfc(x)` |
| Integer functions | `factorial`, `ncr(n,r)`, `npr(n,r)`, `gcd`, `lcm`, `mod(a,b)` |
| Descriptive statistics | `sum`, `product`, `mean`, `median`, `variance`, `stdev`, `sample_variance`, `sample_stdev`, `min`, `max`, `rms`, `geomean`, `harmonic_mean`; each takes a list. Also `quantile(data,p)` and `percentile(data,p)` |
| Relationships | `covariance(x,y)`, `correlation(x,y)`, `linreg(x,y)` |
| Discrete distributions | `binomial_pmf(k,n,p)`, `binomial_cdf(k,n,p)`, `poisson_pmf(k,lambda)`, `poisson_cdf(k,lambda)` |
| Continuous distributions | `normal_pdf(x[,mean,sigma])`, `normal_cdf(x[,mean,sigma])`, `normal_quantile(p[,mean,sigma])`; further families and parameters are listed below |
| Vectors | `dot(a,b)`, `cross(a,b)`, `norm(a)`, `normalize(a)` |
| Matrices | `det(A)`, `inverse(A)`, `transpose(A)`, `trace(A)`, `matmul(A,B)`, `matadd(A,B)`, `matsub(A,B)`, `outer(a,b)`, `solve_linear(A,b)`, `least_squares(A,b)`, `matrix_power(A,n)`, `eigenvalues(A)`, `singular_values(A)`, `rank(A)`, `condition(A)` |
| Numerical calculus | `integrate(expr,x,a,b)`, `root(expr,x,a,b)`, `derivative_at(expr,x,a)`, `summation(expr,k,start,end)`, `product_range(expr,k,start,end)` |
| Symbolic algebra | `simplify(expr)`, `expand(expr)`, `factor(expr)`, `substitute(expr,x,value)` |
| Symbolic calculus | `diff(expr,x[,order])`, `integrate(expr,x)` |
| Equations | `solve(x^2-5*x+6=0,x)`; an omitted `=0` means solve the expression equal to zero |

Lists use `[1,2,3]`, matrices `[[1,2],[3,4]]`. Scalar extensions can be nested:
`sqrt(mean([9,9]))`, `mean([sqrt(4),ncr(4,2)]) + det([[1,0],[0,2]])`.
Use explicit vector/matrix functions for nonscalar arithmetic.
Use parentheses for function calls: `sin(2)^2` squares the sine of 2.
A single trailing comma is allowed in argument lists, vectors, and matrices,
such as `mean([1,2,3,])`. Missing interior values and incorrect arity are errors.

## Conventions

Trigonometric functions use radians by default. Use degree functions or explicit
angle units. `atan2(y,x)` returns a radian angle and rejects `(0,0)`.
Temperature conversions also accept written `degree`/`degrees Celsius`,
`Fahrenheit`, or `Kelvin`, with case-insensitive names; for example,
`86 degrees Fahrenheit to Celsius` gives `30 °C` with the temperature offset.
`ln(x)` means natural logarithm and `log(x)` means base 10 in both engines.
The numerical helper `log(x,base)` accepts a custom base; use
`ln(x)/ln(base)` for the corresponding symbolic expression.
`cbrt(x)` chooses the real cube root for negative real inputs. Complex scalar
functions use the scalar engine's principal branches.

`variance` and `stdev` are population statistics. `sample_variance` and
`sample_stdev` use the sample correction and require at least two observations.
`covariance` also uses the population divisor. Paired-data operations require
equal list lengths with at least two observations. `linreg(x,y)` returns the
slope, intercept, and R² for `y = slope*x + intercept`.
Quantiles use linear interpolation at position `p*(n-1)` in sorted data;
`quantile` takes `p` in `[0,1]` and `percentile` takes it in `[0,100]`.
`round` rounds halfway cases away from zero. Statistical data and
matrix entries must be dimensionless real numbers; scalar unit arithmetic is
available separately.

Each continuous distribution family provides `_pdf`, `_cdf`, and `_quantile`
functions. The first argument is the observation for PDF/CDF or the probability
for a quantile. Remaining arguments are:

| Family | Remaining arguments |
| --- | --- |
| `normal` | `mean,sigma`, or omit both for the standard normal |
| `student_t` | `df` |
| `chi2` | `df` |
| `exponential` | `rate` |
| `uniform` | `lower,upper` |
| `beta` | `alpha,beta` |
| `gamma` | `shape,scale` |
| `weibull` | `shape,scale` |
| `f` | `df1,df2` |

For example, `gamma_cdf(2,1,2)` uses shape 1 and scale 2.
Discrete counts must be exact nonnegative integers. Quantile endpoints that
would produce infinity return an error.

`solve_linear(A,b)` requires a square matrix with a numerically unique solution.
`least_squares(A,b)` accepts rectangular matrices and uses an SVD pseudoinverse.
`rank` uses a relative floating-point threshold; `condition` is the ratio of
largest to smallest singular value. Eigenvalues of a real matrix may be complex.

Exact scalar rational/integer calculations preserve precision. Exact fractional
JSON values use canonical improper fraction strings, even when human output is
a mixed number: display `-8 8/9` has value `"-80/9"` and kind `"rational"`.
Exact integers outside `[-(2^53-1), 2^53-1]` also use JSON strings to preserve
their value in clients that parse JSON numbers through IEEE-754.
Numerical statistics, distributions, matrices and calculus use `f64`, report `exact:false`,
and may reject overflow, ill-conditioning or loss of reliable precision.
An exact displayed integer must not be inferred merely from an approximate
result happening to look integral. Finite precision cannot resolve every narrow
feature or nearly singular problem.

Symbolic simplification and integration use real-variable domain restrictions;
equation solutions may be complex. Conditions such as `x != 0` survive algebraic
cancellation and appear both in display output and in the JSON result. An
antiderivative includes an arbitrary integration constant. Substitution does
not erase restrictions belonging to the replacement expression.

Symbolic integration supports common elementary forms and selected integration
by parts. Polynomial solving supports rational-coefficient linear and quadratic
equations, plus higher degrees when bounded rational-root search leaves a
remainder of degree at most two. Factorization uses bounded rational
root extraction; a residual factor may still be reducible. General
transcendental solving and arbitrary symbolic integration are unsupported.

Numerical integration requires a finite interval and a numerically resolvable
integrand. Root finding requires a valid finite bracket and verified residual;
it does not silently accept a pole as a root. For difficult problems, split
intervals around known features or use an analytically appropriate expression.
`derivative_at` uses forward automatic differentiation of the numerical
expression and checks nonsmooth points.

Numerical calculus uses a real scalar grammar: arithmetic, powers, `sin`, `cos`,
`tan`, inverse and hyperbolic variants, `sind`, `cosd`, `tand`, `sqrt`, `cbrt`,
`exp`, `ln`, `log`, `log10`, `log2`, `abs`, `sign`, `floor`, `ceil`, `round`,
`trunc`, `erf`, `erfc`, `gamma`, and binary `min`, `max`, `atan2`, `hypot`, `pow`.
Function arguments use parentheses; `log` accepts an optional second argument
for the base. The integration/summation variable and `pi`, `tau`, `e` are the
available identifiers. Bounds and evaluation points are dimensionless real
scalar expressions.

## Bounds

Inputs are limited to 4096 bytes and 64 nesting levels. Lists allow at most 512
items, matrices at most 32 by 32, and iterative/symbolic expansion has explicit
budgets. Worker memory is capped and every request has a wall-clock timeout.
Large exact integers are supported within these resource limits, not without
bound. Integer helper arguments have magnitude at most `2^53-1`; `factorial(n)`
limits `n` to 5000, combinatorial helpers allow at most 5000 multiplication steps,
and `matrix_power` accepts integer exponents from -10000 to 10000.
Symbolic polynomial degree is limited to 64, integration by parts to polynomial
degree 16, derivative order to 16, and exact symbolic coefficients to 8192 bits.
Currency exchange rates are unavailable offline and never fetched.
