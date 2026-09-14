# Local numerical conversion patch

Upstream: `fend-core` 1.5.8, as published on crates.io. Its version, license,
package manifest, and other source files are retained. Root Cargo.toml selects
this copy with `[patch.crates-io]`; the Cargo registry cache is unchanged.

Numerical source changes are confined to `src/num/bigrat.rs`; two corresponding
display expectations are updated in `tests/integration_tests.rs`.
The original `BigRat::from_f64` (upstream
lines 223–246) multiplied a float by `u64::MAX` and cast it to `u128`, which
saturated large finite values and truncated sufficiently small values to zero.
For example `sinh(228)` became approximately `1.8446744e19`, while its correct
order of magnitude is `5.2253078e98`.

The replacement decodes the IEEE-754 sign, significand, and exponent into an
arbitrary-size rational, including subnormal values. NaN and infinity return an
error. This preserves every finite input float exactly at that conversion step;
it does not claim that the preceding floating computation was exact.

`BigRat::into_f64` (original lines 209–221) also needs scaled conversion when an
exactly decoded subnormal's denominator exceeds the finite float range. It
normalizes both integers to at most 54 significant bits and applies the exponent
in two stages at the subnormal/overflow boundaries. Unrepresentable nonzero
results return an error rather than silently underflowing to zero or overflowing
to infinity. Ordinary finite numerator/denominator conversions keep their path.

Independent application tests in `tests/engine_audit.rs` cover large positive
and negative hyperbolic values, tiny trigonometric/hyperbolic values, nested
subnormal results, a complex identity, overflow rejection, and all previous
scientific/symbolic suites. `omni-float-conversion.patch` records the source delta
against the untouched published crate for review or a future upstream fix.

Validation: 44 upstream unit tests pass (one pre-existing ignored), including
exact bit roundtrips over extreme and subnormal floats and 128 seeded finite bit
patterns. Two integration assertions (`log10_10000` and `log_10000`) formerly
expected `approx. 4.0000000000`; this corrected conversion renders `approx. 4`
instead. Their expectations are updated to preserve the same approximate
numerical answer while matching the corrected display, using `test_eval` to
verify that the corrected display now roundtrips (the previous helper required
its display to change when parsed). These intentional
cosmetic changes are included in the upstream-ready patch. All other published
tests are unchanged. The application's scientific and symbolic suites pass.
The complete updated integration suite passes: 1,129 tests, with five
pre-existing ignored tests.
