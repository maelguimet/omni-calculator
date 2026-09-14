//! Generated arithmetic trees checked against an independent rational oracle.
use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::Zero;
use omni_calc::engine;
use std::{
    str::FromStr,
    time::{Duration, Instant},
};

fn next(seed: &mut u64) -> u64 {
    *seed ^= *seed << 13;
    *seed ^= *seed >> 7;
    *seed ^= *seed << 17;
    *seed
}

fn tree(seed: &mut u64, depth: usize) -> (String, BigRational) {
    if depth == 0 {
        let numerator = (next(seed) % 101) as i64 - 50;
        let denominator = (next(seed) % 17 + 1) as i64;
        return (
            format!("({numerator}/{denominator})"),
            BigRational::new(numerator.into(), denominator.into()),
        );
    }
    let (left, a) = tree(seed, depth - 1);
    let (right, b) = tree(seed, depth - 1);
    match next(seed) % 4 {
        0 => (format!("({left}+{right})"), a + b),
        1 => (format!("({left}-{right})"), a - b),
        2 => (format!("({left}*{right})"), a * b),
        _ if !b.is_zero() => (format!("({left}/{right})"), a / b),
        _ => (format!("({left}+{right})"), a + b),
    }
}

#[test]
fn generated_rational_trees_preserve_every_bit_and_grouping() {
    let mut seed = 0x93e5_421a_bcc1_902fu64;
    for index in 0..256 {
        let (expression, expected) = tree(&mut seed, 1 + index % 4);
        let result = engine::evaluate(&expression, Instant::now() + Duration::from_secs(1))
            .unwrap_or_else(|error| panic!("{expression}: {error}"));
        assert!(result.exact, "{expression}: {result:?}");
        assert!(
            matches!(result.kind.as_str(), "integer" | "rational" | "number"),
            "{expression}: {result:?}"
        );
        let text = result
            .value
            .as_str()
            .map(str::to_owned)
            .unwrap_or_else(|| result.value.to_string());
        let actual = if let Some((numerator, denominator)) = text.split_once('/') {
            BigRational::new(
                BigInt::from_str(numerator).unwrap(),
                BigInt::from_str(denominator).unwrap(),
            )
        } else if let Some((whole, fraction)) = text.split_once('.') {
            let sign = if text.starts_with('-') { -1 } else { 1 };
            let denominator = BigInt::from(10u32).pow(fraction.len() as u32);
            BigRational::new(
                BigInt::from_str(whole).unwrap() * &denominator
                    + BigInt::from_str(fraction).unwrap() * sign,
                denominator,
            )
        } else {
            BigRational::from_integer(BigInt::from_str(&text).unwrap())
        };
        assert_eq!(actual, expected, "{expression}: {result:?}");
    }
}
