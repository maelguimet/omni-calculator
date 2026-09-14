//! Standalone inference benchmark, deliberately excluding CLI/IPC/math overhead.
//!
//! rustc --edition 2024 -O training/router/bench.rs -o /tmp/omni-router-bench

#[path = "runtime-v1.rs"]
mod router;

use std::hint::black_box;
use std::time::Instant;

fn main() {
    let initial = Instant::now();
    let first = router::reference_confidence(black_box("what is 71+29"));
    let cold_us = initial.elapsed().as_secs_f64() * 1e6;
    let mut durations = Vec::with_capacity(10_000);
    let mut accepted = 0;
    for i in 0..10_000 {
        let input = format!("what is {}+{}", 100_003 + i, 300_007 + i);
        let start = Instant::now();
        let probability = black_box(router::reference_confidence(black_box(&input)));
        durations.push(start.elapsed().as_secs_f64() * 1e6);
        accepted += usize::from(probability >= router::REFERENCE_THRESHOLD);
    }
    durations.sort_by(f64::total_cmp);
    let mean = durations.iter().sum::<f64>() / durations.len() as f64;
    black_box(router::metadata());
    println!(
        "{{\"model\":\"{}\",\"unique_inputs\":{},\"cold_us\":{},\"first_confidence\":{},\"mean_us\":{},\"p50_us\":{},\"p95_us\":{},\"p99_us\":{},\"accepted\":{}}}",
        router::MODEL_ID,
        durations.len(),
        cold_us,
        first,
        mean,
        durations[5_000],
        durations[9_500],
        durations[9_900],
        accepted,
    );
}
