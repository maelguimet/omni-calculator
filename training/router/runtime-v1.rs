//! A trained CPU MLP selecting the typed `evaluate()` input-reference tool.
//!
//! The network predicts only whether a request is a literal mathematical
//! expression. The caller must also validate `planner::literal_expression`.
//! Lower-confidence requests are deferred to the broader local Needle model.

use std::sync::OnceLock;

const FEATURES: usize = 2048;
const HIDDEN: usize = 24;
const OUTPUTS: usize = 2;
const HEADER_BYTES: usize = 48;
const FNV_OFFSET: u64 = 14_695_981_039_346_656_037;
const FNV_PRIME: u64 = 1_099_511_628_211;
const NGRAM_SALT: u64 = 0x9e37_79b9_7f4a_7c15;

/// Selected using validation precision; independent test scores accompany weights.
pub const REFERENCE_THRESHOLD: f64 = 0.98;
pub const MODEL_ID: &str = "omni-reference-router-v1";

const WEIGHTS: &[u8] = include_bytes!("../../assets/router.bin");
static MODEL: OnceLock<Model> = OnceLock::new();

struct Model {
    w1: Vec<f32>,
    b1: Vec<f32>,
    w2: Vec<f32>,
    b2: Vec<f32>,
}

fn model() -> &'static Model {
    MODEL.get_or_init(|| {
        assert_eq!(&WEIGHTS[..8], b"OMNIRTR1", "invalid embedded router magic");
        let header: Vec<u32> = WEIGHTS[8..40]
            .chunks_exact(4)
            .map(|bytes| u32::from_le_bytes(bytes.try_into().unwrap()))
            .collect();
        assert_eq!(
            header,
            [
                1,
                FEATURES as u32,
                HIDDEN as u32,
                OUTPUTS as u32,
                1,
                4,
                1,
                0
            ]
        );
        assert_eq!(
            WEIGHTS.len(),
            HEADER_BYTES + (FEATURES * HIDDEN + HIDDEN + HIDDEN * OUTPUTS + OUTPUTS) * 4
        );
        let parameters: Vec<f32> = WEIGHTS[HEADER_BYTES..]
            .chunks_exact(4)
            .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
            .collect();
        assert!(parameters.iter().all(|x| x.is_finite()));
        let a = FEATURES * HIDDEN;
        let b = a + HIDDEN;
        let c = b + HIDDEN * OUTPUTS;
        Model {
            w1: parameters[..a].to_vec(),
            b1: parameters[a..b].to_vec(),
            w2: parameters[b..c].to_vec(),
            b2: parameters[c..].to_vec(),
        }
    })
}

fn features(input: &str) -> (Vec<usize>, f32) {
    let trimmed = input.trim_matches([' ', '\t', '\r', '\n']);
    let mut bytes = Vec::with_capacity(trimmed.len() + 2);
    bytes.push(0xff);
    bytes.extend(trimmed.bytes().map(|byte| byte.to_ascii_lowercase()));
    bytes.push(0xfe);
    let mut present = [false; FEATURES];
    for width in 1..=4 {
        let initial = FNV_OFFSET ^ (width as u64).wrapping_mul(NGRAM_SALT);
        for gram in bytes.windows(width) {
            let mut hash = initial;
            for byte in gram {
                hash = (hash ^ u64::from(*byte)).wrapping_mul(FNV_PRIME);
            }
            present[hash as usize & (FEATURES - 1)] = true;
        }
    }
    let indices: Vec<usize> = present
        .iter()
        .enumerate()
        .filter_map(|(i, present)| present.then_some(i))
        .collect();
    let scale = 1.0_f32 / (indices.len() as f32).sqrt();
    (indices, scale)
}

/// Return learned probability of the input-reference tool. No math is executed.
pub fn reference_confidence(input: &str) -> f64 {
    if input.len() > 4096 {
        return 0.0;
    }
    let model = model();
    let (indices, scale) = features(input);
    let scale = f64::from(scale);
    let mut hidden = [0.0_f64; HIDDEN];
    for (j, activation) in hidden.iter_mut().enumerate() {
        let weights = &model.w1[j * FEATURES..(j + 1) * FEATURES];
        let mut value = f64::from(model.b1[j]);
        for index in &indices {
            value += f64::from(weights[*index]) * scale;
        }
        *activation = value.max(0.0);
    }
    let mut logits = [0.0_f64; OUTPUTS];
    for (k, logit) in logits.iter_mut().enumerate() {
        *logit = f64::from(model.b2[k]);
        for (j, activation) in hidden.iter().enumerate() {
            *logit += f64::from(model.w2[k * HIDDEN + j]) * activation;
        }
    }
    let margin = logits[1] - logits[0];
    if margin >= 0.0 {
        1.0 / (1.0 + (-margin).exp())
    } else {
        let exponential = margin.exp();
        exponential / (1.0 + exponential)
    }
}

/// Training provenance, checksums, split policy, measured errors, and threshold.
pub fn metadata() -> &'static str {
    include_str!("metadata.json")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn feature_and_inference_parity_with_independent_python_scalar_reference() {
        let fixtures: serde_json::Value =
            serde_json::from_str(include_str!("parity.json")).unwrap();
        for fixture in fixtures.as_array().unwrap() {
            let input = fixture["input"].as_str().unwrap();
            let (indices, scale) = features(input);
            let expected: Vec<usize> = fixture["indices"]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_u64().unwrap() as usize)
                .collect();
            assert_eq!(indices, expected, "feature indices for {input:?}");
            assert_eq!(
                u64::from(scale.to_bits()),
                fixture["scale_bits"].as_u64().unwrap(),
                "exact feature normalization bits for {input:?}"
            );
            let actual = reference_confidence(input);
            let expected = fixture["confidence"].as_f64().unwrap();
            assert!(
                (actual - expected).abs() < 1e-12,
                "inference for {input:?}: {actual} versus {expected}"
            );
        }
    }
}
