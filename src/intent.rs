//! Trained bounded verbal-operation MLP. Numeric spans are copied unchanged.
//!
//! This emits a candidate tool call. The caller must validate it with
//! `planner::expression_for_call` and defer immediately if validation fails.
use crate::{model::Prediction, protocol::ToolCall};
use std::sync::OnceLock;

pub const MODEL_ID: &str = "omni-intent-v1";
pub const THRESHOLD: f64 = 0.98;
const FEATURES: usize = 2048;
const HIDDEN: usize = 32;
const OUTPUTS: usize = 20;
const WEIGHTS: &[u8] = include_bytes!("../assets/intent.bin");
const VOCABULARY: &str = include_str!("../training/intent/vocabulary.txt");
const CUES: &str = include_str!("../training/intent/operation-cues.txt");
const CLASSES: [(&str, usize, bool); OUTPUTS] = [
    ("defer", 0, false),
    ("add", 2, false),
    ("subtract", 2, false),
    ("subtract", 2, true),
    ("multiply", 2, false),
    ("divide", 2, false),
    ("divide", 2, true),
    ("power", 2, false),
    ("sqrt", 1, false),
    ("cbrt", 1, false),
    ("factorial", 1, false),
    ("mod", 2, false),
    ("gcd", 2, false),
    ("lcm", 2, false),
    ("ncr", 2, false),
    ("ncr", 2, true),
    ("npr", 2, false),
    ("npr", 2, true),
    ("percent", 2, false),
    ("abs", 1, false),
];

struct Model {
    w1: Vec<f32>,
    b1: Vec<f32>,
    w2: Vec<f32>,
    b2: Vec<f32>,
}
static MODEL: OnceLock<Model> = OnceLock::new();

fn model() -> &'static Model {
    MODEL.get_or_init(|| {
        assert_eq!(&WEIGHTS[..8], b"OMNIINT1");
        let header: Vec<u32> = WEIGHTS[8..40]
            .chunks_exact(4)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
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
            48 + 4 * (FEATURES * HIDDEN + HIDDEN + HIDDEN * OUTPUTS + OUTPUTS)
        );
        let values: Vec<f32> = WEIGHTS[48..]
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        assert!(values.iter().all(|x| x.is_finite()));
        let a = FEATURES * HIDDEN;
        let b = a + HIDDEN;
        let c = b + HIDDEN * OUTPUTS;
        Model {
            w1: values[..a].to_vec(),
            b1: values[a..b].to_vec(),
            w2: values[b..c].to_vec(),
            b2: values[c..].to_vec(),
        }
    })
}

fn number_end(bytes: &[u8], start: usize) -> Option<usize> {
    let mut i = start;
    if bytes.get(i).is_some_and(|b| matches!(b, b'+' | b'-')) {
        i += 1;
    }
    let before = i;
    while bytes.get(i).is_some_and(u8::is_ascii_digit) {
        i += 1;
    }
    let mut digits = i - before;
    if bytes.get(i) == Some(&b'.') {
        i += 1;
        let before = i;
        while bytes.get(i).is_some_and(u8::is_ascii_digit) {
            i += 1;
        }
        digits += i - before;
    }
    if digits == 0 {
        return None;
    }
    if bytes.get(i).is_some_and(|b| matches!(b, b'e' | b'E')) {
        let mut end = i + 1;
        if bytes.get(end).is_some_and(|b| matches!(b, b'+' | b'-')) {
            end += 1;
        }
        let digits = end;
        while bytes.get(end).is_some_and(u8::is_ascii_digit) {
            end += 1;
        }
        if end > digits {
            i = end;
        }
    }
    Some(i)
}

/// Class-independent lexical and complexity gate, never an operation selector.
fn extract(input: &str) -> Option<Vec<(usize, usize)>> {
    if input.is_empty() || input.len() > 512 || !input.is_ascii() {
        return None;
    }
    let bytes = input.as_bytes();
    let mut i = 0;
    let mut spans = Vec::with_capacity(2);
    let mut cues = 0;
    while i < bytes.len() {
        match bytes[i] {
            b if b.is_ascii_whitespace() || b == b'?' => i += 1,
            b',' => {
                if i > 0
                    && i + 1 < bytes.len()
                    && bytes[i - 1].is_ascii_digit()
                    && bytes[i + 1].is_ascii_digit()
                {
                    return None;
                }
                i += 1;
            }
            b'%' => {
                if spans.last().is_none_or(|&(_, end)| end != i) {
                    return None;
                }
                cues += 1;
                i += 1;
            }
            b if b.is_ascii_digit() || matches!(b, b'+' | b'-' | b'.') => {
                if i > 0 && !bytes[i - 1].is_ascii_whitespace() && bytes[i - 1] != b',' {
                    return None;
                }
                let end = number_end(bytes, i)?;
                if end < bytes.len()
                    && !(bytes[end].is_ascii_whitespace()
                        || matches!(bytes[end], b',' | b'?' | b'%'))
                {
                    return None;
                }
                spans.push((i, end));
                if spans.len() > 2 {
                    return None;
                }
                i = end;
            }
            b if b.is_ascii_alphabetic() || b == b'\'' => {
                let start = i;
                i += 1;
                while i < bytes.len() && (bytes[i].is_ascii_alphabetic() || bytes[i] == b'\'') {
                    i += 1;
                }
                let word = input[start..i].to_ascii_lowercase();
                if !VOCABULARY.split_whitespace().any(|item| item == word) {
                    return None;
                }
                cues += usize::from(CUES.split_whitespace().any(|item| item == word));
            }
            _ => return None,
        }
    }
    (!spans.is_empty() && cues == 1).then_some(spans)
}

fn mask(input: &str, spans: &[(usize, usize)]) -> String {
    let mut text = String::with_capacity(input.len());
    let mut previous = 0;
    for &(start, end) in spans {
        text.push_str(&input[previous..start]);
        text.push('#');
        previous = end;
    }
    text.push_str(&input[previous..]);
    text
}

fn features(text: &str) -> (Vec<usize>, f32) {
    let mut bytes = Vec::with_capacity(text.len() + 2);
    bytes.push(0xff);
    bytes.extend(
        text.trim_matches([' ', '\t', '\r', '\n'])
            .bytes()
            .map(|b| b.to_ascii_lowercase()),
    );
    bytes.push(0xfe);
    let mut present = [false; FEATURES];
    for width in 1..=4 {
        let initial =
            14_695_981_039_346_656_037_u64 ^ (width as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
        for gram in bytes.windows(width) {
            let mut hash = initial;
            for byte in gram {
                hash = (hash ^ u64::from(*byte)).wrapping_mul(1_099_511_628_211);
            }
            present[hash as usize & (FEATURES - 1)] = true;
        }
    }
    let indices: Vec<usize> = present
        .iter()
        .enumerate()
        .filter_map(|(i, p)| p.then_some(i))
        .collect();
    let scale = 1.0_f32 / (indices.len() as f32).sqrt();
    (indices, scale)
}

fn probabilities(masked: &str) -> [f64; OUTPUTS] {
    let model = model();
    let (indices, scale) = features(masked);
    let scale = f64::from(scale);
    let mut hidden = [0.0; HIDDEN];
    for (j, activation) in hidden.iter_mut().enumerate() {
        let mut value = f64::from(model.b1[j]);
        for index in &indices {
            value += f64::from(model.w1[j * FEATURES + index]) * scale;
        }
        *activation = value.max(0.0);
    }
    let mut output = [0.0; OUTPUTS];
    for (k, logit) in output.iter_mut().enumerate() {
        *logit = f64::from(model.b2[k]);
        for (j, activation) in hidden.iter().enumerate() {
            *logit += f64::from(model.w2[k * HIDDEN + j]) * activation;
        }
    }
    let peak = output.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    for value in &mut output {
        *value = (*value - peak).exp();
    }
    let total = output.iter().sum::<f64>();
    for value in &mut output {
        *value /= total;
    }
    output
}

/// Predict a function and operand order using learned weights. Arguments are
/// copies of original scalar spans; this does not validate mathematical domains.
pub fn predict(input: &str) -> Option<Prediction> {
    let spans = extract(input)?;
    let probs = probabilities(&mask(input, &spans));
    let (class, &confidence) = probs.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1))?;
    let (function, arity, reverse) = CLASSES[class];
    if class == 0 || confidence < THRESHOLD || spans.len() != arity {
        return None;
    }
    let mut arguments: Vec<&str> = spans
        .iter()
        .map(|&(start, end)| &input[start..end])
        .collect();
    if reverse {
        arguments.reverse();
    }
    Some(Prediction {
        calls: vec![ToolCall {
            name: "function".into(),
            arguments: serde_json::json!({"function":function,"arguments":arguments}),
        }],
        confidence: Some(confidence),
    })
}

pub fn metadata() -> &'static str {
    include_str!("../training/intent/metadata.json")
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn python_extraction_features_and_scalar_inference_match() {
        let cases: serde_json::Value =
            serde_json::from_str(include_str!("../training/intent/parity.json")).unwrap();
        for case in cases.as_array().unwrap() {
            let input = case["input"].as_str().unwrap();
            let actual = extract(input);
            if case["spans"].is_null() {
                assert!(actual.is_none(), "{input}");
                continue;
            }
            let expected: Vec<(usize, usize)> = case["spans"]
                .as_array()
                .unwrap()
                .iter()
                .map(|a| {
                    (
                        a[0].as_u64().unwrap() as usize,
                        a[1].as_u64().unwrap() as usize,
                    )
                })
                .collect();
            assert_eq!(actual.as_ref(), Some(&expected), "{input}");
            let masked = mask(input, &expected);
            assert_eq!(masked, case["masked"]);
            let (indices, scale) = features(&masked);
            let expected_indices: Vec<usize> = case["indices"]
                .as_array()
                .unwrap()
                .iter()
                .map(|n| n.as_u64().unwrap() as usize)
                .collect();
            assert_eq!(indices, expected_indices);
            assert_eq!(
                u64::from(scale.to_bits()),
                case["scale_bits"].as_u64().unwrap()
            );
            for (actual, expected) in probabilities(&masked)
                .iter()
                .zip(case["probabilities"].as_array().unwrap())
            {
                assert!(
                    (actual - expected.as_f64().unwrap()).abs() < 1e-12,
                    "{input}: {actual} vs {expected}"
                );
            }
        }
    }
}
