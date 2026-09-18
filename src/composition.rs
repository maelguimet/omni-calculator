//! Learned templates for bounded compositions. The model selects the complete
//! operation structure; scalar text is copied, and original-source validation
//! is mandatory before any candidate can execute.
use crate::{intent, model::Prediction, protocol::ToolCall};
use serde::Deserialize;
use std::fmt::Write;
use std::sync::OnceLock;

pub const MODEL_ID: &str = "omni-composition-v1";
pub const THRESHOLD: f64 = 0.90;
const FEATURES: usize = 2048;
const HIDDEN: usize = 32;
const WEIGHTS: &[u8] = include_bytes!("../assets/composition.bin");
const VOCABULARY: &str = include_str!("../training/composition/vocabulary.txt");

#[derive(Deserialize)]
struct Class {
    function: String,
    arity: usize,
    arguments: Vec<String>,
}

struct Model {
    weights: Vec<f32>,
    classes: Vec<Class>,
}

fn model() -> &'static Model {
    static MODEL: OnceLock<Model> = OnceLock::new();
    MODEL.get_or_init(|| {
        let classes: Vec<Class> =
            serde_json::from_str(include_str!("../training/composition/classes.json")).unwrap();
        assert_eq!(&WEIGHTS[..8], b"OMNICMP1");
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
                classes.len() as u32,
                1,
                4,
                2,
                0
            ]
        );
        assert_eq!(
            WEIGHTS.len(),
            48 + 4 * (FEATURES * HIDDEN + HIDDEN + (HIDDEN + 1) * classes.len())
        );
        let weights: Vec<f32> = WEIGHTS[48..]
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        assert!(weights.iter().all(|w| w.is_finite()));
        Model { weights, classes }
    })
}

fn extract(input: &str) -> Option<Vec<(usize, usize)>> {
    if input.is_empty() || input.len() > 512 || !input.is_ascii() {
        return None;
    }
    let bytes = input.as_bytes();
    let mut spans = Vec::with_capacity(4);
    let mut i = 0;
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
            b if b.is_ascii_digit() || matches!(b, b'+' | b'-' | b'.') => {
                if i > 0 && !bytes[i - 1].is_ascii_whitespace() && bytes[i - 1] != b',' {
                    return None;
                }
                let end = intent::number_end(bytes, i)?;
                if end < bytes.len()
                    && !(bytes[end].is_ascii_whitespace() || matches!(bytes[end], b',' | b'?'))
                {
                    return None;
                }
                spans.push((i, end));
                if spans.len() > 4 {
                    return None;
                }
                i = end;
            }
            b if b.is_ascii_alphabetic() || b == b'\'' => {
                let begin = i;
                while i < bytes.len() && (bytes[i].is_ascii_alphabetic() || bytes[i] == b'\'') {
                    i += 1;
                }
                let word = input[begin..i].to_ascii_lowercase();
                if !VOCABULARY.split_whitespace().any(|allowed| allowed == word) {
                    return None;
                }
            }
            _ => return None,
        }
    }
    (spans.len() >= 2).then_some(spans)
}

fn probabilities(input: &str, spans: &[(usize, usize)]) -> Vec<f64> {
    let model = model();
    let masked = intent::mask(input, spans);
    let mut positioned = format!("{masked} | ");
    let mut position = 0;
    for (i, word) in masked.split_whitespace().enumerate() {
        if i > 0 {
            positioned.push(' ');
        }
        write!(positioned, "{position}:{word}").unwrap();
        position += word.bytes().filter(|&byte| byte == b'#').count();
    }
    let (indices, scale) = intent::features(&positioned);
    let scale = f64::from(scale);
    let bias = FEATURES * HIDDEN;
    let second = bias + HIDDEN;
    let output_bias = second + HIDDEN * model.classes.len();
    let mut hidden = [0.0; HIDDEN];
    for (j, activation) in hidden.iter_mut().enumerate() {
        let mut value = f64::from(model.weights[bias + j]);
        for index in &indices {
            value += f64::from(model.weights[j * FEATURES + index]) * scale;
        }
        *activation = value.max(0.0);
    }
    let mut output = vec![0.0; model.classes.len()];
    for (k, logit) in output.iter_mut().enumerate() {
        *logit = f64::from(model.weights[output_bias + k]);
        for (j, activation) in hidden.iter().enumerate() {
            *logit += f64::from(model.weights[second + k * HIDDEN + j]) * activation;
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

pub fn predict(input: &str) -> Option<Prediction> {
    let spans = extract(input)?;
    let probabilities = probabilities(input, &spans);
    let (index, &confidence) = probabilities
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))?;
    let class = &model().classes[index];
    if index == 0 || confidence < THRESHOLD || spans.len() != class.arity {
        return None;
    }
    let arguments: Vec<String> = class
        .arguments
        .iter()
        .map(|template| {
            let mut argument = template.clone();
            for (letter, &(begin, end)) in ['a', 'b', 'c', 'd'].iter().zip(&spans) {
                argument = argument.replace(&format!("{{{letter}}}"), &input[begin..end]);
            }
            argument
        })
        .collect();
    Some(Prediction {
        calls: vec![ToolCall {
            name: "function".into(),
            arguments: serde_json::json!({"function":class.function,"arguments":arguments}),
        }],
        confidence: Some(confidence),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn python_lexer_and_scalar_inference_match() {
        let cases: serde_json::Value =
            serde_json::from_str(include_str!("../training/composition/parity.json")).unwrap();
        for case in cases.as_array().unwrap() {
            let input = case["input"].as_str().unwrap();
            let expected: Option<Vec<(usize, usize)>> =
                serde_json::from_value(case["spans"].clone()).unwrap();
            assert_eq!(extract(input), expected, "{input}");
            if let Some(spans) = expected {
                assert_eq!(intent::mask(input, &spans), case["masked"]);
                for (actual, expected) in probabilities(input, &spans)
                    .iter()
                    .zip(case["probabilities"].as_array().unwrap())
                {
                    assert!(
                        (actual - expected.as_f64().unwrap()).abs() < 1e-12,
                        "{input}"
                    );
                }
            }
        }
    }
}
