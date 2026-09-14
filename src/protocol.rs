use crate::{Calculation, error::CalcError};
use serde::{Deserialize, Serialize};
use serde_json::Value;

pub const MAX_INPUT_BYTES: usize = 4096;
pub const MAX_RETRIES: u32 = 20;
pub const MAX_TIMEOUT_MS: u64 = 30_000;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Request {
    pub input: String,
    #[serde(default = "default_retry")]
    pub retry: u32,
    #[serde(default = "default_timeout")]
    pub timeout_ms: u64,
    #[serde(
        default,
        deserialize_with = "present_id",
        skip_serializing_if = "Option::is_none"
    )]
    pub id: Option<Value>,
}

fn present_id<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> Result<Option<Value>, D::Error> {
    Value::deserialize(deserializer).map(Some)
}

pub fn default_retry() -> u32 {
    3
}
pub fn default_timeout() -> u64 {
    5000
}

impl Request {
    pub fn new(input: impl Into<String>) -> Self {
        Self {
            input: input.into(),
            retry: default_retry(),
            timeout_ms: default_timeout(),
            id: None,
        }
    }
    pub fn validate(&self) -> Result<(), CalcError> {
        if self.input.trim().is_empty() {
            return Err(CalcError::invalid("Enter a calculation."));
        }
        if self.input.len() > MAX_INPUT_BYTES {
            return Err(CalcError::limit("Input exceeds 4096 bytes."));
        }
        if self
            .input
            .chars()
            .any(|c| c.is_control() && !c.is_whitespace())
        {
            return Err(CalcError::invalid("Input contains a control character."));
        }
        if self.retry > MAX_RETRIES {
            return Err(CalcError::invalid("--retry must be between 0 and 20."));
        }
        if self.timeout_ms == 0 || self.timeout_ms > MAX_TIMEOUT_MS {
            return Err(CalcError::invalid(
                "--timeout-ms must be between 1 and 30000.",
            ));
        }
        if self
            .id
            .as_ref()
            .is_some_and(|id| id.to_string().len() > 1024)
        {
            return Err(CalcError::invalid("Request id exceeds 1024 bytes."));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Timing {
    pub total_ms: f64,
    pub model_ms: f64,
    pub compute_ms: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ToolCall {
    pub name: String,
    pub arguments: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Response {
    pub schema_version: u32,
    pub ok: bool,
    #[serde(
        default,
        deserialize_with = "present_id",
        skip_serializing_if = "Option::is_none"
    )]
    pub id: Option<Value>,
    pub input: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expression: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call: Option<ToolCall>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<Calculation>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<CalcError>,
    pub attempts: u32,
    pub retries: u32,
    pub timing: Timing,
    pub model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f64>,
}

impl Response {
    pub fn error(request: &Request, error: CalcError) -> Self {
        Self {
            schema_version: 1,
            ok: false,
            id: request.id.clone(),
            input: request.input.clone(),
            expression: None,
            tool_call: None,
            result: None,
            error: Some(error),
            attempts: 0,
            retries: 0,
            timing: Timing::default(),
            model: "none".into(),
            confidence: None,
        }
    }
    pub fn exit_code(&self) -> i32 {
        if self.ok {
            return 0;
        }
        match self
            .error
            .as_ref()
            .map(|e| e.code.as_str())
            .unwrap_or("internal_error")
        {
            "invalid_input" | "unsupported_input" | "ambiguous_input" => 2,
            "math_error" | "unsupported_operation" => 3,
            "model_error" | "model_unavailable" | "retry_exhausted" => 4,
            "resource_limit" | "timeout" => 5,
            _ => 6,
        }
    }
}
