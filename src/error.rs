use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CalcError {
    pub code: String,
    pub message: String,
    pub retryable: bool,
}

impl CalcError {
    pub fn new(code: &str, message: impl Into<String>) -> Self {
        Self {
            code: code.into(),
            message: message.into(),
            retryable: false,
        }
    }
    pub fn invalid(message: impl Into<String>) -> Self {
        Self::new("invalid_input", message)
    }
    pub fn math(message: impl Into<String>) -> Self {
        Self::new("math_error", message)
    }
    pub fn limit(message: impl Into<String>) -> Self {
        Self::new("resource_limit", message)
    }
    pub fn model(message: impl Into<String>) -> Self {
        let mut error = Self::new("model_error", message);
        error.retryable = true;
        error
    }
}

impl std::fmt::Display for CalcError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.code, self.message)
    }
}
impl std::error::Error for CalcError {}
