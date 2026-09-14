pub mod engine;
pub mod error;
#[cfg(feature = "fast-router")]
pub mod intent;
pub mod model;
pub mod planner;
pub mod protocol;
#[cfg(feature = "fast-router")]
pub mod router;
pub mod service;
pub mod symbolic;

use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Calculation {
    pub display: String,
    pub value: Value,
    pub kind: String,
    pub exact: bool,
}

impl Calculation {
    pub fn text(display: impl Into<String>, kind: &str, exact: bool) -> Self {
        let display = display.into();
        Self {
            value: Value::String(display.clone()),
            display,
            kind: kind.into(),
            exact,
        }
    }
}
