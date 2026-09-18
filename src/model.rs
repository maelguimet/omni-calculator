use crate::{error::CalcError, protocol::ToolCall};
use serde_json::Value;
use std::{
    borrow::Cow,
    ffi::{CString, c_char, c_int},
    fs::File,
    path::{Path, PathBuf},
};
#[cfg(feature = "bundled-needle")]
use std::{
    io::Write,
    os::fd::{AsRawFd, FromRawFd},
};

pub const TOOLS: &str = include_str!("../assets/tools.json");
pub const SYSTEM: &str = include_str!("../assets/system.txt");
pub const ENGINE_VERSION: &str = "needle2-2.0.4";

pub struct Prediction {
    pub calls: Vec<ToolCall>,
    pub confidence: Option<f64>,
}

pub trait Provider {
    fn predict(&mut self, input: &str, feedback: Option<&str>) -> Result<Prediction, CalcError>;
    fn name(&self) -> &'static str {
        "needle2"
    }
}

/// A learned one-pass reference router handles literal math. Richer requests
/// lazily initialize the generative model in the same bounded worker process.
pub struct LocalModel {
    needle: Option<Needle>,
    weights: Option<PathBuf>,
    force_needle: bool,
    selected: &'static str,
    ready: bool,
}

impl LocalModel {
    pub fn new(weights: Option<PathBuf>, force_needle: bool) -> Self {
        Self {
            needle: None,
            weights,
            force_needle,
            selected: "needle2",
            ready: false,
        }
    }
    pub fn needle_loaded(&self) -> bool {
        self.needle.is_some()
    }
    pub fn is_ready(&self) -> bool {
        self.ready
    }
}

impl Provider for LocalModel {
    fn predict(&mut self, input: &str, feedback: Option<&str>) -> Result<Prediction, CalcError> {
        #[cfg(feature = "fast-router")]
        if !self.force_needle && feedback.is_none() {
            if crate::planner::literal_expression(input).is_ok() {
                let confidence = crate::router::reference_confidence(input);
                if confidence >= crate::router::REFERENCE_THRESHOLD {
                    self.selected = crate::router::MODEL_ID;
                    self.ready = true;
                    return Ok(Prediction {
                        calls: vec![ToolCall {
                            name: "evaluate".into(),
                            arguments: serde_json::json!({}),
                        }],
                        confidence: Some(confidence),
                    });
                }
            } else {
                let normalized = crate::planner::normalized_request(input);
                // Familiar wrappers carry no mathematical information. If the first
                // classification defers, allow the same learned model to see the
                // existing normalized wording. Always ground against the full input.
                for candidate in [Some(input), (normalized != input).then_some(normalized)]
                    .into_iter()
                    .flatten()
                {
                    if let Some(prediction) = crate::intent::predict(candidate)
                        && prediction.calls.len() == 1
                        && crate::planner::expression_for_call(&prediction.calls[0], input).is_ok()
                    {
                        self.selected = crate::intent::MODEL_ID;
                        self.ready = true;
                        return Ok(prediction);
                    }
                }
                let numeric = crate::planner::model_number_spelling(input);
                if let Some(candidate) = numeric.as_deref()
                    && let Some(prediction) = crate::intent::predict(candidate)
                    && prediction.calls.len() == 1
                    && crate::planner::expression_for_call(&prediction.calls[0], input).is_ok()
                {
                    self.selected = crate::intent::MODEL_ID;
                    self.ready = true;
                    return Ok(prediction);
                }
                for candidate in [
                    Some(input),
                    (normalized != input).then_some(normalized),
                    numeric.as_deref(),
                ]
                .into_iter()
                .flatten()
                {
                    if let Some(prediction) = crate::composition::predict(candidate)
                        && prediction.calls.len() == 1
                        && crate::planner::expression_for_call(&prediction.calls[0], input).is_ok()
                    {
                        self.selected = crate::composition::MODEL_ID;
                        self.ready = true;
                        return Ok(prediction);
                    }
                }
            }
        }
        #[cfg(not(feature = "fast-router"))]
        let _ = self.force_needle;
        self.selected = "needle2";
        if self.needle.is_none() {
            self.needle = Some(Needle::new(self.weights.as_deref())?);
        }
        self.ready = true;
        let needle = self.needle.as_mut().unwrap();
        self.selected = needle.name();
        needle.predict(input, feedback)
    }
    fn name(&self) -> &'static str {
        self.selected
    }
}

#[cfg(feature = "bundled-needle")]
type Init = unsafe extern "C" fn(*const c_char, *const c_char, *const c_char) -> c_int;
type Complete = unsafe extern "C" fn(*const c_char, c_int, *mut c_char, c_int) -> c_int;
type Reset = unsafe extern "C" fn();
#[cfg(feature = "bundled-needle")]
type Load = unsafe extern "C" fn(*const u8, u64) -> c_int;

/// One model instance per worker process. The upstream C API has process-global state.
pub struct Needle {
    _library: libloading::Library,
    _backing: File,
    complete: Complete,
    reset: Reset,
    _weights: Option<Cow<'static, [u8]>>,
    _single_thread: std::marker::PhantomData<std::rc::Rc<()>>,
}

impl Needle {
    #[cfg(not(feature = "bundled-needle"))]
    pub fn new(_: Option<&Path>) -> Result<Self, CalcError> {
        Err(CalcError::new(
            "model_unavailable",
            "This build has no model. Run scripts/setup_model.py and build with the default features.",
        ))
    }

    #[cfg(feature = "bundled-needle")]
    pub fn new(weights: Option<&Path>) -> Result<Self, CalcError> {
        let bytes = include_bytes!(env!("OMNI_NEEDLE_LIBRARY"));
        // The immutable official library is embedded in the executable. A sealed anonymous
        // file lets Linux's loader map it without installation, extraction races, or network.
        let name = CString::new("omni-needle").unwrap();
        let fd = unsafe {
            libc::memfd_create(name.as_ptr(), libc::MFD_CLOEXEC | libc::MFD_ALLOW_SEALING)
        };
        if fd < 0 {
            return Err(CalcError::new(
                "model_unavailable",
                format!(
                    "Cannot create model mapping: {}",
                    std::io::Error::last_os_error()
                ),
            ));
        }
        let mut backing = unsafe { File::from_raw_fd(fd) };
        backing
            .write_all(bytes)
            .map_err(|e| CalcError::new("model_unavailable", e.to_string()))?;
        let seals =
            libc::F_SEAL_WRITE | libc::F_SEAL_GROW | libc::F_SEAL_SHRINK | libc::F_SEAL_SEAL;
        if unsafe { libc::fcntl(backing.as_raw_fd(), libc::F_ADD_SEALS, seals) } < 0 {
            return Err(CalcError::new(
                "model_unavailable",
                "Cannot seal model mapping.",
            ));
        }
        let library =
            unsafe { libloading::Library::new(format!("/proc/self/fd/{}", backing.as_raw_fd())) }
                .map_err(|e| {
                CalcError::new(
                    "model_unavailable",
                    format!("Cannot load bundled Needle: {e}"),
                )
            })?;
        let (init, complete, reset, load): (Init, Complete, Reset, Load) = unsafe {
            (
                *library.get(b"needle_init\0").map_err(symbol_error)?,
                *library.get(b"needle_complete\0").map_err(symbol_error)?,
                *library.get(b"needle_reset\0").map_err(symbol_error)?,
                *library.get(b"needle_load\0").map_err(symbol_error)?,
            )
        };
        let mut weights = weights
            .map(|path| {
                let metadata = path.metadata().map_err(|e| {
                    CalcError::new(
                        "model_unavailable",
                        format!("Cannot read weights {}: {e}", path.display()),
                    )
                })?;
                if !metadata.is_file() || metadata.len() > 256 * 1024 * 1024 {
                    return Err(CalcError::invalid(
                        "Weights must be a Needle 2 .cact file smaller than 256 MiB.",
                    ));
                }
                let data = std::fs::read(path)
                    .map_err(|e| CalcError::new("model_unavailable", e.to_string()))?;
                if data.len() < 4 || data[..4] != [0x83, 0x2a, 0xe1, 0x05] {
                    return Err(CalcError::invalid(
                        "Weights do not have a Needle 2 .cact header.",
                    ));
                }
                if unsafe { load(data.as_ptr(), data.len() as u64) } < 0 {
                    return Err(CalcError::new(
                        "model_unavailable",
                        "Needle rejected the supplied weights.",
                    ));
                }
                Ok(Cow::Owned(data))
            })
            .transpose()?;
        #[cfg(feature = "tuned-model")]
        if weights.is_none() {
            let data: &'static [u8] = include_bytes!(env!("OMNI_TUNED_WEIGHTS"));
            if unsafe { load(data.as_ptr(), data.len() as u64) } < 0 {
                return Err(CalcError::new(
                    "model_unavailable",
                    "Needle rejected the bundled calculator weights.",
                ));
            }
            weights = Some(Cow::Borrowed(data));
        }
        // Mutable only when the optional bundled tune can replace base weights.
        #[cfg(not(feature = "tuned-model"))]
        let weights = weights.take();
        let tools = CString::new(TOOLS).unwrap();
        let system = CString::new(SYSTEM.trim()).unwrap();
        if unsafe { init(system.as_ptr(), tools.as_ptr(), std::ptr::null()) } < 0 {
            return Err(CalcError::new(
                "model_unavailable",
                "Needle could not initialize the calculator tools.",
            ));
        }
        Ok(Self {
            _library: library,
            _backing: backing,
            complete,
            reset,
            _weights: weights,
            _single_thread: std::marker::PhantomData,
        })
    }
}

#[cfg(feature = "bundled-needle")]
fn symbol_error(e: libloading::Error) -> CalcError {
    CalcError::new("model_unavailable", e.to_string())
}

impl Provider for Needle {
    fn name(&self) -> &'static str {
        if self._weights.is_some() {
            "needle2-tuned"
        } else {
            "needle2"
        }
    }
    fn predict(&mut self, input: &str, feedback: Option<&str>) -> Result<Prediction, CalcError> {
        let prompt = match feedback {
            None => input.to_owned(),
            Some(error) => format!(
                "Calculation request: {input}\nPrevious tool call was rejected: {error}\nCorrect the tool call, preserving all quantities and the requested operation."
            ),
        };
        let input = CString::new(prompt).map_err(|_| CalcError::invalid("Input contains NUL."))?;
        let mut buffer = vec![0u8; 65_536];
        let rc = unsafe {
            (self.reset)();
            (self.complete)(
                input.as_ptr(),
                256,
                buffer.as_mut_ptr().cast(),
                buffer.len() as c_int,
            )
        };
        if rc < 0 {
            return Err(CalcError::model(format!("Needle inference failed ({rc}).")));
        }
        let end = buffer
            .iter()
            .position(|&x| x == 0)
            .ok_or_else(|| CalcError::model("Model response exceeded its buffer."))?;
        let raw: Value = serde_json::from_slice(&buffer[..end])
            .map_err(|_| CalcError::model("Needle returned invalid JSON."))?;
        if raw.get("success").and_then(Value::as_bool) != Some(true) {
            return Err(CalcError::model("Needle reported unsuccessful inference."));
        }
        let calls: Vec<ToolCall> = serde_json::from_value(
            raw.get("function_calls")
                .cloned()
                .ok_or_else(|| CalcError::model("Missing tool calls."))?,
        )
        .map_err(|_| CalcError::model("Malformed tool calls."))?;
        if calls.len() > 1 {
            return Err(CalcError::model(
                "Expected exactly one calculation tool call.",
            ));
        }
        let confidence = if self._weights.is_none() {
            raw.get("confidence")
                .and_then(Value::as_f64)
                .filter(|x| x.is_finite())
        } else {
            None
        };
        Ok(Prediction { calls, confidence })
    }
}
