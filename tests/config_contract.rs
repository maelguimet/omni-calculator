use serde_json::Value;
use std::process::Command;

#[test]
fn a_fast_request_does_not_hide_a_missing_or_invalid_model_option() {
    let temp = tempfile::tempdir().unwrap();
    let invalid = temp.path().join("invalid.cact");
    std::fs::write(&invalid, b"not a model").unwrap();
    for daemon in [false, true] {
        for (path, expected) in [
            (temp.path().join("missing.cact"), "model_unavailable"),
            (invalid.clone(), "invalid_input"),
        ] {
            let mut command = Command::new(env!("CARGO_BIN_EXE_calc"));
            command
                .args(["--json", "--model"])
                .arg(path)
                .arg("what is 1 + 1");
            if !daemon {
                command.arg("--no-daemon");
            }
            let result = command.output().unwrap();
            assert!(!result.status.success());
            let reply: Value = serde_json::from_slice(&result.stdout).unwrap();
            assert_eq!(reply["ok"], false);
            assert_eq!(reply["error"]["code"], expected);
            assert!(result.stderr.is_empty());
        }
    }
}
