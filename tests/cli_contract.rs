use serde_json::Value;
use std::{
    io::Write,
    os::unix::ffi::OsStringExt,
    process::{Command, Output, Stdio},
};

fn binary() -> Command {
    Command::new(env!("CARGO_BIN_EXE_calc"))
}
fn json(output: Output) -> Value {
    assert!(output.status.code().is_some_and(|n| n > 0));
    assert!(output.stderr.is_empty(), "{:?}", output.stderr);
    let text = String::from_utf8(output.stdout).unwrap();
    assert_eq!(text.lines().count(), 1, "{text}");
    let value: Value = serde_json::from_str(&text).unwrap();
    assert_eq!(value["ok"], false);
    value
}
#[test]
fn invalid_flags_are_machine_readable() {
    for options in [
        ["--retry", "-1"],
        ["--retry", "21"],
        ["--timeout-ms", "0"],
        ["--threads", "0"],
        ["--unknown", "x"],
    ] {
        let reply = json(binary().args(options).arg("--json").output().unwrap());
        assert_eq!(reply["error"]["code"], "invalid_input");
    }
}
#[test]
fn non_utf8_argv_is_an_error_not_a_panic() {
    let reply = json(
        binary()
            .arg(std::ffi::OsString::from_vec(vec![0xff, 0xfe]))
            .arg("--json")
            .output()
            .unwrap(),
    );
    assert_eq!(reply["error"]["code"], "invalid_input");
}
#[test]
fn invalid_stdin_is_bounded_and_machine_readable() {
    for (data, code) in [
        (vec![0xff], "invalid_input"),
        (vec![b'1'; 4097], "resource_limit"),
        (vec![], "invalid_input"),
    ] {
        let mut child = binary()
            .arg("--json")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        child.stdin.take().unwrap().write_all(&data).unwrap();
        assert_eq!(
            json(child.wait_with_output().unwrap())["error"]["code"],
            code
        );
    }
}
#[test]
fn jsonl_preserves_ids_and_rejects_extra_fields() {
    let mut child = binary()
        .args(["--jsonl", "--no-daemon"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    child.stdin.take().unwrap().write_all(b"{\"input\":\"\",\"id\":\"first\"}\n{\"input\":\"1+1\",\"surprise\":1}\n{\"input\":\"\",\"id\":7}\n").unwrap();
    let output = child.wait_with_output().unwrap();
    let text = String::from_utf8(output.stdout).unwrap();
    let values: Vec<Value> = text
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    assert_eq!(values.len(), 3);
    assert_eq!(values[0]["id"], "first");
    assert_eq!(values[2]["id"], 7);
    assert!(values.iter().all(|v| v["ok"] == false));
}
