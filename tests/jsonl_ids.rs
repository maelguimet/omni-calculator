use serde_json::{Value, json};
use std::{
    io::Write,
    process::{Command, Stdio},
};

#[test]
fn explicit_null_and_ids_on_invalid_objects_are_preserved() {
    let mut child = Command::new(env!("CARGO_BIN_EXE_calc"))
        .args(["--jsonl", "--no-daemon", "--retry", "0"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    {
        let mut input = child.stdin.take().unwrap();
        for record in [
            json!({"input":"1+1","id":null}),
            json!({"input":"1+1","id":"invalid","retry":"three"}),
            json!({"input":"1+1","id":{"key":7},"surprise":true}),
        ] {
            writeln!(input, "{record}").unwrap();
        }
    }
    let result = child.wait_with_output().unwrap();
    assert!(result.stderr.is_empty());
    let records: Vec<Value> = String::from_utf8(result.stdout)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert_eq!(records.len(), 3);
    assert_eq!(records[0].get("id"), Some(&Value::Null));
    assert_eq!(records[1]["id"], "invalid");
    assert_eq!(records[1]["input"], "1+1");
    assert_eq!(records[2]["id"], json!({"key":7}));
    assert_eq!(records[1]["ok"], false);
    assert_eq!(records[2]["ok"], false);
}
