//! Check synthetic tool targets against the independent source validator.
use omni_calc::{planner::expression_for_call, protocol::ToolCall};
use std::io::BufRead;

fn main() {
    let path = std::env::args().nth(1).expect("JSONL training file");
    let file = std::fs::File::open(path).unwrap();
    let mut total = 0;
    let mut failures = 0;
    for line in std::io::BufReader::new(file).lines() {
        let row: serde_json::Value = serde_json::from_str(&line.unwrap()).unwrap();
        let input = row["query"].as_str().unwrap();
        for answer in row["answers"].as_array().unwrap() {
            total += 1;
            let call: ToolCall = serde_json::from_value(answer.clone()).unwrap();
            if let Err(error) = expression_for_call(&call, input) {
                failures += 1;
                if failures <= 20 {
                    eprintln!("{input}: {error:?}");
                }
            }
        }
    }
    println!("{} / {total} grounded targets", total - failures);
    if failures != 0 {
        std::process::exit(1);
    }
}
