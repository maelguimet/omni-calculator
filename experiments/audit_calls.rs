//! Link against the already compiled application library to audit tool calls.
//! This is an experiment utility, not a second calculator implementation.
use std::io::{self, BufRead};
use std::time::{Duration, Instant};
use omni_calc::{planner::expression_for_call, protocol::ToolCall};
use serde_json::{Value, json};

fn inspect(calls: &Value, input: &str, math: bool) -> Value {
    let Some(calls) = calls.as_array() else { return json!({"error":"calls are not an array"}) };
    if calls.is_empty() { return json!({"rejection":true}); }
    if calls.len()!=1 { return json!({"error":"multiple tool calls"}); }
    let call: ToolCall = match serde_json::from_value(calls[0].clone()) {
        Ok(call)=>call, Err(error)=>return json!({"error":error.to_string()}),
    };
    match expression_for_call(&call,input) {
        Err(error)=>json!({"error":error}),
        Ok(expression)=> {
            let mut output=json!({"expression":expression});
            if math {
                match omni_calc::engine::evaluate(&expression,Instant::now()+Duration::from_millis(50)) {
                    Ok(answer)=>output["result"]=serde_json::to_value(answer).unwrap(),
                    Err(error)=>output["math_error"]=serde_json::to_value(error).unwrap(),
                }
            }
            output
        },
    }
}

fn main() {
    let math=std::env::args().any(|arg|arg=="--math");
    for (index,line) in io::stdin().lock().lines().enumerate() {
        let line=line.unwrap();
        let row:Value=serde_json::from_str(&line).unwrap();
        let input=row.get("query").or_else(||row.get("input")).and_then(Value::as_str).unwrap();
        let source=input.strip_prefix("Calculation request: ").and_then(|s|s.split_once("\nPrevious tool call was rejected:").map(|p|p.0)).unwrap_or(input);
        let expected=inspect(&row["answers"],source,math);
        let predicted=if row.get("response").is_some() {inspect(&row["response"]["function_calls"],source,math)} else {Value::Null};
        println!("{}",json!({"index":index,"query":input,"family":row["_family"],"expected":expected,"predicted":predicted}));
    }
}
