//! Exercise the real Rust model, planner, retry loop and engine without a daemon.
use std::{io::{self,BufRead},path::Path,time::Instant};
use omni_calc::{model::{Needle,Prediction,Provider},planner,protocol::Request,error::CalcError};
use serde_json::{Value,json};

struct Recorded {inner:Needle,traces:Vec<Value>}
impl Provider for Recorded {
    fn predict(&mut self,input:&str,feedback:Option<&str>)->Result<Prediction,CalcError> {
        let started=Instant::now();
        let result=self.inner.predict(input,feedback);
        self.traces.push(match &result {
            Ok(prediction)=>json!({"feedback":feedback,"calls":prediction.calls,"ms":started.elapsed().as_secs_f64()*1000.0}),
            Err(error)=>json!({"feedback":feedback,"error":error,"ms":started.elapsed().as_secs_f64()*1000.0}),
        });
        result
    }
    fn name(&self)->&'static str {self.inner.name()}
}

fn main() {
    let args:Vec<String>=std::env::args().collect();
    assert_eq!(args.len(),2,"usage: pipeline MODEL.cact < cases.jsonl");
    let start=Instant::now();
    let needle=Needle::new(Some(Path::new(&args[1]))).expect("load local Needle weights");
    eprintln!("initialized_ms={:.2}",start.elapsed().as_secs_f64()*1000.0);
    let mut model=Recorded{inner:needle,traces:Vec::new()};
    for (index,line) in io::stdin().lock().lines().enumerate() {
        let row:Value=serde_json::from_str(&line.unwrap()).unwrap();
        let input=row.get("query").or_else(||row.get("input")).and_then(Value::as_str).unwrap();
        let input=input.strip_prefix("Calculation request: ").and_then(|s|s.split_once("\nPrevious tool call was rejected:").map(|p|p.0)).unwrap_or(input);
        let mut request=Request::new(input);
        request.id=Some(json!(index));
        // Keep these experiments bounded; a release deployment uses its CLI limit.
        request.timeout_ms=1500;
        model.traces.clear();
        let response=planner::calculate(&mut model,&request);
        println!("{}",json!({"query":input,"family":row["_family"],"answers":row["answers"],
                             "response":response,"trace":model.traces}));
    }
}
