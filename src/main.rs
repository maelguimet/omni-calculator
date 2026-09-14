use omni_calc::{
    error::CalcError,
    model,
    protocol::{Request, Response},
    service::{self, ClientOptions},
};
use std::{
    io::{self, IsTerminal, Read, Write},
    path::PathBuf,
};

const HELP: &str = r#"calc — local scientific calculator for humans and agents

Usage: calc [OPTIONS] [CALCULATION ...]

  calc what is 1 + 1
  calc wat iz 1+1 --json
  calc 'what is (3+4)*5?' --retry 3
  calc 'integrate(sin(x),x,0,pi)'
  printf '%s\n' '{"id":1,"input":"sqrt(144)"}' | calc --jsonl

Options:
  --json             One structured JSON response (including errors)
  --jsonl            Read JSON requests from stdin; one response per line
  --retry N          Maximum retries after the first attempt (default 3; 0..20)
  --timeout-ms N     Total request budget in milliseconds (default 5000; 1..30000)
  --threads N        Model CPU threads (default 2; 1..8)
  --model PATH       Use local fine-tuned Needle 2 .cact weights
  --backend NAME     auto (default) or needle (force the generative model)
  --no-daemon        Use a fresh, disposable worker for this request
  --explain          Show the interpreted expression and timing on stderr
  --warmup           Load the local model and check a simple calculation
  --status           Report the local worker's status
  --stop             Stop this version/configuration's local worker
  --tools            Print the model's tool schema as JSON
  --help, -h         Show this help
  --version, -V      Print version
  --                 Treat remaining arguments as input

Without arguments, a pipe supplies one calculation. The local worker stops
after two idle minutes. No account, network, Python, or GPU is used at runtime.
Quote expressions containing shell operators such as *, (, ), <, >, or !.
Natural-language interpretation can fail; inspect `expression` in JSON output.
Exit codes: 0 success, 2 input, 3 mathematics, 4 model, 5 limits, 6 service.
"#;

#[derive(Default)]
struct Cli {
    query: Vec<String>,
    json: bool,
    jsonl: bool,
    explain: bool,
    retry: Option<u32>,
    timeout: Option<u64>,
    client: ClientOptions,
    action: Option<String>,
}

fn parse(args: &[String]) -> Result<Cli, CalcError> {
    let mut cli = Cli::default();
    let mut i = 0;
    let mut literal = false;
    while i < args.len() {
        let argument = &args[i];
        if literal {
            cli.query.push(argument.clone());
            i += 1;
            continue;
        }
        let (key, inline) = argument
            .split_once('=')
            .map_or((argument.as_str(), None), |(a, b)| (a, Some(b)));
        match key {
            "--" => literal = true,
            "--json" => cli.json = true,
            "--jsonl" => {
                cli.json = true;
                cli.jsonl = true;
            }
            "--explain" => cli.explain = true,
            "--no-daemon" => cli.client.no_daemon = true,
            "--help" | "-h" | "--version" | "-V" | "--tools" | "--warmup" | "--stop"
            | "--status" => {
                if cli.action.is_some() {
                    return Err(CalcError::invalid(
                        "Choose one of --help, --version, --tools, --warmup, --status, or --stop.",
                    ));
                }
                cli.action = Some(key.to_owned());
            }
            "--retry" | "--timeout-ms" | "--threads" | "--model" | "--backend" => {
                let value = if let Some(value) = inline {
                    value
                } else {
                    i += 1;
                    args.get(i)
                        .map(String::as_str)
                        .ok_or_else(|| CalcError::invalid(format!("{key} requires a value.")))?
                };
                match key {
                    "--backend" => {
                        cli.client.force_needle = match value {
                            "auto" => false,
                            "needle" => true,
                            _ => {
                                return Err(CalcError::invalid(
                                    "--backend must be auto or needle.",
                                ));
                            }
                        };
                    }
                    "--retry" => {
                        cli.retry = Some(value.parse().map_err(|_| {
                            CalcError::invalid("--retry requires a nonnegative integer.")
                        })?)
                    }
                    "--timeout-ms" => {
                        cli.timeout = Some(value.parse().map_err(|_| {
                            CalcError::invalid("--timeout-ms requires a positive integer.")
                        })?)
                    }
                    "--threads" => {
                        cli.client.threads = value.parse().map_err(|_| {
                            CalcError::invalid("--threads requires a positive integer.")
                        })?
                    }
                    _ => cli.client.weights = Some(PathBuf::from(value)),
                }
            }
            _ if key.starts_with('-')
                && !key
                    .as_bytes()
                    .get(1)
                    .is_some_and(|c| c.is_ascii_digit() || *c == b'.') =>
            {
                return Err(CalcError::invalid(format!(
                    "Unknown option {key}. Use -- before an expression beginning with a hyphen."
                )));
            }
            _ => cli.query.push(argument.clone()),
        }
        if inline.is_some()
            && ![
                "--retry",
                "--timeout-ms",
                "--threads",
                "--model",
                "--backend",
            ]
            .contains(&key)
            && key.starts_with('-')
        {
            return Err(CalcError::invalid(format!(
                "{key} does not accept a value."
            )));
        }
        i += 1;
    }
    if !(1..=8).contains(&cli.client.threads) {
        return Err(CalcError::invalid("--threads must be between 1 and 8."));
    }
    if cli.retry.is_some_and(|n| n > 20) {
        return Err(CalcError::invalid("--retry must be between 0 and 20."));
    }
    if cli.timeout.is_some_and(|n| n == 0 || n > 30_000) {
        return Err(CalcError::invalid(
            "--timeout-ms must be between 1 and 30000.",
        ));
    }
    if cli.jsonl && (!cli.query.is_empty() || cli.action.is_some()) {
        return Err(CalcError::invalid(
            "--jsonl reads requests from stdin and cannot take a calculation or action.",
        ));
    }
    if cli.action.is_some() && !cli.query.is_empty() {
        return Err(CalcError::invalid(
            "A control option cannot be combined with a calculation.",
        ));
    }
    Ok(cli)
}

fn emit(response: &Response, json: bool, explain: bool) -> Result<(), io::Error> {
    if json {
        let mut stdout = io::stdout().lock();
        serde_json::to_writer(&mut stdout, response)?;
        writeln!(stdout)?;
    } else if let Some(result) = &response.result {
        writeln!(io::stdout().lock(), "{}", result.display)?;
    } else if let Some(error) = &response.error {
        writeln!(
            io::stderr().lock(),
            "calc: {} ({})",
            error.message,
            error.code
        )?;
    }
    if explain {
        writeln!(
            io::stderr().lock(),
            "expression: {}\nattempts: {}; total: {:.2} ms; model: {:.2} ms; math: {:.2} ms",
            response.expression.as_deref().unwrap_or("none"),
            response.attempts,
            response.timing.total_ms,
            response.timing.model_ms,
            response.timing.compute_ms
        )?;
    }
    Ok(())
}

fn jsonl(cli: &Cli) -> Result<i32, CalcError> {
    let stdin = io::stdin();
    let mut reader = stdin.lock();
    let mut exit = 0;
    loop {
        let line = match service::bounded_line(&mut reader, 64 * 1024) {
            Ok(None) => break,
            Ok(Some(line)) => Ok(line),
            Err(e) if e.code == "resource_limit" => Err(e),
            Err(e) => return Err(e),
        };
        let mut rejected_request = Request::new("");
        let request = line.and_then(|line| {
            let mut value: serde_json::Value = serde_json::from_slice(&line)
                .map_err(|e| CalcError::invalid(format!("Invalid JSONL request: {e}")))?;
            if let Some(s) = value.as_str() {
                value = serde_json::json!({"input":s});
            }
            if let Some(map) = value.as_object_mut() {
                if let Some(input) = map.get("input").and_then(serde_json::Value::as_str)
                    && input.len() <= 4096
                {
                    rejected_request.input = input.to_owned();
                }
                rejected_request.id = map
                    .get("id")
                    .filter(|id| id.to_string().len() <= 1024)
                    .cloned();
                if let Some(retry) = cli.retry {
                    map.entry("retry").or_insert(retry.into());
                }
                if let Some(timeout) = cli.timeout {
                    map.entry("timeout_ms").or_insert(timeout.into());
                }
            }
            serde_json::from_value::<Request>(value)
                .map_err(|e| CalcError::invalid(format!("Invalid JSONL request: {e}")))
        });
        let response = match request {
            Ok(request) => service::request(&request, &cli.client),
            Err(error) => Response::error(&rejected_request, error),
        };
        exit = exit.max(response.exit_code());
        emit(&response, true, cli.explain).map_err(output_error)?;
    }
    Ok(exit)
}

fn run(args: &[String]) -> Result<i32, CalcError> {
    // These process modes are private, never passed through the model.
    if args.first().is_some_and(|s| s == "--internal-worker") {
        service::worker()?;
        return Ok(0);
    }
    if args.first().is_some_and(|s| s == "--internal-serve") {
        let path = args
            .get(1)
            .ok_or_else(|| CalcError::invalid("Missing internal socket path."))?;
        service::serve(std::path::Path::new(path))?;
        return Ok(0);
    }
    let cli = parse(args)?;
    if let Some(action) = cli.action.as_deref() {
        match action {
            "--help" | "-h" => {
                write!(io::stdout().lock(), "{HELP}").map_err(output_error)?;
                return Ok(0);
            }
            "--version" | "-V" => {
                writeln!(
                    io::stdout().lock(),
                    "calc {} ({})",
                    env!("CARGO_PKG_VERSION"),
                    model::ENGINE_VERSION
                )
                .map_err(output_error)?;
                return Ok(0);
            }
            "--tools" => {
                writeln!(io::stdout().lock(), "{}", model::TOOLS.trim()).map_err(output_error)?;
                return Ok(0);
            }
            "--stop" | "--status" => {
                let status = service::control(&cli.client, action == "--stop")?;
                if cli.json {
                    writeln!(io::stdout().lock(), "{status}").map_err(output_error)?;
                } else {
                    writeln!(
                        io::stdout().lock(),
                        "{}",
                        if status["running"] == true {
                            "Local worker is running."
                        } else {
                            "Local worker is stopped."
                        }
                    )
                    .map_err(output_error)?;
                }
                return Ok(0);
            }
            "--warmup" => {
                let mut request = Request::new("1+1");
                if let Some(n) = cli.retry {
                    request.retry = n;
                }
                if let Some(n) = cli.timeout {
                    request.timeout_ms = n;
                }
                let response = service::request(&request, &cli.client);
                emit(&response, cli.json, cli.explain).map_err(output_error)?;
                return Ok(response.exit_code());
            }
            _ => unreachable!(),
        }
    }
    if cli.jsonl {
        return jsonl(&cli);
    }
    let input = if cli.query.is_empty() {
        if io::stdin().is_terminal() {
            return Err(CalcError::invalid(
                "Enter a calculation, or use calc --help.",
            ));
        }
        let mut data = Vec::new();
        io::stdin()
            .take(4097)
            .read_to_end(&mut data)
            .map_err(output_error)?;
        if data.len() > 4096 {
            return Err(CalcError::limit("Input exceeds 4096 bytes."));
        }
        String::from_utf8(data).map_err(|_| CalcError::invalid("Input must be UTF-8."))?
    } else {
        cli.query.join(" ")
    };
    let mut request = Request::new(input.trim());
    if let Some(n) = cli.retry {
        request.retry = n;
    }
    if let Some(n) = cli.timeout {
        request.timeout_ms = n;
    }
    let response = service::request(&request, &cli.client);
    emit(&response, cli.json, cli.explain).map_err(output_error)?;
    Ok(response.exit_code())
}

fn output_error(e: io::Error) -> CalcError {
    CalcError::new(
        if e.kind() == io::ErrorKind::BrokenPipe {
            "output_closed"
        } else {
            "service_error"
        },
        e.to_string(),
    )
}
fn main() {
    let raw: Vec<std::ffi::OsString> = std::env::args_os().skip(1).collect();
    let json = raw.iter().any(|a| a == "--json" || a == "--jsonl");
    let args: Vec<String> = match raw
        .into_iter()
        .map(|a| a.into_string())
        .collect::<Result<_, _>>()
    {
        Ok(args) => args,
        Err(_) => {
            let response = Response::error(
                &Request::new(""),
                CalcError::invalid("Arguments must be UTF-8."),
            );
            let _ = emit(&response, json, false);
            std::process::exit(2);
        }
    };
    let private_worker = args.first().is_some_and(|argument| {
        matches!(argument.as_str(), "--internal-worker" | "--internal-serve")
    });
    let code = match run(&args) {
        Ok(code) => code,
        Err(error) if error.code == "output_closed" => 0,
        Err(error) => {
            let response = Response::error(&Request::new(""), error);
            // Broken pipes are normal when a consumer exits early.
            if let Err(e) = emit(&response, json, false)
                && e.kind() == io::ErrorKind::BrokenPipe
            {
                std::process::exit(0);
            }
            response.exit_code()
        }
    };
    if private_worker {
        // Rust request/socket cleanup and explicit writes have completed. The
        // native library can register C exit handlers whose code was unloaded
        // after failed initialization; do not run them in disposable workers.
        unsafe { libc::_exit(code) };
    }
    std::process::exit(code);
}

#[cfg(test)]
mod tests {
    use super::*;
    fn p(args: &[&str]) -> Result<Cli, CalcError> {
        parse(&args.iter().map(|s| s.to_string()).collect::<Vec<_>>())
    }
    #[test]
    fn flags_work_after_unquoted_input() {
        let c = p(&["what", "is", "1", "+", "1", "--json", "--retry", "0"]).unwrap();
        assert_eq!(c.query.join(" "), "what is 1 + 1");
        assert!(c.json);
        assert_eq!(c.retry, Some(0));
    }
    #[test]
    fn negative_inputs_and_option_terminator() {
        assert_eq!(p(&["-1", "+", "2"]).unwrap().query[0], "-1");
        assert_eq!(p(&["--", "--retry"]).unwrap().query[0], "--retry");
        assert!(p(&["--retry", "-1"]).is_err());
        assert!(p(&["--retry", "21"]).is_err());
        assert!(p(&["--json=false"]).is_err());
    }
}
