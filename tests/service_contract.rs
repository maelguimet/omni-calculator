//! Real CLI/IPC checks use private sockets and owned child processes. Fake peers
//! deliberately fail or delay transport; none of these tests needs model weights.
use omni_calc::{
    Calculation,
    protocol::{Request, Response, Timing},
};
use serde_json::{Value, json};
use std::{
    fs,
    hash::{Hash, Hasher},
    io::{BufRead, BufReader, Write},
    os::{
        fd::{AsRawFd, FromRawFd, OwnedFd},
        unix::{
            fs::{MetadataExt, PermissionsExt},
            net::{UnixListener, UnixStream},
        },
    },
    path::{Path, PathBuf},
    process::{Child, Command, Output, Stdio},
    thread,
    time::{Duration, Instant},
};

struct Runtime {
    directory: tempfile::TempDir,
    weights: PathBuf,
    socket: PathBuf,
}

impl Runtime {
    fn new() -> Self {
        let directory = tempfile::Builder::new()
            .prefix("omni-ipc-")
            .tempdir()
            .unwrap();
        fs::set_permissions(directory.path(), fs::Permissions::from_mode(0o700)).unwrap();
        let weights = directory.path().join("invalid.cact");
        fs::write(&weights, b"not model weights").unwrap();
        // Locate this CLI configuration's private endpoint so fake peers can
        // exercise transport failures without introducing test hooks into calc.
        let exe = Path::new(env!("CARGO_BIN_EXE_calc"))
            .canonicalize()
            .unwrap();
        let meta = exe.metadata().unwrap();
        let mut hash = std::collections::hash_map::DefaultHasher::new();
        exe.hash(&mut hash);
        (
            meta.dev(),
            meta.ino(),
            meta.len(),
            meta.mtime(),
            meta.mtime_nsec(),
        )
            .hash(&mut hash);
        2_u32.hash(&mut hash);
        true.hash(&mut hash);
        let socket = directory
            .path()
            .join(format!("{:016x}.sock", hash.finish()));
        Self {
            directory,
            weights,
            socket,
        }
    }

    fn command(&self) -> Command {
        let mut command = Command::new(env!("CARGO_BIN_EXE_calc"));
        command
            .env("OMNI_RUNTIME_DIR", self.directory.path())
            .env_remove("OMNI_MODEL")
            .env("OMNI_BACKEND", "needle")
            .env("OMNI_THREADS", "2")
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        command
    }

    fn run(&self, args: &[&str]) -> Output {
        let mut command = self.command();
        command.args(args);
        run_bounded(command, Duration::from_secs(3))
    }

    fn listener(&self) -> UnixListener {
        UnixListener::bind(&self.socket).unwrap()
    }

    fn worker(&self) -> OwnedChild {
        let mut command = self.command();
        command
            .arg("--internal-serve")
            .arg(&self.socket)
            // This deliberately failing backend exercises worker lifecycle
            // without model inference. Clients use a valid default config;
            // explicit bad model paths have separate preflight tests below.
            .env("OMNI_MODEL", &self.weights)
            .stdout(Stdio::null())
            .stderr(Stdio::null());
        let mut child = OwnedChild(Some(command.spawn().unwrap()));
        let deadline = Instant::now() + Duration::from_secs(2);
        while !self.socket.exists() {
            assert!(
                child.0.as_mut().unwrap().try_wait().unwrap().is_none(),
                "worker exited at startup"
            );
            assert!(
                Instant::now() < deadline,
                "worker did not bind private socket"
            );
            thread::sleep(Duration::from_millis(2));
        }
        child
    }
}

struct OwnedChild(Option<Child>);
impl Drop for OwnedChild {
    fn drop(&mut self) {
        if let Some(mut child) = self.0.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

fn run_bounded(mut command: Command, limit: Duration) -> Output {
    wait_bounded(OwnedChild(Some(command.spawn().unwrap())), limit)
}

fn wait_bounded(mut child: OwnedChild, limit: Duration) -> Output {
    let deadline = Instant::now() + limit;
    while child.0.as_mut().unwrap().try_wait().unwrap().is_none() {
        assert!(
            Instant::now() < deadline,
            "CLI did not finish within {} ms",
            limit.as_millis()
        );
        thread::sleep(Duration::from_millis(2));
    }
    child.0.take().unwrap().wait_with_output().unwrap()
}

fn one_json(output: &Output) -> Value {
    assert!(
        output.stderr.is_empty(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let text = std::str::from_utf8(&output.stdout).unwrap();
    assert_eq!(text.lines().count(), 1, "{text}");
    serde_json::from_str(text).unwrap()
}

fn accept(listener: &UnixListener) -> UnixStream {
    listener.set_nonblocking(true).unwrap();
    let deadline = Instant::now() + Duration::from_secs(2);
    loop {
        match listener.accept() {
            Ok((stream, _)) => {
                stream
                    .set_read_timeout(Some(Duration::from_secs(1)))
                    .unwrap();
                stream
                    .set_write_timeout(Some(Duration::from_secs(1)))
                    .unwrap();
                return stream;
            }
            Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                assert!(Instant::now() < deadline, "no client reached fake peer");
                thread::sleep(Duration::from_millis(2));
            }
            Err(error) => panic!("accept: {error}"),
        }
    }
}

fn read_request(stream: &UnixStream) -> Request {
    let mut line = String::new();
    BufReader::new(stream).read_line(&mut line).unwrap();
    serde_json::from_str(&line).unwrap()
}

fn answer(request: &Request) -> Response {
    Response {
        schema_version: 1,
        ok: true,
        id: request.id.clone(),
        input: request.input.clone(),
        expression: Some("1+1".into()),
        tool_call: None,
        result: Some(Calculation::text("2", "number", true)),
        error: None,
        attempts: 1,
        retries: 0,
        timing: Timing::default(),
        model: "test-peer".into(),
        confidence: None,
    }
}

#[test]
fn transport_failure_retries_preserve_request_and_budget() {
    let runtime = Runtime::new();
    let listener = runtime.listener();
    let peer = thread::spawn(move || {
        let first = accept(&listener);
        let first_request = read_request(&first);
        drop(first);
        let mut second = accept(&listener);
        let second_request = read_request(&second);
        serde_json::to_writer(&mut second, &answer(&second_request)).unwrap();
        second.write_all(b"\n").unwrap();
        (first_request, second_request)
    });
    let mut command = runtime.command();
    command
        .args(["--jsonl", "--retry", "3", "--timeout-ms", "1000"])
        .stdin(Stdio::piped());
    let mut child = OwnedChild(Some(command.spawn().unwrap()));
    child
        .0
        .as_mut()
        .unwrap()
        .stdin
        .take()
        .unwrap()
        .write_all(b"{\"id\":\"kept\",\"input\":\"1+1\"}\n")
        .unwrap();
    let output = wait_bounded(child, Duration::from_secs(3));
    let response = one_json(&output);
    let (first, second) = peer.join().unwrap();
    assert!(output.status.success(), "{response}");
    assert_eq!(response["result"]["display"], "2");
    assert_eq!(response["id"], "kept");
    assert_eq!(response["attempts"], 2);
    assert_eq!(response["retries"], 1);
    assert_eq!((first.retry, second.retry), (3, 2));
    assert_eq!(first.id, second.id);
    assert_eq!(first.input, second.input);
    assert!(second.timeout_ms <= first.timeout_ms && second.timeout_ms > 0);
}

#[test]
fn transport_retry_exhaustion_is_bounded_and_terminal() {
    let runtime = Runtime::new();
    let listener = runtime.listener();
    let peer = thread::spawn(move || {
        for retries in (0..=3).rev() {
            let stream = accept(&listener);
            assert_eq!(read_request(&stream).retry, retries);
        }
    });
    let output = runtime.run(&["1+1", "--json", "--retry", "3", "--timeout-ms", "1000"]);
    let response = one_json(&output);
    peer.join().unwrap();
    assert_eq!(output.status.code(), Some(4), "{response}");
    assert_eq!(response["error"]["code"], "model_error");
    assert_eq!(response["error"]["retryable"], false);
    assert_eq!(response["attempts"], 4);
    assert_eq!(response["retries"], 3);
}

#[test]
fn total_timeout_is_not_reset_by_fragmented_worker_output() {
    let runtime = Runtime::new();
    let listener = runtime.listener();
    let peer = thread::spawn(move || {
        let mut stream = accept(&listener);
        let request = read_request(&stream);
        let data = serde_json::to_vec(&answer(&request)).unwrap();
        for byte in data.iter().take(50) {
            if stream.write_all(&[*byte]).is_err() {
                break;
            }
            thread::sleep(Duration::from_millis(10));
        }
    });
    let start = Instant::now();
    let output = runtime.run(&["1+1", "--json", "--retry", "3", "--timeout-ms", "50"]);
    let elapsed = start.elapsed();
    peer.join().unwrap();
    let response = one_json(&output);
    assert_eq!(response["error"]["code"], "timeout", "{response}");
    assert!(
        elapsed < Duration::from_millis(250),
        "50 ms budget took {elapsed:?}"
    );
}

#[test]
fn full_socket_backlog_obeys_timeout_without_replacing_live_socket() {
    let runtime = Runtime::new();
    let listener = runtime.listener();
    assert_eq!(unsafe { libc::listen(listener.as_raw_fd(), 1) }, 0);
    // Linux permits backlog+1 pending AF_UNIX connections. No accepts occur.
    let _queued_a = UnixStream::connect(&runtime.socket).unwrap();
    let _queued_b = UnixStream::connect(&runtime.socket).unwrap();
    let inode = runtime.socket.metadata().unwrap().ino();
    let mut command = runtime.command();
    command.args(["1+1", "--json", "--retry", "0", "--timeout-ms", "50"]);
    let output = run_bounded(command, Duration::from_millis(400));
    let response = one_json(&output);
    assert_eq!(response["error"]["code"], "timeout", "{response}");
    assert_eq!(
        runtime.socket.metadata().unwrap().ino(),
        inode,
        "live socket was replaced"
    );
}

#[test]
fn daemon_reuses_process_and_stop_removes_its_socket() {
    let runtime = Runtime::new();
    let mut worker = runtime.worker();
    let pid = worker.0.as_ref().unwrap().id();
    for _ in 0..2 {
        let status = one_json(&runtime.run(&["--status", "--json"]));
        assert_eq!(status["pid"], pid);
        assert_eq!(status["running"], true);
        assert_eq!(
            status["loaded"], false,
            "failed forced Needle initialization is not a ready model"
        );
        let result =
            one_json(&runtime.run(&["1+1", "--json", "--retry", "0", "--timeout-ms", "1000"]));
        assert_eq!(
            result["ok"], false,
            "invalid weights must not compute: {result}"
        );
    }
    let stopped = one_json(&runtime.run(&["--stop", "--json"]));
    assert_eq!(stopped["running"], false);
    let deadline = Instant::now() + Duration::from_secs(1);
    while worker.0.as_mut().unwrap().try_wait().unwrap().is_none() {
        assert!(
            Instant::now() < deadline,
            "--stop did not terminate own worker; socket exists={}, wait={:?}",
            runtime.socket.exists(),
            fs::read_to_string(format!("/proc/{pid}/wchan"))
        );
        thread::sleep(Duration::from_millis(2));
    }
    assert!(!runtime.socket.exists());
    assert_eq!(
        one_json(&runtime.run(&["--status", "--json"]))["running"],
        false
    );
}

#[test]
fn client_timeout_does_not_poison_next_daemon_request() {
    let runtime = Runtime::new();
    let worker = runtime.worker();
    let pid = worker.0.as_ref().unwrap().id();
    let mut blocker = UnixStream::connect(&runtime.socket).unwrap();
    blocker.write_all(b"{").unwrap();
    // Confirm the incomplete first request is accepted ahead of the CLI request.
    thread::sleep(Duration::from_millis(15));
    let start = Instant::now();
    let response = one_json(&runtime.run(&["1+1", "--json", "--retry", "0", "--timeout-ms", "50"]));
    assert_eq!(response["error"]["code"], "timeout");
    assert!(start.elapsed() < Duration::from_millis(250));
    drop(blocker);
    let status = one_json(&runtime.run(&["--status", "--json"]));
    assert_eq!(
        status["pid"], pid,
        "a request timeout replaced a live worker"
    );
    assert_eq!(status["running"], true);
}

#[test]
fn disconnected_status_client_does_not_kill_shared_worker() {
    let runtime = Runtime::new();
    let mut worker = runtime.worker();
    let pid = worker.0.as_ref().unwrap().id();
    let mut stream = UnixStream::connect(&runtime.socket).unwrap();
    stream.write_all(b"{\"control\":\"status\"}\n").unwrap();
    stream.shutdown(std::net::Shutdown::Both).unwrap();
    drop(stream);
    thread::sleep(Duration::from_millis(25));
    assert!(
        worker.0.as_mut().unwrap().try_wait().unwrap().is_none(),
        "status client's closed pipe killed worker"
    );
    assert_eq!(one_json(&runtime.run(&["--status", "--json"]))["pid"], pid);
}

#[test]
fn daemon_cleanup_does_not_unlink_a_replacement_socket() {
    let runtime = Runtime::new();
    let mut worker = runtime.worker();
    let mut original = UnixStream::connect(&runtime.socket).unwrap();
    fs::remove_file(&runtime.socket).unwrap();
    let replacement = runtime.listener();
    let inode = runtime.socket.metadata().unwrap().ino();
    original.write_all(b"{\"control\":\"stop\"}\n").unwrap();
    let deadline = Instant::now() + Duration::from_secs(1);
    while worker.0.as_mut().unwrap().try_wait().unwrap().is_none() {
        assert!(Instant::now() < deadline);
        thread::sleep(Duration::from_millis(2));
    }
    assert_eq!(fs::metadata(&runtime.socket).unwrap().ino(), inode);
    drop(replacement);
}

#[test]
fn missing_model_is_explicit_in_fresh_and_daemon_modes() {
    let runtime = Runtime::new();
    let missing = runtime.directory.path().join("missing.cact");
    for fresh in [false, true] {
        let mut command = runtime.command();
        command
            .args(["1+1", "--json", "--retry", "3"])
            .arg("--model")
            .arg(&missing);
        if fresh {
            command.arg("--no-daemon");
        }
        let response = one_json(&run_bounded(command, Duration::from_secs(3)));
        assert_eq!(response["ok"], false);
        assert_eq!(response["error"]["code"], "model_unavailable", "{response}");
        assert_eq!(response["error"]["retryable"], false);
        assert!(response["result"].is_null());
    }
    assert!(!runtime.socket.exists());
}

#[test]
fn malformed_jsonl_records_recover_without_losing_next_request() {
    let runtime = Runtime::new();
    let mut input = b"not json\n".to_vec();
    input.extend(vec![b'x'; 128 * 1024]);
    input.extend(b"\n\xff\n");
    input.extend(
        serde_json::to_vec(&json!({"id":"valid-shape", "input":"1+1", "retry":0})).unwrap(),
    );
    input.push(b'\n');
    input.extend(b"{\"id\":99,\"input\":\"\"}\n");
    let mut command = runtime.command();
    command
        .args(["--jsonl", "--no-daemon"])
        .arg("--model")
        .arg(&runtime.weights)
        .stdin(Stdio::piped());
    let mut child = OwnedChild(Some(command.spawn().unwrap()));
    child
        .0
        .as_mut()
        .unwrap()
        .stdin
        .take()
        .unwrap()
        .write_all(&input)
        .unwrap();
    let output = wait_bounded(child, Duration::from_secs(3));
    assert!(output.stderr.is_empty());
    let values: Vec<Value> = std::str::from_utf8(&output.stdout)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert_eq!(values.len(), 5);
    assert_eq!(values[0]["error"]["code"], "invalid_input");
    assert_eq!(values[1]["error"]["code"], "resource_limit");
    assert_eq!(values[2]["error"]["code"], "invalid_input");
    assert_eq!(values[3]["id"], "valid-shape");
    assert_eq!(values[3]["input"], "1+1");
    assert!(
        values[3]["error"]["message"]
            .as_str()
            .is_some_and(|message| !message.contains("JSONL"))
    );
    assert_eq!(values[4]["id"], 99);
    assert!(values.iter().all(|response| response["ok"] == false));
}

#[test]
fn mismatched_and_inconsistent_worker_responses_are_not_successes() {
    for case in 0..4 {
        let runtime = Runtime::new();
        let listener = runtime.listener();
        let peer = thread::spawn(move || {
            let mut stream = accept(&listener);
            let request = read_request(&stream);
            let mut response = answer(&request);
            match case {
                0 => response.input = "different request".into(),
                1 => response.id = Some(json!("different id")),
                2 => response.attempts = u32::MAX,
                _ => response.result = None,
            }
            serde_json::to_writer(&mut stream, &response).unwrap();
            stream.write_all(b"\n").unwrap();
        });
        let output = runtime.run(&["1+1", "--json", "--retry", "0"]);
        peer.join().unwrap();
        let response = one_json(&output);
        assert_eq!(output.status.code(), Some(4), "case {case}: {response}");
        assert_eq!(response["error"]["code"], "model_error");
        assert_eq!(response["attempts"], 1);
        assert!(response["result"].is_null());
    }
}

#[test]
fn malformed_control_response_returns_json_error_without_panicking() {
    let runtime = Runtime::new();
    let listener = runtime.listener();
    let peer = thread::spawn(move || {
        let mut stream = accept(&listener);
        let mut line = String::new();
        BufReader::new(&stream).read_line(&mut line).unwrap();
        assert_eq!(line, "{\"control\":\"status\"}\n");
        stream.write_all(b"42\n").unwrap();
    });
    let output = runtime.run(&["--status", "--json"]);
    peer.join().unwrap();
    assert_eq!(output.status.code(), Some(6));
    assert_eq!(one_json(&output)["error"]["code"], "service_error");
}

#[test]
fn unsafe_runtime_files_are_rejected_without_replacement() {
    let runtime = Runtime::new();
    fs::write(&runtime.socket, b"keep this ordinary file").unwrap();
    let response = one_json(&runtime.run(&["1+1", "--json"]));
    assert_eq!(response["error"]["code"], "service_error");
    assert_eq!(
        fs::read(&runtime.socket).unwrap(),
        b"keep this ordinary file"
    );
    fs::remove_file(&runtime.socket).unwrap();
    fs::set_permissions(runtime.directory.path(), fs::Permissions::from_mode(0o755)).unwrap();
    let response = one_json(&runtime.run(&["1+1", "--json"]));
    assert_eq!(response["error"]["code"], "service_error");
    assert!(!runtime.socket.exists());
}

#[test]
fn concurrent_daemon_clients_keep_their_response_ids() {
    let runtime = Runtime::new();
    let worker = runtime.worker();
    let mut clients = Vec::new();
    for id in 0..6 {
        let path = runtime.socket.clone();
        clients.push(thread::spawn(move || {
            let mut stream = UnixStream::connect(path).unwrap();
            stream
                .set_read_timeout(Some(Duration::from_secs(1)))
                .unwrap();
            // Invalid requests must remain isolated too, without loading a model.
            let mut request = Request::new("");
            request.id = Some(json!(id));
            serde_json::to_writer(&mut stream, &request).unwrap();
            stream.write_all(b"\n").unwrap();
            let mut line = String::new();
            BufReader::new(stream).read_line(&mut line).unwrap();
            let response: Value = serde_json::from_str(&line).unwrap();
            assert_eq!(response["id"], id);
            assert_eq!(response["error"]["code"], "invalid_input");
        }));
    }
    for client in clients {
        client.join().unwrap();
    }
    assert_eq!(
        one_json(&runtime.run(&["--status", "--json"]))["pid"],
        worker.0.as_ref().unwrap().id()
    );
}

#[test]
fn closing_stdout_is_quiet_and_does_not_panic() {
    let runtime = Runtime::new();
    for argument in ["--help", "--version", "--tools"] {
        let mut descriptors = [0; 2];
        assert_eq!(
            unsafe { libc::pipe2(descriptors.as_mut_ptr(), libc::O_CLOEXEC) },
            0
        );
        let reader = unsafe { OwnedFd::from_raw_fd(descriptors[0]) };
        let writer = unsafe { OwnedFd::from_raw_fd(descriptors[1]) };
        drop(reader); // The consumer is already gone before the CLI starts.
        let mut command = runtime.command();
        command.arg(argument).stdout(Stdio::from(writer));
        let output = run_bounded(command, Duration::from_secs(1));
        assert!(
            output.status.success(),
            "{argument}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(output.stderr.is_empty());
    }
}
