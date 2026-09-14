use crate::{
    error::CalcError,
    model::LocalModel,
    planner,
    protocol::{Request, Response},
};
use std::{
    fs::{self, OpenOptions},
    hash::{Hash, Hasher},
    io::{self, BufRead, BufReader, Read, Write},
    os::{
        fd::{AsRawFd, FromRawFd, OwnedFd, RawFd},
        unix::{
            ffi::OsStrExt,
            fs::{MetadataExt, OpenOptionsExt, PermissionsExt},
            net::{UnixListener, UnixStream},
            process::CommandExt,
        },
    },
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    sync::mpsc,
    time::{Duration, Instant},
};

const WIRE_LIMIT: usize = 64 * 1024;
const IDLE_MS: i32 = 120_000;

#[derive(Debug, Clone)]
pub struct ClientOptions {
    pub weights: Option<PathBuf>,
    pub threads: u32,
    pub no_daemon: bool,
    pub force_needle: bool,
}
impl Default for ClientOptions {
    fn default() -> Self {
        Self {
            weights: std::env::var_os("OMNI_MODEL").map(PathBuf::from),
            threads: std::env::var("OMNI_THREADS")
                .ok()
                .and_then(|x| x.parse().ok())
                .unwrap_or(2),
            no_daemon: false,
            force_needle: std::env::var("OMNI_BACKEND").is_ok_and(|s| s == "needle"),
        }
    }
}

pub fn request(request: &Request, options: &ClientOptions) -> Response {
    let start = Instant::now();
    if let Err(error) = request.validate() {
        return Response::error(request, error);
    }
    if let Err(error) = validate_options(options) {
        return Response::error(request, error);
    }
    let budget = Duration::from_millis(request.timeout_ms);
    let mut reply = Response::error(
        request,
        CalcError::new("timeout", "Calculation exceeded its time limit."),
    );
    for failures in 0..=request.retry {
        let remaining = budget.saturating_sub(start.elapsed());
        if remaining.as_millis() == 0 {
            reply.ok = false;
            reply.result = None;
            reply.error = Some(CalcError::new(
                "timeout",
                "Calculation exceeded its time limit.",
            ));
            break;
        }
        let mut pending = request.clone();
        pending.retry = request.retry - failures;
        pending.timeout_ms = remaining.as_millis() as u64;
        let result = if options.no_daemon {
            request_once(&pending, options)
        } else {
            request_daemon(&pending, options)
        }
        .and_then(|response| validate_response(&pending, response));
        match result {
            Ok(mut response) => {
                response.attempts += failures;
                response.retries = response.attempts.saturating_sub(1);
                reply = response;
                break;
            }
            Err(error) => {
                let retryable = error.retryable;
                reply = Response::error(request, error);
                reply.attempts = failures + 1;
                reply.retries = failures;
                if !retryable {
                    break;
                }
            }
        }
    }
    if start.elapsed() >= budget {
        reply.ok = false;
        reply.result = None;
        reply.error = Some(CalcError::new(
            "timeout",
            "Calculation exceeded its time limit.",
        ));
    }
    if let Some(error) = &mut reply.error {
        error.retryable = false;
    }
    // End-to-end timing includes startup, model load, local IPC and calculation.
    reply.timing.total_ms = start.elapsed().as_secs_f64() * 1000.0;
    reply
}

fn validate_response(request: &Request, response: Response) -> Result<Response, CalcError> {
    if response.schema_version != 1 || response.input != request.input || response.id != request.id
    {
        return Err(CalcError::model(
            "Local worker returned a response for a different request or protocol.",
        ));
    }
    if response.attempts > request.retry + 1
        || response.retries != response.attempts.saturating_sub(1)
        || response.ok != response.result.is_some()
        || response.ok == response.error.is_some()
        || (response.ok && response.attempts == 0)
        || [
            response.timing.total_ms,
            response.timing.model_ms,
            response.timing.compute_ms,
        ]
        .iter()
        .any(|value| !value.is_finite() || *value < 0.0)
    {
        return Err(CalcError::model(
            "Local worker returned an inconsistent calculation response.",
        ));
    }
    Ok(response)
}

fn validate_options(options: &ClientOptions) -> Result<(), CalcError> {
    if !(1..=8).contains(&options.threads) {
        return Err(CalcError::invalid("--threads must be between 1 and 8."));
    }
    if let Some(path) = &options.weights {
        let mut file = OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_NONBLOCK)
            .open(path)
            .map_err(|error| {
                CalcError::new(
                    "model_unavailable",
                    format!("Cannot read weights {}: {error}", path.display()),
                )
            })?;
        let metadata = file.metadata().map_err(io_error)?;
        if !metadata.is_file() || !(4..=256 * 1024 * 1024).contains(&metadata.len()) {
            return Err(CalcError::invalid(
                "Weights must be a Needle 2 .cact file smaller than 256 MiB.",
            ));
        }
        let mut magic = [0; 4];
        file.read_exact(&mut magic).map_err(io_error)?;
        if magic != [0x83, 0x2a, 0xe1, 0x05] {
            return Err(CalcError::invalid(
                "Weights do not have a Needle 2 .cact header.",
            ));
        }
    }
    Ok(())
}

fn command(options: &ClientOptions) -> Result<Command, CalcError> {
    if !(1..=8).contains(&options.threads) {
        return Err(CalcError::invalid("--threads must be between 1 and 8."));
    }
    let exe = std::env::current_exe().map_err(io_error)?;
    let mut command = Command::new(exe);
    command
        .env("NEEDLE_THREADS", options.threads.to_string())
        .env("OMP_NUM_THREADS", options.threads.to_string())
        .env("DO_NOT_TRACK", "1")
        .env("NEEDLE_TELEMETRY", "0")
        .env(
            "OMNI_BACKEND",
            if options.force_needle {
                "needle"
            } else {
                "auto"
            },
        )
        .stderr(Stdio::null());
    if let Some(weights) = &options.weights {
        command.env("OMNI_MODEL", weights);
    } else {
        command.env_remove("OMNI_MODEL");
    }
    Ok(command)
}

fn request_once(request: &Request, options: &ClientOptions) -> Result<Response, CalcError> {
    let deadline = Instant::now() + Duration::from_millis(request.timeout_ms);
    let mut command = command(options)?;
    command
        .arg("--internal-worker")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped());
    let child = command.spawn().map_err(io_error)?;
    let mut guard = ChildGuard(Some(child));
    let child = guard.0.as_mut().unwrap();
    {
        let mut stdin = DeadlineIo::new(child.stdin.take().unwrap(), deadline)?;
        serde_json::to_writer(&mut stdin, request).map_err(json_error)?;
        stdin.write_all(b"\n").map_err(io_error)?;
    }
    let stdout = DeadlineIo::new(child.stdout.take().unwrap(), deadline)?;
    read_response(BufReader::new(stdout))
}
struct ChildGuard(Option<Child>);
impl Drop for ChildGuard {
    fn drop(&mut self) {
        if let Some(mut child) = self.0.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

fn identity(options: &ClientOptions) -> Result<String, CalcError> {
    let exe = std::env::current_exe().map_err(io_error)?;
    let meta = exe.metadata().map_err(io_error)?;
    let mut h = std::collections::hash_map::DefaultHasher::new();
    exe.hash(&mut h);
    (
        meta.dev(),
        meta.ino(),
        meta.len(),
        meta.mtime(),
        meta.mtime_nsec(),
    )
        .hash(&mut h);
    options.threads.hash(&mut h);
    options.force_needle.hash(&mut h);
    if let Some(weights) = &options.weights {
        let path = weights.canonicalize().map_err(|e| {
            CalcError::new(
                "model_unavailable",
                format!("Cannot open weights {}: {e}", weights.display()),
            )
        })?;
        let meta = path.metadata().map_err(io_error)?;
        path.hash(&mut h);
        (meta.ino(), meta.len(), meta.mtime(), meta.mtime_nsec()).hash(&mut h);
    }
    Ok(format!("{:016x}", h.finish()))
}

pub fn socket_path(options: &ClientOptions) -> Result<PathBuf, CalcError> {
    let uid = unsafe { libc::geteuid() };
    let base = std::env::var_os("OMNI_RUNTIME_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(format!("/tmp/omni-calc-{uid}")));
    match fs::create_dir(&base) {
        Ok(()) => {
            fs::set_permissions(&base, fs::Permissions::from_mode(0o700)).map_err(io_error)?
        }
        Err(e) if e.kind() == io::ErrorKind::AlreadyExists => {}
        Err(e) => return Err(io_error(e)),
    }
    let meta = fs::symlink_metadata(&base).map_err(io_error)?;
    if !meta.is_dir() || meta.uid() != uid || meta.permissions().mode() & 0o077 != 0 {
        return Err(CalcError::new(
            "service_error",
            "Runtime directory must be a real directory owned by you with mode 0700.",
        ));
    }
    let socket = base.join(format!("{}.sock", identity(options)?));
    if socket.as_os_str().len() >= 104 {
        return Err(CalcError::new(
            "service_error",
            "Runtime socket path is too long; set OMNI_RUNTIME_DIR to a shorter private directory.",
        ));
    }
    Ok(socket)
}

fn request_daemon(request: &Request, options: &ClientOptions) -> Result<Response, CalcError> {
    let deadline = Instant::now() + Duration::from_millis(request.timeout_ms);
    let socket = socket_path(options)?;
    let stream = match connect_before(&socket, deadline) {
        Ok(stream) => stream,
        Err(error) if unavailable_socket(&error) => {
            let lock_path = socket.with_extension("lock");
            let lock = OpenOptions::new()
                .create(true)
                .truncate(false)
                .read(true)
                .write(true)
                .mode(0o600)
                .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC)
                .open(lock_path)
                .map_err(io_error)?;
            while unsafe { libc::flock(lock.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } != 0 {
                let error = io::Error::last_os_error();
                if !matches!(
                    error.kind(),
                    io::ErrorKind::WouldBlock | io::ErrorKind::Interrupted
                ) {
                    return Err(io_error(error));
                }
                if Instant::now() >= deadline {
                    return Err(CalcError::new(
                        "timeout",
                        "Timed out waiting for worker startup.",
                    ));
                }
                std::thread::sleep(Duration::from_millis(2));
            }
            // Keep the advisory lock until a new worker is accepting connections.
            match connect_before(&socket, deadline) {
                Ok(stream) => stream,
                Err(error) if unavailable_socket(&error) => {
                    if let Ok(meta) = fs::symlink_metadata(&socket) {
                        use std::os::unix::fs::FileTypeExt;
                        if !meta.file_type().is_socket() {
                            return Err(CalcError::new(
                                "service_error",
                                "Refusing to replace a non-socket runtime file.",
                            ));
                        }
                        fs::remove_file(&socket).map_err(io_error)?;
                    }
                    let mut command = command(options)?;
                    command
                        .arg("--internal-serve")
                        .arg(&socket)
                        .stdin(Stdio::null())
                        .stdout(Stdio::null())
                        .process_group(0);
                    let mut child = command.spawn().map_err(io_error)?;
                    loop {
                        match connect_before(&socket, deadline) {
                            Ok(stream) => break stream,
                            Err(error) if unavailable_socket(&error) => {}
                            Err(error) => {
                                let _ = child.kill();
                                let _ = child.wait();
                                return Err(io_error(error));
                            }
                        }
                        if let Some(status) = child.try_wait().map_err(io_error)? {
                            return Err(CalcError::new(
                                "model_unavailable",
                                format!(
                                    "Calculator worker could not start ({status}). Try --no-daemon for a fresh worker."
                                ),
                            ));
                        }
                        if Instant::now() >= deadline {
                            let _ = child.kill();
                            let _ = child.wait();
                            return Err(CalcError::new(
                                "timeout",
                                "Model startup exceeded its time limit.",
                            ));
                        }
                        std::thread::sleep(Duration::from_millis(2));
                    }
                }
                Err(error) => return Err(io_error(error)),
            }
        }
        Err(error) => return Err(io_error(error)),
    };
    verify_peer(&stream)?;
    let mut stream = DeadlineIo::new(stream, deadline)?;
    serde_json::to_writer(&mut stream, request).map_err(json_error)?;
    stream.write_all(b"\n").map_err(io_error)?;
    read_response(BufReader::new(stream))
}

fn unavailable_socket(error: &io::Error) -> bool {
    matches!(
        error.kind(),
        io::ErrorKind::NotFound | io::ErrorKind::ConnectionRefused
    )
}

fn timeout_io() -> io::Error {
    io::Error::new(io::ErrorKind::TimedOut, "local worker deadline exceeded")
}

fn remaining(deadline: Instant) -> io::Result<Duration> {
    deadline
        .checked_duration_since(Instant::now())
        .filter(|time| !time.is_zero())
        .ok_or_else(timeout_io)
}

fn poll_before(fd: RawFd, events: libc::c_short, deadline: Instant) -> io::Result<()> {
    loop {
        let time = remaining(deadline)?;
        let milliseconds = time
            .as_millis()
            .saturating_add(u128::from(time.subsec_nanos() % 1_000_000 != 0));
        let mut pollfd = libc::pollfd {
            fd,
            events,
            revents: 0,
        };
        let ready =
            unsafe { libc::poll(&mut pollfd, 1, milliseconds.min(i32::MAX as u128) as i32) };
        if ready > 0 {
            if pollfd.revents & libc::POLLNVAL != 0 {
                return Err(io::Error::from_raw_os_error(libc::EBADF));
            }
            return Ok(());
        }
        if ready == 0 {
            return Err(timeout_io());
        }
        let error = io::Error::last_os_error();
        if error.kind() != io::ErrorKind::Interrupted {
            return Err(error);
        }
    }
}

/// AF_UNIX connect can block on a full backlog before SO_RCVTIMEO applies.
/// Nonblocking connection attempts distinguish a busy peer from a stale socket.
fn connect_before(path: &Path, deadline: Instant) -> io::Result<UnixStream> {
    let bytes = path.as_os_str().as_bytes();
    let mut address: libc::sockaddr_un = unsafe { std::mem::zeroed() };
    if bytes.len() >= address.sun_path.len() || bytes.contains(&0) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "invalid Unix socket path",
        ));
    }
    address.sun_family = libc::AF_UNIX as libc::sa_family_t;
    for (slot, byte) in address.sun_path.iter_mut().zip(bytes) {
        *slot = *byte as libc::c_char;
    }
    let length =
        (std::mem::offset_of!(libc::sockaddr_un, sun_path) + bytes.len() + 1) as libc::socklen_t;
    loop {
        remaining(deadline)?;
        let raw = unsafe {
            libc::socket(
                libc::AF_UNIX,
                libc::SOCK_STREAM | libc::SOCK_NONBLOCK | libc::SOCK_CLOEXEC,
                0,
            )
        };
        if raw < 0 {
            return Err(io::Error::last_os_error());
        }
        let fd = unsafe { OwnedFd::from_raw_fd(raw) };
        let connected = unsafe {
            libc::connect(
                fd.as_raw_fd(),
                (&address as *const libc::sockaddr_un).cast(),
                length,
            )
        };
        if connected == 0 {
            return Ok(UnixStream::from(fd));
        }
        let error = io::Error::last_os_error();
        match error.raw_os_error() {
            Some(libc::EINPROGRESS) => {
                poll_before(fd.as_raw_fd(), libc::POLLOUT, deadline)?;
                let mut socket_error: libc::c_int = 0;
                let mut size = std::mem::size_of_val(&socket_error) as libc::socklen_t;
                if unsafe {
                    libc::getsockopt(
                        fd.as_raw_fd(),
                        libc::SOL_SOCKET,
                        libc::SO_ERROR,
                        (&mut socket_error as *mut libc::c_int).cast(),
                        &mut size,
                    )
                } < 0
                {
                    return Err(io::Error::last_os_error());
                }
                if socket_error != 0 {
                    return Err(io::Error::from_raw_os_error(socket_error));
                }
                return Ok(UnixStream::from(fd));
            }
            Some(libc::EAGAIN) | Some(libc::EINTR) => {
                drop(fd);
                std::thread::sleep(remaining(deadline)?.min(Duration::from_millis(2)));
            }
            _ => return Err(error),
        }
    }
}

/// Every read/write checks one absolute deadline; progress cannot renew it.
struct DeadlineIo<T> {
    inner: T,
    deadline: Instant,
}
impl<T: AsRawFd> DeadlineIo<T> {
    fn new(inner: T, deadline: Instant) -> Result<Self, CalcError> {
        let flags = unsafe { libc::fcntl(inner.as_raw_fd(), libc::F_GETFL) };
        if flags < 0
            || unsafe { libc::fcntl(inner.as_raw_fd(), libc::F_SETFL, flags | libc::O_NONBLOCK) }
                < 0
        {
            return Err(io_error(io::Error::last_os_error()));
        }
        Ok(Self { inner, deadline })
    }
}
impl<T: Read + AsRawFd> Read for DeadlineIo<T> {
    fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        loop {
            remaining(self.deadline)?;
            match self.inner.read(buffer) {
                Err(error) if error.kind() == io::ErrorKind::WouldBlock => {
                    poll_before(self.inner.as_raw_fd(), libc::POLLIN, self.deadline)?
                }
                Err(error) if error.kind() == io::ErrorKind::Interrupted => {}
                result => return result,
            }
        }
    }
}
impl<T: Write + AsRawFd> Write for DeadlineIo<T> {
    fn write(&mut self, buffer: &[u8]) -> io::Result<usize> {
        loop {
            remaining(self.deadline)?;
            match self.inner.write(buffer) {
                Err(error) if error.kind() == io::ErrorKind::WouldBlock => {
                    poll_before(self.inner.as_raw_fd(), libc::POLLOUT, self.deadline)?
                }
                Err(error) if error.kind() == io::ErrorKind::Interrupted => {}
                result => return result,
            }
        }
    }
    fn flush(&mut self) -> io::Result<()> {
        self.inner.flush()
    }
}

fn verify_peer(stream: &UnixStream) -> Result<(), CalcError> {
    let mut cred: libc::ucred = unsafe { std::mem::zeroed() };
    let mut size = std::mem::size_of::<libc::ucred>() as libc::socklen_t;
    if unsafe {
        libc::getsockopt(
            stream.as_raw_fd(),
            libc::SOL_SOCKET,
            libc::SO_PEERCRED,
            (&mut cred as *mut libc::ucred).cast(),
            &mut size,
        )
    } != 0
        || cred.uid != unsafe { libc::geteuid() }
    {
        return Err(CalcError::new(
            "service_error",
            "Worker peer ownership check failed.",
        ));
    }
    Ok(())
}

fn read_response(mut reader: impl BufRead) -> Result<Response, CalcError> {
    let data=bounded_line(&mut reader,WIRE_LIMIT)?.ok_or_else(||CalcError::model("The calculator worker exited unexpectedly (timeout, resource limit, or native model failure)."))?;
    serde_json::from_slice(&data)
        .map_err(|_| CalcError::model("Invalid response from local worker."))
}

/// Bounded reads recover at the next line, allowing JSONL to continue after bad input.
pub fn bounded_line(reader: &mut impl BufRead, limit: usize) -> Result<Option<Vec<u8>>, CalcError> {
    let mut out = Vec::new();
    let mut overflow = false;
    let mut seen = false;
    loop {
        let buf = reader.fill_buf().map_err(io_error)?;
        if buf.is_empty() {
            if !seen {
                return Ok(None);
            }
            break;
        }
        seen = true;
        let end = buf.iter().position(|&c| c == b'\n');
        let take = end.map_or(buf.len(), |p| p + 1);
        if !overflow && out.len() + take <= limit {
            out.extend_from_slice(&buf[..take]);
        } else {
            overflow = true;
        }
        reader.consume(take);
        if end.is_some() {
            break;
        }
    }
    if overflow {
        return Err(CalcError::limit(format!(
            "Input line exceeds {limit} bytes."
        )));
    }
    Ok(Some(out))
}

fn hard_limits() {
    // Limits affect only this disposable worker, never the parent shell or system.
    unsafe {
        let memory = libc::rlimit {
            rlim_cur: 768 * 1024 * 1024,
            rlim_max: 768 * 1024 * 1024,
        };
        libc::setrlimit(libc::RLIMIT_AS, &memory);
        let core = libc::rlimit {
            rlim_cur: 0,
            rlim_max: 0,
        };
        libc::setrlimit(libc::RLIMIT_CORE, &core);
    }
}
struct Watchdog {
    done: Option<mpsc::Sender<()>>,
    thread: Option<std::thread::JoinHandle<()>>,
}
impl Watchdog {
    fn new(timeout: u64) -> Self {
        let (tx, rx) = mpsc::channel();
        let thread = std::thread::spawn(move || {
            if rx
                .recv_timeout(Duration::from_millis(timeout + 100))
                .is_err()
            {
                // Native inference is not cooperatively cancellable. End only its worker.
                unsafe { libc::_exit(124) };
            }
        });
        Self {
            done: Some(tx),
            thread: Some(thread),
        }
    }
}
impl Drop for Watchdog {
    fn drop(&mut self) {
        if let Some(done) = self.done.take() {
            let _ = done.send(());
        }
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

fn process(model: &mut Option<LocalModel>, request: &Request) -> Response {
    if let Err(error) = request.validate() {
        return Response::error(request, error);
    }
    let _watchdog = Watchdog::new(request.timeout_ms);
    if model.is_none() {
        let weights = std::env::var_os("OMNI_MODEL").map(PathBuf::from);
        let force_needle = std::env::var("OMNI_BACKEND").is_ok_and(|s| s == "needle");
        *model = Some(LocalModel::new(weights, force_needle));
    }
    planner::calculate(model.as_mut().unwrap(), request)
}

pub fn worker() -> Result<(), CalcError> {
    hard_limits();
    let mut stdin = io::stdin().lock();
    let line = bounded_line(&mut stdin, WIRE_LIMIT)?
        .ok_or_else(|| CalcError::invalid("Missing worker request."))?;
    let request: Request = serde_json::from_slice(&line).map_err(json_error)?;
    let response = process(&mut None, &request);
    let mut stdout = io::stdout().lock();
    serde_json::to_writer(&mut stdout, &response).map_err(json_error)?;
    stdout.write_all(b"\n").map_err(io_error)?;
    Ok(())
}

pub fn serve(path: &Path) -> Result<(), CalcError> {
    hard_limits();
    let listener = UnixListener::bind(path).map_err(io_error)?;
    let metadata = fs::symlink_metadata(path).map_err(io_error)?;
    let _cleanup = SocketCleanup {
        path: path.to_owned(),
        dev: metadata.dev(),
        ino: metadata.ino(),
    };
    fs::set_permissions(path, fs::Permissions::from_mode(0o600)).map_err(io_error)?;
    let mut model = None;
    loop {
        let mut pollfd = libc::pollfd {
            fd: listener.as_raw_fd(),
            events: libc::POLLIN,
            revents: 0,
        };
        let ready = unsafe { libc::poll(&mut pollfd, 1, IDLE_MS) };
        if ready == 0 {
            return Ok(());
        }
        if ready < 0 {
            if io::Error::last_os_error().kind() == io::ErrorKind::Interrupted {
                continue;
            }
            return Err(io_error(io::Error::last_os_error()));
        }
        let (stream, _) = listener.accept().map_err(io_error)?;
        if verify_peer(&stream).is_err() {
            continue;
        }
        let mut stream = DeadlineIo::new(stream, Instant::now() + Duration::from_secs(2))?;
        let line = match bounded_line(&mut BufReader::new(&mut stream), WIRE_LIMIT) {
            Ok(Some(line)) => line,
            _ => continue,
        };
        stream.deadline = Instant::now() + Duration::from_secs(2);
        if line == b"{\"control\":\"stop\"}\n" {
            let _ = stream.write_all(b"{\"ok\":true}\n");
            return Ok(());
        }
        if line == b"{\"control\":\"status\"}\n" {
            let status = serde_json::json!({"ok":true,"loaded":model.as_ref().is_some_and(LocalModel::is_ready),
                "needle_loaded":model.as_ref().is_some_and(LocalModel::needle_loaded),"pid":std::process::id()});
            if serde_json::to_writer(&mut stream, &status).is_ok() {
                let _ = stream.write_all(b"\n");
            }
            continue;
        }
        let request: Request = match serde_json::from_slice(&line) {
            Ok(r) => r,
            Err(_) => continue,
        };
        let response = process(&mut model, &request);
        stream.deadline = Instant::now() + Duration::from_secs(2);
        if serde_json::to_writer(&mut stream, &response).is_ok() {
            let _ = stream.write_all(b"\n");
        }
    }
}
struct SocketCleanup {
    path: PathBuf,
    dev: u64,
    ino: u64,
}
impl Drop for SocketCleanup {
    fn drop(&mut self) {
        if fs::symlink_metadata(&self.path)
            .is_ok_and(|metadata| metadata.dev() == self.dev && metadata.ino() == self.ino)
        {
            let _ = fs::remove_file(&self.path);
        }
    }
}

pub fn control(options: &ClientOptions, stop: bool) -> Result<serde_json::Value, CalcError> {
    let deadline = Instant::now() + Duration::from_secs(2);
    let socket = socket_path(options)?;
    let stream = match connect_before(&socket, deadline) {
        Ok(s) => s,
        Err(error) if unavailable_socket(&error) => {
            return Ok(serde_json::json!({"ok":true,"running":false,"loaded":false}));
        }
        Err(error) => return Err(io_error(error)),
    };
    verify_peer(&stream)?;
    let mut stream = DeadlineIo::new(stream, deadline)?;
    stream
        .write_all(if stop {
            b"{\"control\":\"stop\"}\n"
        } else {
            b"{\"control\":\"status\"}\n"
        })
        .map_err(io_error)?;
    let line = bounded_line(&mut BufReader::new(stream), 1024)?
        .ok_or_else(|| CalcError::new("service_error", "Worker closed control connection."))?;
    let mut result: serde_json::Value = serde_json::from_slice(&line).map_err(json_error)?;
    if !result.is_object()
        || result["ok"] != true
        || (!stop && (!result["pid"].is_u64() || !result["loaded"].is_boolean()))
    {
        return Err(CalcError::new(
            "service_error",
            "Invalid control response from local worker.",
        ));
    }
    result["running"] = serde_json::Value::Bool(!stop);
    Ok(result)
}

fn io_error(error: io::Error) -> CalcError {
    if matches!(
        error.kind(),
        io::ErrorKind::TimedOut | io::ErrorKind::WouldBlock
    ) {
        CalcError::new("timeout", "Local calculator worker timed out.")
    } else if matches!(
        error.kind(),
        io::ErrorKind::BrokenPipe | io::ErrorKind::ConnectionReset | io::ErrorKind::UnexpectedEof
    ) {
        CalcError::model("Local model worker disconnected unexpectedly.")
    } else {
        CalcError::new("service_error", error.to_string())
    }
}
fn json_error(error: serde_json::Error) -> CalcError {
    if let Some(kind) = error.io_error_kind() {
        io_error(io::Error::new(kind, error))
    } else {
        CalcError::new("invalid_input", error.to_string())
    }
}
