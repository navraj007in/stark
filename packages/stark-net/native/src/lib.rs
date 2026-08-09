//! Blocking TCP native provider and reusable loopback echo harness.

use std::collections::HashMap;
use std::io::{Read, Write};
use std::net::{IpAddr, Shutdown, SocketAddr, TcpListener, TcpStream, ToSocketAddrs};
use std::sync::mpsc;
use std::sync::{Mutex, OnceLock};
use std::thread::{self, JoinHandle};
use std::time::Duration;

pub use stark_provider_abi::{
    BorrowedBuffer, BorrowedBufferMut, ProviderStatus, RawOsHandle, RawResourceHandle,
};

pub const TCP_LISTENER_RESOURCE_TYPE: u32 = 0;
pub const TCP_STREAM_RESOURCE_TYPE: u32 = 1;

pub const STATUS_CONNECTION_REFUSED: ProviderStatus = ProviderStatus { code: 1 };
pub const STATUS_TIMED_OUT: ProviderStatus = ProviderStatus { code: 2 };
pub const STATUS_NOT_FOUND: ProviderStatus = ProviderStatus { code: 3 };
pub const STATUS_PERMISSION_DENIED: ProviderStatus = ProviderStatus { code: 4 };
pub const STATUS_ADDRESS_IN_USE: ProviderStatus = ProviderStatus { code: 5 };
pub const STATUS_INVALID_INPUT: ProviderStatus = ProviderStatus { code: 6 };
pub const STATUS_CONNECTION_RESET: ProviderStatus = ProviderStatus { code: 7 };
pub const STATUS_BROKEN_PIPE: ProviderStatus = ProviderStatus { code: 8 };
pub const STATUS_WOULD_BLOCK: ProviderStatus = ProviderStatus { code: 9 };
pub const STATUS_UNSUPPORTED: ProviderStatus = ProviderStatus { code: 10 };
pub const STATUS_OTHER_DECLARED: ProviderStatus = ProviderStatus { code: 11 };
pub const STATUS_DNS_INVALID_HOST: ProviderStatus = ProviderStatus { code: 101 };
pub const STATUS_DNS_NOT_FOUND: ProviderStatus = ProviderStatus { code: 102 };
pub const STATUS_DNS_TEMPORARY_FAILURE: ProviderStatus = ProviderStatus { code: 103 };
pub const STATUS_DNS_TOO_MANY_RESULTS: ProviderStatus = ProviderStatus { code: 104 };
pub const STATUS_DNS_UNSUPPORTED_ADDRESS_FAMILY: ProviderStatus = ProviderStatus { code: 105 };
pub const STATUS_DNS_UNSUPPORTED: ProviderStatus = ProviderStatus { code: 106 };
pub const STATUS_DNS_OTHER: ProviderStatus = ProviderStatus { code: 107 };

const DNS_RECORD_WIDTH: usize = 22;
const DNS_FAMILY_IPV4: u8 = 4;
const DNS_FAMILY_IPV6: u8 = 6;

struct Table {
    next: u64,
    listeners: HashMap<u64, TcpListener>,
    streams: HashMap<u64, TcpStream>,
}

static TABLE: OnceLock<Mutex<Table>> = OnceLock::new();

fn table() -> &'static Mutex<Table> {
    TABLE.get_or_init(|| {
        Mutex::new(Table {
            next: 1,
            listeners: HashMap::new(),
            streams: HashMap::new(),
        })
    })
}

fn abort_contract() -> ! {
    std::process::abort()
}

unsafe fn read_buffer(buffer: BorrowedBuffer) -> &'static [u8] {
    if buffer.len == 0 {
        return &[];
    }
    if buffer.ptr.is_null() {
        abort_contract();
    }
    unsafe { std::slice::from_raw_parts(buffer.ptr, buffer.len) }
}

unsafe fn write_scalar<T: Copy>(out: *mut T, value: T) {
    if out.is_null() {
        abort_contract();
    }
    unsafe {
        *out = value;
    }
}

fn address_from_buffer(buffer: BorrowedBuffer) -> Result<String, ProviderStatus> {
    let bytes = unsafe { read_buffer(buffer) };
    if bytes.is_empty() || bytes.contains(&0) {
        return Err(STATUS_INVALID_INPUT);
    }
    std::str::from_utf8(bytes)
        .map(str::to_owned)
        .map_err(|_| STATUS_INVALID_INPUT)
}

fn hostname_from_buffer(buffer: BorrowedBuffer) -> Result<String, ProviderStatus> {
    let bytes = unsafe { read_buffer(buffer) };
    if bytes.is_empty() || bytes.contains(&0) {
        return Err(STATUS_DNS_INVALID_HOST);
    }
    if bytes.len() > 253 {
        return Err(STATUS_DNS_INVALID_HOST);
    }
    std::str::from_utf8(bytes)
        .map(str::to_owned)
        .map_err(|_| STATUS_DNS_INVALID_HOST)
}

fn map_io_error(error: &std::io::Error) -> ProviderStatus {
    match error.kind() {
        std::io::ErrorKind::ConnectionRefused => STATUS_CONNECTION_REFUSED,
        std::io::ErrorKind::TimedOut => STATUS_TIMED_OUT,
        std::io::ErrorKind::NotFound => STATUS_NOT_FOUND,
        std::io::ErrorKind::PermissionDenied => STATUS_PERMISSION_DENIED,
        std::io::ErrorKind::AddrInUse => STATUS_ADDRESS_IN_USE,
        std::io::ErrorKind::InvalidInput | std::io::ErrorKind::InvalidData => STATUS_INVALID_INPUT,
        std::io::ErrorKind::ConnectionReset => STATUS_CONNECTION_RESET,
        std::io::ErrorKind::BrokenPipe => STATUS_BROKEN_PIPE,
        std::io::ErrorKind::WouldBlock => STATUS_WOULD_BLOCK,
        std::io::ErrorKind::Unsupported => STATUS_UNSUPPORTED,
        _ => STATUS_OTHER_DECLARED,
    }
}

/// The same mapping for a STREAM read or write, where `WouldBlock` means one specific thing.
///
/// **DEV-163.** A provider stream is always blocking; the only non-blocking socket in this file is
/// the test harness's listener. So `read`/`write` can return `WouldBlock` for exactly one reason:
/// `SO_RCVTIMEO`/`SO_SNDTIMEO` expired. Unix reports that as `EAGAIN` (`WouldBlock`) and Windows
/// reports it as `WSAETIMEDOUT` (`TimedOut`) — the same event, two error kinds.
///
/// Passing both through unchanged made a configured `read_timeout` surface as
/// `NetworkError::Interrupted` on Unix and `NetworkError::TimedOut` on Windows, so
/// `stark-http-client` reported "the connection failed" on macOS and Linux and "timed out reading
/// the response" on Windows — for the identical peer. Found by HC13's stalling peer; invisible to
/// every test that used a peer which answers.
fn map_stream_io_error(error: &std::io::Error) -> ProviderStatus {
    match error.kind() {
        std::io::ErrorKind::WouldBlock => STATUS_TIMED_OUT,
        _ => map_io_error(error),
    }
}

fn map_dns_error(error: &std::io::Error) -> ProviderStatus {
    match error.kind() {
        std::io::ErrorKind::InvalidInput | std::io::ErrorKind::InvalidData => {
            STATUS_DNS_INVALID_HOST
        }
        std::io::ErrorKind::NotFound => STATUS_DNS_NOT_FOUND,
        std::io::ErrorKind::TimedOut
        | std::io::ErrorKind::Interrupted
        | std::io::ErrorKind::WouldBlock => STATUS_DNS_TEMPORARY_FAILURE,
        std::io::ErrorKind::Unsupported => STATUS_DNS_UNSUPPORTED,
        _ => STATUS_DNS_OTHER,
    }
}

fn resolve_records(host: &str) -> Result<Vec<[u8; DNS_RECORD_WIDTH]>, ProviderStatus> {
    let mut records = Vec::new();
    let addrs = (host, 0u16)
        .to_socket_addrs()
        .map_err(|error| map_dns_error(&error))?;
    for addr in addrs {
        let mut record = [0u8; DNS_RECORD_WIDTH];
        match addr.ip() {
            IpAddr::V4(v4) => {
                record[0] = DNS_FAMILY_IPV4;
                record[1] = 4;
                record[2..6].copy_from_slice(&v4.octets());
            }
            IpAddr::V6(v6) => {
                record[0] = DNS_FAMILY_IPV6;
                record[1] = 16;
                record[2..18].copy_from_slice(&v6.octets());
            }
        }
        records.push(record);
    }
    if records.is_empty() {
        return Err(STATUS_DNS_NOT_FOUND);
    }
    Ok(records)
}

fn next_id(table: &mut Table) -> u64 {
    let id = table.next;
    table.next = table
        .next
        .checked_add(1)
        .unwrap_or_else(|| abort_contract());
    id
}

fn validate(handle: RawResourceHandle, resource_type: u32) {
    if handle.resource_type != resource_type {
        abort_contract();
    }
}

fn insert_listener(listener: TcpListener) -> RawResourceHandle {
    let mut table = table().lock().unwrap_or_else(|_| abort_contract());
    let id = next_id(&mut table);
    table.listeners.insert(id, listener);
    RawResourceHandle {
        id,
        resource_type: TCP_LISTENER_RESOURCE_TYPE,
    }
}

fn insert_stream(stream: TcpStream) -> RawResourceHandle {
    let mut table = table().lock().unwrap_or_else(|_| abort_contract());
    let id = next_id(&mut table);
    table.streams.insert(id, stream);
    RawResourceHandle {
        id,
        resource_type: TCP_STREAM_RESOURCE_TYPE,
    }
}

/// `stark_tcp_listener_bind`, an ABI v0.1 entry point.
///
/// # Safety
/// `address` must point to `len` initialised bytes the caller owns for the duration of this call, or be zero-length (ABI §9: a call-duration view, never a transfer).
/// every out-pointer must be non-null, properly aligned and owned by the caller for this call; out-slots are written only on success (ABI §4.7).
#[no_mangle]
pub unsafe extern "C" fn stark_tcp_listener_bind(
    address: BorrowedBuffer,
    out_handle: *mut RawResourceHandle,
) -> ProviderStatus {
    let address = match address_from_buffer(address) {
        Ok(address) => address,
        Err(status) => return status,
    };
    let listener = match TcpListener::bind(address) {
        Ok(listener) => listener,
        Err(error) => return map_io_error(&error),
    };
    unsafe { write_scalar(out_handle, insert_listener(listener)) };
    ProviderStatus::SUCCESS
}

/// `stark_tcp_listener_accept`, an ABI v0.1 entry point.
///
/// # Safety
/// `listener` must be a handle this provider issued and has not yet closed; its `resource_type` is checked, but a stale id is not detectable and aborts.
/// every out-pointer must be non-null, properly aligned and owned by the caller for this call; out-slots are written only on success (ABI §4.7).
#[no_mangle]
pub unsafe extern "C" fn stark_tcp_listener_accept(
    listener: RawResourceHandle,
    out_stream: *mut RawResourceHandle,
) -> ProviderStatus {
    validate(listener, TCP_LISTENER_RESOURCE_TYPE);
    let table_guard = table().lock().unwrap_or_else(|_| abort_contract());
    let Some(listener_ref) = table_guard.listeners.get(&listener.id) else {
        abort_contract();
    };
    let cloned = listener_ref
        .try_clone()
        .unwrap_or_else(|_| abort_contract());
    drop(table_guard);
    let (stream, _) = match cloned.accept() {
        Ok(value) => value,
        Err(error) => return map_io_error(&error),
    };
    unsafe { write_scalar(out_stream, insert_stream(stream)) };
    ProviderStatus::SUCCESS
}

/// `stark_tcp_stream_connect`, an ABI v0.1 entry point.
///
/// # Safety
/// `address` must point to `len` initialised bytes the caller owns for the duration of this call, or be zero-length (ABI §9: a call-duration view, never a transfer).
/// every out-pointer must be non-null, properly aligned and owned by the caller for this call; out-slots are written only on success (ABI §4.7).
#[no_mangle]
pub unsafe extern "C" fn stark_tcp_stream_connect(
    address: BorrowedBuffer,
    out_stream: *mut RawResourceHandle,
) -> ProviderStatus {
    let address = match address_from_buffer(address) {
        Ok(address) => address,
        Err(status) => return status,
    };
    let stream = match TcpStream::connect(address) {
        Ok(stream) => stream,
        Err(error) => return map_io_error(&error),
    };
    unsafe { write_scalar(out_stream, insert_stream(stream)) };
    ProviderStatus::SUCCESS
}

/// `stark_tcp_stream_read`, an ABI v0.1 entry point.
///
/// # Safety
/// `stream` must be a handle this provider issued and has not yet closed; its `resource_type` is checked, but a stale id is not detectable and aborts.
/// `out_buffer` must point to `len` writable bytes the caller owns for the duration of this call, or be zero-length; the caller reads it back afterwards, which is the point of the form.
/// every out-pointer must be non-null, properly aligned and owned by the caller for this call; out-slots are written only on success (ABI §4.7).
#[no_mangle]
pub unsafe extern "C" fn stark_tcp_stream_read(
    stream: RawResourceHandle,
    out_buffer: BorrowedBufferMut,
    out_written: *mut u64,
) -> ProviderStatus {
    validate(stream, TCP_STREAM_RESOURCE_TYPE);
    if out_buffer.len > 0 && out_buffer.ptr.is_null() {
        abort_contract();
    }
    let mut table = table().lock().unwrap_or_else(|_| abort_contract());
    let Some(stream_ref) = table.streams.get_mut(&stream.id) else {
        abort_contract();
    };
    let slice = if out_buffer.len == 0 {
        &mut []
    } else {
        unsafe { std::slice::from_raw_parts_mut(out_buffer.ptr, out_buffer.len) }
    };
    let read = match stream_ref.read(slice) {
        Ok(read) => read,
        Err(error) => return map_stream_io_error(&error),
    };
    unsafe { write_scalar(out_written, read as u64) };
    ProviderStatus::SUCCESS
}

/// `stark_tcp_stream_write`, an ABI v0.1 entry point.
///
/// # Safety
/// `stream` must be a handle this provider issued and has not yet closed; its `resource_type` is checked, but a stale id is not detectable and aborts.
/// `data` must point to `len` initialised bytes the caller owns for the duration of this call, or be zero-length (ABI §9: a call-duration view, never a transfer).
/// every out-pointer must be non-null, properly aligned and owned by the caller for this call; out-slots are written only on success (ABI §4.7).
#[no_mangle]
pub unsafe extern "C" fn stark_tcp_stream_write(
    stream: RawResourceHandle,
    data: BorrowedBuffer,
    out_accepted: *mut u64,
) -> ProviderStatus {
    validate(stream, TCP_STREAM_RESOURCE_TYPE);
    let data = unsafe { read_buffer(data) };
    let mut table = table().lock().unwrap_or_else(|_| abort_contract());
    let Some(stream_ref) = table.streams.get_mut(&stream.id) else {
        abort_contract();
    };
    let written = match stream_ref.write(data) {
        Ok(written) => written,
        Err(error) => return map_stream_io_error(&error),
    };
    unsafe { write_scalar(out_accepted, written as u64) };
    ProviderStatus::SUCCESS
}

/// HC4: `stark_tcp_stream_set_read_timeout`, an ABI v0.1 entry point.
///
/// `nanos` is a duration in nanoseconds, and **zero means "no timeout"** — the zero-duration
/// semantics HC4 requires to be specified rather than left implicit. That maps to `None`, which is
/// what `std` uses for a blocking socket.
///
/// A non-zero duration that rounds to zero at the OS's resolution is raised to 1ns rather than
/// silently becoming "block forever": the caller asked for a bound, and the one thing this must not
/// do is turn a bound into its opposite.
///
/// # Safety
/// `stream` must be a handle this provider issued and has not yet closed; its `resource_type` is checked, but a stale id is not detectable and aborts.
#[no_mangle]
pub unsafe extern "C" fn stark_tcp_stream_set_read_timeout(
    stream: RawResourceHandle,
    nanos: u64,
) -> ProviderStatus {
    validate(stream, TCP_STREAM_RESOURCE_TYPE);
    let mut table = table().lock().unwrap_or_else(|_| abort_contract());
    let Some(stream_ref) = table.streams.get_mut(&stream.id) else {
        abort_contract();
    };
    match stream_ref.set_read_timeout(timeout_from_nanos(nanos)) {
        Ok(()) => ProviderStatus::SUCCESS,
        Err(error) => map_io_error(&error),
    }
}

/// HC4: `stark_tcp_stream_set_write_timeout`, an ABI v0.1 entry point.
///
/// # Safety
/// `stream` must be a handle this provider issued and has not yet closed; its `resource_type` is checked, but a stale id is not detectable and aborts.
#[no_mangle]
pub unsafe extern "C" fn stark_tcp_stream_set_write_timeout(
    stream: RawResourceHandle,
    nanos: u64,
) -> ProviderStatus {
    validate(stream, TCP_STREAM_RESOURCE_TYPE);
    let mut table = table().lock().unwrap_or_else(|_| abort_contract());
    let Some(stream_ref) = table.streams.get_mut(&stream.id) else {
        abort_contract();
    };
    match stream_ref.set_write_timeout(timeout_from_nanos(nanos)) {
        Ok(()) => ProviderStatus::SUCCESS,
        Err(error) => map_io_error(&error),
    }
}

/// Zero is "no timeout"; anything else is a real bound, floored at 1ns so a sub-resolution request
/// cannot become unbounded.
fn timeout_from_nanos(nanos: u64) -> Option<std::time::Duration> {
    if nanos == 0 {
        return None;
    }
    let duration = std::time::Duration::from_nanos(nanos);
    if duration.is_zero() {
        return Some(std::time::Duration::from_nanos(1));
    }
    Some(duration)
}

/// `stark_tcp_listener_close`, an ABI v0.1 entry point.
///
/// # Safety
/// `handle` must be a handle this provider issued and has not yet closed; its `resource_type` is checked, but a stale id is not detectable and aborts.
#[no_mangle]
pub unsafe extern "C" fn stark_tcp_listener_close(handle: RawResourceHandle) -> ProviderStatus {
    validate(handle, TCP_LISTENER_RESOURCE_TYPE);
    let mut table = table().lock().unwrap_or_else(|_| abort_contract());
    if table.listeners.remove(&handle.id).is_none() {
        abort_contract();
    }
    ProviderStatus::SUCCESS
}

/// `stark_tcp_stream_close`, an ABI v0.1 entry point.
///
/// # Safety
/// `handle` must be a handle this provider issued and has not yet closed; its `resource_type` is checked, but a stale id is not detectable and aborts.
#[no_mangle]
pub unsafe extern "C" fn stark_tcp_stream_close(handle: RawResourceHandle) -> ProviderStatus {
    validate(handle, TCP_STREAM_RESOURCE_TYPE);
    let mut table = table().lock().unwrap_or_else(|_| abort_contract());
    if let Some(stream) = table.streams.remove(&handle.id) {
        let _ = stream.shutdown(Shutdown::Both);
    } else {
        abort_contract();
    }
    ProviderStatus::SUCCESS
}

#[no_mangle]
pub unsafe extern "C" fn stark_dns_resolve_len(
    host: BorrowedBuffer,
    out_required_len: *mut u64,
    out_count: *mut u64,
) -> ProviderStatus {
    let host = match hostname_from_buffer(host) {
        Ok(host) => host,
        Err(status) => return status,
    };
    let records = match resolve_records(&host) {
        Ok(records) => records,
        Err(status) => return status,
    };
    unsafe {
        write_scalar(out_required_len, (records.len() * DNS_RECORD_WIDTH) as u64);
        write_scalar(out_count, records.len() as u64);
    }
    ProviderStatus::SUCCESS
}

#[no_mangle]
pub unsafe extern "C" fn stark_dns_resolve_fill(
    host: BorrowedBuffer,
    out_records: BorrowedBufferMut,
    out_written_len: *mut u64,
    out_count: *mut u64,
) -> ProviderStatus {
    let host = match hostname_from_buffer(host) {
        Ok(host) => host,
        Err(status) => return status,
    };
    let records = match resolve_records(&host) {
        Ok(records) => records,
        Err(status) => return status,
    };
    let required = records.len() * DNS_RECORD_WIDTH;
    if out_records.len < required {
        return STATUS_DNS_TOO_MANY_RESULTS;
    }
    if required > 0 && out_records.ptr.is_null() {
        abort_contract();
    }
    let output = unsafe { std::slice::from_raw_parts_mut(out_records.ptr, required) };
    for (i, record) in records.iter().enumerate() {
        let start = i * DNS_RECORD_WIDTH;
        output[start..start + DNS_RECORD_WIDTH].copy_from_slice(record);
    }
    unsafe {
        write_scalar(out_written_len, required as u64);
        write_scalar(out_count, records.len() as u64);
    }
    ProviderStatus::SUCCESS
}

/// **HC9 — `stark_tcp_stream_detach`: this provider's half of a cross-provider transfer.**
///
/// Not an ABI v0.1 entry point, and deliberately absent from the provider manifest. No package
/// binds it and lowering never emits it; it is called by another PROVIDER, resolved by the linker
/// inside the one binary every provider is statically linked into. See
/// `stark_provider_abi::RawOsHandle` for why the convention lives there rather than in the
/// manifest's callable surface.
///
/// The visible declaration of this relationship is the consumer's
/// `consumes: [{ "provider": "stark-std-net", "resource": "tcp_stream" }]`, which
/// `provider_abi::validate` and `ProviderSet::select` check from both ends (CD-360).
///
/// # What it does, and what the caller then owes
///
/// Removes the stream from this provider's table and yields the underlying socket **without
/// closing it** — `into_raw_fd`/`into_raw_socket`, not `as_raw_fd`, so no destructor runs. The
/// caller owns the socket from the moment this returns `SUCCESS` and must eventually close it.
///
/// After a successful detach this provider knows nothing about the handle. A subsequent
/// `stark_tcp_stream_close` for it aborts, which is correct: CD-360's lowering clears the caller's
/// drop flag at transfer call entry, so a close reaching here for a detached handle means the
/// ownership rule was broken somewhere, and aborting names it at the point of breach rather than
/// letting a recycled descriptor be closed twice much later.
///
/// A handle this provider does not hold aborts for the same reason `close` does — a stale id is
/// indistinguishable from a forged one.
///
/// # Safety
/// `handle` must be a handle this provider issued and has not yet closed or detached; its
/// `resource_type` is checked, but a stale id is not detectable and aborts.
/// `out_handle` must be non-null, properly aligned and owned by the caller for this call; it is
/// written only on success.
#[no_mangle]
pub unsafe extern "C" fn stark_tcp_stream_detach(
    handle: RawResourceHandle,
    out_handle: *mut RawOsHandle,
) -> ProviderStatus {
    validate(handle, TCP_STREAM_RESOURCE_TYPE);
    let mut table = table().lock().unwrap_or_else(|_| abort_contract());
    let Some(stream) = table.streams.remove(&handle.id) else {
        abort_contract();
    };
    drop(table);
    unsafe { write_scalar(out_handle, RawOsHandle::socket(into_raw_socket(stream))) };
    ProviderStatus::SUCCESS
}

/// The socket, with its Rust owner consumed and no destructor run.
#[cfg(unix)]
fn into_raw_socket(stream: TcpStream) -> i64 {
    use std::os::fd::IntoRawFd;
    stream.into_raw_fd() as i64
}

#[cfg(windows)]
fn into_raw_socket(stream: TcpStream) -> i64 {
    use std::os::windows::io::IntoRawSocket;
    // `SOCKET` is `UINT_PTR`. On 64-bit this is a lossless bit-pattern reinterpretation, and
    // `INVALID_SOCKET` (`(UINT_PTR)(-1)`) arrives as `-1` — which `is_valid_socket` refuses,
    // matching Unix's error value without a per-platform comparison.
    stream.into_raw_socket() as i64
}

/// Whether this provider still holds the stream with this id.
///
/// Test support for the OTHER side of a transfer. After `stark_tcp_stream_detach` the answer must
/// be `false`, and nothing else can observe that: the table is private, and a consuming provider
/// asserting "the socket left the owner" has no other way to say it. Without this, a detach that
/// yielded a duplicate rather than moving the socket would pass every test in both crates.
pub fn holds_stream(id: u64) -> bool {
    table()
        .lock()
        .unwrap_or_else(|_| abort_contract())
        .streams
        .contains_key(&id)
}

#[derive(Debug)]
pub enum HarnessError {
    Io(std::io::Error),
    Timeout(&'static str),
    Protocol(&'static str),
    ThreadPanic,
}

impl From<std::io::Error> for HarnessError {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}

pub struct EchoServer {
    pub address: SocketAddr,
    stop_tx: mpsc::Sender<()>,
    join: JoinHandle<Result<(), HarnessError>>,
}

impl EchoServer {
    pub fn spawn() -> Result<Self, HarnessError> {
        let listener = TcpListener::bind("127.0.0.1:0")?;
        listener.set_nonblocking(true)?;
        let address = listener.local_addr()?;
        let (ready_tx, ready_rx) = mpsc::channel();
        let (stop_tx, stop_rx) = mpsc::channel();
        let join = thread::spawn(move || {
            ready_tx
                .send(())
                .map_err(|_| HarnessError::Protocol("ready signal failed"))?;
            loop {
                if stop_rx.try_recv().is_ok() {
                    return Ok(());
                }
                match listener.accept() {
                    Ok((mut stream, _)) => {
                        thread::spawn(move || {
                            let _ = stream.set_read_timeout(Some(Duration::from_secs(5)));
                            let _ = stream.set_write_timeout(Some(Duration::from_secs(5)));
                            let _ = echo_connection(&mut stream);
                        });
                    }
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                        thread::park_timeout(Duration::from_millis(5));
                    }
                    Err(error) => return Err(HarnessError::Io(error)),
                }
            }
        });
        ready_rx
            .recv_timeout(Duration::from_secs(2))
            .map_err(|_| HarnessError::Timeout("readiness"))?;
        Ok(Self {
            address,
            stop_tx,
            join,
        })
    }

    pub fn shutdown(self) -> Result<(), HarnessError> {
        let _ = self.stop_tx.send(());
        TcpStream::connect_timeout(&self.address, Duration::from_millis(200)).ok();
        self.join.join().map_err(|_| HarnessError::ThreadPanic)?
    }
}

fn echo_connection(stream: &mut TcpStream) -> Result<(), HarnessError> {
    loop {
        let mut len_bytes = [0u8; 8];
        match stream.read_exact(&mut len_bytes) {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(()),
            Err(error) if error.kind() == std::io::ErrorKind::ConnectionReset => return Ok(()),
            Err(error) => return Err(HarnessError::Io(error)),
        }
        let len = u64::from_be_bytes(len_bytes);
        let len = usize::try_from(len).map_err(|_| HarnessError::Protocol("frame too large"))?;
        let mut payload = vec![0; len];
        stream.read_exact(&mut payload)?;
        stream.write_all(&len_bytes)?;
        stream.write_all(&payload)?;
    }
}

pub fn send_frame(
    address: SocketAddr,
    payload: &[u8],
    timeout: Duration,
) -> Result<Vec<u8>, HarnessError> {
    let mut stream = TcpStream::connect_timeout(&address, timeout)?;
    stream.set_read_timeout(Some(timeout))?;
    stream.set_write_timeout(Some(timeout))?;
    stream.write_all(&(payload.len() as u64).to_be_bytes())?;
    stream.write_all(payload)?;
    let mut len_bytes = [0u8; 8];
    stream.read_exact(&mut len_bytes)?;
    let len = usize::try_from(u64::from_be_bytes(len_bytes))
        .map_err(|_| HarnessError::Protocol("frame too large"))?;
    let mut response = vec![0; len];
    stream.read_exact(&mut response)?;
    Ok(response)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    /// **Every test that touches the provider's global table takes this first.**
    ///
    /// `TABLE` is process-global and `cargo test` runs tests in parallel in one process, so these
    /// tests are not independent however carefully each is written. Two of them additionally hand a
    /// raw socket out of the provider (`detach` -> `into_raw_fd` -> `adopt`), which means a live fd
    /// exists outside any Rust owner for a window; a third concurrently opening and closing sockets
    /// makes that window observable.
    ///
    /// The symptom was a detached socket that connected, accepted writes, and then reported EOF
    /// instead of the echo — a failure in a test that was itself correct, roughly one run in five,
    /// and green in twelve consecutive runs before the third test existed. **Serialising the
    /// WRITERS is the fix, not serialising the assertions**: a test that merely opens a socket
    /// perturbs a test that asserts on the table just as much as one that reads it.
    static PROVIDER_TABLE: Mutex<()> = Mutex::new(());

    /// Held for the body of a test. Poisoning is ignored: a panicking test has already failed, and
    /// propagating its poison would convert one failure into every subsequent test failing for an
    /// unrelated reason.
    fn exclusive() -> std::sync::MutexGuard<'static, ()> {
        PROVIDER_TABLE.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn buf(bytes: &[u8]) -> BorrowedBuffer {
        BorrowedBuffer {
            ptr: bytes.as_ptr(),
            len: bytes.len(),
        }
    }

    fn provider_metadata() -> starkc::backend::provider_abi::ProviderMetadata {
        use starkc::backend::provider_abi::{
            AbiParam, FunctionDecl, ProviderIdentity, ProviderMetadata, ScalarTy,
        };
        let listener = "tcp_listener".to_string();
        let stream = "tcp_stream".to_string();
        ProviderMetadata {
            // CD-360: this provider consumes no other provider's resource.
            foreign_resources: Vec::new(),
            identity: ProviderIdentity {
                name: "stark-std-net".to_string(),
                semver: (0, 1, 0),
                abi_version: starkc::backend::provider_abi::ABI_VERSION.to_string(),
            },
            target_triples: vec![
                "aarch64-apple-darwin".to_string(),
                "x86_64-apple-darwin".to_string(),
                "x86_64-unknown-linux-gnu".to_string(),
                "x86_64-pc-windows-msvc".to_string(),
            ],
            capabilities: vec![
                "network-client".to_string(),
                "network-listen".to_string(),
            ],
            resource_types: vec![listener.clone(), stream.clone()],
            functions: vec![
                FunctionDecl {
                    name: "stark_tcp_listener_bind".to_string(),
                    capability: "network-listen".to_string(),
                    params: vec![
                        AbiParam::BufferIn,
                        AbiParam::HandleOut {
                            resource_type: listener.clone(),
                        },
                    ],
                    is_close_for: None,
                    may_block: true,
                },
                FunctionDecl {
                    name: "stark_tcp_listener_accept".to_string(),
                    capability: "network-listen".to_string(),
                    params: vec![
                        AbiParam::HandleBorrowed {
                            resource_type: listener.clone(),
                        },
                        AbiParam::HandleOut {
                            resource_type: stream.clone(),
                        },
                    ],
                    is_close_for: None,
                    may_block: true,
                },
                FunctionDecl {
                    name: "stark_tcp_stream_connect".to_string(),
                    capability: "network-client".to_string(),
                    params: vec![
                        AbiParam::BufferIn,
                        AbiParam::HandleOut {
                            resource_type: stream.clone(),
                        },
                    ],
                    is_close_for: None,
                    may_block: true,
                },
                FunctionDecl {
                    name: "stark_tcp_stream_read".to_string(),
                    capability: "network-client".to_string(),
                    params: vec![
                        AbiParam::HandleBorrowed {
                            resource_type: stream.clone(),
                        },
                        AbiParam::BufferInOut,
                        AbiParam::ScalarOut(ScalarTy::U64),
                    ],
                    is_close_for: None,
                    may_block: true,
                },
                FunctionDecl {
                    name: "stark_tcp_stream_write".to_string(),
                    capability: "network-client".to_string(),
                    params: vec![
                        AbiParam::HandleBorrowed {
                            resource_type: stream.clone(),
                        },
                        AbiParam::BufferIn,
                        AbiParam::ScalarOut(ScalarTy::U64),
                    ],
                    is_close_for: None,
                    may_block: true,
                },
                FunctionDecl {
                    name: "stark_tcp_listener_close".to_string(),
                    capability: "network-listen".to_string(),
                    params: vec![AbiParam::HandleConsumed {
                        resource_type: listener.clone(),
                    }],
                    is_close_for: Some(listener),
                    may_block: false,
                },
                FunctionDecl {
                    name: "stark_tcp_stream_close".to_string(),
                    capability: "network-client".to_string(),
                    params: vec![AbiParam::HandleConsumed {
                        resource_type: stream.clone(),
                    }],
                    is_close_for: Some(stream),
                    may_block: false,
                },
                FunctionDecl {
                    name: "stark_dns_resolve_len".to_string(),
                    capability: "network-client".to_string(),
                    params: vec![
                        AbiParam::BufferIn,
                        AbiParam::ScalarOut(ScalarTy::U64),
                        AbiParam::ScalarOut(ScalarTy::U64),
                    ],
                    is_close_for: None,
                    may_block: true,
                },
                FunctionDecl {
                    name: "stark_dns_resolve_fill".to_string(),
                    capability: "network-client".to_string(),
                    params: vec![
                        AbiParam::BufferIn,
                        AbiParam::BufferInOut,
                        AbiParam::ScalarOut(ScalarTy::U64),
                        AbiParam::ScalarOut(ScalarTy::U64),
                    ],
                    is_close_for: None,
                    may_block: true,
                },
            ],
        }
    }

    fn portable_c_identifier(name: &str) -> bool {
        let mut chars = name.bytes();
        matches!(chars.next(), Some(b'_' | b'a'..=b'z' | b'A'..=b'Z'))
            && chars.all(|c| matches!(c, b'_' | b'a'..=b'z' | b'A'..=b'Z' | b'0'..=b'9'))
    }

    mod linked {
        use super::{BorrowedBuffer, BorrowedBufferMut, ProviderStatus, RawResourceHandle};

        unsafe extern "C" {
            pub fn stark_tcp_listener_bind(
                address: BorrowedBuffer,
                out_handle: *mut RawResourceHandle,
            ) -> ProviderStatus;
            pub fn stark_tcp_listener_accept(
                listener: RawResourceHandle,
                out_stream: *mut RawResourceHandle,
            ) -> ProviderStatus;
            pub fn stark_tcp_stream_connect(
                address: BorrowedBuffer,
                out_stream: *mut RawResourceHandle,
            ) -> ProviderStatus;
            pub fn stark_tcp_stream_read(
                stream: RawResourceHandle,
                out_buffer: BorrowedBufferMut,
                out_written: *mut u64,
            ) -> ProviderStatus;
            pub fn stark_tcp_stream_write(
                stream: RawResourceHandle,
                data: BorrowedBuffer,
                out_accepted: *mut u64,
            ) -> ProviderStatus;
            // DEV-163's regression test drives the deadline through the LINKED symbol, like every
            // other call here, so it exercises the exported entry point rather than the Rust fn.
            pub fn stark_tcp_stream_set_read_timeout(
                stream: RawResourceHandle,
                nanos: u64,
            ) -> ProviderStatus;
            pub fn stark_tcp_listener_close(handle: RawResourceHandle) -> ProviderStatus;
            pub fn stark_tcp_stream_close(handle: RawResourceHandle) -> ProviderStatus;
            pub fn stark_tcp_stream_detach(
                handle: RawResourceHandle,
                out_handle: *mut super::RawOsHandle,
            ) -> ProviderStatus;
            pub fn stark_dns_resolve_len(
                host: BorrowedBuffer,
                out_required_len: *mut u64,
                out_count: *mut u64,
            ) -> ProviderStatus;
            pub fn stark_dns_resolve_fill(
                host: BorrowedBuffer,
                out_records: BorrowedBufferMut,
                out_written_len: *mut u64,
                out_count: *mut u64,
            ) -> ProviderStatus;
        }
    }

    #[test]
    fn metadata_validates_and_symbols_match() {
        let metadata = provider_metadata();
        assert_eq!(starkc::backend::provider_abi::validate(&metadata), Ok(()));
        let declared: HashSet<_> = metadata.functions.iter().map(|f| f.name.as_str()).collect();
        let exported = HashSet::from([
            "stark_tcp_listener_bind",
            "stark_tcp_listener_accept",
            "stark_tcp_stream_connect",
            "stark_tcp_stream_read",
            "stark_tcp_stream_write",
            "stark_tcp_listener_close",
            "stark_tcp_stream_close",
            "stark_dns_resolve_len",
            "stark_dns_resolve_fill",
        ]);
        assert_eq!(declared, exported);
        assert!(declared.iter().all(|name| portable_c_identifier(name)));
        assert_eq!(metadata.resource_types, vec!["tcp_listener", "tcp_stream"]);
    }

    /// **The shipped JSON manifest is the authority (P0.2), so THAT is what the exports are checked
    /// against.**
    ///
    /// `metadata_validates_and_symbols_match` above compares two hand-written literals in this
    /// file. They agree, and they have to — but agreement between a mirror and a list of the same
    /// mirror's contents is precisely the CD-219 failure: both drifted from the real export set
    /// together, and neither could see it. `stark_tcp_stream_set_read_timeout` and its write twin
    /// were added under HC4, are in the JSON manifest, are exported here, and appear in NEITHER
    /// literal above — undetected until this test was written.
    ///
    /// This reads the manifest the compiler actually embeds and compares it against the symbols
    /// this crate actually links, which is the comparison that can fail.
    #[test]
    fn the_shipped_manifest_matches_the_symbols_this_crate_links() {
        let text = include_str!("../../../../starkc/providers/stark-net-native.json");
        let provider = starkc::provider_manifest::parse_provider_manifest(text, "stark-net-native")
            .expect("the shipped manifest must parse");
        assert_eq!(
            starkc::provider_abi::validate(&provider.metadata),
            Ok(()),
            "the shipped manifest must satisfy the ABI validator"
        );

        // Every symbol the manifest declares must resolve at link time. Calling each with a
        // deliberately invalid handle is not viable — several would abort — so the linkage is
        // proven by the `extern` block in `linked` below, which the compiler resolves for the whole
        // module. What this asserts is the SET.
        let declared: HashSet<&str> = provider
            .metadata
            .functions
            .iter()
            .map(|f| f.name.as_str())
            .collect();
        let exported = HashSet::from([
            "stark_tcp_listener_bind",
            "stark_tcp_listener_accept",
            "stark_tcp_stream_connect",
            "stark_tcp_stream_read",
            "stark_tcp_stream_write",
            "stark_tcp_stream_set_read_timeout",
            "stark_tcp_stream_set_write_timeout",
            "stark_tcp_listener_close",
            "stark_tcp_stream_close",
            "stark_dns_resolve_len",
            "stark_dns_resolve_fill",
        ]);
        assert_eq!(
            declared, exported,
            "the shipped manifest and this crate's callable exports must name the same set"
        );

        // HC9's transfer surface is exported but NOT in the manifest, by design — see
        // `stark_tcp_stream_detach`. Asserted explicitly so "absent from the manifest" stays a
        // decision with a test behind it rather than an omission nobody would notice.
        assert!(
            !declared.contains("stark_tcp_stream_detach"),
            "`detach` is a provider-to-provider convention, not a STARK-callable ABI function; \
             putting it in the manifest would place a permanently unreachable symbol into the \
             surface the validator governs"
        );
    }

    #[test]
    fn physical_abi_types_are_from_shared_crate_with_pinned_layout() {
        assert_eq!(
            std::mem::size_of::<ProviderStatus>(),
            std::mem::size_of::<stark_provider_abi::ProviderStatus>()
        );
        assert_eq!(
            std::mem::align_of::<ProviderStatus>(),
            std::mem::align_of::<stark_provider_abi::ProviderStatus>()
        );
        assert_eq!(
            std::mem::size_of::<RawResourceHandle>(),
            std::mem::size_of::<stark_provider_abi::RawResourceHandle>()
        );
        assert_eq!(
            std::mem::size_of::<BorrowedBuffer>(),
            std::mem::size_of::<stark_provider_abi::BorrowedBuffer>()
        );
        assert_eq!(
            std::mem::size_of::<BorrowedBufferMut>(),
            std::mem::size_of::<stark_provider_abi::BorrowedBufferMut>()
        );
    }

    #[test]
    fn status_zero_means_success_and_declared_errors_are_stable() {
        assert_eq!(ProviderStatus::SUCCESS.code, 0);
        assert_eq!(STATUS_OTHER_DECLARED.code, 11);
    }

    #[test]
    fn dns_rejects_invalid_hosts_before_resolver_call() {
        let mut required_len = 123u64;
        let mut count = 456u64;

        let empty =
            unsafe { linked::stark_dns_resolve_len(buf(b""), &mut required_len, &mut count) };
        assert_eq!(empty, STATUS_DNS_INVALID_HOST);
        assert_eq!(required_len, 123);
        assert_eq!(count, 456);

        let nul = unsafe {
            linked::stark_dns_resolve_len(buf(b"exa\0mple"), &mut required_len, &mut count)
        };
        assert_eq!(nul, STATUS_DNS_INVALID_HOST);
        assert_eq!(required_len, 123);
        assert_eq!(count, 456);
    }

    #[test]
    fn harness_echoes_binary_large_repeated_and_parallel_instances() {
        let _exclusive = exclusive();
        let a = EchoServer::spawn().unwrap();
        let b = EchoServer::spawn().unwrap();
        assert!(a.address.ip().is_loopback());
        assert_ne!(a.address, b.address);
        for payload in [
            &b""[..],
            b"x",
            b"a\0b",
            "hello".as_bytes(),
            &[0xFF, 0x00, 0xFE][..],
        ] {
            assert_eq!(
                send_frame(a.address, payload, Duration::from_secs(2)).unwrap(),
                payload
            );
        }
        let large = vec![7u8; 64 * 1024];
        assert_eq!(
            send_frame(a.address, &large, Duration::from_secs(2)).unwrap(),
            large
        );
        a.shutdown().unwrap();
        b.shutdown().unwrap();
    }

    #[test]
    fn provider_loopback_bind_connect_accept_send_receive() {
        let _exclusive = exclusive();
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        drop(listener);
        let address_text = address.to_string();
        let mut listener_handle = RawResourceHandle {
            id: 0,
            resource_type: TCP_LISTENER_RESOURCE_TYPE,
        };
        assert_eq!(
            unsafe {
                linked::stark_tcp_listener_bind(buf(address_text.as_bytes()), &mut listener_handle)
            },
            ProviderStatus::SUCCESS
        );
        let accept_thread = thread::spawn(move || {
            let mut accepted = RawResourceHandle {
                id: 0,
                resource_type: TCP_STREAM_RESOURCE_TYPE,
            };
            assert_eq!(
                unsafe { linked::stark_tcp_listener_accept(listener_handle, &mut accepted) },
                ProviderStatus::SUCCESS
            );
            accepted
        });
        let mut client = RawResourceHandle {
            id: 0,
            resource_type: TCP_STREAM_RESOURCE_TYPE,
        };
        assert_eq!(
            unsafe { linked::stark_tcp_stream_connect(buf(address_text.as_bytes()), &mut client) },
            ProviderStatus::SUCCESS
        );
        let accepted = accept_thread.join().unwrap();
        let mut written = 0;
        assert_eq!(
            unsafe { linked::stark_tcp_stream_write(client, buf(b"abc\0def"), &mut written) },
            ProviderStatus::SUCCESS
        );
        assert_eq!(written, 7);
        let mut out = [0u8; 16];
        let mut read = 0;
        assert_eq!(
            unsafe {
                linked::stark_tcp_stream_read(
                    accepted,
                    BorrowedBufferMut {
                        ptr: out.as_mut_ptr(),
                        len: out.len(),
                    },
                    &mut read,
                )
            },
            ProviderStatus::SUCCESS
        );
        assert_eq!(&out[..read as usize], b"abc\0def");
        assert_eq!(
            unsafe { linked::stark_tcp_stream_close(client) },
            ProviderStatus::SUCCESS
        );
        assert_eq!(
            unsafe { linked::stark_tcp_stream_close(accepted) },
            ProviderStatus::SUCCESS
        );
        assert_eq!(
            unsafe { linked::stark_tcp_listener_close(listener_handle) },
            ProviderStatus::SUCCESS
        );
    }

    /// **HC9's transfer half, proved against a live peer.**
    ///
    /// A detached socket must still WORK — the point of a transfer is that the consuming provider
    /// continues using the connection, so a detach that yields a closed or duplicated descriptor
    /// would pass a shallow "did it return SUCCESS" check and fail at the first handshake byte.
    /// This one detaches, rebuilds a `TcpStream` from the raw handle, and completes a round trip
    /// over it.
    #[test]
    fn a_detached_socket_is_live_and_this_provider_has_forgotten_it() {
        let _exclusive = exclusive();
        let server = EchoServer::spawn().unwrap();
        let address_text = server.address.to_string();

        let mut handle = RawResourceHandle {
            id: 0,
            resource_type: TCP_STREAM_RESOURCE_TYPE,
        };
        assert_eq!(
            unsafe { linked::stark_tcp_stream_connect(buf(address_text.as_bytes()), &mut handle) },
            ProviderStatus::SUCCESS
        );

        let id = handle.id;
        assert!(
            table().lock().unwrap().streams.contains_key(&id),
            "the provider must hold the stream before the transfer"
        );

        let mut detached = RawOsHandle::NONE;
        assert_eq!(
            unsafe { linked::stark_tcp_stream_detach(handle, &mut detached) },
            ProviderStatus::SUCCESS
        );
        assert!(
            detached.is_valid_socket(),
            "a successful detach must yield a usable socket, got {detached:?}"
        );
        assert!(
            !table().lock().unwrap().streams.contains_key(&id),
            "after a transfer the owner must hold nothing: a later close for this handle would be \
             a double release, and the abort that would follow is the correct outcome"
        );

        // The load-bearing part: adopt the raw socket and use it. `into_raw_fd` ran, so no
        // destructor closed it, and this is the only owner.
        let mut adopted = adopt(detached);
        adopted
            .set_read_timeout(Some(Duration::from_secs(5)))
            .unwrap();
        adopted.write_all(&5u64.to_be_bytes()).unwrap();
        adopted.write_all(b"hello").unwrap();
        let mut len_bytes = [0u8; 8];
        adopted.read_exact(&mut len_bytes).unwrap();
        assert_eq!(u64::from_be_bytes(len_bytes), 5);
        let mut echoed = [0u8; 5];
        adopted.read_exact(&mut echoed).unwrap();
        assert_eq!(&echoed, b"hello");

        drop(adopted);
        server.shutdown().unwrap();
    }

    /// Two detaches of one handle must not both succeed. The second finds nothing in the table and
    /// aborts, so it cannot be asserted in-process without killing the test runner — what IS
    /// assertable is that the first removed the entry, which is the property the abort rests on.
    #[test]
    fn a_detach_removes_the_entry_so_a_second_one_cannot_find_it() {
        let _exclusive = exclusive();
        let server = EchoServer::spawn().unwrap();
        let mut handle = RawResourceHandle {
            id: 0,
            resource_type: TCP_STREAM_RESOURCE_TYPE,
        };
        assert_eq!(
            unsafe {
                linked::stark_tcp_stream_connect(
                    buf(server.address.to_string().as_bytes()),
                    &mut handle,
                )
            },
            ProviderStatus::SUCCESS
        );
        let mut detached = RawOsHandle::NONE;
        assert_eq!(
            unsafe { linked::stark_tcp_stream_detach(handle, &mut detached) },
            ProviderStatus::SUCCESS
        );
        assert!(!table().lock().unwrap().streams.contains_key(&handle.id));
        drop(adopt(detached));
        server.shutdown().unwrap();
    }

    #[cfg(unix)]
    fn adopt(handle: RawOsHandle) -> TcpStream {
        use std::os::fd::FromRawFd;
        unsafe { TcpStream::from_raw_fd(handle.value as std::os::fd::RawFd) }
    }

    #[cfg(windows)]
    fn adopt(handle: RawOsHandle) -> TcpStream {
        use std::os::windows::io::FromRawSocket;
        unsafe { TcpStream::from_raw_socket(handle.value as std::os::windows::io::RawSocket) }
    }

    #[test]
    fn malformed_address_and_connection_refused_are_recoverable() {
        let _exclusive = exclusive();
        let mut handle = RawResourceHandle {
            id: 0,
            resource_type: TCP_STREAM_RESOURCE_TYPE,
        };
        assert_eq!(
            unsafe { linked::stark_tcp_stream_connect(buf(b"not a socket address"), &mut handle) },
            STATUS_INVALID_INPUT
        );
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        drop(listener);
        let status = unsafe {
            linked::stark_tcp_stream_connect(buf(address.to_string().as_bytes()), &mut handle)
        };
        assert!(matches!(status.code, 1 | 11));
    }

    /// **DEV-163: an expired socket deadline must report as a TIMEOUT on every platform.**
    ///
    /// `SO_RCVTIMEO` expiring produces two different error kinds:
    ///
    /// ```text
    /// Unix     EAGAIN        -> std::io::ErrorKind::WouldBlock
    /// Windows  WSAETIMEDOUT  -> std::io::ErrorKind::TimedOut
    /// ```
    ///
    /// Passing both through unchanged made `stark-http-client` report "the connection failed" on
    /// Linux and macOS and "timed out reading the response" on Windows — same peer, same STARK
    /// source. An operator reading the Unix message would go and look at the network instead of at
    /// the peer that was deliberately holding the socket.
    ///
    /// The peer here **holds** the accepted connection rather than closing it. Closing would give a
    /// clean EOF, which is a different outcome and the one that already worked — the whole defect
    /// lives in the case where a peer accepts and then says nothing at all.
    #[test]
    fn an_expired_read_deadline_reports_as_a_timeout() {
        let _exclusive = exclusive();
        // The peer is accepted and then simply HELD, in this thread's own scope. No helper thread
        // and no sleep: the 200 ms deadline is the only thing that ends the read, so the test costs
        // its own timeout and nothing more. An earlier version parked a thread for three seconds,
        // which perturbed the scheduling of the whole suite -- see the note on `EchoServer`.
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();

        let mut handle = RawResourceHandle {
            id: 0,
            resource_type: TCP_STREAM_RESOURCE_TYPE,
        };
        assert_eq!(
            unsafe {
                linked::stark_tcp_stream_connect(buf(address.to_string().as_bytes()), &mut handle)
            },
            ProviderStatus::SUCCESS
        );
        // Accepted and kept alive for the rest of the test. Dropping it would close the connection
        // and turn the timeout under test into a clean EOF -- a different outcome, and the one that
        // already worked.
        let _accepted = listener.accept().unwrap();

        assert_eq!(
            unsafe { linked::stark_tcp_stream_set_read_timeout(handle, 200_000_000) },
            ProviderStatus::SUCCESS
        );

        let mut bytes = [0u8; 16];
        let mut written = 0u64;
        let status = unsafe {
            linked::stark_tcp_stream_read(
                handle,
                BorrowedBufferMut {
                    ptr: bytes.as_mut_ptr(),
                    len: bytes.len(),
                },
                &mut written,
            )
        };
        assert_eq!(
            status, STATUS_TIMED_OUT,
            "an expired read deadline must be STATUS_TIMED_OUT. Before DEV-163 this was \
             STATUS_WOULD_BLOCK on Unix, which stark-net maps to NetworkError::Interrupted and \
             every caller above reports as a connection failure"
        );

        unsafe { linked::stark_tcp_stream_close(handle) };
    }
}
