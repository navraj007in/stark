//! Blocking TCP native provider and reusable loopback echo harness.

use std::collections::HashMap;
use std::io::{Read, Write};
use std::net::{Shutdown, SocketAddr, TcpListener, TcpStream};
use std::sync::mpsc;
use std::sync::{Mutex, OnceLock};
use std::thread::{self, JoinHandle};
use std::time::Duration;

pub use stark_provider_abi::{
    BorrowedBuffer, BorrowedBufferMut, ProviderStatus, RawResourceHandle,
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
        Err(error) => return map_io_error(&error),
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
        Err(error) => return map_io_error(&error),
    };
    unsafe { write_scalar(out_accepted, written as u64) };
    ProviderStatus::SUCCESS
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
            capabilities: vec!["tcp".to_string()],
            resource_types: vec![listener.clone(), stream.clone()],
            functions: vec![
                FunctionDecl {
                    name: "stark_tcp_listener_bind".to_string(),
                    capability: "tcp".to_string(),
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
                    capability: "tcp".to_string(),
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
                    capability: "tcp".to_string(),
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
                    capability: "tcp".to_string(),
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
                    capability: "tcp".to_string(),
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
                    capability: "tcp".to_string(),
                    params: vec![AbiParam::HandleConsumed {
                        resource_type: listener.clone(),
                    }],
                    is_close_for: Some(listener),
                    may_block: false,
                },
                FunctionDecl {
                    name: "stark_tcp_stream_close".to_string(),
                    capability: "tcp".to_string(),
                    params: vec![AbiParam::HandleConsumed {
                        resource_type: stream.clone(),
                    }],
                    is_close_for: Some(stream),
                    may_block: false,
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
            pub fn stark_tcp_listener_close(handle: RawResourceHandle) -> ProviderStatus;
            pub fn stark_tcp_stream_close(handle: RawResourceHandle) -> ProviderStatus;
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
        ]);
        assert_eq!(declared, exported);
        assert!(declared.iter().all(|name| portable_c_identifier(name)));
        assert_eq!(metadata.resource_types, vec!["tcp_listener", "tcp_stream"]);
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
    fn harness_echoes_binary_large_repeated_and_parallel_instances() {
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

    #[test]
    fn malformed_address_and_connection_refused_are_recoverable() {
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
}
