//! HC9 — the native TLS 1.2/1.3 client provider, and a reusable TLS peer harness.
//!
//! **Engine: rustls over aws-lc-rs, Profile N (CD-361).** No TLS or cryptographic primitive is
//! implemented here or in STARK. This file is a boundary: it adapts a mature implementation to the
//! Native Provider ABI, and every line of it is either resource bookkeeping, ABI marshalling, or
//! error normalisation.
//!
//! # The transfer, and why its order is not arbitrary
//!
//! `stark_tls_stream_connect` is STARK's first cross-provider transfer (CD-360). It takes a
//! `tcp_stream` the net provider owns, and ownership passes at CALL ENTRY — on success and on
//! failure alike. The consuming provider owes the release.
//!
//! That obligation dictates the shape of the function:
//!
//! ```text
//! detach the socket FIRST, into an owned Rust TcpStream
//!         ↓
//! validate the configuration
//!         ↓
//! handshake
//! ```
//!
//! **Validating before detaching would leak.** The handle is consumed by the ABI whatever this
//! function returns — the caller's drop flag was cleared before the call — so an early return that
//! happens before the socket is adopted leaves it stranded in the net provider's table with nothing
//! left holding a reference to it. Detaching first makes every subsequent error path a plain Rust
//! drop, which closes the socket. There is no cleanup code below for that reason, and its absence
//! is the design rather than an omission.
//!
//! # Root stores
//!
//! CD-361 separated trust-anchor policy from engine selection. HC9 implements `ExplicitRoots` only,
//! which is the policy its controlled fixture uses. `SystemRoots` is HC10's, and is refused by name
//! rather than silently falling back to a bundled set — a client that believes it is checking
//! against the system trust store while checking against something else is worse than one that
//! fails to start.

use std::collections::HashMap;
use std::io::{Read, Write};
use std::net::TcpStream;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

pub use stark_provider_abi::{
    BorrowedBuffer, BorrowedBufferMut, ProviderStatus, RawOsHandle, RawResourceHandle,
};

// The net provider's transfer entry point. Declared, never depended on: see
// `stark_provider_abi::RawOsHandle` for why this is a link-level convention rather than a Cargo
// edge or a manifest function.
unsafe extern "C" {
    fn stark_tcp_stream_detach(
        handle: RawResourceHandle,
        out_handle: *mut RawOsHandle,
    ) -> ProviderStatus;
}

/// The net provider's id for `tcp_stream`. It is that provider's index into its own declared
/// resource list (ABI §7), and a transferred handle keeps its OWNER's id (CD-360) — so this is
/// checked against the incoming handle, not overwritten.
pub const TCP_STREAM_RESOURCE_TYPE: u32 = 1;
/// This provider's id for `tls_stream`: index 0 of its one declared resource.
pub const TLS_STREAM_RESOURCE_TYPE: u32 = 0;

// I/O statuses. Deliberately a separate vocabulary from the net provider's rather than a shared
// one: they are different capabilities with different raw error enums, and a shared numbering
// would imply an interchangeability the type system does not grant.
pub const STATUS_CONNECTION_RESET: ProviderStatus = ProviderStatus { code: 1 };
pub const STATUS_BROKEN_PIPE: ProviderStatus = ProviderStatus { code: 2 };
pub const STATUS_TIMED_OUT: ProviderStatus = ProviderStatus { code: 3 };
pub const STATUS_WOULD_BLOCK: ProviderStatus = ProviderStatus { code: 4 };
pub const STATUS_END_OF_STREAM: ProviderStatus = ProviderStatus { code: 5 };
pub const STATUS_INVALID_INPUT: ProviderStatus = ProviderStatus { code: 6 };
pub const STATUS_UNSUPPORTED: ProviderStatus = ProviderStatus { code: 7 };
pub const STATUS_OTHER: ProviderStatus = ProviderStatus { code: 8 };

// Handshake and certificate statuses. Each certificate failure gets its OWN code rather than
// collapsing into one "bad certificate": an operator reading a log needs to distinguish a clock
// problem from a trust problem from a name problem, and a client that reports all three the same
// way sends people to the wrong place.
pub const STATUS_HANDSHAKE_FAILED: ProviderStatus = ProviderStatus { code: 20 };
pub const STATUS_CERTIFICATE_INVALID: ProviderStatus = ProviderStatus { code: 21 };
pub const STATUS_CERTIFICATE_EXPIRED: ProviderStatus = ProviderStatus { code: 22 };
pub const STATUS_CERTIFICATE_NOT_YET_VALID: ProviderStatus = ProviderStatus { code: 23 };
pub const STATUS_CERTIFICATE_UNKNOWN_ISSUER: ProviderStatus = ProviderStatus { code: 24 };
pub const STATUS_HOSTNAME_MISMATCH: ProviderStatus = ProviderStatus { code: 25 };
pub const STATUS_PROTOCOL_VERSION_UNSUPPORTED: ProviderStatus = ProviderStatus { code: 26 };
pub const STATUS_HANDSHAKE_TIMEOUT: ProviderStatus = ProviderStatus { code: 27 };
pub const STATUS_PEER_CLOSED_DURING_HANDSHAKE: ProviderStatus = ProviderStatus { code: 28 };
pub const STATUS_INVALID_SERVER_NAME: ProviderStatus = ProviderStatus { code: 29 };
pub const STATUS_INVALID_CONFIGURATION: ProviderStatus = ProviderStatus { code: 30 };

/// `min_version`/`max_version` and `stark_tls_stream_peer_version`'s output share one encoding.
pub const VERSION_TLS12: u32 = 0;
pub const VERSION_TLS13: u32 = 1;
/// `peer_version` before a version is negotiated. Cannot occur through the STARK surface, which
/// only hands out a `TlsStream` after a completed handshake; declared so the value is not invented
/// at the call site.
pub const VERSION_UNKNOWN: u32 = 0xFFFF_FFFF;

pub const ROOTS_SYSTEM: u32 = 0;
pub const ROOTS_BUNDLED: u32 = 1;
pub const ROOTS_EXPLICIT: u32 = 2;

/// The largest PEM blob a caller may hand in as an explicit root set.
///
/// A bound rather than a trust in the caller: this crosses an ABI, `len` is attacker-influenced in
/// the general case, and rustls-pemfile will happily work through as much input as it is given.
/// 1 MiB is roughly a thousand root certificates — far past any legitimate explicit-root set and
/// far short of a memory problem.
pub const MAX_ROOTS_PEM_BYTES: usize = 1024 * 1024;

/// The longest server name accepted. DNS's own limit is 253 octets; nothing legitimate is longer,
/// and `ServerName::try_from` would reject it anyway. Checked here so the refusal names the input.
pub const MAX_SERVER_NAME_BYTES: usize = 253;

struct Entry {
    stream: rustls::StreamOwned<rustls::ClientConnection, TcpStream>,
}

struct Table {
    next: u64,
    streams: HashMap<u64, Entry>,
}

static TABLE: OnceLock<Mutex<Table>> = OnceLock::new();

fn table() -> &'static Mutex<Table> {
    TABLE.get_or_init(|| {
        Mutex::new(Table {
            next: 1,
            streams: HashMap::new(),
        })
    })
}

/// A contract violation the ABI cannot express as a status: a forged handle, a poisoned lock, an
/// exhausted id space. Aborting rather than returning is ABI §13 — a recoverable status here would
/// let a caller continue on a resource model that has already been broken.
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

fn validate(handle: RawResourceHandle, resource_type: u32) {
    if handle.resource_type != resource_type {
        abort_contract();
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

fn insert(entry: Entry) -> RawResourceHandle {
    let mut table = table().lock().unwrap_or_else(|_| abort_contract());
    let id = next_id(&mut table);
    table.streams.insert(id, entry);
    RawResourceHandle {
        id,
        resource_type: TLS_STREAM_RESOURCE_TYPE,
    }
}

/// Ordinary post-handshake I/O errors.
///
/// `TimedOut` and `WouldBlock` stay DISTINCT from `EndOfStream`, which HC4 requires: a socket that
/// went quiet and a peer that closed are different events, and a client that conflates them either
/// retries a finished response or reports a completed one as truncated.
fn map_io_error(error: &std::io::Error) -> ProviderStatus {
    match error.kind() {
        std::io::ErrorKind::ConnectionReset => STATUS_CONNECTION_RESET,
        std::io::ErrorKind::BrokenPipe => STATUS_BROKEN_PIPE,
        std::io::ErrorKind::TimedOut => STATUS_TIMED_OUT,
        std::io::ErrorKind::WouldBlock => STATUS_WOULD_BLOCK,
        std::io::ErrorKind::UnexpectedEof => STATUS_END_OF_STREAM,
        std::io::ErrorKind::InvalidInput | std::io::ErrorKind::InvalidData => {
            // rustls reports a protocol failure as InvalidData carrying a `rustls::Error`. After
            // the handshake that is still a TLS failure, not a byte-level one, so the specific
            // cause is recovered rather than flattened.
            rustls_error(error).map_or(STATUS_INVALID_INPUT, map_tls_error)
        }
        std::io::ErrorKind::Unsupported => STATUS_UNSUPPORTED,
        _ => STATUS_OTHER,
    }
}

/// The `rustls::Error` inside an `io::Error`, if there is one.
fn rustls_error(error: &std::io::Error) -> Option<&rustls::Error> {
    error.get_ref()?.downcast_ref::<rustls::Error>()
}

/// **The error taxonomy, and the reason it is exhaustive by hand.**
///
/// Each certificate rejection reaches STARK as its own status. rustls has both a plain and a
/// `*Context` form of the three time/name failures — the context form carries the observed and
/// expected values — and they mean the SAME thing to a caller, so both map to one status. Matching
/// only the plain forms is the mistake this comment exists to prevent: `ExpiredContext` is what a
/// real verifier produces, and a `_ => HandshakeFailed` catch-all would have swallowed all three
/// while the fixture tests still passed on their names.
fn map_tls_error(error: &rustls::Error) -> ProviderStatus {
    use rustls::CertificateError as C;
    match error {
        rustls::Error::InvalidCertificate(cert) => match cert {
            C::Expired | C::ExpiredContext { .. } => STATUS_CERTIFICATE_EXPIRED,
            C::NotValidYet | C::NotValidYetContext { .. } => STATUS_CERTIFICATE_NOT_YET_VALID,
            C::NotValidForName | C::NotValidForNameContext { .. } => STATUS_HOSTNAME_MISMATCH,
            C::UnknownIssuer => STATUS_CERTIFICATE_UNKNOWN_ISSUER,
            _ => STATUS_CERTIFICATE_INVALID,
        },
        rustls::Error::InvalidCertRevocationList(_) => STATUS_CERTIFICATE_INVALID,
        rustls::Error::NoCertificatesPresented => STATUS_CERTIFICATE_INVALID,
        rustls::Error::PeerIncompatible(_) => STATUS_PROTOCOL_VERSION_UNSUPPORTED,
        _ => STATUS_HANDSHAKE_FAILED,
    }
}

/// A handshake-phase `io::Error`, which may be a TLS failure in an I/O wrapper or a genuine
/// transport event.
fn map_handshake_error(error: &std::io::Error) -> ProviderStatus {
    if let Some(tls) = rustls_error(error) {
        return map_tls_error(tls);
    }
    match error.kind() {
        // A peer that goes away mid-handshake is its own case. It is the observable signature of a
        // server that rejected the ClientHello without sending an alert, and reporting it as a
        // generic handshake failure loses the one fact that distinguishes it.
        std::io::ErrorKind::UnexpectedEof | std::io::ErrorKind::ConnectionAborted => {
            STATUS_PEER_CLOSED_DURING_HANDSHAKE
        }
        std::io::ErrorKind::ConnectionReset => STATUS_CONNECTION_RESET,
        std::io::ErrorKind::TimedOut | std::io::ErrorKind::WouldBlock => STATUS_HANDSHAKE_TIMEOUT,
        _ => STATUS_HANDSHAKE_FAILED,
    }
}

/// Zero is "no timeout"; anything else is a real bound, floored at 1ns so a sub-resolution request
/// cannot silently become unbounded. Identical to the net provider's rule, deliberately: two
/// providers disagreeing about what a zero duration means would make the STARK-level policy
/// unstatable.
fn timeout_from_nanos(nanos: u64) -> Option<Duration> {
    if nanos == 0 {
        return None;
    }
    let duration = Duration::from_nanos(nanos);
    if duration.is_zero() {
        return Some(Duration::from_nanos(1));
    }
    Some(duration)
}

fn versions_for(
    min: u32,
    max: u32,
) -> Result<Vec<&'static rustls::SupportedProtocolVersion>, ProviderStatus> {
    if min > max {
        return Err(STATUS_INVALID_CONFIGURATION);
    }
    let mut out = Vec::new();
    for code in [VERSION_TLS12, VERSION_TLS13] {
        if code < min || code > max {
            continue;
        }
        out.push(match code {
            VERSION_TLS12 => &rustls::version::TLS12,
            _ => &rustls::version::TLS13,
        });
    }
    // An empty set is a range naming no version this build supports. Refused rather than handed to
    // rustls, whose own error for it would surface as a generic handshake failure.
    if out.is_empty() {
        return Err(STATUS_INVALID_CONFIGURATION);
    }
    Ok(out)
}

/// **HC10 — the platform trust store.**
///
/// CD-361's point, and the one that defused the strongest argument for `native-tls`: system roots
/// can be used WITHOUT handing the protocol to SChannel, Secure Transport or OpenSSL. This loads
/// trust anchors from the platform and hands them to rustls; certificate validation stays
/// rustls-owned, so there is still exactly one verifier to reason about and qualify.
///
/// **A partial load is a failure, not a smaller trust set.** `rustls-native-certs` reports both the
/// anchors it read and the errors it hit, and it is tempting to take what parsed and continue. That
/// silently shrinks trust: the connection that then fails is some unrelated endpoint whose issuer
/// happened to be in the part that did not load, and it fails as `UnknownIssuer` far from the cause.
/// An empty store is refused for the same reason — validating against nothing rejects everything,
/// which reads to a caller as "the whole internet is untrusted".
fn system_root_store() -> Result<rustls::RootCertStore, ProviderStatus> {
    let loaded = rustls_native_certs::load_native_certs();
    if !loaded.errors.is_empty() {
        return Err(STATUS_INVALID_CONFIGURATION);
    }
    if loaded.certs.is_empty() {
        return Err(STATUS_INVALID_CONFIGURATION);
    }
    let mut store = rustls::RootCertStore::empty();
    let (added, ignored) = store.add_parsable_certificates(loaded.certs);
    // `add_parsable_certificates` is the right call HERE and the wrong one for explicit roots: a
    // platform store legitimately contains anchors this verifier cannot use (unsupported algorithms,
    // expired legacy roots), and refusing the whole store for one of them would make TLS unusable on
    // an ordinary machine. An explicit set is small and hand-written, so there a skip is a typo.
    // What is NOT tolerated is ending up with nothing usable.
    if added == 0 {
        let _ = ignored;
        return Err(STATUS_INVALID_CONFIGURATION);
    }
    Ok(store)
}

fn root_store(policy: u32, pem: &[u8]) -> Result<rustls::RootCertStore, ProviderStatus> {
    match policy {
        // CD-361: root acquisition is POLICY, separate from engine selection.
        ROOTS_SYSTEM => system_root_store(),
        // Still refused. A bundled set means vendoring a CA list into the binary and owning its
        // update cadence — a distribution decision, not an implementation gap, and one nobody has
        // taken. Refused by name rather than quietly falling back to the system store, because a
        // caller who asked for a pinned bundle and got the machine's store got the opposite of the
        // property they wanted.
        ROOTS_BUNDLED => Err(STATUS_UNSUPPORTED),
        ROOTS_EXPLICIT => {
            if pem.is_empty() || pem.len() > MAX_ROOTS_PEM_BYTES {
                return Err(STATUS_INVALID_CONFIGURATION);
            }
            let mut reader = std::io::BufReader::new(pem);
            let mut store = rustls::RootCertStore::empty();
            let mut added = 0usize;
            for cert in rustls_pemfile::certs(&mut reader) {
                let Ok(cert) = cert else {
                    return Err(STATUS_INVALID_CONFIGURATION);
                };
                // One bad anchor fails the whole set. `add_parsable_certificates` would skip it and
                // continue, which means a typo silently shrinks the trust set -- and a trust set
                // that is quietly smaller than intended fails much later, as an unrelated
                // UnknownIssuer against a certificate that should have verified.
                if store.add(cert).is_err() {
                    return Err(STATUS_INVALID_CONFIGURATION);
                }
                added += 1;
            }
            if added == 0 {
                return Err(STATUS_INVALID_CONFIGURATION);
            }
            Ok(store)
        }
        _ => Err(STATUS_INVALID_CONFIGURATION),
    }
}

/// Adopts a detached socket. `-1` is refused on every platform — see
/// `RawOsHandle::is_valid_socket`.
#[cfg(unix)]
fn adopt(handle: RawOsHandle) -> Option<TcpStream> {
    use std::os::fd::FromRawFd;
    handle
        .is_valid_socket()
        .then(|| unsafe { TcpStream::from_raw_fd(handle.value as std::os::fd::RawFd) })
}

#[cfg(windows)]
fn adopt(handle: RawOsHandle) -> Option<TcpStream> {
    use std::os::windows::io::FromRawSocket;
    handle.is_valid_socket().then(|| unsafe {
        TcpStream::from_raw_socket(handle.value as std::os::windows::io::RawSocket)
    })
}

/// **`stark_tls_stream_connect` — the cross-provider transfer (CD-360).**
///
/// Consumes a `tcp_stream` the net provider owns and produces a `tls_stream` this provider owns.
/// Ownership of the TCP side passes at call entry and does not return on failure; the socket is
/// adopted before anything can fail, so every error path below closes it by dropping it.
///
/// # Safety
/// `tcp` must be a live `tcp_stream` handle the net provider issued; its `resource_type` is
/// checked, but a stale id is not detectable and aborts.
/// `server_name` and `roots_pem` must each point to `len` initialised bytes the caller owns for the
/// duration of this call, or be zero-length (ABI §9).
/// `out_stream` must be non-null, properly aligned and owned by the caller for this call; it is
/// written only on success (ABI §4.7).
#[no_mangle]
pub unsafe extern "C" fn stark_tls_stream_connect(
    tcp: RawResourceHandle,
    server_name: BorrowedBuffer,
    roots_pem: BorrowedBuffer,
    min_version: u32,
    max_version: u32,
    root_policy: u32,
    handshake_timeout_nanos: u64,
    out_stream: *mut RawResourceHandle,
) -> ProviderStatus {
    // A transferred handle carries its OWNER's type id, unchanged (CD-360). Checking it against
    // this provider's own would reject every legitimate transfer.
    validate(tcp, TCP_STREAM_RESOURCE_TYPE);

    // STEP 1, before any validation that could return: take the socket. See the module header —
    // the handle is consumed whatever happens, so anything that returns before this point strands
    // the socket in the net provider's table.
    let mut detached = RawOsHandle::NONE;
    let status = unsafe { stark_tcp_stream_detach(tcp, &mut detached) };
    if !status.is_success() {
        // Unreachable today: the net provider's detach either succeeds or aborts. If a future owner
        // makes it fallible, this is the one path on which the socket is neither ours nor released,
        // and the honest report is that the transfer failed rather than a TLS-shaped error.
        return STATUS_OTHER;
    }
    let Some(socket) = adopt(detached) else {
        return STATUS_OTHER;
    };

    // STEP 2. From here every `return` drops `socket`, which closes it. That is the whole cleanup
    // story for the failure path, and it is why there is no cleanup code.
    let name_bytes = unsafe { read_buffer(server_name) };
    if name_bytes.is_empty() || name_bytes.len() > MAX_SERVER_NAME_BYTES {
        return STATUS_INVALID_SERVER_NAME;
    }
    let Ok(name_text) = std::str::from_utf8(name_bytes) else {
        return STATUS_INVALID_SERVER_NAME;
    };
    // SNI is mandatory for hostnames and hostname verification is not optional: `ServerName`
    // parsing is what makes both true, since rustls verifies against exactly this value.
    let Ok(name) = rustls_pki_types::ServerName::try_from(name_text) else {
        return STATUS_INVALID_SERVER_NAME;
    };
    let name = name.to_owned();

    let versions = match versions_for(min_version, max_version) {
        Ok(versions) => versions,
        Err(status) => return status,
    };
    let roots = match root_store(root_policy, unsafe { read_buffer(roots_pem) }) {
        Ok(roots) => roots,
        Err(status) => return status,
    };

    // The crypto provider is named EXPLICITLY rather than taken from rustls' process default.
    // CD-361 made aws-lc-rs a decision; a process default is ambient state that a future
    // `install_default` call anywhere in the binary could change without touching this file.
    let provider = Arc::new(rustls::crypto::aws_lc_rs::default_provider());
    let config = match rustls::ClientConfig::builder_with_provider(provider)
        .with_protocol_versions(&versions)
    {
        Ok(builder) => builder.with_root_certificates(roots).with_no_client_auth(),
        Err(_) => return STATUS_INVALID_CONFIGURATION,
    };

    let conn = match rustls::ClientConnection::new(Arc::new(config), name) {
        Ok(conn) => conn,
        Err(error) => return map_tls_error(&error),
    };

    let mut stream = rustls::StreamOwned::new(conn, socket);
    if let Err(status) = handshake(&mut stream, timeout_from_nanos(handshake_timeout_nanos)) {
        return status;
    }

    unsafe { write_scalar(out_stream, insert(Entry { stream })) };
    ProviderStatus::SUCCESS
}

/// Drives the handshake to completion under a TOTAL deadline.
///
/// **A socket read timeout is not a handshake timeout.** HC4 states the distinction and this is
/// where it bites first: a per-read bound is an idle bound, so a peer dribbling one handshake byte
/// at a time stays under it forever. The deadline is therefore held on a monotonic clock and the
/// per-read timeout is recomputed from what remains of it, which bounds the whole phase rather than
/// each read within it.
///
/// The timeouts are cleared on success. They belong to the handshake, and leaving them installed
/// would silently give the caller's first `read` a bound it never asked for — HC4's "timeout
/// configuration must survive TLS wrapping or be deliberately transferred", resolved as
/// *deliberately not transferred*: the STARK package sets its own afterwards.
fn handshake(
    stream: &mut rustls::StreamOwned<rustls::ClientConnection, TcpStream>,
    timeout: Option<Duration>,
) -> Result<(), ProviderStatus> {
    let deadline = timeout.map(|t| Instant::now() + t);

    while stream.conn.is_handshaking() {
        if let Some(deadline) = deadline {
            let Some(remaining) = deadline.checked_duration_since(Instant::now()) else {
                return Err(STATUS_HANDSHAKE_TIMEOUT);
            };
            // Floored at 1ms: a remaining budget that rounds to zero at the OS's resolution would
            // be installed as "no timeout" by most platforms, turning the last instant of a bounded
            // handshake into an unbounded one.
            let slice = remaining.max(Duration::from_millis(1));
            if stream.sock.set_read_timeout(Some(slice)).is_err()
                || stream.sock.set_write_timeout(Some(slice)).is_err()
            {
                return Err(STATUS_OTHER);
            }
        }

        match stream.conn.complete_io(&mut stream.sock) {
            Ok((0, 0)) if stream.conn.is_handshaking() => {
                // No bytes moved and still handshaking. rustls normally reports this as
                // UnexpectedEof; if it ever does not, this prevents a spin.
                return Err(STATUS_PEER_CLOSED_DURING_HANDSHAKE);
            }
            Ok(_) => {}
            Err(error) => {
                let timed_out = matches!(
                    error.kind(),
                    std::io::ErrorKind::TimedOut | std::io::ErrorKind::WouldBlock
                );
                if timed_out && deadline.is_some() {
                    return Err(STATUS_HANDSHAKE_TIMEOUT);
                }
                return Err(map_handshake_error(&error));
            }
        }
    }

    if stream.sock.set_read_timeout(None).is_err() || stream.sock.set_write_timeout(None).is_err() {
        return Err(STATUS_OTHER);
    }
    Ok(())
}

/// `stark_tls_stream_read`, an ABI v0.1 entry point.
///
/// A return of `0` written is end-of-stream, the same convention `stark_tcp_stream_read` uses. For
/// TLS that means a `close_notify` was received — a CLEAN close, distinguishable from a truncated
/// connection, which arrives as `UnexpectedEof` and maps to `EndOfStream` as a status rather than a
/// zero-length success.
///
/// # Safety
/// `stream` must be a handle this provider issued and has not yet closed; its `resource_type` is
/// checked, but a stale id is not detectable and aborts.
/// `out_buffer` must point to `len` writable bytes the caller owns for the duration of this call,
/// or be zero-length.
/// `out_written` must be non-null, properly aligned and owned by the caller for this call.
#[no_mangle]
pub unsafe extern "C" fn stark_tls_stream_read(
    stream: RawResourceHandle,
    out_buffer: BorrowedBufferMut,
    out_written: *mut u64,
) -> ProviderStatus {
    validate(stream, TLS_STREAM_RESOURCE_TYPE);
    if out_buffer.len > 0 && out_buffer.ptr.is_null() {
        abort_contract();
    }
    let mut table = table().lock().unwrap_or_else(|_| abort_contract());
    let Some(entry) = table.streams.get_mut(&stream.id) else {
        abort_contract();
    };
    let slice = if out_buffer.len == 0 {
        &mut []
    } else {
        unsafe { std::slice::from_raw_parts_mut(out_buffer.ptr, out_buffer.len) }
    };
    let read = match entry.stream.read(slice) {
        Ok(read) => read,
        // A peer that vanishes without close_notify is a truncation. rustls surfaces it as
        // UnexpectedEof, and reporting it as a clean zero-length read would let a caller treat a
        // cut-off response as a complete one -- the exact confusion HC7's close-delimited body
        // rules exist to keep decidable.
        Err(error) => return map_io_error(&error),
    };
    unsafe { write_scalar(out_written, read as u64) };
    ProviderStatus::SUCCESS
}

/// `stark_tls_stream_write`, an ABI v0.1 entry point. May accept fewer bytes than offered; the
/// package loops.
///
/// # Safety
/// `stream` must be a handle this provider issued and has not yet closed; its `resource_type` is
/// checked, but a stale id is not detectable and aborts.
/// `data` must point to `len` initialised bytes the caller owns for the duration of this call, or
/// be zero-length (ABI §9).
/// `out_accepted` must be non-null, properly aligned and owned by the caller for this call.
#[no_mangle]
pub unsafe extern "C" fn stark_tls_stream_write(
    stream: RawResourceHandle,
    data: BorrowedBuffer,
    out_accepted: *mut u64,
) -> ProviderStatus {
    validate(stream, TLS_STREAM_RESOURCE_TYPE);
    let data = unsafe { read_buffer(data) };
    let mut table = table().lock().unwrap_or_else(|_| abort_contract());
    let Some(entry) = table.streams.get_mut(&stream.id) else {
        abort_contract();
    };
    let written = match entry.stream.write(data) {
        Ok(written) => written,
        Err(error) => return map_io_error(&error),
    };
    // rustls buffers into its own record layer, so a `write` that returns without a flush can leave
    // ciphertext unsent indefinitely -- a request that is never delivered while the client waits
    // for its response. Flush failures on a socket that just accepted a write are reportable, not
    // ignorable.
    if let Err(error) = entry.stream.flush() {
        return map_io_error(&error);
    }
    unsafe { write_scalar(out_accepted, written as u64) };
    ProviderStatus::SUCCESS
}

/// `stark_tls_stream_set_read_timeout`. Zero nanoseconds means "no timeout"; see
/// [`timeout_from_nanos`].
///
/// # Safety
/// `stream` must be a handle this provider issued and has not yet closed; its `resource_type` is
/// checked, but a stale id is not detectable and aborts.
#[no_mangle]
pub unsafe extern "C" fn stark_tls_stream_set_read_timeout(
    stream: RawResourceHandle,
    nanos: u64,
) -> ProviderStatus {
    validate(stream, TLS_STREAM_RESOURCE_TYPE);
    let mut table = table().lock().unwrap_or_else(|_| abort_contract());
    let Some(entry) = table.streams.get_mut(&stream.id) else {
        abort_contract();
    };
    match entry
        .stream
        .sock
        .set_read_timeout(timeout_from_nanos(nanos))
    {
        Ok(()) => ProviderStatus::SUCCESS,
        Err(error) => map_io_error(&error),
    }
}

/// `stark_tls_stream_set_write_timeout`.
///
/// # Safety
/// `stream` must be a handle this provider issued and has not yet closed; its `resource_type` is
/// checked, but a stale id is not detectable and aborts.
#[no_mangle]
pub unsafe extern "C" fn stark_tls_stream_set_write_timeout(
    stream: RawResourceHandle,
    nanos: u64,
) -> ProviderStatus {
    validate(stream, TLS_STREAM_RESOURCE_TYPE);
    let mut table = table().lock().unwrap_or_else(|_| abort_contract());
    let Some(entry) = table.streams.get_mut(&stream.id) else {
        abort_contract();
    };
    match entry
        .stream
        .sock
        .set_write_timeout(timeout_from_nanos(nanos))
    {
        Ok(()) => ProviderStatus::SUCCESS,
        Err(error) => map_io_error(&error),
    }
}

/// `stark_tls_stream_peer_version` — the negotiated protocol version.
///
/// Purely observational, so it takes a BORROWED handle. It exists because "TLS 1.3 was negotiated"
/// is otherwise unobservable from STARK, and a qualification claim that a TLS 1.2 and a TLS 1.3
/// handshake both succeeded cannot be made by a test that cannot tell them apart.
///
/// # Safety
/// `stream` must be a handle this provider issued and has not yet closed; its `resource_type` is
/// checked, but a stale id is not detectable and aborts.
/// `out_version` must be non-null, properly aligned and owned by the caller for this call.
#[no_mangle]
pub unsafe extern "C" fn stark_tls_stream_peer_version(
    stream: RawResourceHandle,
    out_version: *mut u32,
) -> ProviderStatus {
    validate(stream, TLS_STREAM_RESOURCE_TYPE);
    let table = table().lock().unwrap_or_else(|_| abort_contract());
    let Some(entry) = table.streams.get(&stream.id) else {
        abort_contract();
    };
    let version = match entry.stream.conn.protocol_version() {
        Some(rustls::ProtocolVersion::TLSv1_3) => VERSION_TLS13,
        Some(rustls::ProtocolVersion::TLSv1_2) => VERSION_TLS12,
        _ => VERSION_UNKNOWN,
    };
    unsafe { write_scalar(out_version, version) };
    ProviderStatus::SUCCESS
}

/// `stark_tls_stream_close` — **one close for both effects.**
///
/// HC9's normative rule: `TlsStream` owns the TLS state and the socket, and one close performs the
/// TLS shutdown and the socket close. So this sends `close_notify`, then drops, and the socket's
/// own destructor closes it. There is deliberately no second path: the net provider's
/// `tcp_stream_close` is never called for a transferred socket, because after the detach the net
/// provider holds nothing to close.
///
/// A `close_notify` that cannot be written is ignored. The peer is gone or the socket is broken;
/// the release must still happen, and a close that can fail is a close a caller has to handle,
/// which is how handles get leaked.
///
/// # Safety
/// `handle` must be a handle this provider issued and has not yet closed; its `resource_type` is
/// checked, but a stale id is not detectable and aborts.
#[no_mangle]
pub unsafe extern "C" fn stark_tls_stream_close(handle: RawResourceHandle) -> ProviderStatus {
    validate(handle, TLS_STREAM_RESOURCE_TYPE);
    let mut table = table().lock().unwrap_or_else(|_| abort_contract());
    let Some(mut entry) = table.streams.remove(&handle.id) else {
        abort_contract();
    };
    drop(table);
    entry.stream.conn.send_close_notify();
    let _ = entry.stream.conn.complete_io(&mut entry.stream.sock);
    drop(entry);
    ProviderStatus::SUCCESS
}

/// How many TLS streams this provider currently holds. Test support: "cleanup happened" is
/// otherwise unobservable, and a leak is exactly the failure HC9 must rule out.
pub fn live_stream_count() -> usize {
    table()
        .lock()
        .unwrap_or_else(|_| abort_contract())
        .streams
        .len()
}

// ---------------------------------------------------------------------------------------------
// Controlled TLS peer harness
// ---------------------------------------------------------------------------------------------

/// A local TLS server with a controlled certificate chain.
///
/// HC13 requires qualification not to depend on public internet services, and HC9's negative cases
/// are unreachable without one: nobody operates an expired-certificate endpoint for testing, and a
/// hostname-mismatch case needs a server that presents a name of the tester's choosing. The
/// certificates come from `../fixtures`, generated once with absolute validity windows so an
/// "expired" fixture stays expired — see `fixtures/generate.sh`.
pub mod harness {
    use super::*;
    use std::net::{SocketAddr, TcpListener};
    use std::sync::mpsc;
    use std::thread::{self, JoinHandle};

    /// What the peer does after accepting a TCP connection.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum PeerBehaviour {
        /// Complete the handshake and echo length-prefixed frames, exactly as the net provider's
        /// `EchoServer` does, so a TLS lifecycle test reads like the TCP one.
        Echo,
        /// Complete the handshake, then echo each frame in SEVERAL writes with a flush between
        /// them. Each flush emits its own TLS record, so the client's `read` returns short and a
        /// caller that assumed one read per response gets a truncated answer.
        EchoFragmented,
        /// Accept the TCP connection and then send NOTHING, ever. The client must bound the
        /// handshake itself; a per-read socket timeout would not save it from a peer that dribbles.
        Silent,
        /// Accept the TCP connection and close it immediately, without an alert. This is what a
        /// server refusing a ClientHello outright looks like on the wire.
        CloseImmediately,
    }

    /// Which protocol versions the peer offers. A client asserting "TLS 1.3 was negotiated" proves
    /// nothing unless the peer could have chosen otherwise, and vice versa.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum PeerVersions {
        Both,
        Tls12Only,
        Tls13Only,
    }

    pub struct PeerConfig {
        /// The certificate chain the peer presents, PEM. A chain missing its intermediate is how
        /// the "missing intermediate" case is built — the fixture provides both forms.
        pub chain_pem: &'static str,
        pub key_pem: &'static str,
        pub behaviour: PeerBehaviour,
        pub versions: PeerVersions,
    }

    impl PeerConfig {
        /// The happy path: `server.cert.pem` for `stark.test`, signed by the fixture root.
        pub fn valid() -> Self {
            Self {
                chain_pem: include_str!("../../fixtures/server.cert.pem"),
                key_pem: include_str!("../../fixtures/server.key.pem"),
                behaviour: PeerBehaviour::Echo,
                versions: PeerVersions::Both,
            }
        }

        pub fn with_chain(mut self, chain_pem: &'static str, key_pem: &'static str) -> Self {
            self.chain_pem = chain_pem;
            self.key_pem = key_pem;
            self
        }

        pub fn behaving(mut self, behaviour: PeerBehaviour) -> Self {
            self.behaviour = behaviour;
            self
        }

        pub fn offering(mut self, versions: PeerVersions) -> Self {
            self.versions = versions;
            self
        }
    }

    /// The fixture root, for the client's `ExplicitRoots` policy (CD-361).
    pub const CA_PEM: &str = include_str!("../../fixtures/ca.cert.pem");
    /// A root the fixture chain is NOT signed by, for the untrusted case.
    pub const ROGUE_CA_PEM: &str = include_str!("../../fixtures/rogue-ca.cert.pem");

    pub struct TlsPeer {
        pub address: SocketAddr,
        stop: Arc<std::sync::atomic::AtomicBool>,
        join: Option<JoinHandle<()>>,
    }

    impl TlsPeer {
        pub fn spawn(config: PeerConfig) -> std::io::Result<Self> {
            let listener = TcpListener::bind("127.0.0.1:0")?;
            listener.set_nonblocking(true)?;
            let address = listener.local_addr()?;

            let server_config = Arc::new(build_server_config(&config));
            let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
            let stop_thread = Arc::clone(&stop);
            let (ready_tx, ready_rx) = mpsc::channel();

            let join = thread::spawn(move || {
                let _ = ready_tx.send(());
                while !stop_thread.load(std::sync::atomic::Ordering::Relaxed) {
                    match listener.accept() {
                        Ok((socket, _)) => {
                            let server_config = Arc::clone(&server_config);
                            let behaviour = config.behaviour;
                            thread::spawn(move || serve(socket, server_config, behaviour));
                        }
                        Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                            thread::sleep(Duration::from_millis(5));
                        }
                        Err(_) => return,
                    }
                }
            });
            ready_rx
                .recv_timeout(Duration::from_secs(2))
                .map_err(|_| std::io::Error::other("peer did not signal readiness"))?;
            Ok(Self {
                address,
                stop,
                join: Some(join),
            })
        }
    }

    impl Drop for TlsPeer {
        /// Shutdown in `Drop` rather than an explicit method: a test that fails partway through
        /// would skip the explicit call and leave a listener and its threads behind, and the next
        /// test in the same process would then be running against an unknown number of peers.
        fn drop(&mut self) {
            self.stop.store(true, std::sync::atomic::Ordering::Relaxed);
            // Unblock the accept loop's sleep by connecting once.
            let _ = std::net::TcpStream::connect_timeout(&self.address, Duration::from_millis(200));
            if let Some(join) = self.join.take() {
                let _ = join.join();
            }
        }
    }

    fn build_server_config(config: &PeerConfig) -> rustls::ServerConfig {
        let chain: Vec<_> = rustls_pemfile::certs(&mut config.chain_pem.as_bytes())
            .collect::<Result<_, _>>()
            .expect("fixture chain must parse");
        let key = rustls_pemfile::private_key(&mut config.key_pem.as_bytes())
            .expect("fixture key must parse")
            .expect("fixture key must be present");
        let versions: Vec<&'static rustls::SupportedProtocolVersion> = match config.versions {
            PeerVersions::Both => vec![&rustls::version::TLS12, &rustls::version::TLS13],
            PeerVersions::Tls12Only => vec![&rustls::version::TLS12],
            PeerVersions::Tls13Only => vec![&rustls::version::TLS13],
        };
        rustls::ServerConfig::builder_with_provider(Arc::new(
            rustls::crypto::aws_lc_rs::default_provider(),
        ))
        .with_protocol_versions(&versions)
        .expect("the fixture peer's version set must be supported")
        .with_no_client_auth()
        .with_single_cert(chain, key)
        .expect("fixture chain and key must match")
    }

    fn serve(socket: TcpStream, config: Arc<rustls::ServerConfig>, behaviour: PeerBehaviour) {
        match behaviour {
            PeerBehaviour::CloseImmediately => {
                let _ = socket.shutdown(std::net::Shutdown::Both);
                return;
            }
            PeerBehaviour::Silent => {
                // Hold the connection open, sending nothing. The client's handshake deadline is the
                // only thing that ends this.
                thread::sleep(Duration::from_secs(30));
                return;
            }
            PeerBehaviour::Echo | PeerBehaviour::EchoFragmented => {}
        }

        // **The accepted socket must be put back into blocking mode explicitly.**
        //
        // The listener is non-blocking so the accept loop can poll its stop flag. On Linux an
        // accepted socket does not inherit that; on macOS and the BSDs it DOES. Inherited, every
        // handshake read returns `WouldBlock`, `complete_io` fails at once, and the peer closes —
        // which reaches the client as "peer closed during handshake" for EVERY test, including the
        // ones expecting a certificate verdict. Observed exactly that: ten failures, all status 28,
        // the whole suite finishing in 0.33s because nothing ever waited for anything.
        let _ = socket.set_nonblocking(false);
        let _ = socket.set_read_timeout(Some(Duration::from_secs(10)));
        let _ = socket.set_write_timeout(Some(Duration::from_secs(10)));
        let Ok(conn) = rustls::ServerConnection::new(config) else {
            return;
        };
        let mut stream = rustls::StreamOwned::new(conn, socket);
        if stream.conn.complete_io(&mut stream.sock).is_err() {
            return;
        }

        loop {
            let mut len_bytes = [0u8; 8];
            if stream.read_exact(&mut len_bytes).is_err() {
                return;
            }
            let Ok(len) = usize::try_from(u64::from_be_bytes(len_bytes)) else {
                return;
            };
            if len > 1024 * 1024 {
                return;
            }
            let mut payload = vec![0u8; len];
            if stream.read_exact(&mut payload).is_err() {
                return;
            }
            let ok = match behaviour {
                PeerBehaviour::EchoFragmented => {
                    write_fragmented(&mut stream, &len_bytes, &payload)
                }
                _ => stream
                    .write_all(&len_bytes)
                    .and_then(|_| stream.write_all(&payload))
                    .and_then(|_| stream.flush())
                    .is_ok(),
            };
            if !ok {
                return;
            }
        }
    }

    /// One frame across several TLS records, with a pause between them.
    ///
    /// The flush is what makes this a real fragmentation test: without it rustls coalesces the
    /// writes into one record and the client sees a single read, which would prove nothing.
    fn write_fragmented(
        stream: &mut rustls::StreamOwned<rustls::ServerConnection, TcpStream>,
        len_bytes: &[u8; 8],
        payload: &[u8],
    ) -> bool {
        let mut all = Vec::with_capacity(8 + payload.len());
        all.extend_from_slice(len_bytes);
        all.extend_from_slice(payload);
        for piece in all.chunks(3) {
            if stream.write_all(piece).is_err() || stream.flush().is_err() {
                return false;
            }
            thread::sleep(Duration::from_millis(5));
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::harness::{PeerBehaviour, PeerConfig, PeerVersions, TlsPeer, CA_PEM, ROGUE_CA_PEM};
    use super::*;
    use std::collections::HashSet;

    const SERVER_NAME: &str = "stark.test";

    /// **`live_stream_count` is process-global, and `cargo test` runs tests in parallel.**
    ///
    /// Any test asserting "a failed handshake left no resource" is comparing a global count against
    /// a snapshot, so another test holding a live stream at that instant fails it. Observed: the
    /// handshake-deadline test timed out correctly and then failed on a count of 1 that belonged to
    /// a different test entirely.
    ///
    /// The fix is a serialization lock rather than dropping the assertion, because leak-freedom on
    /// every failure path is one of the two properties HC9 exists to establish.
    ///
    /// **Every test that CREATES a stream holds it, not only those that assert on the count.**
    /// Locking just the assertions is not enough and looks like it is: a concurrent test closing
    /// its own stream moves the global count under an assertion that is holding the lock. Observed
    /// exactly that — `before` was 2 and the count read 1, in a test that had leaked nothing.
    ///
    /// Poisoning is ignored: a panicking test has already failed, and refusing the lock afterwards
    /// would convert one failure into a cascade that hides which test was first.
    static LIFECYCLE: Mutex<()> = Mutex::new(());

    fn lifecycle_lock() -> std::sync::MutexGuard<'static, ()> {
        LIFECYCLE.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn buf(bytes: &[u8]) -> BorrowedBuffer {
        BorrowedBuffer {
            ptr: bytes.as_ptr(),
            len: bytes.len(),
        }
    }

    /// A real `tcp_stream` handle from the NET provider — the only legitimate input to a transfer.
    /// Fabricating one here would test the wrong thing: what HC9 must prove is that a handle the
    /// net provider issued survives the crossing.
    fn connect_tcp(address: std::net::SocketAddr) -> RawResourceHandle {
        let text = address.to_string();
        let mut handle = RawResourceHandle {
            id: 0,
            resource_type: TCP_STREAM_RESOURCE_TYPE,
        };
        let status = unsafe {
            stark_net_native::stark_tcp_stream_connect(buf(text.as_bytes()), &mut handle)
        };
        assert_eq!(status, ProviderStatus::SUCCESS, "the TCP peer must accept");
        handle
    }

    /// One TLS connect attempt against a peer, with everything but the varying parts fixed.
    struct Attempt {
        roots: &'static str,
        server_name: &'static str,
        min: u32,
        max: u32,
        policy: u32,
        timeout_nanos: u64,
    }

    impl Attempt {
        fn new() -> Self {
            Self {
                roots: CA_PEM,
                server_name: SERVER_NAME,
                min: VERSION_TLS12,
                max: VERSION_TLS13,
                policy: ROOTS_EXPLICIT,
                timeout_nanos: 10_000_000_000,
            }
        }

        fn run(&self, address: std::net::SocketAddr) -> (ProviderStatus, RawResourceHandle) {
            let tcp = connect_tcp(address);
            let mut out = RawResourceHandle {
                id: 0,
                resource_type: TLS_STREAM_RESOURCE_TYPE,
            };
            let status = unsafe {
                stark_tls_stream_connect(
                    tcp,
                    buf(self.server_name.as_bytes()),
                    buf(self.roots.as_bytes()),
                    self.min,
                    self.max,
                    self.policy,
                    self.timeout_nanos,
                    &mut out,
                )
            };
            (status, out)
        }
    }

    fn round_trip(stream: RawResourceHandle, payload: &[u8]) -> Vec<u8> {
        let mut frame = Vec::new();
        frame.extend_from_slice(&(payload.len() as u64).to_be_bytes());
        frame.extend_from_slice(payload);

        let mut sent = 0usize;
        while sent < frame.len() {
            let mut accepted = 0u64;
            let status =
                unsafe { stark_tls_stream_write(stream, buf(&frame[sent..]), &mut accepted) };
            assert_eq!(status, ProviderStatus::SUCCESS, "write must succeed");
            assert!(accepted > 0, "a successful write must make progress");
            sent += accepted as usize;
        }

        // Read until the whole frame is back. THIS is the fragmentation-tolerant shape: a caller
        // that read once and stopped would pass against a peer that answers in one record and fail
        // against one that does not.
        let mut received: Vec<u8> = Vec::new();
        let expected = 8 + payload.len();
        while received.len() < expected {
            let mut chunk = [0u8; 64];
            let mut written = 0u64;
            let status = unsafe {
                stark_tls_stream_read(
                    stream,
                    BorrowedBufferMut {
                        ptr: chunk.as_mut_ptr(),
                        len: chunk.len(),
                    },
                    &mut written,
                )
            };
            assert_eq!(status, ProviderStatus::SUCCESS, "read must succeed");
            assert_ne!(written, 0, "the peer closed before the frame completed");
            received.extend_from_slice(&chunk[..written as usize]);
        }
        assert_eq!(
            u64::from_be_bytes(received[..8].try_into().unwrap()) as usize,
            payload.len()
        );
        received[8..].to_vec()
    }

    // -----------------------------------------------------------------------------------------
    // Manifest agreement
    // -----------------------------------------------------------------------------------------

    /// The shipped manifest is the authority (P0.2), so the exports are checked against IT rather
    /// than against a second literal in this file — the mirror-agreeing-with-its-own-mirror failure
    /// CD-219 recorded.
    #[test]
    fn the_shipped_manifest_validates_and_matches_the_symbols_this_crate_links() {
        let text = include_str!("../../../../starkc/providers/stark-tls-native.json");
        let provider = starkc::provider_manifest::parse_provider_manifest(text, "stark-tls-native")
            .expect("the shipped manifest must parse");
        assert_eq!(
            starkc::provider_abi::validate(&provider.metadata),
            Ok(()),
            "the shipped manifest must satisfy the ABI validator"
        );

        let declared: HashSet<&str> = provider
            .metadata
            .functions
            .iter()
            .map(|f| f.name.as_str())
            .collect();
        let exported = HashSet::from([
            "stark_tls_stream_connect",
            "stark_tls_stream_read",
            "stark_tls_stream_write",
            "stark_tls_stream_set_read_timeout",
            "stark_tls_stream_set_write_timeout",
            "stark_tls_stream_peer_version",
            "stark_tls_stream_close",
        ]);
        assert_eq!(declared, exported);
        assert_eq!(provider.metadata.resource_types, vec!["tls_stream"]);
    }

    /// CD-360's declaration, from this side: the manifest must say it consumes the net provider's
    /// `tcp_stream`, name the owner exactly, and claim no close for it.
    #[test]
    fn the_manifest_declares_exactly_one_foreign_consumption_and_claims_no_close_for_it() {
        let text = include_str!("../../../../starkc/providers/stark-tls-native.json");
        let provider = starkc::provider_manifest::parse_provider_manifest(text, "stark-tls-native")
            .expect("the shipped manifest must parse");

        let foreign = &provider.metadata.foreign_resources;
        assert_eq!(
            foreign.len(),
            1,
            "exactly one foreign consumption: {foreign:?}"
        );
        assert_eq!(foreign[0].provider, "stark-std-net");
        assert_eq!(foreign[0].resource, "tcp_stream");

        assert!(
            provider
                .metadata
                .functions
                .iter()
                .all(|f| f.is_close_for.as_deref() != Some("tcp_stream")),
            "a provider may not declare a close for a resource it does not own (CD-360)"
        );
        assert!(
            provider
                .metadata
                .functions
                .iter()
                .any(|f| f.is_close_for.as_deref() == Some("tls_stream")),
            "the resource this provider DOES own must have exactly one close"
        );
    }

    /// The status vocabulary the manifest declares must be the set this file can actually produce.
    /// A code declared and never emitted is a STARK enum variant no program can ever match; a code
    /// emitted and not declared is a contract violation at runtime.
    #[test]
    fn every_declared_status_is_one_this_provider_can_emit() {
        let text = include_str!("../../../../starkc/providers/stark-tls-native.json");
        let provider = starkc::provider_manifest::parse_provider_manifest(text, "stark-tls-native")
            .expect("the shipped manifest must parse");

        let declared: HashSet<u32> = provider
            .status_binding
            .declared_codes()
            .map(|(code, _)| *code)
            .collect();
        let emitted: HashSet<u32> = [
            STATUS_CONNECTION_RESET,
            STATUS_BROKEN_PIPE,
            STATUS_TIMED_OUT,
            STATUS_WOULD_BLOCK,
            STATUS_END_OF_STREAM,
            STATUS_INVALID_INPUT,
            STATUS_UNSUPPORTED,
            STATUS_OTHER,
            STATUS_HANDSHAKE_FAILED,
            STATUS_CERTIFICATE_INVALID,
            STATUS_CERTIFICATE_EXPIRED,
            STATUS_CERTIFICATE_NOT_YET_VALID,
            STATUS_CERTIFICATE_UNKNOWN_ISSUER,
            STATUS_HOSTNAME_MISMATCH,
            STATUS_PROTOCOL_VERSION_UNSUPPORTED,
            STATUS_HANDSHAKE_TIMEOUT,
            STATUS_PEER_CLOSED_DURING_HANDSHAKE,
            STATUS_INVALID_SERVER_NAME,
            STATUS_INVALID_CONFIGURATION,
        ]
        .iter()
        .map(|s| s.code)
        .collect();
        assert_eq!(declared, emitted);
        assert!(
            !declared.contains(&0),
            "0 is SUCCESS and must never be a declared recoverable error"
        );
    }

    // -----------------------------------------------------------------------------------------
    // The happy path, and the transfer it rests on
    // -----------------------------------------------------------------------------------------

    /// **The proving case.** A TCP stream the net provider created crosses into this provider, a
    /// verified TLS session is established over it, application bytes round-trip, and one close
    /// releases both layers.
    #[test]
    fn a_verified_session_round_trips_and_one_close_releases_both_layers() {
        let _serial = lifecycle_lock();
        let peer = TlsPeer::spawn(PeerConfig::valid()).unwrap();
        let before = live_stream_count();

        let (status, stream) = Attempt::new().run(peer.address);
        assert_eq!(
            status,
            ProviderStatus::SUCCESS,
            "the handshake must succeed"
        );
        assert_eq!(
            live_stream_count(),
            before + 1,
            "a successful handshake holds exactly one resource"
        );

        assert_eq!(round_trip(stream, b"hello over tls"), b"hello over tls");
        // Binary, including a NUL: TLS carries bytes, and a length-prefixed frame must not be
        // truncated at one.
        assert_eq!(round_trip(stream, b"a\0b\xff"), b"a\0b\xff");

        assert_eq!(
            unsafe { stark_tls_stream_close(stream) },
            ProviderStatus::SUCCESS
        );
        assert_eq!(
            live_stream_count(),
            before,
            "one close must release the TLS state AND the socket: there is no second close path"
        );
    }

    /// A payload larger than one TLS record (16 KiB) forces the record layer to split, and forces
    /// the caller's read loop to reassemble. A single-read client passes every small test and fails
    /// here.
    #[test]
    fn a_payload_larger_than_one_record_round_trips_intact() {
        let _serial = lifecycle_lock();
        let peer = TlsPeer::spawn(PeerConfig::valid()).unwrap();
        let (status, stream) = Attempt::new().run(peer.address);
        assert_eq!(status, ProviderStatus::SUCCESS);

        let payload: Vec<u8> = (0..70_000u32).map(|i| (i % 251) as u8).collect();
        assert_eq!(round_trip(stream, &payload), payload);

        assert_eq!(
            unsafe { stark_tls_stream_close(stream) },
            ProviderStatus::SUCCESS
        );
    }

    /// Encrypted reads arriving in many small records must reassemble. The peer flushes every three
    /// bytes, so the client cannot complete a frame in one read however large its buffer.
    #[test]
    fn fragmented_encrypted_reads_reassemble() {
        let _serial = lifecycle_lock();
        let peer =
            TlsPeer::spawn(PeerConfig::valid().behaving(PeerBehaviour::EchoFragmented)).unwrap();
        let (status, stream) = Attempt::new().run(peer.address);
        assert_eq!(status, ProviderStatus::SUCCESS);

        assert_eq!(
            round_trip(stream, b"fragmented-across-records"),
            b"fragmented-across-records"
        );

        assert_eq!(
            unsafe { stark_tls_stream_close(stream) },
            ProviderStatus::SUCCESS
        );
    }

    /// **The version claim, made checkable.** "TLS 1.3 succeeded" is only evidence if the peer
    /// could have offered 1.2 and the client can tell which it got.
    #[test]
    fn tls13_and_tls12_each_succeed_and_are_distinguishable() {
        let _serial = lifecycle_lock();
        for (versions, want) in [
            (PeerVersions::Tls13Only, VERSION_TLS13),
            (PeerVersions::Tls12Only, VERSION_TLS12),
        ] {
            let peer = TlsPeer::spawn(PeerConfig::valid().offering(versions)).unwrap();
            let (status, stream) = Attempt::new().run(peer.address);
            assert_eq!(
                status,
                ProviderStatus::SUCCESS,
                "{versions:?} must handshake"
            );

            let mut got = VERSION_UNKNOWN;
            assert_eq!(
                unsafe { stark_tls_stream_peer_version(stream, &mut got) },
                ProviderStatus::SUCCESS
            );
            assert_eq!(got, want, "negotiated the wrong version for {versions:?}");

            assert_eq!(round_trip(stream, b"ping"), b"ping");
            assert_eq!(
                unsafe { stark_tls_stream_close(stream) },
                ProviderStatus::SUCCESS
            );
        }
    }

    // -----------------------------------------------------------------------------------------
    // Certificate and hostname rejection — each with its own status
    // -----------------------------------------------------------------------------------------

    /// **The negative matrix.** Every case differs from the happy path in exactly one property, so
    /// a failure names one cause. Each asserts a SPECIFIC status: a client that rejects everything
    /// as "bad certificate" is safe and useless to operate.
    #[test]
    fn each_certificate_failure_is_rejected_with_its_own_status() {
        let _serial = lifecycle_lock();
        struct Case {
            what: &'static str,
            chain: &'static str,
            key: &'static str,
            server_name: &'static str,
            want: ProviderStatus,
        }
        let cases = [
            Case {
                what: "a chain to a root we do not trust",
                chain: include_str!("../../fixtures/untrusted.cert.pem"),
                key: include_str!("../../fixtures/untrusted.key.pem"),
                server_name: SERVER_NAME,
                want: STATUS_CERTIFICATE_UNKNOWN_ISSUER,
            },
            Case {
                what: "a certificate whose validity window has passed",
                chain: include_str!("../../fixtures/expired.cert.pem"),
                key: include_str!("../../fixtures/expired.key.pem"),
                server_name: SERVER_NAME,
                want: STATUS_CERTIFICATE_EXPIRED,
            },
            Case {
                what: "a certificate whose validity window has not begun",
                chain: include_str!("../../fixtures/not-yet-valid.cert.pem"),
                key: include_str!("../../fixtures/not-yet-valid.key.pem"),
                server_name: SERVER_NAME,
                want: STATUS_CERTIFICATE_NOT_YET_VALID,
            },
            Case {
                what: "a valid certificate for a DIFFERENT name",
                chain: include_str!("../../fixtures/wrong-host.cert.pem"),
                key: include_str!("../../fixtures/wrong-host.key.pem"),
                server_name: SERVER_NAME,
                want: STATUS_HOSTNAME_MISMATCH,
            },
            Case {
                what: "a leaf served without its intermediate",
                chain: include_str!("../../fixtures/chained.cert.pem"),
                key: include_str!("../../fixtures/chained.key.pem"),
                server_name: SERVER_NAME,
                want: STATUS_CERTIFICATE_UNKNOWN_ISSUER,
            },
        ];

        for case in cases {
            let peer =
                TlsPeer::spawn(PeerConfig::valid().with_chain(case.chain, case.key)).unwrap();
            let before = live_stream_count();
            let mut attempt = Attempt::new();
            attempt.server_name = case.server_name;
            let (status, _) = attempt.run(peer.address);

            assert_eq!(
                status, case.want,
                "{}: expected {:?}, got {:?}",
                case.what, case.want, status
            );
            assert_eq!(
                live_stream_count(),
                before,
                "{}: a failed handshake must leave no TLS resource",
                case.what
            );
        }
    }

    /// The CONTROL for the missing-intermediate case. The same leaf and the same key verify when
    /// the intermediate is sent — so the rejection above is about the missing link, not about the
    /// certificate itself.
    #[test]
    fn the_same_leaf_verifies_once_its_intermediate_is_present() {
        let _serial = lifecycle_lock();
        let peer = TlsPeer::spawn(PeerConfig::valid().with_chain(
            include_str!("../../fixtures/chained-fullchain.cert.pem"),
            include_str!("../../fixtures/chained.key.pem"),
        ))
        .unwrap();
        let (status, stream) = Attempt::new().run(peer.address);
        assert_eq!(
            status,
            ProviderStatus::SUCCESS,
            "a complete chain to the fixture root must verify"
        );
        assert_eq!(
            unsafe { stark_tls_stream_close(stream) },
            ProviderStatus::SUCCESS
        );
    }

    /// The happy-path chain against the WRONG trust anchor. This is the other half of the
    /// untrusted case: the certificate is fine, the roots are not, and the same status results —
    /// which is correct, because from the client's position the two are the same event.
    #[test]
    fn a_valid_chain_against_the_wrong_root_is_refused() {
        let _serial = lifecycle_lock();
        let peer = TlsPeer::spawn(PeerConfig::valid()).unwrap();
        let mut attempt = Attempt::new();
        attempt.roots = ROGUE_CA_PEM;
        let (status, _) = attempt.run(peer.address);
        assert_eq!(status, STATUS_CERTIFICATE_UNKNOWN_ISSUER);
    }

    // -----------------------------------------------------------------------------------------
    // Handshake-phase failures
    // -----------------------------------------------------------------------------------------

    /// A peer that accepts the connection and sends nothing. The bound is the client's, and it is a
    /// TOTAL bound rather than a per-read one — see `handshake`.
    #[test]
    fn a_silent_peer_hits_the_handshake_deadline() {
        let _serial = lifecycle_lock();
        let peer = TlsPeer::spawn(PeerConfig::valid().behaving(PeerBehaviour::Silent)).unwrap();
        let before = live_stream_count();

        let mut attempt = Attempt::new();
        attempt.timeout_nanos = 300_000_000; // 300ms
        let started = std::time::Instant::now();
        let (status, _) = attempt.run(peer.address);
        let elapsed = started.elapsed();

        assert_eq!(status, STATUS_HANDSHAKE_TIMEOUT);
        assert!(
            elapsed < Duration::from_secs(5),
            "the deadline must actually bound the phase, took {elapsed:?}"
        );
        assert_eq!(live_stream_count(), before);
    }

    /// A peer that closes without an alert. Distinguished from a generic handshake failure because
    /// it is the observable signature of a server refusing the ClientHello outright.
    #[test]
    fn a_peer_that_closes_during_the_handshake_is_reported_as_such() {
        let _serial = lifecycle_lock();
        let peer =
            TlsPeer::spawn(PeerConfig::valid().behaving(PeerBehaviour::CloseImmediately)).unwrap();
        let before = live_stream_count();
        let (status, _) = Attempt::new().run(peer.address);
        assert!(
            status == STATUS_PEER_CLOSED_DURING_HANDSHAKE || status == STATUS_CONNECTION_RESET,
            "a peer vanishing mid-handshake must be reported as a close or a reset, got {status:?}"
        );
        assert_eq!(live_stream_count(), before);
    }

    /// A client offering only TLS 1.3 against a peer offering only TLS 1.2. What matters is that it
    /// FAILS and leaks nothing; which of the two handshake-class statuses appears depends on which
    /// side detects the mismatch first, and pinning that would be asserting rustls' internals
    /// rather than STARK's contract.
    #[test]
    fn an_unsatisfiable_version_range_fails_without_leaking() {
        let _serial = lifecycle_lock();
        let peer = TlsPeer::spawn(PeerConfig::valid().offering(PeerVersions::Tls12Only)).unwrap();
        let before = live_stream_count();
        let mut attempt = Attempt::new();
        attempt.min = VERSION_TLS13;
        attempt.max = VERSION_TLS13;
        let (status, _) = attempt.run(peer.address);
        assert!(
            status == STATUS_PROTOCOL_VERSION_UNSUPPORTED || status == STATUS_HANDSHAKE_FAILED,
            "expected a handshake-class failure, got {status:?}"
        );
        assert_eq!(live_stream_count(), before);
    }

    // -----------------------------------------------------------------------------------------
    // Configuration refusals — all of which run AFTER the socket is adopted
    // -----------------------------------------------------------------------------------------

    /// **Every configuration refusal still consumes the TCP handle, and must still close the
    /// socket.** This is the ordering constraint from the module header, made checkable: the peer's
    /// accept loop is not even reached for some of these, so what is asserted is that the net
    /// provider no longer holds the stream and that no TLS resource was created.
    #[test]
    fn a_refused_configuration_consumes_the_transfer_and_leaks_nothing() {
        let _serial = lifecycle_lock();
        struct Case {
            what: &'static str,
            mutate: fn(&mut Attempt),
            want: ProviderStatus,
        }
        let cases = [
            Case {
                what: "an empty server name",
                mutate: |a| a.server_name = "",
                want: STATUS_INVALID_SERVER_NAME,
            },
            Case {
                what: "a server name that is not a valid DNS name",
                mutate: |a| a.server_name = "not a hostname",
                want: STATUS_INVALID_SERVER_NAME,
            },
            Case {
                what: "an inverted version range",
                mutate: |a| {
                    a.min = VERSION_TLS13;
                    a.max = VERSION_TLS12;
                },
                want: STATUS_INVALID_CONFIGURATION,
            },
            Case {
                what: "a root policy this ABI does not define",
                mutate: |a| a.policy = 99,
                want: STATUS_INVALID_CONFIGURATION,
            },
            Case {
                what: "explicit roots with nothing in them",
                mutate: |a| a.roots = "",
                want: STATUS_INVALID_CONFIGURATION,
            },
            Case {
                what: "explicit roots that are not PEM certificates",
                mutate: |a| a.roots = "-----BEGIN CERTIFICATE-----\nnot base64\n",
                want: STATUS_INVALID_CONFIGURATION,
            },
            Case {
                what: "a bundled trust store, which nobody has taken the distribution decision for",
                mutate: |a| a.policy = ROOTS_BUNDLED,
                want: STATUS_UNSUPPORTED,
            },
        ];

        let peer = TlsPeer::spawn(PeerConfig::valid()).unwrap();
        for case in cases {
            let before = live_stream_count();
            let mut attempt = Attempt::new();
            (case.mutate)(&mut attempt);

            let tcp = connect_tcp(peer.address);
            let mut out = RawResourceHandle {
                id: 0,
                resource_type: TLS_STREAM_RESOURCE_TYPE,
            };
            let status = unsafe {
                stark_tls_stream_connect(
                    tcp,
                    buf(attempt.server_name.as_bytes()),
                    buf(attempt.roots.as_bytes()),
                    attempt.min,
                    attempt.max,
                    attempt.policy,
                    attempt.timeout_nanos,
                    &mut out,
                )
            };

            assert_eq!(status, case.want, "{}", case.what);
            assert_eq!(
                live_stream_count(),
                before,
                "{}: no TLS resource may exist after a refusal",
                case.what
            );
            // The transfer happened whatever the verdict: the net provider must no longer hold it.
            // If it did, the socket would be stranded -- unreachable from STARK and never closed.
            assert!(
                !stark_net_native::holds_stream(tcp.id),
                "{}: the socket must have left the net provider even though the call failed \
                 (CD-360: consumption is unconditional)",
                case.what
            );
        }
    }

    /// A root set beyond the declared bound is refused before it is parsed. `len` crosses an ABI
    /// and is not this crate's to trust.
    #[test]
    fn an_oversized_root_set_is_refused_before_parsing() {
        let huge = vec![b'x'; MAX_ROOTS_PEM_BYTES + 1];
        assert!(matches!(
            root_store(ROOTS_EXPLICIT, &huge),
            Err(s) if s == STATUS_INVALID_CONFIGURATION
        ));
    }

    /// A server name longer than DNS permits is refused by length, before `ServerName` parsing, so
    /// the diagnostic names the input rather than the parser.
    #[test]
    fn an_oversized_server_name_is_refused() {
        let _serial = lifecycle_lock();
        let peer = TlsPeer::spawn(PeerConfig::valid()).unwrap();
        let long = "a".repeat(MAX_SERVER_NAME_BYTES + 1);
        let tcp = connect_tcp(peer.address);
        let mut out = RawResourceHandle {
            id: 0,
            resource_type: TLS_STREAM_RESOURCE_TYPE,
        };
        let status = unsafe {
            stark_tls_stream_connect(
                tcp,
                buf(long.as_bytes()),
                buf(CA_PEM.as_bytes()),
                VERSION_TLS12,
                VERSION_TLS13,
                ROOTS_EXPLICIT,
                1_000_000_000,
                &mut out,
            )
        };
        assert_eq!(status, STATUS_INVALID_SERVER_NAME);
    }

    // -----------------------------------------------------------------------------------------
    // Timeouts on an established session
    // -----------------------------------------------------------------------------------------

    /// A read timeout on an established session is TERMINAL for the connection (HC4), and must be
    /// distinguishable from end-of-stream — otherwise a caller treats a stalled response as a
    /// complete one.
    #[test]
    fn a_read_timeout_is_reported_as_a_timeout_not_as_end_of_stream() {
        let _serial = lifecycle_lock();
        let peer = TlsPeer::spawn(PeerConfig::valid()).unwrap();
        let (status, stream) = Attempt::new().run(peer.address);
        assert_eq!(status, ProviderStatus::SUCCESS);

        assert_eq!(
            unsafe { stark_tls_stream_set_read_timeout(stream, 200_000_000) },
            ProviderStatus::SUCCESS
        );
        assert_eq!(
            unsafe { stark_tls_stream_set_write_timeout(stream, 5_000_000_000) },
            ProviderStatus::SUCCESS
        );

        // Nothing was sent, so the echo peer has nothing to say.
        let mut chunk = [0u8; 16];
        let mut written = 0u64;
        let status = unsafe {
            stark_tls_stream_read(
                stream,
                BorrowedBufferMut {
                    ptr: chunk.as_mut_ptr(),
                    len: chunk.len(),
                },
                &mut written,
            )
        };
        assert!(
            status == STATUS_TIMED_OUT || status == STATUS_WOULD_BLOCK,
            "a silent peer under a read timeout must time out, got {status:?}"
        );
        assert_ne!(
            status, STATUS_END_OF_STREAM,
            "a timeout is not an end of stream"
        );

        assert_eq!(
            unsafe { stark_tls_stream_close(stream) },
            ProviderStatus::SUCCESS
        );
    }

    /// Zero nanoseconds is "no timeout", the same rule the net provider uses. Two providers
    /// disagreeing about this would make the STARK-level policy unstatable.
    #[test]
    fn zero_nanoseconds_means_no_timeout_here_as_in_the_net_provider() {
        assert_eq!(timeout_from_nanos(0), None);
        assert_eq!(timeout_from_nanos(1), Some(Duration::from_nanos(1)));
        assert_eq!(
            timeout_from_nanos(1_500_000_000),
            Some(Duration::from_millis(1500))
        );
    }

    /// The version encoding is shared between the two config inputs and the observation output. A
    /// mismatch would make `peer_version` report a value the caller cannot compare against what it
    /// requested.
    #[test]
    fn the_version_encoding_is_ordered_and_shared() {
        const { assert!(VERSION_TLS12 < VERSION_TLS13) };
        assert_eq!(versions_for(VERSION_TLS12, VERSION_TLS13).unwrap().len(), 2);
        assert_eq!(versions_for(VERSION_TLS13, VERSION_TLS13).unwrap().len(), 1);
        assert_eq!(versions_for(VERSION_TLS12, VERSION_TLS12).unwrap().len(), 1);
        assert!(versions_for(VERSION_TLS13, VERSION_TLS12).is_err());
    }

    // -----------------------------------------------------------------------------------------
    // HC10 — the platform trust store
    // -----------------------------------------------------------------------------------------

    /// The machine's own trust store loads and yields usable anchors.
    ///
    /// This asserts the LOAD, not a connection: a test that dialled a public endpoint would make
    /// qualification depend on the internet, which HC13 forbids outright.
    #[test]
    fn the_system_trust_store_loads_and_is_not_empty() {
        let store = system_root_store().expect("the platform trust store must load");
        assert!(
            !store.is_empty(),
            "a system store that validates against nothing rejects everything, which reads to a \
             caller as `the whole internet is untrusted`"
        );
    }

    /// **The property that makes `SystemRoots` meaningful, and it is a NEGATIVE one.**
    ///
    /// The fixture CA is not in any machine's trust store. So the same peer that verifies under
    /// `ExplicitRoots` must be REJECTED under `SystemRoots` — offline, deterministic, and the only
    /// way to show the two policies are actually different rather than one silently falling back to
    /// the other. A `SystemRoots` that quietly used the explicit set would pass every positive test.
    #[test]
    fn the_fixture_ca_is_not_trusted_by_the_system_store() {
        let _serial = lifecycle_lock();
        let peer = TlsPeer::spawn(PeerConfig::valid()).unwrap();
        let before = live_stream_count();

        let mut attempt = Attempt::new();
        attempt.policy = ROOTS_SYSTEM;
        attempt.roots = "";
        let (status, _) = attempt.run(peer.address);

        assert_eq!(
            status, STATUS_CERTIFICATE_UNKNOWN_ISSUER,
            "the fixture root must NOT be reachable through the platform store"
        );
        assert_eq!(live_stream_count(), before);
    }

    /// Under `SystemRoots` the explicit PEM is ignored rather than merged. Merging would mean a
    /// caller could widen the platform's trust set by passing bytes alongside a policy that says it
    /// is using the platform's — the trust set would then not be what the policy names.
    #[test]
    fn explicit_roots_are_ignored_under_the_system_policy() {
        let _serial = lifecycle_lock();
        let peer = TlsPeer::spawn(PeerConfig::valid()).unwrap();

        let mut attempt = Attempt::new();
        attempt.policy = ROOTS_SYSTEM;
        attempt.roots = CA_PEM; // the anchor that WOULD verify, under a policy that must not use it
        let (status, _) = attempt.run(peer.address);

        assert_eq!(
            status, STATUS_CERTIFICATE_UNKNOWN_ISSUER,
            "`SystemRoots` must not merge a caller-supplied anchor: the policy names the trust set"
        );
    }
}
