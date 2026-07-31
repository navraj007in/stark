//! Native file provider for Native Provider ABI v0.1.

use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::sync::{Mutex, OnceLock};

pub use stark_provider_abi::{
    BorrowedBuffer, BorrowedBufferMut, ProviderStatus, RawResourceHandle,
};

pub const FILE_RESOURCE_TYPE: u32 = 0;

/// **The package-facing file resource, distinct from Core `File`'s.**
///
/// `file` (type 0) is Core-owned: `ResourceRegistry::builtin()` binds it to `CoreType::File` on the
/// legacy `MirTy::Core` path, and CD-224 forbids a package from claiming it. A package that wants a
/// file handle it can own, move and have closed automatically therefore needs its own resource
/// identity — not a share of Core's.
///
/// `io_file` is that identity. It is an ordinary A11 host resource: `stark-io` binds it, it lowers
/// to `MirTy::HostResource`, and its close runs from a `Drop` terminator exactly as `tcp_stream`'s
/// does. Nothing about Core `File` changes, no compiler guard is weakened, and the two never share
/// a handle — the type tag is checked on every operation, so a `file` handle passed to an `io_file`
/// entry point aborts rather than being reinterpreted.
///
/// The two share this crate's open-file table because they are the same OS objects; they differ in
/// who owns the STARK-side identity, which is the only thing that was ever in dispute.
pub const IO_FILE_RESOURCE_TYPE: u32 = 1;

pub const STATUS_NOT_FOUND: ProviderStatus = ProviderStatus { code: 1 };
pub const STATUS_PERMISSION_DENIED: ProviderStatus = ProviderStatus { code: 2 };
pub const STATUS_INVALID_INPUT: ProviderStatus = ProviderStatus { code: 3 };
pub const STATUS_INVALID_ENCODING: ProviderStatus = ProviderStatus { code: 4 };
pub const STATUS_IS_DIRECTORY: ProviderStatus = ProviderStatus { code: 5 };
pub const STATUS_ALREADY_EXISTS: ProviderStatus = ProviderStatus { code: 6 };
pub const STATUS_UNSUPPORTED: ProviderStatus = ProviderStatus { code: 7 };
pub const STATUS_OTHER_DECLARED: ProviderStatus = ProviderStatus { code: 8 };

struct Table {
    next: u64,
    files: HashMap<u64, File>,
}

static TABLE: OnceLock<Mutex<Table>> = OnceLock::new();

fn table() -> &'static Mutex<Table> {
    TABLE.get_or_init(|| {
        Mutex::new(Table {
            next: 1,
            files: HashMap::new(),
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

fn path_from_buffer(buffer: BorrowedBuffer) -> Result<String, ProviderStatus> {
    let bytes = unsafe { read_buffer(buffer) };
    if bytes.contains(&0) {
        return Err(STATUS_INVALID_INPUT);
    }
    std::str::from_utf8(bytes)
        .map(str::to_owned)
        .map_err(|_| STATUS_INVALID_ENCODING)
}

unsafe fn write_scalar<T: Copy>(out: *mut T, value: T) {
    if out.is_null() {
        abort_contract();
    }
    unsafe {
        *out = value;
    }
}

fn map_io_error(error: &std::io::Error) -> ProviderStatus {
    match error.kind() {
        std::io::ErrorKind::NotFound => STATUS_NOT_FOUND,
        std::io::ErrorKind::PermissionDenied => STATUS_PERMISSION_DENIED,
        std::io::ErrorKind::InvalidInput => STATUS_INVALID_INPUT,
        std::io::ErrorKind::AlreadyExists => STATUS_ALREADY_EXISTS,
        std::io::ErrorKind::Unsupported => STATUS_UNSUPPORTED,
        _ => STATUS_OTHER_DECLARED,
    }
}

fn insert_file_as(file: File, resource_type: u32) -> RawResourceHandle {
    let mut table = table().lock().unwrap_or_else(|_| abort_contract());
    let id = table.next;
    table.next = table
        .next
        .checked_add(1)
        .unwrap_or_else(|| abort_contract());
    table.files.insert(id, file);
    RawResourceHandle { id, resource_type }
}

fn insert_file(file: File) -> RawResourceHandle {
    insert_file_as(file, FILE_RESOURCE_TYPE)
}

/// **Tag check, not a formality.** ABI §13's `from_raw_checked` compares a handle's `resource_type`
/// against the provider's declared list; this is the provider side of the same guarantee. Passing a
/// Core `file` handle to an `io_file` entry point (or the reverse) aborts here rather than being
/// silently reinterpreted, which is what keeps the two identities from sharing a close path.
fn validate_handle_of(handle: RawResourceHandle, expected: u32) {
    if handle.resource_type != expected {
        abort_contract();
    }
}

fn validate_file_handle(handle: RawResourceHandle) {
    validate_handle_of(handle, FILE_RESOURCE_TYPE);
}

#[no_mangle]
pub unsafe extern "C" fn stark_file_open(
    path: BorrowedBuffer,
    out_handle: *mut RawResourceHandle,
) -> ProviderStatus {
    let path = match path_from_buffer(path) {
        Ok(path) => path,
        Err(status) => return status,
    };
    let file = match File::open(path) {
        Ok(file) => file,
        Err(error) if error.kind() == std::io::ErrorKind::IsADirectory => {
            return STATUS_IS_DIRECTORY
        }
        Err(error) => return map_io_error(&error),
    };
    unsafe { write_scalar(out_handle, insert_file(file)) };
    ProviderStatus::SUCCESS
}

#[no_mangle]
pub unsafe extern "C" fn stark_file_create(
    path: BorrowedBuffer,
    out_handle: *mut RawResourceHandle,
) -> ProviderStatus {
    let path = match path_from_buffer(path) {
        Ok(path) => path,
        Err(status) => return status,
    };
    let file = match OpenOptions::new().write(true).create_new(true).open(path) {
        Ok(file) => file,
        Err(error) if error.kind() == std::io::ErrorKind::IsADirectory => {
            return STATUS_IS_DIRECTORY
        }
        Err(error) => return map_io_error(&error),
    };
    unsafe { write_scalar(out_handle, insert_file(file)) };
    ProviderStatus::SUCCESS
}

#[no_mangle]
pub unsafe extern "C" fn stark_file_read(
    handle: RawResourceHandle,
    out_buffer: BorrowedBufferMut,
    out_written: *mut u64,
    out_eof: *mut bool,
) -> ProviderStatus {
    validate_file_handle(handle);
    if out_buffer.len > 0 && out_buffer.ptr.is_null() {
        abort_contract();
    }
    let mut table = table().lock().unwrap_or_else(|_| abort_contract());
    let Some(file) = table.files.get_mut(&handle.id) else {
        abort_contract();
    };
    let slice = if out_buffer.len == 0 {
        &mut []
    } else {
        unsafe { std::slice::from_raw_parts_mut(out_buffer.ptr, out_buffer.len) }
    };
    let read = match file.read(slice) {
        Ok(read) => read,
        Err(error) => return map_io_error(&error),
    };
    unsafe {
        write_scalar(out_written, read as u64);
        write_scalar(out_eof, read == 0);
    }
    ProviderStatus::SUCCESS
}

#[no_mangle]
pub unsafe extern "C" fn stark_file_write(
    handle: RawResourceHandle,
    data: BorrowedBuffer,
    out_accepted: *mut u64,
) -> ProviderStatus {
    validate_file_handle(handle);
    let data = unsafe { read_buffer(data) };
    let mut table = table().lock().unwrap_or_else(|_| abort_contract());
    let Some(file) = table.files.get_mut(&handle.id) else {
        abort_contract();
    };
    let written = match file.write(data) {
        Ok(written) => written,
        Err(error) => return map_io_error(&error),
    };
    unsafe { write_scalar(out_accepted, written as u64) };
    ProviderStatus::SUCCESS
}

#[no_mangle]
pub unsafe extern "C" fn stark_file_complete(handle: RawResourceHandle) -> ProviderStatus {
    validate_file_handle(handle);
    let mut table = table().lock().unwrap_or_else(|_| abort_contract());
    let Some(file) = table.files.get_mut(&handle.id) else {
        abort_contract();
    };
    file.flush()
        .map(|_| ProviderStatus::SUCCESS)
        .unwrap_or_else(|e| map_io_error(&e))
}

#[no_mangle]
pub unsafe extern "C" fn stark_file_close(handle: RawResourceHandle) -> ProviderStatus {
    validate_file_handle(handle);
    let mut table = table().lock().unwrap_or_else(|_| abort_contract());
    if table.files.remove(&handle.id).is_none() {
        abort_contract();
    }
    ProviderStatus::SUCCESS
}

// ------------------------------------------------------------------------------------------
// `io_file` — the package-facing resource
//
// Six entry points mirroring the `file` set above, tagging and checking `IO_FILE_RESOURCE_TYPE`.
// Separate SYMBOLS because a `FunctionDecl` names one symbol and one resource type, so two
// identities need two declarations; separate symbols are what let the two be declared without
// either one shadowing the other.
//
// The bodies delegate to shared helpers rather than being copied: the OS behaviour is identical,
// and only the identity differs.
// ------------------------------------------------------------------------------------------

fn open_impl(path: BorrowedBuffer, resource_type: u32) -> Result<RawResourceHandle, ProviderStatus> {
    let path = path_from_buffer(path)?;
    match File::open(path) {
        Ok(file) => Ok(insert_file_as(file, resource_type)),
        Err(error) if error.kind() == std::io::ErrorKind::IsADirectory => Err(STATUS_IS_DIRECTORY),
        Err(error) => Err(map_io_error(&error)),
    }
}

fn create_impl(
    path: BorrowedBuffer,
    resource_type: u32,
) -> Result<RawResourceHandle, ProviderStatus> {
    let path = path_from_buffer(path)?;
    match OpenOptions::new().write(true).create_new(true).open(path) {
        Ok(file) => Ok(insert_file_as(file, resource_type)),
        Err(error) if error.kind() == std::io::ErrorKind::IsADirectory => Err(STATUS_IS_DIRECTORY),
        Err(error) => Err(map_io_error(&error)),
    }
}

#[no_mangle]
pub unsafe extern "C" fn stark_iofile_open(
    path: BorrowedBuffer,
    out_handle: *mut RawResourceHandle,
) -> ProviderStatus {
    match open_impl(path, IO_FILE_RESOURCE_TYPE) {
        Ok(handle) => {
            unsafe { write_scalar(out_handle, handle) };
            ProviderStatus::SUCCESS
        }
        Err(status) => status,
    }
}

#[no_mangle]
pub unsafe extern "C" fn stark_iofile_create(
    path: BorrowedBuffer,
    out_handle: *mut RawResourceHandle,
) -> ProviderStatus {
    match create_impl(path, IO_FILE_RESOURCE_TYPE) {
        Ok(handle) => {
            unsafe { write_scalar(out_handle, handle) };
            ProviderStatus::SUCCESS
        }
        Err(status) => status,
    }
}

#[no_mangle]
pub unsafe extern "C" fn stark_iofile_read(
    handle: RawResourceHandle,
    out_buffer: BorrowedBufferMut,
    out_written: *mut u64,
    out_eof: *mut bool,
) -> ProviderStatus {
    validate_handle_of(handle, IO_FILE_RESOURCE_TYPE);
    if out_buffer.len > 0 && out_buffer.ptr.is_null() {
        abort_contract();
    }
    let mut table = table().lock().unwrap_or_else(|_| abort_contract());
    let Some(file) = table.files.get_mut(&handle.id) else {
        abort_contract();
    };
    let slice = if out_buffer.len == 0 {
        &mut []
    } else {
        unsafe { std::slice::from_raw_parts_mut(out_buffer.ptr, out_buffer.len) }
    };
    let read = match file.read(slice) {
        Ok(read) => read,
        Err(error) => return map_io_error(&error),
    };
    unsafe {
        write_scalar(out_written, read as u64);
        write_scalar(out_eof, read == 0);
    }
    ProviderStatus::SUCCESS
}

#[no_mangle]
pub unsafe extern "C" fn stark_iofile_write(
    handle: RawResourceHandle,
    data: BorrowedBuffer,
    out_accepted: *mut u64,
) -> ProviderStatus {
    validate_handle_of(handle, IO_FILE_RESOURCE_TYPE);
    let data = unsafe { read_buffer(data) };
    let mut table = table().lock().unwrap_or_else(|_| abort_contract());
    let Some(file) = table.files.get_mut(&handle.id) else {
        abort_contract();
    };
    let written = match file.write(data) {
        Ok(written) => written,
        Err(error) => return map_io_error(&error),
    };
    unsafe { write_scalar(out_accepted, written as u64) };
    ProviderStatus::SUCCESS
}

#[no_mangle]
pub unsafe extern "C" fn stark_iofile_complete(handle: RawResourceHandle) -> ProviderStatus {
    validate_handle_of(handle, IO_FILE_RESOURCE_TYPE);
    let mut table = table().lock().unwrap_or_else(|_| abort_contract());
    let Some(file) = table.files.get_mut(&handle.id) else {
        abort_contract();
    };
    file.flush()
        .map(|_| ProviderStatus::SUCCESS)
        .unwrap_or_else(|e| map_io_error(&e))
}

#[no_mangle]
pub unsafe extern "C" fn stark_iofile_close(handle: RawResourceHandle) -> ProviderStatus {
    validate_handle_of(handle, IO_FILE_RESOURCE_TYPE);
    let mut table = table().lock().unwrap_or_else(|_| abort_contract());
    if table.files.remove(&handle.id).is_none() {
        abort_contract();
    }
    ProviderStatus::SUCCESS
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;
    use std::io::Write;

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
        let file = "file".to_string();
        let io_file = "io_file".to_string();
        ProviderMetadata {
            identity: ProviderIdentity {
                name: "stark-std-file".to_string(),
                semver: (0, 1, 0),
                abi_version: starkc::backend::provider_abi::ABI_VERSION.to_string(),
            },
            target_triples: vec![
                "aarch64-apple-darwin".to_string(),
                "x86_64-apple-darwin".to_string(),
                "x86_64-unknown-linux-gnu".to_string(),
                "x86_64-pc-windows-msvc".to_string(),
            ],
            capabilities: vec!["filesystem".to_string()],
            resource_types: vec![file.clone(), io_file.clone()],
            functions: vec![
                FunctionDecl {
                    name: "stark_file_open".to_string(),
                    capability: "filesystem".to_string(),
                    params: vec![
                        AbiParam::BufferIn,
                        AbiParam::HandleOut {
                            resource_type: file.clone(),
                        },
                    ],
                    is_close_for: None,
                    may_block: true,
                },
                FunctionDecl {
                    name: "stark_file_create".to_string(),
                    capability: "filesystem".to_string(),
                    params: vec![
                        AbiParam::BufferIn,
                        AbiParam::HandleOut {
                            resource_type: file.clone(),
                        },
                    ],
                    is_close_for: None,
                    may_block: true,
                },
                FunctionDecl {
                    name: "stark_file_read".to_string(),
                    capability: "filesystem".to_string(),
                    params: vec![
                        AbiParam::HandleBorrowed {
                            resource_type: file.clone(),
                        },
                        AbiParam::BufferInOut,
                        AbiParam::ScalarOut(ScalarTy::U64),
                        AbiParam::ScalarOut(ScalarTy::Bool),
                    ],
                    is_close_for: None,
                    may_block: true,
                },
                FunctionDecl {
                    name: "stark_file_write".to_string(),
                    capability: "filesystem".to_string(),
                    params: vec![
                        AbiParam::HandleBorrowed {
                            resource_type: file.clone(),
                        },
                        AbiParam::BufferIn,
                        AbiParam::ScalarOut(ScalarTy::U64),
                    ],
                    is_close_for: None,
                    may_block: true,
                },
                FunctionDecl {
                    name: "stark_file_complete".to_string(),
                    capability: "filesystem".to_string(),
                    params: vec![AbiParam::HandleBorrowed {
                        resource_type: file.clone(),
                    }],
                    is_close_for: None,
                    may_block: true,
                },
                FunctionDecl {
                    name: "stark_file_close".to_string(),
                    capability: "filesystem".to_string(),
                    params: vec![AbiParam::HandleConsumed {
                        resource_type: file.clone(),
                    }],
                    is_close_for: Some(file.clone()),
                    may_block: true,
                },
                FunctionDecl {
                    name: "stark_iofile_open".to_string(),
                    capability: "filesystem".to_string(),
                    params: vec![
                        AbiParam::BufferIn,
                        AbiParam::HandleOut {
                            resource_type: io_file.clone(),
                        },
                    ],
                    is_close_for: None,
                    may_block: true,
                },
                FunctionDecl {
                    name: "stark_iofile_create".to_string(),
                    capability: "filesystem".to_string(),
                    params: vec![
                        AbiParam::BufferIn,
                        AbiParam::HandleOut {
                            resource_type: io_file.clone(),
                        },
                    ],
                    is_close_for: None,
                    may_block: true,
                },
                FunctionDecl {
                    name: "stark_iofile_read".to_string(),
                    capability: "filesystem".to_string(),
                    params: vec![
                        AbiParam::HandleBorrowed {
                            resource_type: io_file.clone(),
                        },
                        AbiParam::BufferInOut,
                        AbiParam::ScalarOut(ScalarTy::U64),
                        AbiParam::ScalarOut(ScalarTy::Bool),
                    ],
                    is_close_for: None,
                    may_block: true,
                },
                FunctionDecl {
                    name: "stark_iofile_write".to_string(),
                    capability: "filesystem".to_string(),
                    params: vec![
                        AbiParam::HandleBorrowed {
                            resource_type: io_file.clone(),
                        },
                        AbiParam::BufferIn,
                        AbiParam::ScalarOut(ScalarTy::U64),
                    ],
                    is_close_for: None,
                    may_block: true,
                },
                FunctionDecl {
                    name: "stark_iofile_complete".to_string(),
                    capability: "filesystem".to_string(),
                    params: vec![AbiParam::HandleBorrowed {
                        resource_type: io_file.clone(),
                    }],
                    is_close_for: None,
                    may_block: true,
                },
                FunctionDecl {
                    name: "stark_iofile_close".to_string(),
                    capability: "filesystem".to_string(),
                    params: vec![AbiParam::HandleConsumed {
                        resource_type: io_file.clone(),
                    }],
                    is_close_for: Some(io_file),
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
            pub fn stark_file_open(
                path: BorrowedBuffer,
                out_handle: *mut RawResourceHandle,
            ) -> ProviderStatus;
            pub fn stark_file_create(
                path: BorrowedBuffer,
                out_handle: *mut RawResourceHandle,
            ) -> ProviderStatus;
            pub fn stark_file_read(
                handle: RawResourceHandle,
                out_buffer: BorrowedBufferMut,
                out_written: *mut u64,
                out_eof: *mut bool,
            ) -> ProviderStatus;
            pub fn stark_file_write(
                handle: RawResourceHandle,
                data: BorrowedBuffer,
                out_accepted: *mut u64,
            ) -> ProviderStatus;
            pub fn stark_file_complete(handle: RawResourceHandle) -> ProviderStatus;
            pub fn stark_file_close(handle: RawResourceHandle) -> ProviderStatus;
        }
    }

    #[test]
    fn metadata_validates_and_symbols_match() {
        let metadata = provider_metadata();
        assert_eq!(starkc::backend::provider_abi::validate(&metadata), Ok(()));
        let declared: HashSet<_> = metadata.functions.iter().map(|f| f.name.as_str()).collect();
        let exported = HashSet::from([
            "stark_file_open",
            "stark_file_create",
            "stark_file_read",
            "stark_file_write",
            "stark_file_complete",
            "stark_file_close",
            "stark_iofile_open",
            "stark_iofile_create",
            "stark_iofile_read",
            "stark_iofile_write",
            "stark_iofile_complete",
            "stark_iofile_close",
        ]);
        assert_eq!(declared, exported);
        assert!(declared.iter().all(|name| portable_c_identifier(name)));
        assert_eq!(metadata.resource_types, vec!["file", "io_file"]);
        assert_eq!(
            metadata
                .functions
                .iter()
                .filter(|f| f.is_close_for.as_deref() == Some("file"))
                .count(),
            1
        );
        // **Exactly one close per resource type.** Two would give a resource two destruction paths,
        // which is the defect A11 §5 rule 4 exists to prevent -- and the reason `io_file` needed its
        // own close symbol rather than sharing `file`'s.
        assert_eq!(
            metadata
                .functions
                .iter()
                .filter(|f| f.is_close_for.as_deref() == Some("io_file"))
                .count(),
            1
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
    fn declared_symbols_are_externally_linkable() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("linked.bin");
        let path = path.to_string_lossy();
        let mut handle = RawResourceHandle {
            id: 0,
            resource_type: FILE_RESOURCE_TYPE,
        };
        assert_eq!(
            unsafe { linked::stark_file_create(buf(path.as_bytes()), &mut handle) },
            ProviderStatus::SUCCESS
        );
        let mut accepted = 0;
        assert_eq!(
            unsafe { linked::stark_file_write(handle, buf(b"x"), &mut accepted) },
            ProviderStatus::SUCCESS
        );
        assert_eq!(accepted, 1);
        assert_eq!(
            unsafe { linked::stark_file_complete(handle) },
            ProviderStatus::SUCCESS
        );
        assert_eq!(
            unsafe { linked::stark_file_close(handle) },
            ProviderStatus::SUCCESS
        );

        let mut read_handle = RawResourceHandle {
            id: 0,
            resource_type: FILE_RESOURCE_TYPE,
        };
        assert_eq!(
            unsafe { linked::stark_file_open(buf(path.as_bytes()), &mut read_handle) },
            ProviderStatus::SUCCESS
        );
        let mut out = [0u8; 1];
        let mut written = 0;
        let mut eof = false;
        assert_eq!(
            unsafe {
                linked::stark_file_read(
                    read_handle,
                    BorrowedBufferMut {
                        ptr: out.as_mut_ptr(),
                        len: out.len(),
                    },
                    &mut written,
                    &mut eof,
                )
            },
            ProviderStatus::SUCCESS
        );
        assert_eq!(written, 1);
        assert_eq!(
            unsafe { linked::stark_file_close(read_handle) },
            ProviderStatus::SUCCESS
        );
    }

    #[test]
    fn status_zero_means_success_and_declared_errors_are_stable() {
        assert_eq!(ProviderStatus::SUCCESS.code, 0);
        assert_eq!(STATUS_OTHER_DECLARED.code, 8);
    }

    #[test]
    fn open_read_binary_and_close() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("binary file.bin");
        std::fs::write(&path, b"a\0b").unwrap();
        let path = path.to_string_lossy();
        let mut handle = RawResourceHandle {
            id: 0,
            resource_type: FILE_RESOURCE_TYPE,
        };
        assert_eq!(
            unsafe { stark_file_open(buf(path.as_bytes()), &mut handle) },
            ProviderStatus::SUCCESS
        );
        let mut out = [0u8; 8];
        let mut written = 0;
        let mut eof = false;
        assert_eq!(
            unsafe {
                stark_file_read(
                    handle,
                    BorrowedBufferMut {
                        ptr: out.as_mut_ptr(),
                        len: out.len(),
                    },
                    &mut written,
                    &mut eof,
                )
            },
            ProviderStatus::SUCCESS
        );
        assert_eq!(&out[..written as usize], b"a\0b");
        assert_eq!(unsafe { stark_file_close(handle) }, ProviderStatus::SUCCESS);
    }

    #[test]
    fn create_write_complete_and_read_back() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("unicode-snowman.txt");
        let path_string = path.to_string_lossy();
        let mut handle = RawResourceHandle {
            id: 0,
            resource_type: FILE_RESOURCE_TYPE,
        };
        assert_eq!(
            unsafe { stark_file_create(buf(path_string.as_bytes()), &mut handle) },
            ProviderStatus::SUCCESS
        );
        let mut accepted = 0;
        assert_eq!(
            unsafe { stark_file_write(handle, buf(b"hello\0world"), &mut accepted) },
            ProviderStatus::SUCCESS
        );
        assert_eq!(accepted, 11);
        assert_eq!(
            unsafe { stark_file_complete(handle) },
            ProviderStatus::SUCCESS
        );
        assert_eq!(unsafe { stark_file_close(handle) }, ProviderStatus::SUCCESS);
        assert_eq!(std::fs::read(path).unwrap(), b"hello\0world");
    }

    #[test]
    fn missing_existing_invalid_and_directory_paths_are_declared_errors() {
        let dir = tempfile::tempdir().unwrap();
        let missing = dir
            .path()
            .join("missing.txt")
            .to_string_lossy()
            .into_owned();
        let mut handle = RawResourceHandle {
            id: 0,
            resource_type: FILE_RESOURCE_TYPE,
        };
        assert_eq!(
            unsafe { stark_file_open(buf(missing.as_bytes()), &mut handle) },
            STATUS_NOT_FOUND
        );
        assert_eq!(
            unsafe { stark_file_open(buf(b"bad\0path"), &mut handle) },
            STATUS_INVALID_INPUT
        );
        let dir_path = dir.path().to_string_lossy();
        let status = unsafe { stark_file_open(buf(dir_path.as_bytes()), &mut handle) };
        if status == ProviderStatus::SUCCESS {
            let mut out = [0u8; 1];
            let mut written = 0;
            let mut eof = false;
            let read_status = unsafe {
                stark_file_read(
                    handle,
                    BorrowedBufferMut {
                        ptr: out.as_mut_ptr(),
                        len: out.len(),
                    },
                    &mut written,
                    &mut eof,
                )
            };
            // Windows reports some directory-as-file reads as `PermissionDenied`; Unix-family
            // hosts usually report `IsADirectory`. Both are declared recoverable statuses.
            assert!(matches!(read_status.code, 2 | 5 | 8));
            assert_eq!(unsafe { stark_file_close(handle) }, ProviderStatus::SUCCESS);
        } else {
            assert!(matches!(status.code, 2 | 5 | 8));
        }
    }

    #[test]
    fn create_existing_reports_already_exists() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("exists.txt");
        File::create(&path).unwrap().write_all(b"x").unwrap();
        let path = path.to_string_lossy();
        let mut handle = RawResourceHandle {
            id: 0,
            resource_type: FILE_RESOURCE_TYPE,
        };
        assert_eq!(
            unsafe { stark_file_create(buf(path.as_bytes()), &mut handle) },
            STATUS_ALREADY_EXISTS
        );
    }
}
