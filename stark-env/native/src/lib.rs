//! Native process args/env provider for Native Provider ABI v0.1.

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProviderStatus {
    pub code: u32,
}

impl ProviderStatus {
    pub const SUCCESS: Self = Self { code: 0 };
    pub const INVALID_NAME: Self = Self { code: 1 };
    pub const INVALID_ENCODING: Self = Self { code: 2 };
    pub const BUFFER_TOO_SMALL: Self = Self { code: 3 };
    pub const UNSUPPORTED: Self = Self { code: 4 };
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct BorrowedBuffer {
    pub ptr: *const u8,
    pub len: usize,
}

#[repr(C)]
#[derive(Debug)]
pub struct BorrowedBufferMut {
    pub ptr: *mut u8,
    pub len: usize,
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

unsafe fn write_bytes(out: BorrowedBufferMut, bytes: &[u8]) -> ProviderStatus {
    if bytes.len() > out.len {
        return ProviderStatus::BUFFER_TOO_SMALL;
    }
    if !bytes.is_empty() && out.ptr.is_null() {
        abort_contract();
    }
    if !bytes.is_empty() {
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), out.ptr, bytes.len());
        }
    }
    ProviderStatus::SUCCESS
}

fn args_bytes() -> Result<Vec<u8>, ProviderStatus> {
    let mut encoded = Vec::new();
    for (idx, arg) in std::env::args_os().enumerate() {
        let arg = arg
            .into_string()
            .map_err(|_| ProviderStatus::INVALID_ENCODING)?;
        if idx > 0 {
            encoded.push(0);
        }
        encoded.extend_from_slice(arg.as_bytes());
    }
    Ok(encoded)
}

fn validate_env_name(bytes: &[u8]) -> Result<&str, ProviderStatus> {
    if bytes.is_empty() || bytes.contains(&b'=') || bytes.contains(&0) {
        return Err(ProviderStatus::INVALID_NAME);
    }
    std::str::from_utf8(bytes).map_err(|_| ProviderStatus::INVALID_ENCODING)
}

#[no_mangle]
pub unsafe extern "C" fn stark_env_args_len(out_required_len: *mut u64) -> ProviderStatus {
    let bytes = match args_bytes() {
        Ok(bytes) => bytes,
        Err(status) => return status,
    };
    let Ok(len) = u64::try_from(bytes.len()) else {
        return ProviderStatus::UNSUPPORTED;
    };
    unsafe { write_scalar(out_required_len, len) };
    ProviderStatus::SUCCESS
}

#[no_mangle]
pub unsafe extern "C" fn stark_env_args_fill(
    out_buffer: BorrowedBufferMut,
    out_written: *mut u64,
) -> ProviderStatus {
    let bytes = match args_bytes() {
        Ok(bytes) => bytes,
        Err(status) => return status,
    };
    let status = unsafe { write_bytes(out_buffer, &bytes) };
    if status != ProviderStatus::SUCCESS {
        return status;
    }
    let Ok(written) = u64::try_from(bytes.len()) else {
        return ProviderStatus::UNSUPPORTED;
    };
    unsafe { write_scalar(out_written, written) };
    ProviderStatus::SUCCESS
}

#[no_mangle]
pub unsafe extern "C" fn stark_env_var_len(
    name: BorrowedBuffer,
    out_present: *mut bool,
    out_required_len: *mut u64,
) -> ProviderStatus {
    let name = match validate_env_name(unsafe { read_buffer(name) }) {
        Ok(name) => name,
        Err(status) => return status,
    };
    match std::env::var_os(name) {
        Some(value) => {
            let value = match value.into_string() {
                Ok(value) => value,
                Err(_) => return ProviderStatus::INVALID_ENCODING,
            };
            let Ok(len) = u64::try_from(value.len()) else {
                return ProviderStatus::UNSUPPORTED;
            };
            unsafe {
                write_scalar(out_present, true);
                write_scalar(out_required_len, len);
            }
        }
        None => unsafe {
            write_scalar(out_present, false);
            write_scalar(out_required_len, 0);
        },
    }
    ProviderStatus::SUCCESS
}

#[no_mangle]
pub unsafe extern "C" fn stark_env_var_fill(
    name: BorrowedBuffer,
    out_buffer: BorrowedBufferMut,
    out_written: *mut u64,
) -> ProviderStatus {
    let name = match validate_env_name(unsafe { read_buffer(name) }) {
        Ok(name) => name,
        Err(status) => return status,
    };
    let Some(value) = std::env::var_os(name) else {
        unsafe { write_scalar(out_written, 0) };
        return ProviderStatus::SUCCESS;
    };
    let value = match value.into_string() {
        Ok(value) => value,
        Err(_) => return ProviderStatus::INVALID_ENCODING,
    };
    let status = unsafe { write_bytes(out_buffer, value.as_bytes()) };
    if status != ProviderStatus::SUCCESS {
        return status;
    }
    let Ok(written) = u64::try_from(value.len()) else {
        return ProviderStatus::UNSUPPORTED;
    };
    unsafe { write_scalar(out_written, written) };
    ProviderStatus::SUCCESS
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    fn provider_metadata() -> starkc::backend::provider_abi::ProviderMetadata {
        use starkc::backend::provider_abi::{
            AbiParam, FunctionDecl, ProviderIdentity, ProviderMetadata, ScalarTy,
        };
        ProviderMetadata {
            identity: ProviderIdentity {
                name: "stark-std-env".to_string(),
                semver: (0, 1, 0),
                abi_version: starkc::backend::provider_abi::ABI_VERSION.to_string(),
            },
            target_triples: vec![
                "aarch64-apple-darwin".to_string(),
                "x86_64-apple-darwin".to_string(),
                "x86_64-unknown-linux-gnu".to_string(),
                "x86_64-pc-windows-msvc".to_string(),
            ],
            capabilities: vec!["process.args".to_string(), "process.env".to_string()],
            resource_types: vec![],
            functions: vec![
                FunctionDecl {
                    name: "stark_env_args_len".to_string(),
                    capability: "process.args".to_string(),
                    params: vec![AbiParam::ScalarOut(ScalarTy::U64)],
                    is_close_for: None,
                    may_block: false,
                },
                FunctionDecl {
                    name: "stark_env_args_fill".to_string(),
                    capability: "process.args".to_string(),
                    params: vec![AbiParam::BufferInOut, AbiParam::ScalarOut(ScalarTy::U64)],
                    is_close_for: None,
                    may_block: false,
                },
                FunctionDecl {
                    name: "stark_env_var_len".to_string(),
                    capability: "process.env".to_string(),
                    params: vec![
                        AbiParam::BufferIn,
                        AbiParam::ScalarOut(ScalarTy::Bool),
                        AbiParam::ScalarOut(ScalarTy::U64),
                    ],
                    is_close_for: None,
                    may_block: false,
                },
                FunctionDecl {
                    name: "stark_env_var_fill".to_string(),
                    capability: "process.env".to_string(),
                    params: vec![
                        AbiParam::BufferIn,
                        AbiParam::BufferInOut,
                        AbiParam::ScalarOut(ScalarTy::U64),
                    ],
                    is_close_for: None,
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

    #[test]
    fn metadata_validates_against_real_abi() {
        assert_eq!(
            starkc::backend::provider_abi::validate(&provider_metadata()),
            Ok(())
        );
    }

    #[test]
    fn declared_symbols_are_portable_unique_and_exported_in_this_crate() {
        let metadata = provider_metadata();
        let declared: HashSet<_> = metadata.functions.iter().map(|f| f.name.as_str()).collect();
        let exported = HashSet::from([
            "stark_env_args_len",
            "stark_env_args_fill",
            "stark_env_var_len",
            "stark_env_var_fill",
        ]);
        assert_eq!(declared, exported);
        assert!(declared.iter().all(|name| portable_c_identifier(name)));
        let _ = stark_env_args_len as unsafe extern "C" fn(*mut u64) -> ProviderStatus;
        let _ = stark_env_args_fill
            as unsafe extern "C" fn(BorrowedBufferMut, *mut u64) -> ProviderStatus;
        let _ = stark_env_var_len
            as unsafe extern "C" fn(BorrowedBuffer, *mut bool, *mut u64) -> ProviderStatus;
        let _ = stark_env_var_fill
            as unsafe extern "C" fn(BorrowedBuffer, BorrowedBufferMut, *mut u64) -> ProviderStatus;
    }

    #[test]
    fn status_zero_means_success_and_errors_are_declared() {
        assert_eq!(ProviderStatus::SUCCESS.code, 0);
        assert_eq!(ProviderStatus::INVALID_NAME.code, 1);
        assert_eq!(ProviderStatus::INVALID_ENCODING.code, 2);
        assert_eq!(ProviderStatus::BUFFER_TOO_SMALL.code, 3);
        assert_eq!(ProviderStatus::UNSUPPORTED.code, 4);
    }

    #[test]
    fn env_absent_present_empty_unicode_and_invalid_name() {
        std::env::set_var("STARK_ENV_NATIVE_EMPTY", "");
        std::env::set_var("STARK_ENV_NATIVE_UNICODE", "snowman");
        let mut present = true;
        let mut len = 999;
        let absent = BorrowedBuffer {
            ptr: b"STARK_ENV_NATIVE_ABSENT".as_ptr(),
            len: 23,
        };
        assert_eq!(
            unsafe { stark_env_var_len(absent, &mut present, &mut len) },
            ProviderStatus::SUCCESS
        );
        assert!(!present);
        assert_eq!(len, 0);
        let empty = BorrowedBuffer {
            ptr: b"STARK_ENV_NATIVE_EMPTY".as_ptr(),
            len: 22,
        };
        assert_eq!(
            unsafe { stark_env_var_len(empty, &mut present, &mut len) },
            ProviderStatus::SUCCESS
        );
        assert!(present);
        assert_eq!(len, 0);
        let invalid = BorrowedBuffer {
            ptr: b"BAD=NAME".as_ptr(),
            len: 8,
        };
        assert_eq!(
            unsafe { stark_env_var_len(invalid, &mut present, &mut len) },
            ProviderStatus::INVALID_NAME
        );
    }

    #[test]
    fn buffer_too_small_does_not_write_past_len() {
        std::env::set_var("STARK_ENV_NATIVE_VALUE", "abcd");
        let name = BorrowedBuffer {
            ptr: b"STARK_ENV_NATIVE_VALUE".as_ptr(),
            len: 22,
        };
        let mut out = [0xAAu8; 3];
        let mut written = 77;
        assert_eq!(
            unsafe {
                stark_env_var_fill(
                    name,
                    BorrowedBufferMut {
                        ptr: out.as_mut_ptr(),
                        len: 2,
                    },
                    &mut written,
                )
            },
            ProviderStatus::BUFFER_TOO_SMALL
        );
        assert_eq!(out, [0xAA; 3]);
        assert_eq!(written, 77);
    }

    #[test]
    fn args_query_and_fill_lengths_match() {
        let mut required = u64::MAX;
        assert_eq!(
            unsafe { stark_env_args_len(&mut required) },
            ProviderStatus::SUCCESS
        );
        let mut bytes = vec![0; required as usize];
        let mut written = u64::MAX;
        assert_eq!(
            unsafe {
                stark_env_args_fill(
                    BorrowedBufferMut {
                        ptr: bytes.as_mut_ptr(),
                        len: bytes.len(),
                    },
                    &mut written,
                )
            },
            ProviderStatus::SUCCESS
        );
        assert_eq!(written, required);
    }
}
