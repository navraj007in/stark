//! Native secure randomness provider for Native Provider ABI v0.1.

pub use stark_provider_abi::{BorrowedBufferMut, ProviderStatus};

pub const STATUS_UNAVAILABLE: ProviderStatus = ProviderStatus { code: 1 };
pub const STATUS_LIMIT_EXCEEDED: ProviderStatus = ProviderStatus { code: 2 };
pub const STATUS_OTHER: ProviderStatus = ProviderStatus { code: 3 };

const MAX_SECURE_FILL: usize = 4096;

fn abort_contract() -> ! {
    std::process::abort()
}

/// `stark_random_secure_fill`, an ABI v0.1 entry point.
///
/// # Safety
/// `output` must point to `len` writable bytes the caller owns for the duration of this call, or be zero-length; the caller reads it back afterwards, which is the point of the form.
#[no_mangle]
pub unsafe extern "C" fn stark_random_secure_fill(output: BorrowedBufferMut) -> ProviderStatus {
    if output.len > MAX_SECURE_FILL {
        return STATUS_LIMIT_EXCEEDED;
    }
    if output.len == 0 {
        return ProviderStatus::SUCCESS;
    }
    if output.ptr.is_null() {
        abort_contract();
    }

    let out = unsafe { std::slice::from_raw_parts_mut(output.ptr, output.len) };
    match fill_from_os(out) {
        Ok(()) => ProviderStatus::SUCCESS,
        Err(status) => status,
    }
}

// **No crates.io dependency, deliberately.**
//
// A provider crate is compiled INTO the generated crate for a user's program, and that build runs
// `cargo generate-lockfile --offline`. An external dependency therefore cannot be resolved on any
// machine whose cargo cache does not already hold it — this crate depended on `getrandom` and every
// CI runner failed with "no matching package named `getrandom` found ... note: offline mode".
// It worked locally only because running the crate's own tests had warmed the cache.
//
// Every other first-party provider (`stark-time`, `stark-env`, `stark-file`, `stark-net`) has
// path-only `[dependencies]`; `stark-file`'s `tempfile` is a DEV-dependency and never enters the
// generated crate's graph. This restores that invariant rather than working around the offline
// build, because the invariant is what makes a first-party provider linkable anywhere.

/// Unix: `/dev/urandom`. Blocking-free after boot, and the interface every supported Unix agrees
/// on. Opened per call rather than held: the bound is 4096 bytes and a long-lived descriptor in a
/// library linked into arbitrary programs is a worse trade than an open.
#[cfg(unix)]
fn fill_from_os(out: &mut [u8]) -> Result<(), ProviderStatus> {
    use std::io::Read;
    let mut source = std::fs::File::open("/dev/urandom").map_err(|_| STATUS_UNAVAILABLE)?;
    // `read_exact`, not `read`: a short read must not silently leave part of the buffer as the
    // zeroes the caller allocated. That failure mode is indistinguishable from success at every
    // layer above, which is exactly what must not happen to key material.
    source.read_exact(out).map_err(|_| STATUS_OTHER)
}

/// Windows: `BCryptGenRandom` with the system-preferred RNG, which needs no algorithm handle.
/// There is no `/dev/urandom`, and this is the documented interface — the same one `getrandom`
/// uses on this platform.
#[cfg(windows)]
fn fill_from_os(out: &mut [u8]) -> Result<(), ProviderStatus> {
    #[link(name = "bcrypt")]
    extern "system" {
        fn BCryptGenRandom(
            algorithm: *mut core::ffi::c_void,
            buffer: *mut u8,
            length: u32,
            flags: u32,
        ) -> i32;
    }
    const BCRYPT_USE_SYSTEM_PREFERRED_RNG: u32 = 0x0000_0002;

    // `MAX_SECURE_FILL` is 4096 and is enforced before this is reached, so the cast cannot truncate.
    // Asserted rather than assumed: a later change to that bound must not silently start requesting
    // fewer bytes than the caller asked for.
    debug_assert!(out.len() <= MAX_SECURE_FILL);
    let status = unsafe {
        BCryptGenRandom(
            core::ptr::null_mut(),
            out.as_mut_ptr(),
            out.len() as u32,
            BCRYPT_USE_SYSTEM_PREFERRED_RNG,
        )
    };
    // STATUS_SUCCESS is 0. Anything else means the OS did not fill the buffer.
    if status == 0 {
        Ok(())
    } else {
        Err(STATUS_UNAVAILABLE)
    }
}

/// Neither Unix nor Windows: report unavailable rather than inventing entropy. `STATUS_UNAVAILABLE`
/// is a declared recoverable status, so a program learns it cannot have secure randomness here
/// instead of receiving something that merely looks random.
#[cfg(not(any(unix, windows)))]
fn fill_from_os(_out: &mut [u8]) -> Result<(), ProviderStatus> {
    Err(STATUS_UNAVAILABLE)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn provider_metadata() -> starkc::backend::provider_abi::ProviderMetadata {
        use starkc::backend::provider_abi::{
            AbiParam, FunctionDecl, ProviderIdentity, ProviderMetadata,
        };
        ProviderMetadata {
            identity: ProviderIdentity {
                name: "stark-std-random".to_string(),
                semver: (0, 1, 0),
                abi_version: starkc::backend::provider_abi::ABI_VERSION.to_string(),
            },
            target_triples: vec![
                "aarch64-apple-darwin".to_string(),
                "x86_64-apple-darwin".to_string(),
                "x86_64-unknown-linux-gnu".to_string(),
                "x86_64-pc-windows-msvc".to_string(),
            ],
            capabilities: vec!["random".to_string()],
            resource_types: vec![],
            functions: vec![FunctionDecl {
                name: "stark_random_secure_fill".to_string(),
                capability: "random".to_string(),
                params: vec![AbiParam::BufferInOut],
                is_close_for: None,
                may_block: false,
            }],
        }
    }

    #[test]
    fn metadata_validates_against_provider_abi() {
        starkc::backend::provider_abi::validate(&provider_metadata())
            .expect("random provider metadata must be ABI-valid");
    }

    #[test]
    fn secure_fill_accepts_zero_length() {
        let mut byte = 7u8;
        let status = unsafe {
            stark_random_secure_fill(BorrowedBufferMut {
                ptr: &mut byte,
                len: 0,
            })
        };
        assert_eq!(status, ProviderStatus::SUCCESS);
        assert_eq!(byte, 7);
    }

    #[test]
    fn secure_fill_fills_requested_buffer() {
        let mut bytes = [0u8; 32];
        let status = unsafe {
            stark_random_secure_fill(BorrowedBufferMut {
                ptr: bytes.as_mut_ptr(),
                len: bytes.len(),
            })
        };
        assert_eq!(status, ProviderStatus::SUCCESS);
        assert!(
            bytes.iter().any(|b| *b != 0),
            "all-zero output is astronomically unlikely and indicates the provider did not write"
        );
    }

    #[test]
    fn secure_fill_rejects_provider_limit_without_writing() {
        let mut bytes = [0xA5u8; MAX_SECURE_FILL + 1];
        let status = unsafe {
            stark_random_secure_fill(BorrowedBufferMut {
                ptr: bytes.as_mut_ptr(),
                len: bytes.len(),
            })
        };
        assert_eq!(status, STATUS_LIMIT_EXCEEDED);
        assert!(bytes.iter().all(|b| *b == 0xA5));
    }
}
