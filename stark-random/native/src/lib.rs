//! Native secure randomness provider for Native Provider ABI v0.1.

pub use stark_provider_abi::{BorrowedBufferMut, ProviderStatus};

pub const STATUS_UNAVAILABLE: ProviderStatus = ProviderStatus { code: 1 };
pub const STATUS_LIMIT_EXCEEDED: ProviderStatus = ProviderStatus { code: 2 };
pub const STATUS_OTHER: ProviderStatus = ProviderStatus { code: 3 };

const MAX_SECURE_FILL: usize = 4096;

fn abort_contract() -> ! {
    std::process::abort()
}

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
    match getrandom::fill(out) {
        Ok(()) => ProviderStatus::SUCCESS,
        Err(error) if error.raw_os_error().is_none() => STATUS_UNAVAILABLE,
        Err(_) => STATUS_OTHER,
    }
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
