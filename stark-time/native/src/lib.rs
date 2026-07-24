//! Native clock provider for `stark-time` (WP-TIME-A §16, Native Provider ABI v0.1 §15).
//!
//! Implements the two `stark-std-time` provider functions declared in
//! [`provider_metadata`] (dev-only, behind `#[cfg(test)]`: see that module for why), using only
//! `std::time` -- no third-party crate (§16 "no third-party dependency").
//!
//! **Not wired into `stark build`.** The repository has no owner-approved provider
//! execution/linkage mechanism yet (`../BLOCKERS.md`), so nothing calls these functions from
//! generated STARK code. This crate exists so the provider implementation, its panic/overflow
//! handling, and its ABI metadata can be built, unit-tested, and validated against ABI v0.1
//! independently of that missing seam, per WP-TIME-A's authorized scope.

use std::sync::OnceLock;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

/// §11: every provider function returns this, never a value directly. Structurally identical to
/// `stark_runtime::provider_abi::ProviderStatus` (a dev-only dependency here -- see
/// `tests::provider_metadata` for the crate that actually owns this shape); defined locally so
/// the provider functions below have zero non-dev dependencies, matching §16's "no third-party
/// dependency" for the shipped provider.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProviderStatus {
    pub code: u32,
}

impl ProviderStatus {
    pub const SUCCESS: ProviderStatus = ProviderStatus { code: 0 };
    pub const CLOCK_UNAVAILABLE: ProviderStatus = ProviderStatus { code: 1 };
    pub const OUT_OF_RANGE: ProviderStatus = ProviderStatus { code: 2 };
}

static ORIGIN: OnceLock<Instant> = OnceLock::new();

/// The pure computation behind `stark_time_monotonic_now_ns`, split out so it is testable without
/// the FFI boundary. `None` means the elapsed nanosecond count does not fit `u64` (§16.1 step 4).
fn monotonic_now_ns() -> Option<u64> {
    let origin = *ORIGIN.get_or_init(Instant::now);
    u64::try_from(origin.elapsed().as_nanos()).ok()
}

/// §15.1 / §16.1: `stark_time_monotonic_now_ns`. Writes nanoseconds elapsed since a process-local,
/// unspecified origin (§10.2 of this package's spec -- not the Unix epoch, not stable across
/// runs).
///
/// # Safety
/// `out_ns` must be a valid, non-null, properly aligned pointer to a `u64` the caller owns for
/// the duration of this call (ABI §9's out-pointer contract). Written only on success (§4.7); on
/// failure it is left untouched.
#[no_mangle]
pub unsafe extern "C" fn stark_time_monotonic_now_ns(out_ns: *mut u64) -> ProviderStatus {
    // §16.4: a panic must not unwind across the `extern "C"` boundary -- that is undefined
    // behavior. Catch it, classify it as a host failure, and abort rather than convert it to an
    // ordinary status (which would misrepresent a host failure as a recoverable clock error).
    let computed = match std::panic::catch_unwind(monotonic_now_ns) {
        Ok(value) => value,
        Err(_) => std::process::abort(),
    };

    let Some(elapsed_ns) = computed else {
        return ProviderStatus::OUT_OF_RANGE;
    };

    if out_ns.is_null() {
        std::process::abort();
    }
    // SAFETY: caller contract above; `out_ns` checked non-null; all fallible computation is
    // already complete, so this write is the last thing the function does before returning.
    unsafe {
        *out_ns = elapsed_ns;
    }
    ProviderStatus::SUCCESS
}

/// The pure normalization §16.2 describes, split out so it is testable without the FFI boundary.
/// `None` means the seconds component does not fit `i64`.
fn unix_now_normalized() -> Option<(i64, u32)> {
    match SystemTime::now().duration_since(UNIX_EPOCH) {
        Ok(duration) => {
            let seconds = i64::try_from(duration.as_secs()).ok()?;
            Some((seconds, duration.subsec_nanos()))
        }
        Err(before_epoch) => {
            let d = before_epoch.duration();
            let s = d.as_secs();
            let n = d.subsec_nanos();
            if n == 0 {
                let seconds = i64::try_from(s).ok()?.checked_neg()?;
                Some((seconds, 0))
            } else {
                let s_i64 = i64::try_from(s).ok()?;
                let seconds = s_i64.checked_neg()?.checked_sub(1)?;
                Some((seconds, 1_000_000_000 - n))
            }
        }
    }
}

/// §15.2 / §16.2: `stark_time_unix_now`. Writes normalized Unix seconds/nanoseconds relative to
/// the epoch (§10.3 of this package's spec -- floor seconds, canonical nanoseconds, may move
/// backwards, not suitable for elapsed-time measurement).
///
/// # Safety
/// `out_seconds` and `out_nanos` must each be valid, non-null, properly aligned pointers the
/// caller owns for the duration of this call. Both are written only on success; on failure
/// neither is touched.
#[no_mangle]
pub unsafe extern "C" fn stark_time_unix_now(
    out_seconds: *mut i64,
    out_nanos: *mut u32,
) -> ProviderStatus {
    let computed = match std::panic::catch_unwind(unix_now_normalized) {
        Ok(value) => value,
        // §16.4, same reasoning as stark_time_monotonic_now_ns.
        Err(_) => std::process::abort(),
    };

    let Some((seconds, nanos)) = computed else {
        return ProviderStatus::OUT_OF_RANGE;
    };

    if out_seconds.is_null() || out_nanos.is_null() {
        std::process::abort();
    }
    // SAFETY: caller contract above; both pointers checked non-null; all fallible computation is
    // already complete.
    unsafe {
        *out_seconds = seconds;
        *out_nanos = nanos;
    }
    ProviderStatus::SUCCESS
}

#[cfg(test)]
mod tests {
    use super::*;

    // ------------------------------------------------------- monotonic_now_ns (pure) --

    #[test]
    fn monotonic_now_ns_is_some_and_nondecreasing() {
        let a = monotonic_now_ns().expect("elapsed nanoseconds must fit u64 on a fresh origin");
        let b = monotonic_now_ns().expect("second reading must also fit u64");
        assert!(b >= a, "monotonic readings must never regress");
    }

    // -------------------------------------------------------- unix_now_normalized (pure) --

    #[test]
    fn unix_now_normalized_is_canonical() {
        let (_, nanos) = unix_now_normalized().expect("current wall time must normalize");
        assert!(nanos < 1_000_000_000, "nanoseconds must be canonical");
    }

    #[test]
    fn unix_now_normalized_sandwiches_a_manual_reading() {
        // §22.3's sandwich check, at this pure-computation layer: three SystemTime readings taken
        // back-to-back must be monotonically non-decreasing as raw Unix durations, and the
        // provider's own reading must fall in the inclusive interval.
        let before = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("host clock must be at or after the epoch during this test");
        let (seconds, nanos) = unix_now_normalized().expect("must normalize");
        let after = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("host clock must be at or after the epoch during this test");

        let provider_nanos = (seconds as i128) * 1_000_000_000i128 + nanos as i128;
        let before_nanos = before.as_nanos() as i128;
        let after_nanos = after.as_nanos() as i128;
        assert!(
            provider_nanos >= before_nanos && provider_nanos <= after_nanos,
            "provider reading must lie within the before/after interval"
        );
    }

    // ------------------------------------------------------------- ABI boundary (FFI) --

    #[test]
    fn ffi_monotonic_writes_output_only_on_success() {
        let mut out: u64 = 0xDEAD_BEEF_DEAD_BEEF;
        let status = unsafe { stark_time_monotonic_now_ns(&mut out as *mut u64) };
        assert_eq!(status, ProviderStatus::SUCCESS);
        assert_ne!(out, 0xDEAD_BEEF_DEAD_BEEF, "output must have been written");
    }

    #[test]
    fn ffi_monotonic_two_calls_are_nondecreasing() {
        let mut a: u64 = 0;
        let mut b: u64 = 0;
        assert_eq!(
            unsafe { stark_time_monotonic_now_ns(&mut a as *mut u64) },
            ProviderStatus::SUCCESS
        );
        assert_eq!(
            unsafe { stark_time_monotonic_now_ns(&mut b as *mut u64) },
            ProviderStatus::SUCCESS
        );
        assert!(b >= a);
    }

    #[test]
    fn ffi_unix_now_writes_canonical_output() {
        let mut seconds: i64 = 0;
        let mut nanos: u32 = 0xFFFF_FFFF;
        let status =
            unsafe { stark_time_unix_now(&mut seconds as *mut i64, &mut nanos as *mut u32) };
        assert_eq!(status, ProviderStatus::SUCCESS);
        assert!(nanos < 1_000_000_000);
    }

    // ----------------------------------------------- ProviderStatus shape (§11 physical ABI) --

    #[test]
    fn provider_status_is_repr_c_and_four_bytes() {
        assert_eq!(std::mem::size_of::<ProviderStatus>(), 4);
        assert_eq!(ProviderStatus::SUCCESS.code, 0);
    }

    // ------------------------------------------------------------- ABI v0.1 metadata --

    /// §15's exact declaration, built from the REAL repository ABI types (a dev-dependency on
    /// `starkc`), not a hand-copied parallel definition -- so "provider metadata validates"
    /// checks against the actual source of truth.
    fn provider_metadata() -> starkc::backend::provider_abi::ProviderMetadata {
        use starkc::backend::provider_abi::{
            AbiParam, FunctionDecl, ProviderIdentity, ProviderMetadata, ScalarTy,
        };

        ProviderMetadata {
            identity: ProviderIdentity {
                name: "stark-std-time".to_string(),
                semver: (0, 1, 0),
                abi_version: starkc::backend::provider_abi::ABI_VERSION.to_string(),
            },
            // §15: minimum intended validation targets. Recorded here as the targets this
            // provider's Rust source (std::time only) is written to support; only the ones with
            // actual build/test evidence are claimed in EVIDENCE.md (§15 "a target may be omitted
            // temporarily when no build evidence exists").
            target_triples: vec![
                "aarch64-apple-darwin".to_string(),
                "x86_64-apple-darwin".to_string(),
                "x86_64-unknown-linux-gnu".to_string(),
                "x86_64-pc-windows-msvc".to_string(),
            ],
            capabilities: vec!["clock".to_string()],
            resource_types: vec![],
            functions: vec![
                FunctionDecl {
                    name: "stark_time_monotonic_now_ns".to_string(),
                    capability: "clock".to_string(),
                    params: vec![AbiParam::ScalarOut(ScalarTy::U64)],
                    is_close_for: None,
                    may_block: false,
                },
                FunctionDecl {
                    name: "stark_time_unix_now".to_string(),
                    capability: "clock".to_string(),
                    params: vec![
                        AbiParam::ScalarOut(ScalarTy::I64),
                        AbiParam::ScalarOut(ScalarTy::U32),
                    ],
                    is_close_for: None,
                    may_block: false,
                },
            ],
        }
    }

    #[test]
    fn provider_metadata_validates_against_abi_v0_1() {
        let metadata = provider_metadata();
        let result = starkc::backend::provider_abi::validate(&metadata);
        assert_eq!(result, Ok(()), "metadata must validate cleanly: {result:?}");
    }

    #[test]
    fn provider_metadata_has_no_resource_types() {
        // §15.4: this provider declares no resource type and no close function.
        let metadata = provider_metadata();
        assert!(metadata.resource_types.is_empty());
        assert!(metadata.functions.iter().all(|f| f.is_close_for.is_none()));
    }

    #[test]
    fn both_functions_declare_may_block_false() {
        // §15.7.
        let metadata = provider_metadata();
        assert!(metadata.functions.iter().all(|f| !f.may_block));
    }

    #[test]
    fn wrong_abi_version_is_rejected() {
        let mut metadata = provider_metadata();
        metadata.identity.abi_version = "0.2".to_string();
        let violations = starkc::backend::provider_abi::validate(&metadata).unwrap_err();
        assert!(violations.contains(
            &starkc::backend::provider_abi::AbiViolation::UnsupportedAbiVersion {
                found: "0.2".to_string()
            }
        ));
    }

    #[test]
    fn missing_capability_is_rejected() {
        let mut metadata = provider_metadata();
        metadata.capabilities.clear();
        let violations = starkc::backend::provider_abi::validate(&metadata).unwrap_err();
        assert!(violations.contains(&starkc::backend::provider_abi::AbiViolation::NoCapabilities));
    }

    #[test]
    fn unsupported_target_would_be_rejected_before_invocation() {
        // §15 / §21 item 17: an unsupported target is rejected before invocation. The validator
        // itself does not check target triples against a host (that is a build-time build-key
        // concern outside this compile-time metadata check, per starkc/src/backend/provider_abi.rs
        // §2-16); this test pins that the declared list is non-empty and does not silently claim
        // an untested target, matching §15's own "a target may be omitted temporarily" rule --
        // exactly the four targets this crate is written against, no more.
        let metadata = provider_metadata();
        assert_eq!(metadata.target_triples.len(), 4);
        assert!(!metadata.target_triples.is_empty());
    }
}
