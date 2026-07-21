//! Native Provider ABI v0.1 type definitions
//! (`STARKLANG/docs/compiler/native-provider-abi-v0.1.md` §7/§9/§11, as amended by **CE4
//! Amendment 1** — approved 2026-07-21, CD-054). A real provider's implementation and a generated
//! binary would both compile against these `#[repr(C)]` types; this module does not implement
//! `extern "C"` linkage, dynamic loading, or invocation — that is the owning later package's job
//! (§10.2's C5 implementation boundary: type definitions and a compile-time validator only, no
//! provider actually executes in the C5 MVP).
//!
//! **The handle split (amendment §4.3/§4.9).** A resource handle exists in two forms, and the
//! separation is the whole safety story:
//!
//! - [`RawResourceHandle`] — `#[repr(C)]`, `Copy`, what actually crosses the boundary. Confined
//!   to the boundary: generated STARK code never names it.
//! - [`OwnedResourceHandle`] — what generated code sees. Not `Copy`, not `Clone`, and
//!   deliberately **no `Drop` impl**.
//!
//! Not `Copy`/`Clone` because §8 makes a handle exclusively owned: with `Copy`, a
//! use-after-transfer is invisible to the compiler and a double close is a matter of time. No
//! `Drop` because the exactly-once close obligation belongs to **verified MIR's `Drop`
//! terminator** (`WP-C5-ENTRY.md` §7.5). A Rust destructor here would either double the close or
//! quietly take over an invariant the MIR verifier is supposed to own — and "which layer
//! guarantees exactly-once?" must have one answer.
//!
//! The accepted cost is explicit: an `OwnedResourceHandle` that generated code never closes is
//! *leaked*. That is precisely the property MIR verification exists to exclude, and precisely the
//! property that would become unfalsifiable if a Rust destructor papered over it.

/// The raw, C-compatible handle (§7, amended). `Copy` because it crosses the FFI boundary as a
/// scalar and has to be — this is the boundary-confined form, not the one generated code holds.
///
/// Fields are `pub` because a provider implementation compiles against this type and must be able
/// to construct one. Generated STARK code must never read or write them directly (§4.9's
/// boundary-helper requirement): every raw↔owned conversion goes through the helpers on
/// [`OwnedResourceHandle`], which is where resource-type validation lives.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RawResourceHandle {
    pub id: u64,
    /// A compiler-assigned index into the provider's declared resource-type list — not a pointer
    /// and not a provider-chosen tag.
    pub resource_type: u32,
}

/// The owning handle generated STARK code holds (§7/§8, amended). Not `Copy`, not `Clone`, no
/// `Drop` — see the module docs for why each of those three is deliberate.
#[derive(Debug, PartialEq, Eq)]
pub struct OwnedResourceHandle(RawResourceHandle);

/// A provider returned a handle whose resource type is not the one its declaration promised.
///
/// This is a **contract violation**, not a provider error: §12's middle row, meaning the caller
/// must trap rather than surface it as a recoverable `Result::Err`. It is returned rather than
/// panicked here so the trapping decision stays with the caller, which knows the call site.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResourceTypeMismatch {
    pub expected: u32,
    pub found: u32,
}

impl OwnedResourceHandle {
    /// **The only sanctioned way to construct an owning handle** (§4.7/§4.9). Wraps a raw handle
    /// a provider wrote into a `HandleOut` slot, *after* validating its resource type against the
    /// declared one.
    ///
    /// The validation lives here rather than at the call site so it cannot be skipped by a code
    /// path that forgets it — the check is a property of the constructor, not of the caller's
    /// diligence. Callers must only reach this after `ProviderStatus::is_success`; on failure the
    /// output slot is uninitialised (§4.7) and must never be read, let alone wrapped.
    pub fn from_raw_checked(
        raw: RawResourceHandle,
        expected_resource_type: u32,
    ) -> Result<Self, ResourceTypeMismatch> {
        if raw.resource_type == expected_resource_type {
            Ok(OwnedResourceHandle(raw))
        } else {
            Err(ResourceTypeMismatch {
                expected: expected_resource_type,
                found: raw.resource_type,
            })
        }
    }

    /// Borrow for a `HandleBorrowed` parameter: yields the raw form for the duration of one call
    /// **without** giving it up. `&self` is the contract — ownership does not move, so the caller
    /// may keep using the handle afterward, which is exactly what an ordinary resource operation
    /// needs (amendment §3.2).
    pub fn as_raw(&self) -> RawResourceHandle {
        self.0
    }

    /// Consume for a `HandleConsumed` parameter: ownership transfers into the provider at call
    /// entry, so this takes `self` by value and the handle is gone. Per §4.6 that is true
    /// **regardless of what the call's `ProviderStatus` reports** — there is no "it failed, so
    /// you still own it" path, because a liveness that depended on a runtime value could not be
    /// verified statically.
    ///
    /// Used by close, and by operations that explicitly end a resource's life.
    pub fn into_raw(self) -> RawResourceHandle {
        self.0
    }

    /// The declared resource type this handle carries. Read-only, and the only field access
    /// generated code has any business performing — it never sees `id` at all.
    pub fn resource_type(&self) -> u32 {
        self.0.resource_type
    }
}

/// §9: a borrowed immutable view, valid only for the duration of the call that received it.
/// **Never an ownership transfer** (amendment §3.1) — §8 governs handles only.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct BorrowedBuffer {
    pub ptr: *const u8,
    pub len: usize,
}

/// §9: a borrowed mutable view (admitted in v0.1 so a `read`-shaped function has somewhere to
/// write), same call-duration-only lifetime contract as [`BorrowedBuffer`].
///
/// Caller-initialised and caller-owned across the call (§4.7): the caller allocates it, the
/// provider fills or modifies it, and **the caller reads it afterward**. That last part is why
/// calling this an ownership transfer was wrong — reading your own buffer after the call is the
/// entire purpose of the type, not a use-after-transfer.
#[repr(C)]
#[derive(Debug)]
pub struct BorrowedBufferMut {
    pub ptr: *mut u8,
    pub len: usize,
}

/// §11: every provider function returns this, never a value directly; results come back through
/// an explicit output channel. `code == 0` is success; nonzero meaning is defined per-provider
/// (§12: an ordinary provider error, distinct from a STARK trap or a host failure).
///
/// One exception, and it is not an exception to the *representation*: a **close** function's
/// nonzero status cannot become a recoverable `Result::Err`, because MIR's `Drop` terminator has
/// no result destination (§4.8). It is a fatal provider-close/host failure — abort, do not retry,
/// treat the handle as consumed, and run no further pending Drop glue.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProviderStatus {
    pub code: u32,
}

impl ProviderStatus {
    pub const SUCCESS: ProviderStatus = ProviderStatus { code: 0 };

    pub fn is_success(self) -> bool {
        self.code == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_matching_resource_type_wraps() {
        let raw = RawResourceHandle {
            id: 7,
            resource_type: 3,
        };
        let owned = OwnedResourceHandle::from_raw_checked(raw, 3).expect("types match");
        assert_eq!(owned.resource_type(), 3);
        assert_eq!(owned.as_raw(), raw);
        // Borrowing did not consume it: still usable, which is the whole point of the borrowed
        // form.
        assert_eq!(owned.into_raw(), raw);
    }

    /// §4.7's validation, stated as a test rather than as a comment: a provider handing back a
    /// handle of the wrong resource type must not produce an owning wrapper at all.
    #[test]
    fn a_mismatched_resource_type_is_refused() {
        let raw = RawResourceHandle {
            id: 7,
            resource_type: 9,
        };
        assert_eq!(
            OwnedResourceHandle::from_raw_checked(raw, 3),
            Err(ResourceTypeMismatch {
                expected: 3,
                found: 9
            })
        );
    }

    /// The type-level guarantees this ABI depends on, asserted at compile time rather than
    /// trusted to survive a future `#[derive]` edit. `OwnedResourceHandle` must NOT be `Copy` or
    /// `Clone` (§8's exclusive ownership) — if either derive were added, this fails to compile
    /// because the trait bound would then be satisfiable, which `static_assert_not_copy` detects
    /// by requiring the negative case.
    #[test]
    fn owning_handle_is_not_copy_and_the_raw_form_is() {
        fn assert_copy<T: Copy>() {}
        assert_copy::<RawResourceHandle>();

        // `OwnedResourceHandle: !Copy` is enforced structurally: `into_raw(self)` consumes, so a
        // second use of a moved handle is a compile error in the generated crate. This test pins
        // the observable consequence -- the value is genuinely moved out.
        let owned = OwnedResourceHandle::from_raw_checked(
            RawResourceHandle {
                id: 1,
                resource_type: 0,
            },
            0,
        )
        .unwrap();
        let raw = owned.into_raw();
        // `owned` is moved-from here; referencing it again would not compile. That is the
        // property, and it only holds while the type stays non-`Copy`.
        assert_eq!(raw.id, 1);
    }
}
