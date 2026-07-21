//! Native Provider ABI v0.1 type definitions (`STARKLANG/docs/compiler/native-provider-abi-v0.1.md`,
//! §7/§9/§11 -- PROPOSED, owner CE4 review pending). A real provider's implementation and a
//! generated binary would both compile against these `#[repr(C)]` types; this module does not
//! implement `extern "C"` linkage, dynamic loading, or invocation -- that is the owning later
//! package's job (§10.2's C5 implementation boundary: type definitions and a compile-time
//! validator only, no provider actually executes in the C5 MVP).

/// §7: the ABI's only cross-boundary resource-carrying type. Opaque to generated code -- never
/// constructed except as a provider function's return value, never read field-by-field.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResourceHandle {
    pub id: u64,
    pub resource_type: u32,
}

/// §9: a borrowed immutable view, valid only for the duration of the call that received it.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct BorrowedBuffer {
    pub ptr: *const u8,
    pub len: usize,
}

/// §9: a borrowed mutable view (admitted in v0.1 so a `read`-shaped function has somewhere to
/// write), same call-duration-only lifetime contract as `BorrowedBuffer`.
#[repr(C)]
#[derive(Debug)]
pub struct BorrowedBufferMut {
    pub ptr: *mut u8,
    pub len: usize,
}

/// §11: every provider function returns this, never a value directly; results come back through
/// an out-parameter. `code == 0` is success; nonzero meaning is defined per-provider (§12: an
/// ordinary provider error, distinct from a STARK trap or a host failure).
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
