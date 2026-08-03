//! Compiler-independent physical Native Provider ABI v0.1 boundary types.
//!
//! This crate intentionally contains only C-compatible physical boundary declarations. It has no
//! metadata validator, resolver, MIR, package, backend, or runtime ownership logic.

pub const ABI_VERSION: &str = "0.1";

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

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RawResourceHandle {
    pub id: u64,
    pub resource_type: u32,
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

/// **HC9 — the physical currency of a cross-provider resource transfer.**
///
/// CD-360 ruled what a transfer MEANS: a `HandleConsumed` of a foreign resource takes ownership at
/// call entry, on success and on failure alike, and the consuming provider releases the underlying
/// native resource if it fails before producing its own handle. That ruling settled the ownership
/// question completely and left one thing unstated, because no transfer had yet been implemented:
/// **how the consuming provider obtains the native object.**
///
/// A `RawResourceHandle` is an index into the OWNING provider's private table. The consumer cannot
/// read that table — it is a `static` in another crate — so the handle alone conveys ownership
/// without conveying the socket. This type, and the convention below, are the missing half.
///
/// # The detach convention
///
/// A provider that owns a resource and permits it to be transferred exports one additional
/// `extern "C"` entry point per transferable resource:
///
/// ```text
/// stark_<resource>_detach(RawResourceHandle, *mut RawOsHandle) -> ProviderStatus
/// ```
///
/// It removes the entry from the owner's table and yields the underlying OS object, transferring
/// responsibility for closing it to the caller. After a successful detach the owner knows nothing
/// about the handle: a later close for it would abort, which is correct and is exactly what
/// CD-360's lowering guarantees never happens.
///
/// **Why a link-level convention rather than a manifest function.** A provider manifest describes
/// the STARK-callable surface — what `provider_api` may bind and what the compiler may lower a call
/// to. `detach` is callable by neither: no package binds it, and lowering never emits it. Declaring
/// it in the manifest would put a symbol into the surface the validator governs while leaving it
/// permanently unreachable from STARK, which misrepresents both. The declaration that IS visible to
/// the compiler is the consumer's `consumes: [{provider, resource}]`, which `provider_abi::validate`
/// and `ProviderSet::select` already check from both ends (CD-360).
///
/// **Why the linker rather than a Cargo dependency.** Every provider in a build is statically
/// linked into one generated binary, so an `extern "C"` declaration of the owner's symbol resolves
/// at link time with no dependency edge and no path assumption. A Cargo dependency would hardcode
/// the owner's location into the consumer's manifest — precisely the coupling CD-363 spent its
/// effort deleting.
///
/// # Recorded limitation
///
/// A missing detach symbol is a LINK error, named by symbol, not a compiler diagnostic. The
/// compiler knows a transfer was declared; it does not know the owner published the means. Closing
/// that would mean the manifest carrying a transfer surface distinct from the callable surface, and
/// that is a larger change than HC9 should make on its own. Written down rather than discovered.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RawOsHandle {
    /// The OS object, widened to 64 bits so one layout serves every platform: a `RawFd` (`c_int`)
    /// on Unix, a `SOCKET` (`UINT_PTR`) on Windows.
    ///
    /// Signed, deliberately. Unix reports failure as `-1`, and a type that cannot represent it
    /// would force every producer to invent a sentinel. Windows' `INVALID_SOCKET` is
    /// `(UINT_PTR)(-1)`, which round-trips through this as `-1` on 64-bit and is the same value
    /// again on the way back — so the two platforms' invalid handles agree rather than requiring
    /// per-platform comparison.
    pub value: i64,
    /// What `value` IS, so a consumer cannot mistake a file descriptor for a socket. Values are
    /// assigned by [`RawOsHandleKind`]; the field is a `u32` rather than an enum because an
    /// `enum` crossing the ABI with an unknown discriminant is undefined behaviour, and a
    /// provider on the other side of the boundary is not this crate's code.
    pub kind: u32,
}

/// The `kind` discriminants [`RawOsHandle`] can carry.
///
/// Not a Rust `enum`: see [`RawOsHandle::kind`]. A consumer must compare against these constants
/// and refuse anything it does not recognise, rather than transmuting.
pub struct RawOsHandleKind;

impl RawOsHandleKind {
    /// No handle. The value of `kind` when a detach fails; `value` is then meaningless.
    pub const NONE: u32 = 0;
    /// A connected stream socket: `RawFd` on Unix, `SOCKET` on Windows.
    pub const SOCKET: u32 = 1;
    /// A file descriptor or `HANDLE` that is not a socket. Reserved; nothing produces it yet.
    pub const FILE: u32 = 2;
}

impl RawOsHandle {
    /// The "no handle" value, for initialising an out-slot before a detach that may fail.
    pub const NONE: RawOsHandle = RawOsHandle {
        value: -1,
        kind: RawOsHandleKind::NONE,
    };

    /// A socket handle.
    pub fn socket(value: i64) -> RawOsHandle {
        RawOsHandle {
            value,
            kind: RawOsHandleKind::SOCKET,
        }
    }

    /// Whether this is a socket whose value could plausibly be one.
    ///
    /// `-1` is refused on every platform: it is Unix's error return and, widened, Windows'
    /// `INVALID_SOCKET`. A consumer that adopts it would call `close(-1)` at some later point and
    /// blame the wrong operation.
    pub fn is_valid_socket(self) -> bool {
        self.kind == RawOsHandleKind::SOCKET && self.value >= 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn physical_layouts_are_pinned() {
        assert_eq!(std::mem::size_of::<ProviderStatus>(), 4);
        assert_eq!(std::mem::align_of::<ProviderStatus>(), 4);
        assert_eq!(std::mem::size_of::<RawResourceHandle>(), 16);
        assert_eq!(std::mem::align_of::<RawResourceHandle>(), 8);
        assert_eq!(std::mem::size_of::<BorrowedBuffer>(), 16);
        assert_eq!(std::mem::align_of::<BorrowedBuffer>(), 8);
        assert_eq!(std::mem::size_of::<BorrowedBufferMut>(), 16);
        assert_eq!(std::mem::align_of::<BorrowedBufferMut>(), 8);
        assert_eq!(std::mem::size_of::<RawOsHandle>(), 16);
        assert_eq!(std::mem::align_of::<RawOsHandle>(), 8);
    }

    /// `-1` is Unix's error return AND, widened, Windows' `INVALID_SOCKET`. Both must be refused by
    /// the same comparison, or a consumer adopts an invalid socket and closes it much later.
    #[test]
    fn the_invalid_socket_value_is_refused_on_every_platform() {
        assert!(!RawOsHandle::NONE.is_valid_socket());
        assert!(!RawOsHandle::socket(-1).is_valid_socket());
        assert!(RawOsHandle::socket(0).is_valid_socket());
        assert!(RawOsHandle::socket(7).is_valid_socket());
        // Windows' INVALID_SOCKET is (UINT_PTR)(-1); on 64-bit that is u64::MAX, which is -1 here.
        assert_eq!(u64::MAX as i64, -1);
    }

    /// A non-socket kind is not a socket however plausible its value, and an unknown kind is not
    /// anything. A consumer that only checked `value >= 0` would adopt a file descriptor.
    #[test]
    fn a_kind_is_checked_before_a_value() {
        assert!(!RawOsHandle {
            value: 7,
            kind: RawOsHandleKind::FILE
        }
        .is_valid_socket());
        assert!(!RawOsHandle {
            value: 7,
            kind: 9999
        }
        .is_valid_socket());
    }
}
