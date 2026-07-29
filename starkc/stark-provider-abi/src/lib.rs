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
    }
}
