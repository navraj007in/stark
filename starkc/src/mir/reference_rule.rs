//! **AS4 — the single authority for "does this type STORE a reference?"**
//!
//! RB0 recorded three implementations of "carries a reference" and AS0 measured them across 33
//! samples spanning every `MirTy` variant. The three-way summary said *"they agree except on
//! `FnPtr`"*, which hid the pairwise fact that decides the consolidation:
//!
//! ```text
//! lower::ty_carries_ref  ==  emit::ty_contains_ref     on EVERY sample, FnPtr included
//! emit::ty_carries_reference differs, and ONLY on FnPtr
//! ```
//!
//! So this was never three implementations of one rule. It was **two implementations of one rule**
//! (merged here) plus **one near neighbour asking a different question**
//! ([`super::emit_types::mentions_a_reference`], renamed to say so).
//!
//! # The two questions, and why `FnPtr` separates them
//!
//! | | stores a reference | mentions a reference |
//! | --- | --- | --- |
//! | `&T` | yes | yes |
//! | `(String, &str)` | yes | yes |
//! | `fn(&T) -> Int32` | **no** — the value is a code address | **yes** — the signature names one |
//!
//! A Rust `fn(&T)` is **higher-ranked** (`for<'a> fn(&'a T)`), so it needs no lifetime parameter;
//! that is why the storage question answers `no` and why the lowering guard it feeds (E0106,
//! "would emitting this need a lifetime the backend cannot generate") is right to.
//!
//! The disagreement is **live, not vestigial**: `let g: fn(&Int32) -> Int32 = takes;` is
//! constructible, lowers, and runs in all three engines — unlike `Core(File)`, which DEV-196 showed
//! unreachable. It is kept because the two answers are both correct for their own consumer, which
//! `tests/as4_reference_rule.rs` establishes end to end.

use super::MirTy;

/// **Does a value of this type STORE a reference?**
///
/// Distinct from [`super::emit_types::mentions_a_reference`], which also counts references named in
/// a function pointer's signature. Substituting one for the other is exactly what near-neighbour
/// naming exists to prevent (AS4 work item 2).
///
/// Consumers: MIR lowering's Display guard for droppable borrow-carrying composites, and the
/// backend's `derives_for`.
pub(crate) fn stores_a_reference(ty: &MirTy) -> bool {
    match ty {
        MirTy::Ref { .. } => true,
        MirTy::Struct(_, args)
        | MirTy::Enum(_, args)
        | MirTy::Tuple(args)
        | MirTy::Core(_, args) => args.iter().any(stores_a_reference),
        MirTy::Array(elem, _) | MirTy::Slice(elem) => stores_a_reference(elem),

        // **A function pointer stores a code address, not a reference.** This is the one arm that
        // differs from `mentions_a_reference`, and the difference is the whole reason both exist.
        //
        // **EXHAUSTIVE ON PURPOSE — no property-bearing wildcard.** This ASSERTS a property rather
        // than declining to optimise one, so a wildcard would make every future variant claim the
        // property is absent — the shape that classified `HostResource` as `Copy` and leaked every
        // resource with the suite green.
        MirTy::FnPtr { .. }
        | MirTy::Int8
        | MirTy::Int16
        | MirTy::Int32
        | MirTy::Int64
        | MirTy::UInt8
        | MirTy::UInt16
        | MirTy::UInt32
        | MirTy::UInt64
        | MirTy::Float32
        | MirTy::Float64
        | MirTy::Bool
        | MirTy::Char
        | MirTy::Unit
        | MirTy::Never
        | MirTy::Str
        | MirTy::String
        | MirTy::HostResource(_) => false,
    }
}
