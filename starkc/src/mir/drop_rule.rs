//! **AS4 — the single authority for "does destroying a value of this type run anything?"**
//!
//! This file owns one semantic property. Before it, the rule existed twice — `lower::ty_needs_drop`
//! and `verify::mir_needs_drop` — claiming to answer the same question and disagreeing on 14
//! `MirTy::Core` variants (`AS4-DROP-RULE-MEASUREMENT.md` §6). Both are now delegates.
//!
//! # What is shared and what is supplied
//!
//! The caller cannot supply the parts that drift. The `MirTy` recursion, the `CoreType`
//! classification and `HostResource` live **here**; a caller supplies only the facts that genuinely
//! differ by phase, because lowering is a *producer* of the very table the verifier consumes:
//!
//! ```text
//! shared (this file)          per-phase (DropFacts)
//!   MirTy recursion             has a user destructor?
//!   CoreType classification     instantiated struct fields
//!   HostResource = true         instantiated enum variants
//!   container recursion
//! ```
//!
//! Passing the `CoreType` table as a callback would have moved the easiest-to-drift part into two
//! adapters and called it consolidation.
//!
//! # How the classification was decided
//!
//! Not by picking a winner. One rule, applied to the measured evidence:
//!
//! > **If lowering cannot construct the representation, preserve the verifier's answer.**
//!
//! - **Source-reachable cases → measured semantics decide.** `CharsIter` is `false` by owner ruling
//!   CD-387 (DEV-195): a borrowed `&str` cursor owning nothing that destruction could release.
//! - **Unreachable legacy shapes → the verifier's answer stands.** `Core(File)` is `true`. DEV-196
//!   established `mir_ty` refuses the type outright, so no source program produces it and the
//!   choice has no behavioural constituency to preserve. It is also where A11 is going: Core and
//!   package resources converge on `HostResource`, whose `Drop` must invoke its validated close
//!   exactly once. Legacy answer and future semantics agree.
//! - **`HostResource` → approved A11 semantics decide.** `true`.
//!
//! **This merge is behaviour-neutral, and that is checkable rather than asserted.** For every
//! `CoreType` `mir_ty` can actually construct — `Box`, `CharsIter`, `HashMap`, `HashSet`, `Iter`,
//! `KeysIter`, `Vec`, `VecIter` — lowering and the verifier already agreed after CD-387. Adopting
//! this table changes lowering's answer only on representations it cannot produce.

use super::{EnumRef, MirTy};
use crate::hir::{CoreType, ItemId};

/// The phase-dependent facts about a user nominal. Lowering answers them from HIR while it is still
/// building the MIR type table; the verifier answers them from the finished table.
pub(crate) trait DropFacts {
    /// Does an `impl Drop` govern this nominal at this instantiation?
    fn has_user_destructor(&self, item: ItemId, args: &[MirTy]) -> bool;
    /// The struct's field types at this instantiation, or `None` if unknown.
    fn struct_fields(&self, item: ItemId, args: &[MirTy]) -> Option<Vec<MirTy>>;
    /// The enum's per-variant payload types at this instantiation, or `None` if unknown.
    fn enum_variants(&self, item: ItemId, args: &[MirTy]) -> Option<Vec<Vec<MirTy>>>;
}

/// **Does destroying a value of this type run anything?**
///
/// Distinct from its two near neighbours, which answer different questions and must not be
/// substituted for it (AS4 work item 2):
///
/// | Question | Where | `HostResource` |
/// | --- | --- | --- |
/// | requires drop glue (precise) | **here** | true |
/// | MAY require drop glue (conservative) | `verify::may_need_drop` | true |
/// | has a USER-written destructor | `lower::ty_has_user_destructor_guarded` | **false** |
pub(crate) fn requires_drop_glue_with(ty: &MirTy, facts: &impl DropFacts) -> bool {
    match ty {
        MirTy::String => true,
        MirTy::Core(core, _) => core_requires_drop_glue(*core),
        // A11 §5: a host resource's drop IS its provider close.
        MirTy::HostResource(_) => true,
        MirTy::Struct(item, args) => {
            facts.has_user_destructor(*item, args)
                || facts
                    .struct_fields(*item, args)
                    .is_some_and(|fields| fields.iter().any(|f| requires_drop_glue_with(f, facts)))
        }
        MirTy::Enum(EnumRef::User(item), args) => {
            facts.has_user_destructor(*item, args)
                || facts.enum_variants(*item, args).is_some_and(|variants| {
                    variants
                        .iter()
                        .any(|v| v.iter().any(|f| requires_drop_glue_with(f, facts)))
                })
        }
        MirTy::Enum(_, args) => args.iter().any(|a| requires_drop_glue_with(a, facts)),
        MirTy::Tuple(elems) => elems.iter().any(|e| requires_drop_glue_with(e, facts)),
        MirTy::Array(elem, _) => requires_drop_glue_with(elem, facts),

        // **EXHAUSTIVE ON PURPOSE — no property-bearing wildcard.** A new `MirTy` variant must be
        // classified at this one authority, and every consumer inherits the decision. That is what
        // AS4 work item 3 means: a wildcard here would let a variant acquire a drop answer nobody
        // chose.
        //
        // Scalars own nothing; `Str`/`Slice` are unsized and reachable only behind a `Ref`, which
        // borrows rather than owns; a fn value is a bare pointer.
        MirTy::Int8
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
        | MirTy::Slice(_)
        | MirTy::Ref { .. }
        | MirTy::FnPtr { .. } => false,
    }
}

/// The `CoreType` classification, shared rather than supplied — see this module's header for why.
///
/// **EXHAUSTIVE ON PURPOSE.** A new `CoreType` must be classified here rather than inherit a
/// default, which is the property a producer census cannot give: a future producer then cannot
/// enlarge the reachable set without using a variant whose semantics were already chosen.
pub(crate) fn core_requires_drop_glue(core: CoreType) -> bool {
    match core {
        // CD-387 (DEV-195): a borrowed `&str` cursor yielding `Char` by value. The native runtime
        // wraps `std::str::Chars<'a>` and the backend emits it as intrinsically borrow-carrying, so
        // it owns nothing destruction could release.
        // **CD-388 (RB0 Q1): every constructible iterator is a borrowed cursor owning nothing.**
        //
        // `VecIter<'a,T>{slice,index}`, `KeysIter<'a,K>{keys,index}`, `Iter` (emits AS `KeysIter`)
        // and `CharsIter<'a>{inner: Chars<'a>}` are the same shape: no owned allocation, no
        // resource, no Rust `Drop`, and `DropPlan::Noop`. CD-387 ruled `CharsIter` requires no glue
        // on exactly that reasoning; `AS4-RB0-Q1-ITERATORS.md` measured the other three and found
        // no basis for the historical asymmetry.
        //
        // Effect: fewer drop units and flags in generated MIR. No observable change to any program
        // — the plan was `Noop` either way — which is why this needed its own CD rather than
        // riding along with the authority merge.
        CoreType::CharsIter | CoreType::VecIter | CoreType::KeysIter | CoreType::Iter => false,
        // **Legacy representation, deliberately `true`.** DEV-196: `mir_ty` refuses `Core(File)`
        // outright, so no source program produces it and this preserves the verifier's answer with
        // no behavioural constituency to disturb. A11 sends `File` to `HostResource`, whose `Drop`
        // invokes its validated close exactly once — so the legacy answer and the approved
        // destination agree. The hand-built WP-C7.8.4 provider path is unaffected: it closes with
        // `stark_file_close` (`HandleConsumed`) and does not acquire a `Drop` because a
        // type-property function says the type requires destruction.
        CoreType::File => true,
        CoreType::String
        | CoreType::Vec
        | CoreType::Box
        | CoreType::Option
        | CoreType::Result
        | CoreType::Range
        | CoreType::RangeInclusive
        | CoreType::SplitIter
        | CoreType::HashMap
        | CoreType::HashSet
        | CoreType::ValuesIter
        | CoreType::MapIter
        | CoreType::FilterIter
        | CoreType::Random
        | CoreType::IOError
        | CoreType::Ordering => true,
    }
}
