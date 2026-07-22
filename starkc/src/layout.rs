//! WP-C5.3e — the target-layout contract (CD-058, CD-067).
//!
//! `07-Modules-and-Packages.md` LAYOUT-QUERY-001 says `size_of<T>`/`align_of<T>` return "positive
//! **target-contract** values", and LAYOUT-ABI-001 says those values "may differ between named
//! targets and compiler versions". A layout query therefore answers from a *declared contract*,
//! not from a measurement of whatever representation a host compiler happened to choose. Offsets,
//! niches, discriminant representation, pointer values and physical addresses are all explicitly
//! unobservable, so nothing a STARK program can do depends on the contract matching the host.
//!
//! Before this module the three engines each answered differently and only a relations-only
//! placeholder test hid it: the HIR oracle returned a hardcoded `8` without even reading the
//! queried type, the MIR interpreter's `reference_layout` returned `(8, 8)` for everything, and
//! the native backend emitted `core::mem::size_of::<RustTy>()` — the host's `repr(Rust)`
//! representation. On the spec's reading the *native* engine was the non-conforming one.
//!
//! # The contract is not the backend's representation (CD-067)
//!
//! The values here are a **declared** property of a named target. They are deliberately NOT
//! required to equal the physical layout any particular backend chooses, and the generated crate
//! asserts nothing about its own. Generated nominals stay `repr(Rust)` and remain free to reorder
//! fields and use niches — a STARK program cannot observe either, and binding the contract to
//! generated Rust would make one transitional backend part of the language definition, obstructing
//! a later backend that implements the same contract over a different representation.
//!
//! LAYOUT-ABI-001 is explicit that equal `size_of`/`align_of` does not establish interoperation
//! compatibility, so the three contracts stay separate: this observable language contract, a
//! backend's private representation, and the separately versioned provider ABI.
//!
//! # The algorithm
//!
//! [`TargetLayout::aggregate`] is a declaration-order rule with natural alignment and tail
//! padding; [`TargetLayout::sum`] is a discriminant followed by the widest variant. They are
//! chosen for being deterministic, order-preserving and simple to state — properties that make the
//! contract *testable on its own terms*, which is where falsifiability actually comes from. Their
//! resemblance to the C rules is a consequence of those properties, not a commitment to them, and
//! nothing in the compiler may treat it as one.
//!
//! What makes this falsifiable: a versioned identity ([`TargetLayout::identity`]), one combinator
//! implementation, exact frozen values in the C5 layout matrix, and mutation tests that change a
//! primitive entry or an aggregate rule and break three-engine agreement.
//!
//! # One algorithm, two type walkers
//!
//! The combinators here are the single derivation of the *algorithm*. Each engine still walks its
//! own type representation into them, because those representations genuinely differ (the HIR
//! oracle has checker `Ty`, MIR and the backend have `MirTy`) and no single walk could serve both.
//! That is the same producer/consumer split as `TypeContext::is_copy` (CD-065), and it gets the
//! same treatment: an empirical agreement check rather than a pretended shared walk.

use crate::mir::{MirTy, TypeContext};

/// A type's contract size and alignment, in bytes.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Layout {
    pub size: u64,
    pub align: u64,
}

impl Layout {
    pub const fn new(size: u64, align: u64) -> Self {
        Self { size, align }
    }
}

/// Refused: the contract does not describe this type. Every engine turns this into its own kind
/// of failure rather than inventing a number, so a type outside the contract cannot be answered
/// differently by two engines.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LayoutError(pub String);

impl std::fmt::Display for LayoutError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// The primitive types the contract fixes directly. Kept separate from `MirTy` so the HIR oracle,
/// which has no `MirTy`, can reach the same table.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Scalar {
    Int8,
    Int16,
    Int32,
    Int64,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
    Float32,
    Float64,
    Bool,
    Char,
    Unit,
    /// `&T` / `&mut T`. Core v1 has no unsized references outside `str`/slices, so this is the
    /// thin-pointer case only.
    Reference,
    /// A function value. Core v1 has no closures, so this is a bare code pointer.
    FnValue,
}

/// The identity a layout answer is accountable to (CD-067). Carried into the build key and the
/// build report, and checked against the requested target before any engine answers a query, so a
/// value can always be attributed to a specific contract at a specific revision.
///
/// Three parts, because they change for different reasons:
/// - `target_contract` names the target whose values these are (LAYOUT-ABI-001's "named targets");
/// - `layout_contract_version` is the CONTRACT's version — bumped when a declared value or rule
///   changes, which is an observable change to programs;
/// - `compiler_layout_revision` is the implementation's revision — bumped when this compiler's
///   realisation of the same contract changes without changing any declared value.
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct LayoutIdentity {
    pub target_contract: String,
    pub layout_contract_version: u32,
    pub compiler_layout_revision: u32,
}

impl std::fmt::Display for LayoutIdentity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} (layout_contract_version {}, compiler_layout_revision {})",
            self.target_contract, self.layout_contract_version, self.compiler_layout_revision
        )
    }
}

/// A named target's layout contract (LAYOUT-ABI-001's "named targets").
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TargetLayout {
    /// Which contract, at which version, these values belong to.
    pub identity: LayoutIdentity,
    pub int8: Layout,
    pub int16: Layout,
    pub int32: Layout,
    pub int64: Layout,
    pub uint8: Layout,
    pub uint16: Layout,
    pub uint32: Layout,
    pub uint64: Layout,
    pub float32: Layout,
    pub float64: Layout,
    pub bool_: Layout,
    pub char_: Layout,
    pub unit: Layout,
    pub reference: Layout,
    /// A function value (`MirTy::FnPtr`). Core v1 has no closures, so this is a bare code pointer.
    pub fn_value: Layout,
    /// The discriminant's own layout in a tagged sum. Declared, not inherited from any backend.
    pub enum_tag: Layout,
}

impl TargetLayout {
    /// `stark-64-v1` — the only named target contract Core v1 declares.
    pub fn stark64_v1() -> Self {
        Self {
            identity: LayoutIdentity {
                target_contract: "stark-64-v1".to_string(),
                layout_contract_version: 1,
                compiler_layout_revision: 1,
            },
            int8: Layout::new(1, 1),
            int16: Layout::new(2, 2),
            int32: Layout::new(4, 4),
            int64: Layout::new(8, 8),
            uint8: Layout::new(1, 1),
            uint16: Layout::new(2, 2),
            uint32: Layout::new(4, 4),
            uint64: Layout::new(8, 8),
            float32: Layout::new(4, 4),
            float64: Layout::new(8, 8),
            bool_: Layout::new(1, 1),
            // `Char` is a Unicode scalar value; the contract declares four bytes.
            char_: Layout::new(4, 4),
            // A zero-sized type: LAYOUT-QUERY-001 explicitly permits a target to report size zero.
            unit: Layout::new(0, 1),
            reference: Layout::new(8, 8),
            fn_value: Layout::new(8, 8),
            enum_tag: Layout::new(4, 4),
        }
    }

    pub fn scalar(&self, scalar: Scalar) -> Layout {
        match scalar {
            Scalar::Int8 => self.int8,
            Scalar::Int16 => self.int16,
            Scalar::Int32 => self.int32,
            Scalar::Int64 => self.int64,
            Scalar::UInt8 => self.uint8,
            Scalar::UInt16 => self.uint16,
            Scalar::UInt32 => self.uint32,
            Scalar::UInt64 => self.uint64,
            Scalar::Float32 => self.float32,
            Scalar::Float64 => self.float64,
            Scalar::Bool => self.bool_,
            Scalar::Char => self.char_,
            Scalar::Unit => self.unit,
            Scalar::Reference => self.reference,
            Scalar::FnValue => self.fn_value,
        }
    }

    /// The declared aggregate rule: fields are placed in DECLARATION order, each at the next
    /// offset satisfying its own alignment; the whole takes the strictest field alignment; and the
    /// size is rounded up to that alignment so arrays of the type stay aligned.
    ///
    /// An empty aggregate is `(0, 1)`. Nothing here is derived from a backend: this is the rule
    /// the contract declares, and a backend's physical layout may reorder or pack freely because
    /// no STARK program can observe that it did (CD-067).
    pub fn aggregate(&self, fields: impl IntoIterator<Item = Layout>) -> Layout {
        let mut offset: u64 = 0;
        let mut align: u64 = 1;
        for field in fields {
            offset = round_up(offset, field.align);
            offset += field.size;
            align = align.max(field.align);
        }
        Layout::new(round_up(offset, align), align)
    }

    /// An array's stride is its element's contract SIZE, which [`TargetLayout::aggregate`] has
    /// already rounded up to the element's alignment — so no separate padding term is needed.
    pub fn array(&self, elem: Layout, len: u64) -> Layout {
        Layout::new(elem.size * len, elem.align)
    }

    /// The declared sum rule: a discriminant followed by the widest variant, where each variant
    /// is laid out by the aggregate rule.
    ///
    /// A fieldless enum reduces to the tag alone, so `Ordering` is the tag's `(4, 4)`. The
    /// contract always declares a real discriminant; a backend is still free to niche-optimise its
    /// own representation, which is unobservable.
    pub fn sum(&self, variants: impl IntoIterator<Item = Layout>) -> Layout {
        let mut payload = Layout::new(0, 1);
        for variant in variants {
            payload.size = payload.size.max(variant.size);
            payload.align = payload.align.max(variant.align);
        }
        payload.size = round_up(payload.size, payload.align);
        self.aggregate([self.enum_tag, payload])
    }

    /// The `MirTy` walker — used by the MIR interpreter and by the native backend, which share a
    /// type language. The HIR oracle has its own walker over checker types.
    pub fn layout_of(&self, ty: &MirTy, types: &TypeContext) -> Result<Layout, LayoutError> {
        Ok(match ty {
            MirTy::Int8 => self.int8,
            MirTy::Int16 => self.int16,
            MirTy::Int32 => self.int32,
            MirTy::Int64 => self.int64,
            MirTy::UInt8 => self.uint8,
            MirTy::UInt16 => self.uint16,
            MirTy::UInt32 => self.uint32,
            MirTy::UInt64 => self.uint64,
            MirTy::Float32 => self.float32,
            MirTy::Float64 => self.float64,
            MirTy::Bool => self.bool_,
            MirTy::Char => self.char_,
            MirTy::Unit => self.unit,
            MirTy::Ref { .. } => self.reference,
            MirTy::FnPtr { .. } => self.fn_value,
            MirTy::Tuple(elems) => {
                let mut fields = Vec::with_capacity(elems.len());
                for elem in elems {
                    fields.push(self.layout_of(elem, types)?);
                }
                self.aggregate(fields)
            }
            MirTy::Array(elem, len) => {
                let elem = self.layout_of(elem, types)?;
                self.array(elem, *len)
            }
            MirTy::Struct(item, args) => {
                let field_tys = types
                    .struct_fields
                    .get(&(item.0, args.clone()))
                    .cloned()
                    .ok_or_else(|| LayoutError(format!("no field table for struct #{}", item.0)))?;
                let mut fields = Vec::with_capacity(field_tys.len());
                for field in &field_tys {
                    fields.push(self.layout_of(field, types)?);
                }
                self.aggregate(fields)
            }
            MirTy::Enum(enum_ref, args) => {
                let table = crate::mir::drop_plan::variant_payloads(enum_ref, args, types)
                    .ok_or_else(|| LayoutError(format!("no variant table for {ty:?}")))?;
                let mut variants = Vec::with_capacity(table.len());
                for payload in &table {
                    let mut fields = Vec::with_capacity(payload.len());
                    for field in payload {
                        fields.push(self.layout_of(field, types)?);
                    }
                    variants.push(self.aggregate(fields));
                }
                self.sum(variants)
            }
            // Refused rather than guessed. `String`, `Vec`, `HashMap`, the iterators and `Never`
            // have no contract entry: they are owning runtime types whose representation the
            // backend does not admit either, so answering here would create a divergence between
            // an interpreter that answered and a backend that could not build.
            other => {
                return Err(LayoutError(format!(
                    "the {} layout contract does not describe {other:?}: owning runtime types and \
                     unsized types have no contract entry, and inventing one would let an \
                     interpreter answer a query the backend cannot compile",
                    self.identity.target_contract
                )))
            }
        })
    }
}

impl Default for TargetLayout {
    fn default() -> Self {
        Self::stark64_v1()
    }
}

/// Select a contract by target name. CD-067 requires a build to REJECT a requested target the
/// compiler has no contract for, rather than silently answering from a default.
pub fn contract_for(target: &str) -> Result<TargetLayout, LayoutError> {
    match target {
        "stark-64-v1" => Ok(TargetLayout::stark64_v1()),
        other => Err(LayoutError(format!(
            "no layout contract named `{other}`; this compiler declares `stark-64-v1`. A layout \
             query cannot be answered from a default contract, because the answer is observable \
             and target-specific (LAYOUT-ABI-001)"
        ))),
    }
}

fn round_up(value: u64, align: u64) -> u64 {
    debug_assert!(align > 0, "alignment is always at least 1");
    let rem = value % align;
    if rem == 0 {
        value
    } else {
        value + (align - rem)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hir::ItemId;
    use crate::mir::EnumRef;

    fn t() -> TargetLayout {
        TargetLayout::stark64_v1()
    }

    #[test]
    fn primitives_take_their_contract_values() {
        let t = t();
        let types = TypeContext::default();
        assert_eq!(
            t.layout_of(&MirTy::Int32, &types).unwrap(),
            Layout::new(4, 4)
        );
        assert_eq!(
            t.layout_of(&MirTy::Int64, &types).unwrap(),
            Layout::new(8, 8)
        );
        assert_eq!(
            t.layout_of(&MirTy::Bool, &types).unwrap(),
            Layout::new(1, 1)
        );
        assert_eq!(
            t.layout_of(&MirTy::Char, &types).unwrap(),
            Layout::new(4, 4)
        );
        assert_eq!(
            t.layout_of(&MirTy::Unit, &types).unwrap(),
            Layout::new(0, 1)
        );
    }

    /// The padding rule, which is the whole reason the algorithm is stated rather than assumed:
    /// `(Int8, Int32)` is 8 bytes, not 5 — the `Int32` starts at offset 4 and the total rounds up
    /// to the strictest alignment.
    #[test]
    fn aggregates_pad_between_fields_and_at_the_end() {
        let t = t();
        let types = TypeContext::default();
        let ty = MirTy::Tuple(vec![MirTy::Int8, MirTy::Int32]);
        assert_eq!(t.layout_of(&ty, &types).unwrap(), Layout::new(8, 4));

        // Trailing padding alone: `(Int32, Int8)` is 5 bytes of content in 8.
        let ty = MirTy::Tuple(vec![MirTy::Int32, MirTy::Int8]);
        assert_eq!(t.layout_of(&ty, &types).unwrap(), Layout::new(8, 4));

        // Declaration order matters under the C rule -- this is exactly what `repr(Rust)` is
        // free to optimise away and `repr(C)` is not.
        let ty = MirTy::Tuple(vec![MirTy::Int8, MirTy::Int8, MirTy::Int32]);
        assert_eq!(t.layout_of(&ty, &types).unwrap(), Layout::new(8, 4));
    }

    #[test]
    fn an_empty_aggregate_is_zero_sized_with_alignment_one() {
        let t = t();
        let types = TypeContext::default();
        assert_eq!(
            t.layout_of(&MirTy::Tuple(vec![]), &types).unwrap(),
            Layout::new(0, 1)
        );
    }

    #[test]
    fn an_array_strides_by_the_elements_contract_size() {
        let t = t();
        let types = TypeContext::default();
        let ty = MirTy::Array(Box::new(MirTy::Int32), 4);
        assert_eq!(t.layout_of(&ty, &types).unwrap(), Layout::new(16, 4));
        // A padded element strides by its PADDED size, not its content size.
        let ty = MirTy::Array(Box::new(MirTy::Tuple(vec![MirTy::Int32, MirTy::Int8])), 3);
        assert_eq!(t.layout_of(&ty, &types).unwrap(), Layout::new(24, 4));
    }

    #[test]
    fn a_struct_reads_its_field_table() {
        let t = t();
        let mut types = TypeContext::default();
        types
            .struct_fields
            .insert((1, vec![]), vec![MirTy::Bool, MirTy::Int64]);
        let ty = MirTy::Struct(ItemId(1), vec![]);
        assert_eq!(t.layout_of(&ty, &types).unwrap(), Layout::new(16, 8));
    }

    /// A fieldless enum is the tag alone — so `Ordering` is `(4, 4)`.
    #[test]
    fn a_fieldless_enum_is_its_tag() {
        let t = t();
        let mut types = TypeContext::default();
        types
            .enum_variants
            .insert((1, vec![]), vec![vec![], vec![], vec![]]);
        let ty = MirTy::Enum(EnumRef::User(ItemId(1)), vec![]);
        assert_eq!(t.layout_of(&ty, &types).unwrap(), Layout::new(4, 4));
        assert_eq!(
            t.layout_of(&MirTy::Enum(EnumRef::CoreOrdering, vec![]), &types)
                .unwrap(),
            Layout::new(4, 4)
        );
    }

    /// A payload-carrying enum is tag + union-of-variants, and the union takes the LARGEST
    /// variant and the STRICTEST alignment, not the first or the last.
    #[test]
    fn a_payload_enum_is_a_tag_plus_the_widest_variant() {
        let t = t();
        let mut types = TypeContext::default();
        types.enum_variants.insert(
            (1, vec![]),
            vec![vec![], vec![MirTy::Int8], vec![MirTy::Int64]],
        );
        let ty = MirTy::Enum(EnumRef::User(ItemId(1)), vec![]);
        // tag (4,4) then an 8-aligned payload: the tag's 4 bytes are followed by 4 of padding.
        assert_eq!(t.layout_of(&ty, &types).unwrap(), Layout::new(16, 8));
    }

    /// `Option<T>` under the contract carries a real discriminant. A backend is free to
    /// niche-optimise its own representation to 4 bytes; that difference is unobservable and
    /// deliberately NOT asserted anywhere (CD-067).
    #[test]
    fn core_option_carries_a_real_tag_rather_than_a_niche() {
        let t = t();
        let types = TypeContext::default();
        let ty = MirTy::Enum(EnumRef::CoreOption, vec![MirTy::Int32]);
        assert_eq!(t.layout_of(&ty, &types).unwrap(), Layout::new(8, 4));
    }

    #[test]
    fn owning_runtime_types_are_refused_rather_than_guessed() {
        let t = t();
        let types = TypeContext::default();
        assert!(t.layout_of(&MirTy::String, &types).is_err());
        assert!(t
            .layout_of(
                &MirTy::Core(crate::hir::CoreType::Vec, vec![MirTy::Int32]),
                &types
            )
            .is_err());
    }

    // ---- CD-067's mutation set: the contract is falsifiable on its OWN terms ----
    //
    // The rejected alternative was to assert the contract equals Rust's physical layout, which
    // would have tested agreement with one backend rather than the contract's coherence. These
    // change a manifest entry or a combinator rule and show the answer moves — which is what the
    // frozen three-engine layout matrix then detects. Each mutation is one an editor could
    // plausibly make by accident.

    /// Mutation: change a primitive's declared size. Every aggregate containing it must move.
    #[test]
    fn mutating_a_primitive_entry_changes_every_aggregate_containing_it() {
        let types = TypeContext::default();
        let mut t = t();
        let before = t
            .layout_of(&MirTy::Tuple(vec![MirTy::Int32, MirTy::Int32]), &types)
            .unwrap();
        t.int32 = Layout::new(8, 8);
        let after = t
            .layout_of(&MirTy::Tuple(vec![MirTy::Int32, MirTy::Int32]), &types)
            .unwrap();
        assert_ne!(before, after, "a primitive entry must be load-bearing");
        assert_eq!(after, Layout::new(16, 8));
    }

    /// Mutation: drop the inter-field padding rule (place fields at the running offset without
    /// re-aligning).
    ///
    /// The case is chosen deliberately. `(Int8, Int64)` does NOT distinguish the two rules — both
    /// give 16, because the trailing round-up hides the missing gap. `(Int8, Int32, Int8)` does:
    /// 12 correctly, 8 under the mutant. A mutation test that cannot fail is worse than none, and
    /// the first version of this test picked the case that could not.
    #[test]
    fn dropping_the_field_alignment_rule_changes_the_answer() {
        let t = t();
        let fields = [t.int8, t.int32, t.int8];
        let correct = t.aggregate(fields);
        // The mutant: accumulate sizes with no per-field alignment.
        let mut offset = 0;
        let mut align = 1u64;
        for f in fields {
            offset += f.size;
            align = align.max(f.align);
        }
        let mutant = Layout::new(round_up(offset, align), align);
        assert_eq!(correct, Layout::new(12, 4));
        assert_eq!(mutant, Layout::new(8, 4));
        assert_ne!(
            correct, mutant,
            "per-field alignment must change the answer, or the rule is untested"
        );
    }

    /// Mutation: drop the TRAILING padding rule. `(Int32, Int8)` would be 5, which would then
    /// make an array of it stride wrongly — the reason the rule exists.
    #[test]
    fn dropping_the_trailing_padding_rule_breaks_array_striding() {
        let t = t();
        let elem = t.aggregate([t.int32, t.int8]);
        assert_eq!(elem, Layout::new(8, 4));
        assert_eq!(t.array(elem, 3), Layout::new(24, 4));
        // The mutant: no final round-up.
        let mutant_elem = Layout::new(5, 4);
        assert_ne!(
            t.array(elem, 3),
            t.array(mutant_elem, 3),
            "trailing padding must be observable through array stride"
        );
    }

    /// Mutation: make the sum rule take the FIRST variant instead of the widest. The declared
    /// rule must be what decides, not variant order.
    #[test]
    fn the_sum_rule_takes_the_widest_variant_not_the_first() {
        let t = t();
        let narrow = t.aggregate([t.int8]);
        let wide = t.aggregate([t.int64]);
        let correct = t.sum([narrow, wide]);
        let mutant = t.aggregate([t.enum_tag, narrow]);
        assert_eq!(correct, Layout::new(16, 8));
        assert_ne!(correct, mutant, "variant width must decide the sum's size");
    }

    /// Mutation: change the declared tag. Every enum in the matrix moves, including the fieldless
    /// ones, which is what makes `Ordering`'s frozen `(4, 4)` a real assertion.
    #[test]
    fn mutating_the_enum_tag_changes_every_enum() {
        let mut types = TypeContext::default();
        types
            .enum_variants
            .insert((1, vec![]), vec![vec![], vec![], vec![]]);
        let ty = MirTy::Enum(EnumRef::User(ItemId(1)), vec![]);
        let mut t = t();
        assert_eq!(t.layout_of(&ty, &types).unwrap(), Layout::new(4, 4));
        t.enum_tag = Layout::new(8, 8);
        assert_eq!(t.layout_of(&ty, &types).unwrap(), Layout::new(8, 8));
    }

    /// Field ORDER is part of the contract. Two aggregates with the same fields in different
    /// orders may differ, which is exactly the freedom `repr(Rust)` has and the contract does not.
    #[test]
    fn declaration_order_is_part_of_the_contract() {
        let t = t();
        let types = TypeContext::default();
        let a = t
            .layout_of(
                &MirTy::Tuple(vec![MirTy::Int8, MirTy::Int64, MirTy::Int8]),
                &types,
            )
            .unwrap();
        let b = t
            .layout_of(
                &MirTy::Tuple(vec![MirTy::Int8, MirTy::Int8, MirTy::Int64]),
                &types,
            )
            .unwrap();
        assert_eq!(a, Layout::new(24, 8));
        assert_eq!(b, Layout::new(16, 8));
        assert_ne!(a, b, "reordering fields must be able to change the answer");
    }

    /// An unknown target name is REJECTED, not defaulted: a layout answer is observable and
    /// target-specific, so silently answering from `stark-64-v1` would report values for a target
    /// nobody asked about.
    #[test]
    fn an_unknown_target_contract_is_rejected_rather_than_defaulted() {
        assert!(contract_for("stark-64-v1").is_ok());
        let err = contract_for("stark-32-v1").unwrap_err();
        assert!(err.0.contains("no layout contract named"), "{}", err.0);
    }

    #[test]
    fn the_contract_identity_names_its_target_and_both_versions() {
        let t = t();
        assert_eq!(t.identity.target_contract, "stark-64-v1");
        assert_eq!(t.identity.layout_contract_version, 1);
        assert_eq!(t.identity.compiler_layout_revision, 1);
        assert!(t.identity.to_string().contains("stark-64-v1"));
    }

    /// Every contract value must be usable: a zero alignment would make `round_up` meaningless
    /// and a zero size for a non-`Unit` type would break array striding.
    #[test]
    fn the_contract_is_internally_well_formed() {
        let t = t();
        for scalar in [
            Scalar::Int8,
            Scalar::Int16,
            Scalar::Int32,
            Scalar::Int64,
            Scalar::UInt8,
            Scalar::UInt16,
            Scalar::UInt32,
            Scalar::UInt64,
            Scalar::Float32,
            Scalar::Float64,
            Scalar::Bool,
            Scalar::Char,
            Scalar::Unit,
            Scalar::Reference,
            Scalar::FnValue,
        ] {
            let l = t.scalar(scalar);
            assert!(l.align > 0, "{scalar:?} has zero alignment");
            assert!(
                l.align.is_power_of_two(),
                "{scalar:?} alignment must be a power of two"
            );
            assert!(
                l.size % l.align == 0,
                "{scalar:?} size must be a multiple of its alignment"
            );
            // LAYOUT-QUERY-001 allows size zero only for a zero-sized type.
            assert!(
                l.size > 0 || scalar == Scalar::Unit,
                "{scalar:?} is zero-sized"
            );
        }
        assert!(t.enum_tag.align > 0);
    }
}
