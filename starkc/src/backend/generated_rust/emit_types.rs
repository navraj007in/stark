//! WP-C5.2a — §6.2 type mapping and constant emission ("primitive values and constants").
//! `WP-C5.1.md`'s `MirTy` matrix records the full IN/OUT split -- aggregates, enums, references,
//! and every `Core(CoreType, _)` payload return `Unsupported` from [`emit_ty`] until WP-C5.2b/c
//! (aggregates/control flow) or WP-C5.3 (enums) implement their §6.2 rows. `String`/`str` are
//! deferred alongside output support (WP-C5.2c/d, wherever `RuntimeFn` calls first lower), since
//! a constant string with nothing to do with it yet is not independently useful to prove.

use super::{mangle, BackendDiagnostic};
use crate::mir::{Constant, MirProgram, MirTy, TypeContext};

/// The generated field name for a struct field at declaration index `i`. Positional rather than
/// the STARK source name: MIR carries fields by index (`Projection::Field(u32)`) and has already
/// discarded the names, so inventing them back would mean re-deriving information the backend is
/// explicitly not supposed to reconstruct (§5.2, "no backend semantic reconstruction").
pub fn field_name(i: u32) -> String {
    format!("f{i}")
}

/// The variant payload table for ANY enum, core or user — the single source the generated
/// definition, the discriminant match, and every payload projection all read.
///
/// WP-C5.3d-1b: re-exported from `mir::drop_plan`, which now owns the one derivation the drop
/// plan, the interpreter's place typing, and this module all read: `Option` is `None = 0` /
/// `Some(T) = 1`; `Result` is `Ok(T) = 0` / `Err(E) = 1`; `Ordering` is three fieldless variants
/// `Less`/`Equal`/`Greater`. Sharing it is what stops the consumers from disagreeing about a
/// variant index.
pub use crate::mir::drop_plan::variant_payloads;

/// The generated Rust type name for any nominal instance, core or user. Core enums get a
/// distinct key space (`option`/`result`/`ordering`) so they cannot collide with a user item id.
pub fn nominal_type_name(ty: &MirTy) -> Option<String> {
    use crate::mir::EnumRef;
    match ty {
        MirTy::Struct(item, args) => Some(mangle::type_name_for_nominal(item.0, args)),
        MirTy::Enum(EnumRef::User(item), args) => Some(mangle::type_name_for_nominal(item.0, args)),
        MirTy::Enum(core, args) => {
            let tag = match core {
                EnumRef::CoreOption => "option",
                EnumRef::CoreResult => "result",
                EnumRef::CoreOrdering => "ordering",
                EnumRef::User(_) => unreachable!("handled above"),
            };
            Some(mangle::type_name_for_core_enum(tag, args))
        }
        _ => None,
    }
}

/// The generated variant name for the enum variant at declaration index `v`. Positional for the
/// same reason [`field_name`] is: MIR carries variants by index (`AggKind::EnumVariant`,
/// `Projection::VariantField`, and a `SwitchInt` on a discriminant), having discarded the source
/// names.
pub fn variant_name(v: u32) -> String {
    format!("V{v}")
}

/// WP-C6.1f "borrow-carrying nominals": how lifetimes are spelled in a given syntactic position.
///
/// A generated nominal holding a reference needs a lifetime parameter, and Rust spells it two ways.
/// Inside the type's own DECLARATION every lifetime must be the named parameter `'a` — `'_` is not
/// allowed in a field type, which has no enclosing binder to infer from. Everywhere else (locals,
/// slots, signatures, helper arguments) `'_` lets rustc infer it, which is what keeps the change
/// from spreading into every use site's own lifetime bookkeeping.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum LifetimePosition {
    /// A use site: `Name<'_>`, `&T`. rustc infers.
    Use,
    /// Inside the generated declaration of a borrow-carrying nominal: `Name<'a>`, `&'a T`.
    Declaration,
}

impl LifetimePosition {
    /// The lifetime argument list for a borrow-carrying nominal in this position.
    fn nominal_args(self) -> &'static str {
        match self {
            LifetimePosition::Use => "<'_>",
            LifetimePosition::Declaration => "<'a>",
        }
    }
    /// The lifetime prefix for a reference in this position (`&` vs `&'a `).
    fn ref_prefix(self, mutable: bool) -> &'static str {
        match (self, mutable) {
            (LifetimePosition::Use, false) => "&",
            (LifetimePosition::Use, true) => "&mut ",
            (LifetimePosition::Declaration, false) => "&'a ",
            (LifetimePosition::Declaration, true) => "&'a mut ",
        }
    }
}

pub fn emit_ty(ty: &MirTy) -> Result<String, BackendDiagnostic> {
    emit_ty_at(ty, LifetimePosition::Use)
}

/// Whether a nominal instance needs a lifetime parameter: exactly when its type arguments carry a
/// borrow. Its generated definition then reads `Name<'a>` and every use `Name<'_>`.
/// Which position a nominal's own field/payload types are rendered in: `Declaration` (so their
/// references name the type's `'a`) only when the type actually declares one. A nominal WITHOUT a
/// lifetime parameter has no `'a` in scope, so its fields must stay in `Use` position.
fn decl_position(ty: &MirTy) -> LifetimePosition {
    if nominal_needs_lifetime(ty) {
        LifetimePosition::Declaration
    } else {
        LifetimePosition::Use
    }
}

pub fn nominal_needs_lifetime(ty: &MirTy) -> bool {
    match ty {
        MirTy::Struct(_, args) | MirTy::Enum(_, args) | MirTy::Core(_, args) => {
            args.iter().any(ty_carries_reference)
        }
        _ => false,
    }
}

pub fn emit_ty_at(ty: &MirTy, at: LifetimePosition) -> Result<String, BackendDiagnostic> {
    let lt = if nominal_needs_lifetime(ty) {
        at.nominal_args()
    } else {
        ""
    };
    Ok(match ty {
        MirTy::Int8 => "i8".to_string(),
        MirTy::Int16 => "i16".to_string(),
        MirTy::Int32 => "i32".to_string(),
        MirTy::Int64 => "i64".to_string(),
        MirTy::UInt8 => "u8".to_string(),
        MirTy::UInt16 => "u16".to_string(),
        MirTy::UInt32 => "u32".to_string(),
        MirTy::UInt64 => "u64".to_string(),
        MirTy::Float32 => "f32".to_string(),
        MirTy::Float64 => "f64".to_string(),
        MirTy::Bool => "bool".to_string(),
        MirTy::Char => "char".to_string(),
        MirTy::Unit => "()".to_string(),
        // §6.2 offers "generated concrete tuple or named internal aggregate; choose one
        // canonical form". A Rust tuple is chosen: it needs no generated definition, no name to
        // keep deterministic, and no reachability walk to decide which shapes to emit. The cost
        // is that `Projection::Field` has two Rust syntaxes depending on base type, which
        // `emit_places::TyEnv` resolves.
        MirTy::Tuple(elems) => {
            let rendered: Result<Vec<String>, _> =
                elems.iter().map(|e| emit_ty_at(e, at)).collect();
            let rendered = rendered?;
            // A 1-tuple needs Rust's trailing comma to stay a tuple rather than a parenthesised
            // expression; `()` for the empty case coincides with Unit, which is correct.
            match rendered.len() {
                0 => "()".to_string(),
                1 => format!("({},)", rendered[0]),
                _ => format!("({})", rendered.join(", ")),
            }
        }
        MirTy::Array(elem, n) => format!("[{}; {n}]", emit_ty_at(elem, at)?),
        // WP-C5.3d-1a: the ephemeral borrowed-call reference lane (CD-062). Admitted ONLY in the
        // bounded shapes the lane allows -- a reference-typed function parameter, and a
        // same-block borrow consumed without being stored, returned, or carried across blocks.
        // The pre-emission validator is what enforces that; this only says how one is spelled.
        MirTy::Ref { mutable, inner } => {
            format!("{}{}", at.ref_prefix(*mutable), emit_ty_at(inner, at)?)
        }
        MirTy::Struct(item, args) => {
            format!("{}{lt}", mangle::type_name_for_nominal(item.0, args))
        }
        // WP-C5.3b: user enums only. `Option`/`Result`/`Ordering` (`EnumRef::Core*`) are
        // WP-C5.3c, where they arrive together with match/`?` lowering rather than being
        // half-supported here.
        // WP-C5.3b user enums and WP-C5.3c core enums take the SAME representation: a generated
        // Rust enum with positional tuple variants. §6.2 offered ordinary `Option<T>`/`Result`
        // "if all observable semantics match"; a generated enum is used instead so that one
        // mechanism covers every enum, and so no Rust drop glue exists for a type MIR is
        // responsible for destroying. See CD-060.
        MirTy::Enum(..) => format!(
            "{}{lt}",
            nominal_type_name(ty).ok_or_else(|| {
                BackendDiagnostic::Unsupported(format!("no generated name for enum {ty:?}"))
            })?
        ),
        // WP-C5.4c (§7.1/§7.2): a non-capturing function value is a typed Rust function pointer.
        // This is exactly the calling convention emitted STARK functions already use -- the
        // signature's parameter types are `emit_ty` VALUE types (`emit_param_list`), and the
        // ValueSlot for a non-Copy parameter is an internal body detail, not part of the ABI -- so
        // a `Constant::FnPtr`/sentinel coerces to this type with no wrapper. No `dyn Fn`, closure,
        // `Box`, or raw pointer (§7.1).
        MirTy::FnPtr { params, ret } => {
            let mut ps = Vec::with_capacity(params.len());
            for p in params {
                ps.push(emit_ty_at(p, at)?);
            }
            format!("fn({}) -> {}", ps.join(", "), emit_ty_at(ret, at)?)
        }
        other => {
            return Err(BackendDiagnostic::Unsupported(format!(
                "MirTy {other:?} has no C5.3a generated-Rust representation yet -- enums land in \
                 WP-C5.3b, Option/Result in WP-C5.3c, references/slices/String/Vec are outside \
                 the C5 subset; see WP-C5.1.md's MirTy matrix"
            )))
        }
    })
}

/// Whether MIR classifies `ty` as `Copy`.
///
/// CD-065: the rule is `TypeContext::is_copy`, shared with `mir::verify`. It had been written out
/// here identically — the same defect shape CD-064 closed for destruction order, and the one §7.4
/// most cares about, since a backend that broadened the Copy set from Rust traits would duplicate
/// a value MIR says must move.
pub fn mir_ty_is_copy(ty: &MirTy, types: &TypeContext) -> bool {
    types.is_copy(ty)
}

/// Whether a local of this type is backed by a `ValueSlot`. **The single rule**, shared by the
/// signature emitter, the local declarations, and place emission — three sites that must agree or
/// the generated crate will not even compile (a parameter bound under one convention and read
/// under the other).
///
/// Slot-backed means non-`Copy` AND not a reference. References are ephemeral under the C5.3d-1a
/// lane and have no liveness to track; wrapping one would also make a destructor's `&mut Self`
/// receiver project through a slot instead of through the reference.
/// Whether a type carries a borrow anywhere inside it (OWN-CARRY-001's structural provenance):
/// a reference, or a tuple/array/instantiation containing one. Declared reference *fields* are
/// forbidden by 03 rule 1 and rejected by the front end, so a nominal only carries a borrow through
/// its type arguments.
/// WP-C6.1f "borrow-carrying nominals": the two shapes that still fail, refused **before rustc**.
///
/// Generated nominals now carry lifetime parameters (`Name<'a>` in their declaration, `Name<'_>` at
/// use sites), which makes most borrow-carrying instances work — `Option<&T>`, nested ones, and
/// ones inside tuples all build and run. Two do not, and both fail as `E0502` in the GENERATED
/// crate, so they must be refused here rather than surfacing as errors in code the user never
/// wrote:
///
/// 1. **A slot-backed borrow-carrying nominal.** A non-`Copy` instance (a user struct or enum at a
///    reference) lives in a `ValueSlot`, whose `drop_value`/`take` need `&mut` on the slot — while
///    the reference it stores still borrows its referent's slot immutably. Rust treats those as
///    overlapping across the local's whole lexical region and rejects the program, even though MIR
///    drops the borrower first. Removing the slot is not an escape: it also carries MOVE liveness,
///    and without it the mover fails instead.
///
/// 2. **A function returning a borrow-carrying nominal.** The elided output lifetime ties the
///    result to the input reference for the caller's whole region, which then conflicts with the
///    referent's own slot drop at scope end.
///
/// Both are the `ValueSlot`-versus-Rust-borrow-region tension the C6.1f-a matrix flagged as this
/// package's central design question (§5). It did not appear for plain references or tuples —
/// neither is slot-backed — and it is isolated to these two shapes.
fn refuse_borrow_carrying_nominals(program: &MirProgram) -> Result<(), BackendDiagnostic> {
    // WP-C6.1g-a landing boundary (owner ruling, 2026-07-24). Structural Copy (OWN-COPY-001,
    // amended) makes a `Copy` borrow-carrying nominal non-slot-backed, so it flows through the
    // CD-095 aggregate path and works **in a local and across blocks**. Two shapes stay refused
    // pre-rustc:
    //
    //   1. A **Move** borrow-carrying nominal LOCAL (owned non-`Copy` field, `&mut` field, or a
    //      `Drop` impl): still slot-backed, and the slot pins the borrow across the dispatch loop
    //      (E0502).
    //   2. ANY function whose **return type** is a borrow-carrying nominal, regardless of Copy.
    //      Returning a borrow through the dispatch loop and then consuming it (e.g. `Option::unwrap`,
    //      whose panic-branch match extends the borrow's region) conflicts with the referent's
    //      block-0 assignment — and this fails for `Move` referents identically, so it is a general
    //      borrow-through-return limitation, NOT specific to Copy. `wrap -> H<&P>` happens to build,
    //      but that is an accidental backend-shape success, not supported semantics, so it is
    //      refused uniformly. Uniform borrow-carrier returns are the separate option-2 package
    //      (general borrow-through-return / dispatch-loop linearisation).
    //
    // A plain reference return (`fn f(r: &P) -> &P`) is `MirTy::Ref`, not a nominal, so
    // `nominal_needs_lifetime` is false and it stays supported.
    for body in &program.bodies {
        if nominal_needs_lifetime(&body.ret) {
            return Err(BackendDiagnostic::Unsupported(format!(
                "returning the borrow-carrying nominal `{}` is not representable yet: a borrow \
                 returned through the dispatch loop and then consumed conflicts with the \
                 referent's assignment (E0502/E0506) — this fails for Move referents too, so it is \
                 a general borrow-through-return limitation, not a Copy issue. Borrow-carrying \
                 nominals in LOCALS are supported; a plain reference return is supported",
                crate::mir::dump_ty(&body.ret)
            )));
        }
        for local in &body.locals {
            if nominal_needs_lifetime(&local.ty) && is_slot_backed(&local.ty, &program.types) {
                return Err(BackendDiagnostic::Unsupported(format!(
                    "the Move borrow-carrying nominal `{}` is not representable yet: a non-`Copy` \
                     instance lives in a `ValueSlot`, whose destruction needs `&mut` on the slot \
                     while the reference it stores still borrows its referent (E0502). A `Copy` \
                     borrow-carrying nominal (all fields recursively Copy, no Drop) is supported",
                    crate::mir::dump_ty(&local.ty)
                )));
            }
        }
    }
    Ok(())
}

pub fn ty_carries_reference(ty: &MirTy) -> bool {
    match ty {
        MirTy::Ref { .. } => true,
        MirTy::Tuple(elements) => elements.iter().any(ty_carries_reference),
        MirTy::Array(element, _) | MirTy::Slice(element) => ty_carries_reference(element),
        MirTy::Struct(_, args) | MirTy::Enum(_, args) | MirTy::Core(_, args) => {
            args.iter().any(ty_carries_reference)
        }
        MirTy::FnPtr { params, ret } => {
            params.iter().any(ty_carries_reference) || ty_carries_reference(ret)
        }
        _ => false,
    }
}

pub fn is_slot_backed(ty: &MirTy, types: &TypeContext) -> bool {
    !mir_ty_is_copy(ty, types) && !matches!(ty, MirTy::Ref { .. })
}

/// §6.3: one Rust definition per reachable concrete nominal instance, emitted in the type
/// context's own (`BTreeMap`, therefore deterministic) order.
///
/// **FLAGGED READING, CE4-shaped — see `COMPILER-STATE.md` CD-056.** §6.3 says "do not derive
/// `Clone`, `Copy`, `Eq`, `Ord`, or `Hash` as a shortcut for STARK semantics", while §7.4 says a
/// MIR copy is emitted only for a MIR-`Copy` type and the backend must not broaden that set. A
/// STARK struct with an `impl Copy` needs SOME mechanism for `Operand::Copy` to read it twice.
/// The reading taken here: deriving `Clone`/`Copy` on exactly the instances MIR classifies
/// `Copy` is not a "shortcut for STARK semantics" — MIR decides, the derive follows, and the set
/// is neither broadened nor narrowed. No other trait is derived; `Eq`/`Ord`/`Hash` remain
/// forbidden because those WOULD substitute Rust behaviour for STARK's.
///
/// If the owner reads §6.3 as forbidding this outright, the alternative is an explicit generated
/// copy helper per nominal, and the change is confined to [`derives_for`].
pub fn emit_nominal_definitions(program: &MirProgram) -> Result<String, BackendDiagnostic> {
    let mut out = String::new();
    refuse_borrow_carrying_nominals(program)?;

    // WP-C5.3c: core enums have no `TypeContext` entry -- their variants are DERIVED from their
    // type arguments -- so the reachable instances are collected from the program's own types.
    for ty in collect_core_enum_instances(program) {
        let MirTy::Enum(enum_ref, args) = &ty else {
            continue;
        };
        let variants = variant_payloads(enum_ref, args, &program.types).ok_or_else(|| {
            BackendDiagnostic::Unsupported(format!("no variant table for {ty:?}"))
        })?;
        let name = nominal_type_name(&ty).expect("core enum always names");
        out.push_str(&format!("// STARK nominal: {}\n", crate::mir::dump_ty(&ty)));
        if let Some(derives) = derives_for(&ty, &program.types) {
            out.push_str(&format!("{derives}\n"));
        }
        // WP-C6.1f: a borrow-carrying instance is declared with a lifetime parameter, and its
        // payload types are rendered in DECLARATION position so their references name it.
        let generics = if nominal_needs_lifetime(&ty) {
            "<'a>"
        } else {
            ""
        };
        out.push_str(&format!("enum {name}{generics} {{\n"));
        for (v, payload) in variants.iter().enumerate() {
            let fields: Result<Vec<String>, _> = payload
                .iter()
                .map(|f| emit_ty_at(f, decl_position(&ty)))
                .collect();
            out.push_str(&format!(
                "    {}({}),\n",
                variant_name(v as u32),
                fields?.join(", ")
            ));
        }
        out.push_str("}\n\n");
    }

    for ((item, args), variants) in &program.types.enum_variants {
        let name = mangle::type_name_for_nominal(*item, args);
        let rendered: Vec<String> = args.iter().map(crate::mir::dump_ty).collect();
        out.push_str(&format!(
            "// STARK nominal: enum#{item}[{}]\n",
            rendered.join(", ")
        ));
        let ty = MirTy::Enum(
            crate::mir::EnumRef::User(crate::hir::ItemId(*item)),
            args.clone(),
        );
        if let Some(derives) = derives_for(&ty, &program.types) {
            out.push_str(&format!("{derives}\n"));
        }
        let generics = if nominal_needs_lifetime(&ty) {
            "<'a>"
        } else {
            ""
        };
        out.push_str(&format!("enum {name}{generics} {{\n"));
        for (v, payload) in variants.iter().enumerate() {
            let fields: Result<Vec<String>, _> = payload
                .iter()
                .map(|f| emit_ty_at(f, decl_position(&ty)))
                .collect();
            // Every variant is a TUPLE variant, including empty ones (`V0()` is legal Rust).
            // Uniformity removes a special case from construction (`V0()`), from patterns
            // (`V0(..)` matches it), and from the discriminant match -- a unit variant would
            // need different syntax in all three.
            out.push_str(&format!(
                "    {}({}),\n",
                variant_name(v as u32),
                fields?.join(", ")
            ));
        }
        out.push_str("}\n\n");
    }
    for ((item, args), fields) in &program.types.struct_fields {
        let name = mangle::type_name_for_nominal(*item, args);
        let rendered: Vec<String> = args.iter().map(crate::mir::dump_ty).collect();
        out.push_str(&format!(
            "// STARK nominal: struct#{item}[{}]\n",
            rendered.join(", ")
        ));
        let ty = MirTy::Struct(crate::hir::ItemId(*item), args.clone());
        if let Some(derives) = derives_for(&ty, &program.types) {
            out.push_str(&format!("{derives}\n"));
        }
        let generics = if nominal_needs_lifetime(&ty) {
            "<'a>"
        } else {
            ""
        };
        out.push_str(&format!("struct {name}{generics} {{\n"));
        for (i, field_ty) in fields.iter().enumerate() {
            out.push_str(&format!(
                "    {}: {},\n",
                field_name(i as u32),
                emit_ty_at(field_ty, decl_position(&ty))?
            ));
        }
        out.push_str("}\n\n");
    }
    Ok(out)
}

/// Every distinct core-enum instance the program mentions, in a deterministic order. Walks local
/// declarations and the nominal type tables, recursing through aggregates so an `Option` nested
/// inside a struct field or another enum's payload is still found.
fn collect_core_enum_instances(program: &MirProgram) -> Vec<MirTy> {
    let mut found: std::collections::BTreeMap<String, MirTy> = std::collections::BTreeMap::new();
    let visit = |ty: &MirTy, found: &mut std::collections::BTreeMap<String, MirTy>| {
        walk_ty(ty, found);
    };
    for body in &program.bodies {
        for local in &body.locals {
            visit(&local.ty, &mut found);
        }
        visit(&body.ret, &mut found);
        for param in &body.params {
            visit(param, &mut found);
        }
    }
    for fields in program.types.struct_fields.values() {
        for ty in fields {
            visit(ty, &mut found);
        }
    }
    for variants in program.types.enum_variants.values() {
        for payload in variants {
            for ty in payload {
                visit(ty, &mut found);
            }
        }
    }
    found.into_values().collect()
}

fn walk_ty(ty: &MirTy, found: &mut std::collections::BTreeMap<String, MirTy>) {
    match ty {
        MirTy::Enum(crate::mir::EnumRef::User(_), args) | MirTy::Struct(_, args) => {
            for arg in args {
                walk_ty(arg, found);
            }
        }
        MirTy::Enum(_, args) => {
            found.insert(crate::mir::dump_ty(ty), ty.clone());
            for arg in args {
                walk_ty(arg, found);
            }
        }
        MirTy::Tuple(elems) => {
            for elem in elems {
                walk_ty(elem, found);
            }
        }
        MirTy::Array(elem, _) | MirTy::Slice(elem) | MirTy::Ref { inner: elem, .. } => {
            walk_ty(elem, found)
        }
        _ => {}
    }
}

/// The single point where the flagged §6.3-vs-§7.4 reading is applied; see
/// [`emit_nominal_definitions`].
fn derives_for(ty: &MirTy, types: &TypeContext) -> Option<&'static str> {
    if mir_ty_is_copy(ty, types) {
        // `Clone` is required by Rust for `Copy`, not chosen independently.
        Some("#[derive(Clone, Copy)]")
    } else {
        None
    }
}

/// WP-C5.3d-0: the storage type for a non-`Copy` local — `ValueSlot<T>` rather than a bare `T`.
/// See `stark-runtime/src/slot.rs` for why ordinary Rust ownership cannot express MIR liveness
/// inside the block-dispatch loop.
pub fn emit_slot_ty(ty: &MirTy) -> Result<String, BackendDiagnostic> {
    Ok(format!("stark_runtime::slot::ValueSlot<{}>", emit_ty(ty)?))
}

/// An arbitrary-but-valid value of `ty`, used only to give a local declaration a starting value
/// in multi-block bodies (WP-C5.2c) -- NOT a claim about what the local logically holds. Real
/// MIR liveness (what a lowering-bug read-before-write would violate) is a property of the
/// `__bb` state machine that rustc's definite-assignment analysis cannot see across match arms;
/// see `emit_bodies.rs`'s block-dispatch-loop doc comment for why every local must be
/// default-initialised once more than one block exists.
pub fn default_value_expr(ty: &MirTy, types: &TypeContext) -> Result<String, BackendDiagnostic> {
    Ok(match ty {
        MirTy::Int8
        | MirTy::Int16
        | MirTy::Int32
        | MirTy::Int64
        | MirTy::UInt8
        | MirTy::UInt16
        | MirTy::UInt32
        | MirTy::UInt64 => "0".to_string(),
        MirTy::Float32 | MirTy::Float64 => "0.0".to_string(),
        MirTy::Bool => "false".to_string(),
        MirTy::Char => "'\\0'".to_string(),
        MirTy::Unit => "()".to_string(),
        MirTy::Tuple(elems) => {
            let rendered: Result<Vec<String>, _> =
                elems.iter().map(|e| default_value_expr(e, types)).collect();
            let rendered = rendered?;
            match rendered.len() {
                0 => "()".to_string(),
                1 => format!("({},)", rendered[0]),
                _ => format!("({})", rendered.join(", ")),
            }
        }
        MirTy::Array(elem, n) => {
            let element = default_value_expr(elem, types)?;
            if mir_ty_is_copy(elem, types) {
                format!("[{element}; {n}]")
            } else {
                // `[expr; N]` requires a `Copy` element. `from_fn` builds N independent values
                // without materialising N copies of the expression in the generated source,
                // which a literal element list would (and which would explode for large N).
                format!("core::array::from_fn(|_| {element})")
            }
        }
        MirTy::Struct(item, args) => {
            let fields = types
                .struct_fields
                .get(&(item.0, args.clone()))
                .ok_or_else(|| {
                    BackendDiagnostic::Unsupported(format!(
                        "no type-context entry for struct instance {ty:?}"
                    ))
                })?;
            let name = mangle::type_name_for_nominal(item.0, args);
            let mut parts = Vec::with_capacity(fields.len());
            for (i, field_ty) in fields.iter().enumerate() {
                parts.push(format!(
                    "{}: {}",
                    field_name(i as u32),
                    default_value_expr(field_ty, types)?
                ));
            }
            format!("{name} {{ {} }}", parts.join(", "))
        }
        MirTy::Enum(enum_ref, args) => {
            let variants = variant_payloads(enum_ref, args, types).ok_or_else(|| {
                BackendDiagnostic::Unsupported(format!("no variant table for enum instance {ty:?}"))
            })?;
            let payload = variants.first().cloned().ok_or_else(|| {
                BackendDiagnostic::Unsupported(format!("enum instance {ty:?} declares no variants"))
            })?;
            let name = nominal_type_name(ty).ok_or_else(|| {
                BackendDiagnostic::Unsupported(format!("no generated name for {ty:?}"))
            })?;
            let mut parts = Vec::with_capacity(payload.len());
            for field_ty in &payload {
                parts.push(default_value_expr(field_ty, types)?);
            }
            // The FIRST variant, arbitrarily -- like every other default value here, this exists
            // only so a multi-block body's locals have a starting value, and says nothing about
            // what the local logically holds. MIR liveness governs that.
            format!("{name}::{}({})", variant_name(0), parts.join(", "))
        }
        // WP-C5.4c (§7.4/§7.6): a bare Rust function pointer has no language default, but the CFG
        // dispatch loop default-initialises every non-parameter local. The default is the ABORTING
        // sentinel for this exact signature -- NOT null/zero/transmute and NOT an arbitrary real
        // function (§7.4): a sentinel that returned a value could hide a use-before-init defect,
        // whereas this one aborts if MIR liveness is ever wrong. A `FnPtr` local is Copy, so it is
        // never slot-backed and this is the only place it acquires a starting value.
        MirTy::FnPtr { .. } => mangle::fn_sentinel_name(ty),
        other => {
            return Err(BackendDiagnostic::Unsupported(format!(
                "MirTy {other:?} has no WP-C5.3b default-value representation yet"
            )))
        }
    })
}

// ------------------------------------------------- WP-C5.4c function sentinels --

/// §7.5: collect every DISTINCT `MirTy::FnPtr` reachable in the program, keyed by canonical
/// `dump_ty` so identical signatures collapse and the set is deterministically ordered. Descends
/// recursively through composite types, so a function pointer nested in a tuple, array, reference,
/// nominal type argument (`Option<fn(..)>`), or another function-pointer signature is found. The
/// type sources are every body's params/return/locals and every struct field / enum payload in the
/// `TypeContext` (fields and payloads are where a function pointer hides without being a bare
/// local type).
pub fn collect_fnptr_signatures(program: &MirProgram) -> std::collections::BTreeMap<String, MirTy> {
    let mut out = std::collections::BTreeMap::new();
    for body in &program.bodies {
        for p in &body.params {
            walk_ty_for_fnptr(p, &mut out);
        }
        walk_ty_for_fnptr(&body.ret, &mut out);
        for local in &body.locals {
            walk_ty_for_fnptr(&local.ty, &mut out);
        }
    }
    for fields in program.types.struct_fields.values() {
        for f in fields {
            walk_ty_for_fnptr(f, &mut out);
        }
    }
    for variants in program.types.enum_variants.values() {
        for payload in variants {
            for f in payload {
                walk_ty_for_fnptr(f, &mut out);
            }
        }
    }
    out
}

fn walk_ty_for_fnptr(ty: &MirTy, out: &mut std::collections::BTreeMap<String, MirTy>) {
    match ty {
        MirTy::FnPtr { params, ret } => {
            out.insert(crate::mir::dump_ty(ty), ty.clone());
            for p in params {
                walk_ty_for_fnptr(p, out);
            }
            walk_ty_for_fnptr(ret, out);
        }
        MirTy::Tuple(elems) => {
            for e in elems {
                walk_ty_for_fnptr(e, out);
            }
        }
        MirTy::Array(inner, _) | MirTy::Slice(inner) | MirTy::Ref { inner, .. } => {
            walk_ty_for_fnptr(inner, out)
        }
        MirTy::Struct(_, args) | MirTy::Enum(_, args) | MirTy::Core(_, args) => {
            for a in args {
                walk_ty_for_fnptr(a, out);
            }
        }
        _ => {}
    }
}

/// §7.4/§7.5: emit one aborting sentinel per distinct function-pointer signature, before ordinary
/// bodies (item order is not semantically relevant, but generated source order must be
/// deterministic — the `BTreeMap` guarantees it). Every parameter is accepted and ignored; the
/// body aborts immediately, so `std::process::abort()`'s `!` coerces to the declared return type
/// and the sentinel can never return an arbitrary value that would mask a use-before-init defect.
pub fn emit_fn_sentinels(program: &MirProgram) -> Result<String, BackendDiagnostic> {
    let sigs = collect_fnptr_signatures(program);
    let mut out = String::new();
    for ty in sigs.values() {
        let MirTy::FnPtr { params, ret } = ty else {
            continue;
        };
        let name = mangle::fn_sentinel_name(ty);
        let mut param_decls = Vec::with_capacity(params.len());
        for p in params {
            param_decls.push(format!("_: {}", emit_ty(p)?));
        }
        out.push_str(&format!(
            "// WP-C5.4c aborting sentinel for {}\n\
             fn {name}({}) -> {} {{ std::process::abort() }}\n",
            crate::mir::dump_ty(ty),
            param_decls.join(", "),
            emit_ty(ret)?,
        ));
    }
    Ok(out)
}

/// A Rust expression, not necessarily a single literal token (e.g. a `Char` constant becomes
/// `char::from_u32(...).unwrap()`, and a negative float becomes a unary-negation expression) --
/// callers must not assume the result can be used only where a bare literal is legal, only that
/// it is valid wherever a Rust expression is.
pub fn emit_constant(c: &Constant) -> Result<String, BackendDiagnostic> {
    Ok(match c {
        Constant::Bool(b) => b.to_string(),
        Constant::Unit => "()".to_string(),
        Constant::Int(value, MirTy::Char) => emit_char_constant(*value)?,
        Constant::Int(value, ty) => {
            let suffix = int_suffix(ty)?;
            format!("{value}{suffix}")
        }
        Constant::Float(value, ty) => emit_float_constant(*value, ty)?,
        // WP-C5.4c (§8.1): a function value is the generated function item's name, coerced to the
        // declared `fn(..) -> ..` pointer type by its use context. NOT an address, string lookup,
        // switch table, closure, or environment wrapper (§8.1). The C5.4a linkage preflight has
        // already proven this instance resolves to exactly one body with matching identity (§8.2),
        // so this mapping is safe without re-searching -- identical to how `Callee::Instance`
        // names its target.
        Constant::FnPtr(instance) => mangle::function_name_for_symbol(&instance.symbol),
        other => {
            return Err(BackendDiagnostic::Unsupported(format!(
                "Constant {other:?} has no C5.2a generated-Rust representation yet -- `Str` lands \
                 alongside String/output support"
            )))
        }
    })
}

fn int_suffix(ty: &MirTy) -> Result<&'static str, BackendDiagnostic> {
    Ok(match ty {
        MirTy::Int8 => "i8",
        MirTy::Int16 => "i16",
        MirTy::Int32 => "i32",
        MirTy::Int64 => "i64",
        MirTy::UInt8 => "u8",
        MirTy::UInt16 => "u16",
        MirTy::UInt32 => "u32",
        MirTy::UInt64 => "u64",
        other => {
            return Err(BackendDiagnostic::Unsupported(format!(
                "integer constant with non-integer MirTy {other:?}"
            )))
        }
    })
}

/// A `Char` constant is `Constant::Int(codepoint, MirTy::Char)` (`mir::lower`'s own encoding,
/// f-3b: "a Char literal is its Unicode scalar codepoint, typed Char") -- there is no Rust
/// numeric-literal suffix for `char`, so this reconstructs the value via `char::from_u32`. The
/// `.unwrap()` cannot fail for a program that reached verified MIR: the front end already
/// guarantees every `Char` constant is a valid Unicode scalar value: a failure here would mean a
/// STARK compiler defect upstream of this backend, not a reachable user-facing condition.
fn emit_char_constant(codepoint: i128) -> Result<String, BackendDiagnostic> {
    let codepoint = u32::try_from(codepoint).map_err(|_| {
        BackendDiagnostic::Unsupported(format!(
            "Char constant codepoint {codepoint} does not fit in u32 -- verified MIR should be \
             unreachable here"
        ))
    })?;
    Ok(format!("(char::from_u32({codepoint}u32).unwrap())"))
}

fn emit_float_constant(value: f64, ty: &MirTy) -> Result<String, BackendDiagnostic> {
    let typed_f64 = format_f64_literal_typed(value);
    match ty {
        MirTy::Float64 => Ok(typed_f64),
        // No f32 round-trip literal formatter is implemented here; casting an already-typed f64
        // expression preserves the value without one, at the cost of one extra cast expression.
        MirTy::Float32 => Ok(format!("(({typed_f64}) as f32)")),
        other => Err(BackendDiagnostic::Unsupported(format!(
            "float constant with non-float MirTy {other:?}"
        ))),
    }
}

/// Rust's `Debug` formatting for `f64` (unlike `Display`) always includes a decimal point or
/// exponent, so the result is guaranteed to parse back as a Rust float literal once suffixed --
/// and, per `std`, is already the shortest string that round-trips to the same bit pattern.
/// NaN/infinity have no Rust literal syntax at all, so those branches return an already-typed
/// `f64::NAN`/`f64::INFINITY`/`f64::NEG_INFINITY` expression instead of a bare literal --
/// callers must not append a further type suffix to this function's result.
fn format_f64_literal_typed(value: f64) -> String {
    if value.is_nan() {
        "f64::NAN".to_string()
    } else if value.is_infinite() {
        if value.is_sign_positive() {
            "f64::INFINITY".to_string()
        } else {
            "f64::NEG_INFINITY".to_string()
        }
    } else {
        format!("{value:?}f64")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn compiles_as_a_rust_expression(src: &str) -> bool {
        // A cheap, dependency-free "is this syntactically a Rust expression" check: parenthesize
        // it inside a const-eval-free position and let `rustc --edition 2021 --crate-type lib`
        // parse-check a throwaway file. Skipped if rustc is unavailable, matching the project's
        // existing `rustc_available()` convention (`tests/spike_genrust.rs`,
        // `tests/native_c5_1b_skeleton.rs`) rather than failing the unit test suite in an
        // environment with no Rust toolchain.
        let Ok(status) = std::process::Command::new("rustc")
            .arg("--version")
            .output()
        else {
            return true;
        };
        if !status.status.success() {
            return true;
        }
        let dir = std::env::temp_dir().join(format!(
            "stark_c5_2a_lit_check_{}_{}",
            std::process::id(),
            src.len()
        ));
        let _ = std::fs::create_dir_all(&dir);
        let file = dir.join("check.rs");
        let _ = std::fs::write(
            &file,
            format!("#[allow(dead_code)]\nfn f() {{ let _ = {src}; }}\n"),
        );
        let ok = std::process::Command::new("rustc")
            .arg("--edition")
            .arg("2021")
            .arg("--crate-type")
            .arg("lib")
            .arg("-o")
            .arg(dir.join("out.rlib"))
            .arg(&file)
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false);
        let _ = std::fs::remove_dir_all(&dir);
        ok
    }

    #[test]
    fn int_constants_round_trip_through_rustc() {
        for (value, ty, expected_suffix) in [
            (0i128, MirTy::Int32, "i32"),
            (-5i128, MirTy::Int8, "i8"),
            (255i128, MirTy::UInt8, "u8"),
            (u64::MAX as i128, MirTy::UInt64, "u64"),
        ] {
            let out = emit_constant(&Constant::Int(value, ty)).unwrap();
            assert!(out.ends_with(expected_suffix), "{out}");
            assert!(compiles_as_a_rust_expression(&out), "{out}");
        }
    }

    #[test]
    fn bool_and_unit_constants() {
        assert_eq!(emit_constant(&Constant::Bool(true)).unwrap(), "true");
        assert_eq!(emit_constant(&Constant::Bool(false)).unwrap(), "false");
        assert_eq!(emit_constant(&Constant::Unit).unwrap(), "()");
    }

    #[test]
    fn char_constant_reconstructs_via_from_u32() {
        let out = emit_constant(&Constant::Int(0x41, MirTy::Char)).unwrap(); // 'A'
        assert!(out.contains("char::from_u32"), "{out}");
        assert!(compiles_as_a_rust_expression(&out), "{out}");
    }

    #[test]
    fn float_constants_including_edge_cases_compile() {
        for (value, ty) in [
            (0.0f64, MirTy::Float64),
            (-0.0f64, MirTy::Float64),
            (3.5f64, MirTy::Float32),
            (f64::NAN, MirTy::Float64),
            (f64::INFINITY, MirTy::Float64),
            (f64::NEG_INFINITY, MirTy::Float32),
        ] {
            let out = emit_constant(&Constant::Float(value, ty)).unwrap();
            assert!(compiles_as_a_rust_expression(&out), "{out}");
        }
    }

    #[test]
    fn unsupported_constants_are_reported_not_guessed() {
        assert!(matches!(
            emit_constant(&Constant::Str("x".to_string())),
            Err(BackendDiagnostic::Unsupported(_))
        ));
    }
}
