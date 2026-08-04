//! WP-C4.2..C4.5d — typed HIR → MIR lowering.
//!
//! Lowers the supported subset of Core v1 into STARK MIR v0.1 (see `mir.md`, APPROVED
//! CD-028): literals and locals; unary/binary operations (trapping ones as `Checked`
//! terminators, short-circuit `&&`/`||` as control flow); blocks and assignments (incl.
//! compound); functions and direct calls, monomorphised generic instances (C4.5c), function
//! values and indirect calls (CD-021); methods/associated fns/trait dispatch (C4.5a); array
//! indexing via proof tokens and real references (C4.5b); `if`/`while`/`loop`/
//! `for`-over-range, `break`/`continue`, `return`; tuples, arrays, structs, and enums
//! (incl. `Option`/`Result` as logical enums per CD-028); shallow pattern matching via
//! `Discriminant` + `SwitchInt`; ownership and drop elaboration (C4.5d) — per-unit
//! `DropFlag`-guarded `Drop` terminators at scope/early exits, overwrite, discard, and
//! `drop(x)`, with dtor instances discovered into the worklist.
//!
//! Everything outside the subset returns a clean `LowerError::Unsupported` naming the C4.5
//! owner — no construct is silently mislowered (charter: nothing unsupported reaches a backend
//! silently).
//!
//! Evaluation order (CD-007/CD-010) is preserved structurally: operands, call arguments, and
//! aggregate fields are lowered left to right into temporaries; assignment lowers RHS before
//! resolving the LHS place; conditions/scrutinees lower before their branches.

use super::*;
use crate::ast::{AssignOp, BinOp, Lit, Primitive, UnOp};
use crate::hir::{self, Builtin, ExprId, Hir, ItemId, ItemKind, Res, StmtKind};
use crate::literal;
use crate::mir::provider_lower::ProviderLowering;
use crate::source::SourceFile;
use crate::typecheck::{Ty, TypeTables};
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::sync::Arc;

pub struct LowerError {
    pub what: String,
    pub span: Span,
}

/// LIMIT-MIR-MONO-INSTANCES — the named compiler-resource limit on monomorphised function
/// instances per program (contract §2: recursive or explosive instantiation must fail through
/// a named limit with a compiler-limit diagnostic, never an arbitrary crash; resource
/// classification per C2.9). The value is a capacity choice, not semantics; raising it is not
/// a contract change. It also indirectly bounds the type-nesting depth polymorphic recursion
/// can build (each runaway instance nests one constructor deeper), keeping the recursive type
/// converters within stack budget — revisit both together if the value changes.
pub const LIMIT_MIR_MONO_INSTANCES: usize = 512;

/// LIMIT-MIR-TYPE-DEPTH — the named limit on type-constructor nesting depth.
///
/// `LIMIT_MIR_MONO_INSTANCES` was documented as bounding nesting depth *indirectly*, one
/// constructor per runaway instance, "keeping the recursive type converters within stack budget".
/// An indirect bound is exactly what breaks across platforms: 512-deep nesting fits the 8 MB stack
/// Linux and macOS give a test thread, and overflows the 1 MB Windows gives one. So
/// `polymorphic_recursion_trips_the_named_instance_limit` died by `STATUS_STACK_OVERFLOW` on Windows
/// while passing elsewhere — the two limits racing, with the platform deciding the winner.
///
/// Its own contract says it must fail "deterministically, never by memory exhaustion or stack
/// overflow", so the fix is to make the depth bound DIRECT rather than to give the test a bigger
/// stack.
///
/// 64 is far above anything ordinary code writes — `Option<Vec<(Int32, String)>>` is 4 — and far
/// below what any supported platform's stack cannot absorb. Like the instance limit it is a capacity
/// choice, not semantics.
pub const LIMIT_MIR_TYPE_DEPTH: usize = 64;

/// The nesting depth of a MIR type, computed ITERATIVELY and abandoned as soon as the limit is
/// exceeded.
///
/// Iterative because a recursive depth check would overflow on exactly the inputs it exists to
/// reject — which is the whole failure being fixed — and early-exit so it stays cheap.
fn mir_ty_depth_exceeds(ty: &MirTy, limit: usize) -> bool {
    let mut stack = vec![(ty, 0usize)];
    while let Some((ty, depth)) = stack.pop() {
        if depth > limit {
            return true;
        }
        match ty {
            MirTy::Tuple(elems) => stack.extend(elems.iter().map(|e| (e, depth + 1))),
            MirTy::Array(elem, _) | MirTy::Slice(elem) => stack.push((elem.as_ref(), depth + 1)),
            MirTy::Ref { inner, .. } => stack.push((inner.as_ref(), depth + 1)),
            MirTy::Struct(_, args) | MirTy::Enum(_, args) | MirTy::Core(_, args) => {
                stack.extend(args.iter().map(|a| (a, depth + 1)))
            }
            MirTy::FnPtr { params, ret } => {
                stack.extend(params.iter().map(|p| (p, depth + 1)));
                stack.push((ret.as_ref(), depth + 1));
            }
            _ => {}
        }
    }
    false
}

/// The type arguments a discovered callee is monomorphised at.
fn fn_key_type_args(key: &FnKey) -> Vec<&MirTy> {
    match key {
        FnKey::Top(_, args) => args.iter().collect(),
        FnKey::ImplFn {
            type_args,
            method_args,
            ..
        } => type_args.iter().chain(method_args.iter()).collect(),
        FnKey::TraitDefault {
            self_args,
            method_args,
            ..
        } => self_args.iter().chain(method_args.iter()).collect(),
    }
}

/// Does `ty` mention a user struct/enum anywhere? Comparisons on such types dispatch through
/// the user's `Eq`/`Ord` impl (C4.5e); MIR's structural `BinOp` must not be emitted for them.
fn ty_mentions_user_nominal(ty: &MirTy) -> bool {
    match ty {
        MirTy::Struct(..) | MirTy::Enum(EnumRef::User(_), _) => true,
        MirTy::Enum(_, args) | MirTy::Core(_, args) => args.iter().any(ty_mentions_user_nominal),
        MirTy::Tuple(elems) => elems.iter().any(ty_mentions_user_nominal),
        MirTy::Array(elem, _) | MirTy::Slice(elem) => ty_mentions_user_nominal(elem),
        MirTy::Ref { inner, .. } => ty_mentions_user_nominal(inner),
        // **EXHAUSTIVE ON PURPOSE.** The arm asserts "no user impl governs comparison of this
        // type", which is a licence to emit a structural `BinOp`; a wildcard grants that licence to
        // every variant nobody has classified yet. FnPtr comparison is rejected upstream by the
        // checker (TYPE-FN-001), and a host resource is not comparable at all — neither reaches a
        // structural `BinOp`, so both are false because nothing dispatches, not by omission.
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
        | MirTy::String
        | MirTy::FnPtr { .. }
        | MirTy::HostResource(_) => false,
    }
}

/// Does `ty` carry a reference (borrow) anywhere below the top level? A slot-backed (droppable)
/// composite whose field read returns a borrow needs a generated lifetime the backend does not emit
/// yet (E0106) — so Display of such a composite (`(String, &str, i32)`) is refused for now.
fn ty_carries_ref(ty: &MirTy) -> bool {
    match ty {
        MirTy::Ref { .. } => true,
        MirTy::Enum(_, args) | MirTy::Core(_, args) | MirTy::Struct(_, args) => {
            args.iter().any(ty_carries_ref)
        }
        MirTy::Tuple(elems) => elems.iter().any(ty_carries_ref),
        MirTy::Array(elem, _) | MirTy::Slice(elem) => ty_carries_ref(elem),
        // **EXHAUSTIVE ON PURPOSE.** "Carries no borrow" is a claim about a type, not a decision to
        // skip one, so it must be stated per variant.
        //
        // **This is the third copy of one rule**, after `emit_types::ty_carries_reference` and
        // `emit_types::ty_contains_ref`, and it does not agree with the first: that one descends
        // into a `FnPtr`'s parameters and return, this one calls every fn value borrow-free. The
        // difference is defensible — a Rust `fn(&T)` is higher-ranked and needs no lifetime
        // parameter, which is the only thing this predicate guards (E0106) — but it is an
        // agreement the three copies have never been checked against each other for. Recorded
        // rather than silently harmonised; unifying them is its own change.
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
        | MirTy::String
        | MirTy::FnPtr { .. }
        | MirTy::HostResource(_) => false,
    }
}

fn unsupported<T>(what: impl Into<String>, span: Span) -> Result<T, LowerError> {
    Err(LowerError {
        what: what.into(),
        span,
    })
}

/// `unsupported`'s error without the `Result` wrapper, for `ok_or_else`.
fn unsupported_err(what: impl Into<String>, span: Span) -> LowerError {
    LowerError {
        what: what.into(),
        span,
    }
}

/// WP-C7.8.8 step 6: the `MirTy` for an ABI scalar. The ABI's `ScalarTy` is a closed set, so this
/// is total -- there is no fallback arm to get wrong.
fn scalar_mir_ty(t: crate::provider_abi::ScalarTy) -> MirTy {
    use crate::provider_abi::ScalarTy as S;
    match t {
        S::U8 => MirTy::UInt8,
        S::U16 => MirTy::UInt16,
        S::U32 => MirTy::UInt32,
        S::U64 => MirTy::UInt64,
        S::I8 => MirTy::Int8,
        S::I16 => MirTy::Int16,
        S::I32 => MirTy::Int32,
        S::I64 => MirTy::Int64,
        S::Bool => MirTy::Bool,
        S::F32 => MirTy::Float32,
        S::F64 => MirTy::Float64,
    }
}

/// The zero value an out-slot local is initialised to before its address is taken.
fn zero_of(ty: &MirTy, span: Span) -> Result<Operand, LowerError> {
    Ok(match ty {
        MirTy::Bool => Operand::Const(Constant::Bool(false)),
        MirTy::Float32 | MirTy::Float64 => Operand::Const(Constant::Float(0.0, ty.clone())),
        MirTy::Int8
        | MirTy::Int16
        | MirTy::Int32
        | MirTy::Int64
        | MirTy::UInt8
        | MirTy::UInt16
        | MirTy::UInt32
        | MirTy::UInt64 => Operand::Const(Constant::Int(0, ty.clone())),
        _ => return unsupported("provider out-slot of a non-scalar type", span),
    })
}

/// C4.5f-3c: per-program metadata for multi-file / multi-package lowering. Every item knows
/// its defining file (so spans and name reads land in the right source) and its module path
/// (so canonical symbols are package/module-qualified: `⟨package⟩::⟨module⟩::name@[args]`).
struct ProgramMeta {
    /// Interned files; index = `FileId`. The entry file is `FileId(0)`.
    files: Vec<Arc<SourceFile>>,
    /// `item.0` → (defining file, module path from the root, outermost first).
    items: HashMap<u32, (FileId, Vec<String>)>,
    /// Every item reachable from the root, modules included (deterministic walk order).
    all_items: Vec<ItemId>,
    /// WP-C6.1g-a (OWN-COPY-001, amended): items that are `Copy` when their type arguments are —
    /// impl-`Copy` plus structurally eligible. The single source both `FnLowerer::is_copy` and
    /// `TypeContext::is_copy` consult, mirroring the front end's `copy_eligible_types` so the
    /// engines cannot disagree about Copy-ness.
    copy_eligible: std::collections::HashSet<u32>,
    /// WP-C6.2c: associated-type bindings, keyed by `(implementing nominal item, assoc name)` to
    /// the impl's bound HIR type. Lets the monomorphiser resolve a projection `T::Item` once `T`'s
    /// concrete nominal is known, mirroring the front end's `assoc_projections`.
    assoc_projections: HashMap<(u32, String), hir::TypeId>,
}

impl ProgramMeta {
    fn build(hir: &Hir, entry: &Arc<SourceFile>) -> Result<Self, LowerError> {
        let mut files: Vec<Arc<SourceFile>> = vec![entry.clone()];
        let mut by_name: HashMap<String, FileId> = HashMap::new();
        by_name.insert(entry.name.clone(), FileId(0));
        let mut intern = |file: &Arc<SourceFile>, files: &mut Vec<Arc<SourceFile>>| -> FileId {
            if let Some(&id) = by_name.get(&file.name) {
                return id;
            }
            let id = FileId(files.len() as u32);
            files.push(file.clone());
            by_name.insert(file.name.clone(), id);
            id
        };

        let root_items = match &hir.root {
            hir::Root::Program(items) => items.clone(),
            _ => return unsupported("non-program root", Span { lo: 0, hi: 0 }),
        };
        let mut items: HashMap<u32, (FileId, Vec<String>)> = HashMap::new();
        let mut all_items: Vec<ItemId> = Vec::new();
        let mut stack: Vec<(ItemId, Vec<String>)> =
            root_items.iter().rev().map(|&i| (i, Vec::new())).collect();
        while let Some((item_id, path)) = stack.pop() {
            let file_id = match hir.item_files.get(&item_id) {
                Some(f) => intern(f, &mut files),
                None => FileId(0),
            };
            items.insert(item_id.0, (file_id, path.clone()));
            all_items.push(item_id);
            if let ItemKind::Mod {
                name,
                items: Some(children),
            } = &hir.item(item_id).kind
            {
                // The mod's name span reads in the file DECLARING the mod (this item's own
                // file); dependency-package wrappers use synthetic spans resolved by name.
                let mod_name = if let Some(s) = hir.synthetic_spans.get(name) {
                    s.clone()
                } else {
                    let src = &files[file_id.0 as usize].src;
                    src.get(name.lo as usize..name.hi as usize)
                        .unwrap_or("?")
                        .to_string()
                };
                // DEV-114 (packet 2, option B): crossing a PACKAGE boundary restarts identity.
                //
                // TYPE-NOMINAL-001 defines identity as "canonical package instance + module path +
                // item name": the package instance is the ROOT of identity, so a dependency edge is
                // not a module-path segment. Before this, a package's prefix was whatever chain of
                // modules happened to reach it first — `model::leaf` when reached from the root,
                // `logic::model::leaf` when reached through `logic`, and (because dependency
                // iteration walked a `HashMap`) which one you got varied per process.
                //
                // A synthetic span marks a dependency-package wrapper; a plain `mod` has a real span
                // in its declaring file. So the reset is exact rather than heuristic.
                let crosses_package_boundary = hir.synthetic_spans.contains_key(name);
                let mut child_path = if crosses_package_boundary {
                    Vec::new()
                } else {
                    path
                };
                child_path.push(mod_name);
                for &child in children.iter().rev() {
                    stack.push((child, child_path.clone()));
                }
            }
        }
        let copy_eligible = crate::typecheck::copy_eligible_types(hir)
            .into_iter()
            .map(|id| id.0)
            .collect();
        // WP-C6.2c: index every impl's associated-type bindings by implementing nominal + name.
        let mut assoc_projections: HashMap<(u32, String), hir::TypeId> = HashMap::new();
        for &item_id in &all_items {
            let ItemKind::Impl {
                items: impl_items, ..
            } = &hir.item(item_id).kind
            else {
                continue;
            };
            let Some(nominal) = impl_self_item(hir, item_id) else {
                continue;
            };
            let file_id = match hir.item_files.get(&item_id) {
                Some(f) => *by_name.get(&f.name).unwrap_or(&FileId(0)),
                None => FileId(0),
            };
            let src = &files[file_id.0 as usize].src;
            for impl_item in impl_items {
                if let hir::ImplItem::AssocType { name, ty } = impl_item {
                    let assoc_name = src
                        .get(name.lo as usize..name.hi as usize)
                        .unwrap_or("?")
                        .to_string();
                    assoc_projections.insert((nominal.0, assoc_name), *ty);
                }
            }
        }
        Ok(ProgramMeta {
            files,
            items,
            all_items,
            copy_eligible,
            assoc_projections,
        })
    }

    fn item_file(&self, item: ItemId) -> FileId {
        self.items
            .get(&item.0)
            .map(|(f, _)| *f)
            .unwrap_or(FileId(0))
    }

    fn item_src(&self, item: ItemId) -> &str {
        &self.files[self.item_file(item).0 as usize].src
    }

    /// Read a span belonging to `item`'s file.
    fn item_text(&self, item: ItemId, span: Span) -> &str {
        self.item_src(item)
            .get(span.lo as usize..span.hi as usize)
            .unwrap_or("?")
    }

    /// `"dep::inner::"` for a nested item; empty for root items.
    fn symbol_prefix(&self, item: ItemId) -> String {
        match self.items.get(&item.0) {
            Some((_, path)) if !path.is_empty() => format!("{}::", path.join("::")),
            _ => String::new(),
        }
    }
}

/// Lower a whole program (entry `main` plus every transitively-called supported function).
pub fn lower_program(
    hir: &Hir,
    tables: &TypeTables,
    file: Arc<SourceFile>,
) -> Result<MirProgram, LowerError> {
    lower_program_with_providers(hir, tables, file, ProviderLowering::none())
}

/// WP-C7.8.8 step 6: `lower_program`, plus the provider calls the program may emit.
///
/// A separate entry point rather than a new parameter on `lower_program`: every existing caller
/// binds no provider, and an added argument at ~20 sites would be churn that says nothing. The
/// arena reaches `MirProgram::provider_calls` verbatim -- lowering performs no selection and
/// validates no metadata, because A10 3 puts both before this point.
pub fn lower_program_with_providers(
    hir: &Hir,
    tables: &TypeTables,
    file: Arc<SourceFile>,
    providers: &ProviderLowering,
) -> Result<MirProgram, LowerError> {
    // C4.5f-3c: multi-file/multi-package metadata — per-item files, module paths, and the
    // full (module-nested included) item list.
    let meta = ProgramMeta::build(hir, &file)?;

    // `main` is the ROOT `main` (executable-mode selection, CD-017): module/package `main`s
    // do not qualify.
    let mut main = None;
    for &item_id in &meta.all_items {
        if let ItemKind::Fn(def) = &hir.item(item_id).kind {
            if meta.symbol_prefix(item_id).is_empty()
                && meta.item_text(item_id, def.sig.name) == "main"
            {
                main = Some(item_id);
            }
        }
    }
    let Some(main) = main else {
        return unsupported("program without a `main` function", Span { lo: 0, hi: 0 });
    };

    // A11: resolve the synthesized nominal NAMES to item ids, now that `meta` can map an item to its
    // file and read its name text, and complete the close bindings the driver could only half-build.
    // Cloned because `providers` is shared and this fills `resource_items` in place.
    let mut providers = providers.clone();
    let resolved_closes = if providers.resource_nominal_names.is_empty() {
        Vec::new()
    } else {
        let names: Vec<(u32, String)> = meta
            .all_items
            .iter()
            .filter_map(|&item| match &hir.item(item).kind {
                ItemKind::Enum { name, .. } | ItemKind::Struct { name, .. } => {
                    Some((item.0, meta.item_text(item, *name).to_string()))
                }
                _ => None,
            })
            .collect();
        providers
            .resolve_nominals(|nominal| {
                names
                    .iter()
                    .find(|(_, n)| n == nominal)
                    .map(|(id, _)| ItemId(*id))
            })
            .map_err(|what| LowerError {
                what,
                span: Span { lo: 0, hi: 0 },
            })?
    };
    let providers = &providers;

    let mut program = MirProgram {
        files: meta.files.clone(),
        bodies: Vec::new(),
        types: TypeContext::default(),
        mir_version: MIR_VERSION.to_string(),
        runtime_surface: MIR_RUNTIME_SURFACE.to_string(),
        // A10: resolved before lowering (A10 3) and carried verbatim. Empty for every program
        // that binds no provider.
        provider_calls: providers.arena.clone(),
        // A11 §5: selected at RESOLUTION, carried here verbatim. Empty while no resource is bound,
        // which is every program that touches no host resource.
        provider_closes: resolved_closes.clone(),
        // A11: sorted, so the program's identity is a function of the manifest rather than of
        // iteration order -- the same property CD-213 gave capabilities.
        resource_bindings: providers
            .resource_items
            .iter()
            .map(|(resource, item)| {
                (
                    resource.clone(),
                    crate::mir::HostResourceNominal::Item(*item),
                )
            })
            .collect(),
    };

    // Populate the nominal type context (struct fields, user-enum variant payloads) for every
    // non-generic nominal — module-nested ones included (f-3c) — so the verifier/backends can
    // resolve projections. Each nominal gets a probe keyed to itself so field-type spans read
    // in the nominal's own file.
    // WP-C6.1g-a: the structural+impl Copy eligibility set, shared with the front end via
    // `ProgramMeta::copy_eligible` (computed once from `copy_eligible_types`).
    program.types.copy_eligible_items = meta.copy_eligible.iter().copied().collect();
    // A11 §5: `drop_plan::plan_for` resolves destruction from the TYPE alone, and a host resource's
    // destruction IS its close -- so the close has to be reachable from the type context, exactly as
    // a nominal's `Drop` impl is.
    for binding in &resolved_closes {
        program
            .types
            .host_resource_closes
            .insert(binding.resource.clone(), binding.close);
    }
    for &item_id in &meta.all_items {
        let probe = FnLowerer::with_providers(
            hir,
            tables,
            &meta,
            FnKey::Top(item_id, Vec::new()),
            providers,
        );
        // A1 (CD-031): record which non-generic nominals carry an `impl Copy` (V-COPY-1).
        if matches!(
            &hir.item(item_id).kind,
            ItemKind::Struct { generics, .. } | ItemKind::Enum { generics, .. }
                if generics.is_empty()
        ) && probe.type_has_copy_impl(item_id)
        {
            program.types.copy_types.insert((item_id.0, Vec::new()));
        }
        match &hir.item(item_id).kind {
            ItemKind::Struct {
                fields, generics, ..
            } if generics.is_empty() => {
                let mut tys = Vec::new();
                let mut ok = true;
                for f in fields {
                    // Field HIR types convert through the same path as everything else.
                    match probe.hir_field_ty(f.ty) {
                        Ok(t) => tys.push(t),
                        Err(_) => {
                            ok = false;
                            break;
                        }
                    }
                }
                if ok {
                    program
                        .types
                        .struct_fields
                        .insert((item_id.0, Vec::new()), tys);
                }
            }
            ItemKind::Enum {
                variants, generics, ..
            } if generics.is_empty() => {
                let mut all = Vec::new();
                let mut ok = true;
                for v in variants {
                    let payload: Vec<hir::TypeId> = match &v.kind {
                        hir::VariantKind::Unit => Vec::new(),
                        hir::VariantKind::Tuple(tys) => tys.clone(),
                        hir::VariantKind::Struct(fields) => fields.iter().map(|f| f.ty).collect(),
                    };
                    let mut tys = Vec::new();
                    for ty_id in payload {
                        match probe.hir_field_ty(ty_id) {
                            Ok(t) => tys.push(t),
                            Err(_) => {
                                ok = false;
                                break;
                            }
                        }
                    }
                    if !ok {
                        break;
                    }
                    all.push(tys);
                }
                if ok {
                    program
                        .types
                        .enum_variants
                        .insert((item_id.0, Vec::new()), all);
                }
            }
            _ => {}
        }
    }

    // Deterministic, deduplicating instance discovery (contract §2): worklist from `main`,
    // keyed by canonical symbol (top fns, impl methods/assoc fns, trait defaults — C4.5a;
    // module/package-qualified per f-3c).
    let mut queued: BTreeMap<String, ()> = BTreeMap::new();
    let mut worklist = VecDeque::new();
    let main_key = FnKey::Top(main, Vec::new());
    queued.insert(key_symbol(hir, &meta, &main_key)?, ());
    worklist.push_back(main_key);
    let mut bodies = Vec::new();
    while let Some(key) = worklist.pop_front() {
        let mut lowerer = FnLowerer::with_providers(hir, tables, &meta, key.clone(), providers);
        let body = lowerer.lower_body()?;
        // C4.5d: dtor symbols this body's drop glue dispatches through.
        program
            .types
            .drop_impls
            .append(&mut lowerer.drop_impl_symbols);
        // WP-C6.3d: `Eq` instances the map ops dispatch key identity through.
        program.types.eq_impls.append(&mut lowerer.eq_impl_symbols);
        for callee in lowerer.discovered_callees {
            // LIMIT-MIR-TYPE-DEPTH, checked BEFORE `key_symbol`, which renders the type
            // RECURSIVELY (`symbol_ty`) and is the recursion that overflowed. Checking here rather
            // than inside it keeps the guard ahead of every recursive consumer -- drop planning,
            // dumping and the backends all walk the same type.
            if let Some(deep) = fn_key_type_args(&callee)
                .into_iter()
                .find(|t| mir_ty_depth_exceeds(t, LIMIT_MIR_TYPE_DEPTH))
            {
                let _ = deep;
                return Err(LowerError {
                    what: format!(
                        "program exceeds the compiler resource limit LIMIT-MIR-TYPE-DEPTH \
                         ({LIMIT_MIR_TYPE_DEPTH} nested type constructors); recursive generic \
                         instantiation cannot be compiled"
                    ),
                    span: Span { lo: 0, hi: 0 },
                });
            }
            let symbol = key_symbol(hir, &meta, &callee)?;
            if queued.insert(symbol, ()).is_none() {
                // C4.5c: the named instance limit — polymorphic recursion or explosive
                // generic instantiation fails here deterministically, never by exhaustion.
                if queued.len() > LIMIT_MIR_MONO_INSTANCES {
                    return Err(LowerError {
                        what: format!(
                            "program exceeds the compiler resource limit \
                             LIMIT-MIR-MONO-INSTANCES ({LIMIT_MIR_MONO_INSTANCES} monomorphised \
                             function instances); recursive generic instantiation cannot be \
                             compiled"
                        ),
                        span: Span { lo: 0, hi: 0 },
                    });
                }
                worklist.push_back(callee);
            }
        }
        bodies.push(body);
    }
    bodies.sort_by(|a, b| a.instance.symbol.cmp(&b.instance.symbol));
    program.bodies = bodies;
    // C4.5c: register every generic nominal instantiation reachable from the lowered bodies
    // in the type context, so the verifier and backends can resolve its projections.
    register_reachable_nominal_instances(hir, tables, &meta, &mut program, providers)?;
    Ok(program)
}

/// Field/variant payload types for one monomorphised nominal instance.
enum NominalFields {
    Struct(Vec<MirTy>),
    Enum(Vec<Vec<MirTy>>),
}

fn nominal_instance_fields(
    hir: &Hir,
    tables: &TypeTables,
    meta: &ProgramMeta,
    item: ItemId,
    args: &[MirTy],
    providers: &ProviderLowering,
) -> Result<NominalFields, LowerError> {
    let span0 = Span { lo: 0, hi: 0 };
    // The probe is keyed to the nominal itself, so field-type spans read in ITS file (f-3c).
    //
    // **Provider-aware (CD-234).** This probe builds the variant-payload table, and a provider-blind
    // one recorded `Result<TcpStream, E>`'s payload as the enum SHELL while the body lowered the same
    // type as a `HostResource` -- MIR-0004 at every construction and MIR-0005 at every call between
    // them. The binding has to be visible everywhere a nominal's representation is decided, not only
    // where bodies are lowered.
    let mut probe =
        FnLowerer::with_providers(hir, tables, meta, FnKey::Top(item, Vec::new()), providers);
    let generics = match &hir.item(item).kind {
        ItemKind::Struct { generics, .. } | ItemKind::Enum { generics, .. } => generics,
        _ => return unsupported("nominal instance of a non-nominal item", span0),
    };
    if generics.len() != args.len() {
        return unsupported(
            "nominal type instantiated with the wrong number of type arguments",
            span0,
        );
    }
    for (param, ty) in generics.iter().zip(args) {
        let name = meta.item_text(item, param.name).to_string();
        probe.param_subst.insert(name, ty.clone());
    }
    match &hir.item(item).kind {
        ItemKind::Struct { fields, .. } => {
            let tys = fields
                .iter()
                .map(|f| probe.hir_field_ty(f.ty))
                .collect::<Result<Vec<_>, _>>()?;
            Ok(NominalFields::Struct(tys))
        }
        ItemKind::Enum { variants, .. } => {
            let mut all = Vec::new();
            for v in variants {
                let payload: Vec<hir::TypeId> = match &v.kind {
                    hir::VariantKind::Unit => Vec::new(),
                    hir::VariantKind::Tuple(tys) => tys.clone(),
                    hir::VariantKind::Struct(fields) => fields.iter().map(|f| f.ty).collect(),
                };
                let tys = payload
                    .iter()
                    .map(|&t| probe.hir_field_ty(t))
                    .collect::<Result<Vec<_>, _>>()?;
                all.push(tys);
            }
            Ok(NominalFields::Enum(all))
        }
        _ => unreachable!("guarded above"),
    }
}

/// Walk every type that appears in the lowered bodies' locals and register a type-context
/// entry for each generic nominal instantiation encountered, closing over field types
/// (a `Pair<Int32>` field of type `Option<Point>` registers nothing new, but a field of
/// another generic nominal recurses). Non-generic nominals keep their up-front entries.
fn register_reachable_nominal_instances(
    hir: &Hir,
    tables: &TypeTables,
    meta: &ProgramMeta,
    program: &mut MirProgram,
    providers: &ProviderLowering,
) -> Result<(), LowerError> {
    use std::collections::BTreeSet;
    let mut visit: Vec<MirTy> = Vec::new();
    for body in &program.bodies {
        for decl in &body.locals {
            visit.push(decl.ty.clone());
        }
        // WP-C5.3e: a `LayoutQuery` names a type that need not appear in ANY local — nothing in
        // `size_of::<Pair<Int32>>()` constructs a `Pair<Int32>`. Without this, the queried
        // nominal has no field table and the layout walk fails at run time on a program the front
        // end accepted. Found by the DEV-100 composite-substitution fixture.
        for block in &body.blocks {
            for (statement, _) in &block.statements {
                if let Statement::Assign(_, Rvalue::LayoutQuery { ty, .. }) = statement {
                    visit.push(ty.clone());
                }
            }
        }
    }
    let mut seen: BTreeSet<(u32, Vec<MirTy>)> = BTreeSet::new();
    while let Some(ty) = visit.pop() {
        match ty {
            MirTy::Struct(item, args) | MirTy::Enum(EnumRef::User(item), args) => {
                for a in &args {
                    visit.push(a.clone());
                }
                if args.is_empty() || !seen.insert((item.0, args.clone())) {
                    continue;
                }
                match nominal_instance_fields(hir, tables, meta, item, &args, providers)? {
                    NominalFields::Struct(tys) => {
                        for t in &tys {
                            visit.push(t.clone());
                        }
                        program.types.struct_fields.insert((item.0, args), tys);
                    }
                    NominalFields::Enum(variants) => {
                        for v in &variants {
                            for t in v {
                                visit.push(t.clone());
                            }
                        }
                        program.types.enum_variants.insert((item.0, args), variants);
                    }
                }
            }
            MirTy::Enum(_, args) | MirTy::Core(_, args) | MirTy::Tuple(args) => {
                for a in args {
                    visit.push(a);
                }
            }
            MirTy::Array(elem, _) | MirTy::Slice(elem) => visit.push(*elem),
            MirTy::Ref { inner, .. } => visit.push(*inner),
            MirTy::FnPtr { params, ret } => {
                for p in params {
                    visit.push(p);
                }
                visit.push(*ret);
            }
            _ => {}
        }
    }
    Ok(())
}

// ------------------------------------------------------------------ fn lowering --

/// Identity of a lowerable function body (C4.5a). Canonical symbols derive from this key;
/// discovery deduplicates by symbol.
#[derive(Clone, Debug)]
pub enum FnKey {
    /// Top-level `fn`, monomorphised at the given concrete type arguments (empty for
    /// non-generic fns). Arguments are always fully concrete: the discovering caller applies
    /// its own substitution before constructing the key (C4.5c).
    Top(ItemId, Vec<MirTy>),
    /// A method or associated function inside an `impl` block (`items[member]`), monomorphised
    /// at the IMPL-level type arguments (A1: the nominal instantiation's args; empty for
    /// non-generic impls). Always fully concrete, like `Top`.
    ImplFn {
        impl_item: ItemId,
        member: u32,
        type_args: Vec<MirTy>,
        /// WP-C4.7-8.4: the METHOD's own generic arguments for this instantiation, in the
        /// method's declaration order (empty when the method declares none). Separate from
        /// `type_args`, which are the IMPL's — a method on a generic nominal can be generic in
        /// both, and the two substitutions must not be conflated.
        ///
        /// `FnKey` is lowering-internal: it appears nowhere in `mir.md`, so extending it is not
        /// a contract change and needs no CE3. The rendered `Instance.symbol` does change for
        /// generic methods, but §2 states symbols are "deterministic and injective for identical
        /// inputs; NOT a stable external ABI".
        method_args: Vec<MirTy>,
    },
    /// An un-overridden trait default method, monomorphised for one implementing nominal
    /// instantiation (A1: `self_args` are the nominal's concrete type arguments).
    TraitDefault {
        trait_item: ItemId,
        member: u32,
        self_item: ItemId,
        self_args: Vec<MirTy>,
        /// WP-C4.7-9 audit: the DEFAULT METHOD's own generic arguments for this instantiation
        /// (empty when it declares none) — the `TraitDefault` counterpart of
        /// `ImplFn::method_args`.
        method_args: Vec<MirTy>,
    },
}

/// The item's declared name, read in the item's own file (f-3c).
fn item_name_text<'a>(hir: &Hir, meta: &'a ProgramMeta, item: ItemId) -> Option<&'a str> {
    let span = match &hir.item(item).kind {
        ItemKind::Fn(def) => def.sig.name,
        ItemKind::Struct { name, .. }
        | ItemKind::Enum { name, .. }
        | ItemKind::Trait { name, .. } => *name,
        _ => return None,
    };
    Some(meta.item_text(item, span))
}

fn impl_self_item(hir: &Hir, impl_item: ItemId) -> Option<ItemId> {
    let ItemKind::Impl { self_ty, .. } = &hir.item(impl_item).kind else {
        return None;
    };
    match &hir.ty(*self_ty).kind {
        hir::TypeKind::Path {
            res: Res::Item(item),
            ..
        } => Some(*item),
        _ => None,
    }
}

/// Deterministic canonical symbol for a body (contract §2: injective for identical inputs;
/// not a stable external ABI). f-3c: `⟨package/module path⟩::name@[args]` — every name reads
/// in its declaring item's own file, and module-nested items carry their path, so equal
/// names in different modules/packages stay distinct.
fn key_symbol(hir: &Hir, meta: &ProgramMeta, key: &FnKey) -> Result<String, LowerError> {
    let span0 = Span { lo: 0, hi: 0 };
    match key {
        FnKey::Top(item, type_args) => {
            let name = item_name_text(hir, meta, *item).ok_or_else(|| LowerError {
                what: "unnamed top-level fn".into(),
                span: span0,
            })?;
            let args_text = type_args
                .iter()
                .map(|t| symbol_ty(hir, meta, t))
                .collect::<Vec<_>>()
                .join(", ");
            Ok(format!("{}{name}@[{args_text}]", meta.symbol_prefix(*item)))
        }
        FnKey::ImplFn {
            impl_item,
            member,
            type_args,
            method_args,
        } => {
            let ItemKind::Impl { trait_, items, .. } = &hir.item(*impl_item).kind else {
                return unsupported("FnKey::ImplFn on non-impl", span0);
            };
            let self_item = impl_self_item(hir, *impl_item).ok_or_else(|| LowerError {
                what: "impl self type is not a nominal item".into(),
                span: span0,
            })?;
            let type_name = item_name_text(hir, meta, self_item).unwrap_or("?");
            let hir::ImplItem::Fn { def, .. } = &items[*member as usize] else {
                return unsupported("FnKey::ImplFn member is not a fn", span0);
            };
            let method = meta.item_text(*impl_item, def.sig.name);
            let prefix = meta.symbol_prefix(self_item);
            // A1: the impl-level instantiation renders inside the brackets — the non-generic
            // form stays `@[]`, keeping pre-A1 symbols stable.
            let args_text = type_args
                .iter()
                .map(|t| symbol_ty(hir, meta, t))
                .collect::<Vec<_>>()
                .join(", ");
            // WP-C4.7-8.4: a method's OWN arguments render in a second bracket so impl-level and
            // method-level instantiations stay distinguishable and the symbol stays injective.
            // A method with no own generics renders exactly as before.
            let method_text = if method_args.is_empty() {
                String::new()
            } else {
                format!(
                    "::<{}>",
                    method_args
                        .iter()
                        .map(|t| symbol_ty(hir, meta, t))
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            };
            match trait_ {
                None => Ok(format!(
                    "{prefix}{type_name}::{method}@[{args_text}]{method_text}"
                )),
                Some(trait_ref) => {
                    let trait_name = match trait_ref.res {
                        Res::Item(t) => item_name_text(hir, meta, t).unwrap_or("?"),
                        // C4.5d: compiler-known trait impls (`impl Drop for T`) render their
                        // source-level trait name — symbols stay injective and readable.
                        Res::CoreTrait(_) => meta.item_text(*impl_item, trait_ref.path.span),
                        _ => "?",
                    };
                    Ok(format!(
                        "{prefix}{type_name}::{trait_name}::{method}@[{args_text}]{method_text}"
                    ))
                }
            }
        }
        FnKey::TraitDefault {
            trait_item,
            member,
            self_item,
            self_args,
            method_args,
        } => {
            let trait_name = item_name_text(hir, meta, *trait_item).unwrap_or("?");
            let type_name = item_name_text(hir, meta, *self_item).unwrap_or("?");
            let method_text = if method_args.is_empty() {
                String::new()
            } else {
                format!(
                    "::<{}>",
                    method_args
                        .iter()
                        .map(|t| symbol_ty(hir, meta, t))
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            };
            let ItemKind::Trait { items, .. } = &hir.item(*trait_item).kind else {
                return unsupported("FnKey::TraitDefault on non-trait", span0);
            };
            let hir::TraitItem::Method { sig, .. } = &items[*member as usize] else {
                return unsupported("FnKey::TraitDefault member is not a method", span0);
            };
            let method = meta.item_text(*trait_item, sig.name);
            let prefix = meta.symbol_prefix(*self_item);
            if self_args.is_empty() {
                Ok(format!(
                    "{trait_name}::{method}@[{prefix}{type_name}]{method_text}"
                ))
            } else {
                let args_text = self_args
                    .iter()
                    .map(|t| symbol_ty(hir, meta, t))
                    .collect::<Vec<_>>()
                    .join(", ");
                Ok(format!(
                    "{trait_name}::{method}@[{prefix}{type_name}<{args_text}>]{method_text}"
                ))
            }
        }
    }
}

/// WP-C6.2e: render a `MirTy` for a canonical symbol's type arguments using CONTENT-BASED nominal
/// identity — the nominal's module/package path plus its source name — instead of its raw `ItemId`
/// index. `dump_ty`'s `struct#N`/`enum#N` embed the item index, which is assigned by the item walk
/// order and therefore shifts when dependencies are declared in a different order (§21 forbids
/// that: a clean rebuild, relocation, or dependency reorder must not change semantic symbol
/// identity, and no path/index artifact may enter it). Non-nominal shapes already render from
/// content, so they delegate to `dump_ty`'s structure here.
fn symbol_ty(hir: &Hir, meta: &ProgramMeta, ty: &MirTy) -> String {
    let generic = |head: String, args: &[MirTy]| -> String {
        if args.is_empty() {
            head
        } else {
            let inner = args
                .iter()
                .map(|a| symbol_ty(hir, meta, a))
                .collect::<Vec<_>>()
                .join(", ");
            format!("{head}<{inner}>")
        }
    };
    match ty {
        // The `struct#`/`enum#` heads are kept (they keep a user nominal distinct from an
        // identically-named core type — a user MAY declare `struct Vec`), but the numeric `ItemId`
        // that followed them is replaced with the nominal's content path so the head is
        // order-stable.
        MirTy::Struct(item, args) => {
            let name = item_name_text(hir, meta, *item).unwrap_or("?");
            generic(format!("struct#{}{name}", meta.symbol_prefix(*item)), args)
        }
        MirTy::Enum(EnumRef::User(item), args) => {
            let name = item_name_text(hir, meta, *item).unwrap_or("?");
            generic(format!("enum#{}{name}", meta.symbol_prefix(*item)), args)
        }
        MirTy::Enum(EnumRef::CoreOption, args) => generic("Option".to_string(), args),
        MirTy::Enum(EnumRef::CoreResult, args) => generic("Result".to_string(), args),
        MirTy::Enum(EnumRef::CoreOrdering, args) => generic("Ordering".to_string(), args),
        MirTy::Core(core, args) => generic(format!("{core:?}"), args),
        // A11 Q5: `hostres#<provider>/<resource>@<nominal content path>`. The nominal is rendered by
        // CONTENT PATH, never by `ItemId` -- CD-108 established that ordering-dependent indices must
        // not reach canonical identity, and a host resource is no exception.
        //
        // Two nominals bound to one provider resource render differently, and one nominal bound
        // through different providers renders differently. Both are deliberate: A11 7's negative
        // cases turn on telling those apart.
        MirTy::HostResource(r) => {
            // A11 Q5: the nominal is rendered by CONTENT PATH, never by `ItemId` -- CD-108 established
            // that ordering-dependent indices must not reach canonical identity.
            let nominal = match r.nominal {
                crate::mir::HostResourceNominal::Core(c) => format!("core:{c:?}"),
                crate::mir::HostResourceNominal::Item(item) => {
                    let name = item_name_text(hir, meta, item).unwrap_or("?");
                    format!("{}{name}", meta.symbol_prefix(item))
                }
            };
            format!("hostres#{}/{}@{nominal}", r.provider, r.resource)
        }
        MirTy::Tuple(elems) => {
            let inner = elems
                .iter()
                .map(|e| symbol_ty(hir, meta, e))
                .collect::<Vec<_>>()
                .join(", ");
            format!("({inner})")
        }
        MirTy::Array(elem, len) => format!("[{}; {len}]", symbol_ty(hir, meta, elem)),
        MirTy::Slice(elem) => format!("[{}]", symbol_ty(hir, meta, elem)),
        MirTy::Ref { mutable, inner } => format!(
            "&{}{}",
            if *mutable { "mut " } else { "" },
            symbol_ty(hir, meta, inner)
        ),
        MirTy::FnPtr { params, ret } => format!(
            "fn({}) -> {}",
            params
                .iter()
                .map(|p| symbol_ty(hir, meta, p))
                .collect::<Vec<_>>()
                .join(", "),
            symbol_ty(hir, meta, ret)
        ),
        simple => format!("{simple:?}"),
    }
}

/// A2/DEV-070: how a `match` treats its scrutinee.
#[derive(Clone, Copy, PartialEq, Eq)]
enum MatchMode {
    /// Owned scrutinee, consumed by the match (C4.5d): temp materialization, move-out
    /// bindings, unbound-droppable temps, arm-end drops.
    Consuming,
    /// Scrutinee read through a shared reference (`match *self`): matched in place — no move,
    /// no poison, no drops; bindings must be `Copy` (read by copy).
    ByRef,
}

struct LoopTargets {
    continue_target: BlockId,
    break_target: BlockId,
    /// Scope-stack depth at loop entry (C4.5d): `break`/`continue` drop every scope at this
    /// depth or deeper before jumping out of / restarting the loop.
    scope_depth: usize,
    /// A7: for a `loop` in value position, the local that `break <value>` writes before jumping
    /// to the break target — the loop expression's value is read from it at the exit block.
    /// `None` for statement-position loops and for `while`/`for` (both are Unit-typed).
    value_target: Option<LocalId>,
}

/// One drop-tracked unit of a droppable local (C4.5d): a sub-place (pure field path from the
/// local root) that drops as a whole, guarded by its own `DropFlag`. Units are the outermost
/// sub-places whose types stop static decomposition — a type with its own `Drop` impl, an
/// enum (variant known only at runtime), or an array — reached by descending through
/// dtor-less structs and tuples. A whole-value glue drop is observably the ordered sequence
/// of its units' glue drops, which is what makes partial moves representable: moving one
/// unit out clears exactly that unit's flag.
/// A5 (CD-038): one step of a drop-unit path. TYPED for the same reason move paths are — once
/// constant array indices share the space with struct/tuple fields, a raw `u32` sequence can no
/// longer say which kind it meant.
#[derive(Clone, PartialEq, Eq, Debug)]
enum DropStep {
    Field(u32),
    ConstIndex(u64),
}

impl DropStep {
    fn projection(&self) -> Projection {
        match self {
            DropStep::Field(i) => Projection::Field(*i),
            DropStep::ConstIndex(i) => Projection::ConstIndex(*i),
        }
    }
}

#[derive(Clone)]
struct DropUnit {
    path: Vec<DropStep>,
    ty: MirTy,
    flag: LocalId,
}

struct FnLowerer<'a> {
    hir: &'a Hir,
    tables: &'a TypeTables,
    /// WP-C7.8.8 step 6: the provider calls this program may lower. Empty for every program that
    /// binds no provider, which is almost all of them.
    providers: &'a ProviderLowering,
    /// f-3c: program-wide file/module metadata (per-item files and paths).
    meta: &'a ProgramMeta,
    /// The source of the file DEFINING this body's item — body spans read here.
    src: &'a str,
    file: FileId,
    key: FnKey,
    /// Concrete `Self` type for method/trait-default bodies (C4.5a).
    self_subst: Option<MirTy>,
    /// Concrete types for the body's own generic parameters, from the instance's type
    /// arguments (C4.5c monomorphisation). Empty for non-generic bodies.
    param_subst: HashMap<String, MirTy>,
    locals: Vec<LocalDecl>,
    local_map: HashMap<u32, LocalId>,
    blocks: Vec<Option<BasicBlock>>,
    current: BlockId,
    current_statements: Vec<(Statement, SourceInfo)>,
    loops: Vec<LoopTargets>,
    /// 0.1-A13 (WP-C7.9 Packet D): which stream the output operation being lowered right now
    /// writes to. Set for the duration of one `eprint`/`eprintln` call and restored afterwards;
    /// `Stdout` everywhere else, including inside any function body those calls invoke.
    out_channel: OutChannel,
    discovered_callees: Vec<FnKey>,
    /// C4.5d: drop units per droppable user/param local, keyed by MIR local index.
    drop_info: HashMap<u32, Vec<DropUnit>>,
    /// C4.5d: lexical scope stack; each entry lists that scope's droppable locals in
    /// declaration order (drops emit in reverse at scope exit).
    scopes: Vec<Vec<LocalId>>,
    /// C4.5d: `(item, args) → dtor instance symbol` for every `Drop` impl this body's glue
    /// can reach; merged into `TypeContext::drop_impls` by `lower_program`.
    drop_impl_symbols: BTreeMap<(u32, Vec<MirTy>), String>,
    /// WP-C6.3d: selected `Eq::eq` symbols for nominal map KEYS; merged into `TypeContext::eq_impls`.
    eq_impl_symbols: BTreeMap<(u32, Vec<MirTy>), String>,
}

impl<'a> FnLowerer<'a> {
    fn with_providers(
        hir: &'a Hir,
        tables: &'a TypeTables,
        meta: &'a ProgramMeta,
        key: FnKey,
        providers: &'a ProviderLowering,
    ) -> Self {
        // f-3c: the body's spans and text reads belong to the DEFINING item's file.
        let owner = match &key {
            FnKey::Top(item, _) => *item,
            FnKey::ImplFn { impl_item, .. } => *impl_item,
            FnKey::TraitDefault { trait_item, .. } => *trait_item,
        };
        let file = meta.item_file(owner);
        let src: &'a str = &meta.files[file.0 as usize].src;
        FnLowerer {
            hir,
            tables,
            providers,
            meta,
            src,
            file,
            key,
            self_subst: None,
            param_subst: HashMap::new(),
            locals: Vec::new(),
            local_map: HashMap::new(),
            blocks: vec![None],
            current: BlockId(0),
            current_statements: Vec::new(),
            loops: Vec::new(),
            out_channel: OutChannel::Stdout,
            discovered_callees: Vec::new(),
            drop_info: HashMap::new(),
            scopes: Vec::new(),
            drop_impl_symbols: BTreeMap::new(),
            eq_impl_symbols: BTreeMap::new(),
        }
    }

    fn text(&self, span: Span) -> &'a str {
        &self.src[span.lo as usize..span.hi as usize]
    }

    fn info(&self, span: Span) -> SourceInfo {
        SourceInfo {
            file: self.file,
            span,
            origin: Origin::UserCode,
        }
    }

    fn synthetic(&self, span: Span, kind: SyntheticKind) -> SourceInfo {
        SourceInfo {
            file: self.file,
            span,
            origin: Origin::Synthetic(kind),
        }
    }

    // ---- block plumbing ----

    fn new_block(&mut self) -> BlockId {
        self.blocks.push(None);
        BlockId((self.blocks.len() - 1) as u32)
    }

    fn emit(&mut self, stmt: Statement, info: SourceInfo) {
        self.current_statements.push((stmt, info));
    }

    /// Seal the current block with `term` and switch to `next`.
    fn terminate(&mut self, term: Terminator, info: SourceInfo, next: BlockId) {
        let statements = std::mem::take(&mut self.current_statements);
        let sealed = BasicBlock {
            statements,
            terminator: (term, info),
        };
        self.blocks[self.current.0 as usize] = Some(sealed);
        self.current = next;
    }

    /// WP-C7.8.8 step 6: the `ProviderCallId` for an item, if it is a synthesized binding.
    ///
    /// Resolved by qualified name because bindings are computed from the manifest before parsing,
    /// so no item id exists when they are built. The map is empty for every program that binds no
    /// provider, so this costs one lookup against an empty `BTreeMap` in the ordinary case.
    fn provider_call_for(&self, item: ItemId) -> Option<ProviderCallId> {
        if self.providers.is_empty() {
            return None;
        }
        let ItemKind::Fn(def) = &self.hir.item(item).kind else {
            return None;
        };
        let name = self.meta.item_text(item, def.sig.name);
        self.providers.call_for(name)
    }

    /// Lowers a call to a synthesized binding into `Callee::Provider`.
    ///
    /// The shape is fixed by the ABI and by what the emitter already does (`emit_provider.rs`):
    /// out-slots are **caller-owned locals** passed as `&mut`, the call's `dest` receives the raw
    /// `ProviderStatus` code, and the emitter writes slots back only on status zero. So the STARK
    /// `Result<T, E>` is built here, after the call, from the slots.
    fn lower_provider_call(
        &mut self,
        id: ProviderCallId,
        args: &[ExprId],
        dest: Place,
        span: Span,
    ) -> Result<(), LowerError> {
        let call = self
            .providers
            .arena
            .get(id.0 as usize)
            .ok_or_else(|| unsupported_err("provider call id out of range", span))?
            .clone();

        // A non-empty status vocabulary means some nonzero code is a RECOVERABLE error that must
        // become `Err(e)`. That needs the raw error enum's variant for each code, which synthesis
        // derived from the same vocabulary the emitter dispatches on. If the vocabulary is non-empty
        // and no mapping reached lowering, refuse rather than approximate: emitting `Ok` regardless
        // would turn a declared, recoverable failure into a successful call returning an unwritten
        // slot.
        let declares_recoverable = call.status_binding.declared_codes().next().is_some();
        let error_mapping: Option<(String, BTreeMap<u32, u32>)> = self
            .providers
            .error_mapping_for(id)
            .map(|(ty, v)| (ty.to_string(), v.clone()));
        if declares_recoverable && error_mapping.is_none() {
            return unsupported(
                "provider call declares recoverable statuses but no raw error mapping reached \
                 lowering",
                span,
            );
        }

        let mut ops: Vec<Operand> = Vec::with_capacity(call.function.params.len());
        let mut slots: Vec<(LocalId, MirTy)> = Vec::new();
        // A11: newly-owned handle outputs, kept apart from scalar out-slots because their liveness
        // rule is different -- a scalar slot is merely written, a handle slot becomes OWNED.
        let mut handle_outs: Vec<(LocalId, MirTy)> = Vec::new();
        let mut next_arg = 0usize;

        for param in &call.function.params {
            match param {
                crate::provider_abi::AbiParam::ScalarOut(ty) => {
                    // Caller-owned slot. Initialised because MIR has no uninitialised-read story
                    // for a local whose address is taken; the emitter's `MaybeUninit` discipline is
                    // what actually keeps the value unread before a zero status.
                    let mir_ty = scalar_mir_ty(*ty);
                    let slot = self.new_temp(mir_ty.clone());
                    self.emit(
                        Statement::Assign(Place::local(slot), Rvalue::Use(zero_of(&mir_ty, span)?)),
                        self.info(span),
                    );
                    let r = self.new_temp(MirTy::Ref {
                        mutable: true,
                        inner: Box::new(mir_ty.clone()),
                    });
                    self.emit(
                        Statement::Assign(
                            Place::local(r),
                            Rvalue::RefOf {
                                mutable: true,
                                place: Place::local(slot),
                            },
                        ),
                        self.info(span),
                    );
                    ops.push(Operand::Move(Place::local(r)));
                    slots.push((slot, mir_ty));
                }
                // A11/CD-234: a BORROWED handle. The caller keeps ownership, so this is `&R` and the
                // argument is an ordinary shared borrow of the caller's place -- never a move, which
                // would consume the resource the call only reads.
                // DEV-146: `resource_type` is deliberately unbound. The expected type is derived
                // from the OPERAND below rather than rebuilt from the declaration, so the
                // declaration's resource name is not needed here — see the comment on the
                // coercion.
                crate::provider_abi::AbiParam::HandleBorrowed {
                    resource_type: _resource_type,
                } => {
                    let Some(&a) = args.get(next_arg) else {
                        return unsupported(
                            "provider call argument count disagrees with the declaration",
                            span,
                        );
                    };
                    next_arg += 1;
                    let op = self.lower_expr_to_operand(a)?;
                    // DEV-146: weaken `&mut R` to `&R` here, the same as every other call site.
                    //
                    // This arm used to push the operand with NO expected-type coercion, so a
                    // wrapper declaring `fn write(stream: &mut TcpStream, ..)` and forwarding to
                    // the raw binding passed `&mut` where the derived signature wants `&`. The
                    // front end accepted it; MIR-0005 then rejected the call. Accepted-but-
                    // unbuildable — and it made `&mut` UNUSABLE as a package API shape over any
                    // bound resource, which is a language-surface consequence, not a lowering
                    // detail.
                    //
                    // DEV-133's comment predicted exactly this: it routed all six coercion sites
                    // through `weaken_ref_to` and warned that "whichever site was forgotten would
                    // keep this defect". Provider calls were the seventh, and were forgotten —
                    // invisibly, because no first-party package called a resource function until
                    // `stark-net` did.
                    //
                    // The expected type is derived from the OPERAND rather than rebuilt from the
                    // declaration: if what we hold is `&mut X`, what the borrowed-handle slot wants
                    // is `&X`. Reconstructing `HostResourceTy` here would be a second copy of
                    // `provider_sig`'s mapping and could drift from it; this cannot.
                    let held = match &op {
                        Operand::Copy(place) | Operand::Move(place)
                            if place.projection.is_empty() =>
                        {
                            Some(self.locals[place.local.0 as usize].ty.clone())
                        }
                        _ => None,
                    };
                    let op = match held {
                        Some(MirTy::Ref {
                            mutable: true,
                            inner,
                        }) => {
                            let expected = MirTy::Ref {
                                mutable: false,
                                inner,
                            };
                            self.weaken_ref_to(op, &expected, span)?
                        }
                        _ => op,
                    };
                    ops.push(op);
                }
                // A11 §8: a CONSUMED handle. Ownership transfers at call entry and does not return
                // on failure, so the argument is a move and the caller's drop flag is cleared --
                // otherwise the implicit close would run on a handle the provider already owns.
                crate::provider_abi::AbiParam::HandleConsumed { resource_type } => {
                    let Some(&a) = args.get(next_arg) else {
                        return unsupported(
                            "provider call argument count disagrees with the declaration",
                            span,
                        );
                    };
                    next_arg += 1;
                    let _ = resource_type;
                    ops.push(self.lower_expr_to_operand(a)?);
                }
                // A11/CD-234: a NEWLY-OWNED handle out. The argument names the DESTINATION place
                // (the WP-C7.8.4 convention: a `&mut` to a slot-backed resource cannot work, because
                // the slot is dead until the provider writes it). The destination starts dead and
                // becomes live only on status zero.
                crate::provider_abi::AbiParam::HandleOut { resource_type } => {
                    let Some(ty) = self
                        .providers
                        .resource_ty(resource_type, &call.provider.name)
                    else {
                        return unsupported(
                            format!(
                                "provider resource `{resource_type}` has no bound nominal, so its \
                                 handle output has no STARK type"
                            ),
                            span,
                        );
                    };
                    let slot = self.new_temp(ty.clone());
                    // NO initialisation. CD-234: a host-resource slot begins dead, and no default,
                    // aggregate or placeholder may make it live -- only this call's success does.
                    ops.push(Operand::Move(Place::local(slot)));
                    handle_outs.push((slot, ty));
                }
                _ => {
                    // Every remaining form takes its value from the STARK call's own argument list,
                    // in declaration order. Synthesis derived the signature from these same params,
                    // so the counts agree by construction -- but a mismatch is a compiler defect
                    // worth naming rather than an index panic.
                    let Some(&a) = args.get(next_arg) else {
                        return unsupported(
                            "provider call argument count disagrees with the declaration",
                            span,
                        );
                    };
                    next_arg += 1;
                    ops.push(self.lower_expr_to_operand(a)?);
                }
            }
        }

        // `dest` is the raw ProviderStatus code (see `emit_provider::emit_call`), NOT the STARK
        // value. UInt32 because that is `ProviderStatus::code`.
        let status = self.new_temp(MirTy::UInt32);
        let after = self.new_block();
        self.terminate(
            Terminator::Call {
                callee: Callee::Provider(id),
                args: ops,
                dest: Place::local(status),
                target: after,
            },
            self.info(span),
            after,
        );

        // Handle outputs join the result payload. They are appended after the scalar out-slots,
        // matching `provider_sig`'s derivation order, so a signature deriving
        // `Result<(UInt64, TcpStream), E>` and the MIR that produces it agree by construction.
        slots.extend(handle_outs.iter().cloned());

        // With an EMPTY vocabulary, every nonzero code is a contract violation the emitter aborts on
        // before returning. Control reaching here therefore means status zero and written slots:
        // `Ok` is the only outcome, and that is a fact about the emitted Rust rather than optimism.
        // No branch is emitted, because there is no other reachable arm to branch to.
        let Some((error_ty, variant_of_code)) = error_mapping else {
            self.assign_provider_ok(dest, &slots, span)?;
            return Ok(());
        };

        // A non-empty vocabulary means the status has to be examined. `SwitchInt` rather than a
        // chain of comparisons, because the arms ARE a closed set of declared integer codes.
        let error_item = self.nominal_item_by_name(&error_ty).ok_or_else(|| {
            unsupported_err(
                format!("raw error type `{error_ty}` is not among the program's items"),
                span,
            )
        })?;

        let ok_block = self.new_block();
        let join = self.new_block();
        // One block per declared code. Distinct blocks rather than one shared block, because each
        // constructs a DIFFERENT variant -- merging them is the channel collapse Packet 1 1.2
        // forbids in the emitter, for the same reason.
        let mut arms: Vec<(u128, BlockId)> = Vec::new();
        let mut to_fill: Vec<(BlockId, Option<u32>)> = Vec::new();
        for (code, variant) in &variant_of_code {
            let b = self.new_block();
            arms.push((u128::from(*code), b));
            to_fill.push((b, Some(*variant)));
        }
        arms.push((u128::from(crate::provider_bind::STATUS_SUCCESS), ok_block));
        to_fill.push((ok_block, None));

        // `otherwise` is UNREACHABLE, never a fallback error. An undeclared nonzero code already
        // aborted inside the emitted call, so no value flows here -- and a `_ =>` arm mapped to some
        // generic package error is exactly the collapse the three-channel rule exists to prevent.
        let unreachable_block = self.new_block();
        self.terminate(
            Terminator::SwitchInt {
                scrut: Operand::Copy(Place::local(status)),
                arms,
                otherwise: unreachable_block,
            },
            self.info(span),
            unreachable_block,
        );
        self.terminate(Terminator::Unreachable, self.info(span), to_fill[0].0);

        for i in 0..to_fill.len() {
            let variant = to_fill[i].1;
            match variant {
                Some(v) => {
                    let error_ty = MirTy::Enum(EnumRef::User(error_item), Vec::new());
                    let raw = self.new_temp(error_ty.clone());
                    self.emit(
                        Statement::Assign(
                            Place::local(raw),
                            Rvalue::Aggregate(
                                AggKind::EnumVariant(EnumRef::User(error_item), v),
                                Vec::new(),
                            ),
                        ),
                        self.info(span),
                    );
                    // DEV-125: `read_place`, not a hand-built `Move`. A provider's error enum is
                    // fieldless, so it is structurally `Copy` and moving it contradicts its type.
                    // The slot reads below already went through `read_place`; this one did not,
                    // which is the whole shape of the defect — an operand chosen by the site rather
                    // than by the type.
                    let payload = self.read_place(Place::local(raw), &error_ty, span)?;
                    self.emit(
                        Statement::Assign(
                            dest.clone(),
                            Rvalue::Aggregate(
                                AggKind::EnumVariant(EnumRef::CoreResult, 1),
                                vec![payload],
                            ),
                        ),
                        self.info(span),
                    );
                }
                None => self.assign_provider_ok(dest.clone(), &slots, span)?,
            }
            let next = to_fill.get(i + 1).map_or(join, |(b, _)| *b);
            self.terminate(Terminator::Goto { target: join }, self.info(span), next);
        }

        Ok(())
    }

    /// `dest = Ok(<out-slots>)`: unit for none, the value for one, a tuple for several.
    fn assign_provider_ok(
        &mut self,
        dest: Place,
        slots: &[(LocalId, MirTy)],
        span: Span,
    ) -> Result<(), LowerError> {
        let payload = match slots.len() {
            0 => vec![Operand::Const(Constant::Unit)],
            _ => slots
                .iter()
                .map(|(slot, ty)| self.read_place(Place::local(*slot), ty, span))
                .collect::<Result<Vec<_>, _>>()?,
        };
        let payload = if slots.len() > 1 {
            let tuple_ty = MirTy::Tuple(slots.iter().map(|(_, t)| t.clone()).collect());
            let tmp = self.new_temp(tuple_ty.clone());
            self.emit(
                Statement::Assign(
                    Place::local(tmp),
                    Rvalue::Aggregate(AggKind::Tuple, payload),
                ),
                self.info(span),
            );
            // DEV-125, the same defect one line apart: a tuple of scalar out-slots — `(Bool,
            // UInt64)` for `var_len` — is `Copy` because its elements are.
            vec![self.read_place(Place::local(tmp), &tuple_ty, span)?]
        } else {
            payload
        };
        self.emit(
            Statement::Assign(
                dest,
                Rvalue::Aggregate(AggKind::EnumVariant(EnumRef::CoreResult, 0), payload),
            ),
            self.info(span),
        );
        Ok(())
    }

    /// The `ItemId` of a top-level nominal by source name — resolves a synthesized raw error enum,
    /// whose name lowering knows from the manifest but whose id only the parser assigned.
    fn nominal_item_by_name(&self, name: &str) -> Option<ItemId> {
        self.meta
            .all_items
            .iter()
            .copied()
            .find(|&item| match &self.hir.item(item).kind {
                ItemKind::Enum { name: n, .. } | ItemKind::Struct { name: n, .. } => {
                    self.meta.item_text(item, *n) == name
                }
                _ => false,
            })
    }

    fn new_temp(&mut self, ty: MirTy) -> LocalId {
        self.locals.push(LocalDecl {
            ty,
            kind: LocalKind::Temp,
        });
        LocalId((self.locals.len() - 1) as u32)
    }

    // ---- types ----

    fn mir_ty(&self, ty: &Ty, span: Span) -> Result<MirTy, LowerError> {
        Ok(match ty {
            Ty::Primitive(p) => match p {
                Primitive::Int8 => MirTy::Int8,
                Primitive::Int16 => MirTy::Int16,
                Primitive::Int32 => MirTy::Int32,
                Primitive::Int64 => MirTy::Int64,
                Primitive::UInt8 => MirTy::UInt8,
                Primitive::UInt16 => MirTy::UInt16,
                Primitive::UInt32 => MirTy::UInt32,
                Primitive::UInt64 => MirTy::UInt64,
                Primitive::Float32 => MirTy::Float32,
                Primitive::Float64 => MirTy::Float64,
                Primitive::Bool => MirTy::Bool,
                Primitive::Unit => MirTy::Unit,
                // f-3b: Char (a Unicode scalar codepoint value).
                Primitive::Char => MirTy::Char,
                // A1 (CD-031): first-class text types.
                Primitive::String => MirTy::String,
                Primitive::Str => MirTy::Str,
                _ => return unsupported(format!("type {p:?} (C4.5)"), span),
            },
            Ty::Struct(item, args) => MirTy::Struct(
                *item,
                args.iter()
                    .map(|a| self.mir_ty(a, span))
                    .collect::<Result<Vec<_>, _>>()?,
            ),
            // **CD-234: a provider-bound nominal is its HostResource here too.** This is the main
            // typed-expression path, and the one that decided `Result<TcpStream, E>`'s payload --
            // so leaving it produced the shell wherever the checker's type flowed and the resource
            // form wherever a provider signature did, failing to unify at every boundary between.
            Ty::Enum(item, _) if self.providers.nominal_types.contains_key(&item.0) => {
                self.providers.nominal_types[&item.0].clone()
            }
            Ty::Enum(item, args) => MirTy::Enum(
                EnumRef::User(*item),
                args.iter()
                    .map(|a| self.mir_ty(a, span))
                    .collect::<Result<Vec<_>, _>>()?,
            ),
            Ty::Core(crate::hir::CoreType::Option, args) => {
                let inner = args
                    .iter()
                    .map(|a| self.mir_ty(a, span))
                    .collect::<Result<Vec<_>, _>>()?;
                MirTy::Enum(EnumRef::CoreOption, inner)
            }
            Ty::Core(crate::hir::CoreType::Result, args) => {
                let inner = args
                    .iter()
                    .map(|a| self.mir_ty(a, span))
                    .collect::<Result<Vec<_>, _>>()?;
                MirTy::Enum(EnumRef::CoreResult, inner)
            }
            // A2 (CE3): the prelude `Ordering` enum as a logical MIR enum, fieldless.
            Ty::Core(crate::hir::CoreType::Ordering, _) => {
                MirTy::Enum(EnumRef::CoreOrdering, Vec::new())
            }
            // A1 (CD-031), C4.5e-2: Vec<T> is an opaque runtime type.
            Ty::Core(crate::hir::CoreType::Vec, args) => {
                let inner = args
                    .iter()
                    .map(|a| self.mir_ty(a, span))
                    .collect::<Result<Vec<_>, _>>()?;
                MirTy::Core(crate::hir::CoreType::Vec, inner)
            }
            // 0.1-A7 (WP-C4.7-6.1): `Box<T>` is an OPAQUE OWNING runtime type. Deliberately NOT
            // lowered transparently as `T`: a transparent Box would make recursive types through
            // `Box` (e.g. `struct Node { next: Option<Box<Node>> }`) infinitely sized.
            Ty::Core(crate::hir::CoreType::Box, args) => {
                let inner = args
                    .iter()
                    .map(|a| self.mir_ty(a, span))
                    .collect::<Result<Vec<_>, _>>()?;
                MirTy::Core(crate::hir::CoreType::Box, inner)
            }
            // 0.1-A2 (C4.5f-2): the borrowing Vec iterator.
            Ty::Core(crate::hir::CoreType::VecIter, args) => {
                let inner = args
                    .iter()
                    .map(|a| self.mir_ty(a, span))
                    .collect::<Result<Vec<_>, _>>()?;
                MirTy::Core(crate::hir::CoreType::VecIter, inner)
            }
            // 0.1-A5 (A4-2d): the string chars iterator (no type args).
            Ty::Core(crate::hir::CoreType::CharsIter, _) => {
                MirTy::Core(crate::hir::CoreType::CharsIter, Vec::new())
            }
            // 0.1-A3 (f-3a): HashMap and its keys iterator.
            Ty::Core(crate::hir::CoreType::HashMap, args) => {
                let inner = args
                    .iter()
                    .map(|a| self.mir_ty(a, span))
                    .collect::<Result<Vec<_>, _>>()?;
                MirTy::Core(crate::hir::CoreType::HashMap, inner)
            }
            // DEV-116: `HashSet<T>`. V19 is normative in `std-full` and the HIR oracle runs it;
            // the refusal here was a MIR gap, which §4.3 forbids recording as a non-Core exclusion.
            Ty::Core(crate::hir::CoreType::HashSet, args) => {
                let inner = args
                    .iter()
                    .map(|a| self.mir_ty(a, span))
                    .collect::<Result<Vec<_>, _>>()?;
                MirTy::Core(crate::hir::CoreType::HashSet, inner)
            }
            // DEV-116-B: `Iter<T>` is `HashSet::iter`'s cursor. The TWO-argument form is
            // `HashMap::iter`, which is a different (unimplemented) surface, so only the
            // single-argument form is admitted and the other keeps its refusal.
            Ty::Core(crate::hir::CoreType::Iter, args) if args.len() == 1 => {
                let inner = args
                    .iter()
                    .map(|a| self.mir_ty(a, span))
                    .collect::<Result<Vec<_>, _>>()?;
                MirTy::Core(crate::hir::CoreType::Iter, inner)
            }
            Ty::Core(crate::hir::CoreType::KeysIter, args) => {
                let inner = args
                    .iter()
                    .map(|a| self.mir_ty(a, span))
                    .collect::<Result<Vec<_>, _>>()?;
                MirTy::Core(crate::hir::CoreType::KeysIter, inner)
            }
            // DEV-151: the EMPTY tuple is `MirTy::Unit`, not `MirTy::Tuple(vec![])`. MIR has one
            // canonical unit type and every synthesized site uses it; only a written-out `()` in
            // source reached here, so `fn f() -> Result<(), E>` produced a return type that no
            // constructed value could ever match. It stayed invisible while nothing lowered such a
            // body — DEV-151's method-dispatch repair is what first lowered one.
            Ty::Tuple(elems) if elems.is_empty() => MirTy::Unit,
            Ty::Tuple(elems) => MirTy::Tuple(
                elems
                    .iter()
                    .map(|e| self.mir_ty(e, span))
                    .collect::<Result<Vec<_>, _>>()?,
            ),
            Ty::Array(elem, len) => MirTy::Array(Box::new(self.mir_ty(elem, span)?), *len),
            // 0.1-A6 (A4 slicing): unsized slice type — appears only behind Ref.
            Ty::Slice(elem) => MirTy::Slice(Box::new(self.mir_ty(elem, span)?)),
            // A4: a `Range<T>` value is represented as the tuple `(start, end, inclusive)`. The
            // iteration site distinguishes it from a genuine 3-tuple by the iter's front-end
            // type (`Ty::Range`), so no nominal MIR identity is needed.
            Ty::Range(elem) => {
                let e = self.mir_ty(elem, span)?;
                MirTy::Tuple(vec![e.clone(), e, MirTy::Bool])
            }
            Ty::Fn { params, ret } => MirTy::FnPtr {
                params: params
                    .iter()
                    .map(|p| self.mir_ty(p, span))
                    .collect::<Result<Vec<_>, _>>()?,
                ret: Box::new(self.mir_ty(ret, span)?),
            },
            Ty::Never => MirTy::Never,
            // C4.5b-2: real reference types (the interim by-value peel is gone).
            Ty::Ref { mutable, inner } => MirTy::Ref {
                mutable: *mutable,
                inner: Box::new(self.mir_ty(inner, span)?),
            },
            // C4.5c: the body's own generic parameters resolve through the instance's
            // type arguments; `Self` through the receiver type as before.
            Ty::Param(name) => match self.param_subst.get(name) {
                Some(concrete) => concrete.clone(),
                None if name == "Self" => match &self.self_subst {
                    Some(self_ty) => self_ty.clone(),
                    None => return unsupported("Self outside a method body", span),
                },
                // WP-C6.2c: a projection `T::Item` monomorphises by resolving the base parameter to
                // its concrete nominal, then the associated type through that nominal's impl binding.
                None if name.contains("::") => match self.resolve_projection_mir_ty(name, span) {
                    Some(ty) => ty,
                    None => {
                        return unsupported(
                            format!("unbound generic parameter {name} (C4.5)"),
                            span,
                        )
                    }
                },
                None => {
                    return unsupported(format!("unbound generic parameter {name} (C4.5)"), span)
                }
            },
            _ => return unsupported(format!("type {ty:?} (C4.5)"), span),
        })
    }

    /// WP-C6.2c: resolve a projection parameter name (`"T::Item"` or `"Self::Item"`) to a concrete
    /// MirTy. The base resolves through the active substitution to a nominal, and the associated
    /// type through that nominal's impl binding (`ProgramMeta::assoc_projections`). Returns `None`
    /// when the base is not a known nominal or the binding is absent.
    fn resolve_projection_mir_ty(&self, name: &str, span: Span) -> Option<MirTy> {
        let (base, assoc) = name.split_once("::")?;
        let base_ty = if base == "Self" {
            self.self_subst.clone()?
        } else {
            self.param_subst.get(base).cloned()?
        };
        let nominal = match base_ty {
            MirTy::Struct(item, _) => item.0,
            MirTy::Enum(EnumRef::User(item), _) => item.0,
            _ => return None,
        };
        let binding = *self
            .meta
            .assoc_projections
            .get(&(nominal, assoc.to_string()))?;
        self.hir_field_ty(binding).ok().or_else(|| {
            let _ = span;
            None
        })
    }

    /// Convert an HIR type node (struct field / enum payload declarations) to a MirTy.
    fn hir_field_ty(&self, ty_id: hir::TypeId) -> Result<MirTy, LowerError> {
        let node = self.hir.ty(ty_id);
        let span = node.span;
        match &node.kind {
            hir::TypeKind::Primitive(p) => self.mir_ty(&Ty::Primitive(*p), span),
            hir::TypeKind::Path { res, args, .. } => match res {
                Res::Item(item) => {
                    let converted_args = match args {
                        Some(list) => list
                            .args
                            .iter()
                            .map(|a| match a {
                                hir::GenericArg::Type(t) => self.hir_field_ty(*t),
                                _ => unsupported("field type argument (C4.5)", span),
                            })
                            .collect::<Result<Vec<_>, _>>()?,
                        None => Vec::new(),
                    };
                    match &self.hir.item(*item).kind {
                        ItemKind::Struct { .. } => Ok(MirTy::Struct(*item, converted_args)),
                        // **CD-234: a provider-bound nominal is a HostResource, not its shell.**
                        // The zero-variant enum exists so the source form is opaque; the moment it
                        // has a binding, MIR must see the resource. Without this the same type is
                        // `Enum(User(id))` here and `HostResource` in a provider signature, and every
                        // boundary between them fails to unify (MIR-0004/MIR-0005).
                        ItemKind::Enum { .. }
                            if self.providers.nominal_types.contains_key(&item.0) =>
                        {
                            Ok(self.providers.nominal_types[&item.0].clone())
                        }
                        ItemKind::Enum { .. } => {
                            Ok(MirTy::Enum(EnumRef::User(*item), converted_args))
                        }
                        // Naming the kind: a bare "field type form" cannot distinguish an
                        // unsupported alias from a path that resolved somewhere unexpected, and
                        // bisecting the difference by hand costs real time.
                        other => unsupported(
                            format!(
                                "field type form (C4.5): {}",
                                match other {
                                    ItemKind::Fn(_) => "fn",
                                    ItemKind::Trait { .. } => "trait",
                                    ItemKind::Impl { .. } => "impl",
                                    ItemKind::Const { .. } => "const",
                                    ItemKind::TypeAlias { .. } => "type alias",
                                    ItemKind::Mod { .. } => "mod",
                                    ItemKind::Use(_) => "use",
                                    _ => "other",
                                }
                            ),
                            span,
                        ),
                    }
                }
                // C4.5c: a generic parameter in a field/signature position resolves through
                // the active substitution (nominal-instantiation registration or a
                // monomorphised body's own parameters).
                Res::TypeParam => {
                    let name = self.text(span);
                    match self.param_subst.get(name) {
                        Some(concrete) => Ok(concrete.clone()),
                        None => unsupported(
                            format!("unbound generic parameter {name} in field type (C4.5)"),
                            span,
                        ),
                    }
                }
                Res::CoreType(core) => {
                    let inner = match args {
                        Some(list) => list
                            .args
                            .iter()
                            .map(|a| match a {
                                hir::GenericArg::Type(t) => self.hir_field_ty(*t),
                                _ => unsupported("field type argument (C4.5)", span),
                            })
                            .collect::<Result<Vec<_>, _>>()?,
                        None => Vec::new(),
                    };
                    match core {
                        crate::hir::CoreType::Option => Ok(MirTy::Enum(EnumRef::CoreOption, inner)),
                        crate::hir::CoreType::Result => Ok(MirTy::Enum(EnumRef::CoreResult, inner)),
                        crate::hir::CoreType::Ordering => {
                            Ok(MirTy::Enum(EnumRef::CoreOrdering, Vec::new()))
                        }
                        // A1: runtime container types in signature/field position.
                        crate::hir::CoreType::String => Ok(MirTy::String),
                        // 0.1-A7: `Box<T>` in FIELD position is what makes a recursive type
                        // finitely representable (`struct Node { next: Option<Box<Node>> }`) —
                        // the box is an opaque owning handle, so the field's size does not
                        // depend on `Node`'s.
                        crate::hir::CoreType::Box
                        | crate::hir::CoreType::Vec
                        | crate::hir::CoreType::HashMap
                        | crate::hir::CoreType::HashSet
                        | crate::hir::CoreType::VecIter
                        | crate::hir::CoreType::KeysIter
                        | crate::hir::CoreType::Iter
                        | crate::hir::CoreType::CharsIter => Ok(MirTy::Core(*core, inner)),
                        _ => unsupported("core field type (C4.5)", span),
                    }
                }
                Res::SelfType => match &self.self_subst {
                    Some(self_ty) => Ok(self_ty.clone()),
                    None => unsupported("Self type outside a method context", span),
                },
                _ => unsupported("field type path (C4.5)", span),
            },
            hir::TypeKind::Ref { mutable, inner } => Ok(MirTy::Ref {
                mutable: *mutable,
                inner: Box::new(self.hir_field_ty(*inner)?),
            }),
            // DEV-099 (WP-C5.3e): an ARRAY in a type position. Previously this fell through to
            // "field type form (C4.5)", so `size_of::<[Int32; 4]>()` failed to lower even though
            // arrays are inside the C5.3a aggregate subset and the layout exit matrix requires
            // fixed-array coverage.
            //
            // The length is read the same way `typecheck::convert_hir_type` reads it — HIR stores
            // only the length's SPAN, so both sides parse the literal text. The checker falls back
            // to 0 on a non-literal length; refusing here instead is deliberate, because a wrong
            // length silently changes an observable layout answer whereas a refusal is
            // deterministic and visible.
            hir::TypeKind::Array { elem, len } => {
                let text = self.text(*len);
                let Ok(count) = text.parse::<u64>() else {
                    return unsupported(
                        format!("array length `{text}` is not a literal count (C4.5)"),
                        span,
                    );
                };
                Ok(MirTy::Array(Box::new(self.hir_field_ty(*elem)?), count))
            }
            // DEV-151: same canonicalisation as `mir_ty` — an empty tuple is `Unit`.
            hir::TypeKind::Tuple(elems) if elems.is_empty() => Ok(MirTy::Unit),
            hir::TypeKind::Tuple(elems) => Ok(MirTy::Tuple(
                elems
                    .iter()
                    .map(|e| self.hir_field_ty(*e))
                    .collect::<Result<Vec<_>, _>>()?,
            )),
            hir::TypeKind::Fn { params, ret } => Ok(MirTy::FnPtr {
                params: params
                    .iter()
                    .map(|p| self.hir_field_ty(*p))
                    .collect::<Result<Vec<_>, _>>()?,
                ret: Box::new(match ret {
                    Some(r) => self.hir_field_ty(*r)?,
                    None => MirTy::Unit,
                }),
            }),
            // DEV-153: an unsized SLICE in a type position. `mir_ty` has had this arm all along;
            // this function did not, because it only ever saw struct fields and enum payloads —
            // and Core v1 forbids reference-typed fields, so `&[T]` could not reach it.
            //
            // DEV-151(a) changed that. Opening method dispatch on a host-resource receiver routed
            // a method's DECLARED parameter types through here for the first time, and
            // `fn write_all(&mut self, input: &[UInt8])` refused to lower — while the identical
            // free function `write_all(&mut stream, input)` built fine. A repair widening what can
            // be reached will expose whatever the newly reachable path never handled; that is the
            // cost of the DEV-151 class, not an argument against fixing it.
            hir::TypeKind::Slice(elem) => Ok(MirTy::Slice(Box::new(self.hir_field_ty(*elem)?))),
            _ => unsupported("field type form (C4.5)", span),
        }
    }

    fn expr_mir_ty(&self, expr: ExprId) -> Result<MirTy, LowerError> {
        let span = self.hir.expr(expr).span;
        let ty = self
            .tables
            .expr_types
            .get(&expr)
            .cloned()
            .unwrap_or(Ty::Error);
        self.mir_ty(&ty, span)
    }

    /// Copy-vs-move for reads (contract §5), the PRODUCER side.
    ///
    /// **DEV-128: the rule itself is `mir::mir_ty_is_copy`; this supplies only the nominal set.**
    /// This was a full second copy of that match, byte-identical apart from reading
    /// `meta.copy_eligible` instead of `TypeContext::copy_eligible_items`. Two implementations of
    /// one rule meant every fix had to be applied twice, and repeatedly was not: `HostResource` was
    /// corrected five separate times across the family, CD-240 fixed one copy of the wildcard
    /// defect and left the other, and DEV-125/DEV-127 were operand decisions taken against this
    /// predicate and rejected by the other.
    ///
    /// The two SETS stay separate — they are read at different times, and `lower_program` fills the
    /// consumer's from this one. Only the rule is shared, which is the part that was drifting.
    fn is_copy(&self, ty: &MirTy) -> bool {
        crate::mir::mir_ty_is_copy(ty, &|item| self.meta.copy_eligible.contains(&item))
    }

    /// Read a place as an operand. C4.5d: a `Move` out of a drop-tracked local clears the
    /// flags of every unit the moved place covers, so the value's drop responsibility
    /// transfers with it (scope-exit drops are flag-guarded and skip it).
    fn read_place(&mut self, place: Place, ty: &MirTy, span: Span) -> Result<Operand, LowerError> {
        if self.is_copy(ty) {
            return Ok(Operand::Copy(place));
        }
        if let Some(units) = self.drop_info.get(&place.local.0) {
            let mut prefix: Vec<DropStep> = Vec::new();
            for proj in &place.projection {
                match proj {
                    Projection::Field(i) => prefix.push(DropStep::Field(*i)),
                    // A5 (CD-038): a statically known array element IS a nameable sub-place, so
                    // moving one out clears exactly its own flag and leaves its siblings alive.
                    // That is what makes by-value array iteration and consuming array patterns
                    // representable at all.
                    Projection::ConstIndex(i) => prefix.push(DropStep::ConstIndex(*i)),
                    _ => {
                        return unsupported(
                            "move through a non-field projection of a drop-tracked local (C4.5)",
                            span,
                        )
                    }
                }
            }
            // A place strictly inside a unit is inside a Drop-implementing value (or an
            // enum/array unit): moving out of it is not legal Core; defense in depth here.
            if units
                .iter()
                .any(|u| u.path.len() < prefix.len() && prefix[..u.path.len()] == u.path[..])
            {
                return unsupported("move out of a value whose type implements Drop", span);
            }
            self.set_flags_under(place.local.0, &prefix, false, span);
        }
        Ok(Operand::Move(place))
    }

    // ---- drop elaboration (C4.5d) ----

    /// Does a value of `ty` require drop glue (its own or any transitive `Drop` impl)?
    fn ty_needs_drop(&self, ty: &MirTy, span: Span) -> Result<bool, LowerError> {
        Ok(match ty {
            MirTy::Struct(item, args) => {
                // A1: a Drop impl on a generic nominal drops per instantiation — the dtor
                // instance is monomorphised at the same type arguments.
                if self.type_has_drop_impl(*item) {
                    true
                } else {
                    let fields = nominal_instance_fields(
                        self.hir,
                        self.tables,
                        self.meta,
                        *item,
                        args,
                        self.providers,
                    )?;
                    let NominalFields::Struct(tys) = fields else {
                        return unsupported("struct item with enum fields shape", span);
                    };
                    let mut any = false;
                    for t in &tys {
                        any = any || self.ty_needs_drop(t, span)?;
                    }
                    any
                }
            }
            MirTy::Enum(EnumRef::User(item), args) => {
                if self.type_has_drop_impl(*item) {
                    true
                } else {
                    let fields = nominal_instance_fields(
                        self.hir,
                        self.tables,
                        self.meta,
                        *item,
                        args,
                        self.providers,
                    )?;
                    let NominalFields::Enum(variants) = fields else {
                        return unsupported("enum item with struct fields shape", span);
                    };
                    let mut any = false;
                    for v in &variants {
                        for t in v {
                            any = any || self.ty_needs_drop(t, span)?;
                        }
                    }
                    any
                }
            }
            MirTy::Enum(_, args) => {
                let mut any = false;
                for t in args {
                    any = any || self.ty_needs_drop(t, span)?;
                }
                any
            }
            MirTy::Tuple(elems) => {
                let mut any = false;
                for t in elems {
                    any = any || self.ty_needs_drop(t, span)?;
                }
                any
            }
            MirTy::Array(elem, _) => self.ty_needs_drop(elem, span)?,
            // A1 (CD-031): String and Vec ALWAYS require runtime drop glue (buffer reclaim;
            // Vec also drops elements). Both are leaf drop units — `collect_drop_units`' `_`
            // arm makes them units, and the interp's `drop_in_place` reclaims/element-drops.
            // 0.1-A2: the iterator likewise (cursor/borrow release; T: Copy means no element
            // destructors — glue is observably a no-op).
            // 0.1-A7: a `Box<T>` always needs glue — it owns a heap allocation to release, and
            // drops the contained `T` exactly once first.
            MirTy::String
            | MirTy::Core(crate::hir::CoreType::Box, _)
            | MirTy::Core(crate::hir::CoreType::Vec, _)
            | MirTy::Core(crate::hir::CoreType::VecIter, _)
            | MirTy::Core(crate::hir::CoreType::HashMap, _)
            | MirTy::Core(crate::hir::CoreType::HashSet, _)
            | MirTy::Core(crate::hir::CoreType::KeysIter, _)
            | MirTy::Core(crate::hir::CoreType::Iter, _) => true,
            // **A11 §5: a host resource ALWAYS needs drop — its drop IS its provider close.**
            //
            // The FIFTH `MirTy` catch-all to swallow this variant, after `dump_ty`, `emit_ty`,
            // `default_value_expr`, `TypeContext::is_copy` and `FnLowerer::is_copy` -- and the most
            // consequential, because it silently disabled the whole close mechanism: no drop unit,
            // so no drop flag, so no `Drop` terminator, so no close. Every resource leaked, and
            // nothing complained. `c788_lifecycle_e2e` found it by observing the generated code
            // rather than trusting that the parts were wired together.
            //
            // Note that `Core(File, _)` is deliberately absent from this list too: SELECT-C keeps
            // `File` on the legacy path, where `c784_file_e2e` closes it through explicit MIR rather
            // than through drop elaboration.
            MirTy::HostResource(_) => true,

            // **EXHAUSTIVE ON PURPOSE — do not restore a wildcard here.**
            //
            // "Needs no drop glue" is indistinguishable from a leak, and this is the predicate the
            // wildcard cost the most: it is what silently disabled the close mechanism above.
            // `verify::may_need_drop` is the second copy of this rule and is hardened the same way.
            //
            // The `Core` arms are spelled out individually because the wildcard was hiding a real
            // asymmetry inside them: `VecIter`/`KeysIter`/`Iter` need glue while `CharsIter`,
            // `SplitIter`, `ValuesIter`, `MapIter` and `FilterIter` do not. That is preserved
            // exactly as it stood rather than harmonised here — whether the second group is right
            // is a question for whoever owns iterator lowering, and it is now written down instead
            // of being a side effect of arm ordering. `Core(File, _)` is false for the reason
            // given above (SELECT-C's legacy path), not by omission.
            MirTy::Core(
                crate::hir::CoreType::String
                | crate::hir::CoreType::Option
                | crate::hir::CoreType::Result
                | crate::hir::CoreType::Range
                | crate::hir::CoreType::RangeInclusive
                | crate::hir::CoreType::CharsIter
                | crate::hir::CoreType::SplitIter
                | crate::hir::CoreType::ValuesIter
                | crate::hir::CoreType::MapIter
                | crate::hir::CoreType::FilterIter
                | crate::hir::CoreType::Random
                | crate::hir::CoreType::IOError
                | crate::hir::CoreType::File
                | crate::hir::CoreType::Ordering,
                _,
            ) => false,
            // Scalars own nothing; `Str`/`Slice` are unsized and appear only behind a `Ref`, which
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
        })
    }

    /// Decompose a droppable type into drop units: descend through dtor-less structs and
    /// tuples; a type with its own `Drop` impl, an enum, or an array is one unit.
    fn collect_drop_units(
        &self,
        ty: &MirTy,
        path: &mut Vec<DropStep>,
        out: &mut Vec<(Vec<DropStep>, MirTy)>,
        span: Span,
    ) -> Result<(), LowerError> {
        if !self.ty_needs_drop(ty, span)? {
            return Ok(());
        }
        match ty {
            MirTy::Struct(item, args) if !self.type_has_drop_impl(*item) => {
                let fields = nominal_instance_fields(
                    self.hir,
                    self.tables,
                    self.meta,
                    *item,
                    args,
                    self.providers,
                )?;
                let NominalFields::Struct(tys) = fields else {
                    return unsupported("struct item with enum fields shape", span);
                };
                for (i, fty) in tys.iter().enumerate() {
                    path.push(DropStep::Field(i as u32));
                    self.collect_drop_units(fty, path, out, span)?;
                    path.pop();
                }
            }
            MirTy::Tuple(elems) => {
                for (i, ety) in elems.iter().enumerate() {
                    path.push(DropStep::Field(i as u32));
                    self.collect_drop_units(ety, path, out, span)?;
                    path.pop();
                }
            }
            // A5 (CD-038): a fixed-length array decomposes into PER-ELEMENT units, now that
            // `ConstIndex` can name one. Without this the array is a single unit, so moving one
            // element out (by-value iteration, or an array pattern) and then dropping the array
            // would destroy the moved-out element a second time.
            MirTy::Array(elem, len) => {
                let elem = (**elem).clone();
                for i in 0..*len {
                    path.push(DropStep::ConstIndex(i));
                    self.collect_drop_units(&elem, path, out, span)?;
                    path.pop();
                }
            }
            _ => out.push((path.clone(), ty.clone())),
        }
        Ok(())
    }

    /// Find `impl Drop for <item>`'s `drop` method, as a lowerable key + canonical symbol.
    fn drop_impl_key(
        &self,
        item: ItemId,
        type_args: &[MirTy],
    ) -> Result<Option<(FnKey, String)>, LowerError> {
        for (idx, candidate) in self.hir.items.iter().enumerate() {
            let ItemKind::Impl {
                trait_: Some(trait_ref),
                items,
                ..
            } = &candidate.kind
            else {
                continue;
            };
            if !matches!(trait_ref.res, Res::CoreTrait(crate::hir::CoreTrait::Drop)) {
                continue;
            }
            let impl_item = ItemId(idx as u32);
            if impl_self_item(self.hir, impl_item) != Some(item) {
                continue;
            }
            for (member, impl_member) in items.iter().enumerate() {
                let hir::ImplItem::Fn { def, .. } = impl_member else {
                    continue;
                };
                if self.meta.item_text(impl_item, def.sig.name) != "drop" {
                    continue;
                }
                let key = FnKey::ImplFn {
                    impl_item,
                    member: member as u32,
                    type_args: type_args.to_vec(),
                    // A `Drop::drop` never declares its own generics.
                    method_args: Vec::new(),
                };
                let symbol = key_symbol(self.hir, self.meta, &key)?;
                return Ok(Some((key, symbol)));
            }
        }
        Ok(None)
    }

    /// WP-C6.3e (CD-136): the place an AGGREGATE Display arm projects into, with any reference
    /// layers peeled off.
    ///
    /// CD-135 made an owning `Vec` element arrive as `&T`, which means an aggregate element —
    /// `Vec<(String, Int32)>`, `Vec<Option<String>>` — now reaches the tuple/array/`Option`/`Result`
    /// arms BEHIND a reference. Projecting `Field(0)`/`ConstIndex`/`Discriminant` straight onto that
    /// is ill-formed MIR (MIR-0003/0008/0010): the verifier rejects it, so the user saw a compiler
    /// internal error instead of either a render or a named refusal. Peeling restores the value
    /// place those projections require. (The `String`/`str`/`Vec`/user-nominal arms deliberately do
    /// NOT peel — they consume the reference itself.)
    fn deref_place(mut place: Place, layers: u32) -> Place {
        for _ in 0..layers {
            place.projection.push(Projection::Deref);
        }
        place
    }

    /// WP-C6.3e: a `&Vec<T>` operand for the Display renderer, whether `place` holds the `Vec`
    /// itself or ALREADY holds a reference to one. The recursive case makes the difference real: a
    /// `Vec<Vec<T>>` element arrives as `&Vec<T>`, and borrowing that again would build `&&Vec<T>`,
    /// which the verifier rejects (MIR-0004).
    fn vec_ref_for_display(
        &mut self,
        place: &Place,
        ty: &MirTy,
        layers: u32,
        ref_ty: &MirTy,
        span: Span,
    ) -> Result<Operand, LowerError> {
        if layers > 0 {
            return self.read_place(place.clone(), ty, span);
        }
        let temp = self.new_temp(ref_ty.clone());
        self.emit(
            Statement::Assign(
                Place::local(temp),
                Rvalue::RefOf {
                    mutable: false,
                    place: place.clone(),
                },
            ),
            self.info(span),
        );
        Ok(Operand::Copy(Place::local(temp)))
    }

    /// WP-C6.3d (STD-HASH-001): the selected `Eq::eq` instance for a nominal used as a map KEY.
    /// Structurally identical to [`Self::drop_impl_key`] — same impl scan, same `FnKey`/symbol
    /// construction — differing only in the trait and member name it looks for.
    fn eq_impl_key(
        &self,
        item: ItemId,
        type_args: &[MirTy],
    ) -> Result<Option<(FnKey, String)>, LowerError> {
        for (idx, candidate) in self.hir.items.iter().enumerate() {
            let ItemKind::Impl {
                trait_: Some(trait_ref),
                items,
                ..
            } = &candidate.kind
            else {
                continue;
            };
            if !matches!(trait_ref.res, Res::CoreTrait(crate::hir::CoreTrait::Eq)) {
                continue;
            }
            let impl_item = ItemId(idx as u32);
            if impl_self_item(self.hir, impl_item) != Some(item) {
                continue;
            }
            for (member, impl_member) in items.iter().enumerate() {
                let hir::ImplItem::Fn { def, .. } = impl_member else {
                    continue;
                };
                if self.meta.item_text(impl_item, def.sig.name) != "eq" {
                    continue;
                }
                let key = FnKey::ImplFn {
                    impl_item,
                    member: member as u32,
                    type_args: type_args.to_vec(),
                    // An `Eq::eq` never declares its own generics.
                    method_args: Vec::new(),
                };
                let symbol = key_symbol(self.hir, self.meta, &key)?;
                return Ok(Some((key, symbol)));
            }
        }
        Ok(None)
    }

    /// WP-C6.3d: record the `Eq` instance a map KEY type dispatches identity through, and queue its
    /// body for lowering. A primitive/`String` key needs no entry: it has no user impl, and its
    /// structural comparison IS its lawful `Eq`.
    fn discover_eq_impl(&mut self, key_ty: &MirTy) -> Result<(), LowerError> {
        let (MirTy::Struct(item, args) | MirTy::Enum(EnumRef::User(item), args)) = key_ty else {
            return Ok(());
        };
        let (item, args) = (*item, args.clone());
        if self.eq_impl_symbols.contains_key(&(item.0, args.clone())) {
            return Ok(());
        }
        if let Some((key, symbol)) = self.eq_impl_key(item, &args)? {
            self.eq_impl_symbols.insert((item.0, args.clone()), symbol);
            self.discovered_callees.push(key);
        }
        Ok(())
    }

    /// Discover every dtor instance `ty`'s drop glue can invoke: record its symbol for the
    /// type context and queue its body for lowering.
    fn discover_drop_impls(&mut self, ty: &MirTy) -> Result<(), LowerError> {
        let mut visited = std::collections::BTreeSet::new();
        self.discover_drop_impls_guarded(ty, &mut visited)
    }

    /// 0.1-A7: the walk needs a cycle guard because `Box<T>` makes types RECURSIVE. Without it,
    /// `struct Node { next: Option<Box<Node>> }` walks Node → Option<Box<Node>> → Box<Node> →
    /// Node forever and overflows the stack (observed while adding the Box surface). The guard is
    /// on the type rather than a depth limit: a type's dtor instances only need discovering once,
    /// so revisiting is pure waste even where it would terminate.
    fn discover_drop_impls_guarded(
        &mut self,
        ty: &MirTy,
        visited: &mut std::collections::BTreeSet<MirTy>,
    ) -> Result<(), LowerError> {
        if !visited.insert(ty.clone()) {
            return Ok(());
        }
        match ty {
            MirTy::Struct(item, args) | MirTy::Enum(EnumRef::User(item), args) => {
                let (item, args) = (*item, args.clone());
                if !self.drop_impl_symbols.contains_key(&(item.0, args.clone())) {
                    if let Some((key, symbol)) = self.drop_impl_key(item, &args)? {
                        self.drop_impl_symbols
                            .insert((item.0, args.clone()), symbol);
                        self.discovered_callees.push(key);
                    }
                }
                match nominal_instance_fields(
                    self.hir,
                    self.tables,
                    self.meta,
                    item,
                    &args,
                    self.providers,
                )? {
                    NominalFields::Struct(tys) => {
                        for t in &tys {
                            self.discover_drop_impls_guarded(t, visited)?;
                        }
                    }
                    NominalFields::Enum(variants) => {
                        for v in &variants {
                            for t in v {
                                self.discover_drop_impls_guarded(t, visited)?;
                            }
                        }
                    }
                }
            }
            MirTy::Enum(_, args) | MirTy::Tuple(args) => {
                for t in args.clone() {
                    self.discover_drop_impls_guarded(&t, visited)?;
                }
            }
            MirTy::Array(elem, _) => self.discover_drop_impls_guarded(&elem.clone(), visited)?,
            // 0.1-A7: descend into runtime containers' element types. A `Box<Tag>`'s drop glue
            // runs `Tag`'s destructor, so `Tag`'s dtor instance must be discovered and lowered —
            // without this the `Drop` terminator fires and silently finds no dtor registered.
            // Applies to every `Core` container uniformly (Vec's elements reach the same glue).
            MirTy::Core(_, args) => {
                for t in args.clone() {
                    self.discover_drop_impls_guarded(&t, visited)?;
                }
            }
            _ => {}
        }
        Ok(())
    }

    /// Register a droppable local: create per-unit flags initialized to `init`, record it in
    /// `drop_info` and the current scope, and discover glue's dtor instances. No-op for
    /// non-droppable types.
    fn register_droppable_local(
        &mut self,
        mir_local: LocalId,
        ty: &MirTy,
        init: bool,
        span: Span,
    ) -> Result<(), LowerError> {
        if !self.ty_needs_drop(ty, span)? {
            return Ok(());
        }
        self.discover_drop_impls(ty)?;
        let mut raw = Vec::new();
        self.collect_drop_units(ty, &mut Vec::new(), &mut raw, span)?;
        let mut units = Vec::new();
        for (path, uty) in raw {
            self.locals.push(LocalDecl {
                ty: MirTy::Bool,
                kind: LocalKind::DropFlag,
            });
            let flag = LocalId((self.locals.len() - 1) as u32);
            self.emit(
                Statement::Assign(
                    Place::local(flag),
                    Rvalue::Use(Operand::Const(Constant::Bool(init))),
                ),
                self.synthetic(span, SyntheticKind::DropFlagInit),
            );
            units.push(DropUnit {
                path,
                ty: uty,
                flag,
            });
        }
        self.drop_info.insert(mir_local.0, units);
        if let Some(scope) = self.scopes.last_mut() {
            scope.push(mir_local);
        }
        Ok(())
    }

    /// Emit flag assignments for every unit of `local` whose path starts with `prefix`.
    fn set_flags_under(&mut self, local: u32, prefix: &[DropStep], value: bool, span: Span) {
        let flags: Vec<LocalId> = match self.drop_info.get(&local) {
            Some(units) => units
                .iter()
                .filter(|u| u.path.len() >= prefix.len() && u.path[..prefix.len()] == *prefix)
                .map(|u| u.flag)
                .collect(),
            None => return,
        };
        for flag in flags {
            self.emit(
                Statement::Assign(
                    Place::local(flag),
                    Rvalue::Use(Operand::Const(Constant::Bool(value))),
                ),
                self.synthetic(span, SyntheticKind::DropFlagInit),
            );
        }
    }

    /// Emit `switch flag { true → Drop(place) }` for one unit of `local`.
    fn emit_guarded_drop(&mut self, local: u32, unit: &DropUnit, span: Span) {
        let info = self.synthetic(span, SyntheticKind::DropElaboration);
        let drop_block = self.new_block();
        let join = self.new_block();
        self.terminate(
            Terminator::SwitchInt {
                scrut: Operand::Copy(Place::local(unit.flag)),
                arms: vec![(1, drop_block)],
                otherwise: join,
            },
            info,
            drop_block,
        );
        let place = Place {
            local: LocalId(local),
            projection: unit.path.iter().map(DropStep::projection).collect(),
        };
        self.terminate(
            Terminator::Drop {
                place,
                target: join,
            },
            info,
            join,
        );
    }

    /// A12 / `DEFECT-C788-LOOP-TEMP`: every drop unit of `local` has now been accounted for, so
    /// its storage is dead.
    ///
    /// Emitted at the END of a local's drop sequence, unconditionally, because at that point the
    /// statement is true on **every** path into it and the operation is idempotent: the local was
    /// either dropped whole (storage already dead), moved out whole (already dead), or emptied unit
    /// by unit (partially moved, and this is the only thing that can finish it).
    ///
    /// That last case is the defect. A local whose field was moved out is left partially moved
    /// forever, which no straight-line program notices and which a loop back edge turns into an
    /// abort on the next iteration's assignment.
    fn emit_storage_dead(&mut self, local: u32, reason: StorageEnd, span: Span) {
        self.emit(
            Statement::StorageDead(Place::local(LocalId(local)), reason),
            self.synthetic(span, SyntheticKind::DropElaboration),
        );
    }

    /// Emit flag-guarded drops for every scope at `from_depth` or deeper — innermost scope
    /// first, locals in reverse declaration order, units in reverse order. Does not pop the
    /// scope stack (early exits leave the stack intact for the code that follows).
    ///
    /// A12: each local's units are followed by one storage end, so a local emptied unit by unit
    /// does not stay partially moved into the next iteration.
    fn emit_scope_drops_from(&mut self, from_depth: usize, span: Span) {
        let plan: Vec<(u32, Vec<DropUnit>)> = self.scopes[from_depth.min(self.scopes.len())..]
            .iter()
            .rev()
            .flat_map(|scope| scope.iter().rev())
            .filter_map(|local| {
                self.drop_info
                    .get(&local.0)
                    .map(|units| (local.0, units.iter().rev().cloned().collect()))
            })
            .collect();
        for (local, units) in plan {
            for unit in &units {
                self.emit_guarded_drop(local, unit, span);
            }
            self.emit_storage_dead(local, StorageEnd::Accounted, span);
        }
    }

    /// C4.5d: assignment with overwrite drops. Per the abstract machine (CD-012), the new
    /// value installs before the old is destroyed: any drop units the destination covers are
    /// saved into temporaries (guarded by their flags), the store happens, the saved old
    /// values drop (same guards, reverse order), and the flags flip true.
    fn lower_overwriting_assign(
        &mut self,
        place: Place,
        rhs_op: Operand,
        span: Span,
    ) -> Result<(), LowerError> {
        let covered: Vec<DropUnit> = match self.drop_info.get(&place.local.0) {
            Some(units) => {
                let mut prefix: Vec<DropStep> = Vec::new();
                let mut pure = true;
                for proj in &place.projection {
                    match proj {
                        Projection::Field(i) => prefix.push(DropStep::Field(*i)),
                        Projection::ConstIndex(i) => prefix.push(DropStep::ConstIndex(*i)),
                        _ => {
                            pure = false;
                            break;
                        }
                    }
                }
                if !pure {
                    return unsupported(
                        "assignment through a non-field projection of a drop-tracked local (C4.5)",
                        span,
                    );
                }
                units
                    .iter()
                    .filter(|u| {
                        u.path.len() >= prefix.len() && u.path[..prefix.len()] == prefix[..]
                    })
                    .cloned()
                    .collect()
            }
            None => Vec::new(),
        };
        if covered.is_empty() {
            self.emit(
                Statement::Assign(place, Rvalue::Use(rhs_op)),
                self.info(span),
            );
            return Ok(());
        }
        let info = self.synthetic(span, SyntheticKind::DropElaboration);
        // Save old unit values into temps, each guarded by its (still-old) flag.
        let mut saved: Vec<(DropUnit, LocalId)> = Vec::new();
        for unit in &covered {
            let tmp = self.new_temp(unit.ty.clone());
            let take_block = self.new_block();
            let join = self.new_block();
            self.terminate(
                Terminator::SwitchInt {
                    scrut: Operand::Copy(Place::local(unit.flag)),
                    arms: vec![(1, take_block)],
                    otherwise: join,
                },
                info,
                take_block,
            );
            let unit_place = Place {
                local: place.local,
                projection: unit.path.iter().map(DropStep::projection).collect(),
            };
            self.emit(
                Statement::Assign(Place::local(tmp), Rvalue::Use(Operand::Move(unit_place))),
                info,
            );
            self.terminate(Terminator::Goto { target: join }, info, join);
            saved.push((unit.clone(), tmp));
        }
        // Install the new value.
        self.emit(
            Statement::Assign(place.clone(), Rvalue::Use(rhs_op)),
            self.info(span),
        );
        // Destroy the saved old values (reverse order), guarded by the same flags.
        for (unit, tmp) in saved.iter().rev() {
            let drop_block = self.new_block();
            let join = self.new_block();
            self.terminate(
                Terminator::SwitchInt {
                    scrut: Operand::Copy(Place::local(unit.flag)),
                    arms: vec![(1, drop_block)],
                    otherwise: join,
                },
                info,
                drop_block,
            );
            self.terminate(
                Terminator::Drop {
                    place: Place::local(*tmp),
                    target: join,
                },
                info,
                join,
            );
        }
        // The destination is now initialized.
        for unit in &covered {
            self.emit(
                Statement::Assign(
                    Place::local(unit.flag),
                    Rvalue::Use(Operand::Const(Constant::Bool(true))),
                ),
                self.synthetic(span, SyntheticKind::DropFlagInit),
            );
        }

        // **DEV-158: the storage holds a complete value again — if every unit is live.**
        //
        // Step 1 above moved the covered units out, which makes a slot-backed local PARTIAL. Step 2
        // wrote them back. Nothing said so, and the next whole-value use aborted with
        // "mutable access to a dead slot: the slot is PARTIAL". The reference interpreter has no
        // slot model and accepted the same program, so this was a three-engine divergence: green
        // under `stark test`, aborting only in a native build, at runtime.
        //
        // **The condition is EVERY unit of the local, not the ones this assignment covered.** A
        // sibling unit may have been moved out earlier and never restored, in which case the
        // storage is legitimately still partial and must stay that way. Deriving wholeness from
        // coverage alone was tried and rejected: `RequestBuilder` has three droppable fields, so
        // `builder.body = bytes` covers one of three and the shortcut would have left exactly the
        // case that motivated this still broken, while looking like a fix.
        //
        // MIR already holds per-unit liveness as ordinary locals, so the guard is a conjunction of
        // the local's drop flags. That is the whole safety argument for `ValueSlot::mark_whole`,
        // whose precondition this discharges — the storage type cannot check it, and folding
        // per-unit liveness into it is the conflation the three-state design exists to prevent.
        let all_units: Vec<DropUnit> = match self.drop_info.get(&place.local.0) {
            Some(units) => units.clone(),
            None => Vec::new(),
        };
        if !all_units.is_empty() {
            let mark = self.new_block();
            let join = self.new_block();
            for (index, unit) in all_units.iter().enumerate() {
                // The last check falls through to `mark`; every earlier one to the next check. Any
                // false flag short-circuits to `join`, leaving the slot partial.
                let next = if index + 1 == all_units.len() {
                    mark
                } else {
                    self.new_block()
                };
                self.terminate(
                    Terminator::SwitchInt {
                        scrut: Operand::Copy(Place::local(unit.flag)),
                        arms: vec![(1, next)],
                        otherwise: join,
                    },
                    info,
                    next,
                );
            }
            self.emit(Statement::StorageWhole(Place::local(place.local)), info);
            self.terminate(Terminator::Goto { target: join }, info, join);
        }
        Ok(())
    }

    /// Drop a definitely-initialized temporary holding a discarded droppable value.
    fn emit_temp_drop(&mut self, temp: LocalId, span: Span) {
        let info = self.synthetic(span, SyntheticKind::DropElaboration);
        let join = self.new_block();
        self.terminate(
            Terminator::Drop {
                place: Place::local(temp),
                target: join,
            },
            info,
            join,
        );
    }

    fn type_has_copy_impl(&self, item: ItemId) -> bool {
        self.hir.items.iter().any(|candidate| {
            if let ItemKind::Impl {
                trait_: Some(trait_ref),
                self_ty,
                ..
            } = &candidate.kind
            {
                let is_copy = matches!(trait_ref.res, Res::CoreTrait(crate::hir::CoreTrait::Copy));
                let matches_item = matches!(
                    self.hir.ty(*self_ty).kind,
                    hir::TypeKind::Path { res: Res::Item(impl_item), .. } if impl_item == item
                );
                is_copy && matches_item
            } else {
                false
            }
        })
    }

    fn type_has_drop_impl(&self, item: ItemId) -> bool {
        self.hir.items.iter().any(|candidate| {
            if let ItemKind::Impl {
                trait_: Some(trait_ref),
                self_ty,
                ..
            } = &candidate.kind
            {
                let is_drop = matches!(trait_ref.res, Res::CoreTrait(crate::hir::CoreTrait::Drop));
                let matches_item = matches!(
                    self.hir.ty(*self_ty).kind,
                    hir::TypeKind::Path { res: Res::Item(impl_item), .. } if impl_item == item
                );
                is_drop && matches_item
            } else {
                false
            }
        })
    }

    // ---- function ----

    /// Concrete MirTy for a nominal item (struct or enum).
    fn nominal_ty(&self, item: ItemId, span: Span) -> Result<MirTy, LowerError> {
        // CD-234: a provider-bound nominal is its HostResource everywhere, not only in the type
        // path. Two sites construct the enum shell and BOTH must consult the binding -- patching one
        // left the variant-payload table disagreeing with the provider signature.
        if let Some(ty) = self.providers.nominal_types.get(&item.0) {
            return Ok(ty.clone());
        }
        match &self.hir.item(item).kind {
            ItemKind::Struct { .. } => Ok(MirTy::Struct(item, Vec::new())),
            ItemKind::Enum { .. } => Ok(MirTy::Enum(EnumRef::User(item), Vec::new())),
            _ => unsupported("nominal item is neither struct nor enum", span),
        }
    }

    /// Resolve this lowerer's `FnKey` to (signature, body block, receiver self-type).
    fn fn_parts(&self) -> Result<(&'a hir::FnSig, hir::BlockId, Option<MirTy>), LowerError> {
        let span0 = Span { lo: 0, hi: 0 };
        match &self.key {
            FnKey::Top(item, _) => match &self.hir.item(*item).kind {
                ItemKind::Fn(def) => Ok((&def.sig, def.body, None)),
                _ => unsupported("FnKey::Top on non-fn", span0),
            },
            FnKey::ImplFn {
                impl_item,
                member,
                type_args,
                ..
            } => {
                let ItemKind::Impl { items, .. } = &self.hir.item(*impl_item).kind else {
                    return unsupported("FnKey::ImplFn on non-impl", span0);
                };
                let hir::ImplItem::Fn { def, .. } = &items[*member as usize] else {
                    return unsupported("impl member is not a fn", span0);
                };
                let self_item = impl_self_item(self.hir, *impl_item).ok_or_else(|| LowerError {
                    what: "impl self type is not nominal".into(),
                    span: span0,
                })?;
                let self_ty = if type_args.is_empty() {
                    self.nominal_ty(self_item, span0)?
                } else {
                    // A1: the concrete instantiation of the impl's nominal.
                    match &self.hir.item(self_item).kind {
                        ItemKind::Struct { .. } => MirTy::Struct(self_item, type_args.clone()),
                        ItemKind::Enum { .. } => {
                            MirTy::Enum(EnumRef::User(self_item), type_args.clone())
                        }
                        _ => return unsupported("impl self type is not nominal", span0),
                    }
                };
                Ok((&def.sig, def.body, Some(self_ty)))
            }
            FnKey::TraitDefault {
                trait_item,
                member,
                self_item,
                self_args,
                ..
            } => {
                let ItemKind::Trait { items, .. } = &self.hir.item(*trait_item).kind else {
                    return unsupported("FnKey::TraitDefault on non-trait", span0);
                };
                let hir::TraitItem::Method {
                    sig,
                    body: Some(body),
                } = &items[*member as usize]
                else {
                    return unsupported("trait member has no default body", span0);
                };
                let self_ty = if self_args.is_empty() {
                    self.nominal_ty(*self_item, span0)?
                } else {
                    match &self.hir.item(*self_item).kind {
                        ItemKind::Struct { .. } => MirTy::Struct(*self_item, self_args.clone()),
                        ItemKind::Enum { .. } => {
                            MirTy::Enum(EnumRef::User(*self_item), self_args.clone())
                        }
                        _ => return unsupported("trait self type is not nominal", span0),
                    }
                };
                Ok((sig, *body, Some(self_ty)))
            }
        }
    }

    /// A1: infer a generic nominal's instantiation at an associated-fn CALL by unifying the
    /// fn's declared parameter/return types (written in impl generics) against the call's
    /// concrete argument/result types, then substituting the impl's written self arguments.
    fn infer_assoc_fn_instantiation(
        &self,
        impl_item: ItemId,
        member: u32,
        call_expr: ExprId,
        call_args: &[ExprId],
        span: Span,
    ) -> Result<Vec<MirTy>, LowerError> {
        let ItemKind::Impl { items, self_ty, .. } = &self.hir.item(impl_item).kind else {
            return unsupported("assoc-fn impl is not an impl", span);
        };
        let hir::ImplItem::Fn { def, .. } = &items[member as usize] else {
            return unsupported("assoc-fn member is not a fn", span);
        };
        let mut bound: std::collections::HashMap<String, MirTy> = Default::default();
        // Params against argument types, then the return against the call's result type.
        for (p, &arg) in def.sig.params.iter().zip(call_args) {
            let concrete = self.expr_mir_ty(arg)?;
            self.bind_written_ty(impl_item, p.ty, &concrete, &mut bound);
        }
        if let hir::RetTy::Ty(ret) = &def.sig.ret {
            let concrete = self.expr_mir_ty(call_expr)?;
            self.bind_written_ty(impl_item, *ret, &concrete, &mut bound);
        }
        // The impl's written self arguments (bare params) give the nominal's instantiation.
        let hir::TypeKind::Path {
            args: Some(written),
            ..
        } = &self.hir.ty(*self_ty).kind
        else {
            return unsupported("generic impl self type has no written arguments", span);
        };
        let mut out = Vec::new();
        for arg in &written.args {
            let hir::GenericArg::Type(t) = arg else {
                return unsupported("non-type impl self argument", span);
            };
            let name = self.meta.item_text(impl_item, self.hir.ty(*t).span);
            match bound.get(name) {
                Some(concrete) => out.push(concrete.clone()),
                None => {
                    return unsupported(
                        format!(
                            "cannot infer the instantiation of `{name}` for this associated-fn call"
                        ),
                        span,
                    )
                }
            }
        }
        Ok(out)
    }

    /// Structural one-way unification: walk the WRITTEN HIR type against a concrete MirTy and
    /// bind each generic-parameter name encountered. Mismatched shapes are ignored (the
    /// checker already validated the call).
    fn bind_written_ty(
        &self,
        impl_item: ItemId,
        written: hir::TypeId,
        concrete: &MirTy,
        bound: &mut std::collections::HashMap<String, MirTy>,
    ) {
        let node = self.hir.ty(written);
        match (&node.kind, concrete) {
            (
                hir::TypeKind::Path {
                    res: Res::TypeParam,
                    ..
                },
                _,
            ) => {
                let name = self.meta.item_text(impl_item, node.span).to_string();
                bound.entry(name).or_insert_with(|| concrete.clone());
            }
            (
                hir::TypeKind::Path {
                    args: Some(list), ..
                },
                MirTy::Struct(_, cargs)
                | MirTy::Enum(EnumRef::User(_), cargs)
                | MirTy::Enum(EnumRef::CoreOption, cargs)
                | MirTy::Enum(EnumRef::CoreResult, cargs)
                | MirTy::Core(_, cargs),
            ) => {
                for (w, c) in list.args.iter().zip(cargs) {
                    if let hir::GenericArg::Type(t) = w {
                        self.bind_written_ty(impl_item, *t, c, bound);
                    }
                }
            }
            (hir::TypeKind::Ref { inner, .. }, MirTy::Ref { inner: cinner, .. }) => {
                self.bind_written_ty(impl_item, *inner, cinner, bound);
            }
            (hir::TypeKind::Tuple(elems), MirTy::Tuple(celems)) => {
                for (w, c) in elems.iter().zip(celems) {
                    self.bind_written_ty(impl_item, *w, c, bound);
                }
            }
            _ => {}
        }
    }

    /// A1: the impl-generic substitution for an `ImplFn` instance — map each impl generic
    /// parameter to its concrete type by aligning the impl's WRITTEN self-type arguments
    /// (which must be bare parameter names, e.g. `impl<T> Holder<T>`) with the instantiation.
    fn impl_generic_subst(
        &self,
        impl_item: ItemId,
        type_args: &[MirTy],
    ) -> Result<Vec<(String, MirTy)>, LowerError> {
        let span0 = Span { lo: 0, hi: 0 };
        let ItemKind::Impl {
            generics, self_ty, ..
        } = &self.hir.item(impl_item).kind
        else {
            return unsupported("impl_generic_subst on non-impl", span0);
        };
        if generics.is_empty() {
            return Ok(Vec::new());
        }
        let hir::TypeKind::Path {
            args: Some(written),
            ..
        } = &self.hir.ty(*self_ty).kind
        else {
            return unsupported("generic impl self type has no written arguments", span0);
        };
        // WP-C4.7-8.5: align the impl's WRITTEN self arguments with the instantiation
        // STRUCTURALLY, so a non-bare head (`impl<T> Holder<Option<T>>` against
        // `Holder<Option<Int32>>`) binds `T := Int32` instead of being refused. This mirrors the
        // checker's `unify_impl_ty`; the two must agree about which impls apply, or the front end
        // would admit programs lowering then rejects.
        let written_args: Vec<hir::TypeId> = written
            .args
            .iter()
            .map(|arg| match arg {
                hir::GenericArg::Type(t) => Ok(*t),
                _ => unsupported("non-type impl self argument", span0),
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut subst: Vec<(String, MirTy)> = Vec::new();
        for (i, written_ty) in written_args.iter().enumerate() {
            let concrete = type_args.get(i).cloned().ok_or_else(|| LowerError {
                what: "impl instantiated with too few type arguments".into(),
                span: span0,
            })?;
            self.bind_written_impl_arg(impl_item, *written_ty, &concrete, &mut subst)?;
        }
        Ok(subst)
    }

    /// WP-C4.7-8.5: one-way structural match of an impl's WRITTEN self argument against the
    /// instantiation's concrete type, accumulating parameter bindings.
    ///
    /// A bare parameter binds directly. A constructed type (`Option<T>`, `Vec<T>`, `(T, U)`,
    /// `&T`) recurses into the matching position of the concrete type. Anything else is a
    /// concrete-vs-concrete position that carries no binding and needs no check — the checker
    /// already established that this impl applies, and lowering is only recovering the
    /// substitution it implies.
    fn bind_written_impl_arg(
        &self,
        impl_item: ItemId,
        written: hir::TypeId,
        concrete: &MirTy,
        subst: &mut Vec<(String, MirTy)>,
    ) -> Result<(), LowerError> {
        let node = self.hir.ty(written);
        let span = node.span;
        match &node.kind {
            hir::TypeKind::Path { res, args, .. } => {
                if matches!(res, Res::TypeParam) {
                    let name = self.meta.item_text(impl_item, span).to_string();
                    if !subst.iter().any(|(n, _)| *n == name) {
                        subst.push((name, concrete.clone()));
                    }
                    return Ok(());
                }
                let Some(list) = args else {
                    return Ok(());
                };
                // Descend into the concrete type's arguments positionally.
                let concrete_args: &[MirTy] = match concrete {
                    MirTy::Struct(_, a) | MirTy::Enum(_, a) | MirTy::Core(_, a) => a,
                    _ => return Ok(()),
                };
                for (i, arg) in list.args.iter().enumerate() {
                    let hir::GenericArg::Type(t) = arg else {
                        continue;
                    };
                    let Some(c) = concrete_args.get(i) else {
                        continue;
                    };
                    self.bind_written_impl_arg(impl_item, *t, c, subst)?;
                }
                Ok(())
            }
            hir::TypeKind::Ref { inner, .. } => match concrete {
                MirTy::Ref { inner: c, .. } => {
                    self.bind_written_impl_arg(impl_item, *inner, c, subst)
                }
                _ => Ok(()),
            },
            hir::TypeKind::Tuple(elems) => match concrete {
                MirTy::Tuple(cs) => {
                    for (t, c) in elems.iter().zip(cs) {
                        self.bind_written_impl_arg(impl_item, *t, c, subst)?;
                    }
                    Ok(())
                }
                _ => Ok(()),
            },
            _ => Ok(()),
        }
    }

    /// WP-C6.2a: the ONE place a `FnKey` becomes an `Instance`.
    ///
    /// `Instance` identity is `(item, type_args, symbol)`, and the frozen canonical-identity
    /// contract requires a reference to a callable to carry the SAME triple as the body that
    /// defines it. The defining item is the callable's definition — the function item for a
    /// top-level fn, the `impl` item for an implementation method, the trait item for an
    /// un-overridden default — never the receiver nominal, which is separate contextual
    /// information. Call sites previously passed the receiver nominal, so every method / trait /
    /// operator / associated-function call produced one canonical symbol with two item identities;
    /// the linkage preflight (correctly) refused them all. Routing BOTH `MirBody.instance` and
    /// every `FnKey`-derived `Callee::Instance` through this constructor removes the defect class
    /// rather than its individual manifestations.
    ///
    /// Note `ImplFn` carries impl-level `type_args` AND the method's own `method_args`; the
    /// instance's `type_args` are the IMPL's, matching what the body records (the method's own
    /// arguments are already folded into the canonical `symbol`).
    fn instance_from_key(&self, key: &FnKey) -> Result<Instance, LowerError> {
        let symbol = key_symbol(self.hir, self.meta, key)?;
        let (item, type_args) = match key {
            FnKey::Top(item, type_args) => (*item, type_args.clone()),
            FnKey::ImplFn {
                impl_item,
                type_args,
                ..
            } => (*impl_item, type_args.clone()),
            FnKey::TraitDefault {
                trait_item,
                self_args,
                ..
            } => (*trait_item, self_args.clone()),
        };
        Ok(Instance {
            item,
            type_args,
            symbol,
        })
    }

    fn lower_body(&mut self) -> Result<MirBody, LowerError> {
        let key = self.key.clone();
        let (sig, body_block, self_ty) = self.fn_parts()?;
        let sig_span = sig.span;
        // C4.5c: a generic top-level fn lowers once per concrete instantiation; the key's
        // type arguments substitute for the signature's own generic parameters throughout.
        if !sig.generics.is_empty() {
            match &key {
                FnKey::Top(_, type_args) if type_args.len() == sig.generics.len() => {
                    for (param, ty) in sig.generics.iter().zip(type_args.iter()) {
                        self.param_subst
                            .insert(self.text(param.name).to_string(), ty.clone());
                    }
                }
                FnKey::Top(..) => {
                    return unsupported(
                        "generic fn instantiated with the wrong number of type arguments",
                        sig_span,
                    );
                }
                // WP-C4.7-8.4: a method's OWN generic parameters substitute from the key's
                // `method_args`, which the CALL SITE supplied from the checker's per-call-site
                // recording. Impl-level parameters are bound separately below, from
                // `type_args` — a method on a generic nominal can be generic in both.
                FnKey::ImplFn { method_args, .. } if method_args.len() == sig.generics.len() => {
                    for (param, ty) in sig.generics.iter().zip(method_args.iter()) {
                        self.param_subst
                            .insert(self.text(param.name).to_string(), ty.clone());
                    }
                }
                FnKey::ImplFn { .. } => {
                    return unsupported(
                        "generic method instantiated with the wrong number of type arguments",
                        sig_span,
                    );
                }
                // WP-C4.7-9 audit: a trait DEFAULT method's own generic parameters substitute
                // from the key's `method_args`, exactly as an impl method's do.
                FnKey::TraitDefault { method_args, .. }
                    if method_args.len() == sig.generics.len() =>
                {
                    for (param, ty) in sig.generics.iter().zip(method_args.iter()) {
                        self.param_subst
                            .insert(self.text(param.name).to_string(), ty.clone());
                    }
                }
                FnKey::TraitDefault { .. } => {
                    return unsupported(
                        "generic trait-default method instantiated with the wrong number of type arguments",
                        sig_span,
                    );
                }
            }
        }
        // A1: an impl's generic parameters substitute from the instance's type arguments
        // (aligned through the impl's written self-type arguments).
        if let FnKey::ImplFn {
            impl_item,
            type_args,
            ..
        } = &key
        {
            if !type_args.is_empty() {
                for (name, ty) in self.impl_generic_subst(*impl_item, type_args)? {
                    self.param_subst.insert(name, ty);
                }
            }
        }
        self.self_subst = self_ty.clone();

        // Signature types: top fns use the checker's grounded fn_types; methods derive from
        // the HIR signature (concrete for impls; Self-substituted for trait defaults).
        let (params_no_recv, ret) = match (&key, self_ty.as_ref()) {
            (FnKey::Top(item, _), _) => {
                let (param_tys, ret_ty) = self
                    .tables
                    .fn_types
                    .get(item)
                    .cloned()
                    .unwrap_or((Vec::new(), Ty::Primitive(Primitive::Unit)));
                let ret = self.mir_ty(&ret_ty, sig_span)?;
                let params = param_tys
                    .iter()
                    .map(|t| self.mir_ty(t, sig_span))
                    .collect::<Result<Vec<_>, _>>()?;
                (params, ret)
            }
            _ => {
                let params = sig
                    .params
                    .iter()
                    .map(|p| self.hir_field_ty(p.ty))
                    .collect::<Result<Vec<_>, _>>()?;
                let ret = match &sig.ret {
                    hir::RetTy::Unit => MirTy::Unit,
                    hir::RetTy::Ty(t) => self.hir_field_ty(*t)?,
                    hir::RetTy::Never(_) => return unsupported("never-returning method", sig_span),
                };
                (params, ret)
            }
        };

        // Local 0 = return place; then the receiver (if any); then params.
        self.locals.push(LocalDecl {
            ty: ret.clone(),
            kind: LocalKind::Return,
        });
        let mut body_params: Vec<MirTy> = Vec::new();
        match (sig.receiver, self_ty.clone()) {
            (Some(receiver), Some(recv_self_ty)) => {
                // C4.5b-2: real receivers. `&self`/`&mut self` locals are Ref-typed; `self`
                // (by value) stays the plain type.
                let recv_ty = match receiver {
                    hir::Receiver::Ref => MirTy::Ref {
                        mutable: false,
                        inner: Box::new(recv_self_ty),
                    },
                    hir::Receiver::RefMut => MirTy::Ref {
                        mutable: true,
                        inner: Box::new(recv_self_ty),
                    },
                    hir::Receiver::Value => recv_self_ty,
                };
                self.locals.push(LocalDecl {
                    ty: recv_ty.clone(),
                    kind: LocalKind::Param(0),
                });
                if let Some(recv_local) = sig.receiver_local {
                    self.local_map
                        .insert(recv_local.0, LocalId((self.locals.len() - 1) as u32));
                }
                body_params.push(recv_ty);
            }
            (Some(_), None) => {
                return unsupported("receiver without a self type", sig_span);
            }
            (None, _) => {}
        }
        // C4.5d: the fn-level scope owns receiver/params — they are initialized by the
        // caller (flags start true) and drop at fn exit after the body's own scopes.
        self.scopes.push(Vec::new());
        if let (Some(hir::Receiver::Value), Some(recv_ty)) = (sig.receiver, self_ty.as_ref()) {
            if let Some(recv_local) = sig.receiver_local {
                let mir_local = *self.local_map.get(&recv_local.0).expect("receiver mapped");
                let recv_ty = recv_ty.clone();
                self.register_droppable_local(mir_local, &recv_ty, true, sig_span)?;
            }
        }
        for (param, ty) in sig.params.iter().zip(params_no_recv.iter()) {
            self.locals.push(LocalDecl {
                ty: ty.clone(),
                kind: LocalKind::Param(body_params.len() as u32),
            });
            let mir_local = LocalId((self.locals.len() - 1) as u32);
            self.local_map.insert(param.local.0, mir_local);
            let ty_cloned = ty.clone();
            self.register_droppable_local(mir_local, &ty_cloned, true, param.name)?;
            body_params.push(ty.clone());
        }
        let params = body_params;

        let body_span = self.hir.block(body_block).span;
        let tail = self.lower_block_value(body_block)?;
        if let Some(op) = tail {
            // C6.1f-b2: a tail expression is a return-position expected-type boundary, so
            // `fn g(m: &mut P) -> &P { m }` weakens rather than mismatching.
            let op = self.weaken_ref_to(op, &ret, body_span)?;
            self.emit(
                Statement::Assign(Place::local(LocalId(0)), Rvalue::Use(op)),
                self.synthetic(body_span, SyntheticKind::ReturnSlot),
            );
        } else if matches!(ret, MirTy::Unit) {
            self.emit(
                Statement::Assign(
                    Place::local(LocalId(0)),
                    Rvalue::Use(Operand::Const(Constant::Unit)),
                ),
                self.synthetic(body_span, SyntheticKind::ReturnSlot),
            );
        }
        // C4.5d: the fn-level (receiver/param) scope drops last, after the return value has
        // moved into Local(0).
        self.emit_scope_drops_from(self.scopes.len().saturating_sub(1), body_span);
        self.scopes.pop();
        let exit_info = self.synthetic(body_span, SyntheticKind::ReturnSlot);
        let after = self.new_block();
        self.terminate(Terminator::Return, exit_info, after);
        // Seal the trailing (unreachable) block.
        let final_info = self.synthetic(body_span, SyntheticKind::ReturnSlot);
        let dummy = self.new_block();
        self.terminate(Terminator::Unreachable, final_info, dummy);
        self.blocks.pop(); // drop the never-used dummy slot

        let blocks = self
            .blocks
            .drain(..)
            .map(|b| b.expect("every allocated block must be sealed"))
            .collect();
        // WP-C6.2a: the body's identity comes from the SAME constructor every call reference uses.
        let instance = self.instance_from_key(&key)?;
        Ok(MirBody {
            instance,
            params,
            ret,
            locals: std::mem::take(&mut self.locals),
            blocks,
            entry: BlockId(0),
        })
    }

    // ---- statements/blocks ----

    /// Lower a block; returns its tail value (if any). `None` also covers diverged paths.
    /// C4.5d: each HIR block is a drop scope — its droppable locals drop at block exit in
    /// reverse declaration order, after the tail value (if any) has moved out.
    fn lower_block_value(&mut self, block_id: hir::BlockId) -> Result<Option<Operand>, LowerError> {
        self.scopes.push(Vec::new());
        let block = self.hir.block(block_id);
        let block_span = block.span;
        for &stmt in &block.stmts {
            self.lower_stmt(stmt)?;
        }
        let mut tail_op = match block.tail {
            Some(tail) => {
                let op = self.lower_expr_operand_or_unit(tail)?;
                // Materialize a place-reading tail into a temp before this scope's drops:
                // the value (or copy) must be taken before the locals it may read from are
                // destroyed or poisoned.
                match op {
                    Some(op @ (Operand::Copy(_) | Operand::Move(_)))
                        if !self.scopes.last().map(Vec::is_empty).unwrap_or(true) =>
                    {
                        let ty = self.expr_mir_ty(tail)?;
                        let tmp = self.new_temp(ty.clone());
                        self.emit(
                            Statement::Assign(Place::local(tmp), Rvalue::Use(op)),
                            self.synthetic(block_span, SyntheticKind::DropElaboration),
                        );
                        Some(self.read_place(Place::local(tmp), &ty, block_span)?)
                    }
                    other => other,
                }
            }
            None => None,
        };
        let depth = self.scopes.len() - 1;
        self.emit_scope_drops_from(depth, block_span);
        self.scopes.pop();
        if let Some(op) = tail_op.take() {
            return Ok(Some(op));
        }
        Ok(None)
    }

    fn lower_stmt(&mut self, stmt_id: hir::StmtId) -> Result<(), LowerError> {
        let stmt = self.hir.stmt(stmt_id);
        let span = stmt.span;
        match &self.hir.stmt(stmt_id).kind {
            StmtKind::Empty => Ok(()),
            StmtKind::Expr { expr, .. } => {
                let op = self.lower_expr_operand_or_unit(*expr)?;
                // C4.5d: a discarded droppable value drops immediately (abstract-machine
                // temporary destruction; oracle-confirmed timing).
                if let Some(op) = op {
                    let ty = self.expr_mir_ty(*expr)?;
                    if self.ty_needs_drop(&ty, span)? {
                        self.discover_drop_impls(&ty)?;
                        let tmp = self.new_temp(ty);
                        self.emit(
                            Statement::Assign(Place::local(tmp), Rvalue::Use(op)),
                            self.synthetic(span, SyntheticKind::DropElaboration),
                        );
                        self.emit_temp_drop(tmp, span);
                    }
                }
                Ok(())
            }
            StmtKind::Let {
                name, local, init, ..
            } => {
                let ty = self
                    .tables
                    .local_types
                    .get(local)
                    .cloned()
                    .unwrap_or(Ty::Error);
                let mir_ty = self.mir_ty(&ty, *name)?;
                self.locals.push(LocalDecl {
                    ty: mir_ty.clone(),
                    kind: LocalKind::User(self.text(*name).to_string()),
                });
                let mir_local = LocalId((self.locals.len() - 1) as u32);
                self.local_map.insert(local.0, mir_local);
                // C4.5d: flags start false (registered before the initializer, so an early
                // exit inside it skips this local's drops) and flip true after init.
                self.register_droppable_local(mir_local, &mir_ty, false, *name)?;
                if let Some(init) = init {
                    let value = self.lower_expr_to_operand(*init)?;
                    // C6.1f-b2: `let r: &P = m;` where `m: &mut P`.
                    let value = self.weaken_ref_to(value, &mir_ty, *name)?;
                    self.emit(
                        Statement::Assign(Place::local(mir_local), Rvalue::Use(value)),
                        self.info(span),
                    );
                    self.set_flags_under(mir_local.0, &[], true, *name);
                }
                Ok(())
            }
            StmtKind::Return(value) => {
                if let Some(value) = value {
                    let op = self.lower_expr_to_operand(*value)?;
                    // C6.1f-b2: `fn g(m: &mut P) -> &P { return m; }`.
                    let ret_ty = self.locals[0].ty.clone();
                    let op = self.weaken_ref_to(op, &ret_ty, span)?;
                    self.emit(
                        Statement::Assign(Place::local(LocalId(0)), Rvalue::Use(op)),
                        self.info(span),
                    );
                } else {
                    self.emit(
                        Statement::Assign(
                            Place::local(LocalId(0)),
                            Rvalue::Use(Operand::Const(Constant::Unit)),
                        ),
                        self.info(span),
                    );
                }
                // C4.5d: early return drops every live scope (innermost first) after the
                // return value has moved into Local(0).
                self.emit_scope_drops_from(0, span);
                let dead = self.new_block();
                self.terminate(Terminator::Return, self.info(span), dead);
                Ok(())
            }
            StmtKind::Break(None) => {
                let Some(targets) = self.loops.last() else {
                    return unsupported("break outside a loop", span);
                };
                let target = targets.break_target;
                let depth = targets.scope_depth;
                // C4.5d: leaving the loop drops every scope inside it.
                self.emit_scope_drops_from(depth, span);
                let dead = self.new_block();
                self.terminate(Terminator::Goto { target }, self.info(span), dead);
                Ok(())
            }
            StmtKind::Break(Some(value)) => {
                // A7: `break <value>` — evaluate the value (before the scope drops, since it may
                // read locals in those scopes), write it into the innermost loop's value target
                // (a value-position `loop`), then drop scopes and jump out. If there is no value
                // target (a `while`/`for`, or a statement-position loop), the value type-checks
                // as Unit; lower it for its side effects and discard.
                let Some(targets) = self.loops.last() else {
                    return unsupported("break outside a loop", span);
                };
                let target = targets.break_target;
                let depth = targets.scope_depth;
                let value_target = targets.value_target;
                let op = self.lower_expr_to_operand(*value)?;
                if let Some(local) = value_target {
                    self.emit(
                        Statement::Assign(Place::local(local), Rvalue::Use(op)),
                        self.info(span),
                    );
                }
                self.emit_scope_drops_from(depth, span);
                let dead = self.new_block();
                self.terminate(Terminator::Goto { target }, self.info(span), dead);
                Ok(())
            }
            StmtKind::Continue => {
                let Some(targets) = self.loops.last() else {
                    return unsupported("continue outside a loop", span);
                };
                let target = targets.continue_target;
                let depth = targets.scope_depth;
                // C4.5d: restarting the loop drops the current iteration's scopes.
                self.emit_scope_drops_from(depth, span);
                let dead = self.new_block();
                self.terminate(Terminator::Goto { target }, self.info(span), dead);
                Ok(())
            }
            StmtKind::Item(_) => unsupported("nested item (C4.5)", span),
            StmtKind::Error => unsupported("error statement", span),
        }
    }

    // ---- expressions ----

    /// Lower an expression that may be Unit-typed control flow (statement position or block
    /// tail). Returns `Some(op)` only for value-producing expressions.
    fn lower_expr_operand_or_unit(&mut self, expr: ExprId) -> Result<Option<Operand>, LowerError> {
        let ty = self.tables.expr_types.get(&expr);
        let is_unit = matches!(ty, Some(Ty::Primitive(Primitive::Unit)))
            || matches!(ty, Some(Ty::Never))
            || ty.is_none();
        if is_unit {
            self.lower_unit_expr(expr)?;
            Ok(None)
        } else {
            Ok(Some(self.lower_expr_to_operand(expr)?))
        }
    }

    /// Unit-typed (or diverging) expressions in statement/tail position.
    fn lower_unit_expr(&mut self, expr: ExprId) -> Result<(), LowerError> {
        let span = self.hir.expr(expr).span;
        match &self.hir.expr(expr).kind {
            hir::ExprKind::If {
                cond,
                then_block,
                else_,
            } => {
                let cond_op = self.lower_expr_to_operand(*cond)?;
                let then_block_id = self.new_block();
                let join = self.new_block();
                let else_block_id = if else_.is_some() {
                    self.new_block()
                } else {
                    join
                };
                self.terminate(
                    Terminator::SwitchInt {
                        scrut: cond_op,
                        arms: vec![(1, then_block_id)],
                        otherwise: else_block_id,
                    },
                    self.info(span),
                    then_block_id,
                );
                self.lower_block_value(*then_block)?;
                self.terminate(Terminator::Goto { target: join }, self.info(span), join);
                if let Some(else_expr) = else_ {
                    self.current = else_block_id;
                    self.lower_unit_expr(*else_expr)?;
                    self.terminate(Terminator::Goto { target: join }, self.info(span), join);
                }
                self.current = join;
                Ok(())
            }
            hir::ExprKind::While { cond, body } => {
                let header = self.new_block();
                let body_block = self.new_block();
                let exit = self.new_block();
                self.terminate(Terminator::Goto { target: header }, self.info(span), header);
                let cond_op = self.lower_expr_to_operand(*cond)?;
                self.terminate(
                    Terminator::SwitchInt {
                        scrut: cond_op,
                        arms: vec![(1, body_block)],
                        otherwise: exit,
                    },
                    self.info(span),
                    body_block,
                );
                self.loops.push(LoopTargets {
                    continue_target: header,
                    break_target: exit,
                    scope_depth: self.scopes.len(),
                    value_target: None,
                });
                self.lower_block_value(*body)?;
                self.loops.pop();
                self.terminate(Terminator::Goto { target: header }, self.info(span), exit);
                Ok(())
            }
            hir::ExprKind::Loop { body } => {
                let body_block = self.new_block();
                let exit = self.new_block();
                self.terminate(
                    Terminator::Goto { target: body_block },
                    self.info(span),
                    body_block,
                );
                self.loops.push(LoopTargets {
                    continue_target: body_block,
                    break_target: exit,
                    scope_depth: self.scopes.len(),
                    value_target: None,
                });
                self.lower_block_value(*body)?;
                self.loops.pop();
                self.terminate(
                    Terminator::Goto { target: body_block },
                    self.info(span),
                    exit,
                );
                Ok(())
            }
            hir::ExprKind::For {
                var,
                local,
                iter,
                body,
            } => {
                let (lo, hi, inclusive) = match &self.hir.expr(*iter).kind {
                    hir::ExprKind::Range { lo, hi, inclusive } => (*lo, *hi, *inclusive),
                    _ => {
                        // 0.1-A2 (C4.5f-2): `for x in v.iter()` — borrowing Vec iteration.
                        // 0.1-A3 (f-3a): `for k in m.keys()` — borrowing key iteration.
                        let iter_ty = self.expr_mir_ty(*iter)?;
                        if let MirTy::Core(
                            core @ (crate::hir::CoreType::VecIter
                            | crate::hir::CoreType::KeysIter
                            | crate::hir::CoreType::Iter),
                            args,
                        ) = &iter_ty
                        {
                            let elem = args.first().cloned().unwrap_or(MirTy::Unit);
                            let elem_ref = MirTy::Ref {
                                mutable: false,
                                inner: Box::new(elem),
                            };
                            let next_rt = match core {
                                crate::hir::CoreType::VecIter => RuntimeFn::VecIterNext,
                                crate::hir::CoreType::Iter => RuntimeFn::HashSetIterNext,
                                _ => RuntimeFn::HashMapKeysIterNext,
                            };
                            return self.lower_for_over_iter(
                                *var,
                                *local,
                                *iter,
                                *body,
                                iter_ty.clone(),
                                elem_ref,
                                next_rt,
                                span,
                                false,
                            );
                        }
                        // `for x in &v`: the expression is a BORROW of the Vec, not a cursor, so
                        // the cursor is built from it here. Identical lowering to `v.iter()` from
                        // this point on — same `VecIterNew`, same `VecIterNext`, same `&T` item —
                        // because it is the same iteration; only the spelling differed.
                        if let MirTy::Ref { inner, .. } = &iter_ty {
                            if let MirTy::Core(crate::hir::CoreType::Vec, args) = inner.as_ref() {
                                let elem = args.first().cloned().unwrap_or(MirTy::Unit);
                                let elem_ref = MirTy::Ref {
                                    mutable: false,
                                    inner: Box::new(elem.clone()),
                                };
                                let cursor_ty =
                                    MirTy::Core(crate::hir::CoreType::VecIter, vec![elem]);
                                return self.lower_for_over_iter(
                                    *var,
                                    *local,
                                    *iter,
                                    *body,
                                    cursor_ty,
                                    elem_ref,
                                    RuntimeFn::VecIterNext,
                                    span,
                                    true,
                                );
                            }
                        }
                        // 0.1-A5 (C4.6 A4-2d): `for c in s.chars()` — `Char` by VALUE (not a
                        // reference). The iterator is a borrowed snapshot over the string's
                        // chars; `Next` yields `Option<Char>`.
                        if matches!(iter_ty, MirTy::Core(crate::hir::CoreType::CharsIter, _)) {
                            return self.lower_for_over_iter(
                                *var,
                                *local,
                                *iter,
                                *body,
                                iter_ty,
                                MirTy::Char,
                                RuntimeFn::CharsIterNext,
                                span,
                                false,
                            );
                        }
                        // A1: `for x in it` over a USER Iterator impl — desugar to repeated
                        // `it.next()` instance calls yielding `Option<Item>` by value.
                        if let MirTy::Struct(item, targs)
                        | MirTy::Enum(EnumRef::User(item), targs) = &iter_ty
                        {
                            let (item, targs) = (*item, targs.clone());
                            return self.lower_for_over_user_iter(
                                *var, *local, *iter, *body, item, targs, span,
                            );
                        }
                        // A4: `for i in r` where `r` is a range VALUE (`Ty::Range`) — the
                        // front-end type distinguishes it from a genuine 3-tuple. The inclusive
                        // flag is a runtime field, so the loop condition branches on it.
                        if matches!(self.tables.expr_types.get(iter), Some(Ty::Range(_))) {
                            return self
                                .lower_for_over_range_value(*var, *local, *iter, *body, span);
                        }
                        // WP-C4.7-9 audit: `for x in a` over a fixed-length ARRAY. The checker
                        // accepts it and the oracle runs it, so MIR refusing it was an internal
                        // inconsistency, not a language boundary.
                        if let MirTy::Array(elem, len) = &iter_ty {
                            let (elem, len) = ((**elem).clone(), *len);
                            return self
                                .lower_for_over_array(*var, *local, *iter, *body, elem, len, span);
                        }
                        return unsupported(
                            "for over a non-range, non-Vec iterator (a later increment)",
                            span,
                        );
                    }
                };
                let elem_ty = self.expr_mir_ty(lo)?;
                let lo_op = self.lower_expr_to_operand(lo)?;
                let hi_op = self.lower_expr_to_operand(hi)?;
                // Materialize the bound once (evaluation order: lo then hi, once each).
                let bound = self.new_temp(elem_ty.clone());
                self.emit(
                    Statement::Assign(Place::local(bound), Rvalue::Use(hi_op)),
                    self.synthetic(span, SyntheticKind::ForLoopDesugar),
                );
                self.locals.push(LocalDecl {
                    ty: elem_ty.clone(),
                    kind: LocalKind::User(self.text(*var).to_string()),
                });
                let induction = LocalId((self.locals.len() - 1) as u32);
                self.local_map.insert(local.0, induction);
                self.emit(
                    Statement::Assign(Place::local(induction), Rvalue::Use(lo_op)),
                    self.synthetic(span, SyntheticKind::ForLoopDesugar),
                );

                let header = self.new_block();
                let body_block = self.new_block();
                let latch = self.new_block();
                let exit = self.new_block();
                self.terminate(Terminator::Goto { target: header }, self.info(span), header);
                let cmp = self.new_temp(MirTy::Bool);
                let cmp_op = if inclusive {
                    MirBinOp::Le
                } else {
                    MirBinOp::Lt
                };
                self.emit(
                    Statement::Assign(
                        Place::local(cmp),
                        Rvalue::BinOp(
                            cmp_op,
                            Operand::Copy(Place::local(induction)),
                            Operand::Copy(Place::local(bound)),
                        ),
                    ),
                    self.synthetic(span, SyntheticKind::ForLoopDesugar),
                );
                self.terminate(
                    Terminator::SwitchInt {
                        scrut: Operand::Copy(Place::local(cmp)),
                        arms: vec![(1, body_block)],
                        otherwise: exit,
                    },
                    self.synthetic(span, SyntheticKind::ForLoopDesugar),
                    body_block,
                );
                self.loops.push(LoopTargets {
                    continue_target: latch,
                    break_target: exit,
                    scope_depth: self.scopes.len(),
                    value_target: None,
                });
                self.lower_block_value(*body)?;
                self.loops.pop();
                self.terminate(
                    Terminator::Goto { target: latch },
                    self.synthetic(span, SyntheticKind::ForLoopDesugar),
                    latch,
                );
                // Latch: step = induction + 1 (checked), then induction = step, back to header.
                let step = self.new_temp(elem_ty);
                let copy_block = self.new_block();
                let induction_ty = self.locals[induction.0 as usize].ty.clone();
                self.terminate(
                    Terminator::Checked {
                        op: CheckedOp::Add,
                        args: vec![
                            Operand::Copy(Place::local(induction)),
                            Operand::Const(Constant::Int(1, induction_ty)),
                        ],
                        dest: step,
                        target: copy_block,
                        trap: TrapInfo {
                            category: TrapCategory::IntegerOverflow,
                            source: self.synthetic(span, SyntheticKind::ForLoopDesugar),
                        },
                    },
                    self.synthetic(span, SyntheticKind::ForLoopDesugar),
                    copy_block,
                );
                self.emit(
                    Statement::Assign(
                        Place::local(induction),
                        Rvalue::Use(Operand::Copy(Place::local(step))),
                    ),
                    self.synthetic(span, SyntheticKind::ForLoopDesugar),
                );
                self.terminate(
                    Terminator::Goto { target: header },
                    self.synthetic(span, SyntheticKind::ForLoopDesugar),
                    exit,
                );
                Ok(())
            }
            hir::ExprKind::Assign { op, lhs, rhs } => {
                // A1 (CD-031), C4.5e-2: `v[i] = x` on a Vec is `old = VecReplace(&mut v, i, x)`
                // then drop `old` (install-then-destroy, CD-012) — not a place assignment.
                if matches!(op, AssignOp::Assign) {
                    if let hir::ExprKind::Index { base, index } = &self.hir.expr(*lhs).kind {
                        let (peeled, _) = Self::peel_refs(self.expr_mir_ty(*base)?);
                        if let MirTy::Core(crate::hir::CoreType::Vec, elem_args) = &peeled {
                            let elem = elem_args.first().cloned().unwrap_or(MirTy::Unit);
                            return self.lower_vec_index_set(*base, *index, elem, *rhs, span);
                        }
                    }
                }
                // Evaluation order: RHS before LHS place (CD-007).
                let rhs_op = self.lower_expr_to_operand(*rhs)?;
                let place = self.lower_place(*lhs)?;
                match op {
                    AssignOp::Assign => {
                        // C6.1f-b2: an assignment destination is an expected-type boundary.
                        let want = self.expr_mir_ty(*lhs)?;
                        let rhs_op = self.weaken_ref_to(rhs_op, &want, span)?;
                        self.lower_overwriting_assign(place, rhs_op, span)?;
                        Ok(())
                    }
                    compound => {
                        let ty = self.expr_mir_ty(*lhs)?;
                        let current = self.read_place(place.clone(), &ty, span)?;
                        let bin = match compound {
                            AssignOp::AddAssign => BinOp::Add,
                            AssignOp::SubAssign => BinOp::Sub,
                            AssignOp::MulAssign => BinOp::Mul,
                            AssignOp::DivAssign => BinOp::Div,
                            AssignOp::RemAssign => BinOp::Rem,
                            AssignOp::PowAssign => BinOp::Pow,
                            AssignOp::BitAndAssign => BinOp::BitAnd,
                            AssignOp::BitOrAssign => BinOp::BitOr,
                            AssignOp::BitXorAssign => BinOp::BitXor,
                            AssignOp::ShlAssign => BinOp::Shl,
                            AssignOp::ShrAssign => BinOp::Shr,
                            AssignOp::Assign => unreachable!("handled above"),
                        };
                        let result = self.lower_arith_operands(bin, current, rhs_op, &ty, span)?;
                        self.emit(
                            Statement::Assign(place, Rvalue::Use(result)),
                            self.info(span),
                        );
                        Ok(())
                    }
                }
            }
            hir::ExprKind::Block(block) => {
                self.lower_block_value(*block)?;
                Ok(())
            }
            hir::ExprKind::Match { .. } => {
                self.lower_match(expr, None)?;
                Ok(())
            }
            hir::ExprKind::Call { .. } => {
                self.lower_call(expr, None)?;
                Ok(())
            }
            hir::ExprKind::Try(_) => {
                let _ = self.lower_expr_to_operand(expr)?;
                Ok(())
            }
            hir::ExprKind::Tuple(elems) if elems.is_empty() => Ok(()),
            other => unsupported(
                format!("unit expression form (C4.5): {}", expr_kind_name(other)),
                span,
            ),
        }
    }

    /// Lower a value-producing expression to an operand (temps as needed, L-to-R order).
    fn lower_expr_to_operand(&mut self, expr: ExprId) -> Result<Operand, LowerError> {
        let span = self.hir.expr(expr).span;
        match &self.hir.expr(expr).kind {
            hir::ExprKind::Lit(lit) => self.lower_lit(expr, lit),
            hir::ExprKind::Path { res, .. } => match res {
                Res::Local(local) | Res::SelfValue(local) => {
                    let mir_local = *self.local_map.get(&local.0).ok_or_else(|| LowerError {
                        what: "use of unknown local".to_string(),
                        span,
                    })?;
                    let ty = self.locals[mir_local.0 as usize].ty.clone();
                    self.read_place(Place::local(mir_local), &ty, span)
                }
                // A named function used as a function value (CD-021 item 16; generic fns
                // monomorphise through the recorded instantiation, C4.5c / CD-021 item 21).
                Res::Item(item) => match &self.hir.item(*item).kind {
                    ItemKind::Const { value, .. } => self.lower_expr_to_operand(*value),
                    ItemKind::Fn(_) => {
                        let instance = self.top_fn_instance(*item, expr, span)?;
                        Ok(Operand::Const(Constant::FnPtr(instance)))
                    }
                    _ => unsupported("path form in value position (C4.5)", span),
                },
                Res::Builtin(Builtin::None) => Ok(self.aggregate_to_temp(
                    expr,
                    AggKind::EnumVariant(EnumRef::CoreOption, 0),
                    Vec::new(),
                    span,
                )?),
                // A2 (CE3): `Ordering::Less/Equal/Greater` construct the logical `CoreOrdering`
                // enum with the fixed discriminants Less=0, Equal=1, Greater=2.
                Res::Builtin(
                    variant @ (Builtin::OrderingLess
                    | Builtin::OrderingEqual
                    | Builtin::OrderingGreater),
                ) => {
                    let disc = match variant {
                        Builtin::OrderingLess => 0,
                        Builtin::OrderingEqual => 1,
                        Builtin::OrderingGreater => 2,
                        _ => unreachable!(),
                    };
                    Ok(self.aggregate_to_temp(
                        expr,
                        AggKind::EnumVariant(EnumRef::CoreOrdering, disc),
                        Vec::new(),
                        span,
                    )?)
                }
                // Unit enum variant in value position (`Shape::Point`).
                Res::Variant(item, variant) => Ok(self.aggregate_to_temp(
                    expr,
                    AggKind::EnumVariant(EnumRef::User(*item), *variant),
                    Vec::new(),
                    span,
                )?),
                _ => unsupported("path form in value position (C4.5)", span),
            },
            hir::ExprKind::Unary { op, operand } => {
                let ty = self.expr_mir_ty(expr)?;
                if let UnOp::Ref { mutable } = op {
                    // 0.1-A6 (A4 slicing): `&base[lo..hi]` — a slice view, not a place borrow.
                    // Trap provenance is the INDEX expression's span (`a[1..9]`), matching the
                    // oracle, not the enclosing `&…`.
                    if let hir::ExprKind::Index { base, index } = &self.hir.expr(*operand).kind {
                        if matches!(self.tables.expr_types.get(index), Some(Ty::Range(_))) {
                            let index_span = self.hir.expr(*operand).span;
                            return self.lower_make_slice(*base, *index, *mutable, index_span);
                        }
                        // **`&v[i]` on a Vec — a borrow of the element, not a value read.**
                        //
                        // A Vec is an opaque runtime type, not a projectable place: there is no
                        // `Projection::Index` into one, which is why `&v[i]` could not be written
                        // at all while `&a[i]` on an ARRAY has always worked. Reading by value
                        // (`v[i]`) is separately refused for a non-`Copy` element by E0106, since
                        // that would move out of the Vec — so without this form an owning element
                        // was reachable only through `v.get(i)` or iteration.
                        //
                        // `VecGetRef` already yields `Option<&T>` and never traps; the bounds
                        // failure becomes the `None` arm, and indexing traps on out-of-bounds, so
                        // that arm raises `IndexOutOfBounds` — the same category and the same
                        // observable behaviour as `v[i]`, reached by a different route.
                        let (peeled, _) = Self::peel_refs(self.expr_mir_ty(*base)?);
                        if let MirTy::Core(crate::hir::CoreType::Vec, elem_args) = &peeled {
                            let elem = elem_args.first().cloned().unwrap_or(MirTy::Unit);
                            // Trap provenance is the ENCLOSING `&…` span, not the index
                            // expression's — the opposite of the range/slice case just above, and
                            // not a matter of taste: the oracle reaches this through `expr_place`
                            // under `UnOp::Ref` and reports the borrow, so a differential run
                            // disagreed by exactly one column until this matched it. Three-engine
                            // agreement is the authority on provenance; either span would have
                            // been defensible alone.
                            return self
                                .lower_vec_index_borrow(*base, *index, elem, *mutable, span);
                        }
                    }
                    // C4.5b-2: `&expr` / `&mut expr` — borrow of a place, NOT a value read.
                    // f-3a: a non-place operand (e.g. `&String::from("x")`) materializes into
                    // a temp first, mirroring the method-receiver auto-borrow path.
                    let place = match self.lower_place(*operand) {
                        Ok(place) => place,
                        Err(_) => {
                            let inner_ty = self.expr_mir_ty(*operand)?;
                            let value = self.lower_expr_to_operand(*operand)?;
                            let temp = self.new_temp(inner_ty);
                            self.emit(
                                Statement::Assign(Place::local(temp), Rvalue::Use(value)),
                                self.info(span),
                            );
                            Place::local(temp)
                        }
                    };
                    let dest = self.new_temp(ty.clone());
                    self.emit(
                        Statement::Assign(
                            Place::local(dest),
                            Rvalue::RefOf {
                                mutable: *mutable,
                                place,
                            },
                        ),
                        self.info(span),
                    );
                    return self.read_place(Place::local(dest), &ty, span);
                }
                if matches!(op, UnOp::Deref) {
                    // `*r` as a value: place = r + Deref. A1: a non-place operand (e.g. a
                    // method-call result `*h.get()`) materializes into a temp first.
                    let operand_ty = self.expr_mir_ty(*operand)?;
                    let mut place = self.place_or_temp(*operand, &operand_ty, span)?;
                    place.projection.push(Projection::Deref);
                    return self.read_place(place, &ty, span);
                }
                let inner = self.lower_expr_to_operand(*operand)?;
                match op {
                    UnOp::Not => {
                        let dest = self.new_temp(ty);
                        self.emit(
                            Statement::Assign(
                                Place::local(dest),
                                Rvalue::UnOp(MirUnOp::Not, inner),
                            ),
                            self.info(span),
                        );
                        Ok(Operand::Copy(Place::local(dest)))
                    }
                    UnOp::Neg => match ty {
                        MirTy::Float32 | MirTy::Float64 => {
                            let dest = self.new_temp(ty);
                            self.emit(
                                Statement::Assign(
                                    Place::local(dest),
                                    Rvalue::UnOp(MirUnOp::FloatNeg, inner),
                                ),
                                self.info(span),
                            );
                            Ok(Operand::Copy(Place::local(dest)))
                        }
                        _ => {
                            let dest = self.new_temp(ty);
                            let after = self.new_block();
                            self.terminate(
                                Terminator::Checked {
                                    op: CheckedOp::Neg,
                                    args: vec![inner],
                                    dest,
                                    target: after,
                                    trap: TrapInfo {
                                        category: TrapCategory::IntegerOverflow,
                                        source: self.info(span),
                                    },
                                },
                                self.info(span),
                                after,
                            );
                            Ok(Operand::Copy(Place::local(dest)))
                        }
                    },
                    // A5: `~a` is `a ^ all-ones`. For a signed width the mask is −1 (i128
                    // all-ones, giving !a = −a−1); for an unsigned width W it is `(1<<W)−1`
                    // (giving `(!a) & mask`). Both agree with the oracle's `UnOp::BitNot` and
                    // stay in range, so no trap is owed. Desugaring to BitXor avoids a
                    // type-carrying MIR unary op.
                    UnOp::BitNot => {
                        let mask = match &ty {
                            MirTy::Int8 | MirTy::Int16 | MirTy::Int32 | MirTy::Int64 => -1_i128,
                            MirTy::UInt8 => i128::from(u8::MAX),
                            MirTy::UInt16 => i128::from(u16::MAX),
                            MirTy::UInt32 => i128::from(u32::MAX),
                            MirTy::UInt64 => i128::from(u64::MAX),
                            _ => return unsupported("bitwise-not on a non-integer type", span),
                        };
                        let dest = self.new_temp(ty.clone());
                        self.emit(
                            Statement::Assign(
                                Place::local(dest),
                                Rvalue::BinOp(
                                    MirBinOp::BitXor,
                                    inner,
                                    Operand::Const(Constant::Int(mask, ty)),
                                ),
                            ),
                            self.info(span),
                        );
                        Ok(Operand::Copy(Place::local(dest)))
                    }
                    _ => unsupported("unary operator (C4.5)", span),
                }
            }
            hir::ExprKind::Binary { op, lhs, rhs } => match op {
                BinOp::And | BinOp::Or => self.lower_short_circuit(*op, *lhs, *rhs, span),
                _ => {
                    let lhs_ty = self.expr_mir_ty(*lhs)?;
                    // A1 (CD-031): String/str comparison routes through StrEq/StrCmp, never a
                    // structural BinOp (V-STR-2). Handled before eager operand lowering so the
                    // operands are borrowed as `&str`, not moved.
                    if matches!(
                        op,
                        BinOp::Eq | BinOp::Ne | BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge
                    ) && Self::is_text_ty(&lhs_ty)
                    {
                        return self.lower_string_comparison(*op, *lhs, *rhs, span);
                    }
                    // A3 (C4.6, CD-033): `==`/`!=` on a (non-generic) user nominal dispatches
                    // through its `Eq::eq` impl — the oracle does the same. Handled before eager
                    // operand lowering so both sides are borrowed as `&Self`, not moved. Ordered
                    // comparisons on a user nominal (`Ord`) still wait on the CE3 `Ordering`
                    // runtime-surface amendment; generic-nominal Eq waits on A1.
                    if matches!(op, BinOp::Eq | BinOp::Ne) {
                        if let MirTy::Struct(item, targs)
                        | MirTy::Enum(EnumRef::User(item), targs) =
                            &Self::peel_refs(lhs_ty.clone()).0
                        {
                            let (item, targs) = (*item, targs.clone());
                            return self.lower_user_eq(item, &targs, *op, *lhs, *rhs, span);
                        }
                    }
                    // A3 Ord (A2 amendment, CE3): ordered comparison on a (non-generic) user
                    // nominal dispatches through `Ord::cmp`, then maps the returned `Ordering`
                    // discriminant to the comparison's `Bool`.
                    if matches!(op, BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge) {
                        if let MirTy::Struct(item, targs)
                        | MirTy::Enum(EnumRef::User(item), targs) =
                            &Self::peel_refs(lhs_ty.clone()).0
                        {
                            let (item, targs) = (*item, targs.clone());
                            return self.lower_user_ord(item, &targs, *op, *lhs, *rhs, span);
                        }
                    }
                    let lhs_op = self.lower_expr_to_operand(*lhs)?;
                    let rhs_op = self.lower_expr_to_operand(*rhs)?;
                    match op {
                        BinOp::Add
                        | BinOp::Sub
                        | BinOp::Mul
                        | BinOp::Div
                        | BinOp::Rem
                        | BinOp::Pow
                        | BinOp::BitAnd
                        | BinOp::BitOr
                        | BinOp::BitXor
                        | BinOp::Shl
                        | BinOp::Shr => {
                            self.lower_arith_operands(*op, lhs_op, rhs_op, &lhs_ty, span)
                        }
                        BinOp::Eq | BinOp::Ne | BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge => {
                            // Direct user-nominal `==`/`!=` was already routed to `Eq::eq` above
                            // (A3). What remains here and still dispatches through a user impl:
                            // ordered comparisons on a user nominal (`Ord`, awaiting the CE3
                            // `Ordering` runtime-surface amendment) and any comparison on a
                            // COMPOUND type that merely contains a user nominal (needs structural
                            // + impl dispatch). Both would silently diverge from the oracle under
                            // a structural `BinOp`, so they stay unsupported.
                            if ty_mentions_user_nominal(&lhs_ty) {
                                return unsupported(
                                    "ordered/compound comparison on a user-defined type dispatches through its Ord/Eq impl (A3 Ord pending the Ordering amendment)",
                                    span,
                                );
                            }
                            let mir_op = match op {
                                BinOp::Eq => MirBinOp::Eq,
                                BinOp::Ne => MirBinOp::Ne,
                                BinOp::Lt => MirBinOp::Lt,
                                BinOp::Le => MirBinOp::Le,
                                BinOp::Gt => MirBinOp::Gt,
                                BinOp::Ge => MirBinOp::Ge,
                                _ => unreachable!(),
                            };
                            let dest = self.new_temp(MirTy::Bool);
                            self.emit(
                                Statement::Assign(
                                    Place::local(dest),
                                    Rvalue::BinOp(mir_op, lhs_op, rhs_op),
                                ),
                                self.info(span),
                            );
                            Ok(Operand::Copy(Place::local(dest)))
                        }
                        _ => unsupported("binary operator (C4.5)", span),
                    }
                }
            },
            hir::ExprKind::Call { .. } => {
                let ty = self.expr_mir_ty(expr)?;
                let dest = self.new_temp(ty);
                self.lower_call(expr, Some(Place::local(dest)))?;
                let ty = self.locals[dest.0 as usize].ty.clone();
                self.read_place(Place::local(dest), &ty, span)
            }
            hir::ExprKind::If {
                cond,
                then_block,
                else_,
            } => {
                let Some(else_expr) = else_ else {
                    // A7: a `then`-only `if` is Unit-typed even in value position; lower it for
                    // its effects and yield Unit.
                    self.lower_unit_expr(expr)?;
                    return Ok(Operand::Const(Constant::Unit));
                };
                let ty = self.expr_mir_ty(expr)?;
                let dest = self.new_temp(ty);
                let cond_op = self.lower_expr_to_operand(*cond)?;
                let then_id = self.new_block();
                let else_id = self.new_block();
                let join = self.new_block();
                self.terminate(
                    Terminator::SwitchInt {
                        scrut: cond_op,
                        arms: vec![(1, then_id)],
                        otherwise: else_id,
                    },
                    self.info(span),
                    then_id,
                );
                let then_value = self.lower_block_value(*then_block)?;
                if let Some(v) = then_value {
                    self.emit(
                        Statement::Assign(Place::local(dest), Rvalue::Use(v)),
                        self.info(span),
                    );
                }
                self.terminate(Terminator::Goto { target: join }, self.info(span), else_id);
                let else_value = self.lower_expr_to_operand(*else_expr)?;
                self.emit(
                    Statement::Assign(Place::local(dest), Rvalue::Use(else_value)),
                    self.info(span),
                );
                self.terminate(Terminator::Goto { target: join }, self.info(span), join);
                let ty = self.locals[dest.0 as usize].ty.clone();
                self.read_place(Place::local(dest), &ty, span)
            }
            hir::ExprKind::Block(block) => {
                let value = self.lower_block_value(*block)?;
                value.ok_or_else(|| LowerError {
                    what: "block in value position yielded no value".to_string(),
                    span,
                })
            }
            hir::ExprKind::Tuple(elems) if elems.is_empty() => {
                // DEV-112 / TYPE-PRIM-001: `()` IS `Unit`, so it lowers to the Unit constant, not
                // to a zero-field tuple aggregate. Without this the checker (which canonicalises
                // the type) and lowering disagree, and the verifier says so:
                // "aggregate Tuple assigned to incompatible type Unit" (MIR-0004).
                Ok(Operand::Const(Constant::Unit))
            }
            hir::ExprKind::Tuple(elems) => {
                let ops = elems
                    .iter()
                    .map(|&e| self.lower_expr_to_operand(e))
                    .collect::<Result<Vec<_>, _>>()?;
                self.aggregate_to_temp(expr, AggKind::Tuple, ops, span)
            }
            hir::ExprKind::Array(elems) => {
                let elem_ty = match self.expr_mir_ty(expr)? {
                    MirTy::Array(elem, _) => *elem,
                    other => other,
                };
                let ops = elems
                    .iter()
                    .map(|&e| self.lower_expr_to_operand(e))
                    .collect::<Result<Vec<_>, _>>()?;
                self.aggregate_to_temp(expr, AggKind::Array(elem_ty), ops, span)
            }
            // A4: a range in value position (`let r = lo..hi`) builds the `(start, end,
            // inclusive)` tuple. Evaluation order lo-then-hi, once each.
            hir::ExprKind::Range { lo, hi, inclusive } => {
                let lo_op = self.lower_expr_to_operand(*lo)?;
                let hi_op = self.lower_expr_to_operand(*hi)?;
                let ty = self.expr_mir_ty(expr)?;
                let dest = self.new_temp(ty);
                self.emit(
                    Statement::Assign(
                        Place::local(dest),
                        Rvalue::Aggregate(
                            AggKind::Tuple,
                            vec![lo_op, hi_op, Operand::Const(Constant::Bool(*inclusive))],
                        ),
                    ),
                    self.info(span),
                );
                let ty = self.locals[dest.0 as usize].ty.clone();
                self.read_place(Place::local(dest), &ty, span)
            }
            // A7: `[value; count]` — value evaluated once, replicated `count` times (count is a
            // const carried by the array type; the value is `Copy`, so replicating the operand
            // matches the oracle's clone-per-element).
            hir::ExprKind::Repeat { value, .. } => {
                let (elem_ty, count) = match self.expr_mir_ty(expr)? {
                    MirTy::Array(elem, len) => (*elem, len as usize),
                    other => {
                        return unsupported(format!("repeat of non-array type {other:?}"), span)
                    }
                };
                let op = self.lower_expr_to_operand(*value)?;
                if matches!(op, Operand::Move(_)) {
                    return unsupported("repeat of a non-Copy value", span);
                }
                let ops = vec![op; count];
                self.aggregate_to_temp(expr, AggKind::Array(elem_ty), ops, span)
            }
            hir::ExprKind::StructLit { res, fields, .. } => {
                // Enum struct-variant literal (`Shape::Circle { radius: 2.0 }`).
                if let Res::Variant(item, variant) = res {
                    let field_order = self.variant_field_order(res, *variant)?;
                    let mut by_name: Vec<(String, Operand)> = Vec::new();
                    for field in fields {
                        let value = match field.expr {
                            Some(e) => self.lower_expr_to_operand(e)?,
                            None => {
                                return unsupported(
                                    "shorthand field in variant literal (C4.5)",
                                    field.name,
                                )
                            }
                        };
                        by_name.push((self.text(field.name).to_string(), value));
                    }
                    let mut ordered = Vec::new();
                    for name in &field_order {
                        let Some(pos) = by_name.iter().position(|(n, _)| n == name) else {
                            return unsupported("variant literal missing a field", span);
                        };
                        ordered.push(by_name.remove(pos).1);
                    }
                    return self.aggregate_to_temp(
                        expr,
                        AggKind::EnumVariant(EnumRef::User(*item), *variant),
                        ordered,
                        span,
                    );
                }
                let Res::Item(item) = res else {
                    return unsupported("struct literal path (C4.5)", span);
                };
                let ItemKind::Struct {
                    fields: decl_fields,
                    ..
                } = &self.hir.item(*item).kind
                else {
                    return unsupported("struct literal of non-struct", span);
                };
                // Lower field initializers in WRITTEN order (evaluation order), then arrange
                // into declaration order for the aggregate.
                let mut by_name: Vec<(String, Operand)> = Vec::new();
                for field in fields {
                    let value = match field.expr {
                        Some(e) => self.lower_expr_to_operand(e)?,
                        None => {
                            // Shorthand `Point { x }` — read the same-named local.
                            let name_text = self.text(field.name).to_string();
                            let local = self
                                .local_map
                                .iter()
                                .find_map(|(hir_local, mir_local)| {
                                    let decl = &self.locals[mir_local.0 as usize];
                                    if let LocalKind::User(n) = &decl.kind {
                                        if *n == name_text {
                                            return Some(*mir_local);
                                        }
                                    }
                                    let _ = hir_local;
                                    None
                                })
                                .ok_or_else(|| LowerError {
                                    what: "shorthand field with no matching local".to_string(),
                                    span: field.name,
                                })?;
                            let ty = self.locals[local.0 as usize].ty.clone();
                            self.read_place(Place::local(local), &ty, span)?
                        }
                    };
                    by_name.push((self.text(field.name).to_string(), value));
                }
                let decl_names: Vec<String> = decl_fields
                    .iter()
                    .map(|f| self.meta.item_text(*item, f.name).to_string())
                    .collect();
                let mut ordered = Vec::new();
                for name in &decl_names {
                    let Some(pos) = by_name.iter().position(|(n, _)| n == name) else {
                        return unsupported("struct literal missing a field", span);
                    };
                    ordered.push(by_name.remove(pos).1);
                }
                self.aggregate_to_temp(expr, AggKind::Struct(*item), ordered, span)
            }
            hir::ExprKind::Field { base, name, .. } => {
                let (mut place, peeled) = self.lower_place_autoderef(*base)?;
                let MirTy::Struct(item, _) = peeled else {
                    return unsupported("field access on non-struct (C4.5)", span);
                };
                let ItemKind::Struct { fields, .. } = &self.hir.item(item).kind else {
                    return unsupported("field access on non-struct item", span);
                };
                let name_text = self.text(*name);
                let Some(index) = fields
                    .iter()
                    .position(|f| self.meta.item_text(item, f.name) == name_text)
                else {
                    return unsupported("unknown field", span);
                };
                place.projection.push(Projection::Field(index as u32));
                let field_ty = self.expr_mir_ty(expr)?;
                self.read_place(place, &field_ty, span)
            }
            hir::ExprKind::TupleField { base, index } => {
                let idx: u32 = self.text(*index).parse().map_err(|_| LowerError {
                    what: "bad tuple index".to_string(),
                    span,
                })?;
                let mut place = self.lower_place(*base)?;
                place.projection.push(Projection::Field(idx));
                let field_ty = self.expr_mir_ty(expr)?;
                self.read_place(place, &field_ty, span)
            }
            hir::ExprKind::Index { base, index } => {
                // A1 (CD-031), C4.5e-2: `v[i]` on a Vec is a runtime-checked VecIndexGet (Copy
                // element, V-COPY-1); arrays/slices keep the CheckIndex proof discipline.
                let (peeled, _) = Self::peel_refs(self.expr_mir_ty(*base)?);
                if let MirTy::Core(crate::hir::CoreType::Vec, elem_args) = &peeled {
                    let elem = elem_args.first().cloned().unwrap_or(MirTy::Unit);
                    let recv = self.borrow_vec_receiver(*base, false, elem.clone(), span)?;
                    let idx = self.lower_expr_to_operand(*index)?;
                    let idx = self.widen_index_to_u64(idx, *index, span)?;
                    let dest = self.new_temp(elem.clone());
                    self.emit_runtime_call(
                        RuntimeFn::VecIndexGet,
                        vec![recv, idx],
                        Place::local(dest),
                        span,
                    );
                    return self.read_place(Place::local(dest), &elem, span);
                }
                let place = self.lower_index_place(*base, *index, span)?;
                let elem_ty = self.expr_mir_ty(expr)?;
                self.read_place(place, &elem_ty, span)
            }
            hir::ExprKind::Match { .. } => {
                let ty = self.expr_mir_ty(expr)?;
                let dest = self.new_temp(ty);
                self.lower_match(expr, Some(Place::local(dest)))?;
                let ty = self.locals[dest.0 as usize].ty.clone();
                self.read_place(Place::local(dest), &ty, span)
            }
            // A numeric `as` cast — a checked terminator (all casts are checked; widening
            // never traps, narrowing traps CastFailure on overflow).
            hir::ExprKind::Cast { expr: inner, .. } => {
                let inner = *inner;
                let to = self.expr_mir_ty(expr)?;
                let value = self.lower_expr_to_operand(inner)?;
                self.cast_to_temp(value, to, span)
            }
            // C4.5e-3: the `?` try operator on Option/Result.
            hir::ExprKind::Try(inner) => self.lower_try(*inner, span),
            // A7: `while`/`for` are Unit-typed even in value position — lower for effects,
            // yield Unit.
            hir::ExprKind::While { .. } | hir::ExprKind::For { .. } => {
                self.lower_unit_expr(expr)?;
                Ok(Operand::Const(Constant::Unit))
            }
            // A7: `loop` in value position. A Unit-typed loop lowers as a statement and yields
            // Unit. A non-Unit loop carries its value through `break <value>`: every break
            // writes the result local (the type system guarantees no plain `break` here), and
            // the exit block reads it.
            hir::ExprKind::Loop { body } => {
                let ty = self.expr_mir_ty(expr)?;
                if matches!(ty, MirTy::Unit) {
                    self.lower_unit_expr(expr)?;
                    return Ok(Operand::Const(Constant::Unit));
                }
                let result = self.new_temp(ty.clone());
                let body_block = self.new_block();
                let exit = self.new_block();
                self.terminate(
                    Terminator::Goto { target: body_block },
                    self.info(span),
                    body_block,
                );
                self.loops.push(LoopTargets {
                    continue_target: body_block,
                    break_target: exit,
                    scope_depth: self.scopes.len(),
                    value_target: Some(result),
                });
                self.lower_block_value(*body)?;
                self.loops.pop();
                self.terminate(
                    Terminator::Goto { target: body_block },
                    self.info(span),
                    exit,
                );
                self.read_place(Place::local(result), &ty, span)
            }
            _ => unsupported("expression form (C4.5)", span),
        }
    }

    /// C4.5e-3: lower `e?`. The operand is materialized into a temp (not a scope-registered
    /// local — both switch arms consume it, so no drop elaboration is owed). The Some/Ok
    /// payload becomes the expression's value; None/Err propagates as an early return of the
    /// enclosing function's own Option/Result, after dropping live scopes.
    fn lower_try(&mut self, inner: ExprId, span: Span) -> Result<Operand, LowerError> {
        let inner_ty = self.expr_mir_ty(inner)?;
        // `err_payload_ty` is `None` for `Option`, whose `None` variant carries nothing. It decides
        // the A12 storage-end reason on the propagating path, below.
        let (enum_ref, ok_variant, payload_ty, err_payload_ty) = match &inner_ty {
            MirTy::Enum(er @ EnumRef::CoreOption, args) => (
                *er,
                1u32,
                args.first().cloned().unwrap_or(MirTy::Unit),
                None,
            ),
            MirTy::Enum(er @ EnumRef::CoreResult, args) => (
                *er,
                0u32,
                args.first().cloned().unwrap_or(MirTy::Unit),
                Some(args.get(1).cloned().unwrap_or(MirTy::Unit)),
            ),
            other => {
                return unsupported(format!("`?` on a non-Option/Result type {other:?}"), span)
            }
        };
        // A12: which storage end a path needs, given what it moved out of the scrutinee temp. A
        // non-`Copy` move leaves the storage partially moved; a `Copy` read or no read at all
        // leaves it whole, holding a value whose active variant owns nothing.
        let storage_end_after = |lower: &Self, moved: Option<&MirTy>| match moved {
            Some(ty) if !lower.is_copy(ty) => StorageEnd::Accounted,
            _ => StorageEnd::OwnsNothing,
        };
        // The enclosing function's return type (Local(0)) — the propagated shape.
        let ret_ty = self.locals[0].ty.clone();

        let scrut = self.new_temp(inner_ty.clone());
        let value = self.lower_expr_to_operand(inner)?;
        self.emit(
            Statement::Assign(Place::local(scrut), Rvalue::Use(value)),
            self.info(span),
        );
        let disc = self.new_temp(MirTy::Int64);
        self.emit(
            Statement::Assign(
                Place::local(disc),
                Rvalue::Discriminant(Place::local(scrut)),
            ),
            self.info(span),
        );
        let ok_block = self.new_block();
        let err_block = self.new_block();
        self.terminate(
            Terminator::SwitchInt {
                scrut: Operand::Copy(Place::local(disc)),
                arms: vec![(u128::from(ok_variant), ok_block)],
                otherwise: err_block,
            },
            self.info(span),
            err_block,
        );

        // Err/None path: build the propagated return value, drop scopes, return.
        let propagated = match enum_ref {
            EnumRef::CoreOption => {
                // None (variant 0, no payload).
                Rvalue::Aggregate(AggKind::EnumVariant(EnumRef::CoreOption, 0), Vec::new())
            }
            EnumRef::CoreResult => {
                // Err(payload) — read the Err payload (variant 1, field 0) out of scrut.
                //
                // DEV-125: this said `move` unconditionally. `?` on a `Result<T, E>` with a
                // fieldless (therefore `Copy`) `E` — every provider error enum — moved a value
                // whose type says reading leaves it intact.
                //
                // `storage_end_after` above already branches on `is_copy` of this very type to
                // decide the A12 storage-end reason, so the distinction was present in this
                // function and the operand ignored it.
                let err_ty = err_payload_ty.clone().unwrap_or(MirTy::Unit);
                let err_payload = self.read_place(
                    Place {
                        local: scrut,
                        projection: vec![Projection::VariantField(1, 0)],
                    },
                    &err_ty,
                    span,
                )?;
                Rvalue::Aggregate(
                    AggKind::EnumVariant(EnumRef::CoreResult, 1),
                    vec![err_payload],
                )
            }
            EnumRef::User(_) | EnumRef::CoreOrdering => {
                unreachable!("? only on Option/Result")
            }
        };
        // The propagated value must match the function's return type nominally; both share the
        // logical-enum representation, so the aggregate types against ret_ty directly.
        let _ = &ret_ty;
        self.emit(
            Statement::Assign(Place::local(LocalId(0)), propagated),
            self.info(span),
        );
        // A12: the scrutinee temp is spent — `Err`'s payload was moved into the propagated value,
        // or `None` carried nothing. This path returns, so it is not the one that reused a live
        // slot; it is ended anyway, because leaving one path's storage state to depend on the fact
        // that nothing happens afterwards is how the reused path came to be missed.
        let reason = storage_end_after(self, err_payload_ty.as_ref());
        self.emit_storage_dead(scrut.0, reason, span);
        self.emit_scope_drops_from(0, span);
        // Seal the err block with Return and continue lowering in the Ok/Some block (Return
        // has no CFG edge, so `ok_block` is only the continuation point, not a successor).
        self.terminate(Terminator::Return, self.info(span), ok_block);

        // Ok/Some path: the expression's value is the payload.
        let payload_place = Place {
            local: scrut,
            projection: vec![Projection::VariantField(ok_variant, 0)],
        };
        // A12 / the stark-json requalification finding: **this** is the path that reused a live
        // slot. `?` builds its own scrutinee temp rather than going through `lower_match`, so
        // `consume_variant_payload`'s storage end never covered it — and the Ok path continues
        // executing, so inside a loop the next `?` on the same expression wrote over a partially
        // moved slot and the runtime refused it. `stark-json`'s parser is built on `?` in loops,
        // which is why requalifying it found this and none of the sixteen match shapes did.
        //
        // The payload is materialised into a temp FIRST. `read_place` returns an *operand*, and the
        // move it describes does not happen until the caller consumes it — so ending the storage
        // straight after the call put the storage end BEFORE the move, and the slot was still whole
        // when it ran. Forcing the move into a statement here makes the order the code claims.
        let read = self.read_place(payload_place, &payload_ty, span)?;
        let payload = self.new_temp(payload_ty.clone());
        self.emit(
            Statement::Assign(Place::local(payload), Rvalue::Use(read)),
            self.info(span),
        );
        let reason = storage_end_after(self, Some(&payload_ty));
        self.emit_storage_dead(scrut.0, reason, span);
        self.read_place(Place::local(payload), &payload_ty, span)
    }

    fn lower_lit(&mut self, expr: ExprId, lit: &Lit) -> Result<Operand, LowerError> {
        let span = self.hir.expr(expr).span;
        match lit {
            Lit::Bool(value) => Ok(Operand::Const(Constant::Bool(*value))),
            Lit::Int { base, suffix } => {
                let ty = self.expr_mir_ty(expr)?;
                let value = literal::parse_int_literal(self.text(span), *base, *suffix)
                    .ok_or_else(|| LowerError {
                        what: "unparseable integer literal".to_string(),
                        span,
                    })?;
                Ok(Operand::Const(Constant::Int(value, ty)))
            }
            Lit::Float { suffix } => {
                let ty = self.expr_mir_ty(expr)?;
                let value =
                    literal::parse_float_literal(self.text(span), *suffix).ok_or_else(|| {
                        LowerError {
                            what: "unparseable float literal".to_string(),
                            span,
                        }
                    })?;
                // CD-140: a `Float32` literal is the nearest BINARY32 value, carried in the
                // f64 constant. 07 NUM-FLOAT-LIT-001 converts a decimal literal directly to the
                // DESTINATION format, so `0.1f32` denotes the f32 nearest 0.1, not the f64 one.
                // Storing the f64 made the constant observably wider than its type: `0.1f32 as
                // Float64` yielded `0.1` here and `0.10000000149011612` in the HIR oracle.
                let value = if matches!(ty, MirTy::Float32) {
                    f64::from(value as f32)
                } else {
                    value
                };
                Ok(Operand::Const(Constant::Float(value, ty)))
            }
            // A1 (CD-031): a decoded UTF-8 `&str` literal.
            Lit::Str { .. } => {
                let value = match literal::eval_lit_value(*lit, self.text(span)) {
                    Some(crate::literal::LitValue::Str(s)) => s,
                    _ => {
                        return unsupported("unparseable string literal", span);
                    }
                };
                Ok(Operand::Const(Constant::Str(value)))
            }
            // f-3b: a Char literal is its Unicode scalar codepoint, typed Char.
            Lit::Char => match literal::eval_lit_value(*lit, self.text(span)) {
                Some(crate::literal::LitValue::Char(c)) => Ok(Operand::Const(Constant::Int(
                    i128::from(u32::from(c)),
                    MirTy::Char,
                ))),
                _ => unsupported("unparseable char literal", span),
            },
        }
    }

    fn lower_short_circuit(
        &mut self,
        op: BinOp,
        lhs: ExprId,
        rhs: ExprId,
        span: Span,
    ) -> Result<Operand, LowerError> {
        let dest = self.new_temp(MirTy::Bool);
        let lhs_op = self.lower_expr_to_operand(lhs)?;
        let rhs_block = self.new_block();
        let short_block = self.new_block();
        let join = self.new_block();
        let (on_true, on_false, short_value) = match op {
            BinOp::And => (rhs_block, short_block, false),
            BinOp::Or => (short_block, rhs_block, true),
            _ => unreachable!(),
        };
        self.terminate(
            Terminator::SwitchInt {
                scrut: lhs_op,
                arms: vec![(1, on_true)],
                otherwise: on_false,
            },
            self.synthetic(span, SyntheticKind::ShortCircuit),
            rhs_block,
        );
        let rhs_op = self.lower_expr_to_operand(rhs)?;
        self.emit(
            Statement::Assign(Place::local(dest), Rvalue::Use(rhs_op)),
            self.synthetic(span, SyntheticKind::ShortCircuit),
        );
        self.terminate(
            Terminator::Goto { target: join },
            self.synthetic(span, SyntheticKind::ShortCircuit),
            short_block,
        );
        self.emit(
            Statement::Assign(
                Place::local(dest),
                Rvalue::Use(Operand::Const(Constant::Bool(short_value))),
            ),
            self.synthetic(span, SyntheticKind::ShortCircuit),
        );
        self.terminate(
            Terminator::Goto { target: join },
            self.synthetic(span, SyntheticKind::ShortCircuit),
            join,
        );
        Ok(Operand::Copy(Place::local(dest)))
    }

    fn lower_arith_operands(
        &mut self,
        op: BinOp,
        lhs: Operand,
        rhs: Operand,
        operand_ty: &MirTy,
        span: Span,
    ) -> Result<Operand, LowerError> {
        let is_float = matches!(operand_ty, MirTy::Float32 | MirTy::Float64);
        if is_float {
            match op {
                // CD-139: NUM-FLOAT-OP-001 makes ALL FIVE of these TOTAL. Division by zero
                // yields the IEEE infinity or NaN rather than trapping, and `%` with a zero
                // divisor yields NaN, so `Div`/`Rem` owe no check and join the others here.
                // They previously lowered to `CheckedOp::FloatDiv`/`FloatRem` with a
                // `DivideByZero` trap under CD-006 — an owner ruling on `03-Type-System.md`
                // wording that WP-C2.9 replaced nine hours later with the explicit paired rules
                // NUM-INT-DIV-001 (integer zero division traps) and NUM-FLOAT-OP-001 (floating
                // zero division does not). The owner has ruled CD-006 SUPERSEDED by succession
                // of authority, not reversed on its merits.
                BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Rem => {
                    let mir_op = match op {
                        BinOp::Add => MirBinOp::FloatAdd,
                        BinOp::Sub => MirBinOp::FloatSub,
                        BinOp::Mul => MirBinOp::FloatMul,
                        BinOp::Div => MirBinOp::FloatDiv,
                        BinOp::Rem => MirBinOp::FloatRem,
                        _ => unreachable!(),
                    };
                    let dest = self.new_temp(operand_ty.clone());
                    self.emit(
                        Statement::Assign(Place::local(dest), Rvalue::BinOp(mir_op, lhs, rhs)),
                        self.info(span),
                    );
                    return Ok(Operand::Copy(Place::local(dest)));
                }
                _ => unreachable!(),
            }
        }
        // A5: pure (non-trapping) bitwise ops.
        if let Some(mir_op) = match op {
            BinOp::BitAnd => Some(MirBinOp::BitAnd),
            BinOp::BitOr => Some(MirBinOp::BitOr),
            BinOp::BitXor => Some(MirBinOp::BitXor),
            _ => None,
        } {
            let dest = self.new_temp(operand_ty.clone());
            self.emit(
                Statement::Assign(Place::local(dest), Rvalue::BinOp(mir_op, lhs, rhs)),
                self.info(span),
            );
            return Ok(Operand::Copy(Place::local(dest)));
        }
        let (checked, category) = match op {
            BinOp::Add => (CheckedOp::Add, TrapCategory::IntegerOverflow),
            BinOp::Sub => (CheckedOp::Sub, TrapCategory::IntegerOverflow),
            BinOp::Mul => (CheckedOp::Mul, TrapCategory::IntegerOverflow),
            // The terminator carries the DEFAULT category only. NUM-INT-DIV-001 gives `/` and `%`
            // a second failure — signed `MIN op -1` — which the checked evaluation overrides to
            // `IntegerOverflow`, exactly as a bad shift count overrides to `InvalidShift` below.
            BinOp::Div => (CheckedOp::Div, TrapCategory::DivideByZero),
            BinOp::Rem => (CheckedOp::Rem, TrapCategory::DivideByZero),
            // A5: shifts trap on an invalid count / non-representable left shift; `**` traps on
            // overflow or a negative exponent. Both surface as IntegerOverflow (matching the
            // oracle's category — the differential compares category, not message).
            BinOp::Shl => (CheckedOp::Shl, TrapCategory::IntegerOverflow),
            BinOp::Shr => (CheckedOp::Shr, TrapCategory::IntegerOverflow),
            BinOp::Pow => (CheckedOp::Pow, TrapCategory::IntegerOverflow),
            _ => unreachable!(),
        };
        self.checked_to_temp(checked, vec![lhs, rhs], operand_ty.clone(), category, span)
    }

    fn checked_to_temp(
        &mut self,
        op: CheckedOp,
        args: Vec<Operand>,
        ty: MirTy,
        category: TrapCategory,
        span: Span,
    ) -> Result<Operand, LowerError> {
        let dest = self.new_temp(ty);
        let after = self.new_block();
        self.terminate(
            Terminator::Checked {
                op,
                args,
                dest,
                target: after,
                trap: TrapInfo {
                    category,
                    source: self.info(span),
                },
            },
            self.info(span),
            after,
        );
        Ok(Operand::Copy(Place::local(dest)))
    }

    fn aggregate_to_temp(
        &mut self,
        expr: ExprId,
        kind: AggKind,
        operands: Vec<Operand>,
        span: Span,
    ) -> Result<Operand, LowerError> {
        let ty = self.expr_mir_ty(expr)?;
        let dest = self.new_temp(ty);
        self.emit(
            Statement::Assign(Place::local(dest), Rvalue::Aggregate(kind, operands)),
            self.info(span),
        );
        let ty = self.locals[dest.0 as usize].ty.clone();
        self.read_place(Place::local(dest), &ty, span)
    }

    /// Peel reference layers off a type (for nominal lookup / field access through `&self`,
    /// per Core's one-level method auto-deref — we peel all layers since Core never nests refs).
    fn peel_refs(ty: MirTy) -> (MirTy, u32) {
        let mut layers = 0;
        let mut t = ty;
        while let MirTy::Ref { inner, .. } = t {
            t = *inner;
            layers += 1;
        }
        (t, layers)
    }

    /// WP-C6.1f-b2: the built-in **expected-type reference weakening**, `&mut T` → `&T`.
    ///
    /// 03-Type-System "Reference Coercions" makes `&mut T -> &T` normative, and a function
    /// parameter, an annotated `let`, an assignment destination and a return position are all
    /// **expected-type boundaries** — TYPE-METHOD-002 excludes argument-position auto-borrow,
    /// auto-dereference and *user-defined* coercion, not this fixed built-in set.
    ///
    /// Per the C6.1f-b2 ruling the `&mut` is **re-borrowed, not moved**, so the source stays
    /// usable afterwards. Each re-borrow is a *temporary* borrow that ends with its statement
    /// (03, "References and Lifetimes" rule 4), so no borrow duration changes and Core v1's
    /// lexical rule is untouched — the same property b1 relied on.
    ///
    /// It also covers the **same-mutability** case: passing `&mut T` where `&mut T` is expected
    /// must re-borrow too, or the source reference would be *moved* and a second use would fail
    /// V-MOVE-1 — the MIR-level twin of the E0100 that borrowck used to raise.
    ///
    /// A no-op unless the operand really is a `&mut T` place read at a reference-typed boundary.
    fn weaken_ref_to(
        &mut self,
        op: Operand,
        expected: &MirTy,
        span: Span,
    ) -> Result<Operand, LowerError> {
        let MirTy::Ref {
            mutable: want_mut,
            inner: want,
        } = expected
        else {
            return Ok(op);
        };
        let (Operand::Copy(place) | Operand::Move(place)) = &op else {
            return Ok(op);
        };
        // Only a whole reference local is handled; a projected place is not a `&mut T` value.
        if !place.projection.is_empty() {
            return Ok(op);
        }
        // **DEV-133: unsize `&[T; N]` to `&[T]`.**
        //
        // The checker accepts `let s: &[UInt8] = &[b];` and the oracle executes it, but lowering
        // emitted the array reference unchanged, so the assignment's declared and actual types
        // differed by unsizing alone and MIR-0004 rejected it. Accepted-but-unbuildable — the same
        // class as DEV-132, a different mechanism: that one failed to preserve place context
        // through indexing, this one omits a coercion outright.
        //
        // Handled here rather than at a new seam because this function is ALREADY the place where
        // an operand is coerced to an expected reference type — it does `&mut T` -> `&T` — and all
        // six coercion sites (let, call argument, receiver, return, return-expression, assignment
        // RHS) route through it. A separate hook would have to be added to each, and whichever site
        // was forgotten would keep this defect.
        //
        // `SliceNew` is the existing primitive and already accepts an `&[T; N]` receiver: no new
        // `RuntimeFn`, no new `MirTy`, no amendment. Bounds are the whole array, exclusive.
        // The local's type is cloned out before any `&mut self` call below: holding a borrow into
        // `self.locals` across `new_temp`/`emit_runtime_call` does not compile.
        let have_ty = self.locals[place.local.0 as usize].ty.clone();
        if let MirTy::Slice(want_elem) = want.as_ref() {
            if let MirTy::Ref {
                mutable: have_mut,
                inner: have,
            } = &have_ty
            {
                if let MirTy::Array(have_elem, len) = have.as_ref() {
                    if have_elem == want_elem && !*want_mut && !*have_mut {
                        let slice_ty = MirTy::Ref {
                            mutable: false,
                            inner: Box::new(MirTy::Slice(want_elem.clone())),
                        };
                        let dest = self.new_temp(slice_ty.clone());
                        let len = *len;
                        self.emit_runtime_call(
                            RuntimeFn::SliceNew,
                            vec![
                                op.clone(),
                                Operand::Const(Constant::Int(0, MirTy::UInt64)),
                                Operand::Const(Constant::Int(len as i128, MirTy::UInt64)),
                                Operand::Const(Constant::Bool(false)),
                            ],
                            Place::local(dest),
                            span,
                        );
                        return self.read_place(Place::local(dest), &slice_ty, span);
                    }
                }
            }
        }
        let have = match &self.locals[place.local.0 as usize].ty {
            MirTy::Ref {
                mutable: true,
                inner,
            } => inner.clone(),
            _ => return Ok(op),
        };
        if have != *want {
            return Ok(op);
        }
        let mut deref = place.clone();
        deref.projection.push(Projection::Deref);
        let ref_ty = MirTy::Ref {
            mutable: *want_mut,
            inner: want.clone(),
        };
        let temp = self.new_temp(ref_ty.clone());
        self.emit(
            Statement::Assign(
                Place::local(temp),
                Rvalue::RefOf {
                    mutable: *want_mut,
                    place: deref,
                },
            ),
            self.info(span),
        );
        self.read_place(Place::local(temp), &ref_ty, span)
    }

    /// A place for `base`, auto-dereffed: if `base`'s type is a reference, the returned place
    /// carries the needed `Deref` projections and the returned type is the referent.
    fn lower_place_autoderef(&mut self, base: ExprId) -> Result<(Place, MirTy), LowerError> {
        let base_ty = self.expr_mir_ty(base)?;
        let (peeled, layers) = Self::peel_refs(base_ty.clone());
        // WP-C6.1f "returning a reference": the base may be a non-place VALUE — most importantly a
        // call returning a reference (`pick(&a, &b).field`, `f(&x).method()`). A returned
        // reference is a first-class value now, so projecting through one must work without an
        // intervening `let`. Materialise the value into a temp and project through that, exactly as
        // the `RefOf` and method-receiver paths already do for non-place operands. The temp carries
        // the reference; the derefs below reach its referent.
        let mut place = match self.lower_place(base) {
            Ok(place) => place,
            Err(_) => {
                let value = self.lower_expr_to_operand(base)?;
                let temp = self.new_temp(base_ty);
                self.emit(
                    Statement::Assign(Place::local(temp), Rvalue::Use(value)),
                    self.info(self.hir.expr(base).span),
                );
                Place::local(temp)
            }
        };
        for _ in 0..layers {
            place.projection.push(Projection::Deref);
        }
        Ok((place, peeled))
    }

    /// Lower an expression used as an assignable/projectable place.
    fn lower_place(&mut self, expr: ExprId) -> Result<Place, LowerError> {
        let span = self.hir.expr(expr).span;
        match &self.hir.expr(expr).kind {
            hir::ExprKind::Path { res, .. } => match res {
                Res::Local(local) | Res::SelfValue(local) => {
                    let mir_local = *self.local_map.get(&local.0).ok_or_else(|| LowerError {
                        what: "unknown local in place position".to_string(),
                        span,
                    })?;
                    Ok(Place::local(mir_local))
                }
                _ => unsupported("place form (C4.5)", span),
            },
            hir::ExprKind::Field { base, name, .. } => {
                let (mut place, peeled) = self.lower_place_autoderef(*base)?;
                let MirTy::Struct(item, _) = peeled else {
                    return unsupported("field place on non-struct (C4.5)", span);
                };
                let ItemKind::Struct { fields, .. } = &self.hir.item(item).kind else {
                    return unsupported("field place on non-struct item", span);
                };
                let name_text = self.text(*name);
                let Some(index) = fields
                    .iter()
                    .position(|f| self.meta.item_text(item, f.name) == name_text)
                else {
                    return unsupported("unknown field", span);
                };
                place.projection.push(Projection::Field(index as u32));
                Ok(place)
            }
            hir::ExprKind::TupleField { base, index } => {
                let idx: u32 = self.text(*index).parse().map_err(|_| LowerError {
                    what: "bad tuple index".to_string(),
                    span,
                })?;
                let mut place = self.lower_place(*base)?;
                place.projection.push(Projection::Field(idx));
                Ok(place)
            }
            hir::ExprKind::Index { base, index } => self.lower_index_place(*base, *index, span),
            hir::ExprKind::Unary {
                op: UnOp::Deref,
                operand,
            } => {
                let mut place = self.lower_place(*operand)?;
                place.projection.push(Projection::Deref);
                Ok(place)
            }
            other => unsupported(
                format!("place expression (C4.5): {}", expr_kind_name(other)),
                span,
            ),
        }
    }

    /// 0.1-A6 (A4 slicing): `&base[range]` — a shared slice view over an Array/Vec/slice
    /// referent via `SliceNew`, which traps IndexOutOfBounds on a negative, inverted, or
    /// out-of-range bound (06-Standard-Library behavioral requirement). Evaluation order:
    /// base, then the range (lo before hi via the A4-2a range-tuple lowering).
    fn lower_make_slice(
        &mut self,
        base: ExprId,
        index: ExprId,
        mutable: bool,
        span: Span,
    ) -> Result<Operand, LowerError> {
        let (peeled, layers) = Self::peel_refs(self.expr_mir_ty(base)?);
        let elem = match &peeled {
            MirTy::Array(elem, _) | MirTy::Slice(elem) => (**elem).clone(),
            MirTy::Core(crate::hir::CoreType::Vec, args) => {
                args.first().cloned().unwrap_or(MirTy::Unit)
            }
            other => return unsupported(format!("slicing {other:?}"), span),
        };
        // Base reference: pass an existing reference through (a `&[T]` re-slice or `&Vec`),
        // else borrow the owned Array/Vec place (shared).
        // 0.1-A8: an EXCLUSIVE view borrows the base exclusively; a shared one borrows shared.
        let base_ref = if layers > 0 {
            self.lower_expr_to_operand(base)?
        } else {
            let place = self.place_or_temp(base, &peeled, span)?;
            let ref_ty = MirTy::Ref {
                mutable,
                inner: Box::new(peeled.clone()),
            };
            let temp = self.new_temp(ref_ty.clone());
            self.emit(
                Statement::Assign(Place::local(temp), Rvalue::RefOf { mutable, place }),
                self.info(span),
            );
            self.read_place(Place::local(temp), &ref_ty, span)?
        };
        // The range tuple (start, end, inclusive) — materialize once, read its fields.
        let range_ty = self.expr_mir_ty(index)?;
        let bound_ty = match &range_ty {
            MirTy::Tuple(fields) => fields.first().cloned().unwrap_or(MirTy::Int32),
            other => return unsupported(format!("slice index is not a range: {other:?}"), span),
        };
        let range_op = self.lower_expr_to_operand(index)?;
        let range_local = self.new_temp(range_ty);
        self.emit(
            Statement::Assign(Place::local(range_local), Rvalue::Use(range_op)),
            self.info(span),
        );
        let field = |i: u32| Place {
            local: range_local,
            projection: vec![Projection::Field(i)],
        };
        let _ = &bound_ty;
        let slice_ty = MirTy::Ref {
            mutable,
            inner: Box::new(MirTy::Slice(Box::new(elem))),
        };
        let dest = self.new_temp(slice_ty.clone());
        self.emit_runtime_call(
            if mutable {
                RuntimeFn::SliceNewMut
            } else {
                RuntimeFn::SliceNew
            },
            vec![
                base_ref,
                Operand::Copy(field(0)),
                Operand::Copy(field(1)),
                Operand::Copy(field(2)),
            ],
            Place::local(dest),
            span,
        );
        self.read_place(Place::local(dest), &slice_ty, span)
    }

    /// C4.5b-1: `base[index]` place with the CE3 proof-token discipline. Evaluation order:
    /// base before index (CD-007). Only fixed-length arrays here — Vec indexing stays on the
    /// runtime surface (mutable length, contract §6) and slices arrive with references
    /// (C4.5b-2).
    fn lower_index_place(
        &mut self,
        base: ExprId,
        index: ExprId,
        span: Span,
    ) -> Result<Place, LowerError> {
        let base_ty = self.expr_mir_ty(base)?;
        // 0.1-A6 (A4 slicing): `s[i]` on a `&[T]` — the indexable place is the slice referent
        // (`s` + Deref); CheckIndex checks against the VIEW length at runtime.
        if let MirTy::Ref { inner, .. } = &base_ty {
            if matches!(**inner, MirTy::Slice(_)) {
                let mut place = self.place_or_temp(base, &base_ty, span)?;
                place.projection.push(Projection::Deref);
                let index_op = self.lower_expr_to_operand(index)?;
                self.locals.push(LocalDecl {
                    ty: MirTy::Int64,
                    kind: LocalKind::IndexProof,
                });
                let proof = LocalId((self.locals.len() - 1) as u32);
                let after = self.new_block();
                self.terminate(
                    Terminator::Checked {
                        op: CheckedOp::CheckIndex,
                        args: vec![Operand::Copy(place.clone()), index_op],
                        dest: proof,
                        target: after,
                        trap: TrapInfo {
                            category: TrapCategory::IndexOutOfBounds,
                            source: self.info(span),
                        },
                    },
                    self.info(span),
                    after,
                );
                place.projection.push(Projection::Index(proof));
                return Ok(place);
            }
        }
        // **DEV-132: `v[i]` in PLACE position borrows the element; it does not read it.**
        //
        // This arm used to fall through to the refusal below, so `&v[i].field` was lowered by the
        // VALUE path instead — `VecIndexGet` into a temp, then project the field off it. That is a
        // by-value read of the element, so V-COPY-1 required `Copy` and MIR-0016 refused every
        // `Vec<NonCopy>`. The refusal was correct for the MIR emitted; emitting it was the defect.
        // A borrow never needed the element by value.
        //
        // `VecGetRef` is the existing primitive for exactly this — `(&Vec<T>, u64) -> Option<&T>`,
        // verified, no `Copy` requirement, already what `v.get(i)` lowers to. No new runtime
        // function, no new `MirTy`, no amendment.
        //
        // **The `Option` is unwrapped by TRAPPING, not by yielding `None`.** `get` and `[]` differ
        // precisely there: `get` returns `None` out of bounds, `v[i]` traps. Lowering the place
        // through `get`'s primitive must not quietly inherit `get`'s out-of-bounds behaviour, so
        // the `None` edge terminates in `Trap(IndexOutOfBounds)` — the same category the array and
        // slice paths raise, leaving observable out-of-bounds behaviour unchanged.
        //
        // The resulting place derefs a SHARED reference, so a write through it is already refused
        // by V-REF-1 (MIR-0014). Making a borrow representable here does not make it assignable,
        // which is the property the negative controls pin.
        // Peeled, because the base is as often a `&Vec<T>` PARAMETER as an owned local — which is
        // the form `stark-mime` uses and the form the first cut of this arm missed, matching the
        // unpeeled type and falling through to the by-value path for exactly the case that
        // motivated the change. `borrow_vec_receiver` peels again for itself.
        let (peeled_base, _) = Self::peel_refs(base_ty.clone());
        if let MirTy::Core(crate::hir::CoreType::Vec, elem_args) = &peeled_base {
            let elem = elem_args.first().cloned().unwrap_or(MirTy::Unit);
            let recv = self.borrow_vec_receiver(base, false, elem.clone(), span)?;
            let index_op = self.lower_expr_to_operand(index)?;
            let index_op = self.widen_index_to_u64(index_op, index, span)?;

            let elem_ref = MirTy::Ref {
                mutable: false,
                inner: Box::new(elem),
            };
            let opt_ty = MirTy::Enum(EnumRef::CoreOption, vec![elem_ref]);
            let opt = self.new_temp(opt_ty);
            self.emit_runtime_call(
                RuntimeFn::VecGetRef,
                vec![recv, index_op],
                Place::local(opt),
                span,
            );

            let disc = self.new_temp(MirTy::Int64);
            self.emit(
                Statement::Assign(Place::local(disc), Rvalue::Discriminant(Place::local(opt))),
                self.info(span),
            );
            let in_bounds = self.new_block();
            let out_of_bounds = self.new_block();
            self.terminate(
                Terminator::SwitchInt {
                    scrut: Operand::Copy(Place::local(disc)),
                    arms: vec![(1, in_bounds)],
                    otherwise: out_of_bounds,
                },
                self.info(span),
                out_of_bounds,
            );
            self.terminate(
                Terminator::Trap {
                    info: TrapInfo {
                        category: TrapCategory::IndexOutOfBounds,
                        source: self.info(span),
                    },
                    message: None,
                },
                self.info(span),
                in_bounds,
            );

            return Ok(Place {
                local: opt,
                projection: vec![Projection::VariantField(1, 0), Projection::Deref],
            });
        }
        if !matches!(base_ty, MirTy::Array(..)) {
            return unsupported(format!("indexing {base_ty:?} (slices are C4.5b-2)"), span);
        }
        let base_place = match self.lower_place(base) {
            Ok(place) => place,
            Err(_) => {
                let value = self.lower_expr_to_operand(base)?;
                let temp = self.new_temp(base_ty.clone());
                self.emit(
                    Statement::Assign(Place::local(temp), Rvalue::Use(value)),
                    self.info(span),
                );
                Place::local(temp)
            }
        };
        let index_op = self.lower_expr_to_operand(index)?;
        self.locals.push(LocalDecl {
            ty: MirTy::Int64,
            kind: LocalKind::IndexProof,
        });
        let proof = LocalId((self.locals.len() - 1) as u32);
        let after = self.new_block();
        self.terminate(
            Terminator::Checked {
                op: CheckedOp::CheckIndex,
                args: vec![Operand::Copy(base_place.clone()), index_op],
                dest: proof,
                target: after,
                trap: TrapInfo {
                    category: TrapCategory::IndexOutOfBounds,
                    source: self.info(span),
                },
            },
            self.info(span),
            after,
        );
        let mut place = base_place;
        place.projection.push(Projection::Index(proof));
        Ok(place)
    }

    // ---- calls ----

    fn lower_call(&mut self, expr: ExprId, dest: Option<Place>) -> Result<(), LowerError> {
        let span = self.hir.expr(expr).span;
        let hir::ExprKind::Call { callee, args } = &self.hir.expr(expr).kind else {
            return unsupported("not a call", span);
        };
        let callee = *callee;
        let args = args.clone();

        // Resolve destination (unit calls get a throwaway unit temp).
        let dest = match dest {
            Some(place) => place,
            None => Place::local(self.new_temp(MirTy::Unit)),
        };

        // C4.5a: method call — `receiver.method(args)`.
        if let hir::ExprKind::Field { base, name, .. } = &self.hir.expr(callee).kind {
            let base = *base;
            let name_span = *name;
            return self.lower_method_call(base, name_span, &args, dest, span, expr);
        }

        match &self.hir.expr(callee).kind {
            hir::ExprKind::Path { res, .. } => match res {
                // 0.1-A13 (WP-C7.9 Packet D): `eprint`/`eprintln` join this arm rather than getting
                // one of their own. They are the same operation on a different stream — same
                // arity, same type dispatch, same `Display` invocation, same composite
                // decomposition — and the channel is applied where an operation becomes a call
                // (`on_current_channel`). Before this they had no lowering at all, so MIR refused
                // every program that used them and native emitted nothing.
                Res::Builtin(
                    builtin @ (Builtin::Println
                    | Builtin::Print
                    | Builtin::Eprintln
                    | Builtin::Eprint),
                ) => {
                    if args.len() != 1 {
                        return unsupported("print/println arity", span);
                    }
                    let arg_ty = self.expr_mir_ty(args[0])?;
                    let is_println = matches!(builtin, Builtin::Println | Builtin::Eprintln);
                    let restore = self.out_channel;
                    self.out_channel = match builtin {
                        Builtin::Eprint | Builtin::Eprintln => OutChannel::Stderr,
                        _ => OutChannel::Stdout,
                    };
                    let lowered = self.lower_output_call(args[0], arg_ty, is_println, dest, span);
                    self.out_channel = restore;
                    lowered
                }
                Res::Builtin(ctor @ (Builtin::Some | Builtin::Ok | Builtin::Err)) => {
                    let (enum_ref, variant) = match ctor {
                        Builtin::Some => (EnumRef::CoreOption, 1),
                        Builtin::Ok => (EnumRef::CoreResult, 0),
                        Builtin::Err => (EnumRef::CoreResult, 1),
                        _ => unreachable!(),
                    };
                    let ops = args
                        .iter()
                        .map(|&a| self.lower_expr_to_operand(a))
                        .collect::<Result<Vec<_>, _>>()?;
                    self.emit(
                        Statement::Assign(
                            dest,
                            Rvalue::Aggregate(AggKind::EnumVariant(enum_ref, variant), ops),
                        ),
                        self.info(span),
                    );
                    Ok(())
                }
                // A1 (CD-031): owned String construction.
                Res::Builtin(Builtin::StringNew) => {
                    self.emit_runtime_call(RuntimeFn::StringNew, vec![], dest, span);
                    Ok(())
                }
                Res::Builtin(Builtin::StringFrom) => {
                    let ops = args
                        .iter()
                        .map(|&a| self.lower_expr_to_operand(a))
                        .collect::<Result<Vec<_>, _>>()?;
                    self.emit_runtime_call(RuntimeFn::StringFromStr, ops, dest, span);
                    Ok(())
                }
                Res::Builtin(Builtin::CharFromU32) => {
                    let ops = args
                        .iter()
                        .map(|&a| self.lower_expr_to_operand(a))
                        .collect::<Result<Vec<_>, _>>()?;
                    self.emit_runtime_call(RuntimeFn::CharFromU32, ops, dest, span);
                    Ok(())
                }
                // A1 (CD-031), C4.5e-2: Vec construction.
                // 0.1-A7 (WP-C4.7-6.1): `Box::new(v)` moves `v` into a fresh allocation;
                // `Box::into_inner(b)` consumes the box and transfers the value out WITHOUT
                // dropping it, releasing the allocation. Both consume their argument exactly
                // once, so ordinary operand lowering (which moves a non-Copy value) is correct
                // and no drop is owed on either side.
                Res::Builtin(Builtin::BoxNew) => {
                    if args.len() != 1 {
                        return unsupported("Box::new arity", span);
                    }
                    let value = self.lower_expr_to_operand(args[0])?;
                    self.emit_runtime_call(RuntimeFn::BoxNew, vec![value], dest, span);
                    Ok(())
                }
                Res::Builtin(Builtin::BoxIntoInner) => {
                    if args.len() != 1 {
                        return unsupported("Box::into_inner arity", span);
                    }
                    let boxed = self.lower_expr_to_operand(args[0])?;
                    self.emit_runtime_call(RuntimeFn::BoxIntoInner, vec![boxed], dest, span);
                    Ok(())
                }
                Res::Builtin(Builtin::VecNew) => {
                    self.emit_runtime_call(RuntimeFn::VecNew, vec![], dest, span);
                    Ok(())
                }
                Res::Builtin(Builtin::VecWithCapacity) => {
                    let ops = args
                        .iter()
                        .map(|&a| self.lower_expr_to_operand(a))
                        .collect::<Result<Vec<_>, _>>()?;
                    self.emit_runtime_call(RuntimeFn::VecWithCapacity, ops, dest, span);
                    Ok(())
                }
                // 0.1-A3 (f-3a): HashMap construction. User-Drop K/V excluded at method
                // dispatch (the constructor's dest type is checked there on first use).
                Res::Builtin(Builtin::HashMapNew) => {
                    self.emit_runtime_call(RuntimeFn::HashMapNew, vec![], dest, span);
                    Ok(())
                }
                // DEV-116: the same for `HashSet::new`. Element restrictions are enforced at method
                // dispatch, where the element type is known from the receiver.
                Res::Builtin(Builtin::HashSetNew) => {
                    self.emit_runtime_call(RuntimeFn::HashSetNew, vec![], dest, span);
                    Ok(())
                }
                // A1 (CD-031): `panic(msg)` → an unconditional Trap carrying the message.
                Res::Builtin(Builtin::Panic) => {
                    let message = match args.first() {
                        Some(&arg) => Some(self.str_operand_for(arg, span)?),
                        None => None,
                    };
                    let info = self.info(span);
                    let dead = self.new_block();
                    self.terminate(
                        Terminator::Trap {
                            info: TrapInfo {
                                category: TrapCategory::Panic,
                                source: info,
                            },
                            message,
                        },
                        info,
                        dead,
                    );
                    Ok(())
                }
                // A1: `assert(cond)` → trap AssertFailure when the condition is false.
                // f-3b: `assert_eq(a, b)` / `assert_ne(a, b)` on comparable scalars — compare,
                // trap AssertFailure on mismatch. The trap carries no message: the comparator
                // matches compiler-generated traps by category fragment; the oracle's
                // formatted left/right message is a diagnostic nicety (recorded cosmetic gap).
                Res::Builtin(kind @ (Builtin::AssertEq | Builtin::AssertNe)) => {
                    if args.len() != 2 {
                        return unsupported("assert_eq/assert_ne arity", span);
                    }
                    let lhs_ty = self.expr_mir_ty(args[0])?;
                    let cond = if Self::is_text_ty(&lhs_ty) {
                        self.lower_string_comparison(BinOp::Eq, args[0], args[1], span)?
                    } else {
                        if ty_mentions_user_nominal(&lhs_ty) {
                            return unsupported(
                                "assert_eq/ne on a user-defined type dispatches through its \
                                 Eq impl (a later increment)",
                                span,
                            );
                        }
                        let a = self.lower_expr_to_operand(args[0])?;
                        let b = self.lower_expr_to_operand(args[1])?;
                        let eq = self.new_temp(MirTy::Bool);
                        self.emit(
                            Statement::Assign(Place::local(eq), Rvalue::BinOp(MirBinOp::Eq, a, b)),
                            self.info(span),
                        );
                        Operand::Copy(Place::local(eq))
                    };
                    // assert_eq passes on equal (1); assert_ne passes on unequal (0).
                    let pass_key = if matches!(kind, Builtin::AssertEq) {
                        1
                    } else {
                        0
                    };
                    let info = self.info(span);
                    let ok_block = self.new_block();
                    let fail_block = self.new_block();
                    self.terminate(
                        Terminator::SwitchInt {
                            scrut: cond,
                            arms: vec![(pass_key, ok_block)],
                            otherwise: fail_block,
                        },
                        info,
                        fail_block,
                    );
                    self.terminate(
                        Terminator::Trap {
                            info: TrapInfo {
                                category: TrapCategory::AssertFailure,
                                source: info,
                            },
                            message: None,
                        },
                        info,
                        ok_block,
                    );
                    self.emit(
                        Statement::Assign(dest, Rvalue::Use(Operand::Const(Constant::Unit))),
                        info,
                    );
                    Ok(())
                }
                Res::Builtin(Builtin::Assert) => {
                    if args.len() != 1 {
                        return unsupported("assert arity", span);
                    }
                    let cond = self.lower_expr_to_operand(args[0])?;
                    let info = self.info(span);
                    let ok_block = self.new_block();
                    let fail_block = self.new_block();
                    self.terminate(
                        Terminator::SwitchInt {
                            scrut: cond,
                            arms: vec![(1, ok_block)],
                            otherwise: fail_block,
                        },
                        info,
                        fail_block,
                    );
                    self.terminate(
                        Terminator::Trap {
                            info: TrapInfo {
                                category: TrapCategory::AssertFailure,
                                source: info,
                            },
                            message: None,
                        },
                        info,
                        ok_block,
                    );
                    self.emit(
                        Statement::Assign(dest, Rvalue::Use(Operand::Const(Constant::Unit))),
                        info,
                    );
                    Ok(())
                }
                // C4.5d: explicit early destruction — `drop(x)` moves the value and runs
                // its glue immediately (no-op for non-droppable types).
                Res::Builtin(Builtin::Drop) => {
                    if args.len() != 1 {
                        return unsupported("drop() arity", span);
                    }
                    let ty = self.expr_mir_ty(args[0])?;
                    let op = self.lower_expr_to_operand(args[0])?;
                    if self.ty_needs_drop(&ty, span)? {
                        self.discover_drop_impls(&ty)?;
                        let tmp = self.new_temp(ty);
                        self.emit(
                            Statement::Assign(Place::local(tmp), Rvalue::Use(op)),
                            self.info(span),
                        );
                        self.emit_temp_drop(tmp, span);
                    }
                    self.emit(
                        Statement::Assign(dest, Rvalue::Use(Operand::Const(Constant::Unit))),
                        self.info(span),
                    );
                    Ok(())
                }
                // MIR amendment A4 (CD-036): `size_of::<T>()` / `align_of::<T>()` are
                // **target-layout queries** (06-Standard-Library; LAYOUT-QUERY-001), so the
                // queried type must SURVIVE into MIR — a backend answers them from its own
                // target layout, and it cannot do that from a MIR that erased `T`. (WP-C4.6
                // A4-1 emitted `Const 8` here, which the C4.7 audit found type-erasing.)
                // `hir_field_ty` applies the active `param_subst`, so `size_of::<T>()` inside a
                // monomorphised generic body records the INSTANTIATION's concrete type.
                Res::Builtin(builtin @ (Builtin::SizeOf | Builtin::AlignOf)) => {
                    let hir::ExprKind::Path {
                        turbofish: Some(generic_args),
                        ..
                    } = &self.hir.expr(callee).kind
                    else {
                        // The checker requires exactly one generic argument (T is not
                        // inferable), so this is unreachable for checked programs.
                        return unsupported("layout query without a type argument", span);
                    };
                    let [hir::GenericArg::Type(ty_id)] = generic_args.args.as_slice() else {
                        return unsupported("layout query type argument form", span);
                    };
                    let queried = self.hir_field_ty(*ty_id)?;
                    let kind = match builtin {
                        Builtin::SizeOf => LayoutKind::SizeOf,
                        _ => LayoutKind::AlignOf,
                    };
                    self.emit(
                        Statement::Assign(dest, Rvalue::LayoutQuery { kind, ty: queried }),
                        self.info(span),
                    );
                    Ok(())
                }
                Res::Builtin(_) => unsupported("builtin (C4.5)", span),
                Res::Item(item) => {
                    // C4.5c: generic callees resolve to a concrete monomorphised instance
                    // through the checker's recorded instantiation.
                    let item = *item;
                    // WP-C7.8.8 step 6: a synthesized provider binding is an ordinary item to
                    // everything before this point -- resolution, type checking and the borrow
                    // checker all saw a normal function. Only here does it become a provider call,
                    // which is what kept the front end free of provider special cases.
                    if let Some(id) = self.provider_call_for(item) {
                        return self.lower_provider_call(id, &args, dest, span);
                    }
                    let instance = self.top_fn_instance(item, callee, span)?;
                    // C6.1f-b2: a parameter is an expected-type boundary, so `&mut T` weakens to
                    // `&T` here. The checker's grounded signature supplies the expected types.
                    // The checker's grounded signature supplies the expected types, resolved under
                    // the CALLEE's own substitution (C6.1f-b2 generic-callee completion).
                    let param_tys =
                        self.callee_param_types(item, &instance.type_args.clone(), span)?;
                    let mut ops = Vec::with_capacity(args.len());
                    for (i, &a) in args.iter().enumerate() {
                        let op = self.lower_expr_to_operand(a)?;
                        ops.push(match param_tys.get(i).and_then(|t| t.clone()) {
                            Some(expected) => self.weaken_ref_to(op, &expected, span)?,
                            None => op,
                        });
                    }
                    let after = self.new_block();
                    self.terminate(
                        Terminator::Call {
                            callee: Callee::Instance(instance),
                            args: ops,
                            dest,
                            target: after,
                        },
                        self.info(span),
                        after,
                    );
                    Ok(())
                }
                Res::Variant(item, variant) => {
                    let ops = args
                        .iter()
                        .map(|&a| self.lower_expr_to_operand(a))
                        .collect::<Result<Vec<_>, _>>()?;
                    self.emit(
                        Statement::Assign(
                            dest,
                            Rvalue::Aggregate(
                                AggKind::EnumVariant(EnumRef::User(*item), *variant),
                                ops,
                            ),
                        ),
                        self.info(span),
                    );
                    Ok(())
                }
                // C4.5a: associated function (`Point::new(3, 4)`). A1: on a GENERIC nominal
                // (`Holder::make(7)`), the instantiation is inferred by unifying the fn's
                // declared signature against the call's concrete argument/result types.
                Res::AssociatedFn(nominal, name_span) => {
                    let nominal = *nominal;
                    let name_text = self.text(*name_span).to_string();
                    // Locate first (empty args), then infer and rebuild the key.
                    let Some((located, _receiver)) =
                        self.find_impl_fn(nominal, &name_text, /*receiverless=*/ true, &[])
                    else {
                        return unsupported(
                            format!("associated function {name_text} not found"),
                            span,
                        );
                    };
                    let nominal_generic = matches!(
                        &self.hir.item(nominal).kind,
                        ItemKind::Struct { generics, .. } | ItemKind::Enum { generics, .. }
                            if !generics.is_empty()
                    );
                    let key = if !nominal_generic {
                        located
                    } else {
                        let FnKey::ImplFn {
                            impl_item, member, ..
                        } = located
                        else {
                            return unsupported(
                                "associated fn on a generic nominal via a trait default",
                                span,
                            );
                        };
                        let type_args = self
                            .infer_assoc_fn_instantiation(impl_item, member, expr, &args, span)?;
                        FnKey::ImplFn {
                            impl_item,
                            member,
                            type_args,
                            // Associated-fn calls with their own generics are handled by the
                            // same recording; none is present when the fn declares none.
                            method_args: Vec::new(),
                        }
                    };
                    let instance = self.instance_from_key(&key)?;
                    self.discovered_callees.push(key);
                    let ops = args
                        .iter()
                        .map(|&a| self.lower_expr_to_operand(a))
                        .collect::<Result<Vec<_>, _>>()?;
                    let after = self.new_block();
                    self.terminate(
                        Terminator::Call {
                            callee: Callee::Instance(instance),
                            args: ops,
                            dest,
                            target: after,
                        },
                        self.info(span),
                        after,
                    );
                    Ok(())
                }
                // Indirect call through a function value (CD-021 item 17).
                Res::Local(_) | Res::SelfValue(_) => {
                    let fn_op = self.lower_expr_to_operand(callee)?;
                    let ops = args
                        .iter()
                        .map(|&a| self.lower_expr_to_operand(a))
                        .collect::<Result<Vec<_>, _>>()?;
                    let after = self.new_block();
                    self.terminate(
                        Terminator::Call {
                            callee: Callee::FnValue(fn_op),
                            args: ops,
                            dest,
                            target: after,
                        },
                        self.info(span),
                        after,
                    );
                    Ok(())
                }
                // WP-C6.2b / DEV-102: fully qualified trait call `Trait::method(&recv, args...)`.
                // 03-Type-System TYPE-METHOD-001: "Trait methods can always be called in
                // fully-qualified function form ... since a method is an ordinary function whose
                // first parameter is the receiver", and the form "bypasses trait-name lookup but
                // still requires a unique coherent impl". Selection is therefore filtered to the
                // NAMED trait (`find_trait_impl_fn`) rather than reusing ordinary method lookup,
                // which takes any in-scope trait supplying the name — that difference is the whole
                // point of the form, since it is how §18's ambiguity case is disambiguated.
                //
                // Because the receiver is written explicitly, no auto-borrow or auto-deref applies
                // (TYPE-METHOD-002 governs `recv.m()` only): every argument, receiver included,
                // lowers as an ordinary operand in source order.
                Res::TraitMember(trait_item, member) => {
                    let (trait_item, member) = (*trait_item, *member);
                    let ItemKind::Trait {
                        items: trait_items, ..
                    } = &self.hir.item(trait_item).kind
                    else {
                        return unsupported("fully qualified call through a non-trait path", span);
                    };
                    let Some(hir::TraitItem::Method { sig, .. }) = trait_items.get(member as usize)
                    else {
                        return unsupported(
                            "fully qualified call to a non-method trait member",
                            span,
                        );
                    };
                    let receiver_kind = sig.receiver.unwrap_or(hir::Receiver::Ref);
                    if sig.receiver.is_none() {
                        // Defensive only: the checker already rejects `Trait::assoc()` with E0005
                        // ("qualified trait method requires a receiver"), since without a receiver
                        // argument the implementing type is not recoverable. Kept so a future
                        // front-end relaxation surfaces here rather than mis-selecting an impl.
                        return unsupported(
                            "fully qualified call to a receiverless trait member",
                            span,
                        );
                    }
                    let name_text = self.meta.item_text(trait_item, sig.name).to_string();
                    let Some((&recv_expr, _rest)) = args.split_first() else {
                        return unsupported(
                            "fully qualified trait call without a receiver argument",
                            span,
                        );
                    };
                    let (peeled, _) = Self::peel_refs(self.expr_mir_ty(recv_expr)?);
                    let (nominal, nominal_args) = match &peeled {
                        MirTy::Struct(item, a) | MirTy::Enum(EnumRef::User(item), a) => {
                            (*item, a.clone())
                        }
                        other => {
                            return unsupported(
                                format!(
                                    "fully qualified trait call on non-nominal receiver {other:?}"
                                ),
                                span,
                            )
                        }
                    };
                    let Some(key) =
                        self.find_trait_impl_fn(nominal, trait_item, &name_text, &nominal_args)
                    else {
                        return unsupported(
                            format!("no impl of the named trait supplies {name_text}"),
                            span,
                        );
                    };
                    // Method-level generic arguments come from the checker's per-call-site
                    // recording, exactly as for `recv.m::<T>(...)` (WP-C4.7-8.4).
                    let method_args = match self.tables.generic_insts.get(&expr) {
                        Some(tys) => tys
                            .iter()
                            .map(|t| self.mir_ty(t, span))
                            .collect::<Result<Vec<_>, _>>()?,
                        None => Vec::new(),
                    };
                    let key = match key {
                        FnKey::ImplFn {
                            impl_item,
                            member,
                            type_args,
                            ..
                        } => FnKey::ImplFn {
                            impl_item,
                            member,
                            type_args,
                            method_args,
                        },
                        FnKey::TraitDefault {
                            trait_item,
                            member,
                            self_item,
                            self_args,
                            ..
                        } => FnKey::TraitDefault {
                            trait_item,
                            member,
                            self_item,
                            self_args,
                            method_args,
                        },
                        other => other,
                    };
                    let instance = self.instance_from_key(&key)?;
                    self.discovered_callees.push(key);
                    // C6.1f-b2: the receiver argument is an expected-type boundary too — a
                    // `&self` method reached through a `&mut` receiver weakens to `&Self`.
                    let recv_expected = MirTy::Ref {
                        mutable: matches!(receiver_kind, hir::Receiver::RefMut),
                        inner: Box::new(peeled.clone()),
                    };
                    let mut ops = Vec::with_capacity(args.len());
                    for (i, &a) in args.iter().enumerate() {
                        let op = self.lower_expr_to_operand(a)?;
                        ops.push(if i == 0 {
                            self.weaken_ref_to(op, &recv_expected, span)?
                        } else {
                            op
                        });
                    }
                    let after = self.new_block();
                    self.terminate(
                        Terminator::Call {
                            callee: Callee::Instance(instance),
                            args: ops,
                            dest,
                            target: after,
                        },
                        self.info(span),
                        after,
                    );
                    Ok(())
                }
                _ => unsupported("callee form (C4.5)", span),
            },
            _ => unsupported("indirect callee expression (C4.5)", span),
        }
    }

    /// C4.5c: resolve a use of a top-level fn item (call callee or fn-value position) to its
    /// concrete monomorphised instance. Generic fns consume the checker's recorded
    /// instantiation for the referencing expression; this body's own substitution is applied
    /// so the resulting type arguments are always fully concrete, even for generic-to-generic
    /// calls whose recorded arguments mention the caller's parameters.
    /// WP-C6.1f-b2: the callee's declared parameter types, ground under the **callee's own**
    /// generic substitution.
    ///
    /// These are the expected types at an argument boundary, which is where `&mut T` -> `&T`
    /// weakening applies. For a generic callee the checker's `fn_types` entry still mentions the
    /// callee's OWN parameters (`Ty::Param("T")`), and the caller's substitution cannot ground them
    /// — resolving against it either fails or, worse, silently picks up a same-named parameter of
    /// the enclosing generic body. The call's concrete arguments are already computed for the
    /// instance, and they are in the callee's generic declaration order, so they are exactly the
    /// substitution needed.
    ///
    /// Resolution stays best-effort per parameter: an entry that still cannot be ground yields
    /// `None`, meaning no weakening for that argument — never a lowering failure. A coercion is an
    /// optimisation of expressiveness, not a correctness requirement, and MIR verification remains
    /// the backstop.
    fn callee_param_types(
        &mut self,
        item: ItemId,
        type_args: &[MirTy],
        span: Span,
    ) -> Result<Vec<Option<MirTy>>, LowerError> {
        let Some((param_tys, _)) = self.tables.fn_types.get(&item).cloned() else {
            return Ok(Vec::new());
        };
        // The generic parameter NAMES belong to the callee's own file, so they are read with
        // `item_text` rather than the current body's file (DEV-101).
        let names: Vec<String> = match &self.hir.item(item).kind {
            ItemKind::Fn(def) => def
                .sig
                .generics
                .iter()
                .map(|g| self.meta.item_text(item, g.name).to_string())
                .collect(),
            _ => Vec::new(),
        };
        let subst: HashMap<String, MirTy> =
            names.into_iter().zip(type_args.iter().cloned()).collect();
        // `mir_ty` resolves `Ty::Param` through `self.param_subst`, so the callee's map is swapped
        // in for the resolution and restored immediately after.
        let saved = std::mem::replace(&mut self.param_subst, subst);
        let resolved = param_tys
            .iter()
            .map(|t| self.mir_ty(t, span).ok())
            .collect();
        self.param_subst = saved;
        Ok(resolved)
    }

    fn top_fn_instance(
        &mut self,
        item: ItemId,
        use_expr: ExprId,
        span: Span,
    ) -> Result<Instance, LowerError> {
        let ItemKind::Fn(def) = &self.hir.item(item).kind else {
            let kind = match &self.hir.item(item).kind {
                ItemKind::Struct { .. } => "struct",
                ItemKind::Enum { .. } => "enum",
                ItemKind::Trait { .. } => "trait",
                ItemKind::Impl { .. } => "impl",
                ItemKind::Mod { .. } => "module",
                ItemKind::Use { .. } => "use",
                ItemKind::Const { .. } => "const",
                ItemKind::TypeAlias { .. } => "type alias",
                ItemKind::Model(_) => "model",
                ItemKind::Fn(_) => "function",
            };
            let item_name = item_name_text(self.hir, self.meta, item).unwrap_or("<unnamed>");
            return unsupported(
                format!(
                    "use of a non-function item as a function: {kind} {item_name} at `{}`",
                    self.text(span)
                ),
                span,
            );
        };
        let type_args = if def.sig.generics.is_empty() {
            Vec::new()
        } else {
            let Some(recorded) = self.tables.generic_insts.get(&use_expr) else {
                // The checker records every accepted use of a generic fn (undetermined ones
                // are E0004-rejected before lowering), so a miss is a pipeline invariant
                // violation, not a user error — still reported cleanly, never mislowered.
                return unsupported("generic fn use without a recorded instantiation", span);
            };
            recorded
                .iter()
                .map(|t| self.mir_ty(t, span))
                .collect::<Result<Vec<_>, _>>()?
        };
        let key = FnKey::Top(item, type_args.clone());
        let symbol = key_symbol(self.hir, self.meta, &key)?;
        self.discovered_callees.push(key);
        Ok(Instance {
            item,
            type_args,
            symbol,
        })
    }

    /// C4.5a method resolution: inherent impls first, then trait impls, then trait defaults —
    /// mirroring `typecheck::resolve_method`'s precedence for the non-generic subset. When
    /// `receiverless`, only receiverless fns match (associated-function position).
    /// A3: borrow `expr` as `&Self` (or pass an existing reference through), materializing a
    /// temp for a non-place operand — the operand shape `Eq::eq`/`Ord::cmp` dispatch needs.
    fn borrow_value_ref(&mut self, expr: ExprId, span: Span) -> Result<Operand, LowerError> {
        let (peeled, layers) = Self::peel_refs(self.expr_mir_ty(expr)?);
        if layers > 0 {
            return self.lower_expr_to_operand(expr);
        }
        let place = self.place_or_temp(expr, &peeled, span)?;
        let ref_ty = MirTy::Ref {
            mutable: false,
            inner: Box::new(peeled),
        };
        let temp = self.new_temp(ref_ty.clone());
        self.emit(
            Statement::Assign(
                Place::local(temp),
                Rvalue::RefOf {
                    mutable: false,
                    place,
                },
            ),
            self.info(span),
        );
        self.read_place(Place::local(temp), &ref_ty, span)
    }

    /// A3 (C4.6, CD-033): lower `a == b` / `a != b` on a user nominal to a call of its
    /// `Eq::eq(&self, &other) -> Bool` impl (`!=` negates). Evaluation order is left-then-right,
    /// both borrowed — matching the HIR oracle's `Eq::eq` dispatch.
    #[allow(clippy::too_many_arguments)]
    fn lower_user_eq(
        &mut self,
        nominal: ItemId,
        type_args: &[MirTy],
        op: BinOp,
        lhs: ExprId,
        rhs: ExprId,
        span: Span,
    ) -> Result<Operand, LowerError> {
        let Some((key, _receiver)) = self.find_impl_fn(nominal, "eq", false, type_args) else {
            return unsupported("`==`/`!=` on a user type without an `Eq` impl", span);
        };
        let lhs_ref = self.borrow_value_ref(lhs, span)?;
        let rhs_ref = self.borrow_value_ref(rhs, span)?;
        let instance = self.instance_from_key(&key)?;
        self.discovered_callees.push(key);
        let eq_dest = self.new_temp(MirTy::Bool);
        let after = self.new_block();
        self.terminate(
            Terminator::Call {
                callee: Callee::Instance(instance),
                args: vec![lhs_ref, rhs_ref],
                dest: Place::local(eq_dest),
                target: after,
            },
            self.info(span),
            after,
        );
        if matches!(op, BinOp::Eq) {
            self.read_place(Place::local(eq_dest), &MirTy::Bool, span)
        } else {
            let neq = self.new_temp(MirTy::Bool);
            self.emit(
                Statement::Assign(
                    Place::local(neq),
                    Rvalue::UnOp(MirUnOp::Not, Operand::Copy(Place::local(eq_dest))),
                ),
                self.info(span),
            );
            Ok(Operand::Copy(Place::local(neq)))
        }
    }

    /// A3 Ord (Amendment A2, CE3): lower `a < b` / `<=` / `>` / `>=` on a user nominal to a call
    /// of its `Ord::cmp(&self, &other) -> Ordering`, then map the returned `Ordering`
    /// discriminant (Less=0, Equal=1, Greater=2) to the comparison's `Bool`:
    /// `<` → `d == 0`, `<=` → `d != 2`, `>` → `d == 2`, `>=` → `d != 0` — matching the oracle.
    /// Operands are borrowed left-then-right (`&Self`), never moved.
    #[allow(clippy::too_many_arguments)]
    fn lower_user_ord(
        &mut self,
        nominal: ItemId,
        type_args: &[MirTy],
        op: BinOp,
        lhs: ExprId,
        rhs: ExprId,
        span: Span,
    ) -> Result<Operand, LowerError> {
        let Some((key, _receiver)) = self.find_impl_fn(nominal, "cmp", false, type_args) else {
            return unsupported(
                "ordered comparison on a user type without an `Ord` impl",
                span,
            );
        };
        let lhs_ref = self.borrow_value_ref(lhs, span)?;
        let rhs_ref = self.borrow_value_ref(rhs, span)?;
        let instance = self.instance_from_key(&key)?;
        self.discovered_callees.push(key);
        // cmp(&a, &b) -> Ordering
        let ord_ty = MirTy::Enum(EnumRef::CoreOrdering, Vec::new());
        let ord_dest = self.new_temp(ord_ty.clone());
        let after = self.new_block();
        self.terminate(
            Terminator::Call {
                callee: Callee::Instance(instance),
                args: vec![lhs_ref, rhs_ref],
                dest: Place::local(ord_dest),
                target: after,
            },
            self.info(span),
            after,
        );
        // Read the discriminant, then compare against the fixed variant index.
        let disc = self.new_temp(MirTy::Int64);
        self.emit(
            Statement::Assign(
                Place::local(disc),
                Rvalue::Discriminant(Place::local(ord_dest)),
            ),
            self.info(span),
        );
        let (mir_op, rhs_disc) = match op {
            BinOp::Lt => (MirBinOp::Eq, 0), // Less
            BinOp::Le => (MirBinOp::Ne, 2), // not Greater
            BinOp::Gt => (MirBinOp::Eq, 2), // Greater
            BinOp::Ge => (MirBinOp::Ne, 0), // not Less
            _ => unreachable!("lower_user_ord on a non-ordered operator"),
        };
        let result = self.new_temp(MirTy::Bool);
        self.emit(
            Statement::Assign(
                Place::local(result),
                Rvalue::BinOp(
                    mir_op,
                    Operand::Copy(Place::local(disc)),
                    Operand::Const(Constant::Int(rhs_disc, MirTy::Int64)),
                ),
            ),
            self.info(span),
        );
        Ok(Operand::Copy(Place::local(result)))
    }

    fn find_impl_fn(
        &self,
        nominal: ItemId,
        name: &str,
        receiverless: bool,
        type_args: &[MirTy],
    ) -> Option<(FnKey, Option<hir::Receiver>)> {
        let mut inherent: Option<(FnKey, Option<hir::Receiver>)> = None;
        let mut via_trait: Option<(FnKey, Option<hir::Receiver>)> = None;
        let mut via_default: Option<(FnKey, Option<hir::Receiver>)> = None;
        for (idx, item) in self.hir.items.iter().enumerate() {
            let ItemKind::Impl { trait_, items, .. } = &item.kind else {
                continue;
            };
            let impl_item = ItemId(idx as u32);
            if impl_self_item(self.hir, impl_item) != Some(nominal) {
                continue;
            }
            for (member, impl_member) in items.iter().enumerate() {
                let hir::ImplItem::Fn { def, .. } = impl_member else {
                    continue;
                };
                if self.meta.item_text(impl_item, def.sig.name) != name {
                    continue;
                }
                if receiverless != def.sig.receiver.is_none() {
                    continue;
                }
                let hit = (
                    FnKey::ImplFn {
                        impl_item,
                        member: member as u32,
                        type_args: type_args.to_vec(),
                        // `find_impl_fn` locates the member; the CALL SITE supplies any
                        // method-level arguments, since they vary per call.
                        method_args: Vec::new(),
                    },
                    def.sig.receiver,
                );
                if trait_.is_none() {
                    inherent.get_or_insert(hit);
                } else {
                    via_trait.get_or_insert(hit);
                }
            }
            // Trait defaults: only when this impl does NOT override the method.
            if let Some(trait_ref) = trait_ {
                if let Res::Item(trait_item) = trait_ref.res {
                    let overridden = items.iter().any(|m| {
                        matches!(m, hir::ImplItem::Fn { def, .. }
                            if self.meta.item_text(impl_item, def.sig.name) == name)
                    });
                    if !overridden {
                        if let ItemKind::Trait {
                            items: trait_items, ..
                        } = &self.hir.item(trait_item).kind
                        {
                            for (member, trait_member) in trait_items.iter().enumerate() {
                                let hir::TraitItem::Method { sig, body: Some(_) } = trait_member
                                else {
                                    continue;
                                };
                                if self.meta.item_text(trait_item, sig.name) != name {
                                    continue;
                                }
                                if receiverless != sig.receiver.is_none() {
                                    continue;
                                }
                                via_default.get_or_insert((
                                    FnKey::TraitDefault {
                                        trait_item,
                                        member: member as u32,
                                        self_item: nominal,
                                        self_args: type_args.to_vec(),
                                        // Filled by the CALL SITE, like `ImplFn::method_args`.
                                        method_args: Vec::new(),
                                    },
                                    sig.receiver,
                                ));
                            }
                        }
                    }
                }
            }
        }
        inherent.or(via_trait).or(via_default)
    }

    /// WP-C6.2b: locate the implementation of a **specific** trait's method for `nominal`.
    ///
    /// `find_impl_fn` answers "what does `recv.m()` mean here", so it prefers inherent methods and
    /// accepts any in-scope trait supplying the name. A fully qualified `Trait::m(&recv)` asks a
    /// different question — TYPE-METHOD-001's "bypasses trait-name lookup but still requires a
    /// unique coherent impl" — and must ignore both inherent methods and other traits. Keeping the
    /// two lookups separate is what lets `A::go(&s)` and `B::go(&s)` disambiguate the §18 ambiguity
    /// case instead of both resolving to whichever impl is found first.
    ///
    /// Coherence (one impl of a trait per type) is enforced upstream, so the first match is the
    /// unique one; an overriding `impl` member wins over the trait's default body, as elsewhere.
    fn find_trait_impl_fn(
        &self,
        nominal: ItemId,
        trait_item: ItemId,
        name: &str,
        type_args: &[MirTy],
    ) -> Option<FnKey> {
        for (idx, item) in self.hir.items.iter().enumerate() {
            let ItemKind::Impl {
                trait_: Some(trait_ref),
                items,
                ..
            } = &item.kind
            else {
                continue;
            };
            let impl_item = ItemId(idx as u32);
            if impl_self_item(self.hir, impl_item) != Some(nominal) {
                continue;
            }
            if trait_ref.res != Res::Item(trait_item) {
                continue;
            }
            for (member, impl_member) in items.iter().enumerate() {
                let hir::ImplItem::Fn { def, .. } = impl_member else {
                    continue;
                };
                if self.meta.item_text(impl_item, def.sig.name) != name
                    || def.sig.receiver.is_none()
                {
                    continue;
                }
                return Some(FnKey::ImplFn {
                    impl_item,
                    member: member as u32,
                    type_args: type_args.to_vec(),
                    method_args: Vec::new(),
                });
            }
            // Not overridden by this impl — the trait's own default body.
            if let ItemKind::Trait {
                items: trait_items, ..
            } = &self.hir.item(trait_item).kind
            {
                for (member, trait_member) in trait_items.iter().enumerate() {
                    let hir::TraitItem::Method { sig, body: Some(_) } = trait_member else {
                        continue;
                    };
                    if self.meta.item_text(trait_item, sig.name) != name || sig.receiver.is_none() {
                        continue;
                    }
                    return Some(FnKey::TraitDefault {
                        trait_item,
                        member: member as u32,
                        self_item: nominal,
                        self_args: type_args.to_vec(),
                        method_args: Vec::new(),
                    });
                }
            }
        }
        None
    }

    /// Lower `receiver.method(args)` — evaluation order: receiver first (CD-007/CD-010).
    fn lower_method_call(
        &mut self,
        base: ExprId,
        name_span: Span,
        args: &[ExprId],
        dest: Place,
        span: Span,
        // WP-C4.7-8.4: the CALL expression, which keys the checker's per-call-site record of the
        // method's own generic arguments.
        call_expr: ExprId,
    ) -> Result<(), LowerError> {
        let base_ty = self.expr_mir_ty(base)?;
        let (peeled_ty, base_ref_layers) = Self::peel_refs(base_ty.clone());
        // WP-C4.7-6.2: `Ord::cmp` on a PRIMITIVE receiver (06's `impl Ord for Int32`, "and
        // similar for other types"). Checked BEFORE the String/Vec/HashMap dispatches below,
        // because `String` is a primitive receiver for this purpose and would otherwise be
        // claimed by the String runtime surface (which has no `cmp` entry).
        if self.text(name_span) == "cmp"
            && args.len() == 1
            && matches!(
                peeled_ty,
                MirTy::Int8
                    | MirTy::Int16
                    | MirTy::Int32
                    | MirTy::Int64
                    | MirTy::UInt8
                    | MirTy::UInt16
                    | MirTy::UInt32
                    | MirTy::UInt64
                    // DEV-075: `Char` is ordered by Unicode scalar value. `Bool` is deliberately
                    // absent — it is not `Ord`, and the checker rejects `Bool::cmp` outright.
                    | MirTy::Char
                    | MirTy::String
                    | MirTy::Str
            )
        {
            return self.lower_primitive_cmp(base, args[0], dest, span);
        }
        // DEV-DISPLAY-DISPATCH: `Display::fmt` on a standard-library receiver. Checked BEFORE the
        // String/str dispatch below, which has no `fmt` entry of its own and would refuse the
        // call.
        //
        // This is the concrete tail of generic `Display` dispatch, not a separate feature: by the
        // time MIR sees `x.fmt()` inside `fn show<T: Display>(x: &T)`, monomorphisation has ground
        // `T` down to a concrete type. When that type is a user nominal the ordinary nominal path
        // further below already resolves its `impl Display` — but when it is a PRIMITIVE there is
        // no impl item to find, because 06-Standard-Library declares those impls and no source
        // file writes them. The runtime call below is the lowering of exactly those declarations.
        if self.text(name_span) == "fmt" && args.is_empty() {
            if let Some(kind) = FmtReceiver::of(&peeled_ty) {
                return self.lower_display_fmt(base, &peeled_ty, base_ref_layers, kind, dest, span);
            }
        }
        // 0.1-A7: the method spelling `b.into_inner()` (the qualified `Box::into_inner(b)` form
        // goes through the builtin arm in `lower_call`). Core v1 has NO `Deref` trait, so this is
        // the ONLY way to get the value out of a box — `*b` is not a Core construct.
        if let MirTy::Core(crate::hir::CoreType::Box, _) = &peeled_ty {
            let name = self.text(name_span);
            if name != "into_inner" || !args.is_empty() {
                return unsupported(format!("Box method {name}"), span);
            }
            let boxed = self.lower_expr_to_operand(base)?;
            self.emit_runtime_call(RuntimeFn::BoxIntoInner, vec![boxed], dest, span);
            return Ok(());
        }
        // A1 (CD-031): methods on the runtime text types dispatch to the RuntimeFn surface.
        if matches!(peeled_ty, MirTy::String | MirTy::Str) {
            return self.lower_string_method_call(base, &peeled_ty, name_span, args, dest, span);
        }
        // A1 (CD-031), C4.5e-2: Vec methods dispatch to the Vec RuntimeFn surface.
        if let MirTy::Core(crate::hir::CoreType::Vec, elem_args) = &peeled_ty {
            let elem = elem_args.first().cloned().unwrap_or(MirTy::Unit);
            return self.lower_vec_method_call(base, elem, name_span, args, dest, span);
        }
        // 0.1-A3 (f-3a): HashMap methods dispatch to the map RuntimeFn surface.
        if let MirTy::Core(crate::hir::CoreType::HashMap, kv_args) = &peeled_ty {
            let kv = kv_args.clone();
            return self.lower_map_method_call(base, kv, name_span, args, dest, span);
        }
        if let MirTy::Core(crate::hir::CoreType::HashSet, elem_args) = &peeled_ty {
            let elem = elem_args.first().cloned().unwrap_or(MirTy::Unit);
            return self.lower_set_method_call(base, elem, name_span, args, dest, span);
        }
        // 0.1-A6 (A4 slicing): slice methods — `len`/`is_empty` on a `&[T]` receiver.
        if matches!(peeled_ty, MirTy::Slice(_)) {
            let name = self.text(name_span);
            let rt = match name {
                "len" => RuntimeFn::SliceLen,
                "is_empty" => RuntimeFn::SliceIsEmpty,
                other => return unsupported(format!("slice method {other}"), span),
            };
            // The receiver expression is the `&[T]`/`&mut [T]` value itself (peel found ref
            // layers). WP-C4.7-8.6: read it by COPY rather than move. `len`/`is_empty` only read
            // through the reference — the MIR-level equivalent of a shared reborrow — so moving
            // it would consume an exclusive view and make a second use of the same `&mut [T]`
            // local fail V-MOVE-1, which is not what the language says (`s.len(); s[0]` is legal,
            // and the oracle accepts it since DEV-082).
            let recv = match self.lower_place(base) {
                Ok(place) => Operand::Copy(place),
                Err(_) => self.lower_expr_to_operand(base)?,
            };
            self.emit_runtime_call(rt, vec![recv], dest, span);
            return Ok(());
        }
        // C4.5e-3: Option/Result inspection and extraction methods; A4: value combinators.
        if let MirTy::Enum(enum_ref @ (EnumRef::CoreOption | EnumRef::CoreResult), ty_args) =
            &peeled_ty
        {
            return self.lower_option_result_method_call(
                base, *enum_ref, ty_args, name_span, args, dest, span,
            );
        }
        // A1: methods on generic nominal instantiations monomorphise at the receiver's
        // concrete type arguments (impl-level substitution).
        let (nominal, nominal_args) = match &peeled_ty {
            MirTy::Struct(item, args) | MirTy::Enum(EnumRef::User(item), args) => {
                (*item, args.clone())
            }
            // DEV-151: a host resource IS a nominal for method-dispatch purposes. `HostResourceTy`
            // carries the item of the synthesized zero-variant enum the package declared (CD-234),
            // and an `impl TcpStream { fn set_read_timeout(&mut self, ..) }` hangs off exactly that
            // item — so the ordinary nominal path resolves it with no special case beyond naming
            // the item. Falling into `other` refused every method a resource package declares,
            // which made CD-346's `&mut self` ruling unbuildable AT THE CALL SITE while the
            // declaration itself qualified. Resources take no type arguments, hence the empty list.
            MirTy::HostResource(resource) => match &resource.nominal {
                crate::mir::HostResourceNominal::Item(item) => (*item, Vec::new()),
                crate::mir::HostResourceNominal::Core(_) => {
                    return unsupported(
                        "method call on a Core host resource (CD-235 sequencing)".to_string(),
                        span,
                    );
                }
            },
            other => {
                return unsupported(
                    format!("method call on non-nominal receiver {other:?} (C4.5b+)"),
                    span,
                )
            }
        };
        let name_text = self.text(name_span).to_string();
        let Some((key, receiver)) = self.find_impl_fn(nominal, &name_text, false, &nominal_args)
        else {
            return unsupported(format!("method {name_text} not found (C4.5b+)"), span);
        };
        // WP-C4.7-8.4: attach this call site's METHOD-level generic arguments. `find_impl_fn`
        // locates the member and supplies the IMPL-level arguments; the method's own vary per
        // call, so they come from the checker's recording keyed by this call expression. The
        // recorded types are grounded but may still mention the ENCLOSING body's parameters, so
        // the active substitution applies — the same treatment top-level generic calls get.
        let recorded_method_args = match self.tables.generic_insts.get(&call_expr) {
            Some(tys) => tys
                .iter()
                .map(|t| self.mir_ty(t, span))
                .collect::<Result<Vec<_>, _>>()?,
            None => Vec::new(),
        };
        let key = match key {
            FnKey::ImplFn {
                impl_item,
                member,
                type_args,
                ..
            } => FnKey::ImplFn {
                impl_item,
                member,
                type_args,
                method_args: recorded_method_args,
            },
            FnKey::TraitDefault {
                trait_item,
                member,
                self_item,
                self_args,
                ..
            } => FnKey::TraitDefault {
                trait_item,
                member,
                self_item,
                self_args,
                method_args: recorded_method_args,
            },
            other => other,
        };
        // Receiver operand FIRST (normative order), before arguments. C4.5b-2: real borrows.
        let receiver_op = match receiver {
            Some(kind @ (hir::Receiver::Ref | hir::Receiver::RefMut)) => {
                let mutable = matches!(kind, hir::Receiver::RefMut);
                // WP-C6.1f-b1. TYPE-METHOD-002: auto-dereference "examines `S`, then repeatedly
                // removes one leading `&`/`&mut`; at each level receiver matching tries by-value,
                // shared-borrow, then exclusive-borrow form". So a receiver that is ALREADY a
                // reference is dereferenced and **re-borrowed at the method's required
                // mutability** — it is not passed through.
                //
                // Passing it through (the pre-C6.1f-b1 behaviour) was wrong twice over:
                //   * it never adjusted `&mut T` to `&T`, so a `&self` method reached through a
                //     `&mut` receiver failed MIR verification with an argument-type mismatch; and
                //   * it MOVED the reference — `&mut T` is not `Copy` — so `m.bump(); m.bump();`
                //     failed V-MOVE-1 on the second call.
                //
                // Each re-borrow is a *temporary* borrow that ends with its statement
                // (03-Type-System, "References and Lifetimes" rule 4: "`f(&x); g(&mut x);` is
                // legal"), so this introduces no borrow the checker has not already approved —
                // and indeed the front end already accepted both shapes; only lowering did not
                // express them. Deriving the place through `lower_place_autoderef` also makes
                // NESTED reference receivers fall out for free, since it peels every layer.
                let place = match self.lower_place_autoderef(base) {
                    Ok((place, _)) => place,
                    Err(_) => {
                        // Non-place receiver (`make().method()`): materialise the base at its own
                        // type — which may itself be a reference — then project down to the
                        // referent, so the temp's type and the projection stay consistent.
                        let value = self.lower_expr_to_operand(base)?;
                        let base_ty = self.expr_mir_ty(base)?;
                        let temp = self.new_temp(base_ty);
                        self.emit(
                            Statement::Assign(Place::local(temp), Rvalue::Use(value)),
                            self.info(span),
                        );
                        let mut place = Place::local(temp);
                        for _ in 0..base_ref_layers {
                            place.projection.push(Projection::Deref);
                        }
                        place
                    }
                };
                let ref_ty = MirTy::Ref {
                    mutable,
                    inner: Box::new(peeled_ty.clone()),
                };
                let temp = self.new_temp(ref_ty.clone());
                self.emit(
                    Statement::Assign(Place::local(temp), Rvalue::RefOf { mutable, place }),
                    self.info(span),
                );
                self.read_place(Place::local(temp), &ref_ty, span)?
            }
            Some(hir::Receiver::Value) => self.lower_expr_to_operand(base)?,
            None => {
                return unsupported("method-syntax call to a receiverless fn", span);
            }
        };
        let instance = self.instance_from_key(&key)?;
        self.discovered_callees.push(key);
        let mut ops = vec![receiver_op];
        for &arg in args {
            ops.push(self.lower_expr_to_operand(arg)?);
        }
        let after = self.new_block();
        self.terminate(
            Terminator::Call {
                callee: Callee::Instance(instance),
                args: ops,
                dest,
                target: after,
            },
            self.info(span),
            after,
        );
        Ok(())
    }

    /// A1 (CD-031): lower a method call on a `String`/`str` receiver to the RuntimeFn surface.
    fn lower_string_method_call(
        &mut self,
        base: ExprId,
        peeled_ty: &MirTy,
        name_span: Span,
        args: &[ExprId],
        dest: Place,
        span: Span,
    ) -> Result<(), LowerError> {
        let name = self.text(name_span).to_string();
        let is_string = matches!(peeled_ty, MirTy::String);
        // 0.1-A5 (A4-2d): `chars()` on `str`/`String` → a `CharsIter` over a `&str` snapshot.
        if name == "chars" {
            let str_op = self.str_operand_for(base, span)?;
            self.emit_runtime_call(RuntimeFn::CharsIterNew, vec![str_op], dest, span);
            return Ok(());
        }
        // `bytes()` on `str`/`String` → immutable `&[UInt8]`; owned strings first snapshot to
        // `&str`, matching `chars()` and comparison lowering.
        if name == "bytes" {
            let str_op = self.str_operand_for(base, span)?;
            self.emit_runtime_call(RuntimeFn::StrBytes, vec![str_op], dest, span);
            return Ok(());
        }
        // (runtime fn, receiver mutability). str methods take the `&str` value directly.
        let (rt, recv_mut) = match (is_string, name.as_str()) {
            (true, "as_str") => (RuntimeFn::StringAsStr, Some(false)),
            (true, "len") => (RuntimeFn::StringLen, Some(false)),
            (true, "is_empty") => (RuntimeFn::StringIsEmpty, Some(false)),
            (true, "clone") => (RuntimeFn::StringClone, Some(false)),
            (true, "contains") => (RuntimeFn::StringContains, Some(false)),
            (true, "push_str") => (RuntimeFn::StringPushStr, Some(true)),
            (true, "push") => (RuntimeFn::StringPushChar, Some(true)),
            (true, "pop") => (RuntimeFn::StringPopChar, Some(true)),
            (true, "clear") => (RuntimeFn::StringClear, Some(true)),
            (false, "len") => (RuntimeFn::StrLen, None),
            (false, "is_empty") => (RuntimeFn::StrIsEmpty, None),
            (false, "to_string") => (RuntimeFn::StrToString, None),
            (false, "substring") => (RuntimeFn::StrSubstring, None),
            _ => {
                return unsupported(
                    format!("method {name} on {peeled_ty:?} (a later C4.5e sub-slice)"),
                    span,
                )
            }
        };
        // Receiver operand.
        let recv_op = match recv_mut {
            Some(mutable) => self.borrow_string_receiver(base, mutable, span)?,
            None => self.lower_expr_to_operand(base)?, // `&str` value, passed through
        };
        let mut ops = vec![recv_op];
        for &arg in args {
            ops.push(self.lower_expr_to_operand(arg)?);
        }
        self.emit_runtime_call(rt, ops, dest, span);
        Ok(())
    }

    /// Is `ty` a `String` or a `str` behind any reference depth (A1 comparison routing)?
    fn is_text_ty(ty: &MirTy) -> bool {
        matches!(Self::peel_refs(ty.clone()).0, MirTy::String | MirTy::Str)
    }

    /// A `&str` operand for `expr` (an owned/borrowed `String` is converted via `StringAsStr`).
    fn str_operand_for(&mut self, expr: ExprId, span: Span) -> Result<Operand, LowerError> {
        let peeled = Self::peel_refs(self.expr_mir_ty(expr)?).0;
        if matches!(peeled, MirTy::Str) {
            return self.lower_expr_to_operand(expr);
        }
        // String / &String → borrow then snapshot to &str.
        let recv = self.borrow_string_receiver(expr, false, span)?;
        let str_ty = MirTy::Ref {
            mutable: false,
            inner: Box::new(MirTy::Str),
        };
        let tmp = self.new_temp(str_ty.clone());
        self.emit_runtime_call(RuntimeFn::StringAsStr, vec![recv], Place::local(tmp), span);
        self.read_place(Place::local(tmp), &str_ty, span)
    }

    /// WP-C4.7-6.2: `a.cmp(&b)` on a primitive, producing a `CoreOrdering` value.
    ///
    /// Strategy (no new MIR shape, no new `RuntimeFn`): compute the SAME comparisons the `<` and
    /// `==` operator paths already lower — including routing `String`/`str` through `StrCmp`, so
    /// `a.cmp(&b)` and `a < b` cannot disagree — then select the variant with a two-step branch:
    ///
    /// ```text
    ///   if a < b        -> Ordering::Less
    ///   else if a == b  -> Ordering::Equal
    ///   else            -> Ordering::Greater
    /// ```
    ///
    /// This is the inverse of `lower_user_ord`, which CALLS a user `cmp` and switches on the
    /// resulting discriminant; here we compute the comparison and CONSTRUCT the value. Both
    /// operands are read into temps before any branching, so each is evaluated exactly once,
    /// receiver before argument (EXEC-ONCE-001 / the normative evaluation order).
    fn lower_primitive_cmp(
        &mut self,
        base: ExprId,
        other: ExprId,
        dest: Place,
        span: Span,
    ) -> Result<(), LowerError> {
        let (ty, _) = Self::peel_refs(self.expr_mir_ty(base)?);
        let info = self.info(span);
        let is_str = matches!(ty, MirTy::String | MirTy::Str);

        let (lt, eq) = if is_str {
            let a = self.str_operand_for(base, span)?;
            let b = self.str_operand_for(other, span)?;
            let cmp = self.new_temp(MirTy::Int64);
            self.emit_runtime_call(RuntimeFn::StrCmp, vec![a, b], Place::local(cmp), span);
            let zero = Operand::Const(Constant::Int(0, MirTy::Int64));
            let lt = self.new_temp(MirTy::Bool);
            self.emit(
                Statement::Assign(
                    Place::local(lt),
                    Rvalue::BinOp(MirBinOp::Lt, Operand::Copy(Place::local(cmp)), zero.clone()),
                ),
                info,
            );
            let eq = self.new_temp(MirTy::Bool);
            self.emit(
                Statement::Assign(
                    Place::local(eq),
                    Rvalue::BinOp(MirBinOp::Eq, Operand::Copy(Place::local(cmp)), zero),
                ),
                info,
            );
            (lt, eq)
        } else {
            let a = self.scalar_value_operand(base, &ty, span)?;
            let b = self.scalar_value_operand(other, &ty, span)?;
            let lt = self.new_temp(MirTy::Bool);
            self.emit(
                Statement::Assign(
                    Place::local(lt),
                    Rvalue::BinOp(MirBinOp::Lt, a.clone(), b.clone()),
                ),
                info,
            );
            let eq = self.new_temp(MirTy::Bool);
            self.emit(
                Statement::Assign(Place::local(eq), Rvalue::BinOp(MirBinOp::Eq, a, b)),
                info,
            );
            (lt, eq)
        };

        let less_block = self.new_block();
        let not_less_block = self.new_block();
        let equal_block = self.new_block();
        let greater_block = self.new_block();
        let join = self.new_block();

        self.terminate(
            Terminator::SwitchInt {
                scrut: Operand::Copy(Place::local(lt)),
                arms: vec![(1, less_block)],
                otherwise: not_less_block,
            },
            info,
            less_block,
        );
        self.assign_ordering_variant(dest.clone(), 0, join, info);

        self.current = not_less_block;
        self.terminate(
            Terminator::SwitchInt {
                scrut: Operand::Copy(Place::local(eq)),
                arms: vec![(1, equal_block)],
                otherwise: greater_block,
            },
            info,
            equal_block,
        );
        self.assign_ordering_variant(dest.clone(), 1, join, info);

        self.current = greater_block;
        self.assign_ordering_variant(dest, 2, join, info);

        self.current = join;
        Ok(())
    }

    /// Read a scalar operand for `cmp`, dereferencing a `&Self` argument to its referent — the
    /// comparison is between the VALUES, not the references.
    fn scalar_value_operand(
        &mut self,
        expr: ExprId,
        ty: &MirTy,
        span: Span,
    ) -> Result<Operand, LowerError> {
        let (peeled, layers) = Self::peel_refs(self.expr_mir_ty(expr)?);
        if layers == 0 {
            let place = self.place_or_temp(expr, ty, span)?;
            return self.read_place(place, ty, span);
        }
        let mut place = self.place_or_temp(expr, &self.expr_mir_ty(expr)?.clone(), span)?;
        for _ in 0..layers {
            place.projection.push(Projection::Deref);
        }
        self.read_place(place, &peeled, span)
    }

    /// Assign one fieldless `Ordering` variant and jump to `join`, sealing the current block.
    fn assign_ordering_variant(
        &mut self,
        dest: Place,
        variant: u32,
        join: BlockId,
        info: SourceInfo,
    ) {
        self.emit(
            Statement::Assign(
                dest,
                Rvalue::Aggregate(
                    AggKind::EnumVariant(EnumRef::CoreOrdering, variant),
                    Vec::new(),
                ),
            ),
            info,
        );
        self.terminate(Terminator::Goto { target: join }, info, join);
    }

    /// Lower a `String`/`str` comparison to `StrEq`/`StrCmp` (A1). `==`/`!=` use `StrEq`;
    /// ordered comparisons derive from `StrCmp`'s −1/0/+1 against zero.
    fn lower_string_comparison(
        &mut self,
        op: BinOp,
        lhs: ExprId,
        rhs: ExprId,
        span: Span,
    ) -> Result<Operand, LowerError> {
        let a = self.str_operand_for(lhs, span)?;
        let b = self.str_operand_for(rhs, span)?;
        match op {
            BinOp::Eq | BinOp::Ne => {
                let eq = self.new_temp(MirTy::Bool);
                self.emit_runtime_call(RuntimeFn::StrEq, vec![a, b], Place::local(eq), span);
                if matches!(op, BinOp::Eq) {
                    self.read_place(Place::local(eq), &MirTy::Bool, span)
                } else {
                    let neq = self.new_temp(MirTy::Bool);
                    self.emit(
                        Statement::Assign(
                            Place::local(neq),
                            Rvalue::UnOp(MirUnOp::Not, Operand::Copy(Place::local(eq))),
                        ),
                        self.info(span),
                    );
                    Ok(Operand::Copy(Place::local(neq)))
                }
            }
            BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge => {
                let cmp = self.new_temp(MirTy::Int64);
                self.emit_runtime_call(RuntimeFn::StrCmp, vec![a, b], Place::local(cmp), span);
                let mir_op = match op {
                    BinOp::Lt => MirBinOp::Lt,
                    BinOp::Le => MirBinOp::Le,
                    BinOp::Gt => MirBinOp::Gt,
                    BinOp::Ge => MirBinOp::Ge,
                    _ => unreachable!(),
                };
                let dest = self.new_temp(MirTy::Bool);
                self.emit(
                    Statement::Assign(
                        Place::local(dest),
                        Rvalue::BinOp(
                            mir_op,
                            Operand::Copy(Place::local(cmp)),
                            Operand::Const(Constant::Int(0, MirTy::Int64)),
                        ),
                    ),
                    self.info(span),
                );
                Ok(Operand::Copy(Place::local(dest)))
            }
            _ => unsupported("non-comparison string binop", span),
        }
    }

    /// C4.5e-3: lower `is_some`/`is_none`/`is_ok`/`is_err`/`unwrap` on an Option/Result
    /// receiver. Inspection reads the discriminant; `unwrap` switches on it, extracting the
    /// payload on the expected variant and trapping otherwise.
    #[allow(clippy::too_many_arguments)]
    fn lower_option_result_method_call(
        &mut self,
        base: ExprId,
        enum_ref: EnumRef,
        args: &[MirTy],
        name_span: Span,
        call_args: &[ExprId],
        dest: Place,
        span: Span,
    ) -> Result<(), LowerError> {
        let name = self.text(name_span).to_string();
        let is_option = matches!(enum_ref, EnumRef::CoreOption);
        // (ok discriminant, trap for the wrong variant on unwrap).
        let (ok_variant, unwrap_trap) = if is_option {
            (1u32, TrapCategory::UnwrapNone)
        } else {
            (0u32, TrapCategory::UnwrapErr)
        };
        let place = self.materialize_enum_receiver(base, &enum_ref, args, span)?;

        match name.as_str() {
            "is_some" | "is_ok" => self.emit_discriminant_eq(place, ok_variant, dest, span),
            "is_none" | "is_err" => {
                let other = if ok_variant == 1 { 0 } else { 1 };
                self.emit_discriminant_eq(place, other, dest, span);
            }
            "unwrap" => {
                let payload_ty = args.first().cloned().unwrap_or(MirTy::Unit);
                let disc = self.new_temp(MirTy::Int64);
                self.emit(
                    Statement::Assign(Place::local(disc), Rvalue::Discriminant(place.clone())),
                    self.info(span),
                );
                let ok_block = self.new_block();
                let trap_block = self.new_block();
                self.terminate(
                    Terminator::SwitchInt {
                        scrut: Operand::Copy(Place::local(disc)),
                        arms: vec![(u128::from(ok_variant), ok_block)],
                        otherwise: trap_block,
                    },
                    self.info(span),
                    trap_block,
                );
                let info = self.info(span);
                self.terminate(
                    Terminator::Trap {
                        info: TrapInfo {
                            category: unwrap_trap,
                            source: info,
                        },
                        message: None,
                    },
                    info,
                    ok_block,
                );
                let mut payload_place = place;
                payload_place
                    .projection
                    .push(Projection::VariantField(ok_variant, 0));
                let value = self.read_place(payload_place, &payload_ty, span)?;
                self.emit(Statement::Assign(dest, Rvalue::Use(value)), self.info(span));
            }
            // A4: `unwrap_or(default)` — Some/Ok → payload, None/Err → the eagerly-evaluated
            // default. Both branches assign `dest` and join. The default is evaluated once,
            // before the switch (matching by-value argument evaluation). Non-droppable payload
            // only for now: a droppable payload/default would need drop-of-unused elaboration
            // (the discarded branch's value), owned by a later increment.
            "unwrap_or" => {
                let payload_ty = args.first().cloned().unwrap_or(MirTy::Unit);
                // WP-C4.7-8.1: droppable payloads/defaults are supported. `unwrap_or` DISCARDS
                // exactly one of two values, and the discarded one owes a destructor. The
                // oracle's timing was pinned empirically first (§0.6), and DEV-076 had to be
                // fixed before this could be written at all — the oracle used to double-drop the
                // payload and never drop the default, so matching it would have encoded a double
                // drop into the backend contract.
                //
                // Pinned semantics: the default is evaluated ONCE before the switch (Core has no
                // laziness, so it is evaluated whether or not it is used); on Some/Ok the payload
                // is yielded and the default is dropped **at the call site**, not at end of
                // scope; on None/Err the default is yielded, and a `Result`'s displaced `Err`
                // payload is dropped there.
                let payload_needs_drop = self.ty_needs_drop(&payload_ty, span)?;
                let err_ty = if is_option {
                    MirTy::Unit
                } else {
                    args.get(1).cloned().unwrap_or(MirTy::Unit)
                };
                let err_needs_drop = !is_option && self.ty_needs_drop(&err_ty, span)?;
                let consuming = payload_needs_drop || err_needs_drop;
                if payload_needs_drop {
                    self.discover_drop_impls(&payload_ty)?;
                }
                if err_needs_drop {
                    self.discover_drop_impls(&err_ty)?;
                }
                // Consuming a payload out of a DROP-TRACKED local through a `VariantField`
                // projection is refused outright (C4.5d). `lower_match` solved this by
                // materializing the scrutinee into a fresh temp: the move clears the source
                // local's drop flags, and a temp is never auto-dropped, so ownership transfers
                // exactly once. Reuse that, rather than inventing a second discipline.
                let place = if consuming {
                    let recv_ty = MirTy::Enum(enum_ref, args.to_vec());
                    let value = self.read_place(place, &recv_ty, span)?;
                    let temp = self.new_temp(recv_ty);
                    self.emit(
                        Statement::Assign(Place::local(temp), Rvalue::Use(value)),
                        self.info(span),
                    );
                    Place::local(temp)
                } else {
                    place
                };
                let Some(&default_expr) = call_args.first() else {
                    return unsupported("unwrap_or expects one argument", span);
                };
                let default_op = self.lower_expr_to_operand(default_expr)?;
                // A droppable default needs a named temp so the unused-path drop has a place to
                // name; a non-droppable one is used directly, keeping the common lowering
                // byte-identical to before this change.
                let default_op = if payload_needs_drop {
                    let temp = self.new_temp(payload_ty.clone());
                    self.emit(
                        Statement::Assign(Place::local(temp), Rvalue::Use(default_op)),
                        self.info(span),
                    );
                    Operand::Move(Place::local(temp))
                } else {
                    default_op
                };
                let disc = self.new_temp(MirTy::Int64);
                self.emit(
                    Statement::Assign(Place::local(disc), Rvalue::Discriminant(place.clone())),
                    self.info(span),
                );
                let ok_block = self.new_block();
                let else_block = self.new_block();
                let join = self.new_block();
                self.terminate(
                    Terminator::SwitchInt {
                        scrut: Operand::Copy(Place::local(disc)),
                        arms: vec![(u128::from(ok_variant), ok_block)],
                        otherwise: else_block,
                    },
                    self.info(span),
                    ok_block,
                );
                // Ok/Some arm: move the payload into dest, then DROP THE UNUSED DEFAULT here —
                // the oracle destroys it at the call, not at end of scope.
                let mut payload_place = place.clone();
                payload_place
                    .projection
                    .push(Projection::VariantField(ok_variant, 0));
                let payload = self.read_place(payload_place, &payload_ty, span)?;
                self.emit(
                    Statement::Assign(dest.clone(), Rvalue::Use(payload)),
                    self.info(span),
                );
                if payload_needs_drop {
                    let Operand::Move(default_place) = &default_op else {
                        return unsupported("droppable unwrap_or default is not a place", span);
                    };
                    let temp = default_place.local;
                    self.emit_temp_drop(temp, span);
                }
                self.terminate(
                    Terminator::Goto { target: join },
                    self.info(span),
                    else_block,
                );
                // None/Err arm: yield the default. A `Result`'s displaced `Err` payload is
                // discarded exactly as the default is on the other path, so it drops here.
                if err_needs_drop {
                    let mut err_place = place;
                    err_place
                        .projection
                        .push(Projection::VariantField(1 - ok_variant, 0));
                    let err_val = self.read_place(err_place, &err_ty, span)?;
                    let err_temp = self.new_temp(err_ty.clone());
                    self.emit(
                        Statement::Assign(Place::local(err_temp), Rvalue::Use(err_val)),
                        self.info(span),
                    );
                    self.emit_temp_drop(err_temp, span);
                }
                self.emit(
                    Statement::Assign(dest, Rvalue::Use(default_op)),
                    self.info(span),
                );
                self.terminate(Terminator::Goto { target: join }, self.info(span), join);
            }
            // A4: value combinators `map` / `and_then` (Option + Result) and `map_err` (Result).
            "map" | "and_then" | "map_err" => {
                if name == "map_err" && is_option {
                    return unsupported("Option has no map_err", span);
                }
                let Some(&fn_expr) = call_args.first() else {
                    return unsupported(format!("{name} expects one function argument"), span);
                };
                return self
                    .lower_opt_res_combinator(&name, enum_ref, args, place, fn_expr, dest, span);
            }
            _ => {
                return unsupported(
                    format!("Option/Result method {name} (a later C4.5e sub-slice)"),
                    span,
                )
            }
        }
        Ok(())
    }

    /// A4: emit an indirect call through a function-value operand, returning the result operand.
    fn emit_fn_value_call(
        &mut self,
        fn_op: Operand,
        arg: Operand,
        ret_ty: MirTy,
        span: Span,
    ) -> Result<Operand, LowerError> {
        let dest = self.new_temp(ret_ty);
        let after = self.new_block();
        self.terminate(
            Terminator::Call {
                callee: Callee::FnValue(fn_op),
                args: vec![arg],
                dest: Place::local(dest),
                target: after,
            },
            self.info(span),
            after,
        );
        let ty = self.locals[dest.0 as usize].ty.clone();
        self.read_place(Place::local(dest), &ty, span)
    }

    /// A4: lower `Option`/`Result` `map` / `and_then` / `map_err`. Each switches on the active
    /// variant, applies the function value `f` to the relevant payload, and rebuilds the result
    /// enum — passing the other variant through unchanged. Every payload is moved exactly once
    /// (into `f` or into the rebuilt variant), so no drop-of-unused arises; the non-droppable
    /// gate is retained for parity with `unwrap`/`unwrap_or` until droppable Option/Result
    /// value-methods are elaborated as a whole.
    #[allow(clippy::too_many_arguments)]
    fn lower_opt_res_combinator(
        &mut self,
        name: &str,
        enum_ref: EnumRef,
        args: &[MirTy],
        place: Place,
        fn_expr: ExprId,
        dest: Place,
        span: Span,
    ) -> Result<(), LowerError> {
        let is_option = matches!(enum_ref, EnumRef::CoreOption);
        // Payload types of each variant before transformation.
        let (ok_variant, ok_ty, other_variant, other_ty) = if is_option {
            (
                1u32,
                args.first().cloned().unwrap_or(MirTy::Unit),
                0u32,
                MirTy::Unit,
            )
        } else {
            (
                0u32,
                args.first().cloned().unwrap_or(MirTy::Unit),
                1u32,
                args.get(1).cloned().unwrap_or(MirTy::Unit),
            )
        };
        // `map_err` transforms the ERROR (Result only); the others transform the ok payload.
        let transform_err = name == "map_err";
        let (xform_variant, xform_ty, passthru_variant, passthru_ty) = if transform_err {
            (other_variant, other_ty.clone(), ok_variant, ok_ty.clone())
        } else {
            (ok_variant, ok_ty.clone(), other_variant, other_ty.clone())
        };
        if [&ok_ty, &other_ty]
            .iter()
            .any(|t| self.ty_needs_drop(t, span).unwrap_or(true))
        {
            return unsupported(
                "Option/Result combinator on a droppable payload type (a later increment)",
                span,
            );
        }
        let fn_op = self.lower_expr_to_operand(fn_expr)?;
        let fn_ret = match Self::peel_refs(self.expr_mir_ty(fn_expr)?).0 {
            MirTy::FnPtr { ret, .. } => *ret,
            other => {
                return unsupported(format!("combinator argument is not a fn: {other:?}"), span)
            }
        };
        let disc = self.new_temp(MirTy::Int64);
        self.emit(
            Statement::Assign(Place::local(disc), Rvalue::Discriminant(place.clone())),
            self.info(span),
        );
        let xform_block = self.new_block();
        let passthru_block = self.new_block();
        let join = self.new_block();
        self.terminate(
            Terminator::SwitchInt {
                scrut: Operand::Copy(Place::local(disc)),
                arms: vec![(u128::from(xform_variant), xform_block)],
                otherwise: passthru_block,
            },
            self.info(span),
            xform_block,
        );
        // Transform arm: move the relevant payload out, apply `f`.
        let mut payload_place = place.clone();
        payload_place
            .projection
            .push(Projection::VariantField(xform_variant, 0));
        let payload = self.read_place(payload_place, &xform_ty, span)?;
        let mapped = self.emit_fn_value_call(fn_op, payload, fn_ret.clone(), span)?;
        // `and_then`'s `f` returns the whole result enum; `map`/`map_err` wrap it in the variant.
        let xform_value = if name == "and_then" {
            Rvalue::Use(mapped)
        } else {
            Rvalue::Aggregate(AggKind::EnumVariant(enum_ref, xform_variant), vec![mapped])
        };
        self.emit(
            Statement::Assign(dest.clone(), xform_value),
            self.info(span),
        );
        self.terminate(
            Terminator::Goto { target: join },
            self.info(span),
            passthru_block,
        );
        // Pass-through arm: rebuild the untouched variant from its moved payload (Option's
        // None has no payload).
        let passthru_value = if is_option && passthru_variant == 0 {
            Rvalue::Aggregate(AggKind::EnumVariant(enum_ref, 0), Vec::new())
        } else {
            let mut p = place;
            p.projection
                .push(Projection::VariantField(passthru_variant, 0));
            let payload = self.read_place(p, &passthru_ty, span)?;
            Rvalue::Aggregate(
                AggKind::EnumVariant(enum_ref, passthru_variant),
                vec![payload],
            )
        };
        self.emit(Statement::Assign(dest, passthru_value), self.info(span));
        self.terminate(Terminator::Goto { target: join }, self.info(span), join);
        Ok(())
    }

    /// Emit `dest = (discriminant(place) == variant)`.
    fn emit_discriminant_eq(&mut self, place: Place, variant: u32, dest: Place, span: Span) {
        let disc = self.new_temp(MirTy::Int64);
        self.emit(
            Statement::Assign(Place::local(disc), Rvalue::Discriminant(place)),
            self.info(span),
        );
        self.emit(
            Statement::Assign(
                dest,
                Rvalue::BinOp(
                    MirBinOp::Eq,
                    Operand::Copy(Place::local(disc)),
                    Operand::Const(Constant::Int(i128::from(variant), MirTy::Int64)),
                ),
            ),
            self.info(span),
        );
    }

    /// A place holding the Option/Result receiver, auto-dereffed. A place-expression base is
    /// used directly; a value-expression base (e.g. a call result) is materialized into a temp.
    fn materialize_enum_receiver(
        &mut self,
        base: ExprId,
        enum_ref: &EnumRef,
        args: &[MirTy],
        span: Span,
    ) -> Result<Place, LowerError> {
        if let Ok((place, _)) = self.lower_place_autoderef(base) {
            return Ok(place);
        }
        let ty = MirTy::Enum(*enum_ref, args.to_vec());
        let op = self.lower_expr_to_operand(base)?;
        let temp = self.new_temp(ty);
        self.emit(
            Statement::Assign(Place::local(temp), Rvalue::Use(op)),
            self.info(span),
        );
        Ok(Place::local(temp))
    }

    /// A1 (CD-031), C4.5e-2: lower a method call on a `Vec<T>` receiver to the Vec RuntimeFn
    /// surface. Iteration (`iter`) is deferred to an owner-reviewed surface bump (STARK's
    /// `.iter()` is by-reference `&T` — A1 reserved it).
    fn lower_vec_method_call(
        &mut self,
        base: ExprId,
        elem: MirTy,
        name_span: Span,
        args: &[ExprId],
        dest: Place,
        span: Span,
    ) -> Result<(), LowerError> {
        let name = self.text(name_span).to_string();
        if name == "as_slice" {
            if !args.is_empty() {
                return unsupported("Vec::as_slice with arguments", span);
            }
            let recv = self.borrow_vec_receiver(base, false, elem, span)?;
            let len_temp = self.new_temp(MirTy::UInt64);
            self.emit_runtime_call(
                RuntimeFn::VecLen,
                vec![recv.clone()],
                Place::local(len_temp),
                span,
            );
            self.emit_runtime_call(
                RuntimeFn::SliceNew,
                vec![
                    recv,
                    Operand::Const(Constant::Int(0, MirTy::UInt64)),
                    Operand::Copy(Place::local(len_temp)),
                    Operand::Const(Constant::Bool(false)),
                ],
                dest,
                span,
            );
            return Ok(());
        }
        // (runtime fn, receiver mutability).
        let (rt, recv_mut) = match name.as_str() {
            "push" => (RuntimeFn::VecPush, true),
            "pop" => (RuntimeFn::VecPop, true),
            "remove" => (RuntimeFn::VecRemove, true),
            "clear" => {
                // A1 §5a: `clear()` on a droppable element type must not hide destructors in
                // the opaque runtime op — it lowers to a pop-and-drop loop instead.
                if self.ty_needs_drop(&elem, span)? {
                    return self.lower_vec_clear_droppable(base, elem, dest, span);
                }
                (RuntimeFn::VecClear, true)
            }
            "len" => (RuntimeFn::VecLen, false),
            "is_empty" => (RuntimeFn::VecIsEmpty, false),
            // 0.1-A3 (C4.6 A6): by-reference iteration via a TRUE borrowed cursor — Next indexes
            // the live Vec and yields an interior `&T`, so the element type need NOT be Copy
            // (the earlier snapshot representation required it; this does not).
            "iter" => (RuntimeFn::VecIterNew, false),
            // 0.1-A4 (C4.6 A4-2b): checked interior access — `Option<&T>`/`Option<&mut T>`,
            // returns `None` on out-of-bounds (never traps). Any element type (yields a
            // reference, not a value).
            "get" => (RuntimeFn::VecGetRef, false),
            "get_mut" => (RuntimeFn::VecGetMutRef, true),
            _ => return unsupported(format!("Vec::{name} (a later C4.5e sub-slice)"), span),
        };
        let recv = self.borrow_vec_receiver(base, recv_mut, elem.clone(), span)?;
        let mut ops = vec![recv];
        for &arg in args {
            ops.push(self.lower_expr_to_operand(arg)?);
        }
        self.emit_runtime_call(rt, ops, dest, span);
        Ok(())
    }

    /// `v.clear()` for a droppable element type: pop-and-drop each element at a visible `Drop`
    /// terminator (A1 §5a — no `RuntimeFn` runs a user destructor). `VecPop` returns
    /// `Option<T>`; the loop drops each `Some(x)` and stops at `None`.
    fn lower_vec_clear_droppable(
        &mut self,
        base: ExprId,
        elem: MirTy,
        dest: Place,
        span: Span,
    ) -> Result<(), LowerError> {
        self.discover_drop_impls(&elem)?;
        let opt_ty = MirTy::Enum(EnumRef::CoreOption, vec![elem.clone()]);
        let header = self.new_block();
        let body_block = self.new_block();
        let exit = self.new_block();
        self.terminate(Terminator::Goto { target: header }, self.info(span), header);
        // header: pop → Option<T>, switch on discriminant.
        let popped = self.new_temp(opt_ty.clone());
        let recv = self.borrow_vec_receiver(base, true, elem.clone(), span)?;
        self.emit_runtime_call(RuntimeFn::VecPop, vec![recv], Place::local(popped), span);
        let disc = self.new_temp(MirTy::Int64);
        self.emit(
            Statement::Assign(
                Place::local(disc),
                Rvalue::Discriminant(Place::local(popped)),
            ),
            self.info(span),
        );
        // discriminant 1 = Some → body; 0 = None → exit.
        self.terminate(
            Terminator::SwitchInt {
                scrut: Operand::Copy(Place::local(disc)),
                arms: vec![(1, body_block)],
                otherwise: exit,
            },
            self.info(span),
            body_block,
        );
        // body: extract the payload into a temp, drop it, loop.
        let elem_temp = self.new_temp(elem.clone());
        self.emit(
            Statement::Assign(
                Place::local(elem_temp),
                Rvalue::Use(Operand::Move(Place {
                    local: popped,
                    projection: vec![Projection::VariantField(1, 0)],
                })),
            ),
            self.info(span),
        );
        self.emit_temp_drop(elem_temp, span);
        self.terminate(Terminator::Goto { target: header }, self.info(span), exit);
        // exit: clear() returns Unit.
        self.emit(
            Statement::Assign(dest, Rvalue::Use(Operand::Const(Constant::Unit))),
            self.info(span),
        );
        Ok(())
    }

    /// `&v[i]` / `&mut v[i]`: borrow a Vec element.
    ///
    /// A Vec element is not a MIR place, so this cannot be a `RefOf`. `VecGetRef`/`VecGetMutRef`
    /// yield `Option<&T>`, and the `None` arm — which is exactly the out-of-bounds case — raises
    /// `IndexOutOfBounds`, matching what `v[i]` does for a `Copy` element. The `Some` payload is
    /// reached through a trailing `VariantField`, which CD-126 made borrowable.
    fn lower_vec_index_borrow(
        &mut self,
        base: ExprId,
        index: ExprId,
        elem: MirTy,
        mutable: bool,
        span: Span,
    ) -> Result<Operand, LowerError> {
        let recv = self.borrow_vec_receiver(base, mutable, elem.clone(), span)?;
        let idx = self.lower_expr_to_operand(index)?;
        let idx = self.widen_index_to_u64(idx, index, span)?;
        let elem_ref_ty = MirTy::Ref {
            mutable,
            inner: Box::new(elem),
        };
        let opt_ty = MirTy::Enum(EnumRef::CoreOption, vec![elem_ref_ty.clone()]);
        let opt = self.new_temp(opt_ty);
        let rt = if mutable {
            RuntimeFn::VecGetMutRef
        } else {
            RuntimeFn::VecGetRef
        };
        self.emit_runtime_call(rt, vec![recv, idx], Place::local(opt), span);

        let disc = self.new_temp(MirTy::Int64);
        self.emit(
            Statement::Assign(Place::local(disc), Rvalue::Discriminant(Place::local(opt))),
            self.info(span),
        );
        let some_blk = self.new_block();
        let oob_blk = self.new_block();
        self.terminate(
            Terminator::SwitchInt {
                scrut: Operand::Copy(Place::local(disc)),
                arms: vec![(1, some_blk)],
                otherwise: oob_blk,
            },
            self.info(span),
            oob_blk,
        );
        // `None` is the out-of-bounds case, and indexing traps on it.
        let info = self.info(span);
        self.terminate(
            Terminator::Trap {
                info: TrapInfo {
                    category: TrapCategory::IndexOutOfBounds,
                    source: info,
                },
                message: None,
            },
            info,
            some_blk,
        );
        let mut payload = Place::local(opt);
        payload.projection.push(Projection::VariantField(1, 0));
        self.read_place(payload, &elem_ref_ty, span)
    }

    /// 0.1-A2 (C4.5f-2): `for value in v.iter() { body }`. Desugar:
    /// ```text
    /// it = <iter expr>            // VecIterNew(&v) via the method-call lowering
    /// header:
    ///   nxt = VecIterNext(&mut it)     // Option<&T>
    ///   switch discriminant(nxt) [Some → body_bb] else exit
    /// body_bb:
    ///   value: &T = copy nxt.v1.0     // `copy`/`move` by the payload's type, never fixed
    ///   ...body (own scope)...
    ///   goto header
    /// exit:
    /// ```
    /// The loop variable is a `&T` interior reference into the iterator's frame local; the
    /// f-1 frame generations guard it if it ever escapes. `T: Copy` (V-COPY-1) was checked
    /// when `iter()` lowered. The iterator local is registered droppable (no-op glue).
    #[allow(clippy::too_many_arguments)]
    /// A4: `for i in r` where `r` is a range VALUE. The range is the tuple `(start, end,
    /// inclusive)`; `inclusive` is a runtime `Bool`, so the loop condition is
    /// `i < end || (inclusive && i == end)`, lowered as a two-step branch (no boolean algebra).
    fn lower_for_over_range_value(
        &mut self,
        var: Span,
        var_local: crate::hir::LocalId,
        iter: ExprId,
        body: hir::BlockId,
        span: Span,
    ) -> Result<(), LowerError> {
        let range_ty = self.expr_mir_ty(iter)?;
        let elem_ty = match &range_ty {
            MirTy::Tuple(fields) => fields.first().cloned().unwrap_or(MirTy::Unit),
            other => return unsupported(format!("range value is not a tuple: {other:?}"), span),
        };
        // Materialize the range value once, then read start/end/inclusive from its fields.
        let range_op = self.lower_expr_to_operand(iter)?;
        let range_local = self.new_temp(range_ty);
        self.emit(
            Statement::Assign(Place::local(range_local), Rvalue::Use(range_op)),
            self.synthetic(span, SyntheticKind::ForLoopDesugar),
        );
        let field = |i: u32| Place {
            local: range_local,
            projection: vec![Projection::Field(i)],
        };
        let bound = self.new_temp(elem_ty.clone());
        self.emit(
            Statement::Assign(Place::local(bound), Rvalue::Use(Operand::Copy(field(1)))),
            self.synthetic(span, SyntheticKind::ForLoopDesugar),
        );
        let incl = self.new_temp(MirTy::Bool);
        self.emit(
            Statement::Assign(Place::local(incl), Rvalue::Use(Operand::Copy(field(2)))),
            self.synthetic(span, SyntheticKind::ForLoopDesugar),
        );
        self.locals.push(LocalDecl {
            ty: elem_ty.clone(),
            kind: LocalKind::User(self.text(var).to_string()),
        });
        let induction = LocalId((self.locals.len() - 1) as u32);
        self.local_map.insert(var_local.0, induction);
        self.emit(
            Statement::Assign(
                Place::local(induction),
                Rvalue::Use(Operand::Copy(field(0))),
            ),
            self.synthetic(span, SyntheticKind::ForLoopDesugar),
        );

        let header = self.new_block();
        let check_eq = self.new_block();
        let check_incl = self.new_block();
        let body_block = self.new_block();
        let latch = self.new_block();
        let exit = self.new_block();
        let syn = |s: &Self| s.synthetic(span, SyntheticKind::ForLoopDesugar);

        self.terminate(Terminator::Goto { target: header }, syn(self), header);
        // header: i < end ? body : check_eq
        let lt = self.new_temp(MirTy::Bool);
        self.emit(
            Statement::Assign(
                Place::local(lt),
                Rvalue::BinOp(
                    MirBinOp::Lt,
                    Operand::Copy(Place::local(induction)),
                    Operand::Copy(Place::local(bound)),
                ),
            ),
            syn(self),
        );
        self.terminate(
            Terminator::SwitchInt {
                scrut: Operand::Copy(Place::local(lt)),
                arms: vec![(1, body_block)],
                otherwise: check_eq,
            },
            syn(self),
            check_eq,
        );
        // check_eq: i == end ? check_incl : exit
        let eq = self.new_temp(MirTy::Bool);
        self.emit(
            Statement::Assign(
                Place::local(eq),
                Rvalue::BinOp(
                    MirBinOp::Eq,
                    Operand::Copy(Place::local(induction)),
                    Operand::Copy(Place::local(bound)),
                ),
            ),
            syn(self),
        );
        self.terminate(
            Terminator::SwitchInt {
                scrut: Operand::Copy(Place::local(eq)),
                arms: vec![(1, check_incl)],
                otherwise: exit,
            },
            syn(self),
            check_incl,
        );
        // check_incl: inclusive ? body : exit
        self.terminate(
            Terminator::SwitchInt {
                scrut: Operand::Copy(Place::local(incl)),
                arms: vec![(1, body_block)],
                otherwise: exit,
            },
            syn(self),
            body_block,
        );
        self.loops.push(LoopTargets {
            continue_target: latch,
            break_target: exit,
            scope_depth: self.scopes.len(),
            value_target: None,
        });
        self.lower_block_value(body)?;
        self.loops.pop();
        self.terminate(Terminator::Goto { target: latch }, syn(self), latch);
        // latch: i = i + 1 (checked), back to header.
        let step = self.new_temp(elem_ty);
        let copy_block = self.new_block();
        let induction_ty = self.locals[induction.0 as usize].ty.clone();
        self.terminate(
            Terminator::Checked {
                op: CheckedOp::Add,
                args: vec![
                    Operand::Copy(Place::local(induction)),
                    Operand::Const(Constant::Int(1, induction_ty)),
                ],
                dest: step,
                target: copy_block,
                trap: TrapInfo {
                    category: TrapCategory::IntegerOverflow,
                    source: syn(self),
                },
            },
            syn(self),
            copy_block,
        );
        self.emit(
            Statement::Assign(
                Place::local(induction),
                Rvalue::Use(Operand::Copy(Place::local(step))),
            ),
            syn(self),
        );
        self.terminate(Terminator::Goto { target: header }, syn(self), exit);
        Ok(())
    }

    /// A1: `for x in it` over a USER `Iterator` impl — desugar to a loop of `it.next()`
    /// instance calls (`&mut self`), switching on the returned `Option<Item>` discriminant and
    /// binding the loop variable BY VALUE from the `Some` payload (matching the oracle).
    #[allow(clippy::too_many_arguments)]
    /// WP-C4.7-9 audit: `for x in a` over a fixed-length array — a counting loop that reads one
    /// element per iteration through the ordinary `CheckIndex` proof discipline.
    ///
    /// Elements are read by COPY. Iterating by value out of an array would move each element out
    /// of the same local, and `Projection::Index` necessarily collapses to the whole local in the
    /// verifier's move dataflow (a dynamic proof names no statically-known sub-place), so the
    /// next iteration's `CheckIndex` would read a possibly-moved place. That is the same root
    /// cause recorded for droppable array PATTERNS, and it needs a constant-index projection
    /// form (a CE3 shape change) rather than a workaround here — so a droppable element type is
    /// a clean, precise `Unsupported`.
    #[allow(clippy::too_many_arguments)]
    fn lower_for_over_array(
        &mut self,
        var: Span,
        var_local: crate::hir::LocalId,
        iter: ExprId,
        body: hir::BlockId,
        elem: MirTy,
        len: u64,
        span: Span,
    ) -> Result<(), LowerError> {
        // WP-C6.1d (owner ruling, Option (a)): by-value iteration over a NON-`Copy` fixed array is
        // lowered by unconditional unrolling. The dynamic-index/copy path below is unsound for a
        // non-`Copy` element (the array would still own and re-destroy it), and a dynamic index
        // cannot name a `ConstIndex` for move tracking — but a fixed `[T; N]` has statically known
        // positions, so each element moves via `ConstIndex(i)` (the same machinery consuming array
        // patterns use). The `Copy` path is preserved unchanged for `Copy` elements.
        if !self.is_copy(&elem) {
            return self.lower_for_over_array_unrolled(var, var_local, iter, body, elem, len, span);
        }
        let info = self.synthetic(span, SyntheticKind::ForLoopDesugar);
        // The array itself, materialized once (EXEC-FOR-001: the iterable evaluates once).
        let arr_ty = MirTy::Array(Box::new(elem.clone()), len);
        let arr_place = self.place_or_temp(iter, &arr_ty, span)?;
        // Counter.
        let idx = self.new_temp(MirTy::Int64);
        self.emit(
            Statement::Assign(
                Place::local(idx),
                Rvalue::Use(Operand::Const(Constant::Int(0, MirTy::Int64))),
            ),
            info,
        );
        let header = self.new_block();
        let body_block = self.new_block();
        let exit = self.new_block();
        self.terminate(Terminator::Goto { target: header }, info, header);
        // header: `idx < len` ?
        let cond = self.new_temp(MirTy::Bool);
        self.emit(
            Statement::Assign(
                Place::local(cond),
                Rvalue::BinOp(
                    MirBinOp::Lt,
                    Operand::Copy(Place::local(idx)),
                    Operand::Const(Constant::Int(i128::from(len), MirTy::Int64)),
                ),
            ),
            info,
        );
        self.terminate(
            Terminator::SwitchInt {
                scrut: Operand::Copy(Place::local(cond)),
                arms: vec![(1, body_block)],
                otherwise: exit,
            },
            info,
            body_block,
        );
        // body: bind `var` to `arr[idx]` (proof-checked), run the body, increment, loop.
        self.locals.push(LocalDecl {
            ty: MirTy::Int64,
            kind: LocalKind::IndexProof,
        });
        let proof = LocalId((self.locals.len() - 1) as u32);
        let after_check = self.new_block();
        self.terminate(
            Terminator::Checked {
                op: CheckedOp::CheckIndex,
                args: vec![
                    Operand::Copy(arr_place.clone()),
                    Operand::Copy(Place::local(idx)),
                ],
                dest: proof,
                target: after_check,
                trap: TrapInfo {
                    category: TrapCategory::IndexOutOfBounds,
                    source: info,
                },
            },
            info,
            after_check,
        );
        let mut elem_place = arr_place;
        elem_place.projection.push(Projection::Index(proof));
        self.locals.push(LocalDecl {
            ty: elem.clone(),
            kind: LocalKind::User(self.text(var).to_string()),
        });
        let bound = LocalId((self.locals.len() - 1) as u32);
        self.local_map.insert(var_local.0, bound);
        self.emit(
            Statement::Assign(Place::local(bound), Rvalue::Use(Operand::Copy(elem_place))),
            info,
        );
        // `continue` must reach the INCREMENT, not the header — jumping straight back would
        // re-test the same index forever. So the loop's continue target is a latch block that
        // increments and then falls into the header. (Caught by the control-flow test: without
        // it, `continue` spun until the interpreter's fuel ran out.)
        let latch = self.new_block();
        let scope_depth = self.scopes.len();
        self.loops.push(LoopTargets {
            continue_target: latch,
            break_target: exit,
            scope_depth,
            value_target: None,
        });
        self.scopes.push(Vec::new());
        self.lower_block_value(body)?;
        self.emit_scope_drops_from(scope_depth, span);
        self.scopes.pop();
        self.loops.pop();
        self.terminate(Terminator::Goto { target: latch }, info, latch);
        // latch: idx += 1 — a checked add, like every other integer arithmetic in MIR.
        let after_incr = self.new_block();
        self.terminate(
            Terminator::Checked {
                op: CheckedOp::Add,
                args: vec![
                    Operand::Copy(Place::local(idx)),
                    Operand::Const(Constant::Int(1, MirTy::Int64)),
                ],
                dest: idx,
                target: after_incr,
                trap: TrapInfo {
                    category: TrapCategory::IntegerOverflow,
                    source: info,
                },
            },
            info,
            after_incr,
        );
        self.terminate(Terminator::Goto { target: header }, info, exit);
        Ok(())
    }

    /// WP-C6.1d (owner ruling, Option (a)): by-value iteration over a NON-`Copy` fixed array,
    /// lowered by UNCONDITIONAL unrolling. The iterable is moved ONCE into a dedicated
    /// per-element-drop-tracked array local; each of the `N` statically-known elements is moved out
    /// with `ConstIndex(i)` (which clears exactly its own drop flag, via `read_place`) into a FRESH
    /// binding local per iteration; and the loop body is lowered once per element.
    ///
    /// Scope nesting is `array owner` ⊃ `iteration binding` ⊃ `body locals`, so cleanup order is
    /// body locals → current binding → remaining array elements: normal completion / `continue` /
    /// `break` / `return` / `?` all drop the current binding (and body) via the loop's
    /// binding-scope depth, and the array owner's scope drop at `exit` destroys the still-live
    /// elements (reverse index order, per the array `DropPlan`). A trap aborts and performs no
    /// cleanup. No `CheckIndex`, dynamic `Index`, element copy, or runtime array iterator appears.
    #[allow(clippy::too_many_arguments)]
    fn lower_for_over_array_unrolled(
        &mut self,
        var: Span,
        var_local: crate::hir::LocalId,
        iter: ExprId,
        body: hir::BlockId,
        elem: MirTy,
        len: u64,
        span: Span,
    ) -> Result<(), LowerError> {
        let info = self.synthetic(span, SyntheticKind::ForLoopDesugar);
        let arr_ty = MirTy::Array(Box::new(elem.clone()), len);
        // Point 1: evaluate the iterable EXACTLY ONCE and take ownership into a dedicated array
        // local, then register its per-element `ConstIndex` drop units (initially live).
        let arr_value = self.lower_expr_to_operand(iter)?;
        let arr_local = self.new_temp(arr_ty.clone());
        self.emit(
            Statement::Assign(Place::local(arr_local), Rvalue::Use(arr_value)),
            info,
        );
        let array_scope_depth = self.scopes.len();
        self.scopes.push(Vec::new());
        self.register_droppable_local(arr_local, &arr_ty, true, span)?;

        // The block every early/normal exit converges on; the array owner's remaining elements are
        // destroyed here.
        let exit = self.new_block();

        for i in 0..len {
            // Point 2: move element `i` out via `ConstIndex(i)` — `read_place` clears its flag.
            let mut elem_place = Place::local(arr_local);
            elem_place.projection.push(Projection::ConstIndex(i));
            let value = self.read_place(elem_place, &elem, span)?;
            // Point 3: a FRESH binding local per iteration; remap the HIR loop variable to it
            // before lowering this iteration's copy of the body.
            self.locals.push(LocalDecl {
                ty: elem.clone(),
                kind: LocalKind::User(self.text(var).to_string()),
            });
            let bound = LocalId((self.locals.len() - 1) as u32);
            self.local_map.insert(var_local.0, bound);
            self.emit(
                Statement::Assign(Place::local(bound), Rvalue::Use(value)),
                info,
            );

            // Iteration binding scope (owns the yielded value for this iteration).
            let binding_scope_depth = self.scopes.len();
            self.scopes.push(Vec::new());
            self.register_droppable_local(bound, &elem, true, span)?;

            // `continue` restarts at the NEXT unrolled iteration; the last iteration continues to
            // `exit`. `break` leaves the loop at `exit`. Both drop from the binding scope down.
            let next = if i + 1 < len { self.new_block() } else { exit };
            self.loops.push(LoopTargets {
                continue_target: next,
                break_target: exit,
                scope_depth: binding_scope_depth,
                value_target: None,
            });
            self.lower_block_value(body)?;
            // Normal completion of this iteration: drop the binding (and any body locals).
            self.emit_scope_drops_from(binding_scope_depth, span);
            self.scopes.pop();
            self.loops.pop();
            self.terminate(Terminator::Goto { target: next }, info, next);
        }
        if len == 0 {
            // The iterable was still evaluated once; no elements, so jump straight to the owner's
            // (no-op) cleanup.
            self.terminate(Terminator::Goto { target: exit }, info, exit);
        }

        // At `exit`: destroy the array owner's remaining live elements (none after full
        // consumption; the unconsumed tail after a `break`).
        self.emit_scope_drops_from(array_scope_depth, span);
        self.scopes.pop();
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn lower_for_over_user_iter(
        &mut self,
        var: Span,
        var_local: crate::hir::LocalId,
        iter: ExprId,
        body: hir::BlockId,
        item: ItemId,
        targs: Vec<MirTy>,
        span: Span,
    ) -> Result<(), LowerError> {
        let Some((key, receiver)) = self.find_impl_fn(item, "next", false, &targs) else {
            return unsupported(
                "for over a non-range, non-Vec iterator without an Iterator impl",
                span,
            );
        };
        if !matches!(receiver, Some(hir::Receiver::RefMut)) {
            return unsupported("Iterator::next must take &mut self", span);
        }
        let iter_ty = self.expr_mir_ty(iter)?;
        // The Item type: the located `next`'s declared `Option<Item>` return, evaluated under
        // the impl-level substitution of this instantiation.
        let opt_ty = self.impl_fn_ret_ty(&key, span)?;
        let MirTy::Enum(EnumRef::CoreOption, opt_args) = &opt_ty else {
            return unsupported("Iterator::next must return Option", span);
        };
        let elem = opt_args.first().cloned().unwrap_or(MirTy::Unit);
        // WP-C4.7-8.2: a droppable `Item` is supported. Each yielded value is destroyed at the
        // END OF ITS OWN ITERATION — pinned against the oracle first (§0.6): a three-element
        // loop over a printing-destructor Item observes body, value, DROP, body, value, DROP, …
        // rather than three drops at loop exit. `break` also destroys the current iteration's
        // value before leaving.
        let elem_needs_drop = self.ty_needs_drop(&elem, span)?;
        let instance = self.instance_from_key(&key)?;
        self.discovered_callees.push(key);

        // Materialize the iterator into a registered local (it may itself be droppable).
        let it_op = self.lower_expr_to_operand(iter)?;
        self.locals.push(LocalDecl {
            ty: iter_ty.clone(),
            kind: LocalKind::Temp,
        });
        let it_local = LocalId((self.locals.len() - 1) as u32);
        self.register_droppable_local(it_local, &iter_ty, false, span)?;
        self.emit(
            Statement::Assign(Place::local(it_local), Rvalue::Use(it_op)),
            self.synthetic(span, SyntheticKind::ForLoopDesugar),
        );
        self.set_flags_under(it_local.0, &[], true, span);

        let header = self.new_block();
        let body_block = self.new_block();
        let exit = self.new_block();
        self.terminate(
            Terminator::Goto { target: header },
            self.synthetic(span, SyntheticKind::ForLoopDesugar),
            header,
        );
        // header: nxt = next(&mut it); switch on its discriminant.
        let iter_ref_ty = MirTy::Ref {
            mutable: true,
            inner: Box::new(iter_ty),
        };
        let iter_ref = self.new_temp(iter_ref_ty);
        self.emit(
            Statement::Assign(
                Place::local(iter_ref),
                Rvalue::RefOf {
                    mutable: true,
                    place: Place::local(it_local),
                },
            ),
            self.synthetic(span, SyntheticKind::ForLoopDesugar),
        );
        let nxt = self.new_temp(opt_ty);
        let after = self.new_block();
        self.terminate(
            Terminator::Call {
                callee: Callee::Instance(instance),
                args: vec![Operand::Copy(Place::local(iter_ref))],
                dest: Place::local(nxt),
                target: after,
            },
            self.synthetic(span, SyntheticKind::ForLoopDesugar),
            after,
        );
        let disc = self.new_temp(MirTy::Int64);
        self.emit(
            Statement::Assign(Place::local(disc), Rvalue::Discriminant(Place::local(nxt))),
            self.synthetic(span, SyntheticKind::ForLoopDesugar),
        );
        self.terminate(
            Terminator::SwitchInt {
                scrut: Operand::Copy(Place::local(disc)),
                arms: vec![(1, body_block)],
                otherwise: exit,
            },
            self.synthetic(span, SyntheticKind::ForLoopDesugar),
            body_block,
        );
        // body: bind the Item loop variable by value, run the body, loop.
        self.locals.push(LocalDecl {
            ty: elem.clone(),
            kind: LocalKind::User(self.text(var).to_string()),
        });
        let bound = LocalId((self.locals.len() - 1) as u32);
        self.local_map.insert(var_local.0, bound);
        // DEV-124, the by-value half: a user `Iterator` may yield a `Copy` `Item` (`Int32`) or a
        // non-`Copy` one (`String`), and only the second may be moved. `read_place` decides from
        // `elem`; the hand-built `Move` this replaces asserted the answer for both.
        let payload_op = self.read_place(
            Place {
                local: nxt,
                projection: vec![Projection::VariantField(1, 0)],
            },
            &elem,
            span,
        )?;
        self.emit(
            Statement::Assign(Place::local(bound), Rvalue::Use(payload_op)),
            self.synthetic(span, SyntheticKind::ForLoopDesugar),
        );
        // The loop's `scope_depth` is captured BEFORE the per-iteration scope is pushed, so the
        // existing `break`/`continue` handling — which drops every scope from `scope_depth`
        // onward — destroys the current iteration's value on those paths without any special
        // casing. That ordering is the whole trick; pushing the scope first would leave the
        // value alive on `break`.
        let scope_depth = self.scopes.len();
        self.loops.push(LoopTargets {
            continue_target: header,
            break_target: exit,
            scope_depth,
            value_target: None,
        });
        self.scopes.push(Vec::new());
        if elem_needs_drop {
            // Registered with flags FALSE and then set true: the binding is initialized by the
            // move above, and the flag must not be live before that point.
            self.register_droppable_local(bound, &elem, false, span)?;
            self.set_flags_under(bound.0, &[], true, span);
        }
        self.lower_block_value(body)?;
        // Normal end of iteration: destroy this iteration's value before looping back.
        self.emit_scope_drops_from(scope_depth, span);
        self.scopes.pop();
        self.loops.pop();
        self.terminate(
            Terminator::Goto { target: header },
            self.synthetic(span, SyntheticKind::ForLoopDesugar),
            exit,
        );
        // The registered iterator local drops with its enclosing scope (flag live).
        Ok(())
    }

    /// A1: an `ImplFn` instance's declared return type, evaluated under the impl-level
    /// substitution of the key's type arguments (scratch save/restore of the active substs).
    fn impl_fn_ret_ty(&mut self, key: &FnKey, span: Span) -> Result<MirTy, LowerError> {
        let FnKey::ImplFn {
            impl_item,
            member,
            type_args,
            ..
        } = key
        else {
            return unsupported("impl_fn_ret_ty on a non-impl key", span);
        };
        let ItemKind::Impl { items, .. } = &self.hir.item(*impl_item).kind else {
            return unsupported("impl_fn_ret_ty on non-impl", span);
        };
        let hir::ImplItem::Fn { def, .. } = &items[*member as usize] else {
            return unsupported("impl member is not a fn", span);
        };
        let ret_id = match &def.sig.ret {
            hir::RetTy::Ty(t) => *t,
            hir::RetTy::Unit => return Ok(MirTy::Unit),
            hir::RetTy::Never(_) => return unsupported("never-returning method", span),
        };
        let saved_params = self.param_subst.clone();
        let saved_self = self.self_subst.clone();
        for (name, ty) in self.impl_generic_subst(*impl_item, type_args)? {
            self.param_subst.insert(name, ty);
        }
        let result = self.hir_field_ty(ret_id);
        self.param_subst = saved_params;
        self.self_subst = saved_self;
        result
    }

    #[allow(clippy::too_many_arguments)]
    fn lower_for_over_iter(
        &mut self,
        var: Span,
        var_local: crate::hir::LocalId,
        iter: ExprId,
        body: hir::BlockId,
        iter_ty: MirTy,
        elem_ref: MirTy,
        next_rt: RuntimeFn,
        span: Span,
        build_cursor_from_borrow: bool,
    ) -> Result<(), LowerError> {
        // `elem_ref` is the type the loop variable binds to and the `Next` Option's payload:
        // `&T` for Vec/HashMap iteration, `Char` (by value) for `chars()`.
        let opt_ty = MirTy::Enum(EnumRef::CoreOption, vec![elem_ref.clone()]);

        // The cursor owns a borrow of the iterable, so give it a scope that ends at the loop exit
        // rather than registering it in the surrounding source block. Normal exhaustion and
        // `break` converge on `exit`, where this scope is cleaned up; `continue` keeps the cursor
        // live. This is semantically significant for generated Rust: dropping the cursor here ends
        // its shared borrow before a following mutation of the source collection.
        let iterator_scope_depth = self.scopes.len();
        self.scopes.push(Vec::new());

        // Materialize the iterator into a registered droppable local.
        //
        // `build_cursor_from_borrow` is the `for x in &v` form: the expression is a borrow of the
        // Vec, not a cursor, so the cursor is constructed here from it — the same `VecIterNew` the
        // `.iter()` method call emits. Built INSIDE the iterator scope, so the cursor's lifetime is
        // the loop's exactly as in the `.iter()` spelling; the two forms differ in what the user
        // wrote and in nothing else.
        let it_op = if build_cursor_from_borrow {
            let vec_ref = self.lower_expr_to_operand(iter)?;
            let cursor = self.new_temp(iter_ty.clone());
            self.emit_runtime_call(
                RuntimeFn::VecIterNew,
                vec![vec_ref],
                Place::local(cursor),
                span,
            );
            Operand::Move(Place::local(cursor))
        } else {
            self.lower_expr_to_operand(iter)?
        };
        self.locals.push(LocalDecl {
            ty: iter_ty.clone(),
            kind: LocalKind::Temp,
        });
        let it_local = LocalId((self.locals.len() - 1) as u32);
        self.register_droppable_local(it_local, &iter_ty, false, span)?;
        self.emit(
            Statement::Assign(Place::local(it_local), Rvalue::Use(it_op)),
            self.synthetic(span, SyntheticKind::ForLoopDesugar),
        );
        self.set_flags_under(it_local.0, &[], true, span);

        let header = self.new_block();
        let body_block = self.new_block();
        let exit = self.new_block();
        self.terminate(
            Terminator::Goto { target: header },
            self.synthetic(span, SyntheticKind::ForLoopDesugar),
            header,
        );

        // header: nxt = VecIterNext(&mut it); switch on its discriminant.
        let iter_ref_ty = MirTy::Ref {
            mutable: true,
            inner: Box::new(iter_ty),
        };
        let iter_ref = self.new_temp(iter_ref_ty.clone());
        self.emit(
            Statement::Assign(
                Place::local(iter_ref),
                Rvalue::RefOf {
                    mutable: true,
                    place: Place::local(it_local),
                },
            ),
            self.synthetic(span, SyntheticKind::ForLoopDesugar),
        );
        let nxt = self.new_temp(opt_ty);
        self.emit_runtime_call(
            next_rt,
            vec![Operand::Copy(Place::local(iter_ref))],
            Place::local(nxt),
            span,
        );
        let disc = self.new_temp(MirTy::Int64);
        self.emit(
            Statement::Assign(Place::local(disc), Rvalue::Discriminant(Place::local(nxt))),
            self.synthetic(span, SyntheticKind::ForLoopDesugar),
        );
        self.terminate(
            Terminator::SwitchInt {
                scrut: Operand::Copy(Place::local(disc)),
                arms: vec![(1, body_block)],
                otherwise: exit,
            },
            self.synthetic(span, SyntheticKind::ForLoopDesugar),
            body_block,
        );

        // body: bind the &T loop variable, run the body in its own scope, loop.
        self.locals.push(LocalDecl {
            ty: elem_ref.clone(),
            kind: LocalKind::User(self.text(var).to_string()),
        });
        let bound = LocalId((self.locals.len() - 1) as u32);
        self.local_map.insert(var_local.0, bound);
        // **DEV-124: the operand follows the payload's TYPE, not the desugar's opinion.**
        //
        // This was a hand-built `Operand::Move`, moving out of a `Copy` payload — `&T` for
        // `.iter()`, `Char` for `.chars()`. Nothing ever observed it, because the `Option` temp is
        // overwritten at the top of every iteration before anything can read the emptied place:
        // unobservable rather than harmless by design, which is why it survived. INV-MOVE-001
        // found it on its first run.
        //
        // The repair is not "write `copy`" — that is the same mistake with the other constant.
        // Routing through `read_place` makes the operand a function of `elem_ref`, which is what
        // the law requires, and picks up the drop-flag transfer on the move path for free. The
        // by-value sibling below now reads the same way.
        let payload_op = self.read_place(
            Place {
                local: nxt,
                projection: vec![Projection::VariantField(1, 0)],
            },
            &elem_ref,
            span,
        )?;
        self.emit(
            Statement::Assign(Place::local(bound), Rvalue::Use(payload_op)),
            self.synthetic(span, SyntheticKind::ForLoopDesugar),
        );
        self.loops.push(LoopTargets {
            continue_target: header,
            break_target: exit,
            scope_depth: self.scopes.len(),
            value_target: None,
        });
        self.lower_block_value(body)?;
        self.loops.pop();
        self.terminate(
            Terminator::Goto { target: header },
            self.synthetic(span, SyntheticKind::ForLoopDesugar),
            exit,
        );
        self.emit_scope_drops_from(iterator_scope_depth, span);
        self.scopes.pop();
        Ok(())
    }

    /// `v[i] = x` (A1 §5c): `old = VecReplace(&mut v, i, x)`, then drop `old` when the element
    /// type is droppable (install-then-destroy per CD-012; the RHS is already installed by the
    /// time the old value is destroyed).
    fn lower_vec_index_set(
        &mut self,
        base: ExprId,
        index: ExprId,
        elem: MirTy,
        rhs: ExprId,
        span: Span,
    ) -> Result<(), LowerError> {
        // Evaluation order: RHS, then receiver/index (CD-007 keeps RHS first).
        let rhs_op = self.lower_expr_to_operand(rhs)?;
        let recv = self.borrow_vec_receiver(base, true, elem.clone(), span)?;
        let idx = self.lower_expr_to_operand(index)?;
        let idx = self.widen_index_to_u64(idx, index, span)?;
        let old = self.new_temp(elem.clone());
        self.emit_runtime_call(
            RuntimeFn::VecReplace,
            vec![recv, idx, rhs_op],
            Place::local(old),
            span,
        );
        if self.ty_needs_drop(&elem, span)? {
            self.discover_drop_impls(&elem)?;
            self.emit_temp_drop(old, span);
        }
        Ok(())
    }

    /// Coerce a Vec index operand to `UInt64` (the schematic Vec-op index type), inserting a
    /// widening checked cast if the checker did not already type it `UInt64`.
    fn widen_index_to_u64(
        &mut self,
        idx: Operand,
        index_expr: ExprId,
        span: Span,
    ) -> Result<Operand, LowerError> {
        if matches!(self.expr_mir_ty(index_expr)?, MirTy::UInt64) {
            Ok(idx)
        } else {
            self.cast_to_temp(idx, MirTy::UInt64, span)
        }
    }

    /// 0.1-A3 (f-3a): lower a method call on a `HashMap<K, V>` receiver to the map RuntimeFn
    /// surface. The A1 §5a honesty rule stands: no runtime op runs a user destructor —
    /// user-`Drop` K/V types are excluded (`insert`'s replaced value is RETURNED and dropped
    /// by the caller at a visible Drop, the `VecReplace` pattern; String/Vec K/V are fine
    /// since their glue is unobservable buffer reclaim).
    fn lower_map_method_call(
        &mut self,
        base: ExprId,
        kv: Vec<MirTy>,
        name_span: Span,
        args: &[ExprId],
        dest: Place,
        span: Span,
    ) -> Result<(), LowerError> {
        let name = self.text(name_span).to_string();
        let (k, v) = (
            kv.first().cloned().unwrap_or(MirTy::Unit),
            kv.get(1).cloned().unwrap_or(MirTy::Unit),
        );
        // Honesty exclusion: K/V with USER Drop impls would make map internals run
        // destructors invisibly.
        if self.ty_has_user_drop(&k) || self.ty_has_user_drop(&v) {
            return unsupported(
                "HashMap over user-Drop key/value types (reserved — std-full)",
                span,
            );
        }
        let (rt, recv_mut) = match name.as_str() {
            "insert" => (RuntimeFn::HashMapInsert, true),
            "get" => (RuntimeFn::HashMapGet, false),
            "len" => (RuntimeFn::HashMapLen, false),
            "is_empty" => (RuntimeFn::HashMapIsEmpty, false),
            "contains_key" => (RuntimeFn::HashMapContainsKey, false),
            // CD-180. `remove` returns `Option<V>`; like `insert`'s replaced value it is dropped by
            // the discard-drop machinery at a visible Drop terminator when the result is unused.
            "remove" => (RuntimeFn::HashMapRemove, true),
            "clear" => (RuntimeFn::HashMapClear, true),
            "keys" => (RuntimeFn::HashMapKeysIterNew, false),
            _ => return unsupported(format!("HashMap::{name} (reserved — std-full)"), span),
        };
        // WP-C6.3d (STD-HASH-001): key identity is the KEY TYPE'S lawful `Eq`, not structural
        // comparison. Record the selected instance so both the MIR interpreter and the backend
        // dispatch through it; a primitive/`String` key records nothing and keeps its structural
        // comparison, which for those types IS its lawful `Eq`.
        self.discover_eq_impl(&k)?;
        let recv = self.borrow_map_receiver(base, recv_mut, &k, &v, span)?;
        let mut ops = vec![recv];
        for &arg in args {
            ops.push(self.lower_expr_to_operand(arg)?);
        }
        // `insert` returns the replaced `Option<V>` into `dest`; in statement position the
        // discard-drop machinery (StmtKind::Expr) drops it at a visible Drop terminator —
        // the VecReplace pattern, keeping destructors out of the runtime op.
        self.emit_runtime_call(rt, ops, dest, span);
        Ok(())
    }

    /// Does `ty` transitively contain a USER `Drop` impl (as opposed to the unobservable
    /// String/Vec buffer glue)?
    fn ty_has_user_drop(&self, ty: &MirTy) -> bool {
        let mut visited = std::collections::BTreeSet::new();
        self.ty_has_user_drop_guarded(ty, &mut visited)
    }

    fn ty_has_user_drop_guarded(
        &self,
        ty: &MirTy,
        visited: &mut std::collections::BTreeSet<MirTy>,
    ) -> bool {
        if !visited.insert(ty.clone()) {
            return false;
        }
        match ty {
            MirTy::Struct(item, _) | MirTy::Enum(EnumRef::User(item), _) => {
                if self.type_has_drop_impl(*item) {
                    return true;
                }
                // Conservative: user nominals could nest droppables; check fields.
                match nominal_instance_fields(
                    self.hir,
                    self.tables,
                    self.meta,
                    *item,
                    match ty {
                        MirTy::Struct(_, a) | MirTy::Enum(_, a) => a,
                        _ => unreachable!(),
                    },
                    self.providers,
                ) {
                    Ok(NominalFields::Struct(tys)) => tys
                        .iter()
                        .any(|t| self.ty_has_user_drop_guarded(t, visited)),
                    Ok(NominalFields::Enum(vs)) => vs
                        .iter()
                        .any(|v| v.iter().any(|t| self.ty_has_user_drop_guarded(t, visited))),
                    Err(_) => true, // unresolvable: be conservative
                }
            }
            MirTy::Enum(_, args) | MirTy::Core(_, args) | MirTy::Tuple(args) => args
                .iter()
                .any(|t| self.ty_has_user_drop_guarded(t, visited)),
            MirTy::Array(elem, _) => self.ty_has_user_drop_guarded(elem, visited),

            // **EXHAUSTIVE ON PURPOSE.** "No user `Drop` impl governs this type" is an assertion,
            // and this is the narrowest of the drop predicates: it asks specifically about a USER
            // impl, not about glue.
            //
            // A host resource is false here ON PURPOSE and this is the one place that distinction
            // matters: its close is provider-driven, established by A11 §5 rather than by an
            // `impl Drop` the program wrote. It needs drop (`ty_needs_drop`, `may_need_drop`,
            // `mir_needs_drop` all say true) without having a user destructor.
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
            | MirTy::String
            | MirTy::Slice(_)
            | MirTy::Ref { .. }
            | MirTy::FnPtr { .. }
            | MirTy::HostResource(_) => false,
        }
    }

    /// Build a `&HashMap`/`&mut HashMap` receiver operand.
    /// DEV-116: `HashSet<T>` methods. The element IS the key, so this mirrors
    /// `lower_map_method_call` exactly — including `discover_eq_impl`, which is what makes
    /// STD-HASH-001 true here: uniqueness is decided by the element type's lawful `Eq`, dispatched
    /// through the recorded instance, never by structural comparison standing in for it.
    fn lower_set_method_call(
        &mut self,
        base: ExprId,
        elem: MirTy,
        name_span: Span,
        args: &[ExprId],
        dest: Place,
        span: Span,
    ) -> Result<(), LowerError> {
        let name = self.text(name_span).to_string();
        // CE4 (CD-132), unchanged for sets: a user `Drop` element would make set internals run
        // destructors invisibly, and entry Drop order is deliberately unspecified.
        if self.ty_has_user_drop(&elem) {
            return unsupported(
                "HashSet over user-Drop element types (reserved — std-full)",
                span,
            );
        }
        let (rt, recv_mut) = match name.as_str() {
            "insert" => (RuntimeFn::HashSetInsert, true),
            "remove" => (RuntimeFn::HashSetRemove, true),
            "contains" => (RuntimeFn::HashSetContains, false),
            "len" => (RuntimeFn::HashSetLen, false),
            "is_empty" => (RuntimeFn::HashSetIsEmpty, false),
            "clear" => (RuntimeFn::HashSetClear, true),
            // DEV-116-B. A BORROWING cursor: `iter(&self) -> Iter<T>` yielding `&T`, so the
            // receiver is a shared borrow and elements are never moved out of the set.
            "iter" => (RuntimeFn::HashSetIterNew, false),
            _ => return unsupported(format!("HashSet::{name} (reserved — std-full)"), span),
        };
        self.discover_eq_impl(&elem)?;
        let recv = self.borrow_set_receiver(base, recv_mut, &elem, span)?;
        let mut ops = vec![recv];
        for &arg in args {
            ops.push(self.lower_expr_to_operand(arg)?);
        }
        self.emit_runtime_call(rt, ops, dest, span);
        Ok(())
    }

    /// DEV-147: a receiver that is ALREADY a reference must be REBORROWED, not passed through.
    ///
    /// `&mut T` is not `Copy`, so handing the caller's reference straight to the callee lowers to a
    /// `Move` of it. Once per call is harmless; inside a loop it is not — the back-edge sees the
    /// parameter possibly-moved and MIR-0007 refuses the next iteration:
    ///
    /// ```text
    /// fn push_all(out: &mut Vec<UInt8>, ..) { while i < n { out.push(..); .. } }
    ///   -> MIR-0007 push_all@[] bb6: move from possibly-moved place _1[]
    /// ```
    ///
    /// The checker accepted it and the HIR oracle executed it correctly, so this was
    /// accepted-but-unbuildable — and it blocked "append into a caller's buffer in a loop", the
    /// shape of every serializer and encoder.
    ///
    /// A reborrow is `&mut *base`, which is exactly what the `layers == 0` path already builds one
    /// deref further down. Returning `None` means "pass through unchanged", which stays correct for
    /// a SHARED base: `&T` is `Copy`, so reading it moves nothing and there is nothing to fix.
    ///
    /// # DEV-149: the gate is the BASE's mutability, not the RECEIVER's
    ///
    /// DEV-147 first narrowed this to `mutable` — the mutability the METHOD wants. That was the
    /// wrong axis, and it left half the defect in place. A `&mut` base calling a `&self` method is
    /// still broken in both ways at once:
    ///
    /// ```text
    /// fn count(v: &mut Vec<UInt8>) -> UInt64 { v.len() }
    ///   -> MIR-0005 bb0: expected Ref { mutable: false, .. }, found Ref { mutable: true, .. }
    ///   -> MIR-0007 bb4: move from possibly-moved place _1[]
    /// ```
    ///
    /// The reborrow fixes both, because `&*base` from a `&mut` base IS the weakening: it produces
    /// the shared reference the callee wants without consuming the caller's. So the gate is whether
    /// the BASE is `&mut` (is there a non-`Copy` reference at risk of being moved), and the ref
    /// built takes the RECEIVER's mutability (what does the callee actually want).
    ///
    /// The one case that must still refuse is a `&T` base under a `&mut` receiver: lowering must
    /// not invent exclusivity the checker refused. That falls out of gating on the base.
    fn reborrow_reference_receiver(
        &mut self,
        base: ExprId,
        mutable: bool,
        span: Span,
    ) -> Result<Option<Operand>, LowerError> {
        let base_ty = self.expr_mir_ty(base)?;
        let MirTy::Ref {
            mutable: base_mut,
            inner,
        } = &base_ty
        else {
            return Ok(None);
        };
        // Only a `&mut` base can be moved by being read, and only a `&mut` base can be weakened.
        // A `&T` base needs neither: it is `Copy`, and where `&mut` is wanted it is a mutability
        // error the checker already refused — lowering must not invent the capability.
        if !*base_mut {
            return Ok(None);
        }
        let ref_ty = MirTy::Ref {
            mutable,
            inner: inner.clone(),
        };
        let Ok(mut place) = self.lower_place(base) else {
            // A non-place base (a call result, say) has no caller reference to preserve, so moving
            // the temporary is already correct.
            return Ok(None);
        };
        place.projection.push(Projection::Deref);
        let temp = self.new_temp(ref_ty.clone());
        self.emit(
            Statement::Assign(Place::local(temp), Rvalue::RefOf { mutable, place }),
            self.info(span),
        );
        Ok(Some(self.read_place(Place::local(temp), &ref_ty, span)?))
    }

    fn borrow_set_receiver(
        &mut self,
        base: ExprId,
        mutable: bool,
        elem: &MirTy,
        span: Span,
    ) -> Result<Operand, LowerError> {
        let (_, layers) = Self::peel_refs(self.expr_mir_ty(base)?);
        if layers > 0 {
            // DEV-147: reborrow a `&mut` receiver rather than moving the caller's reference.
            if let Some(reborrowed) = self.reborrow_reference_receiver(base, mutable, span)? {
                return Ok(reborrowed);
            }
            return self.lower_expr_to_operand(base);
        }
        let set_ty = MirTy::Core(crate::hir::CoreType::HashSet, vec![elem.clone()]);
        let place = self.place_or_temp(base, &set_ty, span)?;
        let ref_ty = MirTy::Ref {
            mutable,
            inner: Box::new(set_ty),
        };
        let temp = self.new_temp(ref_ty.clone());
        self.emit(
            Statement::Assign(Place::local(temp), Rvalue::RefOf { mutable, place }),
            self.info(span),
        );
        // DEV-127: `read_place`, matching `borrow_map_receiver` three lines below — its sibling
        // already read this way and this one did not. A `&HashSet<T>` receiver is a SHARED
        // reference and therefore `Copy`, so moving it contradicts its type; the `&mut` form is
        // unaffected because `&mut` is not `Copy`, which is why only the shared spellings fired.
        self.read_place(Place::local(temp), &ref_ty, span)
    }

    fn borrow_map_receiver(
        &mut self,
        base: ExprId,
        mutable: bool,
        k: &MirTy,
        v: &MirTy,
        span: Span,
    ) -> Result<Operand, LowerError> {
        let (_, layers) = Self::peel_refs(self.expr_mir_ty(base)?);
        if layers > 0 {
            // DEV-147: reborrow a `&mut` receiver rather than moving the caller's reference.
            if let Some(reborrowed) = self.reborrow_reference_receiver(base, mutable, span)? {
                return Ok(reborrowed);
            }
            return self.lower_expr_to_operand(base);
        }
        let map_ty = MirTy::Core(crate::hir::CoreType::HashMap, vec![k.clone(), v.clone()]);
        let place = self.place_or_temp(base, &map_ty, span)?;
        let ref_ty = MirTy::Ref {
            mutable,
            inner: Box::new(map_ty),
        };
        let temp = self.new_temp(ref_ty.clone());
        self.emit(
            Statement::Assign(Place::local(temp), Rvalue::RefOf { mutable, place }),
            self.info(span),
        );
        self.read_place(Place::local(temp), &ref_ty, span)
    }

    /// Build a `&Vec`/`&mut Vec` receiver operand: pass a reference base through, or borrow an
    /// owned `Vec` place.
    fn borrow_vec_receiver(
        &mut self,
        base: ExprId,
        mutable: bool,
        elem: MirTy,
        span: Span,
    ) -> Result<Operand, LowerError> {
        let (_, layers) = Self::peel_refs(self.expr_mir_ty(base)?);
        if layers > 0 {
            // DEV-147: reborrow a `&mut` receiver rather than moving the caller's reference.
            if let Some(reborrowed) = self.reborrow_reference_receiver(base, mutable, span)? {
                return Ok(reborrowed);
            }
            return self.lower_expr_to_operand(base);
        }
        let vec_ty = MirTy::Core(crate::hir::CoreType::Vec, vec![elem]);
        let place = self.place_or_temp(base, &vec_ty, span)?;
        let ref_ty = MirTy::Ref {
            mutable,
            inner: Box::new(vec_ty),
        };
        let temp = self.new_temp(ref_ty.clone());
        self.emit(
            Statement::Assign(Place::local(temp), Rvalue::RefOf { mutable, place }),
            self.info(span),
        );
        self.read_place(Place::local(temp), &ref_ty, span)
    }

    /// A place for `base`, materializing non-place expressions (call results, literals) into
    /// a temp — receivers and `&expr` operands borrow through this.
    fn place_or_temp(&mut self, base: ExprId, ty: &MirTy, span: Span) -> Result<Place, LowerError> {
        match self.lower_place(base) {
            Ok(place) => Ok(place),
            Err(_) => {
                let value = self.lower_expr_to_operand(base)?;
                let temp = self.new_temp(ty.clone());
                self.emit(
                    Statement::Assign(Place::local(temp), Rvalue::Use(value)),
                    self.info(span),
                );
                Ok(Place::local(temp))
            }
        }
    }

    /// Build a `&String`/`&mut String` receiver operand: pass a reference base through, or
    /// borrow an owned `String` place.
    fn borrow_string_receiver(
        &mut self,
        base: ExprId,
        mutable: bool,
        span: Span,
    ) -> Result<Operand, LowerError> {
        let (_, layers) = Self::peel_refs(self.expr_mir_ty(base)?);
        if layers > 0 {
            // DEV-147: reborrow a `&mut` receiver rather than moving the caller's reference.
            if let Some(reborrowed) = self.reborrow_reference_receiver(base, mutable, span)? {
                return Ok(reborrowed);
            }
            // Already a shared reference to the String — pass it through.
            return self.lower_expr_to_operand(base);
        }
        let place = self.place_or_temp(base, &MirTy::String, span)?;
        let ref_ty = MirTy::Ref {
            mutable,
            inner: Box::new(MirTy::String),
        };
        let temp = self.new_temp(ref_ty.clone());
        self.emit(
            Statement::Assign(Place::local(temp), Rvalue::RefOf { mutable, place }),
            self.info(span),
        );
        self.read_place(Place::local(temp), &ref_ty, span)
    }

    /// 0.1-A13 (WP-C7.9 Packet D): `print`/`println`/`eprint`/`eprintln`.
    ///
    /// One lowering for all four. They differ only in which stream the selected operation
    /// writes to, which `on_current_channel` applies where an operation becomes a call — so the
    /// type dispatch below (str, `Ordering`, a user `Display` impl, composites, `Float32`, the
    /// widened primitives) is written once and cannot drift between the two channels.
    fn lower_output_call(
        &mut self,
        arg: ExprId,
        arg_ty: MirTy,
        is_println: bool,
        dest: Place,
        span: Span,
    ) -> Result<(), LowerError> {
        let args = [arg];
        // A1 (CD-031): printing a `&str` / `String` routes to Print(ln)Str, after
        // an implicit `as_str` for an owned/borrowed String.
        let peeled = Self::peel_refs(arg_ty.clone()).0;
        if matches!(peeled, MirTy::Str | MirTy::String) {
            let str_op = if matches!(peeled, MirTy::String) {
                let recv = self.borrow_string_receiver(args[0], false, span)?;
                let str_ty = MirTy::Ref {
                    mutable: false,
                    inner: Box::new(MirTy::Str),
                };
                let tmp = self.new_temp(str_ty.clone());
                self.emit_runtime_call(RuntimeFn::StringAsStr, vec![recv], Place::local(tmp), span);
                self.read_place(Place::local(tmp), &str_ty, span)?
            } else {
                self.lower_expr_to_operand(args[0])?
            };
            let rt = if is_println {
                RuntimeFn::PrintlnStr
            } else {
                RuntimeFn::PrintStr
            };
            self.emit_runtime_call(rt, vec![str_op], dest, span);
            return Ok(());
        }
        // A4: `println(Ordering)` (Display, deferred from Amendment A2). No runtime
        // op — a discriminant switch prints the variant name via `Print(ln)Str`.
        if matches!(peeled, MirTy::Enum(EnumRef::CoreOrdering, _)) {
            return self.lower_print_ordering(args[0], is_println, dest, span);
        }
        // DEV-089: a user nominal (struct/enum) with its own `Display` impl prints
        // through that impl — an ordinary static call to the selected `Display::fmt`,
        // whose returned `String` is what is printed. The checker's E0500 guarantees
        // any non-standard type reaching here has such an impl.
        if matches!(peeled, MirTy::Struct(..) | MirTy::Enum(EnumRef::User(_), _)) {
            return self.lower_print_display(args[0], &arg_ty, is_println, dest, span);
        }
        // WP-C6.3e: a displayable COMPOSITE (tuple/array of primitive elements in this
        // slice) is rendered as a SEQUENCE of primitive print ops matching the
        // interpreter's `Display for Value` — no runtime-surface change.
        if matches!(
            peeled,
            MirTy::Tuple(_)
                | MirTy::Array(..)
                | MirTy::Enum(EnumRef::CoreOption, _)
                | MirTy::Enum(EnumRef::CoreResult, _)
                | MirTy::Core(crate::hir::CoreType::Vec, _)
        ) {
            return self.lower_print_composite(args[0], &arg_ty, is_println, dest, span);
        }
        // DEV-105: a `Float32` keeps its DECLARED width — widening first would print
        // the shortest round-trip of the f64 it became, not of the f32 that was
        // written. It is the one primitive that must not pass through
        // `widen_for_print`.
        if matches!(peeled, MirTy::Float32) {
            let value = self.lower_expr_to_operand(args[0])?;
            let rt = if is_println {
                RuntimeFn::PrintlnFloat32
            } else {
                RuntimeFn::PrintFloat32
            };
            self.emit_runtime_call(rt, vec![value], dest, span);
            return Ok(());
        }
        let value = self.lower_expr_to_operand(args[0])?;
        let (runtime, widened) = self.widen_for_print(value, &arg_ty, span)?;
        // This site builds its terminator directly rather than through `emit_runtime_call`, so the
        // channel redirect is applied explicitly — it is the one output operation that would
        // otherwise always write to stdout.
        let runtime = match (runtime, is_println) {
            (PrintKind::Int, true) => RuntimeFn::PrintlnInt64,
            (PrintKind::Int, false) => RuntimeFn::PrintInt64,
            (PrintKind::UInt, true) => RuntimeFn::PrintlnUInt64,
            (PrintKind::UInt, false) => RuntimeFn::PrintUInt64,
            (PrintKind::Bool, true) => RuntimeFn::PrintlnBool,
            (PrintKind::Bool, false) => RuntimeFn::PrintBool,
            (PrintKind::Float, true) => RuntimeFn::PrintlnFloat64,
            (PrintKind::Float, false) => RuntimeFn::PrintFloat64,
            (PrintKind::Char, true) => RuntimeFn::PrintlnChar,
            (PrintKind::Char, false) => RuntimeFn::PrintChar,
        };
        let runtime = self.on_current_channel(runtime);
        let after = self.new_block();
        self.terminate(
            Terminator::Call {
                callee: Callee::Runtime(runtime),
                args: vec![widened],
                dest,
                target: after,
            },
            self.info(span),
            after,
        );
        Ok(())
    }

    /// Emit a `Call` to a runtime op with a fresh successor block.
    fn emit_runtime_call(&mut self, rt: RuntimeFn, ops: Vec<Operand>, dest: Place, span: Span) {
        let rt = self.on_current_channel(rt);
        let after = self.new_block();
        self.terminate(
            Terminator::Call {
                callee: Callee::Runtime(rt),
                args: ops,
                dest,
                target: after,
            },
            self.info(span),
            after,
        );
    }

    /// 0.1-A13 (WP-C7.9 Packet D): an output operation, redirected to whichever channel is being
    /// lowered right now.
    ///
    /// `eprint`/`eprintln` differ from `print`/`println` in exactly one respect — the sink — and in
    /// nothing else: the same argument types dispatch the same way, `Ordering` prints its variant
    /// name the same way, a user `Display` impl is invoked the same way, and a composite is
    /// decomposed the same way. Threading a channel argument through all of that dispatch would
    /// have meant editing every site that selects an output operation and hoping none was missed.
    /// Redirecting at the single point where an operation becomes a call cannot miss one.
    ///
    /// Every non-output operation passes through untouched.
    fn on_current_channel(&self, rt: RuntimeFn) -> RuntimeFn {
        if self.out_channel == OutChannel::Stdout {
            return rt;
        }
        match rt {
            RuntimeFn::PrintlnStr => RuntimeFn::EprintlnStr,
            RuntimeFn::PrintStr => RuntimeFn::EprintStr,
            RuntimeFn::PrintlnInt64 => RuntimeFn::EprintlnInt64,
            RuntimeFn::PrintInt64 => RuntimeFn::EprintInt64,
            RuntimeFn::PrintlnUInt64 => RuntimeFn::EprintlnUInt64,
            RuntimeFn::PrintUInt64 => RuntimeFn::EprintUInt64,
            RuntimeFn::PrintlnBool => RuntimeFn::EprintlnBool,
            RuntimeFn::PrintBool => RuntimeFn::EprintBool,
            RuntimeFn::PrintlnFloat64 => RuntimeFn::EprintlnFloat64,
            RuntimeFn::PrintFloat64 => RuntimeFn::EprintFloat64,
            RuntimeFn::PrintlnFloat32 => RuntimeFn::EprintlnFloat32,
            RuntimeFn::PrintFloat32 => RuntimeFn::EprintFloat32,
            RuntimeFn::PrintlnChar => RuntimeFn::EprintlnChar,
            RuntimeFn::PrintChar => RuntimeFn::EprintChar,
            other => other,
        }
    }

    /// A4: `print`/`println` of an `Ordering` value — a discriminant switch that prints the
    /// variant name (`Less`/`Equal`/`Greater`) via `Print(ln)Str`. No runtime op is added.
    fn lower_print_ordering(
        &mut self,
        arg: ExprId,
        is_println: bool,
        dest: Place,
        span: Span,
    ) -> Result<(), LowerError> {
        let ord_ty = MirTy::Enum(EnumRef::CoreOrdering, Vec::new());
        let place = self.place_or_temp(arg, &ord_ty, span)?;
        let disc = self.new_temp(MirTy::Int64);
        self.emit(
            Statement::Assign(Place::local(disc), Rvalue::Discriminant(place)),
            self.info(span),
        );
        let rt = if is_println {
            RuntimeFn::PrintlnStr
        } else {
            RuntimeFn::PrintStr
        };
        let less = self.new_block();
        let equal = self.new_block();
        let greater = self.new_block();
        let join = self.new_block();
        self.terminate(
            Terminator::SwitchInt {
                scrut: Operand::Copy(Place::local(disc)),
                arms: vec![(0, less), (1, equal), (2, greater)],
                otherwise: join,
            },
            self.info(span),
            less,
        );
        // Each variant block prints its name and jumps to `join`. `terminate` seals the current
        // block and advances to the next; after the switch, `current == less`.
        for (name, next) in [("Less", equal), ("Equal", greater), ("Greater", join)] {
            self.terminate(
                Terminator::Call {
                    callee: Callee::Runtime(rt),
                    args: vec![Operand::Const(Constant::Str(name.to_string()))],
                    dest: dest.clone(),
                    target: join,
                },
                self.info(span),
                next,
            );
        }
        Ok(())
    }

    /// WP-C6.3e: emit `PrintStr` of a fixed string literal (the structural punctuation of a
    /// composite's canonical rendering).
    fn print_str_lit(&mut self, text: &str, span: Span) {
        let dest = Place::local(self.new_temp(MirTy::Unit));
        self.emit_runtime_call(
            RuntimeFn::PrintStr,
            vec![Operand::Const(Constant::Str(text.to_string()))],
            dest,
            span,
        );
    }

    /// WP-C6.3e: emit the canonical Display rendering of the value at `place` as a SEQUENCE of print
    /// ops (no trailing newline), matching the interpreter's `Display for Value`. Reuses the
    /// primitive `Print*` ops, so no runtime-surface change. This slice handles primitive elements
    /// and fixed-structure `Tuple`/`Array` of them; `Vec` (a runtime loop), `Option`/`Result`/`Box`,
    /// `str`/`String` elements, and nested user-`Display` land in follow-on slices.
    fn emit_display_value(
        &mut self,
        place: Place,
        ty: &MirTy,
        span: Span,
    ) -> Result<(), LowerError> {
        let (peeled, layers) = Self::peel_refs(ty.clone());
        match &peeled {
            MirTy::Int8
            | MirTy::Int16
            | MirTy::Int32
            | MirTy::Int64
            | MirTy::UInt8
            | MirTy::UInt16
            | MirTy::UInt32
            | MirTy::UInt64
            | MirTy::Bool
            | MirTy::Float32
            | MirTy::Float64
            | MirTy::Char => {
                // DEV-105 CLOSED: a `Float32` element renders at its DECLARED width through the
                // width-preserving op, in every composite context (tuple, array, `Option`,
                // `Result`, `Vec`) — the same selection the scalar path makes. The former refusal
                // here existed only because the widening path was wrong.
                if matches!(peeled, MirTy::Float32) {
                    let value = self.read_place(place, &peeled, span)?;
                    let dest = Place::local(self.new_temp(MirTy::Unit));
                    self.emit_runtime_call(RuntimeFn::PrintFloat32, vec![value], dest, span);
                    return Ok(());
                }
                let op = self.read_place(place, &peeled, span)?;
                let (kind, widened) = self.widen_for_print(op, &peeled, span)?;
                let rt = match kind {
                    PrintKind::Int => RuntimeFn::PrintInt64,
                    PrintKind::UInt => RuntimeFn::PrintUInt64,
                    PrintKind::Bool => RuntimeFn::PrintBool,
                    PrintKind::Float => RuntimeFn::PrintFloat64,
                    PrintKind::Char => RuntimeFn::PrintChar,
                };
                let dest = Place::local(self.new_temp(MirTy::Unit));
                self.emit_runtime_call(rt, vec![widened], dest, span);
                Ok(())
            }
            // A `String` element renders its raw bytes (NO quotes — `Display for Value` line 501),
            // via `&String -> as_str -> &str -> PrintStr`. The element is BORROWED, never moved: the
            // owning composite keeps it and drops it after the whole render (CD-120 Contract C).
            MirTy::String => {
                let str_ref_ty = MirTy::Ref {
                    mutable: false,
                    inner: Box::new(MirTy::Str),
                };
                // `&String`: pass a reference element through; borrow an owned `String` place.
                let string_ref = if layers > 0 {
                    self.read_place(place, ty, span)?
                } else {
                    let owned_ref_ty = MirTy::Ref {
                        mutable: false,
                        inner: Box::new(MirTy::String),
                    };
                    let t = self.new_temp(owned_ref_ty);
                    self.emit(
                        Statement::Assign(
                            Place::local(t),
                            Rvalue::RefOf {
                                mutable: false,
                                place,
                            },
                        ),
                        self.info(span),
                    );
                    Operand::Copy(Place::local(t))
                };
                let str_tmp = self.new_temp(str_ref_ty);
                self.emit_runtime_call(
                    RuntimeFn::StringAsStr,
                    vec![string_ref],
                    Place::local(str_tmp),
                    span,
                );
                let dest = Place::local(self.new_temp(MirTy::Unit));
                self.emit_runtime_call(
                    RuntimeFn::PrintStr,
                    vec![Operand::Copy(Place::local(str_tmp))],
                    dest,
                    span,
                );
                Ok(())
            }
            // A `str` element is `&str` (str is unsized — it appears behind a reference). Print its
            // bytes directly. A reference is Copy, so this reads without disturbing the composite.
            MirTy::Str => {
                let str_op = if layers > 0 {
                    self.read_place(place, ty, span)?
                } else {
                    let str_ref_ty = MirTy::Ref {
                        mutable: false,
                        inner: Box::new(MirTy::Str),
                    };
                    let t = self.new_temp(str_ref_ty);
                    self.emit(
                        Statement::Assign(
                            Place::local(t),
                            Rvalue::RefOf {
                                mutable: false,
                                place,
                            },
                        ),
                        self.info(span),
                    );
                    Operand::Copy(Place::local(t))
                };
                let dest = Place::local(self.new_temp(MirTy::Unit));
                self.emit_runtime_call(RuntimeFn::PrintStr, vec![str_op], dest, span);
                Ok(())
            }
            MirTy::Tuple(elems) => {
                let place = Self::deref_place(place, layers);
                self.print_str_lit("(", span);
                for (i, elem_ty) in elems.iter().enumerate() {
                    if i > 0 {
                        self.print_str_lit(", ", span);
                    }
                    let mut field = place.clone();
                    field.projection.push(Projection::Field(i as u32));
                    self.emit_display_value(field, elem_ty, span)?;
                }
                self.print_str_lit(")", span);
                Ok(())
            }
            MirTy::Array(elem, n) => {
                // The renderer UNROLLS one print sequence per element, so a large array would emit
                // proportional MIR / generated Rust. Bounded until a loop- or iterator-based renderer
                // lands (the same loop `Vec` will need); above the cap it is refused, not silently
                // quadratic. Small heterogeneous tuples stay unrolled (their arity is tiny).
                const MAX_UNROLL: u64 = 64;
                if *n > MAX_UNROLL {
                    return unsupported(
                        format!(
                            "Display of an array longer than {MAX_UNROLL} ({n} elements) is deferred: \
                             the renderer unrolls per element; a loop-based renderer is a later slice"
                        ),
                        span,
                    );
                }
                let place = Self::deref_place(place, layers);
                self.print_str_lit("[", span);
                for i in 0..*n {
                    if i > 0 {
                        self.print_str_lit(", ", span);
                    }
                    let mut element = place.clone();
                    element.projection.push(Projection::ConstIndex(i));
                    self.emit_display_value(element, elem, span)?;
                }
                self.print_str_lit("]", span);
                Ok(())
            }
            // `None` / `Some(v)` (discriminant None=0, Some=1). A discriminant switch prints the
            // variant, recursing into the `Some` payload.
            MirTy::Enum(EnumRef::CoreOption, args) => {
                let inner = args.first().cloned().unwrap_or(MirTy::Unit);
                // The `Some` payload is rendered by BORROW where it is a leaf owner (`String`, a user
                // `Display` nominal) — the backend's trailing variant-field borrow (WP-C6.3e) yields
                // `&payload` for any type. A Copy payload is read by value. A deeper non-Copy payload
                // (e.g. a tuple owning a `String`) needs a non-trailing variant-field VALUE read and
                // is refused by the backend there (cleanly, pre-rustc) — no lowering gate needed.
                let place = Self::deref_place(place, layers);
                let disc = self.new_temp(MirTy::Int64);
                self.emit(
                    Statement::Assign(Place::local(disc), Rvalue::Discriminant(place.clone())),
                    self.info(span),
                );
                let none_blk = self.new_block();
                let some_blk = self.new_block();
                let join = self.new_block();
                self.terminate(
                    Terminator::SwitchInt {
                        scrut: Operand::Copy(Place::local(disc)),
                        arms: vec![(0, none_blk), (1, some_blk)],
                        otherwise: join,
                    },
                    self.info(span),
                    none_blk,
                );
                self.print_str_lit("None", span);
                self.terminate(Terminator::Goto { target: join }, self.info(span), some_blk);
                self.print_str_lit("Some(", span);
                let mut payload = place.clone();
                payload.projection.push(Projection::VariantField(1, 0));
                self.emit_display_value(payload, &inner, span)?;
                self.print_str_lit(")", span);
                self.terminate(Terminator::Goto { target: join }, self.info(span), join);
                Ok(())
            }
            // `Ok(v)` / `Err(e)` (discriminant Ok=0, Err=1).
            MirTy::Enum(EnumRef::CoreResult, args) => {
                let ok_ty = args.first().cloned().unwrap_or(MirTy::Unit);
                let err_ty = args.get(1).cloned().unwrap_or(MirTy::Unit);
                // As for `Option`: a leaf owner payload (`String`, a user `Display` nominal) renders
                // by the backend's trailing variant-field borrow; a Copy payload by value; a deeper
                // non-Copy payload is refused by the backend (pre-rustc). No lowering gate needed.
                let place = Self::deref_place(place, layers);
                let disc = self.new_temp(MirTy::Int64);
                self.emit(
                    Statement::Assign(Place::local(disc), Rvalue::Discriminant(place.clone())),
                    self.info(span),
                );
                let ok_blk = self.new_block();
                let err_blk = self.new_block();
                let join = self.new_block();
                self.terminate(
                    Terminator::SwitchInt {
                        scrut: Operand::Copy(Place::local(disc)),
                        arms: vec![(0, ok_blk), (1, err_blk)],
                        otherwise: join,
                    },
                    self.info(span),
                    ok_blk,
                );
                self.print_str_lit("Ok(", span);
                let mut ok_payload = place.clone();
                ok_payload.projection.push(Projection::VariantField(0, 0));
                self.emit_display_value(ok_payload, &ok_ty, span)?;
                self.print_str_lit(")", span);
                self.terminate(Terminator::Goto { target: join }, self.info(span), err_blk);
                self.print_str_lit("Err(", span);
                let mut err_payload = place.clone();
                err_payload.projection.push(Projection::VariantField(1, 0));
                self.emit_display_value(err_payload, &err_ty, span)?;
                self.print_str_lit(")", span);
                self.terminate(Terminator::Goto { target: join }, self.info(span), join);
                Ok(())
            }
            // A `Vec<T>` renders `[e0, e1, …]` with a runtime LOOP (its length is dynamic, unlike a
            // fixed array), built against CD-120's contracts: the per-element print-op sequence is
            // emitted in index order (Contract A), and a trap in an element leaves exactly the
            // prefix printed so far (Contract B — the loop emits, never buffers). `VecIndexGet`
            // yields the element by COPY (V-COPY-1), so this slice requires a Copy element; a
            // non-Copy element (String/Box/Vec) is a later slice. The owning Vec's own destructor
            // runs after the whole render, in `lower_print_composite` (Contract C).
            MirTy::Core(crate::hir::CoreType::Vec, elem_args) => {
                let elem = elem_args.first().cloned().unwrap_or(MirTy::Unit);
                // A Copy element is read BY VALUE (`VecIndexGet`); an owning element (`String`, …)
                // is read BY REFERENCE (`VecGetRef`), because copying it out would duplicate a value
                // the Vec still owns. Both render through the same recursion.
                let elem_is_copy = self.is_copy(&elem);
                // CD-127: a user-nominal element renders through its `Display::fmt`, whose returned
                // `String` is borrowed then dropped PER ITERATION. That loop-carried borrow used to
                // be rejected (E0502) because the dispatch loop gave rustc no borrow precision
                // inside loops; structured control-flow emission fixed it, so no gate remains.
                let vec_ty = MirTy::Core(crate::hir::CoreType::Vec, vec![elem.clone()]);
                let ref_ty = MirTy::Ref {
                    mutable: false,
                    inner: Box::new(vec_ty),
                };
                // A FRESH shared borrow per runtime read (length, then each element): the owning Vec
                // is dropped after the render (Contract C), so a single reused `&Vec` held across the
                // loop would still be live at that drop (E0502). Each short borrow dies at its call.
                let len = self.new_temp(MirTy::UInt64);
                {
                    let vref = self.vec_ref_for_display(&place, ty, layers, &ref_ty, span)?;
                    self.emit_runtime_call(RuntimeFn::VecLen, vec![vref], Place::local(len), span);
                }
                let idx = self.new_temp(MirTy::UInt64);
                self.emit(
                    Statement::Assign(
                        Place::local(idx),
                        Rvalue::Use(Operand::Const(Constant::Int(0, MirTy::UInt64))),
                    ),
                    self.info(span),
                );
                self.print_str_lit("[", span);
                let header = self.new_block();
                let body = self.new_block();
                let exit = self.new_block();
                self.terminate(Terminator::Goto { target: header }, self.info(span), header);
                // header: `idx < len` ?
                let cond = self.new_temp(MirTy::Bool);
                self.emit(
                    Statement::Assign(
                        Place::local(cond),
                        Rvalue::BinOp(
                            MirBinOp::Lt,
                            Operand::Copy(Place::local(idx)),
                            Operand::Copy(Place::local(len)),
                        ),
                    ),
                    self.info(span),
                );
                self.terminate(
                    Terminator::SwitchInt {
                        scrut: Operand::Copy(Place::local(cond)),
                        arms: vec![(1, body)],
                        otherwise: exit,
                    },
                    self.info(span),
                    body,
                );
                // body: a separator `", "` before every element but the first.
                let sep = self.new_block();
                let render = self.new_block();
                let is_first = self.new_temp(MirTy::Bool);
                self.emit(
                    Statement::Assign(
                        Place::local(is_first),
                        Rvalue::BinOp(
                            MirBinOp::Eq,
                            Operand::Copy(Place::local(idx)),
                            Operand::Const(Constant::Int(0, MirTy::UInt64)),
                        ),
                    ),
                    self.info(span),
                );
                self.terminate(
                    Terminator::SwitchInt {
                        scrut: Operand::Copy(Place::local(is_first)),
                        arms: vec![(1, render)],
                        otherwise: sep,
                    },
                    self.info(span),
                    sep,
                );
                self.print_str_lit(", ", span);
                self.terminate(Terminator::Goto { target: render }, self.info(span), render);
                // render: read the element, display it, then `idx += 1` and loop. Each read takes a
                // FRESH shared borrow (see above) that dies at its runtime call.
                {
                    let vref = self.vec_ref_for_display(&place, ty, layers, &ref_ty, span)?;
                    if elem_is_copy {
                        let elem_tmp = self.new_temp(elem.clone());
                        self.emit_runtime_call(
                            RuntimeFn::VecIndexGet,
                            vec![vref, Operand::Copy(Place::local(idx))],
                            Place::local(elem_tmp),
                            span,
                        );
                        self.emit_display_value(Place::local(elem_tmp), &elem, span)?;
                    } else {
                        // `VecGetRef` yields `Option<&T>` and never traps — `idx < len` holds here,
                        // so the `None` arm is unreachable, but it is still a real discriminant
                        // switch rather than an assumption. The `Some` payload is a `&T` reached
                        // through a trailing `VariantField`, which CD-126 made borrowable.
                        let elem_ref_ty = MirTy::Ref {
                            mutable: false,
                            inner: Box::new(elem.clone()),
                        };
                        let opt_ty = MirTy::Enum(EnumRef::CoreOption, vec![elem_ref_ty.clone()]);
                        let opt = self.new_temp(opt_ty);
                        self.emit_runtime_call(
                            RuntimeFn::VecGetRef,
                            vec![vref, Operand::Copy(Place::local(idx))],
                            Place::local(opt),
                            span,
                        );
                        let disc = self.new_temp(MirTy::Int64);
                        self.emit(
                            Statement::Assign(
                                Place::local(disc),
                                Rvalue::Discriminant(Place::local(opt)),
                            ),
                            self.info(span),
                        );
                        let some_blk = self.new_block();
                        let after = self.new_block();
                        self.terminate(
                            Terminator::SwitchInt {
                                scrut: Operand::Copy(Place::local(disc)),
                                arms: vec![(1, some_blk)],
                                otherwise: after,
                            },
                            self.info(span),
                            some_blk,
                        );
                        let mut payload = Place::local(opt);
                        payload.projection.push(Projection::VariantField(1, 0));
                        self.emit_display_value(payload, &elem_ref_ty, span)?;
                        self.terminate(Terminator::Goto { target: after }, self.info(span), after);
                    }
                }
                // idx += 1 — a checked add, like every other integer arithmetic in MIR.
                let after_incr = self.new_block();
                self.terminate(
                    Terminator::Checked {
                        op: CheckedOp::Add,
                        args: vec![
                            Operand::Copy(Place::local(idx)),
                            Operand::Const(Constant::Int(1, MirTy::UInt64)),
                        ],
                        dest: idx,
                        target: after_incr,
                        trap: TrapInfo {
                            category: TrapCategory::IntegerOverflow,
                            source: self.info(span),
                        },
                    },
                    self.info(span),
                    after_incr,
                );
                self.terminate(Terminator::Goto { target: header }, self.info(span), exit);
                self.print_str_lit("]", span);
                Ok(())
            }
            // A nested user nominal with its own `Display` impl: call its `fmt(&self) -> String` on
            // the element BORROWED IN PLACE (the owning composite keeps it and drops it later —
            // Contract C), print the returned String (no newline — an element), then drop that
            // String. Same machinery as the top-level `lower_print_display`, without the arg-drop.
            MirTy::Struct(item, args) | MirTy::Enum(EnumRef::User(item), args) => {
                let nominal = *item;
                let nominal_args = args.clone();
                let Some((key, receiver)) = self.find_impl_fn(nominal, "fmt", false, &nominal_args)
                else {
                    return unsupported(
                        "Display::fmt not found for a composite element (only standard-library and \
                         user `Display` types render inside a composite)",
                        span,
                    );
                };
                if !matches!(receiver, Some(hir::Receiver::Ref)) {
                    return unsupported("Display::fmt with a non-&self receiver", span);
                }
                // `&element`: a reference element passes through; an owned element is borrowed.
                let recv_op = if layers > 0 {
                    self.read_place(place, ty, span)?
                } else {
                    let ref_ty = MirTy::Ref {
                        mutable: false,
                        inner: Box::new(peeled.clone()),
                    };
                    let ref_tmp = self.new_temp(ref_ty.clone());
                    self.emit(
                        Statement::Assign(
                            Place::local(ref_tmp),
                            Rvalue::RefOf {
                                mutable: false,
                                place,
                            },
                        ),
                        self.info(span),
                    );
                    self.read_place(Place::local(ref_tmp), &ref_ty, span)?
                };
                let instance = self.instance_from_key(&key)?;
                self.discovered_callees.push(key);
                let str_result = self.new_temp(MirTy::String);
                let after_fmt = self.new_block();
                self.terminate(
                    Terminator::Call {
                        callee: Callee::Instance(instance),
                        args: vec![recv_op],
                        dest: Place::local(str_result),
                        target: after_fmt,
                    },
                    self.info(span),
                    after_fmt,
                );
                // `String::as_str` then `PrintStr` (no newline — a composite element).
                let str_ref_ty = MirTy::Ref {
                    mutable: false,
                    inner: Box::new(MirTy::String),
                };
                let str_ref = self.new_temp(str_ref_ty);
                self.emit(
                    Statement::Assign(
                        Place::local(str_ref),
                        Rvalue::RefOf {
                            mutable: false,
                            place: Place::local(str_result),
                        },
                    ),
                    self.info(span),
                );
                let as_str_ty = MirTy::Ref {
                    mutable: false,
                    inner: Box::new(MirTy::Str),
                };
                let as_str = self.new_temp(as_str_ty.clone());
                self.emit_runtime_call(
                    RuntimeFn::StringAsStr,
                    vec![Operand::Copy(Place::local(str_ref))],
                    Place::local(as_str),
                    span,
                );
                let str_op = self.read_place(Place::local(as_str), &as_str_ty, span)?;
                let dest = Place::local(self.new_temp(MirTy::Unit));
                self.emit_runtime_call(RuntimeFn::PrintStr, vec![str_op], dest, span);
                // Drop the formatting String (the element itself is dropped by the owning composite).
                let after_str_drop = self.new_block();
                self.terminate(
                    Terminator::Drop {
                        place: Place::local(str_result),
                        target: after_str_drop,
                    },
                    self.info(span),
                    after_str_drop,
                );
                Ok(())
            }
            other => unsupported(
                format!("Display of {other:?} inside a composite is a later C6.3e slice"),
                span,
            ),
        }
    }

    /// WP-C6.3e: lower `print`/`println` of a displayable composite (tuple/array). The value is
    /// materialised into a temporary so its fields/elements can be projected, rendered as a print
    /// sequence, then a trailing newline for `println`. This slice restricts to `Copy` composites
    /// (tuple/array of primitives), which own nothing to drop.
    fn lower_print_composite(
        &mut self,
        arg: ExprId,
        arg_ty: &MirTy,
        is_println: bool,
        dest: Place,
        span: Span,
    ) -> Result<(), LowerError> {
        let (peeled, _) = Self::peel_refs(arg_ty.clone());
        // A droppable (non-Copy) composite — a tuple/array/`Option`/`Result`/`Vec` that OWNS
        // `String`/`Box`/`Vec` elements — is supported: `emit_display_value` renders each element
        // in place (borrowing owners, never moving out of `tmp`), and the whole composite is dropped
        // after the render (CD-120 Contract C). `emit_display_value` is the real filter — an element
        // it cannot render (e.g. a user `Drop` struct, or a `Vec` of non-Copy elements) is a clean
        // `Unsupported` there. A `&Vec`/`&String` composite never reaches here (the typechecker
        // rejects `println(&x)`, E0500), so an owned composite is moved into `tmp` and this lowering
        // owns the sole copy to drop.
        let droppable = !self.is_copy(&peeled);
        // A DROPPABLE (slot-backed) composite that also carries a borrow — `(String, &str, i32)` —
        // reads its fields through a generated projection wrapper whose return borrows the slot; the
        // backend does not emit the lifetime that ties them (E0106). Refuse until generated lifetimes
        // land. A COPY borrow-carrying composite (`(&str, i32)`) is fine: no slot, no wrapper.
        if droppable && ty_carries_ref(&peeled) {
            return unsupported(
                "Display of a droppable composite that also carries a borrowed element (e.g. \
                 `&str` beside an owned field) needs generated lifetimes — a later C6.3e slice",
                span,
            );
        }
        let value = self.lower_expr_to_operand(arg)?;
        let tmp = self.new_temp(peeled.clone());
        self.emit(
            Statement::Assign(Place::local(tmp), Rvalue::Use(value)),
            self.info(span),
        );
        self.emit_display_value(Place::local(tmp), &peeled, span)?;
        if is_println {
            let d = Place::local(self.new_temp(MirTy::Unit));
            self.emit_runtime_call(
                RuntimeFn::PrintlnStr,
                vec![Operand::Const(Constant::Str(String::new()))],
                d,
                span,
            );
        }
        // CD-120 Contract C: a droppable composite's destructor runs AFTER the whole render
        // (including the trailing newline), never interleaved with the printed bytes.
        if droppable {
            let after_drop = self.new_block();
            self.terminate(
                Terminator::Drop {
                    place: Place::local(tmp),
                    target: after_drop,
                },
                self.info(span),
                after_drop,
            );
        }
        self.emit(
            Statement::Assign(dest, Rvalue::Use(Operand::Const(Constant::Unit))),
            self.info(span),
        );
        Ok(())
    }

    /// DEV-089 (WP-C4.7 close-out, §4): lower `print`/`println` of a user nominal that has its
    /// own `Display` impl. Emitted as ordinary visible MIR — a static `Callee::Instance` call to
    /// the selected `Display::fmt` (so user code, traps and provenance stay visible), then the
    /// existing `StringAsStr` + `Print(ln)Str` runtime surface, then a visible `Drop` of the
    /// formatting `String` and (for an owned by-value argument) the argument itself. No new MIR
    /// shape, no new `RuntimeFn`, no runtime-surface bump.
    /// DEV-DISPLAY-DISPATCH: lower `x.fmt()` where `x`'s type is one of the standard library's
    /// own `Display` implementors (06-Standard-Library declares `impl Display for Int32` and
    /// "similar for other types"; no source file writes those blocks, so `find_impl_fn` can never
    /// find them).
    ///
    /// **Ownership.** `Display::fmt` takes `&self`, so this must not consume the receiver. Every
    /// scalar here is `Copy`, so `read_place` yields a `Copy` operand and the receiver is
    /// untouched; `String` is NOT `Copy`, so it is read through a shared reference and cloned.
    /// That is what makes `let a = x.fmt(); let b = x.fmt();` legal at this level rather than a
    /// V-MOVE-1 failure on the second call.
    fn lower_display_fmt(
        &mut self,
        base: ExprId,
        peeled_ty: &MirTy,
        base_ref_layers: u32,
        kind: FmtReceiver,
        dest: Place,
        span: Span,
    ) -> Result<(), LowerError> {
        // The receiver expression as a place AT ITS OWN TYPE — which may itself be a reference.
        // A non-place receiver (`make_int().fmt()`) is materialised into a temp first, exactly as
        // the ordinary method-receiver path does.
        let base_place = match self.lower_place(base) {
            Ok(place) => place,
            Err(_) => {
                let value = self.lower_expr_to_operand(base)?;
                let base_ty = self.expr_mir_ty(base)?;
                let temp = self.new_temp(base_ty);
                self.emit(
                    Statement::Assign(Place::local(temp), Rvalue::Use(value)),
                    self.info(span),
                );
                Place::local(temp)
            }
        };
        // `str` is unsized: the referent is never a place a `RefOf` may take, so the `&str` that
        // already exists one level up is what gets read. Everything else projects all the way down
        // to the value (TYPE-METHOD-002 auto-dereference).
        if let FmtReceiver::StrSlice = kind {
            let str_ref_ty = MirTy::Ref {
                mutable: false,
                inner: Box::new(MirTy::Str),
            };
            let ref_place = Self::deref_place(base_place, base_ref_layers.saturating_sub(1));
            let str_op = self.read_place(ref_place, &str_ref_ty, span)?;
            self.emit_runtime_call(RuntimeFn::StrToString, vec![str_op], dest, span);
            return Ok(());
        }
        let place = Self::deref_place(base_place, base_ref_layers);
        match kind {
            FmtReceiver::Unit => {
                self.emit_runtime_call(RuntimeFn::FmtUnit, Vec::new(), dest, span);
            }
            FmtReceiver::Scalar => {
                let value = self.read_place(place, peeled_ty, span)?;
                let (print_kind, widened) = self.widen_for_print(value, peeled_ty, span)?;
                let rt = match print_kind {
                    PrintKind::Int => RuntimeFn::FmtInt64,
                    PrintKind::UInt => RuntimeFn::FmtUInt64,
                    PrintKind::Bool => RuntimeFn::FmtBool,
                    PrintKind::Char => RuntimeFn::FmtChar,
                    PrintKind::Float => RuntimeFn::FmtFloat64,
                };
                self.emit_runtime_call(rt, vec![widened], dest, span);
            }
            FmtReceiver::Float32 => {
                let value = self.read_place(place, peeled_ty, span)?;
                self.emit_runtime_call(RuntimeFn::FmtFloat32, vec![value], dest, span);
            }
            // `&String -> StringAsStr -> &str -> StrToString`, the same borrow-then-copy shape
            // `emit_display_value` uses for a `String` element. The receiver is never moved.
            FmtReceiver::StringOwned => {
                let owned_ref_ty = MirTy::Ref {
                    mutable: false,
                    inner: Box::new(MirTy::String),
                };
                let string_ref = self.new_temp(owned_ref_ty.clone());
                self.emit(
                    Statement::Assign(
                        Place::local(string_ref),
                        Rvalue::RefOf {
                            mutable: false,
                            place,
                        },
                    ),
                    self.info(span),
                );
                let str_ref_ty = MirTy::Ref {
                    mutable: false,
                    inner: Box::new(MirTy::Str),
                };
                let str_tmp = self.new_temp(str_ref_ty);
                self.emit_runtime_call(
                    RuntimeFn::StringAsStr,
                    vec![Operand::Copy(Place::local(string_ref))],
                    Place::local(str_tmp),
                    span,
                );
                self.emit_runtime_call(
                    RuntimeFn::StrToString,
                    vec![Operand::Copy(Place::local(str_tmp))],
                    dest,
                    span,
                );
            }
            // Handled above, before the receiver place was dereferenced.
            FmtReceiver::StrSlice => unreachable!("str receiver is lowered before this match"),
        }
        Ok(())
    }

    fn lower_print_display(
        &mut self,
        arg: ExprId,
        arg_ty: &MirTy,
        is_println: bool,
        dest: Place,
        span: Span,
    ) -> Result<(), LowerError> {
        let (peeled, ref_layers) = Self::peel_refs(arg_ty.clone());
        let (nominal, nominal_args) = match &peeled {
            MirTy::Struct(item, args) | MirTy::Enum(EnumRef::User(item), args) => {
                (*item, args.clone())
            }
            other => return unsupported(format!("Display print on non-nominal {other:?}"), span),
        };
        let Some((key, receiver)) = self.find_impl_fn(nominal, "fmt", false, &nominal_args) else {
            return unsupported("Display::fmt not found for printed type", span);
        };
        if !matches!(receiver, Some(hir::Receiver::Ref)) {
            // The canonical `Display::fmt` is `&self`. Anything else is outside this path.
            return unsupported("Display::fmt with a non-&self receiver", span);
        }

        // The by-value argument gets its own storage so `&self` can borrow it and its destructor
        // runs at the call (ordinary by-value call ownership). A `&self`/`&mut self` argument
        // expression (`println(&x)`) is already a reference we do not own — borrow through it and
        // owe no drop.
        let (recv_op, owned_arg) = if ref_layers > 0 {
            (self.lower_expr_to_operand(arg)?, None)
        } else {
            let value = self.lower_expr_to_operand(arg)?;
            let arg_tmp = self.new_temp(peeled.clone());
            self.emit(
                Statement::Assign(Place::local(arg_tmp), Rvalue::Use(value)),
                self.info(span),
            );
            let ref_ty = MirTy::Ref {
                mutable: false,
                inner: Box::new(peeled.clone()),
            };
            let ref_tmp = self.new_temp(ref_ty.clone());
            self.emit(
                Statement::Assign(
                    Place::local(ref_tmp),
                    Rvalue::RefOf {
                        mutable: false,
                        place: Place::local(arg_tmp),
                    },
                ),
                self.info(span),
            );
            (
                self.read_place(Place::local(ref_tmp), &ref_ty, span)?,
                Some(arg_tmp),
            )
        };

        // Ordinary static call to the selected `Display::fmt(&self) -> String`.
        let instance = self.instance_from_key(&key)?;
        self.discovered_callees.push(key);
        let str_result = self.new_temp(MirTy::String);
        let after_fmt = self.new_block();
        self.terminate(
            Terminator::Call {
                callee: Callee::Instance(instance),
                args: vec![recv_op],
                dest: Place::local(str_result),
                target: after_fmt,
            },
            self.info(span),
            after_fmt,
        );

        // `String::as_str` then the existing `Print(ln)Str` runtime op.
        let str_ref_ty = MirTy::Ref {
            mutable: false,
            inner: Box::new(MirTy::String),
        };
        let str_ref = self.new_temp(str_ref_ty);
        self.emit(
            Statement::Assign(
                Place::local(str_ref),
                Rvalue::RefOf {
                    mutable: false,
                    place: Place::local(str_result),
                },
            ),
            self.info(span),
        );
        let as_str_ty = MirTy::Ref {
            mutable: false,
            inner: Box::new(MirTy::Str),
        };
        let as_str = self.new_temp(as_str_ty.clone());
        self.emit_runtime_call(
            RuntimeFn::StringAsStr,
            vec![Operand::Copy(Place::local(str_ref))],
            Place::local(as_str),
            span,
        );
        let str_op = self.read_place(Place::local(as_str), &as_str_ty, span)?;
        let rt = if is_println {
            RuntimeFn::PrintlnStr
        } else {
            RuntimeFn::PrintStr
        };
        self.emit_runtime_call(rt, vec![str_op], dest, span);

        // Visible Drop of the formatting String (after its bytes were submitted), then of the
        // by-value argument (its destructor is observable; the oracle drops it here too).
        let after_str_drop = self.new_block();
        self.terminate(
            Terminator::Drop {
                place: Place::local(str_result),
                target: after_str_drop,
            },
            self.info(span),
            after_str_drop,
        );
        // The by-value argument's destructor runs here (observable — the oracle drops it too), but
        // ONLY when it is droppable. A `Copy` printed type (e.g. a plain `struct P { v: Int32 }`
        // whose `Display` is by `&self`) has no destructor; emitting a `Drop` on it is a no-op the
        // interpreter ignores but the native backend refuses (Copy has no slot), so skip it.
        if let Some(arg_tmp) = owned_arg {
            if !self.is_copy(&peeled) {
                let after_arg_drop = self.new_block();
                self.terminate(
                    Terminator::Drop {
                        place: Place::local(arg_tmp),
                        target: after_arg_drop,
                    },
                    self.info(span),
                    after_arg_drop,
                );
            }
        }
        Ok(())
    }

    fn widen_for_print(
        &mut self,
        value: Operand,
        ty: &MirTy,
        span: Span,
    ) -> Result<(PrintKind, Operand), LowerError> {
        match ty {
            MirTy::Bool => Ok((PrintKind::Bool, value)),
            MirTy::Char => Ok((PrintKind::Char, value)),
            MirTy::Float64 => Ok((PrintKind::Float, value)),
            MirTy::Int64 => Ok((PrintKind::Int, value)),
            MirTy::UInt64 => Ok((PrintKind::UInt, value)),
            MirTy::Int8 | MirTy::Int16 | MirTy::Int32 => {
                let widened = self.cast_to_temp(value, MirTy::Int64, span)?;
                Ok((PrintKind::Int, widened))
            }
            MirTy::UInt8 | MirTy::UInt16 | MirTy::UInt32 => {
                let widened = self.cast_to_temp(value, MirTy::UInt64, span)?;
                Ok((PrintKind::UInt, widened))
            }
            // DEV-105 CLOSED (0.1-A9): a `Float32` never reaches here — both the scalar and the
            // composite path select `PrintFloat32`/`PrintlnFloat32`, which preserve the declared
            // width. Widening it would print the shortest round-trip of the f64 it became rather
            // than of the `Float32` that was written, which PRINT-DISPLAY-001 forbids.
            MirTy::Float32 => unsupported(
                "internal: `Float32` must select the width-preserving print op, not be widened \
                 (DEV-105 / 0.1-A9)",
                span,
            ),
            _ => unsupported("print/println of this type (C4.5)", span),
        }
    }

    fn cast_to_temp(
        &mut self,
        value: Operand,
        to: MirTy,
        span: Span,
    ) -> Result<Operand, LowerError> {
        // Widening casts cannot fail; still lowered as `Checked Cast` per the contract (all
        // casts are checked terminators — uniformity over cleverness in v0.1).
        self.checked_to_temp(
            CheckedOp::Cast,
            vec![value],
            to,
            TrapCategory::CastFailure,
            span,
        )
    }

    // ---- match ----

    /// A2/DEV-070: does this scrutinee expression read a place THROUGH a shared reference
    /// (`match *self`, `match self.state` behind `&self`)? Such a match must not move from —
    /// and poison — the borrowed place; it matches by reference instead (`MatchMode::ByRef`).
    fn scrutinee_reads_through_ref(&self, expr: ExprId) -> bool {
        match &self.hir.expr(expr).kind {
            hir::ExprKind::Unary {
                op: UnOp::Deref, ..
            } => true,
            hir::ExprKind::Field { base, .. } | hir::ExprKind::TupleField { base, .. } => {
                matches!(self.tables.expr_types.get(base), Some(Ty::Ref { .. }))
                    || self.scrutinee_reads_through_ref(*base)
            }
            _ => false,
        }
    }

    fn lower_match(&mut self, expr: ExprId, dest: Option<Place>) -> Result<(), LowerError> {
        let span = self.hir.expr(expr).span;
        let hir::ExprKind::Match { scrutinee, arms } = &self.hir.expr(expr).kind else {
            return unsupported("not a match", span);
        };
        let scrutinee = *scrutinee;
        let arms: Vec<_> = arms.iter().map(|a| (a.pat, a.body)).collect();

        let scrut_ty = self.expr_mir_ty(scrutinee)?;
        // A2/DEV-070: a scrutinee read through a shared reference is matched BY REFERENCE — no
        // move, no poison, no arm-end drops; the referent stays owned by the caller. Consumption
        // depends on the scrutinee, not on a blanket rule (CE3 requirement): owned scrutinees
        // keep the C4.5d consuming semantics below. User-`Drop` scrutinee types stay unsupported
        // by-ref (the oracle's legacy clone would run the dtor on the clone; front-end move-
        // out-of-borrow checking is the real fix and is recorded as a front-end gap).
        let by_ref = self.scrutinee_reads_through_ref(scrutinee);
        let (scrut_place, mode) = if by_ref {
            if self.ty_has_user_drop(&scrut_ty) {
                return unsupported(
                    "match through a reference on a user-Drop type (front-end move-out-of-borrow gap)",
                    span,
                );
            }
            (self.lower_place(scrutinee)?, MatchMode::ByRef)
        } else {
            // Materialize the scrutinee once. The initial move clears the source local's drop
            // flag (if it was a registered droppable local), so the scrutinee temp — not the
            // source — is what the arms consume; a temp is never auto-dropped, so no
            // double-drop can occur.
            let scrut_local = self.new_temp(scrut_ty.clone());
            let scrut_value = self.lower_expr_to_operand(scrutinee)?;
            self.emit(
                Statement::Assign(Place::local(scrut_local), Rvalue::Use(scrut_value)),
                self.synthetic(span, SyntheticKind::MatchDesugar),
            );
            (Place::local(scrut_local), MatchMode::Consuming)
        };

        let join = self.new_block();
        match &scrut_ty {
            // Flat enum arms keep the proven, drop-elaborated C4.5d path.
            MirTy::Enum(enum_ref, args) if self.enum_arms_are_flat(&arms) => self
                .lower_enum_match(
                    *enum_ref,
                    args.clone(),
                    scrut_place,
                    mode,
                    &arms,
                    dest,
                    join,
                    span,
                )?,
            MirTy::Bool
            | MirTy::Int8
            | MirTy::Int16
            | MirTy::Int32
            | MirTy::Int64
            | MirTy::UInt8
            | MirTy::UInt16
            | MirTy::UInt32
            | MirTy::UInt64
            | MirTy::Char => self.lower_int_match(scrut_place, &arms, dest, join, span)?,
            // A2-2: everything else — tuple/array/struct/Float/&str scrutinees and NESTED enum
            // patterns — routes to the general recursive engine. WP-C4.7-8.3b: a DROPPABLE
            // scrutinee is handled there too, by generalizing C4.5d's drop-unit decomposition to
            // arbitrary pattern trees (unbound leaves consumed into temps, bindings registered).
            MirTy::Enum(..)
            | MirTy::Tuple(_)
            | MirTy::Array(..)
            | MirTy::Struct(..)
            | MirTy::Float32
            | MirTy::Float64
            | MirTy::Ref { .. } => {
                self.lower_general_match(scrut_place, scrut_ty, mode, &arms, dest, join, span)?
            }
            _ => return unsupported("match scrutinee type (C4.5)", span),
        }
        self.current = join;
        Ok(())
    }

    fn lower_int_match(
        &mut self,
        scrut: Place,
        arms: &[(hir::PatId, ExprId)],
        dest: Option<Place>,
        join: BlockId,
        span: Span,
    ) -> Result<(), LowerError> {
        // Chain: literal arms become SwitchInt cases; the first wildcard/binding arm is the
        // fallthrough. (Usefulness/exhaustiveness were verified upstream.)
        let mut cases: Vec<(u128, hir::PatId, ExprId)> = Vec::new();
        let mut default: Option<(hir::PatId, ExprId)> = None;
        for &(pat, body) in arms {
            match &self.hir.pat(pat).kind {
                hir::PatKind::Lit(lit) => {
                    let pat_span = self.hir.pat(pat).span;
                    let value = match lit {
                        Lit::Bool(b) => {
                            if *b {
                                1
                            } else {
                                0
                            }
                        }
                        Lit::Int { base, suffix } => {
                            literal::parse_int_literal(self.text(pat_span), *base, *suffix)
                                .ok_or_else(|| LowerError {
                                    what: "unparseable literal pattern".to_string(),
                                    span: pat_span,
                                })? as u128
                        }
                        // A2: a Char literal pattern is its Unicode scalar codepoint (the same
                        // representation Char literals lower to as expressions).
                        Lit::Char => match literal::eval_lit_value(*lit, self.text(pat_span)) {
                            Some(crate::literal::LitValue::Char(c)) => u128::from(u32::from(c)),
                            _ => return unsupported("unparseable char literal pattern", pat_span),
                        },
                        _ => return unsupported("literal pattern form (C4.5)", pat_span),
                    };
                    cases.push((value, pat, body));
                }
                hir::PatKind::Wild | hir::PatKind::Binding { .. } => {
                    if default.is_none() {
                        default = Some((pat, body));
                    }
                }
                _ => return unsupported("pattern form (C4.5)", self.hir.pat(pat).span),
            }
        }
        let Some((default_pat, default_body)) = default else {
            return unsupported("integer match without a default arm (C4.5)", span);
        };

        let case_blocks: Vec<BlockId> = cases.iter().map(|_| self.new_block()).collect();
        let default_block = self.new_block();
        let switch_arms = cases
            .iter()
            .zip(&case_blocks)
            .map(|((value, _, _), block)| (*value, *block))
            .collect();
        self.terminate(
            Terminator::SwitchInt {
                scrut: Operand::Copy(scrut.clone()),
                arms: switch_arms,
                otherwise: default_block,
            },
            self.synthetic(span, SyntheticKind::MatchDesugar),
            default_block,
        );

        // Default arm (binding binds the scrutinee — always Copy for scalar scrutinees).
        // A ByRef place is a deref of a ref-to-scalar; peeling recovers the scalar type.
        if let hir::PatKind::Binding { name, local } = &self.hir.pat(default_pat).kind {
            let ty = Self::peel_refs(self.locals[scrut.local.0 as usize].ty.clone()).0;
            self.locals.push(LocalDecl {
                ty,
                kind: LocalKind::User(self.text(*name).to_string()),
            });
            let bound = LocalId((self.locals.len() - 1) as u32);
            self.local_map.insert(local.0, bound);
            self.emit(
                Statement::Assign(
                    Place::local(bound),
                    Rvalue::Use(Operand::Copy(scrut.clone())),
                ),
                self.synthetic(span, SyntheticKind::MatchDesugar),
            );
        }
        self.lower_arm_into(default_body, &dest, join, span)?;

        for ((_, _, body), block) in cases.iter().zip(&case_blocks) {
            self.current = *block;
            self.lower_arm_into(*body, &dest, join, span)?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::too_many_arguments)]
    fn lower_enum_match(
        &mut self,
        enum_ref: EnumRef,
        scrut_args: Vec<MirTy>,
        scrut: Place,
        mode: MatchMode,
        arms: &[(hir::PatId, ExprId)],
        dest: Option<Place>,
        join: BlockId,
        span: Span,
    ) -> Result<(), LowerError> {
        let disc = self.new_temp(MirTy::Int64);
        self.emit(
            Statement::Assign(Place::local(disc), Rvalue::Discriminant(scrut.clone())),
            self.synthetic(span, SyntheticKind::MatchDesugar),
        );

        struct ArmPlan {
            variant: u128,
            block: BlockId,
            pat: hir::PatId,
            body: ExprId,
        }
        let mut plans: Vec<ArmPlan> = Vec::new();
        let mut default: Option<(hir::PatId, ExprId)> = None;
        for &(pat, body) in arms {
            let pat_span = self.hir.pat(pat).span;
            let variant = match &self.hir.pat(pat).kind {
                hir::PatKind::Wild | hir::PatKind::Binding { .. } => {
                    if default.is_none() {
                        default = Some((pat, body));
                    }
                    continue;
                }
                hir::PatKind::Path { res, .. }
                | hir::PatKind::TupleVariant { res, .. }
                | hir::PatKind::Struct { res, .. } => match res {
                    Res::Variant(_, v) => *v as u128,
                    Res::Builtin(Builtin::None) => 0,
                    Res::Builtin(Builtin::Some) => 1,
                    Res::Builtin(Builtin::Ok) => 0,
                    Res::Builtin(Builtin::Err) => 1,
                    // A2 (CE3): Ordering variants (fieldless), discriminants Less=0/Equal=1/Greater=2.
                    Res::Builtin(Builtin::OrderingLess) => 0,
                    Res::Builtin(Builtin::OrderingEqual) => 1,
                    Res::Builtin(Builtin::OrderingGreater) => 2,
                    _ => return unsupported("enum pattern resolution (C4.5)", pat_span),
                },
                _ => return unsupported("pattern form in enum match (C4.5)", pat_span),
            };
            plans.push(ArmPlan {
                variant,
                block: self.new_block(),
                pat,
                body,
            });
        }
        let otherwise = if default.is_some() {
            self.new_block()
        } else {
            // Exhaustive over variants (verified upstream): route "otherwise" to Unreachable.
            self.new_block()
        };
        let switch_arms = plans.iter().map(|p| (p.variant, p.block)).collect();
        self.terminate(
            Terminator::SwitchInt {
                scrut: Operand::Copy(Place::local(disc)),
                arms: switch_arms,
                otherwise,
            },
            self.synthetic(span, SyntheticKind::MatchDesugar),
            otherwise,
        );

        if let Some((default_pat, default_body)) = default {
            self.scopes.push(Vec::new());
            let depth = self.scopes.len() - 1;
            let scrut_ty = MirTy::Enum(enum_ref, scrut_args.clone());
            if let hir::PatKind::Binding { name, local } = &self.hir.pat(default_pat).kind {
                // Catch-all binding: bind the whole scrutinee. Consuming: move it in and
                // register it to drop at arm end. ByRef: the whole value must be Copy to bind
                // (a non-Copy whole-value binding would move out of the borrow).
                if mode == MatchMode::ByRef && !self.is_copy(&scrut_ty) {
                    return unsupported(
                        "binding a non-Copy scrutinee through a shared reference",
                        span,
                    );
                }
                self.locals.push(LocalDecl {
                    ty: scrut_ty.clone(),
                    kind: LocalKind::User(self.text(*name).to_string()),
                });
                let bound = LocalId((self.locals.len() - 1) as u32);
                self.local_map.insert(local.0, bound);
                let value = self.read_place(scrut.clone(), &scrut_ty, span)?;
                self.emit(
                    Statement::Assign(Place::local(bound), Rvalue::Use(value)),
                    self.synthetic(span, SyntheticKind::MatchDesugar),
                );
                if mode == MatchMode::Consuming {
                    self.register_droppable_local(bound, &scrut_ty, true, span)?;
                }
            } else if mode == MatchMode::Consuming {
                // Wildcard `_` catch-all: the scrutinee is dropped whole at arm end.
                // (ByRef: the referent stays owned by the caller — nothing to drop.)
                self.drop_whole_scrutinee_at_arm_end(scrut.clone(), &scrut_ty, span)?;
            }
            self.lower_arm_body_scoped(default_body, &dest, join, depth, span)?;
        } else {
            let next = self.new_block();
            self.terminate(
                Terminator::Unreachable,
                self.synthetic(span, SyntheticKind::MatchDesugar),
                next,
            );
            self.blocks.pop();
        }

        let plans: Vec<_> = plans
            .into_iter()
            .map(|p| (p.variant, p.block, p.pat, p.body))
            .collect();
        for (variant, block, pat, body) in plans {
            self.current = block;
            self.scopes.push(Vec::new());
            let depth = self.scopes.len() - 1;
            // C4.5d match-drop: consume the active variant's payload — bound fields into
            // registered binding locals, unbound droppable fields into registered temps — so
            // the scrutinee is fully accounted for and everything drops at arm end.
            self.consume_variant_payload(
                enum_ref,
                &scrut_args,
                scrut.clone(),
                mode,
                variant as u32,
                pat,
                span,
            )?;
            self.lower_arm_body_scoped(body, &dest, join, depth, span)?;
        }
        Ok(())
    }

    /// A2-2: is every arm within the FLAT shapes the drop-elaborated `lower_enum_match` path
    /// supports (top-level Wild/Binding, or a variant pattern whose sub-patterns are all
    /// Wild/Binding/shorthand)? Anything else routes to the general engine.
    fn enum_arms_are_flat(&self, arms: &[(hir::PatId, ExprId)]) -> bool {
        arms.iter().all(|&(pat, _)| match &self.hir.pat(pat).kind {
            hir::PatKind::Wild | hir::PatKind::Binding { .. } | hir::PatKind::Path { .. } => true,
            hir::PatKind::TupleVariant { pats, .. } => pats.iter().all(|&p| {
                matches!(
                    self.hir.pat(p).kind,
                    hir::PatKind::Wild | hir::PatKind::Binding { .. }
                )
            }),
            hir::PatKind::Struct { fields, .. } => fields.iter().all(|f| match f.pat {
                None => true,
                Some(p) => matches!(
                    self.hir.pat(p).kind,
                    hir::PatKind::Wild | hir::PatKind::Binding { .. }
                ),
            }),
            _ => false,
        })
    }

    /// A2-2: the GENERAL pattern engine — sequential per-arm test-and-bind, fully recursive
    /// over pattern structure (tuples, arrays, structs, nested variants, Char/Float/String
    /// literals). Restricted to scrutinee types without drop obligations in Consuming mode
    /// (droppable + nested is the recorded residual); ByRef mode enforces Copy-only bindings.
    #[allow(clippy::too_many_arguments)]
    fn lower_general_match(
        &mut self,
        scrut: Place,
        scrut_ty: MirTy,
        mode: MatchMode,
        arms: &[(hir::PatId, ExprId)],
        dest: Option<Place>,
        join: BlockId,
        span: Span,
    ) -> Result<(), LowerError> {
        for &(pat, body) in arms {
            let fail = self.new_block();
            self.emit_pattern_test(pat, &scrut, &scrut_ty, fail, span)?;
            self.scopes.push(Vec::new());
            let depth = self.scopes.len() - 1;
            // WP-C4.7-8.3b: consuming a droppable scrutinee decomposes it completely — whatever
            // the pattern DISCARDS still owes a destructor. The unbound walk runs first so that
            // arm-end drops (reverse registration order) destroy the bindings first and the
            // discarded leaves after, which is the order the oracle produces (DEV-080).
            if mode == MatchMode::Consuming {
                self.consume_unbound_leaves(pat, &scrut, &scrut_ty, span)?;
            }
            self.bind_pattern(pat, &scrut, &scrut_ty, mode, span)?;
            self.lower_arm_body_scoped(body, &dest, join, depth, span)?;
            self.current = fail;
        }
        // Exhaustiveness was verified upstream; a fall-off is unreachable.
        let next = self.new_block();
        self.terminate(
            Terminator::Unreachable,
            self.synthetic(span, SyntheticKind::MatchDesugar),
            next,
        );
        self.blocks.pop();
        Ok(())
    }

    /// Emit the recursive TEST for `pat` against `place`: on mismatch jump to `fail`; on match
    /// fall through in `self.current`. Emits no bindings (the bind phase re-walks the pattern).
    fn emit_pattern_test(
        &mut self,
        pat: hir::PatId,
        place: &Place,
        ty: &MirTy,
        fail: BlockId,
        span: Span,
    ) -> Result<(), LowerError> {
        let pat_span = self.hir.pat(pat).span;
        match &self.hir.pat(pat).kind {
            hir::PatKind::Wild | hir::PatKind::Binding { .. } => Ok(()),
            hir::PatKind::Lit(lit) => {
                let text = self.text(pat_span).to_string();
                match ty {
                    MirTy::Bool
                    | MirTy::Char
                    | MirTy::Int8
                    | MirTy::Int16
                    | MirTy::Int32
                    | MirTy::Int64
                    | MirTy::UInt8
                    | MirTy::UInt16
                    | MirTy::UInt32
                    | MirTy::UInt64 => {
                        let value = match lit {
                            Lit::Bool(b) => i128::from(*b),
                            Lit::Int { base, suffix } => literal::parse_int_literal(
                                &text, *base, *suffix,
                            )
                            .ok_or_else(|| LowerError {
                                what: "unparseable literal pattern".to_string(),
                                span: pat_span,
                            })?,
                            Lit::Char => match literal::eval_lit_value(*lit, &text) {
                                Some(crate::literal::LitValue::Char(c)) => i128::from(u32::from(c)),
                                _ => {
                                    return unsupported(
                                        "unparseable char literal pattern",
                                        pat_span,
                                    )
                                }
                            },
                            _ => return unsupported("literal/type mismatch in pattern", pat_span),
                        };
                        let eq = self.new_temp(MirTy::Bool);
                        self.emit(
                            Statement::Assign(
                                Place::local(eq),
                                Rvalue::BinOp(
                                    MirBinOp::Eq,
                                    Operand::Copy(place.clone()),
                                    Operand::Const(Constant::Int(value, ty.clone())),
                                ),
                            ),
                            self.synthetic(span, SyntheticKind::MatchDesugar),
                        );
                        self.branch_on(eq, fail, span);
                        Ok(())
                    }
                    // A2-2: Float literal patterns — spec-exact IEEE equality, matching the
                    // oracle's structural comparison.
                    MirTy::Float32 | MirTy::Float64 => {
                        let Lit::Float { suffix } = lit else {
                            return unsupported("literal/type mismatch in pattern", pat_span);
                        };
                        let value =
                            literal::parse_float_literal(&text, *suffix).ok_or_else(|| {
                                LowerError {
                                    what: "unparseable float literal pattern".to_string(),
                                    span: pat_span,
                                }
                            })?;
                        let eq = self.new_temp(MirTy::Bool);
                        self.emit(
                            Statement::Assign(
                                Place::local(eq),
                                Rvalue::BinOp(
                                    MirBinOp::Eq,
                                    Operand::Copy(place.clone()),
                                    Operand::Const(Constant::Float(value, ty.clone())),
                                ),
                            ),
                            self.synthetic(span, SyntheticKind::MatchDesugar),
                        );
                        self.branch_on(eq, fail, span);
                        Ok(())
                    }
                    // A2-2: String literal patterns on a `&str` scrutinee — content equality
                    // via `StrEq` (never a structural BinOp, V-STR-2).
                    MirTy::Ref { inner, .. } if matches!(**inner, MirTy::Str) => {
                        let Lit::Str { .. } = lit else {
                            return unsupported("literal/type mismatch in pattern", pat_span);
                        };
                        let value = match literal::eval_lit_value(*lit, &text) {
                            Some(crate::literal::LitValue::Str(s)) => s,
                            _ => {
                                return unsupported("unparseable string literal pattern", pat_span)
                            }
                        };
                        let eq = self.new_temp(MirTy::Bool);
                        self.emit_runtime_call(
                            RuntimeFn::StrEq,
                            vec![
                                Operand::Copy(place.clone()),
                                Operand::Const(Constant::Str(value)),
                            ],
                            Place::local(eq),
                            span,
                        );
                        self.branch_on(eq, fail, span);
                        Ok(())
                    }
                    other => unsupported(
                        format!("literal pattern on scrutinee type {other:?}"),
                        pat_span,
                    ),
                }
            }
            hir::PatKind::Path { res, .. } => {
                let variant = self.variant_of_res(res, pat_span)?;
                self.emit_discriminant_test(place, variant, fail, span);
                Ok(())
            }
            hir::PatKind::TupleVariant { res, pats, .. } => {
                let res = *res;
                let pats = pats.clone();
                let variant = self.variant_of_res(&res, pat_span)?;
                self.emit_discriminant_test(place, variant, fail, span);
                let (enum_ref, args) = match ty {
                    MirTy::Enum(er, args) => (*er, args.clone()),
                    other => {
                        return unsupported(
                            format!("variant pattern on non-enum {other:?}"),
                            pat_span,
                        )
                    }
                };
                let payload_tys = self.variant_payload_types(enum_ref, &args, variant, span)?;
                for (i, &sub) in pats.iter().enumerate() {
                    let field_ty = payload_tys.get(i).cloned().unwrap_or(MirTy::Unit);
                    let mut sub_place = place.clone();
                    sub_place
                        .projection
                        .push(Projection::VariantField(variant, i as u32));
                    self.emit_pattern_test(sub, &sub_place, &field_ty, fail, span)?;
                }
                Ok(())
            }
            hir::PatKind::Struct { res, fields, .. } => {
                let res = *res;
                let fields: Vec<(Span, Option<hir::PatId>, Option<crate::hir::LocalId>)> =
                    fields.iter().map(|f| (f.name, f.pat, f.local)).collect();
                match ty {
                    MirTy::Enum(er, args) => {
                        let (er, args) = (*er, args.clone());
                        let variant = self.variant_of_res(&res, pat_span)?;
                        self.emit_discriminant_test(place, variant, fail, span);
                        let payload_tys = self.variant_payload_types(er, &args, variant, span)?;
                        let order = self.variant_field_order(&res, variant)?;
                        for (name_span, sub, _) in &fields {
                            let Some(sub) = sub else { continue };
                            let name_text = self.text(*name_span).to_string();
                            let Some(index) = order.iter().position(|n| *n == name_text) else {
                                return unsupported("unknown variant field", *name_span);
                            };
                            let field_ty = payload_tys.get(index).cloned().unwrap_or(MirTy::Unit);
                            let mut sub_place = place.clone();
                            sub_place
                                .projection
                                .push(Projection::VariantField(variant, index as u32));
                            self.emit_pattern_test(*sub, &sub_place, &field_ty, fail, span)?;
                        }
                        Ok(())
                    }
                    MirTy::Struct(item, args) => {
                        let (item, args) = (*item, args.clone());
                        let field_tys = match nominal_instance_fields(
                            self.hir,
                            self.tables,
                            self.meta,
                            item,
                            &args,
                            self.providers,
                        )? {
                            NominalFields::Struct(tys) => tys,
                            NominalFields::Enum(_) => {
                                return unsupported("struct pattern on enum item", pat_span)
                            }
                        };
                        for (name_span, sub, _) in &fields {
                            let Some(sub) = sub else { continue };
                            let index = self.struct_field_index(item, *name_span)?;
                            let field_ty = field_tys.get(index).cloned().unwrap_or(MirTy::Unit);
                            let mut sub_place = place.clone();
                            sub_place.projection.push(Projection::Field(index as u32));
                            self.emit_pattern_test(*sub, &sub_place, &field_ty, fail, span)?;
                        }
                        Ok(())
                    }
                    other => unsupported(
                        format!("struct pattern on scrutinee type {other:?}"),
                        pat_span,
                    ),
                }
            }
            hir::PatKind::Tuple(pats) => {
                let pats = pats.clone();
                let elem_tys = match ty {
                    MirTy::Tuple(elems) => elems.clone(),
                    other => {
                        return unsupported(
                            format!("tuple pattern on scrutinee type {other:?}"),
                            pat_span,
                        )
                    }
                };
                for (i, &sub) in pats.iter().enumerate() {
                    let elem_ty = elem_tys.get(i).cloned().unwrap_or(MirTy::Unit);
                    let mut sub_place = place.clone();
                    sub_place.projection.push(Projection::Field(i as u32));
                    self.emit_pattern_test(sub, &sub_place, &elem_ty, fail, span)?;
                }
                Ok(())
            }
            hir::PatKind::Array(pats) => {
                let pats = pats.clone();
                let elem_ty = match ty {
                    MirTy::Array(elem, _) => (**elem).clone(),
                    other => {
                        return unsupported(
                            format!("array pattern on scrutinee type {other:?}"),
                            pat_span,
                        )
                    }
                };
                for (i, &sub) in pats.iter().enumerate() {
                    let sub_place = self.array_elem_place(place, ty, i, span)?;
                    self.emit_pattern_test(sub, &sub_place, &elem_ty, fail, span)?;
                }
                Ok(())
            }
            hir::PatKind::Error => unsupported("error pattern", pat_span),
        }
    }

    /// Emit the recursive BIND for a matched `pat`: bindings read out of the scrutinee
    /// (Copy per `read_place`; ByRef enforces Copy-only). Tests were already emitted.
    /// WP-C4.7-8.3b: move every droppable sub-place of `place` that the pattern does NOT bind
    /// into a registered temp, so it is destroyed at arm end.
    ///
    /// This is `consume_variant_payload`'s flat rule generalized to an arbitrary pattern tree:
    /// a consuming match decomposes the scrutinee completely, and whatever the pattern discards
    /// still owes a destructor. It runs BEFORE the binding walk, because arm-end drops run in
    /// reverse registration order and the oracle destroys bound bindings first, then the
    /// discarded leaves (DEV-080 — established empirically, and the three-field `(a, _, c)` case
    /// distinguishes this rule from plain reverse-field order).
    ///
    /// `Lit`/`Path` patterns bind nothing and cover a leaf that was already tested: a unit
    /// variant carries no payload, and a literal's type is `Copy`, so neither owes a drop.
    fn consume_unbound_leaves(
        &mut self,
        pat: hir::PatId,
        place: &Place,
        ty: &MirTy,
        span: Span,
    ) -> Result<(), LowerError> {
        if !self.ty_needs_drop(ty, span)? {
            return Ok(());
        }
        let pat_span = self.hir.pat(pat).span;
        match &self.hir.pat(pat).kind {
            // A binding takes ownership of this whole sub-place in the bind walk.
            hir::PatKind::Binding { .. } => Ok(()),
            // `_` discards this sub-place wholesale: move it into a temp that drops at arm end.
            hir::PatKind::Wild => {
                self.discover_drop_impls(ty)?;
                let value = self.read_place(place.clone(), ty, span)?;
                let tmp = self.new_temp(ty.clone());
                self.emit(
                    Statement::Assign(Place::local(tmp), Rvalue::Use(value)),
                    self.synthetic(span, SyntheticKind::MatchDesugar),
                );
                self.register_droppable_local(tmp, ty, false, span)?;
                self.set_flags_under(tmp.0, &[], true, span);
                Ok(())
            }
            hir::PatKind::Lit(_) | hir::PatKind::Path { .. } => Ok(()),
            hir::PatKind::TupleVariant { res, pats, .. } => {
                let res = *res;
                let pats = pats.clone();
                let variant = self.variant_of_res(&res, pat_span)?;
                let (enum_ref, args) = match ty {
                    MirTy::Enum(er, args) => (*er, args.clone()),
                    _ => return unsupported("variant pattern on non-enum", pat_span),
                };
                let payload_tys = self.variant_payload_types(enum_ref, &args, variant, span)?;
                for (i, &sub) in pats.iter().enumerate() {
                    let field_ty = payload_tys.get(i).cloned().unwrap_or(MirTy::Unit);
                    let mut sub_place = place.clone();
                    sub_place
                        .projection
                        .push(Projection::VariantField(variant, i as u32));
                    self.consume_unbound_leaves(sub, &sub_place, &field_ty, span)?;
                }
                Ok(())
            }
            hir::PatKind::Tuple(pats) => {
                let pats = pats.clone();
                let elems = match ty {
                    MirTy::Tuple(elems) => elems.clone(),
                    _ => return unsupported("tuple pattern on non-tuple", pat_span),
                };
                for (i, &sub) in pats.iter().enumerate() {
                    let elem_ty = elems.get(i).cloned().unwrap_or(MirTy::Unit);
                    let mut sub_place = place.clone();
                    sub_place.projection.push(Projection::Field(i as u32));
                    self.consume_unbound_leaves(sub, &sub_place, &elem_ty, span)?;
                }
                Ok(())
            }
            hir::PatKind::Struct { res, fields, .. } => {
                let res = *res;
                let fields: Vec<(Span, Option<hir::PatId>, Option<crate::hir::LocalId>)> =
                    fields.iter().map(|f| (f.name, f.pat, f.local)).collect();
                // Field types + the index of each named field, for both the struct-nominal and
                // the struct-shaped-enum-variant forms, mirroring `bind_pattern`'s split.
                let (field_tys, indices, base_proj): (Vec<MirTy>, Vec<usize>, Vec<Projection>) =
                    match ty {
                        MirTy::Enum(er, args) => {
                            let (er, args) = (*er, args.clone());
                            let variant = self.variant_of_res(&res, pat_span)?;
                            let tys = self.variant_payload_types(er, &args, variant, span)?;
                            let order = self.variant_field_order(&res, variant)?;
                            let mut idx = Vec::new();
                            for (name_span, _, _) in &fields {
                                let name_text = self.text(*name_span).to_string();
                                let Some(i) = order.iter().position(|n| *n == name_text) else {
                                    return unsupported("unknown variant field", *name_span);
                                };
                                idx.push(i);
                            }
                            let proj: Vec<Projection> = (0..tys.len())
                                .map(|i| Projection::VariantField(variant, i as u32))
                                .collect();
                            (tys, idx, proj)
                        }
                        MirTy::Struct(item, args) => {
                            let (item, args) = (*item, args.clone());
                            let tys = match nominal_instance_fields(
                                self.hir,
                                self.tables,
                                self.meta,
                                item,
                                &args,
                                self.providers,
                            )? {
                                NominalFields::Struct(tys) => tys,
                                NominalFields::Enum(_) => {
                                    return unsupported("struct pattern on enum item", pat_span)
                                }
                            };
                            let mut idx = Vec::new();
                            for (name_span, _, _) in &fields {
                                idx.push(self.struct_field_index(item, *name_span)?);
                            }
                            let proj: Vec<Projection> = (0..tys.len())
                                .map(|i| Projection::Field(i as u32))
                                .collect();
                            (tys, idx, proj)
                        }
                        _ => return unsupported("struct pattern on scrutinee type", pat_span),
                    };
                let mut mentioned = vec![false; field_tys.len()];
                for ((_, sub, shorthand), index) in fields.iter().zip(indices) {
                    mentioned[index] = true;
                    let field_ty = field_tys.get(index).cloned().unwrap_or(MirTy::Unit);
                    let mut sub_place = place.clone();
                    sub_place.projection.push(base_proj[index].clone());
                    match (sub, shorthand) {
                        // A sub-pattern may itself discard leaves; a shorthand binds, so the
                        // bind walk owns it.
                        (Some(sub), _) => {
                            self.consume_unbound_leaves(*sub, &sub_place, &field_ty, span)?
                        }
                        (None, Some(_)) => {}
                        (None, None) => {}
                    }
                }
                // Fields the pattern never mentions are discarded and still owe a drop.
                for (index, was_mentioned) in mentioned.iter().enumerate() {
                    if *was_mentioned {
                        continue;
                    }
                    let field_ty = field_tys.get(index).cloned().unwrap_or(MirTy::Unit);
                    if !self.ty_needs_drop(&field_ty, span)? {
                        continue;
                    }
                    self.discover_drop_impls(&field_ty)?;
                    let mut sub_place = place.clone();
                    sub_place.projection.push(base_proj[index].clone());
                    let value = self.read_place(sub_place, &field_ty, span)?;
                    let tmp = self.new_temp(field_ty.clone());
                    self.emit(
                        Statement::Assign(Place::local(tmp), Rvalue::Use(value)),
                        self.synthetic(span, SyntheticKind::MatchDesugar),
                    );
                    self.register_droppable_local(tmp, &field_ty, false, span)?;
                    self.set_flags_under(tmp.0, &[], true, span);
                }
                Ok(())
            }
            // Array patterns are normative (`02:291`) and decompose exactly like tuples: each
            // element position is a sub-place, and whatever the pattern does not bind is
            // discarded and owes a destructor. `array_elem_place` mints the index proof an
            // element place needs — the same helper the binding walk uses.
            hir::PatKind::Array(pats) => {
                let pats = pats.clone();
                let elem_ty = match ty {
                    MirTy::Array(elem, _) => (**elem).clone(),
                    _ => return unsupported("array pattern on non-array", pat_span),
                };
                for (i, &sub) in pats.iter().enumerate() {
                    let sub_place = self.array_elem_place(place, ty, i, span)?;
                    self.consume_unbound_leaves(sub, &sub_place, &elem_ty, span)?;
                }
                Ok(())
            }
            hir::PatKind::Error => Ok(()),
        }
    }

    fn bind_pattern(
        &mut self,
        pat: hir::PatId,
        place: &Place,
        ty: &MirTy,
        mode: MatchMode,
        span: Span,
    ) -> Result<(), LowerError> {
        let pat_span = self.hir.pat(pat).span;
        match &self.hir.pat(pat).kind {
            hir::PatKind::Wild | hir::PatKind::Lit(_) | hir::PatKind::Path { .. } => Ok(()),
            hir::PatKind::Binding { name, local } => {
                let (name, local) = (self.text(*name).to_string(), *local);
                let bind_by_ref = mode == MatchMode::ByRef && !self.is_copy(ty);
                let local_ty = if bind_by_ref {
                    MirTy::Ref {
                        mutable: false,
                        inner: Box::new(ty.clone()),
                    }
                } else {
                    ty.clone()
                };
                self.locals.push(LocalDecl {
                    ty: local_ty,
                    kind: LocalKind::User(name),
                });
                let bound = LocalId((self.locals.len() - 1) as u32);
                self.local_map.insert(local.0, bound);
                if bind_by_ref {
                    self.emit(
                        Statement::Assign(
                            Place::local(bound),
                            Rvalue::RefOf {
                                mutable: false,
                                place: place.clone(),
                            },
                        ),
                        self.synthetic(span, SyntheticKind::MatchDesugar),
                    );
                } else {
                    let value = self.read_place(place.clone(), ty, span)?;
                    self.emit(
                        Statement::Assign(Place::local(bound), Rvalue::Use(value)),
                        self.synthetic(span, SyntheticKind::MatchDesugar),
                    );
                }
                // WP-C4.7-8.3b: Consuming — the binding owns the moved-in value and drops at
                // arm-scope end, exactly as the flat path's `bind_field_local` does.
                if mode == MatchMode::Consuming {
                    self.register_droppable_local(bound, ty, true, span)?;
                }
                Ok(())
            }
            hir::PatKind::TupleVariant { res, pats, .. } => {
                let res = *res;
                let pats = pats.clone();
                let variant = self.variant_of_res(&res, pat_span)?;
                let (enum_ref, args) = match ty {
                    MirTy::Enum(er, args) => (*er, args.clone()),
                    _ => return unsupported("variant pattern on non-enum", pat_span),
                };
                let payload_tys = self.variant_payload_types(enum_ref, &args, variant, span)?;
                for (i, &sub) in pats.iter().enumerate() {
                    let field_ty = payload_tys.get(i).cloned().unwrap_or(MirTy::Unit);
                    let mut sub_place = place.clone();
                    sub_place
                        .projection
                        .push(Projection::VariantField(variant, i as u32));
                    self.bind_pattern(sub, &sub_place, &field_ty, mode, span)?;
                }
                Ok(())
            }
            hir::PatKind::Struct { res, fields, .. } => {
                let res = *res;
                let fields: Vec<(Span, Option<hir::PatId>, Option<crate::hir::LocalId>)> =
                    fields.iter().map(|f| (f.name, f.pat, f.local)).collect();
                match ty {
                    MirTy::Enum(er, args) => {
                        let (er, args) = (*er, args.clone());
                        let variant = self.variant_of_res(&res, pat_span)?;
                        let payload_tys = self.variant_payload_types(er, &args, variant, span)?;
                        let order = self.variant_field_order(&res, variant)?;
                        for (name_span, sub, shorthand) in &fields {
                            let name_text = self.text(*name_span).to_string();
                            let Some(index) = order.iter().position(|n| *n == name_text) else {
                                return unsupported("unknown variant field", *name_span);
                            };
                            let field_ty = payload_tys.get(index).cloned().unwrap_or(MirTy::Unit);
                            let mut sub_place = place.clone();
                            sub_place
                                .projection
                                .push(Projection::VariantField(variant, index as u32));
                            match (sub, shorthand) {
                                (Some(sub), _) => {
                                    self.bind_pattern(*sub, &sub_place, &field_ty, mode, span)?
                                }
                                (None, Some(local)) => self.bind_shorthand(
                                    name_text, *local, &sub_place, &field_ty, mode, span,
                                )?,
                                (None, None) => {}
                            }
                        }
                        Ok(())
                    }
                    MirTy::Struct(item, args) => {
                        let (item, args) = (*item, args.clone());
                        let field_tys = match nominal_instance_fields(
                            self.hir,
                            self.tables,
                            self.meta,
                            item,
                            &args,
                            self.providers,
                        )? {
                            NominalFields::Struct(tys) => tys,
                            NominalFields::Enum(_) => {
                                return unsupported("struct pattern on enum item", pat_span)
                            }
                        };
                        for (name_span, sub, shorthand) in &fields {
                            let index = self.struct_field_index(item, *name_span)?;
                            let field_ty = field_tys.get(index).cloned().unwrap_or(MirTy::Unit);
                            let mut sub_place = place.clone();
                            sub_place.projection.push(Projection::Field(index as u32));
                            match (sub, shorthand) {
                                (Some(sub), _) => {
                                    self.bind_pattern(*sub, &sub_place, &field_ty, mode, span)?
                                }
                                (None, Some(local)) => {
                                    let name = self.text(*name_span).to_string();
                                    self.bind_shorthand(
                                        name, *local, &sub_place, &field_ty, mode, span,
                                    )?
                                }
                                (None, None) => {}
                            }
                        }
                        Ok(())
                    }
                    _ => unsupported("struct pattern on scrutinee type", pat_span),
                }
            }
            hir::PatKind::Tuple(pats) => {
                let pats = pats.clone();
                let elem_tys = match ty {
                    MirTy::Tuple(elems) => elems.clone(),
                    _ => return unsupported("tuple pattern on non-tuple", pat_span),
                };
                for (i, &sub) in pats.iter().enumerate() {
                    let elem_ty = elem_tys.get(i).cloned().unwrap_or(MirTy::Unit);
                    let mut sub_place = place.clone();
                    sub_place.projection.push(Projection::Field(i as u32));
                    self.bind_pattern(sub, &sub_place, &elem_ty, mode, span)?;
                }
                Ok(())
            }
            hir::PatKind::Array(pats) => {
                let pats = pats.clone();
                let elem_ty = match ty {
                    MirTy::Array(elem, _) => (**elem).clone(),
                    _ => return unsupported("array pattern on non-array", pat_span),
                };
                for (i, &sub) in pats.iter().enumerate() {
                    let sub_place = self.array_elem_place(place, ty, i, span)?;
                    self.bind_pattern(sub, &sub_place, &elem_ty, mode, span)?;
                }
                Ok(())
            }
            hir::PatKind::Error => unsupported("error pattern", pat_span),
        }
    }

    /// A shorthand struct-field binding (`Point { x }`): bind `x` to the field's value.
    fn bind_shorthand(
        &mut self,
        name: String,
        hir_local: crate::hir::LocalId,
        place: &Place,
        ty: &MirTy,
        mode: MatchMode,
        span: Span,
    ) -> Result<(), LowerError> {
        // CE1: a non-`Copy` field matched through a shared reference binds BY REFERENCE, exactly as
        // the named form does in `bind_field_local`. This used to refuse instead, and the
        // asymmetry was invisible while the front end refused the same programs first: once
        // typecheck started binding them by reference, `Wrap { h: h }` lowered and `Wrap { h }`
        // did not — the same program, written two ways, with only one of them compiling.
        let bind_by_ref = mode == MatchMode::ByRef && !self.is_copy(ty);
        let local_ty = if bind_by_ref {
            MirTy::Ref {
                mutable: false,
                inner: Box::new(ty.clone()),
            }
        } else {
            ty.clone()
        };
        self.locals.push(LocalDecl {
            ty: local_ty,
            kind: LocalKind::User(name),
        });
        let bound = LocalId((self.locals.len() - 1) as u32);
        self.local_map.insert(hir_local.0, bound);
        if bind_by_ref {
            self.emit(
                Statement::Assign(
                    Place::local(bound),
                    Rvalue::RefOf {
                        mutable: false,
                        place: place.clone(),
                    },
                ),
                self.synthetic(span, SyntheticKind::MatchDesugar),
            );
        } else {
            let value = self.read_place(place.clone(), ty, span)?;
            self.emit(
                Statement::Assign(Place::local(bound), Rvalue::Use(value)),
                self.synthetic(span, SyntheticKind::MatchDesugar),
            );
        }
        // DEV-081 (WP-C4.7-8.3b): a shorthand binding owns the moved-in value exactly as a named
        // one does, and must drop at arm-scope end. This registration was missing entirely, so
        // `match p { P { a, b } => … }` over droppable fields moved them out and then destroyed
        // NEITHER — a leak, not a double drop, which is why nothing failed loudly.
        if mode == MatchMode::Consuming {
            self.register_droppable_local(bound, ty, true, span)?;
        }
        Ok(())
    }

    /// The resolved variant index of an enum pattern's path resolution.
    fn variant_of_res(&self, res: &Res, pat_span: Span) -> Result<u32, LowerError> {
        Ok(match res {
            Res::Variant(_, v) => *v,
            Res::Builtin(Builtin::None) => 0,
            Res::Builtin(Builtin::Some) => 1,
            Res::Builtin(Builtin::Ok) => 0,
            Res::Builtin(Builtin::Err) => 1,
            Res::Builtin(Builtin::OrderingLess) => 0,
            Res::Builtin(Builtin::OrderingEqual) => 1,
            Res::Builtin(Builtin::OrderingGreater) => 2,
            _ => return unsupported("enum pattern resolution (C4.5)", pat_span),
        })
    }

    /// Emit `if discriminant(place) != variant goto fail`.
    fn emit_discriminant_test(&mut self, place: &Place, variant: u32, fail: BlockId, span: Span) {
        let disc = self.new_temp(MirTy::Int64);
        self.emit(
            Statement::Assign(Place::local(disc), Rvalue::Discriminant(place.clone())),
            self.synthetic(span, SyntheticKind::MatchDesugar),
        );
        let eq = self.new_temp(MirTy::Bool);
        self.emit(
            Statement::Assign(
                Place::local(eq),
                Rvalue::BinOp(
                    MirBinOp::Eq,
                    Operand::Copy(Place::local(disc)),
                    Operand::Const(Constant::Int(i128::from(variant), MirTy::Int64)),
                ),
            ),
            self.synthetic(span, SyntheticKind::MatchDesugar),
        );
        self.branch_on(eq, fail, span);
    }

    /// Branch: `eq == 1` falls through to a fresh pass block; otherwise jump to `fail`.
    fn branch_on(&mut self, eq: LocalId, fail: BlockId, span: Span) {
        let pass = self.new_block();
        self.terminate(
            Terminator::SwitchInt {
                scrut: Operand::Copy(Place::local(eq)),
                arms: vec![(1, pass)],
                otherwise: fail,
            },
            self.synthetic(span, SyntheticKind::MatchDesugar),
            pass,
        );
    }

    /// An array element place at a CONSTANT index: `CheckIndex` mints the proof (statically
    /// in-bounds — the checker verified the pattern length against the array length).
    /// A5 (CD-038): an array element at a STATICALLY KNOWN index — a pattern position or a
    /// desugared iteration step. Emits `Projection::ConstIndex`, which the verifier bounds-checks
    /// against the array's compile-time length, so no `CheckIndex` terminator and no
    /// `IndexProof` local are needed.
    ///
    /// This is what makes consuming array patterns work: a proof-backed `Index` forces move
    /// analysis to treat the whole array as one unit (a dynamic proof names no
    /// statically-known sub-place), so moving one element out poisoned every other element.
    /// Dynamic source indexing — `a[i]` for a runtime `i` — still uses `CheckIndex` + `Index`.
    fn array_elem_place(
        &mut self,
        base: &Place,
        array_ty: &MirTy,
        index: usize,
        span: Span,
    ) -> Result<Place, LowerError> {
        let _ = (array_ty, span);
        let mut place = base.clone();
        place.projection.push(Projection::ConstIndex(index as u64));
        Ok(place)
    }

    /// The declared index of a struct field by name.
    fn struct_field_index(&self, item: ItemId, name_span: Span) -> Result<usize, LowerError> {
        let ItemKind::Struct { fields, .. } = &self.hir.item(item).kind else {
            return unsupported("field pattern on non-struct item", name_span);
        };
        let name_text = self.text(name_span);
        fields
            .iter()
            .position(|f| self.meta.item_text(item, f.name) == name_text)
            .ok_or_else(|| LowerError {
                what: "unknown field".to_string(),
                span: name_span,
            })
    }

    /// The payload field types of one variant of a matched enum instance.
    fn variant_payload_types(
        &self,
        enum_ref: EnumRef,
        scrut_args: &[MirTy],
        variant: u32,
        span: Span,
    ) -> Result<Vec<MirTy>, LowerError> {
        match enum_ref {
            EnumRef::CoreOption => Ok(if variant == 1 {
                vec![scrut_args.first().cloned().unwrap_or(MirTy::Unit)]
            } else {
                Vec::new()
            }),
            EnumRef::CoreResult => Ok(vec![scrut_args
                .get(variant as usize)
                .cloned()
                .unwrap_or(MirTy::Unit)]),
            // A2 (CE3): Ordering's three variants are all fieldless.
            EnumRef::CoreOrdering => Ok(Vec::new()),
            EnumRef::User(item) => {
                match nominal_instance_fields(
                    self.hir,
                    self.tables,
                    self.meta,
                    item,
                    scrut_args,
                    self.providers,
                )? {
                    NominalFields::Enum(variants) => {
                        Ok(variants.get(variant as usize).cloned().unwrap_or_default())
                    }
                    NominalFields::Struct(_) => {
                        unsupported("enum instance resolved to struct fields", span)
                    }
                }
            }
        }
    }

    /// Consume the active variant's payload of a match arm (C4.5d match-drop). Every payload
    /// field is moved out of the scrutinee: a bound field into a registered binding local; an
    /// unbound (Wild / unmentioned) droppable field into a registered temp so it drops at arm
    /// end; an unbound non-droppable field is simply abandoned in the (never-dropped) scrutinee
    /// temp.
    #[allow(clippy::too_many_arguments)]
    fn consume_variant_payload(
        &mut self,
        enum_ref: EnumRef,
        scrut_args: &[MirTy],
        scrut: Place,
        mode: MatchMode,
        variant: u32,
        pat: hir::PatId,
        span: Span,
    ) -> Result<(), LowerError> {
        let payload_tys = self.variant_payload_types(enum_ref, scrut_args, variant, span)?;

        // WP-C6.1c (owner ruling, refined Option A). A MULTI-FIELD variant payload containing a
        // non-`Copy` field cannot be moved out field-by-field: an enum payload has no raw
        // projection, so the second `VariantField` move hits a partial slot (the CD-070 boundary).
        // Decompose such a payload into ONE canonical tuple aggregate first, then move fields as
        // ordinary tuple `Field` projections (raw-projectable — C6.1b). Single-field and all-`Copy`
        // payloads keep the direct `VariantField` path: they already work, and decomposing them
        // would churn every `Option`/`Result` match for no gain.
        let use_tuple = mode == MatchMode::Consuming
            && payload_tys.len() > 1
            && payload_tys.iter().any(|t| !self.is_copy(t));
        // A12 / `DEFECT-C788-LOOP-TEMP`. A consuming match empties the scrutinee temp, and how it
        // empties it decides what the temp's storage looks like afterwards:
        //
        // - a non-`Copy` payload field is MOVED out, leaving the storage partially moved. Nothing
        //   downstream ever finished it, so the temp stayed live forever — invisible in
        //   straight-line code, fatal on the second iteration of a loop.
        // - a payload that is empty or entirely `Copy` is not moved out at all, so the storage
        //   still holds the whole value, and the same reassignment fails for the opposite reason.
        //
        // Both are handled below, after consumption. The two catch-all arms already reset the temp
        // — a binding moves the whole value out, and `drop_whole_scrutinee_at_arm_end` reads it
        // whole — which is why only the variant-payload path was affected.
        let ends_storage = mode == MatchMode::Consuming && scrut.projection.is_empty();
        let payload_was_moved = payload_tys.iter().any(|t| !self.is_copy(t));
        let source = if use_tuple {
            self.materialize_consumed_variant_payload(scrut.clone(), variant, &payload_tys, span)?
        } else {
            scrut.clone()
        };
        // The already-projected place of payload field `i`: a tuple `Field` on the decomposition,
        // else a `VariantField` on the scrutinee. Registration ORDER (below) is unchanged, so
        // arm-end Drop order is preserved (DEV-080).
        let field_place = |i: u32| -> Place {
            let mut p = source.clone();
            if use_tuple {
                p.projection.push(Projection::Field(i));
            } else {
                p.projection.push(Projection::VariantField(variant, i));
            }
            p
        };

        match &self.hir.pat(pat).kind {
            hir::PatKind::TupleVariant { pats, .. } => {
                let pats = pats.clone();
                // WP-C4.7-8.3 (DEV-080): UNBOUND fields are consumed FIRST, bound ones second.
                // Arm-end drops run in reverse registration order, so registering the unbound
                // (wildcard / unmentioned) leaves first makes the bindings drop first — in
                // reverse binding order — and the discarded leaves after them, which is the
                // order the oracle produces. Consuming in a single field-order pass interleaved
                // them and gave plain reverse-FIELD order: for `Two::Pair(a, _)` over
                // `(Tag{1}, Tag{2})` that printed `2, 1` where the oracle prints `1, 2`. The
                // divergence was invisible until the V-MOVE-1 fix (DEV-079) let such programs
                // verify at all.
                let is_binding = |lower: &Self, sub: &hir::PatId| {
                    matches!(lower.hir.pat(*sub).kind, hir::PatKind::Binding { .. })
                };
                for pass_binds in [false, true] {
                    for (i, sub) in pats.iter().enumerate() {
                        if is_binding(self, sub) != pass_binds {
                            continue;
                        }
                        let field_ty = payload_tys.get(i).cloned().unwrap_or(MirTy::Unit);
                        self.consume_field(
                            field_place(i as u32),
                            mode,
                            &field_ty,
                            Some(*sub),
                            span,
                        )?;
                    }
                }
            }
            hir::PatKind::Struct { fields, res, .. } => {
                let res = *res;
                // Collect owned (name-span, sub-pat, shorthand-local) to release the HIR borrow.
                let fields: Vec<(Span, Option<hir::PatId>, Option<crate::hir::LocalId>)> =
                    fields.iter().map(|f| (f.name, f.pat, f.local)).collect();
                let field_order = self.variant_field_order(&res, variant)?;
                let mut mentioned = vec![false; payload_tys.len()];
                for (name_span, field_pat, field_local) in &fields {
                    let name_text = self.text(*name_span).to_string();
                    let Some(index) = field_order.iter().position(|n| *n == name_text) else {
                        return unsupported("unknown variant field", *name_span);
                    };
                    if index < mentioned.len() {
                        mentioned[index] = true;
                    }
                    let field_ty = payload_tys.get(index).cloned().unwrap_or(MirTy::Unit);
                    match (field_pat, field_local) {
                        (Some(sub), _) => self.consume_field(
                            field_place(index as u32),
                            mode,
                            &field_ty,
                            Some(*sub),
                            span,
                        )?,
                        (None, Some(local)) => self.bind_field_local(
                            field_place(index as u32),
                            mode,
                            name_text,
                            *local,
                            &field_ty,
                            span,
                        )?,
                        (None, None) => {}
                    }
                }
                // Unmentioned droppable fields still drop at arm end (Consuming only).
                for (i, ty) in payload_tys.iter().enumerate() {
                    if !mentioned[i] {
                        self.consume_field(field_place(i as u32), mode, ty, None, span)?;
                    }
                }
            }
            // Unit variant (`None`, `E::Empty`) — no payload.
            hir::PatKind::Path { .. } => {}
            _ => {}
        }

        // A12: account for the scrutinee temp's STORAGE, now that its units are accounted for.
        if ends_storage {
            // The C6.1c decomposition temp is emptied field by field, so it is left partially moved
            // exactly as the scrutinee would be — it is a second compiler-generated temporary on
            // the same reassignment path, and missing it kept the multi-field payload case failing
            // after the single-field one was fixed. (`use_tuple` implies at least one non-`Copy`
            // field, so this temp is never `Whole` here.)
            if use_tuple {
                self.emit_storage_dead(source.local.0, StorageEnd::Accounted, span);
            }
            if payload_was_moved {
                // Something non-`Copy` came out, so the storage is partially moved (or already
                // emptied whole, by the C6.1c decomposition). Either way this ends it, and it is
                // idempotent on the already-dead case.
                self.emit_storage_dead(scrut.local.0, StorageEnd::Accounted, span);
            } else {
                // Nothing came out, so the whole value is still in the temp — but this variant's
                // payload is empty or entirely `Copy`, so it owns nothing and the storage can
                // simply end. It must NOT be dropped instead: a whole-value drop runs the enum's
                // glue for EVERY variant, including one holding a host resource this arm never
                // had, which the backend rightly refuses ("must be emitted by the Drop terminator,
                // not by generic drop glue"). That is the `Err` arm of every
                // `Result<Resource, E>`.
                self.emit_storage_dead(scrut.local.0, StorageEnd::OwnsNothing, span);
            }
        }
        Ok(())
    }

    /// WP-C6.1c: decompose the active variant's payload into ONE tuple aggregate (owner ruling,
    /// refined Option A). Reads EVERY payload field in declaration order into a single
    /// `Aggregate(Tuple, [...])` statement and returns the tuple temporary. It does NOT register
    /// that temporary for whole-value Drop — its fields are consumed individually by the caller, so
    /// it never owes (and must never take) a whole-tuple drop over partially-moved storage.
    ///
    /// This is a MIR CANONICALISATION using only existing operations (`Rvalue::Aggregate`,
    /// `Projection::VariantField`): no new MIR variant, no changed operation meaning, no verifier
    /// change. The generated-Rust backend recognises this exact statement-local shape and emits one
    /// destructuring `match` (whole-enum `take()` + tuple rebuild); after it, movement is ordinary
    /// tuple-field partial-move machinery.
    fn materialize_consumed_variant_payload(
        &mut self,
        scrut: Place,
        variant: u32,
        payload_tys: &[MirTy],
        span: Span,
    ) -> Result<Place, LowerError> {
        let mut operands = Vec::with_capacity(payload_tys.len());
        for (i, field_ty) in payload_tys.iter().enumerate() {
            if self.ty_needs_drop(field_ty, span)? {
                self.discover_drop_impls(field_ty)?;
            }
            let mut place = scrut.clone();
            place
                .projection
                .push(Projection::VariantField(variant, i as u32));
            operands.push(self.read_place(place, field_ty, span)?);
        }
        let tuple_local = self.new_temp(MirTy::Tuple(payload_tys.to_vec()));
        self.emit(
            Statement::Assign(
                Place::local(tuple_local),
                Rvalue::Aggregate(AggKind::Tuple, operands),
            ),
            self.synthetic(span, SyntheticKind::MatchDesugar),
        );
        Ok(Place::local(tuple_local))
    }

    /// Consume one variant payload field given its sub-pattern (`None` = unbound/Wild).
    ///
    /// WP-C6.1c: `field_place` is the ALREADY-PROJECTED place of the field — either
    /// `scrut.VariantField(v, i)` (ByRef, and the single-field/all-Copy consuming fast paths) or a
    /// `tuple.Field(i)` on the decomposed payload tuple (multi-field consuming). The caller chooses;
    /// this reads from it uniformly.
    fn consume_field(
        &mut self,
        field_place: Place,
        mode: MatchMode,
        field_ty: &MirTy,
        sub: Option<hir::PatId>,
        span: Span,
    ) -> Result<(), LowerError> {
        match sub.map(|s| &self.hir.pat(s).kind) {
            Some(hir::PatKind::Binding { name, local }) => {
                let (name, local) = (self.text(*name).to_string(), *local);
                self.bind_field_local(field_place, mode, name, local, field_ty, span)
            }
            Some(hir::PatKind::Wild) | None => {
                // ByRef: nothing is consumed — the referent keeps ownership of every payload.
                if mode == MatchMode::Consuming && self.ty_needs_drop(field_ty, span)? {
                    self.discover_drop_impls(field_ty)?;
                    let value = self.read_place(field_place, field_ty, span)?;
                    let tmp = self.new_temp(field_ty.clone());
                    self.emit(
                        Statement::Assign(Place::local(tmp), Rvalue::Use(value)),
                        self.synthetic(span, SyntheticKind::MatchDesugar),
                    );
                    self.register_droppable_local(tmp, field_ty, false, span)?;
                    self.set_flags_under(tmp.0, &[], true, span);
                }
                Ok(())
            }
            Some(_) => unsupported("nested pattern in match arm (C4.5)", span),
        }
    }

    /// Bind a variant payload field to a fresh binding local. Consuming: move it in and
    /// register it to drop at arm end. ByRef: Copy fields are read by copy; non-Copy fields bind
    /// as shared references to the payload.
    fn bind_field_local(
        &mut self,
        field_place: Place,
        mode: MatchMode,
        name: String,
        hir_local: crate::hir::LocalId,
        field_ty: &MirTy,
        span: Span,
    ) -> Result<(), LowerError> {
        let bind_by_ref = mode == MatchMode::ByRef && !self.is_copy(field_ty);
        let local_ty = if bind_by_ref {
            MirTy::Ref {
                mutable: false,
                inner: Box::new(field_ty.clone()),
            }
        } else {
            field_ty.clone()
        };
        self.locals.push(LocalDecl {
            ty: local_ty,
            kind: LocalKind::User(name),
        });
        let bound = LocalId((self.locals.len() - 1) as u32);
        self.local_map.insert(hir_local.0, bound);
        if bind_by_ref {
            self.emit(
                Statement::Assign(
                    Place::local(bound),
                    Rvalue::RefOf {
                        mutable: false,
                        place: field_place,
                    },
                ),
                self.synthetic(span, SyntheticKind::MatchDesugar),
            );
        } else {
            let value = self.read_place(field_place, field_ty, span)?;
            self.emit(
                Statement::Assign(Place::local(bound), Rvalue::Use(value)),
                self.synthetic(span, SyntheticKind::MatchDesugar),
            );
        }
        // Consuming: the binding owns the moved-in value (flag true), drops at arm-scope end.
        if mode == MatchMode::Consuming {
            self.register_droppable_local(bound, field_ty, true, span)?;
        }
        Ok(())
    }

    /// Wildcard `_` catch-all: drop the whole scrutinee at arm end (move it into a registered
    /// temp). No-op if the scrutinee isn't droppable.
    fn drop_whole_scrutinee_at_arm_end(
        &mut self,
        scrut: Place,
        scrut_ty: &MirTy,
        span: Span,
    ) -> Result<(), LowerError> {
        if !self.ty_needs_drop(scrut_ty, span)? {
            return Ok(());
        }
        self.discover_drop_impls(scrut_ty)?;
        let value = self.read_place(scrut, scrut_ty, span)?;
        let tmp = self.new_temp(scrut_ty.clone());
        self.emit(
            Statement::Assign(Place::local(tmp), Rvalue::Use(value)),
            self.synthetic(span, SyntheticKind::MatchDesugar),
        );
        self.register_droppable_local(tmp, scrut_ty, false, span)?;
        self.set_flags_under(tmp.0, &[], true, span);
        Ok(())
    }

    /// Lower a match arm body inside its drop scope: compute the arm value into `dest`, drop the
    /// arm scope (bindings + unbound-payload temps), then jump to `join`.
    fn lower_arm_body_scoped(
        &mut self,
        body: ExprId,
        dest: &Option<Place>,
        join: BlockId,
        depth: usize,
        span: Span,
    ) -> Result<(), LowerError> {
        match dest {
            Some(place) => {
                let value = self.lower_expr_to_operand(body)?;
                self.emit(
                    Statement::Assign(place.clone(), Rvalue::Use(value)),
                    self.info(span),
                );
            }
            None => {
                self.lower_expr_operand_or_unit(body)?;
            }
        }
        self.emit_scope_drops_from(depth, span);
        self.scopes.pop();
        let dead = self.new_block();
        self.terminate(Terminator::Goto { target: join }, self.info(span), dead);
        self.blocks.pop();
        Ok(())
    }

    fn variant_field_order(&self, res: &Res, variant: u32) -> Result<Vec<String>, LowerError> {
        match res {
            Res::Variant(item, _) => {
                let ItemKind::Enum { variants, .. } = &self.hir.item(*item).kind else {
                    return Ok(Vec::new());
                };
                let v = &variants[variant as usize];
                Ok(match &v.kind {
                    hir::VariantKind::Struct(fields) => fields
                        .iter()
                        .map(|f| self.meta.item_text(*item, f.name).to_string())
                        .collect(),
                    _ => Vec::new(),
                })
            }
            _ => Ok(Vec::new()),
        }
    }

    fn lower_arm_into(
        &mut self,
        body: ExprId,
        dest: &Option<Place>,
        join: BlockId,
        span: Span,
    ) -> Result<(), LowerError> {
        match dest {
            Some(place) => {
                let value = self.lower_expr_to_operand(body)?;
                self.emit(
                    Statement::Assign(place.clone(), Rvalue::Use(value)),
                    self.info(span),
                );
            }
            None => {
                self.lower_expr_operand_or_unit(body)?;
            }
        }
        let dead = self.new_block();
        self.terminate(Terminator::Goto { target: join }, self.info(span), dead);
        self.blocks.pop();
        Ok(())
    }
}

#[derive(Clone, Copy)]
enum PrintKind {
    Int,
    UInt,
    Bool,
    Char,
    Float,
}

/// DEV-DISPLAY-DISPATCH: which lowering a `Display::fmt` receiver takes when the implementation
/// is the standard library's rather than a user `impl` block.
///
/// The match in [`FmtReceiver::of`] is deliberately **total over `MirTy`**: a new MIR type is a
/// question about whether it is `Display`, and the answer belongs here rather than in a `_` arm
/// that silently says "no".
#[derive(Clone, Copy)]
enum FmtReceiver {
    /// The integer widths, `Bool`, `Char` and `Float64` — everything `widen_for_print` accepts.
    Scalar,
    /// `Float32` renders at its DECLARED width (DEV-105), so it must not be widened first.
    Float32,
    /// `String`. Cloned rather than moved: `fmt(&self)` must leave the receiver usable.
    StringOwned,
    /// `str`, which is only ever reached behind a reference.
    StrSlice,
    /// `Unit` renders as `()`.
    Unit,
}

impl FmtReceiver {
    fn of(ty: &MirTy) -> Option<FmtReceiver> {
        match ty {
            MirTy::Int8
            | MirTy::Int16
            | MirTy::Int32
            | MirTy::Int64
            | MirTy::UInt8
            | MirTy::UInt16
            | MirTy::UInt32
            | MirTy::UInt64
            | MirTy::Bool
            | MirTy::Char
            | MirTy::Float64 => Some(FmtReceiver::Scalar),
            MirTy::Float32 => Some(FmtReceiver::Float32),
            MirTy::String => Some(FmtReceiver::StringOwned),
            MirTy::Str => Some(FmtReceiver::StrSlice),
            MirTy::Unit => Some(FmtReceiver::Unit),
            // Not standard-library `Display` receivers. A nominal is handled by the ordinary
            // impl-resolution path in `lower_method_call`, which is where a user
            // `impl Display for Point` is found; the rest have no `Display` at all in Core v1
            // (`Ordering`/`IOError` render only through `print`/`println`, whose composite
            // renderer walks them structurally — they have no `fmt` returning a `String`).
            MirTy::Never
            | MirTy::Struct(_, _)
            | MirTy::Enum(_, _)
            | MirTy::Tuple(_)
            | MirTy::Array(_, _)
            | MirTy::Slice(_)
            | MirTy::Ref { .. }
            | MirTy::FnPtr { .. }
            | MirTy::Core(_, _)
            | MirTy::HostResource(_) => None,
        }
    }
}

/// 0.1-A13 (WP-C7.9 Packet D): which stream an output operation writes to.
///
/// PROC-STREAM-001 gives a program two independent streams; before Packet D the compiler could
/// only lower one of them, so `eprint`/`eprintln` were accepted by the front end and executable by
/// no engine below it.
#[derive(Clone, Copy, PartialEq, Eq)]
enum OutChannel {
    Stdout,
    Stderr,
}

fn expr_kind_name(kind: &hir::ExprKind) -> &'static str {
    match kind {
        hir::ExprKind::Lit(_) => "Lit",
        hir::ExprKind::Path { .. } => "Path",
        hir::ExprKind::Unary { .. } => "Unary",
        hir::ExprKind::Binary { .. } => "Binary",
        hir::ExprKind::Call { .. } => "Call",
        hir::ExprKind::Field { .. } => "Field",
        hir::ExprKind::TupleField { .. } => "TupleField",
        hir::ExprKind::Index { .. } => "Index",
        hir::ExprKind::Tuple(_) => "Tuple",
        hir::ExprKind::Array(_) => "Array",
        hir::ExprKind::StructLit { .. } => "StructLit",
        hir::ExprKind::If { .. } => "If",
        hir::ExprKind::Match { .. } => "Match",
        hir::ExprKind::Loop { .. } => "Loop",
        hir::ExprKind::While { .. } => "While",
        hir::ExprKind::For { .. } => "For",
        hir::ExprKind::Block(_) => "Block",
        hir::ExprKind::Assign { .. } => "Assign",
        hir::ExprKind::Range { .. } => "Range",
        hir::ExprKind::Try(_) => "Try",
        hir::ExprKind::Cast { .. } => "Cast",
        hir::ExprKind::Repeat { .. } => "Repeat",
        hir::ExprKind::Error => "Error",
    }
}
