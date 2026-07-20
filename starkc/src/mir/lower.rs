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

/// Does `ty` mention a user struct/enum anywhere? Comparisons on such types dispatch through
/// the user's `Eq`/`Ord` impl (C4.5e); MIR's structural `BinOp` must not be emitted for them.
fn ty_mentions_user_nominal(ty: &MirTy) -> bool {
    match ty {
        MirTy::Struct(..) | MirTy::Enum(EnumRef::User(_), _) => true,
        MirTy::Enum(_, args) | MirTy::Core(_, args) => args.iter().any(ty_mentions_user_nominal),
        MirTy::Tuple(elems) => elems.iter().any(ty_mentions_user_nominal),
        MirTy::Array(elem, _) | MirTy::Slice(elem) => ty_mentions_user_nominal(elem),
        MirTy::Ref { inner, .. } => ty_mentions_user_nominal(inner),
        // FnPtr comparison is rejected upstream by the checker (TYPE-FN-001).
        _ => false,
    }
}

fn unsupported<T>(what: impl Into<String>, span: Span) -> Result<T, LowerError> {
    Err(LowerError {
        what: what.into(),
        span,
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
                let mut child_path = path;
                child_path.push(mod_name);
                for &child in children.iter().rev() {
                    stack.push((child, child_path.clone()));
                }
            }
        }
        Ok(ProgramMeta {
            files,
            items,
            all_items,
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

    let mut program = MirProgram {
        files: meta.files.clone(),
        bodies: Vec::new(),
        types: TypeContext::default(),
        mir_version: MIR_VERSION.to_string(),
        runtime_surface: MIR_RUNTIME_SURFACE.to_string(),
    };

    // Populate the nominal type context (struct fields, user-enum variant payloads) for every
    // non-generic nominal — module-nested ones included (f-3c) — so the verifier/backends can
    // resolve projections. Each nominal gets a probe keyed to itself so field-type spans read
    // in the nominal's own file.
    for &item_id in &meta.all_items {
        let probe = FnLowerer::new(hir, tables, &meta, FnKey::Top(item_id, Vec::new()));
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
        let mut lowerer = FnLowerer::new(hir, tables, &meta, key.clone());
        let body = lowerer.lower_body()?;
        // C4.5d: dtor symbols this body's drop glue dispatches through.
        program
            .types
            .drop_impls
            .append(&mut lowerer.drop_impl_symbols);
        for callee in lowerer.discovered_callees {
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
    register_reachable_nominal_instances(hir, tables, &meta, &mut program)?;
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
) -> Result<NominalFields, LowerError> {
    let span0 = Span { lo: 0, hi: 0 };
    // The probe is keyed to the nominal itself, so field-type spans read in ITS file (f-3c).
    let mut probe = FnLowerer::new(hir, tables, meta, FnKey::Top(item, Vec::new()));
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
) -> Result<(), LowerError> {
    use std::collections::BTreeSet;
    let mut visit: Vec<MirTy> = Vec::new();
    for body in &program.bodies {
        for decl in &body.locals {
            visit.push(decl.ty.clone());
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
                match nominal_instance_fields(hir, tables, meta, item, &args)? {
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
    },
    /// An un-overridden trait default method, monomorphised for one implementing nominal
    /// instantiation (A1: `self_args` are the nominal's concrete type arguments).
    TraitDefault {
        trait_item: ItemId,
        member: u32,
        self_item: ItemId,
        self_args: Vec<MirTy>,
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
                .map(super::dump_ty)
                .collect::<Vec<_>>()
                .join(", ");
            Ok(format!("{}{name}@[{args_text}]", meta.symbol_prefix(*item)))
        }
        FnKey::ImplFn {
            impl_item,
            member,
            type_args,
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
                .map(super::dump_ty)
                .collect::<Vec<_>>()
                .join(", ");
            match trait_ {
                None => Ok(format!("{prefix}{type_name}::{method}@[{args_text}]")),
                Some(trait_ref) => {
                    let trait_name = match trait_ref.res {
                        Res::Item(t) => item_name_text(hir, meta, t).unwrap_or("?"),
                        // C4.5d: compiler-known trait impls (`impl Drop for T`) render their
                        // source-level trait name — symbols stay injective and readable.
                        Res::CoreTrait(_) => meta.item_text(*impl_item, trait_ref.path.span),
                        _ => "?",
                    };
                    Ok(format!(
                        "{prefix}{type_name}::{trait_name}::{method}@[{args_text}]"
                    ))
                }
            }
        }
        FnKey::TraitDefault {
            trait_item,
            member,
            self_item,
            self_args,
        } => {
            let trait_name = item_name_text(hir, meta, *trait_item).unwrap_or("?");
            let type_name = item_name_text(hir, meta, *self_item).unwrap_or("?");
            let ItemKind::Trait { items, .. } = &hir.item(*trait_item).kind else {
                return unsupported("FnKey::TraitDefault on non-trait", span0);
            };
            let hir::TraitItem::Method { sig, .. } = &items[*member as usize] else {
                return unsupported("FnKey::TraitDefault member is not a method", span0);
            };
            let method = meta.item_text(*trait_item, sig.name);
            let prefix = meta.symbol_prefix(*self_item);
            if self_args.is_empty() {
                Ok(format!("{trait_name}::{method}@[{prefix}{type_name}]"))
            } else {
                let args_text = self_args
                    .iter()
                    .map(super::dump_ty)
                    .collect::<Vec<_>>()
                    .join(", ");
                Ok(format!(
                    "{trait_name}::{method}@[{prefix}{type_name}<{args_text}>]"
                ))
            }
        }
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
#[derive(Clone)]
struct DropUnit {
    path: Vec<u32>,
    ty: MirTy,
    flag: LocalId,
}

struct FnLowerer<'a> {
    hir: &'a Hir,
    tables: &'a TypeTables,
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
    discovered_callees: Vec<FnKey>,
    /// C4.5d: drop units per droppable user/param local, keyed by MIR local index.
    drop_info: HashMap<u32, Vec<DropUnit>>,
    /// C4.5d: lexical scope stack; each entry lists that scope's droppable locals in
    /// declaration order (drops emit in reverse at scope exit).
    scopes: Vec<Vec<LocalId>>,
    /// C4.5d: `(item, args) → dtor instance symbol` for every `Drop` impl this body's glue
    /// can reach; merged into `TypeContext::drop_impls` by `lower_program`.
    drop_impl_symbols: BTreeMap<(u32, Vec<MirTy>), String>,
}

impl<'a> FnLowerer<'a> {
    fn new(hir: &'a Hir, tables: &'a TypeTables, meta: &'a ProgramMeta, key: FnKey) -> Self {
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
            discovered_callees: Vec::new(),
            drop_info: HashMap::new(),
            scopes: Vec::new(),
            drop_impl_symbols: BTreeMap::new(),
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
            Ty::Core(crate::hir::CoreType::KeysIter, args) => {
                let inner = args
                    .iter()
                    .map(|a| self.mir_ty(a, span))
                    .collect::<Result<Vec<_>, _>>()?;
                MirTy::Core(crate::hir::CoreType::KeysIter, inner)
            }
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
                None => {
                    return unsupported(format!("unbound generic parameter {name} (C4.5)"), span)
                }
            },
            _ => return unsupported(format!("type {ty:?} (C4.5)"), span),
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
                        ItemKind::Enum { .. } => {
                            Ok(MirTy::Enum(EnumRef::User(*item), converted_args))
                        }
                        _ => unsupported("field type form (C4.5)", span),
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
                        crate::hir::CoreType::Vec
                        | crate::hir::CoreType::HashMap
                        | crate::hir::CoreType::VecIter
                        | crate::hir::CoreType::KeysIter
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

    /// Copy-vs-move for reads (contract §5): primitives, fn values, and shared refs are Copy;
    /// tuples/arrays/Option/Result of Copy are Copy; user structs/enums are Move (an explicit
    /// `impl Copy` is not visible to lowering in the scalar core — conservative, and harmless
    /// here because no scalar-core type requires drop).
    fn is_copy(&self, ty: &MirTy) -> bool {
        match ty {
            // C4.5e-0 (DEV-068): user nominals with an `impl Copy` are Copy — the front end
            // has already validated the all-Copy-fields / no-Drop rules for the impl to
            // exist, so lowering consults the impl's presence only. Without one they stay
            // Move (an unmarked all-Copy-field struct is still Move in STARK).
            MirTy::Struct(item, _) | MirTy::Enum(EnumRef::User(item), _) => {
                self.type_has_copy_impl(*item)
            }
            MirTy::Enum(_, args) => args.iter().all(|a| self.is_copy(a)),
            MirTy::Tuple(elems) => elems.iter().all(|e| self.is_copy(e)),
            MirTy::Array(elem, _) => self.is_copy(elem),
            MirTy::Ref { mutable, .. } => !*mutable,
            MirTy::Slice(_) | MirTy::Core(..) | MirTy::String => false,
            _ => true,
        }
    }

    /// Read a place as an operand. C4.5d: a `Move` out of a drop-tracked local clears the
    /// flags of every unit the moved place covers, so the value's drop responsibility
    /// transfers with it (scope-exit drops are flag-guarded and skip it).
    fn read_place(&mut self, place: Place, ty: &MirTy, span: Span) -> Result<Operand, LowerError> {
        if self.is_copy(ty) {
            return Ok(Operand::Copy(place));
        }
        if let Some(units) = self.drop_info.get(&place.local.0) {
            let mut prefix: Vec<u32> = Vec::new();
            for proj in &place.projection {
                match proj {
                    Projection::Field(i) => prefix.push(*i),
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
                    let fields =
                        nominal_instance_fields(self.hir, self.tables, self.meta, *item, args)?;
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
                    let fields =
                        nominal_instance_fields(self.hir, self.tables, self.meta, *item, args)?;
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
            MirTy::String
            | MirTy::Core(crate::hir::CoreType::Vec, _)
            | MirTy::Core(crate::hir::CoreType::VecIter, _)
            | MirTy::Core(crate::hir::CoreType::HashMap, _)
            | MirTy::Core(crate::hir::CoreType::KeysIter, _) => true,
            _ => false,
        })
    }

    /// Decompose a droppable type into drop units: descend through dtor-less structs and
    /// tuples; a type with its own `Drop` impl, an enum, or an array is one unit.
    fn collect_drop_units(
        &self,
        ty: &MirTy,
        path: &mut Vec<u32>,
        out: &mut Vec<(Vec<u32>, MirTy)>,
        span: Span,
    ) -> Result<(), LowerError> {
        if !self.ty_needs_drop(ty, span)? {
            return Ok(());
        }
        match ty {
            MirTy::Struct(item, args) if !self.type_has_drop_impl(*item) => {
                let fields =
                    nominal_instance_fields(self.hir, self.tables, self.meta, *item, args)?;
                let NominalFields::Struct(tys) = fields else {
                    return unsupported("struct item with enum fields shape", span);
                };
                for (i, fty) in tys.iter().enumerate() {
                    path.push(i as u32);
                    self.collect_drop_units(fty, path, out, span)?;
                    path.pop();
                }
            }
            MirTy::Tuple(elems) => {
                for (i, ety) in elems.iter().enumerate() {
                    path.push(i as u32);
                    self.collect_drop_units(ety, path, out, span)?;
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
                };
                let symbol = key_symbol(self.hir, self.meta, &key)?;
                return Ok(Some((key, symbol)));
            }
        }
        Ok(None)
    }

    /// Discover every dtor instance `ty`'s drop glue can invoke: record its symbol for the
    /// type context and queue its body for lowering.
    fn discover_drop_impls(&mut self, ty: &MirTy) -> Result<(), LowerError> {
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
                match nominal_instance_fields(self.hir, self.tables, self.meta, item, &args)? {
                    NominalFields::Struct(tys) => {
                        for t in &tys {
                            self.discover_drop_impls(t)?;
                        }
                    }
                    NominalFields::Enum(variants) => {
                        for v in &variants {
                            for t in v {
                                self.discover_drop_impls(t)?;
                            }
                        }
                    }
                }
            }
            MirTy::Enum(_, args) | MirTy::Tuple(args) => {
                for t in args.clone() {
                    self.discover_drop_impls(&t)?;
                }
            }
            MirTy::Array(elem, _) => self.discover_drop_impls(&elem.clone())?,
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
    fn set_flags_under(&mut self, local: u32, prefix: &[u32], value: bool, span: Span) {
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
            projection: unit.path.iter().map(|&i| Projection::Field(i)).collect(),
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

    /// Emit flag-guarded drops for every scope at `from_depth` or deeper — innermost scope
    /// first, locals in reverse declaration order, units in reverse order. Does not pop the
    /// scope stack (early exits leave the stack intact for the code that follows).
    fn emit_scope_drops_from(&mut self, from_depth: usize, span: Span) {
        let plan: Vec<(u32, DropUnit)> = self.scopes[from_depth.min(self.scopes.len())..]
            .iter()
            .rev()
            .flat_map(|scope| scope.iter().rev())
            .flat_map(|local| {
                self.drop_info
                    .get(&local.0)
                    .into_iter()
                    .flat_map(|units| units.iter().rev())
                    .map(|u| (local.0, u.clone()))
            })
            .collect();
        for (local, unit) in plan {
            self.emit_guarded_drop(local, &unit, span);
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
                let mut prefix: Vec<u32> = Vec::new();
                let mut pure = true;
                for proj in &place.projection {
                    match proj {
                        Projection::Field(i) => prefix.push(*i),
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
                projection: unit.path.iter().map(|&i| Projection::Field(i)).collect(),
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
        let mut subst = Vec::new();
        for (i, arg) in written.args.iter().enumerate() {
            let hir::GenericArg::Type(t) = arg else {
                return unsupported("non-type impl self argument", span0);
            };
            let hir::TypeKind::Path {
                res: Res::TypeParam,
                ..
            } = &self.hir.ty(*t).kind
            else {
                return unsupported(
                    "generic impl with a non-parameter self argument (e.g. impl Holder<Vec<T>>)",
                    span0,
                );
            };
            let name = self
                .meta
                .item_text(impl_item, self.hir.ty(*t).span)
                .to_string();
            let concrete = type_args.get(i).cloned().ok_or_else(|| LowerError {
                what: "impl instantiated with too few type arguments".into(),
                span: span0,
            })?;
            subst.push((name, concrete));
        }
        Ok(subst)
    }

    fn lower_body(&mut self) -> Result<MirBody, LowerError> {
        let symbol = key_symbol(self.hir, self.meta, &self.key)?;
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
                _ => {
                    // A1 lowers IMPL-level generics; a method's OWN generic parameters
                    // (`fn map<U>(&self, ...)`) still need per-call-site instantiation
                    // recording and remain a later increment.
                    return unsupported(
                        "method with its own generic parameters (a later increment)",
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
        let (instance_item, instance_type_args) = match &self.key {
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
        Ok(MirBody {
            instance: Instance {
                item: instance_item,
                type_args: instance_type_args,
                symbol,
            },
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
                            core @ (crate::hir::CoreType::VecIter | crate::hir::CoreType::KeysIter),
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
                            );
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
            _ => unsupported("unit expression form (C4.5)", span),
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
                Res::Item(item) => {
                    let instance = self.top_fn_instance(*item, expr, span)?;
                    Ok(Operand::Const(Constant::FnPtr(instance)))
                }
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
                            if *mutable {
                                return unsupported(
                                    "mutable slice views (&mut base[range]) are reserved",
                                    span,
                                );
                            }
                            let index_span = self.hir.expr(*operand).span;
                            return self.lower_make_slice(*base, *index, index_span);
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
        let (enum_ref, ok_variant, payload_ty) = match &inner_ty {
            MirTy::Enum(er @ EnumRef::CoreOption, args) => {
                (*er, 1u32, args.first().cloned().unwrap_or(MirTy::Unit))
            }
            MirTy::Enum(er @ EnumRef::CoreResult, args) => {
                (*er, 0u32, args.first().cloned().unwrap_or(MirTy::Unit))
            }
            other => {
                return unsupported(format!("`?` on a non-Option/Result type {other:?}"), span)
            }
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
                // Err(payload) — move the Err payload (variant 1, field 0) out of scrut.
                let err_payload = Operand::Move(Place {
                    local: scrut,
                    projection: vec![Projection::VariantField(1, 0)],
                });
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
        self.emit_scope_drops_from(0, span);
        // Seal the err block with Return and continue lowering in the Ok/Some block (Return
        // has no CFG edge, so `ok_block` is only the continuation point, not a successor).
        self.terminate(Terminator::Return, self.info(span), ok_block);

        // Ok/Some path: the expression's value is the payload.
        let payload_place = Place {
            local: scrut,
            projection: vec![Projection::VariantField(ok_variant, 0)],
        };
        let out = self.read_place(payload_place, &payload_ty, span)?;
        Ok(out)
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
                BinOp::Add | BinOp::Sub | BinOp::Mul => {
                    let mir_op = match op {
                        BinOp::Add => MirBinOp::FloatAdd,
                        BinOp::Sub => MirBinOp::FloatSub,
                        BinOp::Mul => MirBinOp::FloatMul,
                        _ => unreachable!(),
                    };
                    let dest = self.new_temp(operand_ty.clone());
                    self.emit(
                        Statement::Assign(Place::local(dest), Rvalue::BinOp(mir_op, lhs, rhs)),
                        self.info(span),
                    );
                    return Ok(Operand::Copy(Place::local(dest)));
                }
                BinOp::Div | BinOp::Rem => {
                    let checked = if matches!(op, BinOp::Div) {
                        CheckedOp::FloatDiv
                    } else {
                        CheckedOp::FloatRem
                    };
                    return self.checked_to_temp(
                        checked,
                        vec![lhs, rhs],
                        operand_ty.clone(),
                        TrapCategory::DivideByZero,
                        span,
                    );
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

    /// A place for `base`, auto-dereffed: if `base`'s type is a reference, the returned place
    /// carries the needed `Deref` projections and the returned type is the referent.
    fn lower_place_autoderef(&mut self, base: ExprId) -> Result<(Place, MirTy), LowerError> {
        let base_ty = self.expr_mir_ty(base)?;
        let (peeled, layers) = Self::peel_refs(base_ty);
        let mut place = self.lower_place(base)?;
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
            _ => unsupported("place expression (C4.5)", span),
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
        let base_ref = if layers > 0 {
            self.lower_expr_to_operand(base)?
        } else {
            let place = self.place_or_temp(base, &peeled, span)?;
            let ref_ty = MirTy::Ref {
                mutable: false,
                inner: Box::new(peeled.clone()),
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
            mutable: false,
            inner: Box::new(MirTy::Slice(Box::new(elem))),
        };
        let dest = self.new_temp(slice_ty.clone());
        self.emit_runtime_call(
            RuntimeFn::SliceNew,
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
        if !matches!(base_ty, MirTy::Array(..)) {
            return unsupported(
                format!("indexing {base_ty:?} (Vec is runtime-surface; slices are C4.5b-2)"),
                span,
            );
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
            return self.lower_method_call(base, name_span, &args, dest, span);
        }

        match &self.hir.expr(callee).kind {
            hir::ExprKind::Path { res, .. } => match res {
                Res::Builtin(builtin @ (Builtin::Println | Builtin::Print)) => {
                    if args.len() != 1 {
                        return unsupported("print/println arity", span);
                    }
                    let arg_ty = self.expr_mir_ty(args[0])?;
                    let is_println = matches!(builtin, Builtin::Println);
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
                            self.emit_runtime_call(
                                RuntimeFn::StringAsStr,
                                vec![recv],
                                Place::local(tmp),
                                span,
                            );
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
                    let value = self.lower_expr_to_operand(args[0])?;
                    let (runtime, widened) = self.widen_for_print(value, &arg_ty, span)?;
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
                // A1 (CD-031), C4.5e-2: Vec construction.
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
                    let instance = self.top_fn_instance(*item, callee, span)?;
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
                        }
                    };
                    let symbol = key_symbol(self.hir, self.meta, &key)?;
                    let instance_args = match &key {
                        FnKey::ImplFn { type_args, .. } => type_args.clone(),
                        _ => Vec::new(),
                    };
                    self.discovered_callees.push(key);
                    let ops = args
                        .iter()
                        .map(|&a| self.lower_expr_to_operand(a))
                        .collect::<Result<Vec<_>, _>>()?;
                    let after = self.new_block();
                    self.terminate(
                        Terminator::Call {
                            callee: Callee::Instance(Instance {
                                item: nominal,
                                type_args: instance_args,
                                symbol,
                            }),
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
    fn top_fn_instance(
        &mut self,
        item: ItemId,
        use_expr: ExprId,
        span: Span,
    ) -> Result<Instance, LowerError> {
        let ItemKind::Fn(def) = &self.hir.item(item).kind else {
            return unsupported("use of a non-function item as a function", span);
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
        let symbol = key_symbol(self.hir, self.meta, &key)?;
        self.discovered_callees.push(key);
        let eq_dest = self.new_temp(MirTy::Bool);
        let after = self.new_block();
        self.terminate(
            Terminator::Call {
                callee: Callee::Instance(Instance {
                    item: nominal,
                    type_args: type_args.to_vec(),
                    symbol,
                }),
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
        let symbol = key_symbol(self.hir, self.meta, &key)?;
        self.discovered_callees.push(key);
        // cmp(&a, &b) -> Ordering
        let ord_ty = MirTy::Enum(EnumRef::CoreOrdering, Vec::new());
        let ord_dest = self.new_temp(ord_ty.clone());
        let after = self.new_block();
        self.terminate(
            Terminator::Call {
                callee: Callee::Instance(Instance {
                    item: nominal,
                    type_args: type_args.to_vec(),
                    symbol,
                }),
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

    /// Lower `receiver.method(args)` — evaluation order: receiver first (CD-007/CD-010).
    fn lower_method_call(
        &mut self,
        base: ExprId,
        name_span: Span,
        args: &[ExprId],
        dest: Place,
        span: Span,
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
                    | MirTy::String
                    | MirTy::Str
            )
        {
            return self.lower_primitive_cmp(base, args[0], dest, span);
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
        // 0.1-A6 (A4 slicing): slice methods — `len`/`is_empty` on a `&[T]` receiver.
        if matches!(peeled_ty, MirTy::Slice(_)) {
            let name = self.text(name_span);
            let rt = match name {
                "len" => RuntimeFn::SliceLen,
                "is_empty" => RuntimeFn::SliceIsEmpty,
                other => return unsupported(format!("slice method {other}"), span),
            };
            // The receiver expression is the `&[T]` value itself (peel found ref layers).
            let recv = self.lower_expr_to_operand(base)?;
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
        // Receiver operand FIRST (normative order), before arguments. C4.5b-2: real borrows.
        let receiver_op = match receiver {
            Some(kind @ (hir::Receiver::Ref | hir::Receiver::RefMut)) => {
                let mutable = matches!(kind, hir::Receiver::RefMut);
                if base_ref_layers > 0 {
                    // The receiver expression is already a reference (`self` inside a method
                    // forwarding to a sibling): pass the reference value itself.
                    self.lower_expr_to_operand(base)?
                } else {
                    // Borrow the receiver place (materialize a temp for non-place receivers,
                    // e.g. a call result used as `make().method()`).
                    let place = match self.lower_place(base) {
                        Ok(place) => place,
                        Err(_) => {
                            let value = self.lower_expr_to_operand(base)?;
                            let temp = self.new_temp(peeled_ty.clone());
                            self.emit(
                                Statement::Assign(Place::local(temp), Rvalue::Use(value)),
                                self.info(span),
                            );
                            Place::local(temp)
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
            }
            Some(hir::Receiver::Value) => self.lower_expr_to_operand(base)?,
            None => {
                return unsupported("method-syntax call to a receiverless fn", span);
            }
        };
        let symbol = key_symbol(self.hir, self.meta, &key)?;
        self.discovered_callees.push(key);
        let mut ops = vec![receiver_op];
        for &arg in args {
            ops.push(self.lower_expr_to_operand(arg)?);
        }
        let after = self.new_block();
        self.terminate(
            Terminator::Call {
                callee: Callee::Instance(Instance {
                    item: nominal,
                    type_args: nominal_args.clone(),
                    symbol,
                }),
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
                if self.ty_needs_drop(&payload_ty, span)? {
                    return unsupported(
                        "unwrap_or on a droppable payload type (a later increment)",
                        span,
                    );
                }
                let Some(&default_expr) = call_args.first() else {
                    return unsupported("unwrap_or expects one argument", span);
                };
                let default_op = self.lower_expr_to_operand(default_expr)?;
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
                // Ok/Some arm: move the payload into dest.
                let mut payload_place = place;
                payload_place
                    .projection
                    .push(Projection::VariantField(ok_variant, 0));
                let payload = self.read_place(payload_place, &payload_ty, span)?;
                self.emit(
                    Statement::Assign(dest.clone(), Rvalue::Use(payload)),
                    self.info(span),
                );
                self.terminate(
                    Terminator::Goto { target: join },
                    self.info(span),
                    else_block,
                );
                // None/Err arm: the default.
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

    /// 0.1-A2 (C4.5f-2): `for value in v.iter() { body }`. Desugar:
    /// ```text
    /// it = <iter expr>            // VecIterNew(&v) via the method-call lowering
    /// header:
    ///   nxt = VecIterNext(&mut it)     // Option<&T>
    ///   switch discriminant(nxt) [Some → body_bb] else exit
    /// body_bb:
    ///   value: &T = move nxt.v1.0
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
        if self.ty_needs_drop(&elem, span)? {
            // The yielded value would need per-iteration drop elaboration; recorded residual.
            return unsupported(
                "user Iterator yielding a droppable Item type (a later increment)",
                span,
            );
        }
        let symbol = key_symbol(self.hir, self.meta, &key)?;
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
                callee: Callee::Instance(Instance {
                    item,
                    type_args: targs,
                    symbol,
                }),
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
        self.emit(
            Statement::Assign(
                Place::local(bound),
                Rvalue::Use(Operand::Move(Place {
                    local: nxt,
                    projection: vec![Projection::VariantField(1, 0)],
                })),
            ),
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
    ) -> Result<(), LowerError> {
        // `elem_ref` is the type the loop variable binds to and the `Next` Option's payload:
        // `&T` for Vec/HashMap iteration, `Char` (by value) for `chars()`.
        let opt_ty = MirTy::Enum(EnumRef::CoreOption, vec![elem_ref.clone()]);

        // Materialize the iterator into a registered droppable local.
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
        self.emit(
            Statement::Assign(
                Place::local(bound),
                Rvalue::Use(Operand::Move(Place {
                    local: nxt,
                    projection: vec![Projection::VariantField(1, 0)],
                })),
            ),
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
            "keys" => (RuntimeFn::HashMapKeysIterNew, false),
            _ => return unsupported(format!("HashMap::{name} (reserved — std-full)"), span),
        };
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
                ) {
                    Ok(NominalFields::Struct(tys)) => tys.iter().any(|t| self.ty_has_user_drop(t)),
                    Ok(NominalFields::Enum(vs)) => vs
                        .iter()
                        .any(|v| v.iter().any(|t| self.ty_has_user_drop(t))),
                    Err(_) => true, // unresolvable: be conservative
                }
            }
            MirTy::Enum(_, args) | MirTy::Core(_, args) | MirTy::Tuple(args) => {
                args.iter().any(|t| self.ty_has_user_drop(t))
            }
            MirTy::Array(elem, _) => self.ty_has_user_drop(elem),
            _ => false,
        }
    }

    /// Build a `&HashMap`/`&mut HashMap` receiver operand.
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
            // Already a reference to the String — pass it through.
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

    /// Emit a `Call` to a runtime op with a fresh successor block.
    fn emit_runtime_call(&mut self, rt: RuntimeFn, ops: Vec<Operand>, dest: Place, span: Span) {
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
            MirTy::Float32 => {
                let widened = self.cast_to_temp(value, MirTy::Float64, span)?;
                Ok((PrintKind::Float, widened))
            }
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
            // patterns — routes to the general recursive engine. Consuming mode requires a
            // drop-free scrutinee (droppable + nested is the recorded residual: it needs the
            // C4.5d drop-unit decomposition generalized to arbitrary pattern trees).
            MirTy::Enum(..)
            | MirTy::Tuple(_)
            | MirTy::Array(..)
            | MirTy::Struct(..)
            | MirTy::Float32
            | MirTy::Float64
            | MirTy::Ref { .. } => {
                if mode == MatchMode::Consuming && self.ty_needs_drop(&scrut_ty, span)? {
                    return unsupported(
                        "droppable scrutinee with nested/general patterns (A2 residual)",
                        span,
                    );
                }
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
                // DEV-072 CLOSED (WP-C4.7-5): borrowck now rejects this in the front end
                // (E0101), using the SAME by-reference classification as `lower_match`, so a
                // checked program can no longer reach here. The guard is kept deliberately as
                // defense in depth — the charter's rule is that nothing unsupported reaches a
                // backend silently, and a guard that is unreachable-by-construction costs
                // nothing while a missing one would mislower a move out of a borrow.
                if mode == MatchMode::ByRef && !self.is_copy(ty) {
                    return unsupported(
                        "binding a non-Copy payload through a shared reference (unreachable for checked programs since DEV-072; defense in depth)",
                        pat_span,
                    );
                }
                let (name, local) = (self.text(*name).to_string(), *local);
                self.locals.push(LocalDecl {
                    ty: ty.clone(),
                    kind: LocalKind::User(name),
                });
                let bound = LocalId((self.locals.len() - 1) as u32);
                self.local_map.insert(local.0, bound);
                let value = self.read_place(place.clone(), ty, span)?;
                self.emit(
                    Statement::Assign(Place::local(bound), Rvalue::Use(value)),
                    self.synthetic(span, SyntheticKind::MatchDesugar),
                );
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
        if mode == MatchMode::ByRef && !self.is_copy(ty) {
            return unsupported(
                "binding a non-Copy payload through a shared reference (front-end move-out-of-borrow gap)",
                span,
            );
        }
        self.locals.push(LocalDecl {
            ty: ty.clone(),
            kind: LocalKind::User(name),
        });
        let bound = LocalId((self.locals.len() - 1) as u32);
        self.local_map.insert(hir_local.0, bound);
        let value = self.read_place(place.clone(), ty, span)?;
        self.emit(
            Statement::Assign(Place::local(bound), Rvalue::Use(value)),
            self.synthetic(span, SyntheticKind::MatchDesugar),
        );
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
    fn array_elem_place(
        &mut self,
        base: &Place,
        array_ty: &MirTy,
        index: usize,
        span: Span,
    ) -> Result<Place, LowerError> {
        let _ = array_ty;
        self.locals.push(LocalDecl {
            ty: MirTy::Int64,
            kind: LocalKind::IndexProof,
        });
        let proof = LocalId((self.locals.len() - 1) as u32);
        let after = self.new_block();
        self.terminate(
            Terminator::Checked {
                op: CheckedOp::CheckIndex,
                args: vec![
                    Operand::Copy(base.clone()),
                    Operand::Const(Constant::Int(index as i128, MirTy::Int64)),
                ],
                dest: proof,
                target: after,
                trap: TrapInfo {
                    category: TrapCategory::IndexOutOfBounds,
                    source: self.synthetic(span, SyntheticKind::MatchDesugar),
                },
            },
            self.synthetic(span, SyntheticKind::MatchDesugar),
            after,
        );
        let mut place = base.clone();
        place.projection.push(Projection::Index(proof));
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
                match nominal_instance_fields(self.hir, self.tables, self.meta, item, scrut_args)? {
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
        match &self.hir.pat(pat).kind {
            hir::PatKind::TupleVariant { pats, .. } => {
                let pats = pats.clone();
                for (i, sub) in pats.iter().enumerate() {
                    let field_ty = payload_tys.get(i).cloned().unwrap_or(MirTy::Unit);
                    self.consume_field(
                        scrut.clone(),
                        mode,
                        variant,
                        i as u32,
                        &field_ty,
                        Some(*sub),
                        span,
                    )?;
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
                            scrut.clone(),
                            mode,
                            variant,
                            index as u32,
                            &field_ty,
                            Some(*sub),
                            span,
                        )?,
                        (None, Some(local)) => self.bind_field_local(
                            scrut.clone(),
                            mode,
                            variant,
                            index as u32,
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
                        self.consume_field(scrut.clone(), mode, variant, i as u32, ty, None, span)?;
                    }
                }
            }
            // Unit variant (`None`, `E::Empty`) — no payload.
            hir::PatKind::Path { .. } => {}
            _ => {}
        }
        Ok(())
    }

    /// Consume one variant payload field given its sub-pattern (`None` = unbound/Wild).
    #[allow(clippy::too_many_arguments)]
    fn consume_field(
        &mut self,
        scrut: Place,
        mode: MatchMode,
        variant: u32,
        index: u32,
        field_ty: &MirTy,
        sub: Option<hir::PatId>,
        span: Span,
    ) -> Result<(), LowerError> {
        match sub.map(|s| &self.hir.pat(s).kind) {
            Some(hir::PatKind::Binding { name, local }) => {
                let (name, local) = (self.text(*name).to_string(), *local);
                self.bind_field_local(scrut, mode, variant, index, name, local, field_ty, span)
            }
            Some(hir::PatKind::Wild) | None => {
                // ByRef: nothing is consumed — the referent keeps ownership of every payload.
                if mode == MatchMode::Consuming && self.ty_needs_drop(field_ty, span)? {
                    self.discover_drop_impls(field_ty)?;
                    let mut place = scrut;
                    place
                        .projection
                        .push(Projection::VariantField(variant, index));
                    let value = self.read_place(place, field_ty, span)?;
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
    /// register it to drop at arm end. ByRef: the field must be `Copy` (read by copy; a
    /// non-Copy binding would move out of the borrow — a front-end move-out-of-borrow gap,
    /// recorded, keeps such programs out of MIR).
    #[allow(clippy::too_many_arguments)]
    fn bind_field_local(
        &mut self,
        scrut: Place,
        mode: MatchMode,
        variant: u32,
        index: u32,
        name: String,
        hir_local: crate::hir::LocalId,
        field_ty: &MirTy,
        span: Span,
    ) -> Result<(), LowerError> {
        if mode == MatchMode::ByRef && !self.is_copy(field_ty) {
            return unsupported(
                "binding a non-Copy payload through a shared reference (front-end move-out-of-borrow gap)",
                span,
            );
        }
        self.locals.push(LocalDecl {
            ty: field_ty.clone(),
            kind: LocalKind::User(name),
        });
        let bound = LocalId((self.locals.len() - 1) as u32);
        self.local_map.insert(hir_local.0, bound);
        let mut place = scrut;
        place
            .projection
            .push(Projection::VariantField(variant, index));
        let value = self.read_place(place, field_ty, span)?;
        self.emit(
            Statement::Assign(Place::local(bound), Rvalue::Use(value)),
            self.synthetic(span, SyntheticKind::MatchDesugar),
        );
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
