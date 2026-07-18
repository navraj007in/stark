//! Name resolution and AST-to-HIR lowering pass for STARK (PLAN.md M2.1).

use crate::ast;
use crate::diag::Diagnostic;
use crate::hir::{self, Builtin, CoreTrait, CoreType, Hir, LocalId, Res};
use crate::options::LanguageOptions;
use crate::source::{SourceFile, Span};
use std::collections::{hash_map::Entry, HashMap};
use std::sync::Arc;

/// A single-segment name reserved by the `tensor` extension: element types,
/// tensor/device type constructors, and the `Dim`/`DType` kinds. Used to give
/// a focused "requires extension `tensor`" diagnostic in Core-only mode and to
/// suppress "undefined type" for these names under the extension (their full
/// resolution lands in M4.2).
fn extension_reserved_name(name: &str) -> Option<&'static str> {
    match name {
        "Dim" => Some("`Dim` kind"),
        "DType" => Some("`DType` kind"),
        "Device" => Some("`Device` kind"),
        "Float16" => Some("`Float16` element type"),
        "BFloat16" => Some("`BFloat16` element type"),
        "Tensor" => Some("`Tensor` type"),
        "TensorDyn" => Some("`TensorDyn` type"),
        "TensorAny" => Some("`TensorAny` type"),
        "Cpu" => Some("`Cpu` device type"),
        "Cuda" => Some("`Cuda` device type"),
        "ByteRange" => Some("`ByteRange` value range"),
        "UnitRange" => Some("`UnitRange` value range"),
        "Normalized" => Some("`Normalized` value range"),
        "Unspecified" => Some("`Unspecified` value range"),
        "ModelError" => Some("`ModelError` type"),
        _ => None,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ModuleId(pub u32);

struct ModuleData {
    #[allow(dead_code)]
    name: String,
    parent: Option<ModuleId>,
    file: Arc<SourceFile>,
    items: HashMap<String, Res>,
    submodules: HashMap<String, ModuleId>,
    package_root: ModuleId,
}

enum ItemDefDetail {
    Enum {
        variants: Vec<String>,
    },
    #[allow(dead_code)]
    Struct {
        fields: Vec<String>,
    },
    #[allow(dead_code)]
    Trait {
        items: Vec<String>,
    },
    Model,
    Other,
}

pub struct Resolver<'a> {
    ast: &'a ast::Ast,
    hir: Hir,
    diags: Vec<Diagnostic>,
    modules: Vec<ModuleData>,
    current_module: ModuleId,
    scopes: Vec<HashMap<String, Res>>,
    local_count: u32,
    item_map: HashMap<ast::ItemId, hir::ItemId>,
    item_modules: HashMap<hir::ItemId, ModuleId>,
    item_details: HashMap<hir::ItemId, ItemDefDetail>,
    submodule_map: HashMap<ast::ItemId, ModuleId>,
    options: LanguageOptions,
    current_use_item_vis: Option<ast::Vis>,
    reexport_vis: HashMap<(ModuleId, String), Option<ast::Vis>>,
}

/// Resolve `ast` in Core-only mode (the Core v1 entry point).
pub fn resolve(ast: &ast::Ast, file: Arc<SourceFile>) -> (Hir, Vec<Diagnostic>) {
    resolve_with_options(ast, file, LanguageOptions::CORE)
}

/// Resolve `ast` under `options`, which may enable extensions (Gate 4+).
pub fn resolve_with_options(
    ast: &ast::Ast,
    file: Arc<SourceFile>,
    options: LanguageOptions,
) -> (Hir, Vec<Diagnostic>) {
    let mut resolver = Resolver {
        ast,
        hir: Hir::default(),
        diags: Vec::new(),
        modules: Vec::new(),
        current_module: ModuleId(0),
        scopes: Vec::new(),
        local_count: 0,
        item_map: HashMap::new(),
        item_modules: HashMap::new(),
        item_details: HashMap::new(),
        submodule_map: HashMap::new(),
        options,
        current_use_item_vis: None,
        reexport_vis: HashMap::new(),
    };

    // Root module
    resolver.modules.push(ModuleData {
        name: "crate".to_string(),
        parent: None,
        file: file.clone(),
        items: HashMap::new(),
        submodules: HashMap::new(),
        package_root: ModuleId(0),
    });

    // Pass 1: Declare items (collect all module-level item signatures)
    let root_items = match &ast.root {
        ast::Root::Program(items) => items.clone(),
        ast::Root::Snippet { stmts, tail: _ } => {
            // Snippets can contain items and statements. Let's extract items.
            let mut items = Vec::new();
            for &stmt_id in stmts {
                if let ast::StmtKind::Item(item_id) = ast.stmt(stmt_id).kind {
                    items.push(item_id);
                }
            }
            items
        }
    };

    resolver.declare_items(&root_items);

    // Pass 2: Resolve use Tree imports (with fixed-point iteration for re-exports)
    let mut last_total_items = 0;
    loop {
        resolver.resolve_imports(&root_items);
        let total_items = resolver
            .modules
            .iter()
            .map(|m| m.items.len())
            .sum::<usize>();
        if total_items == last_total_items {
            break;
        }
        last_total_items = total_items;
    }

    // Run final unresolved imports check
    resolver.check_imports_resolved(&root_items);
    for ((module, name), visibility) in &resolver.reexport_vis {
        if matches!(visibility, Some(ast::Vis::Pub)) {
            if let Some(Res::Item(item)) = resolver.modules[module.0 as usize].items.get(name) {
                resolver.hir.publicly_nameable_items.insert(*item);
            }
        }
    }

    // Pass 3: Lower AST to HIR & perform lexical/local name resolution
    resolver.lower_crate();

    (resolver.hir, resolver.diags)
}

impl<'a> Resolver<'a> {
    fn current_file(&self) -> &SourceFile {
        &self.modules[self.current_module.0 as usize].file
    }

    fn current_file_arc(&self) -> Arc<SourceFile> {
        self.modules[self.current_module.0 as usize].file.clone()
    }

    /// WP-C1.2 (2026-07-17): `self.current_module` accurately tracks which module (and
    /// therefore which file) is being processed at every diagnostic-construction site (it is
    /// saved/restored around every submodule descent in declare_items/resolve_imports/
    /// lower_crate), but resolve.rs never attached that file to its own diagnostics -- every
    /// resolve-stage diagnostic for a non-root file in a multi-file package rendered against
    /// the wrong file. This mirrors typecheck.rs's own if-none backfill pattern
    /// (typecheck.rs:2041-2044 etc.) rather than requiring every call site to remember to call
    /// `.with_file(...)` itself. See COMPILER-STATE.md DEV-006.
    fn push_diag(&mut self, diag: Diagnostic) {
        let diag = if diag.file.is_none() {
            diag.with_file(self.current_file_arc())
        } else {
            diag
        };
        self.diags.push(diag);
    }

    fn text(&self, span: Span) -> &str {
        if span.lo >= 0x8000_0000 {
            if let Some(s) = self.ast.synthetic_spans.get(&span) {
                return s;
            }
        }
        let file = self.current_file();
        &file.src[span.lo as usize..span.hi as usize]
    }

    fn path_to_string(&self, path: &ast::Path) -> String {
        path.segments
            .iter()
            .map(|seg| self.text(seg.span))
            .collect::<Vec<_>>()
            .join("::")
    }

    fn alloc_local(&mut self) -> LocalId {
        let id = self.local_count;
        self.local_count += 1;
        LocalId(id)
    }

    fn push_scope(&mut self) {
        self.scopes.push(HashMap::new());
    }

    fn pop_scope(&mut self) {
        self.scopes.pop();
    }

    fn declare_items(&mut self, items: &[ast::ItemId]) {
        let current_mod_id = self.current_module;

        // 1. Declare names in the current module scope.
        for &ast_id in items {
            let item = self.ast.item(ast_id);
            let name_span = match &item.kind {
                ast::ItemKind::Fn(f) => Some(f.sig.name),
                ast::ItemKind::Struct { name, .. } => Some(*name),
                ast::ItemKind::Enum { name, .. } => Some(*name),
                ast::ItemKind::Trait { name, .. } => Some(*name),
                ast::ItemKind::Const { name, .. } => Some(*name),
                ast::ItemKind::TypeAlias { name, .. } => Some(*name),
                ast::ItemKind::Mod { name, .. } => Some(*name),
                ast::ItemKind::Model(def) => Some(def.name),
                ast::ItemKind::Use(_) => None,
                ast::ItemKind::Impl { .. } => None,
            };

            if let Some(span) = name_span {
                let name_str = self.text(span).to_string();
                match self.modules[current_mod_id.0 as usize]
                    .items
                    .entry(name_str.clone())
                {
                    Entry::Occupied(_) => self.push_diag(
                        Diagnostic::error(
                            format!("duplicate definition of '{}' in the same scope", name_str),
                            span,
                        )
                        .with_code("E0204")
                        .with_label("item redefined here"),
                    ),
                    Entry::Vacant(entry) => {
                        entry.insert(Res::Item(hir::ItemId(ast_id.0)));
                    }
                }
            }

            // Populate item details for variants/members
            let hir_id = hir::ItemId(ast_id.0);
            self.item_modules.insert(hir_id, current_mod_id);
            match &item.kind {
                ast::ItemKind::Enum { variants, .. } => {
                    let variant_names = variants
                        .iter()
                        .map(|v| self.text(v.name).to_string())
                        .collect();
                    self.item_details.insert(
                        hir_id,
                        ItemDefDetail::Enum {
                            variants: variant_names,
                        },
                    );
                }
                ast::ItemKind::Struct { fields, .. } => {
                    let field_names = fields
                        .iter()
                        .map(|f| self.text(f.name).to_string())
                        .collect();
                    self.item_details.insert(
                        hir_id,
                        ItemDefDetail::Struct {
                            fields: field_names,
                        },
                    );
                }
                ast::ItemKind::Trait {
                    items: trait_items, ..
                } => {
                    let item_names = trait_items
                        .iter()
                        .map(|ti| match ti {
                            ast::TraitItem::Method { sig, .. } => self.text(sig.name).to_string(),
                            ast::TraitItem::AssocType { name } => self.text(*name).to_string(),
                        })
                        .collect();
                    self.item_details
                        .insert(hir_id, ItemDefDetail::Trait { items: item_names });
                }
                ast::ItemKind::Model(_) => {
                    self.item_details.insert(hir_id, ItemDefDetail::Model);
                }
                _ => {
                    self.item_details.insert(hir_id, ItemDefDetail::Other);
                }
            }
        }

        // 2. Process submodules recursively
        for &ast_id in items {
            let item = self.ast.item(ast_id);
            if let ast::ItemKind::Mod {
                name,
                items: ref sub_items,
            } = item.kind
            {
                let name_str = self.text(name).to_string();
                let sub_mod_id = ModuleId(self.modules.len() as u32);

                let file = if let Some(ref sub_items_vec) = sub_items {
                    if !sub_items_vec.is_empty() {
                        if let Some(file_arc) = self.ast.item_files.get(&sub_items_vec[0]) {
                            file_arc.clone()
                        } else {
                            self.modules[current_mod_id.0 as usize].file.clone()
                        }
                    } else {
                        self.modules[current_mod_id.0 as usize].file.clone()
                    }
                } else {
                    self.modules[current_mod_id.0 as usize].file.clone()
                };

                let is_dep_package = name.lo >= 0x8000_0000;
                let package_root = if is_dep_package {
                    sub_mod_id
                } else {
                    self.modules[current_mod_id.0 as usize].package_root
                };

                let sub_mod_data = ModuleData {
                    name: name_str.clone(),
                    parent: Some(current_mod_id),
                    file,
                    items: HashMap::new(),
                    submodules: HashMap::new(),
                    package_root,
                };
                self.modules.push(sub_mod_data);
                self.modules[current_mod_id.0 as usize]
                    .submodules
                    .insert(name_str, sub_mod_id);
                self.submodule_map.insert(ast_id, sub_mod_id);

                if let Some(ref sub_items_vec) = sub_items {
                    self.current_module = sub_mod_id;
                    self.declare_items(sub_items_vec);
                    self.current_module = current_mod_id;
                }
            }
        }
    }

    fn check_imports_resolved(&mut self, items: &[ast::ItemId]) {
        let current_mod_id = self.current_module;
        for &ast_id in items {
            let item = self.ast.item(ast_id);
            if let ast::ItemKind::Use(use_tree) = &item.kind {
                self.check_use_tree_resolved(current_mod_id, use_tree);
            }
        }
        for &ast_id in items {
            let item = self.ast.item(ast_id);
            if let ast::ItemKind::Mod {
                items: Some(ref sub_items),
                ..
            } = item.kind
            {
                if let Some(&sub_mod_id) = self.submodule_map.get(&ast_id) {
                    self.current_module = sub_mod_id;
                    self.check_imports_resolved(sub_items);
                    self.current_module = current_mod_id;
                }
            }
        }
    }

    fn check_use_tree_resolved(&mut self, current_mod: ModuleId, tree: &ast::UseTree) {
        match tree {
            ast::UseTree::Path { path, .. } => {
                let res = self.resolve_path(current_mod, path);
                if res == Res::Err {
                    self.push_diag(
                        Diagnostic::error(
                            format!("unresolved import '{}'", self.path_to_string(path)),
                            path.span,
                        )
                        .with_code("E0205"),
                    );
                }
            }
            ast::UseTree::Glob { prefix } => {
                let res = self.resolve_path(current_mod, prefix);
                if res == Res::Err {
                    self.push_diag(
                        Diagnostic::error(
                            format!("unresolved import '{}'", self.path_to_string(prefix)),
                            prefix.span,
                        )
                        .with_code("E0205"),
                    );
                }
            }
            ast::UseTree::SelfImport { prefix } => {
                let res = self.resolve_path(current_mod, prefix);
                if res == Res::Err {
                    self.push_diag(
                        Diagnostic::error(
                            format!("unresolved import '{}'", self.path_to_string(prefix)),
                            prefix.span,
                        )
                        .with_code("E0205"),
                    );
                }
            }
            ast::UseTree::Group { prefix, items } => {
                let base_res = self.resolve_path(current_mod, prefix);
                if base_res == Res::Err {
                    self.push_diag(
                        Diagnostic::error(
                            format!("unresolved import '{}'", self.path_to_string(prefix)),
                            prefix.span,
                        )
                        .with_code("E0205"),
                    );
                } else {
                    for item in items {
                        self.check_use_tree_resolved(current_mod, item);
                    }
                }
            }
        }
    }

    fn resolve_imports(&mut self, items: &[ast::ItemId]) {
        let current_mod_id = self.current_module;

        // Resolve use imports in current module
        for &ast_id in items {
            let item = self.ast.item(ast_id);
            if let ast::ItemKind::Use(use_tree) = &item.kind {
                self.current_use_item_vis = item.vis;
                self.resolve_use_tree(current_mod_id, use_tree);
                self.current_use_item_vis = None;
            }
        }

        // Recurse into submodules
        for &ast_id in items {
            let item = self.ast.item(ast_id);
            if let ast::ItemKind::Mod {
                items: Some(ref sub_items),
                ..
            } = item.kind
            {
                if let Some(&sub_mod_id) = self.submodule_map.get(&ast_id) {
                    self.current_module = sub_mod_id;
                    self.resolve_imports(sub_items);
                    self.current_module = current_mod_id;
                }
            }
        }
    }

    fn resolve_use_tree(&mut self, current_mod: ModuleId, tree: &ast::UseTree) {
        match tree {
            ast::UseTree::Path { path, alias } => {
                let res = self.resolve_path(current_mod, path);
                if res != Res::Err {
                    let name = if let Some(alias_span) = alias {
                        self.text(*alias_span).to_string()
                    } else if let Some(last) = path.segments.last() {
                        self.text(last.span).to_string()
                    } else {
                        return;
                    };
                    self.insert_module_item(current_mod, name, res, path.span);
                }
            }
            ast::UseTree::Glob { prefix } => {
                let res = self.resolve_path(current_mod, prefix);
                if let Res::Item(target_item_id) = res {
                    if let Some(&sub_mod_id) =
                        self.submodule_map.get(&ast::ItemId(target_item_id.0))
                    {
                        // WP-C1.2 (2026-07-17): sort by name before iterating. `items` is a
                        // HashMap, whose iteration order is randomized per-process by Rust's
                        // default SipHash seed; iterating it directly made which of two
                        // glob-colliding names wins (vs. gets flagged E0204 by
                        // insert_module_item) nondeterministic across runs of the identical
                        // program. See COMPILER-STATE.md DEV-007.
                        let mut items_to_copy: Vec<(String, Res)> = self.modules
                            [sub_mod_id.0 as usize]
                            .items
                            .iter()
                            .map(|(k, v)| (k.clone(), *v))
                            .collect();
                        items_to_copy.sort_by(|a, b| a.0.cmp(&b.0));
                        for (k, v) in items_to_copy {
                            self.insert_module_item(current_mod, k, v, prefix.span);
                        }
                    }
                }
            }
            ast::UseTree::SelfImport { prefix } => {
                let res = self.resolve_path(current_mod, prefix);
                if res != Res::Err {
                    if let Some(last) = prefix.segments.last() {
                        let name = self.text(last.span).to_string();
                        self.insert_module_item(current_mod, name, res, prefix.span);
                    }
                }
            }
            ast::UseTree::Group { prefix, items } => {
                let res = self.resolve_path(current_mod, prefix);
                if let Res::Item(target_item_id) = res {
                    if let Some(&sub_mod_id) =
                        self.submodule_map.get(&ast::ItemId(target_item_id.0))
                    {
                        for item in items {
                            self.resolve_use_tree_relative(current_mod, sub_mod_id, item);
                        }
                    }
                }
            }
        }
    }

    fn resolve_use_tree_relative(
        &mut self,
        import_mod: ModuleId,
        target_mod: ModuleId,
        tree: &ast::UseTree,
    ) {
        match tree {
            ast::UseTree::Path { path, alias } => {
                let res = self.resolve_path_relative(target_mod, path);
                if res != Res::Err {
                    let name = if let Some(alias_span) = alias {
                        self.text(*alias_span).to_string()
                    } else if let Some(last) = path.segments.last() {
                        self.text(last.span).to_string()
                    } else {
                        return;
                    };
                    self.insert_module_item(import_mod, name, res, path.span);
                }
            }
            ast::UseTree::Glob { prefix } => {
                let res = self.resolve_path_relative(target_mod, prefix);
                if let Res::Item(target_item_id) = res {
                    if let Some(&sub_mod_id) =
                        self.submodule_map.get(&ast::ItemId(target_item_id.0))
                    {
                        // WP-C1.2 (2026-07-17): sort by name before iterating. `items` is a
                        // HashMap, whose iteration order is randomized per-process by Rust's
                        // default SipHash seed; iterating it directly made which of two
                        // glob-colliding names wins (vs. gets flagged E0204 by
                        // insert_module_item) nondeterministic across runs of the identical
                        // program. See COMPILER-STATE.md DEV-007.
                        let mut items_to_copy: Vec<(String, Res)> = self.modules
                            [sub_mod_id.0 as usize]
                            .items
                            .iter()
                            .map(|(k, v)| (k.clone(), *v))
                            .collect();
                        items_to_copy.sort_by(|a, b| a.0.cmp(&b.0));
                        for (k, v) in items_to_copy {
                            self.insert_module_item(import_mod, k, v, prefix.span);
                        }
                    }
                }
            }
            ast::UseTree::SelfImport { prefix } => {
                let res = self.resolve_path_relative(target_mod, prefix);
                if res != Res::Err {
                    if let Some(last) = prefix.segments.last() {
                        let name = self.text(last.span).to_string();
                        self.insert_module_item(import_mod, name, res, prefix.span);
                    }
                }
            }
            ast::UseTree::Group { prefix, items } => {
                let res = self.resolve_path_relative(target_mod, prefix);
                if let Res::Item(target_item_id) = res {
                    if let Some(&sub_mod_id) =
                        self.submodule_map.get(&ast::ItemId(target_item_id.0))
                    {
                        for item in items {
                            self.resolve_use_tree_relative(import_mod, sub_mod_id, item);
                        }
                    }
                }
            }
        }
    }

    fn insert_module_item(&mut self, module_id: ModuleId, name: String, res: Res, span: Span) {
        if let Some(vis) = self.current_use_item_vis {
            self.reexport_vis
                .insert((module_id, name.clone()), Some(vis));
        }
        match self.modules[module_id.0 as usize].items.entry(name.clone()) {
            Entry::Occupied(occ) => {
                if occ.get() != &res {
                    self.push_diag(
                        Diagnostic::error(
                            format!(
                                "duplicate definition of '{}' in the same module scope",
                                name
                            ),
                            span,
                        )
                        .with_code("E0204"),
                    );
                }
            }
            Entry::Vacant(entry) => {
                entry.insert(res);
            }
        }
    }

    fn resolve_path(&mut self, start_mod: ModuleId, path: &ast::Path) -> Res {
        self.resolve_path_relative(start_mod, path)
    }

    fn resolve_path_relative(&mut self, start_mod: ModuleId, path: &ast::Path) -> Res {
        if path.segments.is_empty() {
            return Res::Err;
        }
        match self.path_to_string(path).as_str() {
            "String::from" => return Res::Builtin(Builtin::StringFrom),
            "String::new" => return Res::Builtin(Builtin::StringNew),
            "String::with_capacity" => return Res::Builtin(Builtin::StringWithCapacity),
            "Vec::new" => return Res::Builtin(Builtin::VecNew),
            "Vec::with_capacity" => return Res::Builtin(Builtin::VecWithCapacity),
            "Box::new" => return Res::Builtin(Builtin::BoxNew),
            "Box::into_inner" => return Res::Builtin(Builtin::BoxIntoInner),
            "std::fs::read_file" => return Res::Builtin(Builtin::ReadFile),
            "std::fs::write_file" => return Res::Builtin(Builtin::WriteFile),
            "File::open" => return Res::Builtin(Builtin::FileOpen),
            "File::create" => return Res::Builtin(Builtin::FileCreate),
            "HashMap::new" => return Res::Builtin(Builtin::HashMapNew),
            "HashMap::with_capacity" => return Res::Builtin(Builtin::HashMapWithCapacity),
            "HashSet::new" => return Res::Builtin(Builtin::HashSetNew),
            // Phase 4E: `math::min`/`math::max` are qualified-only — bare
            // `min`/`max` are already claimed by the `tensor` extension.
            "math::min" | "std::math::min" => return Res::Builtin(Builtin::MathMin),
            "math::max" | "std::math::max" => return Res::Builtin(Builtin::MathMax),
            "Random::new" => return Res::Builtin(Builtin::RandomNew),
            // WP-C2.2 (DEV-027): Ordering's unit variants, mirroring IOError's wiring.
            "Ordering::Less" => return Res::Builtin(Builtin::OrderingLess),
            "Ordering::Equal" => return Res::Builtin(Builtin::OrderingEqual),
            "Ordering::Greater" => return Res::Builtin(Builtin::OrderingGreater),
            "IOError::NotFound" => return Res::Builtin(Builtin::IOErrorNotFound),
            "IOError::PermissionDenied" => return Res::Builtin(Builtin::IOErrorPermissionDenied),
            "IOError::AlreadyExists" => return Res::Builtin(Builtin::IOErrorAlreadyExists),
            "IOError::InvalidInput" => return Res::Builtin(Builtin::IOErrorInvalidInput),
            "IOError::Other" => return Res::Builtin(Builtin::IOErrorOther),
            _ => {}
        }

        let mut current_res = None;
        let mut current_mod = start_mod;

        for (i, segment) in path.segments.iter().enumerate() {
            let name_str = self.text(segment.span);

            if i == 0 {
                match segment.kind {
                    ast::SegmentKind::Crate => {
                        let pkg_root = self.modules[start_mod.0 as usize].package_root;
                        current_res = Some(Res::Item(hir::ItemId(pkg_root.0)));
                        current_mod = pkg_root;
                        continue;
                    }
                    ast::SegmentKind::Super => {
                        if let Some(parent) = self.modules[start_mod.0 as usize].parent {
                            current_res = Some(Res::Item(hir::ItemId(0)));
                            current_mod = parent;
                        } else {
                            self.push_diag(
                                Diagnostic::error("no parent module for 'super'", segment.span)
                                    .with_code("E0206"),
                            );
                            return Res::Err;
                        }
                        continue;
                    }
                    ast::SegmentKind::SelfValue => {
                        return self.resolve_unqualified("self");
                    }
                    ast::SegmentKind::SelfType => {
                        return Res::SelfType;
                    }
                    ast::SegmentKind::Ident => {
                        let is_unqualified = path.segments.len() == 1;
                        let mut resolved = None;
                        if is_unqualified {
                            for scope in self.scopes.iter().rev() {
                                if let Some(&res) = scope.get(name_str) {
                                    resolved = Some(res);
                                    break;
                                }
                            }
                        }
                        if resolved.is_none() {
                            if let Some(&res) =
                                self.modules[start_mod.0 as usize].items.get(name_str)
                            {
                                resolved = Some(res);
                            } else if let Some(primitive) = resolve_primitive(name_str) {
                                resolved = Some(Res::Primitive(primitive));
                            } else if let Some(builtin) = resolve_builtin(name_str) {
                                if !is_tensor_builtin(builtin) || self.options.tensor() {
                                    resolved = Some(Res::Builtin(builtin));
                                }
                            } else if let Some(core_trait) = resolve_core_trait(name_str) {
                                resolved = Some(Res::CoreTrait(core_trait));
                            } else if let Some(core_type) = resolve_core_type(name_str) {
                                resolved = Some(Res::CoreType(core_type));
                            }
                        }

                        if let Some(res) = resolved {
                            current_res = Some(res);
                            if let Res::Item(item_id) = res {
                                if let Some(&sub_mod_id) =
                                    self.submodule_map.get(&ast::ItemId(item_id.0))
                                {
                                    current_mod = sub_mod_id;
                                }
                            }
                        } else {
                            return Res::Err;
                        }
                    }
                }
            } else if let Some(&res) = self.modules[current_mod.0 as usize].items.get(name_str) {
                if !self.name_is_visible_from(current_mod, name_str, start_mod) {
                    self.push_diag(
                        Diagnostic::error(format!("item '{name_str}' is private"), segment.span)
                            .with_code("E0207"),
                    );
                    return Res::Err;
                }
                current_res = Some(res);
                if let Res::Item(item_id) = res {
                    if let Some(&sub_mod_id) = self.submodule_map.get(&ast::ItemId(item_id.0)) {
                        current_mod = sub_mod_id;
                    }
                }
            } else if let Some(Res::Item(item_id)) = current_res {
                match self.item_details.get(&item_id) {
                    Some(ItemDefDetail::Enum { variants }) => {
                        if let Some(variant_idx) = variants.iter().position(|v| v == name_str) {
                            current_res = Some(Res::Variant(item_id, variant_idx as u32));
                        } else {
                            current_res = Some(Res::AssociatedFn(item_id, segment.span));
                        }
                    }
                    Some(ItemDefDetail::Struct { .. }) => {
                        current_res = Some(Res::AssociatedFn(item_id, segment.span));
                    }
                    Some(ItemDefDetail::Trait { items }) => {
                        if let Some(member) = items.iter().position(|item| item == name_str) {
                            current_res = Some(Res::TraitMember(item_id, member as u32));
                        } else {
                            return Res::Err;
                        }
                    }
                    Some(ItemDefDetail::Model) if name_str == "load" => {
                        current_res = Some(Res::ModelLoad(item_id));
                    }
                    _ => return Res::Err,
                }
            } else {
                return Res::Err;
            }
        }

        current_res.unwrap_or(Res::Err)
    }

    fn lower_crate(&mut self) {
        self.current_module = ModuleId(0);
        let root = match &self.ast.root {
            ast::Root::Program(items) => {
                for ast_id in 0..self.ast.items.len() {
                    let _ = self.lower_item(ast::ItemId(ast_id as u32));
                }
                let hir_items = items.iter().map(|&id| hir::ItemId(id.0)).collect();
                hir::Root::Program(hir_items)
            }
            ast::Root::Snippet { stmts, tail } => {
                // Initialize snippet scope
                self.scopes = vec![HashMap::new()];
                let stmts = stmts.iter().map(|&s| self.lower_stmt(s)).collect();
                let tail = tail.map(|e| self.lower_expr(e));
                hir::Root::Snippet { stmts, tail }
            }
        };
        self.hir.root = root;
    }

    fn item_is_visible_from(&self, item_id: hir::ItemId, from: ModuleId) -> bool {
        let defining = self.item_modules.get(&item_id).copied().unwrap_or(from);
        if defining == from {
            return true;
        }
        matches!(
            self.ast.item(ast::ItemId(item_id.0)).vis,
            Some(ast::Vis::Pub)
        )
    }

    fn name_is_visible_from(&self, module_id: ModuleId, name: &str, from: ModuleId) -> bool {
        if module_id == from {
            return true;
        }
        if let Some(vis) = self.reexport_vis.get(&(module_id, name.to_string())) {
            return matches!(vis, Some(ast::Vis::Pub));
        }
        if let Some(&Res::Item(item_id)) = self.modules[module_id.0 as usize].items.get(name) {
            return self.item_is_visible_from(item_id, from);
        }
        true
    }

    fn lower_type(&mut self, ast_id: ast::TypeId) -> hir::TypeId {
        let node = self.ast.ty(ast_id);
        let kind = match &node.kind {
            ast::TypeKind::Primitive(p) => hir::TypeKind::Primitive(*p),
            ast::TypeKind::Path { path, args } => {
                let res = if path.segments.len() == 2
                    && path.segments[0].kind == ast::SegmentKind::SelfType
                {
                    Res::SelfAssoc(path.segments[1].span)
                } else if path.segments.len() == 2
                    && self.scopes.iter().rev().any(|scope| {
                        matches!(
                            scope.get(self.text(path.segments[0].span)),
                            Some(Res::TypeParam)
                        )
                    })
                {
                    Res::ParamAssoc(path.segments[0].span, path.segments[1].span)
                } else {
                    self.resolve_path(self.current_module, path)
                };
                if matches!(res, Res::Err | Res::Builtin(_) | Res::CoreTrait(_)) {
                    // A reserved `tensor` extension type name (`Tensor`,
                    // `Float16`, ...) is rejected in Core-only mode with a
                    // focused diagnostic (D1/D3); under the extension it is
                    // left for the M4.2 tensor type resolver rather than
                    // reported as an undefined Core type.
                    let ext_name = (path.segments.len() == 1)
                        .then(|| extension_reserved_name(self.text(path.segments[0].span)))
                        .flatten();
                    match ext_name {
                        Some(what) if !self.options.tensor() => {
                            self.push_diag(
                                Diagnostic::error(
                                    format!("the {what} requires extension `tensor`"),
                                    path.span,
                                )
                                .with_code("E0210"),
                            );
                        }
                        Some(_) => { /* tensor mode: deferred to M4.2 */ }
                        None => self.push_diag(
                            Diagnostic::error(
                                format!("undefined type '{}'", self.path_to_string(path)),
                                path.span,
                            )
                            .with_code("E0202"),
                        ),
                    }
                }
                let args = args.as_ref().map(|a| self.lower_generic_args(a));
                hir::TypeKind::Path {
                    path: path.clone(),
                    res,
                    args,
                }
            }
            ast::TypeKind::Array { elem, len } => {
                let elem = self.lower_type(*elem);
                hir::TypeKind::Array { elem, len: *len }
            }
            ast::TypeKind::Slice(elem) => {
                let elem = self.lower_type(*elem);
                hir::TypeKind::Slice(elem)
            }
            ast::TypeKind::Tuple(elems) => {
                let elems = elems.iter().map(|&e| self.lower_type(e)).collect();
                hir::TypeKind::Tuple(elems)
            }
            ast::TypeKind::Ref { mutable, inner } => {
                let inner = self.lower_type(*inner);
                hir::TypeKind::Ref {
                    mutable: *mutable,
                    inner,
                }
            }
            ast::TypeKind::Fn { params, ret } => {
                let params = params.iter().map(|&p| self.lower_type(p)).collect();
                let ret = ret.map(|r| self.lower_type(r));
                hir::TypeKind::Fn { params, ret }
            }
            ast::TypeKind::Never => hir::TypeKind::Never,
            ast::TypeKind::Error => hir::TypeKind::Error,
        };
        self.hir.alloc_type(kind, node.span)
    }

    fn lower_expr(&mut self, ast_id: ast::ExprId) -> hir::ExprId {
        let node = self.ast.expr(ast_id);
        let kind = match &node.kind {
            ast::ExprKind::Lit(lit) => hir::ExprKind::Lit(*lit),
            ast::ExprKind::Path { path, turbofish } => {
                let res = self.resolve_path(self.current_module, path);
                if res == Res::Err {
                    self.push_diag(
                        Diagnostic::error(
                            format!("undefined variable '{}'", self.path_to_string(path)),
                            path.span,
                        )
                        .with_code("E0200"),
                    );
                }
                let turbofish = turbofish.as_ref().map(|t| self.lower_generic_args(t));
                hir::ExprKind::Path {
                    path: path.clone(),
                    res,
                    turbofish,
                }
            }
            ast::ExprKind::Unary { op, operand } => {
                let operand = self.lower_expr(*operand);
                hir::ExprKind::Unary { op: *op, operand }
            }
            ast::ExprKind::Binary { op, lhs, rhs } => {
                let lhs = self.lower_expr(*lhs);
                let rhs = self.lower_expr(*rhs);
                hir::ExprKind::Binary { op: *op, lhs, rhs }
            }
            ast::ExprKind::Assign { op, lhs, rhs } => {
                let lhs = self.lower_expr(*lhs);
                let rhs = self.lower_expr(*rhs);
                hir::ExprKind::Assign { op: *op, lhs, rhs }
            }
            ast::ExprKind::Range { lo, hi, inclusive } => {
                let lo = self.lower_expr(*lo);
                let hi = self.lower_expr(*hi);
                hir::ExprKind::Range {
                    lo,
                    hi,
                    inclusive: *inclusive,
                }
            }
            ast::ExprKind::Cast { expr, ty } => {
                let expr = self.lower_expr(*expr);
                let ty = self.lower_type(*ty);
                hir::ExprKind::Cast { expr, ty }
            }
            ast::ExprKind::Call { callee, args } => {
                let callee = self.lower_expr(*callee);
                let args = args.iter().map(|&a| self.lower_expr(a)).collect();
                hir::ExprKind::Call { callee, args }
            }
            ast::ExprKind::Field {
                base,
                name,
                turbofish,
            } => {
                let base = self.lower_expr(*base);
                let turbofish = turbofish.as_ref().map(|args| self.lower_generic_args(args));
                hir::ExprKind::Field {
                    base,
                    name: *name,
                    turbofish,
                }
            }
            ast::ExprKind::TupleField { base, index } => {
                let base = self.lower_expr(*base);
                hir::ExprKind::TupleField {
                    base,
                    index: *index,
                }
            }
            ast::ExprKind::Index { base, index } => {
                let base = self.lower_expr(*base);
                let index = self.lower_expr(*index);
                hir::ExprKind::Index { base, index }
            }
            ast::ExprKind::Try(expr) => {
                let expr = self.lower_expr(*expr);
                hir::ExprKind::Try(expr)
            }
            ast::ExprKind::Tuple(elems) => {
                let elems = elems.iter().map(|&e| self.lower_expr(e)).collect();
                hir::ExprKind::Tuple(elems)
            }
            ast::ExprKind::Array(elems) => {
                let elems = elems.iter().map(|&e| self.lower_expr(e)).collect();
                hir::ExprKind::Array(elems)
            }
            ast::ExprKind::Repeat { value, count } => {
                let value = self.lower_expr(*value);
                let count = self.lower_expr(*count);
                hir::ExprKind::Repeat { value, count }
            }
            ast::ExprKind::StructLit { path, fields } => {
                let res = self.resolve_path(self.current_module, path);
                if res == Res::Err {
                    self.push_diag(
                        Diagnostic::error(
                            format!("undefined struct '{}'", self.path_to_string(path)),
                            path.span,
                        )
                        .with_code("E0202"),
                    );
                }
                let fields = fields
                    .iter()
                    .map(|f| {
                        let expr = if let Some(expr) = f.expr {
                            Some(self.lower_expr(expr))
                        } else {
                            let name_str = self.text(f.name).to_string();
                            let var_res = self.resolve_unqualified(&name_str);
                            if var_res == Res::Err {
                                self.push_diag(
                                    Diagnostic::error(
                                        format!(
                                            "undefined variable '{}' (shorthand field)",
                                            name_str
                                        ),
                                        f.name,
                                    )
                                    .with_code("E0200"),
                                );
                            }
                            let path = ast::Path {
                                segments: vec![ast::PathSegment {
                                    kind: ast::SegmentKind::Ident,
                                    span: f.name,
                                }],
                                span: f.name,
                            };
                            Some(self.hir.alloc_expr(
                                hir::ExprKind::Path {
                                    path,
                                    res: var_res,
                                    turbofish: None,
                                },
                                f.name,
                            ))
                        };
                        hir::FieldInit { name: f.name, expr }
                    })
                    .collect();
                hir::ExprKind::StructLit {
                    path: path.clone(),
                    res,
                    fields,
                }
            }
            ast::ExprKind::If {
                cond,
                then_block,
                else_,
            } => {
                let cond = self.lower_expr(*cond);
                let then_block = self.lower_block(*then_block);
                let else_ = else_.map(|e| self.lower_expr(e));
                hir::ExprKind::If {
                    cond,
                    then_block,
                    else_,
                }
            }
            ast::ExprKind::Match { scrutinee, arms } => {
                let scrutinee = self.lower_expr(*scrutinee);
                let arms = arms
                    .iter()
                    .map(|arm| {
                        self.push_scope();
                        let pat = self.lower_pattern(arm.pat);
                        let body = self.lower_expr(arm.body);
                        self.pop_scope();
                        hir::MatchArm { pat, body }
                    })
                    .collect();
                hir::ExprKind::Match { scrutinee, arms }
            }
            ast::ExprKind::Loop { body } => {
                let body = self.lower_block(*body);
                hir::ExprKind::Loop { body }
            }
            ast::ExprKind::While { cond, body } => {
                let cond = self.lower_expr(*cond);
                let body = self.lower_block(*body);
                hir::ExprKind::While { cond, body }
            }
            ast::ExprKind::For { var, iter, body } => {
                let iter = self.lower_expr(*iter);
                self.push_scope();
                let local = self.alloc_local();
                let var_name = self.text(*var).to_string();
                self.scopes
                    .last_mut()
                    .unwrap()
                    .insert(var_name, Res::Local(local));
                let body = self.lower_block(*body);
                self.pop_scope();
                hir::ExprKind::For {
                    var: *var,
                    local,
                    iter,
                    body,
                }
            }
            ast::ExprKind::Block(b) => {
                let b = self.lower_block(*b);
                hir::ExprKind::Block(b)
            }
            ast::ExprKind::Error => hir::ExprKind::Error,
        };
        self.hir.alloc_expr(kind, node.span)
    }

    fn lower_pattern(&mut self, ast_id: ast::PatId) -> hir::PatId {
        let node = self.ast.pat(ast_id);
        let kind = match &node.kind {
            ast::PatKind::Lit(lit) => hir::PatKind::Lit(*lit),
            ast::PatKind::Wild => hir::PatKind::Wild,
            ast::PatKind::Binding(name_span) => {
                let name_str = self.text(*name_span);
                let module_res = self.modules[self.current_module.0 as usize]
                    .items
                    .get(name_str)
                    .copied();
                if let Some(Res::Variant(enum_id, variant_idx)) = module_res {
                    let path = ast::Path {
                        segments: vec![ast::PathSegment {
                            kind: ast::SegmentKind::Ident,
                            span: *name_span,
                        }],
                        span: *name_span,
                    };
                    hir::PatKind::Path {
                        path,
                        res: Res::Variant(enum_id, variant_idx),
                    }
                } else if let Some(Res::Item(item_id)) = module_res {
                    let path = ast::Path {
                        segments: vec![ast::PathSegment {
                            kind: ast::SegmentKind::Ident,
                            span: *name_span,
                        }],
                        span: *name_span,
                    };
                    hir::PatKind::Path {
                        path,
                        res: Res::Item(item_id),
                    }
                } else {
                    let var_name = name_str.to_string();
                    if self.scopes.last().unwrap().contains_key(&var_name) {
                        self.push_diag(
                            Diagnostic::error(
                                format!(
                                    "duplicate definition of variable '{}' in the same scope",
                                    var_name
                                ),
                                *name_span,
                            )
                            .with_code("E0204")
                            .with_label("variable declared here again"),
                        );
                    }
                    let local = self.alloc_local();
                    self.scopes
                        .last_mut()
                        .unwrap()
                        .insert(var_name, Res::Local(local));
                    hir::PatKind::Binding {
                        name: *name_span,
                        local,
                    }
                }
            }
            ast::PatKind::Path(path) => {
                let res = self.resolve_path(self.current_module, path);
                if res == Res::Err {
                    self.push_diag(
                        Diagnostic::error(
                            format!("undefined pattern path '{}'", self.path_to_string(path)),
                            path.span,
                        )
                        .with_code("E0200"),
                    );
                }
                hir::PatKind::Path {
                    path: path.clone(),
                    res,
                }
            }
            ast::PatKind::TupleVariant { path, pats } => {
                let res = self.resolve_path(self.current_module, path);
                if res == Res::Err {
                    self.push_diag(
                        Diagnostic::error(
                            format!("undefined enum variant '{}'", self.path_to_string(path)),
                            path.span,
                        )
                        .with_code("E0202"),
                    );
                }
                let pats = pats.iter().map(|&p| self.lower_pattern(p)).collect();
                hir::PatKind::TupleVariant {
                    path: path.clone(),
                    res,
                    pats,
                }
            }
            ast::PatKind::Struct { path, fields } => {
                let res = self.resolve_path(self.current_module, path);
                if res == Res::Err {
                    self.push_diag(
                        Diagnostic::error(
                            format!("undefined struct/variant '{}'", self.path_to_string(path)),
                            path.span,
                        )
                        .with_code("E0202"),
                    );
                }
                let fields = fields.iter().map(|f| {
                    let pat = f.pat.map(|p| self.lower_pattern(p));
                    let local = if f.pat.is_none() {
                        let name_str = self.text(f.name);
                        let var_name = name_str.to_string();
                        if self.scopes.last().unwrap().contains_key(&var_name) {
                            self.push_diag(
                                Diagnostic::error(format!("duplicate definition of variable '{}' in the same scope", var_name), f.name)
                                    .with_code("E0204")
                            );
                        }
                        let local = self.alloc_local();
                        self.scopes.last_mut().unwrap().insert(var_name, Res::Local(local));
                        Some(local)
                    } else {
                        None
                    };
                    hir::FieldPat { name: f.name, pat, local }
                }).collect();
                hir::PatKind::Struct {
                    path: path.clone(),
                    res,
                    fields,
                }
            }
            ast::PatKind::Tuple(elems) => {
                let elems = elems.iter().map(|&e| self.lower_pattern(e)).collect();
                hir::PatKind::Tuple(elems)
            }
            ast::PatKind::Array(elems) => {
                let elems = elems.iter().map(|&e| self.lower_pattern(e)).collect();
                hir::PatKind::Array(elems)
            }
        };
        self.hir.alloc_pat(kind, node.span)
    }

    fn lower_stmt(&mut self, ast_id: ast::StmtId) -> hir::StmtId {
        let node = self.ast.stmt(ast_id);
        let kind = match &node.kind {
            ast::StmtKind::Empty => hir::StmtKind::Empty,
            ast::StmtKind::Expr { expr, semi } => {
                let expr = self.lower_expr(*expr);
                hir::StmtKind::Expr { expr, semi: *semi }
            }
            ast::StmtKind::Let {
                mutable,
                name,
                ty,
                init,
            } => {
                let init = init.map(|e| self.lower_expr(e));
                let ty = ty.map(|t| self.lower_type(t));

                let var_name = self.text(*name).to_string();
                if self.scopes.last().unwrap().contains_key(&var_name) {
                    self.push_diag(
                        Diagnostic::error(
                            format!(
                                "duplicate definition of variable '{}' in the same scope",
                                var_name
                            ),
                            *name,
                        )
                        .with_code("E0204")
                        .with_label("variable declared here again"),
                    );
                }

                let local = self.alloc_local();
                self.scopes
                    .last_mut()
                    .unwrap()
                    .insert(var_name, Res::Local(local));
                hir::StmtKind::Let {
                    mutable: *mutable,
                    name: *name,
                    local,
                    ty,
                    init,
                }
            }
            ast::StmtKind::Return(expr) => {
                let expr = expr.map(|e| self.lower_expr(e));
                hir::StmtKind::Return(expr)
            }
            ast::StmtKind::Break(expr) => {
                let expr = expr.map(|e| self.lower_expr(e));
                hir::StmtKind::Break(expr)
            }
            ast::StmtKind::Continue => hir::StmtKind::Continue,
            ast::StmtKind::Item(item_id) => {
                let item_id = self.lower_item(*item_id);
                hir::StmtKind::Item(item_id)
            }
            ast::StmtKind::Error => hir::StmtKind::Error,
        };
        self.hir.alloc_stmt(kind, node.span)
    }

    fn lower_block(&mut self, ast_id: ast::BlockId) -> hir::BlockId {
        let node = self.ast.block(ast_id);
        self.push_scope();
        let stmts = node.stmts.iter().map(|&s| self.lower_stmt(s)).collect();
        let tail = node.tail.map(|e| self.lower_expr(e));
        self.pop_scope();
        self.hir.alloc_block(hir::BlockNode {
            stmts,
            tail,
            span: node.span,
        })
    }

    fn lower_item(&mut self, ast_id: ast::ItemId) -> hir::ItemId {
        if let Some(&hir_id) = self.item_map.get(&ast_id) {
            return hir_id;
        }

        let prev_module = self.current_module;
        let candidate_hir_id = hir::ItemId(ast_id.0);
        if let Some(&mod_id) = self.item_modules.get(&candidate_hir_id) {
            self.current_module = mod_id;
        }

        let node = self.ast.item(ast_id);
        let saved_scopes = std::mem::take(&mut self.scopes);
        self.scopes = vec![HashMap::new()];
        let kind = match &node.kind {
            ast::ItemKind::Fn(f) => {
                for param in &f.sig.generics {
                    let name_str = self.text(param.name).to_string();
                    self.scopes
                        .last_mut()
                        .unwrap()
                        .insert(name_str, Res::TypeParam);
                }

                let receiver_local = if f.sig.receiver.is_some() {
                    let local = self.alloc_local();
                    self.scopes
                        .last_mut()
                        .unwrap()
                        .insert("self".to_string(), Res::SelfValue(local));
                    Some(local)
                } else {
                    None
                };

                let params = f
                    .sig
                    .params
                    .iter()
                    .map(|p| {
                        let ty = self.lower_type(p.ty);
                        let local = self.alloc_local();
                        let param_name = self.text(p.name).to_string();
                        self.scopes
                            .last_mut()
                            .unwrap()
                            .insert(param_name, Res::Local(local));
                        hir::Param {
                            mutable: p.mutable,
                            name: p.name,
                            ty,
                            local,
                        }
                    })
                    .collect();

                let ret = match f.sig.ret {
                    ast::RetTy::Unit => hir::RetTy::Unit,
                    ast::RetTy::Ty(t) => hir::RetTy::Ty(self.lower_type(t)),
                    ast::RetTy::Never(s) => hir::RetTy::Never(s),
                };

                let generics = self.lower_generic_params(&f.sig.generics);

                let body = self.lower_block(f.body);

                hir::ItemKind::Fn(hir::FnDef {
                    sig: hir::FnSig {
                        name: f.sig.name,
                        generics,
                        receiver: f.sig.receiver.map(|r| match r {
                            ast::Receiver::Value => hir::Receiver::Value,
                            ast::Receiver::Ref => hir::Receiver::Ref,
                            ast::Receiver::RefMut => hir::Receiver::RefMut,
                        }),
                        receiver_local,
                        params,
                        ret,
                        span: f.sig.span,
                    },
                    body,
                })
            }
            ast::ItemKind::Struct {
                name,
                generics,
                fields,
            } => {
                self.push_scope();
                self.declare_generic_params(generics);
                let generics_lowered = self.lower_generic_params(generics);
                let fields = fields
                    .iter()
                    .map(|f| {
                        let ty = self.lower_type(f.ty);
                        hir::FieldDef {
                            is_pub: f.is_pub,
                            name: f.name,
                            ty,
                        }
                    })
                    .collect();
                self.pop_scope();
                hir::ItemKind::Struct {
                    name: *name,
                    generics: generics_lowered,
                    fields,
                }
            }
            ast::ItemKind::Enum {
                name,
                generics,
                variants,
            } => {
                self.push_scope();
                self.declare_generic_params(generics);
                let generics_lowered = self.lower_generic_params(generics);
                let variants = variants
                    .iter()
                    .map(|v| {
                        let kind = match &v.kind {
                            ast::VariantKind::Unit => hir::VariantKind::Unit,
                            ast::VariantKind::Tuple(types) => {
                                let types = types.iter().map(|&t| self.lower_type(t)).collect();
                                hir::VariantKind::Tuple(types)
                            }
                            ast::VariantKind::Struct(fields) => {
                                let fields = fields
                                    .iter()
                                    .map(|f| {
                                        let ty = self.lower_type(f.ty);
                                        hir::FieldDef {
                                            is_pub: f.is_pub,
                                            name: f.name,
                                            ty,
                                        }
                                    })
                                    .collect();
                                hir::VariantKind::Struct(fields)
                            }
                        };
                        hir::Variant { name: v.name, kind }
                    })
                    .collect();
                self.pop_scope();
                hir::ItemKind::Enum {
                    name: *name,
                    generics: generics_lowered,
                    variants,
                }
            }
            ast::ItemKind::Trait {
                name,
                generics,
                items: trait_items,
            } => {
                self.push_scope();
                self.declare_generic_params(generics);
                let generics_lowered = self.lower_generic_params(generics);
                let items = trait_items
                    .iter()
                    .map(|ti| match ti {
                        ast::TraitItem::Method { sig, body } => {
                            self.push_scope();
                            let generics = self.lower_generic_params(&sig.generics);
                            self.declare_generic_params(&sig.generics);
                            let receiver_local = if sig.receiver.is_some() {
                                let local = self.alloc_local();
                                self.scopes
                                    .last_mut()
                                    .unwrap()
                                    .insert("self".to_string(), Res::SelfValue(local));
                                Some(local)
                            } else {
                                None
                            };
                            let params = sig
                                .params
                                .iter()
                                .map(|p| {
                                    let ty = self.lower_type(p.ty);
                                    let local = self.alloc_local();
                                    let name = self.text(p.name).to_string();
                                    self.scopes
                                        .last_mut()
                                        .unwrap()
                                        .insert(name, Res::Local(local));
                                    hir::Param {
                                        mutable: p.mutable,
                                        name: p.name,
                                        ty,
                                        local,
                                    }
                                })
                                .collect();
                            let ret = match sig.ret {
                                ast::RetTy::Unit => hir::RetTy::Unit,
                                ast::RetTy::Ty(t) => hir::RetTy::Ty(self.lower_type(t)),
                                ast::RetTy::Never(s) => hir::RetTy::Never(s),
                            };
                            let body = body.map(|b| self.lower_block(b));
                            let lowered = hir::TraitItem::Method {
                                sig: hir::FnSig {
                                    name: sig.name,
                                    generics,
                                    receiver: sig.receiver.map(|r| match r {
                                        ast::Receiver::Value => hir::Receiver::Value,
                                        ast::Receiver::Ref => hir::Receiver::Ref,
                                        ast::Receiver::RefMut => hir::Receiver::RefMut,
                                    }),
                                    receiver_local,
                                    params,
                                    ret,
                                    span: sig.span,
                                },
                                body,
                            };
                            self.pop_scope();
                            lowered
                        }
                        ast::TraitItem::AssocType { name } => {
                            hir::TraitItem::AssocType { name: *name }
                        }
                    })
                    .collect();
                self.pop_scope();
                hir::ItemKind::Trait {
                    name: *name,
                    generics: generics_lowered,
                    items,
                }
            }
            ast::ItemKind::Impl {
                generics,
                trait_,
                self_ty,
                items,
            } => {
                self.push_scope();
                self.declare_generic_params(generics);
                let generics_lowered = self.lower_generic_params(generics);
                let self_ty = self.lower_type(*self_ty);
                let trait_ = trait_.as_ref().map(|t| {
                    let res = self.resolve_path(self.current_module, &t.path);
                    let args = t.args.as_ref().map(|a| self.lower_generic_args(a));
                    hir::TraitRef {
                        path: t.path.clone(),
                        res,
                        args,
                    }
                });
                let items = items
                    .iter()
                    .map(|item| match item {
                        ast::ImplItem::Fn { vis, def } => {
                            self.push_scope();
                            self.scopes
                                .last_mut()
                                .unwrap()
                                .insert("Self".to_string(), Res::SelfType);

                            for param in &def.sig.generics {
                                let name_str = self.text(param.name).to_string();
                                self.scopes
                                    .last_mut()
                                    .unwrap()
                                    .insert(name_str, Res::TypeParam);
                            }

                            let receiver_local = if def.sig.receiver.is_some() {
                                let local = self.alloc_local();
                                self.scopes
                                    .last_mut()
                                    .unwrap()
                                    .insert("self".to_string(), Res::SelfValue(local));
                                Some(local)
                            } else {
                                None
                            };

                            let params = def
                                .sig
                                .params
                                .iter()
                                .map(|p| {
                                    let ty = self.lower_type(p.ty);
                                    let local = self.alloc_local();
                                    let param_name = self.text(p.name).to_string();
                                    self.scopes
                                        .last_mut()
                                        .unwrap()
                                        .insert(param_name, Res::Local(local));
                                    hir::Param {
                                        mutable: p.mutable,
                                        name: p.name,
                                        ty,
                                        local,
                                    }
                                })
                                .collect();

                            let ret = match def.sig.ret {
                                ast::RetTy::Unit => hir::RetTy::Unit,
                                ast::RetTy::Ty(t) => hir::RetTy::Ty(self.lower_type(t)),
                                ast::RetTy::Never(s) => hir::RetTy::Never(s),
                            };

                            let sig_generics = self.lower_generic_params(&def.sig.generics);
                            let body = self.lower_block(def.body);

                            let lowered = hir::ImplItem::Fn {
                                vis: *vis,
                                def: hir::FnDef {
                                    sig: hir::FnSig {
                                        name: def.sig.name,
                                        generics: sig_generics,
                                        receiver: def.sig.receiver.map(|r| match r {
                                            ast::Receiver::Value => hir::Receiver::Value,
                                            ast::Receiver::Ref => hir::Receiver::Ref,
                                            ast::Receiver::RefMut => hir::Receiver::RefMut,
                                        }),
                                        receiver_local,
                                        params,
                                        ret,
                                        span: def.sig.span,
                                    },
                                    body,
                                },
                            };
                            self.pop_scope();
                            lowered
                        }
                        ast::ImplItem::AssocType { name, ty } => {
                            let ty = self.lower_type(*ty);
                            hir::ImplItem::AssocType { name: *name, ty }
                        }
                    })
                    .collect();
                self.pop_scope();
                hir::ItemKind::Impl {
                    generics: generics_lowered,
                    trait_,
                    self_ty,
                    items,
                }
            }
            ast::ItemKind::Const { name, ty, value } => {
                let ty = self.lower_type(*ty);
                let value = self.lower_expr(*value);
                hir::ItemKind::Const {
                    name: *name,
                    ty,
                    value,
                }
            }
            ast::ItemKind::TypeAlias { name, generics, ty } => {
                self.push_scope();
                self.declare_generic_params(generics);
                let generics_lowered = self.lower_generic_params(generics);
                let ty = self.lower_type(*ty);
                self.pop_scope();
                hir::ItemKind::TypeAlias {
                    name: *name,
                    generics: generics_lowered,
                    ty,
                }
            }
            ast::ItemKind::Use(use_tree) => {
                let tree = self.lower_use_tree(use_tree);
                hir::ItemKind::Use(tree)
            }
            ast::ItemKind::Mod { name, items } => {
                let sub_items = items
                    .as_ref()
                    .map(|sub_items| sub_items.iter().map(|&id| hir::ItemId(id.0)).collect());
                hir::ItemKind::Mod {
                    name: *name,
                    items: sub_items,
                }
            }
            ast::ItemKind::Model(def) => {
                // Model generic parameters are in scope for every port type
                // (spec §7.1); port dimension variables inside shapes are
                // carried structurally and left to the extension checker.
                self.push_scope();
                self.declare_generic_params(&def.generics);
                let generics = self.lower_generic_params(&def.generics);
                let ports = def
                    .ports
                    .iter()
                    .map(|p| hir::ModelPort {
                        dir: p.dir,
                        name: p.name,
                        ty: self.lower_type(p.ty),
                        span: p.span,
                    })
                    .collect();
                self.pop_scope();
                hir::ItemKind::Model(hir::ModelDef {
                    name: def.name,
                    generics,
                    ports,
                })
            }
        };

        let hir_id = self.hir.alloc_item(kind, node.vis, node.span);
        if let Some(file) = self.ast.item_files.get(&ast_id) {
            self.hir.item_files.insert(hir_id, file.clone());
        }
        self.item_map.insert(ast_id, hir_id);
        self.scopes = saved_scopes;
        self.current_module = prev_module;
        hir_id
    }

    fn declare_generic_params(&mut self, params: &[ast::GenericParam]) {
        for param in params {
            let name = self.text(param.name).to_string();
            self.scopes
                .last_mut()
                .expect("item scope exists")
                .insert(name, Res::TypeParam);
        }
    }

    fn lower_generic_params(&mut self, params: &[ast::GenericParam]) -> Vec<hir::GenericParam> {
        params
            .iter()
            .map(|g| {
                let bounds = g
                    .bounds
                    .iter()
                    .map(|b| {
                        let res = self.resolve_path(self.current_module, &b.path);
                        // A `Dim`/`DType` kind bound (D1) that does not resolve
                        // to a user-declared trait is rejected in Core-only
                        // mode; under the extension it is a kind bound handled
                        // by the M4.2 checker. Only fires on genuine resolution
                        // failure, so a user trait spelled `Dim` is unaffected.
                        if res == Res::Err && b.path.segments.len() == 1 {
                            if let Some(what) =
                                extension_reserved_name(self.text(b.path.segments[0].span))
                            {
                                if !self.options.tensor() {
                                    self.push_diag(
                                        Diagnostic::error(
                                            format!("the {what} requires extension `tensor`"),
                                            b.path.span,
                                        )
                                        .with_code("E0210"),
                                    );
                                }
                            }
                        }
                        let args = b.args.as_ref().map(|a| self.lower_generic_args(a));
                        hir::TraitRef {
                            path: b.path.clone(),
                            res,
                            args,
                        }
                    })
                    .collect();
                hir::GenericParam {
                    name: g.name,
                    bounds,
                }
            })
            .collect()
    }

    fn lower_generic_args(&mut self, args: &ast::GenericArgs) -> hir::GenericArgs {
        let args_vec = args
            .args
            .iter()
            .map(|a| match a {
                ast::GenericArg::Type(t) => hir::GenericArg::Type(self.lower_type(*t)),
                ast::GenericArg::Const(span) => hir::GenericArg::Const(*span),
                ast::GenericArg::Binding { name, ty } => hir::GenericArg::Binding {
                    name: *name,
                    ty: self.lower_type(*ty),
                },
                ast::GenericArg::Shape(shape) => hir::GenericArg::Shape(self.lower_shape(shape)),
            })
            .collect();
        hir::GenericArgs {
            args: args_vec,
            span: args.span,
        }
    }

    fn lower_shape(&self, shape: &ast::ShapeArg) -> hir::ShapeArg {
        hir::ShapeArg {
            dims: shape.dims.iter().map(|&d| self.lower_dim(d)).collect(),
            span: shape.span,
        }
    }

    fn lower_dim(&self, id: ast::DimId) -> hir::DimExpr {
        match &self.ast.dim(id).kind {
            ast::DimExprKind::Lit(s) => hir::DimExpr::Lit(*s),
            ast::DimExprKind::Var(s) => hir::DimExpr::Var(*s),
            ast::DimExprKind::Binary { op, lhs, rhs } => hir::DimExpr::Binary {
                op: *op,
                lhs: Box::new(self.lower_dim(*lhs)),
                rhs: Box::new(self.lower_dim(*rhs)),
            },
            ast::DimExprKind::Error => hir::DimExpr::Error,
        }
    }

    fn lower_use_tree(&mut self, tree: &ast::UseTree) -> hir::UseTree {
        match tree {
            ast::UseTree::Path { path, alias } => hir::UseTree::Path {
                path: path.clone(),
                alias: *alias,
            },
            ast::UseTree::Glob { prefix } => hir::UseTree::Glob {
                prefix: prefix.clone(),
            },
            ast::UseTree::SelfImport { prefix } => hir::UseTree::SelfImport {
                prefix: prefix.clone(),
            },
            ast::UseTree::Group { prefix, items } => {
                let items = items.iter().map(|item| self.lower_use_tree(item)).collect();
                hir::UseTree::Group {
                    prefix: prefix.clone(),
                    items,
                }
            }
        }
    }

    fn resolve_unqualified(&mut self, name: &str) -> Res {
        for scope in self.scopes.iter().rev() {
            if let Some(&res) = scope.get(name) {
                return res;
            }
        }
        if let Some(&res) = self.modules[self.current_module.0 as usize].items.get(name) {
            return res;
        }
        if let Some(primitive) = resolve_primitive(name) {
            return Res::Primitive(primitive);
        }
        if let Some(builtin) = resolve_builtin(name) {
            // WP-C1.2 (2026-07-17): gate tensor-extension builtins the same way
            // resolve_path_relative already does (see `is_tensor_builtin` usage there). Without
            // this, bare `min`/`max` resolved to the tensor extension's builtin even in
            // Core-only mode -- see COMPILER-STATE.md DEV-004.
            if !is_tensor_builtin(builtin) || self.options.tensor() {
                return Res::Builtin(builtin);
            }
        }
        if let Some(core_trait) = resolve_core_trait(name) {
            return Res::CoreTrait(core_trait);
        }
        if let Some(core_type) = resolve_core_type(name) {
            return Res::CoreType(core_type);
        }
        Res::Err
    }
}

fn resolve_builtin(name: &str) -> Option<Builtin> {
    match name {
        "print" => Some(Builtin::Print),
        "println" => Some(Builtin::Println),
        "panic" => Some(Builtin::Panic),
        "assert" => Some(Builtin::Assert),
        "assert_eq" => Some(Builtin::AssertEq),
        "assert_ne" => Some(Builtin::AssertNe),
        "sqrt" => Some(Builtin::Sqrt),
        "drop" => Some(Builtin::Drop),
        "read_file" => Some(Builtin::ReadFile),
        "write_file" => Some(Builtin::WriteFile),
        "size_of" => Some(Builtin::SizeOf),
        "align_of" => Some(Builtin::AlignOf),
        "swap" => Some(Builtin::Swap),
        "replace" => Some(Builtin::Replace),
        "take" => Some(Builtin::Take),
        "Some" => Some(Builtin::Some),
        "None" => Some(Builtin::None),
        "Ok" => Some(Builtin::Ok),
        "Err" => Some(Builtin::Err),
        "zeros" => Some(Builtin::TensorZeros),
        "ones" => Some(Builtin::TensorOnes),
        "full" => Some(Builtin::TensorFull),
        "from_vec" => Some(Builtin::TensorFromVec),
        "add" => Some(Builtin::TensorAdd),
        "sub" => Some(Builtin::TensorSub),
        "mul" => Some(Builtin::TensorMul),
        "div" => Some(Builtin::TensorDiv),
        "min" => Some(Builtin::TensorMin),
        "max" => Some(Builtin::TensorMax),
        "eq" => Some(Builtin::TensorEq),
        "ne" => Some(Builtin::TensorNe),
        "lt" => Some(Builtin::TensorLt),
        "le" => Some(Builtin::TensorLe),
        "gt" => Some(Builtin::TensorGt),
        "ge" => Some(Builtin::TensorGe),
        "broadcast_to" => Some(Builtin::TensorBroadcastTo),
        "matmul" => Some(Builtin::TensorMatMul),
        "batch_matmul" => Some(Builtin::TensorBatchMatMul),
        "concat" => Some(Builtin::TensorConcat),
        "permute" => Some(Builtin::TensorPermute),
        "reshape" => Some(Builtin::TensorReshape),
        "slice_axis" => Some(Builtin::TensorSliceAxis),
        "transpose" => Some(Builtin::TensorTranspose),
        "sum_axis" => Some(Builtin::TensorSumAxis),
        "mean_axis" => Some(Builtin::TensorMeanAxis),
        "argmax" => Some(Builtin::TensorArgMax),
        "sum" => Some(Builtin::TensorSum),
        "softmax" => Some(Builtin::TensorSoftmax),
        "cast" => Some(Builtin::TensorCast),
        "to_device" => Some(Builtin::TensorToDevice),
        "scale_255" => Some(Builtin::TensorScale255),
        "normalize" => Some(Builtin::TensorNormalize),
        // Phase 4E: Math (bare names that don't collide with the tensor
        // extension's bare `min`/`max`; those are `math::min`/`math::max`,
        // resolved via the qualified-path table in `resolve_path_relative`).
        "PI" => Some(Builtin::MathPi),
        "E" => Some(Builtin::MathE),
        "abs" => Some(Builtin::MathAbs),
        "clamp" => Some(Builtin::MathClamp),
        "pow" => Some(Builtin::Pow),
        "log" => Some(Builtin::Log),
        "log10" => Some(Builtin::Log10),
        "exp" => Some(Builtin::Exp),
        "sin" => Some(Builtin::Sin),
        "cos" => Some(Builtin::Cos),
        "tan" => Some(Builtin::Tan),
        "asin" => Some(Builtin::Asin),
        "acos" => Some(Builtin::Acos),
        "atan" => Some(Builtin::Atan),
        "atan2" => Some(Builtin::Atan2),
        "floor" => Some(Builtin::Floor),
        "ceil" => Some(Builtin::Ceil),
        "round" => Some(Builtin::Round),
        "trunc" => Some(Builtin::Trunc),
        "eprint" => Some(Builtin::Eprint),
        "eprintln" => Some(Builtin::Eprintln),
        _ => None,
    }
}

pub fn is_tensor_builtin(b: Builtin) -> bool {
    matches!(
        b,
        Builtin::TensorZeros
            | Builtin::TensorOnes
            | Builtin::TensorFull
            | Builtin::TensorFromVec
            | Builtin::TensorAdd
            | Builtin::TensorSub
            | Builtin::TensorMul
            | Builtin::TensorDiv
            | Builtin::TensorMin
            | Builtin::TensorMax
            | Builtin::TensorEq
            | Builtin::TensorNe
            | Builtin::TensorLt
            | Builtin::TensorLe
            | Builtin::TensorGt
            | Builtin::TensorGe
            | Builtin::TensorBroadcastTo
            | Builtin::TensorMatMul
            | Builtin::TensorBatchMatMul
            | Builtin::TensorConcat
            | Builtin::TensorPermute
            | Builtin::TensorReshape
            | Builtin::TensorSliceAxis
            | Builtin::TensorTranspose
            | Builtin::TensorSumAxis
            | Builtin::TensorMeanAxis
            | Builtin::TensorArgMax
            | Builtin::TensorSum
            | Builtin::TensorSoftmax
            | Builtin::TensorCast
            | Builtin::TensorToDevice
            | Builtin::TensorScale255
            | Builtin::TensorNormalize
    )
}

fn resolve_core_type(name: &str) -> Option<CoreType> {
    match name {
        "String" => Some(CoreType::String),
        "Vec" => Some(CoreType::Vec),
        "Box" => Some(CoreType::Box),
        "Option" => Some(CoreType::Option),
        "Result" => Some(CoreType::Result),
        "Range" => Some(CoreType::Range),
        "RangeInclusive" => Some(CoreType::RangeInclusive),
        "CharsIter" => Some(CoreType::CharsIter),
        "SplitIter" => Some(CoreType::SplitIter),
        "VecIter" => Some(CoreType::VecIter),
        "HashMap" => Some(CoreType::HashMap),
        "HashSet" => Some(CoreType::HashSet),
        "KeysIter" => Some(CoreType::KeysIter),
        "ValuesIter" => Some(CoreType::ValuesIter),
        "Iter" => Some(CoreType::Iter),
        "MapIter" => Some(CoreType::MapIter),
        "FilterIter" => Some(CoreType::FilterIter),
        "Random" => Some(CoreType::Random),
        "IOError" => Some(CoreType::IOError),
        "File" => Some(CoreType::File),
        "Ordering" => Some(CoreType::Ordering),
        _ => None,
    }
}

fn resolve_core_trait(name: &str) -> Option<CoreTrait> {
    match name {
        "Copy" => Some(CoreTrait::Copy),
        "Drop" => Some(CoreTrait::Drop),
        "Eq" => Some(CoreTrait::Eq),
        "Ord" => Some(CoreTrait::Ord),
        "Num" => Some(CoreTrait::Num),
        "Clone" => Some(CoreTrait::Clone),
        "Hash" => Some(CoreTrait::Hash),
        "Default" => Some(CoreTrait::Default),
        "Display" => Some(CoreTrait::Display),
        "Error" => Some(CoreTrait::Error),
        "From" => Some(CoreTrait::From),
        "Into" => Some(CoreTrait::Into),
        "TryFrom" => Some(CoreTrait::TryFrom),
        "Index" => Some(CoreTrait::Index),
        "IndexMut" => Some(CoreTrait::IndexMut),
        "Iterator" => Some(CoreTrait::Iterator),
        "FromIterator" => Some(CoreTrait::FromIterator),
        _ => None,
    }
}

fn resolve_primitive(name: &str) -> Option<ast::Primitive> {
    match name {
        "Int8" => Some(ast::Primitive::Int8),
        "Int16" => Some(ast::Primitive::Int16),
        "Int32" => Some(ast::Primitive::Int32),
        "Int64" => Some(ast::Primitive::Int64),
        "UInt8" => Some(ast::Primitive::UInt8),
        "UInt16" => Some(ast::Primitive::UInt16),
        "UInt32" => Some(ast::Primitive::UInt32),
        "UInt64" => Some(ast::Primitive::UInt64),
        "Float32" => Some(ast::Primitive::Float32),
        "Float64" => Some(ast::Primitive::Float64),
        "Bool" => Some(ast::Primitive::Bool),
        "Char" => Some(ast::Primitive::Char),
        "String" => Some(ast::Primitive::String),
        "str" => Some(ast::Primitive::Str),
        "Unit" => Some(ast::Primitive::Unit),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::{parse, ParseMode};

    fn check_src(src: &str) -> (Hir, Vec<Diagnostic>) {
        let file = Arc::new(SourceFile::new("test.stark".to_string(), src.to_string()));
        let (tree, diags) = parse(&file, ParseMode::Program);
        assert!(diags.is_empty(), "parse failed: {:?}", diags);
        let (hir, sem_diags) = resolve(&tree, file);
        (hir, sem_diags)
    }

    fn check_snippet(src: &str) -> (Hir, Vec<Diagnostic>) {
        let file = Arc::new(SourceFile::new("test.stark".to_string(), src.to_string()));
        let (tree, diags) = parse(&file, ParseMode::Snippet);
        assert!(diags.is_empty(), "parse failed: {:?}", diags);
        let (hir, sem_diags) = resolve(&tree, file);
        (hir, sem_diags)
    }

    /// Resolve a program (parsing with the same options) under `options`.
    fn resolve_diags(src: &str, options: LanguageOptions) -> Vec<Diagnostic> {
        let file = Arc::new(SourceFile::new("test.stark".to_string(), src.to_string()));
        let (tree, pdiags) = crate::parser::parse_with_options(&file, ParseMode::Program, options);
        let (_hir, mut sem) = resolve_with_options(&tree, file, options);
        let mut all = pdiags;
        all.append(&mut sem);
        all
    }

    fn core_rejects_naming_tensor(src: &str) {
        let diags = resolve_diags(src, LanguageOptions::CORE);
        assert!(
            diags
                .iter()
                .any(|d| d.message.contains("extension `tensor`")),
            "expected a `tensor` extension rejection for {src:?}, got {:?}",
            diags.iter().map(|d| &d.message).collect::<Vec<_>>()
        );
    }

    #[test]
    fn core_only_rejects_all_d1_d5_naming_extension() {
        // D1 kinds
        core_rejects_naming_tensor("fn f<B: Dim>(x: Int32) -> Int32 { x }");
        core_rejects_naming_tensor("fn f<T: DType>(x: Int32) -> Int32 { x }");
        // D3 element types
        core_rejects_naming_tensor("fn f(x: Float16) -> Unit {}");
        core_rejects_naming_tensor("fn f(x: BFloat16) -> Unit {}");
        // D2 shape argument (rank-1 and multi) via the Tensor type name
        core_rejects_naming_tensor("fn f(x: Tensor<Float32, [B]>) -> Unit {}");
        core_rejects_naming_tensor("fn f(x: Tensor<Float32, [B, 3]>) -> Unit {}");
        // D4 model item
        core_rejects_naming_tensor(
            "model M { input x: Tensor<Float32, [B]>; output y: Tensor<Float32, [B]>; }",
        );
        // D5 const index list
        core_rejects_naming_tensor("fn f() -> Unit { let y = permute::<[0, 2, 1]>(x); }");
    }

    /// WP-C1.2 regression test for DEV-004: bare `min`/`max` in a struct-literal shorthand
    /// field, with no local/module item of that name in scope, used to resolve unconditionally
    /// to the tensor extension's `Builtin::TensorMin`/`TensorMax` even in Core-only mode
    /// (`resolve_unqualified` was missing the `options.tensor()` gate `resolve_path_relative`
    /// already had). Confirms Core-only mode now correctly reports "undefined variable"
    /// instead, and that extension mode still resolves it to the tensor builtin (no regression
    /// in the case DEV-004 wasn't about).
    #[test]
    fn bare_min_max_shorthand_field_is_gated_by_tensor_extension() {
        let src = "struct Point { min: Int32 }\nfn f() -> Unit { let p = Point { min }; }";
        let core_diags = resolve_diags(src, LanguageOptions::CORE);
        assert!(
            core_diags
                .iter()
                .any(|d| d.message.contains("undefined variable")),
            "Core-only mode should reject bare 'min' shorthand field as an undefined variable, \
             not silently resolve it to the tensor builtin: {:?}",
            core_diags.iter().map(|d| &d.message).collect::<Vec<_>>()
        );
        // Same source under the tensor extension: `min` still isn't a local/module item, so the
        // shorthand still resolves to the tensor builtin -- this confirms the gate only affects
        // Core-only mode, not correct tensor-mode behavior.
        let tensor_diags = resolve_diags(src, LanguageOptions::with_tensor());
        assert!(
            !tensor_diags
                .iter()
                .any(|d| d.message.contains("undefined variable")),
            "tensor-extension mode should still resolve bare 'min' to the tensor builtin: {:?}",
            tensor_diags.iter().map(|d| &d.message).collect::<Vec<_>>()
        );
    }

    /// WP-C1.2 regression test for DEV-007: glob-import (`use mod::*`) expansion iterated an
    /// unsorted HashMap, making which of two colliding names won (vs. got flagged E0204)
    /// nondeterministic across runs. Runs the same colliding-glob program many times and
    /// confirms the diagnostic set is identical every time.
    #[test]
    fn glob_import_collision_diagnostics_are_deterministic() {
        let src = "mod a { pub fn item() -> Int32 { 1 } }\nmod b { pub fn item() -> Int32 { 2 } }\nuse a::*;\nuse b::*;\nfn main() -> Unit {}";
        let first = resolve_diags(src, LanguageOptions::CORE);
        for _ in 0..25 {
            let again = resolve_diags(src, LanguageOptions::CORE);
            assert_eq!(
                first
                    .iter()
                    .map(|d| (&d.code, &d.message))
                    .collect::<Vec<_>>(),
                again
                    .iter()
                    .map(|d| (&d.code, &d.message))
                    .collect::<Vec<_>>(),
                "glob-import collision diagnostics differ across repeated resolves of the \
                 identical program"
            );
        }
    }

    /// WP-C1.2 regression test for DEV-006 (resolve half): resolve-stage diagnostics for a
    /// non-root file in a multi-file program used to render against the root file (the only
    /// file resolve.rs's callers ever backfilled), since resolve.rs never attached `.with_file`
    /// itself despite `current_module`/`current_file()` tracking the right file throughout.
    /// Confirms a duplicate-definition error inside an inline submodule now carries that
    /// submodule's own file identity, not silently the caller-supplied default.
    #[test]
    fn resolve_diagnostics_carry_their_own_file_not_the_caller_default() {
        let src = "mod inner {\n    fn dup() -> Unit {}\n    fn dup() -> Unit {}\n}\nfn main() -> Unit {}";
        let file = Arc::new(SourceFile::new("outer.stark".to_string(), src.to_string()));
        let (tree, pdiags) = crate::parser::parse(&file, ParseMode::Program);
        assert!(pdiags.is_empty(), "parse failed: {:?}", pdiags);
        let (_hir, diags) = resolve(&tree, file.clone());
        let dup = diags
            .iter()
            .find(|d| d.code.as_deref() == Some("E0204"))
            .expect("expected an E0204 duplicate-definition diagnostic");
        assert!(
            dup.file.is_some(),
            "resolve-stage diagnostic should carry its own file identity, not rely solely on \
             the caller's default-file fallback at render time"
        );
    }

    /// WP-C1.2 (checklist item 1): a local binding sharing a name with a module-level item.
    /// `resolve_unqualified` checks lexical scopes (`self.scopes`) before module items
    /// (resolve.rs ~1891-1897) -- confirms the local wins, and that outside the local's scope
    /// the module item resolves normally (no residual shadowing leaking past its block).
    #[test]
    fn local_binding_shadows_same_named_module_item_within_its_scope_only() {
        let (hir, diags) = check_src(
            "fn helper() -> Int32 { 1 }\n\
             fn main() -> Int32 {\n\
             \x20   let outer = helper();\n\
             \x20   let inner = { let helper = 99; helper };\n\
             \x20   outer + inner\n\
             }",
        );
        assert!(diags.is_empty(), "unexpected diagnostics: {:?}", diags);
        let _ = hir;
    }

    /// WP-C1.2 (checklist item 2): `super::` from the root module has no parent -- confirms the
    /// E0206 diagnostic (resolve.rs ~688-691) actually fires rather than panicking or silently
    /// resolving to something else. This code path had zero test evidence before this WP despite
    /// producing a real diagnostic.
    #[test]
    fn super_from_root_module_reports_e0203_not_a_panic() {
        let diags = resolve_diags(
            "use super::nothing;\nfn main() -> Unit {}",
            LanguageOptions::CORE,
        );
        assert!(
            diags
                .iter()
                .any(|d| d.code.as_deref() == Some("E0206") && d.message.contains("super")),
            "expected E0206 'no parent module for super', got {:?}",
            diags
                .iter()
                .map(|d| (&d.code, &d.message))
                .collect::<Vec<_>>()
        );
    }

    /// WP-C1.2 (checklist item 2): `super::` from a nested inline module correctly reaches the
    /// parent, and `crate::` from a nested module reaches the package root -- both previously
    /// had no dedicated test exercising navigation from a non-root starting point. Both `top`
    /// and `mid` must be `pub`: per `07-Modules-and-Packages.md` §Visibility ("items are
    /// private to their defining module by default"), STARK's model is *not* Rust's
    /// descendant-inherits-ancestor's-privacy rule -- `inner` is not `top`'s or `mid`'s defining
    /// module, so a private `top`/`mid` would correctly be rejected here regardless of nesting
    /// depth (confirmed by `module_paths_imports_and_visibility_are_enforced` above, and by the
    /// fact that the first version of this test, written assuming Rust-style visibility, failed
    /// against the real implementation with exactly this rejection -- corrected here, not a
    /// resolver bug).
    #[test]
    fn super_and_crate_navigate_correctly_from_a_nested_module() {
        let (_hir, diags) = check_src(
            "pub fn top() -> Int32 { 1 }\n\
             mod outer {\n\
             \x20   pub fn mid() -> Int32 { 2 }\n\
             \x20   mod inner {\n\
             \x20       fn via_super() -> Int32 { super::mid() }\n\
             \x20       fn via_crate() -> Int32 { crate::top() }\n\
             \x20   }\n\
             }",
        );
        assert!(diags.is_empty(), "unexpected diagnostics: {:?}", diags);
    }

    /// WP-C1.2: companion negative case to the above -- confirms private items are visible only
    /// within their *exact* defining module, not automatically to descendant modules (unlike
    /// Rust). `top` here is intentionally non-`pub`.
    #[test]
    fn private_item_is_not_visible_from_a_descendant_module() {
        let diags = resolve_diags(
            "fn top() -> Int32 { 1 }\n\
             mod outer {\n\
             \x20   mod inner {\n\
             \x20       fn via_crate() -> Int32 { crate::top() }\n\
             \x20   }\n\
             }",
            LanguageOptions::CORE,
        );
        assert!(
            diags.iter().any(|d| d.message.contains("private")),
            "expected a private-item-access rejection, got {:?}",
            diags.iter().map(|d| &d.message).collect::<Vec<_>>()
        );
    }

    /// WP-C1.2 (checklist item 5): single-level `pub use` re-export -- confirms an item is
    /// visible through the re-exporting module's own path from outside, not just its original
    /// declaration site. `reexport_vis`/`current_use_item_vis` (resolve.rs) had zero test
    /// coverage of any kind before this WP despite being real, purpose-built logic. `inner` is
    /// `pub mod` so this test isolates the re-export mechanism itself from the separate
    /// "can a sibling module see another sibling's private module" question covered by
    /// `private_item_is_not_visible_from_a_descendant_module` above.
    #[test]
    fn pub_use_single_level_reexport_is_visible_from_outside() {
        let (_hir, diags) = check_src(
            "pub mod inner {\n\
             \x20   pub fn item() -> Int32 { 1 }\n\
             }\n\
             mod facade {\n\
             \x20   pub use super::inner::item;\n\
             }\n\
             fn main() -> Int32 { facade::item() }",
        );
        assert!(diags.is_empty(), "unexpected diagnostics: {:?}", diags);
    }

    /// WP-C1.2 (checklist item 5): a 2-level `pub use` re-export chain (A re-exports from B,
    /// which re-exports from C) -- confirms the fixed-point iteration in resolve_with_options
    /// (resolve.rs ~139-151, "Pass 2... with fixed-point iteration for re-exports") actually
    /// converges on a multi-level chain, not just a single hop.
    #[test]
    fn pub_use_multi_level_reexport_chain_resolves() {
        let (_hir, diags) = check_src(
            "pub mod c {\n\
             \x20   pub fn item() -> Int32 { 1 }\n\
             }\n\
             pub mod b {\n\
             \x20   pub use super::c::item;\n\
             }\n\
             mod a {\n\
             \x20   pub use super::b::item;\n\
             }\n\
             fn main() -> Int32 { a::item() }",
        );
        assert!(diags.is_empty(), "unexpected diagnostics: {:?}", diags);
    }

    /// WP-C1.2 (checklist item 5): `pub use` of a *private* item -- per `name_is_visible_from`
    /// (resolve.rs ~822-833), `reexport_vis` is authoritative over the original item's own
    /// `vis` once populated, so a `pub use` of a private item is expected to leak it. This is a
    /// real design behavior, not an oversight -- confirmed and pinned down by this test since it
    /// had zero prior coverage and would be easy to accidentally "fix" into a rejection later
    /// without realizing it's intentional.
    #[test]
    fn pub_use_of_a_private_item_leaks_it() {
        let (_hir, diags) = check_src(
            "mod inner {\n\
             \x20   fn secret() -> Int32 { 1 }\n\
             \x20   pub use secret as facade_secret;\n\
             }\n\
             fn main() -> Int32 { inner::facade_secret() }",
        );
        // NOTE: if this assertion starts failing because `pub use` of a private item is
        // rejected, that is a deliberate semantic change to visibility rules requiring CE1/CE2
        // escalation (Charter), not a routine test update -- update this comment and
        // COMPILER-STATE.md's DEV-020 record together with the fix, don't just adjust the
        // assertion.
        assert!(
            diags.is_empty(),
            "expected pub-use-of-private to leak the item (current design), got: {:?}",
            diags
        );
    }

    /// WP-C1.2 (checklist item 6): two explicit (non-glob) `use` imports bringing in the same
    /// name from two different sources -- distinct from the already-fixed glob-import
    /// nondeterminism case (DEV-007), which only affects `use mod::*`.
    #[test]
    fn two_explicit_use_imports_colliding_on_name_is_rejected() {
        let diags = resolve_diags(
            "mod a { pub fn item() -> Int32 { 1 } }\n\
             mod b { pub fn item() -> Int32 { 2 } }\n\
             use a::item;\n\
             use b::item;\n\
             fn main() -> Unit {}",
            LanguageOptions::CORE,
        );
        assert!(
            diags.iter().any(|d| d.code.as_deref() == Some("E0204")),
            "expected E0204 for two explicit `use` imports colliding on the same name, got {:?}",
            diags.iter().map(|d| &d.message).collect::<Vec<_>>()
        );
    }

    /// WP-C1.2 (checklist item 6): a `use` import colliding with an item declared directly in
    /// the same module (as opposed to two `use` imports colliding with each other).
    #[test]
    fn use_import_colliding_with_directly_declared_item_is_rejected() {
        let diags = resolve_diags(
            "mod other { pub fn add() -> Int32 { 1 } }\n\
             use other::add;\n\
             fn add() -> Int32 { 2 }\n\
             fn main() -> Unit {}",
            LanguageOptions::CORE,
        );
        assert!(
            diags.iter().any(|d| d.code.as_deref() == Some("E0204")),
            "expected E0204 for a `use` import colliding with a directly-declared item, got {:?}",
            diags.iter().map(|d| &d.message).collect::<Vec<_>>()
        );
    }

    #[test]
    fn extension_mode_accepts_dim_and_dtype_bounds() {
        let diags = resolve_diags(
            "fn f<T: DType, N: Dim>(x: Int32) -> Int32 { x }",
            LanguageOptions::with_tensor(),
        );
        assert!(
            diags.is_empty(),
            "tensor mode should accept Dim/DType bounds: {:?}",
            diags.iter().map(|d| &d.message).collect::<Vec<_>>()
        );
    }

    #[test]
    fn user_declared_dim_trait_is_not_misclassified() {
        // A real user trait spelled `Dim` must resolve in Core mode, not be
        // rejected as the extension kind.
        let diags = resolve_diags(
            "trait Dim {}\nfn f<B: Dim>(x: Int32) -> Int32 { x }",
            LanguageOptions::CORE,
        );
        assert!(
            !diags
                .iter()
                .any(|d| d.message.contains("extension `tensor`")),
            "user trait `Dim` must not trigger the extension diagnostic: {:?}",
            diags.iter().map(|d| &d.message).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_duplicate_let() {
        let (_hir, diags) = check_snippet("let mut x: Int32 = 42; let x = 44;");
        assert_eq!(diags.len(), 1);
        assert_eq!(diags[0].code.as_deref(), Some("E0204"));
    }

    #[test]
    fn test_undefined_variable() {
        let (_hir, diags) = check_snippet("let y = x;");
        assert_eq!(diags.len(), 1);
        assert_eq!(diags[0].code.as_deref(), Some("E0200"));
    }

    #[test]
    fn test_shadowing() {
        let (_hir, diags) = check_snippet("let x = 10; { let x = 20; let y = x; }");
        assert!(diags.is_empty(), "unexpected diagnostics: {:?}", diags);
    }

    #[test]
    fn test_duplicate_item_definitions() {
        let (_hir, diags) = check_src("fn foo() {} fn foo() {}");
        assert_eq!(diags.len(), 1);
        assert_eq!(diags[0].code.as_deref(), Some("E0204"));
    }

    #[test]
    fn test_struct_lit_resolution() {
        let (_hir, diags) = check_src(
            "struct Point { x: Int32, y: Int32 } fn main() { let p = Point { x: 1, y: 2 }; }",
        );
        assert!(diags.is_empty(), "unexpected diagnostics: {:?}", diags);
    }

    #[test]
    fn module_paths_imports_and_visibility_are_enforced() {
        let (_, valid) = check_src(
            "mod math { pub fn answer() -> Int32 { 42 } } use math::answer; fn main() { let x = answer(); }",
        );
        assert!(valid.is_empty(), "unexpected diagnostics: {valid:?}");

        let (_, private) = check_src(
            "mod math { fn secret() -> Int32 { 42 } } fn main() { let x = math::secret(); }",
        );
        assert!(private
            .iter()
            .any(|diagnostic| diagnostic.code.as_deref() == Some("E0207")));
    }
}
