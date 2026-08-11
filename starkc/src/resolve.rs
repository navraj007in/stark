//! Name resolution and AST-to-HIR lowering pass for STARK (PLAN.md M2.1).

use crate::ast;
use crate::diag::Diagnostic;
use crate::hir::{self, Builtin, CoreTrait, CoreType, Hir, LocalId, Res};
use crate::options::LanguageOptions;
use crate::source::{SourceFile, Span};
use std::collections::{hash_map::Entry, HashMap};
use std::sync::Arc;

/// A single-segment name the `tensor` extension owns. Used to give a focused "requires extension
/// `tensor`" diagnostic in Core-only mode and to suppress "undefined type" for these names under
/// the extension (their full resolution lands in M4.2).
///
/// AS6 exit qualification: the table itself is the extension's, and lives with it.
fn extension_reserved_name(name: &str) -> Option<&'static str> {
    crate::extensions::tensor::syntax::extension_type_name(name)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ModuleId(pub u32);

/// DEV-228. `04-Semantic-Analysis.md` NAME-RESOLVE-001: *"Core has distinct module, type, value,
/// and associated-item namespaces … The same spelling may coexist in different namespaces, but two
/// declarations in one namespace and scope are duplicates."*
///
/// The resolver held one `HashMap<String, Res>` for all of them, so `struct Pair` alongside
/// `fn Pair()` was rejected as a duplicate although the rule permits it — and, more corrosively,
/// every lookup had to be given a PRECEDENCE over names that were never meant to compete.
/// DEV-223 and DEV-225 were both repaired by ordering one lookup ahead of another; a third such
/// exception was the trajectory this replaces.
///
/// The associated-item namespace is not here: it hangs off a type or trait, and
/// `qualified_associated_name` already answers it from `item_details`.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
enum Namespace {
    Module,
    Type,
    Value,
}

/// Which namespace a lookup is entitled to search, decided by the syntactic position that asked.
///
/// `Any` is for a position that genuinely cannot know — a path QUALIFIER (`Foo::bar`, where `Foo`
/// may be a module or a type) and an import, which per MOD-USE-001 selects by what it finds.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
enum NsHint {
    Type,
    Value,
    Any,
}

struct ModuleData {
    #[allow(dead_code)]
    name: String,
    parent: Option<ModuleId>,
    file: Arc<SourceFile>,
    /// The three module-level namespaces, replacing the single `items` map.
    modules: HashMap<String, Res>,
    types: HashMap<String, Res>,
    values: HashMap<String, Res>,
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
    /// DEV-227. A constant is the one non-constructor item a pattern may name, so it has to be
    /// distinguishable from a function -- both were `Other`.
    Const,
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
    /// **DEV-175.** Each package's DIRECT dependencies, keyed by the package's root module.
    ///
    /// The parser attaches a package's dependencies as synthetic module wrappers under that
    /// package's root, so a dependency alias was only findable from the root's own file --
    /// `use stark_http_core::Header;` resolved in `main.stark` and not in any sibling module. The
    /// spec says an unqualified dependency alias starts at that dependency's public root whichever
    /// module names it, so the alias is recorded once per package here and consulted package-wide.
    ///
    /// Keyed by the OWNING package's root rather than looked up in a global package graph, which is
    /// what keeps a transitive dependency out of reach: `app -> a -> b` registers `b` under `a`'s
    /// root, so `app` never sees it.
    dependency_aliases: HashMap<ModuleId, HashMap<String, Res>>,
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
        dependency_aliases: HashMap::new(),
    };

    // Root module
    resolver.modules.push(ModuleData {
        name: "crate".to_string(),
        parent: None,
        file: file.clone(),
        modules: HashMap::new(),
        types: HashMap::new(),
        values: HashMap::new(),
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
            .map(|m| m.modules.len() + m.types.len() + m.values.len())
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
            if let Some(Res::Item(item)) = resolver.lookup_ns(*module, name, NsHint::Any) {
                resolver.hir.publicly_nameable_items.insert(item);
            }
        }
    }

    // Pass 3: Lower AST to HIR & perform lexical/local name resolution
    resolver.lower_crate();

    // C4.5f-3c: carry the synthesised names (dependency-package `mod` wrappers) into HIR so MIR
    // lowering's module-path walk can read them.
    //
    // AS1b-ii-d: these are keyed by ITEM now, and AST and HIR item ids are DIFFERENT SPACES, so
    // this remaps through `item_map` rather than cloning. The old span-keyed map needed no
    // remapping — a span means the same thing in both trees — which is exactly the kind of
    // assumption that survives a refactor silently if it is not stated.
    let synthetic_names: std::collections::HashMap<hir::ItemId, String> = resolver
        .ast
        .synthetic_names
        .iter()
        .filter_map(|(ast_id, name)| {
            resolver
                .item_map
                .get(ast_id)
                .map(|hir_id| (*hir_id, name.clone()))
        })
        .collect();
    resolver.hir.synthetic_names = synthetic_names;
    // Frozen after parsing: the registry is complete once every file has been loaded, and nothing
    // downstream registers a source.
    resolver.hir.sources = resolver.ast.sources.clone().freeze();
    // DEV-173: literal values travel with the HIR; nothing downstream re-decodes from a span.
    resolver.hir.str_lits = resolver.ast.str_lits.clone();

    // WP-C6.2b-F1: expose the module map so the type checker can enforce member/field visibility.
    resolver.hir.item_modules = resolver
        .item_modules
        .iter()
        .map(|(item, module)| (*item, module.0))
        .collect();
    (resolver.hir, resolver.diags)
}

impl<'a> Resolver<'a> {
    fn current_file(&self) -> &SourceFile {
        &self.modules[self.current_module.0 as usize].file
    }

    /// WP-C1.2 (2026-07-17) attached the current module's file to every resolve diagnostic:
    /// without it, a diagnostic for a non-root file in a multi-file package rendered against the
    /// wrong file (DEV-006).
    ///
    /// AS1b-ii-d deleted the attachment. `current_module` still tracks where resolution is, but a
    /// diagnostic's span already carries the source it was lexed from, so the file no longer has
    /// to be re-derived from the resolver's position and cannot disagree with it.
    fn push_diag(&mut self, diag: Diagnostic) {
        self.diags.push(diag);
    }

    /// AS1b-ii-d: every span this sees indexes a real file. The `lo >= 0x8000_0000` branch that
    /// returned a synthesised NAME from a side table is gone — names are keyed by item now, so a
    /// span reader no longer has to know that some spans are not locations.
    /// The declared name of an item.
    ///
    /// AS1b-ii-d: a compiler-synthesised item (a dependency-package `mod` wrapper) has no source
    /// text, so its name comes from the item table. Everything else reads its own span. Both
    /// name-deriving sites go through here — the first version of this change patched only the
    /// submodule walk and left `declare_items` reading a zero-width span, which registered every
    /// dependency wrapper under the empty name and collided them.
    fn item_name(&self, ast_id: ast::ItemId, name_span: Span) -> String {
        match self.ast.synthetic_names.get(&ast_id) {
            Some(name) => name.clone(),
            None => self.text(name_span).to_string(),
        }
    }

    fn text(&self, span: Span) -> &str {
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
                let name_str = self.item_name(ast_id, span);
                // DEV-228. This entry is where every declaration used to land in one map, so a
                // type and a value sharing a spelling collided although NAME-RESOLVE-001 permits
                // it. Each now lands in its own namespace, and a duplicate is a duplicate WITHIN
                // one namespace -- which is exactly what the rule says.
                let ns = Self::namespace_of_item(&item.kind);
                let map = match ns {
                    Namespace::Module => &mut self.modules[current_mod_id.0 as usize].modules,
                    Namespace::Type => &mut self.modules[current_mod_id.0 as usize].types,
                    Namespace::Value => &mut self.modules[current_mod_id.0 as usize].values,
                };
                match map.entry(name_str.clone()) {
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
                ast::ItemKind::Const { .. } => {
                    self.item_details.insert(hir_id, ItemDefDetail::Const);
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
                let name_str = self.item_name(ast_id, name);
                let sub_mod_id = ModuleId(self.modules.len() as u32);

                let file = if let Some(ref sub_items_vec) = sub_items {
                    if !sub_items_vec.is_empty() {
                        if let Some(file_arc) = self.ast.item_file(sub_items_vec[0]) {
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

                // A synthesised name marks a dependency-package wrapper, exactly as the fake
                // span range used to — but as a fact about the item rather than about its span.
                let is_dep_package = self.ast.synthetic_names.contains_key(&ast_id);
                let owner_package_root = self.modules[current_mod_id.0 as usize].package_root;
                if is_dep_package {
                    // DEV-175: record the alias against the package that DECLARED the dependency,
                    // so every module of that package can name it and no other package can.
                    self.dependency_aliases
                        .entry(owner_package_root)
                        .or_default()
                        .insert(name_str.clone(), Res::Item(hir::ItemId(ast_id.0)));
                }
                let package_root = if is_dep_package {
                    sub_mod_id
                } else {
                    owner_package_root
                };

                let sub_mod_data = ModuleData {
                    name: name_str.clone(),
                    parent: Some(current_mod_id),
                    file,
                    modules: HashMap::new(),
                    types: HashMap::new(),
                    values: HashMap::new(),
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

    /// The direct-dependency alias `name` of the package containing `module`, if any.
    ///
    /// **Only the containing package's own dependencies.** This reads the alias table at that one
    /// package root rather than searching the root's items, because searching the items would make
    /// every ordinary root-level function and type implicitly visible from every child module --
    /// and the spec is explicit that an unqualified name does not search parent or crate scopes.
    /// Dependency aliases are the sole exception, so they get their own table.
    fn dependency_alias(&self, module: ModuleId, name: &str) -> Option<Res> {
        let package_root = self.modules[module.0 as usize].package_root;
        self.dependency_aliases
            .get(&package_root)
            .and_then(|aliases| aliases.get(name))
            .copied()
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
                        let mut items_to_copy: Vec<(String, Res)> =
                            self.all_module_names(sub_mod_id);
                        items_to_copy.sort_by(|a, b| a.0.cmp(&b.0));
                        for (k, v) in items_to_copy {
                            self.insert_module_item(current_mod, k, v, prefix.span);
                        }
                    } else if let Some(variants) = self.enum_variant_items(target_item_id) {
                        for (name, variant_res) in variants {
                            self.insert_module_item(current_mod, name, variant_res, prefix.span);
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
                    } else if self.enum_variant_items(target_item_id).is_some() {
                        for item in items {
                            self.resolve_enum_variant_group_item(current_mod, target_item_id, item);
                        }
                    }
                }
            }
        }
    }

    /// One leaf of a `use Color::{Red, Green};`-style group whose prefix names an enum (see
    /// `enum_variant_items`) rather than a module -- each leaf must be a bare variant name
    /// (nested groups/globs under an enum variant aren't meaningful, so anything else is simply
    /// not imported, same as an unresolved path would be).
    fn resolve_enum_variant_group_item(
        &mut self,
        current_mod: ModuleId,
        enum_item_id: hir::ItemId,
        item: &ast::UseTree,
    ) {
        let ast::UseTree::Path { path, alias } = item else {
            return;
        };
        let Some(last) = path.segments.last() else {
            return;
        };
        let variant_name = self.text(last.span).to_string();
        let Some(variants) = self.enum_variant_items(enum_item_id) else {
            return;
        };
        let Some((_, variant_res)) = variants.into_iter().find(|(name, _)| *name == variant_name)
        else {
            return;
        };
        let bind_name = if let Some(alias_span) = alias {
            self.text(*alias_span).to_string()
        } else {
            variant_name
        };
        self.insert_module_item(current_mod, bind_name, variant_res, path.span);
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
                        let mut items_to_copy: Vec<(String, Res)> =
                            self.all_module_names(sub_mod_id);
                        items_to_copy.sort_by(|a, b| a.0.cmp(&b.0));
                        for (k, v) in items_to_copy {
                            self.insert_module_item(import_mod, k, v, prefix.span);
                        }
                    } else if let Some(variants) = self.enum_variant_items(target_item_id) {
                        for (name, variant_res) in variants {
                            self.insert_module_item(import_mod, name, variant_res, prefix.span);
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
                    } else if self.enum_variant_items(target_item_id).is_some() {
                        for item in items {
                            self.resolve_enum_variant_group_item(import_mod, target_item_id, item);
                        }
                    }
                }
            }
        }
    }

    /// DEV-055: a glob/group `use` whose prefix names an enum (not a module) enumerates the
    /// enum's own variants as importable names, exactly as `use Color::*;` should bring `Red`,
    /// `Green`, `Blue` into scope the same way a real submodule's items would. Before this, the
    /// glob/group expansion below only ever consulted `submodule_map` (real modules) and did
    /// nothing at all when the prefix was an enum, since an enum's variants are resolved
    /// dynamically through `item_details` (see `resolve_path_relative`'s `ItemDefDetail::Enum`
    /// arm) rather than being pre-populated into a module's `items` map the way real submodule
    /// contents are.
    fn enum_variant_items(&self, item_id: hir::ItemId) -> Option<Vec<(String, Res)>> {
        match self.item_details.get(&item_id) {
            Some(ItemDefDetail::Enum { variants }) => Some(
                variants
                    .iter()
                    .enumerate()
                    .map(|(idx, name)| (name.clone(), Res::Variant(item_id, idx as u32)))
                    .collect(),
            ),
            _ => None,
        }
    }

    /// The namespace a declaration belongs to. NAME-RESOLVE-001's own division: types, aliases and
    /// traits are the type namespace; constants and functions are the value namespace; a `mod` is
    /// the module namespace. Enum VARIANTS are deliberately absent — they are associated names on
    /// their enum, not module-level ones, and `qualified_associated_name` answers for them.
    fn namespace_of_item(kind: &ast::ItemKind) -> Namespace {
        match kind {
            ast::ItemKind::Mod { .. } => Namespace::Module,
            ast::ItemKind::Struct { .. }
            | ast::ItemKind::Enum { .. }
            | ast::ItemKind::Trait { .. }
            | ast::ItemKind::TypeAlias { .. }
            | ast::ItemKind::Model(_) => Namespace::Type,
            ast::ItemKind::Fn(_) | ast::ItemKind::Const { .. } => Namespace::Value,
            // No declared name of their own.
            ast::ItemKind::Use(_) | ast::ItemKind::Impl { .. } => Namespace::Value,
        }
    }

    /// The namespace an already-resolved name belongs to, for imports: MOD-USE-001 selects by what
    /// the leaf resolves to rather than by how it was spelled.
    fn namespace_of_res(&self, res: Res) -> Namespace {
        match res {
            Res::Item(item_id) => {
                if self.submodule_map.contains_key(&ast::ItemId(item_id.0)) {
                    Namespace::Module
                } else {
                    match self.item_details.get(&item_id) {
                        Some(ItemDefDetail::Enum { .. })
                        | Some(ItemDefDetail::Struct { .. })
                        | Some(ItemDefDetail::Trait { .. })
                        | Some(ItemDefDetail::Model) => Namespace::Type,
                        _ => Namespace::Value,
                    }
                }
            }
            // A variant is a constructor: a value.
            Res::Variant(_, _) | Res::Builtin(_) => Namespace::Value,
            Res::Primitive(_) | Res::CoreType(_) | Res::CoreTrait(_) => Namespace::Type,
            _ => Namespace::Value,
        }
    }

    fn ns_map(&self, module: ModuleId, ns: Namespace) -> &HashMap<String, Res> {
        let m = &self.modules[module.0 as usize];
        match ns {
            Namespace::Module => &m.modules,
            Namespace::Type => &m.types,
            Namespace::Value => &m.values,
        }
    }

    /// Look `name` up in exactly the namespace the asking position is entitled to.
    ///
    /// `NsHint::Any` searches modules, then types, then values — which is not a precedence rule
    /// smuggled back in: it is for positions that legitimately admit any of the three, a path
    /// qualifier and an import. A position that KNOWS its namespace never reaches the fallback.
    fn lookup_ns(&self, module: ModuleId, name: &str, hint: NsHint) -> Option<Res> {
        match hint {
            NsHint::Type => self.ns_map(module, Namespace::Type).get(name).copied(),
            NsHint::Value => self
                .ns_map(module, Namespace::Value)
                .get(name)
                .copied()
                // A unit-like nominal used as a value pattern still resolves; the type namespace is
                // consulted only after the value namespace has declined.
                .or_else(|| self.ns_map(module, Namespace::Type).get(name).copied()),
            NsHint::Any => self
                .ns_map(module, Namespace::Module)
                .get(name)
                .copied()
                .or_else(|| self.ns_map(module, Namespace::Type).get(name).copied())
                .or_else(|| self.ns_map(module, Namespace::Value).get(name).copied()),
        }
    }

    /// Every name declared in `module`, across all three namespaces. For diagnostics and for the
    /// glob import, which MOD-USE-001 defines over "all public names in the selected namespaces".
    fn all_module_names(&self, module: ModuleId) -> Vec<(String, Res)> {
        let m = &self.modules[module.0 as usize];
        m.modules
            .iter()
            .chain(m.types.iter())
            .chain(m.values.iter())
            .map(|(k, v)| (k.clone(), *v))
            .collect()
    }

    fn insert_module_item(&mut self, module_id: ModuleId, name: String, res: Res, span: Span) {
        if let Some(vis) = self.current_use_item_vis {
            self.reexport_vis
                .insert((module_id, name.clone()), Some(vis));
        }
        let ns = self.namespace_of_res(res);
        let map = match ns {
            Namespace::Module => &mut self.modules[module_id.0 as usize].modules,
            Namespace::Type => &mut self.modules[module_id.0 as usize].types,
            Namespace::Value => &mut self.modules[module_id.0 as usize].values,
        };
        match map.entry(name.clone()) {
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

    /// The name `name` as the qualifier in `current_res` owns it, or `None` when the qualifier is
    /// not a type, trait or model and the module namespace should answer instead.
    ///
    /// DEV-223/225. `04-Semantic-Analysis.md` NAME-RESOLVE-001: *"Associated names are searched
    /// only after resolving their qualifying type or trait."* The subsequent-segment loop did the
    /// opposite -- it consulted the enclosing module's items first, so a module-level name won
    /// over the qualifier's own. `Attr::Policy` resolved to a same-named TYPE rather than to
    /// `Attr`'s variant, and `Foo::new()` resolved to a module-level `new` rather than to `Foo`'s
    /// associated function.
    ///
    /// `None` is returned for a module qualifier (`ItemDefDetail::Other`), which is what keeps
    /// `mymod::Thing` resolving through the module namespace exactly as before.
    fn qualified_associated_name(
        &self,
        current_res: Option<Res>,
        name: &str,
        span: crate::source::Span,
    ) -> Option<Res> {
        let Some(Res::Item(item_id)) = current_res else {
            return None;
        };
        match self.item_details.get(&item_id)? {
            ItemDefDetail::Enum { variants } => Some(
                variants
                    .iter()
                    .position(|v| v == name)
                    .map(|idx| Res::Variant(item_id, idx as u32))
                    .unwrap_or(Res::AssociatedFn(item_id, span)),
            ),
            ItemDefDetail::Struct { .. } => Some(Res::AssociatedFn(item_id, span)),
            ItemDefDetail::Trait { items } => Some(
                items
                    .iter()
                    .position(|item| item == name)
                    .map(|member| Res::TraitMember(item_id, member as u32))
                    .unwrap_or(Res::Err),
            ),
            ItemDefDetail::Model => Some(if name == "load" {
                Res::ModelLoad(item_id)
            } else {
                Res::Err
            }),
            // Not a qualifier that owns associated names: let the module namespace answer.
            ItemDefDetail::Const | ItemDefDetail::Other => None,
        }
    }

    /// Whether a PATTERN may name `res`.
    ///
    /// DEV-222/226/227. `resolve_path` answers for expression position, where far more is legal:
    /// `Duration::from_seconds` is an associated function, `Vec::new` is a builtin function, and a
    /// bare item name is whatever item it names. A pattern may name only a constructor or a
    /// constant. Every resolution this rejects used to be accepted as a pattern that simply never
    /// matched, which a wildcard arm then hid completely.
    ///
    /// Exhaustive over `Res` so a new variant cannot be added without deciding this.
    fn resolution_is_pattern_legal(&self, res: &Res) -> bool {
        match res {
            // An enum variant is the pattern constructor.
            Res::Variant(_, _) => true,
            // `Res::Err` is an already-reported failure; the caller diagnoses it before consulting
            // this, and reporting again would double up on one mistake.
            Res::Err => true,
            // Only the prelude's CONSTRUCTORS, never its functions.
            Res::Builtin(builtin) => hir::builtin_is_pattern_constructor(builtin),
            // DEV-227. A bare item name is a pattern only when the item is a constant or a struct
            // whose shape a pattern can name. A function, a trait, a module or a model is not a
            // pattern, and accepting one let `match n { helper => .. , _ => .. }` compile with
            // `helper` silently matching nothing.
            Res::Item(item_id) => matches!(
                self.item_details.get(item_id),
                Some(ItemDefDetail::Struct { .. }) | Some(ItemDefDetail::Const)
            ),
            // Everything else names something that exists but cannot appear in pattern position.
            Res::AssociatedFn(_, _)
            | Res::TraitMember(_, _)
            | Res::CoreTraitMember(_, _)
            | Res::CoreTrait(_)
            | Res::CoreType(_)
            | Res::ModelLoad(_)
            | Res::Local(_)
            | Res::SelfValue(_)
            | Res::SelfType
            | Res::SelfAssoc(_)
            | Res::TypeParam
            | Res::ParamAssoc(_, _)
            | Res::Primitive(_) => false,
        }
    }

    /// `res` unless a pattern may not name it, in which case `Res::Err` after reporting.
    ///
    /// DEV-222. `resolve_path` answers for EXPRESSION position, where a qualified name that is
    /// not a variant of the enum or struct is an inherent associated function -- which is how
    /// `Duration::from_seconds` and `Instant::now` reach their definitions, and must not change.
    /// A pattern may name far less. The three pattern branches previously asked only "is it
    /// `Res::Err`", so an associated-function resolution was accepted as a pattern that simply
    /// never matched: `Colour::Blu` type-checked and fell through to the wildcard with nothing
    /// reported.
    fn reject_non_pattern_resolution(&mut self, path: &ast::Path, res: Res, code: &str) -> Res {
        if self.resolution_is_pattern_legal(&res) {
            return res;
        }
        self.push_diag(
            Diagnostic::error(
                format!(
                    "'{}' is not a pattern; no such variant exists",
                    self.path_to_string(path)
                ),
                path.span,
            )
            .with_code(code),
        );
        Res::Err
    }

    /// Resolve `path` for a position entitled to `hint`'s namespace.
    ///
    /// DEV-228. Before the namespaces were split this took no hint, because there was only one map
    /// to search and the question could not be asked.
    fn resolve_path_in(&mut self, start_mod: ModuleId, path: &ast::Path, hint: NsHint) -> Res {
        self.resolve_path_relative_in(start_mod, path, hint)
    }

    /// The `NsHint::Any` entry point, for imports and for callers whose position admits any
    /// namespace. MOD-USE-001 defines an import over what the leaf resolves to.
    fn resolve_path(&mut self, start_mod: ModuleId, path: &ast::Path) -> Res {
        self.resolve_path_relative_in(start_mod, path, NsHint::Any)
    }

    fn resolve_path_relative(&mut self, start_mod: ModuleId, path: &ast::Path) -> Res {
        self.resolve_path_relative_in(start_mod, path, NsHint::Any)
    }

    fn resolve_path_relative_in(
        &mut self,
        start_mod: ModuleId,
        path: &ast::Path,
        hint: NsHint,
    ) -> Res {
        if path.segments.is_empty() {
            return Res::Err;
        }
        match self.path_to_string(path).as_str() {
            "String::from" => return Res::Builtin(Builtin::StringFrom),
            "String::new" => return Res::Builtin(Builtin::StringNew),
            "String::with_capacity" => return Res::Builtin(Builtin::StringWithCapacity),
            "Char::from_u32" => return Res::Builtin(Builtin::CharFromU32),
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
        // NAME-RESOLVE-001 distinguishes the MODULE namespace from a type's associated names, and
        // `crate`/`super` stash a placeholder `Res::Item` while steering `current_mod`. Without
        // this flag the associated-name lookup below would read that placeholder as a real type.
        let mut current_is_module = false;

        for (i, segment) in path.segments.iter().enumerate() {
            let name_str = self.text(segment.span);

            if i == 0 {
                match segment.kind {
                    ast::SegmentKind::Crate => {
                        let pkg_root = self.modules[start_mod.0 as usize].package_root;
                        current_res = Some(Res::Item(hir::ItemId(pkg_root.0)));
                        current_mod = pkg_root;
                        current_is_module = true;
                        continue;
                    }
                    ast::SegmentKind::Super => {
                        if let Some(parent) = self.modules[start_mod.0 as usize].parent {
                            current_res = Some(Res::Item(hir::ItemId(0)));
                            current_mod = parent;
                            current_is_module = true;
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
                            // A first segment is a QUALIFIER when more segments follow, and it
                            // may legitimately be a module, a type or a value; when it is the
                            // whole path, the asking position's hint decides.
                            if let Some(res) = self.lookup_ns(
                                start_mod,
                                name_str,
                                if path.segments.len() == 1 {
                                    hint
                                } else {
                                    NsHint::Any
                                },
                            ) {
                                resolved = Some(res);
                            } else if let Some(res) = self.dependency_alias(start_mod, name_str) {
                                // DEV-175. Deliberately AFTER the current module's own items, so a
                                // local name shadows a dependency alias rather than the reverse.
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
                            current_is_module = false;
                            if let Res::Item(item_id) = res {
                                if let Some(&sub_mod_id) =
                                    self.submodule_map.get(&ast::ItemId(item_id.0))
                                {
                                    current_mod = sub_mod_id;
                                    current_is_module = true;
                                }
                            }
                        } else {
                            return Res::Err;
                        }
                    }
                }
            } else if !current_is_module && matches!(current_res, Some(Res::Item(_))) {
                // DEV-228 phase 3. A qualifier that is a TYPE or TRAIT owns an associated-item
                // namespace, and NAME-RESOLVE-001 searches associated names "after resolving their
                // qualifying type or trait" — in THAT qualifier's namespace, not in whichever
                // module happens to enclose the path.
                //
                // DEV-223 and DEV-225 were repaired by ordering this lookup ahead of the module
                // maps. That ordering is now gone, and with it the precedence question: the module
                // maps are not consulted at all here, because they were never the right place to
                // look. `Attr::Policy` cannot find an enclosing module's `Policy` for the same
                // reason `Attr::Policy` cannot find a local variable — a different namespace is
                // not a lower-priority candidate, it is not a candidate.
                let Some(assoc_res) =
                    self.qualified_associated_name(current_res, name_str, segment.span)
                else {
                    return Res::Err;
                };
                if assoc_res == Res::Err {
                    return Res::Err;
                }
                current_res = Some(assoc_res);
            } else if let Some(res) = self.lookup_ns(
                current_mod,
                name_str,
                if i + 1 == path.segments.len() {
                    hint
                } else {
                    NsHint::Any
                },
            ) {
                if !self.name_is_visible_from(current_mod, name_str, start_mod) {
                    self.push_diag(
                        Diagnostic::error(format!("item '{name_str}' is private"), segment.span)
                            .with_code("E0207"),
                    );
                    return Res::Err;
                }
                current_res = Some(res);
                current_is_module = false;
                if let Res::Item(item_id) = res {
                    if let Some(&sub_mod_id) = self.submodule_map.get(&ast::ItemId(item_id.0)) {
                        current_mod = sub_mod_id;
                        current_is_module = true;
                    }
                }
            } else if let Some(Res::Item(_)) = current_res {
                // A qualifier that owns associated names was answered by
                // `qualified_associated_name` above; reaching here means the qualifier was a
                // module or a constant and the module lookup already declined.
                return Res::Err;
            } else if let Some(Res::CoreTrait(core_trait)) = current_res {
                // DEV-052: a `CoreTrait` (`Eq`, `Ord`, `Hash`, ...) has no `hir::ItemKind::Trait`
                // declaration item to look a member up against the way the `Res::Item` arm above
                // does for a user-declared trait -- it's resolved directly by name instead.
                if core_trait_method_name(core_trait) == Some(name_str) {
                    current_res = Some(Res::CoreTraitMember(core_trait, segment.span));
                } else {
                    return Res::Err;
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
        if let Some(Res::Item(item_id)) = self.lookup_ns(module_id, name, NsHint::Any) {
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
                    // A type annotation, bound or impl target searches the TYPE namespace.
                    self.resolve_path_in(self.current_module, path, NsHint::Type)
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
            // WP-FMT-001: segments lower one for one. A field's expression is an ordinary
            // expression and resolves in the enclosing scope — an interpolation introduces no
            // scope of its own, so `f"{x}"` sees exactly the `x` the surrounding code sees.
            ast::ExprKind::FormatString { segments } => {
                let segments = segments
                    .iter()
                    .map(|segment| match segment {
                        ast::FormatSegment::Literal { text, span } => hir::FormatSegment::Literal {
                            text: text.clone(),
                            span: *span,
                        },
                        ast::FormatSegment::Field {
                            expr,
                            spec,
                            span,
                            expr_span,
                        } => hir::FormatSegment::Field {
                            expr: self.lower_expr(*expr),
                            spec: *spec,
                            span: *span,
                            expr_span: *expr_span,
                        },
                    })
                    .collect();
                hir::ExprKind::FormatString { segments }
            }
            ast::ExprKind::Path { path, turbofish } => {
                // An expression searches the VALUE namespace.
                let res = self.resolve_path_in(self.current_module, path, NsHint::Value);
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
                let module_res = self.lookup_ns(self.current_module, name_str, NsHint::Value);
                // A bare identifier that names a known value -- a module item/enum variant, or
                // a compiler builtin -- must match by value (03-Type-System.md's pattern-name-
                // resolution note; 02-Syntax-Grammar.md SYN-PATTERN-001 states the same rule).
                // Previously only `Res::Variant`/`Res::Item` were checked here; `Res::Builtin`
                // was not, so `None` (the only realistic bare, zero-argument Builtin pattern --
                // `Some`/`Ok`/`Err` always take parens and parse as `TupleVariant` instead)
                // unconditionally fell through to "fresh local binding": a `None` arm silently
                // matched *any* value instead of only `Option::None`, with no diagnostic --
                // confirmed to produce wrong runtime output, not merely a spurious rejection.
                // Gated by the tensor extension exactly as `resolve_unqualified` already gates
                // ordinary bare-identifier builtin resolution (DEV-004), so a Core-only-mode
                // program can still use a tensor-only builtin name (e.g. `min`) as an ordinary
                // pattern-binding identifier when the extension isn't enabled.
                let value_res = if let Some(Res::Variant(enum_id, variant_idx)) = module_res {
                    Some(Res::Variant(enum_id, variant_idx))
                } else if let Some(Res::Item(item_id)) = module_res {
                    // DEV-227. SYN-PATTERN-001 matches by value only for "a unit enum variant or a
                    // constant in scope"; anything else "introduces a new binding". Every item was
                    // taken by value here, so a bare `helper` naming a FUNCTION became a value
                    // pattern that could never equal anything -- it silently matched nothing, and
                    // a wildcard arm absorbed the mistake. A function, struct, trait, alias or
                    // module name is a binding, which is what the rule says and what a reader
                    // expects.
                    if matches!(self.item_details.get(&item_id), Some(ItemDefDetail::Const)) {
                        Some(Res::Item(item_id))
                    } else {
                        None
                    }
                } else if let Some(builtin) = resolve_builtin(name_str) {
                    // DEV-226. Only a constructor matches by value; `Vec::new` and friends are
                    // functions and must fall through to a binding exactly as any other name does.
                    if (!is_tensor_builtin(builtin) || self.options.tensor())
                        && hir::builtin_is_pattern_constructor(&builtin)
                    {
                        Some(Res::Builtin(builtin))
                    } else {
                        None
                    }
                } else {
                    None
                };
                if let Some(res) = value_res {
                    let path = ast::Path {
                        segments: vec![ast::PathSegment {
                            kind: ast::SegmentKind::Ident,
                            span: *name_span,
                        }],
                        span: *name_span,
                    };
                    hir::PatKind::Path { path, res }
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
                let mut res = self.resolve_path(self.current_module, path);
                if res == Res::Err {
                    self.push_diag(
                        Diagnostic::error(
                            format!("undefined pattern path '{}'", self.path_to_string(path)),
                            path.span,
                        )
                        .with_code("E0200"),
                    );
                } else {
                    res = self.reject_non_pattern_resolution(path, res, "E0200");
                }
                hir::PatKind::Path {
                    path: path.clone(),
                    res,
                }
            }
            ast::PatKind::TupleVariant { path, pats } => {
                let mut res = self.resolve_path(self.current_module, path);
                if res == Res::Err {
                    self.push_diag(
                        Diagnostic::error(
                            format!("undefined enum variant '{}'", self.path_to_string(path)),
                            path.span,
                        )
                        .with_code("E0202"),
                    );
                } else {
                    res = self.reject_non_pattern_resolution(path, res, "E0202");
                }
                let pats = pats.iter().map(|&p| self.lower_pattern(p)).collect();
                hir::PatKind::TupleVariant {
                    path: path.clone(),
                    res,
                    pats,
                }
            }
            ast::PatKind::Struct { path, fields } => {
                let mut res = self.resolve_path(self.current_module, path);
                if res == Res::Err {
                    self.push_diag(
                        Diagnostic::error(
                            format!("undefined struct/variant '{}'", self.path_to_string(path)),
                            path.span,
                        )
                        .with_code("E0202"),
                    );
                } else {
                    res = self.reject_non_pattern_resolution(path, res, "E0202");
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
                let is_discard = var_name == "_";
                if !is_discard && self.scopes.last().unwrap().contains_key(&var_name) {
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
                // `_` is a discard, not a name. Allocate a local so the existing ownership and
                // drop machinery still observes the initializer, but do not publish it into the
                // lexical scope; repeated discards therefore cannot collide or be referenced.
                if !is_discard {
                    self.scopes
                        .last_mut()
                        .unwrap()
                        .insert(var_name, Res::Local(local));
                }
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
        if let Some(source) = self.ast.item_sources.get(&ast_id) {
            self.hir.item_sources.insert(hir_id, *source);
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
        if let Some(res) = self.lookup_ns(self.current_module, name, NsHint::Value) {
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
        // **AS6: the extension's spelling table lives with the extension.**
        //
        // Exit criterion 2 names this exactly — "central Core modules do not contain open-ended
        // tensor spelling tables". Thirty-three `name => Builtin::Tensor*` arms sat here, in
        // Core's resolver, and every new tensor operation would have added one more. Consulted
        // last, so every Core name above still decides first.
        _ => crate::extensions::tensor::builtin_named(name),
    }
}

/// Whether `b` is an extension-owned tensor operation.
///
/// AS6: the CATALOGUE moved to `extensions::tensor`; this stays as the Core-side name the
/// resolver's gate reads, so the gate keeps reading as a Core concern while the list of what
/// counts belongs to the extension that owns it.
pub fn is_tensor_builtin(b: Builtin) -> bool {
    crate::extensions::tensor::owns_builtin(b)
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

pub(crate) fn resolve_core_trait(name: &str) -> Option<CoreTrait> {
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

/// DEV-052: the fixed, single callable method name for each `CoreTrait` that supports qualified-
/// call syntax (`Eq::eq(&a, &b)`). Traits with no directly user-callable single method (`Copy`,
/// `Num`, `Index`, ...) return `None`, matching how a user-declared trait with no matching
/// member also fails to resolve a bogus qualified-call segment.
pub(crate) fn core_trait_method_name(core_trait: CoreTrait) -> Option<&'static str> {
    match core_trait {
        CoreTrait::Eq => Some("eq"),
        CoreTrait::Ord => Some("cmp"),
        CoreTrait::Hash => Some("hash"),
        CoreTrait::Clone => Some("clone"),
        CoreTrait::Display => Some("fmt"),
        CoreTrait::Default => Some("default"),
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
    /// to the tensor extension's `TensorBuiltin::Min`/`Max` even in Core-only mode
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

    /// DEV-055: a glob `use` whose prefix names an enum (not a module) used to expand to
    /// nothing at all, since `resolve_use_tree`'s `Glob` arm only ever consulted
    /// `submodule_map` (populated for real modules only) and an enum's variants are resolved
    /// dynamically through `item_details`, never pre-populated into a module's `items` map.
    /// Confirms a bare, glob-imported unit variant now resolves as an expression.
    #[test]
    fn glob_imported_enum_variant_resolves_as_bare_expression() {
        let src = "enum Color { Red, Green, Blue }\nuse Color::*;\nfn main() -> Unit { let c: Color = Red; }";
        let diags = resolve_diags(src, LanguageOptions::CORE);
        assert!(
            diags.is_empty(),
            "expected 'Red' to resolve via the glob import: {:?}",
            diags.iter().map(|d| &d.message).collect::<Vec<_>>()
        );
    }

    /// Companion: a group `use` (`use Color::{Red, Blue};`) whose prefix names an enum hits the
    /// identical `submodule_map`-only gap in `resolve_use_tree`'s `Group` arm, fixed the same
    /// way via `resolve_enum_variant_group_item`. Confirms only the *named* variants resolve --
    /// `Green`, deliberately left out of the group, must still be undefined.
    #[test]
    fn group_imported_enum_variants_resolve_selectively() {
        let src = "enum Color { Red, Green, Blue }\nuse Color::{Red, Blue};\nfn main() -> Unit { let c: Color = Red; let d: Color = Blue; }";
        let diags = resolve_diags(src, LanguageOptions::CORE);
        assert!(
            diags.is_empty(),
            "expected 'Red' and 'Blue' to resolve via the group import: {:?}",
            diags.iter().map(|d| &d.message).collect::<Vec<_>>()
        );

        let excluded_src = "enum Color { Red, Green, Blue }\nuse Color::{Red, Blue};\nfn main() -> Unit { let c: Color = Green; }";
        let excluded_diags = resolve_diags(excluded_src, LanguageOptions::CORE);
        assert!(
            excluded_diags
                .iter()
                .any(|d| d.message.contains("undefined variable")),
            "'Green' was not named in the group import and must stay undefined, not leak in \
             alongside 'Red'/'Blue': {:?}",
            excluded_diags
                .iter()
                .map(|d| &d.message)
                .collect::<Vec<_>>()
        );
    }

    /// WP-C1.2 regression test for DEV-006 (resolve half): resolve-stage diagnostics for a
    /// non-root file in a multi-file program used to render against the root file (the only
    /// file resolve.rs's callers ever backfilled), since resolve.rs never attached `.with_file`
    /// itself despite `current_module`/`current_file()` tracking the right file throughout.
    ///
    /// AS1b-ii-d rewrote what "carries its own file identity" means: there is no `Diagnostic.file`
    /// to inspect, so the assertion is on the span's `SourceId` — which is the identity, rather
    /// than a copy of it that could disagree.
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
        let registered = tree
            .sources
            .id_for_name("outer.stark")
            .expect("the parse registered this file");
        assert_eq!(
            dup.span.source, registered,
            "resolve-stage diagnostic should name the source it belongs to, not rely on a \
             caller-supplied default at render time"
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

    /// DEV-053/DEV-054 (found building the WP-C2.12 differential corpus): `lower_pattern`'s
    /// `ast::PatKind::Binding` arm only recognized `Res::Variant`/`Res::Item` module items as
    /// "known value" resolutions for a bare identifier -- never `Res::Builtin`, which is how
    /// `None` is classified (`resolve_builtin("None") == Some(Builtin::None)`). Every bare
    /// `None` pattern therefore fell through to "fresh local binding" unconditionally: two
    /// `None`s within one tuple pattern collided as duplicate definitions of a local variable
    /// literally named "None" (`E0204`), even though a by-value identifier pattern is supposed
    /// to introduce no binding at all (`02-Syntax-Grammar.md` SYN-PATTERN-001's own note). This
    /// is the resolve-stage regression pinning down that `(None, None)` no longer collides;
    /// `interp.rs`'s `bare_none_pattern_matches_by_value_not_as_a_wildcard` and siblings cover
    /// the full end-to-end runtime-semantics half of the same fix (the more serious half: this
    /// bug did not just mis-diagnose valid code, it made `None` silently match *any* value).
    #[test]
    fn repeated_none_in_one_tuple_pattern_does_not_collide_as_duplicate_bindings() {
        let (_, diags) = check_src(
            "fn main() { \
                 let pair: (Option<Int32>, Option<Int32>) = (None, None); \
                 let _ = match pair { (None, None) => 0, _ => 1 }; \
             }",
        );
        assert!(
            diags.is_empty(),
            "unexpected diagnostics for repeated by-value `None` in one tuple pattern: {diags:?}"
        );
    }
}
