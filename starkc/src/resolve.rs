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

    // Pass 2: Resolve use Tree imports
    resolver.resolve_imports(&root_items);

    // Pass 3: Lower AST to HIR & perform lexical/local name resolution
    resolver.lower_crate();

    (resolver.hir, resolver.diags)
}

impl<'a> Resolver<'a> {
    fn current_file(&self) -> &SourceFile {
        &self.modules[self.current_module.0 as usize].file
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
                    Entry::Occupied(_) => self.diags.push(
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



    fn resolve_imports(&mut self, items: &[ast::ItemId]) {
        let current_mod_id = self.current_module;

        // Resolve use imports in current module
        for &ast_id in items {
            let item = self.ast.item(ast_id);
            if let ast::ItemKind::Use(use_tree) = &item.kind {
                self.resolve_use_tree(current_mod_id, use_tree);
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
                        let items_to_copy: Vec<(String, Res)> = self.modules[sub_mod_id.0 as usize]
                            .items
                            .iter()
                            .map(|(k, v)| (k.clone(), *v))
                            .collect();
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
                        let items_to_copy: Vec<(String, Res)> = self.modules[sub_mod_id.0 as usize]
                            .items
                            .iter()
                            .map(|(k, v)| (k.clone(), *v))
                            .collect();
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
        match self.modules[module_id.0 as usize].items.entry(name.clone()) {
            Entry::Occupied(_) => self.diags.push(
                Diagnostic::error(
                    format!(
                        "duplicate definition of '{}' in the same module scope",
                        name
                    ),
                    span,
                )
                .with_code("E0204"),
            ),
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
                            self.diags.push(
                                Diagnostic::error("no parent module for 'super'", segment.span)
                                    .with_code("E0203"),
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
                if let Res::Item(item_id) = res {
                    if !self.item_is_visible_from(item_id, start_mod) {
                        self.diags.push(
                            Diagnostic::error(
                                format!("item '{name_str}' is private"),
                                segment.span,
                            )
                            .with_code("E0203"),
                        );
                        return Res::Err;
                    }
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
                            self.diags.push(
                                Diagnostic::error(
                                    format!("the {what} requires extension `tensor`"),
                                    path.span,
                                )
                                .with_code("E0210"),
                            );
                        }
                        Some(_) => { /* tensor mode: deferred to M4.2 */ }
                        None => self.diags.push(
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
                    self.diags.push(
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
                    self.diags.push(
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
                                self.diags.push(
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
                        self.diags.push(
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
                    self.diags.push(
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
                    self.diags.push(
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
                    self.diags.push(
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
                            self.diags.push(
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
                    self.diags.push(
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
                let sub_items = items.as_ref().map(|sub_items| {
                    sub_items.iter().map(|&id| hir::ItemId(id.0)).collect()
                });
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
                                    self.diags.push(
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
            return Res::Builtin(builtin);
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
            .any(|diagnostic| diagnostic.code.as_deref() == Some("E0203")));
    }
}
