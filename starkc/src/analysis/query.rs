use super::*;
use crate::ast;
use crate::hir;
use crate::typecheck::{ExtensionTy, Ty};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SourceLocation {
    pub source: SourceId,
    pub span: Span,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EnclosingContext {
    pub source: SourceId,
    pub item: Option<QueryHandle>,
    pub module: Option<QueryHandle>,
    pub package: Option<String>,
}

#[derive(Clone, Copy)]
struct NodeRecord {
    handle: QueryHandle,
    location: SourceLocation,
    item: Option<QueryHandle>,
    module: Option<QueryHandle>,
}

#[derive(Clone, Copy)]
struct SymbolOccurrence {
    location: SourceLocation,
    symbol: QueryHandle,
}

#[derive(Default)]
pub(crate) struct QueryIndex {
    nodes: Vec<NodeRecord>,
    definitions: HashMap<QueryHandle, SourceLocation>,
    references: HashMap<QueryHandle, Vec<SourceLocation>>,
    occurrences: Vec<SymbolOccurrence>,
}

impl QueryIndex {
    pub(crate) fn build(
        analysis: u64,
        ast: &Ast,
        hir: Option<&Hir>,
        sources: &SourceMap,
        symbols: &SymbolIndex,
    ) -> Self {
        let mut index = Self::default();
        let root = sources.files.first().map(|source| source.id);
        if let Some(root) = root {
            AstIndexer::new(analysis, ast, sources, &mut index).walk_root(root);
        }
        if let (Some(root), Some(hir)) = (root, hir) {
            HirIndexer::new(analysis, hir, sources, &mut index).walk_root(root);
        }
        for symbol in symbols.symbols() {
            let location = SourceLocation {
                source: symbol.source,
                span: symbol.span,
            };
            index.definitions.insert(symbol.definition, location);
            index.occurrences.push(SymbolOccurrence {
                location,
                symbol: symbol.definition,
            });
        }
        for references in index.references.values_mut() {
            references
                .sort_by_key(|location| (location.source.0, location.span.lo, location.span.hi));
            references.dedup();
        }
        index
    }

    pub(crate) fn contains(&self, handle: QueryHandle) -> bool {
        self.nodes.iter().any(|node| node.handle == handle)
            || self.definitions.contains_key(&handle)
    }

    pub(crate) fn node_at(
        &self,
        domain: QueryDomain,
        source: SourceId,
        offset: u32,
    ) -> Option<QueryHandle> {
        self.nodes
            .iter()
            .filter(|node| {
                node.handle.domain == domain
                    && node.location.source == source
                    && contains(node.location.span, offset)
            })
            .min_by_key(|node| {
                (
                    node.location.span.hi.saturating_sub(node.location.span.lo),
                    kind_rank(node.handle.kind),
                )
            })
            .map(|node| node.handle)
    }

    pub(crate) fn symbol_at(&self, source: SourceId, offset: u32) -> Option<QueryHandle> {
        self.occurrences
            .iter()
            .filter(|occurrence| {
                occurrence.location.source == source && contains(occurrence.location.span, offset)
            })
            .min_by_key(|occurrence| {
                occurrence
                    .location
                    .span
                    .hi
                    .saturating_sub(occurrence.location.span.lo)
            })
            .map(|occurrence| occurrence.symbol)
    }

    pub(crate) fn definition(&self, symbol: QueryHandle) -> Option<SourceLocation> {
        self.definitions.get(&symbol).copied()
    }

    pub(crate) fn references(&self, symbol: QueryHandle) -> Vec<SourceLocation> {
        self.references.get(&symbol).cloned().unwrap_or_default()
    }

    pub(crate) fn type_of(
        &self,
        analysis: &ProjectAnalysis,
        handle: QueryHandle,
    ) -> Option<String> {
        if handle.domain != QueryDomain::Hir {
            return None;
        }
        let tables = analysis.type_tables.as_ref();
        match handle.kind {
            QueryKind::Expression => tables
                .and_then(|tables| tables.expr_types.get(&hir::ExprId(handle.slot)))
                .map(|ty| render_ty(analysis, ty)),
            QueryKind::Local => tables
                .and_then(|tables| tables.local_types.get(&hir::LocalId(handle.slot)))
                .map(|ty| render_ty(analysis, ty)),
            QueryKind::Type => self
                .node(handle)
                .and_then(|node| source_text(analysis, node.location)),
            QueryKind::Item => self.signature(analysis, handle),
            QueryKind::Statement | QueryKind::Pattern | QueryKind::Block => None,
        }
    }

    pub(crate) fn signature(
        &self,
        analysis: &ProjectAnalysis,
        handle: QueryHandle,
    ) -> Option<String> {
        if handle.domain != QueryDomain::Hir || handle.kind != QueryKind::Item {
            return None;
        }
        let hir = analysis.hir.as_ref()?;
        let item = hir.items.get(handle.slot as usize)?;
        let node = self.node(handle)?;
        let text = source_text(
            analysis,
            SourceLocation {
                source: node.location.source,
                span: item.span,
            },
        )?;
        Some(trim_item_signature(&text).to_string())
    }

    pub(crate) fn enclosing(
        &self,
        analysis: &ProjectAnalysis,
        handle: QueryHandle,
    ) -> Option<EnclosingContext> {
        let node = self.node(handle)?;
        let package = match &analysis.source_map.get(node.location.source)?.provenance {
            SourceProvenance::Root { package } | SourceProvenance::Module { package } => {
                package.clone()
            }
        };
        Some(EnclosingContext {
            source: node.location.source,
            item: node.item,
            module: node.module,
            package,
        })
    }

    fn node(&self, handle: QueryHandle) -> Option<&NodeRecord> {
        self.nodes.iter().find(|node| node.handle == handle)
    }

    fn add_node(
        &mut self,
        handle: QueryHandle,
        source: SourceId,
        span: Span,
        item: Option<QueryHandle>,
        module: Option<QueryHandle>,
    ) {
        if !self.nodes.iter().any(|node| node.handle == handle) {
            self.nodes.push(NodeRecord {
                handle,
                location: SourceLocation { source, span },
                item,
                module,
            });
        }
    }

    fn define(&mut self, symbol: QueryHandle, source: SourceId, span: Span) {
        let location = SourceLocation { source, span };
        self.definitions.entry(symbol).or_insert(location);
        self.occurrences.push(SymbolOccurrence { location, symbol });
    }

    fn reference(&mut self, symbol: QueryHandle, source: SourceId, span: Span) {
        let location = SourceLocation { source, span };
        self.references.entry(symbol).or_default().push(location);
        self.occurrences.push(SymbolOccurrence { location, symbol });
    }
}

fn query_handle(analysis: u64, domain: QueryDomain, kind: QueryKind, slot: u32) -> QueryHandle {
    QueryHandle {
        analysis,
        domain,
        kind,
        slot,
    }
}

fn contains(span: Span, offset: u32) -> bool {
    if span.lo == span.hi {
        offset == span.lo
    } else {
        span.lo <= offset && offset < span.hi
    }
}

fn kind_rank(kind: QueryKind) -> u8 {
    match kind {
        QueryKind::Local => 0,
        QueryKind::Expression => 1,
        QueryKind::Type => 2,
        QueryKind::Pattern => 3,
        QueryKind::Statement => 4,
        QueryKind::Block => 5,
        QueryKind::Item => 6,
    }
}

fn source_text(analysis: &ProjectAnalysis, location: SourceLocation) -> Option<String> {
    let source = analysis.source_map.get(location.source)?;
    source
        .file
        .src
        .get(location.span.lo as usize..location.span.hi as usize)
        .map(str::to_owned)
}

fn trim_item_signature(text: &str) -> &str {
    let boundary = text
        .char_indices()
        .find_map(|(index, ch)| matches!(ch, '{' | ';').then_some(index))
        .unwrap_or(text.len());
    text[..boundary].trim()
}

fn render_ty(analysis: &ProjectAnalysis, ty: &Ty) -> String {
    match ty {
        Ty::Primitive(primitive) => primitive.name().to_string(),
        Ty::Struct(item, args) | Ty::Enum(item, args) => render_nominal(analysis, *item, args),
        Ty::Core(core, args) => {
            let name = match core {
                hir::CoreType::String => "String",
                hir::CoreType::Vec => "Vec",
                hir::CoreType::Box => "Box",
                hir::CoreType::Option => "Option",
                hir::CoreType::Result => "Result",
                hir::CoreType::Range => "Range",
                hir::CoreType::RangeInclusive => "RangeInclusive",
                hir::CoreType::CharsIter => "CharsIter",
                hir::CoreType::SplitIter => "SplitIter",
                hir::CoreType::VecIter => "VecIter",
                hir::CoreType::HashMap => "HashMap",
                hir::CoreType::HashSet => "HashSet",
                hir::CoreType::KeysIter => "KeysIter",
                hir::CoreType::ValuesIter => "ValuesIter",
                hir::CoreType::Iter => "Iter",
                hir::CoreType::MapIter => "MapIter",
                hir::CoreType::FilterIter => "FilterIter",
                hir::CoreType::Random => "Random",
                hir::CoreType::IOError => "IOError",
                hir::CoreType::File => "File",
                hir::CoreType::Ordering => "Ordering",
            };
            render_args(name, args, analysis)
        }
        Ty::Ref { mutable, inner } => format!(
            "&{}{}",
            if *mutable { "mut " } else { "" },
            render_ty(analysis, inner)
        ),
        Ty::Tuple(items) => format!(
            "({})",
            items
                .iter()
                .map(|item| render_ty(analysis, item))
                .collect::<Vec<_>>()
                .join(", ")
        ),
        Ty::Array(item, len) => format!("[{}; {len}]", render_ty(analysis, item)),
        Ty::Slice(item) => format!("[{}]", render_ty(analysis, item)),
        Ty::Fn { params, ret } => format!(
            "fn({}) -> {}",
            params
                .iter()
                .map(|param| render_ty(analysis, param))
                .collect::<Vec<_>>()
                .join(", "),
            render_ty(analysis, ret)
        ),
        Ty::Range(item) => format!("Range<{}>", render_ty(analysis, item)),
        Ty::Never => "!".to_string(),
        Ty::Param(name) => name.clone(),
        Ty::Infer(_) => "_".to_string(),
        Ty::Extension(extension) => match extension.as_ref() {
            ExtensionTy::Tensor(tensor) => format!("{tensor:?}"),
            ExtensionTy::Model(model) => {
                item_name(analysis, model.item_id).unwrap_or_else(|| "<model>".to_string())
            }
            ExtensionTy::ModelError => "<model-error>".to_string(),
        },
        Ty::Error => "<error>".to_string(),
    }
}

fn render_nominal(analysis: &ProjectAnalysis, item: hir::ItemId, args: &[Ty]) -> String {
    let name = item_name(analysis, item).unwrap_or_else(|| "<item>".to_string());
    render_args(&name, args, analysis)
}

fn render_args(name: &str, args: &[Ty], analysis: &ProjectAnalysis) -> String {
    if args.is_empty() {
        name.to_string()
    } else {
        format!(
            "{name}<{}>",
            args.iter()
                .map(|arg| render_ty(analysis, arg))
                .collect::<Vec<_>>()
                .join(", ")
        )
    }
}

fn item_name(analysis: &ProjectAnalysis, item: hir::ItemId) -> Option<String> {
    analysis
        .symbols
        .symbols()
        .iter()
        .find(|symbol| {
            symbol.definition.domain == QueryDomain::Hir
                && symbol.definition.kind == QueryKind::Item
                && symbol.definition.slot == item.0
        })
        .map(|symbol| symbol.name.clone())
}

struct AstIndexer<'a, 'q> {
    analysis: u64,
    ast: &'a Ast,
    sources: &'a SourceMap,
    index: &'q mut QueryIndex,
    seen: HashSet<QueryHandle>,
}

impl<'a, 'q> AstIndexer<'a, 'q> {
    fn new(analysis: u64, ast: &'a Ast, sources: &'a SourceMap, index: &'q mut QueryIndex) -> Self {
        Self {
            analysis,
            ast,
            sources,
            index,
            seen: HashSet::new(),
        }
    }

    fn walk_root(&mut self, root_source: SourceId) {
        match &self.ast.root {
            ast::Root::Program(items) => {
                for item in items {
                    self.item(*item, root_source, None);
                }
            }
            ast::Root::Snippet { stmts, tail } => {
                for stmt in stmts {
                    self.stmt(*stmt, root_source, None);
                }
                if let Some(tail) = tail {
                    self.expr(*tail, root_source, None);
                }
            }
        }
    }

    fn item(
        &mut self,
        id: ast::ItemId,
        fallback_source: SourceId,
        parent_module: Option<QueryHandle>,
    ) {
        let source = self
            .ast
            .item_files
            .get(&id)
            .and_then(|file| self.sources.id_for_name(&file.name))
            .unwrap_or(fallback_source);
        let handle = query_handle(self.analysis, QueryDomain::Syntax, QueryKind::Item, id.0);
        if !self.seen.insert(handle) {
            return;
        }
        let item = self.ast.item(id);
        self.index
            .add_node(handle, source, item.span, Some(handle), parent_module);
        let module = matches!(item.kind, ast::ItemKind::Mod { .. })
            .then_some(handle)
            .or(parent_module);
        match &item.kind {
            ast::ItemKind::Fn(def) => {
                for param in &def.sig.params {
                    self.ty(param.ty, source, Some(handle), module);
                }
                if let ast::RetTy::Ty(ty) = def.sig.ret {
                    self.ty(ty, source, Some(handle), module);
                }
                self.block(def.body, source, Some(handle), module);
            }
            ast::ItemKind::Struct { fields, .. } => {
                for field in fields {
                    self.ty(field.ty, source, Some(handle), module);
                }
            }
            ast::ItemKind::Enum { variants, .. } => {
                for variant in variants {
                    match &variant.kind {
                        ast::VariantKind::Tuple(types) => {
                            for ty in types {
                                self.ty(*ty, source, Some(handle), module);
                            }
                        }
                        ast::VariantKind::Struct(fields) => {
                            for field in fields {
                                self.ty(field.ty, source, Some(handle), module);
                            }
                        }
                        ast::VariantKind::Unit => {}
                    }
                }
            }
            ast::ItemKind::Trait { items, .. } => {
                for trait_item in items {
                    if let ast::TraitItem::Method { sig, body } = trait_item {
                        for param in &sig.params {
                            self.ty(param.ty, source, Some(handle), module);
                        }
                        if let ast::RetTy::Ty(ty) = sig.ret {
                            self.ty(ty, source, Some(handle), module);
                        }
                        if let Some(body) = body {
                            self.block(*body, source, Some(handle), module);
                        }
                    }
                }
            }
            ast::ItemKind::Impl { self_ty, items, .. } => {
                self.ty(*self_ty, source, Some(handle), module);
                for impl_item in items {
                    match impl_item {
                        ast::ImplItem::Fn { def, .. } => {
                            for param in &def.sig.params {
                                self.ty(param.ty, source, Some(handle), module);
                            }
                            if let ast::RetTy::Ty(ty) = def.sig.ret {
                                self.ty(ty, source, Some(handle), module);
                            }
                            self.block(def.body, source, Some(handle), module);
                        }
                        ast::ImplItem::AssocType { ty, .. } => {
                            self.ty(*ty, source, Some(handle), module)
                        }
                    }
                }
            }
            ast::ItemKind::Const { ty, value, .. } => {
                self.ty(*ty, source, Some(handle), module);
                self.expr(*value, source, Some(handle));
            }
            ast::ItemKind::TypeAlias { ty, .. } => self.ty(*ty, source, Some(handle), module),
            ast::ItemKind::Mod {
                items: Some(items), ..
            } => {
                for child in items {
                    self.item(*child, source, module);
                }
            }
            ast::ItemKind::Model(model) => {
                for port in &model.ports {
                    self.ty(port.ty, source, Some(handle), module);
                }
            }
            ast::ItemKind::Use(_) | ast::ItemKind::Mod { items: None, .. } => {}
        }
    }

    fn ty(
        &mut self,
        id: ast::TypeId,
        source: SourceId,
        item: Option<QueryHandle>,
        module: Option<QueryHandle>,
    ) {
        let handle = query_handle(self.analysis, QueryDomain::Syntax, QueryKind::Type, id.0);
        if !self.seen.insert(handle) {
            return;
        }
        let ty = self.ast.ty(id);
        self.index.add_node(handle, source, ty.span, item, module);
        match &ty.kind {
            ast::TypeKind::Array { elem, .. }
            | ast::TypeKind::Slice(elem)
            | ast::TypeKind::Ref { inner: elem, .. } => self.ty(*elem, source, item, module),
            ast::TypeKind::Tuple(types) => {
                for ty in types {
                    self.ty(*ty, source, item, module);
                }
            }
            ast::TypeKind::Fn { params, ret } => {
                for ty in params {
                    self.ty(*ty, source, item, module);
                }
                if let Some(ty) = ret {
                    self.ty(*ty, source, item, module);
                }
            }
            ast::TypeKind::Primitive(_)
            | ast::TypeKind::Path { .. }
            | ast::TypeKind::Never
            | ast::TypeKind::Error => {}
        }
    }

    fn expr(&mut self, id: ast::ExprId, source: SourceId, item: Option<QueryHandle>) {
        let handle = query_handle(
            self.analysis,
            QueryDomain::Syntax,
            QueryKind::Expression,
            id.0,
        );
        if !self.seen.insert(handle) {
            return;
        }
        let expr = self.ast.expr(id);
        let module = item.and_then(|item| self.index.node(item).and_then(|node| node.module));
        self.index.add_node(handle, source, expr.span, item, module);
        match &expr.kind {
            ast::ExprKind::Unary { operand, .. } | ast::ExprKind::Try(operand) => {
                self.expr(*operand, source, item)
            }
            ast::ExprKind::Binary { lhs, rhs, .. }
            | ast::ExprKind::Assign { lhs, rhs, .. }
            | ast::ExprKind::Range {
                lo: lhs, hi: rhs, ..
            }
            | ast::ExprKind::Index {
                base: lhs,
                index: rhs,
            } => {
                self.expr(*lhs, source, item);
                self.expr(*rhs, source, item);
            }
            ast::ExprKind::Cast { expr, ty } => {
                self.expr(*expr, source, item);
                self.ty(*ty, source, item, module);
            }
            ast::ExprKind::Call { callee, args } => {
                self.expr(*callee, source, item);
                for arg in args {
                    self.expr(*arg, source, item);
                }
            }
            ast::ExprKind::Field { base, .. } | ast::ExprKind::TupleField { base, .. } => {
                self.expr(*base, source, item)
            }
            ast::ExprKind::Tuple(values) | ast::ExprKind::Array(values) => {
                for value in values {
                    self.expr(*value, source, item);
                }
            }
            ast::ExprKind::Repeat { value, count } => {
                self.expr(*value, source, item);
                self.expr(*count, source, item);
            }
            ast::ExprKind::StructLit { fields, .. } => {
                for field in fields {
                    if let Some(expr) = field.expr {
                        self.expr(expr, source, item);
                    }
                }
            }
            ast::ExprKind::If {
                cond,
                then_block,
                else_,
            } => {
                self.expr(*cond, source, item);
                self.block(*then_block, source, item, module);
                if let Some(expr) = else_ {
                    self.expr(*expr, source, item);
                }
            }
            ast::ExprKind::Match { scrutinee, arms } => {
                self.expr(*scrutinee, source, item);
                for arm in arms {
                    self.pat(arm.pat, source, item, module);
                    self.expr(arm.body, source, item);
                }
            }
            ast::ExprKind::Loop { body } | ast::ExprKind::Block(body) => {
                self.block(*body, source, item, module)
            }
            ast::ExprKind::While { cond, body } => {
                self.expr(*cond, source, item);
                self.block(*body, source, item, module);
            }
            ast::ExprKind::For { iter, body, .. } => {
                self.expr(*iter, source, item);
                self.block(*body, source, item, module);
            }
            ast::ExprKind::Lit(_) | ast::ExprKind::Path { .. } | ast::ExprKind::Error => {}
        }
    }

    fn stmt(&mut self, id: ast::StmtId, source: SourceId, item: Option<QueryHandle>) {
        let handle = query_handle(
            self.analysis,
            QueryDomain::Syntax,
            QueryKind::Statement,
            id.0,
        );
        if !self.seen.insert(handle) {
            return;
        }
        let stmt = self.ast.stmt(id);
        let module = item.and_then(|item| self.index.node(item).and_then(|node| node.module));
        self.index.add_node(handle, source, stmt.span, item, module);
        match &stmt.kind {
            ast::StmtKind::Expr { expr, .. } => self.expr(*expr, source, item),
            ast::StmtKind::Let { ty, init, .. } => {
                if let Some(ty) = ty {
                    self.ty(*ty, source, item, module);
                }
                if let Some(init) = init {
                    self.expr(*init, source, item);
                }
            }
            ast::StmtKind::Return(expr) | ast::StmtKind::Break(expr) => {
                if let Some(expr) = expr {
                    self.expr(*expr, source, item);
                }
            }
            ast::StmtKind::Item(item_id) => self.item(*item_id, source, module),
            ast::StmtKind::Empty | ast::StmtKind::Continue | ast::StmtKind::Error => {}
        }
    }

    fn block(
        &mut self,
        id: ast::BlockId,
        source: SourceId,
        item: Option<QueryHandle>,
        module: Option<QueryHandle>,
    ) {
        let handle = query_handle(self.analysis, QueryDomain::Syntax, QueryKind::Block, id.0);
        if !self.seen.insert(handle) {
            return;
        }
        let block = self.ast.block(id);
        self.index
            .add_node(handle, source, block.span, item, module);
        for stmt in &block.stmts {
            self.stmt(*stmt, source, item);
        }
        if let Some(tail) = block.tail {
            self.expr(tail, source, item);
        }
    }

    fn pat(
        &mut self,
        id: ast::PatId,
        source: SourceId,
        item: Option<QueryHandle>,
        module: Option<QueryHandle>,
    ) {
        let handle = query_handle(self.analysis, QueryDomain::Syntax, QueryKind::Pattern, id.0);
        if !self.seen.insert(handle) {
            return;
        }
        let pat = self.ast.pat(id);
        self.index.add_node(handle, source, pat.span, item, module);
        match &pat.kind {
            ast::PatKind::TupleVariant { pats, .. }
            | ast::PatKind::Tuple(pats)
            | ast::PatKind::Array(pats) => {
                for pat in pats {
                    self.pat(*pat, source, item, module);
                }
            }
            ast::PatKind::Struct { fields, .. } => {
                for field in fields {
                    if let Some(pat) = field.pat {
                        self.pat(pat, source, item, module);
                    }
                }
            }
            ast::PatKind::Lit(_)
            | ast::PatKind::Wild
            | ast::PatKind::Binding(_)
            | ast::PatKind::Path(_) => {}
        }
    }
}

struct HirIndexer<'a, 'q> {
    analysis: u64,
    hir: &'a Hir,
    sources: &'a SourceMap,
    index: &'q mut QueryIndex,
    seen: HashSet<QueryHandle>,
}

impl<'a, 'q> HirIndexer<'a, 'q> {
    fn new(analysis: u64, hir: &'a Hir, sources: &'a SourceMap, index: &'q mut QueryIndex) -> Self {
        Self {
            analysis,
            hir,
            sources,
            index,
            seen: HashSet::new(),
        }
    }

    fn walk_root(&mut self, root_source: SourceId) {
        match &self.hir.root {
            hir::Root::Program(items) => {
                for item in items {
                    self.item(*item, root_source, None);
                }
            }
            hir::Root::Snippet { stmts, tail } => {
                for stmt in stmts {
                    self.stmt(*stmt, root_source, None, None);
                }
                if let Some(tail) = tail {
                    self.expr(*tail, root_source, None, None);
                }
            }
        }
    }

    fn item(
        &mut self,
        id: hir::ItemId,
        fallback_source: SourceId,
        parent_module: Option<QueryHandle>,
    ) {
        let source = self
            .hir
            .item_files
            .get(&id)
            .and_then(|file| self.sources.id_for_name(&file.name))
            .unwrap_or(fallback_source);
        let handle = query_handle(self.analysis, QueryDomain::Hir, QueryKind::Item, id.0);
        if !self.seen.insert(handle) {
            return;
        }
        let item = self.hir.item(id);
        self.index
            .add_node(handle, source, item.span, Some(handle), parent_module);
        let module = matches!(item.kind, hir::ItemKind::Mod { .. })
            .then_some(handle)
            .or(parent_module);
        if let Some(name) = item_name_span(item) {
            self.index.define(handle, source, name);
        }
        match &item.kind {
            hir::ItemKind::Fn(def) => self.fn_def(def, source, handle, module),
            hir::ItemKind::Struct { fields, .. } => {
                for field in fields {
                    self.ty(field.ty, source, Some(handle), module);
                }
            }
            hir::ItemKind::Enum { variants, .. } => {
                for variant in variants {
                    match &variant.kind {
                        hir::VariantKind::Tuple(types) => {
                            for ty in types {
                                self.ty(*ty, source, Some(handle), module);
                            }
                        }
                        hir::VariantKind::Struct(fields) => {
                            for field in fields {
                                self.ty(field.ty, source, Some(handle), module);
                            }
                        }
                        hir::VariantKind::Unit => {}
                    }
                }
            }
            hir::ItemKind::Trait { items, .. } => {
                for trait_item in items {
                    if let hir::TraitItem::Method { sig, body } = trait_item {
                        self.fn_sig(sig, source, handle, module);
                        if let Some(body) = body {
                            self.block(*body, source, Some(handle), module);
                        }
                    }
                }
            }
            hir::ItemKind::Impl { self_ty, items, .. } => {
                self.ty(*self_ty, source, Some(handle), module);
                for impl_item in items {
                    match impl_item {
                        hir::ImplItem::Fn { def, .. } => self.fn_def(def, source, handle, module),
                        hir::ImplItem::AssocType { ty, .. } => {
                            self.ty(*ty, source, Some(handle), module)
                        }
                    }
                }
            }
            hir::ItemKind::Const { ty, value, .. } => {
                self.ty(*ty, source, Some(handle), module);
                self.expr(*value, source, Some(handle), module);
            }
            hir::ItemKind::TypeAlias { ty, .. } => self.ty(*ty, source, Some(handle), module),
            hir::ItemKind::Mod {
                items: Some(items), ..
            } => {
                for child in items {
                    self.item(*child, source, module);
                }
            }
            hir::ItemKind::Model(model) => {
                for port in &model.ports {
                    self.ty(port.ty, source, Some(handle), module);
                }
            }
            hir::ItemKind::Use(_) | hir::ItemKind::Mod { items: None, .. } => {}
        }
    }

    fn fn_def(
        &mut self,
        def: &hir::FnDef,
        source: SourceId,
        item: QueryHandle,
        module: Option<QueryHandle>,
    ) {
        self.fn_sig(&def.sig, source, item, module);
        self.block(def.body, source, Some(item), module);
    }

    fn fn_sig(
        &mut self,
        sig: &hir::FnSig,
        source: SourceId,
        item: QueryHandle,
        module: Option<QueryHandle>,
    ) {
        if let Some(local) = sig.receiver_local {
            self.define_local(local, source, sig.name, Some(item), module);
        }
        for param in &sig.params {
            self.define_local(param.local, source, param.name, Some(item), module);
            self.ty(param.ty, source, Some(item), module);
        }
        if let hir::RetTy::Ty(ty) = sig.ret {
            self.ty(ty, source, Some(item), module);
        }
    }

    fn ty(
        &mut self,
        id: hir::TypeId,
        source: SourceId,
        item: Option<QueryHandle>,
        module: Option<QueryHandle>,
    ) {
        let handle = query_handle(self.analysis, QueryDomain::Hir, QueryKind::Type, id.0);
        if !self.seen.insert(handle) {
            return;
        }
        let ty = self.hir.ty(id);
        self.index.add_node(handle, source, ty.span, item, module);
        match &ty.kind {
            hir::TypeKind::Path { path, res, .. } => self.resolution(*res, source, path.span),
            hir::TypeKind::Array { elem, .. }
            | hir::TypeKind::Slice(elem)
            | hir::TypeKind::Ref { inner: elem, .. } => self.ty(*elem, source, item, module),
            hir::TypeKind::Tuple(types) => {
                for ty in types {
                    self.ty(*ty, source, item, module);
                }
            }
            hir::TypeKind::Fn { params, ret } => {
                for ty in params {
                    self.ty(*ty, source, item, module);
                }
                if let Some(ty) = ret {
                    self.ty(*ty, source, item, module);
                }
            }
            hir::TypeKind::Primitive(_) | hir::TypeKind::Never | hir::TypeKind::Error => {}
        }
    }

    fn expr(
        &mut self,
        id: hir::ExprId,
        source: SourceId,
        item: Option<QueryHandle>,
        module: Option<QueryHandle>,
    ) {
        let handle = query_handle(self.analysis, QueryDomain::Hir, QueryKind::Expression, id.0);
        if !self.seen.insert(handle) {
            return;
        }
        let expr = self.hir.expr(id);
        self.index.add_node(handle, source, expr.span, item, module);
        match &expr.kind {
            hir::ExprKind::Path { path, res, .. } => self.resolution(*res, source, path.span),
            hir::ExprKind::Unary { operand, .. } | hir::ExprKind::Try(operand) => {
                self.expr(*operand, source, item, module)
            }
            hir::ExprKind::Binary { lhs, rhs, .. }
            | hir::ExprKind::Assign { lhs, rhs, .. }
            | hir::ExprKind::Range {
                lo: lhs, hi: rhs, ..
            }
            | hir::ExprKind::Index {
                base: lhs,
                index: rhs,
            } => {
                self.expr(*lhs, source, item, module);
                self.expr(*rhs, source, item, module);
            }
            hir::ExprKind::Cast { expr, ty } => {
                self.expr(*expr, source, item, module);
                self.ty(*ty, source, item, module);
            }
            hir::ExprKind::Call { callee, args } => {
                self.expr(*callee, source, item, module);
                for arg in args {
                    self.expr(*arg, source, item, module);
                }
            }
            hir::ExprKind::Field { base, .. } | hir::ExprKind::TupleField { base, .. } => {
                self.expr(*base, source, item, module)
            }
            hir::ExprKind::Tuple(values) | hir::ExprKind::Array(values) => {
                for value in values {
                    self.expr(*value, source, item, module);
                }
            }
            hir::ExprKind::Repeat { value, count } => {
                self.expr(*value, source, item, module);
                self.expr(*count, source, item, module);
            }
            hir::ExprKind::StructLit {
                path, res, fields, ..
            } => {
                self.resolution(*res, source, path.span);
                for field in fields {
                    if let Some(expr) = field.expr {
                        self.expr(expr, source, item, module);
                    }
                }
            }
            hir::ExprKind::If {
                cond,
                then_block,
                else_,
            } => {
                self.expr(*cond, source, item, module);
                self.block(*then_block, source, item, module);
                if let Some(expr) = else_ {
                    self.expr(*expr, source, item, module);
                }
            }
            hir::ExprKind::Match { scrutinee, arms } => {
                self.expr(*scrutinee, source, item, module);
                for arm in arms {
                    self.pat(arm.pat, source, item, module);
                    self.expr(arm.body, source, item, module);
                }
            }
            hir::ExprKind::Loop { body } | hir::ExprKind::Block(body) => {
                self.block(*body, source, item, module)
            }
            hir::ExprKind::While { cond, body } => {
                self.expr(*cond, source, item, module);
                self.block(*body, source, item, module);
            }
            hir::ExprKind::For {
                var,
                local,
                iter,
                body,
            } => {
                self.define_local(*local, source, *var, item, module);
                self.expr(*iter, source, item, module);
                self.block(*body, source, item, module);
            }
            hir::ExprKind::Lit(_) | hir::ExprKind::Error => {}
        }
    }

    fn stmt(
        &mut self,
        id: hir::StmtId,
        source: SourceId,
        item: Option<QueryHandle>,
        module: Option<QueryHandle>,
    ) {
        let handle = query_handle(self.analysis, QueryDomain::Hir, QueryKind::Statement, id.0);
        if !self.seen.insert(handle) {
            return;
        }
        let stmt = self.hir.stmt(id);
        self.index.add_node(handle, source, stmt.span, item, module);
        match &stmt.kind {
            hir::StmtKind::Expr { expr, .. } => self.expr(*expr, source, item, module),
            hir::StmtKind::Let {
                name,
                local,
                ty,
                init,
                ..
            } => {
                self.define_local(*local, source, *name, item, module);
                if let Some(ty) = ty {
                    self.ty(*ty, source, item, module);
                }
                if let Some(init) = init {
                    self.expr(*init, source, item, module);
                }
            }
            hir::StmtKind::Return(expr) | hir::StmtKind::Break(expr) => {
                if let Some(expr) = expr {
                    self.expr(*expr, source, item, module);
                }
            }
            hir::StmtKind::Item(item_id) => self.item(*item_id, source, module),
            hir::StmtKind::Empty | hir::StmtKind::Continue | hir::StmtKind::Error => {}
        }
    }

    fn block(
        &mut self,
        id: hir::BlockId,
        source: SourceId,
        item: Option<QueryHandle>,
        module: Option<QueryHandle>,
    ) {
        let handle = query_handle(self.analysis, QueryDomain::Hir, QueryKind::Block, id.0);
        if !self.seen.insert(handle) {
            return;
        }
        let block = self.hir.block(id);
        self.index
            .add_node(handle, source, block.span, item, module);
        for stmt in &block.stmts {
            self.stmt(*stmt, source, item, module);
        }
        if let Some(tail) = block.tail {
            self.expr(tail, source, item, module);
        }
    }

    fn pat(
        &mut self,
        id: hir::PatId,
        source: SourceId,
        item: Option<QueryHandle>,
        module: Option<QueryHandle>,
    ) {
        let handle = query_handle(self.analysis, QueryDomain::Hir, QueryKind::Pattern, id.0);
        if !self.seen.insert(handle) {
            return;
        }
        let pat = self.hir.pat(id);
        self.index.add_node(handle, source, pat.span, item, module);
        match &pat.kind {
            hir::PatKind::Binding { name, local } => {
                self.define_local(*local, source, *name, item, module)
            }
            hir::PatKind::Path { path, res }
            | hir::PatKind::TupleVariant { path, res, .. }
            | hir::PatKind::Struct { path, res, .. } => {
                self.resolution(*res, source, path.span);
                match &pat.kind {
                    hir::PatKind::TupleVariant { pats, .. } => {
                        for pat in pats {
                            self.pat(*pat, source, item, module);
                        }
                    }
                    hir::PatKind::Struct { fields, .. } => {
                        for field in fields {
                            if let Some(local) = field.local {
                                self.define_local(local, source, field.name, item, module);
                            }
                            if let Some(pat) = field.pat {
                                self.pat(pat, source, item, module);
                            }
                        }
                    }
                    _ => {}
                }
            }
            hir::PatKind::Tuple(pats) | hir::PatKind::Array(pats) => {
                for pat in pats {
                    self.pat(*pat, source, item, module);
                }
            }
            hir::PatKind::Lit(_) | hir::PatKind::Wild | hir::PatKind::Error => {}
        }
    }

    fn define_local(
        &mut self,
        local: hir::LocalId,
        source: SourceId,
        span: Span,
        item: Option<QueryHandle>,
        module: Option<QueryHandle>,
    ) {
        let handle = query_handle(self.analysis, QueryDomain::Hir, QueryKind::Local, local.0);
        self.index.add_node(handle, source, span, item, module);
        self.index.define(handle, source, span);
    }

    fn resolution(&mut self, res: Res, source: SourceId, span: Span) {
        let symbol = match res {
            Res::Local(local) | Res::SelfValue(local) => Some(query_handle(
                self.analysis,
                QueryDomain::Hir,
                QueryKind::Local,
                local.0,
            )),
            Res::Item(item)
            | Res::Variant(item, _)
            | Res::TraitMember(item, _)
            | Res::AssociatedFn(item, _)
            | Res::ModelLoad(item) => Some(query_handle(
                self.analysis,
                QueryDomain::Hir,
                QueryKind::Item,
                item.0,
            )),
            Res::Primitive(_)
            | Res::SelfType
            | Res::SelfAssoc(_)
            | Res::TypeParam
            | Res::ParamAssoc(_, _)
            | Res::Builtin(_)
            | Res::CoreTrait(_)
            | Res::CoreTraitMember(_, _)
            | Res::CoreType(_)
            | Res::Err => None,
        };
        if let Some(symbol) = symbol {
            self.index.reference(symbol, source, span);
        }
    }
}

fn item_name_span(item: &hir::ItemNode) -> Option<Span> {
    match &item.kind {
        hir::ItemKind::Fn(def) => Some(def.sig.name),
        hir::ItemKind::Struct { name, .. }
        | hir::ItemKind::Enum { name, .. }
        | hir::ItemKind::Trait { name, .. }
        | hir::ItemKind::Const { name, .. }
        | hir::ItemKind::TypeAlias { name, .. }
        | hir::ItemKind::Mod { name, .. } => Some(*name),
        hir::ItemKind::Model(model) => Some(model.name),
        hir::ItemKind::Impl { .. } | hir::ItemKind::Use(_) => None,
    }
}
