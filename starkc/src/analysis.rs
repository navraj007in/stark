//! Shared project analysis for compiler and tooling consumers.
//!
//! This is the semantic entry point. Consumers that only need syntax (the
//! formatter and `starkc parse`) may call the parser directly; compilation,
//! validation, and tooling should use [`analyze_project`].

use crate::ast::Ast;
use crate::diag::{Diagnostic, Severity};
use crate::hir::{Hir, ItemKind, Res};
use crate::options::{ExtensionSet, LanguageOptions};
use crate::package::PackageGraph;
use crate::parser::{parse_package_graph, parse_with_options, ParseMode};
use crate::source::{SourceFile, Span};
use crate::typecheck::{analyze_with_options, TypeTables};
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

static NEXT_ANALYSIS_ID: AtomicU64 = AtomicU64::new(1);

/// Input accepted by the shared compiler pipeline.
#[derive(Clone)]
pub enum ProjectInput {
    Source {
        file: Arc<SourceFile>,
        mode: ParseMode,
    },
    Package(Arc<PackageGraph>),
}

impl ProjectInput {
    pub fn source(file: Arc<SourceFile>, mode: ParseMode) -> Self {
        Self::Source { file, mode }
    }

    pub fn program(file: Arc<SourceFile>) -> Self {
        Self::source(file, ParseMode::Program)
    }

    pub fn snippet(file: Arc<SourceFile>) -> Self {
        Self::source(file, ParseMode::Snippet)
    }

    pub fn package(graph: PackageGraph) -> Self {
        Self::Package(Arc::new(graph))
    }
}

/// Stable source identity within one analysis result.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SourceId(u32);

/// How a source file entered an analysis.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SourceProvenance {
    Root { package: Option<String> },
    Module { package: Option<String> },
}

#[derive(Clone)]
pub struct SourceRecord {
    pub id: SourceId,
    pub file: Arc<SourceFile>,
    pub provenance: SourceProvenance,
}

/// All physical sources participating in the project analysis.
#[derive(Default)]
pub struct SourceMap {
    files: Vec<SourceRecord>,
    by_name: HashMap<String, SourceId>,
}

impl SourceMap {
    pub fn files(&self) -> &[SourceRecord] {
        &self.files
    }

    pub fn get(&self, id: SourceId) -> Option<&SourceRecord> {
        self.files.get(id.0 as usize)
    }

    pub fn id_for_name(&self, name: &str) -> Option<SourceId> {
        self.by_name.get(name).copied()
    }
}

/// Entity category encoded by a stable query handle.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum QueryKind {
    Item,
    Expression,
    Type,
    Statement,
    Pattern,
    Block,
    Local,
}

/// Opaque identity stable for the lifetime of its [`ProjectAnalysis`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct QueryHandle {
    analysis: u64,
    kind: QueryKind,
    slot: u32,
}

impl QueryHandle {
    pub fn kind(self) -> QueryKind {
        self.kind
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SymbolKind {
    Function,
    Struct,
    Enum,
    Trait,
    Constant,
    TypeAlias,
    Module,
    Model,
}

pub struct Symbol {
    pub name: String,
    pub kind: SymbolKind,
    pub definition: QueryHandle,
    pub source: SourceId,
    pub span: Span,
}

#[derive(Default)]
pub struct SymbolIndex {
    symbols: Vec<Symbol>,
    by_name: HashMap<String, Vec<usize>>,
}

impl SymbolIndex {
    pub fn symbols(&self) -> &[Symbol] {
        &self.symbols
    }

    pub fn named(&self, name: &str) -> impl Iterator<Item = &Symbol> {
        self.by_name
            .get(name)
            .into_iter()
            .flatten()
            .filter_map(|index| self.symbols.get(*index))
    }
}

/// Name-resolution facts keyed by stable handles.
#[derive(Default)]
pub struct ResolutionTables {
    pub expressions: HashMap<QueryHandle, Res>,
    pub types: HashMap<QueryHandle, Res>,
}

/// One owned, coherent view of a project through semantic analysis.
pub struct ProjectAnalysis {
    id: u64,
    pub options: LanguageOptions,
    pub extensions: ExtensionSet,
    pub package_graph: Option<Arc<PackageGraph>>,
    pub root_file: Arc<SourceFile>,
    pub source_map: SourceMap,
    pub ast: Ast,
    pub hir: Option<Hir>,
    pub resolutions: ResolutionTables,
    pub type_tables: Option<TypeTables>,
    pub diagnostics: Vec<Diagnostic>,
    pub symbols: SymbolIndex,
}

impl ProjectAnalysis {
    pub fn has_errors(&self) -> bool {
        self.diagnostics
            .iter()
            .any(|diag| diag.severity == Severity::Error)
    }

    pub(crate) fn handle(&self, kind: QueryKind, slot: u32) -> Option<QueryHandle> {
        let len = match kind {
            QueryKind::Item => self.hir.as_ref()?.items.len(),
            QueryKind::Expression => self.hir.as_ref()?.exprs.len(),
            QueryKind::Type => self.hir.as_ref()?.types.len(),
            QueryKind::Statement => self.hir.as_ref()?.stmts.len(),
            QueryKind::Pattern => self.hir.as_ref()?.pats.len(),
            QueryKind::Block => self.hir.as_ref()?.blocks.len(),
            QueryKind::Local => self
                .type_tables
                .as_ref()
                .map_or(0, |tables| tables.local_types.len()),
        };
        ((slot as usize) < len).then_some(QueryHandle {
            analysis: self.id,
            kind,
            slot,
        })
    }

    pub fn owns_handle(&self, handle: QueryHandle) -> bool {
        handle.analysis == self.id && self.handle(handle.kind, handle.slot) == Some(handle)
    }
}

/// Run the canonical parse → resolve → typecheck pipeline.
pub fn analyze_project(input: ProjectInput, options: LanguageOptions) -> ProjectAnalysis {
    let id = NEXT_ANALYSIS_ID.fetch_add(1, Ordering::Relaxed);
    let (root_file, package_graph, ast, mut diagnostics) = match input {
        ProjectInput::Source { file, mode } => {
            let (ast, diagnostics) = parse_with_options(&file, mode, options);
            (file, None, ast, diagnostics)
        }
        ProjectInput::Package(graph) => {
            let package = &graph.packages[&graph.root_package_name];
            match std::fs::read_to_string(&package.entry) {
                Ok(source) => {
                    let file = Arc::new(SourceFile::new(
                        package.entry.to_string_lossy().into_owned(),
                        source,
                    ));
                    let (ast, diagnostics) = parse_package_graph(&graph, options);
                    (file, Some(graph), ast, diagnostics)
                }
                Err(error) => {
                    let file = Arc::new(SourceFile::new(
                        package.entry.to_string_lossy().into_owned(),
                        "",
                    ));
                    let diagnostic = Diagnostic::error(
                        format!("failed to read entry file: {error}"),
                        Span { lo: 0, hi: 0 },
                    )
                    .with_file(file.clone());
                    (file, Some(graph), Ast::default(), vec![diagnostic])
                }
            }
        }
    };

    let mut hir = None;
    let mut type_tables = None;
    if !has_errors(&diagnostics) {
        let (resolved, resolve_diagnostics) =
            crate::resolve::resolve_with_options(&ast, root_file.clone(), options);
        diagnostics.extend(resolve_diagnostics);
        if !has_errors(&diagnostics) {
            let checked = analyze_with_options(&resolved, root_file.clone(), options);
            diagnostics.extend(checked.diagnostics);
            type_tables = Some(checked.tables);
        }
        hir = Some(resolved);
    }

    let source_map = build_source_map(&root_file, &ast, hir.as_ref(), package_graph.as_deref());
    let resolutions = hir
        .as_ref()
        .map(|hir| build_resolutions(id, hir))
        .unwrap_or_default();
    let symbols = hir
        .as_ref()
        .map(|hir| build_symbols(id, hir, &ast, &source_map))
        .unwrap_or_default();

    ProjectAnalysis {
        id,
        options,
        extensions: options.extensions,
        package_graph,
        root_file,
        source_map,
        ast,
        hir,
        resolutions,
        type_tables,
        diagnostics,
        symbols,
    }
}

fn has_errors(diagnostics: &[Diagnostic]) -> bool {
    diagnostics
        .iter()
        .any(|diag| diag.severity == Severity::Error)
}

fn build_source_map(
    root: &Arc<SourceFile>,
    ast: &Ast,
    hir: Option<&Hir>,
    graph: Option<&PackageGraph>,
) -> SourceMap {
    let mut files = vec![root.clone()];
    files.extend(ast.item_files.values().cloned());
    if let Some(hir) = hir {
        files.extend(hir.item_files.values().cloned());
    }
    let mut seen = HashSet::new();
    files.retain(|file| seen.insert(file.name.clone()));

    let root_package = graph.map(|graph| graph.root_package_name.clone());
    let mut map = SourceMap::default();
    for file in files {
        let id = SourceId(map.files.len() as u32);
        let package = graph.and_then(|graph| {
            graph
                .packages
                .iter()
                .find(|(_, package)| {
                    package
                        .entry
                        .parent()
                        .is_some_and(|parent| file.name.starts_with(&*parent.to_string_lossy()))
                })
                .map(|(name, _)| name.clone())
        });
        let provenance = if file.name == root.name {
            SourceProvenance::Root {
                package: root_package.clone(),
            }
        } else {
            SourceProvenance::Module { package }
        };
        map.by_name.insert(file.name.clone(), id);
        map.files.push(SourceRecord {
            id,
            file,
            provenance,
        });
    }
    map
}

fn handle(analysis: u64, kind: QueryKind, slot: usize) -> QueryHandle {
    QueryHandle {
        analysis,
        kind,
        slot: slot as u32,
    }
}

fn build_resolutions(analysis: u64, hir: &Hir) -> ResolutionTables {
    let mut tables = ResolutionTables::default();
    for (slot, expr) in hir.exprs.iter().enumerate() {
        if let crate::hir::ExprKind::Path { res, .. } = expr.kind {
            tables
                .expressions
                .insert(handle(analysis, QueryKind::Expression, slot), res);
        }
    }
    for (slot, ty) in hir.types.iter().enumerate() {
        if let crate::hir::TypeKind::Path { res, .. } = ty.kind {
            tables
                .types
                .insert(handle(analysis, QueryKind::Type, slot), res);
        }
    }
    tables
}

fn build_symbols(analysis: u64, hir: &Hir, ast: &Ast, sources: &SourceMap) -> SymbolIndex {
    let mut index = SymbolIndex::default();
    for (slot, item) in hir.items.iter().enumerate() {
        let (kind, span) = match &item.kind {
            ItemKind::Fn(def) => (SymbolKind::Function, def.sig.name),
            ItemKind::Struct { name, .. } => (SymbolKind::Struct, *name),
            ItemKind::Enum { name, .. } => (SymbolKind::Enum, *name),
            ItemKind::Trait { name, .. } => (SymbolKind::Trait, *name),
            ItemKind::Const { name, .. } => (SymbolKind::Constant, *name),
            ItemKind::TypeAlias { name, .. } => (SymbolKind::TypeAlias, *name),
            ItemKind::Mod { name, .. } => (SymbolKind::Module, *name),
            ItemKind::Model(model) => (SymbolKind::Model, model.name),
            ItemKind::Impl { .. } | ItemKind::Use(_) => continue,
        };
        let item_id = crate::hir::ItemId(slot as u32);
        let file = hir
            .item_files
            .get(&item_id)
            .map(Arc::as_ref)
            .unwrap_or_else(|| sources.files[0].file.as_ref());
        let name = ast.synthetic_spans.get(&span).cloned().or_else(|| {
            file.src
                .get(span.lo as usize..span.hi as usize)
                .map(str::to_owned)
        });
        let Some(name) = name else { continue };
        let Some(source) = sources.id_for_name(&file.name) else {
            continue;
        };
        let symbol = Symbol {
            name: name.clone(),
            kind,
            definition: handle(analysis, QueryKind::Item, slot),
            source,
            span,
        };
        let symbol_slot = index.symbols.len();
        index.symbols.push(symbol);
        index.by_name.entry(name).or_default().push(symbol_slot);
    }
    index
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn analysis_owns_semantic_results_and_stable_handles() {
        let file = Arc::new(SourceFile::new("main.stark", "fn main() { let x = 1; }"));
        let analysis = analyze_project(ProjectInput::program(file), LanguageOptions::CORE);
        assert!(!analysis.has_errors(), "{:?}", analysis.diagnostics);
        assert!(analysis.hir.is_some());
        assert!(analysis.type_tables.is_some());
        assert_eq!(analysis.source_map.files().len(), 1);
        let item = analysis.handle(QueryKind::Item, 0).unwrap();
        assert!(analysis.owns_handle(item));
        assert_eq!(analysis.symbols.named("main").count(), 1);
    }

    #[test]
    fn handles_do_not_cross_analysis_sessions() {
        let file = Arc::new(SourceFile::new("main.stark", "fn main() {}"));
        let first = analyze_project(ProjectInput::program(file.clone()), LanguageOptions::CORE);
        let second = analyze_project(ProjectInput::program(file), LanguageOptions::CORE);
        assert!(!second.owns_handle(first.handle(QueryKind::Item, 0).unwrap()));
    }

    #[test]
    fn parse_errors_preserve_partial_analysis() {
        let file = Arc::new(SourceFile::new("broken.stark", "fn main("));
        let analysis = analyze_project(ProjectInput::program(file), LanguageOptions::CORE);
        assert!(analysis.has_errors());
        assert!(analysis.hir.is_none());
        assert!(analysis.type_tables.is_none());
        assert_eq!(analysis.source_map.files().len(), 1);
    }
}
