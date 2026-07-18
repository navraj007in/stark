//! Shared project analysis for compiler and tooling consumers.
//!
//! This is the semantic entry point. Consumers that only need syntax (the
//! formatter and `starkc parse`) may call the parser directly; compilation,
//! validation, and tooling should use [`analyze_project`].

use crate::ast::Ast;
use crate::diag::{Diagnostic, DiagnosticBatch, Severity};
use crate::hir::{Hir, ItemKind, Res};
use crate::options::{ExtensionSet, LanguageOptions};
use crate::package::PackageGraph;
use crate::parser::{parse_package_graph, parse_with_options, ParseMode};
use crate::source::{SourceFile, Span};
use crate::typecheck::{analyze_with_options, TypeTables};
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

mod query;
pub use query::{EnclosingContext, SourceLocation};

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
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SourceId(u32);

impl SourceId {
    pub fn as_u32(self) -> u32 {
        self.0
    }
}

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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum QueryDomain {
    Syntax,
    Hir,
}

/// Opaque identity stable for the lifetime of its [`ProjectAnalysis`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct QueryHandle {
    pub(crate) analysis: u64,
    pub(crate) domain: QueryDomain,
    pub(crate) kind: QueryKind,
    pub(crate) slot: u32,
}

impl QueryHandle {
    pub fn domain(self) -> QueryDomain {
        self.domain
    }

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
    queries: query::QueryIndex,
}

impl ProjectAnalysis {
    pub fn has_errors(&self) -> bool {
        self.diagnostics
            .iter()
            .any(|diag| diag.severity == Severity::Error)
    }

    pub fn owns_handle(&self, handle: QueryHandle) -> bool {
        handle.analysis == self.id && self.queries.contains(handle)
    }

    pub fn syntax_at(&self, source: SourceId, offset: u32) -> Option<QueryHandle> {
        self.queries.node_at(QueryDomain::Syntax, source, offset)
    }

    pub fn hir_at(&self, source: SourceId, offset: u32) -> Option<QueryHandle> {
        self.queries.node_at(QueryDomain::Hir, source, offset)
    }

    pub fn symbol_at(&self, source: SourceId, offset: u32) -> Option<QueryHandle> {
        self.queries.symbol_at(source, offset)
    }

    pub fn definition(&self, symbol: QueryHandle) -> Option<SourceLocation> {
        self.owns_handle(symbol)
            .then(|| self.queries.definition(symbol))
            .flatten()
    }

    pub fn references(&self, symbol: QueryHandle) -> Vec<SourceLocation> {
        if self.owns_handle(symbol) {
            self.queries.references(symbol)
        } else {
            Vec::new()
        }
    }

    pub fn type_of(&self, handle: QueryHandle) -> Option<String> {
        self.owns_handle(handle)
            .then(|| self.queries.type_of(self, handle))
            .flatten()
    }

    pub fn signature(&self, handle: QueryHandle) -> Option<String> {
        self.owns_handle(handle)
            .then(|| self.queries.signature(self, handle))
            .flatten()
    }

    pub fn enclosing(&self, handle: QueryHandle) -> Option<EnclosingContext> {
        self.owns_handle(handle)
            .then(|| self.queries.enclosing(self, handle))
            .flatten()
    }

    pub fn public_items(&self) -> Vec<&Symbol> {
        let Some(hir) = self.hir.as_ref() else {
            return Vec::new();
        };
        self.symbols
            .symbols()
            .iter()
            .filter(|symbol| {
                symbol.definition.domain == QueryDomain::Hir
                    && symbol.definition.kind == QueryKind::Item
                    && matches!(
                        hir.items[symbol.definition.slot as usize].vis,
                        Some(crate::ast::Vis::Pub)
                    )
            })
            .collect()
    }

    pub fn document_symbols(&self, source: SourceId) -> Vec<&Symbol> {
        self.symbols
            .symbols()
            .iter()
            .filter(|symbol| symbol.source == source)
            .collect()
    }

    pub fn workspace_symbols(&self, query: &str) -> Vec<&Symbol> {
        let query = query.to_ascii_lowercase();
        self.symbols
            .symbols()
            .iter()
            .filter(|symbol| symbol.name.to_ascii_lowercase().contains(&query))
            .collect()
    }

    pub fn diagnostic_batch(&self, source_versions: &HashMap<SourceId, i64>) -> DiagnosticBatch {
        let root = self
            .source_map
            .id_for_name(&self.root_file.name)
            .expect("analysis root source must exist in its source map");
        DiagnosticBatch::from_compiler_diagnostics(
            &self.diagnostics,
            &self.source_map,
            root,
            source_versions,
            if self.options.tensor() {
                vec!["tensor".to_string()]
            } else {
                Vec::new()
            },
        )
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
    let queries = query::QueryIndex::build(id, &ast, hir.as_ref(), &source_map, &symbols);

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
        queries,
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
    let mut non_root = files.split_off(1);
    non_root.sort_by(|left, right| left.name.cmp(&right.name));
    files.extend(non_root);

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
        domain: QueryDomain::Hir,
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
        let item = analysis.symbols.named("main").next().unwrap().definition;
        assert!(analysis.owns_handle(item));
        assert_eq!(analysis.symbols.named("main").count(), 1);
    }

    #[test]
    fn handles_do_not_cross_analysis_sessions() {
        let file = Arc::new(SourceFile::new("main.stark", "fn main() {}"));
        let first = analyze_project(ProjectInput::program(file.clone()), LanguageOptions::CORE);
        let second = analyze_project(ProjectInput::program(file), LanguageOptions::CORE);
        let first_item = first.symbols.named("main").next().unwrap().definition;
        assert!(!second.owns_handle(first_item));
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

    #[test]
    fn diagnostic_json_matches_the_schema_v1_golden() {
        let file = Arc::new(SourceFile::new("golden.stark", "fn main() { missing; }\n"));
        let analysis = analyze_project(ProjectInput::program(file), LanguageOptions::with_tensor());
        let source = analysis.source_map.id_for_name("golden.stark").unwrap();
        let batch = analysis.diagnostic_batch(&HashMap::from([(source, 9)]));
        assert_eq!(
            batch.to_json(&analysis.source_map),
            concat!(
                "{\"schemaVersion\":1,\"tool\":\"starkc\",\"toolVersion\":\"0.1.0\",",
                "\"extensions\":[\"tensor\"],",
                "\"sources\":[{\"sourceId\":0,\"file\":\"golden.stark\",",
                "\"kind\":\"root\",\"package\":null}],",
                "\"diagnostics\":[{\"code\":\"E0200\",\"severity\":\"error\",",
                "\"message\":\"undefined variable 'missing'\",",
                "\"primary\":{\"sourceId\":0,\"file\":\"golden.stark\",",
                "\"span\":{\"startByte\":12,\"endByte\":19},\"label\":null},",
                "\"related\":[],\"notes\":[],\"help\":[],\"sourceVersion\":9,",
                "\"ruleId\":null,\"deviationId\":null}]}"
            )
        );
    }

    #[test]
    fn position_symbol_type_and_enclosing_queries_share_stable_handles() {
        let source = "pub struct User { id: Int32 }\n\
                      fn helper(x: Int32) -> Int32 { x }\n\
                      pub fn main() { let value = helper(1); println(value); }\n";
        let file = Arc::new(SourceFile::new("main.stark", source));
        let analysis = analyze_project(ProjectInput::program(file), LanguageOptions::CORE);
        assert!(!analysis.has_errors(), "{:?}", analysis.diagnostics);
        let source_id = analysis.source_map.id_for_name("main.stark").unwrap();

        let helper_call = source.rfind("helper").unwrap() as u32;
        let syntax = analysis.syntax_at(source_id, helper_call).unwrap();
        let hir = analysis.hir_at(source_id, helper_call).unwrap();
        assert_eq!(syntax.domain(), QueryDomain::Syntax);
        assert_eq!(hir.domain(), QueryDomain::Hir);

        let helper = analysis.symbol_at(source_id, helper_call).unwrap();
        assert_eq!(
            analysis.signature(helper).as_deref(),
            Some("fn helper(x: Int32) -> Int32")
        );
        let definition = analysis.definition(helper).unwrap();
        assert_eq!(
            &source[definition.span.lo as usize..definition.span.hi as usize],
            "helper"
        );
        assert_eq!(analysis.references(helper).len(), 1);

        let value_reference = source.rfind("value").unwrap() as u32;
        let value = analysis.symbol_at(source_id, value_reference).unwrap();
        assert_eq!(value.kind(), QueryKind::Local);
        assert_eq!(analysis.type_of(value).as_deref(), Some("Int32"));
        let literal = source.find("helper(1)").unwrap() as u32 + "helper(".len() as u32;
        let literal_expr = analysis.hir_at(source_id, literal).unwrap();
        assert_eq!(analysis.type_of(literal_expr).as_deref(), Some("Int32"));
        assert_eq!(
            analysis.type_of(helper),
            analysis.signature(helper),
            "item type rendering uses its source-like signature"
        );
        let enclosing = analysis.enclosing(value).unwrap();
        let main = analysis.symbols.named("main").next().unwrap().definition;
        assert_eq!(enclosing.item, Some(main));

        let public = analysis
            .public_items()
            .into_iter()
            .map(|symbol| symbol.name.as_str())
            .collect::<Vec<_>>();
        assert_eq!(public, vec!["User", "main"]);
        assert_eq!(analysis.document_symbols(source_id).len(), 3);
        assert_eq!(analysis.workspace_symbols("help").len(), 1);
    }

    #[test]
    fn definitions_and_references_preserve_cross_file_provenance() {
        let unique = format!(
            "stark-c24-{}-{}",
            std::process::id(),
            NEXT_ANALYSIS_ID.load(Ordering::Relaxed)
        );
        let directory = std::env::temp_dir().join(unique);
        std::fs::create_dir_all(&directory).unwrap();
        let root_path = directory.join("main.stark");
        let child_path = directory.join("child.stark");
        let root_source = "mod child;\nfn main() { println(child::answer()); }\n";
        std::fs::write(&root_path, root_source).unwrap();
        std::fs::write(&child_path, "pub fn answer() -> Int32 { 42 }\n").unwrap();

        let file = Arc::new(SourceFile::new(
            root_path.to_string_lossy().into_owned(),
            root_source,
        ));
        let mut analysis = analyze_project(ProjectInput::program(file), LanguageOptions::CORE);
        assert!(!analysis.has_errors(), "{:?}", analysis.diagnostics);
        let root_id = analysis
            .source_map
            .id_for_name(&root_path.to_string_lossy())
            .unwrap();
        let child_id = analysis
            .source_map
            .id_for_name(&child_path.to_string_lossy())
            .unwrap();
        let call = root_source.find("answer").unwrap() as u32;
        let symbol = analysis.symbol_at(root_id, call).unwrap();
        let child_definition_offset = "pub fn ".len() as u32;
        assert_eq!(
            analysis
                .syntax_at(child_id, child_definition_offset)
                .unwrap()
                .domain(),
            QueryDomain::Syntax
        );
        assert_eq!(
            analysis
                .hir_at(child_id, child_definition_offset)
                .unwrap()
                .domain(),
            QueryDomain::Hir
        );
        assert_eq!(analysis.definition(symbol).unwrap().source, child_id);
        assert_eq!(
            analysis.references(symbol),
            vec![SourceLocation {
                source: root_id,
                span: Span {
                    lo: root_source.find("child::answer").unwrap() as u32,
                    hi: (root_source.find("child::answer").unwrap() + "child::answer".len()) as u32,
                },
            }]
        );
        assert!(analysis.enclosing(symbol).unwrap().module.is_some());
        assert_eq!(analysis.document_symbols(child_id).len(), 1);

        let root_file = analysis.source_map.get(root_id).unwrap().file.clone();
        let child_file = analysis.source_map.get(child_id).unwrap().file.clone();
        analysis.diagnostics.push(
            Diagnostic::error("cross-file contract violation", Span::new(0, 3))
                .with_file(root_file)
                .with_code("E0999")
                .with_label("primary")
                .with_related(child_file, Span::new(7, 13), "declared here")
                .with_note("transport note")
                .with_help("transport help")
                .with_rule_id("TEST-RULE-001")
                .with_deviation_id("DEV-TEST"),
        );
        let versions = HashMap::from([(root_id, 17)]);
        let batch = analysis.diagnostic_batch(&versions);
        let transported = batch.diagnostics.last().unwrap();
        assert_eq!(transported.primary.source, root_id);
        assert_eq!(transported.related[0].location.source, child_id);
        assert_eq!(transported.source_version, Some(17));
        assert_eq!(transported.rule_id.as_deref(), Some("TEST-RULE-001"));
        assert_eq!(transported.deviation_id.as_deref(), Some("DEV-TEST"));
        assert_eq!(
            batch
                .sources
                .iter()
                .find(|source| source.id == child_id)
                .unwrap()
                .kind,
            crate::diag::DiagnosticSourceKind::Module
        );
        let first_json = batch.to_json(&analysis.source_map);
        let second_json = batch.to_json(&analysis.source_map);
        assert_eq!(first_json, second_json);
        assert!(first_json.contains("\"sourceVersion\":17"));
        assert!(first_json.contains(&format!("\"sourceId\":{}", child_id.as_u32())));
        assert!(batch.render(&analysis.source_map).contains("related: "));

        std::fs::remove_dir_all(directory).unwrap();
    }
}
