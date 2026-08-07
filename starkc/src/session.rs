//! AS2 — one compiler session, one pipeline.
//!
//! Before this module, eleven production entry points assembled the semantic pipeline and six of
//! them did it by hand: parse, resolve, typecheck, decide what counts as failure, and render
//! diagnostics, each in its own way. `AS0-BASELINE-AND-INVENTORY.md` §3 has the exact set, and
//! §3.2's characterization baseline records what they actually did — they were not identical, so
//! consolidation had to *choose* rather than merely deduplicate.
//!
//! A `CompilerSession` owns the inputs (package graph or single source, provider overlays, language
//! options) and exposes the pipeline as named operations. **Command-line parsing and presentation
//! stay outside**: this module decides what happened, not how to print it. That is why
//! [`CompileFailure`] renders diagnostics but leaves the trailing
//! `"<package>: package compilation failed"` line to the CLI that owns that wording.
//!
//! This is not an incremental query engine and holds no state between commands.

use std::sync::Arc;

use crate::analysis::{analyze_project, ProjectAnalysis, ProjectInput};
use crate::diag::{Diagnostic, Severity};
use crate::hir::Hir;
use crate::interp::{Execution, RuntimeError};
use crate::options::LanguageOptions;
use crate::package::PackageGraph;
use crate::source::SourceFile;
use crate::typecheck::TypeTables;

/// The inputs a compilation needs, before any of it has run.
pub struct CompilerSession {
    input: ProjectInput,
    options: LanguageOptions,
}

impl CompilerSession {
    /// A package and its dependency graph.
    pub fn for_package(graph: PackageGraph, options: LanguageOptions) -> Self {
        Self {
            input: ProjectInput::package(graph),
            options,
        }
    }

    /// A package whose sources are partly supplied in memory — provider synthesis, or an editor's
    /// unsaved buffer. Overlays decide CONTENT; they never change a source's identity.
    pub fn for_package_with_overlays(
        graph: PackageGraph,
        overlays: std::collections::HashMap<std::path::PathBuf, String>,
        options: LanguageOptions,
    ) -> Self {
        Self {
            input: ProjectInput::package_with_overlays(graph, overlays),
            options,
        }
    }

    /// A single source file with no package around it. Its name is the path, which is
    /// `SourceFile::name`'s documented contract for this case and **not** an inconsistency to be
    /// consolidated away (AS0 finding D4).
    pub fn for_source(file: Arc<SourceFile>, options: LanguageOptions) -> Self {
        Self::for_source_in_mode(file, crate::parser::ParseMode::Program, options)
    }

    /// A single source parsed in an explicit mode. The editor compiles whichever mode the user
    /// selected, so the mode is a session input rather than something a caller applies before
    /// handing the pipeline a parse tree.
    pub fn for_source_in_mode(
        file: Arc<SourceFile>,
        mode: crate::parser::ParseMode,
        options: LanguageOptions,
    ) -> Self {
        Self {
            input: ProjectInput::source(file, mode),
            options,
        }
    }

    pub fn options(&self) -> LanguageOptions {
        self.options
    }

    /// parse → resolve → typecheck, through the one pipeline. Returns everything the analysis
    /// produced, including diagnostics, whether or not it succeeded.
    pub fn analyze(self) -> ProjectAnalysis {
        analyze_project(self.input, self.options)
    }

    /// [`Self::analyze`] plus the success/failure decision, so no caller re-derives "did it fail"
    /// from the diagnostic list. Six entry points each had their own copy of that test.
    pub fn check(self) -> Result<CheckedProgram, CompileFailure> {
        let analysis = self.analyze();
        if analysis
            .diagnostics
            .iter()
            .any(|d| d.severity == Severity::Error)
        {
            return Err(CompileFailure {
                analysis: Box::new(analysis),
            });
        }
        // A successful analysis always produces both, but the types are `Option`, so state the
        // dependency rather than unwrapping and hoping.
        if analysis.hir.is_none() || analysis.type_tables.is_none() {
            return Err(CompileFailure {
                analysis: Box::new(analysis),
            });
        }
        Ok(CheckedProgram { analysis })
    }
}

/// An analysis that produced no errors, and therefore has HIR and type tables.
pub struct CheckedProgram {
    analysis: ProjectAnalysis,
}

impl CheckedProgram {
    pub fn analysis(&self) -> &ProjectAnalysis {
        &self.analysis
    }

    pub fn hir(&self) -> &Hir {
        self.analysis
            .hir
            .as_ref()
            .expect("a checked program has HIR")
    }

    pub fn tables(&self) -> &TypeTables {
        self.analysis
            .type_tables
            .as_ref()
            .expect("a checked program has type tables")
    }

    pub fn root_file(&self) -> &Arc<SourceFile> {
        &self.analysis.root_file
    }

    /// The root as a REGISTERED source — identity and file together.
    ///
    /// AS1b-ii: execution and lowering take this rather than a bare `Arc`, so neither can be
    /// handed a file this compilation never registered.
    pub fn root_source(&self) -> crate::source::RegisteredSource {
        self.analysis
            .ast
            .sources
            .id_for_name(&self.analysis.root_file.name)
            .and_then(|id| self.analysis.ast.sources.get(id))
            .expect("a checked program's root is registered")
            .clone()
    }

    /// Non-error diagnostics that survived a successful check — warnings a caller may still print.
    pub fn diagnostics(&self) -> &[Diagnostic] {
        &self.analysis.diagnostics
    }

    /// Execute on the typed-HIR reference interpreter.
    pub fn execute_hir(&self) -> Result<Execution, RuntimeError> {
        crate::interp::run(self.hir(), self.root_source(), self.tables())
    }

    /// Lower to monomorphised MIR.
    pub fn lower_mir(&self) -> Result<crate::mir::MirProgram, crate::mir::lower::LowerError> {
        crate::mir::lower::lower_program(self.hir(), self.tables(), self.root_source())
    }
}

/// Which phase of the pipeline rejected the program.
///
/// Analysis stops at the first phase that produced an error (AS0 finding D1), so exactly one phase
/// is responsible. Callers used to know this only because they had run the phases themselves; it is
/// recoverable from what the analysis produced, so the session states it rather than making every
/// caller keep its own pipeline to find out.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum FailedPhase {
    Parse,
    Resolve,
    Typecheck,
}

impl FailedPhase {
    /// The word used in `"<n> <phase> error(s)"`, the format
    /// `docs/WP8_3_TEST_FRAMEWORK_IMPLEMENTATION.md` documents for standalone program failures.
    pub fn label(self) -> &'static str {
        match self {
            FailedPhase::Parse => "parse",
            FailedPhase::Resolve => "resolve",
            FailedPhase::Typecheck => "typecheck",
        }
    }
}

/// An analysis that produced at least one error.
///
/// The analysis is boxed: a `ProjectAnalysis` is around two kilobytes, and this is the `Err` half of
/// every `check()`, so returning it inline would move that much on each call and on each `?`.
pub struct CompileFailure {
    analysis: Box<ProjectAnalysis>,
}

impl CompileFailure {
    pub fn analysis(&self) -> &ProjectAnalysis {
        &self.analysis
    }

    /// The phase that rejected the program, inferred from how far the pipeline got: resolve
    /// produces HIR, typecheck produces type tables, and neither runs after an earlier error.
    pub fn phase(&self) -> FailedPhase {
        match (
            self.analysis.hir.is_some(),
            self.analysis.type_tables.is_some(),
        ) {
            (false, _) => FailedPhase::Parse,
            (true, false) => FailedPhase::Resolve,
            (true, true) => FailedPhase::Typecheck,
        }
    }

    /// How many diagnostics are actually errors.
    ///
    /// Callers that reported `"<n> parse error(s)"` from the whole diagnostic list counted warnings
    /// among the errors. This counts what the message claims.
    pub fn error_count(&self) -> usize {
        self.analysis
            .diagnostics
            .iter()
            .filter(|d| d.severity == Severity::Error)
            .count()
    }

    /// `"<n> <phase> error(s)"` — the documented standalone-program failure format.
    pub fn summary(&self) -> String {
        format!("{} {} error(s)", self.error_count(), self.phase().label())
    }

    pub fn diagnostics(&self) -> &[Diagnostic] {
        &self.analysis.diagnostics
    }

    pub fn root_file(&self) -> &Arc<SourceFile> {
        &self.analysis.root_file
    }

    /// The root as a REGISTERED source — identity and file together.
    ///
    /// AS1b-ii: execution and lowering take this rather than a bare `Arc`, so neither can be
    /// handed a file this compilation never registered.
    pub fn root_source(&self) -> crate::source::RegisteredSource {
        self.analysis
            .ast
            .sources
            .id_for_name(&self.analysis.root_file.name)
            .and_then(|id| self.analysis.ast.sources.get(id))
            .expect("a checked program's root is registered")
            .clone()
    }

    /// Every diagnostic, rendered in pipeline order — parse, then resolve, then typecheck.
    ///
    /// **This is a deliberate ordering change** for the entry points that used to print typecheck
    /// diagnostics first and parse/resolve ones after (`checked.diagnostics.chain(diags)`). Because
    /// analysis stops at the first phase that produced an error (AS0 finding D1), the two orders
    /// can only differ in where earlier-phase *warnings* land. Pipeline order is the one worth
    /// keeping: it reads in the order the compiler did the work.
    pub fn render(&self) -> String {
        let root = self.root_file();
        let mut out = String::new();
        for diagnostic in &self.analysis.diagnostics {
            out.push_str(&diagnostic.render(root));
        }
        out
    }
}
