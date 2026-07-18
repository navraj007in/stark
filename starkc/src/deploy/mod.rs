//! Bounded deployment lowering for one checked inference pipeline (Gate 5).
//!
//! `starkc deploy` runs the normal tensor-enabled front end over a pipeline,
//! verifies the live ONNX artifact against the STARK `model` declaration with
//! the Gate 4 importer, then lowers the entry's reachable call graph into the
//! backend-oriented [`ir::DeploymentProgram`]. Host emission (M5.2) consumes
//! that program; this module never depends on the inference backend crate.
//!
//! Deployment diagnostics use the `E06xx` range (see the `lower` module); the
//! normative front end never emits those codes.

pub mod emit;
pub mod ir;
mod lower;

pub use emit::{emit, write_project, EmittedFile};
pub use ir::{
    DeployModel, DeployOp, DeployParam, DeployPort, DeployStmt, DeployTy, DeploymentFunction,
    DeploymentProgram, ScalarLit, TensorShape, ValueId,
};

use crate::analysis::{analyze_project, ProjectInput};
use crate::diag::Diagnostic;
use crate::options::LanguageOptions;
use crate::source::SourceFile;
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// What `starkc deploy` reports on success (plan §5.1).
pub struct DeploymentSummary {
    pub out_dir: PathBuf,
    pub model_sha256: String,
    pub entry: String,
}

/// Full `starkc deploy` pipeline: lower, emit, and write the host project.
/// Generation itself never downloads dependencies or invokes Cargo (plan §5.1).
pub fn deploy(
    pipeline: &Path,
    model: &Path,
    entry: &str,
    out: &Path,
    force: bool,
) -> Result<DeploymentSummary, DeployError> {
    let program = lower_pipeline(pipeline, model, entry)?;
    let files = emit(&program);
    write_project(out, &files, force).map_err(DeployError::Io)?;
    Ok(DeploymentSummary {
        out_dir: out.to_path_buf(),
        model_sha256: program.model.artifact_sha256,
        entry: entry.to_string(),
    })
}

/// A deployment failure, carrying enough context for the CLI to render it.
pub enum DeployError {
    /// Filesystem or artifact read failure.
    Io(String),
    /// ONNX decode/verification failure (Gate 4 importer error surface).
    Onnx(String),
    /// Front-end or lowering diagnostics against `file` (rendered by the CLI).
    Diagnostics {
        file: Arc<SourceFile>,
        diagnostics: Vec<Diagnostic>,
    },
}

impl DeployError {
    /// Render the failure to a string suitable for stderr.
    pub fn render(&self) -> String {
        match self {
            DeployError::Io(message) => format!("Error: {message}\n"),
            DeployError::Onnx(message) => format!("Error: {message}\n"),
            DeployError::Diagnostics { file, diagnostics } => {
                let mut out = String::new();
                for diagnostic in diagnostics {
                    out.push_str(&diagnostic.render(file));
                }
                out
            }
        }
    }
}

/// Lower `<pipeline>` + `<model.onnx>` into a [`DeploymentProgram`], selecting
/// `entry` as the inference entry point. Pure: reads inputs, checks, and
/// lowers; it never writes files, invokes Cargo, or downloads anything.
pub fn lower_pipeline(
    pipeline: &Path,
    model: &Path,
    entry: &str,
) -> Result<DeploymentProgram, DeployError> {
    let source = std::fs::read_to_string(pipeline)
        .map_err(|e| DeployError::Io(format!("cannot read `{}`: {e}", pipeline.display())))?;
    let file = Arc::new(SourceFile::new(pipeline.to_string_lossy(), source.clone()));

    // 1. Normal tensor-enabled front end. Any diagnostic aborts before lowering.
    let options = LanguageOptions::with_tensor();
    let analysis = analyze_project(ProjectInput::program(file.clone()), options);
    let mut diagnostics = analysis.diagnostics;
    if diagnostics
        .iter()
        .any(|diag| diag.severity == crate::diag::Severity::Error)
    {
        return Err(DeployError::Diagnostics { file, diagnostics });
    }
    let hir = analysis.hir.expect("successful analysis has HIR");
    let tables = analysis
        .type_tables
        .expect("successful analysis has type tables");

    // 2. Read + hash the live ONNX artifact.
    let (signature, bytes) =
        crate::onnx::read_signature(model).map_err(|e| DeployError::Onnx(e.to_string()))?;
    let artifact_sha256 = sha256_hex(&bytes);

    if signature.inputs.len() != 1 || signature.outputs.len() != 1 {
        return Err(DeployError::Onnx(format!(
            "the deployment prototype supports single-input/single-output models; \
                     `{}` has {} input(s) and {} output(s)",
            model.display(),
            signature.inputs.len(),
            signature.outputs.len()
        )));
    }

    // 3. Lower the reachable call graph (also validates the entry ABI
    //    and selects the single model declaration).
    let graph = match lower::lower_reachable(&hir, &tables, &file, entry) {
        Ok(graph) => graph,
        Err(diags) => {
            diagnostics.extend(diags);
            return Err(DeployError::Diagnostics { file, diagnostics });
        }
    };

    // 4. Gate 4 live artifact/declaration verification, before we
    //    commit the signature into the program (plan §4.4).
    let report = crate::onnx::verify_declaration_source(
        &signature,
        &source,
        &file.name,
        Some(&graph.model_type_name),
    )
    .map_err(|e| DeployError::Onnx(e.to_string()))?;
    if !report.is_match() {
        let mut message = String::from("ONNX artifact does not match the model declaration:");
        for difference in &report.differences {
            message.push_str(&format!("\n  - {difference}"));
        }
        return Err(DeployError::Onnx(message));
    }

    let model = ir::model_from_signature(graph.model_type_name, &signature, artifact_sha256);

    Ok(DeploymentProgram {
        compiler_version: env!("CARGO_PKG_VERSION").to_string(),
        entry: entry.to_string(),
        model,
        functions: graph.functions,
    })
}

/// Lowercase hex SHA-256, matching the importer's artifact-hash convention.
fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut out = String::with_capacity(digest.len() * 2);
    for byte in digest {
        out.push_str(&format!("{byte:02x}"));
    }
    out
}
