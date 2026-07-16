use super::importer::{read_signature, DType, Dimension, ModelSignature, Port};
use super::OnnxError;
use crate::ast::{PortDir, Primitive};
use crate::hir::{self, GenericArg, ItemKind, TypeKind};
use crate::options::LanguageOptions;
use crate::parser::{parse_with_options, ParseMode};
use crate::source::{SourceFile, Span};
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::sync::Arc;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Difference {
    pub category: String,
    pub port_kind: Option<String>,
    pub port: Option<String>,
    pub axis: Option<usize>,
    pub expected: Option<String>,
    pub actual: Option<String>,
    pub description: String,
}

impl std::fmt::Display for Difference {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.description)
    }
}

impl Difference {
    pub fn to_json(&self) -> String {
        let mut fields = Vec::new();
        fields.push(format!("\"category\":\"{}\"", escape_json(&self.category)));
        if let Some(kind) = &self.port_kind {
            fields.push(format!("\"port_kind\":\"{}\"", escape_json(kind)));
        }
        if let Some(port) = &self.port {
            fields.push(format!("\"port\":\"{}\"", escape_json(port)));
        }
        if let Some(axis) = self.axis {
            fields.push(format!("\"axis\":{}", axis));
        }
        if let Some(expected) = &self.expected {
            fields.push(format!("\"expected\":\"{}\"", escape_json(expected)));
        }
        if let Some(actual) = &self.actual {
            fields.push(format!("\"actual\":\"{}\"", escape_json(actual)));
        }
        format!("{{{}}}", fields.join(","))
    }
}

pub fn escape_json(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
        .replace('\t', "\\t")
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct VerificationReport {
    pub differences: Vec<Difference>,
}

impl VerificationReport {
    pub fn is_match(&self) -> bool {
        self.differences.is_empty()
    }
}

#[derive(Clone, Debug)]
struct DeclarationSignature {
    inputs: Vec<DeclarationPort>,
    outputs: Vec<DeclarationPort>,
}

#[derive(Clone, Debug)]
struct DeclarationPort {
    name: String,
    dtype: DType,
    dimensions: Option<Vec<DeclarationDimension>>,
}

#[derive(Clone, Debug)]
enum DeclarationDimension {
    Static(u64),
    Dynamic(String),
}

pub fn verify_declaration_file(
    artifact: &Path,
    declaration: &Path,
    model_name: Option<&str>,
) -> Result<VerificationReport, OnnxError> {
    let (signature, _) = read_signature(artifact)?;
    let source = fs::read_to_string(declaration).map_err(|error| {
        OnnxError::new(format!(
            "cannot read declaration file `{}`: {error}",
            declaration.display()
        ))
    })?;
    verify_declaration_source(
        &signature,
        &source,
        &declaration.to_string_lossy(),
        model_name,
    )
}

pub fn verify_declaration_source(
    artifact: &ModelSignature,
    source: &str,
    source_name: &str,
    model_name: Option<&str>,
) -> Result<VerificationReport, OnnxError> {
    let declaration = extract_declaration(source, source_name, model_name)?;
    let mut report = VerificationReport::default();
    let mut declaration_to_artifact = HashMap::new();
    let mut artifact_to_declaration = HashMap::new();
    compare_ports(
        "input",
        &artifact.inputs,
        &declaration.inputs,
        &mut declaration_to_artifact,
        &mut artifact_to_declaration,
        &mut report.differences,
    );
    compare_ports(
        "output",
        &artifact.outputs,
        &declaration.outputs,
        &mut declaration_to_artifact,
        &mut artifact_to_declaration,
        &mut report.differences,
    );
    Ok(report)
}

fn extract_declaration(
    source: &str,
    source_name: &str,
    requested_model: Option<&str>,
) -> Result<DeclarationSignature, OnnxError> {
    let file = Arc::new(SourceFile::new(source_name, source));
    let options = LanguageOptions::with_tensor();
    let (ast, parse_diagnostics) = parse_with_options(&file, ParseMode::Program, options);
    if !parse_diagnostics.is_empty() {
        return Err(frontend_error("parse", &parse_diagnostics));
    }
    let (hir, resolution_diagnostics) =
        crate::resolve::resolve_with_options(&ast, file.clone(), options);
    if !resolution_diagnostics.is_empty() {
        return Err(frontend_error("resolve", &resolution_diagnostics));
    }
    let type_diagnostics = crate::typecheck::check_with_options(&hir, file.clone(), options);
    if type_diagnostics
        .iter()
        .any(|diagnostic| diagnostic.severity == crate::diag::Severity::Error)
    {
        return Err(frontend_error("type-check", &type_diagnostics));
    }

    let models = hir
        .items
        .iter()
        .filter_map(|item| match &item.kind {
            ItemKind::Model(definition) => Some((definition, text(&file, definition.name))),
            _ => None,
        })
        .collect::<Vec<_>>();
    let selected = if let Some(requested) = requested_model {
        models
            .iter()
            .find(|(_, name)| *name == requested)
            .copied()
            .ok_or_else(|| {
                OnnxError::new(format!(
                    "declaration contains no model named `{requested}`; available models: {}",
                    model_names(&models)
                ))
            })?
    } else {
        match models.as_slice() {
            [only] => *only,
            [] => return Err(OnnxError::new("declaration contains no model items")),
            _ => {
                return Err(OnnxError::new(format!(
                    "declaration contains multiple models ({}); pass `--model <Name>`",
                    model_names(&models)
                )));
            }
        }
    };

    let original_names = parse_original_name_metadata(source)?;
    let mut inputs = Vec::new();
    let mut outputs = Vec::new();
    for port in &selected.0.ports {
        let identifier = text(&file, port.name).to_string();
        let direction = match port.dir {
            PortDir::Input => "input",
            PortDir::Output => "output",
        };
        let name = original_names
            .get(&(direction.to_string(), identifier.clone()))
            .cloned()
            .unwrap_or(identifier);
        let declaration_port = extract_port(&hir, &file, port.ty, name)?;
        match port.dir {
            PortDir::Input => inputs.push(declaration_port),
            PortDir::Output => outputs.push(declaration_port),
        }
    }
    Ok(DeclarationSignature { inputs, outputs })
}

fn extract_port(
    hir: &hir::Hir,
    file: &SourceFile,
    type_id: hir::TypeId,
    name: String,
) -> Result<DeclarationPort, OnnxError> {
    let node = hir.ty(type_id);
    let TypeKind::Path { path, args, .. } = &node.kind else {
        return Err(OnnxError::new(format!(
            "model port `{name}` is not a tensor type"
        )));
    };
    let type_name = path
        .segments
        .last()
        .map(|segment| text(file, segment.span))
        .unwrap_or("");
    let args = args
        .as_ref()
        .map(|args| args.args.as_slice())
        .unwrap_or(&[]);
    match type_name {
        "Tensor" => {
            let dtype = args
                .first()
                .and_then(|argument| declaration_dtype(hir, file, argument))
                .ok_or_else(|| {
                    OnnxError::new(format!("cannot read dtype of declaration port `{name}`"))
                })?;
            let dimensions = match args.get(1) {
                Some(GenericArg::Shape(shape)) => shape
                    .dims
                    .iter()
                    .map(|dimension| declaration_dimension(file, dimension, &name))
                    .collect::<Result<Vec<_>, _>>()?,
                _ => {
                    return Err(OnnxError::new(format!(
                        "cannot read shape of declaration port `{name}`"
                    )));
                }
            };
            Ok(DeclarationPort {
                name,
                dtype,
                dimensions: Some(dimensions),
            })
        }
        "TensorDyn" => {
            let dtype = args
                .first()
                .and_then(|argument| declaration_dtype(hir, file, argument))
                .ok_or_else(|| {
                    OnnxError::new(format!("cannot read dtype of declaration port `{name}`"))
                })?;
            Ok(DeclarationPort {
                name,
                dtype,
                dimensions: None,
            })
        }
        _ => Err(OnnxError::new(format!(
            "model port `{name}` is not a Tensor or TensorDyn"
        ))),
    }
}

fn declaration_dtype(hir: &hir::Hir, file: &SourceFile, argument: &GenericArg) -> Option<DType> {
    let GenericArg::Type(type_id) = argument else {
        return None;
    };
    match &hir.ty(*type_id).kind {
        TypeKind::Primitive(primitive) => dtype_from_primitive(*primitive),
        TypeKind::Path { path, .. } => path
            .segments
            .last()
            .and_then(|segment| dtype_from_name(text(file, segment.span))),
        _ => None,
    }
}

fn dtype_from_primitive(primitive: Primitive) -> Option<DType> {
    dtype_from_name(primitive.name())
}

fn dtype_from_name(name: &str) -> Option<DType> {
    match name {
        "Int8" => Some(DType::Int8),
        "Int16" => Some(DType::Int16),
        "Int32" => Some(DType::Int32),
        "Int64" => Some(DType::Int64),
        "UInt8" => Some(DType::UInt8),
        "UInt16" => Some(DType::UInt16),
        "UInt32" => Some(DType::UInt32),
        "UInt64" => Some(DType::UInt64),
        "Float16" => Some(DType::Float16),
        "Float32" => Some(DType::Float32),
        "Float64" => Some(DType::Float64),
        "BFloat16" => Some(DType::BFloat16),
        "Bool" => Some(DType::Bool),
        _ => None,
    }
}

fn declaration_dimension(
    file: &SourceFile,
    dimension: &hir::DimExpr,
    port_name: &str,
) -> Result<DeclarationDimension, OnnxError> {
    match dimension {
        hir::DimExpr::Lit(span) => text(file, *span)
            .parse::<u64>()
            .map(DeclarationDimension::Static)
            .map_err(|_| {
                OnnxError::new(format!(
                    "declaration port `{port_name}` has an invalid static dimension"
                ))
            }),
        hir::DimExpr::Var(span) => Ok(DeclarationDimension::Dynamic(
            text(file, *span).to_string(),
        )),
        hir::DimExpr::Binary { .. } => Err(OnnxError::new(format!(
            "declaration port `{port_name}` uses a dimension expression; artifact verification requires a literal or variable"
        ))),
        hir::DimExpr::Error => Err(OnnxError::new(format!(
            "declaration port `{port_name}` has an invalid dimension"
        ))),
    }
}

fn compare_ports(
    direction: &str,
    artifact: &[Port],
    declaration: &[DeclarationPort],
    declaration_to_artifact: &mut HashMap<String, String>,
    artifact_to_declaration: &mut HashMap<String, String>,
    differences: &mut Vec<Difference>,
) {
    if artifact.len() != declaration.len() {
        let desc = format!(
            "{direction} port count differs: artifact has {}, declaration has {}",
            artifact.len(),
            declaration.len()
        );
        differences.push(Difference {
            category: "port_count".to_string(),
            port_kind: Some(direction.to_string()),
            port: None,
            axis: None,
            expected: Some(artifact.len().to_string()),
            actual: Some(declaration.len().to_string()),
            description: desc,
        });
    }
    for (index, (artifact, declaration)) in artifact.iter().zip(declaration).enumerate() {
        let position = index + 1;
        if artifact.name != declaration.name {
            let desc = format!(
                "{direction} {position} name differs: artifact `{}`, declaration `{}`",
                artifact.name, declaration.name
            );
            differences.push(Difference {
                category: "name".to_string(),
                port_kind: Some(direction.to_string()),
                port: Some(declaration.name.clone()),
                axis: None,
                expected: Some(artifact.name.clone()),
                actual: Some(declaration.name.clone()),
                description: desc,
            });
        }
        if artifact.dtype != declaration.dtype {
            let desc = format!(
                "{direction} `{}` dtype differs: artifact {}, declaration {}",
                artifact.name,
                artifact.dtype.stark_name(),
                declaration.dtype.stark_name()
            );
            differences.push(Difference {
                category: "dtype".to_string(),
                port_kind: Some(direction.to_string()),
                port: Some(declaration.name.clone()),
                axis: None,
                expected: Some(artifact.dtype.stark_name().to_string()),
                actual: Some(declaration.dtype.stark_name().to_string()),
                description: desc,
            });
        }
        let Some(declaration_dimensions) = &declaration.dimensions else {
            let desc = format!(
                "{direction} `{}` is TensorDyn in the declaration but rank {} in the artifact",
                artifact.name,
                artifact.dimensions.len()
            );
            differences.push(Difference {
                category: "rank".to_string(),
                port_kind: Some(direction.to_string()),
                port: Some(declaration.name.clone()),
                axis: None,
                expected: Some(artifact.dimensions.len().to_string()),
                actual: Some("dynamic".to_string()),
                description: desc,
            });
            continue;
        };
        if artifact.dimensions.len() != declaration_dimensions.len() {
            let desc = format!(
                "{direction} `{}` rank differs: artifact {}, declaration {}",
                artifact.name,
                artifact.dimensions.len(),
                declaration_dimensions.len()
            );
            differences.push(Difference {
                category: "rank".to_string(),
                port_kind: Some(direction.to_string()),
                port: Some(declaration.name.clone()),
                axis: None,
                expected: Some(artifact.dimensions.len().to_string()),
                actual: Some(declaration_dimensions.len().to_string()),
                description: desc,
            });
            continue;
        }
        for (axis, (artifact_dimension, declaration_dimension)) in artifact
            .dimensions
            .iter()
            .zip(declaration_dimensions)
            .enumerate()
        {
            match (artifact_dimension, declaration_dimension) {
                (
                    Dimension::Static(artifact_value),
                    DeclarationDimension::Static(declaration_value),
                ) if artifact_value == declaration_value => {}
                (
                    Dimension::Static(artifact_value),
                    DeclarationDimension::Static(declaration_value),
                ) => {
                    let desc = format!(
                        "{direction} `{}` dimension {axis} differs: artifact {artifact_value}, declaration {declaration_value}",
                        declaration.name
                    );
                    differences.push(Difference {
                        category: "dimension".to_string(),
                        port_kind: Some(direction.to_string()),
                        port: Some(declaration.name.clone()),
                        axis: Some(axis),
                        expected: Some(artifact_value.to_string()),
                        actual: Some(declaration_value.to_string()),
                        description: desc,
                    });
                }
                (Dimension::Static(artifact_value), DeclarationDimension::Dynamic(variable)) => {
                    let desc = format!(
                        "{direction} `{}` dimension {axis} over-promises dynamic `{variable}`; artifact is static {artifact_value}",
                        declaration.name
                    );
                    differences.push(Difference {
                        category: "dimension".to_string(),
                        port_kind: Some(direction.to_string()),
                        port: Some(declaration.name.clone()),
                        axis: Some(axis),
                        expected: Some(artifact_value.to_string()),
                        actual: Some(variable.clone()),
                        description: desc,
                    });
                }
                (Dimension::Dynamic { .. }, DeclarationDimension::Static(value)) => {
                    let desc = format!(
                        "{direction} `{}` dimension {axis} is dynamic in the artifact but static {value} in the declaration",
                        declaration.name
                    );
                    differences.push(Difference {
                        category: "dimension".to_string(),
                        port_kind: Some(direction.to_string()),
                        port: Some(declaration.name.clone()),
                        axis: Some(axis),
                        expected: Some("dynamic".to_string()),
                        actual: Some(value.to_string()),
                        description: desc,
                    });
                }
                (Dimension::Dynamic { key, .. }, DeclarationDimension::Dynamic(variable)) => {
                    if let Some(previous) = declaration_to_artifact.get(variable) {
                        if previous != key {
                            let desc = format!(
                                "declaration dimension `{variable}` maps to multiple artifact dynamic identities"
                            );
                            differences.push(Difference {
                                category: "conflict".to_string(),
                                port_kind: Some(direction.to_string()),
                                port: Some(declaration.name.clone()),
                                axis: Some(axis),
                                expected: Some(previous.clone()),
                                actual: Some(key.clone()),
                                description: desc,
                            });
                        }
                    } else {
                        declaration_to_artifact.insert(variable.clone(), key.clone());
                    }
                    if let Some(previous) = artifact_to_declaration.get(key) {
                        if previous != variable {
                            let desc = format!(
                                "artifact dynamic identity used by declaration dimensions `{previous}` and `{variable}`"
                            );
                            differences.push(Difference {
                                category: "conflict".to_string(),
                                port_kind: Some(direction.to_string()),
                                port: Some(declaration.name.clone()),
                                axis: Some(axis),
                                expected: Some(previous.clone()),
                                actual: Some(variable.clone()),
                                description: desc,
                            });
                        }
                    } else {
                        artifact_to_declaration.insert(key.clone(), variable.clone());
                    }
                }
            }
        }
    }
}

fn parse_original_name_metadata(
    source: &str,
) -> Result<HashMap<(String, String), String>, OnnxError> {
    let mut names = HashMap::new();
    for line in source.lines() {
        let Some(metadata) = line.trim().strip_prefix("// starkc-onnx-name ") else {
            continue;
        };
        let mut parts = metadata.split_whitespace();
        let (Some(direction), Some(identifier), Some(encoded), None) =
            (parts.next(), parts.next(), parts.next(), parts.next())
        else {
            return Err(OnnxError::new(
                "malformed starkc ONNX name metadata comment",
            ));
        };
        if !matches!(direction, "input" | "output") {
            return Err(OnnxError::new("invalid direction in ONNX name metadata"));
        }
        let decoded = hex_decode(encoded)?;
        let original = String::from_utf8(decoded)
            .map_err(|_| OnnxError::new("ONNX name metadata is not UTF-8"))?;
        let key = (direction.to_string(), identifier.to_string());
        if names.insert(key, original).is_some() {
            return Err(OnnxError::new("duplicate ONNX name metadata entry"));
        }
    }
    Ok(names)
}

fn hex_decode(text: &str) -> Result<Vec<u8>, OnnxError> {
    if text.len() % 2 != 0 {
        return Err(OnnxError::new("ONNX name metadata has invalid hex length"));
    }
    text.as_bytes()
        .chunks_exact(2)
        .map(|pair| {
            let high = hex_value(pair[0])?;
            let low = hex_value(pair[1])?;
            Ok((high << 4) | low)
        })
        .collect()
}

fn hex_value(byte: u8) -> Result<u8, OnnxError> {
    match byte {
        b'0'..=b'9' => Ok(byte - b'0'),
        b'a'..=b'f' => Ok(byte - b'a' + 10),
        _ => Err(OnnxError::new("ONNX name metadata contains invalid hex")),
    }
}

fn frontend_error(stage: &str, diagnostics: &[crate::diag::Diagnostic]) -> OnnxError {
    OnnxError::new(format!(
        "cannot {stage} declaration: {}",
        diagnostics
            .iter()
            .map(|diagnostic| diagnostic.message.as_str())
            .collect::<Vec<_>>()
            .join("; ")
    ))
}

fn model_names(models: &[(&hir::ModelDef, &str)]) -> String {
    models
        .iter()
        .map(|(_, name)| format!("`{name}`"))
        .collect::<Vec<_>>()
        .join(", ")
}

fn text(file: &SourceFile, span: Span) -> &str {
    &file.src[span.lo as usize..span.hi as usize]
}
