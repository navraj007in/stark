use super::OnnxError;
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

#[derive(Clone, Copy, Debug)]
pub struct DecodeLimits {
    pub max_file_size: usize,
    pub max_field_size: usize,
    pub max_depth: usize,
    pub max_inputs: usize,
    pub max_outputs: usize,
    pub max_initializers: usize,
    pub max_rank: usize,
    pub max_name_bytes: usize,
}

pub const DEFAULT_LIMITS: DecodeLimits = DecodeLimits {
    max_file_size: 256 * 1024 * 1024,
    max_field_size: 256 * 1024 * 1024,
    max_depth: 32,
    max_inputs: 4096,
    max_outputs: 4096,
    max_initializers: 1_000_000,
    max_rank: 64,
    max_name_bytes: 4096,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DType {
    Int8,
    Int16,
    Int32,
    Int64,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
    Float16,
    Float32,
    Float64,
    BFloat16,
    Bool,
}

impl DType {
    pub fn stark_name(self) -> &'static str {
        match self {
            DType::Int8 => "Int8",
            DType::Int16 => "Int16",
            DType::Int32 => "Int32",
            DType::Int64 => "Int64",
            DType::UInt8 => "UInt8",
            DType::UInt16 => "UInt16",
            DType::UInt32 => "UInt32",
            DType::UInt64 => "UInt64",
            DType::Float16 => "Float16",
            DType::Float32 => "Float32",
            DType::Float64 => "Float64",
            DType::BFloat16 => "BFloat16",
            DType::Bool => "Bool",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Dimension {
    Static(u64),
    Dynamic {
        /// Stable identity used to preserve equality relationships. Named
        /// ONNX dimensions share a key; each anonymous dimension is unique.
        key: String,
        hint: Option<String>,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Port {
    pub name: String,
    pub dtype: DType,
    pub dimensions: Vec<Dimension>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModelSignature {
    pub inputs: Vec<Port>,
    pub outputs: Vec<Port>,
}

#[derive(Default)]
struct DecodeState {
    anonymous_dimension: usize,
}

struct Reader<'a> {
    bytes: &'a [u8],
    position: usize,
    limits: DecodeLimits,
}

impl<'a> Reader<'a> {
    fn new(bytes: &'a [u8], limits: DecodeLimits) -> Self {
        Self {
            bytes,
            position: 0,
            limits,
        }
    }

    fn done(&self) -> bool {
        self.position == self.bytes.len()
    }

    fn varint(&mut self) -> Result<u64, OnnxError> {
        let mut value = 0u64;
        for shift in (0..70).step_by(7) {
            let byte = *self
                .bytes
                .get(self.position)
                .ok_or_else(|| OnnxError::new("truncated protobuf varint"))?;
            self.position += 1;
            if shift == 63 && byte > 1 {
                return Err(OnnxError::new("protobuf varint exceeds 64 bits"));
            }
            value |= u64::from(byte & 0x7f) << shift;
            if byte & 0x80 == 0 {
                return Ok(value);
            }
        }
        Err(OnnxError::new("protobuf varint is longer than ten bytes"))
    }

    fn tag(&mut self) -> Result<(u32, u8), OnnxError> {
        let tag = self.varint()?;
        let field = u32::try_from(tag >> 3)
            .map_err(|_| OnnxError::new("protobuf field number is out of range"))?;
        if field == 0 {
            return Err(OnnxError::new("protobuf field number zero is invalid"));
        }
        Ok((field, (tag & 7) as u8))
    }

    fn length_delimited(&mut self) -> Result<&'a [u8], OnnxError> {
        let length = usize::try_from(self.varint()?)
            .map_err(|_| OnnxError::new("protobuf field length is out of range"))?;
        if length > self.limits.max_field_size {
            return Err(OnnxError::new(format!(
                "protobuf field exceeds the {} byte limit",
                self.limits.max_field_size
            )));
        }
        let end = self
            .position
            .checked_add(length)
            .ok_or_else(|| OnnxError::new("protobuf field length overflow"))?;
        let bytes = self
            .bytes
            .get(self.position..end)
            .ok_or_else(|| OnnxError::new("truncated length-delimited protobuf field"))?;
        self.position = end;
        Ok(bytes)
    }

    fn skip(&mut self, wire: u8) -> Result<(), OnnxError> {
        match wire {
            0 => {
                self.varint()?;
            }
            1 => self.advance(8)?,
            2 => {
                self.length_delimited()?;
            }
            5 => self.advance(4)?,
            3 | 4 => return Err(OnnxError::new("protobuf group wire types are unsupported")),
            _ => return Err(OnnxError::new(format!("invalid protobuf wire type {wire}"))),
        }
        Ok(())
    }

    fn advance(&mut self, count: usize) -> Result<(), OnnxError> {
        let end = self
            .position
            .checked_add(count)
            .ok_or_else(|| OnnxError::new("protobuf cursor overflow"))?;
        if end > self.bytes.len() {
            return Err(OnnxError::new("truncated fixed-width protobuf field"));
        }
        self.position = end;
        Ok(())
    }

    fn string(&mut self) -> Result<String, OnnxError> {
        let bytes = self.length_delimited()?;
        if bytes.len() > self.limits.max_name_bytes {
            return Err(OnnxError::new(format!(
                "ONNX name exceeds the {} byte limit",
                self.limits.max_name_bytes
            )));
        }
        std::str::from_utf8(bytes)
            .map(str::to_owned)
            .map_err(|_| OnnxError::new("ONNX metadata name is not valid UTF-8"))
    }
}

fn nested<'a>(
    bytes: &'a [u8],
    limits: DecodeLimits,
    depth: usize,
) -> Result<Reader<'a>, OnnxError> {
    if depth > limits.max_depth {
        return Err(OnnxError::new(format!(
            "ONNX metadata exceeds the nesting depth limit of {}",
            limits.max_depth
        )));
    }
    Ok(Reader::new(bytes, limits))
}

pub fn decode_signature(bytes: &[u8], limits: DecodeLimits) -> Result<ModelSignature, OnnxError> {
    if bytes.len() > limits.max_file_size {
        return Err(OnnxError::new(format!(
            "ONNX file exceeds the {} byte limit",
            limits.max_file_size
        )));
    }
    let mut reader = Reader::new(bytes, limits);
    let mut graph = None;
    let mut state = DecodeState::default();
    while !reader.done() {
        let (field, wire) = reader.tag()?;
        if field == 7 {
            require_wire(field, wire, 2)?;
            graph = Some(parse_graph(
                reader.length_delimited()?,
                limits,
                1,
                &mut state,
            )?);
        } else {
            reader.skip(wire)?;
        }
    }
    graph.ok_or_else(|| OnnxError::new("ONNX ModelProto is missing GraphProto field 7"))
}

fn parse_graph(
    bytes: &[u8],
    limits: DecodeLimits,
    depth: usize,
    state: &mut DecodeState,
) -> Result<ModelSignature, OnnxError> {
    let mut reader = nested(bytes, limits, depth)?;
    let mut inputs = Vec::new();
    let mut outputs = Vec::new();
    let mut initializers = HashSet::new();
    let mut initializer_count = 0usize;
    while !reader.done() {
        let (field, wire) = reader.tag()?;
        match field {
            5 => {
                require_wire(field, wire, 2)?;
                if initializer_count >= limits.max_initializers {
                    return Err(OnnxError::new(format!(
                        "ONNX graph exceeds the initializer limit of {}",
                        limits.max_initializers
                    )));
                }
                initializer_count += 1;
                if let Some(name) =
                    parse_tensor_name(reader.length_delimited()?, limits, depth + 1)?
                {
                    initializers.insert(name);
                }
            }
            11 => {
                require_wire(field, wire, 2)?;
                if inputs.len() >= limits.max_inputs {
                    return Err(OnnxError::new(format!(
                        "ONNX graph exceeds the input limit of {}",
                        limits.max_inputs
                    )));
                }
                inputs.push(parse_value_info(
                    reader.length_delimited()?,
                    limits,
                    depth + 1,
                    state,
                )?);
            }
            12 => {
                require_wire(field, wire, 2)?;
                if outputs.len() >= limits.max_outputs {
                    return Err(OnnxError::new(format!(
                        "ONNX graph exceeds the output limit of {}",
                        limits.max_outputs
                    )));
                }
                outputs.push(parse_value_info(
                    reader.length_delimited()?,
                    limits,
                    depth + 1,
                    state,
                )?);
            }
            _ => reader.skip(wire)?,
        }
    }
    inputs.retain(|port| !initializers.contains(&port.name));
    validate_ports("input", &inputs)?;
    validate_ports("output", &outputs)?;
    validate_external_port_names(&inputs, &outputs)?;
    if inputs.is_empty() {
        return Err(OnnxError::new(
            "ONNX graph has no external inputs after initializer filtering",
        ));
    }
    if outputs.is_empty() {
        return Err(OnnxError::new("ONNX graph has no outputs"));
    }
    Ok(ModelSignature { inputs, outputs })
}

fn parse_tensor_name(
    bytes: &[u8],
    limits: DecodeLimits,
    depth: usize,
) -> Result<Option<String>, OnnxError> {
    let mut reader = nested(bytes, limits, depth)?;
    let mut name = None;
    while !reader.done() {
        let (field, wire) = reader.tag()?;
        if field == 8 {
            require_wire(field, wire, 2)?;
            name = Some(reader.string()?);
        } else {
            reader.skip(wire)?;
        }
    }
    Ok(name)
}

fn parse_value_info(
    bytes: &[u8],
    limits: DecodeLimits,
    depth: usize,
    state: &mut DecodeState,
) -> Result<Port, OnnxError> {
    let mut reader = nested(bytes, limits, depth)?;
    let mut name = None;
    let mut tensor = None;
    while !reader.done() {
        let (field, wire) = reader.tag()?;
        match field {
            1 => {
                require_wire(field, wire, 2)?;
                name = Some(reader.string()?);
            }
            2 => {
                require_wire(field, wire, 2)?;
                tensor = Some(parse_type_proto(
                    reader.length_delimited()?,
                    limits,
                    depth + 1,
                    state,
                )?);
            }
            _ => reader.skip(wire)?,
        }
    }
    let name = name.ok_or_else(|| OnnxError::new("ONNX ValueInfoProto is missing its name"))?;
    let (dtype, dimensions) = tensor
        .ok_or_else(|| OnnxError::new(format!("ONNX port `{name}` is missing its TypeProto")))?;
    Ok(Port {
        name,
        dtype,
        dimensions,
    })
}

fn parse_type_proto(
    bytes: &[u8],
    limits: DecodeLimits,
    depth: usize,
    state: &mut DecodeState,
) -> Result<(DType, Vec<Dimension>), OnnxError> {
    let mut reader = nested(bytes, limits, depth)?;
    let mut tensor = None;
    let mut unsupported = None;
    while !reader.done() {
        let (field, wire) = reader.tag()?;
        if field == 1 {
            require_wire(field, wire, 2)?;
            tensor = Some(parse_tensor_type(
                reader.length_delimited()?,
                limits,
                depth + 1,
                state,
            )?);
        } else if matches!(field, 4 | 5 | 7 | 8 | 9) {
            unsupported = Some(match field {
                4 => "sequence",
                5 => "map",
                7 => "opaque",
                8 => "sparse tensor",
                9 => "optional",
                _ => unreachable!(),
            });
            reader.skip(wire)?;
        } else {
            reader.skip(wire)?;
        }
    }
    if let Some(kind) = unsupported {
        return Err(OnnxError::new(format!(
            "unsupported ONNX {kind} graph port"
        )));
    }
    tensor.ok_or_else(|| OnnxError::new("ONNX graph port is not a tensor type"))
}

fn parse_tensor_type(
    bytes: &[u8],
    limits: DecodeLimits,
    depth: usize,
    state: &mut DecodeState,
) -> Result<(DType, Vec<Dimension>), OnnxError> {
    let mut reader = nested(bytes, limits, depth)?;
    let mut dtype = None;
    let mut shape = None;
    while !reader.done() {
        let (field, wire) = reader.tag()?;
        match field {
            1 => {
                require_wire(field, wire, 0)?;
                dtype = Some(map_dtype(reader.varint()?)?);
            }
            2 => {
                require_wire(field, wire, 2)?;
                shape = Some(parse_shape(
                    reader.length_delimited()?,
                    limits,
                    depth + 1,
                    state,
                )?);
            }
            _ => reader.skip(wire)?,
        }
    }
    Ok((
        dtype.ok_or_else(|| OnnxError::new("ONNX tensor port is missing elem_type"))?,
        shape.ok_or_else(|| OnnxError::new("ONNX tensor port is missing its rank/shape"))?,
    ))
}

fn parse_shape(
    bytes: &[u8],
    limits: DecodeLimits,
    depth: usize,
    state: &mut DecodeState,
) -> Result<Vec<Dimension>, OnnxError> {
    let mut reader = nested(bytes, limits, depth)?;
    let mut dimensions = Vec::new();
    while !reader.done() {
        let (field, wire) = reader.tag()?;
        if field == 1 {
            require_wire(field, wire, 2)?;
            if dimensions.len() >= limits.max_rank {
                return Err(OnnxError::new(format!(
                    "ONNX tensor rank exceeds the limit of {}",
                    limits.max_rank
                )));
            }
            dimensions.push(parse_dimension(
                reader.length_delimited()?,
                limits,
                depth + 1,
                state,
            )?);
        } else {
            reader.skip(wire)?;
        }
    }
    Ok(dimensions)
}

fn parse_dimension(
    bytes: &[u8],
    limits: DecodeLimits,
    depth: usize,
    state: &mut DecodeState,
) -> Result<Dimension, OnnxError> {
    let mut reader = nested(bytes, limits, depth)?;
    let mut dimension = None;
    while !reader.done() {
        let (field, wire) = reader.tag()?;
        match field {
            1 => {
                require_wire(field, wire, 0)?;
                let signed = reader.varint()? as i64;
                if signed < 0 {
                    return Err(OnnxError::new(format!(
                        "ONNX static dimension {signed} is negative"
                    )));
                }
                dimension = Some(Dimension::Static(signed as u64));
            }
            2 => {
                require_wire(field, wire, 2)?;
                let name = reader.string()?;
                if name.is_empty() {
                    dimension = Some(anonymous_dimension(state));
                } else {
                    dimension = Some(Dimension::Dynamic {
                        key: format!("named:{name}"),
                        hint: Some(name),
                    });
                }
            }
            _ => reader.skip(wire)?,
        }
    }
    Ok(dimension.unwrap_or_else(|| anonymous_dimension(state)))
}

fn anonymous_dimension(state: &mut DecodeState) -> Dimension {
    state.anonymous_dimension += 1;
    Dimension::Dynamic {
        key: format!("anonymous:{}", state.anonymous_dimension),
        hint: None,
    }
}

fn require_wire(field: u32, found: u8, expected: u8) -> Result<(), OnnxError> {
    if found == expected {
        Ok(())
    } else {
        Err(OnnxError::new(format!(
            "protobuf field {field} has wire type {found}, expected {expected}"
        )))
    }
}

fn map_dtype(value: u64) -> Result<DType, OnnxError> {
    match value {
        1 => Ok(DType::Float32),
        2 => Ok(DType::UInt8),
        3 => Ok(DType::Int8),
        4 => Ok(DType::UInt16),
        5 => Ok(DType::Int16),
        6 => Ok(DType::Int32),
        7 => Ok(DType::Int64),
        9 => Ok(DType::Bool),
        10 => Ok(DType::Float16),
        11 => Ok(DType::Float64),
        12 => Ok(DType::UInt32),
        13 => Ok(DType::UInt64),
        16 => Ok(DType::BFloat16),
        other => Err(OnnxError::new(format!(
            "unsupported ONNX tensor element type {other}"
        ))),
    }
}

fn validate_ports(kind: &str, ports: &[Port]) -> Result<(), OnnxError> {
    let mut names = HashSet::new();
    for port in ports {
        if port.name.is_empty() {
            return Err(OnnxError::new(format!(
                "ONNX {kind} port has an empty name"
            )));
        }
        if !names.insert(port.name.as_str()) {
            return Err(OnnxError::new(format!(
                "ONNX graph contains duplicate {kind} port `{}`",
                port.name
            )));
        }
    }
    Ok(())
}

fn validate_external_port_names(inputs: &[Port], outputs: &[Port]) -> Result<(), OnnxError> {
    let mut names = HashSet::new();
    for port in inputs.iter().chain(outputs) {
        if !names.insert(port.name.as_str()) {
            return Err(OnnxError::new(format!(
                "ONNX graph contains duplicate external port `{}`",
                port.name
            )));
        }
    }
    Ok(())
}

pub fn read_signature(path: &Path) -> Result<(ModelSignature, Vec<u8>), OnnxError> {
    let file = File::open(path).map_err(|error| {
        OnnxError::new(format!(
            "cannot open ONNX file `{}`: {error}",
            path.display()
        ))
    })?;
    let mut bytes = Vec::new();
    file.take((DEFAULT_LIMITS.max_file_size as u64) + 1)
        .read_to_end(&mut bytes)
        .map_err(|error| {
            OnnxError::new(format!(
                "cannot read ONNX file `{}`: {error}",
                path.display()
            ))
        })?;
    if bytes.len() > DEFAULT_LIMITS.max_file_size {
        return Err(OnnxError::new(format!(
            "ONNX file exceeds the {} byte limit",
            DEFAULT_LIMITS.max_file_size
        )));
    }
    let signature = decode_signature(&bytes, DEFAULT_LIMITS)?;
    Ok((signature, bytes))
}

pub fn import_file(input: &Path, output: &Path, force: bool) -> Result<String, OnnxError> {
    if output.exists() && !force {
        return Err(OnnxError::new(format!(
            "output file `{}` already exists; pass `--force` to replace it",
            output.display()
        )));
    }
    let (signature, bytes) = read_signature(input)?;
    let stem = input
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or("model");
    let hash = sha256_hex(&bytes);
    let source = format_declaration(&signature, stem, &hash);
    write_atomic(output, source.as_bytes(), force)?;
    Ok(source)
}

pub fn format_declaration(signature: &ModelSignature, artifact_stem: &str, hash: &str) -> String {
    let model_name = model_identifier(artifact_stem);
    let mut used_ports = HashSet::new();
    let mut port_names = HashMap::new();
    for port in signature.inputs.iter().chain(&signature.outputs) {
        let identifier = unique_identifier(sanitize_identifier(&port.name), &mut used_ports);
        port_names.insert(port.name.clone(), identifier);
    }

    let mut used_dims = HashSet::new();
    let mut dimension_names = HashMap::new();
    for port in signature.inputs.iter().chain(&signature.outputs) {
        for dimension in &port.dimensions {
            if let Dimension::Dynamic { key, hint } = dimension {
                if !dimension_names.contains_key(key) {
                    let base = hint
                        .as_deref()
                        .map(sanitize_identifier)
                        .filter(|name| !name.is_empty())
                        .unwrap_or_else(|| format!("D{}", dimension_names.len() + 1));
                    dimension_names.insert(key.clone(), unique_identifier(base, &mut used_dims));
                }
            }
        }
    }

    let mut source = format!(
        "// Generated by starkc {}; ONNX SHA-256: {hash}\n",
        env!("CARGO_PKG_VERSION")
    );
    for (kind, ports) in [("input", &signature.inputs), ("output", &signature.outputs)] {
        for port in ports {
            let identifier = &port_names[&port.name];
            if identifier != &port.name {
                source.push_str(&format!(
                    "// starkc-onnx-name {kind} {identifier} {}\n",
                    hex_encode(port.name.as_bytes())
                ));
            }
        }
    }
    source.push_str("\nmodel ");
    source.push_str(&model_name);
    if !dimension_names.is_empty() {
        let mut names = dimension_names.values().cloned().collect::<Vec<_>>();
        // HashMap iteration is not stable; recover first-occurrence order.
        names.clear();
        for port in signature.inputs.iter().chain(&signature.outputs) {
            for dimension in &port.dimensions {
                if let Dimension::Dynamic { key, .. } = dimension {
                    let name = &dimension_names[key];
                    if !names.contains(name) {
                        names.push(name.clone());
                    }
                }
            }
        }
        source.push('<');
        source.push_str(
            &names
                .iter()
                .map(|name| format!("{name}: Dim"))
                .collect::<Vec<_>>()
                .join(", "),
        );
        source.push('>');
    }
    source.push_str(" {\n");
    for (direction, ports) in [("input", &signature.inputs), ("output", &signature.outputs)] {
        for port in ports {
            let dimensions = port
                .dimensions
                .iter()
                .map(|dimension| match dimension {
                    Dimension::Static(value) => value.to_string(),
                    Dimension::Dynamic { key, .. } => dimension_names[key].clone(),
                })
                .collect::<Vec<_>>()
                .join(", ");
            source.push_str(&format!(
                "    {direction} {}: Tensor<{}, [{dimensions}]>;\n",
                port_names[&port.name],
                port.dtype.stark_name()
            ));
        }
    }
    source.push_str("}\n");
    source
}

fn sha256_hex(bytes: &[u8]) -> String {
    hex_encode(&Sha256::digest(bytes))
}

fn model_identifier(stem: &str) -> String {
    let mut name = String::new();
    for word in stem.split(|character: char| !character.is_ascii_alphanumeric()) {
        if word.is_empty() {
            continue;
        }
        let mut characters = word.chars();
        if let Some(first) = characters.next() {
            name.push(first.to_ascii_uppercase());
            name.extend(characters.map(|character| character.to_ascii_lowercase()));
        }
    }
    if name.is_empty() || name.as_bytes()[0].is_ascii_digit() {
        name.insert_str(0, "Model");
    }
    if is_keyword(&name) {
        name.push('_');
    }
    name
}

fn sanitize_identifier(name: &str) -> String {
    let mut identifier = String::with_capacity(name.len());
    for (index, character) in name.chars().enumerate() {
        let valid = character.is_ascii_alphanumeric() || character == '_';
        if valid && !(index == 0 && character.is_ascii_digit()) {
            identifier.push(character);
        } else {
            identifier.push('_');
            if index == 0 && character.is_ascii_digit() {
                identifier.push(character);
            }
        }
    }
    if identifier.is_empty() {
        identifier.push_str("value");
    }
    if is_keyword(&identifier) {
        identifier.push('_');
    }
    identifier
}

fn unique_identifier(base: String, used: &mut HashSet<String>) -> String {
    if used.insert(base.clone()) {
        return base;
    }
    for suffix in 2usize.. {
        let candidate = format!("{base}_{suffix}");
        if used.insert(candidate.clone()) {
            return candidate;
        }
    }
    unreachable!()
}

fn is_keyword(name: &str) -> bool {
    matches!(
        name,
        "as" | "async"
            | "await"
            | "break"
            | "const"
            | "continue"
            | "crate"
            | "else"
            | "enum"
            | "false"
            | "fn"
            | "for"
            | "if"
            | "impl"
            | "in"
            | "let"
            | "loop"
            | "match"
            | "mod"
            | "move"
            | "mut"
            | "pub"
            | "ref"
            | "return"
            | "self"
            | "Self"
            | "static"
            | "struct"
            | "super"
            | "trait"
            | "true"
            | "type"
            | "use"
            | "where"
            | "while"
            | "model"
            | "input"
            | "output"
    )
}

fn hex_encode(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);
    for &byte in bytes {
        output.push(HEX[(byte >> 4) as usize] as char);
        output.push(HEX[(byte & 0x0f) as usize] as char);
    }
    output
}

fn write_atomic(path: &Path, bytes: &[u8], force: bool) -> Result<(), OnnxError> {
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("model.stark");
    let mut temporary = PathBuf::new();
    let mut file = None;
    for suffix in 0..1000usize {
        temporary = parent.join(format!(".{file_name}.tmp-{}-{suffix}", std::process::id()));
        match OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)
        {
            Ok(opened) => {
                file = Some(opened);
                break;
            }
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
            Err(error) => {
                return Err(OnnxError::new(format!(
                    "cannot create temporary output beside `{}`: {error}",
                    path.display()
                )));
            }
        }
    }
    let mut file = file.ok_or_else(|| OnnxError::new("cannot allocate a temporary output name"))?;
    let result = (|| {
        file.write_all(bytes)?;
        file.sync_all()?;
        drop(file);
        if force && path.exists() {
            match fs::rename(&temporary, path) {
                Ok(()) => return Ok(()),
                Err(error)
                    if cfg!(windows) && error.kind() == std::io::ErrorKind::AlreadyExists =>
                {
                    fs::remove_file(path)?;
                }
                Err(error) => return Err(error),
            }
        }
        fs::rename(&temporary, path)
    })();
    if let Err(error) = result {
        let _ = fs::remove_file(&temporary);
        return Err(OnnxError::new(format!(
            "cannot write output file `{}`: {error}",
            path.display()
        )));
    }
    Ok(())
}
