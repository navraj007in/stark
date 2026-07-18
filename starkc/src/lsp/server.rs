//! LSP server implementation with message routing and document synchronization.

use crate::analysis::{analyze_project, ProjectInput};
use crate::diag::{DiagnosticBatch, StructuredDiagnostic};
use crate::source::{SourceFile, Span};
use std::collections::{BTreeMap, HashMap};
use std::io::{BufRead, Write};
use std::sync::Arc;

use super::protocol::*;
use super::state::*;

/// LSP server
pub struct Server {
    state: ServerState,
}

impl Server {
    /// Create a new server.
    pub fn new() -> Self {
        Self {
            state: ServerState::new(),
        }
    }

    /// Run the server on stdio.
    pub fn run<R: BufRead, W: Write>(
        &mut self,
        mut reader: R,
        mut writer: W,
    ) -> std::io::Result<()> {
        let mut headers = HashMap::new();
        let mut buffer = String::new();

        loop {
            buffer.clear();
            headers.clear();

            // Read headers
            loop {
                buffer.clear();
                let n = reader.read_line(&mut buffer)?;
                if n == 0 {
                    return Ok(());
                }

                let line = buffer.trim_end();
                if line.is_empty() {
                    break;
                }

                if let Some((key, value)) = line.split_once(':') {
                    headers.insert(key.trim().to_string(), value.trim().to_string());
                }
            }

            // Read content
            if let Some(content_length_str) = headers.get("Content-Length") {
                if let Ok(content_length) = content_length_str.parse::<usize>() {
                    let mut content = vec![0u8; content_length];
                    reader.read_exact(&mut content)?;

                    if let Ok(message_text) = String::from_utf8(content) {
                        match parse_message(&message_text) {
                            Ok(message) => {
                                if !self.handle_message(&message, &mut writer)? {
                                    return Ok(());
                                }
                            }
                            Err(e) => {
                                eprintln!("Failed to parse message: {}", e);
                            }
                        }
                    }
                }
            }
        }
    }

    /// Handle a single message and return false to exit.
    fn handle_message<W: Write>(
        &mut self,
        message: &Message,
        writer: &mut W,
    ) -> std::io::Result<bool> {
        match message {
            Message::Request(req) => self.handle_request(req, writer),
            Message::Notification(notif) => self.handle_notification(notif, writer),
            Message::Response(_) => Ok(true),
        }
    }

    /// Handle a request message.
    fn handle_request<W: Write>(&mut self, req: &Request, writer: &mut W) -> std::io::Result<bool> {
        let response = match req.method.as_str() {
            "initialize" => self.handle_initialize(req.id, &req.params),
            "shutdown" => self.handle_shutdown(req.id, &req.params),
            "textDocument/hover" => self.handle_hover(req.id, &req.params),
            "textDocument/definition" => self.handle_definition(req.id, &req.params),
            "textDocument/references" => self.handle_references(req.id, &req.params),
            "textDocument/formatting" => self.handle_formatting(req.id, &req.params),
            _ => self.error_response(req.id, -32601, "Method not found"),
        };

        self.send_response(&response, writer)?;

        Ok(req.method != "shutdown")
    }

    /// Handle a notification message.
    fn handle_notification<W: Write>(
        &mut self,
        notif: &Notification,
        writer: &mut W,
    ) -> std::io::Result<bool> {
        match notif.method.as_str() {
            "initialized" => Ok(true),
            "textDocument/didOpen" => {
                self.handle_did_open(&notif.params);
                self.publish_cached_diagnostics(&notif.params, writer)?;
                Ok(true)
            }
            "textDocument/didChange" => {
                self.handle_did_change(&notif.params);
                self.publish_cached_diagnostics(&notif.params, writer)?;
                Ok(true)
            }
            "textDocument/didClose" => {
                self.clear_published_diagnostics(&notif.params, writer)?;
                self.handle_did_close(&notif.params);
                Ok(true)
            }
            "textDocument/didSave" => {
                self.handle_did_save(&notif.params);
                self.publish_cached_diagnostics(&notif.params, writer)?;
                Ok(true)
            }
            "exit" => Ok(false),
            _ => Ok(true),
        }
    }

    /// Handle initialize request
    fn handle_initialize(&mut self, id: i64, params: &JsonValue) -> Response {
        if let Some(root_uri) = params.get("rootUri").and_then(|v| v.as_str()) {
            self.state.set_root_uri(root_uri.to_string());
        }

        // `initializationOptions: { "extensions": ["tensor"] }` — matches
        // the CLI's `--extension` flag naming. Unknown names are ignored
        // rather than rejected: a stale/misconfigured client option
        // shouldn't prevent the server from starting.
        if let Some(extensions) = params
            .get("initializationOptions")
            .and_then(|v| v.get("extensions"))
            .and_then(|v| v.as_array())
        {
            let names: Vec<String> = extensions
                .iter()
                .filter_map(|v| v.as_str().map(str::to_string))
                .collect();
            if let Ok(options) = crate::options::options_from_extension_flags(&names) {
                self.state.options = options;
            }
        }

        let mut capabilities = HashMap::new();
        capabilities.insert("textDocumentSync".to_string(), JsonValue::Number(1.0)); // Full
        capabilities.insert("hoverProvider".to_string(), JsonValue::Bool(true));
        capabilities.insert("definitionProvider".to_string(), JsonValue::Bool(true));
        capabilities.insert("referencesProvider".to_string(), JsonValue::Bool(true));
        capabilities.insert(
            "documentFormattingProvider".to_string(),
            JsonValue::Bool(true),
        );

        let mut result = HashMap::new();
        result.insert("capabilities".to_string(), JsonValue::Object(capabilities));

        Response {
            id,
            result: Some(JsonValue::Object(result)),
            error: None,
        }
    }

    /// Handle shutdown request
    fn handle_shutdown(&mut self, id: i64, _params: &JsonValue) -> Response {
        self.state.clear();
        Response {
            id,
            result: Some(JsonValue::Null),
            error: None,
        }
    }

    /// Handle textDocument/didOpen notification
    fn handle_did_open(&mut self, params: &JsonValue) {
        if let Some(text_document) = params.get("textDocument") {
            if let (Some(uri), Some(text), Some(version)) = (
                text_document.get("uri").and_then(|v| v.as_str()),
                text_document.get("text").and_then(|v| v.as_str()),
                text_document.get("version").and_then(|v| v.as_i64()),
            ) {
                self.state
                    .open_document(uri.to_string(), version as i32, text.to_string());
                self.compile_document(uri);
            }
        }
    }

    /// Handle textDocument/didChange notification
    fn handle_did_change(&mut self, params: &JsonValue) {
        if let (Some(uri), Some(version)) = (
            params
                .get("textDocument")
                .and_then(|v| v.get("uri"))
                .and_then(|v| v.as_str()),
            params
                .get("textDocument")
                .and_then(|v| v.get("version"))
                .and_then(|v| v.as_i64()),
        ) {
            if let Some(changes) = params.get("contentChanges").and_then(|v| v.as_array()) {
                if !changes.is_empty() {
                    if let Some(text) = changes[0].get("text").and_then(|v| v.as_str()) {
                        self.state.update_document(
                            uri.to_string(),
                            version as i32,
                            text.to_string(),
                        );
                        self.compile_document(uri);
                    }
                }
            }
        }
    }

    /// Handle textDocument/didClose notification
    fn handle_did_close(&mut self, params: &JsonValue) {
        if let Some(uri) = params
            .get("textDocument")
            .and_then(|v| v.get("uri"))
            .and_then(|v| v.as_str())
        {
            self.state.close_document(uri);
        }
    }

    /// Handle textDocument/didSave notification
    fn handle_did_save(&mut self, params: &JsonValue) {
        if let Some(uri) = params
            .get("textDocument")
            .and_then(|v| v.get("uri"))
            .and_then(|v| v.as_str())
        {
            self.compile_document(uri);
        }
    }

    /// Compile a document and cache results
    fn compile_document(&mut self, uri: &str) {
        if let Some(doc) = self.state.get_document(uri).cloned() {
            let source = Arc::new(SourceFile::new(uri, doc.text.clone()));
            let options = self.state.options;
            let analysis = analyze_project(ProjectInput::program(source), options);

            let result = CompilationResult {
                uri: uri.to_string(),
                version: doc.version,
                analysis,
                last_compiled_at: std::time::SystemTime::now(),
            };

            self.state.cache_compilation_result(result);
        }
    }

    fn publish_cached_diagnostics<W: Write>(
        &self,
        params: &JsonValue,
        writer: &mut W,
    ) -> std::io::Result<()> {
        let uri = params
            .get("textDocument")
            .and_then(|document| document.get("uri"))
            .and_then(JsonValue::as_str);
        let Some(uri) = uri else {
            return Ok(());
        };
        let Some(compilation) = self.state.compilation_cache.get(uri) else {
            return Ok(());
        };
        let root_source = compilation
            .analysis
            .source_map
            .id_for_name(uri)
            .expect("compiled LSP document must be in its source map");
        let versions = HashMap::from([(root_source, i64::from(compilation.version))]);
        let batch = compilation.analysis.diagnostic_batch(&versions);
        self.publish_batch(
            uri,
            compilation.version,
            &compilation.analysis,
            &batch,
            writer,
        )
    }

    fn publish_batch<W: Write>(
        &self,
        root_uri: &str,
        root_version: i32,
        analysis: &crate::analysis::ProjectAnalysis,
        batch: &DiagnosticBatch,
        writer: &mut W,
    ) -> std::io::Result<()> {
        let root_source = analysis
            .source_map
            .id_for_name(root_uri)
            .expect("compiled LSP document must be in its source map");
        let mut grouped: BTreeMap<crate::analysis::SourceId, Vec<&StructuredDiagnostic>> =
            BTreeMap::new();
        grouped.entry(root_source).or_default();
        for diagnostic in &batch.diagnostics {
            grouped
                .entry(diagnostic.primary.source)
                .or_default()
                .push(diagnostic);
        }

        for (source, diagnostics) in grouped {
            let record = analysis
                .source_map
                .get(source)
                .expect("published diagnostic source must exist");
            let uri = source_uri(&record.file.name);
            let version = (source == root_source).then_some(root_version);
            let values = diagnostics
                .into_iter()
                .map(|diagnostic| lsp_diagnostic(analysis, diagnostic))
                .collect();
            let mut params = HashMap::new();
            params.insert("uri".to_string(), JsonValue::String(uri));
            params.insert(
                "version".to_string(),
                version.map_or(JsonValue::Null, |value| JsonValue::Number(f64::from(value))),
            );
            params.insert("diagnostics".to_string(), JsonValue::Array(values));
            self.send_notification(
                "textDocument/publishDiagnostics",
                JsonValue::Object(params),
                writer,
            )?;
        }
        Ok(())
    }

    fn clear_published_diagnostics<W: Write>(
        &self,
        params: &JsonValue,
        writer: &mut W,
    ) -> std::io::Result<()> {
        let Some(uri) = params
            .get("textDocument")
            .and_then(|document| document.get("uri"))
            .and_then(JsonValue::as_str)
        else {
            return Ok(());
        };
        let mut publish = HashMap::new();
        publish.insert("uri".to_string(), JsonValue::String(uri.to_string()));
        publish.insert("diagnostics".to_string(), JsonValue::Array(Vec::new()));
        self.send_notification(
            "textDocument/publishDiagnostics",
            JsonValue::Object(publish),
            writer,
        )
    }

    /// Handle textDocument/hover request
    fn handle_hover(&mut self, id: i64, params: &JsonValue) -> Response {
        if let (Some(uri), Some(line), Some(character)) = (
            params
                .get("textDocument")
                .and_then(|v| v.get("uri"))
                .and_then(|v| v.as_str()),
            params
                .get("position")
                .and_then(|v| v.get("line"))
                .and_then(|v| v.as_i64()),
            params
                .get("position")
                .and_then(|v| v.get("character"))
                .and_then(|v| v.as_i64()),
        ) {
            if let Some(cached) = self.state.compilation_cache.get(uri) {
                if let Some(_type_tables) = &cached.analysis.type_tables {
                    // Find identifiers at position (simplified: just check if any local matches)
                    // In a full implementation, we'd parse the token stream to find the exact identifier
                    let hover_text = format!(
                        "Position: line {}, character {} ({}:{})",
                        line,
                        character,
                        line + 1,
                        character + 1
                    );

                    let mut contents = HashMap::new();
                    contents.insert(
                        "kind".to_string(),
                        JsonValue::String("plaintext".to_string()),
                    );
                    contents.insert("value".to_string(), JsonValue::String(hover_text));

                    let mut result = HashMap::new();
                    result.insert("contents".to_string(), JsonValue::Object(contents));

                    return Response {
                        id,
                        result: Some(JsonValue::Object(result)),
                        error: None,
                    };
                }
            }
        }

        Response {
            id,
            result: Some(JsonValue::Null),
            error: None,
        }
    }

    /// Handle textDocument/definition request
    fn handle_definition(&mut self, id: i64, params: &JsonValue) -> Response {
        if params
            .get("textDocument")
            .and_then(|v| v.get("uri"))
            .is_some()
        {
            return Response {
                id,
                result: Some(JsonValue::Null),
                error: None,
            };
        }

        Response {
            id,
            result: Some(JsonValue::Null),
            error: None,
        }
    }

    /// Handle textDocument/references request
    fn handle_references(&mut self, id: i64, params: &JsonValue) -> Response {
        if params
            .get("textDocument")
            .and_then(|v| v.get("uri"))
            .is_some()
        {
            return Response {
                id,
                result: Some(JsonValue::Array(Vec::new())),
                error: None,
            };
        }

        Response {
            id,
            result: Some(JsonValue::Array(Vec::new())),
            error: None,
        }
    }

    /// Handle textDocument/formatting request. Formats the live (possibly
    /// unsaved) buffer via `formatter::format_file` and returns a single
    /// full-document `TextEdit`. Returns `null` (no edits) if the buffer
    /// does not currently parse cleanly — the formatter has no text to
    /// fall back on for the parts it couldn't build a tree for.
    fn handle_formatting(&mut self, id: i64, params: &JsonValue) -> Response {
        let uri = params
            .get("textDocument")
            .and_then(|v| v.get("uri"))
            .and_then(|v| v.as_str());

        if let Some(uri) = uri {
            if let Some(doc) = self.state.get_document(uri) {
                let text = doc.text.clone();
                let source = SourceFile::new(uri, text.clone());
                if let Ok(formatted) = crate::formatter::format_file(&source, self.state.options) {
                    let (end_line, end_col) = source.line_col(text.len() as u32);
                    let mut range = HashMap::new();
                    range.insert("start".to_string(), lsp_position(0, 0));
                    range.insert(
                        "end".to_string(),
                        lsp_position((end_line - 1) as i64, (end_col - 1) as i64),
                    );
                    let mut edit = HashMap::new();
                    edit.insert("range".to_string(), JsonValue::Object(range));
                    edit.insert("newText".to_string(), JsonValue::String(formatted));
                    return Response {
                        id,
                        result: Some(JsonValue::Array(vec![JsonValue::Object(edit)])),
                        error: None,
                    };
                }
            }
        }

        Response {
            id,
            result: Some(JsonValue::Null),
            error: None,
        }
    }

    /// Send a response
    fn send_response<W: Write>(&self, response: &Response, writer: &mut W) -> std::io::Result<()> {
        let mut obj = HashMap::new();
        obj.insert("jsonrpc".to_string(), JsonValue::String("2.0".to_string()));
        obj.insert("id".to_string(), JsonValue::Number(response.id as f64));

        if let Some(error) = &response.error {
            let mut err_obj = HashMap::new();
            err_obj.insert("code".to_string(), JsonValue::Number(error.code as f64));
            err_obj.insert(
                "message".to_string(),
                JsonValue::String(error.message.clone()),
            );
            obj.insert("error".to_string(), JsonValue::Object(err_obj));
        } else {
            obj.insert(
                "result".to_string(),
                response.result.clone().unwrap_or(JsonValue::Null),
            );
        }

        let content = JsonValue::Object(obj).to_string();
        let header = format!("Content-Length: {}\r\n\r\n", content.len());

        writer.write_all(header.as_bytes())?;
        writer.write_all(content.as_bytes())?;
        writer.flush()?;

        Ok(())
    }

    fn send_notification<W: Write>(
        &self,
        method: &str,
        params: JsonValue,
        writer: &mut W,
    ) -> std::io::Result<()> {
        let mut object = HashMap::new();
        object.insert("jsonrpc".to_string(), JsonValue::String("2.0".to_string()));
        object.insert("method".to_string(), JsonValue::String(method.to_string()));
        object.insert("params".to_string(), params);
        let content = JsonValue::Object(object).to_string();
        let header = format!("Content-Length: {}\r\n\r\n", content.len());
        writer.write_all(header.as_bytes())?;
        writer.write_all(content.as_bytes())?;
        writer.flush()
    }

    /// Create an error response
    fn error_response(&self, id: i64, code: i32, message: &str) -> Response {
        Response {
            id,
            result: None,
            error: Some(ResponseError {
                code,
                message: message.to_string(),
            }),
        }
    }
}

impl Default for Server {
    fn default() -> Self {
        Self::new()
    }
}

fn lsp_position(line: i64, character: i64) -> JsonValue {
    let mut pos = HashMap::new();
    pos.insert("line".to_string(), JsonValue::Number(line as f64));
    pos.insert("character".to_string(), JsonValue::Number(character as f64));
    JsonValue::Object(pos)
}

fn lsp_diagnostic(
    analysis: &crate::analysis::ProjectAnalysis,
    diagnostic: &StructuredDiagnostic,
) -> JsonValue {
    let primary = analysis
        .source_map
        .get(diagnostic.primary.source)
        .expect("LSP diagnostic source must exist");
    let mut value = HashMap::new();
    value.insert(
        "range".to_string(),
        lsp_range(&primary.file, diagnostic.primary.span),
    );
    value.insert(
        "severity".to_string(),
        JsonValue::Number(match diagnostic.severity {
            crate::diag::Severity::Error => 1.0,
            crate::diag::Severity::Warning => 2.0,
        }),
    );
    value.insert(
        "code".to_string(),
        JsonValue::String(diagnostic.code.clone()),
    );
    value.insert(
        "source".to_string(),
        JsonValue::String("starkc".to_string()),
    );
    value.insert(
        "message".to_string(),
        JsonValue::String(match &diagnostic.label {
            Some(label) => format!("{}\n{label}", diagnostic.message),
            None => diagnostic.message.clone(),
        }),
    );
    if !diagnostic.related.is_empty() {
        let related = diagnostic
            .related
            .iter()
            .map(|related| {
                let source = analysis
                    .source_map
                    .get(related.location.source)
                    .expect("LSP related diagnostic source must exist");
                let mut location = HashMap::new();
                location.insert(
                    "uri".to_string(),
                    JsonValue::String(source_uri(&source.file.name)),
                );
                location.insert(
                    "range".to_string(),
                    lsp_range(&source.file, related.location.span),
                );
                let mut information = HashMap::new();
                information.insert("location".to_string(), JsonValue::Object(location));
                information.insert(
                    "message".to_string(),
                    JsonValue::String(related.message.clone()),
                );
                JsonValue::Object(information)
            })
            .collect();
        value.insert("relatedInformation".to_string(), JsonValue::Array(related));
    }
    let mut data = HashMap::new();
    data.insert(
        "sourceVersion".to_string(),
        diagnostic
            .source_version
            .map_or(JsonValue::Null, |version| JsonValue::Number(version as f64)),
    );
    data.insert(
        "sourceId".to_string(),
        JsonValue::Number(diagnostic.primary.source.as_u32() as f64),
    );
    let (source_kind, package) = match &primary.provenance {
        crate::analysis::SourceProvenance::Root { package } => ("root", package),
        crate::analysis::SourceProvenance::Module { package } => ("module", package),
    };
    data.insert(
        "sourceKind".to_string(),
        JsonValue::String(source_kind.to_string()),
    );
    data.insert(
        "package".to_string(),
        package.clone().map_or(JsonValue::Null, JsonValue::String),
    );
    data.insert(
        "extensions".to_string(),
        JsonValue::Array(if analysis.options.tensor() {
            vec![JsonValue::String("tensor".to_string())]
        } else {
            Vec::new()
        }),
    );
    data.insert(
        "ruleId".to_string(),
        diagnostic
            .rule_id
            .clone()
            .map_or(JsonValue::Null, JsonValue::String),
    );
    data.insert(
        "deviationId".to_string(),
        diagnostic
            .deviation_id
            .clone()
            .map_or(JsonValue::Null, JsonValue::String),
    );
    data.insert(
        "help".to_string(),
        JsonValue::Array(
            diagnostic
                .help
                .iter()
                .map(|help| JsonValue::String(help.clone()))
                .collect(),
        ),
    );
    data.insert(
        "notes".to_string(),
        JsonValue::Array(
            diagnostic
                .notes
                .iter()
                .map(|note| JsonValue::String(note.clone()))
                .collect(),
        ),
    );
    value.insert("data".to_string(), JsonValue::Object(data));
    JsonValue::Object(value)
}

fn lsp_range(file: &SourceFile, span: Span) -> JsonValue {
    let (start_line, start_column) = file.line_col(span.lo);
    let (end_line, end_column) = file.line_col(span.hi);
    let mut range = HashMap::new();
    range.insert(
        "start".to_string(),
        lsp_position((start_line - 1) as i64, (start_column - 1) as i64),
    );
    range.insert(
        "end".to_string(),
        lsp_position((end_line - 1) as i64, (end_column - 1) as i64),
    );
    JsonValue::Object(range)
}

fn source_uri(name: &str) -> String {
    if name.contains("://") {
        name.to_string()
    } else {
        format!("file://{name}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_server_creation() {
        let server = Server::new();
        assert_eq!(server.state.open_documents.len(), 0);
    }

    #[test]
    fn publishes_shared_diagnostics_with_document_version() {
        let uri = "file:///diagnostic.stark";
        let mut server = Server::new();
        server
            .state
            .open_document(uri.to_string(), 23, "fn main() { missing; }".to_string());
        server.compile_document(uri);

        let mut document = HashMap::new();
        document.insert("uri".to_string(), JsonValue::String(uri.to_string()));
        let mut params = HashMap::new();
        params.insert("textDocument".to_string(), JsonValue::Object(document));
        let mut output = Vec::new();
        server
            .publish_cached_diagnostics(&JsonValue::Object(params), &mut output)
            .unwrap();

        let message = String::from_utf8(output).unwrap();
        assert!(message.contains("textDocument/publishDiagnostics"));
        assert!(message.contains("\"version\":23"));
        assert!(message.contains("\"sourceVersion\":23"));
        assert!(message.contains("\"code\":\"E0200\""), "{message}");
        assert!(message.contains("\"diagnostics\":[{"), "{message}");
    }
}
