//! LSP server implementation with message routing and document synchronization.

use crate::parser::{parse_with_options, ParseMode};
use crate::resolve::resolve_with_options;
use crate::source::SourceFile;
use crate::typecheck::analyze_with_options;
use std::collections::HashMap;
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
        _writer: &mut W,
    ) -> std::io::Result<bool> {
        match notif.method.as_str() {
            "initialized" => Ok(true),
            "textDocument/didOpen" => {
                self.handle_did_open(&notif.params);
                Ok(true)
            }
            "textDocument/didChange" => {
                self.handle_did_change(&notif.params);
                Ok(true)
            }
            "textDocument/didClose" => {
                self.handle_did_close(&notif.params);
                Ok(true)
            }
            "textDocument/didSave" => {
                self.handle_did_save(&notif.params);
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
            let source = SourceFile::new(uri, doc.text.clone());
            let options = self.state.options;

            // Parse
            let (ast, mut diagnostics) = parse_with_options(&source, ParseMode::Program, options);

            let mut type_tables = None;

            // Try to resolve names if no parse errors
            if diagnostics
                .iter()
                .all(|d| d.severity != crate::diag::Severity::Error)
            {
                let source_arc = Arc::new(source);
                let (hir, resolve_diags) = resolve_with_options(&ast, source_arc.clone(), options);
                diagnostics.extend(resolve_diags);

                // Try to typecheck if no resolve errors
                if diagnostics
                    .iter()
                    .all(|d| d.severity != crate::diag::Severity::Error)
                {
                    let result = analyze_with_options(&hir, source_arc, options);
                    diagnostics.extend(result.diagnostics);
                    type_tables = Some(result.tables);
                }
            }

            let result = CompilationResult {
                uri: uri.to_string(),
                version: doc.version,
                ast: Some(ast),
                hir: None,
                diagnostics,
                type_tables,
                last_compiled_at: std::time::SystemTime::now(),
            };

            self.state.cache_compilation_result(result);
        }
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
                if let Some(_type_tables) = &cached.type_tables {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_server_creation() {
        let server = Server::new();
        assert_eq!(server.state.open_documents.len(), 0);
    }
}
