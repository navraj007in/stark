//! LSP server implementation with message routing and document synchronization.

use crate::analysis::{analyze_project, ProjectInput};
use crate::diag::{DiagnosticBatch, StructuredDiagnostic};
use crate::package::{find_package_root, PackageGraph};
use crate::source::{SourceFile, Span};
use std::collections::{BTreeMap, HashMap};
use std::io::{BufRead, Write};
use std::path::{Path, PathBuf};
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
            "textDocument/completion" => self.handle_completion(req.id, &req.params),
            "textDocument/signatureHelp" => self.handle_signature_help(req.id, &req.params),
            "textDocument/rename" => self.handle_rename(req.id, &req.params),
            "textDocument/documentSymbol" => self.handle_document_symbol(req.id, &req.params),
            "textDocument/semanticTokens/full" => {
                self.handle_semantic_tokens_full(req.id, &req.params)
            }
            "workspace/symbol" => self.handle_workspace_symbol(req.id, &req.params),
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

        // `initializationOptions: { "extensions": ["tensor"] }` matches the
        // CLI's `--extension` flag naming and validation policy.
        if let Some(extensions) = params
            .get("initializationOptions")
            .and_then(|v| v.get("extensions"))
            .and_then(|v| v.as_array())
        {
            let names: Vec<String> = extensions
                .iter()
                .filter_map(|v| v.as_str().map(str::to_string))
                .collect();
            match crate::options::options_from_extension_flags(&names) {
                Ok(options) => self.state.options = options,
                Err(error) => {
                    return self.error_response(id, -32602, &error.to_string());
                }
            }
        }

        let mut capabilities = HashMap::new();
        capabilities.insert(
            "textDocumentSync".to_string(),
            JsonValue::number_from_i64(1),
        ); // Full
        capabilities.insert("hoverProvider".to_string(), JsonValue::Bool(true));
        capabilities.insert("definitionProvider".to_string(), JsonValue::Bool(true));
        capabilities.insert("referencesProvider".to_string(), JsonValue::Bool(true));
        let mut completion = HashMap::new();
        completion.insert("resolveProvider".to_string(), JsonValue::Bool(false));
        completion.insert(
            "triggerCharacters".to_string(),
            JsonValue::Array(vec![
                JsonValue::String(":".to_string()),
                JsonValue::String(".".to_string()),
            ]),
        );
        capabilities.insert(
            "completionProvider".to_string(),
            JsonValue::Object(completion),
        );
        let mut signature_help = HashMap::new();
        signature_help.insert(
            "triggerCharacters".to_string(),
            JsonValue::Array(vec![
                JsonValue::String("(".to_string()),
                JsonValue::String(",".to_string()),
            ]),
        );
        signature_help.insert(
            "retriggerCharacters".to_string(),
            JsonValue::Array(vec![JsonValue::String(",".to_string())]),
        );
        capabilities.insert(
            "signatureHelpProvider".to_string(),
            JsonValue::Object(signature_help),
        );
        capabilities.insert("renameProvider".to_string(), JsonValue::Bool(true));
        capabilities.insert("documentSymbolProvider".to_string(), JsonValue::Bool(true));
        capabilities.insert("workspaceSymbolProvider".to_string(), JsonValue::Bool(true));
        capabilities.insert(
            "semanticTokensProvider".to_string(),
            semantic_tokens_provider_capability(),
        );
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
            let options = self.state.options;
            let analysis = self
                .package_input_for_document(&doc)
                .map(|input| analyze_project(input, options))
                .unwrap_or_else(|| {
                    let source = Arc::new(SourceFile::new(uri, doc.text.clone()));
                    analyze_project(ProjectInput::program(source), options)
                });

            let result = CompilationResult {
                uri: uri.to_string(),
                version: doc.version,
                analysis,
                last_compiled_at: std::time::SystemTime::now(),
                // DEV-213: recorded so `invalidate_package_of` can find this analysis's siblings
                // without touching the filesystem on an edit.
                package_root: self.package_root_for_document(&doc),
            };

            self.state.cache_compilation_result(result);
        }
    }

    /// The package root a document belongs to, or `None` for a loose single file.
    ///
    /// DEV-213 needs this in two places -- to build the project input, and to stamp the cache
    /// entry so invalidation can find siblings. It is one function rather than two so the two
    /// callers cannot answer the question differently; a second copy of "which package is this
    /// URI in" is exactly the duplicated-authority shape `AS8-DA-*` catalogues.
    fn package_root_for_document(&self, doc: &OpenDocument) -> Option<PathBuf> {
        let path = file_uri_to_path(&doc.uri)?;
        let dir = if path.is_dir() {
            path.as_path()
        } else {
            path.parent()?
        };
        find_package_root(dir).ok()
    }

    fn package_input_for_document(&self, doc: &OpenDocument) -> Option<ProjectInput> {
        let manifest = self.package_root_for_document(doc)?;
        let graph = PackageGraph::load_from_root_with_modes(&manifest, false, true).ok()?;
        let overlays = self.open_document_overlays();
        Some(ProjectInput::package_with_overlays(graph, overlays))
    }

    fn open_document_overlays(&self) -> HashMap<PathBuf, String> {
        self.state
            .open_documents
            .values()
            .filter_map(|doc| {
                let path = file_uri_to_path(&doc.uri)?;
                let canonical = path.canonicalize().ok()?;
                Some((canonical, doc.text.clone()))
            })
            .collect()
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
        let Some(root_source) = source_id_for_uri(&compilation.analysis, uri) else {
            return Ok(());
        };
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
        let root_source = source_id_for_uri(analysis, root_uri)
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
            // The document the client asked about is republished under the *client's own* URI.
            // A round trip through the source map would re-render it from the canonicalised
            // disk path, and a workspace reached through a symlink would then get its
            // diagnostics on a URI no open editor holds.
            let uri = if source == root_source {
                root_uri.to_string()
            } else {
                source_uri(&record.file)
            };
            let version = (source == root_source).then_some(root_version);
            let values = diagnostics
                .into_iter()
                .map(|diagnostic| lsp_diagnostic(analysis, diagnostic))
                .collect();
            let mut params = HashMap::new();
            params.insert("uri".to_string(), JsonValue::String(uri));
            params.insert(
                "version".to_string(),
                version.map_or(JsonValue::Null, |value| {
                    JsonValue::number_from_i64(i64::from(value))
                }),
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
        if let Some((cached, source, offset)) = self.document_position(params) {
            if let Some(symbol) = cached.analysis.symbol_at(source, offset) {
                let mut lines = Vec::new();
                if let Some(signature) = cached.analysis.signature(symbol) {
                    lines.push(signature);
                }
                if let Some(ty) = cached.analysis.type_of(symbol) {
                    if !lines.iter().any(|line| line == &ty) {
                        lines.push(ty);
                    }
                }
                if !lines.is_empty() {
                    let mut contents = HashMap::new();
                    contents.insert(
                        "kind".to_string(),
                        JsonValue::String("markdown".to_string()),
                    );
                    contents.insert(
                        "value".to_string(),
                        JsonValue::String(format!("```stark\n{}\n```", lines.join("\n"))),
                    );

                    let mut result = HashMap::new();
                    result.insert("contents".to_string(), JsonValue::Object(contents));

                    return Response {
                        id,
                        result: Some(JsonValue::Object(result)),
                        error: None,
                    };
                }
            }
            if let Some(handle) = cached.analysis.hir_at(source, offset) {
                if let Some(ty) = cached.analysis.type_of(handle) {
                    let mut contents = HashMap::new();
                    contents.insert(
                        "kind".to_string(),
                        JsonValue::String("markdown".to_string()),
                    );
                    contents.insert(
                        "value".to_string(),
                        JsonValue::String(format!("```stark\n{ty}\n```")),
                    );
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
        if let Some((cached, source, offset)) = self.document_position(params) {
            if let Some(symbol) = cached.analysis.symbol_at(source, offset) {
                if let Some(definition) = cached.analysis.definition(symbol) {
                    if let Some(location) = lsp_location(&cached.analysis, definition) {
                        return Response {
                            id,
                            result: Some(location),
                            error: None,
                        };
                    }
                }
            }
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
        if let Some((cached, source, offset)) = self.document_position(params) {
            if let Some(symbol) = cached.analysis.symbol_at(source, offset) {
                let include_declaration = params
                    .get("context")
                    .and_then(|value| value.get("includeDeclaration"))
                    .and_then(JsonValue::as_bool)
                    .unwrap_or(false);
                let mut locations = cached
                    .analysis
                    .references(symbol)
                    .into_iter()
                    .filter_map(|reference| lsp_location(&cached.analysis, reference))
                    .collect::<Vec<_>>();
                if include_declaration {
                    if let Some(definition) = cached
                        .analysis
                        .definition(symbol)
                        .and_then(|definition| lsp_location(&cached.analysis, definition))
                    {
                        locations.push(definition);
                    }
                }
                return Response {
                    id,
                    result: Some(JsonValue::Array(locations)),
                    error: None,
                };
            }
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

    fn document_position(
        &self,
        params: &JsonValue,
    ) -> Option<(&CompilationResult, crate::analysis::SourceId, u32)> {
        let uri = params
            .get("textDocument")
            .and_then(|v| v.get("uri"))
            .and_then(|v| v.as_str())?;
        let line = params
            .get("position")
            .and_then(|v| v.get("line"))
            .and_then(|v| v.as_i64())?;
        let character = params
            .get("position")
            .and_then(|v| v.get("character"))
            .and_then(|v| v.as_i64())?;
        let line = u32::try_from(line).ok()?;
        let character = u32::try_from(character).ok()?;
        let cached = self.state.compilation_cache.get(uri)?;
        let source = source_id_for_uri(&cached.analysis, uri)?;
        let record = cached.analysis.source_map.get(source)?;
        let offset =
            super::position::lsp_position_to_byte_offset(&record.file.src, line, character)?;
        Some((cached, source, offset))
    }

    fn handle_completion(&mut self, id: i64, params: &JsonValue) -> Response {
        let Some((cached, source, offset)) = self.document_position(params) else {
            return Response {
                id,
                result: Some(JsonValue::Null),
                error: None,
            };
        };
        let record = cached
            .analysis
            .source_map
            .get(source)
            .expect("completion source must exist");
        let prefix = completion_prefix(&record.file.src, offset);
        let items = cached
            .analysis
            .completion_candidates(&prefix)
            .into_iter()
            .map(|candidate| completion_item(&candidate))
            .collect::<Vec<_>>();
        let mut list = HashMap::new();
        list.insert("isIncomplete".to_string(), JsonValue::Bool(false));
        list.insert("items".to_string(), JsonValue::Array(items));
        Response {
            id,
            result: Some(JsonValue::Object(list)),
            error: None,
        }
    }

    fn handle_signature_help(&mut self, id: i64, params: &JsonValue) -> Response {
        let Some((cached, source, offset)) = self.document_position(params) else {
            return Response {
                id,
                result: Some(JsonValue::Null),
                error: None,
            };
        };
        let Some(help) = cached.analysis.signature_help_at(source, offset) else {
            return Response {
                id,
                result: Some(JsonValue::Null),
                error: None,
            };
        };
        Response {
            id,
            result: Some(signature_help_result(&help)),
            error: None,
        }
    }

    fn handle_rename(&mut self, id: i64, params: &JsonValue) -> Response {
        let Some(new_name) = params.get("newName").and_then(JsonValue::as_str) else {
            return self.error_response(id, -32602, "rename requires newName");
        };
        let Some((cached, source, offset)) = self.document_position(params) else {
            return Response {
                id,
                result: Some(JsonValue::Null),
                error: None,
            };
        };
        let Some(symbol) = cached.analysis.symbol_at(source, offset) else {
            return Response {
                id,
                result: Some(JsonValue::Null),
                error: None,
            };
        };
        let Some(edits) = cached.analysis.rename_edits(symbol, new_name) else {
            return self.error_response(
                id,
                -32602,
                "rename is not safe for this symbol or requested name",
            );
        };
        let mut changes: HashMap<String, Vec<JsonValue>> = HashMap::new();
        for edit in edits {
            let Some(record) = cached.analysis.source_map.get(edit.source) else {
                continue;
            };
            let mut text_edit = HashMap::new();
            text_edit.insert("range".to_string(), lsp_range(&record.file, edit.span));
            text_edit.insert(
                "newText".to_string(),
                JsonValue::String(new_name.to_string()),
            );
            changes
                .entry(source_uri(&record.file))
                .or_default()
                .push(JsonValue::Object(text_edit));
        }
        let mut change_object = HashMap::new();
        for (uri, edits) in changes {
            change_object.insert(uri, JsonValue::Array(edits));
        }
        let mut workspace_edit = HashMap::new();
        workspace_edit.insert("changes".to_string(), JsonValue::Object(change_object));
        Response {
            id,
            result: Some(JsonValue::Object(workspace_edit)),
            error: None,
        }
    }

    fn handle_document_symbol(&mut self, id: i64, params: &JsonValue) -> Response {
        let Some(uri) = params
            .get("textDocument")
            .and_then(|value| value.get("uri"))
            .and_then(JsonValue::as_str)
        else {
            return Response {
                id,
                result: Some(JsonValue::Null),
                error: None,
            };
        };
        let Some(cached) = self.state.compilation_cache.get(uri) else {
            return Response {
                id,
                result: Some(JsonValue::Array(Vec::new())),
                error: None,
            };
        };
        // `source_id_for_uri`, not `id_for_name`: in a package build the source-map name is the
        // package-relative path, so a URI never matches it by name — it has to fall through to
        // the disk-path comparison, exactly as every position-based handler does.
        let Some(source) = source_id_for_uri(&cached.analysis, uri) else {
            return Response {
                id,
                result: Some(JsonValue::Array(Vec::new())),
                error: None,
            };
        };
        let items = cached
            .analysis
            .document_symbols(source)
            .into_iter()
            .filter_map(|symbol| symbol_information(&cached.analysis, symbol))
            .collect();
        Response {
            id,
            result: Some(JsonValue::Array(items)),
            error: None,
        }
    }

    fn handle_workspace_symbol(&mut self, id: i64, params: &JsonValue) -> Response {
        let query = params
            .get("query")
            .and_then(JsonValue::as_str)
            .unwrap_or("");
        let mut symbols = Vec::new();
        for cached in self.state.compilation_cache.values() {
            symbols.extend(
                cached
                    .analysis
                    .workspace_symbols(query)
                    .into_iter()
                    .filter_map(|symbol| symbol_information(&cached.analysis, symbol)),
            );
        }
        symbols.sort_by_key(|symbol| {
            symbol
                .get("name")
                .and_then(JsonValue::as_str)
                .unwrap_or("")
                .to_string()
        });
        Response {
            id,
            result: Some(JsonValue::Array(symbols)),
            error: None,
        }
    }

    fn handle_semantic_tokens_full(&mut self, id: i64, params: &JsonValue) -> Response {
        let Some(uri) = params
            .get("textDocument")
            .and_then(|value| value.get("uri"))
            .and_then(JsonValue::as_str)
        else {
            return Response {
                id,
                result: Some(JsonValue::Null),
                error: None,
            };
        };
        let Some(cached) = self.state.compilation_cache.get(uri) else {
            return Response {
                id,
                result: Some(semantic_tokens_result(Vec::new())),
                error: None,
            };
        };
        // See `handle_document_symbol`: a package build's source-map names are package-relative.
        let Some(source) = source_id_for_uri(&cached.analysis, uri) else {
            return Response {
                id,
                result: Some(semantic_tokens_result(Vec::new())),
                error: None,
            };
        };
        let Some(record) = cached.analysis.source_map.get(source) else {
            return Response {
                id,
                result: Some(semantic_tokens_result(Vec::new())),
                error: None,
            };
        };
        let encoded = encode_semantic_tokens(&record.file, cached.analysis.semantic_tokens(source));
        Response {
            id,
            result: Some(semantic_tokens_result(encoded)),
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
                    let end = super::position::byte_offset_to_lsp_position(
                        &source.src,
                        text.len() as u32,
                    );
                    let mut range = HashMap::new();
                    range.insert("start".to_string(), lsp_position(0, 0));
                    range.insert("end".to_string(), lsp_position(end.line, end.character));
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
        obj.insert("id".to_string(), JsonValue::number_from_i64(response.id));

        if let Some(error) = &response.error {
            let mut err_obj = HashMap::new();
            err_obj.insert(
                "code".to_string(),
                JsonValue::number_from_i64(i64::from(error.code)),
            );
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

fn lsp_position(line: u32, character: u32) -> JsonValue {
    let mut pos = HashMap::new();
    pos.insert(
        "line".to_string(),
        JsonValue::number_from_i64(i64::from(line)),
    );
    pos.insert(
        "character".to_string(),
        JsonValue::number_from_i64(i64::from(character)),
    );
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
        JsonValue::number_from_i64(match diagnostic.severity {
            crate::diag::Severity::Error => 1,
            crate::diag::Severity::Warning => 2,
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
                    JsonValue::String(source_uri(&source.file)),
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
            .map_or(JsonValue::Null, |version| {
                JsonValue::number_from_i64(version)
            }),
    );
    data.insert(
        "sourceId".to_string(),
        JsonValue::number_from_i64(i64::from(diagnostic.primary.source.as_u32())),
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
    let converted = super::position::span_to_lsp_range(file, span);
    let mut range = HashMap::new();
    range.insert(
        "start".to_string(),
        lsp_position(converted.start.line, converted.start.character),
    );
    range.insert(
        "end".to_string(),
        lsp_position(converted.end.line, converted.end.character),
    );
    JsonValue::Object(range)
}

fn lsp_location(
    analysis: &crate::analysis::ProjectAnalysis,
    location: crate::analysis::SourceLocation,
) -> Option<JsonValue> {
    let record = analysis.source_map.get(location.source)?;
    let mut object = HashMap::new();
    object.insert(
        "uri".to_string(),
        JsonValue::String(source_uri(&record.file)),
    );
    object.insert("range".to_string(), lsp_range(&record.file, location.span));
    Some(JsonValue::Object(object))
}

fn completion_item(candidate: &crate::analysis::CompletionCandidate) -> JsonValue {
    let mut item = HashMap::new();
    item.insert(
        "label".to_string(),
        JsonValue::String(candidate.name.clone()),
    );
    item.insert(
        "kind".to_string(),
        JsonValue::number_from_i64(i64::from(completion_kind(candidate.kind))),
    );
    if let Some(signature) = &candidate.signature {
        item.insert("detail".to_string(), JsonValue::String(signature.clone()));
    }
    item.insert(
        "sortText".to_string(),
        JsonValue::String(candidate.name.clone()),
    );
    JsonValue::Object(item)
}

fn signature_help_result(help: &crate::analysis::SignatureHelp) -> JsonValue {
    let parameters = help
        .parameters
        .iter()
        .map(|parameter| {
            let mut object = HashMap::new();
            object.insert("label".to_string(), JsonValue::String(parameter.clone()));
            JsonValue::Object(object)
        })
        .collect::<Vec<_>>();
    let mut signature = HashMap::new();
    signature.insert(
        "label".to_string(),
        JsonValue::String(help.signature.clone()),
    );
    signature.insert("parameters".to_string(), JsonValue::Array(parameters));

    let mut result = HashMap::new();
    result.insert(
        "signatures".to_string(),
        JsonValue::Array(vec![JsonValue::Object(signature)]),
    );
    result.insert("activeSignature".to_string(), JsonValue::number_from_i64(0));
    result.insert(
        "activeParameter".to_string(),
        JsonValue::number_from_i64(help.active_parameter as i64),
    );
    JsonValue::Object(result)
}

fn symbol_information(
    analysis: &crate::analysis::ProjectAnalysis,
    symbol: &crate::analysis::Symbol,
) -> Option<JsonValue> {
    let location = lsp_location(
        analysis,
        crate::analysis::SourceLocation {
            source: symbol.source,
            span: symbol.span,
        },
    )?;
    let mut item = HashMap::new();
    item.insert("name".to_string(), JsonValue::String(symbol.name.clone()));
    item.insert(
        "kind".to_string(),
        JsonValue::number_from_i64(i64::from(symbol_kind(symbol.kind))),
    );
    item.insert("location".to_string(), location);
    Some(JsonValue::Object(item))
}

fn symbol_kind(kind: crate::analysis::SymbolKind) -> u8 {
    match kind {
        crate::analysis::SymbolKind::Function => 12,
        crate::analysis::SymbolKind::Struct | crate::analysis::SymbolKind::Model => 23,
        crate::analysis::SymbolKind::Enum => 10,
        crate::analysis::SymbolKind::Trait => 11,
        crate::analysis::SymbolKind::Constant => 14,
        crate::analysis::SymbolKind::TypeAlias => 26,
        crate::analysis::SymbolKind::Module => 2,
    }
}

fn semantic_tokens_provider_capability() -> JsonValue {
    let mut legend = HashMap::new();
    legend.insert(
        "tokenTypes".to_string(),
        JsonValue::Array(
            SEMANTIC_TOKEN_TYPES
                .iter()
                .map(|token| JsonValue::String((*token).to_string()))
                .collect(),
        ),
    );
    legend.insert(
        "tokenModifiers".to_string(),
        JsonValue::Array(vec![JsonValue::String("declaration".to_string())]),
    );
    let mut provider = HashMap::new();
    provider.insert("legend".to_string(), JsonValue::Object(legend));
    provider.insert("full".to_string(), JsonValue::Bool(true));
    provider.insert("range".to_string(), JsonValue::Bool(false));
    JsonValue::Object(provider)
}

const SEMANTIC_TOKEN_TYPES: &[&str] = &[
    "namespace",
    "type",
    "struct",
    "enum",
    "interface",
    "function",
    "method",
    "property",
    "parameter",
    "variable",
    "constant",
    "macro",
    "decorator",
];

fn encode_semantic_tokens(
    file: &SourceFile,
    mut tokens: Vec<crate::analysis::SemanticToken>,
) -> Vec<JsonValue> {
    tokens.sort_by_key(|token| (token.span.lo, token.span.hi));
    let mut encoded = Vec::new();
    let mut previous_line = 0u32;
    let mut previous_start = 0u32;
    for token in tokens {
        let range = super::position::span_to_lsp_range(file, token.span);
        if range.start.line != range.end.line {
            continue;
        }
        let delta_line = range.start.line.saturating_sub(previous_line);
        let delta_start = if delta_line == 0 {
            range.start.character.saturating_sub(previous_start)
        } else {
            range.start.character
        };
        let length = range.end.character.saturating_sub(range.start.character);
        if length == 0 {
            continue;
        }
        encoded.push(JsonValue::number_from_i64(i64::from(delta_line)));
        encoded.push(JsonValue::number_from_i64(i64::from(delta_start)));
        encoded.push(JsonValue::number_from_i64(i64::from(length)));
        encoded.push(JsonValue::number_from_i64(i64::from(semantic_token_type(
            token.kind,
        ))));
        encoded.push(JsonValue::number_from_i64(if token.declaration {
            1
        } else {
            0
        }));
        previous_line = range.start.line;
        previous_start = range.start.character;
    }
    encoded
}

fn semantic_tokens_result(data: Vec<JsonValue>) -> JsonValue {
    let mut result = HashMap::new();
    result.insert("data".to_string(), JsonValue::Array(data));
    JsonValue::Object(result)
}

fn semantic_token_type(kind: crate::analysis::SemanticTokenKind) -> u32 {
    match kind {
        crate::analysis::SemanticTokenKind::Namespace => 0,
        crate::analysis::SemanticTokenKind::Type => 1,
        crate::analysis::SemanticTokenKind::Struct => 2,
        crate::analysis::SemanticTokenKind::Enum => 3,
        crate::analysis::SemanticTokenKind::Trait => 4,
        crate::analysis::SemanticTokenKind::Function => 5,
        crate::analysis::SemanticTokenKind::Method => 6,
        crate::analysis::SemanticTokenKind::Field => 7,
        crate::analysis::SemanticTokenKind::Parameter => 8,
        crate::analysis::SemanticTokenKind::Local => 9,
        crate::analysis::SemanticTokenKind::Constant => 10,
        crate::analysis::SemanticTokenKind::Builtin => 11,
        crate::analysis::SemanticTokenKind::Extension => 12,
    }
}

fn completion_kind(kind: crate::analysis::SymbolKind) -> u8 {
    match kind {
        crate::analysis::SymbolKind::Function => 3,
        crate::analysis::SymbolKind::Module => 9,
        crate::analysis::SymbolKind::Enum => 13,
        crate::analysis::SymbolKind::Constant => 21,
        crate::analysis::SymbolKind::Struct | crate::analysis::SymbolKind::Model => 22,
        crate::analysis::SymbolKind::Trait => 8,
        crate::analysis::SymbolKind::TypeAlias => 25,
    }
}

fn completion_prefix(source: &str, offset: u32) -> String {
    let offset = (offset as usize).min(source.len());
    let offset = if source.is_char_boundary(offset) {
        offset
    } else {
        let mut boundary = offset;
        while boundary > 0 && !source.is_char_boundary(boundary) {
            boundary -= 1;
        }
        boundary
    };
    let start = source[..offset]
        .char_indices()
        .rev()
        .find_map(|(index, ch)| (!is_identifier_continue(ch)).then_some(index + ch.len_utf8()))
        .unwrap_or(0);
    source[start..offset].to_string()
}

fn is_identifier_continue(ch: char) -> bool {
    ch == '_' || ch.is_alphanumeric()
}

/// The URI an editor can open for a source file.
///
/// `SourceFile::name` is identity-bearing, not a location: for a package build it is
/// `<package>/<path within the package>` and deliberately never an absolute checkout path
/// (see `source.rs`). Formatting that name as `file://{name}` makes the *package name* the
/// URI authority — `file://my-pkg/src/main.stark` — which no client can resolve, so
/// diagnostics land on a phantom document and every `Location` is unopenable.
/// `disk_path` is the field `source.rs` designates for pointing a human at a file, so
/// prefer it; the name is only a usable location for a single-file compile, where the
/// caller passed a path.
fn source_uri(file: &SourceFile) -> String {
    if let Some(path) = &file.disk_path {
        return path_to_file_uri(path);
    }
    if file.name.contains("://") {
        return file.name.clone();
    }
    path_to_file_uri(Path::new(&file.name))
}

/// Absolutises `path` (relative names come from single-file compiles, where they are relative
/// to the server's working directory) and renders it as a `file://` URI.
///
/// A URI path is always slash-separated and always rooted, which a Windows path is neither:
/// `C:\dir\file.stark` has to become `file:///C:/dir/file.stark`. Emitting the native form
/// leaves backslashes in a JSON string, where they are escape characters, so the message the
/// client receives is not even parseable.
fn path_to_file_uri(path: &Path) -> String {
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()
            .map(|dir| dir.join(path))
            .unwrap_or_else(|_| path.to_path_buf())
    };
    let text = absolute.to_string_lossy().into_owned();
    // Only on Windows: a backslash is a legal character in a Unix file name.
    #[cfg(windows)]
    let text = text.replace('\\', "/");
    let rooted = if text.starts_with('/') {
        text
    } else {
        format!("/{text}")
    };
    format!("file://{}", percent_encode_uri_path(&rooted))
}

/// Percent-encodes the characters that are not legal unescaped in a URI path. The inverse of
/// `percent_decode_uri_path`, which the server applies to every client-supplied URI.
fn percent_encode_uri_path(path: &str) -> String {
    let mut encoded = String::with_capacity(path.len());
    for byte in path.bytes() {
        match byte {
            b'A'..=b'Z'
            | b'a'..=b'z'
            | b'0'..=b'9'
            | b'-'
            | b'.'
            | b'_'
            | b'~'
            | b'/'
            | b'!'
            | b'$'
            | b'&'
            | b'\''
            | b'('
            | b')'
            | b'*'
            | b'+'
            | b','
            | b';'
            | b'='
            | b':'
            | b'@' => encoded.push(byte as char),
            _ => encoded.push_str(&format!("%{byte:02X}")),
        }
    }
    encoded
}

fn file_uri_to_path(uri: &str) -> Option<PathBuf> {
    let path = uri.strip_prefix("file://")?;
    let decoded = percent_decode_uri_path(path);
    // `file:///C:/dir/file.stark` — the leading slash roots the URI path, it is not part of the
    // Windows path underneath it.
    let native = decoded
        .strip_prefix('/')
        .filter(|rest| starts_with_drive_letter(rest))
        .unwrap_or(&decoded);
    Some(PathBuf::from(native))
}

/// Whether `path` opens with a `C:`-style Windows drive prefix.
fn starts_with_drive_letter(path: &str) -> bool {
    let mut chars = path.chars();
    matches!(chars.next(), Some(letter) if letter.is_ascii_alphabetic())
        && chars.next() == Some(':')
}

fn source_id_for_uri(
    analysis: &crate::analysis::ProjectAnalysis,
    uri: &str,
) -> Option<crate::analysis::SourceId> {
    if let Some(source) = analysis.source_map.id_for_name(uri) {
        return Some(source);
    }
    let requested = file_uri_to_path(uri).and_then(|path| path.canonicalize().ok());
    analysis.source_map.files().iter().find_map(|record| {
        let disk = record.file.disk_path.as_ref()?;
        let disk = disk.canonicalize().ok()?;
        (Some(disk) == requested).then_some(record.id)
    })
}

fn percent_decode_uri_path(path: &str) -> String {
    let mut decoded = Vec::with_capacity(path.len());
    let bytes = path.as_bytes();
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index] == b'%' && index + 2 < bytes.len() {
            if let (Some(hi), Some(lo)) = (hex_value(bytes[index + 1]), hex_value(bytes[index + 2]))
            {
                decoded.push((hi << 4) | lo);
                index += 3;
                continue;
            }
        }
        decoded.push(bytes[index]);
        index += 1;
    }
    String::from_utf8_lossy(&decoded).into_owned()
}

fn hex_value(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    }
}

#[cfg(test)]
mod tests {

    /// **AS8 — one `ProjectAnalysis` per open URI, invalidated per URI, merged across URIs.**
    ///
    /// `ServerState::compilation_cache` is keyed by URI and each value owns a WHOLE-PACKAGE
    /// analysis. `update_document` removes only the edited URI's entry, and
    /// `handle_workspace_symbol` merges symbols from EVERY cached analysis. Those three facts
    /// together mean an edit to one file leaves every other open file's cached analysis holding
    /// the OLD symbols for it, and the workspace-symbol response is assembled from both.
    ///
    /// This is why the AS8 profile measured the duplication: the cost is the visible half, and
    /// this is the half that changes an answer.
    #[test]
    fn dev213_editing_one_file_invalidates_every_analysis_of_its_package() {
        let package_dir = std::env::temp_dir().join(format!(
            "as8_lsp_stale_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let src_dir = package_dir.join("src");
        std::fs::create_dir_all(&src_dir).unwrap();
        std::fs::write(
            package_dir.join("starkpkg.json"),
            r#"{"name":"app","version":"0.1.0","entry":"src/main.stark"}"#,
        )
        .unwrap();
        std::fs::write(src_dir.join("main.stark"), "mod child;\nfn main() { }\n").unwrap();
        let child_path = src_dir.join("child.stark");
        std::fs::write(&child_path, "pub fn alpha_symbol() -> Int32 { 1 }\n").unwrap();

        let main_uri = path_to_file_uri(&src_dir.join("main.stark"));
        let child_uri = path_to_file_uri(&child_path);

        let mut server = Server::new();
        // Two files of ONE package open. Each gets its own whole-package analysis.
        server.state.open_document(
            main_uri.clone(),
            1,
            "mod child;\nfn main() { }\n".to_string(),
        );
        server.compile_document(&main_uri);
        server.state.open_document(
            child_uri.clone(),
            1,
            "pub fn alpha_symbol() -> Int32 { 1 }\n".to_string(),
        );
        server.compile_document(&child_uri);
        assert_eq!(
            server.state.compilation_cache.len(),
            2,
            "two open URIs of one package must produce two independent analyses for this to matter"
        );

        // Rename the symbol, and recompile ONLY the edited URI — which is exactly what
        // `didChange` does. Before DEV-213's repair this left `main.stark`'s whole-package
        // analysis untouched and still carrying the OLD name.
        server.state.update_document(
            child_uri.clone(),
            2,
            "pub fn renamed_symbol() -> Int32 { 1 }\n".to_string(),
        );
        server.compile_document(&child_uri);

        // What `handle_workspace_symbol` does: merge symbols across every cached analysis.
        let merged: Vec<String> = server
            .state
            .compilation_cache
            .values()
            .flat_map(|cached| cached.analysis.workspace_symbols("symbol"))
            .map(|symbol| symbol.name.clone())
            .collect();

        assert!(
            merged.iter().any(|name| name == "renamed_symbol"),
            "the edited URI's analysis must see the new name; got {merged:?}"
        );
        // DEV-213, REPAIRED. This assertion is AS8's, with its polarity flipped exactly as AS8's
        // own message instructed -- the test is not deleted, because what it pins is the same
        // fact either way: what `workspace/symbol` can see after a single-file edit.
        //
        // `alpha_symbol` no longer exists anywhere in the package. Before the repair,
        // `main.stark`'s never-invalidated whole-package analysis still carried it and the merged
        // response contained BOTH names. Now the edit sweeps every analysis of the package, so the
        // only surviving answer is the one that is true.
        assert!(
            !merged.iter().any(|name| name == "alpha_symbol"),
            "DEV-213: `alpha_symbol` was renamed and must not survive in ANY cached analysis of \
             this package. Its presence means a sibling URI's whole-package analysis outlived the \
             edit -- the exact defect AS8 demonstrated; got {merged:?}"
        );
        // The sweep must be an invalidation, not a purge of the whole cache: unrelated packages
        // and loose files keep their analyses. `main.stark` is recompiled on demand, so a
        // subsequent request answers correctly rather than answering nothing.
        server.compile_document(&main_uri);
        let after: Vec<String> = server
            .state
            .compilation_cache
            .values()
            .flat_map(|cached| cached.analysis.workspace_symbols("symbol"))
            .map(|symbol| symbol.name.clone())
            .collect();
        assert!(
            after.iter().any(|name| name == "renamed_symbol"),
            "after recompiling the sibling, the package's only symbol must be the new one; \
             got {after:?}"
        );
        assert!(
            !after.iter().any(|name| name == "alpha_symbol"),
            "the stale name must not reappear once the sibling is recompiled; got {after:?}"
        );

        let _ = std::fs::remove_dir_all(&package_dir);
    }
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

    #[test]
    fn package_lsp_analysis_uses_open_document_overlays() {
        let package_dir = std::env::temp_dir().join(format!(
            "stark-lsp-package-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let src_dir = package_dir.join("src");
        std::fs::create_dir_all(&src_dir).unwrap();
        std::fs::write(
            package_dir.join("starkpkg.json"),
            r#"{"name":"app","version":"0.1.0","entry":"src/main.stark"}"#,
        )
        .unwrap();
        std::fs::write(src_dir.join("main.stark"), "mod child;\nfn main() { }\n").unwrap();
        let child_path = src_dir.join("child.stark");
        std::fs::write(&child_path, "pub fn value() -> Int32 { 1 }\n").unwrap();

        let child_uri = path_to_file_uri(&child_path);
        let mut server = Server::new();
        server.state.open_document(
            child_uri.clone(),
            9,
            "pub fn value() -> Int32 { missing }\n".to_string(),
        );
        server.compile_document(&child_uri);

        let cached = server
            .state
            .compilation_cache
            .get(&child_uri)
            .expect("package document must compile into cache");
        assert!(
            cached.analysis.package_graph.is_some(),
            "file inside starkpkg.json package must use package analysis"
        );
        let child_source = source_id_for_uri(&cached.analysis, &child_uri)
            .expect("package source map must resolve opened module URI");
        let versions = HashMap::from([(child_source, 9)]);
        let batch = cached.analysis.diagnostic_batch(&versions);
        assert!(
            batch.diagnostics.iter().any(
                |diagnostic| diagnostic.code == "E0200" && diagnostic.source_version == Some(9)
            ),
            "open module overlay should produce versioned unresolved-symbol diagnostic"
        );
    }

    #[test]
    fn json_rpc_transport_publishes_package_overlay_diagnostics() {
        let package_dir = std::env::temp_dir().join(format!(
            "stark-lsp-transport-package-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let src_dir = package_dir.join("src");
        std::fs::create_dir_all(&src_dir).unwrap();
        std::fs::write(
            package_dir.join("starkpkg.json"),
            r#"{"name":"app","version":"0.1.0","entry":"src/main.stark"}"#,
        )
        .unwrap();
        std::fs::write(src_dir.join("main.stark"), "mod child;\nfn main() { }\n").unwrap();
        let child_path = src_dir.join("child.stark");
        std::fs::write(&child_path, "pub fn value() -> Int32 { 1 }\n").unwrap();
        let child_uri = path_to_file_uri(&child_path);
        let input = format!(
            "{}{}{}",
            frame(r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}"#),
            frame(&did_open(
                &child_uri,
                11,
                "pub fn value() -> Int32 { missing }\n",
            )),
            frame(r#"{"jsonrpc":"2.0","id":2,"method":"shutdown","params":{}}"#),
        );
        let mut server = Server::new();
        let mut output = Vec::new();
        server
            .run(std::io::Cursor::new(input.into_bytes()), &mut output)
            .unwrap();
        let messages = decode_framed_output(&output);
        assert!(
            messages.iter().any(|message| {
                message.contains("textDocument/publishDiagnostics")
                    && message.contains("\"uri\":\"")
                    && message.contains("\"version\":11")
                    && message.contains("\"code\":\"E0200\"")
            }),
            "expected package overlay diagnostic publication, got {messages:?}"
        );
    }

    #[test]
    fn package_lsp_sessions_keep_tensor_extension_isolated() {
        let package_dir = std::env::temp_dir().join(format!(
            "stark-lsp-extension-package-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let src_dir = package_dir.join("src");
        std::fs::create_dir_all(&src_dir).unwrap();
        std::fs::write(
            package_dir.join("starkpkg.json"),
            r#"{"name":"app","version":"0.1.0","entry":"src/main.stark"}"#,
        )
        .unwrap();
        let main_path = src_dir.join("main.stark");
        let tensor_source = "model Resnet50V17<N: Dim> {\n    input data: Tensor<Float32, [N, 3, 224, 224]>;\n    output scores: Tensor<Float32, [N, 1000]>;\n}\n";
        std::fs::write(&main_path, tensor_source).unwrap();
        let uri = path_to_file_uri(&main_path);

        let mut core = Server::new();
        let response = core.handle_initialize(1, &JsonValue::Object(HashMap::new()));
        assert!(response.error.is_none());
        core.state
            .open_document(uri.clone(), 1, tensor_source.to_string());
        core.compile_document(&uri);
        let core_messages: Vec<String> = core.state.compilation_cache[&uri]
            .analysis
            .diagnostics
            .iter()
            .map(|diagnostic| diagnostic.message.clone())
            .collect();
        assert!(
            core_messages
                .iter()
                .any(|message| message.contains("extension `tensor`")),
            "{core_messages:?}"
        );

        let mut tensor = Server::new();
        let response =
            tensor.handle_initialize(2, &initialize_params_with_extensions(vec!["tensor"]));
        assert!(response.error.is_none());
        tensor
            .state
            .open_document(uri.clone(), 2, tensor_source.to_string());
        tensor.compile_document(&uri);
        let tensor_messages: Vec<String> = tensor.state.compilation_cache[&uri]
            .analysis
            .diagnostics
            .iter()
            .map(|diagnostic| diagnostic.message.clone())
            .collect();
        assert!(
            !tensor_messages
                .iter()
                .any(|message| message.contains("extension `tensor`")),
            "{tensor_messages:?}"
        );

        let _ = std::fs::remove_dir_all(package_dir);
    }

    /// A package build names its sources `<package>/<path in package>` — an identity token, not
    /// a location. Every URI the server hands back must still be one the client can open, and
    /// every source lookup must resolve a client URI that will never match those names. Both
    /// failed silently before: diagnostics were published to `file://app/src/main.stark`, whose
    /// *authority* is the package name, and document symbols and semantic tokens came back empty.
    #[test]
    fn package_build_returns_openable_uris_and_whole_file_results() {
        let package_dir = std::env::temp_dir().join(format!(
            "stark-lsp-package-uri-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let src_dir = package_dir.join("src");
        std::fs::create_dir_all(&src_dir).unwrap();
        std::fs::write(
            package_dir.join("starkpkg.json"),
            r#"{"name":"app","version":"0.1.0","entry":"src/main.stark"}"#,
        )
        .unwrap();
        let main_path = src_dir.join("main.stark");
        let source = "fn helper() -> Int32 { 1 }\nfn main() { helper(); }\n";
        std::fs::write(&main_path, source).unwrap();
        let uri = path_to_file_uri(&main_path);

        let mut server = Server::new();
        let mut output = Vec::new();
        server
            .run(
                std::io::Cursor::new(
                    format!(
                        "{}{}{}",
                        frame(r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}"#),
                        frame(&did_open(&uri, 1, source)),
                        frame(r#"{"jsonrpc":"2.0","id":2,"method":"shutdown","params":{}}"#),
                    )
                    .into_bytes(),
                ),
                &mut output,
            )
            .unwrap();
        let published = decode_framed_output(&output)
            .into_iter()
            .find(|message| message.contains("textDocument/publishDiagnostics"))
            .expect("package document must publish diagnostics");
        assert!(
            published.contains(&format!("\"uri\":\"{uri}\"")),
            "diagnostics must be published under the client's own URI, got {published}"
        );

        let cached_uri = uri.clone();
        let mut server = Server::new();
        server
            .state
            .open_document(cached_uri.clone(), 1, source.to_string());
        server.compile_document(&cached_uri);
        assert!(
            server.state.compilation_cache[&cached_uri]
                .analysis
                .package_graph
                .is_some(),
            "test must exercise package analysis, not single-file analysis"
        );

        let document = JsonValue::Object(HashMap::from([(
            "textDocument".to_string(),
            JsonValue::Object(HashMap::from([(
                "uri".to_string(),
                JsonValue::String(cached_uri.clone()),
            )])),
        )]));

        let symbols = server.handle_document_symbol(3, &document).result.unwrap();
        let JsonValue::Array(symbols) = symbols else {
            panic!("documentSymbol must return an array");
        };
        assert!(
            symbols.len() == 2,
            "package document symbols must cover both functions, got {symbols:?}"
        );
        for symbol in &symbols {
            let location_uri = symbol
                .get("location")
                .and_then(|location| location.get("uri"))
                .and_then(JsonValue::as_str)
                .expect("symbol must carry a location URI");
            assert!(
                location_uri.starts_with("file:///"),
                "symbol URI must be an absolute file URI, got {location_uri}"
            );
        }

        let tokens = server
            .handle_semantic_tokens_full(4, &document)
            .result
            .unwrap();
        let data = tokens
            .get("data")
            .and_then(|data| match data {
                JsonValue::Array(values) => Some(values.len()),
                _ => None,
            })
            .expect("semantic tokens must return a data array");
        assert!(data > 0, "package document must produce semantic tokens");

        // `helper` in the call on line 1 — definition resolves back to line 0.
        let call_site = JsonValue::Object(HashMap::from([
            (
                "textDocument".to_string(),
                JsonValue::Object(HashMap::from([(
                    "uri".to_string(),
                    JsonValue::String(cached_uri.clone()),
                )])),
            ),
            (
                "position".to_string(),
                JsonValue::Object(HashMap::from([
                    ("line".to_string(), JsonValue::number_from_i64(1)),
                    ("character".to_string(), JsonValue::number_from_i64(12)),
                ])),
            ),
        ]));
        let definition = server.handle_definition(5, &call_site).result.unwrap();
        let definition_uri = definition
            .get("uri")
            .and_then(JsonValue::as_str)
            .expect("definition must resolve inside a package");
        assert!(
            definition_uri.starts_with("file:///")
                && file_uri_to_path(definition_uri).is_some_and(|path| path.exists()),
            "definition URI must name a file that exists on disk, got {definition_uri}"
        );
    }

    /// The URI a client receives is interpolated into a JSON frame, so a native Windows path
    /// handed over verbatim makes the message unparseable — every backslash reads as an escape.
    #[test]
    fn file_uris_are_rooted_slash_separated_and_round_trip_to_native_paths() {
        let native = if cfg!(windows) {
            r"C:\dir\a b\file.stark"
        } else {
            "/dir/a b/file.stark"
        };
        let uri = path_to_file_uri(Path::new(native));

        assert!(uri.starts_with("file:///"), "URI must be rooted, got {uri}");
        assert!(!uri.contains('\\'), "URI must not carry backslashes: {uri}");
        assert!(
            uri.contains("a%20b"),
            "URI must percent-encode spaces, got {uri}"
        );
        assert_eq!(
            file_uri_to_path(&uri).expect("file URI must convert back to a path"),
            PathBuf::from(native)
        );
    }

    #[test]
    fn initialize_advertises_only_semantically_supported_handlers() {
        let mut server = Server::new();
        let response = server.handle_initialize(1, &JsonValue::Object(HashMap::new()));
        let result = response.result.expect("initialize must return result");
        let capabilities = result
            .get("capabilities")
            .expect("initialize result must include capabilities");

        assert_eq!(
            capabilities.get("documentFormattingProvider"),
            Some(&JsonValue::Bool(true))
        );
        assert_eq!(
            capabilities.get("hoverProvider"),
            Some(&JsonValue::Bool(true))
        );
        assert_eq!(
            capabilities.get("definitionProvider"),
            Some(&JsonValue::Bool(true))
        );
        assert_eq!(
            capabilities.get("referencesProvider"),
            Some(&JsonValue::Bool(true))
        );
        assert!(capabilities.get("completionProvider").is_some());
        assert_eq!(
            capabilities.get("renameProvider"),
            Some(&JsonValue::Bool(true))
        );
        assert_eq!(
            capabilities.get("documentSymbolProvider"),
            Some(&JsonValue::Bool(true))
        );
        assert_eq!(
            capabilities.get("workspaceSymbolProvider"),
            Some(&JsonValue::Bool(true))
        );
        assert!(capabilities.get("semanticTokensProvider").is_some());
        assert!(capabilities.get("signatureHelpProvider").is_some());
        assert_eq!(capabilities.get("inlayHintProvider"), None);
    }

    #[test]
    fn initialize_rejects_unknown_and_duplicate_extensions_like_cli() {
        for (id, extensions, expected) in [
            (11, vec!["unknown"], "unknown extension `unknown`"),
            (
                12,
                vec!["tensor", "tensor"],
                "extension `tensor` enabled more than once",
            ),
        ] {
            let mut server = Server::new();
            let params = initialize_params_with_extensions(extensions);
            let response = server.handle_initialize(id, &params);
            let error = response
                .error
                .expect("initialize must reject invalid extensions");
            assert_eq!(error.code, -32602);
            assert!(
                error.message.contains(expected),
                "expected {expected:?}, got {:?}",
                error.message
            );
            assert_eq!(server.state.options, crate::options::LanguageOptions::CORE);
        }
    }

    #[test]
    fn shutdown_clears_lsp_extension_session_state() {
        let mut server = Server::new();
        let response =
            server.handle_initialize(1, &initialize_params_with_extensions(vec!["tensor"]));
        assert!(response.error.is_none());
        assert!(server.state.options.tensor());

        let shutdown = server.handle_shutdown(2, &JsonValue::Object(HashMap::new()));
        assert!(shutdown.error.is_none());
        assert_eq!(server.state.options, crate::options::LanguageOptions::CORE);

        let response = server.handle_initialize(3, &JsonValue::Object(HashMap::new()));
        assert!(response.error.is_none());
        assert_eq!(server.state.options, crate::options::LanguageOptions::CORE);
    }

    #[test]
    fn hover_uses_compiler_symbol_signature() {
        let uri = "file:///hover.stark";
        let mut server = Server::new();
        server.state.open_document(
            uri.to_string(),
            1,
            "fn helper() -> Int32 { 1 }\nfn main() { helper(); }\n".to_string(),
        );
        server.compile_document(uri);

        let response = server.handle_hover(2, &text_position_params(uri, 1, 12));
        let result = response.result.expect("hover must return a result");
        let contents = result.get("contents").expect("hover must include contents");
        assert_eq!(
            contents.get("value"),
            Some(&JsonValue::String(
                "```stark\nfn helper() -> Int32\n```".to_string()
            ))
        );
    }

    #[test]
    fn definition_and_references_use_resolved_symbol_identity() {
        let uri = "file:///definition.stark";
        let mut server = Server::new();
        server.state.open_document(
            uri.to_string(),
            1,
            "fn helper() -> Int32 { 1 }\nfn main() { helper(); }\n".to_string(),
        );
        server.compile_document(uri);

        let definition = server.handle_definition(3, &text_position_params(uri, 1, 12));
        let definition = definition.result.expect("definition must return a result");
        assert_location_contains(&definition, uri, 0, 3, 0, 9);

        let references = server.handle_references(4, &text_position_params(uri, 1, 12));
        let references = references.result.expect("references must return a result");
        let JsonValue::Array(references) = references else {
            panic!("references result must be an array");
        };
        assert_eq!(references.len(), 1);
        assert_location_contains(&references[0], uri, 1, 12, 1, 18);

        let references =
            server.handle_references(5, &text_position_params_with_declaration(uri, 1, 12, true));
        let references = references.result.expect("references must return a result");
        let JsonValue::Array(references) = references else {
            panic!("references result must be an array");
        };
        assert_eq!(references.len(), 2);
        assert_location_contains(&references[1], uri, 0, 3, 0, 9);
    }

    #[test]
    fn completion_returns_indexed_semantic_symbols() {
        let uri = "file:///completion.stark";
        let mut server = Server::new();
        server.state.open_document(
            uri.to_string(),
            1,
            "fn helper() -> Int32 { 1 }\nfn main() { hel }\n".to_string(),
        );
        server.compile_document(uri);

        let response = server.handle_completion(6, &text_position_params(uri, 1, 15));
        let result = response.result.expect("completion must return a result");
        let items = result
            .get("items")
            .expect("completion list must include items");
        let JsonValue::Array(items) = items else {
            panic!("completion items must be an array");
        };
        let helper = items
            .iter()
            .find(|item| item.get("label") == Some(&JsonValue::String("helper".to_string())))
            .expect("helper completion must be present");
        assert_eq!(
            helper.get("detail"),
            Some(&JsonValue::String("fn helper() -> Int32".to_string()))
        );
    }

    #[test]
    fn signature_help_uses_resolved_callee_and_argument_spans() {
        let uri = "file:///signature.stark";
        let mut server = Server::new();
        server.state.open_document(
            uri.to_string(),
            1,
            concat!(
                "fn outer(left: Int32, right: Int32) -> Int32 { left }\n",
                "fn helper(first: Int32, second: Int32) -> Int32 { first }\n",
                "fn main() { helper(outer(1, 2), 3); }\n",
            )
            .to_string(),
        );
        server.compile_document(uri);

        let response = server.handle_signature_help(6, &text_position_params(uri, 2, 33));
        let result = response
            .result
            .expect("signature help must return a result");
        assert_eq!(
            result.get("activeParameter"),
            Some(&JsonValue::number_from_i64(1))
        );
        let JsonValue::Array(signatures) = result
            .get("signatures")
            .expect("signature help must include signatures")
        else {
            panic!("signatures must be an array");
        };
        let signature = signatures
            .first()
            .expect("signature help must include one signature");
        assert_eq!(
            signature.get("label"),
            Some(&JsonValue::String(
                "fn helper(first: Int32, second: Int32) -> Int32".to_string()
            ))
        );
        let JsonValue::Array(parameters) = signature
            .get("parameters")
            .expect("signature help must include parameters")
        else {
            panic!("parameters must be an array");
        };
        assert_eq!(parameters.len(), 2);
        assert_eq!(
            parameters[1].get("label"),
            Some(&JsonValue::String("second: Int32".to_string()))
        );

        let nested = server.handle_signature_help(7, &text_position_params(uri, 2, 28));
        let nested = nested.result.expect("nested signature help must return");
        assert_eq!(
            nested.get("activeParameter"),
            Some(&JsonValue::number_from_i64(1))
        );
        let JsonValue::Array(signatures) = nested
            .get("signatures")
            .expect("nested signature help must include signatures")
        else {
            panic!("signatures must be an array");
        };
        assert_eq!(
            signatures[0].get("label"),
            Some(&JsonValue::String(
                "fn outer(left: Int32, right: Int32) -> Int32".to_string()
            ))
        );
    }

    #[test]
    fn document_and_workspace_symbols_use_compiler_symbol_index() {
        let uri = "file:///symbols.stark";
        let mut server = Server::new();
        server.state.open_document(
            uri.to_string(),
            1,
            "pub struct User { id: Int32 }\nfn helper() -> Int32 { 1 }\n".to_string(),
        );
        server.compile_document(uri);

        let document = server.handle_document_symbol(7, &text_document_params(uri));
        let document = document
            .result
            .expect("document symbols must return a result");
        let JsonValue::Array(document) = document else {
            panic!("document symbols must be an array");
        };
        assert_eq!(document.len(), 2);
        assert!(document
            .iter()
            .any(|symbol| symbol.get("name") == Some(&JsonValue::String("User".to_string()))));

        let workspace = server.handle_workspace_symbol(8, &workspace_symbol_params("help"));
        let workspace = workspace
            .result
            .expect("workspace symbols must return a result");
        let JsonValue::Array(workspace) = workspace else {
            panic!("workspace symbols must be an array");
        };
        assert_eq!(workspace.len(), 1);
        assert_eq!(
            workspace[0].get("name"),
            Some(&JsonValue::String("helper".to_string()))
        );
    }

    #[test]
    fn rename_uses_resolved_symbol_identity_for_safe_top_level_symbols() {
        let uri = "file:///rename.stark";
        let mut server = Server::new();
        server.state.open_document(
            uri.to_string(),
            1,
            "fn helper() -> Int32 { 1 }\nfn main() { helper(); }\n".to_string(),
        );
        server.compile_document(uri);

        let response = server.handle_rename(9, &rename_params(uri, 1, 12, "renamed"));
        let result = response.result.expect("rename must return a result");
        let changes = result.get("changes").expect("rename must include changes");
        let edits = changes
            .get(uri)
            .expect("rename must include edits for the current uri");
        let JsonValue::Array(edits) = edits else {
            panic!("rename edits must be an array");
        };
        assert_eq!(edits.len(), 2);
        assert_eq!(
            edits[0].get("newText"),
            Some(&JsonValue::String("renamed".to_string()))
        );

        let response = server.handle_rename(10, &rename_params(uri, 1, 12, "main"));
        assert!(response.error.is_some(), "colliding rename must be refused");
    }

    #[test]
    fn semantic_tokens_are_encoded_from_compiler_classification() {
        let uri = "file:///tokens.stark";
        let mut server = Server::new();
        server.state.open_document(
            uri.to_string(),
            1,
            "fn helper() -> Int32 { 1 }\nfn main() { let value = helper(); value; }\n".to_string(),
        );
        server.compile_document(uri);

        let response = server.handle_semantic_tokens_full(11, &text_document_params(uri));
        let result = response.result.expect("semantic tokens must return result");
        let data = result
            .get("data")
            .expect("semantic tokens must include data");
        let JsonValue::Array(data) = data else {
            panic!("semantic token data must be an array");
        };
        assert!(!data.is_empty(), "semantic token data must not be empty");
        assert_eq!(
            data.len() % 5,
            0,
            "semantic tokens are encoded in groups of 5"
        );
    }

    #[test]
    fn json_rpc_transport_handles_initialize_unknown_method_and_shutdown() {
        let input = format!(
            "{}{}{}",
            frame(r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}"#),
            frame(r#"{"jsonrpc":"2.0","id":2,"method":"workspace/unknown","params":{}}"#),
            frame(r#"{"jsonrpc":"2.0","id":3,"method":"shutdown","params":{}}"#),
        );
        let mut server = Server::new();
        let mut output = Vec::new();
        server
            .run(std::io::Cursor::new(input.into_bytes()), &mut output)
            .unwrap();
        let messages = decode_framed_output(&output);
        assert_eq!(messages.len(), 3);
        assert!(messages[0].contains("\"id\":1"));
        assert!(messages[0].contains("\"capabilities\""));
        assert!(messages[1].contains("\"id\":2"));
        assert!(messages[1].contains("\"code\":-32601"));
        assert!(messages[2].contains("\"id\":3"));
        assert!(messages[2].contains("\"result\":null"));
    }

    #[test]
    fn json_rpc_transport_publishes_diagnostics_for_open_document() {
        let uri = "file:///transport.stark";
        let open = format!(
            r#"{{
                "jsonrpc":"2.0",
                "method":"textDocument/didOpen",
                "params":{{
                    "textDocument":{{
                        "uri":"{uri}",
                        "languageId":"stark",
                        "version":7,
                        "text":"fn main() {{ missing; }}"
                    }}
                }}
            }}"#
        );
        let input = format!(
            "{}{}{}",
            frame(r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}"#),
            frame(&open),
            frame(r#"{"jsonrpc":"2.0","id":2,"method":"shutdown","params":{}}"#),
        );
        let mut server = Server::new();
        let mut output = Vec::new();
        server
            .run(std::io::Cursor::new(input.into_bytes()), &mut output)
            .unwrap();
        let messages = decode_framed_output(&output);
        assert!(
            messages.iter().any(
                |message| message.contains("textDocument/publishDiagnostics")
                    && message.contains("\"version\":7")
                    && message.contains("\"code\":\"E0200\"")
            ),
            "expected versioned compiler diagnostic publication, got {messages:?}"
        );
    }

    #[test]
    fn json_rpc_transport_change_clears_diagnostics_and_close_publishes_clear() {
        let uri = "file:///lifecycle.stark";
        let open = did_open(uri, 1, "fn main() { missing; }");
        let change = format!(
            r#"{{
                "jsonrpc":"2.0",
                "method":"textDocument/didChange",
                "params":{{
                    "textDocument":{{"uri":"{uri}","version":2}},
                    "contentChanges":[{{"text":"fn main() {{ }}"}}]
                }}
            }}"#
        );
        let close = format!(
            r#"{{
                "jsonrpc":"2.0",
                "method":"textDocument/didClose",
                "params":{{"textDocument":{{"uri":"{uri}"}}}}
            }}"#
        );
        let input = format!(
            "{}{}{}{}{}",
            frame(r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}"#),
            frame(&open),
            frame(&change),
            frame(&close),
            frame(r#"{"jsonrpc":"2.0","id":2,"method":"shutdown","params":{}}"#),
        );
        let mut server = Server::new();
        let mut output = Vec::new();
        server
            .run(std::io::Cursor::new(input.into_bytes()), &mut output)
            .unwrap();
        let messages = decode_framed_output(&output);
        assert!(
            messages
                .iter()
                .any(|message| message.contains("\"version\":1")
                    && message.contains("\"code\":\"E0200\"")),
            "open should publish the initial error, got {messages:?}"
        );
        assert!(
            messages
                .iter()
                .any(|message| message.contains("\"version\":2")
                    && message.contains("\"diagnostics\":[]")),
            "change should publish cleared diagnostics for version 2, got {messages:?}"
        );
        assert!(
            messages.iter().any(
                |message| message.contains("textDocument/publishDiagnostics")
                    && message.contains("\"diagnostics\":[]")
            ),
            "close should publish a diagnostic clear, got {messages:?}"
        );
    }

    #[test]
    fn json_rpc_transport_malformed_message_does_not_stop_later_request() {
        let input = format!(
            "{}{}",
            frame(r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":"#),
            frame(r#"{"jsonrpc":"2.0","id":2,"method":"shutdown","params":{}}"#),
        );
        let mut server = Server::new();
        let mut output = Vec::new();
        server
            .run(std::io::Cursor::new(input.into_bytes()), &mut output)
            .unwrap();
        let messages = decode_framed_output(&output);
        assert_eq!(messages.len(), 1);
        assert!(messages[0].contains("\"id\":2"));
        assert!(messages[0].contains("\"result\":null"));
    }

    #[test]
    fn json_rpc_transport_exit_notification_stops_without_response() {
        let input = frame(r#"{"jsonrpc":"2.0","method":"exit","params":{}}"#);
        let mut server = Server::new();
        let mut output = Vec::new();
        server
            .run(std::io::Cursor::new(input.into_bytes()), &mut output)
            .unwrap();
        assert!(
            output.is_empty(),
            "exit notification must not produce a response"
        );
    }

    #[test]
    fn json_rpc_transport_serves_semantic_feature_requests() {
        let uri = "file:///features.stark";
        let input = format!(
            "{}{}{}{}{}{}{}{}{}",
            frame(r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}"#),
            frame(&did_open(
                uri,
                1,
                "fn helper(value: Int32, other: Int32) -> Int32 { value }\nfn main() { helper(1, 2); }\n",
            )),
            frame(&request_position(2, "textDocument/hover", uri, 1, 12)),
            frame(&request_position(3, "textDocument/definition", uri, 1, 12)),
            frame(&request_position(4, "textDocument/completion", uri, 1, 15)),
            frame(&request_position(
                5,
                "textDocument/signatureHelp",
                uri,
                1,
                22
            )),
            frame(&request_document(
                6,
                "textDocument/semanticTokens/full",
                uri
            )),
            frame(&request_rename(7, uri, 1, 12, "renamed")),
            frame(r#"{"jsonrpc":"2.0","id":8,"method":"shutdown","params":{}}"#),
        );
        let mut server = Server::new();
        let mut output = Vec::new();
        server
            .run(std::io::Cursor::new(input.into_bytes()), &mut output)
            .unwrap();
        let messages = decode_framed_output(&output);
        assert!(messages.iter().any(|message| {
            message.contains("\"id\":2")
                && message.contains("fn helper(value: Int32, other: Int32) -> Int32")
        }));
        assert!(messages.iter().any(|message| {
            message.contains("\"id\":3")
                && message.contains("\"uri\":\"file:///features.stark\"")
                && message.contains("\"range\"")
        }));
        assert!(messages.iter().any(
            |message| message.contains("\"id\":4") && message.contains("\"label\":\"helper\"")
        ));
        assert!(messages.iter().any(|message| message.contains("\"id\":5")
            && message.contains("\"activeParameter\":1")
            && message.contains("fn helper(value: Int32, other: Int32) -> Int32")));
        assert!(messages
            .iter()
            .any(|message| message.contains("\"id\":6") && message.contains("\"data\"")));
        assert!(messages.iter().any(|message| {
            message.contains("\"id\":7")
                && message.contains("\"changes\"")
                && message.contains("\"newText\":\"renamed\"")
        }));
    }

    #[test]
    fn json_rpc_transport_handles_cancellation_formatting_workspace_symbol_and_rename_failure() {
        let uri = "file:///protocol-more.stark";
        let input = format!(
            "{}{}{}{}{}{}{}",
            frame(r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}"#),
            frame(&did_open(
                uri,
                1,
                "fn helper()->Int32{1}\nfn main(){helper();}\n",
            )),
            frame(r#"{"jsonrpc":"2.0","method":"$/cancelRequest","params":{"id":99}}"#),
            frame(&request_document(2, "textDocument/formatting", uri)),
            frame(
                r#"{"jsonrpc":"2.0","id":3,"method":"workspace/symbol","params":{"query":"help"}}"#
            ),
            frame(&request_rename(4, uri, 1, 10, "1bad")),
            frame(r#"{"jsonrpc":"2.0","id":5,"method":"shutdown","params":{}}"#),
        );
        let mut server = Server::new();
        let mut output = Vec::new();
        server
            .run(std::io::Cursor::new(input.into_bytes()), &mut output)
            .unwrap();
        let messages = decode_framed_output(&output);
        assert!(
            messages.iter().any(|message| message.contains("\"id\":2")
                && message.contains("\"newText\"")
                && message.contains("fn helper() -> Int32")),
            "formatting response missing from {messages:?}"
        );
        assert!(
            messages
                .iter()
                .any(|message| message.contains("\"id\":3")
                    && message.contains("\"name\":\"helper\"")),
            "workspace symbol response missing from {messages:?}"
        );
        assert!(
            messages
                .iter()
                .any(|message| message.contains("\"id\":4") && message.contains("\"code\":-32602")),
            "rename failure response missing from {messages:?}"
        );
        assert!(
            !messages
                .iter()
                .any(|message| message.contains("cancelRequest")),
            "cancel notification must not receive a response"
        );
    }

    fn text_position_params(uri: &str, line: i64, character: i64) -> JsonValue {
        text_position_params_with_declaration(uri, line, character, false)
    }

    fn initialize_params_with_extensions(extensions: Vec<&str>) -> JsonValue {
        let mut options = HashMap::new();
        options.insert(
            "extensions".to_string(),
            JsonValue::Array(
                extensions
                    .into_iter()
                    .map(|extension| JsonValue::String(extension.to_string()))
                    .collect(),
            ),
        );
        let mut params = HashMap::new();
        params.insert(
            "initializationOptions".to_string(),
            JsonValue::Object(options),
        );
        JsonValue::Object(params)
    }

    fn text_position_params_with_declaration(
        uri: &str,
        line: i64,
        character: i64,
        include_declaration: bool,
    ) -> JsonValue {
        let mut document = HashMap::new();
        document.insert("uri".to_string(), JsonValue::String(uri.to_string()));
        let mut position = HashMap::new();
        position.insert("line".to_string(), JsonValue::number_from_i64(line));
        position.insert(
            "character".to_string(),
            JsonValue::number_from_i64(character),
        );
        let mut params = HashMap::new();
        params.insert("textDocument".to_string(), JsonValue::Object(document));
        params.insert("position".to_string(), JsonValue::Object(position));
        let mut context = HashMap::new();
        context.insert(
            "includeDeclaration".to_string(),
            JsonValue::Bool(include_declaration),
        );
        params.insert("context".to_string(), JsonValue::Object(context));
        JsonValue::Object(params)
    }

    fn text_document_params(uri: &str) -> JsonValue {
        let mut document = HashMap::new();
        document.insert("uri".to_string(), JsonValue::String(uri.to_string()));
        let mut params = HashMap::new();
        params.insert("textDocument".to_string(), JsonValue::Object(document));
        JsonValue::Object(params)
    }

    fn workspace_symbol_params(query: &str) -> JsonValue {
        let mut params = HashMap::new();
        params.insert("query".to_string(), JsonValue::String(query.to_string()));
        JsonValue::Object(params)
    }

    fn rename_params(uri: &str, line: i64, character: i64, new_name: &str) -> JsonValue {
        let mut params = match text_position_params(uri, line, character) {
            JsonValue::Object(params) => params,
            _ => unreachable!(),
        };
        params.insert(
            "newName".to_string(),
            JsonValue::String(new_name.to_string()),
        );
        JsonValue::Object(params)
    }

    fn did_open(uri: &str, version: i64, text: &str) -> String {
        format!(
            r#"{{
                "jsonrpc":"2.0",
                "method":"textDocument/didOpen",
                "params":{{
                    "textDocument":{{
                        "uri":"{uri}",
                        "languageId":"stark",
                        "version":{version},
                        "text":"{}"
                    }}
                }}
            }}"#,
            text.replace('\\', "\\\\")
                .replace('"', "\\\"")
                .replace('\n', "\\n")
        )
    }

    fn request_position(id: i64, method: &str, uri: &str, line: i64, character: i64) -> String {
        format!(
            r#"{{
                "jsonrpc":"2.0",
                "id":{id},
                "method":"{method}",
                "params":{{
                    "textDocument":{{"uri":"{uri}"}},
                    "position":{{"line":{line},"character":{character}}},
                    "context":{{"includeDeclaration":false}}
                }}
            }}"#
        )
    }

    fn request_document(id: i64, method: &str, uri: &str) -> String {
        format!(
            r#"{{
                "jsonrpc":"2.0",
                "id":{id},
                "method":"{method}",
                "params":{{"textDocument":{{"uri":"{uri}"}}}}
            }}"#
        )
    }

    fn request_rename(id: i64, uri: &str, line: i64, character: i64, new_name: &str) -> String {
        format!(
            r#"{{
                "jsonrpc":"2.0",
                "id":{id},
                "method":"textDocument/rename",
                "params":{{
                    "textDocument":{{"uri":"{uri}"}},
                    "position":{{"line":{line},"character":{character}}},
                    "newName":"{new_name}"
                }}
            }}"#
        )
    }

    fn assert_location_contains(
        location: &JsonValue,
        expected_uri: &str,
        start_line: u32,
        start_character: u32,
        end_line: u32,
        end_character: u32,
    ) {
        assert_eq!(
            location.get("uri"),
            Some(&JsonValue::String(expected_uri.to_string()))
        );
        let range = location.get("range").expect("location must include range");
        let start = range.get("start").expect("range must include start");
        let end = range.get("end").expect("range must include end");
        assert_eq!(
            start.get("line"),
            Some(&JsonValue::number_from_i64(i64::from(start_line)))
        );
        assert_eq!(
            start.get("character"),
            Some(&JsonValue::number_from_i64(i64::from(start_character)))
        );
        assert_eq!(
            end.get("line"),
            Some(&JsonValue::number_from_i64(i64::from(end_line)))
        );
        assert_eq!(
            end.get("character"),
            Some(&JsonValue::number_from_i64(i64::from(end_character)))
        );
    }

    fn frame(content: &str) -> String {
        format!("Content-Length: {}\r\n\r\n{content}", content.len())
    }

    fn decode_framed_output(output: &[u8]) -> Vec<String> {
        let text = String::from_utf8(output.to_vec()).unwrap();
        let mut rest = text.as_str();
        let mut messages = Vec::new();
        while !rest.is_empty() {
            let Some(header_end) = rest.find("\r\n\r\n") else {
                panic!("missing header terminator in {rest:?}");
            };
            let header = &rest[..header_end];
            let length = header
                .lines()
                .find_map(|line| line.strip_prefix("Content-Length: "))
                .and_then(|value| value.parse::<usize>().ok())
                .expect("missing content length");
            let body_start = header_end + 4;
            let body_end = body_start + length;
            messages.push(rest[body_start..body_end].to_string());
            rest = &rest[body_end..];
        }
        messages
    }
}
