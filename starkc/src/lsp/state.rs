//! LSP server state management.

use crate::ast::Ast;
use crate::diag::Diagnostic as StarkDiagnostic;
use crate::hir::Hir;
use crate::options::LanguageOptions;
use crate::typecheck::TypeTables;
use std::collections::HashMap;

/// Server state with open documents and compilation cache.
pub struct ServerState {
    pub root_uri: Option<String>,
    pub open_documents: HashMap<String, OpenDocument>,
    pub compilation_cache: HashMap<String, CompilationResult>,
    /// Which language extensions (e.g. `tensor`) every subsequent parse
    /// should enable — set once from `initialize`'s `initializationOptions`
    /// (`{"extensions": ["tensor"]}`, matching the CLI's `--extension`
    /// flag naming) and held for the life of the session, mirroring how
    /// `starkc check`/`stark fmt` take extensions as a fixed input rather
    /// than something that varies per request.
    pub options: LanguageOptions,
}

/// Open document with version tracking.
#[derive(Clone)]
pub struct OpenDocument {
    pub uri: String,
    pub version: i32,
    pub text: String,
}

/// Compilation result cache.
pub struct CompilationResult {
    pub uri: String,
    pub version: i32,
    pub ast: Option<Ast>,
    pub hir: Option<Hir>,
    pub diagnostics: Vec<StarkDiagnostic>,
    pub type_tables: Option<TypeTables>,
    pub last_compiled_at: std::time::SystemTime,
}

impl ServerState {
    /// Create a new server state.
    pub fn new() -> Self {
        Self {
            root_uri: None,
            open_documents: HashMap::new(),
            compilation_cache: HashMap::new(),
            options: LanguageOptions::CORE,
        }
    }

    /// Set the root URI.
    pub fn set_root_uri(&mut self, root_uri: String) {
        self.root_uri = Some(root_uri);
    }

    /// Open a document.
    pub fn open_document(&mut self, uri: String, version: i32, text: String) {
        self.open_documents.insert(
            uri.clone(),
            OpenDocument {
                uri: uri.clone(),
                version,
                text,
            },
        );
        // Invalidate cache for this document
        self.compilation_cache.remove(&uri);
    }

    /// Update a document.
    pub fn update_document(&mut self, uri: String, version: i32, text: String) {
        if let Some(doc) = self.open_documents.get_mut(&uri) {
            doc.version = version;
            doc.text = text;
            // Invalidate cache for this document
            self.compilation_cache.remove(&uri);
        }
    }

    /// Close a document.
    pub fn close_document(&mut self, uri: &str) {
        self.open_documents.remove(uri);
        self.compilation_cache.remove(uri);
    }

    /// Get an open document.
    pub fn get_document(&self, uri: &str) -> Option<&OpenDocument> {
        self.open_documents.get(uri)
    }

    /// Cache a compilation result.
    pub fn cache_compilation_result(&mut self, result: CompilationResult) {
        self.compilation_cache.insert(result.uri.clone(), result);
    }

    /// Get a cached compilation result if still valid.
    pub fn get_cached_result(&self, uri: &str, version: i32) -> Option<&CompilationResult> {
        let result = self.compilation_cache.get(uri)?;
        if result.version == version {
            Some(result)
        } else {
            None
        }
    }

    /// Clear all state.
    pub fn clear(&mut self) {
        self.root_uri = None;
        self.open_documents.clear();
        self.compilation_cache.clear();
    }
}

impl Default for ServerState {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_open_document() {
        let mut state = ServerState::new();
        state.open_document(
            "file:///test.stark".to_string(),
            1,
            "fn main() {}".to_string(),
        );

        let doc = state.get_document("file:///test.stark").unwrap();
        assert_eq!(doc.version, 1);
        assert_eq!(doc.text, "fn main() {}");
    }

    #[test]
    fn test_update_document() {
        let mut state = ServerState::new();
        state.open_document(
            "file:///test.stark".to_string(),
            1,
            "fn main() {}".to_string(),
        );
        state.update_document(
            "file:///test.stark".to_string(),
            2,
            "fn test() {}".to_string(),
        );

        let doc = state.get_document("file:///test.stark").unwrap();
        assert_eq!(doc.version, 2);
        assert_eq!(doc.text, "fn test() {}");
    }

    #[test]
    fn test_close_document() {
        let mut state = ServerState::new();
        state.open_document(
            "file:///test.stark".to_string(),
            1,
            "fn main() {}".to_string(),
        );
        state.close_document("file:///test.stark");

        assert!(state.get_document("file:///test.stark").is_none());
    }
}
