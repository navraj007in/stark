//! LSP server state management.

use crate::analysis::ProjectAnalysis;
use crate::options::LanguageOptions;
use std::collections::HashMap;
use std::path::PathBuf;

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
    pub analysis: ProjectAnalysis,
    pub last_compiled_at: std::time::SystemTime,
    /// DEV-213. The package this URI's analysis covers, when it covers one.
    ///
    /// `analysis` is a WHOLE-PACKAGE `ProjectAnalysis`, not an analysis of `uri` alone, and the
    /// cache holds one per open URI. Two open files of one package therefore hold two analyses
    /// that each describe BOTH files, so invalidating only the edited URI leaves the other
    /// carrying a pre-edit view of the file that was just edited. Recording the package here is
    /// what lets `invalidate_package_of` find the siblings.
    ///
    /// `None` for a single-file analysis (no manifest above it). A `None` never matches another
    /// `None`: two unrelated loose files share no package and must not invalidate each other.
    pub package_root: Option<PathBuf>,
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
        // DEV-213: opening adds an overlay, which changes the input to every analysis of this
        // package -- not only this URI's.
        self.invalidate_package_of(&uri);
    }

    /// Update a document.
    pub fn update_document(&mut self, uri: String, version: i32, text: String) {
        if let Some(doc) = self.open_documents.get_mut(&uri) {
            doc.version = version;
            doc.text = text;
            // DEV-213: this is the `didChange` path, and the one the defect was demonstrated on.
            self.invalidate_package_of(&uri);
        }
    }

    /// Close a document.
    pub fn close_document(&mut self, uri: &str) {
        self.open_documents.remove(uri);
        // DEV-213: closing REMOVES an overlay, which changes the package's analysis input just as
        // opening one does.
        self.invalidate_package_of(uri);
    }

    /// DEV-213 -- drop `uri`'s cached analysis, and every sibling analysis of the same package.
    ///
    /// **Why a whole-package sweep rather than one entry.** Each cache value owns a whole-package
    /// `ProjectAnalysis`. `handle_workspace_symbol` merges symbols across every cached value, so a
    /// sibling that was never invalidated contributes its pre-edit view of the edited file and the
    /// response contains a name that no longer exists. Removing only `uri` is what made
    /// `as8_editing_one_file_leaves_other_uris_cached_analyses_stale` pass at HEAD.
    ///
    /// The package is read from the CACHE rather than from the filesystem: the entry being
    /// invalidated already recorded which package it analysed, so this stays a pure in-memory
    /// operation on a hot editor path. A URI with no cached entry has no siblings to find, which
    /// is the correct answer rather than a missed case -- nothing stale can exist for a package
    /// this server has not analysed.
    fn invalidate_package_of(&mut self, uri: &str) {
        let package = self
            .compilation_cache
            .get(uri)
            .and_then(|cached| cached.package_root.clone());
        self.compilation_cache.remove(uri);
        if let Some(package) = package {
            // `Some(p) == Some(p)` only. Single-file analyses carry `None` and are never swept in
            // as siblings of one another.
            self.compilation_cache
                .retain(|_, cached| cached.package_root.as_ref() != Some(&package));
        }
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
        self.options = LanguageOptions::CORE;
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
