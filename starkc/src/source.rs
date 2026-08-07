//! Source files, byte spans, and position mapping.
//!
//! Spans are half-open byte ranges `[lo, hi)` into a single source file.
//! Line and column numbers are 1-based, matching the diagnostic format in
//! `04-Semantic-Analysis.md`. Columns count bytes within the line, which is
//! sufficient until the diagnostics renderer grows Unicode-width awareness.

/// A half-open byte range `[lo, hi)` within one source file.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub struct Span {
    pub lo: u32,
    pub hi: u32,
}

impl Span {
    pub fn new(lo: u32, hi: u32) -> Self {
        debug_assert!(lo <= hi, "span lo {lo} > hi {hi}");
        Span { lo, hi }
    }

    /// A zero-width span at a single offset.
    pub fn point(at: u32) -> Self {
        Span { lo: at, hi: at }
    }

    /// The smallest span covering both `self` and `other`.
    pub fn to(self, other: Span) -> Span {
        Span::new(self.lo.min(other.lo), self.hi.max(other.hi))
    }
}

/// Stable identity for one source within a compilation.
///
/// **AS1b-i moved this here from `analysis`, and moved its allocation earlier.** It lived beside
/// `SourceMap`, which assigned ids *after* parse, resolve and typecheck had already run — so the
/// identity a span needs did not exist at the moment spans are created. `WP-SPAN-SOURCEID.md`
/// describes `SourceId` as groundwork already in place; that was true of the type and not of when
/// it was handed out.
///
/// It belongs in `source` because `Span` is here and will carry one.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SourceId(u32);

impl SourceId {
    pub fn as_u32(self) -> u32 {
        self.0
    }

    /// Only for the one case a registry cannot cover: a root whose parse failed before anything
    /// interned it. Not a general constructor — ids come from [`SourceRegistry::intern`].
    pub(crate) fn from_u32(raw: u32) -> Self {
        SourceId(raw)
    }
}

/// The one allocator of [`SourceId`]. Files are interned as they are loaded, in load order.
///
/// Identity is the file's LOGICAL NAME, which after AS1a is `<package>/<path>` for package sources
/// and the path for a single-file compile — one physical file, one name, one id. Interning the same
/// name twice returns the first id rather than making a second identity, which is the invariant
/// AS1a had to repair by hand in `build_source_map`.
///
/// `SourceMap` is now a *view* over this: it adds provenance and answers lookups, and no longer
/// decides what an id is.
#[derive(Default, Debug, Clone)]
pub struct SourceRegistry {
    files: Vec<std::sync::Arc<SourceFile>>,
    by_name: std::collections::HashMap<String, SourceId>,
}

impl SourceRegistry {
    /// Register `file`, or return the id it already has. Idempotent by logical name.
    ///
    /// The first registration wins: a later `Arc` with the same name is dropped rather than
    /// replacing the first, so an id never silently changes what it denotes mid-compilation.
    pub fn intern(&mut self, file: std::sync::Arc<SourceFile>) -> SourceId {
        if let Some(&id) = self.by_name.get(&file.name) {
            return id;
        }
        let id = SourceId(self.files.len() as u32);
        self.by_name.insert(file.name.clone(), id);
        self.files.push(file);
        id
    }

    pub fn get(&self, id: SourceId) -> Option<&std::sync::Arc<SourceFile>> {
        self.files.get(id.0 as usize)
    }

    pub fn id_for_name(&self, name: &str) -> Option<SourceId> {
        self.by_name.get(name).copied()
    }

    pub fn len(&self) -> usize {
        self.files.len()
    }

    pub fn is_empty(&self) -> bool {
        self.files.is_empty()
    }

    /// Every registered source, in id order.
    pub fn iter(&self) -> impl Iterator<Item = (SourceId, &std::sync::Arc<SourceFile>)> {
        self.files
            .iter()
            .enumerate()
            .map(|(index, file)| (SourceId(index as u32), file))
    }
}

/// A loaded source file with precomputed line starts for position mapping.
#[derive(Debug)]
pub struct SourceFile {
    /// The file's LOGICAL name — what every diagnostic, trap and evidence record shows.
    ///
    /// For a package build this is `<package>/<path within the package>`, never an absolute path
    /// (DEV-113). PKG-IDENTITY-001 requires a package token to be "never an absolute checkout path",
    /// and §15.2 requires trap source names to survive relocation: the same workspace compiled in two
    /// directories must observe identically, which it cannot if the checkout path is baked into
    /// provenance. For a single-file compile the name is whatever the caller passed, which is
    /// usually the path — that path is not identity-bearing there because there is no package.
    pub name: String,
    /// Where the file actually is, when that is known. Used to resolve `mod` declarations and to
    /// point a human at a file; **never** used in identity, provenance or comparison.
    pub disk_path: Option<std::path::PathBuf>,
    pub src: String,
    /// Byte offset of the first character of each line. Always starts with 0.
    line_starts: Vec<u32>,
}

impl SourceFile {
    /// Attaches the on-disk location. Separate from `new` so that all 60-plus existing call sites,
    /// which have no package context, keep compiling unchanged.
    pub fn with_disk_path(mut self, path: impl Into<std::path::PathBuf>) -> Self {
        self.disk_path = Some(path.into());
        self
    }

    /// The directory to resolve `mod` declarations against: the real one when known, else the
    /// directory of the name, which is what a single-file compile has.
    pub fn resolution_dir(&self) -> std::path::PathBuf {
        let basis = self
            .disk_path
            .clone()
            .unwrap_or_else(|| std::path::PathBuf::from(&self.name));
        basis
            .parent()
            .map(|p| p.to_path_buf())
            .unwrap_or_else(|| std::path::PathBuf::from(""))
    }

    pub fn new(name: impl Into<String>, src: impl Into<String>) -> Self {
        let src = src.into();
        assert!(
            u32::try_from(src.len()).is_ok(),
            "source file larger than 4 GiB"
        );
        let mut line_starts = vec![0u32];
        for (i, b) in src.bytes().enumerate() {
            if b == b'\n' {
                line_starts.push(i as u32 + 1);
            }
        }
        SourceFile {
            name: name.into(),
            disk_path: None,
            src,
            line_starts,
        }
    }

    pub fn line_count(&self) -> usize {
        self.line_starts.len()
    }

    /// Map a byte offset to a 1-based (line, column) pair.
    ///
    /// Offsets past the end of the file map to the position just past the
    /// last character, so error spans at EOF render sensibly.
    pub fn line_col(&self, offset: u32) -> (usize, usize) {
        let offset = offset.min(self.src.len() as u32);
        let line = match self.line_starts.binary_search(&offset) {
            Ok(exact) => exact,
            Err(insert) => insert - 1,
        };
        let col = offset - self.line_starts[line];
        (line + 1, col as usize + 1)
    }

    /// The text of a 1-based line, without its trailing newline.
    pub fn line_text(&self, line: usize) -> &str {
        assert!(line >= 1 && line <= self.line_count(), "line out of range");
        let start = self.line_starts[line - 1] as usize;
        let end = self
            .line_starts
            .get(line)
            .map_or(self.src.len(), |&s| s as usize);
        self.src[start..end].trim_end_matches(['\n', '\r'])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn line_col_basics() {
        let f = SourceFile::new("t.stark", "let x = 1;\nlet y = 2;\n");
        assert_eq!(f.line_count(), 3); // two lines + empty final line
        assert_eq!(f.line_col(0), (1, 1));
        assert_eq!(f.line_col(4), (1, 5)); // 'x'
        assert_eq!(f.line_col(10), (1, 11)); // the '\n' belongs to line 1
        assert_eq!(f.line_col(11), (2, 1)); // 'l' of second let
        assert_eq!(f.line_col(15), (2, 5)); // 'y'
    }

    #[test]
    fn line_col_at_and_past_eof() {
        let f = SourceFile::new("t.stark", "ab");
        assert_eq!(f.line_col(2), (1, 3));
        assert_eq!(f.line_col(999), (1, 3));
    }

    #[test]
    fn empty_file() {
        let f = SourceFile::new("t.stark", "");
        assert_eq!(f.line_count(), 1);
        assert_eq!(f.line_col(0), (1, 1));
        assert_eq!(f.line_text(1), "");
    }

    #[test]
    fn line_text_strips_newline_and_cr() {
        let f = SourceFile::new("t.stark", "one\r\ntwo\nthree");
        assert_eq!(f.line_text(1), "one");
        assert_eq!(f.line_text(2), "two");
        assert_eq!(f.line_text(3), "three");
    }

    #[test]
    fn span_join() {
        let a = Span::new(4, 7);
        let b = Span::new(10, 12);
        assert_eq!(a.to(b), Span::new(4, 12));
        assert_eq!(b.to(a), Span::new(4, 12));
    }
}
