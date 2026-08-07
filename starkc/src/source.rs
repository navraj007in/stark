//! Source files, byte spans, and position mapping.
//!
//! Spans are half-open byte ranges `[lo, hi)` into a single source file.
//! Line and column numbers are 1-based, matching the diagnostic format in
//! `04-Semantic-Analysis.md`. Columns count bytes within the line, which is
//! sufficient until the diagnostics renderer grows Unicode-width awareness.

/// A half-open byte range `[lo, hi)` **within a named source** (AS1b-ii).
///
/// The source is part of the span because a byte range without it is meaningless, and
/// `SourceFile::line_col` clamps rather than fails — so a span measured against the wrong file
/// yields a well-formed, plausible, wrong location. DEV-122 is that defect, twice.
///
/// There is deliberately **no** two-argument constructor. Every construction names a source, so a
/// site cannot acquire one by whatever happened to be in scope. `WP-SPAN-SOURCEID.md` §6 names that
/// as the failure mode this change is most likely to introduce.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub struct Span {
    pub source: SourceId,
    pub lo: u32,
    pub hi: u32,
}

impl Span {
    pub fn in_source(source: SourceId, lo: u32, hi: u32) -> Self {
        debug_assert!(lo <= hi, "span lo {lo} > hi {hi}");
        Span { source, lo, hi }
    }

    /// A zero-width span at a single offset in `source`.
    pub fn point_in(source: SourceId, at: u32) -> Self {
        Span {
            source,
            lo: at,
            hi: at,
        }
    }

    /// A span for something with **no meaningful location** in `source` — a synthesised item, an
    /// interpreter invariant failure, a lowering rejection with nothing to point at.
    ///
    /// It resolves to the start of `source`, which is exactly what the `Span { lo: 0, hi: 0 }`
    /// sentinel this replaces already rendered as. **The name is the point:** these sites are
    /// asserting "no location", and the renderer still shows one. Making absence representable means
    /// `Option<Span>` through `Diagnostic` and `RuntimeError`, which is a separate change — see
    /// `AS1B-OPENING-ANALYSIS.md`. Until then this at least says so out loud instead of looking
    /// like a real position.
    pub fn synthetic(source: SourceId) -> Self {
        Span {
            source,
            lo: 0,
            hi: 0,
        }
    }

    /// The smallest span covering both `self` and `other`.
    ///
    /// Both must belong to the same source: a range spanning two files denotes nothing.
    ///
    /// **The check is unconditional, deliberately.** It was a `debug_assert_eq!` with a comment
    /// saying "in release the left source wins" — which meant a release compiler would silently
    /// produce `source A, 100..130` from `A 100..110` joined with `B 120..130`. That is a
    /// plausible, well-formed, wrong location: DEV-122's exact failure class, moved from rendering
    /// to span *composition*, and reachable only in the builds users run. Joining spans across
    /// files is an internal compiler defect with no meaningful recovery, so it fails loudly
    /// everywhere.
    pub fn to(self, other: Span) -> Span {
        assert_eq!(
            self.source, other.source,
            "cannot join spans from different sources"
        );
        Span {
            source: self.source,
            lo: self.lo.min(other.lo),
            hi: self.hi.max(other.hi),
        }
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
}

/// A source together with the identity a **particular compilation** gave it.
///
/// Deliberately not a field on [`SourceFile`]. A `SourceFile` is just bytes with a name and can be
/// reused across sessions; a `SourceId` is registry-local and means nothing outside the compilation
/// that minted it. Storing one inside the other would make a reusable value carry a
/// session-scoped fact.
///
/// **Only [`SourceRegistry::intern`] can build one.** The fields are private and there is no public
/// constructor, so holding a `RegisteredSource` is proof the source was **registered rather than
/// fabricated** — which is what `WP-SPAN-SOURCEID.md` §6 warns about.
///
/// It is *not* proof that the registering registry is the same one a given `Hir` carries. `SourceId`
/// is registry-local, but Rust's type system does not encode registry identity here, and
/// [`Span::in_source`] accepts a raw `SourceId`. Encoding that would need generative lifetimes and
/// is disproportionate to the risk; agreement between a program and its registry is held by
/// behavioural tests instead (`as1b_source_registry`).
///
/// Derefs to the file, so existing `.name`, `.src` and `.line_col()` uses read unchanged.
#[derive(Clone, Debug)]
pub struct RegisteredSource {
    id: SourceId,
    file: std::sync::Arc<SourceFile>,
}

impl RegisteredSource {
    pub fn id(&self) -> SourceId {
        self.id
    }

    pub fn file(&self) -> &std::sync::Arc<SourceFile> {
        &self.file
    }

    /// A span in *this* source. The only way to mint one without naming an id by hand.
    pub fn span(&self, lo: u32, hi: u32) -> Span {
        Span::in_source(self.id, lo, hi)
    }

    /// A span for something with no meaningful position in this source.
    pub fn synthetic_span(&self) -> Span {
        Span::synthetic(self.id)
    }
}

/// Two handles are equal when they name the same source. The id is the identity; the `Arc` is
/// what it resolves to, and within one compilation the registry guarantees those agree.
impl PartialEq for RegisteredSource {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for RegisteredSource {}

impl std::ops::Deref for RegisteredSource {
    type Target = SourceFile;

    fn deref(&self) -> &SourceFile {
        &self.file
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
    files: Vec<RegisteredSource>,
    by_name: std::collections::HashMap<String, SourceId>,
}

impl SourceRegistry {
    /// Register `file`, or return the id it already has. Idempotent by logical name.
    ///
    /// The first registration wins: a later `Arc` with the same name is dropped rather than
    /// replacing the first, so an id never silently changes what it denotes mid-compilation.
    pub fn intern(&mut self, file: std::sync::Arc<SourceFile>) -> RegisteredSource {
        if let Some(&id) = self.by_name.get(&file.name) {
            return self.files[id.0 as usize].clone();
        }
        let id = SourceId(self.files.len() as u32);
        self.by_name.insert(file.name.clone(), id);
        // The only construction of a `RegisteredSource` in the crate.
        self.files.push(RegisteredSource { id, file });
        self.files[id.0 as usize].clone()
    }

    pub fn get(&self, id: SourceId) -> Option<&RegisteredSource> {
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
    pub fn iter(&self) -> impl Iterator<Item = &RegisteredSource> {
        self.files.iter()
    }
}

/// A registered source for the crate's own unit tests.
///
/// Tests need a real `SourceId`, and the rule is that ids come from a registry — so this builds
/// one rather than fabricating a value. The registry is dropped; the handle keeps the `Arc` and
/// the id, which is all a test needs.
#[cfg(test)]
pub(crate) fn registered_for_test(name: &str, src: &str) -> RegisteredSource {
    let mut registry = SourceRegistry::default();
    registry.intern(std::sync::Arc::new(SourceFile::new(name, src)))
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
        // AS1b-ii: a span names a source, so even a unit test registers one rather than
        // conjuring an id.
        let mut registry = SourceRegistry::default();
        let s = registry
            .intern(std::sync::Arc::new(SourceFile::new("t.stark", "")))
            .id();
        let a = Span::in_source(s, 4, 7);
        let b = Span::in_source(s, 10, 12);
        assert_eq!(a.to(b), Span::in_source(s, 4, 12));
        assert_eq!(b.to(a), Span::in_source(s, 4, 12));
    }

    /// Joining across sources must fail in EVERY build, not only in debug.
    ///
    /// With a `debug_assert`, a release compiler produced `A 100..130` from `A 100..110` joined
    /// with `B 120..130` — a well-formed, plausible, wrong location, which is DEV-122's failure
    /// class at composition time. `cargo test` runs in debug, so this test cannot observe the
    /// release build directly; what it pins is that the check is a real `assert` that panics,
    /// rather than an equality the function is free to paper over.
    #[test]
    #[should_panic(expected = "cannot join spans from different sources")]
    fn joining_spans_from_different_sources_panics() {
        let mut registry = SourceRegistry::default();
        let a = registry
            .intern(std::sync::Arc::new(SourceFile::new("a.stark", "")))
            .id();
        let b = registry
            .intern(std::sync::Arc::new(SourceFile::new("b.stark", "")))
            .id();
        assert_ne!(a, b, "the two sources must actually differ");
        let _ = Span::in_source(a, 100, 110).to(Span::in_source(b, 120, 130));
    }
}
