//! AS1b-i acceptance — `SourceId` is allocated once, at load, in one place.
//!
//! The split of AS1b was chosen so this half could be *mechanically* checked rather than argued.
//! `WP-SPAN-SOURCEID.md` §6 warns the second half is easy to make compile while threading a
//! plausible-but-wrong id; the defence against that is that there is exactly one thing entitled to
//! mint an id, and it does so when a file is loaded.
//!
//! What this half does **not** do: `Span` still carries no `SourceId`. That is the second half.

use starkc::analysis::{analyze_project, ProjectInput};
use starkc::options::LanguageOptions;
use starkc::package::PackageGraph;
use starkc::source::{SourceFile, SourceRegistry};
use std::path::Path;
use std::sync::Arc;

#[test]
fn interning_is_idempotent_by_logical_name() {
    let mut registry = SourceRegistry::default();
    let first = registry.intern(Arc::new(SourceFile::new(
        "pkg/src/main.stark",
        "fn main() {}",
    )));
    // A second Arc with the same logical name is the SAME source. Returning a new id here is how a
    // file acquires two identities — the exact defect AS1a had to repair in `build_source_map`.
    let again = registry.intern(Arc::new(SourceFile::new(
        "pkg/src/main.stark",
        "fn main() {}",
    )));
    assert_eq!(first.id(), again.id(), "one logical name, one id");
    assert_eq!(registry.len(), 1);

    let other = registry.intern(Arc::new(SourceFile::new("pkg/src/helper.stark", "")));
    assert_ne!(first.id(), other.id());
    assert_eq!(registry.len(), 2);
}

#[test]
fn the_first_registration_wins_and_an_id_never_changes_meaning() {
    let mut registry = SourceRegistry::default();
    let id = registry
        .intern(Arc::new(SourceFile::new("a.stark", "original")))
        .id();
    // Re-interning the same name with DIFFERENT bytes must not repoint the id. The parser does
    // exactly this: an interpolation sub-parse builds a decoded scratch buffer carrying the
    // enclosing file's name.
    let again = registry
        .intern(Arc::new(SourceFile::new("a.stark", "decoded scratch")))
        .id();
    assert_eq!(id, again);
    assert_eq!(
        registry.get(id).expect("registered").src,
        "original",
        "an id keeps denoting the source it was minted for"
    );
}

#[test]
fn ids_are_dense_and_in_load_order() {
    let mut registry = SourceRegistry::default();
    for i in 0..5 {
        let id = registry
            .intern(Arc::new(SourceFile::new(format!("f{i}.stark"), "")))
            .id();
        assert_eq!(id.as_u32(), i, "ids are dense and follow load order");
    }
    let seen: Vec<u32> = registry.iter().map(|s| s.id().as_u32()).collect();
    assert_eq!(seen, vec![0, 1, 2, 3, 4], "iteration is in id order");
}

/// The mechanical part of the acceptance: nothing outside `source.rs` constructs a `SourceId`.
///
/// `SourceId`'s field is private and `from_u32` is `pub(crate)` and single-purpose, so the compiler
/// enforces most of this. This catches the remaining case — a second allocator added *inside* the
/// crate, which is how `build_source_map` came to be one in the first place.
#[test]
fn only_the_registry_mints_source_ids() {
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut offenders = Vec::new();
    let mut scanned = 0usize;

    fn walk(dir: &Path, out: &mut Vec<std::path::PathBuf>) {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                walk(&path, out);
            } else if path.extension().is_some_and(|e| e == "rs") {
                out.push(path);
            }
        }
    }
    let mut files = Vec::new();
    walk(&src, &mut files);
    files.sort();
    assert!(files.len() > 20, "source scan found only {}", files.len());

    for file in &files {
        let relative = file
            .strip_prefix(src.parent().unwrap())
            .unwrap()
            .to_string_lossy()
            .replace('\\', "/");
        let text = std::fs::read_to_string(file).unwrap().replace("\r\n", "\n");
        // Everything before a top-level `#[cfg(test)]`; tests may construct ids freely.
        let production = match text.find("\n#[cfg(test)]") {
            Some(at) => &text[..at + 1],
            None => &text[..],
        };
        scanned += 1;
        if relative == "src/source.rs" {
            continue;
        }
        for (number, line) in production.lines().enumerate() {
            if line.contains("SourceId(") && !line.contains("pub struct SourceId") {
                offenders.push(format!("{relative}:{}: {}", number + 1, line.trim()));
            }
        }
    }

    assert!(scanned > 20, "only {scanned} files scanned");
    assert!(
        offenders.is_empty(),
        "these construct a SourceId outside the registry:\n  {}\n\n\
         Ids come from `SourceRegistry::intern`, at load. A second allocator is how identity came \
         to be assigned after the front end had already run.",
        offenders.join("\n  ")
    );
}

/// End to end: a real package's source map draws its ids from the registry the parser filled, and
/// every file in it round-trips by id and by name.
#[test]
fn a_package_source_map_is_a_view_over_the_registry() {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .subsec_nanos();
    let root = std::env::temp_dir().join(format!("as1bi_{}_{nanos}", std::process::id()));
    let src = root.join("src");
    std::fs::create_dir_all(&src).unwrap();
    std::fs::write(
        root.join("starkpkg.json"),
        r#"{"name":"probe","version":"0.1.0","entry":"src/main.stark"}"#,
    )
    .unwrap();
    std::fs::write(
        src.join("main.stark"),
        "mod helper;\n\nfn main() {\n    let v: Int32 = helper::seven();\n}\n",
    )
    .unwrap();
    std::fs::write(
        src.join("helper.stark"),
        "pub fn seven() -> Int32 {\n    7\n}\n",
    )
    .unwrap();

    let graph =
        PackageGraph::load_from_root_with_modes(&root.join("starkpkg.json"), false, true).unwrap();
    let analysis = analyze_project(ProjectInput::package(graph), LanguageOptions::CORE);
    assert!(!analysis.has_errors(), "fixture must analyse cleanly");

    // Every source-map record is reachable by its own id and by its name, and the two agree.
    for record in analysis.source_map.files() {
        let by_id = analysis
            .source_map
            .get(record.id)
            .expect("record reachable by its own id");
        assert_eq!(by_id.file.name, record.file.name);
        assert_eq!(
            analysis.source_map.id_for_name(&record.file.name),
            Some(record.id),
            "name lookup agrees with the id"
        );
    }

    // AS1a's invariant still holds through the registry: one physical file, one record.
    let mut names: Vec<&str> = analysis
        .source_map
        .files()
        .iter()
        .map(|r| r.file.name.as_str())
        .collect();
    names.sort();
    assert_eq!(
        names,
        vec!["probe/src/helper.stark", "probe/src/main.stark"]
    );

    let _ = std::fs::remove_dir_all(&root);
}

// ---------------------------------------------------------------------------------------------
// AS1b-ii-b — item source identity, and one authority
// ---------------------------------------------------------------------------------------------

/// `item_sources` names ids; the registry answers. Neither may disagree with the other, and there
/// must be no id in the map that the registry cannot resolve.
///
/// This replaced `item_files: ItemId -> Arc<SourceFile>`, which was a rival source *authority*:
/// consumers pulled a file object out of it and indexed spans against that, rather than against
/// whatever the registry said. An id cannot be a rival authority — it can only name.
#[test]
fn every_item_source_resolves_in_the_registry() {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .subsec_nanos();
    let root = std::env::temp_dir().join(format!("as1bii_{}_{nanos}", std::process::id()));
    let src = root.join("src");
    std::fs::create_dir_all(&src).unwrap();
    std::fs::write(
        root.join("starkpkg.json"),
        r#"{"name":"probe","version":"0.1.0","entry":"src/main.stark"}"#,
    )
    .unwrap();
    std::fs::write(
        src.join("main.stark"),
        "mod helper;\n\nfn main() {\n    let v: Int32 = helper::seven();\n}\n",
    )
    .unwrap();
    std::fs::write(
        src.join("helper.stark"),
        "pub fn seven() -> Int32 {\n    7\n}\n",
    )
    .unwrap();

    let graph =
        PackageGraph::load_from_root_with_modes(&root.join("starkpkg.json"), false, true).unwrap();
    let analysis = analyze_project(ProjectInput::package(graph), LanguageOptions::CORE);
    assert!(!analysis.has_errors(), "fixture must analyse cleanly");
    let hir = analysis.hir.as_ref().expect("hir");

    assert!(
        !hir.item_sources.is_empty(),
        "an empty item-source map would make this check vacuous"
    );
    for (item, id) in &hir.item_sources {
        assert!(
            hir.sources.get(*id).is_some(),
            "item {item:?} names source {id:?}, which the registry cannot resolve"
        );
        // And the accessor agrees with the raw map.
        assert_eq!(
            hir.item_file(*item).map(|f| f.name.as_str()),
            hir.sources.get(*id).map(|f| f.name.as_str()),
        );
    }

    // Items really do span both files — otherwise this proves nothing about cross-file identity.
    let mut names: Vec<&str> = hir
        .item_sources
        .values()
        .filter_map(|id| hir.sources.get(*id))
        .map(|f| f.name.as_str())
        .collect();
    names.sort();
    names.dedup();
    assert_eq!(
        names,
        vec!["probe/src/helper.stark", "probe/src/main.stark"]
    );

    let _ = std::fs::remove_dir_all(&root);
}
