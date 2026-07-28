//! WP-C7.3 — the bounded build cache.
//!
//! **What this is, precisely.** It reuses complete content-addressed generated crates and their
//! Cargo artefacts. It is a bounded build cache, **not fine-grained incremental compilation**:
//! nothing here understands functions, packages or interfaces, and a one-character source edit
//! produces a different key and a cold build. Saying otherwise would promise something the
//! mechanism cannot do.
//!
//! **Why this boundary.** The C7.3 decision gate (CD-188) measured that host Cargo/rustc is 65–68%
//! of a cold build, so a perfect front-end cache would cap at the ~33% that is STARK's own share.
//! Preserving the generated crate delivers 2.1× with no new invalidation logic at all — because the
//! directory was already content-addressed and Cargo already does the invalidation correctly. The
//! cache was, in effect, being deleted immediately after being built.
//!
//! **So the work here is eviction, not caching.** Five successive edits left five crate directories
//! and 34 MB with nothing removing them; that leak is exactly what the deletion was protecting
//! against. What follows is the smallest mechanism that makes retention safe: per-entry metadata,
//! an LRU sweep under a size cap, atomic writes, and a lock so two concurrent builds cannot evict
//! each other's work.

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// The default cap. Applied per cache root — that is, per package's `target/stark/<profile>/` —
/// rather than across every package on the machine, because the cache root is where the entries
/// already live and a machine-wide cache would be a relocation rather than the smallest bounded
/// mechanism. Recorded as an interpretation, not as an equivalence.
pub const DEFAULT_MAX_BYTES: u64 = 2 * 1024 * 1024 * 1024;

/// Secondary hygiene: an entry untouched for this long is evictable even when the cap is not
/// reached. Keeps a rarely-built package from holding stale entries indefinitely.
pub const DEFAULT_MAX_AGE: Duration = Duration::from_secs(30 * 24 * 60 * 60);

/// Written into each entry. Deliberately a tiny hand-rolled format rather than a serialisation
/// dependency: it holds two integers, and a corrupt or absent file must degrade to "evict me"
/// rather than fail a build.
const METADATA_FILE: &str = ".stark-cache-entry";

/// Marks an entry the user asked to keep (`--keep-generated` / `--emit-rust`). Such an entry is
/// still counted toward the cap but is **never evicted**: the user asked for it explicitly, and
/// silently removing something they requested is worse than exceeding a soft cap.
const PINNED_FILE: &str = ".stark-cache-pinned";

#[derive(Clone, Debug)]
pub struct CacheEntry {
    pub path: PathBuf,
    pub last_used: SystemTime,
    pub bytes: u64,
    pub pinned: bool,
}

#[derive(Clone, Debug, Default)]
pub struct CacheStatus {
    pub root: PathBuf,
    pub entries: Vec<CacheEntry>,
}

impl CacheStatus {
    pub fn total_bytes(&self) -> u64 {
        self.entries.iter().map(|e| e.bytes).sum()
    }
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn directory_size(path: &Path) -> u64 {
    let mut total = 0;
    let Ok(entries) = fs::read_dir(path) else {
        return 0;
    };
    for entry in entries.flatten() {
        let Ok(kind) = entry.file_type() else {
            continue;
        };
        if kind.is_dir() {
            total += directory_size(&entry.path());
        } else if let Ok(meta) = entry.metadata() {
            total += meta.len();
        }
    }
    total
}

/// Write a file by writing a temporary beside it and renaming. A build interrupted mid-write must
/// leave either the old metadata or the new one, never a truncated file that a later read would
/// have to guess about.
fn write_atomic(path: &Path, contents: &str) -> std::io::Result<()> {
    let temporary = path.with_extension(format!("tmp{}", std::process::id()));
    {
        let mut file = fs::File::create(&temporary)?;
        file.write_all(contents.as_bytes())?;
        file.sync_all()?;
    }
    fs::rename(&temporary, path)
}

/// Record that `entry` was used now, and how large it is.
pub fn touch(entry_dir: &Path, pinned: bool) {
    let bytes = directory_size(entry_dir);
    let _ = write_atomic(
        &entry_dir.join(METADATA_FILE),
        &format!("last_used={}\nbytes={}\n", now_secs(), bytes),
    );
    let pin = entry_dir.join(PINNED_FILE);
    if pinned {
        let _ = write_atomic(&pin, "requested by --keep-generated/--emit-rust\n");
    }
}

fn read_entry(path: &Path) -> Option<CacheEntry> {
    if !path.is_dir() {
        return None;
    }
    let mut last_used = 0u64;
    let mut bytes = None;
    if let Ok(text) = fs::read_to_string(path.join(METADATA_FILE)) {
        for line in text.lines() {
            match line.split_once('=') {
                Some(("last_used", v)) => last_used = v.trim().parse().unwrap_or(0),
                Some(("bytes", v)) => bytes = v.trim().parse().ok(),
                _ => {}
            }
        }
    }
    // A missing or unreadable metadata file is not an error: the entry is simply treated as
    // oldest and measured directly, so a corrupt entry becomes the FIRST candidate for eviction
    // rather than something that breaks the sweep.
    Some(CacheEntry {
        bytes: bytes.unwrap_or_else(|| directory_size(path)),
        last_used: UNIX_EPOCH + Duration::from_secs(last_used),
        pinned: path.join(PINNED_FILE).exists(),
        path: path.to_path_buf(),
    })
}

/// Every entry under one cache root, newest first.
pub fn status(root: &Path) -> CacheStatus {
    let mut entries: Vec<CacheEntry> = fs::read_dir(root)
        .into_iter()
        .flatten()
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.is_dir())
        .filter_map(|p| read_entry(&p))
        .collect();
    // Newest first: `status` is a human-facing listing, and the entry a user is most likely asking
    // about is the one they just built.
    entries.sort_by_key(|e| std::cmp::Reverse(e.last_used));
    CacheStatus {
        root: root.to_path_buf(),
        entries,
    }
}

/// A whole-root lock, so two concurrent builds cannot sweep each other's entries.
///
/// Deliberately advisory and best-effort: it is a `create_new` sentinel with a staleness timeout,
/// not an OS lock. Failing to acquire it SKIPS the sweep rather than failing the build — a build
/// must never fail because a cache could not be tidied.
struct SweepLock {
    path: PathBuf,
    held: bool,
}

impl SweepLock {
    fn acquire(root: &Path) -> Self {
        let path = root.join(".stark-cache-lock");
        // A lock older than five minutes is assumed abandoned by an interrupted build.
        if let Ok(meta) = fs::metadata(&path) {
            if meta
                .modified()
                .ok()
                .and_then(|m| SystemTime::now().duration_since(m).ok())
                .is_some_and(|age| age > Duration::from_secs(300))
            {
                let _ = fs::remove_file(&path);
            }
        }
        let held = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .is_ok();
        SweepLock { path, held }
    }
}

impl Drop for SweepLock {
    fn drop(&mut self) {
        if self.held {
            let _ = fs::remove_file(&self.path);
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct EvictionReport {
    pub removed: Vec<PathBuf>,
    pub freed_bytes: u64,
    pub remaining_bytes: u64,
}

/// Evict least-recently-used entries until the root is under `max_bytes`, then remove anything
/// older than `max_age`.
///
/// `current` is never evicted — it is the build that just succeeded, and removing it would delete
/// the artefact the caller is about to report. Pinned entries are never evicted either.
pub fn evict(
    root: &Path,
    current: Option<&Path>,
    max_bytes: u64,
    max_age: Duration,
) -> EvictionReport {
    let lock = SweepLock::acquire(root);
    let mut report = EvictionReport::default();
    if !lock.held {
        // Another build is sweeping. Skipping is correct: the cap is a bound to be maintained
        // eventually, not an invariant to enforce by racing.
        report.remaining_bytes = status(root).total_bytes();
        return report;
    }

    let candidates = status(root);
    let protected = |entry: &CacheEntry| -> bool {
        entry.pinned
            || current.is_some_and(|c| {
                c == entry.path
                    || c.canonicalize().ok() == entry.path.canonicalize().ok() && c.exists()
            })
    };

    // Oldest first for eviction.
    let mut ordered: Vec<&CacheEntry> = candidates.entries.iter().collect();
    ordered.sort_by_key(|e| e.last_used);

    let mut total = candidates.total_bytes();
    let now = SystemTime::now();
    for entry in ordered {
        let over_cap = total > max_bytes;
        let too_old = now
            .duration_since(entry.last_used)
            .is_ok_and(|age| age > max_age);
        if !over_cap && !too_old {
            continue;
        }
        if protected(entry) {
            continue;
        }
        if fs::remove_dir_all(&entry.path).is_ok() {
            total = total.saturating_sub(entry.bytes);
            report.freed_bytes += entry.bytes;
            report.removed.push(entry.path.clone());
        }
    }
    report.remaining_bytes = total;
    report
}

/// Remove every entry under the root, including pinned ones. This is the explicit user action, so
/// it does what it says; ordinary eviction is the one that protects pins.
pub fn clean(root: &Path) -> EvictionReport {
    let _lock = SweepLock::acquire(root);
    let mut report = EvictionReport::default();
    for entry in status(root).entries {
        if fs::remove_dir_all(&entry.path).is_ok() {
            report.freed_bytes += entry.bytes;
            report.removed.push(entry.path);
        }
    }
    report
}
