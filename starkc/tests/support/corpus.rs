//! WP-C6.5 §9 — the differential corpus manifest, its strict validation, and the lock.
//!
//! Three separable pieces, in dependency order:
//!
//! 1. [`parse_manifest`] — a deliberately small strict reader for the manifest subset (§9.4
//!    option 2: no network-fetched dependency is added to parse a test manifest, and the workspace
//!    has no TOML parser to reuse). It accepts comments, `[[case]]` headers and
//!    `key = "string"` / `key = ["a", "b"]` / `key = true` — and rejects everything else,
//!    including unknown keys. A parser that skips what it does not understand turns a typo'd
//!    attribute into an attribute nobody checks.
//! 2. [`validate`] — §9.3's rejection list, evaluated against the manifest AND the filesystem.
//! 3. [`verify_lock`] — §9.5's integrity proof: every listed file exists and hashes as recorded,
//!    no unlisted corpus source exists, manifest and generator hashes match, counts match, paths
//!    are `/`-separated and canonically ordered.
//!
//! The corpus is **not** the inherited `exec_snapshots` corpus and does not share its lock (§9.5
//! is explicit about that). This one has its own version line, its own generator, and its own
//! freeze.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

/// Case categories, one per coverage-matrix group (§7.3). A case must name one of these — an
/// unknown category is rejected rather than created on the fly, so the corpus cannot grow a
/// category the matrix does not track.
pub const CATEGORIES: [&str; 8] = [
    "expressions-statements",
    "control-transfer",
    "patterns",
    "values-types",
    "calls-dispatch",
    "ownership-drop",
    "traps",
    "packages-environment",
];

pub const KINDS: [&str; 3] = ["handwritten", "generated", "retained"];
pub const PACKAGE_GRAPHS: [&str; 3] = ["single-file", "package", "workspace"];
pub const OUTCOMES: [&str; 2] = ["completion", "trap"];
pub const ENGINES: [&str; 3] = ["hir", "mir", "native-debug"];

/// The Tier-1 targets C6 qualifies on (`C6-PLATFORM-MATRIX.md`). A case may not require a target
/// outside this set: C6.5's claim is about Tier-1 agreement, and a required target nobody runs is
/// a claim nobody checks.
pub const TARGETS: [&str; 2] = ["aarch64-apple-darwin", "x86_64-unknown-linux-gnu"];

/// §4.4's ALLOWED quarantine reason classes. A quarantine string must begin with one of these and
/// name its authority (`CD-###`). Everything on §4.4's disallowed list — engine disagreement,
/// wrong output, wrong trap category, wrong Drop order, native refusal of an accepted program,
/// nondeterminism, slowness, inconvenient package shape, lateness — is absent by construction:
/// there is no reason class it could be written under.
pub const QUARANTINE_REASONS: [&str; 3] = ["non-core-feature", "external-artifact", "environment"];

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct Case {
    pub case_id: String,
    pub kind: String,
    pub category: String,
    pub subcategories: Vec<String>,
    pub sources: Vec<String>,
    pub package_graph: String,
    /// For a `package`/`workspace` case: the corpus-relative directory holding the ROOT package's
    /// `starkpkg.json`. Explicit rather than inferred from the source list — a convention like
    /// "the shallowest manifest" would silently pick the wrong root the first time a case has two
    /// candidate packages at the same depth.
    pub package_root: Option<String>,
    pub language_options: Vec<String>,
    pub expected_outcome: String,
    pub expected_trap_category: Option<String>,
    pub required_engines: Vec<String>,
    pub required_targets: Vec<String>,
    pub metamorphic_family: Option<String>,
    pub metamorphic_group: Option<String>,
    /// `base` or `transformed` (§13.3). A group is exactly one of each; the harness pairs them by
    /// group, so a group with two bases or two transformed members is a malformed group rather than
    /// a pair that quietly compares something to itself.
    pub metamorphic_role: Option<String>,
    /// The named rule or narrow equivalence that makes the transformation semantics-preserving
    /// (§13.3). Recorded per group because the preconditions are real constraints — scope insertion
    /// is invalid over a `Drop` type, arm reordering is invalid past a catch-all — and a pair whose
    /// precondition is unstated cannot be reviewed.
    pub metamorphic_precondition: Option<String>,
    pub generator_seed: Option<String>,
    pub generator_version: Option<String>,
    pub template_id: Option<String>,
    pub normative_rules: Vec<String>,
    /// The exact stdout all three engines must produce, as lines joined by `\n` (§10.3). Stated in
    /// the manifest rather than left to agreement because a sentinel's whole purpose is to fail
    /// under a wrong implementation — and three engines sharing one wrong rendering agree perfectly.
    /// Empty means "not pinned"; a case whose observation IS its output should pin it.
    pub expected_stdout: Vec<String>,
    /// The §8.8 Drop identities, in the order they must be destroyed. Same reasoning: unanimity on a
    /// wrong Drop schedule is still wrong.
    pub expected_drop_log: Vec<String>,
    pub return_probe: Option<String>,
    pub drop_protocol: bool,
    pub deviation: Option<String>,
    pub quarantine: Option<String>,
}

enum Value {
    Str(String),
    List(Vec<String>),
    Bool(bool),
}

/// The manifest subset, parsed strictly (§9.4).
pub fn parse_manifest(text: &str) -> Result<Vec<Case>, String> {
    let mut cases: Vec<Case> = Vec::new();
    let mut seen_keys: BTreeSet<String> = BTreeSet::new();

    for (index, raw) in text.lines().enumerate() {
        let line_no = index + 1;
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if line == "[[case]]" {
            cases.push(Case::default());
            seen_keys.clear();
            continue;
        }
        if line.starts_with('[') {
            return Err(format!(
                "line {line_no}: only `[[case]]` tables exist in this manifest, found {line:?}"
            ));
        }
        let (key, value_text) = line
            .split_once(" = ")
            .ok_or_else(|| format!("line {line_no}: not a `key = value` line: {line:?}"))?;
        let case = cases
            .last_mut()
            .ok_or_else(|| format!("line {line_no}: `{key}` appears before any `[[case]]`"))?;
        if !seen_keys.insert(key.to_string()) {
            return Err(format!("line {line_no}: duplicate key `{key}` in one case"));
        }
        let value = parse_value(value_text)
            .map_err(|reason| format!("line {line_no}: `{key}`: {reason}"))?;
        assign(case, key, value).map_err(|reason| format!("line {line_no}: {reason}"))?;
    }
    Ok(cases)
}

fn parse_value(text: &str) -> Result<Value, String> {
    let text = text.trim();
    if text == "true" {
        return Ok(Value::Bool(true));
    }
    if text == "false" {
        return Ok(Value::Bool(false));
    }
    if let Some(inner) = text.strip_prefix('[').and_then(|t| t.strip_suffix(']')) {
        return Ok(Value::List(parse_list_items(inner.trim())?));
    }
    Ok(Value::Str(parse_string(text)?))
}

/// A list body, scanned rather than split on `,` — a value may legitimately CONTAIN a comma
/// (`expected_stdout = ["[1, 2, 3]"]` is a rendered array), and splitting first would tear it in
/// half. Quotes delimit; commas outside quotes separate.
fn parse_list_items(body: &str) -> Result<Vec<String>, String> {
    let mut items = Vec::new();
    let mut rest = body;
    loop {
        rest = rest.trim_start();
        if rest.is_empty() {
            return Ok(items);
        }
        let after_open = rest
            .strip_prefix('"')
            .ok_or_else(|| format!("expected a double-quoted string in a list, found {rest:?}"))?;
        let end = after_open
            .find('"')
            .ok_or_else(|| format!("unterminated string in a list: {rest:?}"))?;
        let (item, tail) = after_open.split_at(end);
        if item.contains('\\') {
            return Err(format!(
                "quotes and backslashes are not supported in manifest strings: {item:?}"
            ));
        }
        items.push(item.to_string());
        rest = tail[1..].trim_start();
        match rest.strip_prefix(',') {
            Some(next) => rest = next,
            None if rest.is_empty() => return Ok(items),
            None => return Err(format!("expected `,` between list items, found {rest:?}")),
        }
    }
}

fn parse_string(text: &str) -> Result<String, String> {
    let inner = text
        .strip_prefix('"')
        .and_then(|t| t.strip_suffix('"'))
        .ok_or_else(|| format!("expected a double-quoted string, found {text:?}"))?;
    if inner.contains('"') || inner.contains('\\') {
        // The subset has no escapes on purpose: a manifest value needing one is a sign the field
        // wants restructuring, not that the parser wants a lexer.
        return Err(format!(
            "quotes and backslashes are not supported in manifest strings: {inner:?}"
        ));
    }
    Ok(inner.to_string())
}

fn assign(case: &mut Case, key: &str, value: Value) -> Result<(), String> {
    fn want_str(key: &str, value: Value) -> Result<String, String> {
        match value {
            Value::Str(s) => Ok(s),
            _ => Err(format!("`{key}` must be a string")),
        }
    }
    fn want_list(key: &str, value: Value) -> Result<Vec<String>, String> {
        match value {
            Value::List(items) => Ok(items),
            _ => Err(format!("`{key}` must be a list of strings")),
        }
    }
    fn want_bool(key: &str, value: Value) -> Result<bool, String> {
        match value {
            Value::Bool(b) => Ok(b),
            _ => Err(format!("`{key}` must be `true` or `false`")),
        }
    }

    match key {
        "case_id" => case.case_id = want_str(key, value)?,
        "kind" => case.kind = want_str(key, value)?,
        "category" => case.category = want_str(key, value)?,
        "subcategories" => case.subcategories = want_list(key, value)?,
        "sources" => case.sources = want_list(key, value)?,
        "package_graph" => case.package_graph = want_str(key, value)?,
        "package_root" => case.package_root = Some(want_str(key, value)?),
        "language_options" => case.language_options = want_list(key, value)?,
        "expected_outcome" => case.expected_outcome = want_str(key, value)?,
        "expected_trap_category" => case.expected_trap_category = Some(want_str(key, value)?),
        "required_engines" => case.required_engines = want_list(key, value)?,
        "required_targets" => case.required_targets = want_list(key, value)?,
        "metamorphic_family" => case.metamorphic_family = Some(want_str(key, value)?),
        "metamorphic_group" => case.metamorphic_group = Some(want_str(key, value)?),
        "metamorphic_role" => case.metamorphic_role = Some(want_str(key, value)?),
        "metamorphic_precondition" => case.metamorphic_precondition = Some(want_str(key, value)?),
        "generator_seed" => case.generator_seed = Some(want_str(key, value)?),
        "generator_version" => case.generator_version = Some(want_str(key, value)?),
        "template_id" => case.template_id = Some(want_str(key, value)?),
        "normative_rules" => case.normative_rules = want_list(key, value)?,
        "expected_stdout" => case.expected_stdout = want_list(key, value)?,
        "expected_drop_log" => case.expected_drop_log = want_list(key, value)?,
        "return_probe" => case.return_probe = Some(want_str(key, value)?),
        "drop_protocol" => case.drop_protocol = want_bool(key, value)?,
        "deviation" => case.deviation = Some(want_str(key, value)?),
        "quarantine" => case.quarantine = Some(want_str(key, value)?),
        other => {
            return Err(format!(
                "unknown manifest key `{other}` — add fields only when they carry a real \
                 invariant (§9.2), and teach this parser about them in the same change"
            ))
        }
    }
    Ok(())
}

/// §9.3, evaluated against the manifest and the filesystem together. Returns the FIRST violation:
/// the corpus is small enough that fixing them one at a time is not a burden, and a list of
/// derived complaints from one root cause is noise.
pub fn validate(cases: &[Case], root: &Path) -> Result<(), String> {
    let mut ids: BTreeSet<&str> = BTreeSet::new();
    let mut owned_sources: BTreeMap<&str, &str> = BTreeMap::new();

    for case in cases {
        let id = case.case_id.as_str();
        if id.is_empty() {
            return Err("a case has no `case_id`".to_string());
        }
        if !ids.insert(id) {
            return Err(format!("duplicate case_id `{id}`"));
        }
        if !KINDS.contains(&case.kind.as_str()) {
            return Err(format!("{id}: unknown kind `{}`", case.kind));
        }
        if !CATEGORIES.contains(&case.category.as_str()) {
            return Err(format!("{id}: unknown category `{}`", case.category));
        }
        if !PACKAGE_GRAPHS.contains(&case.package_graph.as_str()) {
            return Err(format!(
                "{id}: unknown package_graph `{}`",
                case.package_graph
            ));
        }
        if !OUTCOMES.contains(&case.expected_outcome.as_str()) {
            return Err(format!(
                "{id}: unknown expected_outcome `{}`",
                case.expected_outcome
            ));
        }
        if case.expected_outcome == "trap" && case.expected_trap_category.is_none() {
            return Err(format!(
                "{id}: a trap case must state `expected_trap_category` — otherwise the corpus \
                 records that it traps but not that it traps for the right reason"
            ));
        }
        if case.sources.is_empty() {
            return Err(format!("{id}: no sources"));
        }
        // A package case must say where its root manifest is, and a single-file case must not
        // pretend to have one.
        match (case.package_graph.as_str(), &case.package_root) {
            ("single-file", None) => {}
            ("single-file", Some(_)) => {
                return Err(format!("{id}: a single-file case has no package_root"))
            }
            (_, None) => {
                return Err(format!(
                "{id}: a `{}` case must name its `package_root` (the directory holding the root \
                     starkpkg.json)",
                case.package_graph
            ))
            }
            (_, Some(root)) => {
                if !root_dir_ok(root) {
                    return Err(format!(
                        "{id}: package_root `{root}` is not a safe relative path"
                    ));
                }
                if !root.starts_with("cases/") && !root.starts_with("metamorphic/") {
                    return Err(format!(
                        "{id}: package_root `{root}` is outside the case tree"
                    ));
                }
            }
        }
        if case.required_engines.is_empty() {
            return Err(format!("{id}: no required_engines"));
        }
        for engine in &case.required_engines {
            if !ENGINES.contains(&engine.as_str()) {
                return Err(format!("{id}: unknown required engine `{engine}`"));
            }
        }
        if case.required_targets.is_empty() {
            return Err(format!("{id}: no required_targets"));
        }
        for target in &case.required_targets {
            if !TARGETS.contains(&target.as_str()) {
                return Err(format!("{id}: unknown required target `{target}`"));
            }
        }
        // A Core case must say which rules it is evidence FOR. An empty list makes the case a
        // program that runs rather than a claim that is checked.
        if case.quarantine.is_none() && case.normative_rules.is_empty() {
            return Err(format!(
                "{id}: a Core case must cite at least one normative rule"
            ));
        }
        // ...and the rule must EXIST. See `spec_rule_ids`: the coverage matrix was built with 69
        // invented identifiers out of 84, so "cites a rule" is only meaningful once the citation is
        // checked against the specification.
        for rule in &case.normative_rules {
            if !spec_rule_ids().contains(rule) {
                return Err(format!(
                    "{id}: `{rule}` is not a rule defined in STARKLANG/docs/spec — a citation that \
                     resolves to nothing is worse than none, because it looks like grounding"
                ));
            }
        }
        if !case.expected_drop_log.is_empty() && !case.drop_protocol {
            return Err(format!(
                "{id}: an expected Drop log needs `drop_protocol = true` — the frames are only \
                 parsed out of stdout for cases that declare the protocol"
            ));
        }
        if case.kind == "generated"
            && (case.generator_seed.is_none()
                || case.generator_version.is_none()
                || case.template_id.is_none())
        {
            return Err(format!(
                "{id}: a generated case must carry generator_seed, generator_version and \
                 template_id — without them it cannot be regenerated and is not reproducible"
            ));
        }
        match (&case.metamorphic_family, &case.metamorphic_group) {
            (None, None) => {}
            (Some(_), Some(_)) => {
                // §13.3 requires the role and the precondition too: without the role the harness
                // cannot tell which member is the base, and without the precondition the claim that
                // the transformation preserves semantics is unreviewable.
                if case.metamorphic_role.is_none() || case.metamorphic_precondition.is_none() {
                    return Err(format!(
                        "{id}: a metamorphic member needs `metamorphic_role` and \
                         `metamorphic_precondition` (§13.3)"
                    ));
                }
                let role = case.metamorphic_role.as_deref().unwrap_or_default();
                if role != "base" && role != "transformed" {
                    return Err(format!(
                        "{id}: metamorphic_role must be base or transformed"
                    ));
                }
            }
            _ => {
                return Err(format!(
                    "{id}: a metamorphic member needs BOTH family and group"
                ))
            }
        }
        if let Some(quarantine) = &case.quarantine {
            let allowed = QUARANTINE_REASONS
                .iter()
                .any(|reason| quarantine.starts_with(reason));
            if !allowed {
                return Err(format!(
                    "{id}: quarantine reason must be one of {QUARANTINE_REASONS:?} (§4.4). A \
                     semantic quarantine — engine disagreement, wrong output, wrong trap, wrong \
                     Drop order, native refusal of an accepted program — is not an available \
                     classification: that is a C6 blocker, and it keeps the gate open"
                ));
            }
            if !quarantine.contains("CD-") {
                return Err(format!(
                    "{id}: quarantine must name its deciding authority (a `CD-###` directive)"
                ));
            }
        }

        for source in &case.sources {
            if source.starts_with('/') || source.contains(':') {
                return Err(format!("{id}: absolute source path `{source}`"));
            }
            if source.contains('\\') {
                return Err(format!(
                    "{id}: source path `{source}` must use `/` separators on every platform"
                ));
            }
            if source.split('/').any(|part| part == "..") {
                return Err(format!(
                    "{id}: source path `{source}` escapes the corpus root"
                ));
            }
            if let Some(owner) = owned_sources.insert(source.as_str(), id) {
                return Err(format!(
                    "{id}: source `{source}` is already owned by case `{owner}` — one file, one \
                     case, or a change to it silently changes two claims"
                ));
            }
            if !root.join(source).is_file() {
                return Err(format!("{id}: source `{source}` does not exist"));
            }
        }
    }

    // Enumeration order is part of the contract (§9.3). `load_manifests` sorts the merged set and
    // checks each FILE's internal order separately; this guards callers that build a case list
    // directly, including every rejection test.
    let sorted: Vec<&str> = {
        let mut ids: Vec<&str> = cases.iter().map(|c| c.case_id.as_str()).collect();
        ids.sort_unstable();
        ids
    };
    let declared: Vec<&str> = cases.iter().map(|c| c.case_id.as_str()).collect();
    if declared != sorted {
        return Err("manifest cases are not in ascending case_id order".to_string());
    }

    // Every corpus source must be claimed. An unlisted `.stark` under `cases/` is a file the
    // corpus runs nothing against and the lock would still hash — evidence-shaped, unchecked.
    for found in enumerate_sources(root)? {
        if !owned_sources.contains_key(found.as_str()) {
            return Err(format!(
                "`{found}` exists in the corpus but no case lists it"
            ));
        }
    }
    Ok(())
}

/// Every corpus source file, corpus-root-relative, `/`-separated, sorted. Only `.stark` files
/// count as sources; the manifest, lock, README and generator are corpus INFRASTRUCTURE and are
/// hashed separately (§9.5) rather than treated as cases.
fn root_dir_ok(path: &str) -> bool {
    !path.starts_with('/')
        && !path.contains(':')
        && !path.contains('\\')
        && !path.split('/').any(|part| part == "..")
}

pub fn enumerate_sources(root: &Path) -> Result<Vec<String>, String> {
    fn walk(dir: &Path, base: &Path, out: &mut Vec<String>) -> Result<(), String> {
        let entries = std::fs::read_dir(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        for entry in entries {
            let path = entry.map_err(|e| e.to_string())?.path();
            if path.is_dir() {
                walk(&path, base, out)?;
            } else if path.extension().and_then(|e| e.to_str()) == Some("stark") {
                let relative = path
                    .strip_prefix(base)
                    .map_err(|e| e.to_string())?
                    .components()
                    .map(|c| c.as_os_str().to_string_lossy().into_owned())
                    .collect::<Vec<_>>()
                    .join("/");
                out.push(relative);
            }
        }
        Ok(())
    }
    let mut out = Vec::new();
    walk(root, root, &mut out)?;
    out.sort();
    Ok(out)
}

pub fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    let digest = Sha256::digest(bytes);
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(digest.len() * 2);
    for &b in digest.iter() {
        out.push(HEX[(b >> 4) as usize] as char);
        out.push(HEX[(b & 0x0f) as usize] as char);
    }
    out
}

/// The parsed `corpus.lock` (§9.5).
pub struct Lock {
    pub headers: BTreeMap<String, String>,
    /// Corpus-root-relative path → SHA-256, in the order the lock lists them.
    pub files: Vec<(String, String)>,
}

pub fn parse_lock(text: &str) -> Result<Lock, String> {
    let mut headers = BTreeMap::new();
    let mut files = Vec::new();
    for (index, raw) in text.lines().enumerate() {
        let line = raw.trim_end();
        if line.trim().is_empty() || line.starts_with('#') {
            continue;
        }
        if let Some((key, value)) = line.split_once(" = ") {
            if headers.insert(key.to_string(), value.to_string()).is_some() {
                return Err(format!("line {}: duplicate lock header `{key}`", index + 1));
            }
            continue;
        }
        let (hash, path) = line
            .split_once("  ")
            .ok_or_else(|| format!("line {}: malformed lock line {line:?}", index + 1))?;
        files.push((path.to_string(), hash.to_string()));
    }
    Ok(Lock { headers, files })
}

/// §9.5's integrity proof. Every clause is a separate error message because "the lock does not
/// match" is not an actionable failure for whoever hits it in CI.
pub fn verify_lock(root: &Path, lock: &Lock, cases: &[Case]) -> Result<(), String> {
    fn header<'a>(lock: &'a Lock, key: &str) -> Result<&'a str, String> {
        lock.headers
            .get(key)
            .map(String::as_str)
            .ok_or_else(|| format!("corpus.lock is missing `{key}`"))
    }

    for (path, _) in &lock.files {
        if path.contains('\\') {
            return Err(format!("lock path `{path}` must use `/` separators"));
        }
    }
    let ordered: Vec<&String> = lock.files.iter().map(|(p, _)| p).collect();
    let mut sorted = ordered.clone();
    sorted.sort();
    if ordered != sorted {
        return Err("corpus.lock file lines are not in canonical (sorted) path order".to_string());
    }

    let actual = enumerate_sources(root)?;
    let locked: BTreeMap<&str, &str> = lock
        .files
        .iter()
        .map(|(p, h)| (p.as_str(), h.as_str()))
        .collect();
    if locked.len() != lock.files.len() {
        return Err("corpus.lock lists a path twice".to_string());
    }
    for path in &actual {
        if !locked.contains_key(path.as_str()) {
            return Err(format!(
                "`{path}` exists in the corpus but corpus.lock does not list it"
            ));
        }
    }
    for (path, expected) in &lock.files {
        let full = root.join(path);
        let bytes = std::fs::read(&full)
            .map_err(|e| format!("corpus.lock lists `{path}`, which cannot be read: {e}"))?;
        let observed = sha256_hex(&bytes);
        if &observed != expected {
            return Err(format!(
                "`{path}` has changed: locked {expected}, found {observed}. A corpus change \
                 requires regenerating corpus.lock AND bumping corpus_version (§9.6)"
            ));
        }
    }

    let manifest_bytes = std::fs::read(root.join("manifest.toml")).map_err(|e| e.to_string())?;
    if header(lock, "manifest_sha256")? != sha256_hex(&manifest_bytes) {
        return Err(
            "manifest.toml has changed but corpus.lock's manifest_sha256 has not".to_string(),
        );
    }
    let generator_bytes = std::fs::read(root.join("generate.py")).map_err(|e| e.to_string())?;
    if header(lock, "generator_sha256")? != sha256_hex(&generator_bytes) {
        return Err(
            "generate.py has changed but corpus.lock's generator_sha256 has not".to_string(),
        );
    }

    let counted = |kind: &str| cases.iter().filter(|c| c.kind == kind).count();
    for (key, expected) in [
        ("case_count", cases.len()),
        ("handwritten_count", counted("handwritten")),
        ("generated_count", counted("generated")),
        ("retained_count", counted("retained")),
        (
            "metamorphic_group_count",
            cases
                .iter()
                .filter_map(|c| c.metamorphic_group.as_deref())
                .collect::<BTreeSet<_>>()
                .len(),
        ),
    ] {
        let recorded: usize = header(lock, key)?
            .parse()
            .map_err(|_| format!("corpus.lock `{key}` is not a number"))?;
        if recorded != expected {
            return Err(format!(
                "corpus.lock `{key}` says {recorded}, the manifest has {expected}"
            ));
        }
    }
    Ok(())
}

/// Every normative rule ID the specification actually defines, parsed from the normative source
/// documents under `STARKLANG/docs/spec/`.
///
/// This exists because of a real failure: phase C6.5-0 built the coverage matrix with **69 of its 84
/// cited rule IDs invented** — `OWN-DROP-001`, `FN-VALUE-001`, `MAP-001`, `TRAP-ABORT-001` and 65
/// others are plausible-looking identifiers that appear in no spec document. The §7.5 exit condition
/// "every row has a normative citation" was satisfied by fabricated text, and nothing checked it
/// (CD-154). A citation nobody verifies is worse than no citation: it *looks* like grounding.
///
/// So citations are now machine-checked, here and for the matrix itself. The generated
/// `STARK-Core-v1.md` is excluded deliberately — it is a compilation of the source documents, so
/// reading it would let a stale generated copy validate an ID the sources no longer define.
pub fn spec_rule_ids() -> BTreeSet<String> {
    let spec_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("repo root")
        .join("STARKLANG/docs/spec");
    let mut ids = BTreeSet::new();
    for entry in std::fs::read_dir(&spec_dir).expect("spec directory") {
        let path = entry.expect("spec entry").path();
        let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
        if !name.ends_with(".md") || name.starts_with("STARK-Core-v1") {
            continue;
        }
        let text = std::fs::read_to_string(&path).expect("spec file");
        ids.extend(rule_ids_in(&text));
    }
    assert!(
        ids.len() > 50,
        "only {} rule IDs found under STARKLANG/docs/spec — the parser or the path is wrong, and a \
         silently empty authority set would make every citation check vacuous",
        ids.len()
    );
    ids
}

/// Rule-ID-shaped tokens in one document: `WORD-WORD-001`, at least two segments before the number.
/// `CD-`/`DEV-` prefixes are governance references, not spec rules, and are excluded.
pub fn rule_ids_in(text: &str) -> BTreeSet<String> {
    let mut ids = BTreeSet::new();
    for token in text.split(|c: char| !(c.is_ascii_alphanumeric() || c == '-')) {
        let Some((head, tail)) = token.rsplit_once('-') else {
            continue;
        };
        let digits = tail.len() == 3 && tail.chars().all(|c| c.is_ascii_digit());
        let shaped = head.contains('-')
            && head
                .chars()
                .all(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || c == '-');
        if digits && shaped && !token.starts_with("CD-") && !token.starts_with("DEV-") {
            ids.insert(token.to_string());
        }
    }
    ids
}

/// The corpus root, as an absolute path — tests must not depend on the process working directory.
pub fn corpus_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/c6-corpus")
}

/// The corpus is described by TWO manifests: `manifest.toml` for hand-authored cases and
/// `generated.manifest.toml` for `--write` output. The split keeps a regeneration from ever
/// touching a case a person wrote, and keeps a person from editing an expectation the generator
/// derived. Each file must be internally sorted; the merged set is sorted for enumeration, so case
/// order does not depend on which file a case came from.
pub fn load_manifests(root: &Path) -> Result<Vec<Case>, String> {
    let mut merged: Vec<Case> = Vec::new();
    for name in ["manifest.toml", "generated.manifest.toml"] {
        let path = root.join(name);
        if !path.is_file() {
            if name == "manifest.toml" {
                return Err(format!("{name} is missing"));
            }
            continue; // the generated half is absent until `generate.py --write` has run
        }
        let text = std::fs::read_to_string(&path).map_err(|e| format!("{name}: {e}"))?;
        let cases = parse_manifest(&text).map_err(|reason| format!("{name}: {reason}"))?;
        let ids: Vec<&str> = cases.iter().map(|c| c.case_id.as_str()).collect();
        let mut sorted = ids.clone();
        sorted.sort_unstable();
        if ids != sorted {
            return Err(format!("{name}: cases are not in ascending case_id order"));
        }
        merged.extend(cases);
    }
    merged.sort_by(|a, b| a.case_id.cmp(&b.case_id));
    Ok(merged)
}

pub fn load() -> (Vec<Case>, Lock) {
    let root = corpus_root();
    let cases = load_manifests(&root).unwrap_or_else(|reason| panic!("manifest: {reason}"));
    validate(&cases, &root).unwrap_or_else(|reason| panic!("manifest: {reason}"));
    let lock_text = std::fs::read_to_string(root.join("corpus.lock")).expect("corpus.lock");
    let lock = parse_lock(&lock_text).unwrap_or_else(|reason| panic!("corpus.lock: {reason}"));
    (cases, lock)
}

// ------------------------------------------------------- §12.6/§12.7 replay control --

/// §12.7's deterministic shard assignment: `u64(SHA-256(case_id)[0..8]) % shard_total`.
///
/// Content-addressed rather than index-based on purpose. An index-based split (`i % total`) changes
/// every case's shard when one case is added, so two shards from adjacent commits are not
/// comparable; a digest split moves only the cases whose digests demand it.
pub fn shard_of(case_id: &str, shard_total: u64) -> u64 {
    assert!(shard_total > 0, "shard_total must be positive");
    let digest = sha256_hex(case_id.as_bytes());
    let prefix = u64::from_str_radix(&digest[..16], 16).expect("hex prefix");
    prefix % shard_total
}

/// §12.6's replay filters, read from the environment.
///
/// **A filtered run is a diagnostic run, not closure evidence** — the plan is explicit, so
/// [`Filters::is_full_evidence`] is recorded in the evidence summary and a filtered summary can never
/// be mistaken for a complete one.
#[derive(Debug, Default, Clone)]
pub struct Filters {
    pub case: Option<String>,
    pub category: Option<String>,
    pub template: Option<String>,
    pub kind: Option<String>,
    pub engine: Option<String>,
    pub shard_index: Option<u64>,
    pub shard_total: Option<u64>,
    pub keep_temp: bool,
}

impl Filters {
    pub fn from_env() -> Self {
        fn var(name: &str) -> Option<String> {
            std::env::var(name).ok().filter(|value| !value.is_empty())
        }
        Self {
            case: var("C6_CASE"),
            category: var("C6_CATEGORY"),
            template: var("C6_TEMPLATE"),
            kind: var("C6_KIND"),
            engine: var("C6_ENGINE"),
            shard_index: var("C6_SHARD_INDEX").and_then(|v| v.parse().ok()),
            shard_total: var("C6_SHARD_TOTAL").and_then(|v| v.parse().ok()),
            keep_temp: var("C6_KEEP_TEMP").is_some(),
        }
    }

    /// True only when nothing narrows the run. Sharding counts as narrowing: a shard is complete
    /// evidence for the shard, and evidence for the corpus only once every shard is merged (§12.7).
    pub fn is_full_evidence(&self) -> bool {
        self.case.is_none()
            && self.category.is_none()
            && self.template.is_none()
            && self.kind.is_none()
            && self.engine.is_none()
            && self.shard_index.is_none()
            && self.shard_total.is_none()
    }

    pub fn selects(&self, case: &Case) -> bool {
        if let Some(id) = &self.case {
            if &case.case_id != id {
                return false;
            }
        }
        if let Some(category) = &self.category {
            if &case.category != category {
                return false;
            }
        }
        if let Some(template) = &self.template {
            if case.template_id.as_deref() != Some(template.as_str()) {
                return false;
            }
        }
        if let Some(kind) = &self.kind {
            if &case.kind != kind {
                return false;
            }
        }
        if let (Some(index), Some(total)) = (self.shard_index, self.shard_total) {
            if shard_of(&case.case_id, total) != index {
                return false;
            }
        }
        true
    }
}
