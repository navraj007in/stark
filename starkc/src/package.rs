use std::collections::HashMap;
use std::path::{Path, PathBuf};

use crate::source_extensions::is_stark_source;

/// **AS5-c: manifests are read by the compiler's one JSON authority.**
///
/// This module owned a second parser. It agreed with the LSP transport's on 3 of 12 constructs,
/// and it was the LESS conformant of the two despite being the one that reads files people write by
/// hand: it rejected every `\u` escape and every exponent number (valid input), while accepting
/// trailing commas, raw control characters and leading-zero numbers (invalid input).
///
/// AS5 CE9 decision: **trailing commas and leading-zero numbers are now rejected.** A manifest is a
/// durable configuration contract, and accepting non-JSON syntax creates compatibility debt for no
/// benefit. `AS0-MANIFEST-STRICTNESS-AUDIT.md` established that every first-party manifest is
/// already strict-clean, so this is a third-party narrowing, not a repository migration.
pub use crate::json::{JsonNumber, JsonValue};

pub fn parse_json(input: &str) -> Result<JsonValue, String> {
    crate::json::parse(input).map_err(|error| error.to_string())
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Version {
    pub major: u64,
    pub minor: u64,
    pub patch: u64,
}

impl Version {
    pub fn parse(s: &str) -> Result<Self, String> {
        let parts: Vec<&str> = s.split('.').collect();
        if parts.len() != 3 {
            return Err(format!("invalid semver: '{}'", s));
        }
        let major = parts[0]
            .parse::<u64>()
            .map_err(|_| format!("invalid major in '{}'", s))?;
        let minor = parts[1]
            .parse::<u64>()
            .map_err(|_| format!("invalid minor in '{}'", s))?;
        let patch = parts[2]
            .parse::<u64>()
            .map_err(|_| format!("invalid patch in '{}'", s))?;
        Ok(Self {
            major,
            minor,
            patch,
        })
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum VersionReq {
    Any,
    Caret(Version),
    Exact(Version),
    Range(Vec<Comparator>),
}

#[derive(Clone, Debug, PartialEq)]
pub enum Comparator {
    Ge(Version),
    Le(Version),
    Gt(Version),
    Lt(Version),
    Eq(Version),
}

impl Comparator {
    pub fn matches(&self, version: &Version) -> bool {
        match self {
            Comparator::Ge(v) => version >= v,
            Comparator::Le(v) => version <= v,
            Comparator::Gt(v) => version > v,
            Comparator::Lt(v) => version < v,
            Comparator::Eq(v) => version == v,
        }
    }
}

impl VersionReq {
    pub fn parse(s: &str) -> Result<Self, String> {
        let s = s.trim();
        if s == "*" || s.is_empty() {
            return Ok(VersionReq::Any);
        }
        if let Some(stripped) = s.strip_prefix('^') {
            let v = Version::parse(stripped)?;
            return Ok(VersionReq::Caret(v));
        }

        let parts: Vec<&str> = s.split(',').collect();
        let mut comparators = Vec::new();
        for part in parts {
            let part = part.trim();
            if let Some(stripped) = part.strip_prefix(">=") {
                comparators.push(Comparator::Ge(Version::parse(stripped.trim())?));
            } else if let Some(stripped) = part.strip_prefix("<=") {
                comparators.push(Comparator::Le(Version::parse(stripped.trim())?));
            } else if let Some(stripped) = part.strip_prefix('>') {
                comparators.push(Comparator::Gt(Version::parse(stripped.trim())?));
            } else if let Some(stripped) = part.strip_prefix('<') {
                comparators.push(Comparator::Lt(Version::parse(stripped.trim())?));
            } else if let Some(stripped) = part.strip_prefix('=') {
                comparators.push(Comparator::Eq(Version::parse(stripped.trim())?));
            } else {
                let v = Version::parse(part)?;
                comparators.push(Comparator::Eq(v));
            }
        }
        if comparators.len() == 1 {
            if let Comparator::Eq(v) = &comparators[0] {
                return Ok(VersionReq::Exact(v.clone()));
            }
        }
        Ok(VersionReq::Range(comparators))
    }

    pub fn matches(&self, version: &Version) -> bool {
        match self {
            VersionReq::Any => true,
            VersionReq::Exact(v) => version == v,
            VersionReq::Caret(v) => {
                if version < v {
                    return false;
                }
                if v.major > 0 {
                    version.major == v.major
                } else if v.minor > 0 {
                    version.major == 0 && version.minor == v.minor
                } else {
                    version.major == 0 && version.minor == 0 && version.patch == v.patch
                }
            }
            VersionReq::Range(comparators) => comparators.iter().all(|c| c.matches(version)),
        }
    }

    pub fn single_major_line(&self) -> Result<u64, String> {
        match self {
            VersionReq::Exact(version) | VersionReq::Caret(version) => Ok(version.major),
            VersionReq::Any => {
                Err("version requirement must identify exactly one major version line".to_string())
            }
            VersionReq::Range(comparators) => {
                if let Some(version) = comparators.iter().find_map(|comparator| match comparator {
                    Comparator::Eq(version) => Some(version),
                    _ => None,
                }) {
                    if comparators
                        .iter()
                        .all(|comparator| comparator.matches(version))
                    {
                        return Ok(version.major);
                    }
                    return Err("version requirement has no satisfiable version".to_string());
                }

                let lower = comparators
                    .iter()
                    .filter_map(|comparator| match comparator {
                        Comparator::Ge(version) | Comparator::Gt(version) => Some(version.major),
                        _ => None,
                    })
                    .max();
                let upper = comparators
                    .iter()
                    .filter_map(|comparator| match comparator {
                        Comparator::Lt(version) if version.minor == 0 && version.patch == 0 => {
                            version.major.checked_sub(1)
                        }
                        Comparator::Lt(version) | Comparator::Le(version) => Some(version.major),
                        _ => None,
                    })
                    .min();
                match (lower, upper) {
                    (Some(lower), Some(upper)) if lower == upper => Ok(lower),
                    _ => Err(
                        "version requirement must identify exactly one major version line"
                            .to_string(),
                    ),
                }
            }
        }
    }
}

fn valid_package_name(name: &str) -> bool {
    (1..=64).contains(&name.len())
        && name
            .bytes()
            .next()
            .is_some_and(|byte| byte.is_ascii_lowercase())
        && name.bytes().all(|byte| {
            byte.is_ascii_lowercase() || byte.is_ascii_digit() || matches!(byte, b'-' | b'_')
        })
}

fn valid_dependency_alias(alias: &str) -> bool {
    const KEYWORDS: &[&str] = &[
        "as", "break", "const", "continue", "else", "enum", "false", "fn", "for", "if", "impl",
        "in", "let", "loop", "match", "mod", "mut", "pub", "return", "self", "Self", "struct",
        "super", "trait", "true", "type", "use", "while",
    ];
    let mut bytes = alias.bytes();
    bytes
        .next()
        .is_some_and(|byte| byte.is_ascii_alphabetic() || byte == b'_')
        && bytes.all(|byte| byte.is_ascii_alphanumeric() || byte == b'_')
        && !KEYWORDS.contains(&alias)
}

pub fn req_to_string(req: &VersionReq) -> String {
    match req {
        VersionReq::Any => "*".to_string(),
        VersionReq::Exact(v) => format!("={}.{}.{}", v.major, v.minor, v.patch),
        VersionReq::Caret(v) => format!("^{}.{}.{}", v.major, v.minor, v.patch),
        VersionReq::Range(comparators) => comparators
            .iter()
            .map(|c| match c {
                Comparator::Ge(v) => format!(">={}.{}.{}", v.major, v.minor, v.patch),
                Comparator::Le(v) => format!("<={}.{}.{}", v.major, v.minor, v.patch),
                Comparator::Gt(v) => format!(">{}.{}.{}", v.major, v.minor, v.patch),
                Comparator::Lt(v) => format!("<{}.{}.{}", v.major, v.minor, v.patch),
                Comparator::Eq(v) => format!("={}.{}.{}", v.major, v.minor, v.patch),
            })
            .collect::<Vec<_>>()
            .join(", "),
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum DependencySource {
    Path(PathBuf),
    Registry(VersionReq),
}

#[derive(Clone, Debug, PartialEq)]
pub struct Dependency {
    pub package: String,
    pub source: DependencySource,
}

#[derive(Clone, Debug)]
pub struct Package {
    pub name: String,
    pub version: Version,
    pub entry: PathBuf,
    pub manifest_path: PathBuf,
    pub dependencies: HashMap<String, Dependency>,
    /// Version of the durable capability vocabulary used by this manifest. Version 1 is the only
    /// vocabulary currently defined. Capability-free manifests default to v1 for compatibility;
    /// manifests that declare an envelope serialize the field explicitly.
    pub capability_vocabulary: u64,
    /// WP-C7.8 (CD-212, Packet 5): the host capabilities this package requires, e.g. `["clock"]`.
    ///
    /// **Declaration is the only admission route.** Packet 5's trust boundary forbids implicit
    /// provider discovery, so a provider is linked if and only if some package asked for its
    /// capability by name. An absent or empty list means the program links no provider at all,
    /// which is the overwhelmingly common case and stays byte-identical to a pre-C7.8 build.
    ///
    /// Sorted and deduplicated at parse time so the requirement set — and therefore the selected
    /// provider set, and therefore the generated manifest — cannot depend on JSON key order.
    pub capabilities: Vec<String>,
    /// WP-C7.8.8 step 1 (CD-225): the package's provider API bindings, if any.
    ///
    /// Absent for the overwhelming majority of packages, which bind nothing.
    pub provider_api: ProviderApi,
}

/// WP-C7.8.8: what a package binds to provider capabilities.
///
/// **It names a callable surface; it never mirrors a signature.** Per CD-224's standing invariant,
/// validated provider metadata is the one authoritative signature, and a binding here carries only
/// identity — capability, symbol, item path — so there is no second copy to drift from. CD-219 is
/// why: a mirrored `unix_now` declared one out-slot where the provider declared two, and metadata
/// validation could not see it because the wrong mirror was internally consistent.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ProviderApi {
    /// Item path → binding. Sorted by path so manifest key order cannot reach the build key.
    pub functions: Vec<ProviderFunctionBinding>,
    /// Package nominal → provider resource. Sorted by nominal, same reason.
    ///
    /// **Core resources are never here.** `file → CoreType::File` is compiler-owned and
    /// undeclarable by any package (CD-224): a Core type whose identity a package manifest could
    /// redefine would put Core's authority over its own types in a package's hands.
    pub resources: Vec<ProviderResourceBinding>,
    /// Capability → the package's raw error type name, sorted by capability.
    ///
    /// The *raw* type only. CD-225 keeps the status-code→public-variant mapping in ordinary STARK,
    /// so this is the minimum identity needed to derive a binding and nothing more.
    pub errors: Vec<(String, String)>,
    /// **HC9 — resources this package's bindings NAME but another package OWNS.** Sorted by
    /// resource.
    ///
    /// CD-360 admitted a provider consuming a foreign resource. A package binding such a function
    /// hits a problem the ruling did not reach: the derived signature's first parameter is a
    /// `TcpStream`, and `TcpStream` is `stark-net`'s nominal, not this package's. Before this,
    /// derivation failed with `UnboundResourceInSignature` — a package could declare a transfer it
    /// could not express.
    ///
    /// The two obvious fixes are both wrong. Binding `tcp_stream` here as an ordinary resource
    /// would synthesize a SECOND `enum TcpStream {}` in this package, a distinct `ItemId` and
    /// therefore a distinct type from the one the net package's calls produce — the program would
    /// hold a handle it could not pass. Inferring the owner from the graph would make a typo
    /// (`tcp_strem`) resolve to nothing and surface far from its cause. So the reference is
    /// **declared**, names the owning package, and resolves to that package's existing nominal.
    pub foreign_resources: Vec<ProviderForeignResourceBinding>,
}

/// **HC9** — one `provider_api.foreign_resources` entry: a resource another package owns, which
/// this package's bound signatures may name.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProviderForeignResourceBinding {
    /// The provider's declared resource-type name, e.g. `tcp_stream`. This is the key the ABI uses,
    /// so it is what the entry is keyed and sorted by.
    pub resource: String,
    /// The DEPENDENCY ALIAS the owning package is imported under, e.g. `stark_net` — not the
    /// package's own name (`stark-net`), because the alias is what appears in a STARK path and a
    /// path is what the derived signature has to render.
    pub package: String,
    /// The nominal that package binds to `resource`, e.g. `TcpStream`.
    pub nominal: String,
}

impl ProviderForeignResourceBinding {
    /// How the nominal renders in a derived signature: `stark_net::TcpStream`.
    ///
    /// Qualified rather than imported. Appending a `use` to the entry file would collide with an
    /// import the package author already wrote — and synthesis has no way to know whether they did,
    /// because it runs before name resolution. A qualified path cannot collide with anything.
    pub fn qualified_nominal(&self) -> String {
        format!("{}::{}", self.package, self.nominal)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProviderFunctionBinding {
    /// The public item path this binds, e.g. `Instant::now_ns` or `env::var_len`.
    pub item_path: String,
    pub capability: String,
    /// The provider symbol, verbatim. Never sanitised (Packet 1 §1.3).
    pub symbol: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProviderResourceBinding {
    /// The package nominal, e.g. `TcpStream`.
    pub nominal: String,
    pub capability: String,
    /// The provider's declared resource-type name, e.g. `tcp_stream`.
    pub resource: String,
}

/// The one logical identity for a file inside a package: `<package>/<path within the package>`,
/// always `/`-joined so the same workspace observes identically on every platform.
///
/// DEV-113-A established this scheme for the parser. AS1a makes it the *only* scheme: a physical
/// source had acquired a second identity because `analyze_project` and three CLI paths each built
/// the entry `SourceFile` from `entry.to_string_lossy()` instead. PKG-IDENTITY-001 requires a
/// package token to be "never an absolute checkout path", and §15.2 requires relocation stability —
/// neither survives a second, path-shaped identity for the same file.
pub fn logical_source_name(package: &str, package_root: &Path, file: &Path) -> String {
    let relative = file
        .strip_prefix(package_root)
        .ok()
        .map(|rel| {
            rel.components()
                .map(|c| c.as_os_str().to_string_lossy().into_owned())
                .collect::<Vec<_>>()
                .join("/")
        })
        .unwrap_or_else(|| {
            file.file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_else(|| "<unknown>".to_string())
        });
    format!("{package}/{relative}")
}

impl Package {
    /// The package's own directory — the manifest's parent.
    pub fn root_dir(&self) -> PathBuf {
        self.manifest_path
            .parent()
            .map(|dir| dir.to_path_buf())
            .unwrap_or_default()
    }

    /// The entry file's single logical identity.
    pub fn entry_logical_name(&self) -> String {
        logical_source_name(&self.name, &self.root_dir(), &self.entry)
    }

    /// **The one way to build a package entry's `SourceFile`.** Logical name for identity, real
    /// disk path for module resolution and for pointing a human at a file.
    ///
    /// AS1a: every caller that needs this file goes through here. Building it by hand is what
    /// produced two `SourceRecord`s for one physical file, made the phantom the only `Root`, and
    /// leaked the checkout path into the native build key.
    pub fn entry_source_file(
        &self,
        contents: impl Into<String>,
    ) -> std::sync::Arc<crate::source::SourceFile> {
        std::sync::Arc::new(
            crate::source::SourceFile::new(self.entry_logical_name(), contents)
                .with_disk_path(self.entry.clone()),
        )
    }

    pub fn from_manifest(path: &Path) -> Result<Self, String> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| format!("failed to read manifest at '{}': {}", path.display(), e))?;

        let json = parse_json(&content)
            .map_err(|e| format!("failed to parse manifest at '{}': {}", path.display(), e))?;

        let obj = json
            .as_object()
            .ok_or_else(|| format!("manifest at '{}' must be a JSON object", path.display()))?;

        let name = obj
            .get("name")
            .ok_or_else(|| format!("missing 'name' in manifest '{}'", path.display()))?
            .as_str()
            .ok_or_else(|| format!("'name' in manifest '{}' must be a string", path.display()))?
            .to_string();

        if !valid_package_name(&name) {
            return Err(format!(
                "invalid package name '{}' in manifest '{}'",
                name,
                path.display()
            ));
        }

        let version_str = obj
            .get("version")
            .ok_or_else(|| format!("missing 'version' in manifest '{}'", path.display()))?
            .as_str()
            .ok_or_else(|| {
                format!(
                    "'version' in manifest '{}' must be a string",
                    path.display()
                )
            })?;
        let version = Version::parse(version_str)
            .map_err(|e| format!("{} in manifest '{}'", e, path.display()))?;

        let entry_str = match obj.get("entry") {
            Some(v) => v.as_str().ok_or_else(|| {
                format!("'entry' in manifest '{}' must be a string", path.display())
            })?,
            None => "src/main.stark",
        };

        let parent_dir = path
            .parent()
            .ok_or("manifest must have a parent directory")?;
        let entry = parent_dir.join(entry_str);
        if !is_stark_source(&entry) {
            return Err(format!(
                "entry file '{}' in manifest '{}' must use .stark or .st",
                entry_str,
                path.display()
            ));
        }

        let entry = entry.canonicalize().map_err(|_| {
            format!(
                "entry file '{}' in manifest '{}' does not exist",
                entry_str,
                path.display()
            )
        })?;

        let mut dependencies = HashMap::new();
        if let Some(deps_val) = obj.get("dependencies") {
            let deps_obj = deps_val.as_object().ok_or_else(|| {
                format!(
                    "'dependencies' in manifest '{}' must be a JSON object",
                    path.display()
                )
            })?;
            for (alias, dep_config_val) in deps_obj {
                if !valid_dependency_alias(alias) {
                    return Err(format!(
                        "dependency alias '{}' in manifest '{}' must be a non-keyword STARK identifier",
                        alias,
                        path.display()
                    ));
                }

                let (package_name, source) = if let Some(version) = dep_config_val.as_str() {
                    if alias.as_str() != alias.to_ascii_lowercase() || !valid_package_name(alias) {
                        return Err(format!(
                            "string dependency '{}' requires the alias to equal its canonical package name",
                            alias
                        ));
                    }
                    let req = VersionReq::parse(version).map_err(|error| {
                        format!(
                            "invalid version requirement '{}' for dependency '{}' in manifest '{}': {}",
                            version,
                            alias,
                            path.display(),
                            error
                        )
                    })?;
                    req.single_major_line()?;
                    (alias.clone(), DependencySource::Registry(req))
                } else {
                    let dep_config = dep_config_val.as_object().ok_or_else(|| {
                        format!(
                            "dependency config for '{}' in manifest '{}' must be a string or JSON object",
                            alias,
                            path.display()
                        )
                    })?;
                    let package_name = dep_config
                        .get("package")
                        .map(|value| {
                            value.as_str().ok_or_else(|| {
                                format!(
                                    "'package' for dependency '{}' in manifest '{}' must be a string",
                                    alias,
                                    path.display()
                                )
                            })
                        })
                        .transpose()?
                        .unwrap_or(alias)
                        .to_string();
                    if !valid_package_name(&package_name) {
                        return Err(format!(
                            "invalid canonical package name '{}' for dependency '{}'",
                            package_name, alias
                        ));
                    }

                    let source = if let Some(dep_path_val) = dep_config.get("path") {
                        let dep_path_str = dep_path_val.as_str().ok_or_else(|| {
                            format!(
                                "'path' for dependency '{}' in manifest '{}' must be a string",
                                alias,
                                path.display()
                            )
                        })?;
                        let dep_dir = parent_dir.join(dep_path_str);
                        let dep_dir = dep_dir.canonicalize().map_err(|_| {
                            format!(
                                "dependency path '{}' for '{}' in manifest '{}' does not exist",
                                dep_path_str,
                                alias,
                                path.display()
                            )
                        })?;
                        DependencySource::Path(dep_dir)
                    } else if let Some(dep_ver_val) = dep_config.get("version") {
                        let dep_ver_str = dep_ver_val.as_str().ok_or_else(|| {
                            format!(
                                "'version' for dependency '{}' in manifest '{}' must be a string",
                                alias,
                                path.display()
                            )
                        })?;
                        let req = VersionReq::parse(dep_ver_str)
                        .map_err(|e| format!("invalid version requirement '{}' for dependency '{}' in manifest '{}': {}", dep_ver_str, alias, path.display(), e))?;
                        req.single_major_line()?;
                        DependencySource::Registry(req)
                    } else {
                        return Err(format!(
                        "dependency '{}' in manifest '{}' must specify either 'path' or 'version'",
                        alias,
                        path.display()
                    ));
                    };
                    (package_name, source)
                };

                dependencies.insert(
                    alias.clone(),
                    Dependency {
                        package: package_name,
                        source,
                    },
                );
            }
        }

        let capability_vocabulary = match obj.get("capability_vocabulary") {
            Some(JsonValue::Number(number)) => number
                .as_u64()
                .filter(|version| *version == 1)
                .ok_or_else(|| {
                    format!(
                        "'capability_vocabulary' in manifest '{}' must be the integer 1",
                        path.display()
                    )
                })?,
            Some(_) => {
                return Err(format!(
                    "'capability_vocabulary' in manifest '{}' must be the integer 1",
                    path.display()
                ))
            }
            None => 1,
        };

        // WP-C7.8: `"capabilities": ["clock", "environment-read"]`. Rejected rather than ignored when
        // malformed -- a typo'd capability that silently vanished would surface much later as a
        // build failure naming a capability nobody could find a requirement for.
        let mut capabilities: Vec<String> = Vec::new();
        if let Some(caps_val) = obj.get("capabilities") {
            let caps = caps_val.as_array().ok_or_else(|| {
                format!(
                    "'capabilities' in manifest '{}' must be a JSON array of strings",
                    path.display()
                )
            })?;
            for cap in caps {
                let name = cap.as_str().ok_or_else(|| {
                    format!(
                        "every entry in 'capabilities' in manifest '{}' must be a string",
                        path.display()
                    )
                })?;
                if name.is_empty() {
                    return Err(format!(
                        "'capabilities' in manifest '{}' contains an empty capability name",
                        path.display()
                    ));
                }
                capabilities.push(name.to_string());
            }
            capabilities.sort();
            capabilities.dedup();
        }

        let provider_api = parse_provider_api(obj, &capabilities, &dependencies, path)?;

        Ok(Self {
            name,
            version,
            entry,
            manifest_path: path.to_path_buf(),
            dependencies,
            capability_vocabulary,
            capabilities,
            provider_api,
        })
    }
}

pub fn get_workspace_root(root_manifest_path: &Path) -> PathBuf {
    if let Some(dir) = root_manifest_path.parent() {
        if let Some(parent) = dir.parent() {
            parent.to_path_buf()
        } else {
            dir.to_path_buf()
        }
    } else {
        PathBuf::from(".")
    }
}

pub fn is_within_workspace(path: &Path, workspace_root: &Path) -> bool {
    path.starts_with(workspace_root)
}

pub fn find_package_root(start_dir: &Path) -> Result<PathBuf, String> {
    let mut current = start_dir
        .canonicalize()
        .map_err(|e| format!("failed to canonicalize start directory: {}", e))?;
    loop {
        let manifest = current.join("starkpkg.json");
        if manifest.exists() {
            return Ok(manifest);
        }
        if let Some(parent) = current.parent() {
            current = parent.to_path_buf();
        } else {
            break;
        }
    }
    Err("missing manifest: starkpkg.json not found in current directory or any parent".to_string())
}

/// Find the local package set belonging to the running installed compiler.
///
/// Both an archive executed in place and a versioned prefix are supported. There is deliberately
/// no environment override or source-checkout fallback: a release qualification must not pass by
/// borrowing packages from the checkout that built it.
pub fn discover_toolchain_package_root(current_exe: Option<&Path>) -> Option<PathBuf> {
    let bin_dir = current_exe?.parent()?;
    for relative in [
        "../lib/stark/current/lib/stark/packages",
        "../lib/stark/packages",
    ] {
        let candidate = bin_dir.join(relative);
        if candidate.is_dir() {
            return Some(
                candidate
                    .canonicalize()
                    .unwrap_or_else(|_| candidate.to_path_buf()),
            );
        }
    }
    None
}

#[derive(Clone, Debug)]
pub struct LockfilePackage {
    pub name: String,
    pub version: Version,
    /// Auditable acquisition origin. Path dependencies use the canonical absolute directory;
    /// registry dependencies use `registry`. Older lockfiles omit this and are upgraded on write.
    pub source: Option<String>,
    pub sha256: String,
    pub dependencies: HashMap<String, String>,
}

#[derive(Clone, Debug)]
pub struct Lockfile {
    pub capability_vocabulary: u64,
    pub packages: HashMap<String, LockfilePackage>,
}

impl Lockfile {
    pub fn parse(content: &str) -> Result<Self, String> {
        let json = parse_json(content)?;
        let obj = json.as_object().ok_or("lockfile must be a JSON object")?;
        let pkgs_val = obj
            .get("packages")
            .ok_or("missing 'packages' in lockfile")?;
        let pkgs_arr = match pkgs_val {
            JsonValue::Array(a) => a,
            _ => return Err("'packages' in lockfile must be an array".to_string()),
        };

        let mut packages = HashMap::new();
        for pkg_val in pkgs_arr {
            let pkg_obj = pkg_val
                .as_object()
                .ok_or("package in lockfile must be a JSON object")?;
            let name = pkg_obj
                .get("name")
                .ok_or("missing name")?
                .as_str()
                .ok_or("name must be string")?
                .to_string();
            let ver_str = pkg_obj
                .get("version")
                .ok_or("missing version")?
                .as_str()
                .ok_or("version must be string")?;
            let version = Version::parse(ver_str)?;
            let sha256 = pkg_obj
                .get("sha256")
                .ok_or("missing sha256")?
                .as_str()
                .ok_or("sha256 must be string")?
                .to_string();
            let source = pkg_obj
                .get("source")
                .map(|value| {
                    value
                        .as_str()
                        .ok_or("source must be string")
                        .map(str::to_string)
                })
                .transpose()?;

            let mut dependencies = HashMap::new();
            if let Some(deps_val) = pkg_obj.get("dependencies") {
                let deps_obj = deps_val.as_object().ok_or("dependencies must be object")?;
                for (d_name, d_ver_val) in deps_obj {
                    dependencies.insert(
                        d_name.clone(),
                        d_ver_val
                            .as_str()
                            .ok_or("dependency version must be string")?
                            .to_string(),
                    );
                }
            }

            packages.insert(
                name.clone(),
                LockfilePackage {
                    name,
                    version,
                    source,
                    sha256,
                    dependencies,
                },
            );
        }
        let capability_vocabulary = obj
            .get("capability_vocabulary")
            .map(|value| {
                value
                    .as_number()
                    .and_then(|number| number.as_u64())
                    .filter(|version| *version == 1)
                    .ok_or("capability_vocabulary must be the integer 1")
            })
            .transpose()?
            .unwrap_or(1);
        Ok(Self {
            capability_vocabulary,
            packages,
        })
    }

    pub fn serialize(&self) -> String {
        let mut lines = Vec::new();
        lines.push("{".to_string());
        lines.push(format!(
            "  \"capability_vocabulary\": {},",
            self.capability_vocabulary
        ));
        lines.push("  \"packages\": [".to_string());

        let mut sorted_packages: Vec<&LockfilePackage> = self.packages.values().collect();
        sorted_packages.sort_by(|a, b| a.name.cmp(&b.name));

        for (i, pkg) in sorted_packages.iter().enumerate() {
            let comma = if i + 1 < sorted_packages.len() {
                ","
            } else {
                ""
            };
            lines.push("    {".to_string());
            lines.push(format!("      \"name\": \"{}\",", pkg.name));
            lines.push(format!(
                "      \"version\": \"{}.{}.{}\",",
                pkg.version.major, pkg.version.minor, pkg.version.patch
            ));
            if let Some(source) = &pkg.source {
                lines.push(format!(
                    "      \"source\": \"{}\",",
                    json_string_contents(source)
                ));
            }
            lines.push(format!("      \"sha256\": \"{}\",", pkg.sha256));
            lines.push("      \"dependencies\": {".to_string());

            let mut sorted_deps: Vec<(&String, &String)> = pkg.dependencies.iter().collect();
            sorted_deps.sort_by(|a, b| a.0.cmp(b.0));
            for (j, (d_name, d_ver)) in sorted_deps.iter().enumerate() {
                let d_comma = if j + 1 < sorted_deps.len() { "," } else { "" };
                lines.push(format!("        \"{}\": \"{}\"{}", d_name, d_ver, d_comma));
            }
            lines.push("      }".to_string());
            lines.push(format!("    }}{}", comma));
        }

        lines.push("  ]".to_string());
        lines.push("}".to_string());
        lines.join("\n")
    }
}

fn json_string_contents(value: &str) -> String {
    let mut escaped = String::new();
    for ch in value.chars() {
        match ch {
            '"' => escaped.push_str("\\\""),
            '\\' => escaped.push_str("\\\\"),
            '\n' => escaped.push_str("\\n"),
            '\r' => escaped.push_str("\\r"),
            '\t' => escaped.push_str("\\t"),
            c if c <= '\u{1f}' => escaped.push_str(&format!("\\u{:04x}", c as u32)),
            c => escaped.push(c),
        }
    }
    escaped
}

struct FileData {
    relative: String,
    content: Vec<u8>,
}

fn get_files_recursive(current: &Path, files: &mut Vec<FileData>) -> Result<(), String> {
    if current.is_file() {
        let file_name = current.file_name().unwrap().to_string_lossy();
        if file_name == "stark.lock" || file_name.starts_with('.') {
            return Ok(());
        }
        let content = std::fs::read(current)
            .map_err(|e| format!("cannot read file '{}': {}", current.display(), e))?;
        files.push(FileData {
            relative: "".to_string(),
            content,
        });
        return Ok(());
    }

    let entries = std::fs::read_dir(current)
        .map_err(|e| format!("cannot read directory '{}': {}", current.display(), e))?;
    for entry in entries {
        let entry = entry.map_err(|e| format!("failed to read entry: {}", e))?;
        let path = entry.path();
        let file_name = path.file_name().unwrap().to_string_lossy();
        if file_name == "stark.lock" || file_name == "target" || file_name.starts_with('.') {
            continue;
        }
        if path.is_file() {
            let content = std::fs::read(&path)
                .map_err(|e| format!("cannot read file '{}': {}", path.display(), e))?;
            files.push(FileData {
                relative: path.file_name().unwrap().to_string_lossy().into_owned(),
                content,
            });
        } else {
            let mut sub_files = Vec::new();
            get_files_recursive(&path, &mut sub_files)?;
            for mut sf in sub_files {
                sf.relative = format!("{}/{}", file_name, sf.relative);
                files.push(sf);
            }
        }
    }
    Ok(())
}

pub fn calculate_dir_sha256(dir: &Path) -> Result<String, String> {
    use sha2::{Digest, Sha256};
    let mut files = Vec::new();
    get_files_recursive(dir, &mut files)?;
    files.sort_by(|a, b| a.relative.cmp(&b.relative));

    let mut hasher = Sha256::new();
    for f in &files {
        hasher.update(f.relative.as_bytes());
        hasher.update(&f.content);
    }
    let digest = hasher.finalize();
    let hex: String = digest.iter().map(|b| format!("{:02x}", b)).collect();
    Ok(hex)
}

fn copy_dir_all(src: &Path, dst: &Path) -> Result<(), String> {
    std::fs::create_dir_all(dst)
        .map_err(|e| format!("failed to create directory '{}': {}", dst.display(), e))?;
    for entry in
        std::fs::read_dir(src).map_err(|e| format!("failed to read '{}': {}", src.display(), e))?
    {
        let entry = entry.map_err(|e| format!("entry error: {}", e))?;
        let path = entry.path();
        let file_name = path.file_name().unwrap();
        let dest_path = dst.join(file_name);
        if path.is_dir() {
            copy_dir_all(&path, &dest_path)?;
        } else {
            std::fs::copy(&path, &dest_path).map_err(|e| {
                format!(
                    "failed to copy from '{}' to '{}': {}",
                    path.display(),
                    dest_path.display(),
                    e
                )
            })?;
        }
    }
    Ok(())
}

pub fn find_highest_compatible_version(
    registry_root: &Path,
    pkg_name: &str,
    req: &VersionReq,
) -> Result<(Version, PathBuf), String> {
    let pkg_dir = registry_root.join(pkg_name);
    if !pkg_dir.exists() {
        return Err(format!(
            "package '{}' matching version requirement '{}' was not found in the workspace \
             registry '{}'. Dependencies in this build resolve from an explicit `path` or that \
             workspace registry; toolchain-supplied packages are not available. Use a path \
             dependency such as \"{}\": {{ \"package\": \"{}\", \"version\": \"{}\", \
             \"path\": \"../{}\" }}",
            pkg_name,
            req_to_string(req),
            registry_root.display(),
            pkg_name,
            pkg_name,
            req_to_string(req),
            pkg_name,
        ));
    }

    let mut highest: Option<(Version, PathBuf)> = None;
    let entries = std::fs::read_dir(&pkg_dir).map_err(|e| {
        format!(
            "failed to read registry directory for '{}': {}",
            pkg_name, e
        )
    })?;

    for entry in entries {
        let entry = entry.map_err(|e| format!("failed to read entry: {}", e))?;
        let name_os = entry.file_name();
        let name_str = name_os.to_string_lossy();
        if let Ok(version) = Version::parse(&name_str) {
            if req.matches(&version) {
                let manifest_path = entry.path().join("starkpkg.json");
                if manifest_path.exists() {
                    if let Some((ref h_ver, _)) = highest {
                        if version > *h_ver {
                            highest = Some((version, manifest_path));
                        }
                    } else {
                        highest = Some((version, manifest_path));
                    }
                }
            }
        }
    }

    highest.ok_or_else(|| {
        format!(
            "no compatible version of '{}' found matching '{}'",
            pkg_name,
            req_to_string(req)
        )
    })
}

#[derive(Clone, Debug)]
pub struct PackageGraph {
    pub root_package_name: String,
    pub packages: HashMap<String, Package>,
    pub workspace_root: PathBuf,
}

impl PackageGraph {
    pub fn load_from_root_with_modes(
        root_manifest_path: &Path,
        locked: bool,
        offline: bool,
    ) -> Result<Self, String> {
        let current_exe = std::env::current_exe().ok();
        let toolchain_root = discover_toolchain_package_root(current_exe.as_deref());
        Self::load_from_root_with_modes_and_toolchain(
            root_manifest_path,
            locked,
            offline,
            toolchain_root.as_deref(),
        )
    }

    /// Explicit-root entry point for hermetic resolver and installed-layout qualification.
    #[doc(hidden)]
    pub fn load_from_root_with_modes_and_toolchain(
        root_manifest_path: &Path,
        locked: bool,
        offline: bool,
        toolchain_root: Option<&Path>,
    ) -> Result<Self, String> {
        // Dependency paths are canonicalized while parsing their manifests. Canonicalize the
        // graph root as well so workspace containment compares paths in the same representation.
        // This is required on Windows, where canonicalization adds the `\\?\` prefix.
        let root_manifest_path = root_manifest_path.canonicalize().map_err(|error| {
            format!(
                "failed to read manifest at '{}': {}",
                root_manifest_path.display(),
                error
            )
        })?;
        let root_package = Package::from_manifest(&root_manifest_path)?;
        let workspace_root = get_workspace_root(&root_manifest_path);

        if !is_within_workspace(&root_manifest_path, &workspace_root) {
            return Err("root package is outside the permitted workspace".to_string());
        }

        let lock_path = root_manifest_path.parent().unwrap().join("stark.lock");
        let existing_lock = if lock_path.exists() {
            let lock_content = std::fs::read_to_string(&lock_path)
                .map_err(|e| format!("failed to read lockfile: {}", e))?;
            Some(Lockfile::parse(&lock_content)?)
        } else {
            None
        };

        // If locked mode, fail if lockfile is missing
        if locked && existing_lock.is_none() {
            return Err(
                "lockfile out of sync: stark.lock must be updated but --locked was passed"
                    .to_string(),
            );
        }

        let mut packages = HashMap::new();
        let root_name = root_package.name.clone();
        packages.insert(root_name.clone(), root_package);

        let mut graph = Self {
            root_package_name: root_name.clone(),
            packages,
            workspace_root,
        };

        let registry_dir = graph.workspace_root.join("tmp/stark_registry");
        let cache_dir = graph.workspace_root.join("tmp/stark_cache");

        let mut resolved_packages = HashMap::new();
        let mut shadowed_toolchain_packages = std::collections::HashSet::new();
        graph.resolve_dependencies_for(
            &root_name,
            &mut Vec::new(),
            locked,
            offline,
            &registry_dir,
            &cache_dir,
            toolchain_root,
            existing_lock.as_ref(),
            &mut resolved_packages,
            &mut shadowed_toolchain_packages,
        )?;

        graph.enforce_root_capability_envelope()?;

        // If not in locked mode, write the updated lockfile
        if !locked {
            let mut lock_pkgs = HashMap::new();
            for (pkg_name, pkg) in &graph.packages {
                if pkg_name == &graph.root_package_name {
                    continue;
                }

                let (source, sha256) = if let Some(resolved_meta) = resolved_packages.get(pkg_name)
                {
                    (
                        Some(resolved_meta.source.clone()),
                        resolved_meta.sha256.clone(),
                    )
                } else {
                    let directory = pkg.manifest_path.parent().ok_or_else(|| {
                        format!(
                            "manifest '{}' has no package directory",
                            pkg.manifest_path.display()
                        )
                    })?;
                    if is_within_workspace(directory, &graph.workspace_root) {
                        (None, String::new())
                    } else {
                        (
                            Some(format!("path:{}", directory.display())),
                            calculate_dir_sha256(directory)?,
                        )
                    }
                };

                let mut dependencies = HashMap::new();
                for (d_name, dependency) in &pkg.dependencies {
                    let d_ver = match &dependency.source {
                        DependencySource::Path(p) => {
                            let p_manifest = p.join("starkpkg.json");
                            let p_pkg = Package::from_manifest(&p_manifest)?;
                            format!(
                                "{}.{}.{}",
                                p_pkg.version.major, p_pkg.version.minor, p_pkg.version.patch
                            )
                        }
                        DependencySource::Registry(_) => {
                            let dep_pkg = graph.packages.get(d_name).ok_or_else(|| {
                                format!("missing resolved dependency '{}'", d_name)
                            })?;
                            format!(
                                "{}.{}.{}",
                                dep_pkg.version.major, dep_pkg.version.minor, dep_pkg.version.patch
                            )
                        }
                    };
                    dependencies.insert(d_name.clone(), d_ver);
                }

                lock_pkgs.insert(
                    pkg_name.clone(),
                    LockfilePackage {
                        name: pkg_name.clone(),
                        version: pkg.version.clone(),
                        source,
                        sha256,
                        dependencies,
                    },
                );
            }
            let new_lock = Lockfile {
                capability_vocabulary: 1,
                packages: lock_pkgs,
            };

            // Check if updated lock differs from existing lock when --locked is passed
            if let Some(ref old_lock) = existing_lock {
                if new_lock.serialize() != old_lock.serialize() && locked {
                    return Err(
                        "lockfile out of sync: stark.lock must be updated but --locked was passed"
                            .to_string(),
                    );
                }
            }

            std::fs::write(&lock_path, new_lock.serialize())
                .map_err(|e| format!("failed to write lockfile: {}", e))?;
        }

        Ok(graph)
    }

    #[allow(clippy::too_many_arguments)]
    fn resolve_dependencies_for(
        &mut self,
        package_name: &str,
        visit_stack: &mut Vec<String>,
        locked: bool,
        offline: bool,
        registry_dir: &Path,
        cache_dir: &Path,
        toolchain_root: Option<&Path>,
        existing_lock: Option<&Lockfile>,
        resolved_packages: &mut HashMap<String, ResolvedMeta>,
        shadowed_toolchain_packages: &mut std::collections::HashSet<String>,
    ) -> Result<(), String> {
        visit_stack.push(package_name.to_string());

        let package = self.packages.get(package_name).unwrap().clone();
        for (dep_alias, dependency) in &package.dependencies {
            if let Some(pos) = visit_stack.iter().position(|x| x == dep_alias) {
                let cycle = visit_stack[pos..].to_vec();
                return Err(format!(
                    "dependency cycle detected: {} -> {}",
                    cycle.join(" -> "),
                    dep_alias
                ));
            }

            match &dependency.source {
                DependencySource::Path(dep_dir) => {
                    let dep_manifest = dep_dir.join("starkpkg.json");
                    if let Some(existing) = self.packages.get(dep_alias) {
                        if existing.manifest_path != dep_manifest {
                            return Err(format!(
                                "duplicate package name '{}': both '{}' and '{}' exist",
                                dep_alias,
                                existing.manifest_path.display(),
                                dep_manifest.display()
                            ));
                        }
                        continue;
                    }

                    if !dep_manifest.exists() {
                        return Err(format!(
                            "missing manifest: dependency '{}' requires '{}' to exist",
                            dep_alias,
                            dep_manifest.display()
                        ));
                    }
                    let dep_pkg = Package::from_manifest(&dep_manifest)?;
                    if dep_pkg.name != dependency.package {
                        return Err(format!(
                            "package name mismatch: dependency config expects '{}', but manifest defines '{}'",
                            dependency.package, dep_pkg.name
                        ));
                    }

                    if resolved_packages
                        .get(package_name)
                        .is_some_and(|meta| meta.source == "toolchain")
                    {
                        let directory = dep_manifest.parent().unwrap();
                        let sha256 = calculate_dir_sha256(directory)?;
                        if let Some(lock_pkg) =
                            existing_lock.and_then(|lock| lock.packages.get(dep_alias))
                        {
                            if locked
                                && (lock_pkg.source.as_deref() != Some("toolchain")
                                    || lock_pkg.sha256 != sha256)
                            {
                                return Err(format!(
                                    "lockfile out of sync for toolchain package '{}': source or content hash changed",
                                    dependency.package
                                ));
                            }
                        }
                        resolved_packages.insert(
                            dep_alias.clone(),
                            ResolvedMeta {
                                source: "toolchain".to_string(),
                                sha256,
                            },
                        );
                    }

                    self.packages.insert(dep_alias.clone(), dep_pkg);
                    self.resolve_dependencies_for(
                        dep_alias,
                        visit_stack,
                        locked,
                        offline,
                        registry_dir,
                        cache_dir,
                        toolchain_root,
                        existing_lock,
                        resolved_packages,
                        shadowed_toolchain_packages,
                    )?;
                }
                DependencySource::Registry(req) => {
                    let locked_package = existing_lock
                        .and_then(|lock| lock.packages.get(dep_alias))
                        .filter(|package| req.matches(&package.version));
                    if locked && locked_package.is_none() {
                        return Err("lockfile out of sync: stark.lock must be updated but --locked was passed".to_string());
                    }
                    if locked
                        && locked_package.is_some_and(|package| {
                            !matches!(
                                package.source.as_deref(),
                                None | Some("registry") | Some("toolchain")
                            )
                        })
                    {
                        return Err(format!(
                            "lockfile out of sync: version dependency '{}' has incompatible source",
                            dependency.package
                        ));
                    }

                    let toolchain_manifest = toolchain_root
                        .map(|root| root.join(&dependency.package).join("starkpkg.json"))
                        .filter(|path| path.is_file());
                    let toolchain_package = toolchain_manifest
                        .as_ref()
                        .map(|path| Package::from_manifest(path))
                        .transpose()?;

                    let locked_to_toolchain = locked_package
                        .is_some_and(|package| package.source.as_deref() == Some("toolchain"));
                    let registry_version = if locked_package.is_some() && !locked_to_toolchain {
                        locked_package.map(|package| package.version.clone())
                    } else if locked_package.is_none() {
                        find_highest_compatible_version(registry_dir, &dependency.package, req)
                            .ok()
                            .map(|(version, _)| version)
                    } else {
                        None
                    };

                    let (version, expected_sha, source, package_dir) = if locked_to_toolchain {
                        let lock_package = locked_package.unwrap();
                        let package = toolchain_package.as_ref().ok_or_else(|| {
                            let version = format!(
                                "{}.{}.{}",
                                lock_package.version.major,
                                lock_package.version.minor,
                                lock_package.version.patch
                            );
                            let root = toolchain_root
                                .map(|root| root.display().to_string())
                                .unwrap_or_else(|| "<undiscovered>".to_string());
                            format!(
                                "locked toolchain package '{} {}' is missing from '{}'",
                                dependency.package, version, root
                            )
                        })?;
                        if package.version != lock_package.version {
                            let version = format!(
                                "{}.{}.{}",
                                lock_package.version.major,
                                lock_package.version.minor,
                                lock_package.version.patch
                            );
                            return Err(format!(
                                "locked toolchain package '{}' requires version '{}', but this toolchain carries '{}'",
                                dependency.package,
                                version,
                                package.version_str()
                            ));
                        }
                        (
                            package.version.clone(),
                            Some(lock_package.sha256.clone()),
                            "toolchain",
                            package.manifest_path.parent().unwrap().to_path_buf(),
                        )
                    } else if let Some(version) = registry_version {
                        if toolchain_package.is_some()
                            && locked_package.is_none()
                            && shadowed_toolchain_packages.insert(dependency.package.clone())
                        {
                            eprintln!(
                                "warning: workspace registry package '{}' shadows the toolchain package",
                                dependency.package
                            );
                        }
                        let ver_str =
                            format!("{}.{}.{}", version.major, version.minor, version.patch);
                        let cached = cache_dir.join(&dependency.package).join(&ver_str);
                        if !cached.exists() {
                            if offline {
                                return Err(format!(
                                    "offline mode: cached package '{} {}' is not available in '{}'",
                                    dependency.package,
                                    ver_str,
                                    cached.display()
                                ));
                            }
                            let registry = registry_dir.join(&dependency.package).join(&ver_str);
                            if !registry.is_dir() {
                                return Err(format!(
                                    "package '{} {}' not found in registry '{}'",
                                    dependency.package,
                                    ver_str,
                                    registry.display()
                                ));
                            }
                            copy_dir_all(&registry, &cached)?;
                        }
                        (
                            version,
                            locked_package.map(|package| package.sha256.clone()),
                            "registry",
                            cached,
                        )
                    } else if let Some(package) = toolchain_package.as_ref() {
                        if !req.matches(&package.version) {
                            return Err(format!(
                                "package '{}' requests version '{}', but toolchain root '{}' carries incompatible version '{}'",
                                dependency.package,
                                req_to_string(req),
                                toolchain_root.unwrap().display(),
                                package.version_str()
                            ));
                        }
                        (
                            package.version.clone(),
                            None,
                            "toolchain",
                            package.manifest_path.parent().unwrap().to_path_buf(),
                        )
                    } else {
                        let outcome = if registry_dir.join(&dependency.package).is_dir() {
                            format!(
                                "no compatible version of '{}' found matching '{}'",
                                dependency.package,
                                req_to_string(req)
                            )
                        } else {
                            format!(
                                "package '{}' matching version requirement '{}' was not found",
                                dependency.package,
                                req_to_string(req)
                            )
                        };
                        return Err(format!(
                            "{} in workspace registry '{}'{}; use an explicit `path` dependency such as \"{}\": {{ \"package\": \"{}\", \"version\": \"{}\", \"path\": \"../{}\" }}",
                            outcome,
                            registry_dir.display(),
                            toolchain_root.map(|root| format!(" or toolchain root '{}'", root.display())).unwrap_or_else(|| "; this build has no discovered toolchain package root".to_string()),
                            dep_alias,
                            dependency.package,
                            req_to_string(req),
                            dependency.package,
                        ));
                    };
                    let ver_str = format!("{}.{}.{}", version.major, version.minor, version.patch);
                    let sha256 = calculate_dir_sha256(&package_dir)?;
                    if let Some(ref exp_sha) = expected_sha {
                        if sha256 != *exp_sha {
                            return Err(format!(
                                "content hash mismatch for {} package '{} {}': expected '{}', found '{}'",
                                source, dependency.package, ver_str, exp_sha, sha256
                            ));
                        }
                    }

                    let dep_manifest = package_dir.join("starkpkg.json");
                    if let Some(existing) = self.packages.get(dep_alias) {
                        if existing.version != version {
                            return Err(format!(
                                "duplicate package name '{}' with conflicting versions: resolved both '{}' and '{}'",
                                dep_alias, existing.version_str(), ver_str
                            ));
                        }
                        if existing.manifest_path != dep_manifest {
                            return Err(format!(
                                "duplicate package name '{}' resolved to different paths",
                                dep_alias
                            ));
                        }
                        continue;
                    }

                    let dep_pkg = Package::from_manifest(&dep_manifest)?;
                    if dep_pkg.name != dependency.package {
                        return Err(format!(
                            "package name mismatch: dependency config expects '{}', but manifest defines '{}'",
                            dependency.package, dep_pkg.name
                        ));
                    }
                    self.packages.insert(dep_alias.clone(), dep_pkg);
                    resolved_packages.insert(
                        dep_alias.clone(),
                        ResolvedMeta {
                            source: source.to_string(),
                            sha256,
                        },
                    );

                    self.resolve_dependencies_for(
                        dep_alias,
                        visit_stack,
                        locked,
                        offline,
                        registry_dir,
                        cache_dir,
                        toolchain_root,
                        existing_lock,
                        resolved_packages,
                        shadowed_toolchain_packages,
                    )?;
                }
            }
        }

        visit_stack.pop();
        Ok(())
    }

    pub fn load_from_root(root_manifest_path: &Path) -> Result<Self, String> {
        Self::load_from_root_with_modes(root_manifest_path, false, false)
    }

    /// WP-P1.6: the root application approves the conservative transitive capability closure.
    /// Provider bindings are the current compiler-emitted host-interface references: every bound
    /// function/resource contributes, regardless of reachability, and the diagnostic preserves
    /// the package plus exact interface path that introduced the capability.
    fn enforce_root_capability_envelope(&self) -> Result<(), String> {
        let root = self
            .packages
            .get(&self.root_package_name)
            .ok_or("package graph has no root package")?;
        let mut contributors: Vec<(&str, &str, String)> = Vec::new();
        for (alias, package) in &self.packages {
            for binding in &package.provider_api.functions {
                contributors.push((
                    binding.capability.as_str(),
                    alias.as_str(),
                    format!("provider_api.functions.{}", binding.item_path),
                ));
            }
            for binding in &package.provider_api.resources {
                contributors.push((
                    binding.capability.as_str(),
                    alias.as_str(),
                    format!("provider_api.resources.{}", binding.nominal),
                ));
            }
        }
        contributors.sort();
        contributors.dedup();

        let missing: Vec<_> = contributors
            .into_iter()
            .filter(|(capability, _, _)| {
                !root
                    .capabilities
                    .iter()
                    .any(|declared| declared == capability)
            })
            .collect();
        if missing.is_empty() {
            return Ok(());
        }
        let details = missing
            .iter()
            .map(|(capability, package, interface)| {
                format!(
                    "  capability '{capability}' derived by package '{package}' from interface reference '{interface}'"
                )
            })
            .collect::<Vec<_>>()
            .join("\n");
        Err(format!(
            "root package '{}' does not declare the complete transitive capability envelope:\n{details}\nadd the named capabilities to its \"capabilities\" array (capability vocabulary v1)",
            self.root_package_name
        ))
    }
}

impl Package {
    pub fn version_str(&self) -> String {
        format!(
            "{}.{}.{}",
            self.version.major, self.version.minor, self.version.patch
        )
    }
}

struct ResolvedMeta {
    source: String,
    sha256: String,
}

/// WP-C7.8.8 step 1: parses and validates `provider_api`.
///
/// Everything checkable **without** provider metadata happens here, at manifest load. The checks
/// that need a selected provider — symbol exists, resource exists, no close bound, derived
/// signature resolvable — happen at provider selection, because they need metadata this function
/// does not have.
fn parse_provider_api(
    obj: &std::collections::HashMap<String, JsonValue>,
    capabilities: &[String],
    dependencies: &HashMap<String, Dependency>,
    path: &Path,
) -> Result<ProviderApi, String> {
    let Some(api_val) = obj.get("provider_api") else {
        return Ok(ProviderApi::default());
    };
    let api = api_val.as_object().ok_or_else(|| {
        format!(
            "'provider_api' in manifest '{}' must be a JSON object",
            path.display()
        )
    })?;

    // A binding is a USE of a capability, so Packet 5's admission rule applies: it must have been
    // declared. Without this, a package could reach a provider it never required.
    let declared = |cap: &str, what: &str| -> Result<(), String> {
        if capabilities.iter().any(|c| c == cap) {
            return Ok(());
        }
        Err(format!(
            "{what} in manifest '{}' binds capability '{cap}', which the package does not declare; \
             add it to \"capabilities\"",
            path.display()
        ))
    };

    let mut functions = Vec::new();
    if let Some(fns_val) = api.get("functions") {
        let fns = fns_val.as_object().ok_or_else(|| {
            format!(
                "'provider_api.functions' in manifest '{}' must be a JSON object",
                path.display()
            )
        })?;
        for (item_path, spec) in fns {
            let (capability, symbol) =
                binding_pair(spec, "symbol", item_path, "provider_api.functions", path)?;
            declared(&capability, &format!("function binding '{item_path}'"))?;
            functions.push(ProviderFunctionBinding {
                item_path: item_path.clone(),
                capability,
                symbol,
            });
        }
    }

    let mut resources = Vec::new();
    if let Some(res_val) = api.get("resources") {
        let res = res_val.as_object().ok_or_else(|| {
            format!(
                "'provider_api.resources' in manifest '{}' must be a JSON object",
                path.display()
            )
        })?;
        for (nominal, spec) in res {
            let (capability, resource) =
                binding_pair(spec, "resource", nominal, "provider_api.resources", path)?;
            declared(&capability, &format!("resource binding '{nominal}'"))?;
            // CD-224: Core resources are compiler-owned. A package declaring `file` would be
            // claiming authority over a Core type, which is exactly what the two-mechanism ruling
            // forbids -- so it is rejected here rather than silently shadowing the built-in.
            if crate::provider_bind::ResourceRegistry::builtin()
                .lookup(&resource)
                .is_some()
            {
                return Err(format!(
                    "manifest '{}' binds resource '{resource}', which is a Core resource owned by \
                     the compiler; a package may not declare it",
                    path.display()
                ));
            }
            resources.push(ProviderResourceBinding {
                nominal: nominal.clone(),
                capability,
                resource,
            });
        }
    }

    // HC9: resources another package owns, which this package's bindings may NAME. Deliberately a
    // separate section from `resources`: an entry here synthesizes no nominal, because the nominal
    // already exists in the owning package and a second one would be a second type.
    let mut foreign_resources = Vec::new();
    if let Some(foreign_val) = api.get("foreign_resources") {
        let foreign = foreign_val.as_object().ok_or_else(|| {
            format!(
                "'provider_api.foreign_resources' in manifest '{}' must be a JSON object",
                path.display()
            )
        })?;
        for (resource, spec) in foreign {
            let spec_obj = spec.as_object().ok_or_else(|| {
                format!(
                    "'provider_api.foreign_resources.{resource}' in manifest '{}' must be a JSON \
                     object",
                    path.display()
                )
            })?;
            let field = |key: &str| -> Result<String, String> {
                spec_obj
                    .get(key)
                    .and_then(|v| v.as_str())
                    .map(str::to_string)
                    .ok_or_else(|| {
                        format!(
                            "'provider_api.foreign_resources.{resource}' in manifest '{}' must \
                             have a string '{key}'",
                            path.display()
                        )
                    })
            };
            let package = field("package")?;
            let nominal = field("nominal")?;
            // The alias must be one the package actually depends on. Otherwise the derived
            // signature renders a path to a package that is not in the graph, and the failure
            // surfaces as an unresolved name inside generated source nobody wrote.
            if !dependencies.contains_key(&package) {
                return Err(format!(
                    "manifest '{}' declares foreign resource '{resource}' in package '{package}', \
                     which is not a dependency; add it to \"dependencies\" under that alias",
                    path.display()
                ));
            }
            // A Core resource is compiler-owned (CD-224), and no package owns one to lend.
            if crate::provider_bind::ResourceRegistry::builtin()
                .lookup(resource)
                .is_some()
            {
                return Err(format!(
                    "manifest '{}' declares foreign resource '{resource}', which is a Core \
                     resource owned by the compiler; it is not another package's to lend",
                    path.display()
                ));
            }
            foreign_resources.push(ProviderForeignResourceBinding {
                resource: resource.clone(),
                package,
                nominal,
            });
        }
    }

    let mut errors = Vec::new();
    if let Some(err_val) = api.get("errors") {
        let errs = err_val.as_object().ok_or_else(|| {
            format!(
                "'provider_api.errors' in manifest '{}' must be a JSON object",
                path.display()
            )
        })?;
        for (capability, ty) in errs {
            let ty = ty.as_str().ok_or_else(|| {
                format!(
                    "'provider_api.errors.{capability}' in manifest '{}' must be a string",
                    path.display()
                )
            })?;
            declared(capability, &format!("error binding for '{capability}'"))?;
            errors.push((capability.clone(), ty.to_string()));
        }
    }

    // Two nominals bound to one resource is rejected, not warned (design §13.3): they would be
    // distinct STARK types that are identical at the boundary, so one would satisfy the other
    // dynamically while failing statically, and each would record its own close for one resource --
    // breaking exactly-once.
    let mut by_resource: HashMap<&str, Vec<&str>> = HashMap::new();
    for r in &resources {
        by_resource
            .entry(r.resource.as_str())
            .or_default()
            .push(r.nominal.as_str());
    }
    let mut collisions: Vec<String> = by_resource
        .iter()
        .filter(|(_, noms)| noms.len() > 1)
        .map(|(res, noms)| {
            let mut sorted = noms.clone();
            sorted.sort();
            format!("'{res}' is bound by {}", sorted.join(", "))
        })
        .collect();
    collisions.sort();
    if let Some(first) = collisions.first() {
        return Err(format!(
            "manifest '{}' binds one provider resource to several nominals: {first}",
            path.display()
        ));
    }

    // Every capability with a bound function needs a raw error type: the derived signature is
    // `Result<_, E>` and there is no E without it.
    for f in &functions {
        if !errors.iter().any(|(cap, _)| cap == &f.capability) {
            return Err(format!(
                "manifest '{}' binds '{}' for capability '{}' but declares no \
                 'provider_api.errors' entry for it",
                path.display(),
                f.item_path,
                f.capability
            ));
        }
    }

    // HC9: a resource cannot be both owned and foreign. Owning it synthesizes a nominal; borrowing
    // it references someone else's. Declaring both would put two nominals behind one resource name
    // -- the same failure the collision check above rejects, arriving by a different route, and
    // the one that would silently produce a handle the program cannot pass anywhere.
    let mut both: Vec<&str> = foreign_resources
        .iter()
        .filter(|f| resources.iter().any(|r| r.resource == f.resource))
        .map(|f| f.resource.as_str())
        .collect();
    both.sort_unstable();
    if let Some(first) = both.first() {
        return Err(format!(
            "manifest '{}' declares resource '{first}' as both owned and foreign; a package either \
             binds a resource's nominal or references the owner's, never both",
            path.display()
        ));
    }

    // Sorted so manifest key order reaches neither the build key nor generated code -- the property
    // CD-213 gave capabilities and CD-205 gave the status vocabulary.
    functions.sort_by(|a, b| a.item_path.cmp(&b.item_path));
    resources.sort_by(|a, b| a.nominal.cmp(&b.nominal));
    foreign_resources.sort_by(|a, b| a.resource.cmp(&b.resource));
    errors.sort();
    Ok(ProviderApi {
        functions,
        resources,
        errors,
        foreign_resources,
    })
}

/// One `{ "capability": ..., "<key>": ... }` binding entry.
fn binding_pair(
    spec: &JsonValue,
    key: &str,
    name: &str,
    section: &str,
    path: &Path,
) -> Result<(String, String), String> {
    let obj = spec.as_object().ok_or_else(|| {
        format!(
            "'{section}.{name}' in manifest '{}' must be a JSON object",
            path.display()
        )
    })?;
    let field = |k: &str| -> Result<String, String> {
        obj.get(k)
            .and_then(|v| v.as_str())
            .filter(|v| !v.is_empty())
            .map(str::to_string)
            .ok_or_else(|| {
                format!(
                    "'{section}.{name}' in manifest '{}' needs a non-empty string '{k}'",
                    path.display()
                )
            })
    };
    Ok((field("capability")?, field(key)?))
}
