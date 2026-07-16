use std::collections::HashMap;
use std::path::{Path, PathBuf};

#[derive(Clone, Debug, PartialEq)]
pub enum JsonValue {
    Null,
    Bool(bool),
    Number(f64),
    String(String),
    Array(Vec<JsonValue>),
    Object(std::collections::HashMap<String, JsonValue>),
}

impl JsonValue {
    pub fn as_str(&self) -> Option<&str> {
        match self {
            JsonValue::String(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_object(&self) -> Option<&std::collections::HashMap<String, JsonValue>> {
        match self {
            JsonValue::Object(o) => Some(o),
            _ => None,
        }
    }
}

pub fn parse_json(input: &str) -> Result<JsonValue, String> {
    let chars: Vec<char> = input.chars().collect();
    let mut pos = 0;

    fn skip_whitespace(chars: &[char], pos: &mut usize) {
        while *pos < chars.len() && chars[*pos].is_whitespace() {
            *pos += 1;
        }
    }

    fn parse_value(chars: &[char], pos: &mut usize) -> Result<JsonValue, String> {
        skip_whitespace(chars, pos);
        if *pos >= chars.len() {
            return Err("Unexpected EOF".to_string());
        }
        match chars[*pos] {
            '{' => parse_object(chars, pos),
            '[' => parse_array(chars, pos),
            '"' => parse_string(chars, pos),
            't' | 'f' => parse_bool(chars, pos),
            'n' => parse_null(chars, pos),
            '-' | '0'..='9' => parse_number(chars, pos),
            c => Err(format!("Unexpected character: {}", c)),
        }
    }

    fn parse_object(chars: &[char], pos: &mut usize) -> Result<JsonValue, String> {
        *pos += 1;
        let mut map = std::collections::HashMap::new();
        loop {
            skip_whitespace(chars, pos);
            if *pos >= chars.len() {
                return Err("Unterminated object".to_string());
            }
            if chars[*pos] == '}' {
                *pos += 1;
                break;
            }
            if chars[*pos] != '"' {
                return Err("Expected string key in object".to_string());
            }
            let key = match parse_string(chars, pos)? {
                JsonValue::String(s) => s,
                _ => unreachable!(),
            };
            skip_whitespace(chars, pos);
            if *pos >= chars.len() || chars[*pos] != ':' {
                return Err("Expected ':' after key in object".to_string());
            }
            *pos += 1;
            let val = parse_value(chars, pos)?;
            map.insert(key, val);
            skip_whitespace(chars, pos);
            if *pos >= chars.len() {
                return Err("Unterminated object".to_string());
            }
            if chars[*pos] == '}' {
                *pos += 1;
                break;
            }
            if chars[*pos] != ',' {
                return Err(format!("Expected ',' or '}}' in object, got '{}'", chars[*pos]));
            }
            *pos += 1;
        }
        Ok(JsonValue::Object(map))
    }

    fn parse_array(chars: &[char], pos: &mut usize) -> Result<JsonValue, String> {
        *pos += 1;
        let mut list = Vec::new();
        loop {
            skip_whitespace(chars, pos);
            if *pos >= chars.len() {
                return Err("Unterminated array".to_string());
            }
            if chars[*pos] == ']' {
                *pos += 1;
                break;
            }
            let val = parse_value(chars, pos)?;
            list.push(val);
            skip_whitespace(chars, pos);
            if *pos >= chars.len() {
                return Err("Unterminated array".to_string());
            }
            if chars[*pos] == ']' {
                *pos += 1;
                break;
            }
            if chars[*pos] != ',' {
                return Err("Expected ',' or ']' in array".to_string());
            }
            *pos += 1;
        }
        Ok(JsonValue::Array(list))
    }

    fn parse_string(chars: &[char], pos: &mut usize) -> Result<JsonValue, String> {
        *pos += 1;
        let mut s = String::new();
        while *pos < chars.len() {
            let c = chars[*pos];
            if c == '"' {
                *pos += 1;
                return Ok(JsonValue::String(s));
            }
            if c == '\\' {
                *pos += 1;
                if *pos >= chars.len() {
                    return Err("Unterminated escape sequence in string".to_string());
                }
                let esc = chars[*pos];
                match esc {
                    '"' => s.push('"'),
                    '\\' => s.push('\\'),
                    '/' => s.push('/'),
                    'b' => s.push('\x08'),
                    'f' => s.push('\x0c'),
                    'n' => s.push('\n'),
                    'r' => s.push('\r'),
                    't' => s.push('\t'),
                    _ => return Err(format!("Unsupported escape: \\{}", esc)),
                }
            } else {
                s.push(c);
            }
            *pos += 1;
        }
        Err("Unterminated string".to_string())
    }

    fn parse_bool(chars: &[char], pos: &mut usize) -> Result<JsonValue, String> {
        if *pos + 4 <= chars.len() && chars[*pos..*pos+4] == ['t', 'r', 'u', 'e'] {
            *pos += 4;
            Ok(JsonValue::Bool(true))
        } else if *pos + 5 <= chars.len() && chars[*pos..*pos+5] == ['f', 'a', 'l', 's', 'e'] {
            *pos += 5;
            Ok(JsonValue::Bool(false))
        } else {
            Err("Expected boolean value".to_string())
        }
    }

    fn parse_null(chars: &[char], pos: &mut usize) -> Result<JsonValue, String> {
        if *pos + 4 <= chars.len() && chars[*pos..*pos+4] == ['n', 'u', 'l', 'l'] {
            *pos += 4;
            Ok(JsonValue::Null)
        } else {
            Err("Expected null value".to_string())
        }
    }

    fn parse_number(chars: &[char], pos: &mut usize) -> Result<JsonValue, String> {
        let start = *pos;
        if *pos < chars.len() && chars[*pos] == '-' {
            *pos += 1;
        }
        while *pos < chars.len() && (chars[*pos].is_ascii_digit() || chars[*pos] == '.') {
            *pos += 1;
        }
        let num_str: String = chars[start..*pos].iter().collect();
        match num_str.parse::<f64>() {
            Ok(n) => Ok(JsonValue::Number(n)),
            Err(e) => Err(format!("Invalid number '{}': {}", num_str, e)),
        }
    }

    let val = parse_value(&chars, &mut pos)?;
    skip_whitespace(&chars, &mut pos);
    if pos < chars.len() {
        return Err("Trailing characters after JSON value".to_string());
    }
    Ok(val)
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
        let major = parts[0].parse::<u64>().map_err(|_| format!("invalid major in '{}'", s))?;
        let minor = parts[1].parse::<u64>().map_err(|_| format!("invalid minor in '{}'", s))?;
        let patch = parts[2].parse::<u64>().map_err(|_| format!("invalid patch in '{}'", s))?;
        Ok(Self { major, minor, patch })
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
        if s.starts_with('^') {
            let v = Version::parse(&s[1..])?;
            return Ok(VersionReq::Caret(v));
        }
        
        let parts: Vec<&str> = s.split(',').collect();
        let mut comparators = Vec::new();
        for part in parts {
            let part = part.trim();
            if part.starts_with(">=") {
                comparators.push(Comparator::Ge(Version::parse(&part[2..].trim())?));
            } else if part.starts_with("<=") {
                comparators.push(Comparator::Le(Version::parse(&part[2..].trim())?));
            } else if part.starts_with('>') {
                comparators.push(Comparator::Gt(Version::parse(&part[1..].trim())?));
            } else if part.starts_with('<') {
                comparators.push(Comparator::Lt(Version::parse(&part[1..].trim())?));
            } else if part.starts_with('=') {
                comparators.push(Comparator::Eq(Version::parse(&part[1..].trim())?));
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
            VersionReq::Range(comparators) => {
                comparators.iter().all(|c| c.matches(version))
            }
        }
    }
}

pub fn req_to_string(req: &VersionReq) -> String {
    match req {
        VersionReq::Any => "*".to_string(),
        VersionReq::Exact(v) => format!("={}.{}.{}", v.major, v.minor, v.patch),
        VersionReq::Caret(v) => format!("^{}.{}.{}", v.major, v.minor, v.patch),
        VersionReq::Range(comparators) => {
            comparators.iter().map(|c| match c {
                Comparator::Ge(v) => format!(">={}.{}.{}", v.major, v.minor, v.patch),
                Comparator::Le(v) => format!("<={}.{}.{}", v.major, v.minor, v.patch),
                Comparator::Gt(v) => format!(">{}.{}.{}", v.major, v.minor, v.patch),
                Comparator::Lt(v) => format!("<{}.{}.{}", v.major, v.minor, v.patch),
                Comparator::Eq(v) => format!("={}.{}.{}", v.major, v.minor, v.patch),
            }).collect::<Vec<_>>().join(", ")
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum DependencySource {
    Path(PathBuf),
    Registry(VersionReq),
}

#[derive(Clone, Debug)]
pub struct Package {
    pub name: String,
    pub version: Version,
    pub entry: PathBuf,
    pub manifest_path: PathBuf,
    pub dependencies: HashMap<String, DependencySource>,
}

impl Package {
    pub fn from_manifest(path: &Path) -> Result<Self, String> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| format!("failed to read manifest at '{}': {}", path.display(), e))?;
        
        let json = parse_json(&content)
            .map_err(|e| format!("failed to parse manifest at '{}': {}", path.display(), e))?;
        
        let obj = json.as_object()
            .ok_or_else(|| format!("manifest at '{}' must be a JSON object", path.display()))?;
        
        let name = obj.get("name")
            .ok_or_else(|| format!("missing 'name' in manifest '{}'", path.display()))?
            .as_str()
            .ok_or_else(|| format!("'name' in manifest '{}' must be a string", path.display()))?
            .to_string();
            
        if name.is_empty() || !name.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_') {
            return Err(format!("invalid package name '{}' in manifest '{}'", name, path.display()));
        }

        let version_str = obj.get("version")
            .ok_or_else(|| format!("missing 'version' in manifest '{}'", path.display()))?
            .as_str()
            .ok_or_else(|| format!("'version' in manifest '{}' must be a string", path.display()))?;
        let version = Version::parse(version_str)
            .map_err(|e| format!("{} in manifest '{}'", e, path.display()))?;

        let entry_str = match obj.get("entry") {
            Some(v) => v.as_str()
                .ok_or_else(|| format!("'entry' in manifest '{}' must be a string", path.display()))?,
            None => "src/main.stark",
        };

        let parent_dir = path.parent().ok_or("manifest must have a parent directory")?;
        let entry = parent_dir.join(entry_str);
        
        let entry = entry.canonicalize()
            .map_err(|_| format!("entry file '{}' in manifest '{}' does not exist", entry_str, path.display()))?;

        let mut dependencies = HashMap::new();
        if let Some(deps_val) = obj.get("dependencies") {
            let deps_obj = deps_val.as_object()
                .ok_or_else(|| format!("'dependencies' in manifest '{}' must be a JSON object", path.display()))?;
            for (dep_name, dep_config_val) in deps_obj {
                let dep_config = dep_config_val.as_object()
                    .ok_or_else(|| format!("dependency config for '{}' in manifest '{}' must be a JSON object", dep_name, path.display()))?;
                
                let source = if let Some(dep_path_val) = dep_config.get("path") {
                    let dep_path_str = dep_path_val.as_str()
                        .ok_or_else(|| format!("'path' for dependency '{}' in manifest '{}' must be a string", dep_name, path.display()))?;
                    let dep_dir = parent_dir.join(dep_path_str);
                    let dep_dir = dep_dir.canonicalize()
                        .map_err(|_| format!("dependency path '{}' for '{}' in manifest '{}' does not exist", dep_path_str, dep_name, path.display()))?;
                    DependencySource::Path(dep_dir)
                } else if let Some(dep_ver_val) = dep_config.get("version") {
                    let dep_ver_str = dep_ver_val.as_str()
                        .ok_or_else(|| format!("'version' for dependency '{}' in manifest '{}' must be a string", dep_name, path.display()))?;
                    let req = VersionReq::parse(dep_ver_str)
                        .map_err(|e| format!("invalid version requirement '{}' for dependency '{}' in manifest '{}': {}", dep_ver_str, dep_name, path.display(), e))?;
                    DependencySource::Registry(req)
                } else {
                    return Err(format!("dependency '{}' in manifest '{}' must specify either 'path' or 'version'", dep_name, path.display()));
                };
                
                dependencies.insert(dep_name.clone(), source);
            }
        }

        Ok(Self {
            name,
            version,
            entry,
            manifest_path: path.to_path_buf(),
            dependencies,
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
    let mut current = start_dir.canonicalize()
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

#[derive(Clone, Debug)]
pub struct LockfilePackage {
    pub name: String,
    pub version: Version,
    pub sha256: String,
    pub dependencies: HashMap<String, String>,
}

#[derive(Clone, Debug)]
pub struct Lockfile {
    pub packages: HashMap<String, LockfilePackage>,
}

impl Lockfile {
    pub fn parse(content: &str) -> Result<Self, String> {
        let json = parse_json(content)?;
        let obj = json.as_object().ok_or("lockfile must be a JSON object")?;
        let pkgs_val = obj.get("packages").ok_or("missing 'packages' in lockfile")?;
        let pkgs_arr = match pkgs_val {
            JsonValue::Array(a) => a,
            _ => return Err("'packages' in lockfile must be an array".to_string()),
        };
        
        let mut packages = HashMap::new();
        for pkg_val in pkgs_arr {
            let pkg_obj = pkg_val.as_object().ok_or("package in lockfile must be a JSON object")?;
            let name = pkg_obj.get("name").ok_or("missing name")?.as_str().ok_or("name must be string")?.to_string();
            let ver_str = pkg_obj.get("version").ok_or("missing version")?.as_str().ok_or("version must be string")?;
            let version = Version::parse(ver_str)?;
            let sha256 = pkg_obj.get("sha256").ok_or("missing sha256")?.as_str().ok_or("sha256 must be string")?.to_string();
            
            let mut dependencies = HashMap::new();
            if let Some(deps_val) = pkg_obj.get("dependencies") {
                let deps_obj = deps_val.as_object().ok_or("dependencies must be object")?;
                for (d_name, d_ver_val) in deps_obj {
                    dependencies.insert(d_name.clone(), d_ver_val.as_str().ok_or("dependency version must be string")?.to_string());
                }
            }
            
            packages.insert(name.clone(), LockfilePackage {
                name,
                version,
                sha256,
                dependencies,
            });
        }
        Ok(Self { packages })
    }

    pub fn serialize(&self) -> String {
        let mut lines = Vec::new();
        lines.push("{".to_string());
        lines.push("  \"packages\": [".to_string());
        
        let mut sorted_packages: Vec<&LockfilePackage> = self.packages.values().collect();
        sorted_packages.sort_by(|a, b| a.name.cmp(&b.name));
        
        for (i, pkg) in sorted_packages.iter().enumerate() {
            let comma = if i + 1 < sorted_packages.len() { "," } else { "" };
            lines.push("    {".to_string());
            lines.push(format!("      \"name\": \"{}\",", pkg.name));
            lines.push(format!("      \"version\": \"{}.{}.{}\",", pkg.version.major, pkg.version.minor, pkg.version.patch));
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
    for entry in std::fs::read_dir(src).map_err(|e| format!("failed to read '{}': {}", src.display(), e))? {
        let entry = entry.map_err(|e| format!("entry error: {}", e))?;
        let path = entry.path();
        let file_name = path.file_name().unwrap();
        let dest_path = dst.join(file_name);
        if path.is_dir() {
            copy_dir_all(&path, &dest_path)?;
        } else {
            std::fs::copy(&path, &dest_path)
                .map_err(|e| format!("failed to copy from '{}' to '{}': {}", path.display(), dest_path.display(), e))?;
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
        return Err(format!("package '{}' not found in registry", pkg_name));
    }
    
    let mut highest: Option<(Version, PathBuf)> = None;
    let entries = std::fs::read_dir(&pkg_dir)
        .map_err(|e| format!("failed to read registry directory for '{}': {}", pkg_name, e))?;
    
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
    
    highest.ok_or_else(|| format!("no compatible version of '{}' found matching '{}'", pkg_name, req_to_string(req)))
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
        let root_package = Package::from_manifest(root_manifest_path)?;
        let workspace_root = get_workspace_root(root_manifest_path);

        if !is_within_workspace(root_manifest_path, &workspace_root) {
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
            return Err("lockfile out of sync: stark.lock must be updated but --locked was passed".to_string());
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
        graph.resolve_dependencies_for(
            &root_name,
            &mut Vec::new(),
            locked,
            offline,
            &registry_dir,
            &cache_dir,
            existing_lock.as_ref(),
            &mut resolved_packages,
        )?;

        // If not in locked mode, write the updated lockfile
        if !locked {
            let mut lock_pkgs = HashMap::new();
            for (pkg_name, pkg) in &graph.packages {
                if pkg_name == &graph.root_package_name {
                    continue;
                }
                
                let sha256 = if let Some(resolved_meta) = resolved_packages.get(pkg_name) {
                    resolved_meta.sha256.clone()
                } else {
                    "".to_string()
                };

                let mut dependencies = HashMap::new();
                for (d_name, d_src) in &pkg.dependencies {
                    let d_ver = match d_src {
                        DependencySource::Path(p) => {
                            let p_manifest = p.join("starkpkg.json");
                            let p_pkg = Package::from_manifest(&p_manifest)?;
                            format!("{}.{}.{}", p_pkg.version.major, p_pkg.version.minor, p_pkg.version.patch)
                        }
                        DependencySource::Registry(_) => {
                            let dep_pkg = graph.packages.get(d_name).ok_or_else(|| format!("missing resolved dependency '{}'", d_name))?;
                            format!("{}.{}.{}", dep_pkg.version.major, dep_pkg.version.minor, dep_pkg.version.patch)
                        }
                    };
                    dependencies.insert(d_name.clone(), d_ver);
                }

                lock_pkgs.insert(pkg_name.clone(), LockfilePackage {
                    name: pkg_name.clone(),
                    version: pkg.version.clone(),
                    sha256,
                    dependencies,
                });
            }
            let new_lock = Lockfile { packages: lock_pkgs };
            
            // Check if updated lock differs from existing lock when --locked is passed
            if let Some(ref old_lock) = existing_lock {
                if new_lock.serialize() != old_lock.serialize() && locked {
                    return Err("lockfile out of sync: stark.lock must be updated but --locked was passed".to_string());
                }
            }

            std::fs::write(&lock_path, new_lock.serialize())
                .map_err(|e| format!("failed to write lockfile: {}", e))?;
        }

        Ok(graph)
    }

    fn resolve_dependencies_for(
        &mut self,
        package_name: &str,
        visit_stack: &mut Vec<String>,
        locked: bool,
        offline: bool,
        registry_dir: &Path,
        cache_dir: &Path,
        existing_lock: Option<&Lockfile>,
        resolved_packages: &mut HashMap<String, ResolvedMeta>,
    ) -> Result<(), String> {
        visit_stack.push(package_name.to_string());

        let package = self.packages.get(package_name).unwrap().clone();
        for (dep_name, dep_source) in &package.dependencies {
            if let Some(pos) = visit_stack.iter().position(|x| x == dep_name) {
                let cycle = visit_stack[pos..].to_vec();
                return Err(format!(
                    "dependency cycle detected: {} -> {}",
                    cycle.join(" -> "), dep_name
                ));
            }

            match dep_source {
                DependencySource::Path(dep_dir) => {
                    let dep_manifest = dep_dir.join("starkpkg.json");
                    if !is_within_workspace(&dep_manifest, &self.workspace_root) {
                        return Err(format!(
                            "dependency '{}' of package '{}' resolves to '{}' which is outside the permitted workspace '{}'",
                            dep_name, package_name, dep_manifest.display(), self.workspace_root.display()
                        ));
                    }

                    if let Some(existing) = self.packages.get(dep_name) {
                        if existing.manifest_path != dep_manifest {
                            return Err(format!(
                                "duplicate package name '{}': both '{}' and '{}' exist",
                                dep_name, existing.manifest_path.display(), dep_manifest.display()
                            ));
                        }
                        continue;
                    }

                    if !dep_manifest.exists() {
                        return Err(format!(
                            "missing manifest: dependency '{}' requires '{}' to exist",
                            dep_name, dep_manifest.display()
                        ));
                    }
                    let dep_pkg = Package::from_manifest(&dep_manifest)?;
                    if dep_pkg.name != *dep_name {
                        return Err(format!(
                            "package name mismatch: dependency config expects '{}', but manifest defines '{}'",
                            dep_name, dep_pkg.name
                        ));
                    }

                    self.packages.insert(dep_name.clone(), dep_pkg);
                    self.resolve_dependencies_for(
                        dep_name,
                        visit_stack,
                        locked,
                        offline,
                        registry_dir,
                        cache_dir,
                        existing_lock,
                        resolved_packages,
                    )?;
                }
                DependencySource::Registry(req) => {
                    // Try to resolve version using existing lock file first
                    let resolved_version = if let Some(lock) = existing_lock {
                        if let Some(lock_pkg) = lock.packages.get(dep_name) {
                            if req.matches(&lock_pkg.version) {
                                Some((lock_pkg.version.clone(), lock_pkg.sha256.clone()))
                            } else {
                                if locked {
                                    return Err("lockfile out of sync: stark.lock must be updated but --locked was passed".to_string());
                                }
                                None
                            }
                        } else {
                            if locked {
                                return Err("lockfile out of sync: stark.lock must be updated but --locked was passed".to_string());
                            }
                            None
                        }
                    } else {
                        None
                    };

                    let (version, expected_sha) = if let Some((ver, sha)) = resolved_version {
                        (ver, Some(sha))
                    } else {
                        // Resolve from registry
                        let (ver, _) = find_highest_compatible_version(registry_dir, dep_name, req)?;
                        (ver, None)
                    };

                    let ver_str = format!("{}.{}.{}", version.major, version.minor, version.patch);
                    let cached_pkg_dir = cache_dir.join(dep_name).join(&ver_str);

                    // If not in cache, copy from registry (or fail if offline)
                    if !cached_pkg_dir.exists() {
                        if offline {
                            return Err(format!(
                                "offline mode: cached package '{} {}' is not available in '{}'",
                                dep_name, ver_str, cached_pkg_dir.display()
                            ));
                        }
                        
                        let reg_pkg_dir = registry_dir.join(dep_name).join(&ver_str);
                        if !reg_pkg_dir.exists() {
                            return Err(format!(
                                "package '{} {}' not found in registry '{}'",
                                dep_name, ver_str, reg_pkg_dir.display()
                            ));
                        }

                        copy_dir_all(&reg_pkg_dir, &cached_pkg_dir)?;
                    }

                    // Calculate and verify content hash
                    let sha256 = calculate_dir_sha256(&cached_pkg_dir)?;
                    if let Some(ref exp_sha) = expected_sha {
                        if sha256 != *exp_sha {
                            return Err(format!(
                                "content hash mismatch for cached package '{} {}': expected '{}', found '{}'",
                                dep_name, ver_str, exp_sha, sha256
                            ));
                        }
                    }

                    let dep_manifest = cached_pkg_dir.join("starkpkg.json");
                    if let Some(existing) = self.packages.get(dep_name) {
                        if existing.version != version {
                            return Err(format!(
                                "duplicate package name '{}' with conflicting versions: resolved both '{}' and '{}'",
                                dep_name, existing.version_str(), ver_str
                            ));
                        }
                        if existing.manifest_path != dep_manifest {
                            return Err(format!(
                                "duplicate package name '{}' resolved to different paths",
                                dep_name
                            ));
                        }
                        continue;
                    }

                    let dep_pkg = Package::from_manifest(&dep_manifest)?;
                    self.packages.insert(dep_name.clone(), dep_pkg);
                    resolved_packages.insert(dep_name.clone(), ResolvedMeta { sha256 });

                    self.resolve_dependencies_for(
                        dep_name,
                        visit_stack,
                        locked,
                        offline,
                        registry_dir,
                        cache_dir,
                        existing_lock,
                        resolved_packages,
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
}

impl Package {
    pub fn version_str(&self) -> String {
        format!("{}.{}.{}", self.version.major, self.version.minor, self.version.patch)
    }
}

struct ResolvedMeta {
    sha256: String,
}
