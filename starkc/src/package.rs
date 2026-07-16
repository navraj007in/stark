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
        *pos += 1; // skip '{'
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
            *pos += 1; // skip ':'
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
            *pos += 1; // skip ','
        }
        Ok(JsonValue::Object(map))
    }

    fn parse_array(chars: &[char], pos: &mut usize) -> Result<JsonValue, String> {
        *pos += 1; // skip '['
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
            *pos += 1; // skip ','
        }
        Ok(JsonValue::Array(list))
    }

    fn parse_string(chars: &[char], pos: &mut usize) -> Result<JsonValue, String> {
        *pos += 1; // skip '"'
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

#[derive(Clone, Debug)]
pub struct Package {
    pub name: String,
    pub version: String,
    pub entry: PathBuf,
    pub manifest_path: PathBuf,
    pub dependencies: HashMap<String, PathBuf>,
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

        let version = obj.get("version")
            .ok_or_else(|| format!("missing 'version' in manifest '{}'", path.display()))?
            .as_str()
            .ok_or_else(|| format!("'version' in manifest '{}' must be a string", path.display()))?
            .to_string();
            
        let parts: Vec<&str> = version.split('.').collect();
        if parts.len() != 3 || !parts.iter().all(|p| !p.is_empty() && p.chars().all(|c| c.is_ascii_digit())) {
            return Err(format!("invalid package version '{}' in manifest '{}'", version, path.display()));
        }

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
                let dep_path_str = dep_config.get("path")
                    .ok_or_else(|| format!("missing 'path' for dependency '{}' in manifest '{}'", dep_name, path.display()))?
                    .as_str()
                    .ok_or_else(|| format!("'path' for dependency '{}' in manifest '{}' must be a string", dep_name, path.display()))?;
                
                let dep_dir = parent_dir.join(dep_path_str);
                let dep_dir = dep_dir.canonicalize()
                    .map_err(|_| format!("dependency path '{}' for '{}' in manifest '{}' does not exist", dep_path_str, dep_name, path.display()))?;
                
                dependencies.insert(dep_name.clone(), dep_dir);
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
pub struct PackageGraph {
    pub root_package_name: String,
    pub packages: HashMap<String, Package>,
    pub workspace_root: PathBuf,
}

impl PackageGraph {
    pub fn load_from_root(root_manifest_path: &Path) -> Result<Self, String> {
        let root_package = Package::from_manifest(root_manifest_path)?;
        let workspace_root = get_workspace_root(root_manifest_path);

        if !is_within_workspace(root_manifest_path, &workspace_root) {
            return Err("root package is outside the permitted workspace".to_string());
        }

        let mut packages = HashMap::new();
        let root_name = root_package.name.clone();
        packages.insert(root_name.clone(), root_package);

        let mut graph = Self {
            root_package_name: root_name.clone(),
            packages,
            workspace_root,
        };

        graph.load_dependencies_for(&root_name, &mut Vec::new())?;

        Ok(graph)
    }

    fn load_dependencies_for(&mut self, package_name: &str, visit_stack: &mut Vec<String>) -> Result<(), String> {
        visit_stack.push(package_name.to_string());

        let package = self.packages.get(package_name).unwrap().clone();
        for (dep_name, dep_dir) in &package.dependencies {
            let dep_manifest = dep_dir.join("starkpkg.json");
            
            if !is_within_workspace(&dep_manifest, &self.workspace_root) {
                return Err(format!(
                    "dependency '{}' of package '{}' resolves to '{}' which is outside the permitted workspace '{}'",
                    dep_name, package_name, dep_manifest.display(), self.workspace_root.display()
                ));
            }

            if let Some(pos) = visit_stack.iter().position(|x| x == dep_name) {
                let cycle = visit_stack[pos..].to_vec();
                return Err(format!(
                    "dependency cycle detected: {} -> {}",
                    cycle.join(" -> "), dep_name
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
            self.load_dependencies_for(dep_name, visit_stack)?;
        }

        visit_stack.pop();
        Ok(())
    }
}
