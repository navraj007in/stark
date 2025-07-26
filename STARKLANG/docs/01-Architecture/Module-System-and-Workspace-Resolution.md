# STARKLANG Module System and Workspace Resolution

This document outlines how STARKLANG organizes and resolves modules, packages, and workspaces. It covers the structure, import resolution, visibility rules, multi-module projects, and workspace tooling support for large-scale development.

---

## 📦 Module System Overview

Modules in STARKLANG are the primary unit of code organization and encapsulation.

### 🔹 Module Definition
- A module is any `.stark` file or directory containing `.stark` source files with a corresponding `starkpkg.json` (optional).
- Files within a directory can be grouped as a module package.

### 🔹 Import Syntax
```stark
import utils.math
import services.auth as authService
```
- Hierarchical module names map directly to folder structures:
  - `utils.math` → `./utils/math.stark` or `./utils/math/index.stark`

### 🔹 Aliasing Imports
```stark
import analytics.reporting as report
```
Allows shorter names and disambiguation.

### 🔹 Module Visibility Rules
- All symbols are **private by default**.
- Use `export` keyword to expose:
```stark
export fn calculate_tax(): ...
export let config: Map<String, Any>
```

---

## 🗃 Module Resolution Mechanism

### Resolution Order:
1. Local project folder (`src/`)
2. Workspace path definitions (`stark.work`)
3. Package Manager cache (`~/.starkpkgs/`)
4. System-level global modules (standard library)

### Resolution Priority
- Closest scoped match wins.
- Circular dependencies raise a compiler error.

---

## 📁 Example Project Structure
```
my-project/
├── src/
│   ├── main.stark
│   ├── utils/
│   │   ├── math.stark
│   │   └── string.stark
│   └── services/
│       └── auth.stark
├── starkpkg.json
├── stark.work
```

### `stark.work` Workspace File:
```toml
[workspace]
members = ["src/utils", "src/services"]
```

---

## 📦 Package Dependencies

Dependencies are declared in `starkpkg.json`:
```json
{
  "name": "my-project",
  "version": "1.0.0",
  "dependencies": {
    "TensorLib": "^0.3.2",
    "CryptoLib": "^1.0.0"
  }
}
```

The compiler resolves from:
1. Local workspace modules
2. Downloaded packages cache

---

## 🧠 Module Re-Exports
```stark
export from utils.math
```
- Allows re-exporting symbols from another module transparently.

---

## 🌐 Standard Library Resolution
Standard modules such as `io`, `net`, `math`, `tensor`, `dataset` are preloaded by the compiler:
```stark
import std.io
import std.tensor
```
Resolved via internal compiler path, not local FS.

---

## 🧪 Future Enhancements
- Module interface files (`*.ifc`) for binary-only reuse
- Lazy loading for performance
- Tree-shaking unused exports
- Symbol graph indexing
- Versioned module imports

---

## ✅ Summary
STARKLANG’s module and workspace system is designed for scalability, modularity, and developer ergonomics. With predictable resolution order, aliasing, strict visibility, and workspace tooling, it enables robust development for both small scripts and enterprise-scale projects.

