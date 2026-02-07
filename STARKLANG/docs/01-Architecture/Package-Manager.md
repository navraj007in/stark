> Non-Normative Note: This document provides architectural guidance and may evolve. The normative Core v1 language specification lives in `docs/spec/`.

# 📦 STARKLANG — Standard Library Layout & Package Manager Specification

This document defines:
1. The modular structure of STARKLANG’s **Standard Library (stdlib)**
2. The design of **starkpkg**, the official package manager, including dependency resolution, packaging, publishing, and tooling integration.

---

## 📌 Design Objectives

| Goal | Mechanism |
|------|-----------|
| Modularity | Domain-specific standard modules |
| Extensibility | Third-party package integration |
| Type Safety | Typed and self-documenting APIs |
| Developer UX | Easy to discover, install, version |
| Cloud-native | Package deployability & metadata standards |

---

## 🗂️ Standard Library Layout

```
stdlib/
├── core/
│   ├── primitives.stark
│   ├── math.stark
│   ├── string.stark
│   └── control.stark
├── collections/
│   ├── list.stark
│   ├── map.stark
│   └── set.stark
├── ai/
│   ├── tensor.stark
│   ├── model.stark
│   ├── dataset.stark
│   └── metrics.stark
├── concurrency/
│   ├── async.stark
│   ├── futures.stark
│   ├── channels.stark
│   └── actor.stark
├── system/
│   ├── io.stark
│   ├── fs.stark
│   ├── time.stark
│   ├── env.stark
│   └── process.stark
├── cloud/
│   ├── logging.stark
│   ├── metrics.stark
│   └── tracing.stark
├── crypto/
│   ├── hash.stark
│   ├── keypair.stark
│   └── signature.stark
└── test/
    └── assert.stark
```

---

## 📚 Example: `stdlib/core/string.stark`

```stark
export fn length(text: String) -> Int
export fn split(text: String, delim: Char) -> List<String>
export fn to_upper(text: String) -> String
```

---

## ✅ Importing Standard Modules

```stark
import core/string
let len = string.length("STARK")
```

---

## 📦 STARK Package Manager — `starkpkg`

### Overview:
`starkpkg` is the official package manager CLI for STARKLANG.

---

## 📁 Package Layout

```
my-awesome-lib/
├── starkpkg.toml
├── src/
│   ├── mod.stark
│   └── mylib.stark
└── tests/
    └── mylib_test.stark
```

---

## 📄 `starkpkg.toml` Structure

```toml
[package]
name = "my-awesome-lib"
version = "0.1.0"
description = "Awesome utilities for STARK"
license = "MIT"
authors = ["Navraj <me@navraj.dev>"]

[dependencies]
core = "1.0.0"
ai/tensor = "1.2.0"
some-lib = "0.3.1"

[build]
entry = "src/mod.stark"
```

---

## 🔧 Package Commands

| Command | Action |
|--------|--------|
| `starkpkg init` | Initialize new package |
| `starkpkg build` | Compile package to bytecode |
| `starkpkg test` | Run tests |
| `starkpkg publish` | Upload to registry |
| `starkpkg install <pkg>` | Add dependency |
| `starkpkg update` | Sync dependencies |
| `starkpkg info` | Show package metadata |

---

## 🔍 Dependency Resolution

- Semantic versioning: `^1.2.0`, `>=1.1.0, <2.0.0`
- Lockfile support: `starkpkg.lock` tracks resolved versions
- Caching: Local registry mirror support

---

## 🌐 STARK Package Registry

| Feature | Description |
|--------|-------------|
| Hosted Index | Git-like registry (like crates.io) |
| Package Search | Name, version, tags, categories |
| Binary Upload | `.starkpkg`, `.starkbc`, `.sig` |
| Auth | Token-based CLI authentication |

---

## 🔒 Security

| Feature | Implementation |
|--------|-----------------|
| Hash Verification | SHA256 checksum |
| Signature Verification | Author public key |
| Registry Audit Trail | Immutable publish logs |

---

## 🚀 Cloud & Serverless Support (Future)

| Feature | Notes |
|--------|-------|
| Deployable Packages | Annotated serverless functions auto-exported |
| `@serverless` macro | Tagged export for cloud API scaffolds |
| Deployment CLI | `starkpkg deploy` to STARK cloud runtime |

---

## 🧪 Testing Utilities

```
test/
├── assert.stark
├── snapshot.stark
└── test_runner.stark
```

Features:
- `assert_eq`, `assert_true`, `expect_fail`
- Snapshot testing (planned)
- Test coverage reporter (planned)

---

## 📌 Package Metadata Schema

| Field | Required | Description |
|-------|----------|-------------|
| name | ✅ | Unique package name |
| version | ✅ | Semver version |
| authors | ✅ | Author list |
| license | ✅ | SPDX format |
| dependencies | ✅ | Declared in `[dependencies]` |
| build.entry | ✅ | Entrypoint path |

---

## ✅ Summary

| Component | Status |
|----------|--------|
| Standard Library Layout | ✅ Defined |
| Package Manifest (`toml`) | ✅ Specified |
| Dependency Resolution | ✅ Supported |
| CLI Tooling | ✅ Outlined |
| Registry Interface | ✅ Designed |
| Testing Framework | ✅ Drafted |
| Serverless Support | ⏳ Planned |

STARKLANG’s stdlib and package system are foundational to its developer experience, ensuring modularity, discoverability, and scalability across cloud-native and AI-first systems.

# 📦 STARKLANG — Standard Library Layout & Package Manager Specification

This document defines:
1. The modular structure of STARKLANG’s **Standard Library (stdlib)**
2. The design of **starkpkg**, the official package manager, including dependency resolution, packaging, publishing, and tooling integration.

---

## 📌 Design Objectives

| Goal | Mechanism |
|------|-----------|
| Modularity | Domain-specific standard modules |
| Extensibility | Third-party package integration |
| Type Safety | Typed and self-documenting APIs |
| Developer UX | Easy to discover, install, version |
| Cloud-native | Package deployability & metadata standards |

---

## 🗂️ Standard Library Layout

```
stdlib/
├── core/
│   ├── primitives.stark
│   ├── math.stark
│   ├── string.stark
│   └── control.stark
├── collections/
│   ├── list.stark
│   ├── map.stark
│   └── set.stark
├── ai/
│   ├── tensor.stark
│   ├── model.stark
│   ├── dataset.stark
│   └── metrics.stark
├── concurrency/
│   ├── async.stark
│   ├── futures.stark
│   ├── channels.stark
│   └── actor.stark
├── system/
│   ├── io.stark
│   ├── fs.stark
│   ├── time.stark
│   ├── env.stark
│   └── process.stark
├── cloud/
│   ├── logging.stark
│   ├── metrics.stark
│   └── tracing.stark
├── crypto/
│   ├── hash.stark
│   ├── keypair.stark
│   └── signature.stark
└── test/
    └── assert.stark
```

---

## 📚 Example: `stdlib/core/string.stark`

```stark
export fn length(text: String) -> Int
export fn split(text: String, delim: Char) -> List<String>
export fn to_upper(text: String) -> String
```

---

## ✅ Importing Standard Modules

```stark
import core/string
let len = string.length("STARK")
```

---

## 📦 STARK Package Manager — `starkpkg`

### Overview:
`starkpkg` is the official package manager CLI for STARKLANG.

---

## 📁 Package Layout

```
my-awesome-lib/
├── starkpkg.toml
├── src/
│   ├── mod.stark
│   └── mylib.stark
└── tests/
    └── mylib_test.stark
```

---

## 📄 `starkpkg.toml` Structure

```toml
[package]
name = "my-awesome-lib"
version = "0.1.0"
description = "Awesome utilities for STARK"
license = "MIT"
authors = ["Navraj <me@navraj.dev>"]

[dependencies]
core = "1.0.0"
ai/tensor = "1.2.0"
some-lib = "0.3.1"

[build]
entry = "src/mod.stark"
```

---

## 🔧 Package Commands

| Command | Action |
|--------|--------|
| `starkpkg init` | Initialize new package |
| `starkpkg build` | Compile package to bytecode |
| `starkpkg test` | Run tests |
| `starkpkg publish` | Upload to registry |
| `starkpkg install <pkg>` | Add dependency |
| `starkpkg update` | Sync dependencies |
| `starkpkg info` | Show package metadata |

---

## 🔍 Dependency Resolution

- Semantic versioning: `^1.2.0`, `>=1.1.0, <2.0.0`
- Lockfile support: `starkpkg.lock` tracks resolved versions
- Caching: Local registry mirror support

---

## 🌐 STARK Package Registry

| Feature | Description |
|--------|-------------|
| Hosted Index | Git-like registry (like crates.io) |
| Package Search | Name, version, tags, categories |
| Binary Upload | `.starkpkg`, `.starkbc`, `.sig` |
| Auth | Token-based CLI authentication |

---

## 🔒 Security

| Feature | Implementation |
|--------|-----------------|
| Hash Verification | SHA256 checksum |
| Signature Verification | Author public key |
| Registry Audit Trail | Immutable publish logs |

---

## 🚀 Cloud & Serverless Support (Future)

| Feature | Notes |
|--------|-------|
| Deployable Packages | Annotated serverless functions auto-exported |
| `@serverless` macro | Tagged export for cloud API scaffolds |
| Deployment CLI | `starkpkg deploy` to STARK cloud runtime |

---

## 🧪 Testing Utilities

```
test/
├── assert.stark
├── snapshot.stark
└── test_runner.stark
```

Features:
- `assert_eq`, `assert_true`, `expect_fail`
- Snapshot testing (planned)
- Test coverage reporter (planned)

---

## 📌 Package Metadata Schema

| Field | Required | Description |
|-------|----------|-------------|
| name | ✅ | Unique package name |
| version | ✅ | Semver version |
| authors | ✅ | Author list |
| license | ✅ | SPDX format |
| dependencies | ✅ | Declared in `[dependencies]` |
| build.entry | ✅ | Entrypoint path |

---

## ✅ Summary

| Component | Status |
|----------|--------|
| Standard Library Layout | ✅ Defined |
| Package Manifest (`toml`) | ✅ Specified |
| Dependency Resolution | ✅ Supported |
| CLI Tooling | ✅ Outlined |
| Registry Interface | ✅ Designed |
| Testing Framework | ✅ Drafted |
| Serverless Support | ⏳ Planned |

STARKLANG’s stdlib and package system are foundational to its developer experience, ensuring modularity, discoverability, and scalability across cloud-native and AI-first systems.
