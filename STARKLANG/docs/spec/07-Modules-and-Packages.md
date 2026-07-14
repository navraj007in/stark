# STARK Modules and Packages Specification

## Overview
This document defines the module system, visibility rules, and package resolution for the STARK core language. It is normative for Core v1.

## Package Layout
A package is a directory containing a `starkpkg.json` manifest at its root.

- The manifest's `entry` field defines the root source file.
- If `entry` is omitted, the default is `src/main.stark`.

All module paths are resolved relative to the package root unless otherwise noted.

## Module Declarations
Modules are declared with the `mod` keyword.

```stark
mod math;

mod inline {
    pub fn add(a: Int32, b: Int32) -> Int32 { a + b }
}
```

### File-Based Modules
A declaration `mod name;` loads one of the following files, in order:
1. `name.stark`
2. `name/mod.stark`

The loaded file defines the contents of the module `name`.

## Module Paths
Paths use `::` separators.

- `crate` refers to the package root module.
- `self` refers to the current module.
- `super` refers to the parent module.

Examples:
```stark
use crate::utils::math::add;
use super::config;
```

## Imports (`use`)
The `use` statement brings names into scope.

```stark
use crate::utils::math::add;
use crate::utils::{math, io};
use crate::utils::math as m;
use crate::utils::*;
```

Rules:
- `use` affects name resolution in the current module only.
- `pub use` re-exports the imported name from the current module.
- Aliases with `as` are local to the current module.

## Visibility
- Items are private to their defining module by default.
- `pub` makes an item visible to parent modules and external modules.
- `priv` explicitly marks an item as private (same as default).

Visibility applies to:
- Functions, structs, enums, traits, impl blocks, consts, type aliases, and modules.

## Name Resolution Order
Within a module, names are resolved in the following order:
1. Local (lexical) scope — block scopes and function parameters, as defined in
   04-Semantic-Analysis.md
2. Items declared in the current module
3. Items brought into scope by `use`
4. Built-in names (prelude)

Unqualified names do not implicitly search parent or crate scopes. Items in
parent modules or the crate root require explicit `super::` or `crate::`
paths. (Note: *lexical* scopes inside a function body do nest — step 1 — but
*module* scopes never nest implicitly.)

## Manifest Schema (`starkpkg.json`)
The manifest is a JSON object with the following fields:

| Field | Type | Required | Rules |
| --- | --- | --- | --- |
| `name` | string | yes | Package name: `[a-z][a-z0-9_-]*`, 1–64 chars. Used as the import root for dependents. |
| `version` | string | yes | Semantic version `MAJOR.MINOR.PATCH` (numeric components; optional `-prerelease` tag). |
| `entry` | string | no | Package-root-relative path to the root source file. Default `src/main.stark`. MUST exist and MUST be inside the package directory. |
| `dependencies` | object | no | Map from package name to a *version constraint string* or a *dependency object*. |

A dependency object has exactly one of:
- `{ "version": "<constraint>" }` — resolved from a registry or cache, or
- `{ "path": "<relative-path>" }` — a local package directory containing its
  own `starkpkg.json`.

Version constraint syntax:
- `"1.2.3"` — exactly that version
- `"^1.2.3"` — `>=1.2.3` and `<2.0.0` (compatible-with)
- `">=1.2, <2.0"` — comma-separated comparator list (`>=`, `>`, `<=`, `<`, `=`),
  all of which must hold

Validation: unknown top-level fields are ignored (forward compatibility);
a missing/invalid required field, a malformed constraint, a `path` escaping
the workspace, or a dependency whose name violates the name rule is a
manifest error and compilation MUST fail.

Example:
```json
{
  "name": "my-app",
  "version": "0.1.0",
  "entry": "src/main.stark",
  "dependencies": {
    "tensor-lib": "^2.1.0",
    "utils": { "path": "../utils" }
  }
}
```

## Packages and Dependencies
Dependencies are declared in `starkpkg.json` under `dependencies`.

Resolution order for external packages:
1. Local package source (the current package)
2. Direct dependencies from `starkpkg.json`
3. Standard library package `std`

Dependency modules are accessed via their package name:
```stark
use TensorLib::tensor::Tensor;
```

### Dependency Version Resolution (Core v1)
- Version strings follow semantic versioning.
- If multiple versions satisfy a constraint, the highest version MUST be selected.
- If no version satisfies a constraint, compilation MUST fail with an error.
- The source of packages (registry, cache, or local path) is implementation-defined, but the chosen version MUST be reported in build output.

## Standard Library
The standard library is available under the `std` package name:
```stark
use std::io;
use std::collections::Vec;
```

## Cycles
- **Package dependency cycles** — direct (`A → A`) or transitive
  (`A → B → C → A`) — are a compile-time error.
- **Module declarations** (`mod`) form a tree rooted at the package entry
  file, so `mod` cycles cannot occur; declaring the same module twice is an
  error.
- **`use` imports** between modules of the same package MAY be mutually
  recursive (module A may `use` items from B and vice versa); this is not a
  cycle error.

## Errors
- Importing an unknown module or item is a compile-time error.
- Accessing a private item outside its module is a compile-time error.
- Ambiguous imports are a compile-time error unless aliased.
## Conformance
A conforming Core v1 implementation MUST follow the requirements in this document. Any deviations or extensions MUST be explicitly documented by the implementation.
