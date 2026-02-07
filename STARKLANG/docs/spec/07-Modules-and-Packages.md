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
1. Local scope
2. Items declared in the current module
3. Items brought into scope by `use`
4. Items in parent modules via explicit `super::` paths
5. Items in `crate` via explicit `crate::` paths

Unqualified names do not implicitly search parent or crate scopes.

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

## Standard Library
The standard library is available under the `std` package name:
```stark
use std::io;
use std::collections::Vec;
```

## Cycles
Direct cyclic module dependencies are a compile-time error.

## Errors
- Importing an unknown module or item is a compile-time error.
- Accessing a private item outside its module is a compile-time error.
- Ambiguous imports are a compile-time error unless aliased.
## Conformance
A conforming Core v1 implementation MUST follow the requirements in this document. Any deviations or extensions MUST be explicitly documented by the implementation.
