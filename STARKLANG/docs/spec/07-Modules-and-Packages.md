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
A declaration `mod name;` has the following candidate files:
1. `name.stark`
2. `name/mod.stark`

The loaded file defines the contents of the module `name`.

**MOD-FILE-001.** Resolution is relative to the declaring source file's
module directory and must remain within the canonical package root after
resolving `.`/`..` and symbolic links. Exactly one candidate above must exist;
if both exist, the declaration is ambiguous and rejected. One canonical file
may define at most one module in a package. Missing files are always errors
outside an explicitly identified conformance-harness input mode.

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

**MOD-PATH-001.** `crate` starts at the current package root, `self` at the
current module, and each `super` moves exactly one parent and is rejected at
the root. An unqualified first segment follows `NAME-RESOLVE-001`; a
dependency alias starts at that dependency's public root. Later segments are
resolved only within the preceding module/type namespace. Filesystem paths,
checkout locations, and import aliases never participate in public-item
identity.

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

**MOD-USE-001.** Nested imports expand as if each leaf were a separate import.
An explicit alias introduces only the alias. A glob imports all public names
in the selected namespaces but never recursively imports another module's
imports. Two imported leaves that introduce the same namespace/name are
ambiguous unless they resolve to the same canonical item; an explicit local
item also conflicts rather than silently winning. Import processing is
independent of declaration and filesystem order.

## Visibility
- Items are private to their defining module by default.
- `pub` makes an item visible to parent modules and external modules.
- `priv` explicitly marks an item as private (same as default).

Visibility applies to:
- Functions, structs, enums, traits, impl blocks, consts, type aliases, and modules.

**MOD-VIS-001.** A private item is usable in its defining module and
descendant modules only. A public item is externally reachable only through a
path whose every module and re-export edge is public. Fields and enum variants
follow their declarations' explicit visibility rules; an `impl` cannot make
its self type or trait more visible.

**MOD-REEXPORT-001.** Every type and trait appearing transitively in a public
function, constant, field, variant, alias, trait item, or implementation
signature must be nameable by consumers through a public canonical path.
Private items and dependency items lacking a public re-export are rejected in
public API. `pub use` cannot re-export an item the re-exporting module is not
permitted to access, and does not create new nominal identity.

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
| `dependencies` | object | no | Map from local import alias to a *version constraint string* or a *dependency object*. |

A dependency object may set `"package"` when the local alias differs from the
manifest package name, and has exactly one source form:
- `{ "version": "<constraint>", "registry": "<identity>"? }`;
- `{ "path": "<relative-path>" }`; or
- `{ "git": "<origin>", "rev": "<immutable-revision>", "subdir": "<path>"? }`.

Version constraint syntax:
- `"1.2.3"` — exactly that version
- `"^1.2.3"` — `>=1.2.3` and `<2.0.0` (compatible-with)
- `">=1.2, <2.0"` — comma-separated comparator list (`>=`, `>`, `<=`, `<`, `=`),
  all of which must hold; omitted minor/patch components are zero

Validation: unknown top-level fields are ignored (forward compatibility);
a missing/invalid required field, a malformed constraint, a `path` escaping
the workspace, or a dependency whose name violates the name rule is a
manifest error and compilation MUST fail.

**PKG-MANIFEST-001.** Manifest input is UTF-8 JSON with one object root.
Duplicate keys are errors. `name`, `version`, `entry`, and `dependencies` are
validated exactly by this section before source loading. Relative paths are
resolved against the manifest directory and confined to the declared
workspace after canonicalization. Unknown fields are retained for tooling but
have no Core semantics. Core v1 has no conditional dependency features.

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
- The selected source, exact version, and package token MUST be reported in build output.

**PKG-RESOLVE-001.** A dependency map key is its local import alias. A string
value names the same package and a registry version constraint. A dependency
object may additionally set `"package"` to the canonical package name and
must select exactly one source:

- `"version"` with optional canonical `"registry"` identity;
- `"path"` to a workspace package; or
- `"git"` with an immutable `"rev"` and optional package-relative `"subdir"`.

Resolution collects the complete graph before selecting versions. All
constraints for one logical source, canonical package name, and major-version
line are intersected; an empty intersection, source disagreement, alias
collision, manifest-name mismatch, or dependency cycle rejects the graph.
Aliases affect imports only and never package identity.

**PKG-VERSION-001.** Semantic versions are ordered by SemVer precedence.
Within one permitted major line, the highest available non-yanked version
satisfying every constraint is selected; build metadata does not affect
precedence and a tie is broken by exact version string then locked content
hash. Prereleases are considered only when a constraint explicitly names a
prerelease. With a valid lockfile, its still-compatible exact selection wins
without re-resolution.

**PKG-IDENTITY-001.** A resolved package-instance token is relocation-stable:

- registry: canonical registry identity, package name, exact version, locked
  content hash;
- git: canonical origin, immutable revision, subdirectory, manifest package
  name/version, locked content hash;
- workspace path: workspace dependency identity, manifest package
  name/version, locked content hash—never an absolute checkout path.

A public item identity appends its canonical module path, item name, item kind,
and normalized generic arguments. Aliases and re-exports preserve it.

**PKG-MULTIVER-001.** At most one exact version may be selected for each
`(logical source identity, package name, major-version line)`. Different major
lines may coexist and have distinct identities, but must be imported through
distinct local aliases. The same exact package instance reached by multiple
paths is shared.

**PKG-LOCK-001.** A reproducible build uses `stark.lock`, generated
deterministically from the resolved graph. It records schema version, every
package-instance token component, exact dependency edges and aliases, and a
cryptographic content hash. Entries are sorted by package token and edges by
alias. A missing lockfile may be generated; a stale, incompatible, ambiguous,
or hash-mismatching lockfile rejects locked/frozen operation and must never be
silently rewritten there.

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

**MOD-CYCLE-001.** Module declarations form a finite tree and duplicate
canonical files or ancestor re-entry are rejected. Import cycles are legal
because item declarations are collected before import resolution, but a
cycle consisting only of unresolved re-export aliases is rejected with a
deterministically ordered cycle. Package cycles are always rejected.

## Executable and target contract

**PROC-MAIN-001.** An executable package must expose exactly one non-generic
private-or-public root item named `main` with no parameters and one of these
return types: `Unit`, `Int32`, `Result<Unit, String>`, or
`Result<Int32, String>`. Async, overloaded, imported, and dependency `main`
items do not qualify. A library need not define `main`.

**PROC-EXIT-001.** Normal `Unit` and `Ok(Unit)` return status 0. `Int32` and
`Ok(Int32)` must be in `0..=255` and return that status; an out-of-range value
traps as `invalid-exit-status`. `Err(message)` writes `message` plus LF to
stderr and returns status 1. A language trap returns status 101 after its
specified diagnostic. Host/process failure has target-defined status and is
not a STARK trap. Normal nonzero statuses are normal termination, not traps.

**PROC-STREAM-001.** At startup stdin is the host-provided byte stream and
stdout/stderr are distinct ordered byte streams; Core v1 exposes no stdin read
API. Standard output operations obey `STD-FORMAT-001`. All successfully
submitted bytes are flushed before normal or language-trap termination.
Startup or flush failure is host/process failure. Core adds no terminal
encoding, color, carriage return, or platform newline conversion.

**LAYOUT-QUERY-001.** `size_of<T>` and `align_of<T>` are the only Core layout
observations. For every `Sized` `T` they return positive target-contract
values (except a target may report size zero for a zero-sized type), are
compile-time/runtime consistent, and satisfy array/field placement needed by
safe execution. Discriminant representation, field offsets, niches, pointer
values, stack/heap choice, and physical addresses are unobservable.

**LAYOUT-ABI-001.** Core v1 promises no stable data layout, calling convention,
symbol mangling, object format, or cross-package native ABI. Layout-query
values may differ between named targets and compiler versions. Interoperation
requires a future explicitly versioned native-provider ABI and cannot infer
compatibility from equal `size_of`/`align_of` results.

**NUM-FLOAT-REPRO-001.** Primitive numeric behavior is target-independent as
defined by the abstract machine. The named target contract may affect only
standard-math last-bit latitude, layout queries, host/process failures, and
external I/O—not primitive arithmetic, casts, NaN comparisons, or signed
zero.

**LIMIT-RESOURCE-001.** Allocation, address-space, stack, call-depth, file-
descriptor, stream, and other host-resource exhaustion are host/process
failures unless an API returns a specified `Result`. Implementations must
prevent host undefined behavior and report the classified failure when the
host permits; exact capacities are implementation/target-defined.

**LIMIT-COMPILER-001.** Beyond semantic limits explicitly stated by Core
(including 255-byte identifiers, tuple arity 16, and `UInt64` array lengths),
an implementation may impose documented finite limits on source size,
nesting, items, constant-evaluation work, object size, modules, packages, and
dependency solving. Exceeding one rejects compilation with a deterministic
limit category and must not masquerade as a syntax/type error, crash, hang,
or change a semantic result. A conformance run declares these limits as part
of its implementation/target contract.

## Errors
- Importing an unknown module or item is a compile-time error.
- Accessing a private item outside its module is a compile-time error.
- Ambiguous imports are a compile-time error unless aliased.
## Conformance
A conforming Core v1 implementation MUST follow the requirements in this document. Any deviations or extensions MUST be explicitly documented by the implementation.
