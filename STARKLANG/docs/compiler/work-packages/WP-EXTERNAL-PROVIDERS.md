# WP-EXTERNAL-PROVIDERS — provider discovery and registration outside the compiler repository

**Status:** DESIGN, not started. **Priority:** P0. **Blocks:** every native capability the compiler
does not already ship — databases above all. **Release-architecture item**, not a database
prerequisite.

> **Governing claim.** A provider can be supplied outside the compiler repository without modifying
> compiler source, while preserving ABI validation, reproducibility, target qualification, and
> explicit trust policy.

---

## 1. Current state, verified

```rust
// src/provider_registry.rs
pub fn first_party() -> Vec<DeclaredProvider> {
    vec![stark_time(), stark_env(), stark_file(), stark_net(), stark_random()]
}

pub fn crate_location(crate_name: &str, repo_root: &Path) -> Option<PathBuf> {
    match crate_name {
        "stark-time-native"   => Some(repo_root.join("stark-time").join("native")),
        // ... four more, hardcoded
        _ => None,
    }
}
```

**Providers are not an ecosystem mechanism. They are compiler-integrated extensions.** Every native
capability is a hardcoded entry in two functions plus a crate under the repo root.

## 2. What that costs

* a database driver requires a change to compiler source;
* nobody outside this repository can publish a provider at all;
* provider versioning is welded to compiler releases;
* trust policy is implicit — the only providers that exist are ones we wrote;
* package resolution cannot discover a native implementation for a capability;
* **the public package system is incomplete for host capabilities**: a STARK package can declare a
  dependency, but a package needing a native provider cannot express or obtain one.

The last point is the one that makes this a release item. `starkpkg.json` describes a package
graph the toolchain can resolve; native capability is the hole in it.

## 3. Shape: manifests, not plugins

**No dynamic loading.** Providers stay statically linked into the generated Cargo workspace, which
is what makes the current safety model work — ABI validation happens before a symbol is ever
referenced, and a build either links or fails. The change is **how a provider is discovered**, not
how it is loaded.

A provider package declares itself:

```json
{
  "name": "stark-postgres-native",
  "version": "0.1.0",
  "provider": {
    "abi": "stark-provider-0.1",
    "crate": "native",
    "capabilities": ["stark.db.postgres"],
    "targets": [
      "x86_64-unknown-linux-gnu",
      "aarch64-apple-darwin",
      "x86_64-pc-windows-msvc"
    ]
  }
}
```

The provider's ABI surface — functions, resource types, capabilities — must be declarable in the
manifest or derivable from the crate, in the same `ProviderMetadata` shape `validate()` already
consumes. **The validator does not change.** Only its input source does: today a hardcoded `Vec`,
tomorrow a parsed manifest. That is the smallest change that removes the hardcoding, and it means
every existing ABI rule keeps applying unaltered.

## 4. Resolution pipeline

```text
resolve provider package        through the existing package graph
validate manifest               shape, ABI version, target triples
validate capability declarations no duplicate capability in the selected set
validate ABI symbols            provider_abi::validate, unchanged
check target support            refuse early with the triple named, not a link error
add crate to generated workspace as a path or registry dependency
pin exact version and checksum
link statically
record provider identity in build metadata
```

Two properties to preserve from today's behaviour:

* **Locations are not part of MIR.** `crate_location`'s doc is explicit: a crate's path is a
  property of the checkout, its name a property of the program. That separation is what keeps a
  verified MIR artefact relocation-stable, and manifest discovery must not break it.
* **Capability conflicts are resolved at selection**, not at link. Two providers offering the same
  capability is a build refusal naming both, not a duplicate-symbol error from the linker.

## 5. Trust policy — explicit, not enforced

Third-party providers execute native code in the user's build and process. They are **not** ordinary
STARK packages, and the manifest must not let them pretend to be.

Four tiers:

```text
pure STARK package            no native code, no provider
first-party native provider   shipped with the compiler, versioned with it
approved third-party provider explicitly enabled, pinned by version and checksum
untrusted / local provider    path-based, development only, never in a release build
```

Pre-alpha policy — deliberately simple, and **no sandboxing**:

* external providers **disabled by default**;
* enabled explicitly in the *application's* manifest, never transitively by a dependency;
* exact version **and checksum** required;
* **no transitive provider activation** — a library cannot pull a native provider into an
  application that did not ask for one;
* provider capabilities listed visibly at the point of enablement;
* release builds record provider hashes in build metadata;
* CI may enforce an allow-list.

> Do not attempt sandboxing. Make native trust **explicit and visible**; that is achievable now and
> is worth more than a partial isolation story that invites misplaced confidence.

## 6. Versioning and compatibility

* `abi: "stark-provider-0.1"` is checked against the compiler's `ABI_VERSION` and refused on
  mismatch, with both versions named.
* A provider declares supported target triples; building for an unlisted triple is refused **before**
  code generation, naming the triple and the provider.
* Provider semver is independent of compiler semver — that independence is the point of the packet
  — but the ABI version is the compatibility boundary between them.
* Duplicate symbols across providers are refused at selection with both providers named. The ABI
  already treats a symbol as a provider-scoped identity; that must not degrade into a linker
  diagnostic.

## 7. Qualification evidence

An external provider must be able to produce the same evidence a first-party one does, or the gate
becomes weaker for exactly the code that deserves it most:

* the seven-step package gate for its STARK-facing package;
* **CD-355** declared-surface coverage — every public callable called;
* **CD-347/348** lifecycle evidence for every resource type it declares, natively, against a live
  peer or fixture;
* recorded provider identity, version and checksum in the qualification output.

This is what makes "third-party" a trust tier rather than a hole in the gate.

## 8. Reproducibility

* exact version + checksum pinning, recorded in the lockfile;
* build metadata records every provider's identity, version, hash and target;
* the same source and lockfile must select the same provider set on the same target;
* provider selection must not depend on ambient machine state beyond the target triple and the
  declared allow-list.

## 9. Exit criteria

**The architectural test, stated as an executable claim:**

> Adding PostgreSQL, MongoDB, MySQL or SQL Server requires **no compiler-source change**.

Concretely:

1. A provider outside the repository is discovered, validated, linked and used, with no edit to
   `provider_registry.rs`.
2. `first_party()` becomes a *default set* expressed the same way an external provider is —
   discovered rather than hardcoded — so there is one mechanism, not two.
3. An ABI-version mismatch, an unsupported target, a duplicate capability and a missing checksum
   are each refused with a diagnostic naming the provider.
4. A provider not enabled in the application manifest cannot be activated by a dependency.
5. A release build records every provider's hash.

## 10. Explicit non-goals

* Dynamic loading or plugin `dlopen`.
* Sandboxing or capability enforcement at runtime.
* A provider registry service or distribution channel.
* Cross-compiling providers the toolchain cannot build for the host.
