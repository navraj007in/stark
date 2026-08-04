---
name: stark-package-authoring
description: Use when creating, extending or reviewing a first-party STARK package under packages/ — including any package that needs host access (clock, filesystem, environment, random, TCP/DNS, TLS) and therefore a native provider crate. Encodes the manifest and layout rules the compiler enforces, the qualification gate a package must pass, and the compiler defects a package author has to code around.
tools: Read, Grep, Glob, Bash, Edit, Write
---

# Authoring a first-party STARK package

Packages live under `packages/`. Each one ships with a `*-consumer` package whose job is to
**execute the surface the package declares** — not to exist.

---

## 1. Layout and manifest

```text
packages/stark-<name>/
  starkpkg.json
  src/lib.stark          # `mod tests;` here means src/tests.stark is part of the package
  src/tests.stark        # fn test_*() — run by `stark test`
  native/                # ONLY if the package needs a host capability
packages/stark-<name>-consumer/
  starkpkg.json
  src/main.stark
```

```json
{
  "name": "stark-example",
  "version": "0.1.0",
  "entry": "src/lib.stark",
  "capabilities": ["tcp", "dns"],
  "dependencies": {
    "stark_other": { "package": "stark-other", "path": "../stark-other", "version": "0.1.0" }
  }
}
```

**Dependency paths must be siblings.** The workspace root is the *parent directory* of your package
(`starkc/src/package.rs`, `get_workspace_root`), so anything outside `packages/` is refused by name
with `resolves to '…' which is outside the permitted workspace`. Use `../stark-other`, never an
absolute path out of the tree. Package names use hyphens; the dependency alias used in `use`
statements uses underscores.

`stark.lock` records names and versions, never paths.

## 2. Host access is declared, never assumed

A package that touches the outside world names the capability in its manifest, and a native
provider crate satisfies it at build time.

| capability | provider crate |
| --- | --- |
| `clock` | `packages/stark-time/native` |
| `filesystem` | `packages/stark-file/native` |
| `process.env`, `process.args` | `packages/stark-env/native` |
| `random` | `packages/stark-random/native` |
| `tcp`, `dns` | `packages/stark-net/native` |
| `tls` | `packages/stark-tls/native` |

**Capability-backed packages cannot run under `stark run`.** The reference and MIR interpreters have
no provider layer at all, so anything reaching the network or the filesystem builds with
`stark build` or not at all. This is a deliberate boundary; do not try to work around it.

If you add a provider crate, read the **stark-layout-verification** skill first — the relative
depths are load-bearing and have broken three times:

- `packages/<name>/native/Cargo.toml` → `../../../starkc/stark-provider-abi`
- `include_str!` inside `native/src/lib.rs` → `../../../../starkc/providers/<name>-native.json`
  (one level deeper: it resolves from the *source file*, not the manifest)
- a new provider needs a manifest at `starkc/providers/<name>-native.json`, whose `crate_path`
  stays relative to the packages root (`stark-<name>/native`) — the compiler finds that root by
  walking up, so do not prefix it with `packages/`.

## 3. The consumer must execute the surface

CD-345 found `stark-net` passing every gate step while `connect`, `read`, `write` and `close` had
never been called by anything — the consumer only formatted addresses, and a build-breaking defect
sat undetected because nothing had ever lowered a call into the raw bindings.

The bar differs by package category:

| category | what the consumer must do |
| --- | --- |
| pure package | execute each principal public behaviour |
| function-shaped provider | **successfully** invoke each capability family |
| resource-shaped provider | **successfully** acquire, use and release every resource type, both release paths (explicit and by drop), against a live peer |
| failure-only environment | a deterministic negative path is allowed, but must be labelled lowering/linking evidence — never lifecycle evidence |

A resource package needs a separate `*-resource-consumer`, because step 5 of the gate runs
`stark run` and the interpreter cannot reach a bound resource.

## 4. The qualification gate

Register the package in `starkc/scripts/qualify-first-party-packages.py` (`CASES`) and run it:

```bash
python3 starkc/scripts/qualify-first-party-packages.py \
  --stark starkc/target/release/stark --repo-root "$PWD"
```

Per package it runs, in order: `stark check` → `stark test` → **declared-surface check** →
`stark fmt --check` → consumer `stark check` → consumer `stark run` (exact stdout match) →
consumer `stark build --no-build-cache` → run the native binary (same stdout) → resource lifecycle
against a live peer, if the package declares resources.

**The declared-surface check (CD-355) is the one that catches real gaps:** every public callable
must actually be *called* by the package's own tests or by a consumer. An item that genuinely
cannot be called yet goes in `surface_blocked` mapped to the open defect that blocks it — and the
check *refuses* an entry whose item has become callable, so fixing the defect forces the waiver out
rather than letting it rot.

## 5. Compiler defects to code around

These are compiler limitations, not package bugs. Code shaped oddly for one of these reasons
deserves a comment saying which:

| id | what it means for your code |
| --- | --- |
| **DEV-160b** | a call whose borrow reaches it from an earlier block does not build natively. Bind the fields to locals first. |
| **DEV-157** | `panic` in value position has no native representation. Nest match arms instead. |
| **DEV-156** | `stark fmt` evicts a struct field's or enum variant's doc comment to after the type. Fold field-level explanation into the type's own doc comment. |
| **DEV-159** | a native build can race its own dependency build. Unreproduced; retry before investigating. |
| **E0105** | iterator combinators (`map`, `filter`, `collect`, `fold`, `any`, `all`, `find`) are refused by the front end. Iterate a borrow (`v.iter()`) in a `for` loop. |

Language reminders that trip package authors: PascalCase primitives (`Int32`, never `i32`);
integer overflow, division by zero, out-of-bounds indexing and failing casts **trap in every build
mode**, so an arithmetic boundary in a parser is a process abort, not a bad error message; no
closures, no `async`, no trait objects.

## 6. Testing a package that parses untrusted input

If the package reads anything from a wire, a file or a user, ordinary malformed-input tests are
**necessary and not sufficient**. Aim tests at the arithmetic boundaries specifically: SEC-HTTP-001
and SEC-HTTP-002 were remote-abort vulnerabilities sitting exactly where a magnitude guard stopped
and a final accumulation still happened, and eleven malformed-input routes could not reach either.

Assert on the **named error**, not merely on failure. Eighteen cases all reporting "the response
was bad" would also pass against a parser that rejected the valid inputs above them.
