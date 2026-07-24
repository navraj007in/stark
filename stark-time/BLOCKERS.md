# std-time v0.1 blockers

## Classification

READY_PACKAGE_PROVIDER

## Repository head

cf54fb09769e28b966035bb7b67a3c824329369d

## Blocked requirement

Real `extern "C"` provider linkage/invocation: STARK generated code calling
`stark_time_monotonic_now_ns` / `stark_time_unix_now` through Native Provider ABI v0.1 and
observing their `ProviderStatus`/output-slot results.

## Evidence

- command: `python3 STARKLANG/tools/build-core-spec.py --check` and reading
  `STARKLANG/docs/compiler/native-provider-abi-v0.1.md` directly.
- source: `starkc/src/backend/provider_abi.rs` module doc: "this is compile-time/build-time
  validation, not a runtime check against an executing provider -- no provider actually executes
  in the C5 MVP (§10.2's implementation boundary)."
- expected (for `WP-TIME-A COMPLETE` per §29.1 and full `std-time v0.1 COMPLETE` per §29.2): a
  documented, owner-approved way for generated Rust (`starkc/src/backend/generated_rust/`) to
  declare an `extern "C"` symbol for a provider function, link a provider crate into the produced
  binary, and call it from a MIR body.
- actual: `COMPILER-STATE.md` (head `cf54fb0`) states, verbatim, under CD-053/CD-054: "No provider
  executes. §10.2's boundary is unchanged." and "the ABI version stays `0.1` (nothing has shipped
  or executed against it)." `starkc/src/backend/generated_rust/` has no provider-call code
  generation; `starkc/src/backend/provider_abi.rs` and `stark-runtime/src/provider_abi.rs`
  contain only metadata validation and `#[repr(C)]` type definitions, never `extern "C"`
  declarations, dynamic loading, or a call site.
- diagnostic: none (this is an absence of a mechanism, not a compiler error).
- The ABI document's own scope boundary (`native-provider-abi-v0.1.md` lines 19-23, citing
  `WP-C5-ENTRY.md` §10.2), quoted exactly: "This document specifies the v0.1 ABI shape and ships
  a compile-time metadata validator plus mock fixtures (§17). It does **not** implement dynamic
  loading, real `extern "C"` linkage, or any provider actually executing... No C5 exit claim may
  say providers run natively." `std-time` is the first package to need a real provider, but
  WP-TIME-A §1.1 explicitly forbids this package from being the one that invents that mechanism.

## §24.1 mandatory provider-execution blocker acknowledgment

- ABI v0.1 metadata validation exists (`starkc/src/backend/provider_abi.rs`, exercised in
  `stark-time/native/src/lib.rs`'s `provider_metadata_validates_against_abi_v0_1` test).
- ABI runtime boundary types exist (`stark-runtime/src/provider_abi.rs`: `ProviderStatus`,
  `RawResourceHandle`, `OwnedResourceHandle`, `BorrowedBuffer(Mut)`).
- Real linkage/invocation is explicitly deferred in the approved ABI document (quoted above).
- `std-time` cannot claim native STARK execution until an owner-approved execution seam lands:
  `Instant::now`, `UnixTimestamp::now`, and `Instant::elapsed` are not implemented, and no test in
  this package claims they work.
- No loader/linker/manifest policy was invented: the native crate is a standalone, unlinked Rust
  library: nothing in `starkc/`, `stark-runtime/`, or any manifest schema was touched to make it
  reachable from generated code.

## Why this package cannot fix it

Implementing this would require inventing a provider loader, linker contract, manifest schema,
symbol-discovery mechanism, Cargo integration policy, or generated-code call shape from inside
this package -- every one of those is explicitly prohibited by WP-TIME-A §1.1, and modifying
`starkc/src/backend/generated_rust/` (native backend provider-call generation) or
`starkc/src/backend/provider_abi.rs` / `stark-runtime/src/provider_abi.rs` (provider ABI types or
rules) is listed as a prohibited modification in WP-TIME-A §7.2. This is compiler-track scope
(the new Gate C0-C10 governance track, `STARKLANG/docs/compiler/COMPILER-ROADMAP.md`), not
package scope.

## Existing approved owner

None found. `COMPILER-STATE.md`'s current header (2026-07-24) lists Gate C6's open work
(WP-C6.2b's remaining findings, C6.2c-e, WP-C6.3 runtime values/collections, C6.4 platform matrix,
C6.5 differential corpus, C6.6 gate exit) and does not mention provider execution/linkage as a
scheduled item. CE4 Amendment 1 (approved at revision 3, CD-054) extended the ABI's *type* model
(the closed `AbiParam` form, the raw/owning handle split, close-function rules) but explicitly did
not authorize execution: "No provider executes; §10.2's boundary is unchanged."

## Minimum next decision

The owner needs to authorize (or point to an existing, not-yet-discovered authorization for) one
bounded provider-execution design: how generated Rust links an `extern "C"` provider function
declared under Native Provider ABI v0.1 into a native STARK binary, and how a MIR body calls it.
Nothing wider -- not a general FFI mechanism, not dynamic provider discovery, not plugin loading
(all separately excluded by WP-TIME-A §4.2 and its own §1.1).

## Work completed safely

- `stark-time/starkpkg.json`, `stark-time/src/lib.stark` -- `TimeError`; `Duration` (all §11
  operations: `zero`, `from_seconds`, `from_millis`, `from_micros`, `from_nanos`, `seconds`,
  `subsec_nanos`, `is_zero`, `checked_add`, `checked_sub`, `as_millis`, `as_micros`, `as_nanos`);
  `Instant::checked_duration_since` (pure, §12.3); `UnixTimestamp` (all §13 operations except
  `now`: `from_unix_seconds`, `from_unix_millis`, `seconds`, `subsec_nanos`, `to_unix_millis`).
  40 `fn test_*()` functions (§19/§20 corpora), all passing under `stark test`.
- `stark-time/native/` -- a standalone Rust crate (`stark-time-native`, no third-party
  dependency) implementing `stark_time_monotonic_now_ns` and `stark_time_unix_now` per §16
  (panic-contained via `catch_unwind` + abort, guarded overflow checks, output written only on
  success), plus the exact §15 `ProviderMetadata` declaration, validated against the repository's
  real `starkc::backend::provider_abi::validate` (a dev-dependency on `starkc`/`stark-runtime`,
  not linked into the shipped provider). 13 Rust unit tests, all passing under `cargo test`.
- Intentionally NOT implemented: `Instant::now()`, `UnixTimestamp::now()`, `Instant::elapsed()` --
  the three frozen-API (§9) members that require the blocked capability above. Their absence is
  the visible boundary of this blocker; everything else in §9 that does not depend on a live
  clock reading is implemented.

## Closure status

PARTIAL — WAITING_PROVIDER_EXECUTION
