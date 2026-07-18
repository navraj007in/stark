# WP-C2.9 — Numeric, Layout, Text, Process, and Package Contracts

Gate: C2 (Reference Execution Semantics and Compiler Service Foundation).

Status: **Completed 2026-07-18.**

## Scope delivered

- strict UTF-8 source ingestion, identifier limits, escape legality, and deliberately
  non-observable parser recovery;
- fixed-width checked integer arithmetic, division/remainder, shifts, numeric casts, IEEE
  binary32/binary64 operations, NaN/signed-zero behavior, and backend reproducibility;
- the explicit decision that primitive floats do not implement `Eq`, `Ord`, or `Hash`;
- UTF-8 string validity, byte offset units, scalar boundaries/iteration, and Unicode 15.1
  locale-independent case conversion;
- deterministic module files, paths, imports, visibility, re-exports, cycles, public-API
  reachability, manifests, dependency sources, SemVer selection, multi-version coexistence,
  package/public-item identity, and lockfiles;
- executable entry signatures, normal/result/trap status mapping, and byte-stream/flush rules;
- the only observable layout queries, the absence of a stable Core ABI, and explicit
  compiler/resource/host-process failure classifications;
- formatting, hash collections, first-class file I/O, conversion, math, random, and conformance
  profile contracts;
- forty-six C2.9 granular rules transferred into normative homes and mechanically checked.

## Approved decisions

Integer operations and casts are checked and build-mode invariant. Primitive float operations
are IEEE and bit-reproducible; standard transcendental functions retain bounded last-bit
latitude. Strings use UTF-8 byte offsets and scalar-boundary validation.

Package identity combines logical source identity, canonical name, exact selected version, and
locked content, never an absolute checkout path. One exact version is selected per
source/name/major line; different major lines may coexist through aliases. Public signatures
may expose only publicly nameable items.

Executables accept four no-argument `main` return forms (`Unit`, `Int32`,
`Result<Unit,String>`, and `Result<Int32,String>`), with deterministic status and stream
mapping. `core-min` is mandatory; `std-full` is optional but indivisible and includes a
first-class non-`Copy` `File`.

## Evidence and scope control

The validator requires every C2.9 rule to occur exactly once in normative sources and remain
complete in the granular inventory. Combined specification generation, the existing
manifest-synchronized parser corpus, governance parsing, conformance reporting, and the full
Rust regression gates are used as transition evidence.

This package freezes contracts only. It does not align the current resolver, package manager,
type checker, interpreter, standard library, CLI, or backend, and it does not claim granular
positive/negative executable evidence. WP-C2.11 owns those corrections after C2.10 completes.
