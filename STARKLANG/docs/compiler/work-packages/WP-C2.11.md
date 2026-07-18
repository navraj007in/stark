# WP-C2.11 — Implementation Alignment and Adversarial Conformance

Status: **Completed 2026-07-18.**

## Scope delivered

C2.11 aligned the reference front end, interpreter, package resolver, command-line runners,
diagnostics, and executable evidence with the decisions frozen by C2.7–C2.10.

The implementation now includes:

- fixed-width checked integer behavior at every primitive width, operand-width shift checks,
  integer-only exponentiation, Float32 rounding at each operation boundary, IEEE division,
  and canonical float display;
- transparent generic type aliases, alias-cycle rejection, unsized-position checks, and direct
  and indirect infinite-size rejection with `Box`/`Vec` indirection;
- compile-time constant subset validation, deterministic dependency-cycle detection,
  compile-time trap conversion, memoized immutable values, and primitive constant patterns;
- conservative trait-overlap checking, trait-associated static function resolution,
  ambiguity rejection, callable builtin `Display` and standard FNV-1a `Hash`;
- generic borrow-carrying aggregates, consuming `File::close`, first-class non-`Copy` file
  resources, collection ownership cleanup, and `Vec::append` source draining;
- decoded byte and Unicode escapes, UTF-8 rejection, byte-indexed string operations,
  scalar-boundary traps, Core split/replace/trim/case behavior, and case expansion;
- canonical package names separated from source aliases, one-major-line constraints,
  distinct-major coexistence through explicit aliases, and public-signature reachability
  including public re-exports;
- all four legal executable entry signatures, target-selection diagnostics, exact normal and
  `Result` status mapping, `Err(String)` stderr behavior, invalid-status traps, and trap status
  101 in both command-line runners.

## Diagnostics and evidence

The collision-free catalogue allocates module/import codes `E0205`–`E0209`, executable target
code `E0214`, constant/type well-formedness codes `E0215`–`E0217`, receiver code `E0304`,
constant-pattern code `E0305`, and warning `W0006`. The parser, resolver, type checker, tests,
and normative catalogue use the same assignments.

`STARKLANG/conformance/core-v1-c2.11-evidence.toml` records function-level positive and negative
evidence for the high-cost C2.11 surface. `check-conformance.py` rejects duplicate/unknown rule
IDs, missing sources, missing evidence categories, and stale function citations. AST span
containment now covers expression, block, statement, type, pattern, and item arenas across the
complete parseable fixture corpus.

## Deviations

DEV-009, DEV-018, DEV-019, DEV-022, DEV-023, and DEV-024 are resolved by this work package.
DEV-017 is closed for the high-cost C2.11 granular surface through the new evidence database;
legacy broad rules remain historical transition records and are not promoted into granular
claims. DEV-036 remains explicitly owned by WP-C2.12, as planned.

## Validation

Required validation comprises Rust formatting, Clippy with warnings denied, the full
workspace/all-targets/all-features test suite, fixture conformance, the legacy and granular
coverage validators, generated-spec regeneration, governance parsing, and whitespace checks.

## Next

WP-C2.12 builds the differential interpreter corpus and removes DEV-036's filename-based module
test-harness bypass before the Gate C2 exit audit.
