# WP-C2.10 — Future-Extension Compatibility Boundaries

Gate: C2 (Reference Execution Semantics and Compiler Service Foundation).

Status: **Completed 2026-07-18.**

## Scope delivered

- the normative `CORE-V1-FUTURE-BOUNDARIES.md` chapter;
- reserved syntax and explicit edition/extension evolution rules;
- an ownership-aware future callable model covering consume-once, exclusive mutation, shared
  repetition, move/shared/exclusive capture, provenance, and deterministic destruction;
- compatibility space for explicit lifetimes, reference fields, variance, lifetime-bearing
  associated types, higher-ranked bounds, and native-boundary erasure;
- explicit exclusion of capturing closures, lifetime syntax, trait objects, macros, and
  compile-time generation from Core v1;
- the normative statement that Core v1 execution is single-threaded and does not claim future
  cross-thread data-race proof;
- required future concurrency subjects including atomics, ordering, thread-safety markers,
  synchronization, shared ownership, thread locals, cross-thread Drop, and failure propagation;
- exclusion of public unsafe/general FFI and a metadata-bound native-provider boundary with
  identity, integrity, ABI, target, provenance, capability, and verification requirements;
- versioned, explicitly enabled extension isolation that cannot alter Core-only behavior;
- six granular rules transferred into normative homes and mechanically checked.

## Approved decision

`CORE-Q-016` is approved. Core v1 remains safe and single-threaded. Capturing closures,
explicit lifetimes, concurrency, trait objects, macros, unsafe code, and general FFI are future
features, not silent implementation extensions. Approved native providers are the only Core
host boundary and must preserve static capability derivation. Post-v1 extensions require
explicit identity/version enablement and cannot change a Core-only program.

## Evidence and scope control

The combined-spec builder and fixture extractor include the new normative chapter. It contains
no `stark` examples, so the synchronized 112-fixture corpus is unchanged. The validator requires
all six C2.10 rules to occur exactly once and remain complete in the inventory.

This work package introduces no parser, type-system, runtime, concurrency, macro, vtable, unsafe,
FFI, provider, or extension implementation. WP-C2.11 owns enforcement/alignment evidence for
the Core exclusions and existing extension isolation.
