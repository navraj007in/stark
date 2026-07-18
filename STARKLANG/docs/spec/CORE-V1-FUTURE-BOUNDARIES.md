# STARK Core v1 Future-Extension Boundaries

## Status and authority

This chapter is normative for Core v1. It defines compatibility constraints and exclusions,
not features that a Core implementation must accept. Syntax or behavior described as future is
rejected in Core v1 unless an explicitly enabled, versioned extension owns it.

## Reserved syntax and evolution

**FUTURE-SYNTAX-001.** The reserved words in `01-Lexical-Grammar.md`, lifetime-token space,
capturing-lambda delimiters, `dyn`, `unsafe`, `extern`, `async`/`await`, macro invocation and
definition space, and attributes are unavailable to Core programs. A reserved word is rejected
where an identifier is required even though Core assigns it no grammar production.

A future Core edition or extension may assign these forms only under an explicit language
edition or extension version. It may not reinterpret a token sequence that is valid Core v1
with different Core behavior. Unknown edition, extension, attribute, macro, or reserved syntax
is rejected rather than ignored. Macros and compile-time code generation are not Core v1 and
must not be introduced as an implicit substitute for package, dependency, or build tooling.

## Capturing closures and explicit lifetimes

**FUTURE-CLOSURE-001.** Core v1 function values and `fn(...)` types are non-capturing. A future
capturing callable model must distinguish at least:

- consumption on one call;
- mutation through exclusive callable access; and
- repeated calls through shared callable access.

Capture analysis must classify each captured place as moved, shared-borrowed, or
exclusive-borrowed; preserve the referent/provenance rules; and make the closure value's own
move, Copy eligibility, borrow region, and deterministic Drop behavior derive from its
captures. Generic callable bounds may generalize today's non-capturing function parameters,
but every valid Core v1 `fn(...)` argument must retain its meaning and dispatch.

Future explicit lifetime parameters may relax Core v1's conservative lexical regions but may
not accept a dangling reference or invalidate a currently valid reference. The reserved design
space includes lifetime arguments, reference fields, variance, lifetime-bearing associated
types, higher-ranked bounds, and explicit erasure at a native ABI boundary. Core v1 accepts none
of that syntax and continues to prohibit declared reference fields and lifetime annotations.

Trait objects and runtime vtables are also outside Core v1. A future object-safety and `dyn`
design must coexist with static generic dispatch and cannot change an existing impl's
coherence, method priority, ownership, or destruction.

## Concurrency boundary

**FUTURE-THREAD-001.** Core v1 program execution is single-threaded. Its ownership and borrowing
rules prove aliasing and memory safety only for that execution model; Core makes no claim that
all current types are safe to send or share between future threads.

A concurrency extension must separately specify thread-safety marker obligations, atomics and
memory ordering, synchronization, shared ownership, thread-local storage, scheduler/process
observations, cross-thread destruction, failure/panic propagation, and concurrent native
providers. It must preserve Core's exactly-once evaluation and destruction for every execution
it accepts and may not introduce a data race into a program accepted as safe.

## Native providers, unsafe code, and FFI

**FUTURE-FFI-001.** Core v1 exposes no public `unsafe`, raw pointer, general FFI, external
calling-convention, or arbitrary dynamic-library interface. Host access occurs only through an
approved native-provider boundary whose package/artifact metadata identifies:

- provider and artifact identity, origin, integrity hash, and version;
- the versioned ABI and supported targets;
- imported type/ownership/error contracts and provenance;
- required capabilities and host resources; and
- verification status and the safe Core wrapper surface.

Provider internals may use host-unsafe mechanisms, but those mechanisms are not Core operations
and cannot weaken Core memory safety, ownership, trap, or capability rules. Pure Core package
capabilities must remain statically derivable from the resolved package/provider graph; a
provider cannot acquire an undeclared capability through a spelling convention or runtime
fallback.

## Extension isolation

**EXT-ISOLATION-001.** Every post-v1 extension has a stable identifier and semantic version and
is enabled explicitly in the package/build contract. Without that enablement, its syntax,
types, builtins, prelude names, providers, and diagnostics are absent and extension constructs
are rejected.

An extension may add explicitly scoped syntax and semantics, but cannot silently change Core
tokenization, name resolution, prelude contents, type identity, coherence, ownership, numeric
results, standard hooks, package identity, process observations, or diagnostics for a source
that uses only Core v1. Packages must declare required extensions and version constraints;
unsupported or conflicting requirements reject resolution. There is no silent fallback from an
extension operation to a Core approximation or host-specific behavior.

## Conformance

A conforming Core v1 implementation must enforce these exclusions and isolation rules. It need
not implement any future feature described here.
