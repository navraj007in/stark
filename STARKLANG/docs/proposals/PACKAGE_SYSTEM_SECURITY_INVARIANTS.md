# Proposal: Package-System Security Invariants and Capability Derivation

**Status:** RATIFIED AND APPLIED — owner approved 2026-07-18 ("apply amendments"); Amendments
A–D applied to the brief the same day in one pass with the §6.4 consistency sweep. Line
anchors and "current text" quotes below refer to the **pre-amendment** revision of the brief
and are retained as the historical record of what changed.
**Date:** 2026-07-18
**Amends:** `STARKLANG/docs/STARK-Ecosystem-Build-Brief-Revised-Sonnet.md` (the ecosystem brief).
**Relationship to other tracks:** creates one small, well-defined deliverable request to the
compiler track (§6.3); changes no Core syntax or semantics; consistent with the brief's rule 2
(no new syntax inside ecosystem work) and §1.5 prohibitions, which this proposal preserves.

---

## 0. Decision summary (what the owner is ratifying)

Three design decisions, converged after multi-model review:

1. **Controlled major-version coexistence, not single-version resolution.** A resolution graph
   selects at most one version per `(source, package name, major-version line)` slot.
   Semver-incompatible majors may coexist but must be visible everywhere (lockfile, tree,
   diffs, diagnostics). Applications may deny coexistence globally or per package name.
2. **Derived capabilities, not merely declared ones.** For pure STARK packages, the host
   capability set is computed from references to host-backed interfaces in the resolved
   graph — conservative, reference-level, deterministic. Manifest declarations become an
   *envelope* cross-checked against the derived set (`derived ⊆ declared` or the build fails).
   The root application approves the derived transitive closure. This supersedes the brief's
   current "capability metadata is informational" stance (WP-P0.2, line 493).
3. **Format-now / operations-later sequencing.** Everything that shapes durable serialized
   formats (identity, lockfile, archive, capability vocabulary) is specified in P0–P1.
   All operational registry machinery (publisher identity, MFA, typosquat defense, advisories,
   quarantine, reputation) stays in Gate P10, unchanged.

The resulting one-line theses:

> Package acquisition never executes code. **STARK proves pure packages are pure**: host
> capabilities are derived statically from the resolved graph, not trusted from publisher
> declarations. Native providers form a separate, explicit trust tier. Incompatible package
> majors may coexist, but never invisibly.

---

## 1. Design rules pinned by this proposal

These resolve the specific ambiguities found during review. They are normative for all
amendments below.

### 1.1 Identity vs. resolution slot — never conflate them

- **Package identity** (what a lockfile entry names, what a hash verifies) is:
  `source/registry identity + package name + exact version + content hash`.
- **Resolution slot** (the resolver's uniqueness constraint) is:
  `(source, package name, major-version line)` — at most one selected version per slot.
- Major version is a property of the *slot*, never a component of *identity*. `foo 1.2` and
  `foo 1.7` are different identities competing for the same slot.

### 1.2 Derivation is reference-level and conservative (v1 rule)

Any reference to a host-backed interface anywhere in a compiled package contributes that
interface's capabilities to the package's derived set — reachable or not, called or not.
No reachability, dead-code, or feature-conditional narrowing in v1. Rationale: precision
makes the derived set sensitive to compiler version and analysis detail, which poisons
determinism and turns capability diffs into noise. Precision may be added later only as a
*narrowing* refinement behind an explicit format-version bump; it can never be removed once
depended on.

### 1.3 Lockfile carries the configuration-independent record

- The lockfile records, per package: the **derived-capability union** across the resolved
  configuration space, the **declared envelope**, and **native origin** (which native
  providers, if any, contribute).
- Per-build (feature- and target-specific) derived sets are a build/report artifact
  (`stark tree --capabilities`, capability report), never lockfile content. Adding a CI
  target must not rewrite the lockfile.

### 1.4 The capability vocabulary is a versioned format

Initial vocabulary (v1):

```text
filesystem-read
filesystem-write
environment-read
network-client
network-listen
clock
randomness
process-execution
native-code
```

"Pure" is the *absence* of all capabilities, not a listed capability. The vocabulary carries
a version field from day one, with defined widening semantics: if a capability is later split
(e.g. `filesystem-read` → finer grains), an older declaration is interpreted as the union of
its successors. No capability is ever silently renamed or removed.

### 1.5 Honest limits are part of the model

- Derivation is authoritative **only** for pure STARK code (no unsafe, no FFI, no scripts —
  already guaranteed by the brief's rules 7 and the Core spec).
- For native providers, STARK can derive *which provider interfaces the STARK side invokes*,
  but cannot prove the native implementation stays within its declared boundary. Native
  packages therefore require the separate trust tier (§5) and their declared capabilities are
  trusted input to the closure, marked as such (`native-origin`).
- Declared-but-underived envelope entries are an audit signal, not a failure (they may serve
  optional features or other targets).

### 1.6 No arbitrary numeric policy defaults

No default `max-depth` or dependency-count limits. Graph-shape policies exist only as
application-configured controls with no shipped default thresholds.

---

## 2. Amendment A — Governing rules (brief §1.4)

The brief currently has 17 governing rules. This amendment **rewrites rule 9** and
**appends rules 18–20**.

**Rule 9, current text (line 98):**

> **Native code must be disclosed in package metadata.** Disclosure is metadata, not a safety
> claim.

**Rule 9, replacement:**

> **Native code is a separate trust tier and must be disclosed in package metadata.** A
> native-backed package discloses its provider identity, artifacts, hashes, targets, and
> declared capabilities, and requires explicit root-application consent. Disclosure is
> metadata, not a proof of runtime behaviour.

**New rules (appended):**

> 18. **Canonical package identity is `source + name + version + content hash`.** A package
>     name alone never identifies a package. Resolution selects at most one version per
>     `(source, name, major-version line)` slot; semver-incompatible majors may coexist but
>     must be visible in the lockfile, dependency tree, update summaries, and diagnostics.
>     Applications may deny multi-major coexistence globally or per package name. Silent
>     duplication, silent source substitution, and silent registry fallback are all wrong
>     implementations.
> 19. **For pure STARK packages, host capabilities are derived, not trusted.** The toolchain
>     computes each package's capability set from references to host-backed interfaces in the
>     resolved graph (reference-level, conservative). A manifest may declare a capability
>     envelope; the build fails if `derived ⊄ declared`. The root application approves the
>     derived transitive closure. Capability metadata is never merely informational once the
>     derivation mechanism (WP-P1.6) lands.
> 20. **Specify durable formats early; defer registry operations.** Anything encoded into
>     package identity, lockfiles, archives, or capability metadata is decided in P0–P1 under
>     escalations E1/E2/E7. Operational registry machinery (publisher identity, MFA,
>     typosquat defense, advisories, quarantine) is designed only at Gate P10.

Also add, immediately below the §1.4 heading, the governing statement:

> **Package-system invariants:** Package acquisition never executes code. A build resolves
> one visible identity per `(source, name, major)` slot. Every package byte is
> content-verified. For pure STARK packages, host capabilities are derived from the resolved
> graph rather than trusted from declarations. Every native component is a disclosed,
> consented trust tier.

---

## 3. Amendment B — §1.5 prohibitions clarification

The §1.5 prohibition on `capability syntax` (line 118) and the bullet "capabilities, effects,
labels, and typestate remain separate research proposals..." (line 129) **stand for the
language**. Add one clarifying sentence to the line-129 bullet so it cannot be read as
contradicting rule 19:

> Package-level capability *metadata and derivation* (rule 19, WP-P1.6) are package-tooling
> analysis over existing imports and add no language surface; the prohibition here is on Core
> *syntax and semantics* for capabilities, which remain a separate research proposal.

---

## 4. Amendment C — Escalation table (brief §2.3)

Append one row:

| ID | Decision | Reason |
|---|---|---|
| E7 | Capability vocabulary, derivation contract, and enforcement semantics | becomes a durable serialized format in manifests and lockfiles; enforcement changes what builds are rejected |

---

## 5. Amendment D — Gate/WP text changes

### 5.1 WP-P0.2 — Sources and manifest contract

- Line 481, current: "registry identity reserved for the later registry, following E0
  compatibility decisions." **Append:** "Registry/source identity is a component of canonical
  package identity (rule 18) and is designed into the lockfile format at E2 time, before any
  registry exists."
- Line 493, current: "capability metadata is informational until a separately approved
  enforcement mechanism exists." **Replace with:** "capability metadata follows the
  envelope/derived contract of rule 19 and WP-P1.6 (E7); until WP-P1.6 lands, manifests may
  carry declared envelopes, and no build enforcement occurs."
- Manifest additions: optional `declared-capabilities` (envelope, vocabulary-versioned);
  explicit alias syntax for intentional multi-source or multi-major coexistence (aliases must
  be visible in both manifest and lockfile).

### 5.2 WP-P0.3 — Restrictions and native disclosure

Append to the normative restrictions list:

- no silent dependency substitution (alias, fork, override, or source change invisible to
  manifest + lockfile);
- no capability entering the graph without appearing in the derived/declared record;
- a pure→native transition in any dependency release is a security-significant change and
  must surface in `stark lock diff` regardless of semver compatibility.

### 5.3 WP-P0.4 — Lockfile contract

Add to the "Define:" list:

- resolution-slot representation: selected version per `(source, name, major)` slot, with
  coexisting majors explicit;
- per-package capability record: derived union (configuration-independent), declared
  envelope, native-origin list (§1.3 of this proposal);
- capability-vocabulary version;
- toolchain version recorded **informationally** (reproducibility aid; not build-blocking by
  default — locked builds may opt into enforcing it);
- alias records for intentional coexistence.

### 5.4 WP-P1.2 — Identity, conflicts, and cache

Add to the "Cover:" list:

- one-version-per-`(source, name, major)` slot enforcement, with incompatible constraints
  inside one major failing resolution with both dependency paths shown;
- coexisting majors resolved, locked, and reported as distinct identities;
- `deny-multiple-majors` policy: global boolean and per-package-name list
  (`"deny-multiple-majors": true` or `["<package-name>", ...]` — package names, not
  categories);
- cross-major type-mixing diagnostic quality: when `foo@1::T` meets `foo@2::T`, the
  diagnostic names both versions and both dependency paths.

### 5.5 WP-P1.4 — Inspection commands

Extend the command list:

```text
stark tree --why <package>[@<major>]
stark tree --duplicates
stark tree --capabilities
stark tree --native
stark lock diff
stark package inspect <archive>
```

Required answers: why a package exists and which direct dependency introduced it; exact
source and content hash; derived vs. declared capabilities; native origin; duplicate majors
present; and, for `stark lock diff`: packages added/removed, versions changed, source or
registry changes, new capabilities, new native code, pure→native transitions.

### 5.6 New WP-P1.6 — Capability derivation model (E7)

Insert after WP-P1.4. WP-P1.5 (scale fixture and gate exit) absorbs it into exit evidence —
see §5.7. Deliverables:

```text
spec/packages/capabilities.md        vocabulary v1, versioning/widening rules
spec/packages/capability-derivation.md   reference-level conservative rule (§1.2)
```

- canonical capability vocabulary (§1.4 of this proposal) with version field;
- mapping table: approved host-backed interface → capability set (initially empty of
  entries — see timing note);
- transitive derived-closure computation over the resolved graph;
- `derived ⊆ declared-envelope` build enforcement; underived declarations reported as audit
  signals;
- application `capability-policy` (`allow` / `deny` lists) evaluated against the **derived
  transitive closure**;
- lockfile capability record per §5.3;
- deterministic capability report (stable ordering, byte-identical across two runs).

**Timing note (why P1, before any privileged package exists):** until Gate P4/P5 land, there
are no host-backed interfaces, so every package must derive to the empty set. That is the
cheapest possible environment to build and verify the machinery in — any nonempty derived set
before P4 is, by construction, a bug. The mapping table populates as P4/P5 introduce
providers and std packages.

### 5.7 WP-P1.5 — gate exit additions

Append to exit evidence:

7. duplicate-major fixture resolves, locks, and reports both majors visibly; the
   `deny-multiple-majors` policy rejects it with both paths named;
8. capability derivation produces the empty set for the entire (pure) fixture graph,
   deterministically, and the report/lockfile record round-trips.

### 5.8 WP-P2.1 — `stark update` behaviour

Amend: `stark update <dependency>` remains the primary form; add `--dry-run`, and require
every update to print, before applying: packages added/removed, versions changed, source
changes, new capabilities, new native code, and lockfile identity changes (same schema as
`stark lock diff`). Whole-graph update requires an explicit flag, never the bare command.

### 5.9 WP-P2.4 — Deterministic package archive

Append requirements:

- a deterministic inventory with a content hash for **every file** in the archive;
- manifest/archive content-mismatch rejection.

Append explicit rejection tests: `../` path traversal; absolute paths; symlinks escaping the
extraction root; duplicate archive paths; case-collision paths; decompression bombs;
oversized file counts; undeclared native assets; hidden executable payloads.

### 5.10 Gate P4 — native provider amendments

- **WP-P4.1 (E3):** the boundary spec must define, per provider interface, its **capability
  mapping** as part of the ABI contract. An interface without a capability mapping cannot be
  approved.
- **WP-P4.2:** wiring contributes each referenced provider interface's capabilities to the
  importing package's derived closure, marked `native-origin`; native-backed packages
  additionally require: explicit root-application consent, supported target list, per-artifact
  hashes, ABI version, provenance (prebuilt vs. reproducibly built), no undeclared dynamic
  loading, no downloading binaries at build/run time, no fallback to an unverified system
  library.
- **WP-P4.4 exit:** add "`stark tree --native` shows the complete native trust boundary of
  the validation fixture."

### 5.11 Gate P10 — no scope change (confirmation)

P10's existing design list already carries the early-invariant set (immutable archives,
content hashes, no code execution, provenance, signed snapshots where practical). This
proposal moves nothing forward out of P10 and adds nothing to it beyond what rule 18 already
fixes earlier (registry identity inside package identity, no silent registry fallback —
already present at WP-P1.2's "no silent fallback" line). Publisher identity, MFA, namespace
transfer, typosquat/dependency-confusion defense, advisories, quarantine, and reputation
remain P10-only, per rule 20.

---

## 6. Consequences and cross-track notes

### 6.1 What becomes mechanically testable

Each invariant maps to executable evidence: slot-uniqueness and coexistence-visibility
(WP-P1.2/P1.5 fixtures), acquisition-never-executes (WP-P0.3 fixtures, already present),
derived-⊆-declared enforcement and empty-set baseline (WP-P1.6), archive hardening
(WP-P2.4 rejection tests), native tier (WP-P4.4 exit). No invariant rests on convention.

### 6.2 What this deliberately does not claim

No runtime sandboxing. No proof about native implementations beyond their declared boundary.
No protection against a malicious *approved* capability use (a package granted
`network-client` can misuse it). The claim is small, visible, reproducible, auditable risk —
plus one categorical guarantee competitors lack: statically proven purity of pure packages.

### 6.3 Compiler-track interface (single deliverable request)

Capability derivation must not create a hard dependency on compiler-track gates. The
ecosystem side owns vocabulary, mapping, closure, policy, and reports. The compiler track's
only deliverable: **emit, per compiled package, the list of referenced host-backed interface
identities** (the resolver already knows every referenced item). This is a metadata-emission
request, to be scheduled by the compiler track's own governance (a small WP or a rider on an
existing one), not an ecosystem-track implementation inside `starkc`.

### 6.4 Consistency-sweep obligations when the brief is amended

When these amendments are applied: renumber/recount the governing rules (17 → 20) anywhere a
count appears; update the §2.3 escalation table (E1–E6 → E1–E7) and any prose citing "six
escalations"; re-verify line-anchored quotes in this proposal against the amended brief; and
confirm WP cross-references (new WP-P1.6, amended P1.5 exit) in any gate-summary tables.

---

## 7. Open questions for the owner (small, non-blocking)

1. **Alias syntax** for intentional multi-major/multi-source coexistence: manifest field
   shape is unspecified here; proposed to be settled inside WP-P0.2's manifest design under
   the existing E1 process.
2. **Toolchain version in lockfile:** recorded informationally with opt-in enforcement
   (§5.3) — confirm, or require hard enforcement in locked builds.
3. **Initial vocabulary ratification:** confirm the v1 capability list (§1.4) or amend before
   it becomes a serialized format at WP-P0.4.

## 8. Ratification mechanics

On approval: (a) record a dated decision entry (this proposal's ID) in the ecosystem track's
STATE.md decision log when WP-E0.0 bootstraps it — or, if ratified before bootstrap, cite
this file from the bootstrap decision log's first entries; (b) apply Amendments A–D to the
brief in one change, running the §6.4 consistency sweep in the same pass; (c) downstream WP
files inherit from the amended brief — no WP file may carry an invariant the brief does not.

**Applied 2026-07-18:** (b) done — all of Amendments A–D landed in
`STARK-Ecosystem-Build-Brief-Revised-Sonnet.md`; sweep confirmed no stale rule counts, no
stale escalation enumerations, no surviving "capability metadata is informational" text, and
all new cross-references (rules 18–20, E7, WP-P1.6) resolve to real content. (a) remains
outstanding by design: the ecosystem STATE.md does not exist yet — WP-E0.0's bootstrap must
cite this file in its opening decision-log entries. (c) is a standing obligation on all
future ecosystem WP files.
