> **ARCHIVED — SUPERSEDED 2026-08-03.** This roadmap was consolidated into the single
> forward plan at `ROADMAP.md` (repository root). It is retained for provenance and for
> the citations that point at it; **do not schedule work from it**. Where it disagrees
> with `ROADMAP.md`, `ROADMAP.md` wins.
>
> Former path: `STARKLANG/docs/compiler/work-packages/WP-PKG-OPS-ROADMAP.md`

---

# WP-PKG-OPS-ROADMAP — Identity and Operability Layer

**Status:** Proposed execution roadmap — companion to `WP-PKG-ROADMAP`
**Prepared:** 2026-07-31
**Repository:** `navraj007in/stark`
**Scope:** Packages and provider surfaces omitted from `WP-PKG-ROADMAP` that are prerequisites for its own milestones
**Relationship:** This document does not reorder `WP-PKG-ROADMAP`. It inserts work into its waves and amends its milestones. Where the two disagree, the merge in §12 is authoritative.

---

## 1. Objective

`WP-PKG-ROADMAP` covers the connective tissue of the ecosystem — data formats, networking, HTTP. What it under-covers is the **identity and operability layer**: content hashing, standard streams, clean shutdown, buffered reading, logging, and argument parsing. These are the packages that separate "a demo that serves a request" from the production-shaped claim Milestone C already makes.

Every item here satisfies at least one of:

1. an existing roadmap item **names it as a dependency without scheduling it** (UUID v3/v5 defers "until hash packages exist"; no hash package exists in any wave);
2. an existing **milestone deliverable cannot be built without it** (a log analyser that cannot read stdin; a server that cannot be stopped);
3. it is a **stated commitment of the wider programme** (deterministic lockfile resolution requires content addressing; `stark verify`'s thesis *is* artifact hashing).

The same disciplines as the parent roadmap apply: native artefact execution over build-only evidence, no unclassified ignores, exclusions stated rather than implied, and package work forbidden from silently becoming compiler redesign.

---

## 2. Status vocabulary

Inherited unchanged from `WP-PKG-ROADMAP` §2 (`READY`, `READY_BOUNDED`, `PROVIDER_EXPANSION`, `LANGUAGE_BLOCKED`, `DEFER`; complexity S/M/L/XL).

One addition:

| Status | Meaning |
|---|---|
| `AUDIT` | Not a package. A bounded question about existing Core/stdlib behaviour that must be answered before a dependent qualification freezes evidence. |

---

# PART A — Identity

## 3. WP-PKG-SHA2 — `stark-sha2` v0.1

**Priority:** P0
**Complexity:** S–M
**Status:** `READY` — pure STARK
**Type:** Pure STARK
**Dependencies:** Byte/vector operations, wrapping integer arithmetic, `stark-hex` for formatting
**Blocks:** UUID v3/v5, HMAC, content addressing, lockfile integrity, any future `stark verify` self-hosting

### Why P0, not a Wave 4 security item

This is the single omission in the parent roadmap that is essential rather than convenient:

- `WP-PKG-UUID1` §8 defers namespace UUIDs "until hash packages exist" — and then no wave contains one;
- the package manager's deterministic-resolution promise requires content-addressed dependency pinning, which is hashing;
- git-dependency verification (P3 of the ecosystem track) requires artifact digests;
- the programme's north star — "verified programs bound to external artifacts" — **is** content hashing. An ecosystem that cannot compute SHA-256 in its own language outsources its founding primitive.

It is also an ideal conformance workload: fixed official vectors, heavy rotate/shift/wrapping-add traffic over exactly the integer semantics C7.9 hardened, and bit-identical output required across all three engines and all Tier-1 platforms.

### Scope

- SHA-256 and SHA-512, one-shot over `&[UInt8]`;
- incremental state object (`Sha256State::new() → update(&[UInt8]) → finalize() -> [UInt8; 32]`) — required, not optional, because file hashing must not demand whole-file buffering;
- digest as fixed-size byte array; hex rendering via `stark-hex`, never an internal formatter;
- chunked-versus-one-shot equivalence tested at every boundary offset (0, 1, 55, 56, 63, 64, 65 bytes and block multiples).

### Rules

- constants and schedule from FIPS 180-4; no novelty anywhere;
- wrapping arithmetic must be **spelled** wrapping — if Core's default integer semantics trap on overflow, the implementation uses the explicit wrapping operations and the package documents that dependency;
- no truncated variants (SHA-224/384) in v0.1 unless a consumer names one;
- no claim of resistance to timing side channels — this is an integrity primitive, and the README says so.

### Deliberately absent

- SHA-1 and MD5, even "for compatibility" — absent from the namespace entirely rather than present-and-discouraged;
- SHA-3/BLAKE families until a consumer exists;
- password hashing (argon2/bcrypt class) — different threat model, different package, different decade of this project.

### Exit criteria

- FIPS 180-4 and NIST CAVP vectors, including the empty input, the 448-bit and 896-bit boundary messages, and a ≥1 MiB input;
- three-engine agreement on digests (HIR, MIR, native);
- incremental/one-shot equivalence at all block boundaries;
- native consumer hashing a real file via `stark-io`, output compared against a platform reference digest;
- Tier-1 CI on the exact commit.

---

## 4. WP-PKG-HMAC1 — `stark-hmac` v0.1

**Priority:** P2
**Complexity:** S
**Status:** Blocked on `stark-sha2`
**Type:** Pure STARK
**Dependencies:** `stark-sha2` incremental interface

### Scope

- HMAC-SHA256 per RFC 2104 over the incremental hash interface;
- keys longer than block size hashed first, exactly as specified;
- RFC 4231 test vectors in full.

### Rules

- generic over hash only if the trait falls out naturally; a concrete `hmac_sha256` is acceptable for v0.1;
- constant-time comparison for MAC verification is **required** — a `verify(expected, computed) -> Bool` that short-circuits on first mismatch is the one place this package's threat model is real. If constant-time comparison is not cleanly expressible, verification is omitted from v0.1 and the README says callers must not compare MACs with `==`.

### Exit criteria

RFC 4231 vectors; three-engine agreement; Tier-1 CI. No native provider involved.

---

## 5. WP-PKG-CADDR1 — Content addressing utility

**Priority:** P2
**Complexity:** S
**Status:** Blocked on `stark-sha2`, `stark-io`
**Type:** Thin composition package (`stark-digest` or a `stark-checksum` v0.2 extension — owner's naming call)

### Scope

- `digest_file(path) -> Result<[UInt8; 32], _>` streaming through the incremental interface with a fixed chunk size;
- `digest_string`, `digest_bytes` conveniences;
- canonical lowercase-hex rendering, one spelling, frozen — this string format is what lockfiles and gate evidence will embed, so it is an interchange format from day one and versioned as such.

### Exit criteria

- file digest agrees with one-shot digest of the same bytes;
- output byte-identical across Tier-1 platforms for CRLF-free fixture files;
- the canonical rendering documented as frozen.

---

# PART B — Standard streams and operability

## 6. WP-PKG-STDIO1 — Standard streams

**Priority:** P0
**Complexity:** S–M
**Status:** `PROVIDER_EXPANSION` — small, over the existing io provider
**Type:** Host-backed; recommended home is `stark-io` v0.2 rather than a new package
**Blocks:** Milestone A's log/file analyser; every pipeline-composable CLI

### Why P0

Milestone A promises Unix-shaped tools, and nothing in either roadmap can read standard input. `cat access.log | stark-analyse` is the composition test the CLI foundation exists to pass; without stdin the milestone produces tools that only open files they are told about, which is a different and lesser claim.

### Scope

- `stdin_read(buf) -> Result<UInt64, IoError>` — bounded read, 0 = EOF;
- `stdout_write(&[UInt8])`, `stderr_write(&[UInt8])` — with explicit partial-write semantics matching `stark-net`'s, plus `write_all` loops in the package layer;
- `stdout_flush()`;
- `is_terminal(stream) -> Bool` if the provider can answer it cheaply; otherwise deferred and stated.

### Design decisions to pin before implementation

1. **Streams are not `HostResource`s in v0.1.** The three standard streams are process-ambient: not opened by the program, not closed by it, and A11's exactly-once close discipline does not apply. They cross the ABI as capability functions, not handles. Recording this explicitly prevents the alternative from being reinvented later without noticing it creates three resources whose "close" is undefined.
2. Bytes, not text, at the provider boundary. UTF-8 decoding happens in package code with the same reported-never-lossy policy as `stark-env`.
3. Interaction with the existing `print`/`println` Core path must be stated: same underlying stream, and interleaving order between Core prints and package writes is observable and therefore specified (both unbuffered at the OS write level, or the buffering rule is documented).

### Exit criteria

- native consumer that reads stdin to EOF, transforms, writes stdout, reports on stderr — exercised in CI by actually piping bytes in and comparing bytes out, all three Tier-1 platforms;
- EOF, zero-length reads, and partial writes observed, not assumed;
- Windows CRLF behaviour pinned (binary mode; no implicit translation);
- Core print interleaving rule documented and tested.

---

## 7. WP-PKG-BUFIO1 — `stark-bufio` v0.1

**Priority:** P1
**Complexity:** S–M
**Status:** `READY` once STDIO1 lands (works over files alone even before it)
**Type:** Pure STARK over existing read surfaces
**Blocks:** line-oriented Milestone A tools

### Scope

- `BufReader` over anything exposing the bounded-read shape (`NativeFile`, stdin), fixed buffer capacity chosen at construction;
- `read_line(&mut String) -> Result<UInt64, _>` — bytes consumed including terminator; 0 = EOF;
- `lines()`-style iteration **only if** the ownership rules allow it cleanly — CD-292's finding that a `Vec` of owning structs is reachable only through `.iter()` makes iterator-returning APIs a known sharp edge; if it fights the borrow rules, ship `read_line` alone and record the ergonomic pressure as evidence for the RB0-adjacent ergonomics item rather than contorting the API;
- `BufWriter` with explicit `flush`, and a documented rule for buffered data at drop (v0.1: **dropped unflushed data is lost, by design** — a destructor has nowhere to put an error, exactly as CD-291 established for `file_close`).

### Design boundary

Line splitting is byte-oriented (`\n`, with `\r\n` tolerated and stripped when the option is set). No charset detection, no BOM handling, no locale.

### Exit criteria

- files with no trailing newline, empty lines, lines longer than the buffer, and CRLF/LF mixtures;
- equivalence: reading a fixture through `BufReader` yields byte-identical content to a one-shot read;
- a line-counting native consumer over both a file and stdin;
- Tier-1 CI.

---

## 8. WP-PKG-SIGNAL1 — Interrupt and shutdown

**Priority:** P1
**Complexity:** M
**Status:** `PROVIDER_EXPANSION` — **SPEC FIRST, mandatory**
**Type:** Host-backed
**Blocks:** any honest claim that Milestone B's server is operable

### Why this cannot be skipped

Milestone B ships a TCP/HTTP server, and neither roadmap gives a program any way to observe Ctrl-C. The only shutdown story is the OS killing the process — which also makes A11's close guarantees vacuous at exactly the moment they matter: on SIGKILL nothing closes; on unhandled SIGINT the runtime's default abort path either runs the close arena or does not, and today that behaviour is **unspecified rather than decided**.

### The spec must answer, before any code

1. **Poll, don't preempt.** v0.1 is a `shutdown_requested() -> Bool` flag the program polls at its loop head — set by SIGINT/SIGTERM (Unix) and console control events (Windows). No callbacks, no handlers running STARK code at signal time, no async signal delivery. This is the entire mechanism; everything richer is deferred.
2. What the runtime does on a **second** interrupt while the program is draining (recommended: default OS behaviour — immediate termination — so an unresponsive program remains killable).
3. What happens to live `HostResource`s on unhandled fatal signals — stated as *no closes run; providers must tolerate OS-level reclamation* — so the A11 guarantee is scoped honestly to orderly execution rather than silently overclaimed.
4. Whether the flag capability appears in the manifest like any other (`signals`), so a program that never asks cannot observe it — recommended yes, for uniformity.

### Exit criteria

- native consumer: accept-loop server that exits its loop, closes its listener through ordinary drop, and prints a drain marker after receiving SIGINT — driven by CI actually sending the signal on all three platforms (Windows via console ctrl event);
- second-interrupt behaviour observed;
- the fatal-signal resource statement added to the A11 documentation, cross-referenced from the closure matrix.

---

## 9. WP-PKG-LOG1 — `stark-log` v0.1

**Priority:** P1
**Complexity:** S
**Status:** `READY` once STDIO1 lands
**Type:** Pure STARK over stderr

### Scope

- levels: `Error`, `Warn`, `Info`, `Debug`;
- a `Logger` value constructed explicitly and passed where needed — **no global logger, no ambient state**; the roadmap's own rule against hidden global RNG state applies with equal force here;
- line format frozen and documented: `LEVEL timestamp message` with the timestamp from `stark-time`'s qualified wall clock, or omitted (a `Logger` option) for deterministic test output;
- output to stderr through the STDIO1 surface; a file sink can arrive in v0.2 via `stark-io` without changing the API.

### Deliberately absent

- structured/JSON logging (revisit when a consumer parses logs, not before);
- log rotation, filtering DSLs, per-module levels;
- any macro/derive surface — that is the serde-class tooling decision and stays behind its gate.

### Exit criteria

- deterministic-mode output byte-compared in tests;
- level filtering observed;
- interleaving with direct stderr writes specified;
- one Milestone A tool converted to it as the consumer proof.

---

## 10. WP-PKG-ARGS1 — `stark-args` v0.1

**Priority:** P1
**Complexity:** S–M
**Status:** `READY` — pure over `stark-env`
**Type:** Pure STARK

### Why it belongs in the plan

Not essential the way SHA-2 is — but all four Milestone A deliverables otherwise hand-roll flag parsing four times, and hand-rolled parsers are where CLI tools grow their inconsistencies. Highest leverage-to-effort ratio in this document after stdio.

### Scope

- long flags (`--verbose`), long options with values (`--limit 10` and `--limit=10`);
- short flags (`-v`) without clustering in v0.1;
- positional arguments, ordered;
- `--` terminator ending flag parsing;
- unknown-flag and missing-value errors naming the exact token;
- generated usage text from the declared surface.

### Deliberately absent

- subcommands (v0.2 — the checksum CLI does not need them);
- clustering (`-abc`), negation (`--no-x`), environment-variable fallbacks, config-file merging.

### Exit criteria

- table-driven vectors including every error path;
- the checksum CLI and one other Milestone A tool ported to it;
- usage text golden-tested;
- Tier-1 CI (pure — the qualification burden is small).

---

# PART C — Audits

## 11. WP-AUDIT-FLOATFMT — Float formatting fidelity

**Priority:** P0 — **sequenced before `WP-PKG-Q1` freezes JSON evidence**
**Complexity:** S (the audit; the fix, if needed, is a separate decision)
**Status:** `AUDIT`

### The question

Does Core produce **shortest round-trip** float formatting (ryu/Grisu class), such that `parse(format(x)) == x` for every finite `Float64`? Or does `stark-json`'s compact encoder emit floats through something ad hoc?

If the latter, JSON round-trip fidelity is quietly broken for a set of values no spot-check will find, and `WP-PKG-Q1` would freeze qualification evidence over a defective encoder — an over-claim of exactly the shape C7.2 was corrected for.

### Required actions

1. locate the float→string path used by `stark-json` and by Core `Display`;
2. property-test round-trip over: powers of two near the subnormal boundary, `0.1`-class repeating fractions, `Float64::MAX`, values differing in the final ULP, and a randomized corpus ≥10⁶ values per engine;
3. pin NaN/Infinity policy in JSON (reject on encode, per RFC 8259 — or state the deviation);
4. verdict recorded as a CD: either *round-trip holds, evidence attached* or *defect, with scope* — in which case Q1 admits `stark-json` with floats excluded from the claim, or waits.

### Exit criteria

The CD exists, Q1's JSON row cites it, and no qualification claim covers float encoding without it.

---

# PART D — Explicitly rejected

Recorded so their absence is a decision, not an oversight:

| Item | Ruling |
|---|---|
| Compression (gzip/deflate) | Rejected for now. HTTP can decline content-encoding indefinitely; no other consumer exists. Revisit only when a named consumer cannot proceed. |
| SHA-1 / MD5 | Rejected outright, including "for git compatibility" — git-dependency verification pins commits by ID through the host git provider; the ecosystem's own digests are SHA-256. |
| Full regex engine | Parent roadmap's deferral affirmed; glob first, combinators maybe, engine only behind a complexity-bound spec. |
| Password hashing | Out of scope for the foreseeable programme. |
| Async signal handlers running STARK code | Rejected for v0.1 by SIGNAL1's poll-only rule; anything richer waits for the concurrency gate. |

---



# PART F — Cryptography

## 16. Governing crypto-track decision

**Status:** `APPROVED`  
**Applies to:** hashing, MAC, randomness, AEAD, signatures, KDFs, key agreement, and TLS

```text
APPROVED:

- Split packages; no stark-crypto façade.
- stark-sha2 is the only immediate GO crypto implementation.
- Secret-bearing key packages use typed A11 affine resources.
- Default AEAD owns nonce generation inside the provider.
- Verification returns Result<(), UndifferentiatedFailure>.
- HKDF is labelled only for high-entropy input material.
- Key-backed crypto has a native-only Tier-N qualification.
- FIPS requirement is CRYPTO0 decision number one.
- TLS uses a complete provider and is never assembled from package primitives.
- Pure public-key or cipher implementations, if built later, are conformance/research workloads rather than the production security path.
- Seal-key binding mode and VerificationContext resource status must be resolved by CRYPTO0 before any API freezes.
```

### Canonical package split

| Package | Role | Qualification |
|---|---|---|
| `stark-sha2` | SHA-256 initially | Tier P |
| `stark-hmac` | HMAC-SHA256 | Tier P for tag computation |
| `stark-random` | deterministic PRNG plus secure random bytes | Tier P / Tier N split |
| `stark-aead` | authenticated encryption | Tier N |
| `stark-signature` | signing and verification | Tier N |
| `stark-kdf` | high-entropy key derivation | Tier N unless separately admitted pure |
| `stark-key-agreement` | key agreement | Tier N |
| `stark-tls` | complete TLS stack | Tier N |

No umbrella façade duplicates these APIs. One operation has one public spelling.

---

## 17. Qualification tiers

### Tier P — Pure differential qualification

Required evidence:

```text
HIR interpreter
MIR interpreter
native executable
three-engine agreement
Linux x64
macOS arm64
Windows x64
authoritative vectors
```

Applies initially to:

- `stark-sha2`;
- HMAC tag computation;
- canonical crypto parsing and encoding helpers.

### Tier N — Native provider qualification

Required evidence:

```text
native execution
Linux x64
macOS arm64
Windows x64
authoritative vectors where applicable
provider-level deterministic test backend
negative and corruption cases
resource-lifecycle evidence
provider identity and version evidence
```

Applies to:

- secure randomness;
- AEAD;
- secret-key import/generation;
- signature generation;
- provider-side verification;
- provider-backed KDFs;
- TLS.

HIR and MIR must reject Tier-N programs as unavailable capabilities. They must not appear as failed differential rows or reduce pure-language conformance statistics.

---

## 18. WP-PKG-CRYPTO0 — Crypto profile and provider selection

**Priority:** P2  
**Complexity:** M  
**Status:** `SPEC FIRST`  
**Blocks:** AEAD, signatures, provider-backed KDF, key agreement, TLS profile selection

No key-bearing crypto API may freeze before CRYPTO0 closes.

### Mandatory decision order

1. FIPS-oriented versus modern-general-purpose profile.
2. Provider and validated-module boundary.
3. Tier-N qualification rules and unavailable-capability diagnostics.
4. Typed secret-resource inventory.
5. Seal-key ABI mode.
6. Verification-key representation.
7. Coupled AEAD algorithm and nonce policy.
8. Signing determinism and per-signature nonce policy.
9. Import, export and key-generation policy.
10. Zeroisation and destruction guarantees.
11. Public failure-collapse rules.
12. Frozen wire formats, limits and provider identity.

### 18.1 FIPS decision first

The first question is whether FIPS-approved operation is required for the first supported profile.

#### Profile F — FIPS-oriented

```text
Hash:       SHA-256
MAC:        HMAC-SHA256
AEAD:       AES-256-GCM
KDF:        HKDF-SHA256
Signature:  algorithm supported in the selected validated module and mode
Random:     validated-module entropy/DRBG path
```

#### Profile M — Modern general-purpose

```text
Hash:          SHA-256
MAC:           HMAC-SHA256
AEAD:          XChaCha20-Poly1305, or ChaCha20-Poly1305 with enforced limits
Signature:     Ed25519
Key agreement: X25519
KDF:           HKDF-SHA256
Random:        OS CSPRNG
```

The chosen profile, provider, mode, version, platform and build identity form part of the artefact identity.

### 18.2 Typed affine secret resources

```stark
pub resource AesGcmKey;
pub resource XChaCha20Poly1305Key;
pub resource Ed25519SigningKey;
pub resource HmacSha256Key;
```

Rules:

- non-Copy;
- non-Clonable unless an explicit provider operation permits it;
- non-exportable by default;
- moved rather than duplicated;
- destroyed exactly once through MIR-owned drop;
- provider zeroisation on close;
- algorithm- and purpose-specific types;
- no generic `SecretKey`.

Private-key export, where required, is a separately admitted capability and never an incidental `bytes()` method.

### 18.3 Seal-key ABI mode

The current provider ABI supports:

- `HandleBorrowed`;
- `HandleConsumed`;
- `HandleOut`.

It does not currently support mutable-borrow handles.

CRYPTO0 must choose:

#### Option A — Shared borrow with provider-internal mutation

```stark
pub fn seal(
    key: &AeadKey,
    plaintext: &[UInt8],
    associated_data: &[UInt8],
) -> Result<SealedMessage, AeadError>;
```

This requires documented internal mutability and provider-side synchronization.

#### Option B — New mutable-handle ABI mode

This is a compiler/ABI change sequenced behind `WP-C7.8-RB0`, with manifest, lowering, verifier, backend and aliasing qualification.

No `&mut AeadKey` API may freeze while the manifest cannot represent or enforce it.

### 18.4 Verification-key representation

Public keys, signatures, digests and nonces remain byte-backed values by default.

```stark
pub struct VerificationKey {
    bytes: Vec<UInt8>,
}
```

A provider-backed parsed-key resource is admitted only where validated parsing or retained native state provides material value. It must be named accordingly, for example `ParsedVerificationKey`, and is an opaque cache/handle rather than a secret-resource claim.

### 18.5 Coupled AEAD and nonce policy

Algorithm and nonce policy are selected together.

#### ChaCha20-Poly1305

- 96-bit nonce;
- random nonces require an enforced per-key usage bound or another uniqueness guarantee;
- unrestricted random generation is not admitted.

#### XChaCha20-Poly1305

- 192-bit nonce;
- provider-generated random nonces are the preferred default where supported.

#### AES-GCM

- strict nonce uniqueness;
- provider-managed uniqueness or the selected validated module's approved policy;
- exact per-key and per-nonce limits inherited from the selected module/profile.

The default v0.1 API does not accept a caller-supplied nonce.

```stark
pub struct SealedMessage {
    pub nonce: Vec<UInt8>,
    pub ciphertext: Vec<UInt8>,
}

pub fn seal(
    key: &AeadKey,
    plaintext: &[UInt8],
    associated_data: &[UInt8],
) -> Result<SealedMessage, AeadError>;

pub fn open(
    key: &AeadKey,
    message: &SealedMessage,
    associated_data: &[UInt8],
) -> Result<Vec<UInt8>, AuthenticationFailed>;
```

Caller-supplied nonce APIs may be admitted later only for named interoperability requirements.

### 18.6 Verification failure shape

```stark
pub fn verify(
    key: &VerificationKey,
    message: &[UInt8],
    signature: &Signature,
) -> Result<(), SignatureInvalid>;
```

```stark
pub fn verify_mac(
    key: &HmacSha256Key,
    message: &[UInt8],
    tag: &MacTag,
) -> Result<(), AuthenticationFailed>;
```

Rules:

- no `Result<Bool, CryptoError>`;
- callers proceed only on `Ok(())`;
- malformed input and mismatch collapse where distinguishing them has no safe application value;
- pure-STARK HMAC may compute tags but must not claim constant-time verification;
- timing-sensitive verification belongs provider-side.

### 18.7 HKDF scope

HKDF is only for key derivation from high-entropy input keying material.

Explicit exclusions:

- password hashing;
- deriving encryption keys directly from user passwords;
- password storage;
- password authentication.

### 18.8 Signing determinism

Where ECDSA is selected, CRYPTO0 must pin:

- random versus deterministic per-signature nonce generation;
- selected-module support for RFC 6979 where relevant;
- approved-mode compatibility;
- failure behaviour when entropy is unavailable;
- deterministic provider fixtures for tests.

No determinism or FIPS claim is inferred from the algorithm name alone.

### 18.9 Zeroisation and destruction

For each secret resource CRYPTO0 records:

- provider object owning the secret;
- whether secret bytes ever enter STARK memory;
- close function and resource tag;
- zeroisation guarantee on close;
- panic/abort behaviour;
- orderly drop behaviour;
- fatal-signal limitations;
- OS reclamation limitations;
- memory-locking claim or explicit non-claim.

A11 exactly-once close applies to orderly execution, not SIGKILL, abort or machine failure.

---

## 19. WP-PKG-SHA2 — `stark-sha2` v0.1

**Priority:** P0  
**Complexity:** M  
**Status:** `READY`  
**Qualification:** Tier P

### Scope

- SHA-256 one-shot;
- SHA-256 incremental state;
- fixed 32-byte digest;
- rendering through `stark-hex`;
- native file-hashing consumer through `stark-io`.

Deferred to v0.2:

- SHA-512;
- SHA-384;
- SHA-224.

### Exit criteria

- authoritative SHA-256 vectors;
- empty input;
- 448-bit padding boundary;
- block boundaries;
- input of at least 1 MiB;
- incremental/one-shot equivalence;
- HIR/MIR/native agreement;
- file digest compared against a platform reference;
- Tier-1 CI on one exact commit.

This remains the only immediate `GO` crypto implementation.

---

## 20. WP-PKG-HMAC1 — `stark-hmac` v0.1

**Priority:** P2  
**Complexity:** S  
**Status:** Blocked on `stark-sha2`

### Scope

- HMAC-SHA256;
- RFC 4231 vectors;
- correct handling of keys longer than the SHA-256 block size;
- Tier-P tag computation;
- provider-side Tier-N verification only where a timing claim is required.

---

## 21. Crypto roadmap insertion

### Wave 1

- `stark-sha2 v0.1` — SHA-256 only, Tier P.

### Wave 2

- `stark-random`;
- `stark-hmac`;
- `WP-PKG-CRYPTO0`.

### Wave 3

- `stark-aead`;
- `stark-signature`;
- provider-backed `stark-kdf`;
- optional `stark-key-agreement`.

### Wave 4

- `stark-tls`, independently provider-backed as a complete protocol implementation.

Pure public-key or cipher implementations may later be research/conformance workloads, never the production security path.


# PART E — Integration

## 12. Merged wave order (authoritative)

Amendments to `WP-PKG-ROADMAP` §31, inserted items in **bold**:

### Wave 0 — Close what exists
1. **`WP-AUDIT-FLOATFMT`** — before Q1 freezes JSON evidence
2. `WP-PKG-Q1` — Tier-1 qualification (JSON row cites the float CD)
3. `WP-PKG-Q2` — close `stark-io`
4. `WP-PKG-Q3` — qualify `stark-time`

### Wave 1 — CLI foundation
5. `stark-env v0.1`
6. **`stark-sha2 v0.1`**
7. **`stark-io v0.2` — standard streams (STDIO1)**
8. **`stark-bufio v0.1`**
9. `stark-uuid v0.1` parse/format
10. `stark-checksum v0.1`
11. **`stark-args v0.1`**
12. `stark-path v0.1`
13. `stark-semver v0.1`
14. **`stark-log v0.1`**

### Wave 2 — Connected foundation
15. **`WP-PKG-SIGNAL1` spec, then implementation** — must close before Milestone B claims an operable server
16. `stark-random v0.1`
17. **`stark-hmac v0.1`**, **content addressing (CADDR1)**
18. UUID v4/v7 generation
19. `stark-net` completion
20. `stark-http-core`, `stark-http-parser`
21. `stark-csv`

Waves 3–5 unchanged, except: **DNS, process spawning, HTTP parser security profile, TLS, and SIGNAL1 share the SPEC FIRST discipline** — the parent roadmap's §36 list gains `signal/shutdown provider`.

## 13. Amended milestones

**Milestone A — STARK CLI Foundation** additionally requires: stdio, bufio, sha2, args, log. Amended deliverables: the checksum CLI grows a `--sha256` mode and accepts stdin; the log analyser reads stdin or a file through `BufReader`; every tool shares `stark-args` and reports through `stark-log`. The composition test is explicit: **each tool must run correctly in a shell pipeline.**

**Milestone B — Connected STARK** additionally requires: SIGNAL1. Amended deliverable: the HTTP server **shuts down cleanly on SIGINT**, draining and closing through ordinary drop — observed in CI, not described in a README.

**Milestone C — Platform STARK** additionally requires: HMAC, content addressing. Amended deliverable: an artifact-digest tool producing the frozen canonical rendering — the first self-hosted step toward `stark verify` computing its own bindings.

## 14. Owner decisions requested

```text
GO (no further architecture needed):
- WP-AUDIT-FLOATFMT
- stark-sha2 v0.1
- stark-io v0.2 stdio surface
- stark-bufio v0.1
- stark-args v0.1
- stark-log v0.1

SPEC FIRST (bounded work package before code):
- signal/shutdown provider (SIGNAL1 §8 questions 1–4)

NAMING CALLS:
- content addressing: stark-digest vs stark-checksum v0.2
- stdio: stark-io v0.2 vs separate stark-stdio (recommendation: stark-io v0.2)

RECORD AS REJECTED (Part D table → decision log)
```

## 15. Summary table

| Order | Item | Priority | Complexity | Status | Blocks |
|---:|---|---|---|---|---|
| 1 | Float formatting audit | P0 | S | AUDIT | Q1's JSON claim |
| 2 | `stark-sha2` | P0 | S–M | READY | UUID v3/v5, HMAC, lockfile, verify thesis |
| 3 | stdio (`stark-io` v0.2) | P0 | S–M | PROVIDER_EXPANSION | Milestone A pipelines |
| 4 | `stark-bufio` | P1 | S–M | READY after stdio | line-oriented tools |
| 5 | `stark-args` | P1 | S–M | READY | CLI consistency |
| 6 | `stark-log` | P1 | S | READY after stdio | Milestone C's "production-shaped" |
| 7 | SIGNAL1 spec + impl | P1 | M | SPEC FIRST | Milestone B operability, A11 scope honesty |
| 8 | `stark-hmac` | P2 | S | Blocked on sha2 | future HTTP auth |
| 9 | Content addressing | P2 | S | Blocked on sha2+io | lockfile, verify self-hosting |
