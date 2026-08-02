# WP-PKG-HOST-CAPABILITIES — Remaining Host Capability Roadmap

**Status:** Proposed execution work package  
**Prepared:** 2026-07-31  
**Repository:** `navraj007in/stark`  
**Scope:** Secure randomness, standard streams, buffered I/O, signal handling, DNS, and process spawning  
**Relationship:** Companion to `WP-PKG-ROADMAP` and `WP-PKG-OPS-ROADMAP`

---

## 1. Objective

Complete the next host-capability layer using the provider machinery already proven by environment access, file I/O, time, and TCP.

The significance of this work is architectural:

> New host capabilities are now primarily package/provider work rather than compiler campaigns.

Each capability should follow the same repeatable path:

```text
manifest capability
provider ABI symbols
native provider implementation
STARK package
source-level consumer
observed-behaviour end-to-end test
Tier-1 qualification
```

The remaining differences are:

1. whether the capability is **function-shaped** or **resource-shaped**;
2. how much specification is required before implementation;
3. whether qualification is Tier P or Tier N.

---

## 2. Governing ruling

```text
APPROVED:

- Random first; function-shaped and smallest.
- Standard streams second; ambient functions, not resources.
- Buffered I/O immediately after stdio.
- Signals remain SPEC FIRST with poll-only delivery.
- DNS uses the host resolver and bounded snapshot output.
- Process spawning remains last and receives a dedicated lifecycle/capture specification.
- Every capability follows the established provider-manifest-package-native-consumer-Tier-1 pattern.
```

Recommended execution order:

1. `stark-random`
2. standard streams in `stark-io`
3. `stark-bufio`
4. signal/shutdown specification and provider
5. DNS resolver
6. process spawning

---

## 3. Qualification vocabulary

### Tier P — Pure differential qualification

Required:

- HIR interpreter;
- MIR interpreter;
- native executable;
- three-engine agreement;
- Linux x64;
- macOS arm64;
- Windows x64.

Applies here to:

- deterministic PRNG logic;
- buffered I/O logic where the source/sink can be injected;
- argument-independent parsing and encoding helpers.

### Tier N — Native provider qualification

Required:

- native execution;
- Linux x64;
- macOS arm64;
- Windows x64;
- deterministic provider test backend where practical;
- negative and boundary cases;
- provider identity/version evidence;
- resource-lifecycle evidence for resource-shaped capabilities.

Applies here to:

- secure randomness;
- standard streams;
- signals;
- DNS;
- process spawning.

HIR and MIR must reject unavailable Tier-N capabilities explicitly rather than counting them as failed differential rows.

---

# PART A — Secure randomness

## 4. WP-PKG-RANDOM1 — `stark-random` v0.1

**Priority:** P0  
**Complexity:** S–M  
**Status:** `READY`  
**Shape:** Function-shaped  
**Qualification:** Tier P for deterministic PRNG; Tier N for secure randomness  
**Blocks:** CRYPTO0, UUID v4, provider-owned AEAD nonces

### 4.1 Provider surface

Recommended native symbol:

```text
stark_random_secure_fill
```

Conceptual provider binding:

```stark
provider_api fn secure_fill(
    output: &mut [UInt8],
) -> Result<(), RandomError>;
```

The provider guarantees one of:

- the full requested buffer is filled;
- the operation fails and no output is treated as valid.

No partial-success public result is admitted.

### 4.2 Native implementation

Use the operating system CSPRNG through a mature abstraction such as the native `getrandom` implementation.

Platform expectations:

- Linux: OS entropy source;
- macOS: OS entropy source;
- Windows: system cryptographic RNG.

No deterministic fallback is permitted.

### 4.3 Package surface

```stark
pub enum RandomError {
    Unavailable,
    LimitExceeded,
    Other,
}

pub fn secure_bytes(
    count: UInt64,
) -> Result<Vec<UInt8>, RandomError>;
```

The package enforces allocation limits before allocating.

### 4.4 Deterministic PRNG

The deterministic half is pure STARK and separate from secure randomness.

Requirements:

- exact frozen algorithm;
- explicit seed;
- reproducible across engines and platforms;
- no hidden global state;
- never described as cryptographically secure.

### 4.5 Exit criteria

- secure fill works on all Tier-1 platforms;
- zero-length request defined and tested;
- maximum request bound tested;
- provider failure produces no usable output;
- deterministic PRNG sequences agree across HIR, MIR, and native;
- native consumer demonstrates UUID/nonces prerequisites;
- exact qualifying commit recorded.

---

# PART B — Standard streams

## 5. WP-PKG-STDIO1 — Standard streams in `stark-io` v0.2

**Priority:** P0  
**Complexity:** S–M  
**Status:** `READY_BOUNDED`  
**Shape:** Function-shaped  
**Qualification:** Tier N  
**Blocks:** Pipeline-composable CLI tools and `stark-bufio`

### 5.1 Ownership ruling

Standard input, output, and error are process-ambient capabilities.

They are:

- not opened by the program;
- not owned by the program;
- not closed by the program;
- not `HostResource`s;
- not governed by A11 exactly-once close.

They cross the ABI as capability functions.

### 5.2 Provider surface

Recommended functions:

```text
stdin.read
stdout.write
stderr.write
stdout.flush
stderr.flush
```

Conceptual package API:

```stark
pub fn stdin_read(
    output: &mut [UInt8],
) -> Result<UInt64, IoError>;

pub fn stdout_write(
    input: &[UInt8],
) -> Result<UInt64, IoError>;

pub fn stderr_write(
    input: &[UInt8],
) -> Result<UInt64, IoError>;

pub fn stdout_write_all(
    input: &[UInt8],
) -> Result<(), IoError>;

pub fn stderr_write_all(
    input: &[UInt8],
) -> Result<(), IoError>;

pub fn stdout_flush() -> Result<(), IoError>;

pub fn stderr_flush() -> Result<(), IoError>;
```

### 5.3 Boundary rules

- bytes, not text, at the provider boundary;
- UTF-8 decoding belongs in package code;
- invalid UTF-8 is reported, never replaced lossily;
- Windows streams use binary mode;
- no implicit CRLF translation;
- `0` from stdin read means EOF;
- partial writes are exposed by `write`;
- `write_all` loops in package code.

### 5.4 Core print interleaving

Before freezing behaviour, inspect the existing Core `print` and `println` path.

Initial safe guarantee:

> STARK preserves call order within one thread. Multiple writes are not atomic, and output from mixed Core and package APIs may interleave at write boundaries.

Do not claim shared buffering or OS-write atomicity without evidence.

### 5.5 Exit criteria

- native consumer reads stdin to EOF;
- transforms bytes;
- writes stdout;
- writes diagnostics to stderr;
- CI pipes actual bytes into the process and compares exact stdout/stderr;
- EOF, zero-length read, and partial-write behaviour tested;
- Windows binary-mode behaviour pinned;
- Core-print interleaving rule documented;
- Tier-1 CI green.

---

# PART C — Buffered I/O

## 6. WP-PKG-BUFIO1 — `stark-bufio` v0.1

**Priority:** P1  
**Complexity:** S–M  
**Status:** `READY` after STDIO1  
**Shape:** Pure package over read/write capabilities  
**Qualification:** Tier P logic plus Tier-N consumers  
**Blocks:** line-oriented CLI tools

### 6.1 Design seam

Do not expand the compiler solely to support a generic reader abstraction.

Before implementation, choose one:

1. concrete wrappers;
2. existing trait/generic abstraction if already cleanly supported;
3. another already-admitted mechanism.

Recommended v0.1:

```text
FileBufReader
StdinBufReader
FileBufWriter
StdoutBufWriter
StderrBufWriter
```

### 6.2 Scope

- fixed-capacity buffer chosen at construction;
- `read_line`;
- buffered byte reads;
- buffered writes;
- explicit `flush`;
- CRLF/LF option;
- no charset detection;
- no BOM handling;
- no locale behaviour.

### 6.3 Drop policy

Buffered data is not guaranteed to flush during drop.

Reason:

- destructors cannot return flush errors;
- silent implicit flush would hide failures;
- CD-291 established the same principle for file close.

Normative rule:

> Unflushed buffered output may be lost. Programs requiring confirmation must call `flush()` explicitly.

### 6.4 Iterator policy

A `lines()` iterator is admitted only if ownership and borrowing work naturally.

If it creates ergonomic or lifetime pressure:

- ship `read_line` first;
- record the limitation;
- do not redesign the compiler merely for iterator syntax.

### 6.5 Exit criteria

- empty file;
- no trailing newline;
- empty lines;
- lines longer than the buffer;
- mixed CRLF/LF;
- file and stdin line-counting consumers;
- one-shot versus buffered byte equivalence;
- explicit flush failure path;
- Tier-1 CI.

---

# PART D — Signal and shutdown

## 7. WP-PKG-SIGNAL1 — Interrupt and orderly shutdown

**Priority:** P1  
**Complexity:** M  
**Status:** `SPEC FIRST`  
**Shape:** Function-shaped  
**Qualification:** Tier N  
**Blocks:** Honest operable-server milestone

### 7.1 v0.1 mechanism

The provider installs a minimal flag-setting handler.

Signals/events:

- SIGINT and SIGTERM on Unix;
- console control event on Windows.

STARK code polls:

```stark
pub fn shutdown_requested() -> Bool;
```

No STARK code runs asynchronously in signal context.

No callbacks, no closures, no handler registration API, and no async delivery.

### 7.2 Required specification decisions

Before implementation, record:

1. when handler installation occurs;
2. what happens if installation fails;
3. second-interrupt behaviour;
4. fatal-signal resource semantics;
5. manifest capability name and declaration rule.

Recommended answers:

- provider initializes when the capability is selected;
- installation failure is explicit, never silently treated as `false`;
- second interrupt restores/defaults to immediate OS termination;
- orderly shutdown runs normal drop;
- fatal signals do not guarantee resource close;
- capability is manifest-declared.

### 7.3 A11 scope

Normative statement:

> A11 exactly-once close applies to orderly execution. It does not guarantee cleanup after SIGKILL, process abort, fatal runtime failure, or machine failure.

Providers must tolerate OS-level reclamation in those cases.

### 7.4 Exit criteria

- native accept-loop server polls the flag;
- CI sends a real signal/event;
- server leaves the loop;
- listener closes through ordinary drop;
- drain marker is observed;
- second-interrupt behaviour tested;
- Unix and Windows behaviour documented;
- Tier-1 CI or explicitly classified runner limitation.

---

# PART E — DNS

## 8. WP-PKG-DNS1 — Host resolver package

**Priority:** P2  
**Complexity:** M  
**Status:** `SPEC FIRST`  
**Shape:** Function-shaped  
**Qualification:** Tier N  
**Blocks:** Hostname-based HTTP client connections

### 8.1 Boundary ruling

DNS v0.1 uses the host resolver.

It does not implement:

- DNS wire protocol;
- UDP queries;
- `/etc/hosts` parsing;
- resolver caching;
- retry policy;
- custom timeouts;
- DNSSEC.

The host resolver owns those behaviours.

### 8.2 Provider shape

Hostname in, bounded ordered snapshot out.

Use a caller-provided buffer with:

- required length query;
- fill operation;
- `LimitExceeded` on truncation;
- never return a silently shortened list.

### 8.3 Entry encoding

Frozen fixed record width: **22 bytes**.

```text
byte 0       family: UInt8
byte 1       address_length: UInt8
bytes 2..18  address: [UInt8; 16]
bytes 18..22 scope_id: UInt32, big-endian
```

Family tags:

```text
4 = IPv4
6 = IPv6
```

Address bytes are network byte order. IPv4 uses the first four address bytes and zero-fills the
remaining twelve. IPv6 uses all sixteen address bytes. `scope_id` is zero in DNS v0.1; nonzero
scope-id support is deferred.

Port is not included. The public package API supplies the port and attaches it to every resolved
address.

### 8.4 Decisions to freeze

- IPv4 and IPv6 family tags: frozen as `4` and `6`.
- IPv4-mapped IPv6 policy: preserve the host resolver output; no remapping in the provider.
- scope-ID handling: record field exists, but DNS v0.1 emits zero and package decoding ignores it.
- duplicate preservation: preserve host resolver output order and entries.
- host resolver ordering: preserve host resolver order.
- canonical-name inclusion or exclusion: excluded.
- empty-success versus `NotFound`: empty result is `NotFound`.
- temporary failure mapping: OS timeout/interrupted/would-block maps to `TemporaryFailure`.
- unsupported-family mapping: unknown family tags decode as `UnsupportedAddressFamily`.
- maximum result count and total bytes: package v0.1 supports at most 32 records, 704 bytes.
- provider status codes: DNS statuses use `101..107` because the current provider vocabulary is
  provider-wide, not per capability, and TCP already owns `1..11`.

### 8.5 Error model

Example:

```stark
pub enum DnsError {
    InvalidHost,
    NotFound,
    TemporaryFailure,
    TooManyResults,
    UnsupportedAddressFamily,
    Unsupported,
    Other(UInt32),
}
```

### 8.6 Exit criteria

- localhost resolution;
- IPv4-only and IPv6-only fixtures where available;
- duplicate/order policy observed;
- invalid hostname;
- not-found;
- buffer-too-small path;
- native source consumer connects using a resolved address;
- Tier-1 CI.

---

# PART F — Process spawning

## 9. WP-PKG-PROCESS1 — Process execution

**Priority:** P3  
**Complexity:** L  
**Status:** `SPEC FIRST`  
**Shape:** Resource-shaped  
**Qualification:** Tier N  
**Dependencies:** environment, stdio, file I/O, A11 resource lifecycle

This is the only capability in this roadmap that should be treated as a substantial work package rather than routine provider repetition.

### 9.1 Resource model

A child process is a `HostResource`.

Conceptual API:

```stark
pub resource Process;

pub fn spawn(
    executable: &str,
    args: &[String],
    options: &SpawnOptions,
) -> Result<Process, ProcessError>;

pub fn wait(
    process: Process,
) -> Result<ExitStatus, ProcessError>;
```

### 9.2 Wait is not automatically the full drop policy

Consuming `wait` is natural, but it does not define what happens when a process is dropped without waiting.

CRYPTO-style lifecycle discipline applies: the behaviour must be explicit before API freeze.

Possible drop policies:

1. detach;
2. terminate;
3. block and wait;
4. non-blocking reap then detach;
5. reject implicit drop through verifier/runtime policy.

Recommended v0.1 direction:

> Dropping an unwaited process detaches it and transfers eventual OS cleanup to the provider/runtime.

This remains a decision, not an assumption.

### 9.3 Capture constraint

Live stdout and stderr pipes multiply resources and introduce deadlock risk in a single-threaded language.

A child can block on a full stdout pipe while the parent blocks reading stderr.

Therefore v0.1 must not expose unrestricted live dual-pipe capture.

Admitted options:

#### Mode A — Inherit parent stdio

- simplest;
- no capture resources;
- no pipe deadlock.

#### Mode B — Provider-managed bounded capture

- provider drains child output internally;
- optional merged stream;
- hard byte limit;
- result returned on `wait`;
- overflow behaviour explicit.

Deferred:

- live streaming of stdout and stderr;
- three-resource process/pipe lifecycle;
- shell pipelines;
- async capture.

### 9.4 Required policy decisions

- no shell interpretation;
- no implicit PATH search unless separately admitted;
- executable path policy;
- argument encoding;
- current directory;
- environment inheritance;
- environment replacement/overrides;
- stdio inheritance;
- capture mode;
- output byte limits;
- timeout policy;
- termination policy;
- exit status model;
- drop/detach policy;
- orphan/reaping policy;
- Windows command-line quoting rules.

### 9.5 Exit criteria

- spawn known executable;
- ordered argument preservation;
- inherited stdio mode;
- provider-managed bounded capture if admitted;
- non-zero exit status;
- missing executable;
- explicit environment policy;
- drop-before-wait behaviour observed;
- no shell expansion;
- Tier-1 CI.

---

# PART G — Integrated schedule

## 10. Execution waves

### Wave 1 — Complete Milestone A foundation

1. `stark-random`
2. standard streams in `stark-io`
3. `stark-bufio`

Expected outcome:

- secure randomness available for later crypto work;
- CLI tools work in shell pipelines;
- line-oriented tools no longer need whole-file buffering.

### Wave 2 — Operable networking

4. signal/shutdown specification
5. signal/shutdown provider
6. DNS resolver alongside networking/HTTP client work

Expected outcome:

- servers can shut down cleanly;
- clients can resolve hostnames;
- Milestone B can make an operability claim honestly.

### Wave 3 — Process capability

7. process lifecycle and capture specification
8. provider implementation
9. package and qualification

Expected outcome:

- synchronous child process execution;
- inherited stdio;
- optional safe provider-managed capture;
- no unresolved pipe-deadlock design.

---

## 11. Parallelisation

Safe parallel tracks:

### Track A — Randomness

- native secure-fill provider;
- pure deterministic PRNG;
- Tier-1 qualification.

### Track B — Standard streams

- provider symbols;
- package surface;
- pipe-driven CI;
- Core-print audit.

### Track C — Specification

- signal decisions;
- DNS encoding/error model;
- process lifecycle/capture model.

`stark-bufio` begins once the stdio package surface is stable.

DNS implementation begins only after its bounded snapshot encoding is frozen.

Process implementation begins only after its lifecycle and capture work package closes.

---

## 12. Acceptance template

Every host capability must record:

```text
Capability name
Function-shaped or resource-shaped
Manifest declaration
Provider symbols
Input and output bounds
Error model
Encoding
Ownership/lifecycle policy
HIR/MIR unavailable-capability behaviour
Native provider identity
Native source consumer
Observed-behaviour e2e
Linux qualification
macOS qualification
Windows qualification
Known exclusions
Exact qualifying commit
```

A capability is not qualified when:

- symbols exist but no source-level package reaches them;
- the native build succeeds but the produced program is not executed;
- only one platform was tested while Tier-1 support is claimed;
- resource drop behaviour is undocumented;
- truncation silently returns incomplete results;
- provider failure is converted into a plausible success value;
- unavailable Tier-N execution is counted as a differential-engine failure.

---

## 13. Final compiler implication

The compiler implication of this roadmap is:

> Host capability addition is now a repeatable platform workflow.

The remaining risk is not whether STARK can represent these capabilities. The risk is whether lifecycle, allocation limits, encoding, interruption, and failure semantics are specified before implementation.

That is the discipline this work package enforces.
