# WP-C7.8 — Native Host Capability Foundation

> **SUPERSEDED 2026-07-28 — do not implement.**
>
> Disposition: **REVISE — conflicts with approved CE4 Native Provider ABI v0.1 and does not fully
> unblock P1.**
>
> Two defects make this document unsafe to amend incrementally:
>
> 1. **§5.1 reopens a decided question and recommends reversing it.** The native resource model
>    was settled by CE4 Amendment 1 (CD-054, 2026-07-21) and is implemented in
>    `starkc/stark-runtime/src/provider_abi.rs`: `RawResourceHandle` at the boundary,
>    `OwnedResourceHandle` inside, no `Copy`, no `Clone`, **no Rust `Drop`**, exactly-once release
>    owned by MIR `Drop` terminators. §5.1's recommended Option A puts destruction back in Rust
>    `Drop`. The document never cites the ABI.
> 2. **§6.5's TCP scope is client-only**, so the §16 closure claim would be false: P1 requires a
>    TCP listener to serve three REST endpoints.
>
> Replaced by:
> - `STARKLANG/docs/compiler/work-packages/WP-C7.8-First-Party-Native-Host-Capabilities.md`
> - `STARKLANG/docs/compiler/work-packages/WP-C7.8.1-DECISION-PACKETS.md`
>
> Retained for provenance. Its §4 principles, §10 test strategy and §12 capability-matrix
> requirement survive into the replacement; its §5 does not.

**Status:** SUPERSEDED (was PROPOSED)  
**Parent gate:** C7  
**Purpose:** Remove the native-capability blocker preventing P1 from executing on the native path.  
**Authority:** Implementation work package. Any escalation-class decision remains subject to the existing project governance model.  
**Primary target:** macOS, Linux, and Windows native builds.

---

## 1. Context

Gate C7 is complete through C7.7 but cannot close because P1 requires host-facing capabilities that the current native backend and runtime do not provide.

The current frontend can recognise and typecheck selected host-facing types and methods, but the native path cannot lower or execute them. For example:

```text
native build does not yet support this program: type Core(File, [])
```

The missing surface includes:

- process arguments;
- environment variables;
- file I/O;
- time;
- TCP networking.

This is not merely a package implementation gap. It is a native representation, lowering, runtime ABI, resource ownership, error propagation, and cross-platform behaviour gap.

WP-C7.8 establishes that foundation. P1 consumes it.

---

## 2. Objective

After WP-C7.8, native STARK programs must be able to:

1. read command-line arguments;
2. read environment variables;
3. open, read, create, and write files;
4. obtain monotonic and wall-clock time;
5. connect to a TCP endpoint and exchange bytes;
6. own and release host-backed resources safely;
7. surface defined STARK errors instead of backend-specific unsupported-type failures;
8. execute consistently on macOS, Linux, and Windows.

---

## 3. Non-goals

WP-C7.8 does not implement:

- HTTP or HTTPS;
- TLS;
- DNS APIs beyond what is minimally required by the chosen TCP connect surface;
- asynchronous I/O;
- event loops;
- filesystem watching;
- directory traversal beyond any minimal support required by tests;
- advanced file metadata;
- UDP;
- sockets exposed as operating-system descriptors;
- public package ergonomics beyond the minimum required to prove the runtime boundary;
- a direct Cranelift or LLVM backend;
- general foreign-function interfaces.

These belong to later package or backend work.

---

## 4. Architectural principles

### 4.1 Host resources are opaque

Host-backed values such as files and TCP streams must not expose operating-system handles directly to STARK programs.

The compiler and runtime must treat them as opaque resource values with defined ownership and destruction semantics.

Conceptually:

```text
STARK File
    ↓
MIR opaque host resource
    ↓
generated Rust wrapper
    ↓
stark-runtime owned host object
```

The precise representation may use typed generated-Rust wrappers, runtime-owned handles, or another equivalent mechanism, provided all required invariants are met.

### 4.2 Ownership follows normal STARK move rules

Host resources are non-`Copy`.

Moving a resource transfers ownership. Using the moved-from value must be rejected by the existing ownership system.

Example:

```stark
let f = File::open("input.txt")?;
let g = f;
f.read_to_string(); // reject: use after move
```

### 4.3 Resource destruction is deterministic

A live host resource must be released when its owning value is destroyed.

The implementation must define behaviour for:

- normal scope exit;
- early return;
- error propagation;
- branch exits;
- loop exits;
- explicit close;
- implicit close during destruction;
- partially initialised values.

Explicit close must be idempotent or return a defined error. It must never cause double-close undefined behaviour.

### 4.4 Errors are STARK values

Operating-system failures must be converted into defined STARK error values.

Generated Rust panics, raw `std::io::Error`, platform-specific error codes, and backend crashes must not escape as the public language behaviour.

### 4.5 Runtime calls are panic-contained

All host-provider entry points must contain Rust panics and return a defined internal runtime failure result.

A panic in host code must not unwind through generated STARK code.

### 4.6 Cross-platform differences are explicit

The public semantics must be platform-neutral where feasible.

Where behaviour necessarily differs, the difference must be:

- documented;
- observable through a defined result;
- tested per platform;
- excluded from stronger portability claims.

### 4.7 Capability support is explicit

Unsupported operations must fail during compilation or through a defined runtime capability error. They must not fail through incidental code-generation errors.

---

## 5. Required design decisions

WP-C7.8.1 must record the following before broad implementation begins.

### 5.1 Resource representation

Choose one of the following or define an equivalent model.

#### Option A — generated-Rust owned wrappers

Generated Rust values directly own runtime wrapper types such as:

```rust
pub struct StarkFile {
    inner: Option<std::fs::File>,
}
```

Advantages:

- Rust ownership naturally enforces one owner;
- destruction can use `Drop`;
- minimal handle registry;
- simple generated-Rust integration.

Risks:

- generated types become coupled to runtime implementation types;
- ABI boundaries are less isolated;
- future direct backends need a different representation.

#### Option B — runtime handle table

Generated code holds a typed numeric or opaque handle managed by `stark-runtime`.

Advantages:

- stable runtime ABI;
- easier future direct-backend integration;
- centralised validation and destruction.

Risks:

- handle lifetime management is more complex;
- stale-handle and double-close protection are required;
- concurrent access requires future design care.

#### Recommended initial direction

Use generated-Rust owned wrappers for C7.8 unless existing backend architecture strongly favours a runtime handle table.

The generated-Rust backend is the current selected implementation path. Avoid designing a speculative universal ABI at the expense of completing the native capability surface.

The design must still isolate compiler-facing operations behind stable runtime functions so a future direct backend can replace the physical representation.

### 5.2 Runtime call result shape

Define an internal result contract capable of representing:

- success with no value;
- success with scalar value;
- success with `String`;
- success with `Vec<u8>`;
- success with `Vec<String>`;
- success with opaque resource;
- typed host error;
- internal runtime failure.

Generated code should convert internal runtime results into the corresponding STARK `Result<T, E>`.

### 5.3 String transfer

Specify:

- encoding: UTF-8;
- behaviour for invalid environment or path text;
- ownership of input strings;
- ownership of returned strings;
- whether runtime calls borrow or clone inputs;
- platform conversion rules.

Returned strings must be owned STARK strings.

### 5.4 Byte-buffer transfer

Specify:

- owned `Vec<u8>` return;
- borrowed slice input for writes;
- maximum size handling;
- partial-read and partial-write behaviour;
- allocation failure behaviour;
- whether reads are bounded or read-to-end.

### 5.5 Error taxonomy

At minimum define stable error categories for:

- not found;
- permission denied;
- already exists;
- invalid input;
- interrupted;
- unexpected end of file;
- connection refused;
- connection reset;
- timed out;
- address unavailable;
- unsupported;
- resource closed;
- other.

Do not expose platform-specific integer codes as the primary API. An optional diagnostic code may be retained for debugging.

### 5.6 Destruction and explicit close

Define:

- whether `close()` consumes `self` or takes `&mut self`;
- whether `close()` returns `Result<(), IoError>`;
- the state after explicit close;
- behaviour of subsequent operations;
- destructor behaviour after explicit close;
- whether failed close leaves the resource logically closed.

Recommended public semantic shape:

```stark
fn close(self) -> Result<(), IoError>
```

Consuming `self` avoids a persistent closed-but-still-live resource state and fits STARK ownership semantics.

---

## 6. Work-package breakdown

## 6.1 WP-C7.8.1 — Native resource and runtime ABI design

### Scope

Define and prove the common architecture required by all later capability slices.

### Required outputs

1. Native representation for opaque host resources.
2. MIR representation or lowering rule for host-backed core types.
3. Runtime-call ABI conventions.
4. Error mapping contract.
5. String and byte-buffer transfer contract.
6. Destruction model.
7. Panic-containment rules.
8. Cross-platform provider interface.
9. Diagnostics for unsupported host types and calls.
10. Test strategy.

### Compiler changes

The compiler must be able to:

- identify host-backed core types;
- admit them in native type lowering;
- generate correct Rust field, local, argument, and return types;
- move them without cloning;
- destroy them exactly once;
- call runtime-provider functions;
- convert runtime results into STARK `Result`;
- reject unsupported host operations with stable diagnostics.

### Runtime changes

Create host-capability modules under `stark-runtime`, for example:

```text
stark-runtime/
  src/
    host/
      mod.rs
      error.rs
      process.rs
      file.rs
      time.rs
      tcp.rs
```

The exact structure may vary, but process, file, time, and TCP implementations must not be mixed into unrelated runtime code.

### Acceptance criteria

- a synthetic opaque resource can be constructed, moved, returned, destroyed, and rejected after move;
- destruction occurs exactly once on normal and early exits;
- a simulated runtime error converts to a STARK error;
- a simulated runtime panic is contained;
- unsupported host types produce a stable compiler diagnostic;
- all tests pass on macOS, Linux, and Windows.

---

## 6.2 WP-C7.8.2 — Process arguments and environment

### Minimum public surface

Names may follow existing package conventions, but capability must be equivalent to:

```stark
fn args() -> Vec<String>;

fn env(name: &str) -> Result<Option<String>, EnvError>;
```

Optional:

```stark
fn env_all() -> Result<Vec<(String, String)>, EnvError>;
```

`env_all` is not required for closure.

### Required semantics

#### `args()`

- returns all process arguments in deterministic order;
- documents whether the executable path is included;
- returns owned strings;
- handles non-UTF-8 platform arguments according to the recorded encoding policy;
- never aliases runtime-owned buffers.

#### `env(name)`

- returns `Ok(Some(value))` when present;
- returns `Ok(None)` when absent;
- returns `Err(...)` only for invalid input, encoding, or provider failure;
- does not expose mutable process-environment operations in this work package.

### Tests

- zero user arguments;
- one argument;
- multiple arguments;
- spaces in arguments;
- Unicode argument;
- missing environment variable;
- present empty environment variable;
- present normal environment variable;
- Unicode environment value where supported;
- invalid variable name behaviour;
- native execution on all three platforms.

### Acceptance criteria

A native STARK program can print or otherwise verify one command-line argument and one environment value on macOS, Linux, and Windows.

---

## 6.3 WP-C7.8.3 — File I/O

### Minimum public surface

Equivalent capability:

```stark
struct File;

impl File {
    fn open(path: &str) -> Result<File, IoError>;
    fn create(path: &str) -> Result<File, IoError>;

    fn read_to_end(&mut self) -> Result<Vec<u8>, IoError>;
    fn read_to_string(&mut self) -> Result<String, IoError>;

    fn write_all(&mut self, data: &[u8]) -> Result<(), IoError>;

    fn close(self) -> Result<(), IoError>;
}
```

An implementation may use associated functions or package-level functions if required by current language limitations, but the semantics must remain equivalent.

### Path semantics

Record:

- accepted path string encoding;
- relative-path base;
- absolute-path behaviour;
- handling of spaces;
- Unicode paths;
- Windows path separators and drive prefixes;
- behaviour for invalid paths.

### File ownership

`File` must be:

- move-only;
- non-`Copy`;
- non-`Clone` unless explicitly justified later;
- automatically closed when destroyed;
- unusable after being moved;
- consumed by explicit close.

### Read semantics

`read_to_end`:

- reads from the current position to end of file;
- returns owned bytes;
- returns defined errors;
- must not silently truncate.

`read_to_string`:

- reads from the current position to end of file;
- requires valid UTF-8;
- returns a defined encoding or invalid-data error otherwise.

### Write semantics

`write_all`:

- attempts to write the full buffer;
- internally handles partial operating-system writes;
- returns only after all bytes are written or an error occurs.

### Tests

Positive:

- open and read an empty file;
- open and read text;
- open and read binary data;
- create and write an empty file;
- create and write text;
- create and write binary data;
- filenames containing spaces;
- Unicode filenames where supported;
- file returned from a function;
- file moved between locals;
- implicit close on scope exit;
- explicit close.

Negative:

- missing file;
- permission denied where reliably constructible;
- open directory as file where platform behaviour is defined;
- invalid UTF-8 through `read_to_string`;
- use after move rejected;
- duplicate close impossible or safely rejected;
- backend unsupported-type error no longer appears for `File`.

Cross-engine:

- HIR and native error category parity where HIR supports the operation;
- byte-for-byte file output parity.

### Acceptance criteria

A native STARK program can:

1. read a file path from an argument;
2. open the file;
3. read its contents;
4. create a second file;
5. write the same bytes;
6. close or drop both resources safely;
7. produce identical output bytes on all supported platforms.

---

## 6.4 WP-C7.8.4 — Time

### Existing work reuse

Reuse the existing `stark-time` types and provider work where possible.

Do not introduce a parallel incompatible time model.

### Minimum public surface

Equivalent capability:

```stark
impl Instant {
    fn now() -> Instant;
}

impl SystemTime {
    fn now() -> SystemTime;
}

fn sleep(duration: Duration) -> Result<(), TimeError>;
```

If wall-clock naming differs in the current spec, use the existing canonical type.

### Required semantics

#### Monotonic time

`Instant::now()` must:

- be monotonic within the guarantees of the host platform;
- not represent calendar time;
- support elapsed-duration calculation through existing checked operations.

#### Wall-clock time

The wall-clock operation must:

- return a documented epoch-based or structured representation;
- preserve existing `UnixTimestamp` rules if already defined;
- report out-of-range values through a defined error rather than overflow.

#### Sleep

`sleep` must:

- reject negative durations if representable;
- define zero-duration behaviour;
- not promise exact wake-up time;
- return a defined provider error if unsupported.

### Tests

- two monotonic samples are non-decreasing;
- elapsed duration is valid;
- wall-clock value falls within a test-controlled tolerance;
- zero-duration sleep succeeds;
- small-duration sleep does not return before a documented tolerance;
- overflow and invalid-duration cases;
- provider panic containment;
- execution on all three platforms.

### Acceptance criteria

A native STARK program can record a monotonic start time, sleep briefly, record an end time, and compute a non-negative elapsed duration.

---

## 6.5 WP-C7.8.5 — TCP

### Minimum public surface

Equivalent capability:

```stark
struct TcpStream;

impl TcpStream {
    fn connect(address: &str) -> Result<TcpStream, NetworkError>;

    fn read(&mut self, max_bytes: usize) -> Result<Vec<u8>, NetworkError>;

    fn write_all(&mut self, data: &[u8]) -> Result<(), NetworkError>;

    fn close(self) -> Result<(), NetworkError>;
}
```

Alternative buffer-oriented APIs are acceptable if already supported by the language, but closure requires receiving and transmitting arbitrary bytes.

### Address format

For the first slice, an address string such as:

```text
127.0.0.1:8080
```

is sufficient.

Hostname support may be included if it comes naturally from the provider, but DNS-specific APIs are not required.

### Required semantics

- `TcpStream` is move-only;
- connect returns a defined error category;
- reads return zero bytes on orderly peer closure;
- writes handle partial operating-system writes;
- explicit close consumes the stream;
- implicit drop closes the stream;
- no raw socket descriptor is exposed;
- blocking I/O is acceptable for C7.8.

### Tests

Use a controlled local test server.

Positive:

- connect to loopback;
- send bytes;
- receive bytes;
- payload containing zero bytes;
- payload larger than one likely operating-system read/write chunk;
- orderly server close;
- move stream between locals;
- return stream from a function;
- implicit and explicit close.

Negative:

- connection refused;
- malformed address;
- invalid port;
- peer reset where reliably testable;
- use after move rejected;
- unsupported-type diagnostics absent.

Cross-platform:

- same echo protocol test on macOS, Linux, and Windows.

### Acceptance criteria

A native STARK program can connect to a local TCP echo server, send a payload, receive the identical payload, and close safely on all three platforms.

---

## 7. MIR and backend requirements

The native backend must support host-backed types in:

- local variables;
- function parameters;
- return values;
- `Result<T, E>`;
- `Option<T>` where required;
- branches;
- loops;
- early returns;
- error propagation;
- aggregate fields if admitted by the current language rules.

The verifier must reject invalid operations, including:

- copying a move-only resource;
- duplicating ownership;
- destroying a moved value;
- invoking a method requiring a live resource after consumption;
- constructing a host resource without an approved runtime operation.

Generated Rust must not rely on accidental derives that make resources cloneable or copyable.

---

## 8. Runtime API requirements

Every runtime provider function must:

1. validate inputs;
2. contain panics;
3. avoid leaking platform-specific implementation objects;
4. return a defined internal result;
5. preserve ownership rules;
6. document blocking behaviour;
7. avoid undefined behaviour;
8. have direct Rust unit tests;
9. have native STARK integration tests;
10. compile without unnecessary third-party dependencies unless explicitly approved.

Suggested internal shape:

```rust
pub enum HostErrorKind {
    NotFound,
    PermissionDenied,
    AlreadyExists,
    InvalidInput,
    Interrupted,
    UnexpectedEof,
    ConnectionRefused,
    ConnectionReset,
    TimedOut,
    AddressUnavailable,
    Unsupported,
    Closed,
    Other,
}

pub struct HostError {
    pub kind: HostErrorKind,
    pub message: String,
    pub platform_code: Option<i64>,
}
```

This is illustrative. Final names must follow repository conventions.

---

## 9. Diagnostics

Replace generic backend errors with capability-specific diagnostics.

Examples:

```text
E-NATIVE-HOST-001: native file support is unavailable for this target
```

```text
E-NATIVE-HOST-002: host resource type `File` cannot be copied
```

```text
E-NATIVE-HOST-003: operation attempted on a consumed host resource
```

```text
E-NATIVE-HOST-004: native runtime does not implement `TcpStream::connect`
```

Diagnostics must include:

- the unsupported type or operation;
- the compilation target;
- whether the limitation is permanent, deferred, or target-specific where known;
- no raw generated-Rust type names unless shown as secondary debugging detail.

---

## 10. Test strategy

## 10.1 Unit tests

Compiler:

- type lowering;
- move eligibility;
- destruction insertion;
- runtime call lowering;
- result conversion;
- diagnostic selection.

Runtime:

- provider success cases;
- provider error mapping;
- invalid input;
- panic containment;
- explicit close;
- implicit drop;
- double-close defence where applicable.

## 10.2 Conformance fixtures

Add positive and negative STARK fixtures for:

- process arguments;
- environment access;
- file ownership;
- file read/write;
- time;
- TCP;
- unsupported operations;
- resource movement and destruction.

## 10.3 Differential tests

Where the HIR interpreter supports the same host operation, compare:

- success result shape;
- error category;
- returned bytes or strings;
- observable file output;
- resource-lifetime behaviour where observable.

Differences caused by platform state must be controlled by test setup.

## 10.4 Platform matrix

All mandatory capability tests must execute on:

- macOS;
- Linux;
- Windows.

Do not infer one platform's behaviour from another.

## 10.5 Leak and lifecycle tests

At minimum:

- repeated file open/drop;
- repeated TCP connect/drop;
- early return with live resource;
- error propagation with live resource;
- resource moved through function call;
- explicit close followed by destructor path;
- no unbounded runtime handle growth if a handle table is used.

---

## 11. Security requirements

- no arbitrary raw pointer or descriptor exposure;
- no path concatenation inside the runtime unless explicitly required;
- no shell execution;
- no command execution;
- no environment mutation;
- no implicit network listener;
- no unsafe code without isolated justification and tests;
- input sizes must be checked before allocation where practical;
- platform error messages must not substitute for stable error categories;
- tests must avoid external network dependencies.

---

## 12. Documentation requirements

Update:

- compiler state;
- native backend capability matrix;
- core package status;
- runtime architecture;
- error taxonomy;
- supported-platform table;
- examples;
- known limitations.

The capability matrix must distinguish:

- frontend-supported;
- HIR-supported;
- MIR-lowered;
- native-runtime-supported;
- cross-platform verified.

Do not use a single “supported” column for all layers.

---

## 13. Exit criteria

WP-C7.8 closes only when all of the following are met.

### Architecture

- [ ] Resource representation is documented.
- [ ] Ownership and destruction semantics are documented.
- [ ] Runtime call and error contracts are documented.
- [ ] String and byte-buffer transfer rules are documented.
- [ ] Unsupported capability behaviour is documented.

### Process and environment

- [ ] Native program reads arguments.
- [ ] Native program reads an environment variable.
- [ ] Verified on macOS, Linux, and Windows.

### File I/O

- [ ] Native program opens and reads a file.
- [ ] Native program creates and writes a file.
- [ ] File resource is move-only.
- [ ] Explicit and implicit close are safe.
- [ ] Verified on macOS, Linux, and Windows.

### Time

- [ ] Native program reads monotonic time.
- [ ] Native program reads wall-clock time.
- [ ] Native program sleeps for a duration.
- [ ] Existing time package semantics are preserved.
- [ ] Verified on macOS, Linux, and Windows.

### TCP

- [ ] Native program connects to loopback TCP.
- [ ] Native program sends arbitrary bytes.
- [ ] Native program receives arbitrary bytes.
- [ ] Stream resource is move-only.
- [ ] Explicit and implicit close are safe.
- [ ] Verified on macOS, Linux, and Windows.

### Quality

- [ ] CI is green.
- [ ] Formatting and lint checks are clean.
- [ ] No unsupported-type backend error remains for the admitted capability surface.
- [ ] Runtime panics are contained.
- [ ] Error categories are stable and tested.
- [ ] Resource leak tests pass.
- [ ] Documentation and capability matrix are updated.

---

## 14. Gate relationship

WP-C7.8 removes the native-capability prerequisite blocking P1.

It does not by itself close Gate C7.

Required sequence:

```text
C7.8.1  Native resource and ABI design
    ↓
C7.8.2  Process arguments and environment
    ↓
C7.8.3  File I/O
    ↓
C7.8.4  Time
    ↓
C7.8.5  TCP
    ↓
Close WP-C7.8
    ↓
Execute and close P1
    ↓
Re-evaluate Gate C7 closure
```

---

## 15. Recommended implementation order

Implement in this order:

1. common error representation;
2. synthetic opaque-resource proof;
3. resource lowering and destruction;
4. runtime-call result conversion;
5. process arguments;
6. environment lookup;
7. file open and destruction;
8. file read;
9. file create and write;
10. monotonic time;
11. wall-clock time;
12. sleep;
13. TCP connect and destruction;
14. TCP read and write;
15. full platform matrix;
16. documentation and closure evidence.

Do not start TCP before file ownership and destruction are proven. File I/O provides the simpler resource-lifecycle test bed.

---

## 16. Closure statement template

When complete, record:

> WP-C7.8 CLOSED. The native backend and runtime now support the host-capability foundation required by P1: process arguments, environment lookup, file I/O, time, and blocking TCP. Host-backed values are move-only, deterministically released, panic-contained, and mapped to stable STARK errors. The admitted surface is verified on macOS, Linux, and Windows. P1 is no longer blocked by native host capability, but Gate C7 remains open until P1 exit criteria are met.

