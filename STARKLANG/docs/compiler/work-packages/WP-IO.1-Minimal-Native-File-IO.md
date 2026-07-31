# WP-IO.1 — Minimal Native File I/O for STARK

**Execution brief for Codex / Claude**  
**Repository:** `navraj007in/stark`  
**Prepared:** 2026-07-31  
**Target:** complete a bounded, genuinely usable synchronous file-I/O slice without claiming the full `stark-io` v0.1 filesystem surface.

---

## 0. Executive directive

Implement and qualify the smallest useful native file-I/O package surface:

```text
File::open
File::create
File::read
File::write
File::flush
File::close

File::read_to_end
File::read_to_string
File::write_all
File::write_str

read
read_text
write
write_text
```

Use the existing first-party `stark-file/native` provider and the existing source-level `provider_api` package-binding mechanism already exercised by the P1 REST workload.

Do not expand into seek, metadata, directories, rename, delete, recursive operations, copy, append, truncate, symlink handling, or full open-options support in this work package.

The required outcome is a **minimal native file-I/O capability that works from ordinary STARK source**, not merely lower-level MIR tests or Rust provider tests.

---

## 1. Starting position

### 1.1 Existing native provider

`stark-file/native` already exports:

```text
stark_file_open
stark_file_create
stark_file_read
stark_file_write
stark_file_complete
stark_file_close
```

The provider already has:

- Native Provider ABI v0.1 metadata;
- opaque `file` resource handles;
- status-code mapping;
- handle-table ownership;
- external symbol linkage tests;
- read/write/flush/close tests;
- metadata validation;
- target declarations for Linux x64, macOS arm64, macOS x64, and Windows x64.

### 1.2 Existing package state

`stark-io` currently provides:

- public API shape;
- `File`, `IOError`, and supporting types;
- pure validation helpers;
- trapping placeholders for host-backed methods.

### 1.3 Known blockers

Current blockers are:

1. Ordinary STARK package source has not yet bound `File` methods to the file provider.
2. Library-package qualification lacks a clean native build mode without an executable `main`.
3. The full `stark-io` v0.1 API requires a much larger provider surface.

This work package addresses only blocker 1 and uses a bounded executable qualification package to avoid blocker 2.

---

## 2. Scope

### 2.1 In scope

```stark
pub resource File;

impl File {
    pub fn open(path: &str) -> Result<File, IOError>;
    pub fn create(path: &str) -> Result<File, IOError>;
    pub fn read(&mut self, buffer: &mut [UInt8]) -> Result<UInt64, IOError>;
    pub fn read_to_end(&mut self, max_bytes: UInt64) -> Result<Vec<UInt8>, IOError>;
    pub fn read_to_string(&mut self, max_bytes: UInt64) -> Result<String, IOError>;
    pub fn write(&mut self, data: &[UInt8]) -> Result<UInt64, IOError>;
    pub fn write_all(&mut self, data: &[UInt8]) -> Result<Unit, IOError>;
    pub fn write_str(&mut self, text: &str) -> Result<UInt64, IOError>;
    pub fn flush(&mut self) -> Result<Unit, IOError>;
    pub fn close(self) -> Result<Unit, IOError>;
}

pub fn read(path: &str, max_bytes: UInt64) -> Result<Vec<UInt8>, IOError>;
pub fn read_text(path: &str, max_bytes: UInt64) -> Result<String, IOError>;
pub fn write(path: &str, data: &[UInt8]) -> Result<Unit, IOError>;
pub fn write_text(path: &str, text: &str) -> Result<Unit, IOError>;
```

### 2.2 Out of scope

Do not implement:

```text
OpenOptions combinations
append
truncate
seek
stream_position
set_len
sync_data
sync_all
metadata
symlink_metadata
exists/is_file/is_dir
directory creation
directory listing
remove_file
remove_dir
remove_dir_all
rename
copy
canonicalize
symlink handling
permissions
file locking
async I/O
memory-mapped I/O
watchers
temporary-file APIs
```

Do not introduce a generic public provider/plugin system.

---

## 3. Required compiler and package binding work

### 3.1 Reuse the existing provider binding mechanism

Inspect the P1 REST workload and compiler path that supports `provider_api`.

Reuse that mechanism for `filesystem`. Do not create a second source-binding system.

The package must bind:

```text
capability: filesystem
resource type: file
provider crate: stark-file/native
```

Application source must call approved package APIs rather than raw arbitrary ABI symbols.

### 3.2 Resource nominal binding

Bind the public STARK `File` resource nominal to provider resource type `file`.

Required properties:

- `File` is move-only and never structurally `Copy`.
- Successful open/create returns one live owned resource.
- Failed open/create returns no live resource.
- Borrowed methods do not consume the resource.
- `close(self)` consumes the resource exactly once.
- Implicit MIR drop closes a still-live resource exactly once.
- Explicit close prevents a later implicit close.
- Use after close is rejected or impossible through ownership semantics.
- A duplicated provider handle cannot be constructed from STARK source.

### 3.3 Provider function mapping

| Package operation | Provider symbol | Resource mode |
|---|---|---|
| `File::open` | `stark_file_open` | handle out |
| `File::create` | `stark_file_create` | handle out |
| `File::read` | `stark_file_read` | borrowed mutable handle |
| `File::write` | `stark_file_write` | borrowed mutable handle |
| `File::flush` | `stark_file_complete` | borrowed mutable handle |
| `File::close` | `stark_file_close` | consumed handle |

The mapping must be declared in provider metadata or package binding records, not inferred from naming conventions.

### 3.4 Buffer mapping

Prove that package bindings lower:

```text
&str
&[UInt8]
&mut [UInt8]
```

to:

```text
BorrowedBuffer
BorrowedBufferMut
```

Required checks:

- zero-length buffers are valid;
- non-zero buffers cannot have null pointers;
- input buffers remain borrowed;
- mutable output buffers are written only within bounds;
- reported byte counts cannot exceed the supplied buffer length;
- STARK-visible lengths use `UInt64` safely;
- host `usize` conversions are checked where needed.

### 3.5 Output-slot discipline

For every provider call:

- output values are read only on success;
- failed creation does not expose an uninitialised handle;
- failed read does not expose undefined byte-count or EOF values;
- failed write does not expose an undefined accepted-byte count;
- output slots comply with current provider ABI rules.

---

## 4. Error model

### 4.1 Required `IOError` subset

```stark
pub enum IOError {
    NotFound,
    PermissionDenied,
    InvalidInput,
    InvalidData,
    IsDirectory,
    AlreadyExists,
    Unsupported,
    LimitExceeded,
    UnexpectedEof,
    WriteZero,
    Other,
}
```

Keep the existing package vocabulary where already frozen.

### 4.2 Provider status mapping

| Provider status | `IOError` |
|---|---|
| `STATUS_NOT_FOUND` | `NotFound` |
| `STATUS_PERMISSION_DENIED` | `PermissionDenied` |
| `STATUS_INVALID_INPUT` | `InvalidInput` |
| `STATUS_INVALID_ENCODING` | `InvalidData` |
| `STATUS_IS_DIRECTORY` | `IsDirectory` |
| `STATUS_ALREADY_EXISTS` | `AlreadyExists` |
| `STATUS_UNSUPPORTED` | `Unsupported` |
| `STATUS_OTHER_DECLARED` | `Other` |

Undeclared status codes remain provider contract violations and must not be silently mapped to `Other`.

### 4.3 Helper-generated errors

- `LimitExceeded` when a bounded read would exceed `max_bytes`.
- `InvalidData` for malformed UTF-8.
- `WriteZero` when `write_all` receives a successful zero-byte write before completion.
- `UnexpectedEof` only where a future exact-read helper requires it.

Do not use lossy UTF-8 conversion.

---

## 5. Method semantics

### 5.1 `File::open`

- Opens an existing file for reading.
- Returns `NotFound` if absent.
- Returns `IsDirectory` where the platform/provider reports it.
- Returns one owned `File` on success.
- Schedules no close on failure.

### 5.2 `File::create`

The current provider uses `write(true).create_new(true)`.

Therefore current semantics are:

> create a new file and fail if it already exists.

Do not document it as truncate-or-create unless provider semantics are deliberately changed. If the frozen package API intended truncate-or-create, resolve that mismatch explicitly before closure.

### 5.3 `File::read`

- Reads at most `buffer.len()` bytes.
- Returns the byte count.
- Zero bytes at EOF is valid.
- Repeated reads continue from the cursor.
- Empty buffer returns zero safely.
- No allocation inside the primitive method.

### 5.4 `File::write`

- Attempts one partial write.
- Returns accepted byte count.
- Does not promise complete transfer.
- Empty input may return zero successfully.
- Repeated writes advance the cursor.

### 5.5 `File::write_all`

Loop until all bytes are written:

```text
written > 0  -> advance
written == 0 before completion -> WriteZero
provider error -> return that IOError
```

The loop must not duplicate or lose the `File` resource.

### 5.6 `File::flush`

Calls `stark_file_complete`.

Document this as flush semantics only. Do not claim durable storage guarantees equivalent to `sync_all`.

### 5.7 `File::close`

Consumes `self` and:

- calls provider close exactly once;
- removes the provider handle;
- suppresses implicit close after success;
- follows the approved resource-consumption rule on close error;
- never leaves a value usable after consumption.

### 5.8 `File::read_to_end`

Use bounded chunked reading:

```text
result = empty Vec<UInt8>
scratch = fixed bounded chunk
loop:
    read scratch
    if count == 0: return result
    if result.len + count > max_bytes: return LimitExceeded
    append exactly count bytes
```

Requirements:

- no unbounded allocation;
- checked length arithmetic;
- no append of uninitialised bytes;
- `max_bytes = 0` succeeds only for an empty file;
- choose and document a fixed chunk size such as 8 KiB or 16 KiB.

### 5.9 `File::read_to_string`

Call `read_to_end(max_bytes)` and perform strict UTF-8 validation.

Invalid UTF-8 returns `InvalidData`.

### 5.10 Whole-file helpers

Whole-file helpers must:

1. open/create the file;
2. perform the operation;
3. close exactly once on success;
4. close on intermediate error where the resource remains live;
5. preserve the primary operation error under an explicit close-error policy.

Recommended policy:

```text
operation succeeds + close fails -> return close error
operation fails + close also fails -> return operation error
```

Record this policy in tests.

---

## 6. Qualification package

Because library-only native qualification is incomplete, create a bounded executable package, for example:

```text
starkc/tests/workloads/io-minimal/
```

It must import `stark-io` and execute real filesystem operations from ordinary STARK source.

Do not qualify solely through Rust provider tests or hand-built MIR.

---

## 7. Required test matrix

### 7.1 Provider tests

Cover:

- create;
- open;
- write;
- flush;
- close;
- reopen;
- read;
- EOF;
- empty read;
- empty write;
- missing path;
- existing-file create failure;
- directory open refusal;
- invalid handle type;
- invalid handle ID;
- symbol/metadata agreement.

### 7.2 Source-level positive tests

1. create → write → flush → close;
2. open → read → close;
3. implicit close at scope end;
4. explicit close suppresses implicit close;
5. `write_all` across multiple partial writes;
6. `read_to_end` across multiple chunks;
7. `read_to_string` valid UTF-8;
8. whole-file `write` then `read`;
9. whole-file `write_text` then `read_text`;
10. empty file;
11. zero-byte write;
12. zero-byte read buffer;
13. repeated open/close;
14. two independent files live simultaneously.

### 7.3 Source-level negative tests

1. missing file → `NotFound`;
2. create existing file → `AlreadyExists`;
3. directory passed to open → `IsDirectory` where portable;
4. invalid UTF-8 → `InvalidData`;
5. content above `max_bytes` → `LimitExceeded`;
6. use after move rejected;
7. use after explicit close rejected;
8. copying `File` rejected;
9. failed creation schedules no close;
10. zero-progress `write_all` → `WriteZero` using a deterministic test provider if needed.

### 7.4 Lifecycle tests

Observe close behaviour directly:

```text
explicit close                    exactly 1 close
implicit drop                     exactly 1 close
error after successful open       exactly 1 close
failed open                       exactly 0 closes
move into helper                  exactly 1 close
early return                      exactly 1 close
? propagation                     exactly 1 close
loop with file resource           no live-slot overwrite
two files                         each closed exactly once
```

Use deterministic instrumentation where the real provider cannot expose close counts.

### 7.5 Evidence class

Provider-backed file operations cannot automatically inherit the ordinary three-engine equality claim.

Record evidence separately as:

```text
front-end acceptance/refusal
MIR/provider-call verification
native debug execution
native release execution
provider ABI tests
```

Do not claim reference-interpreter equivalence unless implemented.

---

## 8. Cross-platform requirements

Qualify on:

```text
linux-x64
macos-arm64
windows-x64
```

Use temporary directories supplied by the test framework.

Do not hard-code:

- `/tmp`;
- slash direction;
- drive letters;
- Unix permission assumptions;
- deletion of still-open files;
- rename-over-existing behaviour;
- directory-open error categories unless deliberately normalised.

---

## 9. Installed-toolchain qualification

Run the executable qualification package through:

1. the in-repository toolchain;
2. the installed toolchain with source-tree fallback disabled.

Use:

```text
STARK_REQUIRE_INSTALLED_RUNTIME=1
```

or the current equivalent.

This must prove:

- runtime discovery works;
- provider discovery follows the runtime tree;
- the installed repository-shaped layout works;
- no `stark-provider-abi` path collision occurs;
- the file provider is installed and linkable.

---

## 10. Documentation updates

Update:

```text
stark-io/README.md
stark-io/BLOCKERS.md
STARKLANG/docs/packages/STARK-IO-v0.1-Codex-Implementation-Spec.md
COMPILER-STATE.md
starkc/docs/compiler/evidence/io-minimal/README.md
```

`BLOCKERS.md` must distinguish:

```text
CLOSED:
- minimal source-level provider binding
- minimal file handle API

OPEN:
- library-only native qualification mode
- full open options
- seek
- durable sync
- metadata
- path and directory operations
- rename/delete/copy
- complete cross-platform IOError vocabulary
```

---

## 11. Suggested commit sequence

```text
IO-01  baseline and exact source-binding path
IO-02  File resource/provider_api binding
IO-03  primitive open/create/read/write/flush/close methods
IO-04  IOError mapping and negative cases
IO-05  bounded read helpers and strict UTF-8
IO-06  write_all and whole-file helpers
IO-07  lifecycle qualification
IO-08  installed-toolchain and Tier-1 qualification
IO-09  evidence and bounded closure
```

Use owner-issued CD numbers only during integration.

Each commit message should state:

```text
problem
finding
change
evidence
claim boundary
remaining unsupported surface
```

---

## 12. Acceptance criteria

The work package closes only when:

1. Ordinary STARK source can call the minimal `stark-io` API.
2. `File` maps to provider resource type `file`.
3. Open/create/read/write/flush/close execute natively.
4. Explicit and implicit close occur exactly once.
5. Failed resource creation schedules no close.
6. Partial reads and writes are represented honestly.
7. `write_all` detects zero progress.
8. `read_to_end` is bounded by `max_bytes`.
9. Invalid UTF-8 returns `InvalidData`.
10. Whole-file helpers close resources on success and error.
11. Native debug and release pass.
12. Linux x64, macOS arm64, and Windows x64 pass.
13. The installed toolchain executes the qualification workload.
14. Provider metadata and linked symbols agree.
15. No claim is made for seek, metadata, directories, rename, delete, copy, append, truncate, or durable sync.
16. Evidence records exact compiler and package commits.

---

## 13. Permitted closure statement

```text
STARK minimal synchronous file I/O is implemented and Tier-1 qualified for
open/create/read/write/flush/close, bounded whole-file reads, strict UTF-8 text
reads, and whole-file write helpers through the first-party filesystem provider.

The broader stark-io v0.1 filesystem surface remains incomplete.
```

Do not use:

```text
full filesystem support
complete stark-io v0.1
production-ready filesystem API
cross-platform filesystem semantics complete
```

---

## 14. Stop conditions

Stop and escalate if implementation reveals:

- provider-bound resource nominals can become `Copy`;
- borrowed file receivers lower as moves;
- explicit close does not suppress implicit drop;
- output slots are read on provider failure;
- package APIs require raw ABI symbol calls from application source;
- a new MIR/runtime surface change is required without approval;
- Windows requires a public semantic policy not already defined;
- existing `File::create` semantics conflict with the frozen public API;
- buffer lowering cannot safely express `&mut [UInt8]`;
- the installed toolchain still mixes runtime and provider trees.

---

## 15. Final instruction

Prioritise one honest, useful slice over broad API coverage.

The success condition is that a normal STARK program can safely:

```text
create a file
write bytes or text
flush and close it
open it again
read bounded bytes or strict UTF-8 text
```

through the native compiler and installed toolchain on all Tier-1 platforms.

Anything beyond that belongs to a later `stark-io` work package.
