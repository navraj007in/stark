# STARK I/O Package v0.1 Codex Implementation Spec

**Status:** Proposed implementation slice  
**Package:** `stark-io`  
**Primary module:** `stark_io`  
**Normative capability:** `file`  
**Current repo provider capability:** `filesystem`  
**Source request:** `/Users/nexper/.codex/attachments/a0508c93-9962-48e8-976c-85dce5241fc9/pasted-text.txt`

This document records the repo-facing implementation target for the attached STARK File and I/O
Package Specification v0.1. It keeps the package name `stark-io`, while treating the existing
`stark-file/native` crate as the first native provider candidate for the file-handle subset.

## Public Contract

`stark-io` provides synchronous, deterministic file and basic filesystem operations for STARK
programs. Public paths are UTF-8 `str` values. Host path objects, file descriptors, Windows handles
and platform errno values are not exposed.

The attached specification says the package is host-backed and requires:

```json
{
  "capabilities": ["file"]
}
```

The current first-party provider registry names the implemented file provider capability
`filesystem`. The source package therefore declares `filesystem` until the registry is renamed or an
alias is introduced.

The v0.1 API defines:

- move-only `File` resources closed exactly once by explicit `File::close` or MIR `Drop`;
- structured `IOError` categories;
- explicit distinction between partial read/write attempts and complete transfer helpers;
- bounded allocation for whole-file reads and directory listing;
- UTF-8 text reads that reject malformed data without lossy decoding;
- append, truncate, seek, flush and durability operations;
- metadata, basic path inspection, directory creation/listing/removal, rename, delete and copy.

## Initial Implementation Slice

The first source package slice is limited to the attached spec's section 26:

```stark
pub resource File;
pub enum IOError { ... }

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

The package also carries pure definitions for `OpenOptions`, `SeekFrom`, `FileType`,
`FileMetadata`, `DirectoryEntry` and lexical path helpers so later slices can fill in the host-backed
calls without changing the public type vocabulary.

## Current Compiler Boundary

The repository has Native Provider ABI v0.1 support and an existing `stark-file/native` provider.
Provider resource lifecycle is exercised in lower-level Rust tests, including
`starkc/tests/c784_file_e2e.rs` and `starkc/tests/c788_resource_lifecycle.rs`.

Source-level STARK package functions still cannot directly invoke the file provider from ordinary
package source. Until that gap closes, `stark-io/src/lib.stark` exposes the API shape and pure
validation helpers, but host-backed methods deliberately trap with a clear message.

## Provider Mapping Target

The existing provider candidate currently exposes:

- `stark_file_open`
- `stark_file_create`
- `stark_file_read`
- `stark_file_write`
- `stark_file_complete`
- `stark_file_close`

It covers the minimal open/create/read/write/flush/close subset, but does not yet cover the full
`stark-io` v0.1 surface:

- open-options combinations such as append, truncate and create-new;
- seek and position;
- file length changes;
- `sync_data` and `sync_all`;
- metadata and symlink metadata;
- path, directory, rename, delete and copy operations;
- the complete `IOError` vocabulary from the package spec.

## Acceptance Criteria

`stark-io` v0.1 is complete when:

1. the frozen public API checks under the compiler;
2. host-backed calls bind through approved provider metadata;
3. explicit and implicit close execute exactly once;
4. failed resource creation never schedules close;
5. bounded whole-file reads reject over-limit input with `LimitExceeded`;
6. invalid UTF-8 maps to `InvalidData`;
7. directory operations enforce entry and depth bounds;
8. Linux x64, macOS arm64 and Windows x64 qualification passes;
9. evidence records one exact compiler commit and package commit.
