# stark-io blockers

## Source-level provider binding

The compiler can validate and emit resource-carrying provider calls in lower-level tests, but normal
STARK package source cannot yet bind `File` methods to `stark-file/native` provider symbols.

Blocked APIs:

- `File::open`
- `File::create`
- `File::read`
- `File::write`
- `File::flush`
- `File::close`
- all whole-file convenience helpers that depend on those methods

## Library package build mode

`stark build` currently reports `program without a main function` for this library package. The
source file parses and reaches native build planning, but there is no separate library/package API
check command to qualify a package without adding an artificial executable entrypoint.

## Provider surface expansion

`stark-file/native` currently covers the first file-handle subset only. The full `stark-io` v0.1
surface also needs provider metadata and native symbols for open options, append, truncate,
create-new, seek, set length, sync, metadata, path operations, directory operations, remove, rename
and bounded copy.
