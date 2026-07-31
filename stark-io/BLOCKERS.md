# stark-io blockers

## RESOLVED: the package has its own resource identity

**The minimal native slice executes.** `io_minimal_executes_from_source_through_stark_io_package`
opens, writes, reads and closes a real file from ordinary STARK source, through the first-party
provider, in a natively built binary.

It did not get there by migrating Core `File`, and the history is worth keeping because the wrong
answer was tried first.

`starkpkg.json` originally bound `NativeFile` to the provider resource **`file`**, which is
Core-owned — `ResourceRegistry::builtin()` maps it to `LegacyCore(CoreType::File)` — and CD-224
forbids a package from claiming it. The slice was made to run by deleting that guard and adding two
string-keyed exemptions to the MIR verifier. That produced the state SELECT-C exists to refuse:
`file` on the `HostResource` path for selected rules while Core `File` kept legacy direct-close
semantics. One resource name, two MIR representations, two destruction paths. All three guards were
restored and the end-to-end test was `#[ignore]`d for one commit.

**The actual fix was to notice the package never needed Core's identity.** `stark-io` now binds
**`io_file`**: its own resource type, declared by the same provider under its own symbols
(`stark_iofile_*`) and its own type tag. It is absent from the builtin registry, so CD-224 admits
it; it is not `LegacyCore`, so MIR-0027 does not fire; it is wholly on the `HostResource` path, so
MIR owns its only close and the A11 §5 rule 4 guard holds **with no exemption anywhere**. A
`NativeFile` is owned, moved, and closed exactly once from a `Drop` terminator — the same lifecycle
`tcp_stream` has.

Two consequences worth knowing:

- **`file_close(file)` takes the handle by value and calls nothing.** Taking ownership *is* the
  close; drop elaboration emits it. Calling the close symbol directly is rejected as a second
  destruction path — correctly, since both would run. A close error is therefore not observable;
  call `file_flush` first if you need to see one.
- **Core `File` is untouched.** Its migration off the legacy `MirTy::Core` path remains open and is
  a three-engine change (checker, reference interpreter, native backend). Nothing here depends on
  it any more.

## Written in the minimal native slice

Normal STARK source reaches the first-party filesystem provider through the `stark-io` package
using the existing `provider_api` binding mechanism — once the binding above is legal.

Surface written:

- provider-bound resource nominal for `file` as `NativeFile`;
- `open_file`, `create_file`;
- `file_read`, `file_write`, `file_flush`, `file_close`;
- `file_read_to_end`, `file_read_to_string`;
- `file_write_all`, `file_write_str`;
- whole-file `read`, `read_text`, `write`, `write_text`.

The public nominal is `NativeFile` rather than `File` because Core reserves `File`. That is now a
naming question rather than a blocker: `NativeFile` is a fully working owned file handle, and the
only thing the Core name would add is the spelling. Adopting it needs Core `File`'s own migration.

## Library package build mode

`stark build` currently reports `program without a main function` for this library package. The
source file parses and reaches native build planning, but there is no separate library/package API
check command to qualify a package without adding an artificial executable entrypoint.

## Provider surface expansion

`stark-file/native` currently covers the first file-handle subset only. The full `stark-io` v0.1
surface also needs provider metadata and native symbols for open options, append, truncate,
create-new, seek, set length, sync, metadata, path operations, directory operations, remove, rename
and bounded copy.

## Still open after this slice

- exact public `File` nominal/method API instead of `NativeFile` plus free wrappers;
- library-only native qualification mode;
- full open options;
- seek;
- durable sync;
- metadata;
- path and directory operations;
- rename/delete/copy;
- complete cross-platform `FileError`/`IOError` vocabulary.
