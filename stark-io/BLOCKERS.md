# stark-io blockers

## THE BLOCKER: Core `file` is still on the legacy path

**The minimal native slice is written but cannot execute, and this is the reason.**

`starkpkg.json` binds the nominal `NativeFile` to the provider resource `file`. That resource is
Core-owned — `ResourceRegistry::builtin()` maps it to `LegacyCore(CoreType::File)` — and CD-224
rejects a package that claims it.

The slice was first made to run by removing three compiler guards: CD-224's manifest check, and two
string-keyed exemptions in the MIR verifier (MIR-0027, and the rule that MIR owns a resource's only
close). Together those put `file` on the `HostResource` path for selected rules while it kept legacy
direct-close semantics — one resource name, two MIR representations, two destruction paths. That is
precisely the half-migration SELECT-C exists to refuse, and it is what CD-235's
`partially_migrated_core` guard was written to catch. The three guards are restored.

**What unblocks it:** migrating `file` off the legacy path WHOLLY — Route B's representation and
lifecycle work. A complete migration is already permitted; only the partial one is refused. When it
lands, remove the `#[ignore]` from `io_minimal_executes_from_source_through_stark_io_package` in
`starkc/tests/c788_starkc_build.rs`. Nothing in this package needs to change.

Everything below is written and compiles. Only the end-to-end execution path is blocked.

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

The public nominal is `NativeFile` instead of the target `File` because the compiler still reserves
Core `File` on the legacy path. Migrating the exact public name remains open — and note that
choosing a different *nominal* does not avoid the blocker above, because the binding still claims
Core's *resource identity*. The nominal and the resource are separate names, and it is the second
one CD-224 governs.

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
