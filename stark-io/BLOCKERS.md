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

## Surface written and executing

Normal STARK source reaches the first-party filesystem provider through the `stark-io` package
using the existing `provider_api` binding mechanism.

**The minimal slice** (CD-290/CD-291):

- provider-bound resource nominal for `io_file` as `NativeFile`;
- `open_file`, `create_file`;
- `file_read`, `file_write`, `file_flush`, `file_close`;
- `file_read_to_end`, `file_read_to_string`;
- `file_write_all`, `file_write_str`;
- whole-file `read`, `read_text`, `write`, `write_text`.

**The expanded surface** (CD-292) — see "Provider surface expansion" below for what each replaced:

- `open_with_options` (with `default_open_options`, `open_options_are_valid`);
- `file_seek`, `file_sync`, `file_set_length`;
- `file_metadata`, `path_metadata`, `path_exists`;
- `path_join` — refuses an absolute child rather than letting it silently replace the base;
- `remove_file`, `rename`, `copy_file`;
- `create_dir`, `remove_dir`, `read_dir`.

Four types that were declared with no operation behind them — `OpenOptions`, `SeekFrom`,
`FileMetadata`, `DirectoryEntry` — are now all consumed.

The public nominal is `NativeFile` rather than `File` because Core reserves `File`. That is now a
naming question rather than a blocker: `NativeFile` is a fully working owned file handle, and the
only thing the Core name would add is the spelling. Adopting it needs Core `File`'s own migration.

## Library package testing — THREE gaps, previously conflated

This section has now been wrong twice in different ways. The original said a library package could
not be tested at all. CD-298 corrected that to "`stark test` works, `stark build` needs an
entrypoint" — which was also wrong, because it was never verified against a package that declares
`provider_api`. Running it is what settled it.

**1. `stark test` does not synthesize `provider_api`.** This is the blocker that actually stops this
package. `native_build.rs` calls `provider_synth::synthesize_with_resources` before the front end
runs, so the generated `*_raw` functions exist for a native build. `cmd_test` in `src/bin/stark.rs`
does no provider handling at all, so under `stark test` every one of them is E0200 "undefined
variable" and the package fails to compile before a single test is discovered:

```
Error: [E0200] undefined variable 'file_open_options_raw'
Error: [E0200] undefined variable 'dir_list_raw'          ... and 17 more
stark-io: package compilation failed
```

Both first-party packages that declare `provider_api` — `stark-io` and `stark-random` — are affected.
`src/tests.stark` here is written and valid: it type-checks and compiles under the native build (the
io e2e tests vendor it and pass). It simply cannot be RUN yet.

Closing this means extracting the synthesis pipeline `native_build.rs` owns — `required_capabilities`
→ provider set → `derive_all` → `synthesize_with_resources` → `merge_layer` — into something
`cmd_test` can call. Note the semantic limit that survives it: `stark test` executes through the
reference interpreter, which cannot perform a provider call. Even with synthesis, only tests that
avoid the provider would run — which is exactly what `src/tests.stark` was written to be.

**2. There is no `#[test]` attribute.** `#` is not in STARK's lexer, so writing one is a lex error
that takes the whole module down with it. Tests are discovered by the `test_` NAME PREFIX, taking no
parameters and no receiver (`test_runner::discover_tests`). `stark-random` had exactly this defect
(CD-297). Necessary to get right, but not sufficient — gap 1 comes first.

**3. `stark build` still requires an entrypoint.** It reports `program without a main function`, so a
library cannot be NATIVELY qualified without an artificial `main`. This is why
`io_minimal_executes_...` and `io_expanded_surface_executes_...` are consumer programs in
`starkc/tests/c788_starkc_build.rs` rather than package tests.

## Provider surface expansion — DELIVERED (CD-292)

This section listed open options, append, truncate, create-new, seek, set length, sync, metadata,
path operations, directory operations, remove, rename and bounded copy as missing. **All of them
landed in CD-292**, which took `io_file` from 6 native symbols to 19 and added the package
operations that consume them:

| Was missing | Now |
| --- | --- |
| open options, append, truncate, create-new | `stark_iofile_open_options` / `open_with_options` |
| seek | `stark_iofile_seek` / `file_seek` |
| set length | `stark_iofile_set_len` / `file_set_length` |
| durable sync | `stark_iofile_sync` / `file_sync` — distinct from `file_flush` |
| metadata | `stark_iofile_metadata`, `stark_iopath_metadata` / `file_metadata`, `path_metadata` |
| path operations | `stark_iopath_exists` / `path_exists`, `path_join` |
| directory operations | `stark_iodir_create` / `_remove` / `_list` / `create_dir`, `remove_dir`, `read_dir` |
| remove, rename, bounded copy | `stark_iofile_remove` / `_rename` / `_copy` |

Exercised end-to-end by `io_expanded_surface_executes_from_source_through_stark_io_package`, which
asserts on observed values — seek positions, metadata lengths, listing composition, cleanup — not on
the absence of an error.

**Deliberately NOT provided, and not a gap:** recursive directory creation and recursive delete.
Both are unbounded effects from one call, and the second is the most destructive filesystem
primitive there is. Callers walk with `read_dir` and act on what they have seen.

## Still open

Everything the previous version of this list named as missing surface is delivered; what remains is
naming, tooling, and one recorded defect.

- **exact public `File` nominal/method API** instead of `NativeFile` plus free wrappers. A naming
  question, not a capability one — it needs Core `File`'s own migration off the legacy
  `MirTy::Core` path, which is a three-engine change.
- **`stark test` does not synthesize `provider_api`** — the blocker that stops this package's own
  tests running, and then separately **library-only NATIVE qualification**. See "Library package
  testing" above; they are three gaps, not one.
- **the `IOError::` status labels name variants Core's `IOError` does not have** —
  `InvalidData`, `IsDirectory`, `Unsupported`. Core has five variants and `Other(String)` absorbs
  the surplus. The strings are consumed at exactly one site, interpolated into a generated-code
  COMMENT (`emit_provider.rs`), so this is a naming defect rather than a conformance break. The
  package's own enum is `FileError`, which does have them. Recorded in CD-290 and unfixed.
- **symlink creation and reading.** `path_metadata` reports `FileType::Symlink` (it deliberately
  does not follow links), but nothing creates or resolves one.
- **no zeroization.** `secure`-style callers aside, `read`/`read_text` leave file contents in
  buffers that are never scrubbed; STARK has no primitive for it today.
- **complete cross-platform `FileError`/`IOError` vocabulary.**
