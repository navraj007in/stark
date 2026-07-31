# stark-io

`stark-io` is the proposed host-backed synchronous file and filesystem package for STARK.

This checkout implements native synchronous file I/O through the first-party `stark-file/native`
provider and the existing `provider_api` package binding path.

Implemented provider-backed surface:

- `open_file`, `create_file`, `open_with_options`
- `file_read`, `file_write`, `file_flush`, `file_sync`, `file_close`
- `file_read_to_end`, `file_read_to_string`
- `file_write_all`, `file_write_str`
- `file_seek`, `file_set_length`, `file_metadata`
- `read`, `read_text`, `write`, `write_text`
- `path_exists`, `path_metadata`, `path_join`, `path_is_valid`, `path_is_absolute`
- `remove_file`, `rename`, `copy_file`
- `create_dir`, `remove_dir`, `read_dir`

`file_close` takes the handle **by value and calls nothing**: `NativeFile` is a host resource, so
MIR owns its only destruction path and drop elaboration emits the close. A close error is therefore
not observable — call `file_flush` (or `file_sync` for durability) first if you need to see one.

The bound resource is `io_file` — this package's own resource identity, NOT Core's `file`. Core
owns `file` and a package may not claim it (CD-224); `io_file` is declared by the same provider
under its own symbols and its own handle tag, so the package gets an owned, moved, exactly-once-closed
handle with no compiler guard weakened. The nominal is `NativeFile`, not `File`: exporting the exact
WP-IO.1 `File` name still requires Core `File`'s own migration off the legacy path. `NativeFile` is an opaque provider resource nominal synthesized from
`starkpkg.json`; ordinary STARK source cannot construct a handle directly.

The attached v0.1 spec names the required capability `file`; this checkout's first-party registry
currently exposes the same provider under `filesystem`, so the manifest uses `filesystem`.

Seek, metadata, directories, rename, delete, copy, append, truncate, open-options combinations and
durable sync are all implemented and executed end-to-end.

Out of scope, and deliberately so in two cases: **recursive directory creation and recursive
delete** — both are unbounded effects from a single call and the second is the most destructive
filesystem primitive there is, so callers walk with `read_dir` and act on what they have seen.
Genuinely not yet done: symlink creation and resolution, async I/O, and the complete cross-platform
`FileError`/`IOError` vocabulary. See `BLOCKERS.md`.
