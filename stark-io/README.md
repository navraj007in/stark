# stark-io

`stark-io` is the proposed host-backed synchronous file and filesystem package for STARK.

This checkout implements a minimal native synchronous file-I/O slice through the first-party
`stark-file/native` provider and the existing `provider_api` package binding path.

Implemented provider-backed surface:

- `open_file`, `create_file`
- `file_read`, `file_write`, `file_flush`, `file_close`
- `file_read_to_end`, `file_read_to_string`
- `file_write_all`, `file_write_str`
- `read`, `read_text`, `write`, `write_text`

The bound resource nominal is currently `NativeFile`, not `File`. The compiler still has a Core
`File` identity on the legacy path, so exporting the exact WP-IO.1 `File` name requires a separate
Core-name migration. `NativeFile` is an opaque provider resource nominal synthesized from
`starkpkg.json`; ordinary STARK source cannot construct a handle directly.

The attached v0.1 spec names the required capability `file`; this checkout's first-party registry
currently exposes the same provider under `filesystem`, so the manifest uses `filesystem`.

This is not the full `stark-io` v0.1 filesystem surface. Seek, metadata, directories, rename,
delete, copy, append, truncate, open-options combinations, durable sync, async I/O, and complete
cross-platform filesystem semantics remain out of scope.
