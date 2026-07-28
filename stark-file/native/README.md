# stark-file native provider

Standalone Native Provider ABI v0.1 crate for the future Core `File` surface.

Paths are UTF-8 byte buffers. Embedded NUL bytes and invalid UTF-8 are rejected as recoverable
`INVALID_ENCODING`/`INVALID_INPUT` statuses. Relative paths are resolved by the host process
working directory at the moment of the explicit provider call; the provider never changes it.
Absolute paths, Unix separators, Windows separators, and Windows drive prefixes are passed to the
host filesystem APIs without runtime normalization. There is no tilde expansion, environment
expansion, shell interpretation, trust policy, or hidden base path.

The ABI declares one resource type, `file`. Ordinary operations borrow handles. `file_complete`
is a recoverable pre-close operation. `file_close` is the mandatory ABI close function and consumes
the handle exactly once.
