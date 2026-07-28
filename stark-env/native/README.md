# stark-env native provider

Standalone Native Provider ABI v0.1 crate for process arguments and environment reads.

The provider is read-only. It does not mutate process environment, perform shell interpretation,
expand variables, or choose hidden defaults. Argument order is `std::env::args_os()` order and
includes argument zero. Arguments and environment values are encoded as UTF-8 bytes. Non-UTF-8
arguments, names, or values are rejected with `INVALID_ENCODING`; lossy replacement is not used.

Argument output is a single NUL-separated byte sequence. Empty arguments are represented by
adjacent separators, and the exact required byte length includes separators between arguments.
Environment reads distinguish absent variables from present-but-empty variables.

Environment names reject empty names and embedded `=` or NUL bytes. Environment case sensitivity
follows the host platform: case-sensitive on Unix-family targets and conventionally
case-insensitive on Windows. The provider performs no normalization.
