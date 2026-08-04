# stark-glob

`stark-glob` provides a pure lexical glob matcher.

Implemented:

- literal matching;
- `*`;
- `?`;
- bracket character classes;
- byte ranges inside character classes;
- explicit `GlobStyle::{Unix, Windows}` separator behavior.

Wildcards do not match path separators. This package is a matcher only; it does not traverse the filesystem.
