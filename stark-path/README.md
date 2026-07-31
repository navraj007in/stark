# stark-path

`stark-path` provides pure lexical path operations.

Implemented:

- explicit `PathStyle::{Unix, Windows}`;
- `join`;
- `components`;
- `parent`;
- `file_name`;
- `stem`;
- `extension`;
- `with_extension`;
- `is_absolute`;
- lexical `normalize`.

This package does not access the filesystem, resolve symlinks, inspect the current directory, or canonicalize paths through host APIs.
