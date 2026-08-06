# stark-args

`stark-args` is a small, pure-STARK command-line argument parser. Callers provide the argument
tokens to parse; a CLI can obtain those tokens from `stark-env` and then pass them here.

Implemented:

- long flags such as `--verbose`;
- long options with a separate value, such as `--limit 10`;
- long options with an equals value, such as `--limit=10`;
- short flags and options such as `-v` and `-l 10`;
- positional arguments;
- `--` terminator;
- required options;
- default option values;
- repeated options where declared;
- deterministic `UnknownOption`, `MissingValue`, `DuplicateOption`, `RequiredOptionMissing`, and
  `InvalidSpec` errors;
- generated usage text.

Excluded from v0.1:

- subcommands;
- short clustering such as `-abc`;
- negated flags such as `--no-color`;
- environment-variable fallbacks;
- config-file merging;
- derive macros, reflection, closures, or trait-object based dispatch.
