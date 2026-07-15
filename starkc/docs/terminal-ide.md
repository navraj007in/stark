# STARK terminal IDE

`starkide` is a dependency-free, terminal-native compiler workbench inspired
by the Turbo C++ blue/white interface.

```bash
cargo run --bin starkide
cargo run --bin starkide -- examples/gate3/01_hello.stark
```

## Editing

- `Ctrl+N`, `Ctrl+O`, `Ctrl+S`: new buffer, open, save
- `Ctrl+B`: rotate through open buffers
- `Ctrl+Z`, `Ctrl+Y`: undo and redo (bounded to 512 snapshots)
- `Shift+Arrow`: select Unicode text across lines
- `Ctrl+X`, `Ctrl+C`, `Ctrl+V`: internal cut, copy, paste
- `Ctrl+F`, `F3`: find and find next with wraparound
- `Ctrl+G`: go to line

New and opened files remain in separate buffers. Closing a dirty buffer or
quitting with unsaved buffers requires Save, Discard, or Cancel confirmation.

## Projects and persistence

The File menu can browse `.stark` files beneath the configured project root
or reopen one of the ten most recent files. Project traversal excludes `.git`
and `target`, is depth- and count-bounded, and therefore remains responsive in
large repositories.

The project root, recent files, parser mode, and messages-pane visibility are
stored in `~/.starkide-state`. Source files and clipboard contents are never
written to that state file.

## Build and run

- `F9`: parse, resolve, type-check, and borrow-check
- `Ctrl+F9`: build and run through the Gate 3 interpreter
- `F4`: jump to the next compiler/runtime diagnostic

Build messages and program output are retained separately and can be selected
from the Run menu. Warnings do not prevent execution; errors do. Runtime
failures carry source locations and remain navigable with `F4`.

## Terminal behavior

The layout recalculates while the terminal is resized, expands the output pane
on larger screens, and keeps the cursor visible through horizontal and
vertical scrolling. Below 48×14 it displays a stable minimum-size message
instead of drawing overlapping controls. Terminal state is restored on normal
exit and panic.
