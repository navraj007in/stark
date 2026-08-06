# stark-fmt v0.1 Evidence

## Audit

- Package: `stark-fmt`
- Package path: `packages/stark-fmt`
- Repository head when this evidence was produced: `b6a4960` (2026-08-04), plus the working-tree
  changes of **DEV-DISPLAY-DISPATCH** — this package does not check, test or build without them,
  which is the point of it
- Platform: macOS host (Darwin 25.5.0), `Australia/Sydney`
- Toolchain: repository `starkc/target/debug/stark`
- Result: `READY` — all qualification steps pass locally; the tree-wide qualification run is the
  authority for the release claim

## What this package is evidence OF

`stark-fmt` is not evidence that formatting is complete. It is evidence that a **generic
`T: Display` bound is usable outside the compiler's own tests**: `Line::value` and `to_string` are
ordinary generic functions whose only tool is `Display::fmt`, and they work for every standard
`Display` primitive and for a user type with its own `impl Display`, in the interpreter and
natively.

Before DEV-DISPLAY-DISPATCH this package could not exist. `x.fmt()` inside
`fn value<T: Display>(self, value: &T)` was rejected with
`[E0302] method 'fmt' not found for type 'T'`.

## Prerequisites

| Capability | Result | Evidence |
|---|---|---|
| `T: Display` bound admits `value.fmt()` | available (DEV-166 fixed) | `Line::value`, `to_string` |
| `Display::fmt` borrows and does not consume | available | `test_value_is_borrowed_not_consumed` |
| by-value `self` receiver, method chaining | available | every `Line` method |
| moving a field out of a by-value `self` | available | `let mut buf = self.buf;` |
| struct field shorthand (`Line { buf }`) | available | every `Line` method |
| `String::new`, `push_str`, `as_str` | available | package tests |
| generic dispatch over primitives AND user nominals | available | `test_every_display_primitive`, `test_user_display_impl` |
| cross-package dependency alias | available | `stark-fmt-consumer` |
| native compilation of a `Display`-generic method | available | consumer native build and run |

## Commands

```bash
# from packages/stark-fmt
../../starkc/target/debug/stark check
# stark-fmt: OK

../../starkc/target/debug/stark test
# test result: ok. 7 passed; 0 failed; 0 ignored

../../starkc/target/debug/stark fmt --check
# (silent, exit 0)

# declared-surface check (CD-355), run through the qualification script's own function
# surface: 12 public callables, all called
```

```bash
# from packages/stark-fmt-consumer
../../starkc/target/debug/stark check
# stark-fmt-consumer: OK

../../starkc/target/debug/stark run
# pkg=stark n=42 r=0.75 ok=true
# v=0.1
# 0.1

../../starkc/target/debug/stark build --no-build-cache
# Built stark-fmt-consumer [debug] -> target/stark/debug/stark-fmt-consumer

./target/stark/debug/stark-fmt-consumer
# pkg=stark n=42 r=0.75 ok=true
# v=0.1
# 0.1
```

Interpreter and native output are byte-identical, which is the claim the qualification gate's
steps 6 and 8 exist to make.

## Test coverage

`src/tests.stark`, 7 cases:

| test | what it pins |
|---|---|
| `test_empty_line_is_empty` | `new().done()` is `""`, not a stray separator |
| `test_text_only` | literal fragments concatenate in order |
| `test_integer_value` | the basic generic dispatch |
| `test_every_display_primitive` | `Int64`, `UInt32`, `Float64`, `Float32`, `Bool`, `Char`, `String`, `&str` through one method — and pins the RENDERING, so a package that formatted every value as `""` would fail |
| `test_user_display_impl` | a user `impl Display` reached through the same generic method |
| `test_value_is_borrowed_not_consumed` | the same value rendered twice and then used a third time |
| `test_to_string_free_function` | the free function over a primitive and a nominal |

The consumer additionally exercises the chained-builder shape end to end and re-uses its
`Display` value after formatting it, so the borrow property is proven natively and not only in
the interpreter.

## Formatter note

The per-method doc comments this package originally carried were relocated by `stark fmt` into the
method BODIES — **DEV-156**, which already records exactly this for a struct field and an impl
method (`COMPILER-STATE.md` CD-365 has the cause: `field_def` never consumes leading comments, and
`delimited_list`'s flat form has no comment awareness). The API documentation therefore lives in the
module doc comment, which survives formatting. This is a workaround, not a preference, and it is the
same one `stark-tls` and the HTTP packages took.
