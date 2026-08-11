# stark-cookie evidence

Baseline SHA: `2cd4a0850646194493d9f3dfe49d9c41a365ec5e`

Baseline branch: `develop`

Worktree clean at baseline: true

Final SHA: `d6b3c79` (the commit that landed the package); this file is amended by the commit that follows it

Every code-derived claim below names the commit it was read from.

## Package tests

- command: `../../starkc/target/debug/stark test` from `packages/stark-cookie`
- count: 34 passed, 0 failed, 0 ignored
- result: PASS
- note: the count is transitive — `stark test` runs `stark-ascii`'s 5 tests as well. This package
  defines 29 `test_*` functions of its own.

## Consumer

- check: PASS
- run: `../../starkc/target/debug/stark run` from `packages/stark-cookie-consumer`
- result: PASS, stdout `COOKIE_CONSUMER_OK`
- surface: all 16 public callables are executed by the consumer or the package tests, not merely
  imported.

## Engine evidence

- interpreter: `stark test` and `stark run` PASS
- native debug: `stark build --no-build-cache` PASS; `./target/stark/debug/stark-cookie-consumer`
  prints `COOKIE_CONSUMER_OK`
- native release: `stark build --release --no-build-cache` PASS;
  `./target/stark/release/stark-cookie-consumer` prints `COOKIE_CONSUMER_OK`
- HIR vs MIR: **not measured separately.** At `2cd4a08`, `starkc/src/bin/stark.rs` exposes no engine
  selector on `stark run` or `stark test`, and no `STARK_ENGINE`-style switch exists, so a package
  cannot pin one interpreter against the other from the package CLI. Claiming distinct HIR and MIR
  results here would be claiming evidence from a run that did not happen. The three configurations
  above are what was measured.

## First-party qualification

- registered: `starkc/scripts/qualify-first-party-packages.py` gains a `PackageCase` for
  `stark-cookie` / `stark-cookie-consumer` with `expected_stdout="COOKIE_CONSUMER_OK\n"`. No
  population count was hand-edited; the case list is the population.
- the gate's three steps were run by hand for this package: `stark test` in the package, `stark run`
  in the consumer with stdout compared, and `stark build --no-build-cache` in the consumer.
- the full script was not run end to end: it requires live local network peers for the `stark-net`
  resource cases, which this session did not stand up.

## CI

- run ID `31454342408` — workflow `CI`, branch `develop`, commit `d6b3c79`
- run ID `31454342423` — workflow `C7.8 Native Capabilities`, branch `develop`, commit `d6b3c79`
- conclusion: not yet available. Both runs were queued/in progress when this was written, and a
  conclusion is not claimed until one exists. The previous run of both workflows on `develop`
  (`31452583578` / `31452583512`) concluded `success`.

## Compiler changes

none

## Provider changes

none

## Host capabilities

none. The package is pure and runs under `stark run`.

## Dependencies

`stark-ascii 0.1.0`, and nothing else. `stark-time` was considered for `Expires` and rejected: at
`2cd4a08`, `packages/stark-time/starkpkg.json` declares `"capabilities": ["clock"]` and
`packages/stark-time/src/lib.stark` exposes `Duration`, `Instant` and `UnixTimestamp` with no date
parser at all. Depending on it would have made a pure package capability-carrying — and therefore
unable to run under `stark run` — in exchange for nothing.

## New deviations

Two compiler defects were found while implementing this package, plus the native gap recorded
under Residuals. All three are now filed on the compiler track as **DEV-222, DEV-223 and DEV-224**
in `starkc/docs/conformance/KNOWN-DEVIATIONS.md`, with a session record in `COMPILER-STATE.md`
taking population A from 8 to 11. None was worked around by changing the compiler.

### DEV-223 (filed 2026-08-11; was COOKIE-DEV-A here) — a variant sharing a name with an in-scope type is reported non-exhaustive

At `2cd4a08`, an exhaustive match over an enum is rejected with `[E0303] non-exhaustive pattern
match` when one variant's name is also the name of a type in scope.

```stark
enum Policy { A, B }

enum Attr {
    Flag,
    Policy(Policy),     // variant name == type name
}

fn render(attr: &Attr) -> String {
    match *attr {       // E0303: non-exhaustive pattern match
        Attr::Flag => String::from("flag"),
        Attr::Policy(p) => match p {
            Policy::A => String::from("a"),
            Policy::B => String::from("b"),
        },
    }
}
```

Renaming the variant and changing nothing else compiles and runs. Stage: name resolution /
exhaustiveness checking. Alternative taken: `CookieAttributeKind::SameSitePolicy` rather than
`SameSite`, since the type `SameSite` is in scope.

### DEV-222 (filed 2026-08-11; was COOKIE-DEV-B here) — a pattern naming a variant that does not exist type-checks and silently never matches

At `2cd4a08`, a pattern naming a variant that does not exist is accepted by the checker and falls
through to the wildcard at runtime instead of being rejected. It is **not** limited to structs: a
misspelled variant on a real enum behaves the same way, which is the far more common shape.

```stark
enum Colour { Red, Green }
// `Colour::Blu` is a typo. `stark check` reports OK; describe(&Colour::Green) prints "wildcard".
```

The struct form found here is the same defect:

```stark
struct Thing { value: Int64 }

fn main() {
    let t = Thing { value: 7i64 };
    let r = &t;
    match *r {
        Thing::Missing(n) => println(n),          // no such variant; `Thing` is a struct
        _other => println("fell through to the wildcard"),
    }
}
```

`stark check` reports `OK` and `stark run` prints `fell through to the wildcard`. Stage: name
resolution / pattern checking. This one has no package-level workaround because it needs no
workaround — it is a missing diagnostic, and its cost is silent wrong behaviour during a
refactor. It was found exactly that way here: after `CookieAttribute` changed from an enum to a
struct, a test still carrying the old `CookieAttribute::MaxAge(seconds)` pattern kept compiling and
began failing at runtime instead of at the type error that should have caught it.

## Residuals

- **The attribute model is a tagged struct, not a sum type with payloads** (filed as **DEV-224**). The natural shape —
  `enum CookieAttribute { Expires(String), MaxAge(Int64), ... }` — cannot be compiled natively at
  this baseline: `stark build` reports `native build does not yet support this program: binding a
  non-Copy scrutinee through a shared reference`, and every reader of an attribute holds one by
  reference out of a `Vec`. A minimal reproducer is an enum with one `String` variant matched
  through `&`; even a `_` pattern fails, because the rejection is about the scrutinee rather than
  the binding. The equivalent tagged struct — a `Copy` kind field plus typed payload fields —
  compiles and runs natively, so that is what v0.1 uses. `CookieAttributeKind` selects which payload
  field is meaningful, and the constructors fix the others so equal attributes always format
  identically.
- **`Expires` is an opaque validated string.** Cookie-date parsing is deliberately out of scope for
  v0.1; see Dependencies above for why `stark-time` is not the answer.
- **`stark fmt` needed two passes to reach a fixed point** on `packages/stark-cookie/src/tests.stark`
  at `2cd4a08`: the first pass reported the file formatted, and `stark fmt --check` still reported it
  non-canonical until a second pass. Both files are canonical as committed.
- **The quoted-value grammar extends RFC 6265.** `cookie-octet` excludes SP, so strict RFC 6265
  would reject `session="hello world"`. This package accepts SP inside DQUOTEs. The extension is
  documented in `README.md` and is the only place the grammar is wider than the RFC.
