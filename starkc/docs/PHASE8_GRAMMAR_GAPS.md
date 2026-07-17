# Phase 8 — Gaps and Limitations Found While Building Tooling

Running log of everything found missing, incomplete, or worked-around while
building Phase 8 tooling (`STARKLANG/docs/PHASE8_PRODUCTION_TOOLING_PLAN.md`)
— language grammar gaps, missing stdlib primitives, and known-incomplete
tooling deliverables alike. Per `CLAUDE.md`, new language features land in
the spec first; none of the language-level gaps below were added to the
grammar as part of Phase 8 — each was worked around in tooling instead (or,
for tooling-only gaps, left as a disclosed limitation) and is listed here so
a future spec/implementation pass can pick it up deliberately.

---

## WP8.1 — LSP Server Foundation

### Hover/go-to-definition/references are protocol stubs, not real features

**What shipped:** the LSP endpoints exist and respond correctly per the
protocol (`textDocument/hover`, `textDocument/definition`,
`textDocument/references` all wired into `handle_request`), and compiled
`TypeTables` are cached per open document — but the handlers themselves
don't use that data yet: hover returns a raw `line:character` position
string instead of the inferred type/signature at that position;
definition/references return `null`/`[]` unconditionally.

**Why deferred:** mapping an LSP cursor position back to the specific
`ExprId`/`ItemId` at that byte offset (and, for hover, rendering its `Ty`
as source-like text) is real work in its own right — not just wiring.
Scoped out of WP8.1's "foundation" pass; flagged in that WP's own
completion doc (`WP8_1_LSP_IMPLEMENTATION.md`) as unbuilt.

**If picked up later:** needs (a) a span→node lookup (walk the HIR/AST for
the innermost node whose span contains the cursor's byte offset — the AST
doesn't currently expose a position index, so this is currently O(n) per
request; fine for now, worth revisiting at scale), (b) a `Ty`-to-source-text
renderer (the formatter's `printer.rs` already has type-printing logic that
could be reused/factored out for this).

---

## WP8.2 — Formatter

### Comments were being silently discarded by the lexer (fixed)

**What was missing:** `lexer.rs`'s `line_comment`/`block_comment` consumed
comment bytes and threw them away — no token, no side table, nothing. The
AST (`ast.rs`) has no comment field on any node. An AST-only formatter
("walk the parsed AST, print it back") would therefore have deleted every
comment in any file it touched. Confirmed materially real: 122 of 202
`.stark` files in this repo use comments.

**Fixed in WP8.2** (user-approved, see conversation): added
`lexer::tokenize_with_comments`, which collects comments as a separate
trivia list (`Comment`/`CommentKind`) alongside the unchanged token stream.
The formatter re-attaches them by source position while printing.

**Still a gap:** comments are *tooling* trivia (formatter-only), not part
of the AST or HIR. Nothing downstream of parsing (resolve, typecheck, the
interpreter) can see them, and there's still no doc-comment (`///`)
*semantics* — `CommentKind::LineDoc`/`LineInnerDoc` are lexically
distinguished but not attached to the item they document anywhere an
API-doc generator could consume. **WP8.5 (Documentation Generator) will
hit this directly** — `///` doc comments need to become first-class
(attached to the `ItemId` they precede, surviving into a form the doc
generator can read) rather than reusing the formatter's position-based
trivia re-attachment, which is a "make text look right when printed," not
a "let other tools query which comment belongs to which item" mechanism.

### No attribute syntax — affects doc-comment "sections" too

Related to the WP8.3 entry below: the plan's doc-comment example
(`06-Standard-Library.md`-adjacent, `PHASE8_PRODUCTION_TOOLING_PLAN.md`
WP8.5) shows Markdown-style `# Arguments`/`# Returns`/`# Example` sections
inside `///` comments — that's just Markdown text inside a comment (no
grammar gap), but extracting *structured* per-parameter docs from it (vs.
just rendering the Markdown blob) is a doc-generator parsing concern, not
solved by anything built so far.

---

## WP8.3 — Test Framework

### Attribute syntax (`#[test]`, `#[ignore]`, ...) doesn't exist

**Plan assumed:**

```stark
#[test]
fn test_addition() {
    assert_eq(2 + 2, 4);
}

#[test]
#[ignore]
fn test_slow_operation() { }
```

**Reality:** Core v1 has no attribute syntax at all — no `#[...]` handling
in the lexer (`#` falls through to "unexpected character"), no attribute
node in the AST (`ast::ItemNode` has no attribute field), nothing in
`01-Lexical-Grammar.md` or `02-Syntax-Grammar.md`.

**Workaround used (user-approved):** naming convention instead of
attributes for test discovery — see `WP8_3_TEST_FRAMEWORK_IMPLEMENTATION.md`
for the exact convention chosen. The plan's own risk table already lists
"clear naming convention (`test_*`)" as an acceptable fallback.

**If added to the grammar later:** would need `01-Lexical-Grammar.md`,
`02-Syntax-Grammar.md` (an `Attribute ::= '#' '[' Path AttributeArgs? ']'`
production attachable to items, likely repeatable), `ast::ItemNode` (an
`attrs: Vec<Attribute>` field), parser support, and probably
`04-Semantic-Analysis.md` (which attributes are recognized/where they're
valid). A real, scoped language feature — not a tooling-only change.

### `assert_eq`/`assert_ne` didn't exist in the stdlib (added); `assert_matches` couldn't be

**What existed before WP8.3:** only `assert(cond: Bool) -> Unit` (traps
with "assertion failed" on `false`) — no value-comparing or
pattern-matching assertion builtins, and nothing in
`06-Standard-Library.md` describing any of the three the plan assumes
(`assert_eq`, `assert_ne`, `assert_matches`).

**`assert_eq`/`assert_ne`: added as new interpreter builtins** (`hir.rs`
`Builtin::AssertEq`/`AssertNe`, `resolve.rs` name mapping, `typecheck.rs`
generic `fn assert_eq<T>(a: T, b: T)`/`assert_ne<T>(a: T, b: T)` via a
shared fresh type variable — same pattern `swap`/`replace`/`take` already
use — and `interp.rs` runtime comparison + a `left`/`right`-showing panic
message). Not a grammar change, just new stdlib surface, so no spec entry
needed beyond `06-Standard-Library.md` eventually documenting them
properly.

**Notable finding made while adding them:** `==`/`!=` (`BinOp::Eq`/`Ne` in
`interp.rs`) are implemented as *pure structural equality on the
interpreter's internal `Value` enum* (`left == right` via Rust's derived
`PartialEq`) — there is no dispatch through a user's `Eq` trait
implementation at runtime. So "Eq" as a trait bound is currently a
type-checker-only concept (checked as a bound, not honored as a dispatched
method at runtime); the interpreter's structural equality happens to match
what a correct `derive(Eq)` would do, but a user-written *custom*
`impl Eq for T` (if that's even expressible — worth checking whether Eq is
a bound users can implement by hand vs. only auto-derived) would silently
be ignored at runtime. `assert_eq`/`assert_ne` inherit this — they compare
`Value`s structurally, same as `==`/`!=` do, so they're consistent with
existing language behavior rather than a new gap, but it's worth a closer
look outside Phase 8's scope since it affects more than test tooling.

**`assert_matches(value, pattern)`: dropped, not implementable as a plain
function.** Patterns are AST/HIR constructs (`PatKind`), not runtime
values — there's no way to pass a "pattern" as a first-class function
argument the way the plan's signature implies (`assert_matches(value,
pattern)` reads like a two-expression call, but pattern-matching needs
actual pattern syntax). This isn't a workaround-able tooling gap the way
`#[test]` naming was — it needs either macros or a dedicated statement
form, neither of which exist. Recommended STARK idiom in the meantime:
`match value { pattern => {}, _ => panic("assertion failed: value did not match pattern") }`.

### `--seed N` (plan's `stark test -- --seed 42`) — omitted, nothing to seed

Core v1 has no randomness primitives anywhere in the grammar or stdlib
(`06-Standard-Library.md` has no `rand`/`Random` entry). The plan's `--seed`
flag exists for deterministic *test* randomness (e.g. property-based
testing, fuzzing inputs) — with no RNG to seed, the flag would be a no-op,
so it was left out of `stark test` entirely rather than shipped as
dead CLI surface. If/when a randomness primitive lands, revisit.

---

## WP8.4 — VS Code Extension

### The LSP server ignored requested language extensions entirely (found and fixed)

Wiring the VS Code extension's tensor-mode toggle through to `starkc lsp`
surfaced that the server hardcoded `LanguageOptions::default()` (Core-only)
in both `compile_document` and `handle_formatting` from WP8.1 — a client
that asked for the `tensor` extension via LSP `initialize` would still get
"requires extension `tensor`" errors on every tensor file. Fixed in a
dedicated commit (`4eda834`, before this WP's own commit): `ServerState`
now carries the session's `LanguageOptions`, read once from `initialize`'s
`initializationOptions.extensions` and reused for every subsequent parse.
See `WP8_4_VSCODE_EXTENSION_IMPLEMENTATION.md` for the verification.

### `stark.generateDocs` not built — WP8.5 doesn't exist yet

The plan lists a `stark.generateDocs` command under WP8.4, but it's a
thin wrapper around `starkc doc` (WP8.5, Documentation Generator), which
hasn't been implemented. Adding a command that fails with "unknown
subcommand" on every invocation would be worse than the command not
existing; deferred to whenever WP8.5 ships.

### VS Code UI behavior not interactively verified

No `code` CLI is available in this environment to launch an Extension
Development Host, so status bar rendering, command palette entries,
format-on-save actually firing on a real save, and hover popups were not
interactively confirmed — only TypeScript correctness (tsc, ESLint,
esbuild bundling) and the raw LSP protocol exchange the client depends on
(verified via a JSON-RPC script bypassing VS Code, the same technique
used for WP8.1/WP8.2). Flagged rather than silently claimed as tested;
real interactive testing is the natural next step once a VS Code-capable
environment is available.

---

*(Add further entries above this line as Phase 8 continues — WP8.5+.)*
