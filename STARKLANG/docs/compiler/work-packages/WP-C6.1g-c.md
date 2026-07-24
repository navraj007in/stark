# WP-C6.1g-c — General Borrow-Through-Return (Dispatch-Loop Linearisation)

**Track:** A (Claude)
**Status:** CLOSED (CD-112, 2026-07-25)
**Blocks:** Gate C6 uniform borrow-carrier returns. Does NOT block WP-C6.1g-a or structural Copy.

## Closure (CD-112)

An acyclic body is emitted as nested labelled blocks instead of one `loop { match __bb }`
(`emit_bodies::linear_order` = reverse-postorder + back-edge detection; later-RPO labels enclose
earlier ones, so every forward `goto`/branch is `break 'bbTarget` and `Return` is
`break 'stark_ret v`). rustc then sees each block run once, so a borrow returned through a function
and consumed across the `Option::unwrap` blocks no longer collides with the referent's assignment. A
body with a real back-edge keeps the dispatch loop. Pure rendering change — MIR, control flow, and
move/borrow/definite-assignment semantics are identical (three-engine agreement holds).

`refuse_borrow_carrying_nominals`' RETURN refusal is lifted; the slot-backed Move borrow-carrying
LOCAL refusal remains. Beyond the motivating `Option<&P>` return, the same change unblocked
owned-`String` `==`/`<`, stored interior `&str`, and `Vec<String>`-style pushes (all previously
E0502/E0506 at rustc). Evidence: `tests/native_c61f_nominals.rs`
(`c61f_returning_a_borrow_carrying_nominal_builds_and_runs` + inline variant), the six `native_c61f_*`
suites, `--lib` 441, and the 16-suite native/differential regression.

## 1. Why

A borrow returned through a function and then consumed across several generated blocks fails native
build:

```stark
fn wrap(r: &P) -> Option<&P> { Some(r) }
fn main() { let p = P { v: 3 }; assert_eq(wrap(&p).unwrap().get(), 3); }  // E0506 / E0502
```

This is **not** a Copy issue — it fails identically for a Move referent (`P` with a `Drop` field →
E0502). `Option::unwrap`'s panic-branch match extends the returned borrow across enough dispatch-loop
blocks that it collides with the referent's block-0 assignment. WP-C6.1g-a refuses all
borrow-carrying-nominal returns pre-rustc as an interim; this package makes them work.

## 2. Root cause

Every generated body is one `loop { match __bb { … } }`. rustc cannot see that a block runs once, so
a borrow held across blocks is treated as live on the back-edge, conflicting with the referent's
(single) assignment. Referent-storage stabilization (slot vs plain local) does **not** resolve it —
it only moves E0506 to E0502. The fix is at the **generated control-flow** level: emit a shape rustc
can linearise per block (e.g. labelled blocks / straight-line successors instead of `loop + match`),
so a once-assigned referent is seen as once-assigned.

## 3. Scope

- Borrow returned through a function as a nominal (`Option<&T>`, `Result<&T, _>`, a user generic),
  then consumed (`unwrap`, `match`, field/method access) across blocks.
- The move/definite-assignment/borrow properties that the current `loop { match }` obscures must be
  preserved exactly (three-engine agreement; the b3 / CD-095 / OWN-RETURN-001 invariants hold).
- No NLL expansion; no change to accepted-language semantics — this is purely how correct MIR is
  rendered to Rust.

## 4. Exit

- All the WP-C6.1g-a §6.3 "uniform returns" shapes build and run natively with three-engine
  agreement, and the `refuse_borrow_carrying_nominals` return-refusal is lifted.
- Full workspace suite, `fmt --check`, strict `clippy` clean.
