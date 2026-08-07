# AS0 item 6 — callable execution-site inventory

**Packet:** AS0 (Sprint 1, PARTIAL) item 6, executed as **AS3's opening checkpoint** per the
2026-08-07 scheduling decision.
**Branch:** `wp-arch-stability/sprint-2`. **Date:** 2026-08-07.
**Status:** COMPLETE. AS0 item 6 closes; items 7 and 10 remain (AS4 and AS8/C10 respectively).

This is an inventory, not a change. Nothing in the compiler is modified by it.

---

## 1. Why this is AS3's opening item, and what it is for

`WP-VALUE-REP-TOTAL`'s A3c/A4 stalled on an assumption: that the callable surface was known. AS3's
exit criterion 1 — *"every executable user-callable use has exactly one record; duplicates and
omissions fail an invariant test"* — cannot be written against a surface nobody has enumerated, and
work item 4 names ten dispatch families whose completeness was asserted rather than measured.

So the question this inventory answers is not "how many call sites are there". It is:

> **For each way a user-authored body can be entered, does the engine consume the selection the type
> checker already made, or does it make its own?**

---

## 2. Headline finding — there are three selection algorithms, not one

| Algorithm | Where | Call sites |
| --- | --- | ---: |
| The checker's, published as `callable_instantiations` + `bound_trait_calls` | `typecheck.rs` | — (the authority) |
| `Interpreter::find_method` — scan the nominal's impls by name | `interp.rs:4196` | **6** |
| `FnLowerer::find_impl_fn` — scan the nominal's impls by name | `mir/lower.rs:6288` | **8** |

The checker selects. Then **both engines select again**, independently, by walking `hir.items`
looking for an impl on the nominal declaring a method with the right name.

`bound_trait_calls` is not a counter-example — it is the evidence. It is a *narrowing hint* threaded
into both re-selection algorithms so they reach the checker's answer more often. `find_impl_fn`'s own
doc comment records why it exists:

> DEV-BOUND-TRAIT-IDENTITY: … Without the filter this answered "the first impl on this nominal
> declaring a method with that name", so a type implementing two same-named traits ran the same body
> for both bounds — **while the type checker had correctly distinguished them.**

That is a compensating mechanism on a duplicated authority, which is the pattern this programme
exists to remove. And the hint is not applied consistently: of `find_impl_fn`'s 8 call sites, **1**
passes a `trait_filter`; the other 7 pass `None`.

---

## 3. The exact set — how a user body can be entered

### 3.1 Interpreter (HIR oracle)

Two functions actually run a user body: `call_callable` (no receiver) and `call_user_method`
(receiver). Every entry goes through one of them — **21 production call sites** (three further sites
are inside `#[cfg(test)]` and are excluded), enumerated:

| Family | Site(s) | Installs the checker-selected env? | Re-selects? |
| --- | --- | :---: | :---: |
| Test-function invocation (`run_item`, used by `test_runner`) | `1347` | no | no |
| Program entry (`main`) | `1663` | n/a — no generics | no |
| Free function call | `3327` | **yes** (`push_callable_env`, `3326`) | no |
| Associated function | `3359` | no | no |
| Qualified trait call — argument path | `3382` | no | no |
| Qualified core-trait call — argument path | `3406` | no | no |
| Qualified trait call — dispatch (`call_qualified_trait`) | `4154` | no | no |
| Qualified core-trait call — dispatch (`call_qualified_core_trait`) | `4190` | no | no |
| Method call | `4118` | **yes** (`push_callable_env`, `4117`) | **yes** (`find_method`, `4082`) |
| Function value / fn-pointer call | `6779` | env captured at creation (`capture_function_value`, `2828`) | no |
| Operator `==` → user `Eq::eq` | `2972` | no | **yes** (`2944`) |
| Operator `<`/`>`/… → user `Ord::cmp` | `3009` | no | **yes** (`2994`) |
| Structural equality (`language_equal`) | `4646` | no | **yes** (`4639`) |
| Iteration → user `Iterator::next` | `6503` | no | **yes** (via `next_for_iterator`) |
| `Display::fmt` (shallow) | `7095` | no | **yes** (`7090`) |
| `Display::fmt` (deep) | `7154` | no | **yes** (`7146`) |
| `Option`/`Result` combinators calling a user function value | `4846`, `4852`, `4857`, `4865`, `4870` | env from the captured value | no |

**Two of the seventeen families install the checker-selected generic environment.**
`push_callable_env` — the function whose doc says *"the interpreter CONSUMES this environment and
never reconstructs one"* — has exactly **two** call sites: `3326` and `4117`.

### 3.2 MIR lowering

`find_impl_fn`'s 8 call sites, and what each is:

| Site | Family | `trait_filter` |
| --- | --- | --- |
| `5796` | method call through a nominal | `None` |
| `6174` | `eq` | `None` |
| `6224` | `cmp` | `None` |
| `6607` | bound-trait method call | **`bound_trait`** |
| `8037` | `next` (iteration) | `None` |
| `9448` | `fmt` | `None` |
| `10072` | `fmt` | `None` |
| `10158` | `fmt` | `None` |

`callable_instantiations` is read at 3 sites (`5949`, `6108`, `6618`). `Callee::Instance` — a call to
a user body — is constructed at 12.

---

## 4. What this means for AS3

### 4.1 The ten families in AS3 work item 4, measured

| Family | Checker selects | Interpreter consumes | MIR consumes |
| --- | :---: | :---: | :---: |
| free calls | yes | **yes** | yes |
| methods | yes | partial — env yes, identity re-selected | re-selects |
| associated functions | yes | no | — |
| function values | yes | **yes** (at capture) | yes |
| trait defaults | yes | no | re-selects |
| qualified calls | yes | no | re-selects |
| equality | yes | no | re-selects |
| ordering | yes | no | re-selects |
| iteration | yes | no | re-selects |
| display | yes | no | re-selects |

Three of ten are clean. The interpreter and MIR agree today — the differential suites say so — but
they agree because two independently-written scans over the same `hir.items` happen to reach the same
answer, not because either is told.

### 4.2 The scope correction AS3 should carry

AS3's work item 3 says *"Make HIR execution and MIR lowering consume `CallableUse`; neither may
reconstruct selection."* This inventory shows what that costs: **14 re-selection call sites across
two algorithms** (6 in the interpreter, 8 in MIR), against **21** production entry points, not a
handful. The packet's estimate is not obviously wrong, but it was never
grounded, and A3c/A4 stalled once already on exactly that.

It also names the natural checkpoint order, since the families are not equally hard. Free calls and
function values already consume selection — they are the *proof the mechanism works*, not work.
Methods are half-done. Equality, ordering, iteration and display are the four that re-select with
**no filter at all** in both engines, and they are where a disagreement between checker and engine
would be silent rather than caught.

### 4.3 One thing this inventory does not establish

It does not show a live wrong-body defect. DEV-BOUND-TRAIT-IDENTITY was one, and it was repaired by
adding the filter. Whether another is reachable today — a nominal with two same-named trait methods
where the un-filtered sites (`eq`, `cmp`, `next`, `fmt`) pick differently from the checker — is a
question for AS3's first checkpoint, not for an inventory. **The absence of a reproduction here is
not evidence of absence**; it is evidence that nobody has looked, which is the same footing DEV-122
was on before AS1b.

---

## 5. Method

Every number above is a `grep` over the current tree, not a reading. The commands are recorded so a
reviewer can re-derive rather than trust:

```bash
# every site that enters a user body, EXCLUDING the #[cfg(test)] module (which begins at
# interp.rs:8133 and holds three further sites). A plain grep over the whole file reports 24.
python3 - <<'EOF'
import pathlib
t = pathlib.Path("starkc/src/interp.rs").read_text()
prod = t[:t.find("\n#[cfg(test)]")]
print(sum(1 for l in prod.split("\n")
          if ('.call_callable(' in l or '.call_user_method(' in l) and 'fn call_' not in l))
EOF
# -> 21

grep -c 'self\.find_method(' starkc/src/interp.rs          # 6   interpreter re-selection
grep -c 'find_impl_fn(' starkc/src/mir/lower.rs            # 9   = 8 call sites + the definition
grep -c 'push_callable_env(' starkc/src/interp.rs          # 3   = 2 call sites + the definition
grep -c 'Callee::Instance' starkc/src/mir/lower.rs         # 12

# where the checker's selection is actually consumed
grep -rn 'callable_instantiations\|bound_trait_calls' starkc/src --include='*.rs' | grep -v typecheck.rs
```

**A correction made during this inventory, recorded because it is the point.** The first draft
counted 19 entry sites from a `self.`-anchored grep and built its family table on that. Re-deriving
the count mechanically found **21**: the grep missed `run_item`'s `interpreter.call_callable`
(test-function invocation, a whole family) and the two qualified-call *dispatch* sites at `4154` and
`4190`, because I had recorded only their argument-evaluation halves in `eval_call`. An inventory
whose purpose is exactness got its own count wrong on the first pass, in the same way A3c/A4 did —
by pattern-matching a surface instead of enumerating it.

---

## 6. AS0 status after this item

| Item | State |
| --- | --- |
| 6 — callable execution-site inventory | **DONE** (this document) |
| 7 — `WP-C7.8-RB0` predicate inventory | outstanding — AS4's opening inventory |
| 10 — `WP-ENGINE-INDEPENDENCE.md` AS0 scope | outstanding — deferred to AS8/C10 by decision |

AS0 exits when 7 and 10 are done or explicitly deferred. Item 10 **is** now explicitly deferred by
owner decision, so AS0's remaining blocker is item 7 alone.
