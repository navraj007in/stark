# WP-C6.1f — Closure Packet

**Package:** General Reference Storage, Reborrowing, and Provenance (Track A)
**Status:** CLOSED 2026-07-24 (CD-099)
**Base at open:** `main` @ CD-088 · **Closure head:** CD-099
**Owner disposition of remaining limitations:** CD-097

C6.1f was opened (CD-088, F3 ruling) because C5 deferred "general references and full
borrow/lifetime representation" to C6 and the entry plan distributed the *consumers* across C6.2b
and C6.3b without assigning the shared prerequisite. This packet records that the prerequisite is
now met.

---

## 1. The ten scope items (`WP-C6.1f.md` §2)

| # | Item | Result | Evidence |
|---|---|---|---|
| 1 | References in user locals | ✅ native | `native_c61f_b3_stored_refs.rs` (CD-093) |
| 2 | Reference flow across basic blocks | ✅ native | `native_c61f_b3_stored_refs.rs` (CD-093) |
| 3 | Shared and mutable reference parameters | ✅ native | already at C6.1f-a; `native_c61f_reborrow.rs` |
| 4 | Nested references and repeated dereference | ✅ native | representation `native_c61f_reborrow.rs` (CD-090); syntax `c61f_nested_refs.rs` (CD-097a). Selection is Track B's per the F4 ruling |
| 5 | Reborrowing and mutable exclusivity | ✅ native | receiver `native_c61f_reborrow.rs` (CD-090); argument `native_c61f_b2_weakening.rs` (CD-092/098) |
| 6 | Reference returns and provenance | ✅ native, provenance front-end-enforced | `native_c61f_ret_refs.rs` (CD-094). Lifetime **precision** deferred → **CD-097 item 2 / WP-C6.1g-b** |
| 7 | Owner move/drop while borrowed | ✅ correctly rejected | `c61f_reference_boundary.rs` |
| 8 | Array/struct/`Box`/`Vec`/slice-derived refs | array/struct ✅; `Box`/`Vec`/slice **scoped out** | tuples/arrays `native_c61f_aggregates.rs` (CD-095); `Box`/`Vec`/slice → **CD-097 item 3 / C6.3** |
| 9 | HIR/MIR/native agreement + negative verifier tests | ✅ | three-engine suites + `c61f_reference_boundary.rs` |
| 10 | No NLL expansion | ✅ pinned | `c61f_reference_boundary.rs` (no-NLL case) |

Borrow-carrying **nominals** (`Option<&T>`, generics at a reference) land for most shapes (CD-096);
two slot-backed shapes are refused pre-rustc under **CD-097 item 1 / WP-C6.1g-a**.

## 2. Exit criteria (`WP-C6.1f.md` §6)

- ✅ Ten items classified; each implemented with native evidence or owner-dispositioned (CD-097).
- ✅ `let r = &p; r.get()` and nested-reference receivers build and run natively — unblocking
  C6.2b's remaining rows.
- ✅ C6.3b prerequisites explicitly scoped out with owner approval (CD-097 item 3).
- ✅ E0101 and the lexical-borrow rejections still fire, pinned by `c61f_reference_boundary.rs`.
- ✅ Three-engine agreement; **full workspace suite exit 0, 68 suites, zero failures**; `fmt --check`
  and strict `clippy` clean.

## 3. Deviations carried out of the package (all owner-dispositioned, CD-097)

| # | Deviation | Disposition |
|---|---|---|
| 1 | Slot-backed borrow-carrying nominal, and returning one — refused pre-rustc | **WP-C6.1g-a** (Track A); lifetime threading first, no `ValueSlot`/CE4 change without a probe. **Blocks Gate C6.** |
| 2 | Returned-reference lifetimes conservative (shared `'a`) | **WP-C6.1g-b** (Track A). Sound; **blocks Gate C6** native-conformance. |
| 3 | `Box`/`Vec`/slice native representability | scoped out to **C6.3** (Track C). **Blocks Gate C6.** |
| 4 | `Box` dereference rejected | **Not a deviation** — Core v1 defines no `Deref`. Removed from the list. |

## 4. Method note

Every capability in this package was **probed by native execution before design**, per the C6.1b
correction reconfirmed at C6.2a. That discipline repeatedly overturned assumptions held before
measuring:
- the general-reference blocker was rustc **definite assignment** (E0381), not the borrow checker
  (b3);
- tuples of references were blocked by **`default_value_expr`**, not by references-in-aggregates
  (CD-095) — the property was *carries a borrow*, not *is a reference*;
- the `ValueSlot`-versus-borrow-region tension the C6.1f-a matrix flagged as the central design
  question turned out to be real **only** for slot-backed borrow-carrying nominals (CD-096), not
  for plain references or tuples.

## 5. Commit trail

CD-089 (matrix) · CD-090 (b1 reborrow) · CD-092 (b2 weakening) · CD-093 (b3 stored refs) ·
CD-094 (returns) · CD-095 (aggregates) · CD-096 (nominals) · CD-097a (b4/b5) · CD-097 (dispositions) ·
CD-098 (generic-callee weakening) · CD-099 (this closure).

**C6.1f closure does not move Gate C6.** WP-C6.1g-a/-b and C6.3 remain explicit Gate-C6 dependencies.
