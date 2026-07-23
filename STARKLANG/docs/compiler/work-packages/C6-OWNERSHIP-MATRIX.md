# C6-OWNERSHIP-MATRIX — Track A / WP-C6.1a

**Status:** COMPLETE (C6.1a) — every normative ownership shape classified
**Track:** A (ownership, partial moves, Drop)
**Base:** `main` @ integration base (Gate C5 closed, `db73afe`)
**Method:** current native outcomes were **probed through the real backend**
(`lower → verify → emit_program::emit`) at the integration base, not asserted from memory. A shape
is `SUPPORTED` if it emits, `BACKEND-REFUSED` if the backend returns `Unsupported` (deterministic
pre-rustc refusal), `FRONT-END` if the front end rejects it (a language rule or a "not yet"
front-end limitation), and `LANGUAGE-RULE` where rejection is permanent Core v1 semantics.

The interpreters (HIR, MIR) accept and execute every row below — they are the reference oracle. The
matrix therefore tracks **native parity**: where native already agrees, and where C6.1 must close a
gap.

Non-`Copy` vehicle types used in probing: `struct S { v: Int32 }` (Move, no Drop) and
`struct D { v: Int32 } impl Drop for D { … }` (Move, Drop-bearing).

---

## 1. Headline finding

C5 native ownership is **substantially more complete than the C5 exit report's terse "beyond the
current control-flow shapes" implied.** All common cross-block movement already works. The concrete
open gaps for C6.1 are **narrow and specific**:

| Gap | Shape | Current | C6.1 owner |
|---|---|---|---|
| G1 | Multi-**unit** enum-payload consuming match / partial move (bind or move ≥2 non-`Copy` payload fields of one variant) | BACKEND-REFUSED | **C6.1c** |
| G2 | Non-`Copy` **array by-value iteration** (`for x in arr`) | FRONT-END ("not yet supported") | **C6.1d** |
| G3 | **Multi-level (depth ≥2) partial move/drop** through a projection chain (`o.a.x`) — only one projection level is implemented | BACKEND-REFUSED | **C6.1b** (newly surfaced by this audit; not in the entry-plan sub-package list) |

Everything else in §3 is already `SUPPORTED` natively or is a permanent language rule.

**G3 is a new finding.** It was not called out in WP-C6-ENTRY §2's re-pin (which named multi-unit
enum moves, wider cross-block moves, and non-`Copy` array iteration). It belongs to C6.1b's "general
non-`Copy` movement" claim and is recorded here so C6.1b's acceptance set includes it.

---

## 2. Column legend

`source construct` · `type shape` · `Copy class` · `MIR shape` · `HIR` · `MIR` · `native (current)`
· `native (target)` · `positive test` · `negative test` · `deviation/gap` · `status`

- HIR/MIR are `exec` for every row (reference oracle accepts and runs).
- `positive`/`negative test` name the intended `starkc/tests/native_c6_1_ownership.rs` cases (and
  existing C5 coverage where already proven).
- `status`: `PARITY` (native already agrees), `OPEN-Gn` (C6.1 must close gap n), `LANG-RULE`.

---

## 3. Ownership acceptance matrix

### 3.1 Locals, parameters, returns

| source construct | type shape | Copy class | MIR shape | native (current) | native (target) | positive test | negative test | gap | status |
|---|---|---|---|---|---|---|---|---|---|
| `let b = a;` (move a local) | `S` | Move | `Use(Move(_))` into slot | SUPPORTED (probe 01) | same | `local_move` | use-after-move rejected | — | PARITY |
| move on `return` | `S` | Move | `Return` slot `take` | SUPPORTED (probe 02) | same | `return_move` | — | — | PARITY |
| move a Drop local | `D` | Move+Drop | slot + `DropPlan` | SUPPORTED (probe 11/12) | same | `drop_local`,`drop_move_return` | drop-after-trap absent | — | PARITY |
| parameter move in / return round-trip | `D` | Move+Drop | call arg `Move`, return `take` | SUPPORTED (probe 22) | same | `roundtrip_move` | — | — | PARITY |
| Copy local read (no move) | `Int32`,`fn(..)->..` | Copy | plain read | SUPPORTED | same | (C5.2/C5.4c) | copied-move mutation guard | — | PARITY |

### 3.2 Aggregate fields (struct / tuple / enum)

| source construct | type shape | Copy class | MIR shape | native (current) | native (target) | positive | negative | gap | status |
|---|---|---|---|---|---|---|---|---|---|
| move one struct field, sibling stays live | `P{a:S,b:S}` | Move | `Field(i)` move, per-unit drop | SUPPORTED (probe 06) | same | `struct_partial_move` | reversed field-drop order | — | PARITY |
| **multi-level** field move (`o.a.x`) | `Outer{a:Inner{x:S},b:S}` | Move | projection depth 2 | **BACKEND-REFUSED** (probe 20) | SUPPORTED | `nested_partial_move` | depth-2 collapse → duplicate-drop | **G3** | OPEN-G3 |
| tuple field move | `(S, Int32)` | Move | `Field(i)` | SUPPORTED (fn-value tuple, C5.4c) | same | `tuple_field_move` | — | — | PARITY |
| single-unit enum payload consuming match | `E{V(D)}` | Move+Drop | `VariantField(0,0)` whole | SUPPORTED (probe 23) | same | `singleunit_enum_move` | — | — | PARITY |
| **multi-unit** enum payload destructure/partial move | `E{V(S,S)}` | Move | `VariantField(0,k)` ≥2 | **BACKEND-REFUSED** (probe 10/24) | SUPPORTED | `enum_multiunit_partial`,`enum_multiunit_bind_both` | collapse-to-whole → duplicate-drop; unbound-sibling & both-bound refusals | **G1** | OPEN-G1 |
| function-value aggregate (struct field of `fn`) | `H{f:fn(..)->..}` | field Copy | `Field` read | SUPPORTED (probe 18) | same | `fnvalue_aggregate` | fn-ptr treated non-Copy | — | PARITY |

### 3.3 Array elements

| source construct | type shape | Copy class | MIR shape | native (current) | native (target) | positive | negative | gap | status |
|---|---|---|---|---|---|---|---|---|---|
| move `arr[i]` (dynamic index) of non-`Copy` | `[S; N]` | Move | dynamic `Index` | **FRONT-END** "cannot move out of an indexed place" (probe 07) | unchanged | — | move-out-of-index rejected | — | LANG-RULE |
| `arr[i]` of `Copy` element (read) | `[Int32; N]` | Copy | `Index` | SUPPORTED | same | (C5.3) | OOB traps | — | PARITY |
| non-`Copy` by-value iteration (`for x in arr`) | `[S; N]` | Move | `ConstIndex` per element | **FRONT-END** "not yet supported" (probe 08) | SUPPORTED | `array_by_value_iter` | unconsumed-element cleanup after break/return; no whole-array drop after full consume; no cleanup after trap | **G2** | OPEN-G2 |
| `Copy` element array iteration | `[Int32; N]` | Copy | element copy | SUPPORTED | same | (C5.3) | — | — | PARITY |

### 3.4 Container payloads (Option/Result; Vec/Box)

| source construct | type shape | Copy class | native (current) | native (target) | positive | negative | gap | status |
|---|---|---|---|---|---|---|---|---|
| `Option`/`Result` payload move (single field) | `Option<S>`,`Result<S,_>` | Move | SUPPORTED (probe 09/17) | same | `option_payload_move`,`question_mark_transfer` | — | — | PARITY |
| wildcard discards a Drop payload (`Some(_)`) | `Option<D>` | Move+Drop | SUPPORTED (probe 19) | same | `wildcard_payload` | omitted drop of discarded | — | PARITY |
| **Vec element** move / ownership | `Vec<S>` | Move | (Vec is not in the C5 native subset) | SUPPORTED | (C6.3 Track C runtime) | Vec non-Copy element cases | — | deferred (C6.3 ⋂ C6.1 API) | OPEN (C6.3) |
| **Box inner value** move / exact inner drop | `Box<D>` | Move | (Box not in C5 native subset) | SUPPORTED | (C6.3 Track C runtime) | Box recursive/double-drop | — | deferred (C6.3 ⋂ C6.1 API) | OPEN (C6.3) |

*Vec/Box ownership is a Track A ↔ Track C interface: Track A freezes the storage/liveness API, Track
C consumes it for non-`Copy` Vec elements, iterator remaining-element Drop, and Box recursive Drop
(WP-C6-ENTRY §7F dependency).*

### 3.5 Pattern bindings, control transfer, control-flow joins

| source construct | type shape | native (current) | native (target) | positive | negative | gap | status |
|---|---|---|---|---|---|---|---|
| pattern binding move (`match … => x`) | `S`/`D` | SUPPORTED (probe 09/23) | same | (covered above) | — | — | PARITY |
| wildcard `_` (no binding) | any | SUPPORTED (probe 19) | same | `wildcard_payload` | — | — | PARITY |
| define block A, move block B | `S`/`D` | SUPPORTED (probe 03/13) | same | `cross_block_move` | — | — | PARITY |
| conditional definition then join (`let a; if…{a=…}else{a=…}`) | `S` | SUPPORTED (probe 14) | same | `conditional_def_join` | use-before-def rejected | — | PARITY |
| move in one branch only | `D` | SUPPORTED via drop flag (probe 13/21) | same | `move_one_branch`,`loop_break_move` | move-in-branch + use-after rejected (probe 15) | — | PARITY |
| loop-carried value + reinit | `S` | SUPPORTED (probe 05/16) | same | `loop_carried`,`move_reinit_use` | — | — | PARITY |
| `break`/`continue` with live value | `D` | SUPPORTED (probe 21) | same | `loop_break_move` | — | — | PARITY |
| `return` / `?` transfer of non-`Copy` | `S`/`D` | SUPPORTED (probe 02/17) | same | `return_move`,`question_mark_transfer` | — | — | PARITY |
| call-argument and return movement | `D` | SUPPORTED (probe 22) | same | `roundtrip_move` | — | — | PARITY |
| recursive legal control flow | `S` | SUPPORTED (C5.4b recursion) | same | (three-engine recursion) | — | — | PARITY |

### 3.6 Destructor-bearing generic nominals

| source construct | type shape | native (current) | native (target) | positive | negative | gap | status |
|---|---|---|---|---|---|---|
| generic nominal with own `Drop`, moved/returned | `D`-like generic `G<T>` | SUPPORTED for scalar `T` (probe 11/12 pattern; C5.3d) | same across admitted `T` | `generic_drop_move` | wrong drop-instance dispatch | — | PARITY (scalar) |
| generic nominal Drop over a **non-`Copy` field `T`** | `G<S>` | ties to G1/G3 when the field participates in a multi-unit/deep move | SUPPORTED | `generic_drop_nonCopy_field` | duplicate/missing field drop | G1/G3 | OPEN (via G1/G3) |

---

## 4. Drop-path summary (feeds C6.1e)

| exit path | current | target |
|---|---|---|
| normal scope/function/loop/match-arm/return/`?`/main | SUPPORTED (C5.3d, probes above) | PARITY |
| aborting trap (overflow/div0/index/cast/panic/explicit) | **no cleanup** (abort, no unwind) — SUPPORTED | PARITY |
| failed pattern binding | SUPPORTED (front-end/MIR) | PARITY |
| IO/provider failure | n/a in C5 (C6.3 Track C) | C6.3 |

C6.1e observes marker **order and counts**, not just exit code.

---

## 5. C6.1a closure

- [x] Locals/params/returns classified (§3.1)
- [x] Tuple/struct/enum fields classified (§3.2)
- [x] Constant-index and dynamic-index array elements classified (§3.3)
- [x] Vec elements / Box inner values classified (§3.4, deferred to C6.3 interface)
- [x] Option/Result payloads classified (§3.4)
- [x] Pattern bindings and wildcards classified (§3.5)
- [x] Branch joins and loop-carried values classified (§3.5)
- [x] break/continue/return/`?` classified (§3.5)
- [x] Function-value aggregates classified (§3.2)
- [x] Destructor-bearing generic nominals classified (§3.6)
- [x] Every normative ownership shape has a current + target outcome and a named test
- [x] Current outcomes are probe-grounded, not assumed

**Open gaps for C6.1 implementation:** **G1** (multi-unit enum payload → C6.1c), **G2** (non-`Copy`
array by-value iteration → C6.1d), **G3** (multi-level partial move → C6.1b, newly surfaced). Vec/Box
ownership is a Track A↔C interface (C6.3). Everything else is at native parity today.

Next Track-A increment: **C6.1b** — general cross-block non-`Copy` movement, whose acceptance set now
explicitly includes **G3** (multi-level projected moves), then **C6.1c** (G1) and **C6.1d** (G2).
