# WP-C7.9 — closure record

**Status:** qualification in progress; this record is complete for the implementation and is
finalised by the Tier-1 CI run.
**Parent gate:** C7 — **CLOSED at CD-274** while this work package was being implemented. WP-C7.9
therefore appends to a closed gate: it does not reopen CD-274, and §6 states exactly what CD-274's
claim did and did not cover.
**Decisions:** owner rulings D1–D7 in `WP-C7.9_Claude_Marathon_Execution_Brief.md` §2.
**MIR amendment:** A13 (trap classification + the stderr half of the runtime surface),
`MIR_RUNTIME_SURFACE` `0.1-A10` → `0.1-A13`, `MIR_VERSION` unchanged at `0.3`.
**Corpus:** 1.4.0 → **1.5.0** (nine new cases, no existing source or expectation changed).

---

## 1. What was fixed, by packet

### Packet A — integer trap correctness

`Div`/`Rem` carried one static trap category, so every failure of either operator reported
`DivideByZero` regardless of cause. Two defects followed:

- **`MIN % -1` did not trap at all** in MIR or native. Both evaluate on an `i128` carrier and then
  range-filter; the mathematical remainder is `0`, which is in range. The program **completed with
  a value** where the specification requires a trap, while the HIR oracle trapped. This is the
  most severe divergence class the project defines, and no maintained case covered the pair.
- **`MIN / -1` trapped with the wrong identity** — `DivideByZero` for an operation whose divisor
  is `-1`.

NUM-INT-DIV-001 says both trap "because the intermediate quotient is not representable", which is
an overflow. The checked evaluation now overrides the terminator's default category for that cause,
exactly as a bad shift count already did (MIR amendment A13). Guard order is identical in the MIR
interpreter and the generated-Rust backend: read each operand once, trap `DivideByZero` on a zero
divisor, trap `IntegerOverflow` on the signed `MIN op -1` pair, then evaluate.

**A third defect, found by writing the tests.** `acc /= -1i32` on `Int32::MIN` **completed in the
oracle and stored `2147483648` in an `Int32`**. `eval_binary` range-checks its result against the
type of the expression it is given, and the compound-assignment path passed the *assignment*
expression — whose type is `Unit`, which has no width, so the check passed vacuously. The LHS is
now the type-carrying expression. No maintained case had ever overflowed through a compound
assignment.

### Packet B — trait implementation conformance

A user-declared trait is an HIR item and its implementations were compared against it. A
`CoreTrait` has no declaration item — every `impl Ord for T` writes its own signature — and
**nothing checked it at all**. `fn cmp(&self, other: &Self) -> Bool` type-checked and reached
execution.

One canonical contract table now covers every Core trait a user can implement (the seven
fixed-signature traits, `Iterator`, `From`, `Into`), checked on receiver form, parameter arity and
types, return type, method generics, associated types, and missing / extra / duplicate items.
Duplicate detection was added to the user-trait path too, where set-difference membership checks
could not see the same name twice.

`Index`/`IndexMut`/`TryFrom`/`Error`/`FromIterator` are deliberately unmodelled and say so: their
signatures range over associated types *and* method-level generics, and no user implementation of
them is supported anywhere in the compiler, so a contract for them would be checked against nothing.

### Packet C — HIR place-aware pattern execution

`match_pattern` took a `&Value` and had nowhere to build a reference from, so every binding was a
clone of the referent. For read-only use that is observationally identical — which is why 17 of
PAT-BIND-001's 19 cases agreed across all three engines while the oracle was wrong — and it parts
company the moment a binding is used *as* a reference. CD-267 pinned the divergence and escalated
it rather than patching it.

The binding mode is now decided once per `match`, from that `match`'s own scrutinee
(`PatternSource::Owned | Borrowed`), and a borrowed source projects places rather than cloning
values. Non-`Copy` components matched through a reference bind `Value::Ref` to the original
storage; `Copy` components still bind by value; owned scrutinees are untouched, including their
`drop_unbound` path, which a borrowed match must *not* run because it consumes nothing.

Enum positional payloads gained a projection — nothing named them before, which is why the payload
of a tuple variant could not be borrowed in place at all.

The pinned divergence case is now a passing four-configuration case rather than being deleted.

### Packet D — stderr as a compared channel

`eprint`/`eprintln` are normative operations that **no engine below the front end performed**: the
oracle wrote them to the host process, MIR had no lowering, native emitted nothing, and the
comparator compared empty-to-empty. A case could agree across three engines while two of them never
performed the operation under test.

Now: the oracle captures them; MIR lowers them through the same type-directed dispatch as
`print`/`println`, redirected at the single point where an operation becomes a call; the runtime
writes and flushes them; and the comparator compares the channel on completion **and before a
trap**.

The trap half needed a separation protocol, because a program's stderr and the runtime's trap
diagnostic share one host stream. The differential runner passes a fresh random token per run and
the runtime emits exactly one record carrying it — everything before that record is the program's.
A fixed delimiter would be forgeable by a program that simply printed it. **Production output is
unchanged**: the token is absent for every real invocation.

### Packet E — accepted-but-unlowerable surfaces

Nine surfaces were accepted by the front end and executable only by the reference interpreter. All
are now refused at type checking with `E0105`, and the audit is in
`WP-C7.9-ACCEPTED-SURFACE-AUDIT.md`. Four of the nine were not in the review's list — they were
found by reading the combinator table in `core_method_signature` rather than the test suite.

Two further surfaces were found and deliberately **not** changed, because their refusal point is
governed by CE4/CD-132: Drop-bearing `HashMap`/`HashSet` entries, and `match *r` where the matched
type has a user `Drop` impl. Both are recorded with tests that fail if the boundary moves.

### Packet F — resource exhaustion

Deep recursion aborted the process by signal. The cap alone could not fix it: measured on a default
8 MiB stack, an ordinary recursive function overflowed at roughly **a hundred** STARK frames, a
depth real programs reach. So execution now runs on a thread with a stack sized for the cap, and
`MAX_CALL_DEPTH` reports exhaustion before the host runs out — a classified host/process failure
under `LIMIT-RESOURCE-001`, never a trap, with its own exit status (2, not the trap's 101).

Native execution is recorded as a bounded limitation (**DEV-120**), per ruling D4.

`RuntimeError` gained a four-way `FailureClass` in place of the `is_trap` boolean, which could not
distinguish a compiler rejection from a compiler defect from a property of the machine.

### Packet G — comparator and qualification hardening

- Release is compared by default: `HIR == MIR == native-debug == native-release`.
- No macro arm returns without comparing; a missing toolchain removes an engine, not the comparison.
- **Trap identity is structural.** Every language trap in the oracle states its category where it is
  raised; the prose normaliser is gone, and a guard test fails if `contains("integer overflow")`-
  style classification returns. Writing this found unclassified sites the guard then caught:
  slice bounds, `Vec::remove`, `Vec` insertion, and `assert_eq`/`assert_ne`.
- `ExpectedOutcome` states a case's outcome — completion, trap, or front-end rejection — as data,
  independently of any engine.
- The shared `canonical_float` gained mutation coverage: six deliberately wrong renderers, each
  required to FAIL the same specification table the real one passes.
- The adversarial probes are committed as **nine** maintained modules:
  `adversarial_integer_semantics`, `adversarial_trait_impls`, `adversarial_patterns`,
  `adversarial_stderr`, `adversarial_boundaries`, `adversarial_hash_bounds`,
  `adversarial_accepted_surface_audit`, `comparator_discipline`, `resource_exhaustion`.

### Packet H — provider evidence classes

The MIR interpreter deliberately does not execute provider calls, so a provider-backed capability
**cannot** be three-engine qualified however well it is tested. The three evidence classes are
stated in `WP-C7.9-EVIDENCE-CLASSES.md`, with a table of which claim is permitted for what. The
audit found the package documents largely honest already; what was missing was a stated rule. The
deterministic interpreter-side provider is recorded as deferred with one owner (ruling D5).

### Packet I — DEV-118

`HashMap<K: Hash + Eq, V>` and `HashSet<T: Hash + Eq>` were unenforced. All three engines accepted
the same invalid instantiations, because the storage scans by `Eq` and never consults a hash — so no
differential could see it, and it would have become a live divergence the moment one engine began
hashing. Enforced now at **type instantiation**, through a general mechanism for
implementation-declared bounds, so `HashMap<Float64, Int32>` is ill-typed wherever it is written.

Fixing it exposed an ordering defect: a function's own generics were installed *after* its signature
types were converted, so `fn build<T: Hash + Eq>() -> HashMap<T, Int32>` saw `T` with no declared
bounds and rejected its own return type.

## 2. Defects found by this work package that the reviews did not report

| # | Defect | Found by |
| --- | --- | --- |
| 1 | Compound assignment skipped the integer range check entirely (`acc /= -1` stored an out-of-range value) | Packet A's compound-assignment case |
| 2 | Function generics were not in scope for the function's own signature types | Packet I's generic-bounds case |
| 3 | Interpreter host-stack exhaustion at ~100 frames, well below any language limit | Packet F's below-the-limit case |
| 4 | Unclassified language traps: slice bounds, `Vec::remove`, `Vec` insert, `assert_eq`/`assert_ne` | Packet G's structural-category guard |
| 5 | Four further HIR-only iterator surfaces (`fold`, `reduce`, `any`/`all`, `find`) | Packet E's audit |
| 6 | `match *r` over a user-`Drop` type is accepted but unlowerable | Packet C's drop-log case |
| 7 | `eprint`/`eprintln` accept only `&str`, unlike `println` | Packet D's rendering cases |

Items 6 and 7 are recorded rather than fixed: both would change what the language accepts, which is
outside this work package's authority.

## 3. Evidence

Filled in at qualification close: local Q1–Q8 totals, the four-configuration corpus replay, and the
Tier-1 CI matrix.

## 4. Deviations and deferrals retained

- **DEV-120** — native call-depth exhaustion is a bounded host limitation (ruling D4).
- Provider-backed capabilities remain verifier/ABI/native qualified (ruling D5).
- The nine refused iterator surfaces (ruling D3), with the audit naming what implementing them needs.
- `WP-C7-Usage-Shape-Qualification` remains adjacent and unabsorbed.
- Two CE4/CD-132-governed refusal points, recorded with guard tests.
- `eprint`'s `&str`-only signature, with a test that fails when it widens.

## 5. What this does not claim

Not that every type-correct STARK program behaves identically across engines. The bounded claim
authorised by ruling D7 is stated in §6 once qualification is green.

## 6. Relationship to CD-274 (Gate C7 closure)

Gate C7 closed at CD-274 while this work package was in flight. Its evidence predates every fix
here, so:

- CD-274's ruling **stands as written** and is not amended;
- the defects in §1 were present in the tree C7 closed over, and three of them (`MIN % -1`,
  compound-assignment overflow, HIR borrowed-payload binding) were live cross-engine divergences at
  that moment;
- this work package's claim is therefore about the tree *after* it, and is stated separately rather
  than folded into C7's.
