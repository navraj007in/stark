# STARK Core v1 Abstract Machine

## Status and authority

This document is normative for Core v1. It is the sole authority for runtime evaluation,
values and places, ownership transfer, temporary lifetime, destruction, references, and
language traps. `03-Type-System.md` defines static legality, `04-Semantic-Analysis.md` defines
required analyses and rejection categories, and `05-Memory-Model.md` summarizes safety
guarantees. None of those documents defines a competing execution model.

The abstract machine does not prescribe an interpreter frame layout, stack or heap placement,
pointer representation, object layout, garbage collector, MIR shape, backend ABI, or optimizer.
A conforming implementation may represent the concepts below in any way that preserves all
specified behavior.

## Terms and machine state

### Values, objects, storage, and ownership

**AM-VALUE-001.** A *value* is a typed Core datum: a scalar, function reference, reference,
aggregate, enum value, or value of an implementation-provided Core library type.

**AM-OBJECT-001.** An *object* is a value together with its identity and lifetime. Object
identity is abstract. It is not a numeric address and does not change merely because ownership
of the object moves between places.

**AM-STORAGE-001.** A *storage location* is an abstract slot capable of holding one object or
being uninitialized. An *allocation* is one or more storage locations whose physical placement
is not observable unless another Core rule explicitly exposes it.

**AM-OWNER-001.** Every live non-`Copy` object has exactly one owner. An owner may be a local,
temporary, aggregate field or element, active enum payload, function parameter, return-transfer
slot, iterator, collection, or another library value whose contract owns the object.

**AM-LOCAL-001.** A *local binding* names a storage location within a lexical scope. A parameter
and method receiver are locals whose initialization occurs on function entry. A local is
*initialized* when its location contains a live object and *moved-from* when ownership has been
transferred out without replacement.

**AM-TEMP-001.** A *temporary* is an unnamed owner created while evaluating a full expression.
Temporaries have the same exactly-once destruction obligations as named locals.

### Places and projections

**EXEC-PLACE-001.** A *place* designates a storage location or a sublocation of an object. Core
places are:

- an initialized local, parameter, or receiver;
- a field or tuple-field projection from a place;
- an array, slice, vector, map-entry, or other indexing projection whose library contract
  defines a place;
- a range projection denoting a slice place;
- dereference of a valid reference.

A projection is part of the place identity; resolving a place does not read or move its stored
value. Field and tuple projections resolve their base first. Index projections resolve the base
place, then evaluate the index. Bounds or key failure is a language trap.

An expression used where a place is required must be one of the place forms above, except that
borrowing an rvalue may first materialize that rvalue in a temporary place.

## Evaluation model

### Results and full expressions

Evaluation of an expression produces exactly one of:

- a value;
- a place when the surrounding context requires a place;
- `return`, carrying a value;
- `break`, optionally carrying a value;
- `continue`;
- propagation from `?`, carrying `None` or `Err`;
- a language trap.

A *full expression* is the outermost expression of an initializer, expression statement,
`return` operand, `break` operand, condition, match scrutinee, loop iterable, aggregate
initializer, assignment operand, or block tail. Operands, call arguments, call receivers,
callees, and aggregate fields/elements are subexpressions of that enclosing full expression,
not independent temporary-destruction boundaries.

**EXEC-ONCE-001.** Every evaluated expression and subexpression is evaluated exactly once.
Static dispatch, method lookup, place resolution, auto-borrowing, and auto-dereferencing must
not re-evaluate source expressions.

**EXEC-DISPATCH-001.** Execution invokes exactly the function, inherent method, trait method,
default method, associated item, or compiler-recognized language hook selected by the normative
static rules. A runtime may not substitute source-order lookup, same-named inherent lookup for a
trait-qualified protocol, structural host equality/ordering, or string-name dispatch with
different semantics. Dispatch selection itself has no Core-visible side effect.

### Value and place contexts

**MOVE-READ-001.** Reading a place in a value context:

- copies the value and leaves the place initialized when its type is `Copy`;
- otherwise transfers ownership from the place and leaves that place moved-from;
- does not move when the construct explicitly borrows the place or a trait/operator rule
  specifies a reference argument.

The static rules reject any read that would move from a prohibited place, conflict with a live
borrow, or read an uninitialized or moved-from place. Runtime execution of a well-typed program
therefore never relies on recovering from such a read.

### Evaluation order

**EXEC-EVAL-001.** Unless a row below states otherwise, subexpressions evaluate left to right in
source order, and each completes—including its side effects, ownership transfers, normal
destruction, control transfer, or trap—before the next begins.

| Expression form | Normative order and result |
| --- | --- |
| Literal, function item, constant, unit | Produce the corresponding value. Constant evaluation is defined by `03-Type-System.md`. |
| Local/path in value context | Resolve the named place, then apply `MOVE-READ-001`. |
| Grouping | Evaluate the grouped expression once. |
| Field or tuple field | Resolve the base place, append the projection, then read only if a value is required. |
| Integer index | Resolve the base place, evaluate the index, check bounds, append the projection, then read only if required. |
| Range index | Resolve the base place, evaluate the range bounds left to right, validate them, and produce a slice place/view. |
| Borrow `&`/`&mut` | Resolve the operand place, materializing an rvalue temporary if permitted, then create a reference without reading the referent. |
| Dereference | Evaluate the reference, validate it, and produce its referent place; read only if required. |
| Other unary operation or cast | Evaluate the operand, then apply the operation. Numeric outcomes are completed by C2.9. |
| Non-short-circuit binary operation | Evaluate the left operand, then the right, then apply the operation. |
| `a && b` | Evaluate `a`; evaluate `b` only when `a` is `true`. |
| `a || b` | Evaluate `a`; evaluate `b` only when `a` is `false`. |
| Range construction | Evaluate the lower bound, then upper bound, then construct the range. |
| Free or associated call | Select the statically named function item without runtime callee evaluation; evaluate arguments left to right; transfer them to parameters; execute the body. |
| Function-valued call | Evaluate the callee expression exactly once and verify that its result is callable; evaluate arguments left to right; transfer them to parameters; invoke the selected function. A callee trap prevents every argument; an argument trap prevents later arguments and body execution. |
| Method call | Evaluate and resolve the receiver exactly once; evaluate arguments left to right; transfer/borrow the receiver and arguments; execute the selected body. |
| Tuple, array, struct, enum payload | Evaluate fields/elements left to right in written order, transferring each completed value into the partial aggregate. |
| Array repeat `[value; count]` | Evaluate `value` once; obtain the compile-time `count`; copy the value `count` times. The static rules require `Copy`. |
| `if` | Evaluate the condition, then only the selected branch. |
| `match` | Evaluate the scrutinee exactly once; test arms in source order; evaluate only the first matching arm. |
| Block | Evaluate statements in order, then the optional tail expression; the block result is the tail value or `Unit`. |
| `loop` | Repeatedly evaluate the body until control transfers. |
| `while` | Evaluate the condition before each iteration and the body only when it is `true`. |
| `for` | Evaluate the iterable once, obtain its iterator, repeatedly obtain one next value and execute one iteration. |
| `?` | Evaluate the operand once; unwrap `Some`/`Ok`, or transfer `None`/`Err` toward the enclosing function. |
| `return`, `break` | Evaluate the optional operand before beginning the control transfer. |
| `continue` | Begin the control transfer without an operand. |
| Assignment | Follow `EXEC-ASSIGN-001`; it is the named right-before-left exception. |

An expression form not admitted by the Core grammar has no Core execution semantics. A legal
Core expression form omitted from this table is a specification defect, not permission for
implementation-defined evaluation.

A temporary function value produced by a function-valued callee expression remains live
through argument evaluation and the invoked body. Unless ownership transfers elsewhere, it is
destroyed with the other temporaries at the end of the enclosing full expression.

**EXEC-FOR-001.** Every value accepted by the static `for`-loop iterator rules is executed
through its selected iterator protocol. The runtime may not narrow this to a privileged list of
container representations. The iterable expression evaluates once, each successful step yields
one iteration value, and no later step occurs after loop control exits.

### Assignment and replacement

**EXEC-ASSIGN-001.** Simple assignment `destination = source` executes in this exact order:

1. evaluate `source` completely and retain ownership of its result;
2. resolve `destination` as a place, including evaluation and validation of its projections;
3. detach the old destination object if the place is initialized;
4. write the new object and make the destination initialized;
5. destroy the detached old object.

The new value is installed before user-observable destruction of the old value begins. If the
old value's destructor traps, execution aborts with the new value already installed, although
subsequent program observation is impossible because Core has no trap recovery.

Compound assignment evaluates the right operand first, resolves the destination second, reads
the old destination value exactly once, applies the corresponding operation, and replaces the
old value using steps 3–5 above.

If source evaluation traps, the destination is not resolved or modified. If destination
resolution traps after the source has produced an owned object, the program aborts; the source
object and all other live values are not destructed because `DROP-ABORT-001` forbids unwinding.

```stark
let mut values = [0, 0];
values[next_index()] = make_value(); // make_value() completes before next_index().
```

```stark
let mut values = [0];
values[1] = String::from("owned"); // RHS is produced; index failure then aborts without Drop.
```

### Aggregate construction

**EXEC-AGG-001.** Aggregate fields and elements become owned by a partial aggregate as each
initializer completes. Declaration order determines layout-independent field identity;
written initializer order determines evaluation order. Duplicate, unknown, or missing required
fields are static errors and do not execute.

Tuple and positional enum payload order is index order. Named struct and enum payload
initializers evaluate in written order even when that differs from declaration order. The
completed object's later field-destruction order is declaration order reversed, not initializer
order reversed.

**EXEC-AGG-002.** If a later initializer performs normal early control transfer, already
completed fields are destroyed in reverse completion order before the transfer continues. If a
later initializer traps, construction aborts and no completed field is destructed, because a
trap never unwinds.

Core v1 has no struct-update syntax. If a later edition adds it, that edition must define its
evaluation, transfer, and failure sequence rather than inheriting one implicitly.

```stark
let value = (Loud::new("completed"), fail()); // A trap in fail() does not unwind completed.
```

### Normal control transfer and partial evaluation

**EXEC-CFLOW-001.** `return`, `break`, `continue`, and `?` propagation are normal control
transfers, not traps. Before a transfer leaves a scope, all live owners in that scope that are
not carried by the transfer are destroyed using the normal order. The carried value is moved
out before cleanup and is not destroyed by the exited scope.

For nested scopes, cleanup proceeds from the innermost exited scope outward. `break` destroys
the current iteration and loop-body state before producing the loop result. `continue`
destroys the current iteration and begins the next. `?` destroys scopes between the operator
and the function boundary before transferring `None`/`Err` to the caller.

## Moves, partial moves, and reinitialization

**OWN-MOVE-001.** Passing a non-`Copy` value to a by-value parameter, storing it in an aggregate,
returning it, breaking with it, propagating it, or binding it by value transfers ownership.
Passing or returning a `Copy` value copies it.

**OWN-PARTIAL-001.** Moving a field from an aggregate transfers only that field and marks that
field moved-from. Remaining initialized fields remain individually usable and destructible.
The whole aggregate cannot be read, moved, or destructured as fully initialized until every
moved field is reinitialized. Moving a field from a type that implements `Drop` is prohibited,
because its destructor requires the complete value.

Moving out of an indexed place is prohibited unless a library operation explicitly owns and
removes that element (for example `Vec::remove` or `pop`). This keeps shifting/container
invariants inside the owning operation.

**OWN-REINIT-001.** Assignment to a moved-from mutable place reinitializes that place. Assigning
all moved fields reconstitutes a fully initialized aggregate. Reinitialization does not destroy
the absent old value.

```stark
let mut text = String::from("first");
let moved = text;
text = String::from("second"); // text is initialized again; no old text remains to drop.
```

## Pattern execution

This section defines the C2.7 ownership/destruction mechanics for the pattern forms present in
Core v1. C2.8 remains authoritative for pattern typing, exhaustiveness, usefulness, and any
additional static restrictions.

**PAT-OWN-001.** A `match` scrutinee in value context is evaluated once. A non-`Copy` place
scrutinee moves into a hidden scrutinee owner; a `Copy` scrutinee is copied. Pattern traversal
follows written source order, recursively left to right. Tuple, array, and positional payload
patterns therefore use index order; named struct and enum patterns use the written field order,
even when it differs from declaration order.

For each binding reached by that traversal:

- a matched `Copy` component is copied into the binding and remains initialized in the hidden
  scrutinee;
- a matched non-`Copy` component moves into the binding and becomes moved-from in the hidden
  scrutinee;
- a reference scrutinee produces the reference projection required by the static pattern rules
  and does not move or copy the referent.

The wildcard `_` creates no binding and does not suppress destruction.

**PAT-DROP-001.** The selected arm's bindings are arm-local owners. On every normal exit from
the arm, binding locals are destroyed in reverse creation order after any arm result has moved
out. Then every still-owned, unbound component of the hidden scrutinee—including copied source
components and wildcarded or omitted fields—is destroyed exactly once in reverse
declaration/index order. For a named pattern, binding creation order is written pattern order,
while residual field destruction is reverse source declaration order. A trap in pattern testing
or the arm aborts without either cleanup.

Failed arm tests create no user-visible bindings and transfer no ownership out of the hidden
scrutinee. Core v1 has no match guards or alternative (`|`) patterns.

```stark
struct Loud { label: String }
impl Drop for Loud { fn drop(&mut self) { println(self.label.as_str()); } }
let pair = (Loud { label: String::from("bound") },
            Loud { label: String::from("wildcard") });
match pair { (kept, _) => println(kept.label.as_str()), }
```

## References and borrow-carrying values

### Identity and derivation

**REF-IDENTITY-001.** A reference is a value containing:

- the identity of a live referent object;
- a projection path and, for a slice, validated bounds;
- shared or exclusive access capability;
- the static validity interval established by the type/borrow rules.

This is an abstract identity, not a promised address or pointer width. Two references alias when
they designate overlapping storage of the same object.

Creating a reference does not copy or move the referent. Moving the reference value preserves
its designation. Moving the referent's owner while a conflicting reference is live is rejected
statically. When an ownership move is legal, object identity is preserved even if physical
storage changes.

### Projection, auto-borrow, and auto-dereference

**REF-PROJECT-001.** Dereference yields the designated place. Field, tuple, index, and slice
projections through a reference extend its projection path and retain the referent's identity.
Reads and writes through the resulting place observe the current object, never a snapshot.

Method auto-borrow creates a reference to the already evaluated receiver place. Auto-dereference
follows references without re-evaluating the receiver. An exclusive receiver requires a mutable
place and preserves write-through behavior.

### Returned and receiver-derived references

**REF-RETURN-001.** A returned reference must designate an object whose validity, under the
static shortest-input-lifetime rules, extends into the caller. Transfer across a call boundary
preserves referent identity and projection. A reference derived from `&self`, `&mut self`, or
another reference parameter designates the caller's object—not a parameter copy or callee-local
storage—and remains valid after the callee's activation ends for the statically permitted
interval.

```stark
struct Cell { value: Int32 }
impl Cell {
    fn value_ref(&self) -> &Int32 { &self.value }
}
let cell = Cell { value: 42 };
let reference = cell.value_ref();
```

### Slice references

**REF-SLICE-001.** Borrowing a range-indexed place creates a view of the original base object.
The view records validated start/end bounds and aliases those elements. Reads observe later
legal writes; writes through an exclusive slice reference update the original object. Creating
or returning a slice never clones its elements merely to satisfy reference semantics.

```stark
struct Numbers { items: Vec<Int32> }
impl Numbers {
    fn middle(&self) -> &[Int32] { &self.items[1..3] }
}
let mut items = Vec::new();
items.push(10);
items.push(20);
items.push(30);
let numbers = Numbers { items };
let middle = numbers.middle();
```

### Borrow-carrying values

**REF-CARRY-001.** A generic aggregate or library value that contains references carries their
referent identities, projections, capabilities, and validity requirements unchanged through
moves, calls, returns, and pattern binding. Moving the carrier does not retarget its references.
Destroying a carrier destroys owned non-reference components but does not destroy referenced
objects.

## Destruction

### Exactly-once rule

**DROP-EXACT-001.** Every initialized owned object that reaches a normal destruction point is
destroyed exactly once. A moved-from or never-initialized place is not destroyed. Moving a value
transfers its destruction obligation to the new owner. `Copy` types cannot implement `Drop`.

For a type implementing `Drop`, its `Drop::drop(&mut self)` body runs first while all
non-moved fields remain readable. After it returns normally, owned fields are recursively
destroyed. Calling `Drop::drop` directly is prohibited; the `drop(value)` function consumes its
argument and performs this sequence.

If a destructor traps, execution aborts immediately; remaining fields, siblings, locals, and
outer scopes are not destructed.

### Deterministic order

**DROP-ORDER-001.** On normal execution:

- expression-statement results are destroyed at the statement boundary;
- temporaries are destroyed in reverse creation order at the end of their full expression;
- block locals are destroyed in reverse declaration order;
- a return/break/propagated result moves out before local cleanup;
- function body locals are destroyed as their scopes exit, then parameters in reverse parameter
  order; a receiver is the first parameter for this purpose;
- tuple and array elements are destroyed in reverse index order;
- struct and named enum-variant fields are destroyed in reverse source declaration order;
- positional enum payloads are destroyed in reverse index order;
- only the active enum payload is destroyed;
- `Option`, `Result`, `Box`, and similar single-payload owners destroy their active payload;
- partially moved aggregates destroy only fields that remain initialized.

The order is defined in terms of source declarations and logical ownership, never a map order,
address order, backend layout, or optimizer choice.

### Temporaries

**EXEC-TEMP-001.** A temporary that is not transferred into another owner lives until the end
of its enclosing full expression and is then destroyed in reverse creation order. An rvalue
materialized so it can be borrowed as a call argument remains live through all later argument
evaluation and through completion of the call. An rvalue materialized as a borrowed method
receiver likewise remains live through argument evaluation and method completion. A temporary
function value used as a callee follows the same call-completion rule.

The abstract machine does not, by itself, extend an rvalue temporary merely because its
reference is stored in a local or returned. C2.8's static borrow/escape rules decide whether
such a reference is legal and must reject any accepted reference whose validity would outlast
the referent storage assigned by those rules. A temporary moved into a local, aggregate,
argument, return value, or other owner ceases to be a temporary and its destruction obligation
transfers.

A block tail value is moved or copied out before the block's locals and remaining temporaries
are destroyed. Temporaries created while evaluating a condition, match scrutinee, loop
iterable, argument, or aggregate initializer follow the more specific ownership rules for that
construct.

### Loops and iterators

**DROP-LOOP-001.** A `for` iteration binding owns the yielded value for exactly one iteration.
After the body exits normally, by `continue`, by `break`, by `return`, or by `?`, the binding
is destroyed unless its value was moved out. Body-local destruction precedes iteration-binding
destruction.

The iterator owns its unconsumed state. On normal loop completion or normal early transfer, the
iterator is destroyed and therefore destroys all remaining owned elements according to its
library contract. A trap aborts without this cleanup.

```stark
for item in values {
    if skip(item) { continue; }
    if stop(item) { break; }
    consume(item);
}
```

### Collections and ownership-discarding operations

**DROP-COLLECTION-001.** A collection owns its stored elements and entries. Destroying or
clearing a sequence/set destroys elements in reverse defined iteration order. Destroying or
clearing a map destroys entries in reverse defined iteration order, destroying each value
before its key. Operations with component-wise return contracts transfer only the returned
components and destroy the remaining removed or rejected components:

- `HashMap::insert` with a new key transfers the supplied key and value into the map and returns
  `None`;
- `HashMap::insert` with an equal existing key retains the stored key and its iteration
  position, installs the supplied new value, transfers the old stored value into
  `Some(old_value)`, and destroys the supplied duplicate key before returning; the returned
  `Option<V>` owns the old value;
- `HashMap::remove(&key)` leaves the lookup reference and referent caller-owned, removes the
  stored entry, destroys the stored key, and transfers the stored value into `Some(value)`;
  absence returns `None` and destroys nothing;
- `HashSet::insert` with a duplicate destroys the rejected supplied value and returns `false`;
  a new value transfers into the set and returns `true`;
- `HashSet::remove(&value)` leaves the lookup reference and referent caller-owned, destroys the
  removed stored value, and returns `true`; absence returns `false`.

For any other removal or replacement API, a component named in the return value transfers to
that return owner; an owned component not returned is destroyed by the operation. Ignoring a
returned owner does not move destruction into the collection operation: the expression-
statement result is a temporary and is destroyed at its normal temporary boundary.

```stark
let mut map: HashMap<String, Loud> = HashMap::new();
map.insert(String::from("key"), Loud::new("old"));
let replaced = map.insert(String::from("key"), Loud::new("new"));
match replaced {
    Some(old_value) => consume(old_value),
    None => {},
}
```

## Focused ordering examples

The following examples isolate ordering rules that are easy for a backend to implement
accidentally.

```stark
fn choose_function() -> fn(Int32) -> Int32 { selected }
let result = choose_function()(make_argument());
```

Here `choose_function` completes before `make_argument`; if it traps, `make_argument` does not
run.

```stark
struct Record { count: Int32, label: String }
let record = Record { count: 3, label: String::from("owned") };
match record {
    Record { label: text, count: n } => consume(text, n),
}
```

The written pattern creates `text` first by moving `label`, then creates `n` by copying
`count`.

```stark
struct Inner { left: Loud, right: Loud }
struct Outer { first: Loud, second: Inner }
let value = make_outer();
match value {
    Outer {
        second: Inner { right: b, left: a },
        first: c,
    } => consume(b, a, c),
}
```

Binding creation order is `b`, `a`, `c`; normal binding destruction order is `c`, `a`, `b`.

### Trap termination

**DROP-ABORT-001.** A language trap is aborting. It performs no stack unwinding, scope cleanup,
temporary cleanup, partial-aggregate cleanup, collection cleanup, or user `Drop` calls for live
objects. Side effects and destruction completed before the trap remain observable. Core v1 has
no catch or recovery mechanism.

`panic(message)` emits its specified panic diagnostic/message and then traps. The precise
process exit-code mapping is owned by C2.9; the abstract exit category is `trap`.

```stark
struct Loud { label: String }
impl Drop for Loud { fn drop(&mut self) { println(self.label.as_str()); } }
let live = Loud { label: String::from("not dropped") };
panic("abort");
```

## Numeric operations and conversions

**NUM-INT-ARITH-001.** Signed integers are fixed-width two's-complement values and unsigned
integers are fixed-width binary values of the width in their type name. Integer addition,
subtraction, multiplication, exponentiation, and unary negation compute the mathematical result
and trap when it is not representable in the result type. This checked behavior is identical in
all build modes and targets. Negating an unsigned value is ill-typed; negating the minimum
signed value traps. Integer exponentiation requires a nonnegative exponent and traps otherwise;
each intermediate multiply is checked.

**NUM-INT-DIV-001.** Integer division by zero and remainder by zero trap. Signed division
truncates toward zero; the remainder satisfies `a == (a / b) * b + (a % b)` and has the sign of
`a` or is zero. Dividing the minimum signed value by `-1`, and taking its remainder by `-1`,
trap because the intermediate quotient is not representable. Unsigned division and remainder
use ordinary Euclidean nonnegative quotient/remainder.

**NUM-SHIFT-001.** A shift count may have any integer type, but its mathematical value must be
nonnegative and strictly less than the bit width of the left operand; otherwise the operation
traps. Left shift traps when the mathematical result is not representable. Unsigned right shift
fills with zero; signed right shift is arithmetic and rounds toward negative infinity. No shift
count is masked or reduced modulo the width.

**NUM-CAST-001.** Numeric conversion is checked as follows:

- integer to integer preserves the mathematical value and traps if the target cannot represent
  it;
- `Float32` to `Float64` is exact; `Float64` to `Float32` rounds once using
  round-to-nearest, ties-to-even and may produce signed infinity;
- integer to float rounds once using round-to-nearest, ties-to-even;
- finite float to integer first truncates toward zero and then traps unless the result is
  representable; NaN and either infinity always trap.

Conversion preserves a floating zero's sign and infinity's sign. A statically evaluated failing
conversion is a compile-time error instead of a runtime trap.

**NUM-FLOAT-OP-001.** Each primitive floating `+`, `-`, `*`, `/`, unary `-`, and comparison
uses IEEE 754 binary32/binary64 with round-to-nearest, ties-to-even. Floating division by zero
does not trap: it produces the IEEE infinity or NaN result. Floating `%` is the correctly
rounded value of `x - trunc(x / y) * y` using the exact mathematical quotient, with the sign of
a nonzero result matching `x`; zero divisor, infinite dividend, or NaN operand produces NaN.
NaN propagates as a quiet NaN; operations that create a NaN produce the canonical quiet NaN
with sign zero and all payload bits other than the quiet bit zero. Negation flips the sign
bit, including for zero and NaN. Implementations may not reassociate operations, contract
multiply-add, flush subnormals, or use a different rounding mode.

Core v1 has no floating `**` operator. Floating exponentiation is the
`std::math::pow(Float64, Float64)` library operation governed by
`STD-MATH-001`; use of `**` with either floating operand is a type error.

For the same declared float type, inputs, and sequence of primitive operations/casts, the
result bits are backend- and target-independent under `NUM-FLOAT-REPRO-001`. Decimal literals
are converted directly to the destination format using
round-to-nearest, ties-to-even, independent of host parsing. Transcendental and other
standard-library math functions follow `STD-MATH-001`; they are not primitive operations and
need not be bit-identical across targets.

## Trap categories

**TRAP-CATEGORY-001.** A *language trap* is a failure explicitly required by a normative Core
rule, including explicit `panic`, bounds failure, invalid range boundary, division by zero,
checked arithmetic failure, and failing checked conversion where the owning numeric/text rule
requires a trap. It records a stable category and the source location of the operation that
failed.

Allocation exhaustion, stack exhaustion, recursion/call-depth exhaustion, unavailable host
services, host I/O failure outside an API's `Result`, target failure, and OS termination are
`host/process failure`, not language traps. Ordinary file/stream failures handled by the
standard-library contract produce `Result` and do not terminate execution. Compiler limits
reject compilation with a classified diagnostic. An implementation must never report an
internal host panic as a specified STARK trap.

## Observable execution and differential comparison

**OBS-COMPARE-001.** For a fixed program, inputs, target contract, enabled extensions, artifact
set, and external environment, execution backends are semantically equal only when all
applicable observations match:

- stdout bytes in order;
- stderr bytes in order;
- abstract exit category (`normal(status)`, `language trap`, `host/process failure`);
- returned Core value for a harnessed function, compared by its Core value semantics;
- language-trap category;
- language-trap source identity and span;
- user-observable `Drop` side effects and their order;
- artifact verification result and artifact identity when verification is part of the execution
  request.

Wall-clock time, physical addresses, allocation strategy, stack depth, object layout, generated
code bytes, and non-normative diagnostic prose are not observations. Numeric target variation,
process status numbers, environmental I/O, and resource failures are compared only after C2.9
classifies them.

## Conformance

A conforming Core v1 implementation must implement every rule in this document. Optimization
may omit work only when it cannot change an observation in `OBS-COMPARE-001`, including
destructor effects and trap order. The interpreter, future MIR interpreter, and native backend
are implementations of this abstract machine; none is an independent semantic authority.
