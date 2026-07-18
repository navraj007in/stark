# STARK Core v1 Memory-Safety Model

## Status and authority

This chapter states Core v1 memory-safety guarantees. Static ownership, borrowing, lifetime,
`Copy`, and `Drop` legality is defined by `03-Type-System.md` and checked as required by
`04-Semantic-Analysis.md`. Runtime values, places, moves, references, temporary lifetime,
destruction order, and traps are defined solely by `CORE-V1-ABSTRACT-MACHINE.md`.

This chapter does not define physical layout. Core v1 makes no general promise that a value is
stack-allocated or heap-allocated, that a reference is a machine pointer, that a slice is two
machine words, or that an enum uses a particular tag. Observable layout and target contracts are
owned by C2.9.

## Safety invariants

For every well-typed Core v1 program that has not terminated through a language trap or an
external failure:

1. every live non-`Copy` object has exactly one owner;
2. a moved-from or uninitialized place cannot be read;
3. a live shared reference permits reads but not mutation through that reference;
4. a live exclusive reference excludes conflicting shared or exclusive access;
5. a reference cannot be used outside its statically valid interval;
6. dereference and projection preserve referent identity and cannot silently produce a
   disconnected snapshot;
7. bounds-checked places cannot access storage outside their designated object or slice;
8. every owned object that reaches normal cleanup is destroyed exactly once;
9. a value is `Copy` only when its complete type satisfies the `Copy` legality rules;
10. normal control transfer preserves ownership and cleanup obligations.

These are language guarantees, not implementation strategies. A compiler may erase borrow
metadata and omit redundant runtime checks after proving that the same observable behavior and
safety invariants remain.

## Ownership and moves

Ownership transfer does not duplicate the transferred object. Passing or returning a non-`Copy`
value, storing it into another owner, binding it by value, or removing it through an owning
collection operation transfers the destruction obligation with the object.

Partial-move legality, prohibited indexed moves, and reinitialization checks are defined by
`03-Type-System.md`. Their runtime effect is defined by abstract-machine rules
`OWN-PARTIAL-001`, `OWN-REINIT-001`, and `DROP-EXACT-001`.

## Borrowing and validity

Core v1 has shared and exclusive references. The static rules conservatively determine their
validity without written lifetime parameters. Returned references must derive from permitted
reference inputs, and borrow-carrying generic values carry the same validity constraints as
their contained references.

Reference identity, projection, method receivers, returned references, slice views, moves of
reference carriers, and caller/callee boundaries are defined by `REF-IDENTITY-001` through
`REF-CARRY-001` in `CORE-V1-ABSTRACT-MACHINE.md`.

## Destruction safety

The type system prohibits a type from implementing both `Copy` and `Drop`, prohibits explicit
calls to `Drop::drop`, and prohibits partial field moves from a type whose destructor requires
the complete value.

The abstract machine defines all destruction points and ordering, including locals,
temporaries, parameters, fields, partial aggregates, loop bindings, iterators, collections,
explicit `drop(value)`, replacement assignment, normal early transfer, and aborting traps. No
physical container order or object layout may substitute for the specified logical order.

## Traps and external failures

A specified language trap preserves memory safety by terminating execution; Core v1 does not
unwind or run remaining destructors. Side effects completed before the trap remain observable.
`TRAP-CATEGORY-001` distinguishes language traps from host panics and external failures.

Allocation exhaustion, stack exhaustion, OS termination, host I/O failure, and target limits
are not memory-safety loopholes and are not automatically STARK traps. C2.9 classifies their
portable guarantees.

## Informative future directions (non-normative)

Written lifetime parameters, reference fields, reference counting, interior mutability,
concurrency, and unsafe/native memory access are not Core v1 features. A future edition must
preserve the invariants above or explicitly version its compatibility boundary.

## Conformance

A conforming implementation must enforce the static rules and preserve these safety invariants
while implementing the abstract machine. Deviations and extensions must be documented and
tested; an implementation representation is never evidence that a different language rule
applies.
