# STARK Type System Specification

## Overview
STARK features a static type system with type inference, ownership semantics, and memory safety guarantees.

## Core Principles
1. **Static Typing**: All types resolved at compile time
2. **Type Inference**: Types can be inferred when unambiguous
3. **Memory Safety**: No null pointer dereferences, no use-after-free
4. **Ownership**: Clear ownership and borrowing rules
5. **Zero-cost Abstractions**: Type safety without runtime overhead

## Primitive Types

### Integer Types
```stark
Int8    // 8-bit signed integer (-128 to 127)
Int16   // 16-bit signed integer (-32,768 to 32,767)
Int32   // 32-bit signed integer (-2^31 to 2^31-1)
Int64   // 64-bit signed integer (-2^63 to 2^63-1)

UInt8   // 8-bit unsigned integer (0 to 255)
UInt16  // 16-bit unsigned integer (0 to 65,535)
UInt32  // 32-bit unsigned integer (0 to 2^32-1)
UInt64  // 64-bit unsigned integer (0 to 2^64-1)
```

Default integer type is `Int32` for literals that fit, `Int64` otherwise.

### Floating Point Types
```stark
Float32  // 32-bit IEEE 754 floating point
Float64  // 64-bit IEEE 754 floating point
```

Default floating point type is `Float64`.

### Other Primitive Types
```stark
Bool    // Boolean: true or false
Char    // Unicode scalar value (32-bit)
Unit    // Unit type: () - represents no meaningful value
str     // String slice (unsized), typically used behind references: &str
```

### String Type
```stark
String  // UTF-8 encoded string, heap-allocated, growable
```

### String Slice Type
```stark
// str is an unsized string slice type. It is used via references.
let s: &str = "hello";
let owned: String = String::from(s);
```

### Never Type
`!` (the never type) is the type of expressions that never produce a value,
such as calls to `panic`. It may appear only as a function return type.

```stark
fn panic(message: &str) -> !
```

Rules:
- An expression of type `!` coerces to any other type. This allows, for example,
  `panic(...)` to appear in one arm of an `if` or `match` whose other arms
  produce a value.
- A function returning `!` MUST NOT return normally.

### Core Type Identity

**TYPE-PRIM-001.** Each named primitive type is distinct. `Unit` and `()` are
two spellings of the same single-inhabitant type. Array identity is the ordered
pair of element type and length; tuple identity is its ordered type sequence;
reference identity includes pointee type and mutability; function identity
includes the ordered parameter types and return type. The never type `!` is
uninhabited and distinct from every other type.

## Composite Types

### Array Types
```stark
[T; N]   // Fixed-size array of N elements of type T (sized)
[T]      // Slice of elements of type T (unsized)
```

A slice `[T]` is an *unsized* view into a contiguous sequence (an array, or the
elements of a `Vec<T>`). Like `str`, it is used behind references:

```stark
let fixed: [Int32; 5] = [1, 2, 3, 4, 5];
let view: &[Int32] = &fixed;          // Array reference coerces to slice reference
let part: &[Int32] = &fixed[1..4];    // Slicing with a range yields &[T]
```

A local variable cannot have the bare unsized type `[T]`; use `[T; N]` for a
fixed-size array or `Vec<T>` (standard library) for a growable one.

Indexing rules:
- `expr[i]` where `i` is an integer denotes a *place* of type `T`
  (bounds-checked; traps on out-of-bounds).
- `expr[r]` where `r` is a `Range` denotes a *place* of the unsized slice type
  `[T]`; traps if the range is out of bounds or inverted. Because `[T]` is
  unsized, a slice place must be borrowed immediately: `&expr[r]` has type
  `&[T]` and `&mut expr[r]` has type `&mut [T]`.

### Range Type
The range operators produce values of the standard library `Range<T>` type:

```stark
let r = 0..10;      // Range<Int32>: 0,1,...,9 (half-open)
let ri = 0..=9;     // RangeInclusive<Int32>: 0,1,...,9
```

Ranges over integer types implement `Iterator` and may be used with `for` loops
and slicing.

### Tuple Types
```stark
()           // Empty tuple (Unit type)
(T,)         // Single-element tuple
(T1, T2)     // Two-element tuple
(T1, T2, T3) // Three-element tuple
// ... up to 16 elements
```

Examples:
```stark
let empty: () = ();
let single: (Int32,) = (42,);
let pair: (Int32, String) = (42, "hello");
```

### Struct Types
```stark
struct Point {
    x: Float64,
    y: Float64
}

struct Person {
    name: String,
    age: Int32,
    pub email: String  // Public field
}
```

Restriction (Core v1): struct and enum declarations MUST NOT declare fields
whose *written* type is a reference type (`&T`, `&mut T`). Instantiating a
generic type with a reference type argument (e.g. `Option<&T>`) is permitted;
the resulting value is *borrow-carrying* — see "References and Lifetimes"
below.

**TYPE-NOMINAL-001.** A struct or enum type has nominal identity:

```text
canonical package instance + module path + item name + normalized generic arguments
```

The canonical package-instance token is supplied by the package identity rules
completed in C2.9. Relocation, import aliases, re-exports, and local type
aliases do not change identity. Two separately declared types remain distinct
even when their fields and variants are structurally identical.

### Enum Types
```stark
enum Color {
    Red,
    Green,
    Blue
}

enum Option<T> {
    Some(T),
    None
}

enum Result<T, E> {
    Ok(T),
    Err(E)
}
```

### Reference Types
```stark
&T       // Immutable reference to T
&mut T   // Mutable reference to T
```

## Function Types
Function types are written with the `fn` keyword and denote *non-capturing*
functions (named functions or function references). Capturing closures are a
future extension.

```stark
fn(T1, T2) -> R    // Function taking T1, T2 and returning R
fn() -> R          // Function taking no parameters, returning R
fn(T)              // Function taking T, returning Unit
```

## Type Aliases
```stark
type Age = Int32;
type Point2D = (Float64, Float64);
type ErrorCode = Int32;
```

**TYPE-ALIAS-001.** A type alias is transparent. After capture-avoiding
substitution of its generic arguments, it denotes exactly its target type and
introduces no nominal identity, constructor, implementation-coherence
boundary, or distinct method set. Alias expansion is recursive; an alias
cycle is a compile-time error. A nominal wrapper requires a distinct struct or
enum declaration.

## Type Well-Formedness and Sizedness

**TYPE-WF-001.** Every value type and every generic parameter in Core v1 is
implicitly `Sized`; Core has no syntax to relax this bound. The only unsized
types are implementation-provided `str` and slice `[T]`, and they may occur
only immediately behind `&` or `&mut`. Bare unsized locals, parameters,
returns, fields, tuple elements, array elements, generic arguments, and
`Box<str>`/`Box<[T]>` are prohibited in Core v1.

A declaration is infinitely sized when its field-type dependency graph has a
cycle containing only direct value edges. References and the
implementation-provided owning indirections `Box<T>` and `Vec<T>` break that
graph. Direct and indirect value recursion is rejected; recursion through one
of those indirections is well formed when all other rules hold.

The following edge types are explicit:

- `[T; 0]` is a sized, inhabited empty array and still requires sized `T`;
- empty structs are sized, inhabited nominal types;
- zero-variant enums are sized, uninhabited nominal types;
- tuples support arities zero through sixteen; `()` is `Unit`;
- array length is a constant `UInt64`; Core imposes no smaller semantic
  maximum, while finite compiler/target resource limits are classified by
  C2.9;
- `!` is uninhabited, may be written only as a function return type, and
  coerces to any expected type.

```text
struct InvalidNode { next: InvalidNode }
struct ValidNode { next: Option<Box<ValidNode>> }
type Meter = Int32;
```

## The Self Type
Within a `trait` or `impl` block, `Self` refers to the implementing type:

```stark
trait Eq {
    fn eq(&self, other: &Self) -> Bool;
}
```

## Ownership and Borrowing

### Ownership Rules
1. Each value has exactly one owner
2. When the owner goes out of scope, the value is dropped
3. Values can be moved (ownership transfer) or borrowed (temporary access)

### Move Semantics
```stark
let a = String::from("hello");
let b = a;  // a is moved to b, a is no longer valid
// print(a);  // Error: use of moved value
```

### Borrowing Rules
**OWN-BORROW-001.** For any overlapping place, a live exclusive borrow
prohibits every other read, write, borrow, move, or destruction through a
different access path; one or more live shared borrows permit other shared
reads/borrows but prohibit writes, exclusive borrows, moves, and destruction.
Disjoint field projections do not overlap. Index projections are treated as
overlapping unless the checker proves distinct constant indices. Reborrowing
cannot grant stronger mutability or outlive the source reference.

1. You can have either one mutable reference or any number of immutable references
2. References must always be valid (no dangling pointers)
3. References cannot outlive the data they refer to

```stark
fn borrow_immutable(s: &String) {
    // Can read but not modify
}

fn borrow_mutable(s: &mut String) {
    // Can read and modify
}

let mut text = String::from("hello");
borrow_immutable(&text);       // OK
borrow_mutable(&mut text);     // OK
```

## References and Lifetimes (Core v1)
Core v1 has no lifetime annotations. Instead, it enforces the following
conservative rules, which are normative:

**OWN-REGION-001.** A reference or borrow-carrying value bound to a local
remains live from creation through the end of that local's lexical block,
unless the complete carrier is consumed earlier by `drop`. Moving a carrier
transfers, rather than ends, its live borrow: the borrow then remains live
through the destination carrier's lexical region or through completion of the
consuming call. A reference parameter is live for the call. An unbound borrow
used as a subexpression remains live through its enclosing full expression;
borrowed rvalue arguments and borrowed temporary method receivers therefore
remain live through call completion. Core v1 performs no last-use shortening
and no temporary-lifetime extension for `let r = &make_value()`; such an
escaping temporary reference is rejected.

**OWN-RETURN-001.** A returned reference or borrow-carrying value must derive,
on every return path, only from reference parameters. Its caller-visible
region is the intersection (the shortest region) of every reference parameter
from which that path's result may derive. Projections into the referent of a
reference parameter are permitted. A local, temporary, by-value parameter,
constant temporary, or a field whose root is one of those non-reference
sources cannot be returned by reference.

1. **No declared reference fields.** Struct, enum variant, and tuple struct
   declarations MUST NOT write a reference type (`&T`, `&mut T`) as a field
   type. (Generic *instantiation* with a reference argument is allowed; see
   Borrow-Carrying Types below.)
2. **References derive from parameters.** A function may return a reference
   (or borrow-carrying value) only if every control-flow path derives it from
   one of its reference parameters (directly, by field/index projection, or by
   calling a function that itself obeys this rule). Returning a reference
   derived from a local variable or a by-value parameter is a compile-time
   error.
3. **Shortest-input-lifetime rule.** The lifetime of a returned reference (or
   borrow-carrying value) is the *shortest* of the lifetimes of all reference
   parameters from which it could have been derived. Callers MUST treat it as
   invalidated as soon as the shortest-lived of those arguments is
   invalidated.
4. **Local borrows are lexically scoped.** A borrow bound to a variable
   (`let r = &x;`) lasts from its creation to the end of its enclosing block
   (or until the borrower is explicitly dropped with `drop(...)`). The borrow
   checker enforces the exclusive/shared rules over that entire region — even
   if the reference is never used again. To end such a borrow early, introduce
   an inner block or call `drop`. A *temporary* borrow that is not bound to a
   variable (e.g. `f(&x);`) ends at the end of its enclosing statement, so
   `f(&x); g(&mut x);` is legal.

```stark
let mut s = String::from("hello");
{
    let r = &s;           // Borrow begins
    println(r.as_str());
}                         // Borrow ends with the block
s.push('!');              // OK: no live borrow
```

Example of rule 3:
```stark
fn longest(x: &str, y: &str) -> &str {
    if x.len() > y.len() { x } else { y }
}
// The returned reference is valid only while BOTH x's and y's referents
// are valid, regardless of which branch was taken.
```

These rules reject some safe programs (that is intentional). Lifetime
annotations that relax rule 3, user structs with declared reference fields,
and non-lexical (last-use) borrow regions are future extensions; because the
lexical rule is strictly more conservative, adopting last-use regions later
cannot break conforming programs.

### Borrow-Carrying Types
**OWN-CARRY-001.** Borrow provenance is structural and transitive through
tuples, generic arguments, enum payloads, function arguments/results, pattern
bindings, and implementation-provided borrowed views. A merge of control-flow
paths carries the union of possible source referents and the most restrictive
capability; its region cannot exceed the intersection of their valid regions.

A type is **borrow-carrying** if it is:
- a reference type `&T` or `&mut T`; or
- a generic type with at least one borrow-carrying type argument
  (e.g. `Option<&T>`, `(Int32, &str)`); or
- an implementation-provided borrowed view type (the standard library's
  borrowed iterators such as `VecIter<T>`, `CharsIter`, `KeysIter<K>`, and
  the slice types behind references).

A borrow-carrying value is treated *as if it were a reference* for all rules
in this section:
- It counts as a live borrow of the value(s) it was derived from — shared for
  `&`-derived values, exclusive for `&mut`-derived values — for the borrow
  region defined by rule 4.
- It MUST NOT be stored in a user-declared struct or enum field, returned
  except under rules 2–3, or otherwise escape the lifetime of its source.
- Binding it with `let` is permitted; the binding is subject to rule 4.

This is a taint rule: borrow-carrying-ness propagates through generic
instantiation and function returns, and the checker tracks the source
variables each borrow-carrying value may borrow from. It is what makes APIs
like `Vec::get(&self, i) -> Option<&T>` sound without lifetime annotations:
the returned `Option<&T>` is an active shared borrow of the `Vec` until it
goes out of scope.

## Type Inference

### Deterministic Local Inference

**TYPE-INFER-001.** Inference is local to one function or constant body and
never infers item signatures. For each unannotated local, the checker
collects equality and trait constraints from its initializer and every use of
that binding until its scope ends or it is shadowed. Consequently, later uses
within that region may constrain the local, but uses outside it and bodies of
called functions may not.

Expected types flow inward from explicit annotations, function parameters,
return types, assignment destinations, aggregate fields, branch/arm
unification, and an enclosing call's expected result. Constraint collection
is independent of traversal or declaration-map iteration order. Solving
proceeds as follows:

1. expand transparent aliases and create fresh variables for every generic
   instantiation;
2. collect exact-type equations, expected-type constraints, associated-type
   obligations, and required trait bounds;
3. repeatedly apply equality unification and uniquely selected associated-type
   normalization to a fixed point;
4. apply permitted coercions only at their explicit coercion sites;
5. default an unconstrained integer literal to `Int32` when representable,
   otherwise `Int64`, and an unconstrained float literal to `Float64`;
6. reject any remaining unconstrained variable, conflicting equation,
   unsatisfied obligation, or non-unique solution as an inference error.

Inference never chooses an implementation or overload merely because it was
declared first. Error-recovery variables may suppress cascaded diagnostics but
must not turn a rejected program into an accepted one.

```stark
let x = 42;             // Inferred as Int32
let y = 3.14;           // Inferred as Float64
let z = [1, 2, 3];      // Inferred as [Int32; 3]
```

### Function Return Types
Function return types are NOT inferred. A function without a `->` annotation
returns `Unit`.

```stark
fn add(a: Int32, b: Int32) -> Int32 {
    a + b
}

fn greet(name: &str) {      // Returns Unit
    println(name);
}
```

### Generic Type Inference

**TYPE-GENERIC-001.** Each generic use receives fresh parameters. Explicit
turbofish arguments bind the corresponding parameters first; omitted
arguments are inferred from value arguments and then from the expected result
type. All occurrences of one parameter must normalize to one type. Associated
type bindings are obligations, not new inference variables, and normalize
only after unique trait selection. If any parameter remains unconstrained,
the call requires explicit arguments.

Generic declarations are compared after alpha-renaming parameters and
capture-avoiding substitution. Bounds are conjunctions independent of source
order. Monomorphization and dictionary passing are implementation strategies,
not type-system differences.

```stark
fn identity<T>(x: T) -> T {
    x
}

let result = identity(42);  // T inferred as Int32
```

## Subtyping and Coercion

Core v1 has no general subtyping. At an expected-type boundary, coercions are
attempted in this deterministic order: identity/alias normalization, never
coercion, mutable-to-shared reference weakening, then array-reference to
slice-reference coercion. A coercion may combine mutable-to-shared weakening
with array-to-slice exactly once (`&mut [T; N]` to `&[T]`); no other coercion
chains, implicit numeric conversions, user conversions, or trait-based
coercions exist.

### Numeric Coercions
No implicit numeric conversions. Explicit casting required:
```stark
let x: Int32 = 42;
let y: Int64 = x as Int64;  // Explicit cast required
```

Cast semantics (`as`):
- Between integer types: value-preserving if in range; a cast whose value does
  not fit the target type is a runtime error and MUST trap.
- Between floating point types: rounds to nearest representable value.
- Integer to float and float to integer: float-to-int truncates toward zero and
  MUST trap if the result does not fit the target type (including NaN/Inf).

### Reference Coercions
```stark
&mut T -> &T        // Mutable reference to immutable reference
&T -> &T            // Same type (identity)
```

### Array to Slice Coercion
**TYPE-COERCE-003.** Array-to-slice coercion is available only for references.
It preserves the original referent, bounds, and borrow region; it never copies
elements or constructs an owned slice. Shared input produces shared output.
Mutable input may produce mutable output or be weakened to shared output.

```stark
&[T; N] -> &[T]     // Array reference to slice reference
&mut [T; N] -> &mut [T]  // Mutable array reference to mutable slice reference
```

### Never Coercion
```stark
! -> T              // The never type coerces to any type
```

## Trait System (Basic)

### Trait Definition
**TRAIT-DEF-001.** A trait declares required methods, optional default method
bodies, and associated types. An implementation must define every associated
type and required method not supplied by a default, with an alpha-equivalent
generic signature, receiver form, parameter types, return type, and bounds.
An implementation may override a default but may not add trait-associated
items. Trait and implementation bodies are type-checked under their declared
bounds.

```stark
trait Display {
    fn fmt(&self) -> String;
}

trait Eq {
    fn eq(&self, other: &Self) -> Bool;
}
```

### Trait Implementation
```stark
impl Display for Int32 {
    fn fmt(&self) -> String {
        // Convert integer to string
    }
}

impl Eq for Point {
    fn eq(&self, other: &Point) -> Bool {
        self.x == other.x && self.y == other.y
    }
}
```

### Associated Types
**TRAIT-ASSOC-001.** An associated item is identified by its declaring trait
and item name. `T::Item` in a trait-bound context is a projection, not a free
name; it normalizes only after a unique applicable implementation is selected.
Fully qualified syntax selects the named trait. A missing associated value,
conflicting binding, unresolved projection, or normalization cycle is a
compile-time error.

Traits may declare associated types; implementations assign them:

```stark
trait Iterator {
    type Item;
    fn next(&mut self) -> Option<Self::Item>;
}

struct Counter { count: Int32 }

impl Iterator for Counter {
    type Item = Int32;
    fn next(&mut self) -> Option<Int32> {
        self.count += 1;
        Some(self.count)
    }
}
```

Bounds may constrain associated types with bindings: `I: Iterator<Item = T>`.

## Type Checking Rules

### Assignment Compatibility
```stark
let x: T = expr;  // expr must have type T or be coercible to T
```

### Function Call Compatibility
```stark
fn f(param: T) { ... }
f(arg);  // arg must have type T or be coercible to T
```

### Arithmetic Operations
```stark
// Binary arithmetic requires same types
let result = x + y;  // x and y must have the same numeric type

// Comparison operators
let cmp = x < y;     // x and y must have the same comparable type
```

### Logical Operations
```stark
let both = a && b;    // a and b must be Bool
let negated = !x;     // x must be Bool
```

### For Loops
`for x in expr` requires `expr` to have a type that implements `Iterator`;
the loop variable binds successive `Item` values. Integer ranges implement
`Iterator`; slices and collections provide `.iter()` methods (see the
standard library specification).

```stark
for i in 0..10 {
    println(i.fmt());
}
```

### Control Flow Typing
- **`if`/`else`**: an `if` with an `else` is an expression whose branches must
  unify to a single type (identical types, or unifiable via the `!` coercion —
  e.g. one branch may `panic`). An `if` *without* `else` has type `Unit`, and
  its branch must also have type `Unit`.
- **`match`**: all arm expressions must unify to a single type (with `!`
  coercion permitted per arm).
- **`loop`**: a `loop` expression's type is the unified type of the operands
  of its `break value;` statements; if every `break` is bare (no value) or the
  loop has no `break`, the type is `Unit` (a `loop` with no `break` has type
  `!`).
- **`while` / `for`**: always have type `Unit`. `break` inside `while`/`for`
  MUST NOT carry a value.
- **Block**: the type of the trailing expression, or `Unit` if there is none.

**TYPE-LOOP-001.** Reachable `break` operands in one `loop` are unified using
the ordinary expected-type and never-coercion rules. A reachable bare
`break` contributes `Unit`. A `loop` with no reachable `break` has type `!`;
otherwise it has the unique unified break type. `while` and `for` have type
`Unit` and reject value-carrying `break`.

### Method Calls and Auto-Borrowing
**TYPE-METHOD-001.** Method selection is independent of declaration and hash-
map iteration order. For each receiver type considered by auto-dereference,
the checker collects every visible, type-applicable inherent candidate. If
that set is nonempty it must contain exactly one candidate and trait
candidates are ignored. Otherwise it collects applicable methods from traits
in lexical scope or the prelude; exactly one must remain after generic
substitution and obligations. Zero candidates is unresolved and multiple
candidates is ambiguous. Fully qualified `Trait::method(receiver, ...)`
bypasses trait-name lookup but still requires a unique coherent impl.

A method call `recv.m(args)` is resolved as follows:

1. **Candidate collection.** Candidates are, in priority order:
   (a) inherent methods — `fn m` in `impl RecvType { ... }` blocks;
   (b) trait methods — `fn m` from any trait that RecvType implements, where
   the trait is *in scope* (defined in, or imported via `use` into, the
   current module, or in the prelude). Inherent methods shadow trait methods
   of the same name.
2. **Ambiguity.** If two or more traits in scope supply an applicable `m` and
   no inherent method exists, the call is a compile-time error. Disambiguate
   with a fully-qualified call, passing the receiver explicitly:
   `TraitName::m(&recv, args)`.
3. **Receiver coercion (auto-borrowing).** Let `S` be the receiver
   expression's type. Candidates are matched trying these receiver forms in
   order: `self: S` (by value), `self: &S` (auto-borrow: the compiler inserts
   `&`), `self: &mut S` (auto-mutable-borrow: the compiler inserts `&mut`;
   requires the receiver to be a mutable place). Auto-borrows follow the
   normal borrow rules.
4. **Auto-dereference.** If `S` is `&T` or `&mut T` and no candidate matches
   on `S`, resolution retries with `T` (one level of dereference, applied
   repeatedly for nested references). Combined with step 3 this allows
   `(&v).len()` and `v.len()` to resolve identically.
5. **Visibility.** Private methods are callable only per the module rules
   (07-Modules-and-Packages.md).

Trait methods can always be called in fully-qualified function form —
`Display::fmt(&x)` — since a method is an ordinary function whose first
parameter is the receiver.

**TYPE-METHOD-002.** Auto-dereference examines `S`, then repeatedly removes
one leading `&`/`&mut`; at each level receiver matching tries by-value,
shared-borrow, then exclusive-borrow form. Selection stops at the first
dereference level with an applicable candidate. Auto-borrow never evaluates
the receiver twice, never moves through a shared reference, and may create
`&mut` only from a mutable place not conflicting with a live borrow. No
argument-position auto-borrow, auto-dereference, or user coercion exists.

### Operators and Traits
Operator expressions on **primitive types** have built-in meaning (Numeric
Semantics below). Equality and ordering on every non-primitive concrete type,
and on generic parameters, require one uniquely selected coherent `Eq` or
`Ord` implementation and dispatch to its method. Arithmetic on a generic
parameter requires `Num` and becomes the corresponding primitive operation
after monomorphization:

| Operator | Non-primitive/generic requirement | Meaning |
| --- | --- | --- |
| `==`, `!=` | `T: Eq` | `Eq::eq(&a, &b)` (negated for `!=`) |
| `<`, `<=`, `>`, `>=` | `T: Ord` | `Ord::cmp(&a, &b)` compared to `Ordering` |
| `+ - * / % **`, bitwise, shifts | generic `T: Num` | primitive operation after monomorphization |

`Num` is a compiler-known marker trait implemented by exactly the built-in
numeric types (`Int8`–`Int64`, `UInt8`–`UInt64`, `Float32`, `Float64`); user
types cannot implement it in Core v1. Operator overloading for user-defined
types (Add/Sub/... traits) is a future extension. `&&`, `||`, and `!` (on
`Bool`) are built-in, short-circuiting, and not overloadable.

```stark
fn max<T: Ord>(a: T, b: T) -> T {
    if a > b { a } else { b }   // a > b desugars to Ord::cmp(&a, &b)
}
```

### Evaluation Order (Core v1)
Runtime evaluation order is defined solely by
`CORE-V1-ABSTRACT-MACHINE.md` (`EXEC-EVAL-001`, `EXEC-ONCE-001`, and
`EXEC-ASSIGN-001`). Type checking must preserve those rules and must not
introduce an evaluation or ownership transfer merely to select a type,
method, or trait implementation.

### Copy and Drop (Soundness Rules)
**OWN-COPY-001.** The built-in scalar primitives except `String`, shared
reference values, function values, `Unit`, and `!` are `Copy`. Exclusive
references are not `Copy`.
Arrays and tuples are `Copy` exactly when every component is `Copy`. A
user-defined struct or enum may implement `Copy` only when every possible
field/payload is `Copy`; generic implementations must express and satisfy the
corresponding bounds. `Box`, `Vec`, `String`, maps, sets, and all types
implementing `Drop` are not `Copy`. `Copy` is a marker selected by the normal
coherence rules and cannot be inferred structurally for a user type without
an implementation.

- `Copy` may be implemented for a type only if **all** of its fields are
  `Copy`. Violations are compile-time errors.
- A type MUST NOT implement both `Copy` and `Drop` (a copyable type would run
  its destructor once per copy).
- `Drop::drop` MUST NOT be called explicitly; use the free function
  `drop(value)`. After `drop(v)`, `v` is moved-from.
- **Partial moves**: moving a field out of a struct is permitted only if the
  struct's type does not implement `Drop`. After a partial move, the whole
  value may no longer be used or moved, but its remaining fields may be.
- **Reinitialization**: assigning a new value to a moved-from `let mut`
  variable makes it valid again (definite-assignment tracking).
- Moving an element out of an indexed place (`v[i]`) is a compile-time error;
  use APIs that transfer ownership explicitly (e.g. `Vec::remove`, `Vec::pop`).

The execution and ordering of moves, copies, replacement, `drop(value)`,
partial-value destruction, and trap termination are defined solely by
`CORE-V1-ABSTRACT-MACHINE.md`.

## Error Types and Handling

### Option Type
```stark
enum Option<T> {
    Some(T),
    None
}
```

### Result Type
```stark
enum Result<T, E> {
    Ok(T),
    Err(E)
}
```

### Error Propagation
```stark
fn might_fail() -> Result<Int32, String> {
    // ...
}

fn caller() -> Result<Int32, String> {
    let value = might_fail()?;  // Early return on error
    Ok(value * 2)
}
```

### Try Operator (`?`) Typing
The try operator is defined for `Result<T, E>` and `Option<T>`:
- If `expr` has type `Result<T, E>`, then `expr?` has type `T` and propagates `Err(E)` to the nearest enclosing function returning `Result<_, E>`.
- If `expr` has type `Option<T>`, then `expr?` has type `T` and propagates `None` to the nearest enclosing function returning `Option<_>`.

The enclosing function's return type must be compatible with the propagated type.

## Generics (Core v1)
Core v1 supports parametric polymorphism for functions, structs, enums, and traits.

Rules:
- Generic parameters are introduced with `<T, U, ...>` after the item name.
- Generic parameters are in scope within the item body and signatures.
- All generic parameters used in an item MUST be declared by that item.
- Instantiation occurs at use sites; Core v1 permits monomorphization or dictionary-passing, but the observable behavior MUST be equivalent.
- Generic arguments in expressions are inferred. When inference is impossible
  (no parameter or usable result type mentions the type parameter), explicit
  arguments are supplied with the turbofish form on a path expression:
  `size_of::<Int32>()`. This is the only expression-level generic-argument
  syntax in Core v1.

```stark
struct Pair<T> {
    first: T,
    second: T
}

fn max<T: Ord>(a: T, b: T) -> T {
    if a > b { a } else { b }
}
```

### Trait Bounds
```stark
fn max<T: Ord>(a: T, b: T) -> T { if a > b { a } else { b } }
```

Rules:
- A bound `T: Trait` requires exactly one applicable coherent
  `impl Trait for T` in the resolved package graph; trait visibility controls
  whether its name may be written, not whether an existing obligation can be
  proven.
- Multiple bounds are allowed: `T: TraitA + TraitB`.
- Bounds may bind associated types: `I: Iterator<Item = T>`.

## Trait Coherence (Core v1)

**TRAIT-COHERENCE-001.** A trait implementation is permitted only when the
current canonical package instance defines either the trait or the head
nominal type after transparent-alias expansion. Type parameters, primitives,
tuples, arrays, references, and function types are not local head types.
Different selected package versions have distinct nominal identities; import
aliases and re-exports do not make a trait or type local. Inherent
implementations are permitted only for a nominal type defined by the current
package.

**TRAIT-COHERENCE-002.** Coherence is checked over the complete resolved
package graph, independent of source order and whether an implementation is
used. Two implementations overlap when there exists a substitution that
unifies their trait and self-type heads and whose declared positive bounds
can simultaneously hold. Bounds are not assumed disjoint merely because no
current type is known to satisfy both. Duplicate and overlapping
implementations are rejected.

Selection for a concrete obligation:

1. expand aliases and normalize already-resolved projections;
2. collect all coherent impl heads that unify with the obligation;
3. instantiate each with fresh variables and recursively prove its positive
   bounds;
4. require exactly one successful candidate;
5. substitute its associated types and continue normalization.

Recursive obligation cycles that do not establish a strictly smaller,
previously proven obligation are rejected. Core v1 has no specialization,
negative implementations, trait objects, or declaration-order priority.
Blanket implementations are permitted only when the overlap test proves them
disjoint from every other implementation.

## Standard Trait Laws

**TRAIT-LAW-001.** Implementations used as Core `Eq`, `Ord`, and `Hash` must
obey these semantic laws:

- `Eq::eq` is reflexive, symmetric, and transitive;
- `Ord::cmp` defines a total order, returns `Equal` exactly when `Eq::eq` is
  true, and is antisymmetric and transitive;
- values equal under `Eq` produce equal `Hash::hash` results;
- hashing an unchanged value is stable for the complete execution of one
  program under one fixed target/environment.

The compiler is required to reject only violations it can prove; it is not
required to prove arbitrary method bodies lawful. Direct calls still execute
the user bodies normally. If an implementation violates a law, operations
whose contract relies on that law—including ordered comparison and hash
collections—have unspecified logical results, but memory safety, ownership,
trap behavior, and exactly-once destruction remain guaranteed. A law
violation never grants undefined behavior. C2.9 separately decides which
floating types, if any, implement these traits.

## Constant Evaluation (Core v1)

**CONST-DECL-001.** A `const` declaration is evaluated at compile time and
must produce a value exactly matching or coercible to its declared type.
Array-repeat counts and constant patterns use the same evaluator; array-type
lengths remain the integer literals required by the Core grammar and are
validated at compile time. Constants are immutable values, not addressable storage locations:
each value-context use produces the constant value, and borrowing a constant
materializes an ordinary temporary governed by the C2.7 temporary rules.
Declaration order does not affect visibility or evaluation.

**CONST-SUBSET-001.** Constant expressions are the following closed,
side-effect-free subset:

- literals, unit, and references to other constants;
- grouping; tuples; arrays; array repeats; and struct or enum construction;
- primitive unary, arithmetic, bitwise, shift, logical, comparison, and range
  operations;
- `if` expressions whose condition and selected branch are constant;
- explicit primitive casts;
- blocks containing only constant expressions and a final constant value.

Every operand follows the abstract-machine evaluation order. Transparent
aliases, generic substitutions, and expected types are resolved before
evaluation. Function or method calls, trait dispatch, local bindings,
assignment, borrowing/dereference, indexing requiring runtime storage,
`match`, loops, `return`/`break`/`continue`, `?`, `panic`, heap-backed
collections, I/O, artifact/native-provider access, and runtime state are
prohibited. No user function is implicitly treated as a constant function.

Floating literals are allowed. Other floating constant operations are allowed
only where C2.9 supplies a deterministic operation/cast result; the constant
evaluator must then produce exactly that runtime result and may not use host
extended precision or host-language defaults.

**CONST-FAIL-001.** Constant dependencies form a graph across the resolved
package graph. Any dependency cycle is a compile-time error with the cycle
reported deterministically. A prohibited form, type mismatch, ambiguous
inference, overflow, division/remainder by zero, invalid shift, invalid cast,
bounds failure, or any other operation that would trap at runtime is instead
a compile-time constant-evaluation error at the failing operation. Both
branches are type-checked, but only the selected branch of a constant `if` is
evaluated.

Evaluation is deterministic and memoization may not change diagnostics or
results. Finite implementation resource limits are governed by C2.9; reaching
one must produce its classified diagnostic and must never be cached as a
semantic constant value.

## Numeric Semantics (Core v1)
- Integer overflow and underflow are runtime errors and MUST trap. This applies
  in every build configuration; there is no mode in which overflow wraps
  silently.
- Division or modulo by zero is a runtime error and MUST trap.
- Floating-point operations follow IEEE-754 semantics (NaN, +/-Inf).

## Type Safety Guarantees

1. **No null pointer dereferences**: Option type prevents null access
2. **No buffer overflows**: Array bounds checking
3. **No use-after-free**: Ownership system prevents dangling pointers
4. **No data races**: Borrowing rules prevent concurrent access violations
5. **No memory leaks**: Automatic memory management through ownership

## Informative Future Directions (Non-Normative)

### Lifetime Parameters
```stark
fn longest<'a>(x: &'a str, y: &'a str) -> &'a str {
    if x.len() > y.len() { x } else { y }
}
```

### Trait Objects
Dynamic dispatch via `dyn Trait` is a future extension; `dyn` is a reserved
keyword.

### Capturing Closures
Lambda expressions that capture their environment are a future extension; the
`fn(...)` function types in Core v1 are non-capturing.

## C2.8 Static-Semantics Examples

These examples are parser fixtures and semantic-review cases; C2.11 supplies
the positive/negative executable evidence for each granular rule.

```stark
type UserId = Int32;
struct User { id: UserId }
struct Recursive { next: Option<Box<Recursive>> }
// struct Invalid { next: Invalid } // rejected: infinitely sized
```

```stark
fn identity<T>(value: T) -> T { value }
let inferred: Int64 = identity(1);
```

```stark
fn inspect(value: &String) { println(value.as_str()); }
inspect(&String::from("temporary"));
// let escaped = &String::from("temporary"); // rejected: no lifetime extension
```

```stark
enum Message { Quit, Count(Int32) }
fn code(message: Message) -> Int32 {
    match message {
        Message::Quit => 0,
        Message::Count(value) => value,
    }
}
```

```stark
const WIDTH: UInt64 = 4;
const ENABLED: Bool = WIDTH == 4;
const SETTINGS: (UInt64, Bool) = (WIDTH, if ENABLED { true } else { false });
```

## Conformance
A conforming Core v1 implementation MUST follow the requirements in this document. Any deviations or extensions MUST be explicitly documented by the implementation.
