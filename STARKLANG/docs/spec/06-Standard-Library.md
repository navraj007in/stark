# STARK Standard Library Specification

## Overview
The STARK standard library provides essential types, functions, and modules for core programming tasks. This specification defines the minimal standard library for the initial implementation.

## Notation
The code blocks in this document are **API notation**, not compilable STARK
source: function signatures are listed without bodies, and `impl` blocks mix
signatures with prose comments. The *signatures and behaviors* are normative;
the listing style is not. (The Core v1 grammar has no body-less function form
outside `trait` blocks.)

## Implementation-Provided Types
Several standard library types (`Box<T>`, `Vec<T>`, `String`, `HashMap<K, V>`)
manage raw memory internally. Raw pointers and `unsafe` code are NOT part of
the Core v1 language surface; these types are *implementation-provided*: their
public APIs are normative, their internal representation is not expressible in
Core v1 source and is implementation-defined. Struct bodies shown for these
types in this document are illustrative only.

## Iterator and View Types
The iterator types returned by the APIs below (`VecIter<T>`, `KeysIter<K>`,
`ValuesIter<V>`, `Iter<K, V>`, `Iter<T>`, `CharsIter`, `SplitIter`,
`MapIter<I, U>`, `FilterIter<I>`) are implementation-provided opaque structs.
Each implements `Iterator` with the obvious `Item`:

| Type | Produced by | `Item` |
| --- | --- | --- |
| `VecIter<T>` | `Vec::iter` | `&T` |
| `KeysIter<K>` | `HashMap::keys` | `&K` |
| `ValuesIter<V>` | `HashMap::values` | `&V` |
| `Iter<K, V>` | `HashMap::iter` | `(&K, &V)` |
| `Iter<T>` | `HashSet::iter` | `&T` |
| `CharsIter` | `String::chars`, `str::chars` | `Char` |
| `SplitIter` | `String::split` | `&str` |
| `MapIter<I, U>` | `Iterator::map` | `U` |
| `FilterIter<I>` | `Iterator::filter` | `I::Item` |

Iterators whose `Item` is a reference (and `CharsIter`/`SplitIter`, which
borrow their source) are **borrow-carrying types** in the sense of
03-Type-System.md: they hold a live borrow of the collection they iterate and
obey all reference rules (generic wrapper storage propagates the borrow,
regions are lexical, and no wrapper may outlive its source).

## Core Module Structure
```
std/
├── core/           // Core language items
├── collections/    // Data structures
├── io/            // Input/output operations
├── string/        // String manipulation
├── math/          // Mathematical operations
├── mem/           // Memory management utilities
├── error/         // Error handling types
└── prelude/       // Automatically imported items
```

## Prelude Module
Automatically imported into every STARK program:

```stark
// Basic types (no import needed)
Int8, Int16, Int32, Int64, UInt8, UInt16, UInt32, UInt64
Float32, Float64, Bool, Char, String, str, Unit

// Essential traits
trait Copy
trait Clone
trait Drop
trait Eq
trait Ord
trait Hash
trait Default
trait Display

// Essential enums
enum Ordering {
    Less,
    Equal,
    Greater
}

enum Option<T> {
    Some(T),
    None
}

enum Result<T, E> {
    Ok(T),
    Err(E)
}

// Essential functions
fn print(value: &str)
fn println(value: &str)
fn panic(message: &str) -> !    // Never returns; see 03-Type-System.md (Never Type)
```

### Core Trait Profile

**STD-TRAIT-001.** The required Core trait identities are `Copy`, `Clone`,
`Drop`, `Eq`, `Ord`, `Hash`, `Default`, `Display`, `Iterator`, `Index`,
`IndexMut`, and compiler-owned `Num`, with the required items shown in this
document. An implementation claiming the Core profile must provide these
canonical items and the primitive/standard-type implementations explicitly
required here. Additional traits or implementations are extensions and may
not change Core resolution or coherence. The semantic laws of `Eq`, `Ord`,
and `Hash` are defined by `TRAIT-LAW-001`; float participation remains owned
by C2.9.

### Canonical Language Hooks

**STD-HOOK-001.** A language hook is recognized by canonical standard-item
identity, never by an unqualified spelling or source declaration order. The
complete Core v1 hook set is:

| Hook | Language use |
| --- | --- |
| `Copy` | implicit place reads |
| `Drop` and `drop` | deterministic destruction and explicit consumption |
| `panic` | language trap |
| `Option` and `Result` | `?` propagation |
| `Iterator` | `for` protocol |
| `Index` and `IndexMut` | indexing places/values |
| `Eq` and `Ord` | generic equality and ordering operators |
| `Num` | generic primitive numeric operators |
| `size_of` and `align_of` | target-layout queries |

All other APIs—including `Clone`, `Default`, `Display`, `Hash`, collection
methods, string methods, math, and I/O—use ordinary name resolution, method
selection, trait dispatch, or implementation-provided bodies. `HashMap` may
require `Hash`, but that does not make `Hash` a syntax hook. User declarations
with a hook's spelling never acquire hook behavior. C2.9 completes the target
results of `size_of`/`align_of` and canonical package identity.

## Core Module (std::core)

### Essential Trait Definitions
```stark
trait Clone {
    fn clone(&self) -> Self;
}

trait Eq {
    fn eq(&self, other: &Self) -> Bool;
}

trait Ord {
    fn cmp(&self, other: &Self) -> Ordering;
}

trait Hash {
    fn hash(&self) -> UInt64;
}

trait Default {
    fn default() -> Self;
}

trait Display {
    fn fmt(&self) -> String;
}

trait Drop {
    fn drop(&mut self);
}

// Copy is a marker trait (no methods); Copy: Clone.
// Soundness rules (all-fields-Copy, Copy/Drop exclusivity) are in
// 03-Type-System.md, "Copy and Drop".
trait Copy

// Num is a compiler-known marker trait implemented by exactly the built-in
// numeric types; it enables arithmetic operators on generic parameters
// (see 03-Type-System.md, "Operators and Traits"). Not user-implementable.
trait Num
```

### Indexing Traits
The `[]` operator desugars to these traits:

```stark
trait Index<Idx> {
    type Output;
    fn index(&self, index: Idx) -> &Self::Output;
}

trait IndexMut<Idx> {
    fn index_mut(&mut self, index: Idx) -> &mut Self::Output;
}
```

### Range Types
The range operators `..` and `..=` produce these types:

```stark
struct Range<T> {
    start: T,
    end: T
}

struct RangeInclusive<T> {
    start: T,
    end: T
}

// Ranges over integer types are iterable
impl Iterator for Range<Int32> {
    type Item = Int32;
    fn next(&mut self) -> Option<Int32>;
}
// ... similarly for the other integer types, and for RangeInclusive
```

### Memory Management
```stark
// Box - heap allocation (implementation-provided; internals opaque)
struct Box<T> { /* implementation-defined */ }

impl<T> Box<T> {
    fn new(value: T) -> Box<T>;
    fn into_inner(self) -> T;
}

// Manual memory management
fn drop<T>(value: T)
fn size_of<T>() -> UInt64      // T not inferable: call as size_of::<Int32>()
fn align_of<T>() -> UInt64     // T not inferable: call as align_of::<Int32>()
```

### Option Type
```stark
enum Option<T> {
    Some(T),
    None
}

impl<T> Option<T> {
    fn is_some(&self) -> Bool;
    fn is_none(&self) -> Bool;
    fn unwrap(self) -> T;
    fn unwrap_or(self, default: T) -> T;
    fn map<U>(self, f: fn(T) -> U) -> Option<U>;
    fn and_then<U>(self, f: fn(T) -> Option<U>) -> Option<U>;
}
```

### Result Type
```stark
enum Result<T, E> {
    Ok(T),
    Err(E)
}

impl<T, E> Result<T, E> {
    fn is_ok(&self) -> Bool;
    fn is_err(&self) -> Bool;
    fn unwrap(self) -> T;
    fn unwrap_or(self, default: T) -> T;
    fn map<U>(self, f: fn(T) -> U) -> Result<U, E>;
    fn map_err<F>(self, f: fn(E) -> F) -> Result<T, F>;
    fn and_then<U>(self, f: fn(T) -> Result<U, E>) -> Result<U, E>;
}
```

## Collections Module (std::collections)

### Vec<T> - Dynamic Array
```stark
// Implementation-provided; internals opaque
struct Vec<T> { /* implementation-defined */ }

impl<T> Vec<T> {
    fn new() -> Vec<T>;
    fn with_capacity(capacity: UInt64) -> Vec<T>;
    fn push(&mut self, item: T);
    fn pop(&mut self) -> Option<T>;
    fn len(&self) -> UInt64;
    fn capacity(&self) -> UInt64;
    fn is_empty(&self) -> Bool;
    fn get(&self, index: UInt64) -> Option<&T>;
    fn get_mut(&mut self, index: UInt64) -> Option<&mut T>;
    fn insert(&mut self, index: UInt64, item: T);
    fn remove(&mut self, index: UInt64) -> T;
    fn clear(&mut self);
    fn append(&mut self, other: &mut Vec<T>);
    fn extend<I: Iterator<Item = T>>(&mut self, iter: I);
    fn iter(&self) -> VecIter<T>;
    fn as_slice(&self) -> &[T];
}

// Index access
impl<T> Index<UInt64> for Vec<T> {
    type Output = T;
    fn index(&self, index: UInt64) -> &T;
}

impl<T> IndexMut<UInt64> for Vec<T> {
    fn index_mut(&mut self, index: UInt64) -> &mut T;
}
```

`Vec::insert` accepts `0..=len`, shifting later elements right, and traps
above `len`; `remove` accepts `0..len`, shifts later elements left, and traps
otherwise. `append` moves all elements in order and leaves `other` empty; the
borrow rules reject passing the same vector as both receiver and argument.
`with_capacity(n)` creates an empty collection with capacity at least `n`;
capacity may grow but never affects equality or iteration. Because this
constructor has no `Result`, inability to reserve is a classified
host/resource failure, not a language trap.

### HashMap<K, V> - Hash Table
```stark
// Implementation-provided; internals opaque
struct HashMap<K, V> { /* implementation-defined */ }

impl<K: Hash + Eq, V> HashMap<K, V> {
    fn new() -> HashMap<K, V>;
    fn with_capacity(capacity: UInt64) -> HashMap<K, V>;
    fn insert(&mut self, key: K, value: V) -> Option<V>;
    fn get(&self, key: &K) -> Option<&V>;
    fn get_mut(&mut self, key: &K) -> Option<&mut V>;
    fn remove(&mut self, key: &K) -> Option<V>;
    fn contains_key(&self, key: &K) -> Bool;
    fn len(&self) -> UInt64;
    fn is_empty(&self) -> Bool;
    fn clear(&mut self);
    fn keys(&self) -> KeysIter<K>;
    fn values(&self) -> ValuesIter<V>;
    fn iter(&self) -> Iter<K, V>;
}
```

### HashSet<T> - Hash Set
```stark
// Implementation-provided; internals opaque
struct HashSet<T> { /* implementation-defined */ }

impl<T: Hash + Eq> HashSet<T> {
    fn new() -> HashSet<T>;
    fn insert(&mut self, value: T) -> Bool;
    fn remove(&mut self, value: &T) -> Bool;
    fn contains(&self, value: &T) -> Bool;
    fn len(&self) -> UInt64;
    fn is_empty(&self) -> Bool;
    fn clear(&mut self);
    fn iter(&self) -> Iter<T>;
}
```

Collection `with_capacity(n)` requests room for at least `n` entries but
exposes no exact bucket count, load factor, seed, or growth schedule. Failure
to reserve is a classified host/resource failure. Capacity and collision
strategy never affect equality, insertion order, returned values, or Drop.

### Iteration Order (Core v1)
`HashMap::keys`/`values`/`iter` and `HashSet::iter` (and any `for` loop over a
`HashMap`/`HashSet`) MUST visit entries in **first-insertion order**:
`insert`ing a key for the first time appends it to the iteration order;
`insert`ing an already-present key updates its value without changing its
position; `remove`ing a key and later re-`insert`ing it places it at the end
(as a new insertion). This requires no bound beyond `Hash + Eq` on the key
type (unlike a key-sorted-order rule, which would additionally require
`K: Ord` — not part of `HashMap`'s/`HashSet`'s bounds). This is normative and
deterministic: two conforming implementations must produce identical
iteration order for the same sequence of insertions/removals, regardless of
internal storage strategy (see "Performance Notes" below, which describes
storage, not iteration order).

**STD-HASH-001.** `HashMap` and `HashSet` determine key identity exclusively
with lawful `Eq` and use `Hash::hash` only to select candidate buckets.
Collisions must be resolved by `Eq`; unequal keys with equal hashes remain
distinct. Replacing an equal key retains the first stored key and its
insertion position. Their observable iteration order is the first-insertion
order above and is independent of hash values, collision strategy, capacity,
target, and process. Primitive/standard-type hash implementations must be
the 64-bit FNV-1a result with offset basis `14695981039346656037` and prime
`1099511628211` over this canonical byte encoding:

- integers use their fixed-width little-endian two's-complement/unsigned bits;
- `Bool` is byte 0 or 1; `Char` is its `UInt32` scalar encoding; `Unit` is
  empty; `String`/`str` are their UTF-8 bytes;
- arrays, tuples, `Vec`, `Option`, and `Result` use domain tags `0x01`,
  `0x02`, `0x03`, `0x04`, and `0x05` respectively; sequences then encode
  element count as `UInt64` little-endian; `Option` encodes `None` as variant
  byte 0 and `Some` as 1, while `Result` encodes `Ok` as variant byte 0 and
  `Err` as 1; every present component is framed by its encoded byte length as
  `UInt64` little-endian followed by those bytes.

Floats do not implement `Hash`; maps, sets, resources, references, and
borrowed views have no standard `Hash` implementation. These rules make a
direct standard `Hash::hash` result backend- and target-independent. User
implementations return their body result normally; law violations have the
behavior in `TRAIT-LAW-001`.

## String Module (std::string)

### String Type
```stark
// Implementation-provided; internals opaque (UTF-8 byte buffer)
struct String { /* implementation-defined */ }

impl String {
    fn new() -> String;               // Empty string
    fn with_capacity(capacity: UInt64) -> String;
    fn from(s: &str) -> String;       // Construct from a string slice/literal
    fn len(&self) -> UInt64;
    fn is_empty(&self) -> Bool;
    fn push(&mut self, ch: Char);
    fn push_str(&mut self, s: &str);
    fn pop(&mut self) -> Option<Char>;
    fn clear(&mut self);
    fn chars(&self) -> CharsIter;
    fn bytes(&self) -> &[UInt8];
    fn as_str(&self) -> &str;
    fn into_bytes(self) -> Vec<UInt8>;
    fn substring(&self, start: UInt64, end: UInt64) -> &str;
    fn contains(&self, pattern: &str) -> Bool;
    fn starts_with(&self, pattern: &str) -> Bool;
    fn ends_with(&self, pattern: &str) -> Bool;
    fn find(&self, pattern: &str) -> Option<UInt64>;
    fn replace(&self, from: &str, to: &str) -> String;
    fn split(&self, delimiter: &str) -> SplitIter;
    fn trim(&self) -> &str;
    fn to_lowercase(&self) -> String;
    fn to_uppercase(&self) -> String;
}

// String literals (&str)
impl str {
    fn len(&self) -> UInt64;
    fn is_empty(&self) -> Bool;
    fn chars(&self) -> CharsIter;
    fn bytes(&self) -> &[UInt8];
    fn to_string(&self) -> String;
    // ... similar methods to String
}
```

**TEXT-UTF8-001.** Every `String` and `str` value contains valid UTF-8.
Literals, file reads, conversions, mutation, concatenation, and native
providers must validate before a value becomes observable. Invalid external
bytes produce the API's error result; no operation creates a partially valid
string or silently substitutes U+FFFD.

**TEXT-INDEX-001.** String offsets and lengths are unsigned UTF-8 byte offsets.
`len`, `find`, `substring`, and split/view boundaries use bytes. Core provides
no `string[index]` element operator. `bytes` yields encoded bytes in order;
`chars` is the scalar-value API.

**TEXT-BOUNDARY-001.** A byte offset used as a string boundary must be at zero,
at the byte length, or at the first byte of a UTF-8 scalar. `substring` traps
when either offset is not a scalar boundary, either is out of range, or start
exceeds end. Search/split operations return only valid boundaries.

**TEXT-ITER-001.** `chars` yields Unicode scalar values in source order;
`bytes` yields each UTF-8 byte in source order. Neither normalizes text or
combines grapheme clusters. `pop` removes and returns the last scalar, and
`push` appends the scalar's UTF-8 encoding.

**TEXT-CASE-001.** Character classification and `to_lowercase`/
`to_uppercase` use Unicode 15.1 Default Case Conversion, independent of
locale. A mapping may expand one scalar to several and preserves their
specified order. Core performs no normalization before or after mapping.

`replace(from,to)` scans left-to-right and replaces non-overlapping matches;
an empty `from` inserts `to` at every scalar boundary including both ends.
`split(delimiter)` preserves trailing empty components for a nonempty
delimiter. An empty delimiter yields one component per Unicode scalar and no
leading or trailing empty component; splitting an empty string yields no
components. `trim` removes the Unicode 15.1 `White_Space` property from both
ends only. These operations never normalize or use locale-sensitive matching.

## Math Module (std::math)

### Basic Operations
```stark
// Constants
const PI: Float64 = 3.141592653589793;
const E: Float64 = 2.718281828459045;

// Basic functions
fn abs<T: Num>(x: T) -> T
fn min<T: Ord>(a: T, b: T) -> T
fn max<T: Ord>(a: T, b: T) -> T
fn clamp<T: Ord>(value: T, min: T, max: T) -> T

// Floating point functions
fn sqrt(x: Float64) -> Float64
fn pow(base: Float64, exp: Float64) -> Float64
fn log(x: Float64) -> Float64
fn log10(x: Float64) -> Float64
fn exp(x: Float64) -> Float64

// Trigonometric functions
fn sin(x: Float64) -> Float64
fn cos(x: Float64) -> Float64
fn tan(x: Float64) -> Float64
fn asin(x: Float64) -> Float64
fn acos(x: Float64) -> Float64
fn atan(x: Float64) -> Float64
fn atan2(y: Float64, x: Float64) -> Float64

// Rounding functions
fn floor(x: Float64) -> Float64
fn ceil(x: Float64) -> Float64
fn round(x: Float64) -> Float64
fn trunc(x: Float64) -> Float64

// Random numbers (simple linear congruential generator)
struct Random {
    seed: UInt64
}

impl Random {
    fn new(seed: UInt64) -> Random;
    fn next_int(&mut self) -> UInt64;
    fn next_float(&mut self) -> Float64;
    fn range(&mut self, min: Int32, max: Int32) -> Int32;
}
```

**STD-MATH-001.** Integer `abs` follows checked integer arithmetic; floating
`abs` clears the sign bit. `min`, `max`, and `clamp` dispatch through lawful
`Ord` and evaluate arguments once. `floor`, `ceil`, `round` (nearest integer,
ties away from zero), and `trunc` produce the exact IEEE result and preserve
an already-integral signed zero. Transcendental functions return the IEEE
special-case result and a finite result within one ulp of the exact real
result (`pow` within two ulp); their last bit may be target-defined. Domain
errors produce NaN rather than a language trap.

**STD-RANDOM-001.** `Random` is the reproducible 64-bit LCG
`state = state * 6364136223846793005 + 1442695040888963407 (mod 2^64)`.
`new(seed)` installs `seed`; `next_int` advances once and returns the state.
`next_float` advances once and returns the correctly rounded `Float64` value
`state / 2^64` in `[0,1)`. `range(min,max)` requires `min < max`, advances
once, and returns `min + state % (max-min)`, hence `min` is inclusive and
`max` exclusive; an invalid range traps without advancing.

## IO Module (std::io)

### Basic IO Operations
```stark
// Standard streams
fn print(text: &str)
fn println(text: &str)
fn eprint(text: &str)     // stderr
fn eprintln(text: &str)   // stderr

// Simple file operations
struct File { /* implementation-defined */ }

impl File {
    fn open(path: &str) -> Result<File, IOError>;
    fn create(path: &str) -> Result<File, IOError>;
    fn read_to_string(&mut self) -> Result<String, IOError>;
    fn write(&mut self, data: &[UInt8]) -> Result<UInt64, IOError>;
    fn write_str(&mut self, text: &str) -> Result<UInt64, IOError>;
    fn close(self) -> Result<Unit, IOError>;
}

// Error types
enum IOError {
    NotFound,
    PermissionDenied,
    AlreadyExists,
    InvalidInput,
    Other(String)
}

// Utility functions
fn read_file(path: &str) -> Result<String, IOError>
fn write_file(path: &str, content: &str) -> Result<Unit, IOError>
```

**STD-IO-001.** The `std-full` profile provides the APIs above and a
first-class, non-`Copy` `File` resource whose ownership may be moved but not
cloned. Ordinary open/create/read/write/close failures return `IOError`;
`NotFound`, `PermissionDenied`, `AlreadyExists`, and `InvalidInput` are used
when the host exposes that distinction, otherwise `Other` carries stable
human-readable context. Reads validate UTF-8. A successful write reports the
number of bytes accepted; callers must handle a short write. Dropping an open
file attempts close but cannot surface a new language trap.

**STD-FORMAT-001.** `Display::fmt` returns valid UTF-8 and is ordinary trait
dispatch. `print`/`eprint` append exactly the argument bytes; `println`/
`eprintln` append those bytes followed by byte `0x0A`, independent of host
newline convention. Successful calls preserve program order. The process
contract flushes submitted stdout/stderr before reporting normal return or a
language trap; a stream write/flush failure is a host/process failure.

Canonical standard `Display` implementations are byte-exact:

- signed/unsigned integers are base-10 ASCII with no leading zeroes and a
  minus sign only for negative values;
- `Bool` is `true` or `false`; `Char`, `String`, and `str` emit their UTF-8
  content; `Unit` is `()`; `Ordering` is `Less`, `Equal`, or `Greater`;
- finite floats use the fewest significant decimal digits that parse back to
  the same declared IEEE value; among equal-length candidates choose the
  numerically closest, then the one with an even final digit. Fixed form is
  used for decimal exponents in `[-4,15]`, otherwise scientific form with one
  leading digit, lowercase `e`, no exponent `+` or leading zeroes, and at
  least `.0` or an exponent. Negative zero is `-0.0`; infinities are
  `inf`/`-inf`; every NaN is `NaN`;
- `IOError` is `NotFound`, `PermissionDenied`, `AlreadyExists`,
  `InvalidInput`, or `Other(<message>)`; `GenericError` emits its message.

No standard formatting is locale-sensitive or contains padding, grouping,
color, debug syntax, or a trailing newline.

## Error Module (std::error)

### Error Trait
```stark
trait Error {
    fn message(&self) -> String;
}

// Standard error types
struct GenericError {
    message: String
}

impl Error for GenericError {
    fn message(&self) -> String {
        self.message.clone()
    }
}
```

Note: error *chaining* (a `source()` method returning a trait object) requires
`dyn Trait` support, which is a future extension. Core v1 errors carry a
message only; richer error types can embed context in their own fields.

## Memory Module (std::mem)

### Memory Utilities
```stark
// Memory operations
fn size_of<T>() -> UInt64      // Call with turbofish: size_of::<Int32>()
fn align_of<T>() -> UInt64     // Call with turbofish: align_of::<Int32>()
fn swap<T>(a: &mut T, b: &mut T)
fn replace<T>(dest: &mut T, src: T) -> T
fn take<T: Default>(dest: &mut T) -> T
```

Raw-pointer copy operations (`copy`, `copy_nonoverlapping`) require raw
pointers and `unsafe`, which are not part of Core v1; they are deferred to a
future extension.

## Iterator Trait (std::iter)

### Basic Iterator Interface
```stark
trait Iterator {
    type Item;
    
    fn next(&mut self) -> Option<Self::Item>;
    
    // Default implementations
    fn count(self) -> UInt64;
    fn collect<C: FromIterator<Self::Item>>(self) -> C;
    fn map<U>(self, f: fn(Self::Item) -> U) -> MapIter<Self, U>;
    fn filter(self, predicate: fn(&Self::Item) -> Bool) -> FilterIter<Self>;
    fn fold<B>(self, init: B, f: fn(B, Self::Item) -> B) -> B;
    fn reduce(self, f: fn(Self::Item, Self::Item) -> Self::Item) -> Option<Self::Item>;
    fn any(self, predicate: fn(Self::Item) -> Bool) -> Bool;
    fn all(self, predicate: fn(Self::Item) -> Bool) -> Bool;
    fn find(self, predicate: fn(&Self::Item) -> Bool) -> Option<Self::Item>;
}

trait FromIterator<T> {
    fn from_iter<I: Iterator<Item = T>>(iter: I) -> Self;
}
```

Limitation (Core v1): the combinator parameters above are *non-capturing*
function types (`fn(...)`). They accept named functions and function
references but cannot capture local variables. Capturing closures are a
future extension; until then, prefer explicit loops when state must be
carried into the operation.

## Conversion Traits

### Basic Conversion
```stark
trait From<T> {
    fn from(value: T) -> Self;
}

trait Into<T> {
    fn into(self) -> T;
}

trait TryFrom<T> {
    type Error;
    fn try_from(value: T) -> Result<Self, Self::Error>;
}

trait TryInto<T> {
    type Error;
    fn try_into(self) -> Result<T, Self::Error>;
}

// String conversion
trait ToString {
    fn to_string(&self) -> String;
}

trait FromStr {
    type Error;
    fn from_str(s: &str) -> Result<Self, Self::Error>;
}
```

**STD-CONVERT-001.** Numeric `From` exists only for conversions that preserve
every source value. Potentially failing numeric conversions use `TryFrom` and
the exact `NUM-CAST-001` value/range rules. `FromStr` accepts the corresponding
Core literal body without a type suffix or surrounding whitespace and returns
`Err` for malformed, non-finite textual forms, overflow, or underflow; it
never silently clamps or wraps. Float underflow means a nonzero mathematical
input whose correctly rounded result would be zero; subnormal nonzero results
are accepted. `as` remains the only trapping numeric conversion syntax.

## Essential Trait Implementations

### Default Implementations
```stark
// Copy trait for basic types
impl Copy for Int8 { }
impl Copy for Int16 { }
impl Copy for Int32 { }
impl Copy for Int64 { }
impl Copy for UInt8 { }
impl Copy for UInt16 { }
impl Copy for UInt32 { }
impl Copy for UInt64 { }
impl Copy for Float32 { }
impl Copy for Float64 { }
impl Copy for Bool { }
impl Copy for Char { }
impl Copy for Unit { }

// Clone for all Copy types (blanket implementation)
impl<T: Copy> Clone for T {
    fn clone(&self) -> T { *self }
}

// Eq trait for basic types
impl Eq for Int32 {
    fn eq(&self, other: &Int32) -> Bool { *self == *other }
}

// Ord trait for basic types
impl Ord for Int32 {
    fn cmp(&self, other: &Int32) -> Ordering {
        if *self < *other { Ordering::Less }
        else if *self > *other { Ordering::Greater }
        else { Ordering::Equal }
    }
}
```

### Primitive Trait and Operator Matrix

**PRIM-TRAIT-001.** The impls shown above are illustrative of *form*. The
table below is normative for *which* primitive types implement `Eq`, `Ord`
and `Hash`, and which operators they admit. It replaces the earlier
"similar for other types" wording, which was not a specification.

| Primitive | `Eq` | `Ord` | `Hash` | `==` `!=` | `<` `<=` `>` `>=` |
| --- | --- | --- | --- | --- | --- |
| `Int8`…`Int64`, `UInt8`…`UInt64` | yes | yes | yes | yes | yes |
| `Char` | yes | yes | yes | yes | yes |
| `String`, `str` | yes | yes | yes | yes | yes |
| `Bool` | yes | **no** | yes | yes | **no** |
| `Float32`, `Float64` | **no** | **no** | **no** | yes | yes |
| `Unit` | no | no | no | no | no |

Three rows carry deliberate asymmetries:

- **`Char` is ordered by Unicode scalar value.** `'a' < 'b'` compares scalar
  values; `Char::cmp` returns the corresponding `Ordering`. This is scalar
  order, **not** locale-sensitive or linguistic collation, and Core v1 offers
  no collation facility.
- **`Bool` is `Eq` and `Hash` but not `Ord`.** `true < false`, the other
  ordered operators, and `Bool::cmp` are compile-time errors. An ordering
  could be defined, but Core v1 has no use for ordering truth values, and
  rejecting the operator is clearer than fixing an arbitrary order.
- **Floats admit the comparison operators but implement no trait.** `<` and
  `==` on `Float64` are built-in IEEE 754 operations (`NUM-FLOAT-*`), and
  remain available. The traits are withheld because IEEE comparison is not an
  equivalence relation or a total order — NaN is unordered and unequal to
  itself — so `Float64` cannot satisfy a `T: Eq` or `T: Ord` bound, and cannot
  be a `HashMap` key.

Operators on primitives have built-in meaning and do not dispatch through
these traits (see 03-Type-System, "Operators and Traits"). The trait columns
therefore govern *generic bounds* — what `T: Ord` accepts — while the operator
columns govern direct use. The two differ only for the float row.

## Conformance Profiles
Two standard-library conformance profiles are defined. A conforming
implementation MUST state which profile it implements.

**STD-PROFILE-001.** `core-min` is required for every Core v1 implementation.
`std-full` is an optional, indivisible advertised capability: claiming it
requires every listed API plus every behavior explicitly stated in this
chapter, including file I/O and `Random`. C2.9 freezes language-relevant
contracts and the API-availability profile; it does not imply semantics for
an edge case this chapter does not state. Such an unstated result is
implementation-defined and cannot be used as cross-backend conformance
evidence until a later standard-library specification assigns it. A missing
host facility prevents the `std-full` claim rather than changing an API's
meaning. Extensions may add profiles but may not call them Core v1 or weaken
these profiles.

### Profile: `core-min` (MVP)
The minimum standard library for Core v1 conformance:
- Prelude: primitive types, `Option`, `Result`, `Ordering`, essential traits
  (`Copy`, `Clone`, `Drop`, `Eq`, `Ord`, `Hash`, `Default`, `Display`,
  `Iterator`, `Index`, `IndexMut`, `Num`), `print`, `println`, `panic`
- `String` and `str` (construction, `len`, `push`/`push_str`, `as_str`,
  `chars`)
- `Vec<T>` (construction, `push`, `pop`, `len`, `get`/`get_mut`, indexing)
- `Range`/`RangeInclusive` with integer `Iterator` impls (for `for` loops)
- `Box<T>`
- `drop`, `size_of`, `align_of`

### Profile: `std-full`
Everything in this document: `core-min` plus `HashMap`, `HashSet`, the
`Iterator` trait with combinators and all iterator/view types, the full
`String` API, the math module, file IO, `std::mem` utilities, the conversion
traits, and the full required-trait implementations for standard types.

Items in *neither* profile (informative future work, not required for any
Core v1 conformance claim): buffered IO, regular expressions, time/date,
threads and concurrency primitives, `Rc`/`Arc`/`RefCell`.

## Informative Platform Considerations (Non-Normative)

### Cross-platform Abstractions
- File path handling
- Directory operations
- Environment variables
- Process spawning (future)

## Behavioral Requirements (Core v1)
- Indexing `Vec<T>` with `[]` MUST perform bounds checking and MUST trap on out-of-bounds access.
- `get`/`get_mut` MUST return `None` for out-of-bounds indices and MUST NOT trap.
- Slicing an array, slice, or `Vec<T>` with a range MUST trap if the range is out of bounds or inverted.
- `String::substring(start, end)` MUST validate UTF-8 boundaries and MUST trap on invalid boundaries or ranges.
- IO functions MUST return `Result` with a non-`Ok` variant on failure and MUST NOT silently ignore errors.
## Conformance
A conforming Core v1 implementation MUST implement at least the `core-min`
profile, MUST state which profile it implements, and MUST follow the
requirements in this document for every item it provides. Any deviations or
extensions MUST be explicitly documented by the implementation.
