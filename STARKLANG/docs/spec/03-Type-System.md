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
let s: &str = "hello"
let owned: String = String::from(s)
```

## Composite Types

### Array Types
```stark
[T; N]   // Fixed-size array of N elements of type T
[T]      // Dynamic array (slice) of elements of type T
```

Examples:
```stark
let fixed: [Int32; 5] = [1, 2, 3, 4, 5]
let dynamic: [Int32] = [1, 2, 3]
```

### Tuple Types
```stark
()           // Empty tuple (Unit type)
(T,)         // Single-element tuple
(T1, T2)     // Two-element tuple
(T1, T2, T3) // Three-element tuple
// ... up to reasonable limit (e.g., 16 elements)
```

Examples:
```stark
let empty: () = ()
let single: (Int32,) = (42,)
let pair: (Int32, String) = (42, "hello")
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
```stark
fn(T1, T2) -> R    // Function taking T1, T2 and returning R
fn() -> R          // Function taking no parameters, returning R
fn(T)              // Function taking T, returning Unit
```

## Type Aliases
```stark
type Age = Int32
type Point2D = (Float64, Float64)
type ErrorCode = Int32
```

## Ownership and Borrowing

### Ownership Rules
1. Each value has exactly one owner
2. When the owner goes out of scope, the value is dropped
3. Values can be moved (ownership transfer) or borrowed (temporary access)

### Move Semantics
```stark
let a = String::new("hello")
let b = a  // a is moved to b, a is no longer valid
// print(a)  // Error: use of moved value
```

### Borrowing Rules
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

let mut text = String::new("hello")
borrow_immutable(&text)        // OK
borrow_mutable(&mut text)      // OK
// borrow_immutable(&text)     // Error: cannot borrow as immutable while mutable borrow exists
```

## Type Inference

### Local Type Inference
```stark
let x = 42          // Inferred as Int32
let y = 3.14        // Inferred as Float64
let z = [1, 2, 3]   // Inferred as [Int32]
```

### Function Return Type Inference
```stark
fn add(a: Int32, b: Int32) {  // Return type inferred as Int32
    a + b
}
```

### Generic Type Inference
```stark
fn identity<T>(x: T) -> T {
    x
}

let result = identity(42)  // T inferred as Int32
```

## Subtyping and Coercion

### Numeric Coercions
No implicit numeric conversions. Explicit casting required:
```stark
let x: Int32 = 42
let y: Int64 = x as Int64  // Explicit cast required
```

### Reference Coercions
```stark
&mut T -> &T        // Mutable reference to immutable reference
&T -> &T            // Same type (identity)
```

### Array to Slice Coercion
```stark
&[T; N] -> &[T]     // Array reference to slice reference
&mut [T; N] -> &mut [T]  // Mutable array reference to mutable slice reference
```

## Trait System (Basic)

### Trait Definition
```stark
trait Display {
    fn fmt(&self) -> String
}

trait Eq {
    fn eq(&self, other: &Self) -> Bool
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

## Type Checking Rules

### Assignment Compatibility
```stark
let x: T = expr  // expr must have type T or be coercible to T
```

### Function Call Compatibility
```stark
fn f(param: T) { ... }
f(arg)  // arg must have type T or be coercible to T
```

### Arithmetic Operations
```stark
// Binary arithmetic requires same types
let result = x + y  // x and y must have the same numeric type

// Comparison operators
let cmp = x < y     // x and y must have the same comparable type
```

### Logical Operations
```stark
let result = a && b  // a and b must be Bool
let result = !x      // x must be Bool
```

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
    let value = might_fail()?  // Early return on error
    Ok(value * 2)
}
```

### Try Operator (`?`) Typing
The try operator is defined for `Result<T, E>` and `Option<T>`:
- If `expr` has type `Result<T, E>`, then `expr?` has type `T` and propagates `Err(E)` to the nearest enclosing function returning `Result<_, E>`.
- If `expr` has type `Option<T>`, then `expr?` has type `T` and propagates `None` to the nearest enclosing function returning `Option<_>`.

The enclosing function's return type must be compatible with the propagated type.

## Type System Extensions (Future)

### Generics
```stark
struct Vec<T> {
    data: [T],
    len: Int32,
    cap: Int32
}

fn max<T: Ord>(a: T, b: T) -> T {
    if a > b { a } else { b }
}
```

### Associated Types
```stark
trait Iterator {
    type Item
    fn next(&mut self) -> Option<Self::Item>
}
```

### Lifetime Parameters
```stark
fn longest<'a>(x: &'a str, y: &'a str) -> &'a str {
    if x.len() > y.len() { x } else { y }
}
```

## Type Safety Guarantees

1. **No null pointer dereferences**: Option type prevents null access
2. **No buffer overflows**: Array bounds checking
3. **No use-after-free**: Ownership system prevents dangling pointers
4. **No data races**: Borrowing rules prevent concurrent access violations
5. **No memory leaks**: Automatic memory management through ownership

## Implementation Notes

### Type Representation
- Primitive types: Direct machine representation
- Composite types: Laid out according to platform ABI
- References: Pointers with compile-time tracking
- Enums: Tagged unions with optimal layout

### Type Checking Algorithm
1. Parse source into AST
2. Build symbol table with declarations
3. Perform type inference using Hindley-Milner algorithm
4. Check type constraints and ownership rules
5. Generate type-annotated AST for code generation
## Conformance
A conforming Core v1 implementation MUST follow the requirements in this document. Any deviations or extensions MUST be explicitly documented by the implementation.
