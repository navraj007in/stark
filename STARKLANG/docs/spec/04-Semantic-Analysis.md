# STARK Semantic Analysis Specification

## Overview
Semantic analysis validates that programs are meaningful according to STARK's language rules. This phase occurs after parsing and before code generation.

## Analysis Phases

### 1. Symbol Table Construction
Build symbol tables for scopes and declarations.

#### Scope Rules
```stark
// Module scope
fn module_function() { }
const MODULE_CONST: Int32 = 42;

fn example() {
    // Function scope
    let local_var = 10;
    
    {
        // Block scope
        let block_var = 20;
        // block_var accessible here
        // local_var accessible here
    }
    // block_var not accessible here
    // local_var still accessible
}
```

#### Name Resolution Order
Name resolution distinguishes *lexical* scopes (inside function bodies) from
*module* scopes.

Within a function body, an unqualified name is resolved lexically:
1. Current block scope
2. Enclosing block scopes (innermost to outermost)
3. Function parameters

If not found lexically, the name is resolved at module level per the module
system rules (see 07-Modules-and-Packages.md):
4. Items declared in the current module
5. Items brought into scope by `use`
6. Built-in names (prelude)

Unqualified names never implicitly search parent modules or the crate root;
those require explicit `super::` or `crate::` paths.

#### Shadowing Rules
```stark
let x = 10;       // x: Int32
{
    let x = "hi"; // Shadows outer x, x: String
    // Inner x takes precedence
}
// Outer x visible again
```

### 2. Type Checking

#### Variable Declarations
```stark
let x: Int32 = 42;     // Type annotation matches literal
let y = 42;            // Type inferred as Int32
let z: String = 42;    // Error: type mismatch
```

#### Function Calls
```stark
fn add(a: Int32, b: Int32) -> Int32 { a + b }

let result = add(1, 2);    // OK: arguments match parameters
let error = add(1.0, 2);   // Error: Float64 cannot be Int32
let error2 = add(1);       // Error: wrong number of arguments
```

#### Assignment Compatibility
```stark
let mut x: Int32 = 10;
x = 20;                    // OK: same type
x = "hello";               // Error: type mismatch

let y: Int32 = 10;
y = 20;                    // Error: y is not mutable
```

### 3. Ownership and Borrowing Analysis

#### Move Semantics Validation
```stark
let s1 = String::from("hello");
let s2 = s1;              // s1 moved to s2
print(s1);                // Error: use of moved value

fn take_ownership(s: String) { }
let s3 = String::from("world");
take_ownership(s3);       // s3 moved into function
print(s3);                // Error: use of moved value
```

#### Borrow Checking
```stark
let mut s = String::from("hello");
let r1 = &s;              // Immutable borrow
let r2 = &s;              // OK: multiple immutable borrows
let r3 = &mut s;          // Error: cannot borrow mutably while immutably borrowed

let mut s2 = String::from("world");
let r4 = &mut s2;         // Mutable borrow
let r5 = &s2;             // Error: cannot borrow immutably while mutably borrowed
let r6 = &mut s2;         // Error: cannot have multiple mutable borrows
```

#### Lifetime Validation
Returned references must obey the Core v1 reference rules defined in
03-Type-System.md ("References and Lifetimes"): a returned reference must
derive from a reference parameter on every path, and callers treat it as
having the shortest lifetime among the reference arguments it may derive from.

```stark
fn invalid_return() -> &Int32 {
    let x = 42;
    &x                    // Error: returning reference to local variable
}

fn valid_return(x: &Int32) -> &Int32 {
    x                     // OK: returning input reference
}
```

### 4. Control Flow Analysis

#### Unreachable Code Detection
```stark
fn example() -> Int32 {
    return 42;
    let x = 10;           // Warning: unreachable code
}
```

#### Return Path Analysis
```stark
fn missing_return() -> Int32 {
    let x = 42;
    // Error: not all paths return a value
}

fn conditional_return(flag: Bool) -> Int32 {
    if flag {
        return 1;
    }
    // Error: missing return in else branch
}

fn valid_return(flag: Bool) -> Int32 {
    if flag {
        1
    } else {
        2
    }                     // OK: both branches return
}
```

#### Break/Continue Validation
```stark
fn invalid_break() {
    break;                // Error: break outside of loop
}

fn valid_break() {
    loop {
        break;            // OK: break inside loop
    }
}
```

### 5. Pattern Matching Analysis

#### Exhaustiveness Checking
```stark
enum Color { Red, Green, Blue }

fn check_color(c: Color) -> String {
    match c {
        Color::Red => "red",
        Color::Green => "green"
        // Error: non-exhaustive pattern (missing Blue)
    }
}

fn valid_match(c: Color) -> String {
    match c {
        Color::Red => "red",
        Color::Green => "green",
        Color::Blue => "blue"     // OK: exhaustive
    }
}
```

Rules:
- A `match` over an enum type is exhaustive if every variant is covered, or a wildcard (`_`) arm exists.
- Tuple patterns are exhaustive if each element position is exhaustive for its type.
- Literal patterns are exhaustive only for finite domains (e.g., `Bool` with `true` and `false`).
- If a match is not exhaustive, it is a compile-time error.

#### Pattern Type Checking
```stark
let x: Int32 = 42;
match x {
    "hello" => "string",          // Error: pattern type mismatch
    42 => "number",               // OK
    _ => "other"
}
```

### 6. Mutability Analysis

#### Mutable Access Validation
Assignment to an initialized immutable variable is an error. (Exception: the
single initializing assignment of a deferred-initialization `let` — see
Initialization Analysis.)

```stark
let x = 42;
x = 43;                   // Error: x is not mutable

let mut y = 42;
y = 43;                   // OK: y is mutable

struct Point { x: Int32, y: Int32 }
let p = Point { x: 1, y: 2 };
p.x = 10;                 // Error: p is not mutable

let mut p2 = Point { x: 1, y: 2 };
p2.x = 10;                // OK: p2 is mutable
```

### 7. Initialization Analysis

#### Use Before Initialization
```stark
let x: Int32;
print(x.fmt());           // Error: use of uninitialized variable

let y: Int32 = if true { 42 } else { 24 };
print(y.fmt());           // OK: y is initialized in all branches

let z: Int32;
if condition {
    z = 42;
}
print(z.fmt());           // Error: z might not be initialized
```

Rules:
- `let name: Type;` declares a variable without initializing it.
- A variable must be definitely assigned before any read.
- All control-flow paths must assign before use.
- **Deferred initialization of immutable variables**: an *uninitialized*
  immutable `let` may be assigned exactly once on each control-flow path;
  this first assignment is initialization, not mutation. Any assignment to an
  already-initialized immutable variable is an error (see Mutability
  Analysis).

#### Double Initialization
```stark
let mut x: Int32 = 42;
x = 43;                   // OK: reassignment
let x = 44;               // Error: redeclaration in same scope
```

### 8. Array Bounds Analysis

#### Static Bounds Checking
```stark
let arr: [Int32; 3] = [1, 2, 3];
let x = arr[2];           // OK: index in bounds
let y = arr[5];           // Error: index out of bounds (if determinable)
```

#### Dynamic Bounds Checking
```stark
let arr: [Int32; 3] = [1, 2, 3];
let idx = get_index();
let x = arr[idx];         // Runtime bounds check required
```

### 9. Error Propagation Analysis

#### Question Mark Operator
```stark
fn might_fail() -> Result<Int32, String> { Ok(42) }

fn caller1() -> Result<Int32, String> {
    let x = might_fail()?;    // OK: compatible error types
    Ok(x * 2)
}

fn caller2() -> Int32 {
    let x = might_fail()?;    // Error: function doesn't return Result
    x * 2
}
```

Rules:
- `expr?` propagates `Err` or `None` to the nearest enclosing function returning `Result<_, E>` or `Option<_>`.
- The enclosing function return type must be compatible with the propagated type.

## Runtime Error Semantics (Core v1)
Static analysis must identify operations whose normative runtime rule can trap and preserve the
source location of that operation. Trap categories, abort behavior, and the absence of
unwinding are defined solely by `CORE-V1-ABSTRACT-MACHINE.md` (`TRAP-CATEGORY-001` and
`DROP-ABORT-001`). Numeric and process-specific classifications are completed by C2.9.

### 10. Trait Constraint Checking

#### Trait Bounds Validation
```stark
trait Display {
    fn fmt(&self) -> String;
}

fn print_it<T: Display>(item: T) {
    print(item.fmt());        // OK: T implements Display
}

struct Point { x: Int32, y: Int32 }

print_it(Point { x: 1, y: 2 });  // Error: Point doesn't implement Display
```

## Error Reporting

### Error Message Format
```
Error: [ERROR_CODE] [BRIEF_DESCRIPTION]
  --> file.stark:line:column
   |
line | source code line
   | ^^^^^ specific location
   |
   = help: detailed explanation
   = note: additional information
```

### Error Categories

#### Type Errors (E0001-E0099)
- E0001: Type mismatch
- E0002: Unknown type
- E0003: Type annotation required
- E0004: Cannot infer type
- E0005: Wrong number of arguments
- E0006: `?` operator in a function that does not return `Result` or `Option`
- E0007: Index out of bounds (determinable at compile time)
- E0008: Integer literal out of range for its type (suffixed literal exceeds its suffix's
  representable range, or an unsuffixed literal exceeds `Int64`)
- E0009: Array repeat count (`[value; count]`) is not a compile-time constant expression

#### Ownership Errors (E0100-E0199)
- E0100: Use of moved value
- E0101: Borrow check failed
- E0102: Lifetime violation
- E0103: Dangling reference

#### Name Resolution Errors (E0200-E0299)
- E0200: Undefined variable
- E0201: Undefined function
- E0202: Undefined type
- E0203: Ambiguous name
- E0204: Duplicate definition in the same scope

#### Control Flow Errors (E0300-E0399)
- E0301: Missing return value
- E0302: Break outside loop
- E0303: Non-exhaustive match

(E0300 is intentionally unassigned: unreachable code is warning W0005, not an
error — see "Unreachable Code Detection" above.)

#### Mutability and Initialization Errors (E0400-E0499)
- E0400: Assignment to immutable binding
- E0401: Use of possibly-uninitialized variable

#### Trait Errors (E0500-E0599)
- E0500: Trait not implemented

### Warning Categories

#### Unused Items (W0001-W0099)
- W0001: Unused variable
- W0002: Unused function
- W0003: Unused import
- W0004: Dead code
- W0005: Unreachable code

#### Style Warnings (W0100-W0199)
- W0100: Non-snake_case variable
- W0101: Non-PascalCase type
- W0102: Missing documentation

## Conformance
A conforming Core v1 implementation MUST follow the requirements in this document. Any deviations or extensions MUST be explicitly documented by the implementation.
