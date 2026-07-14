# STARK Memory Model and Ownership Specification

## Overview
STARK's memory model ensures memory safety without garbage collection through compile-time ownership tracking, similar to Rust but with some simplifications for the initial implementation.

## Core Principles

### 1. Ownership
- Every value has exactly one owner
- When the owner goes out of scope, the value is dropped
- Ownership can be transferred (moved) but not duplicated

### 2. Borrowing
- Values can be borrowed without transferring ownership
- Immutable borrows allow reading
- Mutable borrows allow reading and writing
- Borrowing rules prevent data races at compile time

### 3. Lifetimes
- All references have a lifetime
- References cannot outlive the data they point to
- Core v1 has no lifetime annotations; the conservative rules in
  03-Type-System.md ("References and Lifetimes") apply

## Memory Layout

### Stack Allocation
```stark
// Stack-allocated values
let x: Int32 = 42;              // x stored on stack
let y: [Int32; 4] = [1, 2, 3, 4];  // y stored on stack

struct Point { x: Float64, y: Float64 }
let p: Point = Point { x: 1.0, y: 2.0 };  // p stored on stack
```

### Heap Allocation
```stark
// Heap-allocated values
let s: String = String::from("hello");   // String data on heap
let v: Vec<Int32> = Vec::new();          // Vec data on heap
let b: Box<Int32> = Box::new(42);        // Boxed value on heap
```

### Reference Layout
```stark
// References are pointers (8 bytes on 64-bit)
let x: Int32 = 42;
let r: &Int32 = &x;         // r contains address of x

// Slice references contain pointer + length
let arr: [Int32; 5] = [1, 2, 3, 4, 5];
let slice: &[Int32] = &arr[1..4];  // pointer + length (16 bytes)
```

## Ownership Rules

### Rule 1: Single Ownership
```stark
let s1 = String::from("hello");
let s2 = s1;                // Ownership moved from s1 to s2
// println(s2.as_str());    // OK: s2 owns the string
// println(s1.as_str());    // Error: s1 no longer valid
```

### Rule 2: Automatic Cleanup
```stark
{
    let s = String::from("hello");  // s owns the string
    // String is automatically freed when s goes out of scope
}
```

### Rule 3: Function Parameters
```stark
fn take_ownership(s: String) {
    // s is owned by this function
    println(s.as_str());
    // s is dropped when function returns
}

fn main() {
    let s = String::from("hello");
    take_ownership(s);      // Ownership transferred to function
    // println(s.as_str()); // Error: s no longer valid
}
```

## Move Semantics

### What Gets Moved
Types are moved if they:
1. Don't implement the `Copy` trait
2. Contain non-Copy fields

```stark
// Types that are moved by default
String, Vec<T>, Box<T>, custom structs without Copy

// Types that are copied by default (implement Copy)
Int32, Float64, Bool, Char, &T, [T; N] where T: Copy
```

### Move in Assignments
```stark
let s1 = String::from("hello");
let s2 = s1;                // Move
let s3 = s2.clone();        // Explicit copy (if Clone implemented)
```

### Move in Function Calls
```stark
fn process(s: String) { println(s.as_str()); }

let my_string = String::from("hello");
process(my_string);         // my_string moved into function
// my_string no longer accessible
```

### Move in Returns
```stark
fn create_string() -> String {
    let s = String::from("hello");
    s                       // Ownership moved to caller
}
```

### Partial Moves, Reinitialization, and Indexed Places
The normative rules are in 03-Type-System.md ("Copy and Drop"):

```stark
struct Person { name: String, age: Int32 }

let p = Person { name: String::from("Alice"), age: 30 };
let n = p.name;             // Partial move (Person does not implement Drop)
let a = p.age;              // OK: remaining Copy field still readable
// let q = p;               // Error: p is partially moved

let mut s = String::from("one");
let t = s;                  // s moved out
s = String::from("two");    // OK: reinitialization revalidates s

let v: Vec<String> = make();
// let x = v[0];            // Error: cannot move out of indexed place
let x = &v[0];              // OK: borrow instead (or use Vec::remove/pop)
```

## Borrowing System

### Immutable Borrowing
```stark
fn read_string(s: &String) -> UInt64 {
    s.len()                 // Can read but not modify
}

let my_string = String::from("hello");
let length = read_string(&my_string);  // Borrow my_string
println(my_string.as_str()); // my_string still accessible
```

### Mutable Borrowing
```stark
fn modify_string(s: &mut String) {
    s.push('!');            // Can read and modify
}

let mut my_string = String::from("hello");
modify_string(&mut my_string);        // Mutable borrow
println(my_string.as_str()); // Prints "hello!"
```

### Borrowing Rules
1. **Either** one mutable borrow **OR** any number of immutable borrows
2. References must always be valid

```stark
let mut s = String::from("hello");

// Multiple immutable borrows - OK
let r1 = &s;
let r2 = &s;

// Mutable and immutable borrow - Error
let r3 = &s;
let r4 = &mut s;            // Error: cannot borrow mutably while immutably borrowed

// Multiple mutable borrows - Error
let r5 = &mut s;
let r6 = &mut s;            // Error: cannot borrow mutably twice
```

## Lifetime System

### Lifetime Basics
Core v1 applies the normative reference rules from 03-Type-System.md:
returned references must derive from reference parameters; callers treat a
returned reference as having the *shortest* lifetime among the reference
arguments it may derive from; borrows bound to variables are lexically scoped
(to end of block), while unbound temporary borrows end with their statement.
User struct/enum declarations may not declare reference fields, but generic
types instantiated with references (e.g. `Option<&T>`) are permitted as
*borrow-carrying values* subject to the same rules as references — see
"Borrow-Carrying Types" in 03-Type-System.md.

```stark
fn longest(x: &str, y: &str) -> &str {
    if x.len() > y.len() {
        x
    } else {
        y
    }
}
// The returned reference is treated as valid only while both x's and y's
// referents are valid (shortest-input-lifetime rule).
```

### Lifetime Annotations (Future Feature)
```stark
fn longest<'a>(x: &'a str, y: &'a str) -> &'a str {
    if x.len() > y.len() { x } else { y }
}
```

### Dangling Reference Prevention
```stark
fn invalid() -> &Int32 {
    let x = 42;
    &x                      // Error: returning reference to local variable
}

fn valid(x: &Int32) -> &Int32 {
    x                       // OK: returning input reference
}
```

## Memory Management Strategies

### Stack vs Heap Decision
```stark
// Stack allocated (small, known size)
let point = Point { x: 1.0, y: 2.0 };
let array = [1, 2, 3, 4, 5];

// Heap allocated (dynamic size, large objects)
let string = String::from("hello");
let vector: Vec<Int32> = Vec::new();
let boxed = Box::new(large_object);
```

### Reference Counting (Rc/Arc)
```stark
// Single-threaded reference counting (future feature)
let data = Rc::new(v);
let data2 = data.clone();   // Increment reference count

// Multi-threaded reference counting (future feature)
let shared = Arc::new(v);
let shared2 = shared.clone();
```

## Drop and Destructors

### Automatic Drop
```stark
struct FileHandle {
    path: String,
    handle: Int32
}

impl Drop for FileHandle {
    fn drop(&mut self) {
        // Close file handle
        close_file(self.handle);
    }
}

{
    let file = FileHandle { path: String::from("test.txt"), handle: 42 };
    // file.drop() called automatically when leaving scope
}
```

### Drop Order
```stark
{
    let x = String::from("first");
    let y = String::from("second");
    let z = String::from("third");
    // Drop order: z, y, x (reverse declaration order)
}
```

### Manual Drop
```stark
let s = String::from("hello");
drop(s);                    // Explicitly drop s
// println(s.as_str());     // Error: s has been dropped
```

### Drop Soundness (Core v1)
Normative rules (see also "Copy and Drop" in 03-Type-System.md):
- Destructors run **exactly once** per value — at scope exit, at explicit
  `drop(value)`, or when the owner is consumed. Implementations MUST track
  initialization state (drop flags) for values moved on only some paths.
- `Drop::drop` cannot be called explicitly; only the `drop(value)` free
  function is allowed.
- A type cannot implement both `Copy` and `Drop`.
- Fields cannot be moved out of a value whose type implements `Drop`.
- On a runtime error or `panic`, the program aborts immediately and
  destructors are NOT run (see 04-Semantic-Analysis.md, Runtime Error
  Semantics).

## Copy vs Move Types

### Copy Types
Implement `Copy` trait - assignment creates a copy, not a move:
```stark
// Built-in Copy types
Int8, Int16, Int32, Int64, UInt8, UInt16, UInt32, UInt64
Float32, Float64, Bool, Char, Unit
&T (immutable references), [T; N] where T: Copy

// Note: &mut T is NOT Copy (mutable borrows are exclusive and move)

// Usage
let x: Int32 = 42;
let y = x;                  // x is copied, still accessible
// Both x and y valid here
```

### Move Types
Do not implement `Copy` - assignment moves ownership:
```stark
// Built-in Move types
String, Vec<T>, Box<T>, HashMap<K, V>

// Custom types are Move by default
struct Person {
    name: String,
    age: Int32
}

let p1 = Person { name: String::from("Alice"), age: 30 };
let p2 = p1;                // p1 moved to p2
// println(p1.name.as_str());  // Error: p1 no longer valid
```

## Smart Pointers

### Box<T> - Heap Allocation
```stark
let boxed_int = Box::new(42);
let large_array = Box::new([0; 1000]);  // Allocate large array on heap
```

### Rc<T> - Reference Counting (Future)
```stark
let data = Rc::new(some_value);
let reference1 = data.clone();   // Increment reference count
let reference2 = data.clone();   // Increment reference count
// Data deallocated when all references dropped
```

### RefCell<T> - Interior Mutability (Future)
```stark
let data = RefCell::new(42);
{
    let mut borrowed = data.borrow_mut();  // Runtime borrow check
    *borrowed = 43;
}
// data.borrow() now yields 43
```

## Memory Safety Guarantees

### What STARK Prevents
1. **Null pointer dereferences**: No null pointers, use Option<T>
2. **Buffer overflows**: Array bounds checking
3. **Use after free**: Ownership system prevents dangling pointers
4. **Double free**: Ownership ensures single deallocation
5. **Memory leaks**: Automatic cleanup through ownership
6. **Data races**: Borrowing rules prevent concurrent access

### Runtime Checks
Some checks remain at runtime:
- Array bounds checking (unless optimized away)
- Integer overflow (traps in ALL build configurations; see Numeric Semantics
  in 03-Type-System.md)
- RefCell borrow checking (future feature)

## Performance Considerations

### Zero-Cost Abstractions
- Ownership and borrowing have no runtime cost
- References are just pointers
- Move semantics avoid unnecessary copies

### Optimization Opportunities
- Dead code elimination for unused values
- Lifetime optimization to reduce copies
- Stack allocation for values that do not escape

### Memory Layout Optimization
- Struct field reordering to minimize padding
- Enum layout optimization for tagged unions
- Array and slice bounds check elimination

## Implementation Notes

### Compiler Phases
1. **Ownership Analysis**: Track value ownership through program
2. **Borrow Checking**: Validate borrowing rules
3. **Lifetime Inference**: Determine reference lifetimes
4. **Drop Insertion**: Insert drop calls at scope exits
5. **Memory Layout**: Determine stack vs heap allocation

### Error Messages
```
Error: borrow of moved value
  --> example.stark:10:5
   |
 8 |     let s1 = String::from("hello");
   |         -- move occurs because `s1` has type `String`
 9 |     let s2 = s1;
   |              -- value moved here
10 |     println(s1.as_str());
   |             ^^ value borrowed here after move
```

### Integration with Type System
- Ownership information included in type signatures
- Borrowing constraints encoded in function types
- Lifetime parameters for generic functions (future)

## Future Extensions

### Advanced Features
- Lifetime parameters and annotations
- Reference-carrying structs (requires lifetime annotations)
- Higher-ranked trait bounds
- Associated types with lifetime parameters
- Async/await with proper lifetime handling
## Conformance
A conforming Core v1 implementation MUST follow the requirements in this document. Any deviations or extensions MUST be explicitly documented by the implementation.
