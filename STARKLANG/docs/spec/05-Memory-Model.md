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
- Lifetimes are mostly inferred by the compiler

## Memory Layout

### Stack Allocation
```stark
// Stack-allocated values
let x: Int32 = 42           // x stored on stack
let y: [Int32; 4] = [1,2,3,4]  // y stored on stack

struct Point { x: Float64, y: Float64 }
let p: Point = Point { x: 1.0, y: 2.0 }  // p stored on stack
```

### Heap Allocation
```stark
// Heap-allocated values
let s: String = String::new("hello")     // String data on heap
let v: Vec<Int32> = Vec::new()          // Vec data on heap
let b: Box<Int32> = Box::new(42)        // Boxed value on heap
```

### Reference Layout
```stark
// References are pointers (8 bytes on 64-bit)
let x: Int32 = 42
let r: &Int32 = &x          // r contains address of x

// Slices contain pointer + length
let arr: [Int32; 5] = [1,2,3,4,5]
let slice: &[Int32] = &arr[1..4]  // pointer + length (16 bytes)
```

## Ownership Rules

### Rule 1: Single Ownership
```stark
let s1 = String::new("hello")
let s2 = s1                 // Ownership moved from s1 to s2
// println(s1)              // Error: s1 no longer valid
println(s2)                 // OK: s2 owns the string
```

### Rule 2: Automatic Cleanup
```stark
{
    let s = String::new("hello")  // s owns the string
    // String is automatically freed when s goes out of scope
}
```

### Rule 3: Function Parameters
```stark
fn take_ownership(s: String) {
    // s is owned by this function
    println(s)
    // s is dropped when function returns
}

fn main() {
    let s = String::new("hello")
    take_ownership(s)       // Ownership transferred to function
    // println(s)           // Error: s no longer valid
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
let s1 = String::new("hello")
let s2 = s1                 // Move
let s3 = s2.clone()         // Explicit copy (if Clone implemented)
```

### Move in Function Calls
```stark
fn process(s: String) { ... }

let my_string = String::new("hello")
process(my_string)          // my_string moved into function
// my_string no longer accessible
```

### Move in Returns
```stark
fn create_string() -> String {
    let s = String::new("hello")
    s                       // Ownership moved to caller
}
```

## Borrowing System

### Immutable Borrowing
```stark
fn read_string(s: &String) -> Int32 {
    s.len()                 // Can read but not modify
}

let my_string = String::new("hello")
let length = read_string(&my_string)  // Borrow my_string
println(my_string)          // my_string still accessible
```

### Mutable Borrowing
```stark
fn modify_string(s: &mut String) {
    s.push('!')             // Can read and modify
}

let mut my_string = String::new("hello")
modify_string(&mut my_string)         // Mutable borrow
println(my_string)          // Prints "hello!"
```

### Borrowing Rules
1. **Either** one mutable borrow **OR** any number of immutable borrows
2. References must always be valid

```stark
let mut s = String::new("hello")

// Multiple immutable borrows - OK
let r1 = &s
let r2 = &s
println("{} {}", r1, r2)

// Mutable and immutable borrow - Error
let r3 = &s
let r4 = &mut s             // Error: cannot borrow mutably while immutably borrowed

// Multiple mutable borrows - Error
let r5 = &mut s
let r6 = &mut s             // Error: cannot borrow mutably twice
```

## Lifetime System

### Lifetime Basics
```stark
fn longest(x: &str, y: &str) -> &str {
    if x.len() > y.len() {
        x
    } else {
        y
    }
}
// Compiler infers that return value has same lifetime as input parameters
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
    let x = 42
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
let point = Point { x: 1.0, y: 2.0 }
let array = [1, 2, 3, 4, 5]

// Heap allocated (dynamic size, large objects)
let string = String::new("hello")
let vector = Vec::new()
let boxed = Box::new(large_object)
```

### Reference Counting (Rc/Arc)
```stark
// Single-threaded reference counting (future feature)
let data = Rc::new(vec![1, 2, 3])
let data2 = data.clone()    // Increment reference count

// Multi-threaded reference counting (future feature)
let shared = Arc::new(vec![1, 2, 3])
let shared2 = shared.clone()
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
        close_file(self.handle)
    }
}

{
    let file = FileHandle { path: "test.txt".to_string(), handle: 42 }
    // file.drop() called automatically when leaving scope
}
```

### Drop Order
```stark
{
    let x = String::new("first")
    let y = String::new("second")
    let z = String::new("third")
    // Drop order: z, y, x (reverse declaration order)
}
```

### Manual Drop
```stark
let s = String::new("hello")
drop(s)                     // Explicitly drop s
// println(s)               // Error: s has been dropped
```

## Copy vs Move Types

### Copy Types
Implement `Copy` trait - assignment creates a copy, not a move:
```stark
// Built-in Copy types
Int8, Int16, Int32, Int64, UInt8, UInt16, UInt32, UInt64
Float32, Float64, Bool, Char, Unit
&T (references), [T; N] where T: Copy

// Usage
let x: Int32 = 42
let y = x                   // x is copied, still accessible
println("{} {}", x, y)      // OK: both x and y valid
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

let p1 = Person { name: "Alice".to_string(), age: 30 }
let p2 = p1                 // p1 moved to p2
// println(p1.name)         // Error: p1 no longer valid
```

## Smart Pointers

### Box<T> - Heap Allocation
```stark
let boxed_int = Box::new(42)
let large_array = Box::new([0; 1000])  // Allocate large array on heap
```

### Rc<T> - Reference Counting (Future)
```stark
let data = Rc::new(vec![1, 2, 3])
let reference1 = data.clone()    // Increment reference count
let reference2 = data.clone()    // Increment reference count
// Data deallocated when all references dropped
```

### RefCell<T> - Interior Mutability (Future)
```stark
let data = RefCell::new(42)
{
    let mut borrowed = data.borrow_mut()  // Runtime borrow check
    *borrowed = 43
}
println(data.borrow())      // Prints 43
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
- Integer overflow (in debug mode)
- RefCell borrow checking (future feature)

## Performance Considerations

### Zero-Cost Abstractions
- Ownership and borrowing have no runtime cost
- References are just pointers
- Move semantics avoid unnecessary copies

### Optimization Opportunities
- Dead code elimination for unused values
- Lifetime optimization to reduce copies
- Stack allocation for escaped values when possible

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
 8 |     let s1 = String::new("hello");
   |         -- move occurs because `s1` has type `String`
 9 |     let s2 = s1;
   |              -- value moved here
10 |     println(s1);
   |     ^^ value borrowed here after move
```

### Integration with Type System
- Ownership information included in type signatures
- Borrowing constraints encoded in function types
- Lifetime parameters for generic functions (future)

## Future Extensions

### Advanced Features
- Lifetime parameters and annotations
- Higher-ranked trait bounds
- Associated types with lifetime parameters
- Async/await with proper lifetime handling
## Conformance
A conforming Core v1 implementation MUST follow the requirements in this document. Any deviations or extensions MUST be explicitly documented by the implementation.
