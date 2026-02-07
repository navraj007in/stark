# STARK Standard Library Specification

## Overview
The STARK standard library provides essential types, functions, and modules for core programming tasks. This specification defines the minimal standard library for the initial implementation.

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

// Essential types
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
fn panic(message: &str) -> !
```

## Core Module (std::core)

### Memory Management
```stark
// Box - heap allocation
struct Box<T> {
    ptr: *mut T
}

impl<T> Box<T> {
    fn new(value: T) -> Box<T>
    fn into_inner(self) -> T
}

// Manual memory management
fn drop<T>(value: T)
fn size_of<T>() -> UInt64
fn align_of<T>() -> UInt64
```

### Option Type
```stark
enum Option<T> {
    Some(T),
    None
}

impl<T> Option<T> {
    fn is_some(&self) -> Bool
    fn is_none(&self) -> Bool
    fn unwrap(self) -> T
    fn unwrap_or(self, default: T) -> T
    fn map<U>(self, f: fn(T) -> U) -> Option<U>
    fn and_then<U>(self, f: fn(T) -> Option<U>) -> Option<U>
}
```

### Result Type
```stark
enum Result<T, E> {
    Ok(T),
    Err(E)
}

impl<T, E> Result<T, E> {
    fn is_ok(&self) -> Bool
    fn is_err(&self) -> Bool
    fn unwrap(self) -> T
    fn unwrap_or(self, default: T) -> T
    fn map<U>(self, f: fn(T) -> U) -> Result<U, E>
    fn map_err<F>(self, f: fn(E) -> F) -> Result<T, F>
    fn and_then<U>(self, f: fn(T) -> Result<U, E>) -> Result<U, E>
}
```

## Collections Module (std::collections)

### Vec<T> - Dynamic Array
```stark
struct Vec<T> {
    ptr: *mut T,
    len: UInt64,
    cap: UInt64
}

impl<T> Vec<T> {
    fn new() -> Vec<T>
    fn with_capacity(capacity: UInt64) -> Vec<T>
    fn push(&mut self, item: T)
    fn pop(&mut self) -> Option<T>
    fn len(&self) -> UInt64
    fn capacity(&self) -> UInt64
    fn is_empty(&self) -> Bool
    fn get(&self, index: UInt64) -> Option<&T>
    fn get_mut(&mut self, index: UInt64) -> Option<&mut T>
    fn insert(&mut self, index: UInt64, item: T)
    fn remove(&mut self, index: UInt64) -> T
    fn clear(&mut self)
    fn append(&mut self, other: &mut Vec<T>)
    fn extend(&mut self, iter: impl Iterator<Item = T>)
}

// Index access
impl<T> Index<UInt64> for Vec<T> {
    type Output = T
    fn index(&self, index: UInt64) -> &T
}

impl<T> IndexMut<UInt64> for Vec<T> {
    fn index_mut(&mut self, index: UInt64) -> &mut T
}
```

### HashMap<K, V> - Hash Table
```stark
struct HashMap<K, V> {
    // Internal implementation details
}

impl<K: Hash + Eq, V> HashMap<K, V> {
    fn new() -> HashMap<K, V>
    fn with_capacity(capacity: UInt64) -> HashMap<K, V>
    fn insert(&mut self, key: K, value: V) -> Option<V>
    fn get(&self, key: &K) -> Option<&V>
    fn get_mut(&mut self, key: &K) -> Option<&mut V>
    fn remove(&mut self, key: &K) -> Option<V>
    fn contains_key(&self, key: &K) -> Bool
    fn len(&self) -> UInt64
    fn is_empty(&self) -> Bool
    fn clear(&mut self)
    fn keys(&self) -> KeysIter<K>
    fn values(&self) -> ValuesIter<V>
    fn iter(&self) -> Iter<K, V>
}
```

### HashSet<T> - Hash Set
```stark
struct HashSet<T> {
    map: HashMap<T, Unit>
}

impl<T: Hash + Eq> HashSet<T> {
    fn new() -> HashSet<T>
    fn insert(&mut self, value: T) -> Bool
    fn remove(&mut self, value: &T) -> Bool
    fn contains(&self, value: &T) -> Bool
    fn len(&self) -> UInt64
    fn is_empty(&self) -> Bool
    fn clear(&mut self)
    fn iter(&self) -> Iter<T>
}
```

## String Module (std::string)

### String Type
```stark
struct String {
    bytes: Vec<UInt8>
}

impl String {
    fn new() -> String
    fn with_capacity(capacity: UInt64) -> String
    fn from(s: &str) -> String
    fn len(&self) -> UInt64
    fn is_empty(&self) -> Bool
    fn push(&mut self, ch: Char)
    fn push_str(&mut self, s: &str)
    fn pop(&mut self) -> Option<Char>
    fn clear(&mut self)
    fn chars(&self) -> CharsIter
    fn bytes(&self) -> &[UInt8]
    fn as_str(&self) -> &str
    fn into_bytes(self) -> Vec<UInt8>
    fn substring(&self, start: UInt64, end: UInt64) -> &str
    fn contains(&self, pattern: &str) -> Bool
    fn starts_with(&self, pattern: &str) -> Bool
    fn ends_with(&self, pattern: &str) -> Bool
    fn find(&self, pattern: &str) -> Option<UInt64>
    fn replace(&self, from: &str, to: &str) -> String
    fn split(&self, delimiter: &str) -> SplitIter
    fn trim(&self) -> &str
    fn to_lowercase(&self) -> String
    fn to_uppercase(&self) -> String
}

// String literals (&str)
impl str {
    fn len(&self) -> UInt64
    fn is_empty(&self) -> Bool
    fn chars(&self) -> CharsIter
    fn bytes(&self) -> &[UInt8]
    fn to_string(&self) -> String
    // ... similar methods to String
}
```

## Math Module (std::math)

### Basic Operations
```stark
// Constants
const PI: Float64 = 3.141592653589793
const E: Float64 = 2.718281828459045

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
    fn new(seed: UInt64) -> Random
    fn next_int(&mut self) -> UInt64
    fn next_float(&mut self) -> Float64
    fn range(&mut self, min: Int32, max: Int32) -> Int32
}
```

## IO Module (std::io)

### Basic IO Operations
```stark
// Standard streams
fn print(text: &str)
fn println(text: &str)
fn eprint(text: &str)     // stderr
fn eprintln(text: &str)   // stderr

// Simple file operations
struct File {
    handle: Int32
}

impl File {
    fn open(path: &str) -> Result<File, IOError>
    fn create(path: &str) -> Result<File, IOError>
    fn read_to_string(&mut self) -> Result<String, IOError>
    fn write(&mut self, data: &[UInt8]) -> Result<UInt64, IOError>
    fn write_str(&mut self, text: &str) -> Result<UInt64, IOError>
    fn close(self) -> Result<Unit, IOError>
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

## Error Module (std::error)

### Error Trait
```stark
trait Error {
    fn message(&self) -> String
    fn source(&self) -> Option<&dyn Error>
}

// Standard error types
struct GenericError {
    message: String
}

impl Error for GenericError {
    fn message(&self) -> String {
        self.message.clone()
    }
    
    fn source(&self) -> Option<&dyn Error> {
        None
    }
}
```

## Memory Module (std::mem)

### Memory Utilities
```stark
// Memory operations
fn size_of<T>() -> UInt64
fn align_of<T>() -> UInt64
fn swap<T>(a: &mut T, b: &mut T)
fn replace<T>(dest: &mut T, src: T) -> T
fn take<T: Default>(dest: &mut T) -> T

// Unsafe memory operations (future)
fn copy<T>(src: *const T, dst: *mut T, count: UInt64)
fn copy_nonoverlapping<T>(src: *const T, dst: *mut T, count: UInt64)
```

## Iterator Trait (std::iter)

### Basic Iterator Interface
```stark
trait Iterator {
    type Item
    
    fn next(&mut self) -> Option<Self::Item>
    
    // Default implementations
    fn count(self) -> UInt64
    fn collect<C: FromIterator<Self::Item>>(self) -> C
    fn map<U>(self, f: fn(Self::Item) -> U) -> MapIter<Self, U>
    fn filter(self, predicate: fn(&Self::Item) -> Bool) -> FilterIter<Self>
    fn fold<B>(self, init: B, f: fn(B, Self::Item) -> B) -> B
    fn reduce(self, f: fn(Self::Item, Self::Item) -> Self::Item) -> Option<Self::Item>
    fn any(self, predicate: fn(Self::Item) -> Bool) -> Bool
    fn all(self, predicate: fn(Self::Item) -> Bool) -> Bool
    fn find(self, predicate: fn(&Self::Item) -> Bool) -> Option<Self::Item>
}

trait FromIterator<T> {
    fn from_iter<I: Iterator<Item = T>>(iter: I) -> Self
}
```

## Conversion Traits

### Basic Conversion
```stark
trait From<T> {
    fn from(value: T) -> Self
}

trait Into<T> {
    fn into(self) -> T
}

trait TryFrom<T> {
    type Error
    fn try_from(value: T) -> Result<Self, Self::Error>
}

trait TryInto<T> {
    type Error
    fn try_into(self) -> Result<T, Self::Error>
}

// String conversion
trait ToString {
    fn to_string(&self) -> String
}

trait FromStr {
    type Error
    fn from_str(s: &str) -> Result<Self, Self::Error>
}
```

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

// Clone trait for all types
impl<T: Copy> Clone for T {
    fn clone(&self) -> T { *self }
}

// Eq trait for basic types
impl Eq for Int32 {
    fn eq(&self, other: &Int32) -> Bool { *self == *other }
}
// ... similar for other types

// Ord trait for basic types
impl Ord for Int32 {
    fn cmp(&self, other: &Int32) -> Ordering {
        if *self < *other { Ordering::Less }
        else if *self > *other { Ordering::Greater }
        else { Ordering::Equal }
    }
}
```

## Implementation Priorities

### Phase 1 (MVP)
- Basic types (primitives, String, Option, Result)
- Vec<T> dynamic array
- Basic IO (print, println, simple file operations)
- Essential traits (Copy, Clone, Eq, Ord)

### Phase 2 (Enhanced)
- HashMap<K, V> and HashSet<T>
- Iterator trait and basic iterators
- Math module with common functions
- Enhanced string operations

### Phase 3 (Complete)
- Advanced memory utilities
- Full IO system with buffering
- Regular expressions
- Time and date handling
- Thread and concurrency primitives (future)

## Platform Considerations

### Cross-platform Abstractions
- File path handling
- Directory operations
- Environment variables
- Process spawning (future)

### Performance Notes
- Vec<T> uses exponential growth strategy
- HashMap<T> uses open addressing with Robin Hood hashing
- String operations are UTF-8 aware
- Iterator chains compile to efficient loops
## Conformance
A conforming Core v1 implementation MUST follow the requirements in this document. Any deviations or extensions MUST be explicitly documented by the implementation.
