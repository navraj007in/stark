# 🧠 STARK Type System Specification

This document defines the behavior, mechanics, and philosophy behind STARK's type system. It's designed to blend the safety of static typing, the flexibility of dynamic zones, and the performance of low-level control—without compromising developer joy.

---

## 🔍 Type Inference

STARK uses powerful **static type inference**.

- Most types are inferred unless ambiguity arises.
- Explicit type annotations are optional but supported.
- Inference works across variable declarations, function returns, and lambda expressions.

### Example:

let value = 42          // Inferred: Int32
let name = "Jarvis"     // Inferred: String

🔄 Type Casting & Coercion
STARK distinguishes safe implicit coercion from explicit casting.

Implicit Coercion:
Int8 → Int16 → Int32 → Int64 → Float32 → Float64
Safe widening is automatic.
Explicit Casting:
Use as for narrowing or unsafe conversions.


let b: Float32 = a as Float32
Compiler warnings occur for lossy or unsafe casts unless explicitly acknowledged.

🔐 Mutability Control
All variables and fields are immutable by default.
Use mut to allow changes:

mut let score = 98
score = 100
Structs can have selectively mutable fields for controlled mutation.
❓ Optional and Nullable Types
STARK supports Optional types via T?.

Nullable values must be safely handled using pattern matching or unwrap.
No null dereference allowed at runtime.
Example:

let user: UserProfile?
match user:
    Some(u) => print(u.name)
    None => print("Guest user")
🌀 Union Types
Declare types like Int | String to allow flexible, multi-type variables.
Fully supported in pattern matching for branching logic.

let data: Int | String = "42"
🏷 Type Aliases
Give meaningful names to existing types:


type UserID = Int64
type Score = Float32
Great for domain clarity and abstraction.

🧬 Generics and Parametric Polymorphism
STARK supports generics across functions and composite types.

Example:

fn identity<T>(value: T) -> T:
    return value
No runtime boxing. Generics are compile-time optimized.
⏳ Ownership and Lifetimes (Optional)
For high-performance or low-level control:

Ownership rules can be optionally used.
Lifetimes guide memory safety and deterministic deallocation.
Compiler warns for risky lifetimes or data leaks.
🧩 Pattern Matching on Types
Pattern matching is built into the core type behavior.

Supports destructuring Enum, Tuple, and Union types.
Enables expressive, safe branching logic.
Example:
stark
Copy
match result:
    Ok(value) => print("Done: " + value)
    Err(e) => log("Error: " + e)
📌 Type System Summary
Feature	Status
Static Typing	✅ Strong
Type Inference	✅ Yes
Mutability Control	✅ Yes
Optional/Nullable Types	✅ T? types
Union Types	✅ Native
Generics	✅ Compile-time
Type Aliases	✅ Yes
Pattern Matching Support	✅ Yes
Ownership & Lifetimes	🟡 Optional/Advanced
STARK's type system doesn't just protect your code—it propels your architecture.

Ready for me to wrap this bad boy into a `.md` file for you? One snap of my virtual fingers and it’s yours. Want me to do the honors?





