# 🧱 Composite Types – Structured Data, Elevated Intelligence

While primitives are the atoms of STARK, **composite types** are the molecules that give your programs structure, flexibility, and real-world modeling power. These types allow you to build complex data relationships, reusable structures, and scalable patterns—all while keeping memory layouts tight and predictable.

---

## 📦 Array<T>

A fixed-size, contiguous block of elements of type `T`.

| Feature             | Details                                          |
|---------------------|--------------------------------------------------|
| Memory Layout        | Linear, memory-aligned block                     |
| Access Time          | Constant (`O(1)` indexing)                       |
| Use Case             | ML tensors, buffers, batch data, numerical ops  |

### Example:
```stark
let scores: Array<Float32>[5] = [89.5, 92.3, 76.0, 88.1, 94.6]



🌀 List<T>
A dynamically resizable array, similar to a vector or array list in other languages.

Feature	Details
Memory Layout	Heap-allocated, auto-expanding on demand
Access Time	Amortized O(1) access, O(n) insert/delete (mid)
Use Case	Dynamic datasets, user-generated input, logs
Example:
stark
Copy
let names: List<Char> = List()
names.append('A')
names.append('B')
🧮 Tuple<T1, T2, ..., Tn>
Fixed-size, ordered group of elements of mixed types. Great for returning multiple values or representing structured but ad-hoc groupings.

Feature	Details
Memory Layout	Struct-like, tightly packed
Access Time	Constant (O(1) per index)
Use Case	Function returns, paired data, key-value records
Example:
stark
Copy
let user: Tuple<Int64, Char> = (1001, 'Z')
🔑 Map<K, V>
Key-value pair collection supporting fast lookups.

Feature	Details
Memory Layout	Hash map or tree-backed, depending on usage pattern
Access Time	Average O(1) (hash), O(log n) (tree)
Use Case	Configs, lookup tables, ML label mappings
Example:
stark
Copy
let config: Map<String, Int> = {
    "max_threads": 8,
    "batch_size": 64
}
🧱 Struct
Custom user-defined composite types. Think of it as your domain-specific data model.

Feature	Details
Memory Layout	Compiler-packed, memory-aligned
Access Time	Constant access via field
Use Case	Domain modeling, organized state
Example:
stark
Copy
struct UserProfile:
    id: Int64
    name: String
    score: Float32

let user: UserProfile = UserProfile(id=1001, name="Alice", score=96.5)
🎭 Enum
Tagged union types—define multiple named variants. Ideal for expressing states, variants, or optionals with pattern matching.

Feature	Details
Memory Layout	Variant tag + data payload (tagged union)
Use Case	Option types, state machines, variant data
Example:
stark
Copy
enum Result<T, E>:
    Ok(value: T)
    Err(error: E)

fn divide(a: Float32, b: Float32) -> Result<Float32, String>:
    if b == 0.0:
        return Err("Division by zero")
    return Ok(a / b)
📌 Design Highlights
All composite types are type-safe, immutable by default, and memory-aligned.
Struct and Enum integrate seamlessly with STARK’s pattern matching engine.
List and Map are auto-managed by the runtime allocator, optimized for low-latency access in distributed environments.
🧠 Future Extensions (Planned)
Union Types with Pattern Guards
TaggedRecord types for schema-aware serialization
Immutable Persistent Data Structures (functional-style collections)
Composite types in STARK aren't just data containers — they're smart, scalable abstractions tailored for performance, safety, and cloud-native intelligence.
