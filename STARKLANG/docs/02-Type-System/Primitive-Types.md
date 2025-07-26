# 🧱 Primitive Types – Low-Level Efficiency, High-Level Expressiveness

STARK's primitive types are the building blocks of all computation. They are designed for **predictable memory layout**, **cross-platform compatibility**, and **performance-optimized execution**—especially in AI/ML and cloud-native environments.

Below is the complete list of supported primitive types in STARK, along with their characteristics and intended use cases.

---

## 🔢 Numeric Types

| Type     | Description                   | Size     | Processor Support        | Use Case                              |
|----------|-------------------------------|----------|--------------------------|---------------------------------------|
| `Int8`   | Signed 8-bit integer          | 1 byte   | x86, ARM, RISC-V         | Low-level flags, bit manipulation     |
| `Int16`  | Signed 16-bit integer         | 2 bytes  | x86, ARM                 | Embedded systems, I/O buffers         |
| `Int32`  | Signed 32-bit integer         | 4 bytes  | x86, ARM, SIMD           | General-purpose arithmetic            |
| `Int64`  | Signed 64-bit integer         | 8 bytes  | x86-64, ARM64, SIMD      | Large counters, timestamps, big data  |
| `UInt8`  | Unsigned 8-bit integer        | 1 byte   | x86, ARM, SIMD           | Byte-level data, color values         |

---

## 🔬 Floating Point Types

| Type       | Description                           | Size     | Processor Support          | Use Case                                |
|------------|---------------------------------------|----------|----------------------------|-----------------------------------------|
| `Float32`  | 32-bit IEEE 754 floating point         | 4 bytes  | FPU, SIMD, AI accelerators | ML operations, matrix math              |
| `Float64`  | 64-bit IEEE 754 floating point         | 8 bytes  | FPU, AI compute cores      | Scientific computing, precise modeling  |

---

## 🎛 Boolean and Character Types

| Type   | Description             | Size     | Processor Support | Use Case                        |
|--------|-------------------------|----------|--------------------|---------------------------------|
| `Bool` | Boolean (true or false) | 1 byte   | Universal           | Control flow, logic branches   |
| `Char` | Unicode scalar value    | 4 bytes  | Universal           | Text processing, string parsing|

---

## ✨ Design Principles

- **Memory-Aligned Layouts:** All types are laid out in memory in an alignment-optimized fashion to take advantage of vectorization and caching.
- **Processor-Aware Compilation:** STARKVM compiles and optimizes bytecode instructions based on target hardware (x86, ARM, RISC-V, etc.).
- **Predictable Size:** Unlike some languages that play loose with sizes, STARK guarantees **fixed-size types**—crucial for ML workloads and hardware interfacing.

---

## 🔥 Future Enhancements (Planned)
- Support for **BF16** and **Float16** for AI model compression.
- Native **Quantized Types** (`QInt8`, `QInt32`) for efficient edge inference.
- **Vector Types** like `Vec4<Float32>` for GPU-native operations.

---

## 📌 Example: Type Declaration

```stark
let score: Float32 = 98.6
let flags: UInt8 = 0b11010010
let char: Char = 'Ω'

Primitive types in STARK aren’t just simple data—they’re foundational weapons of high-performance, high-scalability computing.