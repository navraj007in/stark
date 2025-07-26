# 🤖 STARK Language — Core AI Types Specification

This document defines the fundamental AI/ML data structures in STARKLANG. These are not extensions but core language primitives, designed to make AI deployment efficient, safe, and expressive.

---

## 📦 Core AI Types

### 1️⃣ Tensor<T>[...Shape] - Core Primitive

The fundamental data structure for all AI computations in STARK.

```stark
let features: Tensor<Float32>[batch, 128]
let weights: Tensor<Float32>[128, 10] 
let image: Tensor<UInt8>[3, 224, 224]
```

#### Compile-Time Shape Safety:
- Shape mismatches caught at compile time
- Automatic shape inference in operations
- Zero-cost abstractions for tensor operations

#### Core Methods:
```stark
tensor.shape() -> [Int; N]           // Compile-time known shape
tensor.matmul(other: Tensor<T>[M, K]) -> Tensor<T>[N, K]
tensor.map(fn: T -> U) -> Tensor<U>[...Shape]
tensor.reduce_sum(axis: Int) -> Tensor<T>[...ReducedShape]
```

#### Memory Layout:
- Contiguous memory with configurable layouts (row-major, column-major)
- GPU memory support with automatic transfers
- Zero-copy views and slicing

---

### 2️⃣ Dataset<T> - Streaming Data Pipeline

Efficient, lazy-evaluated data processing for AI workflows.

```stark
let train_data: Dataset<(Tensor<Float32>[128], Int32)>
let images: Dataset<Tensor<UInt8>[3, 224, 224]>
```

#### Pipeline Operations:
```stark
dataset
  .batch(32)                                    // Create batches
  .map(|x| preprocess(x))                      // Transform data
  .shuffle(1000)                               // Shuffle with buffer
  .prefetch(4)                                 // Async prefetching
```

#### Memory Efficiency:
- Lazy evaluation - only process when consumed
- Streaming from disk/network without loading everything
- Automatic memory management for large datasets
- GPU pipeline support for zero-copy transfers

---

### 3️⃣ Model<Input, Output> - Inference Engine

Type-safe model abstraction with shape validation and performance optimization.

```stark
let classifier: Model<Tensor<Float32>[batch, 128], Tensor<Float32>[batch, 10]>
let embedder: Model<Vec<String>, Tensor<Float32>[batch, 384]>
```

#### Model Loading:
```stark
let model = load_pytorch_model("classifier.pt")  // Automatic shape inference
let onnx_model = load_onnx_model("model.onnx")
let huggingface_model = load_hf_model("bert-base-uncased")
```

#### Inference Methods:
```stark
model.predict(input: Input) -> Output
model.batch_predict(inputs: Vec<Input>) -> Vec<Output>
model.stream_predict(inputs: Dataset<Input>) -> Dataset<Output>
```

#### Performance Features:
- Automatic batching for throughput optimization
- Device placement (CPU/GPU) with automatic transfers
- Model quantization and compression
- Execution profiling and optimization hints

---

## 🧠 Supporting Types (Future Extensions)

- `LossFunction`: e.g., `CrossEntropy`, `MSE`
- `Optimizer`: e.g., `SGD`, `Adam`
- `MLGraph`: DAG of models/operators
- `Metric`: Evaluation and scoring abstractions

---

## ✨ AI Type Design Philosophy

| Goal                | Mechanism                           |
| ------------------- | ----------------------------------- |
| Type Safety         | Shape-checked tensors               |
| Performance         | Backend-neutral tensor engine       |
| Composability       | Pipelines, map/filter DSL           |
| Portability         | Export to ONNX, deploy as service   |
| ML-Native Semantics | Model + Dataset + Pipeline cohesion |

---

## 📌 Example: Complete AI Deployment Pipeline

```stark
// Load pre-trained model with automatic shape inference
let model = load_pytorch_model("sentiment_classifier.pt")
// Type: Model<Tensor<Float32>[batch, 512], Tensor<Float32>[batch, 3]>

// Create streaming data pipeline
let text_stream: Dataset<String> = load_text_stream("reviews.jsonl")
let processed = text_stream
    .map(|text| tokenize_and_embed(text))  // String -> Tensor<Float32>[512]
    .batch(32)                             // Batch for efficiency
    .prefetch(2)                          // Async prefetching

// Run inference with automatic batching
for batch in processed {
    let predictions = model.predict(batch)  // Tensor<Float32>[32, 3]
    let labels = argmax(predictions, axis=1) // Tensor<Int32>[32]
    
    // Process results
    for (text, label) in zip(batch.unbatch(), labels.unbatch()) {
        handle_classification(text, label)
    }
}
```

---

These core AI types make STARK uniquely suited for production ML deployment, providing type safety, performance optimization, and seamless integration with existing ML ecosystems. They are fundamental language primitives, not library additions, enabling compile-time optimizations and safety guarantees that are impossible in general-purpose languages.


# 🧱 STARK Language — Advanced & Future-Oriented Types

This document outlines additional powerful types in STARKLANG designed for safety, flexibility, and next-gen programming paradigms. These types enhance developer ergonomics, ensure runtime robustness, and support emerging paradigms such as functional programming, streaming data, and safe concurrency.

---

## ✅ 1. Option

A type-safe replacement for null. Guarantees explicit handling of absent values.

```stark
let maybe_user: Option<UserProfile> = None
```

### Variants:

```stark
enum Option<T>:
    Some(value: T)
    None
```

### Methods:

- `.is_some()`
- `.unwrap()` (with panic fallback)
- `.unwrap_or(default: T)`
- `.map(fn)`

---

## ✅ 2. Result\<T, E>

Robust error handling and propagation using algebraic types.

```stark
fn divide(a: Float32, b: Float32) -> Result<Float32, String>:
    if b == 0.0:
        return Err("Division by zero")
    return Ok(a / b)
```

### Variants:

```stark
enum Result<T, E>:
    Ok(value: T)
    Err(error: E)
```

### Methods:

- `.is_ok()`, `.is_err()`
- `.unwrap()`, `.unwrap_or()`
- `.map(fn)`, `.map_err(fn)`

---

## 🔷 3. Union\<T1 | T2 | ...>

A flexible type that can hold one of several types.

```stark
let mixed: Union<Int | Float32 | String>
```

Handled via **pattern matching**:

```stark
match mixed:
  Int(i): print("Int: " + str(i))
  Float32(f): print("Float: " + str(f))
  String(s): print("String: " + s)
```

---

## 🔷 4. Stream

A lazy, async-capable, chainable data processing abstraction.

```stark
let stream: Stream<Int> = range(0, 100)
  .map(fn(x) => x * x)
  .filter(fn(x) => x % 2 == 0)
```

### Methods:

- `.map(fn)`, `.filter(fn)`
- `.reduce(fn, init)`
- `.collect()`

Supports both **lazy evaluation** and **stream-based pipelines**.

---

## 🔷 5. TaggedRecord

Schema-aware struct with tagged metadata.

```stark
record Invoice:
    id: String @tag("primary")
    total: Float32 @tag("currency:USD")
```

Useful for **serialization**, **schema validation**, and **ETL/data pipelines**.

---

## ⚪ 6. Ref / Pointer

Low-level reference to memory-allocated object (useful for FFI/systems integration).

```stark
let ptr: Ref<UserProfile> = ref user
```

Supports dereferencing, pointer equality, and pass-by-ref semantics.

---

## ⚪ 7. Bitfield

Compact bit-based flag structures for systems/embedded use.

```stark
bitfield Flags:
    READ: 0
    WRITE: 1
    EXEC: 2
```

Bitwise manipulation: `flags.set(READ)`, `flags.has(EXEC)`

---

## ⚪ 8. Persistent Collections

Immutable data structures for functional-style programming.

- `PersistentList<T>`
- `PersistentMap<K, V>`

Enable structural sharing and safe concurrent reads.

---

## ⏳ 9. Channel

Concurrency primitive for **actor systems**, **thread isolates**, and message-passing models.

```stark
let ch: Channel<Int> = channel()
spawn fn producer():
    ch.send(42)

let val = ch.recv()
```

---

## ⏳ 10. Future

Async task result holder, supports chaining and callbacks.

```stark
let result: Future<Int> = async fn compute(): return 99
```

Support methods:

- `.await()`
- `.then(fn)`
- `.catch(fn)`

---

These types extend STARKLANG's capability beyond a typical language into a **developer-first, AI-native, concurrency-safe ecosystem**—built for the real-world challenges of the future.

