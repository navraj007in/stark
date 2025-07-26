# STARKLANG — Traits and Type Constraints Specification

This document defines the traits and type constraint system in STARKLANG. Traits are abstract behavior contracts that allow for polymorphism, compile-time checks, and interface-driven design. They are foundational to building reusable libraries, enforcing structural contracts, and enabling type-safe generic programming.

---

## 🧠 What is a Trait?
A **Trait** in STARKLANG is a collection of function signatures that a type must implement. It is analogous to interfaces (in Go/TypeScript), traits (Rust), or typeclasses (Haskell).

### 📌 Trait Syntax
```stark
trait Predictable:
    fn predict(self, input: Tensor<Float32>) -> Tensor<Float32>
```

### 📌 Trait Implementation
```stark
struct LinearModel:
    weights: Tensor<Float32>

impl Predictable for LinearModel:
    fn predict(self, input: Tensor<Float32>) -> Tensor<Float32>:
        return dot(input, self.weights)
```

---

## 📚 Use in Generic Constraints
Traits allow you to restrict generics to types that fulfill specific contracts.

### Example: Generic with Constraint
```stark
fn run_model<T: Predictable>(model: T, data: Tensor<Float32>) -> Tensor<Float32>:
    return model.predict(data)
```

You can also use multiple constraints:
```stark
fn train<T: Predictable + Serializable>(model: T):
    ...
```

---

## 🧩 Built-in Core Traits (Standard Library)
| Trait Name     | Description                             |
|----------------|------------------------------------------|
| `Equatable`    | Supports `==` / `!=` comparisons          |
| `Comparable`   | Supports `<`, `>`, `<=`, `>=` operations |
| `Serializable` | Can be serialized to JSON/YAML/bytes     |
| `Cloneable`    | Supports deep copying via `.clone()`     |
| `Displayable`  | Can be rendered as a string via `.to_string()` |
| `Predictable`  | Used in ML models, defines `.predict()`  |

---

## 🔄 Trait Inheritance
Traits can inherit other traits to compose behaviors:
```stark
trait Exportable: Serializable + Displayable:
    fn export(self, format: String) -> String
```

---

## 🧠 Trait Objects (Future Support)
Planned for advanced polymorphism:
```stark
let model: dyn Predictable = load_model()
model.predict(input)
```
This will allow runtime-dispatched interfaces (boxed trait objects).

---

## ⚠ Compiler Behavior
- Enforces trait implementation at compile time.
- Emits warnings if trait contract not satisfied.
- Trait resolution is monomorphized — no runtime vtables (unless explicitly boxed).

---

## ✅ Trait Implementation Rules
| Rule | Description |
|------|-------------|
| Trait must be declared before usage | Cannot implement undeclared trait |
| Only structs, enums can implement traits | Not primitives directly |
| Traits must have at least one method | Empty traits disallowed |

---

## 🔮 Future Enhancements
| Feature | Purpose | Status |
|--------|---------|--------|
| Trait object boxing (`dyn`) | Dynamic dispatch | Planned |
| Auto derive traits | Reduce boilerplate | Planned |
| Marker traits | No method traits like `Send`, `Sync` | Planned |

---

## 📌 Summary
STARKLANG's trait and constraint system brings powerful compile-time polymorphism, structural contracts, and modular reuse to the language. It empowers developers to write generic, reusable, and extensible code in both low-level systems and high-level AI abstractions.