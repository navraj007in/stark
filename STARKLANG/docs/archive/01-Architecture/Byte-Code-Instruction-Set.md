> Non-Normative Note: This document provides architectural guidance and may evolve. The normative Core v1 language specification lives in `docs/spec/`.

# ⚙ STARKLANG — Bytecode Instruction Set Specification

This specification defines the low-level intermediate representation for executing STARKLANG code within the STARK Virtual Machine (STARKVM).

The bytecode is:
- Stack-based
- Compact and deterministic
- Optimized for static analysis, JIT, and multi-core scheduling

---

## 📦 Bytecode Structure

```
[Opcode] [Operands...]
```

- Opcode: 1-byte or 2-byte integer (extensible)
- Operands: Zero or more values, encoded by operand type

---

## 🧩 Bytecode Categories

### ✅ 1️⃣ Stack Operations

| Opcode         | Description                        |
|----------------|------------------------------------|
| PUSH_CONST val | Push a constant to stack           |
| POP            | Discard top of stack               |
| DUP            | Duplicate top of stack             |
| SWAP           | Swap top 2 values                  |

### ✅ 2️⃣ Arithmetic & Logic

| Opcode        | Description                        |
|---------------|------------------------------------|
| ADD, SUB, MUL, DIV | Binary math ops               |
| MOD, POW, NEG | Additional math operations         |
| AND, OR, XOR, NOT | Bitwise logic                  |
| EQ, NEQ, GT, LT, GTE, LTE | Comparisons             |

### ✅ 3️⃣ Control Flow

| Opcode         | Description                        |
|----------------|------------------------------------|
| JMP addr       | Unconditional jump                 |
| JMP_IF_TRUE    | Jump if top of stack is true       |
| JMP_IF_FALSE   | Jump if top of stack is false      |
| LABEL id       | Logical label (resolved by compiler) |

### ✅ 4️⃣ Variable/Memory Operations

| Opcode         | Description                        |
|----------------|------------------------------------|
| LOAD_VAR index | Load from local/global var         |
| STORE_VAR index| Store to local/global var          |
| LOAD_HEAP addr | Load from heap pointer             |
| STORE_HEAP addr| Store to heap pointer              |

### ✅ 5️⃣ Function Call & Return

| Opcode       | Description                         |
|--------------|-------------------------------------|
| CALL index argc | Call function from symbol table  |
| RETURN       | Return from function                |
| TAILCALL     | Tail-optimized call                 |
| CLOSURE index| Create a closure (planned)          |

### ✅ 6️⃣ Structs, Arrays, Tuples

| Opcode         | Description                          |
|----------------|--------------------------------------|
| NEW_STRUCT id  | Allocate struct on heap              |
| GET_FIELD idx  | Access field in struct               |
| SET_FIELD idx  | Update field in struct               |
| NEW_ARRAY size | Create array                         |
| ARRAY_GET idx  | Access array index                   |
| ARRAY_SET idx  | Set array index                      |
| NEW_TUPLE n    | Pack values into tuple               |
| TUPLE_GET idx  | Access tuple field                   |

### ✅ 7️⃣ Async / Parallel Ops

| Opcode         | Description                        |
|----------------|------------------------------------|
| AWAIT          | Await a future                     |
| SPAWN_FUNC     | Spawn async task                   |
| FUTURE_RESOLVE | Resolve future                     |
| JOIN_TASK      | Wait for task result               |

### ✅ 8️⃣ Channel / Message Passing

| Opcode          | Description                      |
|------------------|----------------------------------|
| NEW_CHANNEL      | Create new channel              |
| SEND_CHANNEL     | Send message                    |
| RECV_CHANNEL     | Receive message                 |
| SELECT_CHANNEL   | Select over multiple channels (planned) |

### ✅ 9️⃣ Actor Model Ops

| Opcode            | Description                       |
|-------------------|-----------------------------------|
| SPAWN_ACTOR       | Create new actor                  |
| SEND_ACTOR        | Send message to actor             |
| ASK_ACTOR         | Actor RPC                         |
| TERMINATE_ACTOR   | Stop actor                        |

### ✅ 🔢 Tensor & AI Primitives

| Opcode            | Description                         |
|-------------------|-------------------------------------|
| TENSOR_NEW        | Allocate tensor                     |
| TENSOR_GET        | Read tensor value                   |
| TENSOR_SET        | Set tensor value                    |
| MODEL_LOAD        | Load ML model                       |
| MODEL_PREDICT     | Run model inference                 |
| TENSOR_MAP        | Map function over tensor            |

### ✅ 🔐 Observability + Cloud Ops (Planned)

| Opcode           | Description                         |
|------------------|-------------------------------------|
| LOG              | Emit log line                       |
| METRIC           | Record metric                       |
| TRACING_SPAN     | Start trace span                    |
| EMIT_EVENT       | Emit event                          |

---

## 📎 Operand Encoding (Compiler Responsibility)

| Operand Type | Description                       |
|--------------|-----------------------------------|
| Constant     | From constant pool                |
| Function ID  | Symbol/function table index       |
| Heap Address | Runtime managed pointer           |
| Field Index  | Byte or short                     |
| Labels       | Compiler resolves to jump targets |

---

## 🧠 Example: Bytecode Flow

STARK Source:
```stark
fn sum(a: Int, b: Int) -> Int:
    return a + b
```

STARK Bytecode:
```
LOAD_VAR 0
LOAD_VAR 1
ADD
RETURN
```

---

## 🔮 Future Extensions

| Group | Description                        |
|-------|------------------------------------|
| IO_* | File/Socket IO                     |
| SYS_* | Runtime diagnostics, syscalls     |
| GPU_* | GPU-accelerated operations        |
| SERIALIZE_* | Structured data encoding    |

---

## ✅ Summary

| Category                | Status   |
|-------------------------|----------|
| Arithmetic & Stack Ops  | ✅ Complete |
| Control Flow & Calls    | ✅ Complete |
| Structs & Collections   | ✅ Complete |
| Async & Futures         | ✅ Complete |
| Channel Messaging       | ✅ Complete |
| Actor Model Support     | ✅ Complete |
| AI/Tensor Ops           | ✅ Drafted |
| Observability & Cloud   | ⏳ Planned |

STARKLANG’s bytecode is lightweight, flexible, and deeply extensible — enabling AI/ML, cloud-native, and concurrent workloads to run at scale.

