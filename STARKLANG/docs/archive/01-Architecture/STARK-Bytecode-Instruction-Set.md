# STARK VM Bytecode Instruction Set Specification

> Non-Normative Note: This document provides architectural guidance and may evolve. The normative Core v1 language specification lives in `docs/spec/`.

## Overview

The STARK Virtual Machine (STARKVM) is a stack-based virtual machine optimized for AI/ML workloads. It features:
- Native tensor operations
- Hardware-accelerated AI primitives
- Efficient memory management for large datasets
- Built-in concurrency support
- Cloud deployment awareness

## Bytecode Format

### File Structure

```
STARK Bytecode File (.starkc)
+-------------------+
| Magic Number (4B) | 0x53544152 ('STAR')
| Version (2B)      | Major.Minor
| Flags (2B)        | Debug info, optimization level
+-------------------+
| Constant Pool     | 
| Size (4B)         |
| Constants...      |
+-------------------+
| Type Table        |
| Size (4B)         |
| Type Entries...   |
+-------------------+
| Function Table    |
| Size (4B)         |
| Functions...      |
+-------------------+
| Global Data       |
| Size (4B)         |
| Initial Values... |
+-------------------+
| Main Entry (4B)   | Offset to main function
+-------------------+
```

### Instruction Encoding

Each instruction uses variable-length encoding:

```
Basic Format (1-5 bytes):
+--------+--------+--------+--------+--------+
| OPCODE | OPERAND1 | OPERAND2 | ... | 
+--------+--------+--------+--------+--------+
  1 byte   0-4 bytes (optional)

Extended Format (for large constants/jumps):
+--------+--------+----------------+
| 0xFF   | OPCODE | EXTENDED_DATA  |
+--------+--------+----------------+
  1 byte   1 byte   4-8 bytes
```

## Register Architecture

STARKVM uses a hybrid stack/register architecture:

- **Evaluation Stack**: For expression evaluation
- **Local Variable Slots**: Fast access to function locals
- **Tensor Registers**: T0-T15 for tensor operations
- **Special Registers**:
  - PC: Program Counter
  - SP: Stack Pointer
  - FP: Frame Pointer
  - TP: Tensor Pool Pointer
  - GP: Global Pointer

## Memory Layout

```
Memory Space (64-bit addressing):
0x0000_0000_0000_0000 +----------------+
                      | Reserved       |
0x0000_0000_0001_0000 +----------------+
                      | Global Data    |
0x0000_0000_1000_0000 +----------------+
                      | Heap           |
                      | (grows up)     |
                      |                |
                      | ...            |
                      |                |
                      | Stack          |
                      | (grows down)   |
0x0000_7FFF_FFFF_0000 +----------------+
                      | Tensor Pool    |
                      | (GPU mapped)   |
0x0000_8000_0000_0000 +----------------+
                      | Model Cache    |
0x0000_A000_0000_0000 +----------------+
                      | Reserved       |
0xFFFF_FFFF_FFFF_FFFF +----------------+
```

## Instruction Set

### 1. Stack Operations (0x00-0x0F)

| Opcode | Mnemonic | Operands | Description |
|--------|----------|----------|-------------|
| 0x00 | NOP | - | No operation |
| 0x01 | DUP | - | Duplicate top of stack |
| 0x02 | DUP2 | - | Duplicate top 2 stack items |
| 0x03 | POP | - | Remove top of stack |
| 0x04 | SWAP | - | Swap top 2 stack items |
| 0x05 | ROT3 | - | Rotate top 3 items |

### 2. Constants & Literals (0x10-0x1F)

| Opcode | Mnemonic | Operands | Description |
|--------|----------|----------|-------------|
| 0x10 | PUSH_I8 | i8 | Push 8-bit integer |
| 0x11 | PUSH_I16 | i16 | Push 16-bit integer |
| 0x12 | PUSH_I32 | i32 | Push 32-bit integer |
| 0x13 | PUSH_I64 | i64 | Push 64-bit integer |
| 0x14 | PUSH_F32 | f32 | Push 32-bit float |
| 0x15 | PUSH_F64 | f64 | Push 64-bit float |
| 0x16 | PUSH_TRUE | - | Push boolean true |
| 0x17 | PUSH_FALSE | - | Push boolean false |
| 0x18 | PUSH_NULL | - | Push null reference |
| 0x19 | PUSH_CONST | u16 | Push from constant pool |
| 0x1A | PUSH_TENSOR | u16 | Push tensor constant |

### 3. Local Variables (0x20-0x2F)

| Opcode | Mnemonic | Operands | Description |
|--------|----------|----------|-------------|
| 0x20 | LOAD_0 | - | Load local variable 0 |
| 0x21 | LOAD_1 | - | Load local variable 1 |
| 0x22 | LOAD_2 | - | Load local variable 2 |
| 0x23 | LOAD_3 | - | Load local variable 3 |
| 0x24 | LOAD | u8 | Load local variable N |
| 0x25 | STORE_0 | - | Store to local variable 0 |
| 0x26 | STORE_1 | - | Store to local variable 1 |
| 0x27 | STORE_2 | - | Store to local variable 2 |
| 0x28 | STORE_3 | - | Store to local variable 3 |
| 0x29 | STORE | u8 | Store to local variable N |

### 4. Arithmetic Operations (0x30-0x3F)

| Opcode | Mnemonic | Operands | Description |
|--------|----------|----------|-------------|
| 0x30 | ADD | - | Pop 2, push sum |
| 0x31 | SUB | - | Pop 2, push difference |
| 0x32 | MUL | - | Pop 2, push product |
| 0x33 | DIV | - | Pop 2, push quotient |
| 0x34 | MOD | - | Pop 2, push remainder |
| 0x35 | POW | - | Pop 2, push power |
| 0x36 | NEG | - | Negate top of stack |
| 0x37 | ABS | - | Absolute value |
| 0x38 | SQRT | - | Square root |
| 0x39 | ADD_F | - | Floating-point add |
| 0x3A | SUB_F | - | Floating-point subtract |
| 0x3B | MUL_F | - | Floating-point multiply |
| 0x3C | DIV_F | - | Floating-point divide |

### 5. Bitwise Operations (0x40-0x4F)

| Opcode | Mnemonic | Operands | Description |
|--------|----------|----------|-------------|
| 0x40 | AND | - | Bitwise AND |
| 0x41 | OR | - | Bitwise OR |
| 0x42 | XOR | - | Bitwise XOR |
| 0x43 | NOT | - | Bitwise NOT |
| 0x44 | SHL | - | Shift left |
| 0x45 | SHR | - | Shift right |
| 0x46 | USHR | - | Unsigned shift right |

### 6. Comparison Operations (0x50-0x5F)

| Opcode | Mnemonic | Operands | Description |
|--------|----------|----------|-------------|
| 0x50 | EQ | - | Equal |
| 0x51 | NE | - | Not equal |
| 0x52 | LT | - | Less than |
| 0x53 | LE | - | Less than or equal |
| 0x54 | GT | - | Greater than |
| 0x55 | GE | - | Greater than or equal |
| 0x56 | CMP | - | Compare (push -1, 0, or 1) |
| 0x57 | IS_NULL | - | Check if null |
| 0x58 | IS_TENSOR | - | Check if tensor type |

### 7. Control Flow (0x60-0x6F)

| Opcode | Mnemonic | Operands | Description |
|--------|----------|----------|-------------|
| 0x60 | JMP | i16 | Unconditional jump |
| 0x61 | JMP_IF | i16 | Jump if true |
| 0x62 | JMP_IFNOT | i16 | Jump if false |
| 0x63 | JMP_EQ | i16 | Jump if equal |
| 0x64 | JMP_NE | i16 | Jump if not equal |
| 0x65 | JMP_LT | i16 | Jump if less than |
| 0x66 | JMP_LONG | i32 | Long jump |
| 0x67 | CALL | u16 | Call function |
| 0x68 | CALL_VIRT | - | Virtual call (vtable) |
| 0x69 | RET | - | Return from function |
| 0x6A | RET_VAL | - | Return with value |
| 0x6B | THROW | - | Throw exception |
| 0x6C | MATCH | u16 | Pattern match jump table |

### 8. Memory Operations (0x70-0x7F)

| Opcode | Mnemonic | Operands | Description |
|--------|----------|----------|-------------|
| 0x70 | NEW | u16 | Allocate object |
| 0x71 | NEW_ARRAY | u16 | Allocate array |
| 0x72 | NEW_TENSOR | - | Allocate tensor |
| 0x73 | LOAD_FIELD | u16 | Load object field |
| 0x74 | STORE_FIELD | u16 | Store object field |
| 0x75 | LOAD_ELEM | - | Load array element |
| 0x76 | STORE_ELEM | - | Store array element |
| 0x77 | ARRAY_LEN | - | Get array length |
| 0x78 | LOAD_GLOBAL | u16 | Load global variable |
| 0x79 | STORE_GLOBAL | u16 | Store global variable |
| 0x7A | MEMCPY | - | Memory copy |
| 0x7B | MEMSET | - | Memory set |

### 9. Tensor Operations (0x80-0x9F)

| Opcode | Mnemonic | Operands | Description |
|--------|----------|----------|-------------|
| 0x80 | TENSOR_CREATE | u8 | Create tensor with shape |
| 0x81 | TENSOR_RESHAPE | - | Reshape tensor |
| 0x82 | TENSOR_TRANSPOSE | - | Transpose tensor |
| 0x83 | TENSOR_SLICE | - | Slice tensor |
| 0x84 | TENSOR_CONCAT | u8 | Concatenate tensors |
| 0x85 | TENSOR_SPLIT | u8 | Split tensor |
| 0x86 | TENSOR_PAD | - | Pad tensor |
| 0x87 | TENSOR_LOAD_REG | u8 | Load to tensor register |
| 0x88 | TENSOR_STORE_REG | u8 | Store from tensor register |
| 0x90 | MATMUL | - | Matrix multiplication |
| 0x91 | CONV2D | - | 2D convolution |
| 0x92 | MAXPOOL2D | - | 2D max pooling |
| 0x93 | AVGPOOL2D | - | 2D average pooling |
| 0x94 | BATCHNORM | - | Batch normalization |
| 0x95 | DROPOUT | - | Dropout layer |
| 0x96 | ACTIVATION | u8 | Activation (ReLU, Sigmoid, etc.) |
| 0x97 | SOFTMAX | - | Softmax operation |
| 0x98 | REDUCE | u8 | Reduce operation (sum, mean, etc.) |
| 0x99 | BROADCAST | - | Broadcast tensor |
| 0x9A | GRAD | - | Compute gradient |
| 0x9B | BACKWARD | - | Backward propagation |

### 10. Model Operations (0xA0-0xAF)

| Opcode | Mnemonic | Operands | Description |
|--------|----------|----------|-------------|
| 0xA0 | MODEL_LOAD | u16 | Load model from cache |
| 0xA1 | MODEL_SAVE | u16 | Save model to cache |
| 0xA2 | MODEL_FORWARD | - | Forward pass |
| 0xA3 | MODEL_TRAIN | - | Training mode |
| 0xA4 | MODEL_EVAL | - | Evaluation mode |
| 0xA5 | OPTIMIZER_STEP | - | Optimizer step |
| 0xA6 | LOSS_COMPUTE | u8 | Compute loss function |

### 11. Concurrency Operations (0xB0-0xBF)

| Opcode | Mnemonic | Operands | Description |
|--------|----------|----------|-------------|
| 0xB0 | SPAWN | u16 | Spawn actor/thread |
| 0xB1 | SEND | - | Send message to actor |
| 0xB2 | RECEIVE | - | Receive message |
| 0xB3 | AWAIT | - | Await async result |
| 0xB4 | YIELD | - | Yield execution |
| 0xB5 | LOCK | u16 | Acquire lock |
| 0xB6 | UNLOCK | u16 | Release lock |
| 0xB7 | ATOMIC_ADD | - | Atomic add |
| 0xB8 | ATOMIC_CAS | - | Compare and swap |
| 0xB9 | BARRIER | - | Memory barrier |

### 12. Cloud/Deployment Operations (0xC0-0xCF)

| Opcode | Mnemonic | Operands | Description |
|--------|----------|----------|-------------|
| 0xC0 | CLOUD_INVOKE | u16 | Invoke cloud function |
| 0xC1 | CLOUD_STORE | - | Store to cloud storage |
| 0xC2 | CLOUD_LOAD | - | Load from cloud storage |
| 0xC3 | METRIC_PUSH | u16 | Push metric |
| 0xC4 | LOG | u8 | Log message |
| 0xC5 | TRACE_ENTER | u16 | Enter trace span |
| 0xC6 | TRACE_EXIT | - | Exit trace span |

### 13. Type Operations (0xD0-0xDF)

| Opcode | Mnemonic | Operands | Description |
|--------|----------|----------|-------------|
| 0xD0 | CAST | u16 | Type cast |
| 0xD1 | INSTANCEOF | u16 | Instance check |
| 0xD2 | TYPE_OF | - | Get runtime type |
| 0xD3 | TENSOR_DTYPE | - | Get tensor data type |
| 0xD4 | TENSOR_SHAPE | - | Get tensor shape |

### 14. System Operations (0xE0-0xEF)

| Opcode | Mnemonic | Operands | Description |
|--------|----------|----------|-------------|
| 0xE0 | PRINT | - | Print to stdout |
| 0xE1 | DEBUG | - | Debug breakpoint |
| 0xE2 | ASSERT | - | Assert condition |
| 0xE3 | PANIC | u16 | Panic with message |
| 0xE4 | GC_HINT | u8 | Garbage collection hint |
| 0xE5 | DEVICE_SEL | u8 | Select compute device |

### 15. Extended Operations (0xF0-0xFF)

| Opcode | Mnemonic | Operands | Description |
|--------|----------|----------|-------------|
| 0xF0 | CALL_NATIVE | u16 | Call native function |
| 0xF1 | CALL_BUILTIN | u16 | Call builtin function |
| 0xF2 | SIMD | u16 | SIMD operation |
| 0xFF | EXTENDED | - | Extended instruction prefix |

## Constant Pool Entries

The constant pool stores compile-time constants:

```
Entry Types:
- 0x01: Integer (i64)
- 0x02: Float (f64)  
- 0x03: String (UTF-8)
- 0x04: Type descriptor
- 0x05: Function reference
- 0x06: Tensor shape
- 0x07: Model metadata
```

## Function Format

```
Function Entry:
+------------------+
| Flags (2B)       | async, generator, etc.
| Params Count (1B)|
| Locals Count (1B)|
| Max Stack (2B)   |
| Code Length (4B) |
| Code Bytes...    |
+------------------+
| Exception Table  |
| Debug Info       |
+------------------+
```

## Optimization Hints

The bytecode includes optimization hints for the JIT compiler:

- Hot loop markers
- Tensor fusion opportunities  
- Parallelization hints
- Memory prefetch hints
- Device placement hints

## Example Bytecode

Simple tensor operation:
```stark
let a = tensor.rand([1000, 1000])
let b = tensor.rand([1000, 1000])
let c = a @ b
```

Compiles to:
```
PUSH_CONST    0x0001        # Shape [1000, 1000]
CALL_BUILTIN  0x0010        # tensor.rand
STORE_0                     # a = ...
PUSH_CONST    0x0001        # Shape [1000, 1000]
CALL_BUILTIN  0x0010        # tensor.rand  
STORE_1                     # b = ...
LOAD_0                      # Load a
LOAD_1                      # Load b
MATMUL                      # Matrix multiply
STORE_2                     # c = ...
```

## Performance Considerations

1. **Tensor Register Allocation**: The T0-T15 registers avoid stack operations for tensor math
2. **Fused Operations**: Common patterns like Conv2D+ReLU+BatchNorm can be fused
3. **Memory Layout**: Tensors are stored in HW-friendly layouts (NCHW, NHWC)
4. **Device Offloading**: Tensor ops automatically offload to GPU/TPU when available
5. **Zero-Copy Operations**: Reshape, transpose, slice avoid copying when possible

## Future Extensions

Reserved opcode ranges for future use:
- 0x9C-0x9F: Advanced tensor operations
- 0xA7-0xAF: Extended model operations  
- 0xC7-0xCF: More cloud primitives
- 0xD5-0xDF: Advanced type operations
- 0xE6-0xEF: System extensions
- 0xF3-0xFE: Reserved for vendor extensions
