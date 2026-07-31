# STARK Feature Examples — Problems, Root Causes, and Required Solutions

**Status:** Implementation brief  
**Date:** 2026-07-31  
**Target compiler branch:** `c79-three-engine-correction`  
**Target compiler commit:** `61d1c18` or later  
**Purpose:** Correct the generated STARK feature-example suite, separate suite defects from compiler defects, and establish a repeatable qualification process.

---

## 1. Background

A feature-demonstration suite was created to exercise the implemented STARK compiler surface through:

- standalone single-file programs;
- compile-fail programs;
- expected runtime-trap programs;
- multi-file packages;
- multi-package dependency graphs;
- consumers of first-party STARK packages.

The first compiler-driven run produced several failures. Those failures fall into four distinct categories:

1. **Example-source defects**  
   The sample program uses syntax or APIs the compiler does not support.

2. **Package-layout or manifest defects**  
   The source is conceptually correct, but the package name, dependency declaration, or module structure is invalid.

3. **Compiler defects**  
   A program passes static checking but crashes the interpreter, or a required semantic check is absent.

4. **Compiler-version mismatch**  
   A test expects a correction introduced by WP-C7.9/CD-275, but the executed compiler binary may have been built from an earlier tree.

These categories must remain separate. An example defect must not be recorded as a compiler bug, and a compiler bug must not be hidden by rewriting the example.

---

# 2. Qualification Precondition

Before evaluating any result, record the exact compiler source and executable used.

Run:

```bash
git branch --show-current
git rev-parse HEAD
realpath /path/to/starkc
```

Required target:

```text
branch: c79-three-engine-correction
commit: 61d1c18 or a later descendant
```

Rebuild the compiler from that checkout before testing:

```bash
cargo build --workspace
```

Do not rely on a previously built `starkc` found through `PATH`.

The following behaviours are expected only after CD-275:

- malformed Core trait implementations are rejected;
- `HashMap` and `HashSet` key bounds are enforced;
- borrowed enum payload patterns work through references;
- unsupported iterator adapters are rejected with `E0105`;
- signed `MIN / -1` and `MIN % -1` trap as `IntegerOverflow`;
- stderr participates in differential comparison.

If these behaviours are absent, first rule out a stale compiler binary.

---

# 3. Single-File Example Problems

## 3.1 Arrays do not expose `.len()` as a method

### Affected example

```text
single_file/08_arrays_tuples_and_indexing.stark
```

### Problem

The example calls:

```stark
values.len()
```

where `values` has type:

```stark
[Int32; 4]
```

The current compiler does not provide a method-call surface for arrays and emits `E0304`.

### Root cause

The example assumed parity between arrays and `Vec<T>`. STARK currently supports fixed-size array indexing and iteration but not an array `.len()` method.

### Required solution

Remove the method call and demonstrate the supported capabilities only:

```stark
fn main() {
    let values: [Int32; 4] = [10, 20, 30, 40];
    let pair: (Int32, Bool) = (42, true);

    if values[0] != 10 || values[3] != 40 {
        panic("array indexing failed");
    }

    if pair.0 != 42 || !pair.1 {
        panic("tuple projection failed");
    }

    let mut sum = 0;
    for value in values {
        sum += value;
    }

    if sum != 100 {
        panic("array iteration failed");
    }

    println("arrays-tuples: PASS");
}
```

### Acceptance criteria

- passes `starkc check`;
- executes in HIR;
- lowers and executes in MIR;
- builds and executes in native debug and release;
- prints exactly:

```text
arrays-tuples: PASS
```

---

## 3.2 `Box<T>` is not dereferenced with unary `*`

### Affected example

```text
single_file/11_box_and_recursive_enum.stark
```

### Problem

The example uses:

```stark
sum(*left)
```

where `left` is `Box<Tree>`.

The compiler does not treat `Box<T>` as a reference and rejects unary dereference.

### Root cause

The example assumed Rust-style `Deref` behaviour. STARK's `Box<T>` is an implementation-provided owning container with explicit APIs.

### Required solution

Use `Box::into_inner()`:

```stark
enum Tree {
    Leaf(Int32),
    Node(Box<Tree>, Box<Tree>)
}

fn sum(tree: Tree) -> Int32 {
    match tree {
        Tree::Leaf(value) => value,
        Tree::Node(left, right) => {
            let left_tree = left.into_inner();
            let right_tree = right.into_inner();
            sum(left_tree) + sum(right_tree)
        }
    }
}

fn main() {
    let tree = Tree::Node(
        Box::new(Tree::Leaf(20)),
        Box::new(Tree::Leaf(22))
    );

    if sum(tree) != 42 {
        panic("Box recursive enum failed");
    }

    println("box-recursive-enum: PASS");
}
```

### Acceptance criteria

- recursive owning enum compiles;
- no use-after-move diagnostic;
- executes through all admitted engines;
- prints `box-recursive-enum: PASS`.

---

## 3.3 Explicit generic invocation requires turbofish syntax

### Affected example

```text
single_file/12_generics_and_user_trait.stark
```

### Problem

Invalid syntax:

```stark
identity<Int32>(42)
```

### Required solution

Use:

```stark
identity::<Int32>(42)
```

### Acceptance criteria

- explicit generic call parses;
- the same generic function may also be invoked through inference;
- native output agrees across debug and release.

---

## 3.4 Primitive integers do not expose `.to_string()`

### Affected example

```text
single_file/13_display_and_formatting.stark
```

### Problem

The example calls:

```stark
self.x.to_string()
```

on `Int32`, but no such primitive method exists.

### Root cause

The example assumed a Rust-like convenience method not present in the admitted STARK surface.

### Required solution

Keep the example focused on user `Display` dispatch:

```stark
struct Point {
    x: Int32,
    y: Int32
}

impl Display for Point {
    fn fmt(&self) -> String {
        if self.x == 20 && self.y == 22 {
            String::from("POINT")
        } else {
            String::from("OTHER")
        }
    }
}

fn main() {
    let point = Point { x: 20, y: 22 };
    println(point);
    println("display: PASS");
}
```

### Acceptance criteria

Expected stdout:

```text
POINT
display: PASS
```

The test must prove that `println(point)` invokes the user implementation rather than structural formatting.

---

## 3.5 `HashMap<String, V>::get` requires `&String`

### Affected example

```text
single_file/14_hashmap_and_hashset.stark
```

### Problem

The example uses:

```stark
counts.get("stark")
```

for a map keyed by `String`.

The current method signature expects `&String`, not `&str`.

### Required solution

Retain the key:

```stark
fn main() {
    let mut counts: HashMap<String, Int32> = HashMap::new();
    let key = String::from("stark");

    counts.insert(key.clone(), 42);

    match counts.get(&key) {
        Some(value) => {
            if *value != 42 {
                panic("HashMap value mismatch");
            }
        }
        None => {
            panic("HashMap lookup failed");
        }
    }

    let mut seen: HashSet<Int32> = HashSet::new();
    seen.insert(7);
    seen.insert(7);
    seen.insert(42);

    if !seen.contains(&42) {
        panic("HashSet contains failed");
    }

    println("hash-collections: PASS");
}
```

### Acceptance criteria

- valid key type satisfies `Hash + Eq`;
- lookup returns `Some(&Int32)`;
- duplicate set insertion remains harmless;
- all engines print `hash-collections: PASS`.

---

## 3.6 `Ordering` cannot be compared using `!=`

### Affected example

```text
single_file/15_custom_eq_ord_clone_default_from.stark
```

### Problem

The example uses:

```stark
score.cmp(&defaulted) != Ordering::Greater
```

but `Ordering` does not currently satisfy `Eq`.

### Required solution

Use pattern matching:

```stark
match score.cmp(&defaulted) {
    Ordering::Greater => {}
    _ => {
        panic("Ord failed");
    }
}
```

The remainder of the example may continue testing:

- `Eq`;
- `Clone`;
- `Default`;
- `From`;
- `Ord`.

### Acceptance criteria

- no equality operation is applied to `Ordering`;
- custom `Ord::cmp` is exercised;
- malformed `Ord` implementations are tested separately as compile-fail cases.

---

## 3.7 Borrowed enum payload matching

### Affected example

```text
single_file/16_match_through_reference.stark
```

### Problem observed

The compiler reported that matching:

```stark
Holder::Value(value)
```

through `&Holder` moves a non-`Copy` `String` out of a borrow.

### Interpretation

This behaviour is expected on a compiler before CD-275 Packet C.

After CD-275, place-aware pattern matching should preserve a projected reference rather than clone or move the payload.

### Required action

First confirm the compiler commit.

If using `61d1c18` or later and the example still fails, record a regression against Packet C.

### Expected positive example

```stark
enum Holder {
    Empty,
    Value(String)
}

fn inspect(holder: &Holder) -> UInt64 {
    match *holder {
        Holder::Empty => 0u64,
        Holder::Value(value) => value.len(),
    }
}

fn main() {
    let holder = Holder::Value(String::from("stark"));

    if inspect(&holder) != 5u64 {
        panic("borrowed payload binding failed");
    }

    if inspect(&holder) != 5u64 {
        panic("borrowed payload was consumed");
    }

    println("borrowed-pattern: PASS");
}
```

### Acceptance criteria

- accepted by the CD-275 compiler;
- payload remains available after the first match;
- HIR, MIR, native debug, and native release agree.

---

# 4. Confirmed Compiler Defect: Unicode `chars()` Interpreter Panic

## 4.1 Minimal reproduction

```stark
fn main() {
    let mut text = String::from("Stark");
    text.push('語');

    for _ in text.as_str().chars() {
    }
}
```

## 4.2 Actual result

- `starkc check` succeeds;
- `starkc run` crashes the host interpreter;
- Rust panic:

```text
called Option::unwrap() on a None value
src/interp.rs:5211
```

ASCII-only character iteration works. Appending a non-ASCII character without iterating also works.

## 4.3 Expected result

The program must iterate Unicode scalar values and terminate normally.

A valid STARK program must never terminate through an unchecked Rust `unwrap()` panic.

## 4.4 Likely root cause

The HIR interpreter's `String`/`str` character iterator likely mixes:

- byte indices;
- Unicode scalar indices;
- character-boundary lookup.

The implementation appears to assume that advancing one iteration corresponds to one byte or that a calculated byte offset always identifies a character boundary.

For multi-byte UTF-8 characters, that assumption becomes false and a lookup returns `None`, which is then unwrapped.

## 4.5 Required solution

Inspect the HIR implementation of:

- `String::chars`;
- `str::chars`;
- `CharsIter::next`;
- any helper at or near `interp.rs:5211`.

Replace unchecked extraction with a correct UTF-8 scalar iteration model.

Preferred implementation shapes:

### Option A — Store a byte cursor

Maintain:

```text
source string
current byte offset
```

For each `next()`:

1. return `None` when offset equals byte length;
2. obtain the remaining substring at the current valid boundary;
3. decode the first Unicode scalar;
4. advance by `char.len_utf8()`;
5. return that scalar.

The cursor must always remain on a valid UTF-8 boundary.

### Option B — Store decoded scalar values

At iterator construction:

1. decode the source into a sequence of `Char` values;
2. store an index into that sequence.

This is simpler but may allocate and may not match intended borrow/view semantics.

Option A is preferable if the runtime model already represents string views by byte range.

## 4.6 Safety requirement

No `unwrap()` may remain on a value derived from user-controlled string content unless an immediately preceding invariant proves it cannot be `None`.

Unexpected internal inconsistency must return an interpreter-invariant failure, not panic the host.

## 4.7 Required regression matrix

Add cases for:

```text
empty string
ASCII only
two-byte scalar
three-byte scalar: 語
four-byte scalar: 😀
mixed ASCII and Unicode
multiple consecutive Unicode characters
repeated iteration over the same source
String::chars
str::chars
literal string chars
mutated String followed by chars
```

For each admitted case compare:

```text
HIR
MIR
native debug
native release
```

Expected observations must include exact emitted scalar sequence or exact count.

## 4.8 Acceptance criteria

- no host panic;
- correct scalar count;
- correct scalar values and order;
- source remains valid after shared iteration;
- repeated iteration is deterministic;
- all executable engines agree.

---

# 5. Compile-Fail Suite Problems

## 5.1 Unsupported iterator test is masked by mutability errors

### Affected example

```text
compile_fail/04_unsupported_iterator_adapter.stark
```

### Problem

The case is intended to prove `E0105` for unsupported iterator adapters, but it instead produces:

```text
E0400 mutable method receiver requires a mutable place
```

The program never reaches the intended adapter refusal.

### Root cause

The example creates or invokes the iterator through a receiver shape that independently requires mutability.

### Required solution

Use an exact source form already maintained by:

```text
starkc/tests/adversarial_accepted_surface_audit.rs
```

Do not invent a new form unless it has been verified to isolate `E0105`.

The negative test must:

1. type the iterator receiver correctly;
2. avoid unrelated borrow or mutability failures;
3. invoke exactly one unsupported surface;
4. assert one `E0105` diagnostic.

### Acceptance criteria

- no `E0400`;
- at least one `E0105`;
- diagnostic points at the unsupported adapter call;
- no MIR lowering attempt occurs.

---

## 5.2 Malformed Core trait implementation compiles

### Affected example

```text
compile_fail/03_invalid_core_trait_signature.stark
```

### Source shape

```stark
impl Ord for Point {
    fn cmp(&self, other: &Point) -> Int32 {
        self.x - other.x
    }
}
```

### Expected result after CD-275

Reject with `E0500` because canonical `Ord::cmp` must return `Ordering`.

### Actual result observed

Program compiles and runs.

### Interpretation

One of the following is true:

1. a pre-CD-275 compiler binary was used;
2. Packet B is incomplete;
3. the example does not resolve to the canonical Core `Ord` identity.

### Required investigation

1. verify the compiler commit;
2. rebuild the exact branch;
3. compare the example against `adversarial_trait_impls.rs`;
4. confirm the canonical trait identity is selected;
5. ensure validation runs before execution/lowering.

### Required compiler solution if reproduced on CD-275

The canonical Core-trait descriptor must validate:

- method name;
- receiver mode;
- arity;
- parameter types;
- return type;
- method generics;
- associated items;
- duplicates;
- required-item presence.

`Ord::cmp -> Int32` must be rejected with `E0500`.

### Acceptance criteria

- exact example is rejected;
- return-type mismatch names expected `Ordering` and actual `Int32`;
- no HIR/MIR/native execution occurs.

---

## 5.3 Invalid collection key type compiles

### Affected example

```text
compile_fail/05_invalid_hash_key.stark
```

### Source shape

```stark
struct Key {
    value: Int32
}

fn main() {
    let map: HashMap<Key, Int32> = HashMap::new();
}
```

`Key` implements neither `Hash` nor `Eq`.

### Expected result after CD-275

The instantiation must be rejected because `HashMap<K, V>` requires:

```text
K: Hash + Eq
```

### Actual result observed

Program compiles and runs.

### Interpretation

As with the trait case, first rule out a stale compiler.

### Required compiler solution if reproduced on CD-275

Generic bound enforcement must run whenever the collection type is instantiated, not only when:

- `insert` is called;
- `get` is called;
- a method is resolved.

It must apply to:

- local variable annotations;
- function parameters;
- return types;
- struct/enum fields;
- nested generic types;
- aliases if supported;
- generic function signatures.

### Acceptance criteria

- `HashMap<Key, Int32>` is rejected immediately;
- `HashSet<Key>` is also rejected;
- a type implementing only `Hash` is rejected;
- a type implementing only `Eq` is rejected;
- a type implementing both is accepted.

---

# 6. Package and Module Problems

## 6.1 Package names and dependency aliases must be identifiers

### Affected tree

```text
multi_package/
```

### Problem

Names such as:

```text
feature-model
feature-logic
```

contain hyphens and are rejected.

### Required solution

Use identifier-safe names consistently:

```text
feature_model
feature_logic
feature_app
```

The dependency alias, declared package name, and imported path must agree.

Example dependency declaration:

```json
{
  "dependencies": {
    "feature_model": {
      "package": "feature_model",
      "path": "../model"
    }
  }
}
```

### Acceptance criteria

- all manifests parse;
- declared package name matches dependency `package`;
- `use feature_model::...` resolves;
- the complete application executes and prints:

```text
multi-package: PASS
```

---

## 6.2 Sibling files require modules, not package-self imports

### Affected tree

```text
multi_file_app/
```

### Problem

The suite tried to reach sibling source files using:

```stark
use feature_multifile_app::InvoiceLine;
```

That is not the intra-package module mechanism.

### Required solution

Entry file:

```stark
pub mod model;
pub mod math;
```

Use module-qualified access:

```stark
model::InvoiceLine
math::invoice_total
```

Sibling module imports:

```stark
use crate::model::InvoiceLine;
use crate::model::line_total;
```

Use `pub mod`, not plain `mod`, when the module must be visible through the package root.

### Corrected structure

```text
multi_file_app/
├── starkpkg.json
└── src/
    ├── main.stark
    ├── model.stark
    └── math.stark
```

### Acceptance criteria

- package checks successfully;
- sibling modules resolve through `crate::`;
- application executes;
- exact output:

```text
multi-file-app: PASS
```

---

## 6.3 First-party package consumers use environment-dependent paths

### Affected trees

```text
package_consumers/json_consumer/
package_consumers/base64_consumer/
```

### Problem

Paths such as:

```text
../../../stark-json
../../../stark-base64
```

resolve only when the suite is placed at a particular location relative to the repository.

### Required solution

Choose one of the following:

### Option A — Place examples inside the repository

Recommended layout:

```text
examples/feature-suite/
packages/json_consumer/
packages/base64_consumer/
```

Use paths relative to repository root structure.

### Option B — Parameterised staging script

Create a script that:

1. accepts the STARK repository root;
2. copies the consumer fixture into a temporary directory;
3. rewrites dependency paths to absolute or correct relative locations;
4. runs check/build/execute;
5. deletes the temporary directory.

### Classification

Missing dependency directories are not a language host-resource failure. They are a **test-environment/configuration failure**.

Reserve `HostResourceFailure` for runtime resource exhaustion or host capability failure as defined by the compiler's execution model.

### Acceptance criteria

- package dependency paths resolve deterministically;
- consumers run from CI and developer machines;
- JSON consumer executes parse and encode;
- Base64 consumer executes encode and decode over bytes.

---

# 7. Test-Suite Classification

Every case must declare one expected class:

```text
Complete
FrontendReject
LanguageTrap
HostResourceFailure
InterpreterInvariantFailure
NativeProviderOnly
```

Do not place runtime traps under `compile_fail`.

Recommended directories:

```text
single_file/
compile_fail/
expected_trap/
multi_file/
multi_package/
package_consumers/
provider_native/
```

Each case should carry metadata:

```toml
name = "int32_min_div_neg_one"
class = "LanguageTrap"
expected_category = "IntegerOverflow"
engines = ["hir", "mir", "native-debug", "native-release"]
```

For positive cases, pin:

- stdout;
- stderr;
- exit status.

For trap cases, pin:

- trap category;
- source location/provenance;
- pre-trap stdout;
- pre-trap stderr.

For frontend rejection, pin:

- diagnostic code;
- primary source span;
- no later-stage execution.

---

# 8. Qualification Runner Requirements

The current simple runner only invokes one command and treats exit code zero as success. Replace or extend it to support the full outcome model.

## 8.1 Positive portable cases

Run through:

```text
HIR
MIR
native debug
native release
```

Assert each against an independently pinned expectation, then compare engines.

## 8.2 Compile-fail cases

Run frontend analysis only.

Assert exact diagnostic code, not merely nonzero exit status.

## 8.3 Expected traps

Assert:

- accepted by frontend;
- expected trap category;
- expected provenance;
- no host panic.

## 8.4 Package cases

Run through the package driver using the manifest as the entry point.

Do not invoke package files as isolated standalone files.

## 8.5 Native-provider cases

Keep separate from three-engine portable cases because the MIR interpreter does not execute providers.

---

# 9. Required Work Sequence

## Phase 1 — Establish compiler identity

- verify branch and commit;
- rebuild `starkc`;
- print absolute compiler path.

## Phase 2 — Correct example source

Fix:

- array `.len()`;
- `Box::into_inner`;
- turbofish syntax;
- primitive `to_string`;
- `String` key lookup;
- `Ordering` comparison;
- module declarations;
- package identifiers;
- dependency paths;
- masked iterator negative test.

## Phase 3 — Reproduce compiler findings

On `61d1c18` or later, rerun:

- Unicode `chars()` panic;
- malformed `Ord`;
- invalid HashMap key;
- borrowed payload match.

## Phase 4 — Fix confirmed compiler defects

At minimum:

- Unicode character iterator host panic.

If reproduced on current head:

- Core-trait conformance validation;
- collection key-bound enforcement;
- borrowed payload place-aware matching.

## Phase 5 — Run complete qualification

Run:

- formatting;
- static checks;
- all example cases;
- all package cases;
- HIR/MIR/native debug/native release;
- Tier-1 CI where applicable.

## Phase 6 — Record evidence

Create:

```text
FEATURE_MATRIX.md
RESULTS.md
KNOWN_LIMITATIONS.md
```

Each row must state:

- case;
- expected outcome;
- actual outcome;
- engine coverage;
- qualifying commit.

---

# 10. Final Acceptance Criteria

The suite is ready to become authoritative when:

1. every positive example compiles;
2. every positive portable example agrees across HIR, MIR, native debug, and native release;
3. every compile-fail example emits its intended diagnostic code;
4. every expected trap emits its intended trap category;
5. no valid example causes a Rust panic;
6. multi-file modules use the compiler's real module mechanism;
7. multi-package dependencies use valid identifiers and matching manifest names;
8. first-party package consumers resolve their dependencies deterministically;
9. WP-C7.9-specific tests are executed with a compiler containing CD-275;
10. all results are pinned to an exact commit.

---

# 11. Current Disposition Summary

| Finding | Classification | Disposition |
|---|---|---|
| Array `.len()` | Example defect | Remove unsupported call |
| Box unary dereference | Example defect | Use `into_inner()` |
| Generic call syntax | Example defect | Use `::<T>` |
| Primitive `.to_string()` | Example defect | Avoid unsupported method |
| HashMap `&str` lookup | Example defect | Use retained `String` key |
| `Ordering !=` | Example defect | Use `match` |
| Borrowed enum payload rejection | Version-sensitive compiler finding | Retest on CD-275 |
| Unicode `chars()` host panic | Confirmed compiler defect | Fix UTF-8 cursor/iteration |
| Iterator negative emits E0400 | Test-design defect | Use canonical E0105 fixture |
| Malformed `Ord` accepted | Version-sensitive compiler finding | Retest, then fix if current |
| Invalid HashMap key accepted | Version-sensitive compiler finding | Retest, then fix if current |
| Hyphenated package names | Manifest defect | Use identifiers |
| Sibling-file package imports | Module-model defect | Use `pub mod` and `crate::` |
| Missing package dependency paths | Environment/configuration defect | Stage inside repo or rewrite paths |

---

## Final ruling

The compiler-driven pass served its intended purpose. It separated:

- unsupported example assumptions;
- package/module mistakes;
- branch-version uncertainty;
- and a real interpreter defect.

The suite should now be corrected and rerun against an explicitly identified CD-275 compiler. The Unicode `chars()` panic should be treated as the only confirmed new compiler defect until the trait, collection-bound, and borrowed-pattern cases are reproduced on the target commit.
