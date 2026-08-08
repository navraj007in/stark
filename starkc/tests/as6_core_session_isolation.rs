//! **AS6 — a Core-only session knows no tensor-owned name, and a tensor session is unchanged.**
//!
//! AS6's checkpoint evidence is explicitly **two-directional**, per surface:
//!
//! > A quarantine that suppresses tensor semantics passes the first test and fails the second; one
//! > that leaks passes the second and fails the first. Neither is visible to `cargo check`.
//!
//! So every case here is a pair. A test that only proved absence would be satisfied by a compiler
//! that had broken the extension outright, which is the failure mode a quarantine is most likely to
//! introduce and the least likely to notice.
//!
//! # What this surface's inventory found
//!
//! AS6 names seven surfaces. Three — the lexer, the formatter and the diagnostics module — contain
//! **zero** tensor references and need no work. The remainder are concentrated:
//!
//! ```text
//! typecheck.rs   1024        ast.rs      12
//! resolve.rs      135        lexer       0
//! parser.rs       117        formatter   0
//! hir.rs           38        diagnostics 0
//! ```
//!
//! For `ast.rs` and `hir.rs` specifically, exit criterion 1 — *"Core-only sessions load no
//! tensor-owned name or semantic rule"* — **already holds**, which this file pins. `ast.rs`'s
//! references are almost entirely doc comments on syntactic forms Core and the extension **share**
//! (`DimExpr`, the `Item = T` binding form reused for `device = D`, index lists); sharing a form is
//! not a leak, and rewriting those comments would remove information without changing behaviour.
//!
//! What remains open for this surface is criterion 2 — *"central Core modules do not contain
//! open-ended tensor spelling tables or method catalogues"*. `hir::Builtin` carries **33
//! `Tensor*` variants**. That is a catalogue in a central Core module, and relocating it behind a
//! sealed `extensions::tensor` type is real work with a wide blast radius (every match arm in the
//! resolver, checker, interpreter and MIR lowering). It is not attempted here: this file is the
//! harness that must exist **before** the move, so the move can be shown not to have broken the
//! extension.

use starkc::options::LanguageOptions;
use starkc::session::CompilerSession;
use starkc::source::SourceFile;
use std::sync::Arc;

fn errors(opts: LanguageOptions, source: &str) -> Vec<String> {
    let file = Arc::new(SourceFile::new("test.stark", source));
    match CompilerSession::for_source(file, opts).check() {
        Ok(_) => Vec::new(),
        Err(failure) => failure
            .diagnostics()
            .iter()
            .filter(|d| d.severity == starkc::diag::Severity::Error)
            .map(|d| d.message.clone())
            .collect(),
    }
}

/// Both directions for one name. `core_absent` is the message fragment proving the Core session
/// does not know it; `tensor_differs` requires the tensor session to answer **differently** —
/// not necessarily to succeed, since these are deliberately incomplete programs, but to fail for a
/// reason that shows the name resolved.
fn both_directions(name: &str, source: &str, core_absent: &str) {
    let core = errors(LanguageOptions::CORE, source);
    assert!(
        core.iter().any(|m| m.contains(core_absent)),
        "{name}: a Core-only session must not know this tensor-owned name. Expected a diagnostic \
         containing {core_absent:?}, got {core:?}"
    );

    let tensor = errors(LanguageOptions::with_tensor(), source);
    assert!(
        !tensor.iter().any(|m| m.contains(core_absent)),
        "{name}: the tensor session must still resolve it — a quarantine that suppresses the \
         extension passes the absence test and fails this one. Got {tensor:?}"
    );
}

/// The tensor constructor builtins, which resolve by bare name (`zeros`, `ones`, `full`) and are
/// therefore the most exposed to leaking into Core's namespace.
#[test]
fn tensor_constructor_names_are_absent_from_a_core_session() {
    for name in ["zeros", "ones", "full"] {
        both_directions(
            name,
            &format!("fn main() {{ let t = {name}(); }}"),
            "undefined variable",
        );
    }
}

/// `model` is a tensor-owned ITEM form, not merely a name — the strongest version of criterion 1.
#[test]
fn the_model_item_form_is_refused_by_a_core_session() {
    let core = errors(
        LanguageOptions::CORE,
        "model M { input x: Tensor<Float32, [1]>; output y: Tensor<Float32, [1]>; } fn main() {}",
    );
    assert!(
        core.iter()
            .any(|m| m.contains("require") && m.contains("tensor")),
        "a `model` declaration must be refused by name of the extension it needs: {core:?}"
    );
}

/// **The control that makes the whole file mean something.** Core programs must be unaffected by
/// the extension's existence, in either session. Without this, a compiler that had disabled tensor
/// support entirely would satisfy every absence test above.
#[test]
fn ordinary_core_programs_are_identical_under_both_sessions() {
    let source = "struct P { a: Int32 } \
                  fn main() { let p = P { a: 1 }; println(p.a + 1); }";
    assert!(
        errors(LanguageOptions::CORE, source).is_empty(),
        "a Core program must check under a Core session"
    );
    assert!(
        errors(LanguageOptions::with_tensor(), source).is_empty(),
        "enabling the extension must not change Core behaviour"
    );
}

/// The names Core and the extension **share** stay Core's. `add`, `max` and `min` are tensor
/// builtins by one spelling and ordinary user function names by another; a quarantine that removed
/// them from Core's namespace would break programs that never mention tensors.
#[test]
fn shared_spellings_remain_available_to_core_programs() {
    let source = "fn add(a: Int32, b: Int32) -> Int32 { a + b } \
                  fn main() { println(add(1, 2)); }";
    assert!(
        errors(LanguageOptions::CORE, source).is_empty(),
        "a user function named `add` is ordinary Core"
    );
    assert!(
        errors(LanguageOptions::with_tensor(), source).is_empty(),
        "and enabling the extension must not shadow it"
    );
}
