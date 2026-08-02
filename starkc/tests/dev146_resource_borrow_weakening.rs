//! **DEV-146: `&mut R` weakens to `&R` at a provider-call boundary, so `&mut` is expressible over
//! a bound resource.**
//!
//! `AbiParam::HandleBorrowed` derives a SHARED reference — a borrowed handle leaves ownership with
//! the caller. That is an ABI fact and does not change. The defect was that the `HandleBorrowed`
//! arm of `lower_provider_call` pushed its operand with NO expected-type coercion, so a package
//! wrapper declaring `fn write(stream: &mut TcpStream, ..)` and forwarding to the raw binding
//! passed `&mut` where `&` was wanted. The front end accepted it; MIR-0005 rejected it:
//!
//! ```text
//! call argument: expected Ref { mutable: false, inner: HostResource(.. "tcp_stream") },
//!                found    Ref { mutable: true,  inner: HostResource(.. "tcp_stream") }
//! ```
//!
//! Accepted-but-unbuildable — the DEV-132/DEV-133 class. DEV-133's own comment predicted it: it
//! routed six coercion sites through `weaken_ref_to` and warned that "whichever site was forgotten
//! would keep this defect". Provider calls were the seventh, and were forgotten invisibly, because
//! no first-party package called a resource function until `stark-net` did.
//!
//! # The ruling this encodes (CD-346)
//!
//! The ABI's shared-borrow derivation and the package's declared signature **need not match**. The
//! compiler weakens, so a package may declare the stricter surface its semantics deserve:
//!
//! * a resource operation that CONSUMES or PRODUCES bytes, or moves a cursor, takes `&mut` —
//!   otherwise a caller may hold two readers of one stream and byte-consumption order stops being
//!   answerable at the call site;
//! * a purely observational operation stays `&`;
//! * neither choice changes what crosses the ABI, which is always the shared form.
//!
//! This is settled once, here, rather than per package: io v0.2 streams, signals, process handles
//! and crypto keys all face the same question.
//!
//! # Why the negative control is the important half
//!
//! Weakening must run in ONE direction. If `&R` could satisfy a `&mut R` parameter, the repair
//! would have handed out exclusive access from a shared borrow — a real aliasing hole, and a much
//! worse defect than the one being fixed.

mod support;

use starkc::parser::{parse, ParseMode};
use starkc::resolve::resolve;
use starkc::source::SourceFile;
use starkc::typecheck;
use std::sync::Arc;

// The end-to-end path — a package calling `stark-net`'s `&mut` wrappers, building, and talking to
// a real listener — is covered by `stark-net`'s own qualification and by the lifecycle e2e. What
// this file pins is the COERCION RULE itself, which is what the repair changed and what every
// future resource package depends on.

fn check_only(src: &str, tag: &str) -> Option<String> {
    let file = Arc::new(SourceFile::new(format!("{tag}.stark"), src.to_string()));
    let (ast, pd) = parse(&file, ParseMode::Program);
    assert!(pd.is_empty(), "{tag}: parse: {pd:?}");
    let (hir, rd) = resolve(&ast, file.clone());
    assert!(rd.is_empty(), "{tag}: resolve: {rd:?}");
    let checked = typecheck::analyze(&hir, file.clone());
    checked
        .diagnostics
        .iter()
        .find(|d| d.severity == starkc::diag::Severity::Error)
        .map(|d| format!("{} {}", d.code.as_deref().unwrap_or("-"), d.message))
}

/// **The negative control, and the important half.** Weakening runs in ONE direction. A shared
/// borrow must never satisfy a `&mut` parameter — that would hand out exclusive access from a
/// shared borrow, a worse defect than the one repaired.
#[test]
fn a_shared_borrow_does_not_satisfy_a_mutable_parameter() {
    let diagnostic = check_only(
        "struct Holder { value: Int32 }\n\
         fn takes_mut(h: &mut Holder) -> Int32 { h.value }\n\
         fn main() {\n\
         \x20   let holder = Holder { value: 1 };\n\
         \x20   let shared = &holder;\n\
         \x20   println(takes_mut(shared));\n\
         }\n",
        "sharednotmut",
    );
    let diagnostic = diagnostic.expect("a shared borrow must not satisfy a `&mut` parameter");
    assert!(
        diagnostic.contains("mutability") || diagnostic.starts_with("E0001"),
        "expected a mutability mismatch, got: {diagnostic}"
    );
}

/// The ordinary direction still works for a non-resource type, so the repair did not disturb the
/// six sites that already routed through `weaken_ref_to`.
#[test]
fn a_mutable_borrow_still_satisfies_a_shared_parameter() {
    assert_eq!(
        check_only(
            "struct Holder { value: Int32 }\n\
             fn takes_shared(h: &Holder) -> Int32 { h.value }\n\
             fn main() {\n\
             \x20   let mut holder = Holder { value: 1 };\n\
             \x20   let exclusive = &mut holder;\n\
             \x20   println(takes_shared(exclusive));\n\
             }\n",
            "muttoshared",
        ),
        None
    );
}

/// A `&mut` receiver forwarding to a `&` parameter through a method, which is the shape
/// `stark-net`'s `TcpStream::write` has.
#[test]
fn a_mutable_receiver_forwards_to_a_shared_parameter() {
    assert_eq!(
        check_only(
            "struct Holder { value: Int32 }\n\
             fn takes_shared(h: &Holder) -> Int32 { h.value }\n\
             impl Holder {\n\
             \x20   fn forward(&mut self) -> Int32 { takes_shared(self) }\n\
             }\n\
             fn main() {\n\
             \x20   let mut holder = Holder { value: 1 };\n\
             \x20   println(holder.forward());\n\
             }\n",
            "receiverforward",
        ),
        None
    );
}
