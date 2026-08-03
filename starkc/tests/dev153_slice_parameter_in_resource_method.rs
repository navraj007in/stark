//! **DEV-153: `hir_field_ty` had no arm for an unsized slice, so a resource method taking `&[T]`
//! could not lower.**
//!
//! ```text
//! owned.write_all("x".bytes())
//!   -> error: native build does not yet support this program: field type form (C4.5)
//!
//! write_all(&mut owned, "x".bytes())    // the IDENTICAL operation, as a free function
//!   -> builds
//! ```
//!
//! `mir_ty` has had a `Ty::Slice` arm all along. `hir_field_ty` did not, because it only ever
//! converted STRUCT FIELDS and ENUM PAYLOADS — and Core v1 forbids reference-typed fields, so
//! `&[T]` could not reach it. The `_` arm that caught it said only "field type form (C4.5)", naming
//! neither the form nor the type, which is why bisecting it took as long as it did; that message
//! now names the kind.
//!
//! # This is DEV-151's second-order cost, and worth naming as such
//!
//! DEV-151(a) opened method dispatch on a host-resource receiver. That routed a method's DECLARED
//! parameter types through `hir_field_ty` for the first time — and immediately met a form it had
//! never had to handle. **A repair that widens what is reachable will expose whatever the newly
//! reachable path never handled.** That is the cost of the DEV-151 class rather than an argument
//! against fixing it: the alternative is the surface staying unreachable and the gap staying
//! invisible, which is precisely how `set_read_timeout` shipped unbuildable.
//!
//! It also could not be reproduced without a resource. A plain `impl Sink { fn absorb(&mut self,
//! input: &[UInt8]) }` builds fine, because a non-resource method's parameters take a different
//! conversion path. So the test below synthesizes a real provider layer rather than reducing to a
//! struct, and that fidelity is the point.

mod support;

use starkc::mir::lower::lower_program;
use starkc::parser::{parse, ParseMode};
use starkc::provider_derive::derive;
use starkc::provider_registry;
use starkc::provider_resolve::ProviderSet;
use starkc::provider_synth::synthesize_with_resources;
use starkc::resolve::resolve;
use starkc::source::SourceFile;
use starkc::typecheck;
use std::collections::BTreeMap;
use std::sync::Arc;

fn host_triple() -> String {
    starkc::native_toolchain::discover(None)
        .expect("a Rust toolchain is required for provider selection")
        .host_triple
}

fn build(src: &str, tag: &str) -> Result<(), String> {
    let file = Arc::new(SourceFile::new(format!("{tag}.stark"), src.to_string()));
    let (ast, pd) = parse(&file, ParseMode::Program);
    assert!(pd.is_empty(), "{tag}: parse: {pd:?}");
    let (hir, rd) = resolve(&ast, file.clone());
    assert!(rd.is_empty(), "{tag}: resolve: {rd:?}");
    let checked = typecheck::analyze(&hir, file.clone());
    if let Some(first) = checked
        .diagnostics
        .iter()
        .find(|d| d.severity == starkc::diag::Severity::Error)
    {
        return Err(format!(
            "CHECK {} {}",
            first.code.as_deref().unwrap_or("-"),
            first.message
        ));
    }
    let program =
        lower_program(&hir, &checked.tables, file).map_err(|e| format!("LOWER: {}", e.what))?;
    starkc::mir::verify::verify_program(&program).map_err(|errors| {
        format!(
            "VERIFY {}",
            errors
                .iter()
                .map(|e| format!("{} {}", e.code, e.message))
                .collect::<Vec<_>>()
                .join("; ")
        )
    })?;
    Ok(())
}

/// A provider layer declaring a `tcp_stream` resource, so the receiver below really is
/// `MirTy::HostResource` and the method's parameters really route through `hir_field_ty`.
fn provider_layer() -> String {
    let set = ProviderSet::select(
        provider_registry::first_party(),
        &host_triple(),
        &["tcp".into()],
    )
    .expect("tcp provider selects for host");
    let connect = set
        .resolve("tcp", "stark_tcp_stream_connect")
        .expect("connect resolves");
    let connect_sig = derive(
        "connect_raw",
        "tcp",
        &connect.function,
        &BTreeMap::from([("tcp_stream".to_string(), "TcpStream".to_string())]),
        &BTreeMap::from([("tcp".to_string(), "RawNetError".to_string())]),
    )
    .expect("connect signature derives");
    synthesize_with_resources(
        &[connect_sig],
        &BTreeMap::from([("tcp".to_string(), connect.status_binding.clone())]),
        &BTreeMap::from([("tcp_stream".to_string(), "TcpStream".to_string())]),
        &BTreeMap::new(),
    )
    .expect("resource nominal and free connect binding synthesize")
    .source
}

/// **The reproducer**, reduced from `stark-net`'s `TcpStream::write_all`.
#[test]
fn a_resource_method_may_take_a_shared_slice() {
    let source = format!(
        "{}\n\
         impl TcpStream {{\n\
         \x20   fn send(&mut self, input: &[UInt8]) -> Int32 {{ input.len() as Int32 }}\n\
         }}\n\
         fn main() {{ }}\n",
        provider_layer()
    );
    if let Err(why) = build(&source, "sliceparam") {
        panic!(
            "a resource method taking `&[UInt8]` must lower — the identical free function always \
             did, and the asymmetry was invisible until DEV-151 opened method dispatch. Got: {why}"
        );
    }
}

/// A MUTABLE slice, the `read(&mut self, output: &mut [UInt8])` shape. `&mut [T]` reaches the same
/// conversion through the `Ref` arm, so both mutabilities are pinned rather than assumed.
#[test]
fn a_resource_method_may_take_a_mutable_slice() {
    let source = format!(
        "{}\n\
         impl TcpStream {{\n\
         \x20   fn recv(&mut self, output: &mut [UInt8]) -> Int32 {{ output.len() as Int32 }}\n\
         }}\n\
         fn main() {{ }}\n",
        provider_layer()
    );
    assert_eq!(build(&source, "mutsliceparam"), Ok(()));
}

/// A slice nested inside another form — the recursion under `Ref` must reach it too, not just a
/// slice sitting directly in parameter position.
#[test]
fn a_slice_nested_under_a_reference_lowers() {
    let source = format!(
        "{}\n\
         impl TcpStream {{\n\
         \x20   fn sized(&mut self, a: &[UInt8], b: &[UInt64]) -> Int32 {{\n\
         \x20       (a.len() + b.len()) as Int32\n\
         \x20   }}\n\
         }}\n\
         fn main() {{ }}\n",
        provider_layer()
    );
    assert_eq!(build(&source, "twoslices"), Ok(()));
}

/// The non-resource path was never broken and must stay unbroken: this is the control that says
/// the repair widened one conversion rather than changing how ordinary methods lower.
#[test]
fn an_ordinary_struct_method_taking_a_slice_still_lowers() {
    assert_eq!(
        build(
            "struct Sink { total: UInt64 }\n\
             impl Sink {\n\
             \x20   fn absorb(&mut self, input: &[UInt8]) -> UInt64 {\n\
             \x20       let mut i = 0u64;\n\
             \x20       while i < input.len() {\n\
             \x20           self.total = self.total + input[i] as UInt64;\n\
             \x20           i = i + 1u64;\n\
             \x20       }\n\
             \x20       self.total\n\
             \x20   }\n\
             }\n\
             fn main() {\n\
             \x20   let mut sink = Sink { total: 0u64 };\n\
             \x20   println(sink.absorb(\"AB\".bytes()));\n\
             }\n",
            "ordinaryslice",
        ),
        Ok(())
    );
}
