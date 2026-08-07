//! **DEV-151: methods on a host-resource receiver lower, and `()` is `MirTy::Unit` everywhere.**
//!
//! Two defects, found together because the first concealed the second.
//!
//! # (a) A resource receiver was not treated as a nominal
//!
//! `lower_method_call` extracts `(nominal, args)` from the receiver type and refused anything that
//! was not a struct or user enum:
//!
//! ```text
//! error: native build does not yet support this program: method call on non-nominal receiver
//!        HostResource(HostResourceTy { nominal: Item(ItemId(381)), provider: "stark-std-net",
//!        resource: "tcp_stream" }) (C4.5b+)
//! ```
//!
//! But a host resource IS a nominal: `HostResourceTy.nominal` holds the item of the synthesized
//! zero-variant enum the package declared (CD-234), and `impl TcpStream { fn set_read_timeout(..) }`
//! hangs off exactly that item. The refusal was a missing match arm, not a missing capability.
//!
//! **What it cost.** CD-346 ruled that a resource operation which consumes bytes or moves a cursor
//! takes `&mut self`, and `stark-net` declared `set_read_timeout`/`set_write_timeout` as methods on
//! that ruling. The package qualified. The methods had never been CALLED by anything native, so
//! nobody learned that calling one could not build. This is CD-345's lesson one level down: CD-347
//! made a package's resource LIFECYCLE executable, and this was a declared surface whose
//! *call sites* were still unexecuted. `stark-http-client` was the first caller, and it failed
//! immediately.
//!
//! # (b) `()` written in source lowered to `Tuple([])`, which nothing else produces
//!
//! Uncovered the moment (a) let those method bodies lower at all:
//!
//! ```text
//! MIR-0004 stark_net::TcpStream::set_read_timeout@[] bb26: assignment:
//!   expected Enum(CoreResult, [Tuple([]), ..]), found Enum(CoreResult, [Unit, ..])
//! ```
//!
//! MIR has ONE canonical unit type, `MirTy::Unit`, and every synthesized site uses it — the empty
//! tuple is never constructed deliberately anywhere in the compiler. Only a written-out `()` in a
//! type annotation reached the tuple arm, so `fn f() -> Result<(), E>` declared a return type that
//! no constructed value could ever match. Both conversion sites (`mir_ty`, `hir_field_ty`) now
//! canonicalise the empty tuple to `Unit`.
//!
//! That this sat undetected says something worth keeping: `Result<(), E>` is an extremely common
//! signature, and it was fine everywhere the body was never lowered. Two unexecuted paths crossing
//! is what made it reachable.

mod support;

use starkc::mir::lower::lower_program;
use starkc::mir::MirTy;
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

fn build(src: &str, tag: &str) -> Result<starkc::mir::MirProgram, String> {
    let file = Arc::new(SourceFile::new(format!("{tag}.stark"), src.to_string()));
    let (ast, pd) = parse(&file, ParseMode::Program);
    assert!(pd.is_empty(), "{tag}: parse: {pd:?}");
    let (hir, rd) = resolve(&ast, file.clone());
    assert!(rd.is_empty(), "{tag}: resolve: {rd:?}");
    let checked = typecheck::analyze(&hir);
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
    let program = lower_program(
        &hir,
        &checked.tables,
        hir.source_named(&file.name).expect("registered"),
    )
    .map_err(|e| format!("LOWER: {}", e.what))?;
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
    Ok(program)
}

// ------------------------------------------------------------ (b) unit canonicalisation --

/// The reproducer for (b), which needs no resource at all — it was simply unreachable until (a)
/// was fixed, because no `-> Result<(), E>` body had ever been lowered from a written-out `()`.
#[test]
fn a_written_unit_return_type_builds() {
    assert_eq!(
        build(
            "enum Failure { Bad }\n\
             fn run(flag: Bool) -> Result<(), Failure> {\n\
             \x20   if flag { Ok(()) } else { Err(Failure::Bad) }\n\
             }\n\
             fn main() {\n\
             \x20   match run(true) {\n\
             \x20       Ok(_v) => { println(1); }\n\
             \x20       Err(_e) => { println(0); }\n\
             \x20   }\n\
             }\n",
            "unitreturn",
        )
        .map(|_| ()),
        Ok(())
    );
}

/// `()` as a plain return type, as a bare local annotation, and nested inside a generic — every
/// route into the type conversion must agree, since disagreement is the whole defect.
#[test]
fn written_unit_agrees_with_synthesized_unit_everywhere() {
    assert_eq!(
        build(
            "fn nothing() -> () {}\n\
             fn also_nothing() {}\n\
             fn wrapped() -> Option<()> { Some(()) }\n\
             fn main() {\n\
             \x20   let a: () = nothing();\n\
             \x20   let b: () = also_nothing();\n\
             \x20   let _pair = (a, b);\n\
             \x20   match wrapped() {\n\
             \x20       Some(_v) => { println(1); }\n\
             \x20       None => { println(0); }\n\
             \x20   }\n\
             }\n",
            "unitroutes",
        )
        .map(|_| ()),
        Ok(())
    );
}

/// The canonicalisation as a direct structural claim: no lowered body may contain `Tuple([])`.
/// Asserted over the MIR rather than through a build, because a build only fails where the two
/// representations happen to MEET — this catches one that has not met anything yet.
#[test]
fn no_lowered_body_contains_an_empty_tuple_type() {
    let program = build(
        // `fn f(v: ()) -> () { v }` is deliberately NOT here: returning a unit-typed PATH is an
        // unrelated lowering limit ("unit expression form (C4.5): Path") that predates this repair,
        // and folding it in would make this test fail for a reason it does not describe.
        "enum Failure { Bad }\n\
         fn run() -> Result<(), Failure> { Ok(()) }\n\
         fn takes_unit(_v: ()) -> Int32 { 1 }\n\
         fn main() {\n\
         \x20   let _r = run();\n\
         \x20   println(takes_unit(()));\n\
         }\n",
        "noemptytuple",
    )
    .expect("expected a clean build");

    fn mentions_empty_tuple(ty: &MirTy) -> bool {
        match ty {
            MirTy::Tuple(elems) => elems.is_empty() || elems.iter().any(mentions_empty_tuple),
            MirTy::Enum(_, args) | MirTy::Core(_, args) => args.iter().any(mentions_empty_tuple),
            MirTy::Struct(_, args) => args.iter().any(mentions_empty_tuple),
            MirTy::Ref { inner, .. } => mentions_empty_tuple(inner),
            MirTy::Array(elem, _) | MirTy::Slice(elem) => mentions_empty_tuple(elem),
            _ => false,
        }
    }

    for body in &program.bodies {
        // Return and parameter types too, not just locals: the defect entered through a written
        // SIGNATURE, so the signature is where it must be pinned.
        for ty in std::iter::once(&body.ret)
            .chain(body.params.iter())
            .chain(body.locals.iter().map(|local| &local.ty))
        {
            assert!(
                !mentions_empty_tuple(ty),
                "{:?}: type {ty:?} carries an empty tuple; `()` must canonicalise to MirTy::Unit, \
                 and a `Tuple([])` here would silently fail to match any constructed unit value",
                body.instance
            );
        }
    }
}

// -------------------------------------------------------- (a) resource method dispatch --

/// A method on a host-resource receiver lowers. The receiver really is `MirTy::HostResource` —
/// the synthesized provider layer declares the nominal (CD-234), which is the exact type the
/// refusal named.
///
/// No live handle is needed: a resource-typed PARAMETER reaches the same dispatch path, and
/// lowering visits every body. The end-to-end evidence — `stark-http-client` calling
/// `set_read_timeout` on a real socket — is `stark-http-client-consumer` under the qualification
/// gate's HTTP peer. What this pins is that the dispatch RESOLVES, which is what was missing.
#[test]
fn a_method_on_a_resource_receiver_lowers() {
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
    let layer = synthesize_with_resources(
        &[connect_sig],
        &BTreeMap::from([("tcp".to_string(), connect.status_binding.clone())]),
        &BTreeMap::from([("tcp_stream".to_string(), "TcpStream".to_string())]),
        &BTreeMap::new(),
    )
    .expect("resource nominal and free connect binding synthesize");

    // Both mutabilities of receiver, because CD-346 rules that resource APIs use BOTH: `&mut self`
    // for anything that moves a cursor or consumes bytes, `&self` for the purely observational.
    let source = format!(
        "{}\n\
         impl TcpStream {{\n\
         \x20   fn observe(&self) -> Int32 {{ 1 }}\n\
         \x20   fn adjust(&mut self) -> Int32 {{ 2 }}\n\
         }}\n\
         fn use_shared(stream: &TcpStream) -> Int32 {{ stream.observe() }}\n\
         fn use_exclusive(stream: &mut TcpStream) -> Int32 {{ stream.adjust() }}\n\
         fn main() {{ }}\n",
        layer.source
    );

    match build(&source, "resourcemethod") {
        Ok(_) => {}
        Err(why) => panic!(
            "a method on a host-resource receiver must lower — a resource nominal IS a nominal, \
             and refusing made CD-346's `&mut self` ruling unbuildable at every call site. Got: \
             {why}"
        ),
    }
}
