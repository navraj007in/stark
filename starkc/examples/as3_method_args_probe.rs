//! AS3 Boundary 4 — does a trait method's OWN generics survive a bound call site?
//!
//! G2 recorded that `x.to::<Int32>()` inside `fn g<T: Conv>` is accepted, and that
//! `CalleeSelection::Bound` carries no `method_args`. Before populating that field, measure what
//! the checker does with the method's generics today — `check_trait_member_call` ignores
//! `sig.generics` entirely, so it is not obvious whether the accepted program is correct or merely
//! unchecked. Measures; asserts nothing.
use starkc::options::LanguageOptions;
use starkc::session::CompilerSession;
use std::sync::Arc;

fn probe(name: &str, src: &str) {
    let file = Arc::new(starkc::source::SourceFile::new("probe.stark", src));
    match CompilerSession::for_source(file, LanguageOptions::CORE).check() {
        Err(f) => {
            let r = f.render();
            let first = r.lines().next().unwrap_or("").to_string();
            println!("{name:<40} REJECTED  {first}");
        }
        Ok(program) => {
            let bound: Vec<String> = program
                .tables()
                .callable_uses
                .iter()
                .filter_map(|u| match &u.selection {
                    starkc::typecheck::CalleeSelection::Bound {
                        member,
                        method_args,
                        ..
                    } => Some(format!("{member}<{} args>", method_args.len())),
                    _ => None,
                })
                .collect();
            match program.execute_hir() {
                Ok(e) => println!(
                    "{name:<40} OK  out={:<12} bound={bound:?}",
                    format!("{:?}", e.output)
                ),
                Err(e) => println!("{name:<40} TRAP  {}", e.message),
            }
        }
    }
}

fn main() {
    // A trait method with its OWN generic parameter, called through a bound.
    let conv = "trait Conv {\n    fn to<U>(&self, x: U) -> U;\n}\n\
                struct C { v: Int32 }\n\
                impl Conv for C {\n    fn to<U>(&self, x: U) -> U { x }\n}\n";

    probe(
        "turbofish supplied",
        &format!(
            "{conv}fn g<T: Conv>(t: T) -> Int32 {{ t.to::<Int32>(1) }}\n\
             fn main() {{ let c: C = C {{ v: 0 }}; println(g(c)); }}\n"
        ),
    );

    probe(
        "turbofish omitted, inferable from arg",
        &format!(
            "{conv}fn g<T: Conv>(t: T) -> Int32 {{ t.to(1) }}\n\
             fn main() {{ let c: C = C {{ v: 0 }}; println(g(c)); }}\n"
        ),
    );

    // The interesting one: does the checker CATCH a turbofish that contradicts the argument?
    probe(
        "turbofish CONTRADICTS the argument",
        &format!(
            "{conv}fn g<T: Conv>(t: T) -> Bool {{ t.to::<Bool>(1) }}\n\
             fn main() {{ let c: C = C {{ v: 0 }}; println(g(c)); }}\n"
        ),
    );

    // And does it catch a return used at the wrong type?
    probe(
        "result used at the WRONG type",
        &format!(
            "{conv}fn g<T: Conv>(t: T) -> Bool {{ t.to::<Int32>(1) }}\n\
             fn main() {{ let c: C = C {{ v: 0 }}; println(g(c)); }}\n"
        ),
    );

    // Arity: the method declares one generic; supply two.
    probe(
        "turbofish arity WRONG (2 for 1)",
        &format!(
            "{conv}fn g<T: Conv>(t: T) -> Int32 {{ t.to::<Int32, Bool>(1) }}\n\
             fn main() {{ let c: C = C {{ v: 0 }}; println(g(c)); }}\n"
        ),
    );

    // Control: the same shapes on a CONCRETE receiver, which goes down the selected-impl path
    // that WP-C4.7-8.4 already fixed. Any divergence below is the bound path's gap, not the
    // language's rule.
    probe(
        "CONCRETE turbofish contradicts arg",
        &format!("{conv}fn main() {{ let c: C = C {{ v: 0 }}; println(c.to::<Bool>(1)); }}\n"),
    );
    probe(
        "CONCRETE arity WRONG (2 for 1)",
        &format!(
            "{conv}fn main() {{ let c: C = C {{ v: 0 }}; println(c.to::<Int32, Bool>(1)); }}\n"
        ),
    );

    // Where exactly is the boundary? G2 recorded an ACCEPTED case, so `U` cannot be rigid in every
    // position. Vary where the method's own generic appears in the signature.
    println!();
    for (label, decl, body) in [
        (
            "U unused in the signature",
            "fn to<U>(&self) -> Int32;",
            "fn to<U>(&self) -> Int32 { 1 }",
        ),
        (
            "U in RETURN only",
            "fn to<U>(&self) -> U;",
            "fn to<U>(&self) -> U { panic(\"x\") }",
        ),
        (
            "U in PARAM only",
            "fn to<U>(&self, x: U) -> Int32;",
            "fn to<U>(&self, x: U) -> Int32 { 1 }",
        ),
        (
            "U in param, arg is a VARIABLE",
            "fn to<U>(&self, x: U) -> Int32;",
            "fn to<U>(&self, x: U) -> Int32 { 1 }",
        ),
    ] {
        let src = format!(
            "trait Conv {{\n    {decl}\n}}\nstruct C {{ v: Int32 }}\nimpl Conv for C {{\n    {body}\n}}\n"
        );
        let call = if label.contains("VARIABLE") {
            "let n: Int32 = 1; t.to::<Int32>(n)"
        } else if label.contains("PARAM") {
            "t.to::<Int32>(1)"
        } else {
            "t.to::<Int32>()"
        };
        probe(
            label,
            &format!(
                "{src}fn g<T: Conv>(t: T) -> Int32 {{ {call} }}\n\
                 fn main() {{ let c: C = C {{ v: 0 }}; println(g(c)); }}\n"
            ),
        );
    }
}
