//! AS3 Boundary 4 — Display characterization. Measures; asserts nothing.
use starkc::options::LanguageOptions;
use starkc::session::CompilerSession;
use std::sync::Arc;

fn probe(name: &str, src: &str) {
    let file = Arc::new(starkc::source::SourceFile::new("probe.stark", src));
    match CompilerSession::for_source(file, LanguageOptions::CORE).check() {
        Err(f) => {
            let r = f.render();
            let first = r.lines().next().unwrap_or("").to_string();
            println!("{name:<34} REJECTED  {first}");
        }
        Ok(program) => {
            let uses = program.tables().callable_uses.len();
            let display_uses = program
                .tables()
                .callable_uses
                .iter()
                .filter(|u| {
                    matches!(
                        u.provenance,
                        starkc::typecheck::DispatchProvenance::CoreTrait {
                            core: starkc::hir::CoreTrait::Display
                        }
                    )
                })
                .count();
            match program.execute_hir() {
                Ok(e) => println!(
                    "{name:<34} OK  out={:<22} uses={uses} display_uses={display_uses}",
                    format!("{:?}", e.output)
                ),
                Err(e) => println!("{name:<34} TRAP  {}", e.message),
            }
        }
    }
}

const D: &str = "struct A { v: Int32 }\n\
impl Display for A {\n    fn fmt(&self) -> String {\n        String::from(\"A!\")\n    }\n}\n\
struct B { v: Int32 }\n\
impl Display for B {\n    fn fmt(&self) -> String {\n        String::from(\"B!\")\n    }\n}\n\
struct W<T> { v: T }\n\
impl<T> Display for W<T> {\n    fn fmt(&self) -> String {\n        String::from(\"W!\")\n    }\n}\n";

fn main() {
    probe(
        "1 top-level nominal",
        &format!("{D}fn main() {{\n    let a: A = A {{ v: 1 }};\n    println(a);\n}}\n"),
    );
    probe("2 tuple, two nominals", &format!("{D}fn main() {{\n    let a: A = A {{ v: 1 }};\n    let b: B = B {{ v: 2 }};\n    println((a, b));\n}}\n"));
    probe("3 same generic, two insts", &format!("{D}fn main() {{\n    let x: W<Int32> = W {{ v: 1 }};\n    let y: W<Bool> = W {{ v: true }};\n    println((x, y));\n}}\n"));
    probe("4 Vec repeated", &format!("{D}fn main() {{\n    let mut v: Vec<A> = Vec::new();\n    v.push(A {{ v: 1 }});\n    v.push(A {{ v: 2 }});\n    println(v);\n}}\n"));
    probe(
        "5 Option",
        &format!(
            "{D}fn main() {{\n    let o: Option<A> = Some(A {{ v: 1 }});\n    println(o);\n}}\n"
        ),
    );
    probe(
        "5 Result",
        &format!(
            "{D}fn main() {{\n    let r: Result<A, B> = Ok(A {{ v: 1 }});\n    println(r);\n}}\n"
        ),
    );
    probe("6 GATE generic T: Display", &format!("{D}fn show<T: Display>(x: T) {{\n    println(x);\n}}\nfn main() {{\n    show(A {{ v: 1 }});\n}}\n"));
    probe(
        "7 nominal inside generic W",
        &format!(
            "{D}fn main() {{\n    let w: W<A> = W {{ v: A {{ v: 1 }} }};\n    println(w);\n}}\n"
        ),
    );
    probe("8 tuple of same nominal", &format!("{D}fn main() {{\n    let a: A = A {{ v: 1 }};\n    let b: A = A {{ v: 2 }};\n    println((a, b));\n}}\n"));
    probe("6 GATE generic pair", &format!("{D}fn show2<P: Display, Q: Display>(x: P, y: Q) {{\n    println((x, y));\n}}\nfn main() {{\n    show2(A {{ v: 1 }}, B {{ v: 2 }});\n}}\n"));

    // AS3 Boundary 4: characterize before assuming.
    probe(
        "G1 trait default via bound",
        "trait Describe {\n    fn text(&self) -> String {\n        String::from(\"default\")\n    }\n}\nstruct A2 { v: Int32 }\nimpl Describe for A2 {}\nfn f<T: Describe>(x: T) -> String {\n    x.text()\n}\nfn main() {\n    println(f(A2 { v: 1 }));\n}\n",
    );
    probe(
        "G2 method generics via bound",
        "trait Conv {\n    fn to<U>(&self) -> Int32;\n}\nstruct A3 { v: Int32 }\nimpl Conv for A3 {\n    fn to<U>(&self) -> Int32 {\n        1\n    }\n}\nfn g<T: Conv>(x: T) -> Int32 {\n    x.to::<Int32>()\n}\nfn main() {\n    println(g(A3 { v: 1 }));\n}\n",
    );
}
