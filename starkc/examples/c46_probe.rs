// C4.6 audit probe: reads a STARK program from a file, reports where the pipeline stops:
// PARSE-ERR / RESOLVE-ERR / TYPECHECK-ERR (front-end scope) vs LOWER-UNSUPPORTED (C4 gap)
// vs VERIFY-ERR vs OK (+ MIR-interp run outcome).
use starkc::mir::interp::run_program;
use starkc::mir::lower::lower_program;
use starkc::mir::verify::verify_program;
use starkc::parser::{parse, ParseMode};
use starkc::resolve::resolve;
use starkc::source::SourceFile;
use starkc::typecheck;
use std::sync::Arc;

fn main() {
    let path = std::env::args().nth(1).unwrap();
    let src = std::fs::read_to_string(&path).unwrap();
    let file = Arc::new(SourceFile::new("probe.stark", src));
    let (ast, pd) = parse(&file, ParseMode::Program);
    if !pd.is_empty() {
        println!("PARSE-ERR: {}", pd[0].message);
        return;
    }
    let (hir, rd) = resolve(&ast, file.clone());
    if !rd.is_empty() {
        println!("RESOLVE-ERR: {}", rd[0].message);
        return;
    }
    let checked = typecheck::analyze(&hir, file.clone());
    if let Some(d) = checked
        .diagnostics
        .iter()
        .find(|d| d.severity == starkc::diag::Severity::Error)
    {
        println!(
            "TYPECHECK-ERR: [{}] {}",
            d.code.as_deref().unwrap_or(""),
            d.message
        );
        return;
    }
    let program = match lower_program(&hir, &checked.tables, file.clone()) {
        Ok(p) => p,
        Err(e) => {
            println!("LOWER-UNSUPPORTED: {}", e.what);
            return;
        }
    };
    match verify_program(&program) {
        Ok(v) => match run_program(v) {
            Ok(exec) => println!("OK: ran, stdout={:?}", exec.output),
            Err(f) => println!("MIR-RUN-ERR: {:?}", f.error),
        },
        Err(errs) => println!("VERIFY-ERR: {:?}", errs.first()),
    }
}
