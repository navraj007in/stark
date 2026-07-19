// A2 probe: run the HIR oracle alone and print its outcome.
use starkc::interp;
use starkc::parser::{parse, ParseMode};
use starkc::resolve::resolve;
use starkc::source::SourceFile;
use starkc::typecheck;
use std::sync::Arc;

fn main() {
    let src = std::fs::read_to_string(std::env::args().nth(1).unwrap()).unwrap();
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
    match interp::run_with_partial_output(&hir, file, &checked.tables) {
        Ok(exec) => println!("ORACLE-OK: {:?}", exec.output),
        Err((e, partial)) => println!("ORACLE-ERR: {:?} (partial {:?})", e.message, partial),
    }
}
