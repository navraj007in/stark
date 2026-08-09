use starkc::analysis::{analyze_project, ProjectInput};
use starkc::options::LanguageOptions;
use starkc::source::SourceFile;
use std::sync::Arc;
fn run(n: usize) {
    let src = format!("fn main() {{ let x = {}; }}", vec!["1"; n].join(" + "));
    let f = Arc::new(SourceFile::new("t.stark", src));
    let a = analyze_project(ProjectInput::program(f), LanguageOptions::CORE);
    println!("  OK n={n} diags={}", a.diagnostics.len());
}
fn main() {
    let n: usize = std::env::args().nth(1).unwrap().parse().unwrap();
    let stack: usize = std::env::args()
        .nth(2)
        .map(|s| s.parse().unwrap())
        .unwrap_or(2 * 1024 * 1024);
    // Reproduce a TEST THREAD: cargo test spawns each test on a thread with a 2 MiB default stack.
    let h = std::thread::Builder::new()
        .stack_size(stack)
        .spawn(move || run(n))
        .unwrap();
    h.join().unwrap();
}
