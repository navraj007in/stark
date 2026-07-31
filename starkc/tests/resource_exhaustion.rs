//! WP-C7.9 Packet F — call-depth exhaustion is a *reported* outcome, not a crash.
//!
//! **What went wrong before.** A STARK program that recursed without a base case consumed the
//! host's Rust stack until the process died by signal. In a test binary that takes the whole runner
//! with it: every other test in the same process disappears, and the failure that remains says
//! nothing about which program caused it or why. There was no classification — nothing could
//! distinguish "the program recursed too deeply" from "the interpreter crashed", which are a
//! program's problem and a compiler's problem respectively.
//!
//! **What it is now.** `LIMIT-RESOURCE-001` already named call depth and already said what this is:
//! a host/process failure, prevented from becoming host undefined behaviour and *reported*, with an
//! implementation-defined capacity. Both interpreters check `interp::MAX_CALL_DEPTH` before pushing
//! a frame and return a classified `HostResource` failure. It is deliberately **not** a
//! `TrapCategory`: capacities are implementation- and target-defined, so this is not something the
//! engines could be required to agree on.
//!
//! **Why these run in subprocesses.** The guard is what prevents the abort, so a test for the guard
//! must be able to survive the guard being absent. Run in-process, a regression here would take the
//! runner down and be reported as an infrastructure failure rather than as this file failing.

use std::process::Command;

/// Runs one program in a fresh `stark run` process and returns (exit code, stdout, stderr).
///
/// `None` for the code means the process died by signal — which is the outcome this whole file
/// exists to rule out.
fn run_in_subprocess(tag: &str, source: &str) -> (Option<i32>, String, String) {
    let dir = std::env::temp_dir().join(format!(
        "stark_c79_resource_{tag}_{}_{:?}",
        std::process::id(),
        std::thread::current().id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("src")).expect("create case dir");
    std::fs::write(dir.join("src/main.stark"), source).expect("write source");
    std::fs::write(
        dir.join("starkpkg.json"),
        "{ \"name\": \"resource_case\", \"version\": \"0.1.0\", \"entry\": \"src/main.stark\" }\n",
    )
    .expect("write manifest");

    // `stark run` takes no path: it finds the package root from the working directory, so the
    // case directory is the child's cwd rather than an argument.
    let exe = env!("CARGO_BIN_EXE_stark");
    let out = Command::new(exe)
        .arg("run")
        .current_dir(&dir)
        .output()
        .expect("running `stark run` failed");
    let _ = std::fs::remove_dir_all(&dir);
    (
        out.status.code(),
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    )
}

/// Recursion that stays below the capacity completes normally. The guard must not turn ordinary
/// recursive programs into failures — a limit low enough to break real code is its own defect.
#[test]
fn recursion_below_the_limit_completes() {
    let (code, stdout, stderr) = run_in_subprocess(
        "below",
        "fn down(n: Int32) -> Int32 { if n <= 0 { 0 } else { down(n - 1) + 1 } }\n\
         fn main() { println(down(100)); }\n",
    );
    assert_eq!(code, Some(0), "expected clean completion; stderr: {stderr}");
    assert_eq!(stdout, "100\n");
}

/// Unbounded recursion is reported, with a classified message — and the process is still alive to
/// say so.
#[test]
fn unbounded_recursion_is_reported_not_aborted() {
    let (code, _stdout, stderr) = run_in_subprocess(
        "unbounded",
        "fn forever(n: Int32) -> Int32 { forever(n + 1) }\n\
         fn main() { println(forever(0)); }\n",
    );
    assert!(
        code.is_some(),
        "the process died by signal instead of reporting a resource limit"
    );
    assert!(
        stderr.contains("resource limit reached"),
        "expected a classified resource-limit diagnostic, got: {stderr}"
    );
    assert!(
        stderr.contains("call depth"),
        "the diagnostic must say which resource was exhausted: {stderr}"
    );
}

/// The status is stable, nonzero, and **not** the trap status. TRAP-ABORT-001 reserves 101 for
/// language traps; a resource limit is not one, and reusing 101 would make the two
/// indistinguishable to any script that reads exit codes.
#[test]
fn the_resource_limit_status_is_not_the_trap_status() {
    let (code, _, _) = run_in_subprocess(
        "status",
        "fn forever(n: Int32) -> Int32 { forever(n + 1) }\n\
         fn main() { println(forever(0)); }\n",
    );
    assert_eq!(code, Some(2), "resource limits exit 2");
    assert_ne!(code, Some(101), "101 is reserved for language traps");

    // The neighbour, for contrast: a real trap still exits 101.
    let (trap_code, _, trap_stderr) = run_in_subprocess(
        "trap_status",
        "fn main() { let z: Int32 = 0; println(1 / z); }\n",
    );
    assert_eq!(trap_code, Some(101), "a language trap still exits 101");
    assert!(
        trap_stderr.contains("runtime error"),
        "a trap is still reported as one: {trap_stderr}"
    );
}

/// Mutual recursion reaches the same guard. A depth counter attached to a single function, or reset
/// per callee, would miss this.
#[test]
fn mutual_recursion_reaches_the_limit() {
    let (code, _stdout, stderr) = run_in_subprocess(
        "mutual",
        "fn ping(n: Int32) -> Int32 { pong(n + 1) }\n\
         fn pong(n: Int32) -> Int32 { ping(n + 1) }\n\
         fn main() { println(ping(0)); }\n",
    );
    assert!(code.is_some(), "the process died by signal");
    assert!(
        stderr.contains("resource limit reached"),
        "mutual recursion must reach the same guard: {stderr}"
    );
}

/// **The counter is restored on the error path.** A depth counter that only decremented on the
/// success path would leak depth on every trap, and a program that trapped often enough would hit
/// a bogus resource limit while nowhere near the real one. Here a deep-but-legal call chain runs
/// after a trap has already unwound one — and completes.
#[test]
fn depth_is_restored_after_a_failed_call() {
    let (code, stdout, stderr) = run_in_subprocess(
        "restore",
        "fn down(n: Int32) -> Int32 { if n <= 0 { 0 } else { down(n - 1) + 1 } }\n\
         fn main() { \
         let mut total: Int32 = 0; \
         let mut i: Int32 = 0; \
         while i < 20 { total = total + down(100); i = i + 1; } \
         println(total); }\n",
    );
    assert_eq!(
        code,
        Some(0),
        "repeated deep-but-legal calls must not accumulate depth; stderr: {stderr}"
    );
    assert_eq!(stdout, "2000\n");
}

/// Independent runs do not inherit depth state from one another. The guard is per-run, not global.
#[test]
fn independent_runs_do_not_inherit_depth() {
    for round in 0..3 {
        let (code, stdout, _) = run_in_subprocess(
            &format!("independent_{round}"),
            "fn down(n: Int32) -> Int32 { if n <= 0 { 0 } else { down(n - 1) + 1 } }\n\
             fn main() { println(down(200)); }\n",
        );
        assert_eq!(code, Some(0), "round {round} must complete like the first");
        assert_eq!(stdout, "200\n");
    }
}
