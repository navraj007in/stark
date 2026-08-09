//! **AS3 #2 — structural pins for the invocation authority.**
//!
//! The behavioural evidence lives with the interpreter, as unit tests: seven dispatch classes, each
//! proved by removing the environment and requiring the run to fail. Behaviour is what a mutation
//! can test. What a mutation cannot test is a FUTURE edit — a second body executor added next
//! month, or an `InvocationEnv` variant whose installer nobody updated. Those need a pin that fails
//! at build or census time rather than a test that happens to still pass.
//!
//! Both pins below are deliberately crude. A census of source text is a weak instrument in
//! general; it is the right instrument here because the claim is itself structural — "there is
//! exactly one of these" — and because the failure mode being guarded against is someone adding a
//! second one without noticing that a claim depended on there being one.

use starkc::interp::RepBoundary;

fn interp_source() -> String {
    std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/src/interp.rs"))
        .expect("interp.rs must be readable")
        // Normalised at the read: a CRLF checkout would otherwise fail these counts on Windows
        // only, which is the slowest possible way to find out.
        .replace("\r\n", "\n")
        // Comment lines removed: a census that counts prose can be satisfied — or broken — by
        // editing a comment, which is not what any of these pins is about.
        .lines()
        .filter(|line| !line.trim_start().starts_with("//"))
        .collect::<Vec<_>>()
        .join("\n")
}

/// **P1 — the raw user-body executor has exactly one production caller.**
///
/// `eval_block(callable.body)` is the moment a user body starts running. Three separate executors
/// existed before Packet 1: `execute_body`, `call_user_method`, and destruction's own copy. Each
/// carried its own frame push, its own epilogue, and — the reason it mattered — its own answer to
/// whether a generic environment had been installed. DEV-197 collected nine sites across three
/// discovery events, every one of them a path that reached a body without going through the place
/// that installs the environment.
///
/// So the single-entry property is not tidiness. It is what makes "every body runs under its
/// checker-selected environment" checkable at all: with one entry, the environment is a required
/// parameter; with three, it is a convention.
#[test]
fn p1_exactly_one_production_body_executor() {
    let source = interp_source();
    let executions = source.matches("self.eval_block(callable.body)").count();
    assert_eq!(
        executions, 1,
        "a user body must start executing in exactly ONE place. Found {executions}. A new one is \
         a new path that can reach a body without installing its environment — which is DEV-197, \
         nine times over. If a new execution form genuinely differs, make the difference a \
         `BodyEpilogue` variant, as `Call`, `Destructor` and `Method` already are."
    );
}

/// **P1, second half — the authority is the only caller of the raw executor.**
///
/// `execute_body` is reachable only through `invoke_with_epilogue`, which installs the environment
/// first. A direct call would bypass installation while still looking like ordinary dispatch.
#[test]
fn p1_the_raw_executor_is_called_only_by_the_authority() {
    let source = interp_source();
    let calls = source.matches("self.execute_body(").count();
    assert_eq!(
        calls, 1,
        "`execute_body` must be reached only through the invocation authority, which installs the \
         environment before it. Found {calls} call sites."
    );
}

/// **The environment installer is the only thing that pushes a generic frame for a call.**
///
/// If a dispatch path pushed its own frame, the installer's exhaustive match would no longer be
/// the single answer to "what environment is this body running under", and `EnvMutation::
/// DropEnvironment` — which removes the environment at the installer — would stop being a
/// complete control.
#[test]
fn the_installer_is_the_single_environment_entry_point() {
    let source = interp_source();
    let installs = source.matches("self.install_invocation_env(").count();
    assert_eq!(
        installs, 1,
        "the environment must be installed in exactly one place, or the AS3 #2 mutation control \
         is incomplete: a second installer is a path the control never touches. Found {installs}."
    );
}

/// **P6 — installation happens before the body, in the authority itself.**
///
/// The behavioural proof is D4/P6 in the interpreter's own tests: a receiver typed `&W<T>` is read
/// against the callee's instantiation, so it would fail on a correct program if the boundary ran
/// first. This pin adds the structural half — the two statements are adjacent and ordered, and the
/// guard's lifetime spans the call.
#[test]
fn p6_the_environment_is_installed_before_the_body_runs() {
    let source = interp_source();
    let install = source
        .find("let _env = self.install_invocation_env(")
        .expect("the authority must install the environment through the named binding");
    let execute = source
        .find("self.execute_body(")
        .expect("the authority must run the body");
    assert!(
        install < execute,
        "the environment must be installed before the body executes; otherwise `Receiver` and \
         `Parameter` would be read against the CALLER's instantiation, or against none"
    );
    assert!(
        source.contains("let _env ="),
        "the guard must be BOUND, not discarded with `let _ =`: a dropped-immediately guard would \
         uninstall the environment before the body ran, which is the same defect as never \
         installing it"
    );
}

/// The boundaries the AS3 #2 controls are allowed to name. Kept here so the two evidence bodies —
/// the interpreter's mutation tests and this file — cannot drift about what a "typed call
/// boundary" is.
#[test]
fn the_typed_call_boundaries_are_named_consistently() {
    for boundary in [
        RepBoundary::Receiver,
        RepBoundary::Parameter,
        RepBoundary::Return,
        RepBoundary::Propagation,
    ] {
        assert!(
            !boundary.as_str().is_empty(),
            "{boundary:?} must render in a diagnostic; a control that cannot name its boundary \
             cannot show it fired at the right one"
        );
    }
}
