//! **DEV-188 — a trait method's own generic parameters were dropped at a bound call site.**
//!
//! `check_trait_member_call` converted the declared signature and never looked at `sig.generics`.
//! So inside `fn g<T: Conv>(t: T)`, a call to `t.to::<Int32>(1)` on `fn to<U>(&self, x: U) -> U`
//! left `U` rigid: the turbofish was discarded, no inference variable was created, and the argument
//! check compared `Int32` against the type parameter `U` itself.
//!
//! **Every trait method that mentions its own generic parameter was therefore uncallable through a
//! generic bound.** Not mis-typed — uncallable, with `type mismatch: expected 'U'` and no way to
//! satisfy it. The same function is the `Self`-receiver path used by a trait default body calling a
//! sibling default, which had the identical defect.
//!
//! The concrete-receiver path (WP-C4.7-8.4) and the trait-default path already bound these
//! correctly, so this is the bound path being brought into line with a rule the language already
//! had — not a new rule.
//!
//! **Why the rejecting tests assert the message and not merely the rejection.** Before the repair
//! these programs were *also* rejected. A test that only checked "does not compile" would have
//! passed against the broken compiler, and would go on passing if the repair were reverted. The
//! diagnostic is the only thing that distinguishes "`U` was bound to `Bool` and the argument does
//! not match" from "`U` was never bound at all".

use starkc::diag::Severity;
use starkc::options::LanguageOptions;
use starkc::session::CompilerSession;
use starkc::source::SourceFile;
use starkc::typecheck::CalleeSelection;
use std::sync::Arc;

/// The trait under test and one implementor. `to` declares its own generic `U`, which is the
/// entire subject: `U` belongs to the METHOD, not to the trait and not to the impl.
const CONV: &str = "trait Conv {\n    fn to<U>(&self, x: U) -> U;\n}\n\
                    struct C { v: Int32 }\n\
                    impl Conv for C {\n    fn to<U>(&self, x: U) -> U { x }\n}\n";

fn compile(source: &str) -> Result<starkc::session::CheckedProgram, Vec<String>> {
    let file = Arc::new(SourceFile::new("test.stark", source));
    match CompilerSession::for_source(file, LanguageOptions::CORE).check() {
        Ok(program) => Ok(program),
        Err(failure) => Err(failure
            .diagnostics()
            .iter()
            .filter(|d| d.severity == Severity::Error)
            .map(|d| format!("{}: {}", d.code.clone().unwrap_or_default(), d.message))
            .collect()),
    }
}

fn rejection(source: &str) -> String {
    match compile(source) {
        Ok(_) => panic!("expected rejection, but the program compiled"),
        Err(errors) => errors.join(" | "),
    }
}

/// The published `method_args` for every `Bound` selection, by member name.
fn bound_method_args(program: &starkc::session::CheckedProgram) -> Vec<(String, usize)> {
    program
        .tables()
        .callable_uses
        .iter()
        .filter_map(|use_| match &use_.selection {
            CalleeSelection::Bound {
                member,
                method_args,
                ..
            } => Some((member.clone(), method_args.len())),
            _ => None,
        })
        .collect()
}

#[test]
fn a_generic_trait_method_is_callable_through_a_bound_with_a_turbofish() {
    let program = compile(&format!(
        "{CONV}fn g<T: Conv>(t: T) -> Int32 {{ t.to::<Int32>(1) }}\n\
         fn main() {{ let c: C = C {{ v: 0 }}; println(g(c)); }}\n"
    ))
    .expect("a turbofished generic trait method must be callable through a bound");
    assert_eq!(
        program.execute_hir().expect("must run").output,
        "1\n",
        "the method returns its argument, so the program prints it"
    );
    assert_eq!(
        bound_method_args(&program),
        vec![("to".to_string(), 1)],
        "the Bound selection must publish the ONE method-level argument this site supplied"
    );
}

#[test]
fn a_generic_trait_method_is_callable_through_a_bound_without_a_turbofish() {
    // The turbofish is for the uninferable case. `t.to(1)` determines `U` from the argument,
    // exactly as an ordinary generic free call does, so requiring one here would be a second rule.
    let program = compile(&format!(
        "{CONV}fn g<T: Conv>(t: T) -> Int32 {{ t.to(1) }}\n\
         fn main() {{ let c: C = C {{ v: 0 }}; println(g(c)); }}\n"
    ))
    .expect("an inferable generic trait method must be callable through a bound");
    assert_eq!(program.execute_hir().expect("must run").output, "1\n");
    assert_eq!(
        bound_method_args(&program),
        vec![("to".to_string(), 1)],
        "an inferred argument is still an argument: the published list must carry the type the \
         call site settled on, not an empty list"
    );
}

#[test]
fn a_turbofish_that_contradicts_the_argument_is_rejected_as_a_bool_mismatch() {
    // The load-bearing assertion is `'Bool'`. Before the repair this said `expected 'U'` — the
    // turbofish had not been applied at all, so the program was rejected for the wrong reason.
    let errors = rejection(&format!(
        "{CONV}fn g<T: Conv>(t: T) -> Bool {{ t.to::<Bool>(1) }}\n\
         fn main() {{ let c: C = C {{ v: 0 }}; println(g(c)); }}\n"
    ));
    assert!(
        errors.contains("expected 'Bool'"),
        "the turbofish must BIND `U` to `Bool` and the argument be checked against it, \
         got: {errors}"
    );
    assert!(
        !errors.contains("expected 'U'"),
        "`U` rigid is the defect itself, got: {errors}"
    );
}

#[test]
fn a_result_used_at_the_wrong_type_is_rejected_against_the_bound_argument() {
    let errors = rejection(&format!(
        "{CONV}fn g<T: Conv>(t: T) -> Bool {{ t.to::<Int32>(1) }}\n\
         fn main() {{ let c: C = C {{ v: 0 }}; println(g(c)); }}\n"
    ));
    assert!(
        errors.contains("expected 'Bool'") && errors.contains("found 'Int32'"),
        "the return type must be the INSTANTIATED `U`, so returning it as `Bool` is the mismatch \
         reported, got: {errors}"
    );
}

#[test]
fn turbofish_arity_is_validated_at_a_bound_call_site() {
    let errors = rejection(&format!(
        "{CONV}fn g<T: Conv>(t: T) -> Int32 {{ t.to::<Int32, Bool>(1) }}\n\
         fn main() {{ let c: C = C {{ v: 0 }}; println(g(c)); }}\n"
    ));
    assert!(
        errors.contains("generic argument count mismatch"),
        "two arguments for one declared parameter must be an ARITY error, which the bound path \
         never reached before because it never counted the parameters, got: {errors}"
    );
}

#[test]
fn the_bound_path_agrees_with_the_concrete_path() {
    // The claim of this repair is symmetry, so it is worth asserting directly rather than by
    // reading the two sets of tests side by side. The same malformed call sites must produce the
    // same diagnostics whether the receiver is a bounded parameter or the concrete nominal.
    for (label, bound_call, concrete_call) in [
        (
            "contradicting turbofish",
            "t.to::<Bool>(1)",
            "c.to::<Bool>(1)",
        ),
        (
            "wrong arity",
            "t.to::<Int32, Bool>(1)",
            "c.to::<Int32, Bool>(1)",
        ),
    ] {
        let through_bound = rejection(&format!(
            "{CONV}fn g<T: Conv>(t: T) -> Bool {{ {bound_call} }}\n\
             fn main() {{ let c: C = C {{ v: 0 }}; println(g(c)); }}\n"
        ));
        let concrete = rejection(&format!(
            "{CONV}fn main() {{ let c: C = C {{ v: 0 }}; let b: Bool = {concrete_call}; \
             println(b); }}\n"
        ));
        assert_eq!(
            through_bound, concrete,
            "{label}: a bounded parameter and a concrete receiver must report the same defect"
        );
    }
}

#[test]
fn a_core_trait_bound_publishes_no_method_arguments() {
    // Not a gap being tolerated: a core trait's contract is expressed in `ContractTy`, which has no
    // way to declare a method-level generic. Empty is the answer. Pinned so that if core traits
    // ever gain method generics, this test fails and forces the publisher to be revisited rather
    // than silently continuing to publish an empty list.
    let program = compile(
        "struct A { v: Int32 }\n\
         impl Display for A {\n    fn fmt(&self) -> String { String::from(\"A!\") }\n}\n\
         fn render<T: Display>(x: T) -> String { x.fmt() }\n\
         fn main() { println(render(A { v: 1 })); }\n",
    )
    .expect("a core-trait bound call must compile");
    assert_eq!(
        program.execute_hir().expect("must run").output,
        "A!\n",
        "the bound call must still reach the impl's body"
    );
    assert_eq!(
        bound_method_args(&program).len(),
        1,
        "one `x.fmt()` through a core bound"
    );
    for (member, count) in bound_method_args(&program) {
        assert_eq!(
            count, 0,
            "core trait member '{member}' cannot declare method generics"
        );
    }
}

#[test]
fn a_trait_default_calling_a_generic_sibling_through_self_is_callable() {
    // The `Self`-receiver path shares `check_trait_member_call`, so it carried the identical
    // defect. A default body calling a generic sibling on `self` is the same shape as the bound
    // call, reached by a different route.
    let program = compile(
        "trait Conv {\n\
         \x20   fn to<U>(&self, x: U) -> U;\n\
         \x20   fn twice(&self) -> Int32 { self.to::<Int32>(2) }\n\
         }\n\
         struct C { v: Int32 }\n\
         impl Conv for C {\n    fn to<U>(&self, x: U) -> U { x }\n}\n\
         fn main() { let c: C = C { v: 0 }; println(c.twice()); }\n",
    )
    .expect("a trait default calling a generic sibling through `self` must compile");
    assert_eq!(program.execute_hir().expect("must run").output, "2\n");
}
