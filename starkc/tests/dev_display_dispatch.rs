//! **DEV-DISPLAY-DISPATCH** — a compiler-known trait bound contributes callable methods through
//! the same candidate-resolution path a user-defined trait bound does.
//!
//! The defect this suite pins: `fn show<T: Display>(x: &T) -> String { x.fmt() }` was rejected
//! with `[E0302] method 'fmt' not found for type 'T'`, while the identical shape over a
//! user-declared trait compiled. `T: Display` was *checked* as a bound and then contributed
//! nothing to method resolution, because resolution looked bounds up by matching
//! `hir::ItemKind::Trait` items and a compiler-known trait has no such item. Method visibility
//! therefore depended on whether a trait happened to be compiler-known — two trait models, not
//! one.
//!
//! What the cases below are for, beyond "it compiles now":
//!
//! * **Coexistence and order** (`core_and_user_bounds_coexist*`) — the two bound sources are
//!   additive and neither shadows the other, in both written orders. A fix that made `Display`
//!   reachable by *preferring* it would pass a basic-dispatch test and fail these.
//! * **Ambiguity** (`same_name_from_two_bounds_is_ambiguous`) — being compiler-known is not a
//!   tie-breaker. Silently preferring either side is the failure mode a single selection step
//!   exists to prevent.
//! * **Ownership** (`fmt_borrows_and_does_not_consume*`) — `Display::fmt` is `&self`. These
//!   would fail if the call were lowered, or move-checked, as consuming; before this work the
//!   move checker had no branch for a bounded-generic receiver at all and consumed every one.
//! * **Engine parity** — every positive case runs through the shared three-engine comparator
//!   (HIR oracle, MIR interpreter, native binary), with stdout pinned independently rather than
//!   taken from one engine. Three engines agreeing on the wrong text still fails.
//! * **Backend restriction** (`native_selects_stark_formatting_not_rusts`) — the native engine
//!   must not satisfy `Display` through Rust's own `Display`/`Debug`/`format!`/`ToString`. The
//!   generated source is read and checked.

mod support;

use starkc::backend::generated_rust::{emit_native_debug, NativeBuildOptions, Profile};
use starkc::mir::lower::lower_program;
use starkc::mir::verify::verify_program;
use support::differential::{
    agree_completing_with_stdout, front_end, rejects_at_typecheck, rustc_available,
};

/// A user trait, used everywhere a "second, ordinary trait" is needed.
const NAMED: &str = "\
trait Named {
    fn name(&self) -> String;
}
";

/// A non-`Copy` nominal with its own `Display` impl, and a `Named` impl so the coexistence cases
/// have something to call. `String` fields make it non-`Copy` by construction. Carries the
/// `Named` declaration too, so a case that uses `Point` needs only this one fragment.
const POINT: &str = "\
trait Named {
    fn name(&self) -> String;
}

struct Point {
    label: String,
    x: Int32,
}

impl Display for Point {
    fn fmt(&self) -> String {
        let mut out = String::new();
        out.push_str(self.label.as_str());
        out.push_str(\"@\");
        let n = self.x.fmt();
        out.push_str(n.as_str());
        out
    }
}

impl Named for Point {
    fn name(&self) -> String {
        let mut out = String::new();
        out.push_str(\"point\");
        out
    }
}
";

// ------------------------------------------------------------------ positive: basic dispatch --

#[test]
fn generic_display_bound_makes_fmt_callable() {
    agree_completing_with_stdout(
        "dd_basic",
        "\
fn render<T: Display>(x: &T) -> String {
    x.fmt()
}

fn main() {
    let n: Int32 = 42;
    println(render(&n).as_str());
}
",
        "42\n",
    );
}

/// The receiver is an OWNED generic parameter and is formatted twice. `Display::fmt` is `&self`,
/// so neither call may consume it — this is the ownership regression guard.
#[test]
fn fmt_borrows_and_does_not_consume_an_owned_parameter() {
    agree_completing_with_stdout(
        "dd_owned",
        "\
fn render_owned<T: Display>(x: T) -> String {
    let rendered = x.fmt();
    let rendered_again = x.fmt();

    let mut out = String::new();
    out.push_str(rendered.as_str());
    out.push_str(\"|\");
    out.push_str(rendered_again.as_str());
    out
}

fn main() {
    let n: Int32 = 7;
    println(render_owned(n).as_str());
}
",
        "7|7\n",
    );
}

/// A NON-COPY value survives generic formatting and is used afterwards. `String` is not `Copy`,
/// so a consuming lowering would be caught here rather than only by the move checker.
#[test]
fn fmt_borrows_and_does_not_consume_a_non_copy_value() {
    agree_completing_with_stdout(
        "dd_non_copy",
        &format!(
            "{POINT}
fn render<T: Display>(x: &T) -> String {{
    x.fmt()
}}

fn main() {{
    let p = Point {{ label: \"a\".to_string(), x: 3 }};
    let first = render(&p);
    let second = render(&p);
    println(first.as_str());
    println(second.as_str());
    println(p.name().as_str());
}}
"
        ),
        "a@3\na@3\npoint\n",
    );
}

/// An AFFINE value — a nominal with a `Drop` impl, which Core v1 forbids from also being `Copy`,
/// so it can be used at most once by move. It is formatted generically and then still used.
///
/// A host provider resource would be the other affine shape, but a resource nominal is a
/// synthesized zero-variant enum with no `Display` impl and no way to write one; the `Drop`-
/// bearing struct is the smallest affine type that can actually carry `Display`. The rendering is
/// a stable opaque label, not the value's internals — what is being proven is the ownership
/// behaviour, not the text.
#[test]
fn fmt_borrows_and_does_not_consume_an_affine_value() {
    agree_completing_with_stdout(
        "dd_affine",
        "\
struct Handle {
    id: Int32,
}

impl Drop for Handle {
    fn drop(&mut self) {
        println(\"released\");
    }
}

impl Display for Handle {
    fn fmt(&self) -> String {
        let mut out = String::new();
        out.push_str(\"Handle\");
        out
    }
}

fn render<T: Display>(x: &T) -> String {
    x.fmt()
}

fn consume(h: Handle) -> Int32 {
    h.id
}

fn main() {
    let h = Handle { id: 5 };
    let rendered = render(&h);
    println(rendered.as_str());
    let id = consume(h);
    println(id.fmt().as_str());
}
",
        "Handle\nreleased\n5\n",
    );
}

// -------------------------------------------------------------------- positive: the primitives --

/// Every standard-library `Display` primitive, through the SAME generic function. If the fix had
/// been a resolver hack keyed on one receiver shape, this is where it would show.
#[test]
fn generic_dispatch_covers_the_display_primitives() {
    agree_completing_with_stdout(
        "dd_primitives",
        "\
fn render<T: Display>(x: &T) -> String {
    x.fmt()
}

fn main() {
    let i8v: Int8 = -8;
    let i16v: Int16 = -16;
    let i32v: Int32 = -32;
    let i64v: Int64 = -64;
    let u8v: UInt8 = 8;
    let u16v: UInt16 = 16;
    let u32v: UInt32 = 32;
    let u64v: UInt64 = 64;
    let f32v: Float32 = 0.5f32;
    let f64v: Float64 = 0.75;
    let bv: Bool = true;
    let cv: Char = 'z';
    let sv: String = \"owned\".to_string();
    println(render(&i8v).as_str());
    println(render(&i16v).as_str());
    println(render(&i32v).as_str());
    println(render(&i64v).as_str());
    println(render(&u8v).as_str());
    println(render(&u16v).as_str());
    println(render(&u32v).as_str());
    println(render(&u64v).as_str());
    println(render(&f32v).as_str());
    println(render(&f64v).as_str());
    println(render(&bv).as_str());
    println(render(&cv).as_str());
    println(render(&sv).as_str());
    println(render(\"slice\").as_str());
}
",
        "-8\n-16\n-32\n-64\n8\n16\n32\n64\n0.5\n0.75\ntrue\nz\nowned\nslice\n",
    );
}

/// The rendering a generic `fmt()` produces is the rendering `println` produces. They share one
/// renderer in every engine; this states the property rather than trusting the sharing.
#[test]
fn generic_fmt_agrees_with_println_rendering() {
    agree_completing_with_stdout(
        "dd_fmt_matches_println",
        "\
fn render<T: Display>(x: &T) -> String {
    x.fmt()
}

fn main() {
    let f: Float64 = 0.1;
    println(f);
    println(render(&f).as_str());
    let c: Char = 'q';
    println(c);
    println(render(&c).as_str());
}
",
        "0.1\n0.1\nq\nq\n",
    );
}

// --------------------------------------------------------- positive: user impls and coexistence --

#[test]
fn a_user_display_impl_dispatches_through_the_bound() {
    agree_completing_with_stdout(
        "dd_user_impl",
        &format!(
            "{POINT}
fn render<T: Display>(x: &T) -> String {{
    x.fmt()
}}

fn main() {{
    let p = Point {{ label: \"origin\".to_string(), x: 0 }};
    println(render(&p).as_str());
}}
"
        ),
        "origin@0\n",
    );
}

/// `T: Display + Named` — a compiler-known bound and a user bound, both contributing. Neither
/// shadows the other.
#[test]
fn core_and_user_bounds_coexist() {
    agree_completing_with_stdout(
        "dd_coexist",
        &format!(
            "{POINT}
fn describe<T: Display + Named>(value: &T) -> String {{
    let rendered = value.fmt();
    let name = value.name();

    let mut out = String::new();
    out.push_str(name.as_str());
    out.push_str(\"=\");
    out.push_str(rendered.as_str());
    out
}}

fn main() {{
    let p = Point {{ label: \"p\".to_string(), x: 1 }};
    println(describe(&p).as_str());
}}
"
        ),
        "point=p@1\n",
    );
}

/// The inverse bound order must behave identically. Declaration order is not a resolution rule.
#[test]
fn core_and_user_bounds_coexist_in_either_order() {
    agree_completing_with_stdout(
        "dd_coexist_inverse",
        &format!(
            "{POINT}
fn describe<T: Named + Display>(value: &T) -> String {{
    let rendered = value.fmt();
    let name = value.name();

    let mut out = String::new();
    out.push_str(name.as_str());
    out.push_str(\"=\");
    out.push_str(rendered.as_str());
    out
}}

fn main() {{
    let p = Point {{ label: \"p\".to_string(), x: 1 }};
    println(describe(&p).as_str());
}}
"
        ),
        "point=p@1\n",
    );
}

/// A second bound that contributes no `fmt` must not disturb the one that does.
#[test]
fn multiple_bounds_with_only_one_provider() {
    agree_completing_with_stdout(
        "dd_multi_bound",
        &format!(
            "{NAMED}
fn render_named<T: Named + Display>(x: &T) -> String {{
    x.fmt()
}}

struct Tag {{ n: Int32 }}

impl Named for Tag {{
    fn name(&self) -> String {{
        let mut out = String::new();
        out.push_str(\"tag\");
        out
    }}
}}

impl Display for Tag {{
    fn fmt(&self) -> String {{
        self.n.fmt()
    }}
}}

fn main() {{
    let t = Tag {{ n: 11 }};
    println(render_named(&t).as_str());
}}
"
        ),
        "11\n",
    );
}

/// Nested generic forwarding: `outer<T: Display>` hands its parameter to `inner<U: Display>`.
/// Exercised through the native backend because monomorphisation and trait-call lowering are
/// where a forwarding defect lives.
#[test]
fn nested_generic_forwarding() {
    agree_completing_with_stdout(
        "dd_nested",
        &format!(
            "{POINT}
fn outer<T: Display>(x: &T) -> String {{
    inner(x)
}}

fn inner<U: Display>(x: &U) -> String {{
    x.fmt()
}}

fn main() {{
    let n: Int32 = 3;
    println(outer(&n).as_str());
    let p = Point {{ label: \"n\".to_string(), x: 4 }};
    println(outer(&p).as_str());
}}
"
        ),
        "3\nn@4\n",
    );
}

/// A bound on the IMPL head, not on the method (WP-C6.2b-F5), reaching a compiler-known trait.
#[test]
fn an_impl_head_bound_reaches_a_core_trait() {
    agree_completing_with_stdout(
        "dd_impl_head_bound",
        "\
struct Wrap<T> {
    value: T,
}

impl<T: Display> Wrap<T> {
    fn render(&self) -> String {
        self.value.fmt()
    }
}

fn main() {
    let w = Wrap { value: 21 };
    println(w.render().as_str());
}
",
        "21\n",
    );
}

// ------------------------------------------------------------------------------ negative cases --

/// No bound at all: the diagnostic must name the bound that is missing, not merely report that a
/// method was "not found".
#[test]
fn a_missing_bound_names_the_trait_to_add() {
    let messages = rejects_at_typecheck(
        "dd_missing_bound",
        "\
fn bad<T>(x: &T) -> String {
    x.fmt()
}

fn main() {}
",
        "E0302",
    );
    assert!(
        messages
            .iter()
            .any(|m| m.contains("requires the bound") && m.contains("Display")),
        "expected the missing `Display` bound to be named, got {messages:?}"
    );
}

/// An UNRELATED bound is present. The answer is still "you need `Display`".
#[test]
fn a_wrong_bound_still_names_display() {
    let messages = rejects_at_typecheck(
        "dd_wrong_bound",
        &format!(
            "{NAMED}
fn bad<T: Named>(x: &T) -> String {{
    x.fmt()
}}

fn main() {{}}
"
        ),
        "E0302",
    );
    assert!(
        messages
            .iter()
            .any(|m| m.contains("requires the bound") && m.contains("Display")),
        "expected the missing `Display` bound to be named, got {messages:?}"
    );
}

/// A method no trait in scope declares gets the ordinary "not found" wording — the missing-bound
/// diagnostic is derived from the traits that exist, not printed for every failed lookup.
#[test]
fn an_unknown_method_on_a_parameter_is_still_not_found() {
    let messages = rejects_at_typecheck(
        "dd_unknown_method",
        &format!(
            "{NAMED}
fn bad<T: Named>(x: &T) -> String {{
    x.wibble()
}}

fn main() {{}}
"
        ),
        "E0302",
    );
    assert!(
        messages.iter().any(|m| m.contains("not found")),
        "expected a plain not-found rejection, got {messages:?}"
    );
    assert!(
        !messages.iter().any(|m| m.contains("requires the bound")),
        "a method no trait declares must not be reported as a missing bound: {messages:?}"
    );
}

/// A CONCRETE type with no `Display` impl is still rejected. Making `Display` reachable from a
/// bound must not make it reachable from nothing.
#[test]
fn a_concrete_type_without_display_is_rejected() {
    rejects_at_typecheck(
        "dd_concrete_no_display",
        "\
struct Opaque {
    x: Int32,
}

fn main() {
    let o = Opaque { x: 1 };
    let s = o.fmt();
    println(s.as_str());
}
",
        "E0302",
    );
}

/// Two bounds declare `fmt` with compatible signatures. The compiler must report ambiguity rather
/// than pick one — and must pick neither for being compiler-known nor for being written first.
#[test]
fn same_name_from_two_bounds_is_ambiguous() {
    for (tag, bounds) in [
        ("dd_ambiguous", "Display + OtherFormat"),
        ("dd_ambiguous_inverse", "OtherFormat + Display"),
    ] {
        let messages = rejects_at_typecheck(
            tag,
            &format!(
                "\
trait OtherFormat {{
    fn fmt(&self) -> String;
}}

fn ambiguous<T: {bounds}>(x: &T) -> String {{
    x.fmt()
}}

fn main() {{}}
"
            ),
            "E0203",
        );
        assert!(
            messages.iter().any(|m| m.contains("ambiguous")),
            "{tag}: expected an ambiguity rejection, got {messages:?}"
        );
    }
}

/// The ambiguity above is resolvable: a qualified trait call names the trait explicitly, and both
/// spellings select the right impl on a type that implements both.
///
/// **Checked through the front end and the HIR oracle only.** A qualified call to a
/// compiler-known trait's method (`Display::fmt(&x)`) has no MIR lowering — it is refused with
/// "callee form (C4.5)" — so this shape cannot be run natively today. That gap is PRE-EXISTING
/// (DEV-052 introduced the qualified CoreTrait path in the front end and the oracle only) and is
/// recorded as a follow-up rather than widened into this work package. What matters here is that
/// the ambiguity above is resolvable at all; the resolution mechanism is not this WP's to build.
#[test]
fn qualified_calls_disambiguate_the_two_traits() {
    let source = "\
trait OtherFormat {
    fn fmt(&self) -> String;
}

struct Both {
    n: Int32,
}

impl Display for Both {
    fn fmt(&self) -> String {
        let mut out = String::new();
        out.push_str(\"display\");
        out
    }
}

impl OtherFormat for Both {
    fn fmt(&self) -> String {
        let mut out = String::new();
        out.push_str(\"other\");
        out
    }
}

fn main() {
    let b = Both { n: 1 };
    println(Display::fmt(&b).as_str());
    println(OtherFormat::fmt(&b).as_str());
}
";
    let front = front_end("dd_qualified", source);
    match support::differential::run_hir("dd_qualified", &front) {
        support::differential::Observation::Completed(done) => {
            assert_eq!(
                String::from_utf8_lossy(&done.stdout_bytes),
                "display\nother\n",
                "dd_qualified: the qualified calls selected the wrong impls"
            );
        }
        other => panic!("dd_qualified: expected normal completion, got {other:#?}"),
    }
}

/// Arity is checked against the Core trait's own contract: `Display::fmt` takes no arguments.
#[test]
fn a_core_bound_method_checks_its_arity() {
    rejects_at_typecheck(
        "dd_arity",
        "\
fn bad<T: Display>(x: &T) -> String {
    x.fmt(1)
}

fn main() {}
",
        "E0005",
    );
}

// ------------------------------------------------------------------------ backend restrictions --

/// The native engine must select STARK's own formatting, not Rust's. The generated crate is read
/// and checked for the runtime call this lowering emits, and against the Rust facilities that
/// would mean the backend had answered the question itself.
#[test]
fn native_selects_stark_formatting_not_rusts() {
    if !rustc_available() {
        eprintln!("SKIP-NATIVE: dd_backend: no rustc.");
        return;
    }
    let source = "\
fn render<T: Display>(x: &T) -> String {
    x.fmt()
}

fn main() {
    let n: Int32 = 5;
    println(render(&n).as_str());
}
";
    let front = front_end("dd_backend", source);
    let program = match lower_program(&front.hir, &front.tables, front.file.clone()) {
        Ok(program) => program,
        Err(error) => panic!("dd_backend: lowering failed: {}", error.what),
    };
    let verified = verify_program(&program).expect("dd_backend: verification failed");
    let target_dir = std::env::temp_dir().join(format!("stark_dd_backend_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&target_dir);
    let artifact = emit_native_debug(
        &verified,
        &NativeBuildOptions {
            target_dir: target_dir.clone(),
            target_contract: "stark-64-v1".to_string(),
            profile: Profile::Debug,
            target_triple: None,
        },
    )
    .expect("dd_backend: native build failed");
    let generated = std::fs::read_to_string(artifact.build_dir.join("src/main.rs"))
        .expect("dd_backend: generated source unreadable");
    let _ = std::fs::remove_dir_all(&target_dir);

    assert!(
        generated.contains("stark_runtime::format::fmt_i64"),
        "the selected `Display::fmt` must lower to STARK's own renderer"
    );
    for forbidden in [
        "format!",
        "std::fmt::Display",
        "std::fmt::Debug",
        "#[derive(Debug",
        "ToString",
    ] {
        assert!(
            !generated.contains(forbidden),
            "the generated crate must not satisfy `Display` through Rust's `{forbidden}`"
        );
    }
}

/// The release profile is a fourth execution mode, not a faster third one (WP-C7.1 §3.6). The
/// generic-dispatch shape is compared across both native profiles as well as the interpreters.
#[test]
fn release_and_debug_native_agree() {
    if !rustc_available() {
        eprintln!("SKIP-NATIVE: dd_profiles: no rustc.");
        return;
    }
    let source = "\
fn render<T: Display>(x: &T) -> String {
    x.fmt()
}

fn main() {
    let n: Int32 = 42;
    let f: Float64 = 0.25;
    println(render(&n).as_str());
    println(render(&f).as_str());
}
";
    let front = front_end("dd_profiles", source);
    let program = match lower_program(&front.hir, &front.tables, front.file.clone()) {
        Ok(program) => program,
        Err(error) => panic!("dd_profiles: lowering failed: {}", error.what),
    };
    let debug = support::differential::run_native_with_profile(
        "dd_profiles",
        "dd_profiles_debug",
        &program,
        Profile::Debug,
    );
    let release = support::differential::run_native_with_profile(
        "dd_profiles",
        "dd_profiles_release",
        &program,
        Profile::Release,
    );
    assert_eq!(
        support::differential::canonical_form(&debug),
        support::differential::canonical_form(&release),
        "debug and release native builds disagreed"
    );
}
