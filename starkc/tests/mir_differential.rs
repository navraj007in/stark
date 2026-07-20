//! WP-C4.4 — HIR vs MIR differential validation.
//!
//! THE Gate C4 comparator (roadmap WP-C4.4): for each frozen workload the scalar core can
//! lower, `HIR interpreter output/failure == MIR interpreter output/failure`. The HIR
//! interpreter is the semantic oracle (charter §1.6 rule 6); the MIR pipeline under test is
//! lower → verify → execute. Success cases compare stdout byte-for-byte and exit status; trap
//! cases require BOTH engines to trap with the MIR trap category consistent with the oracle's
//! trap message, matching user-origin provenance, AND an identical pre-trap stdout prefix
//! (C4.5e-0 — partial output before a failure is observable).
//!
//! Corpus cases come from the frozen `exec_snapshots` corpus (`corpus_version = 1.0.0`);
//! inline programs cover scalar shapes the corpus reaches only via constructs the scalar core
//! defers to C4.5 (fn values, Option/Result, structs without methods).

use starkc::diag::Severity;
use starkc::interp;
use starkc::mir::interp::{run_program, MirFailure, MirRunError};
use starkc::mir::lower::lower_program;
use starkc::mir::verify::verify_program;
use starkc::mir::TrapCategory;
use starkc::parser::{parse, ParseMode};
use starkc::resolve::resolve;
use starkc::source::SourceFile;
use starkc::typecheck;
use std::path::PathBuf;
use std::sync::Arc;

fn corpus_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/exec_snapshots")
}

struct Front {
    hir: starkc::hir::Hir,
    file: Arc<SourceFile>,
    tables: starkc::typecheck::TypeTables,
}

fn front_end(name: &str, source: String) -> Front {
    let file = Arc::new(SourceFile::new(name, source));
    let (ast, pd) = parse(&file, ParseMode::Program);
    assert!(pd.is_empty(), "{name}: parse: {pd:?}");
    let (hir, rd) = resolve(&ast, file.clone());
    assert!(rd.is_empty(), "{name}: resolve: {rd:?}");
    let checked = typecheck::analyze(&hir, file.clone());
    let errors: Vec<_> = checked
        .diagnostics
        .iter()
        .filter(|d| d.severity == Severity::Error)
        .collect();
    assert!(errors.is_empty(), "{name}: typecheck: {errors:?}");
    Front {
        hir,
        file,
        tables: checked.tables,
    }
}

/// The message fragment the HIR oracle uses for each MIR trap category (the observable
/// comparator's trap-category correspondence).
fn oracle_fragment(category: TrapCategory) -> &'static str {
    match category {
        TrapCategory::IntegerOverflow => "integer overflow",
        TrapCategory::DivideByZero => "division by zero",
        TrapCategory::IndexOutOfBounds => "out of bounds",
        TrapCategory::CastFailure => "cast",
        TrapCategory::Panic => "panic",
        TrapCategory::UnwrapNone => "unwrap",
        TrapCategory::UnwrapErr => "unwrap",
        TrapCategory::AssertFailure => "assert",
        TrapCategory::InvalidShift => "invalid shift",
    }
}

fn differential(name: &str, source: String) {
    let front = front_end(name, source);

    // Oracle (failure carries the stdout accumulated before it — C4.5e-0).
    let oracle = interp::run_with_partial_output(&front.hir, front.file.clone(), &front.tables);

    // MIR pipeline: lower -> verify -> execute.
    let program = match lower_program(&front.hir, &front.tables, front.file.clone()) {
        Ok(program) => program,
        Err(e) => panic!("{name}: lowering failed: {} @ {:?}", e.what, e.span),
    };
    let verified = match verify_program(&program) {
        Ok(verified) => verified,
        Err(errors) => panic!("{name}: verifier rejected lowered MIR:\n{errors:#?}"),
    };
    let mir_result = run_program(verified);

    match (oracle, mir_result) {
        (Ok(oracle_exec), Ok(mir_exec)) => {
            assert_eq!(
                mir_exec.output, oracle_exec.output,
                "{name}: DIFFERENTIAL STDOUT MISMATCH\n--- HIR oracle ---\n{}\n--- MIR ---\n{}",
                oracle_exec.output, mir_exec.output
            );
            assert_eq!(
                mir_exec.status, oracle_exec.status,
                "{name}: exit status mismatch"
            );
        }
        (
            Err((oracle_err, oracle_partial)),
            Err(MirFailure {
                error:
                    MirRunError::Trap {
                        category,
                        source,
                        message,
                    },
                output: mir_partial,
            }),
        ) => {
            assert!(
                oracle_err.is_trap,
                "{name}: oracle errored non-trap ({}) but MIR trapped {category:?}",
                oracle_err.message
            );
            // A user message (panic) is compared exactly against the oracle's message; a
            // compiler-generated trap (no message) uses the category-fragment check.
            match &message {
                Some(m) => assert_eq!(
                    m, &oracle_err.message,
                    "{name}: trap message mismatch — MIR {m:?} vs oracle {:?}",
                    oracle_err.message
                ),
                None => assert!(
                    oracle_err.message.contains(oracle_fragment(category)),
                    "{name}: trap category mismatch — MIR {category:?} vs oracle message {:?}",
                    oracle_err.message
                ),
            }
            // C4.5e-0 (review Finding 3): output emitted BEFORE the trap is observable —
            // two programs printing different prefixes before the same trap differ.
            assert_eq!(
                mir_partial, oracle_partial,
                "{name}: PRE-TRAP STDOUT MISMATCH\n--- HIR oracle ---\n{oracle_partial}\n--- MIR ---\n{mir_partial}"
            );
            // Provenance (review correction): a right-category trap at the WRONG location is
            // a lowering bug the comparator must catch. User-origin traps must match the
            // oracle's span exactly (both derive from the same HIR spans); synthetic-origin
            // traps (e.g. for-loop desugar) compare their documented classification instead.
            assert!(
                (source.file.0 as usize) < program.files.len(),
                "{name}: MIR trap carries an invalid FileId"
            );
            match source.origin {
                starkc::mir::Origin::UserCode => assert_eq!(
                    (source.span.lo, source.span.hi),
                    (oracle_err.span.lo, oracle_err.span.hi),
                    "{name}: trap PROVENANCE mismatch — MIR {category:?} at {:?} vs oracle at {:?}",
                    source.span,
                    oracle_err.span
                ),
                starkc::mir::Origin::Synthetic(_) => {}
            }
        }
        (Ok(exec), Err(mir_failure)) => panic!(
            "{name}: oracle succeeded (stdout {:?}) but MIR failed: {:?} (partial stdout {:?})",
            exec.output, mir_failure.error, mir_failure.output
        ),
        (Err((oracle_err, _)), Ok(mir_exec)) => panic!(
            "{name}: oracle trapped ({}) but MIR succeeded with stdout {:?} — A MISSED TRAP",
            oracle_err.message, mir_exec.output
        ),
        (
            Err((oracle_err, _)),
            Err(MirFailure {
                error: MirRunError::Internal(message),
                ..
            }),
        ) => panic!(
            "{name}: MIR internal error ({message}) while oracle said: {}",
            oracle_err.message
        ),
    }
}

#[test]
fn frozen_corpus_scalar_cases_agree() {
    for name in [
        "expr_stmt__01_arithmetic_and_precedence",
        "expr_stmt__03_loops_break_continue",
        "primitive__01_integer_widths_and_overflow_traps",
        "primitive__02_integer_overflow_traps",
        "struct_enum_trait__01_struct_construction_and_methods",
        "struct_enum_trait__02_enum_and_pattern_match",
    ] {
        let path = corpus_dir().join(format!("{name}.stark"));
        let source = std::fs::read_to_string(&path).unwrap();
        differential(&path.to_string_lossy(), source);
    }
}

#[test]
fn function_values_agree() {
    differential(
        "fnval.stark",
        "fn double(x: Int32) -> Int32 { x * 2 } \
         fn apply(f: fn(Int32) -> Int32, v: Int32) -> Int32 { f(v) } \
         fn main() { \
             let f: fn(Int32) -> Int32 = double; \
             println(f(21)); \
             println(apply(double, 5)); \
             println(apply(f, 7)); \
             println(f(f(10))); \
         }"
        .to_string(),
    );
}

#[test]
fn option_result_agree() {
    differential(
        "opt.stark",
        "fn half(n: Int32) -> Option<Int32> { \
             if n % 2 == 0 { Some(n / 2) } else { None } \
         } \
         fn main() { \
             match half(10) { \
                 Some(v) => println(v), \
                 None => println(0 - 1), \
             } \
             match half(7) { \
                 Some(v) => println(v), \
                 None => println(0 - 1), \
             } \
             let r: Result<Int32, Bool> = Ok(4); \
             match r { \
                 Ok(v) => println(v), \
                 Err(flag) => println(flag), \
             } \
         }"
        .to_string(),
    );
}

#[test]
fn structs_and_tuples_agree() {
    differential(
        "structs.stark",
        "struct Point { x: Int32, y: Int32 } \
         fn make(x: Int32, y: Int32) -> Point { Point { x: x, y: y } } \
         fn main() { \
             let p = make(3, 4); \
             println(p.x * p.x + p.y * p.y); \
             let t = (1, true); \
             println(t.0); \
             println(t.1); \
         }"
        .to_string(),
    );
}

#[test]
fn division_by_zero_trap_agrees() {
    differential(
        "divzero.stark",
        "fn main() { \
             let a = 10; \
             let b = 0; \
             println(a / b); \
         }"
        .to_string(),
    );
}

#[test]
fn mid_output_trap_agrees_on_failure_and_pre_trap_output() {
    // A trap after partial output: both engines must trap with the same category AND have
    // printed the same prefix before it (C4.5e-0 — `run_with_partial_output` /
    // `MirFailure.output` made pre-trap stdout observable to the comparator).
    differential(
        "midtrap.stark",
        "fn main() { \
             println(1); \
             let big: Int8 = 120i8; \
             let other: Int8 = 100i8; \
             println(big + other); \
         }"
        .to_string(),
    );
}

#[test]
fn recursion_and_calls_agree() {
    differential(
        "fib.stark",
        "fn fib(n: Int32) -> Int32 { \
             if n < 2 { n } else { fib(n - 1) + fib(n - 2) } \
         } \
         fn main() { \
             let mut i = 0; \
             while i < 10 { \
                 println(fib(i)); \
                 i = i + 1; \
             } \
         }"
        .to_string(),
    );
}

#[test]
fn methods_and_associated_fns_agree() {
    differential(
        "methods.stark",
        "struct Counter { value: Int32 } \
         impl Counter { \
             fn fresh(start: Int32) -> Counter { Counter { value: start } } \
             fn doubled(&self) -> Int32 { self.value * 2 } \
             fn consumed(self) -> Int32 { self.value + 1 } \
         } \
         fn main() { \
             let c = Counter::fresh(20); \
             println(c.doubled()); \
             println(c.doubled()); \
             println(c.consumed()); \
         }"
        .to_string(),
    );
}

#[test]
fn trait_dispatch_default_and_override_agree() {
    differential(
        "traits.stark",
        "trait Describe { \
             fn id(&self) -> Int32; \
             fn twice(&self) -> Int32 { self.id() * 2 } \
         } \
         struct A { n: Int32 } \
         struct B { n: Int32 } \
         impl Describe for A { \
             fn id(&self) -> Int32 { self.n } \
         } \
         impl Describe for B { \
             fn id(&self) -> Int32 { self.n } \
             fn twice(&self) -> Int32 { self.id() * 10 } \
         } \
         fn main() { \
             let a = A { n: 3 }; \
             let b = B { n: 4 }; \
             println(a.twice()); \
             println(b.twice()); \
             println(a.id()); \
         }"
        .to_string(),
    );
}

#[test]
fn array_indexing_reads_writes_and_loops_agree() {
    differential(
        "arrays.stark",
        "fn main() { \
             let mut a = [10, 20, 30, 40]; \
             println(a[0] + a[3]); \
             a[1] = 99; \
             println(a[1]); \
             let mut sum = 0; \
             let mut i = 0; \
             while i < 4 { \
                 sum = sum + a[i]; \
                 i = i + 1; \
             } \
             println(sum); \
         }"
        .to_string(),
    );
}

#[test]
fn array_out_of_bounds_trap_agrees_with_provenance() {
    // The OOB read must trap in BOTH engines with IndexOutOfBounds and the SAME source span
    // (this also exercises the DEV-065 oracle message fix: \"index out of bounds\").
    differential(
        "oob.stark",
        "fn main() { \
             let a = [1, 2, 3]; \
             let i = 7; \
             println(a[i]); \
         }"
        .to_string(),
    );
}

#[test]
fn mut_self_receiver_mutates_caller_local() {
    // C4.5b-2: the defining test for real references — a &mut self method mutates the
    // CALLER's local across the frame boundary, twice, and reads observe both mutations.
    differential(
        "mutrecv.stark",
        "struct Counter { value: Int32 } \
         impl Counter { \
             fn bump(&mut self) { self.value = self.value + 1; } \
             fn get(&self) -> Int32 { self.value } \
         } \
         fn main() { \
             let mut c = Counter { value: 10 }; \
             c.bump(); \
             c.bump(); \
             println(c.get()); \
             println(c.value); \
         }"
        .to_string(),
    );
}

#[test]
fn reference_arguments_and_derefs_agree() {
    differential(
        "refs.stark",
        "fn read_it(r: &Int32) -> Int32 { *r + 1 } \
         fn write_it(r: &mut Int32) { *r = *r * 2; } \
         fn main() { \
             let mut x = 20; \
             println(read_it(&x)); \
             write_it(&mut x); \
             println(x); \
             let r = &x; \
             println(*r); \
         }"
        .to_string(),
    );
}

#[test]
fn reference_to_struct_field_agrees() {
    differential(
        "fieldref.stark",
        "struct P { a: Int32, b: Int32 } \
         fn bump(r: &mut Int32) { *r = *r + 100; } \
         fn main() { \
             let mut p = P { a: 1, b: 2 }; \
             bump(&mut p.a); \
             println(p.a); \
             println(p.b); \
         }"
        .to_string(),
    );
}

// ---- WP-C4.5c: generics and full static dispatch ----

/// Monomorphised generic fns over multiple primitive instantiations, with operator dispatch
/// on the generic parameter resolving per instantiation (`Ord` comparison to integer and
/// float comparisons, `Num` arithmetic to checked-integer vs IEEE-float operations).
#[test]
fn generic_fns_over_primitive_instantiations_agree() {
    differential(
        "generic_prims.stark",
        // DEV-075 / PRIM-TRAIT-001: `largest` is instantiated at `Int32` and `Char` — both are
        // `Ord`. It is NOT instantiated at `Float64` any more: primitive floats implement no
        // `Ord`, so `largest(2.5, 1.5)` is now correctly rejected (IEEE comparison is not a total
        // order). `twice<T: Num>` still covers the float instantiation, since `Num` does include
        // floats — which keeps this test's real subject, multiple primitive instantiations of a
        // bounded generic, fully exercised.
        "fn largest<T: Ord>(a: T, b: T) -> T { if a > b { a } else { b } } \
         fn twice<T: Num>(x: T) -> T { x + x } \
         fn main() { \
             println(largest(3, 7)); \
             println(largest('a', 'z')); \
             println(twice(21)); \
             println(twice(0.25)); \
         }"
        .to_string(),
    );
}

/// A generic fn calling another generic fn with arguments that mention the caller's own
/// parameter: the recorded instantiation composes with the caller's substitution.
#[test]
fn generic_calling_generic_agrees() {
    differential(
        "generic_chain.stark",
        "fn id<T>(x: T) -> T { x } \
         fn pass<T>(x: T) -> T { id(x) } \
         fn main() { println(pass(11)); println(pass(false)); }"
            .to_string(),
    );
}

/// Recursion inside a generic fn stays within one instance (no runaway instantiation).
/// The recursing parameter here is unbounded. That was originally forced by DEV-067 (a bounded
/// parameter at an intra-generic call site was over-rejected); DEV-067 is closed as of
/// WP-C4.7-7, and the bounded form is now covered by `bounded_generic_call_chain_agrees`. This
/// test keeps the unbounded shape deliberately, so the no-runaway-instantiation property stays
/// pinned independently of bound checking.
#[test]
fn generic_recursion_agrees() {
    differential(
        "generic_rec.stark",
        "fn count_down<T>(marker: T, n: Int32) -> Int32 { \
             if n > 0 { count_down(marker, n - 1) } else { 99 } \
         } \
         fn main() { println(count_down(true, 3)); println(count_down(1.5, 2)); }"
            .to_string(),
    );
}

/// Static trait dispatch through a generic parameter's bound: the method call on a `T`
/// receiver resolves to the implementing type's method after substitution. The by-value receiver
/// was originally forced by DEV-067 (a `&T` receiver was over-rejected); DEV-067 is closed as of
/// WP-C4.7-7, and the reference form is now covered by
/// `bounded_generic_method_through_reference_agrees`. Keeping this one by-value preserves
/// coverage of both receiver forms.
#[test]
fn trait_bound_method_dispatch_in_generic_fn_agrees() {
    differential(
        "generic_bound_dispatch.stark",
        "trait Area { fn area(&self) -> Int32; } \
         struct Sq { s: Int32 } \
         struct Rect { w: Int32, h: Int32 } \
         impl Area for Sq { fn area(&self) -> Int32 { self.s * self.s } } \
         impl Area for Rect { fn area(&self) -> Int32 { self.w * self.h } } \
         fn measure<T: Area>(shape: T) -> Int32 { shape.area() } \
         fn main() { \
             let sq = Sq { s: 4 }; \
             let r = Rect { w: 3, h: 5 }; \
             println(measure(sq)); \
             println(measure(r)); \
         }"
        .to_string(),
    );
}

/// Generic user nominals: struct construction, field access through the registered
/// type-context entries, and pattern matching on a generic user enum.
#[test]
fn generic_struct_and_enum_agree() {
    differential(
        "generic_nominals.stark",
        "struct Pair<T> { a: T, b: T } \
         enum MyOpt<T> { None2, Some2(T) } \
         fn main() { \
             let p = Pair { a: 10, b: 32 }; \
             println(p.a + p.b); \
             let q = Pair { a: true, b: false }; \
             if q.a { println(1); } \
             let m = MyOpt::Some2(7); \
             match m { \
                 MyOpt::Some2(v) => println(v), \
                 MyOpt::None2 => println(0), \
             } \
         }"
        .to_string(),
    );
}

/// CD-021 workload item 21: a monomorphised generic fn as a function value — the coercion
/// site's determined instantiation becomes the FnPtr constant's instance.
#[test]
fn monomorphised_generic_fn_value_agrees() {
    differential(
        "generic_fnval.stark",
        "fn id<T>(x: T) -> T { x } \
         fn apply(f: fn(Int32) -> Int32, v: Int32) -> Int32 { f(v) } \
         fn main() { \
             let f: fn(Int32) -> Int32 = id; \
             println(f(41)); \
             println(apply(f, 7)); \
         }"
        .to_string(),
    );
}

// ---- WP-C4.5d: ownership and Drop ----

/// Core drop timing, oracle-confirmed: parameter drop at callee exit, block-scope exit,
/// assignment overwrite (install the new value, then destroy the old — CD-012),
/// immediately-discarded values, explicit `drop(x)`, and fn-exit drops of what remains.
#[test]
fn drop_scope_overwrite_discard_and_builtin_agree() {
    differential(
        "drop_core.stark",
        "struct Loud { id: Int32 } \
         impl Drop for Loud { fn drop(&mut self) { println(9000 + self.id); } } \
         fn consume(v: Loud) { println(100); } \
         fn make(id: Int32) -> Loud { Loud { id: id } } \
         fn main() { \
             let a = Loud { id: 1 }; \
             let b = Loud { id: 2 }; \
             consume(a); \
             println(101); \
             { \
                 let c = Loud { id: 3 }; \
                 println(102); \
             } \
             println(103); \
             let mut d = Loud { id: 4 }; \
             d = Loud { id: 5 }; \
             println(104); \
             Loud { id: 6 }; \
             make(7); \
             println(105); \
             drop(b); \
             println(106); \
         }"
        .to_string(),
    );
}

/// Conditional moves need runtime drop flags; partial moves drop only the remaining
/// sibling units; a `Drop`-implementing value runs its own destructor before its fields
/// (glue order); loop-body scopes drop per iteration and on `break`.
#[test]
fn conditional_partial_moves_and_loop_scopes_agree() {
    differential(
        "drop_flags.stark",
        "struct Loud { id: Int32 } \
         impl Drop for Loud { fn drop(&mut self) { println(9000 + self.id); } } \
         struct Pair { a: Loud, b: Loud } \
         struct Nest { inner: Loud, tag: Int32 } \
         impl Drop for Nest { fn drop(&mut self) { println(8000 + self.tag); } } \
         fn consume(v: Loud) { println(100); } \
         fn main() { \
             let x = Loud { id: 1 }; \
             let y = Loud { id: 2 }; \
             let z = Loud { id: 3 }; \
             println(200); \
             let cond = true; \
             if cond { consume(x); } \
             println(201); \
             let p = Pair { a: Loud { id: 10 }, b: Loud { id: 11 } }; \
             println(202); \
             consume(p.a); \
             println(203); \
             let n = Nest { inner: Loud { id: 20 }, tag: 99 }; \
             println(204); \
             let mut i = 0; \
             while true { \
                 let w = Loud { id: 30 + i }; \
                 if i > 0 { break; } \
                 i = i + 1; \
             } \
             println(205); \
         }"
        .to_string(),
    );
}

/// Early `return` drops live scopes after the return value moves out; a block tail moves
/// its value out before the block's own drops; `if`-as-value creates only the taken arm.
#[test]
fn early_return_tail_moves_and_if_value_drops_agree() {
    differential(
        "drop_exits.stark",
        "struct Loud { id: Int32 } \
         impl Drop for Loud { fn drop(&mut self) { println(9000 + self.id); } } \
         struct Pair { a: Loud, b: Loud } \
         fn pick(c: Bool) -> Loud { \
             let inner = Loud { id: 50 }; \
             if c { return Loud { id: 60 }; } \
             println(300); \
             inner \
         } \
         fn main() { \
             let p = Pair { a: Loud { id: 1 }, b: Loud { id: 2 } }; \
             println(301); \
             let got = { \
                 let t = Loud { id: 70 }; \
                 println(302); \
                 t \
             }; \
             println(303); \
             let r1 = pick(true); \
             println(304); \
             let r2 = pick(false); \
             println(305); \
             let v = if true { Loud { id: 80 } } else { Loud { id: 81 } }; \
             println(306); \
         }"
        .to_string(),
    );
}

/// Droppable payloads inside enums (user enum and Option) drop through runtime-variant
/// glue; a generic struct instantiated with a droppable type decomposes into units through
/// the instantiated field table.
#[test]
fn enum_payload_and_generic_nominal_drops_agree() {
    differential(
        "drop_enum_generic.stark",
        "struct Loud { id: Int32 } \
         impl Drop for Loud { fn drop(&mut self) { println(9000 + self.id); } } \
         enum Holder { Empty, Full(Loud) } \
         struct Box2<T> { a: T, b: T } \
         fn main() { \
             let h = Holder::Full(Loud { id: 40 }); \
             let e = Holder::Empty; \
             let o = Some(Loud { id: 41 }); \
             let g = Box2 { a: Loud { id: 42 }, b: Loud { id: 43 } }; \
             println(400); \
         }"
        .to_string(),
    );
}

/// Traps abort WITHOUT running destructors (abstract machine): both engines trap with the
/// same category and provenance, and neither runs the live local's destructor.
#[test]
fn trap_aborts_without_drops_agree() {
    differential(
        "drop_trap.stark",
        "struct Loud { id: Int32 } \
         impl Drop for Loud { fn drop(&mut self) { println(9000 + self.id); } } \
         fn main() { \
             let a = Loud { id: 1 }; \
             let zero = 0; \
             println(1 / zero); \
         }"
        .to_string(),
    );
}

// ---- WP-C4.5e-0: pre-runtime-values hardening ----

/// DEV-068: a user struct with `impl Copy` is Copy — used twice and field-read after use,
/// it must lower as copies (the field-precise verifier rejected the old always-Move
/// classification as use-after-move, failing valid programs).
#[test]
fn user_copy_impl_struct_is_copy_in_mir() {
    differential(
        "user_copy.stark",
        "struct Point { x: Int32, y: Int32 } \
         impl Copy for Point {} \
         fn take(p: Point) -> Int32 { p.x } \
         fn main() { \
             let p = Point { x: 1, y: 2 }; \
             println(take(p)); \
             println(take(p)); \
             println(p.y); \
         }"
        .to_string(),
    );
}

/// Destructor output emitted before a later trap is part of the observable pre-trap prefix:
/// a wrongly-placed or missing drop before a trap is now a comparator failure, not invisible.
#[test]
fn drop_output_before_a_trap_is_compared() {
    differential(
        "drop_then_trap.stark",
        "struct Loud { id: Int32 } \
         impl Drop for Loud { fn drop(&mut self) { println(9000 + self.id); } } \
         fn main() { \
             { let a = Loud { id: 1 }; } \
             let zero = 0; \
             println(1 / zero); \
         }"
        .to_string(),
    );
}

// ---- WP-C4.5e-1: strings (Amendment A1) ----

/// The two frozen String-based ownership_drop corpus cases now run through both engines —
/// the first String-dependent corpus cases to go differential-green (A1 §9).
#[test]
fn ownership_drop_corpus_string_cases_agree() {
    for name in [
        "ownership_drop__01_move_and_drop_order",
        "ownership_drop__02_shared_borrow_does_not_move",
    ] {
        let path = corpus_dir().join(format!("{name}.stark"));
        let source = std::fs::read_to_string(&path).unwrap();
        differential(&path.to_string_lossy(), source);
    }
}

/// String construction, `as_str`, `push_str`, `len`, and `&str`/`String` printing agree.
#[test]
fn string_construction_and_methods_agree() {
    differential(
        "strings.stark",
        "fn main() { \
             let mut s = String::from(\"foo\"); \
             s.push_str(\"bar\"); \
             println(s.as_str()); \
             println(s.len()); \
             println(\"literal\"); \
             println(s.is_empty()); \
         }"
        .to_string(),
    );
}

/// String/str equality and ordering route through StrEq/StrCmp and agree with the oracle's
/// content comparison.
#[test]
fn string_comparison_agrees() {
    differential(
        "strcmp.stark",
        "fn main() { \
             let a = String::from(\"apple\"); \
             let b = String::from(\"banana\"); \
             println(a.as_str() == b.as_str()); \
             println(a.as_str() != b.as_str()); \
             println(a.as_str() == \"apple\"); \
             println(\"apple\" < \"banana\"); \
             println(\"banana\" >= \"apple\"); \
         }"
        .to_string(),
    );
}

/// `panic(msg)` after partial output: both engines trap `Panic` with the SAME message and the
/// SAME pre-trap stdout (A1 Trap.message + the C4.5e-0 pre-trap comparator).
#[test]
fn panic_with_message_after_output_agrees() {
    differential(
        "panicmsg.stark",
        "fn main() { println(1); println(2); panic(\"boom\"); }".to_string(),
    );
}

/// `assert(false)` traps AssertFailure (no message); `assert(true)` is a no-op.
#[test]
fn assert_agrees() {
    differential(
        "assert.stark",
        "fn main() { assert(1 + 1 == 2); println(1); assert(1 > 2); println(2); }".to_string(),
    );
}

/// A `String` field makes its struct droppable; the user Drop impl reading `self.label.as_str()`
/// runs in the right order, and the String field's own (unobservable) drop follows.
#[test]
fn struct_with_string_field_drops_agree() {
    differential(
        "stringfield.stark",
        "struct Tag { label: String } \
         impl Drop for Tag { fn drop(&mut self) { println(self.label.as_str()); } } \
         fn main() { \
             let a = Tag { label: String::from(\"alpha\") }; \
             let b = Tag { label: String::from(\"beta\") }; \
             println(0); \
         }"
        .to_string(),
    );
}

// ---- WP-C4.5e-2: Vec data surface (Amendment A1) ----

/// Vec construction, push, len, index-read, index-set, pop, remove, is_empty, clear all agree
/// with the oracle (no iteration — by-reference `.iter()` is deferred to an A2 surface bump).
#[test]
fn vec_data_operations_agree() {
    differential(
        "vecops.stark",
        "fn main() { \
             let mut v: Vec<Int32> = Vec::new(); \
             v.push(10); v.push(20); v.push(30); \
             println(v.len()); \
             println(v[0u64]); println(v[2u64]); \
             v[1u64] = 99; \
             println(v[1u64]); \
             let x = v.pop(); \
             match x { Some(n) => println(n), None => println(0) } \
             println(v.len()); \
             let r = v.remove(0u64); \
             println(r); \
             println(v.is_empty()); \
             v.clear(); \
             println(v.is_empty()); \
         }"
        .to_string(),
    );
}

/// A `Vec` of droppable elements drops them in REVERSE index order at scope exit, and
/// `v[i] = x` on a droppable element runs the replaced element's destructor
/// (install-then-destroy). Both match the oracle (validates A1 §5a end to end).
#[test]
fn vec_of_droppable_elements_drops_reverse_order_agree() {
    differential(
        "vecdrop.stark",
        "struct Loud { id: Int32 } \
         impl Drop for Loud { fn drop(&mut self) { println(9000 + self.id); } } \
         fn main() { \
             let mut v: Vec<Loud> = Vec::new(); \
             v.push(Loud { id: 1 }); \
             v.push(Loud { id: 2 }); \
             v.push(Loud { id: 3 }); \
             println(100); \
             v[1u64] = Loud { id: 20 }; \
             println(101); \
         }"
        .to_string(),
    );
}

/// `v.clear()` on a droppable element type destroys every element (via the pop-and-drop loop,
/// A1 §5a — never hidden in an opaque runtime op), matching the oracle.
#[test]
fn vec_clear_droppable_runs_destructors_agree() {
    differential(
        "vecclear.stark",
        "struct Loud { id: Int32 } \
         impl Drop for Loud { fn drop(&mut self) { println(9000 + self.id); } } \
         fn main() { \
             let mut v: Vec<Loud> = Vec::new(); \
             v.push(Loud { id: 1 }); \
             v.push(Loud { id: 2 }); \
             println(100); \
             v.clear(); \
             println(101); \
         }"
        .to_string(),
    );
}

/// A Vec index out of bounds traps IndexOutOfBounds with the call-site provenance, after
/// partial output.
#[test]
fn vec_index_out_of_bounds_traps_agree() {
    differential(
        "vecoob.stark",
        "fn main() { \
             let mut v: Vec<Int32> = Vec::new(); \
             v.push(1); \
             println(0); \
             println(v[5u64]); \
         }"
        .to_string(),
    );
}

// ---- WP-C4.5e-3: `?` operator + Option/Result methods ----

/// The frozen `option_result__01` corpus case (Option construction, match, is_some/is_none)
/// now runs through both engines.
#[test]
fn option_result_01_corpus_case_agrees() {
    let path = corpus_dir().join("option_result__01_option_construction_and_match.stark");
    let source = std::fs::read_to_string(&path).unwrap();
    differential(&path.to_string_lossy(), source);
}

/// `?` propagation on Option and Result (non-droppable payloads), plus `is_ok`/`is_err`/
/// `is_some`/`is_none`/`unwrap`.
#[test]
fn try_operator_and_inspection_methods_agree() {
    differential(
        "tryops.stark",
        "fn checked(n: Int32) -> Result<Int32, Int32> { \
             if n > 0 { Ok(n) } else { Err(0 - n) } \
         } \
         fn doubled(n: Int32) -> Result<Int32, Int32> { let v = checked(n)?; Ok(v * 2) } \
         fn half(n: Int32) -> Option<Int32> { if n % 2 == 0 { Some(n / 2) } else { None } } \
         fn add_one(n: Int32) -> Option<Int32> { let h = half(n)?; Some(h + 1) } \
         fn main() { \
             println(doubled(5).is_ok()); \
             println(doubled(5).unwrap()); \
             println(doubled(-3).is_err()); \
             println(add_one(10).unwrap()); \
             println(add_one(7).is_none()); \
         }"
        .to_string(),
    );
}

/// `unwrap` on `None` traps `UnwrapNone` after partial output; both engines agree on category
/// and pre-trap prefix.
#[test]
fn unwrap_none_traps_agree() {
    differential(
        "unwrapnone.stark",
        "fn half(n: Int32) -> Option<Int32> { if n % 2 == 0 { Some(n / 2) } else { None } } \
         fn main() { println(1); println(half(7).unwrap()); }"
            .to_string(),
    );
}

/// `?` early-returns `Err(e)` all the way out, propagating through a caller that itself uses
/// `?`; the returned error is inspected by the outer function.
#[test]
fn try_propagation_chains_agree() {
    differential(
        "trychain.stark",
        "fn a(n: Int32) -> Result<Int32, Int32> { if n > 0 { Ok(n) } else { Err(0 - n) } } \
         fn b(n: Int32) -> Result<Int32, Int32> { let x = a(n)?; Ok(x + 1) } \
         fn c(n: Int32) -> Result<Int32, Int32> { let y = b(n)?; Ok(y * 2) } \
         fn main() { \
             println(c(3).unwrap()); \
             println(c(-4).is_err()); \
         }"
        .to_string(),
    );
}

// ---- match-drop increment: match on owned Drop-bearing scrutinees ----

/// The frozen `option_result__02` corpus case (`?` propagation + match over a
/// `Result<Int32, String>`) now runs through both engines — the last runtime-values corpus
/// case reachable without interior references.
#[test]
fn option_result_02_corpus_case_agrees() {
    let path = corpus_dir().join("option_result__02_result_and_try_propagation.stark");
    let source = std::fs::read_to_string(&path).unwrap();
    differential(&path.to_string_lossy(), source);
}

/// Match on an owned Drop-bearing scrutinee: a bound payload drops at arm end; an unbound
/// (Wild) payload also drops at arm end; a catch-all binding drops at arm end. All match the
/// oracle's drop timing.
#[test]
fn match_drop_binding_and_wild_and_catchall_agree() {
    let prelude = "struct Loud { id: Int32 } \
                   impl Drop for Loud { fn drop(&mut self) { print(\"d\"); println(self.id); } } \
                   enum E { A(Loud), B(Loud) } \
                   fn make(tag: Int32, id: Int32) -> E { \
                       if tag == 0 { E::A(Loud { id: id }) } else { E::B(Loud { id: id }) } \
                   } ";
    // Bound payload.
    differential(
        "match_bound.stark",
        format!(
            "{prelude} fn main() {{ println(1); \
                 match make(0, 10) {{ E::A(x) => {{ println(2); }} E::B(y) => {{ println(3); }} }} \
                 println(4); }}"
        ),
    );
    // Unbound (Wild) payload still drops at arm end.
    differential(
        "match_wild.stark",
        format!(
            "{prelude} fn main() {{ println(1); \
                 match make(0, 10) {{ E::A(_) => {{ println(2); }} E::B(y) => {{ println(3); }} }} \
                 println(4); }}"
        ),
    );
    // Catch-all binding drops the whole scrutinee at arm end.
    differential(
        "match_catchall.stark",
        format!(
            "{prelude} fn main() {{ println(1); \
                 match make(1, 10) {{ E::A(x) => {{ println(2); }} other => {{ println(3); }} }} \
                 println(4); }}"
        ),
    );
}

/// A match arm that consumes (moves) its bound payload into a call does not double-drop: the
/// move clears the arm-scope drop, so only the callee's drop fires.
#[test]
fn match_arm_that_moves_binding_does_not_double_drop_agree() {
    differential(
        "match_move.stark",
        "struct Loud { id: Int32 } \
         impl Drop for Loud { fn drop(&mut self) { print(\"d\"); println(self.id); } } \
         enum E { A(Loud), B(Loud) } \
         fn consume(v: Loud) { println(100); } \
         fn make(id: Int32) -> E { E::A(Loud { id: id }) } \
         fn main() { \
             println(1); \
             match make(7) { \
                 E::A(x) => { consume(x); } \
                 E::B(y) => { println(2); } \
             } \
             println(9); \
         }"
        .to_string(),
    );
}

// ---- WP-C4.5f-2: by-reference Vec iteration (surface 0.1-A2) ----

/// The frozen `collection_iter__01` corpus case (Vec push/index/iterate with `for value in
/// values.iter()` binding `&Int32`) now runs through both engines.
#[test]
fn collection_iter_01_corpus_case_agrees() {
    let path = corpus_dir().join("collection_iter__01_vec_push_index_iterate.stark");
    let source = std::fs::read_to_string(&path).unwrap();
    differential(&path.to_string_lossy(), source);
}

/// Iteration behaviors: empty Vec (zero iterations), break inside the loop, and mutation of
/// an outer accumulator through the `&T` loop variable's deref.
#[test]
fn vec_iteration_behaviors_agree() {
    differential(
        "iterbehave.stark",
        "fn main() { \
             let empty: Vec<Int32> = Vec::new(); \
             let mut count = 0; \
             for x in empty.iter() { count = count + 1; } \
             println(count); \
             let mut v: Vec<Int32> = Vec::new(); \
             v.push(5); v.push(6); v.push(7); \
             let mut sum = 0; \
             for x in v.iter() { \
                 if *x == 7 { break; } \
                 sum = sum + *x; \
             } \
             println(sum); \
             println(v.len()); \
         }"
        .to_string(),
    );
}

// ---- WP-C4.5f-3c: multi-file / multi-package lowering ----

/// A real two-file program (`mod helper;` loaded from disk): cross-module calls, a
/// cross-module struct with methods and a Drop impl, and module-qualified canonical
/// symbols. Exercises per-item file plumbing (spans/SourceInfo in the right file, name
/// reads in the declaring file) end to end.
#[test]
fn multi_file_module_program_agrees_with_qualified_symbols() {
    let dir = std::env::temp_dir().join(format!("stark_f3c_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let helper_path = dir.join("helper.stark");
    // WIDENED by WP-C4.7-4 (DEV-069 CLOSED): this test was previously pinned to a
    // front-end-safe subset — scalar free functions and literal-free helper bodies, padded so
    // helper spans stayed in-bounds — because the checker and the oracle read every span
    // against the ENTRY file. With per-item file resolution in place it now exercises the
    // shapes that used to fail: a cross-file STRUCT with METHODS, cross-file LITERALS, a
    // cross-file FIELD read, and a cross-file `Drop` impl whose destructor ordering is part of
    // the compared output. The helper is deliberately longer than the entry file (that length
    // difference is what turned a wrong-file read into an out-of-bounds panic).
    std::fs::write(
        &helper_path,
        "pub fn add_self(n: Int32) -> Int32 { n + n }\n\
         pub fn add_both(a: Int32, b: Int32) -> Int32 { add_self(a) + b }\n\
         pub struct Counter { pub n: Int32 }\n\
         impl Counter {\n\
             pub fn doubled(&self) -> Int32 { self.n * 2 }\n\
         }\n\
         impl Drop for Counter { fn drop(&mut self) { println(self.n); } }\n\
         pub fn counter() -> Counter { Counter { n: 31415 } }\n",
    )
    .unwrap();
    let main_path = dir.join("main.stark");
    let main_src = "mod helper;\n\
         fn main() {\n\
             println(helper::add_self(4));\n\
             println(helper::add_both(2, 5));\n\
             let c = helper::counter();\n\
             println(c.doubled());\n\
             println(c.n);\n\
             println(100);\n\
         }\n";
    std::fs::write(&main_path, main_src).unwrap();

    let file = Arc::new(SourceFile::new(
        main_path.to_string_lossy().into_owned(),
        main_src.to_string(),
    ));
    let (ast, pd) = parse(&file, ParseMode::Program);
    assert!(pd.is_empty(), "parse: {pd:?}");
    let (hir, rd) = resolve(&ast, file.clone());
    assert!(rd.is_empty(), "resolve: {rd:?}");
    let checked = typecheck::analyze(&hir, file.clone());
    let errors: Vec<_> = checked
        .diagnostics
        .iter()
        .filter(|d| d.severity == Severity::Error)
        .collect();
    assert!(errors.is_empty(), "typecheck: {errors:?}");

    // Oracle vs MIR.
    let oracle =
        interp::run_with_partial_output(&hir, file.clone(), &checked.tables).expect("oracle runs");
    let program = lower_program(&hir, &checked.tables, file.clone())
        .unwrap_or_else(|e| panic!("lowering failed: {} @ {:?}", e.what, e.span));
    // Module-qualified symbols + a second interned file.
    let symbols: Vec<&str> = program
        .bodies
        .iter()
        .map(|b| b.instance.symbol.as_str())
        .collect();
    assert!(
        symbols.contains(&"helper::add_self@[]"),
        "expected module-qualified fn symbol, got {symbols:?}"
    );
    assert!(
        symbols.contains(&"helper::add_both@[]"),
        "expected the module-qualified fn symbol, got {symbols:?}"
    );
    assert!(
        program.files.len() >= 2,
        "expected the helper file interned in the file table"
    );
    let verified = verify_program(&program).expect("multi-file program verifies");
    let exec = match run_program(verified) {
        Ok(exec) => exec,
        Err(failure) => panic!(
            "MIR failed: {:?} (stdout {:?})",
            failure.error, failure.output
        ),
    };
    assert_eq!(exec.output, oracle.output, "multi-file differential");
    // Pin the value too, not just the agreement: two engines that both produced nothing would
    // "agree" vacuously. 8 / 9 = the free functions; 62830 / 31415 = the cross-file method and
    // field read; 100 = the entry file's own literal; the trailing 31415 is `Counter`'s
    // cross-file destructor firing at end of scope, AFTER main's last statement.
    assert_eq!(
        oracle.output, "8\n9\n62830\n31415\n100\n31415\n",
        "widened multi-file program output"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

/// C4.5 EXIT CHECK: the ENTIRE frozen corpus (`corpus_version` current, all 17 cases) runs
/// equivalently through both interpreters — the WP-C4.5 exit criterion.
#[test]
fn entire_frozen_corpus_agrees() {
    for name in [
        "collection_iter__01_vec_push_index_iterate",
        "collection_iter__02_hashmap_insert_get_iteration_order",
        // corpus_version 1.1.0 (WP-C4.7-9).
        "collection_iter__03_slice_views_and_array_iteration",
        "expr_stmt__01_arithmetic_and_precedence",
        "expr_stmt__02_if_else_and_block_tail",
        "expr_stmt__03_loops_break_continue",
        "expr_stmt__04_match_and_patterns",
        "option_result__01_option_construction_and_match",
        "option_result__02_result_and_try_propagation",
        "multi_file__01_cross_file_execution_and_provenance",
        "option_result__03_box_and_layout_queries",
        "ownership_drop__01_move_and_drop_order",
        "ownership_drop__02_shared_borrow_does_not_move",
        "ownership_drop__03_discarded_values_and_nested_patterns",
        "primitive__01_integer_widths_and_overflow_traps",
        "primitive__02_integer_overflow_traps",
        "primitive__03_float_arithmetic_and_casts",
        "primitive__04_bitwise_shift_pow_and_ordering",
        "struct_enum_trait__01_struct_construction_and_methods",
        "struct_enum_trait__02_enum_and_pattern_match",
        "struct_enum_trait__03_generic_function_and_trait_bound",
        "struct_enum_trait__04_trait_default_and_override",
        "struct_enum_trait__05_generic_methods_and_impl_heads",
    ] {
        let path = corpus_dir().join(format!("{name}.stark"));
        let source = std::fs::read_to_string(&path).unwrap();
        differential(&path.to_string_lossy(), source);
    }
}

// ---- WP-C4.5f-3a: HashMap surface (0.1-A3) ----

/// HashMap new/insert/get/len/is_empty/contains_key agree, including insert's returned
/// `Option<V>` (None on fresh key, Some(old) on overwrite — the A1 honesty rule's visible form).
#[test]
fn hashmap_data_operations_agree() {
    differential(
        "mapops.stark",
        "fn main() { \
             let mut m: HashMap<Int32, Int32> = HashMap::new(); \
             println(m.is_empty()); \
             match m.insert(1, 10) { Some(old) => println(old), None => println(-1), }; \
             match m.insert(1, 11) { Some(old) => println(old), None => println(-1), }; \
             m.insert(2, 20); \
             println(m.len()); \
             println(m.contains_key(&2)); \
             println(m.contains_key(&9)); \
             match m.get(&1) { Some(v) => println(*v), None => println(-1), }; \
             match m.get(&9) { Some(v) => println(*v), None => println(-1), }; \
         }"
        .to_string(),
    );
}

/// `for k in m.keys()` observes deterministic insertion order (CD-009) identically in both
/// engines, including reads back through `get` inside the loop body.
#[test]
fn hashmap_keys_iteration_order_agrees() {
    differential(
        "mapkeys.stark",
        "fn main() { \
             let mut m: HashMap<Int32, Int32> = HashMap::new(); \
             m.insert(3, 30); \
             m.insert(1, 10); \
             m.insert(2, 20); \
             for k in m.keys() { \
                 println(*k); \
                 match m.get(k) { Some(v) => println(*v), None => println(-1), }; \
             } \
         }"
        .to_string(),
    );
}

// ---- WP-C4.5f-3b: Char ops + assert_eq/assert_ne ----

/// Char literals, `println(Char)`, and `String::push`/`pop` of a Char agree.
#[test]
fn char_operations_agree() {
    differential(
        "charops.stark",
        "fn main() { \
             let c = 'x'; \
             println(c); \
             let mut s = String::from(\"hi\"); \
             s.push('!'); \
             println(s.as_str()); \
             match s.pop() { Some(last) => println(last), None => println('?'), }; \
             println(s.as_str()); \
         }"
        .to_string(),
    );
}

/// Passing `assert_eq`/`assert_ne` (scalar and string) are no-ops; a failing `assert_eq`
/// traps AssertFailure after identical pre-trap output in both engines.
#[test]
fn assert_eq_and_ne_agree() {
    differential(
        "asserteq.stark",
        "fn main() { \
             assert_eq(2 + 2, 4); \
             assert_ne(1, 2); \
             assert_eq(\"ab\", \"ab\"); \
             assert_ne(\"ab\", \"cd\"); \
             println(1); \
             assert_eq(3, 5); \
             println(2); \
         }"
        .to_string(),
    );
}

// ---- WP-C4.6 A5: bitwise / shift / power operators ----

/// Bitwise and/or/xor/not, shifts, and `**` all agree with the oracle, including compound-assign
/// forms and a signed arithmetic right shift.
#[test]
fn bit_shift_pow_operators_agree() {
    differential(
        "a5_ops.stark",
        "fn main() { \
             println(6 & 3); \
             println(6 | 3); \
             println(6 ^ 3); \
             let n: Int32 = ~5; \
             println(n); \
             println(1 << 4); \
             println(64 >> 2); \
             let s: Int32 = -16 >> 2; \
             println(s); \
             println(2 ** 10); \
             let mut x = 6; \
             x &= 3; \
             x <<= 2; \
             x **= 2; \
             println(x); \
         }"
        .to_string(),
    );
}

/// An unsigned bitwise-not is width-masked (`~0u8 == 255`), matching the oracle.
#[test]
fn unsigned_bitnot_is_width_masked_agree() {
    differential(
        "a5_ubitnot.stark",
        "fn main() { let a = 5 as UInt8; let b = ~a; println(b); }".to_string(),
    );
}

/// A shift count at/above the operand width traps identically (IntegerOverflow category) in both
/// engines, after identical pre-trap output.
#[test]
fn oversized_shift_count_traps_agree() {
    differential(
        "a5_shift_trap.stark",
        "fn main() { println(1); let k = 40; let x: Int32 = 1 << k; println(x); }".to_string(),
    );
}

/// Integer `**` overflow traps identically after identical pre-trap output.
#[test]
fn pow_overflow_traps_agree() {
    differential(
        "a5_pow_trap.stark",
        "fn main() { println(1); let x: Int32 = 10 ** 10; println(x); }".to_string(),
    );
}

// ---- WP-C4.6 A7: normative expression forms in value position ----

/// `loop { break <value>; }` yields the break value; a plain-`break` loop is Unit.
#[test]
fn loop_break_value_agree() {
    differential(
        "a7_loop_value.stark",
        "fn main() { \
             let mut i = 0; \
             let found = loop { \
                 if i == 3 { break i * 10; } \
                 i = i + 1; \
             }; \
             println(found); \
             let u = loop { break; }; \
             println(7); \
         }"
        .to_string(),
    );
}

/// `[value; count]` repeat, `while`/`if`-without-else in value position all agree.
#[test]
fn repeat_and_unit_value_forms_agree() {
    differential(
        "a7_forms.stark",
        "fn main() { \
             let a = [9; 4]; \
             println(a[0] + a[1] + a[2] + a[3]); \
             let u = while false { }; \
             let v = if true { println(1); }; \
             println(2); \
         }"
        .to_string(),
    );
}

// ---- WP-C4.6 A6: non-Copy Vec iteration (borrowed cursor) ----

/// Iterating a `Vec<String>` by reference (`for s in v.iter()`) agrees with the oracle — the
/// borrowed-cursor iterator hands out `&String` without requiring the element be Copy.
#[test]
fn non_copy_vec_iteration_agrees() {
    differential(
        "a6_noncopy_iter.stark",
        "fn main() { \
             let mut v: Vec<String> = Vec::new(); \
             v.push(String::from(\"alpha\")); \
             v.push(String::from(\"beta\")); \
             v.push(String::from(\"gamma\")); \
             let mut n = 0; \
             for s in v.iter() { \
                 println(s.as_str()); \
                 n = n + s.len() as Int32; \
             } \
             println(n); \
         }"
        .to_string(),
    );
}

// ---- WP-C4.6 A3: user-defined Eq operator dispatch ----

/// `==`/`!=` on a user struct dispatch through its `Eq::eq` impl; operands are BORROWED (no
/// early drop), and a Drop-bearing type drops normally at scope end — all matching the oracle.
#[test]
fn user_struct_eq_dispatch_agrees() {
    differential(
        "a3_struct_eq.stark",
        "struct Tag { id: Int32 } \
         impl Eq for Tag { fn eq(&self, other: &Tag) -> Bool { self.id == other.id } } \
         impl Drop for Tag { fn drop(&mut self) { println(self.id); } } \
         fn main() { \
             let a = Tag { id: 1 }; \
             let b = Tag { id: 1 }; \
             let c = Tag { id: 2 }; \
             if a == b { println(100); } \
             if a != c { println(200); } \
             println(300); \
         }"
        .to_string(),
    );
}

// ---- WP-C4.6 A3 Ord (MIR Amendment A2, CE3): user Ord operator dispatch ----

/// All four ordered operators on a user `Ord` type, covering Less/Equal/Greater results, agree
/// with the oracle. `cmp` returns the logical `CoreOrdering` enum; the operators map its
/// discriminant to the comparison's Bool.
#[test]
fn user_ord_all_operators_agree() {
    differential(
        "a3_ord_ops.stark",
        "struct P { x: Int32 } \
         impl Ord for P { \
             fn cmp(&self, other: &P) -> Ordering { \
                 if self.x < other.x { Ordering::Less } \
                 else if self.x > other.x { Ordering::Greater } \
                 else { Ordering::Equal } \
             } \
         } \
         fn main() { \
             let a = P { x: 1 }; \
             let b = P { x: 2 }; \
             let c = P { x: 1 }; \
             if a < b { println(1); } \
             if b > a { println(2); } \
             if a <= c { println(3); } \
             if a >= c { println(4); } \
             if a >= b { println(90); } else { println(5); } \
             if b <= a { println(91); } else { println(6); } \
         }"
        .to_string(),
    );
}

/// A directly-invoked `cmp` returns an `Ordering` value that round-trips through a `match` by
/// variant, agreeing with the oracle. Uses all THREE variants explicitly since WP-C4.7-7 closed
/// DEV-071 — an all-three-variant `Ordering` match used to be wrongly flagged non-exhaustive, so
/// this test had to carry a `_` arm as a workaround. Dropping the wildcard is what makes it
/// exercise the real exhaustiveness path as well as the MIR
/// `CoreOrdering` construct/return/discriminant-switch path.
#[test]
fn ordering_value_round_trips_through_match_agree() {
    differential(
        "a3_ord_match.stark",
        "struct P { x: Int32 } \
         impl Ord for P { \
             fn cmp(&self, other: &P) -> Ordering { \
                 if self.x < other.x { Ordering::Less } \
                 else if self.x > other.x { Ordering::Greater } \
                 else { Ordering::Equal } \
             } \
         } \
         fn report(o: Ordering) -> Int32 { \
             match o { \
                 Ordering::Less => 10, \
                 Ordering::Greater => 30, \
                 Ordering::Equal => 20, \
             } \
         } \
         fn main() { \
             let a = P { x: 3 }; \
             let b = P { x: 5 }; \
             println(report(a.cmp(&b))); \
             println(report(b.cmp(&a))); \
             println(report(a.cmp(&a))); \
         }"
        .to_string(),
    );
}

/// Ordered comparison BORROWS both operands (no early move/drop) and evaluates left-to-right; a
/// `Drop`-bearing `Ord` type drops both operands once, at scope end, in reverse declaration
/// order — all matching the oracle.
#[test]
fn user_ord_borrows_and_drops_normally_agree() {
    differential(
        "a3_ord_drop.stark",
        "struct Tag { id: Int32 } \
         impl Ord for Tag { \
             fn cmp(&self, other: &Tag) -> Ordering { \
                 if self.id < other.id { Ordering::Less } \
                 else if self.id > other.id { Ordering::Greater } \
                 else { Ordering::Equal } \
             } \
         } \
         impl Drop for Tag { fn drop(&mut self) { println(self.id); } } \
         fn tag(id: Int32) -> Tag { println(id * 10); Tag { id: id } } \
         fn main() { \
             let a = tag(1); \
             let b = tag(2); \
             if a < b { println(100); } \
             println(200); \
         }"
        .to_string(),
    );
}

// ---- WP-C4.6 A4-1: size_of/align_of + Option/Result unwrap_or (no runtime-surface amendment) ----

/// `size_of::<T>()` / `align_of::<T>()` lower to the fixed word constant the reference
/// implementation reports, agreeing with the oracle.
#[test]
fn size_of_align_of_agree() {
    differential(
        "a4_sizeof.stark",
        "fn main() { \
             println(size_of::<Int32>()); \
             println(align_of::<Bool>()); \
             println(size_of::<Int64>() + align_of::<UInt8>()); \
         }"
        .to_string(),
    );
}

/// `Option::unwrap_or` / `Result::unwrap_or` select the payload or the default, agreeing with
/// the oracle for both variants.
#[test]
fn option_result_unwrap_or_agree() {
    differential(
        "a4_unwrap_or.stark",
        "fn main() { \
             let a: Option<Int32> = Some(7); \
             let b: Option<Int32> = None; \
             println(a.unwrap_or(9)); \
             println(b.unwrap_or(9)); \
             let r: Result<Int32, Int32> = Ok(3); \
             let e: Result<Int32, Int32> = Err(1); \
             println(r.unwrap_or(0)); \
             println(e.unwrap_or(0)); \
         }"
        .to_string(),
    );
}

/// Option/Result `map` / `and_then` / `map_err` (function-value combinators) agree with the
/// oracle, including the pass-through variant (`map` on an `Err`, `map_err` on an `Ok`).
#[test]
fn option_result_combinators_agree() {
    differential(
        "a4_combinators.stark",
        "fn dbl(x: Int32) -> Int32 { x * 2 } \
         fn to_opt(x: Int32) -> Option<Int32> { if x > 0 { Some(x + 100) } else { None } } \
         fn to_res(x: Int32) -> Result<Int32, Int32> { if x > 0 { Ok(x + 100) } else { Err(9) } } \
         fn neg(e: Int32) -> Int32 { 0 - e } \
         fn main() { \
             let a: Option<Int32> = Some(5); \
             let b: Option<Int32> = None; \
             println(a.map(dbl).unwrap_or(0)); \
             println(b.map(dbl).unwrap_or(-1)); \
             println(a.and_then(to_opt).unwrap_or(-2)); \
             let r: Result<Int32, Int32> = Ok(7); \
             let e: Result<Int32, Int32> = Err(3); \
             println(r.map(dbl).unwrap_or(0)); \
             println(e.map(dbl).unwrap_or(-4)); \
             println(e.map_err(neg).unwrap_or(-5)); \
             println(r.and_then(to_res).unwrap_or(-6)); \
         }"
        .to_string(),
    );
}

/// A range bound to a value and then iterated (`let r = 0..n; for i in r`) agrees with the
/// oracle — exclusive, inclusive, and empty ranges, with the inclusive flag read at runtime.
#[test]
fn range_value_iteration_agrees() {
    differential(
        "a4_range.stark",
        "fn main() { \
             let r = 0..3; \
             for i in r { println(i); } \
             let ri = 1..=3; \
             for j in ri { println(j * 10); } \
             let empty = 5..5; \
             for k in empty { println(99); } \
             println(0); \
         }"
        .to_string(),
    );
}

// ---- WP-C4.6 A4-2b: Vec::get / get_mut (surface 0.1-A4) ----

/// `Vec::get`/`get_mut` return `Option<&T>`/`Option<&mut T>` and never trap on out-of-bounds
/// (they return `None`); `get_mut` mutates through the reference; the element type need not be
/// Copy. All agree with the oracle.
#[test]
fn vec_get_and_get_mut_agree() {
    differential(
        "a4_vecget.stark",
        "fn main() { \
             let mut v: Vec<Int32> = Vec::new(); \
             v.push(10); v.push(20); v.push(30); \
             match v.get(1) { Some(x) => println(*x), None => println(-1), } \
             match v.get(5) { Some(x) => println(*x), None => println(-1), } \
             match v.get_mut(0) { Some(x) => { *x = 99; }, None => { } } \
             match v.get(0) { Some(x) => println(*x), None => println(-1), } \
             let mut s: Vec<String> = Vec::new(); \
             s.push(String::from(\"hi\")); \
             match s.get(0) { Some(x) => println(x.as_str()), None => println(\"none\"), } \
         }"
        .to_string(),
    );
}

/// `println(Ordering)` prints the variant name (`Less`/`Equal`/`Greater`), agreeing with the
/// oracle's Display. (Amendment A2 deferred this to A4; lowered without a runtime op.)
#[test]
fn println_ordering_agrees() {
    differential(
        "a4_print_ord.stark",
        "struct P { x: Int32 } \
         impl Ord for P { \
             fn cmp(&self, o: &P) -> Ordering { \
                 if self.x < o.x { Ordering::Less } \
                 else if self.x > o.x { Ordering::Greater } \
                 else { Ordering::Equal } \
             } \
         } \
         fn main() { \
             let a = P { x: 1 }; \
             let b = P { x: 2 }; \
             println(a.cmp(&b)); \
             println(b.cmp(&a)); \
             println(a.cmp(&a)); \
         }"
        .to_string(),
    );
}

// ---- WP-C4.6 A4-2d: str/String chars() iteration (surface 0.1-A5) ----

/// `for c in s.chars()` over a `String` and a `str` literal yields each Char in order, agreeing
/// with the oracle.
#[test]
fn chars_iteration_agrees() {
    differential(
        "a4_chars.stark",
        "fn main() { \
             let s = String::from(\"hello\"); \
             let mut n = 0; \
             for c in s.chars() { println(c); n = n + 1; } \
             println(n); \
             for c in \"hi!\".chars() { println(c); } \
         }"
        .to_string(),
    );
}

// ---- WP-C4.6 A4 slicing (surface 0.1-A6) ----

/// Array and Vec slicing: `&a[lo..hi]` (exclusive + inclusive), `len`/`is_empty`, view-relative
/// indexing, re-slicing composition, empty slices, and slice fn parameters all agree.
#[test]
fn slicing_operations_agree() {
    differential(
        "a4_slice.stark",
        "fn total(s: &[Int32]) -> Int32 { \
             let mut t = 0; \
             let mut i = 0; \
             while i < s.len() { t = t + s[i]; i = i + 1; } \
             t \
         } \
         fn main() { \
             let a = [10, 20, 30, 40, 50]; \
             let s = &a[1..4]; \
             println(s.len()); \
             println(s[0]); \
             println(s[2]); \
             let t = &s[1..3]; \
             println(t[0]); \
             let u = &a[0..=2]; \
             println(u.len()); \
             println(total(&a[1..5])); \
             let e = &a[2..2]; \
             println(e.is_empty()); \
             let mut v: Vec<Int32> = Vec::new(); \
             v.push(7); v.push(8); v.push(9); \
             let w = &v[1..3]; \
             println(w[0] + w[1]); \
         }"
        .to_string(),
    );
}

/// An out-of-range slice bound traps IndexOutOfBounds after identical pre-trap output.
#[test]
fn slice_out_of_range_traps_agree() {
    differential(
        "a4_slice_oob.stark",
        "fn main() { let a = [1, 2, 3]; println(0); let s = &a[1..9]; println(s.len()); }"
            .to_string(),
    );
}

/// An inverted slice range traps IndexOutOfBounds.
#[test]
fn slice_inverted_range_traps_agree() {
    differential(
        "a4_slice_inv.stark",
        "fn main() { let a = [1, 2, 3]; let s = &a[2..1]; println(s.len()); }".to_string(),
    );
}

/// Indexing past the VIEW length (even where the base container is longer) traps.
#[test]
fn slice_index_checks_view_length_agree() {
    differential(
        "a4_slice_view.stark",
        "fn main() { let a = [1, 2, 3, 4]; let s = &a[1..3]; println(s[0]); println(s[3]); }"
            .to_string(),
    );
}

// ---- WP-C4.6 A2 (increment 1): DEV-070 non-consuming match + Char literal patterns ----
// CE3-required regression matrix (CD-033 §5): match-through-reference must not move from and
// poison the borrowed place, WITHOUT a blanket "all matches borrow" rule — consumption depends
// on the scrutinee and pattern semantics.

/// (1) The same `&self` method containing `match *self`, called twice — the borrowed place
/// survives the first call. (2a) fieldless enum behind a shared reference.
#[test]
fn match_deref_self_twice_fieldless_agree() {
    differential(
        "a2_m1.stark",
        "enum Color { Red, Green, Blue } \
         impl Color { fn ord(&self) -> Int32 { \
             match *self { Color::Red => 0, Color::Green => 1, Color::Blue => 2, } } } \
         fn main() { let a = Color::Green; println(a.ord()); println(a.ord()); }"
            .to_string(),
    );
}

/// (2b) payload-bearing enum behind a shared reference: Copy payloads bind by copy; the
/// referent is reusable across calls.
#[test]
fn match_deref_self_copy_payload_agree() {
    differential(
        "a2_m2.stark",
        "enum Holder { Empty, Val(Int32) } \
         impl Holder { fn get(&self) -> Int32 { \
             match *self { Holder::Val(x) => x, Holder::Empty => -1, } } } \
         fn main() { let h = Holder::Val(42); println(h.get()); println(h.get()); }"
            .to_string(),
    );
}

/// (2c) a NON-Copy payload behind a shared reference stays untouched under a wildcard
/// sub-pattern — the referent keeps ownership, nothing drops in the arm, twice-callable.
#[test]
fn match_deref_self_noncopy_wildcard_agree() {
    differential(
        "a2_m6.stark",
        "enum Holder { Empty, Val(String) } \
         impl Holder { fn tag(&self) -> Int32 { \
             match *self { Holder::Val(_) => 1, Holder::Empty => 0, } } } \
         fn main() { let h = Holder::Val(String::from(\"y\")); println(h.tag()); println(h.tag()); }"
            .to_string(),
    );
}

/// (3) Copy scrutinees remain reusable after being matched.
#[test]
fn match_copy_scrutinee_reusable_agree() {
    differential(
        "a2_m4.stark",
        "fn main() { let n = 2; \
             match n { 1 => println(10), _ => println(20), } \
             match n { 2 => println(30), _ => println(40), } \
             println(n); }"
            .to_string(),
    );
}

/// (4) owned non-Copy scrutinees still follow the consuming C4.5d semantics: the matched
/// payload drops at arm end, exactly once.
#[test]
fn match_owned_drop_scrutinee_still_consumes_agree() {
    differential(
        "a2_m5.stark",
        "enum E { A(Tag), B } \
         struct Tag { id: Int32 } \
         impl Drop for Tag { fn drop(&mut self) { println(self.id); } } \
         fn main() { let e = E::A(Tag { id: 7 }); \
             match e { E::A(t) => println(100), E::B => println(200), } \
             println(300); }"
            .to_string(),
    );
}

/// Char literal patterns switch on the Unicode scalar codepoint.
#[test]
fn char_literal_patterns_agree() {
    differential(
        "a2_char_pat.stark",
        "fn main() { \
             let c = 'b'; \
             match c { 'a' => println(1), 'b' => println(2), _ => println(0), } \
             match 'z' { 'z' => println(3), _ => println(4), } \
         }"
        .to_string(),
    );
}

// ---- WP-C4.6 A2-2: general + nested pattern lowering ----

/// Tuple scrutinees, array patterns, and deep nesting (`Some((a, Some(b)))`,
/// `((a, b), [c, d])`) all agree with the oracle, including arm ORDER (first match wins).
#[test]
fn nested_and_compound_patterns_agree() {
    differential(
        "a2_nested.stark",
        "fn main() { \
             let t = (1, 2); \
             match t { (0, b) => println(b), (a, b) => println(a + b), } \
             let v: Option<(Int32, Option<Int32>)> = Some((5, Some(7))); \
             match v { \
                 Some((a, Some(b))) => println(a * 100 + b), \
                 Some((a, None)) => println(a), \
                 None => println(-1), \
             } \
             let w: Option<(Int32, Option<Int32>)> = Some((9, None)); \
             match w { \
                 Some((a, Some(b))) => println(a * 100 + b), \
                 Some((a, None)) => println(a), \
                 None => println(-1), \
             } \
             let u = ((1, 2), [3, 4]); \
             match u { ((a, b), [c, d]) => println(a + b + c + d), } \
         }"
        .to_string(),
    );
}

/// Struct patterns — literal sub-patterns select arms in order; bindings and shorthand bind.
#[test]
fn struct_patterns_agree() {
    differential(
        "a2_struct_pat.stark",
        "struct Point { x: Int32, y: Int32 } \
         fn main() { \
             let p = Point { x: 3, y: 4 }; \
             match p { \
                 Point { x: 0, y } => println(y), \
                 Point { x, y: 0 } => println(x), \
                 Point { x, y } => println(x + y), \
             } \
             let q = Point { x: 0, y: 9 }; \
             match q { Point { x: 0, y } => println(y * 10), _ => println(-1), } \
         }"
        .to_string(),
    );
}

/// String literal patterns on a `&str` scrutinee compare by content (StrEq, never structural).
#[test]
fn string_literal_patterns_agree() {
    differential(
        "a2_str_pat.stark",
        "fn main() { \
             let s = String::from(\"beta\"); \
             match s.as_str() { \
                 \"alpha\" => println(1), \
                 \"beta\" => println(2), \
                 _ => println(0), \
             } \
             match \"x\" { \"x\" => println(3), _ => println(4), } \
         }"
        .to_string(),
    );
}

/// Float literal patterns use spec-exact IEEE equality.
#[test]
fn float_literal_patterns_agree() {
    differential(
        "a2_float_pat.stark",
        "fn main() { \
             let x = 1.5; \
             match x { 1.5 => println(1), _ => println(0), } \
             match 2.5 { 1.5 => println(2), _ => println(3), } \
         }"
        .to_string(),
    );
}

// ---- WP-C4.6 A1: generic impl monomorphisation ----

/// Methods, associated fns, a trait impl with a default method, and a Drop impl — all on
/// GENERIC nominals, across two instantiations (Int32 + String, incl. a Vec<T> field) — agree
/// with the oracle end to end, including dtor timing.
#[test]
fn generic_impls_full_matrix_agree() {
    differential(
        "a1_full.stark",
        "struct Stack<T> { items: Vec<T> } \
         impl<T> Stack<T> { \
             fn make() -> Stack<T> { Stack { items: Vec::new() } } \
             fn push_item(&mut self, v: T) { self.items.push(v); } \
             fn size(&self) -> UInt64 { self.items.len() } \
         } \
         trait Describe { fn id(&self) -> Int32; fn twice(&self) -> Int32 { self.id() * 2 } } \
         struct Tagged<T> { v: T, n: Int32 } \
         impl<T> Describe for Tagged<T> { fn id(&self) -> Int32 { self.n } } \
         impl<T> Drop for Tagged<T> { fn drop(&mut self) { println(self.n); } } \
         fn main() { \
             let mut s: Stack<Int32> = Stack::make(); \
             s.push_item(4); \
             s.push_item(5); \
             println(s.size()); \
             let mut t: Stack<String> = Stack::make(); \
             t.push_item(String::from(\"hi\")); \
             println(t.size()); \
             let g = Tagged { v: 9, n: 77 }; \
             println(g.id()); \
             println(g.twice()); \
             println(100); \
         }"
        .to_string(),
    );
}

/// A method returning `&T` through auto-deref (`*h.get()`) and a generic Drop with distinct
/// instantiations dropping in reverse declaration order.
#[test]
fn generic_method_ref_return_and_drop_order_agree() {
    differential(
        "a1_refret.stark",
        "struct Holder<T> { value: T } \
         impl<T> Holder<T> { fn get(&self) -> &T { &self.value } } \
         struct Res<T> { value: T } \
         impl<T> Drop for Res<T> { fn drop(&mut self) { println(0 - 1); } } \
         fn main() { \
             let h = Holder { value: 7 }; \
             println(*h.get()); \
             let a = Res { value: 1 }; \
             let b = Res { value: String::from(\"x\") }; \
             println(50); \
         }"
        .to_string(),
    );
}

/// A user `Iterator` impl drives a `for` loop: `it.next()` instance calls yield
/// `Option<Item>` by value until `None`, agreeing with the oracle.
#[test]
fn user_iterator_for_loop_agrees() {
    differential(
        "a1_user_iter.stark",
        "struct Counter { n: Int32 } \
         impl Iterator for Counter { \
             type Item = Int32; \
             fn next(&mut self) -> Option<Int32> { \
                 if self.n > 0 { self.n = self.n - 1; Some(self.n) } else { None } \
             } \
         } \
         fn main() { let c = Counter { n: 3 }; for x in c { println(x); } println(9); }"
            .to_string(),
    );
}

/// DEV-073 (WP-C4.7-5): a GENERIC impl satisfies a concrete instantiation's operator bound —
/// `impl<T> Eq for W<T>` makes `W<Int32> == W<Int32>` legal. The checker used to reject this
/// (E0500) because it demanded an exact match against an impl self type whose generic arguments
/// had been dropped, so this test could not exist. MIR dispatch was already instantiation-ready
/// from WP-C4.6 A1 and needed no change — which is what this test confirms.
#[test]
fn generic_impl_eq_dispatch_agrees() {
    differential(
        "generic_eq.stark",
        "struct W<T> { v: T } \
         impl<T> Eq for W<T> { fn eq(&self, other: &W<T>) -> Bool { true } } \
         fn main() { \
             let a = W { v: 1 }; \
             let b = W { v: 2 }; \
             if a == b { println(1); } else { println(0); } \
         }"
        .to_string(),
    );
}

/// DEV-073, the iterable half: `impl<T> Iterator for Repeat<T>` makes `Repeat<Int32>` usable in a
/// `for` loop, with the associated `Item = T` substituted to the instantiation's `Int32`. Also
/// rejected (E0001) before the fix.
#[test]
fn generic_user_iterator_for_loop_agrees() {
    differential(
        "generic_iter.stark",
        "struct Repeat<T> { item: T, left: Int32 } \
         impl<T> Iterator for Repeat<T> { \
             type Item = T; \
             fn next(&mut self) -> Option<T> { \
                 if self.left == 0 { None } else { self.left = self.left - 1; Some(self.item) } \
             } \
         } \
         fn main() { \
             let mut r = Repeat { item: 7, left: 3 }; \
             for x in r { println(x); } \
         }"
        .to_string(),
    );
}

/// WP-C4.7-6.2: `Ord::cmp` on a PRIMITIVE receiver. 06-Standard-Library specifies
/// `impl Ord for Int32 { fn cmp(&self, other: &Int32) -> Ordering }` "and similar for other
/// types", and `Ordering` is `core-min` prelude — but `3.cmp(&5)` failed E0304 ("method call on
/// non-struct/enum type"), so the only way to obtain an `Ordering` was a user `Ord` impl.
///
/// The lowering constructs the variant from the same comparisons `<`/`==` already use (and
/// routes `String` through `StrCmp`), which is what makes `a.cmp(&b)` and `a < b` agree by
/// construction rather than by coincidence.
#[test]
fn primitive_cmp_agrees() {
    differential(
        "prim_cmp.stark",
        "fn label(o: Ordering) -> Int32 { \
             match o { \
                 Ordering::Less => 1, \
                 Ordering::Equal => 2, \
                 _ => 3, \
             } \
         } \
         fn main() { \
             println(label(3.cmp(&5))); \
             println(label(5.cmp(&5))); \
             println(label(9.cmp(&5))); \
             println(label(String::from(\"a\").cmp(&String::from(\"b\")))); \
             println(label(String::from(\"zz\").cmp(&String::from(\"zz\")))); \
             let x = 7; \
             println(label(x.cmp(&x))); \
         }"
        .to_string(),
    );
}

/// The `cmp`/`<` consistency property stated as a test rather than assumed: for the same pair,
/// the `Ordering` a `cmp` reports and the answer the ordered operators give must never disagree.
/// Both engines are checked against each other, and the expected output is pinned.
#[test]
fn primitive_cmp_and_ordered_operators_agree() {
    differential(
        "cmp_vs_ops.stark",
        "fn check(a: Int32, b: Int32) -> Int32 { \
             let via_cmp = match a.cmp(&b) { \
                 Ordering::Less => 1, \
                 Ordering::Equal => 2, \
                 _ => 3, \
             }; \
             let via_ops = if a < b { 1 } else { if a == b { 2 } else { 3 } }; \
             if via_cmp == via_ops { 0 } else { 99 } \
         } \
         fn main() { \
             println(check(1, 2)); \
             println(check(2, 2)); \
             println(check(3, 2)); \
             println(check(-5, 5)); \
         }"
        .to_string(),
    );
}

/// DEV-067(b) (WP-C4.7-7): a trait-bound method called through a `&T` RECEIVER. TYPE-METHOD-002
/// requires auto-dereference to peel leading `&`/`&mut` before receiver matching, but the
/// bounded-parameter lookup tested the unpeeled type, so `t.speak()` on `t: &T` with `T: Speak`
/// failed E0302 "method 'speak' not found for type '&T'" while the by-value form worked.
/// Instantiated at two types, so the monomorphised dispatch is exercised, not just the check.
#[test]
fn bounded_generic_method_through_reference_agrees() {
    differential(
        "bound_ref.stark",
        "trait Speak { fn speak(&self) -> Int32; } \
         struct Dog { n: Int32 } \
         struct Cat { n: Int32 } \
         impl Speak for Dog { fn speak(&self) -> Int32 { self.n * 2 } } \
         impl Speak for Cat { fn speak(&self) -> Int32 { self.n * 3 } } \
         fn call_speak<T: Speak>(t: &T) -> Int32 { t.speak() } \
         fn main() { \
             let d = Dog { n: 7 }; \
             let c = Cat { n: 5 }; \
             println(call_speak(&d)); \
             println(call_speak(&c)); \
         }"
        .to_string(),
    );
}

/// DEV-067(a): a bound on a generic parameter is discharged by the ENCLOSING function's own
/// declared bound (TYPE-GENERIC-001). Any generic fn calling another generic fn with a bounded
/// parameter — including plain recursion — used to fail E0500 "type 'T' does not satisfy trait
/// bound 'Ord'" even with `T: Ord` declared on the caller. Bound obligations are checked in a
/// deferred pass, so the fix had to carry the enclosing generic environment with each obligation.
#[test]
fn bounded_generic_call_chain_agrees() {
    differential(
        "bound_chain.stark",
        "fn biggest<T: Ord>(a: T, b: T) -> T { if a < b { b } else { a } } \
         fn chain<T: Ord>(a: T, b: T) -> T { biggest(a, b) } \
         fn deeper<T: Ord>(a: T, b: T) -> T { chain(a, b) } \
         fn main() { \
             println(deeper(3, 9)); \
             println(deeper(40, 2)); \
         }"
        .to_string(),
    );
}

/// 0.1-A7 (WP-C4.7-6.1): `Box<T>` construction and extraction. 06-Standard-Library lists
/// `Box<T>` in `core-min` and gives it exactly `new` and `into_inner`; both reached the front end
/// and the oracle but not MIR. Core v1 has NO `Deref` trait, so `into_inner` is the only way out
/// of a box — see `box_deref_is_rejected` in the front-end tests for the other half of that.
#[test]
fn box_new_and_into_inner_agree() {
    differential(
        "box_basic.stark",
        "fn main() { \
             let b = Box::new(5); \
             println(b.into_inner()); \
             let s = Box::new(String::from(\"hi\")); \
             println(s.into_inner()); \
         }"
        .to_string(),
    );
}

/// The destructor-timing case, which is what makes the Box representation load-bearing rather
/// than cosmetic: `into_inner` transfers the payload OUT without running its destructor (the
/// caller now owns it and drops it at end of scope), while a box that is simply dropped destroys
/// its contents exactly once. Both engines must agree on the exact interleaving, so the printed
/// order — not just the multiset of lines — is the assertion.
#[test]
fn box_drop_timing_agrees() {
    differential(
        "box_drop.stark",
        "struct Tag { id: Int32 } \
         impl Drop for Tag { fn drop(&mut self) { println(self.id); } } \
         fn main() { \
             println(100); \
             let moved = Box::new(Tag { id: 1 }); \
             let t = moved.into_inner(); \
             println(200); \
             let untouched = Box::new(Tag { id: 2 }); \
             println(300); \
         }"
        .to_string(),
    );
}

/// A finite value of a RECURSIVE type. This is the reason `Box<T>` must stay an opaque owning
/// handle instead of being lowered transparently as `T`: with a transparent box, `Node` would
/// contain itself and be infinitely sized. It also pins the cycle guard in drop-instance
/// discovery — without it, walking `Node -> Option<Box<Node>> -> Box<Node> -> Node` overflows
/// the stack (observed while building this).
#[test]
fn box_recursive_type_agrees() {
    differential(
        "box_rec.stark",
        "struct Node { value: Int32, next: Option<Box<Node>> } \
         fn main() { \
             let tail = Node { value: 2, next: None }; \
             let head = Node { value: 1, next: Some(Box::new(tail)) }; \
             println(head.value); \
             match head.next { \
                 Some(b) => { let n = b.into_inner(); println(n.value); } \
                 None => { println(-1); } \
             } \
         }"
        .to_string(),
    );
}

/// WP-C4.7-6.3: expected-type propagation into unsuffixed integer literals must produce the same
/// RUNTIME values in both engines — the widths a literal adopts are observable through `UInt64`
/// arithmetic and through `Vec::get`'s indexing, so this is not merely a checker-side property.
#[test]
fn expected_typed_integer_literals_agree() {
    differential(
        "lit_expected.stark",
        "fn takes_u64(n: UInt64) -> UInt64 { n + 1 } \
         struct S { n: UInt64 } \
         fn main() { \
             println(takes_u64(0)); \
             println(takes_u64(41)); \
             let s = S { n: 3 }; \
             println(s.n); \
             let a: UInt64 = 9; \
             println(a); \
             let mut v: Vec<Int32> = Vec::new(); \
             v.push(7); \
             v.push(8); \
             let index = 1; \
             match v.get(index) { Some(x) => println(*x), None => println(-1), } \
             let unconstrained = 5; \
             println(unconstrained); \
         }"
        .to_string(),
    );
}

/// DEV-075 (owner specification decision, 2026-07-20; PRIM-TRAIT-001): `Char` is totally ordered
/// by **Unicode scalar value**. Before this, the oracle rejected all four ordered operators on
/// `Char` ("invalid binary operation") while MIR executed them correctly — an ENGINE DIVERGENCE
/// that survived only because no test compared an ordered operator on `Char`. That is exactly the
/// class of defect this harness exists to catch, so the four operators and `cmp` are pinned
/// together here.
#[test]
fn char_ordering_agrees() {
    differential(
        "char_ord.stark",
        "fn label(o: Ordering) -> Int32 { \
             match o { \
                 Ordering::Less => 1, \
                 Ordering::Equal => 2, \
                 Ordering::Greater => 3, \
             } \
         } \
         fn main() { \
             if 'a' < 'b' { println(1); } else { println(0); } \
             if 'b' <= 'b' { println(1); } else { println(0); } \
             if 'c' > 'b' { println(1); } else { println(0); } \
             if 'a' >= 'b' { println(1); } else { println(0); } \
             println(label('a'.cmp(&'b'))); \
             println(label('b'.cmp(&'b'))); \
             println(label('z'.cmp(&'b'))); \
             if 'a' == 'a' { println(1); } else { println(0); } \
             if true == true { println(1); } else { println(0); } \
         }"
        .to_string(),
    );
}

/// The scalar-value rule stated as a comparison a *collation* order would get wrong: `'Z'` (0x5A)
/// sorts BEFORE `'a'` (0x61), and a digit before both. Any locale-aware ordering would disagree,
/// so this distinguishes the specified rule from a plausible alternative rather than merely
/// re-testing that comparison works.
#[test]
fn char_ordering_is_scalar_value_not_collation_agrees() {
    differential(
        "char_scalar.stark",
        "fn main() { \
             if 'Z' < 'a' { println(1); } else { println(0); } \
             if '0' < 'A' { println(1); } else { println(0); } \
             if 'a' < 'z' { println(1); } else { println(0); } \
         }"
        .to_string(),
    );
}

/// WP-C4.7-8.1: `unwrap_or` with a DROPPABLE payload/default. The construct discards exactly one
/// of two values, and the discarded one owes a destructor — Core has no laziness, so the default
/// is evaluated whether or not it is used, which is precisely why it always owes one.
///
/// The timing is the assertion, and it is not the obvious answer: the discarded value is
/// destroyed **at the `unwrap_or` call**, not at end of scope. This was pinned against the oracle
/// before the lowering was written (§0.6) — and DEV-076 had to be fixed first, because the oracle
/// used to destroy the payload twice and the default never, so matching it would have written a
/// double drop into the backend contract.
#[test]
fn droppable_unwrap_or_drop_timing_agrees() {
    differential(
        "unwrap_or_drop.stark",
        "struct Tag { id: Int32 } \
         impl Drop for Tag { fn drop(&mut self) { println(self.id); } } \
         fn main() { \
             println(100); \
             let some = Some(Tag { id: 1 }); \
             let taken = some.unwrap_or(Tag { id: 2 }); \
             println(200); \
             println(taken.id); \
             println(300); \
             let none: Option<Tag> = None; \
             let defaulted = none.unwrap_or(Tag { id: 3 }); \
             println(400); \
             println(defaulted.id); \
             println(500); \
         }"
        .to_string(),
    );
}

/// The `Result` form, including the case with no `Option` analogue: on `Err` the default is
/// yielded and the DISPLACED ERROR PAYLOAD is destroyed at the call, discarded exactly as the
/// default is on the other path. Both type arguments carry destructors so neither can hide.
#[test]
fn droppable_result_unwrap_or_drops_the_error_payload_agrees() {
    differential(
        "unwrap_or_res.stark",
        "struct Tag { id: Int32 } \
         impl Drop for Tag { fn drop(&mut self) { println(self.id); } } \
         fn ok_case() -> Result<Tag, Tag> { Ok(Tag { id: 1 }) } \
         fn err_case() -> Result<Tag, Tag> { Err(Tag { id: 9 }) } \
         fn main() { \
             println(100); \
             let a = ok_case().unwrap_or(Tag { id: 2 }); \
             println(200); \
             println(a.id); \
             println(300); \
             let b = err_case().unwrap_or(Tag { id: 3 }); \
             println(400); \
             println(b.id); \
             println(500); \
         }"
        .to_string(),
    );
}

/// A `String` payload — the runtime-type drop path rather than a user `Drop` impl. No destructor
/// output to observe, so this pins that the buffer reclaim happens without a double free and that
/// the two engines agree on the value.
#[test]
fn droppable_unwrap_or_with_runtime_type_agrees() {
    differential(
        "unwrap_or_string.stark",
        "fn main() { \
             let present: Option<String> = Some(String::from(\"hi\")); \
             println(present.unwrap_or(String::from(\"fallback\"))); \
             let absent: Option<String> = None; \
             println(absent.unwrap_or(String::from(\"fallback\"))); \
         }"
        .to_string(),
    );
}

/// WP-C4.7-8.2: a user `Iterator` whose `Item` needs dropping. Each yielded value is destroyed at
/// the END OF ITS OWN ITERATION, not accumulated to loop exit — pinned against the oracle before
/// the lowering was written (§0.6). The printed ORDER is the assertion: body, value, DROP,
/// repeated, rather than three drops trailing after the loop.
#[test]
fn droppable_iterator_item_drop_timing_agrees() {
    differential(
        "iter_item_drop.stark",
        "struct Tag { id: Int32 } \
         impl Drop for Tag { fn drop(&mut self) { println(self.id); } } \
         struct Gen { left: Int32 } \
         impl Iterator for Gen { \
             type Item = Tag; \
             fn next(&mut self) -> Option<Tag> { \
                 if self.left == 0 { None } \
                 else { self.left = self.left - 1; Some(Tag { id: self.left }) } \
             } \
         } \
         fn main() { \
             println(100); \
             let mut g = Gen { left: 3 }; \
             for t in g { println(900); println(t.id); } \
             println(200); \
         }"
        .to_string(),
    );
}

/// The early-exit paths, which is where a per-iteration scope is easy to get wrong: `break` must
/// destroy the CURRENT iteration's value before leaving, and `continue` must destroy it before
/// looping back. Both fall out of capturing the loop's `scope_depth` before pushing the
/// per-iteration scope — so this test is really checking that ordering decision.
#[test]
fn droppable_iterator_item_break_and_continue_agree() {
    let iter_decl = "struct Tag { id: Int32 } \
         impl Drop for Tag { fn drop(&mut self) { println(self.id); } } \
         struct Gen { left: Int32 } \
         impl Iterator for Gen { \
             type Item = Tag; \
             fn next(&mut self) -> Option<Tag> { \
                 if self.left == 0 { None } \
                 else { self.left = self.left - 1; Some(Tag { id: self.left }) } \
             } \
         } ";
    differential(
        "iter_item_break.stark",
        format!(
            "{iter_decl} fn main() {{ \
                 println(100); \
                 let mut g = Gen {{ left: 5 }}; \
                 for t in g {{ println(900); if t.id == 2 {{ break; }} }} \
                 println(200); \
             }}"
        ),
    );
    differential(
        "iter_item_continue.stark",
        format!(
            "{iter_decl} fn main() {{ \
                 println(100); \
                 let mut g = Gen {{ left: 4 }}; \
                 for t in g {{ if t.id == 2 {{ continue; }} println(900); }} \
                 println(200); \
             }}"
        ),
    );
}

/// DEV-079 (WP-C4.7-8.3): an enum variant with TWO OR MORE droppable payload fields. V-MOVE-1's
/// move dataflow collapsed every non-`Field` projection to the whole local, so moving the second
/// payload field looked like a second move of the same place: lowering accepted the program and
/// verification rejected it with MIR-0007. That is an internal inconsistency between two
/// components meant to be independent readings of the same contract — strictly worse than a clean
/// `Unsupported` — and it applied to EVERY multi-droppable-field variant, wildcard or not.
///
/// No corpus case had a variant carrying two droppable fields, which is why A2's "general pattern
/// engine" and C4.5d's match-drop elaboration both signed off over it.
#[test]
fn enum_variant_with_two_droppable_fields_agrees() {
    differential(
        "two_drop_fields.stark",
        "struct Tag { id: Int32 } \
         impl Drop for Tag { fn drop(&mut self) { println(self.id); } } \
         enum Two { Pair(Tag, Tag), Empty } \
         fn main() { \
             let v = Two::Pair(Tag { id: 1 }, Tag { id: 2 }); \
             match v { \
                 Two::Pair(a, b) => { println(a.id); println(b.id); } \
                 Two::Empty => { println(0); } \
             } \
         }"
        .to_string(),
    );
    // The runtime-type payload form: no user destructor to print, but still two droppable fields
    // moved out of one local.
    differential(
        "two_drop_strings.stark",
        "enum Two { Pair(String, String), Empty } \
         fn main() { \
             let v = Two::Pair(String::from(\"x\"), String::from(\"y\")); \
             match v { \
                 Two::Pair(a, b) => { println(a); println(b); } \
                 Two::Empty => { println(0); } \
             } \
         }"
        .to_string(),
    );
}

/// DEV-080 (WP-C4.7-8.3): arm-end drop ORDER when some payload fields are bound and others are
/// discarded. The oracle destroys all BOUND bindings first, in reverse binding order, and the
/// discarded leaves after them. MIR used plain reverse-FIELD order.
///
/// The three-field `(a, _, c)` case is the discriminating one: the expected order is `c`, `a`,
/// then the wildcard leaf — matching neither plain reverse-field order (`c`, wildcard, `a`) nor
/// declaration order, so it pins the actual rule rather than a coincidence. This divergence was
/// unobservable until DEV-079 was fixed, because such programs could not verify at all.
#[test]
fn variant_payload_drop_order_with_wildcards_agrees() {
    let decl = "struct Tag { id: Int32 } \
         impl Drop for Tag { fn drop(&mut self) { println(self.id); } } ";
    differential(
        "drop_order_wild.stark",
        format!(
            "{decl} enum Two {{ Pair(Tag, Tag), Empty }} \
             fn main() {{ \
                 let v = Two::Pair(Tag {{ id: 1 }}, Tag {{ id: 2 }}); \
                 match v {{ \
                     Two::Pair(a, _) => {{ println(900); println(a.id); }} \
                     Two::Empty => {{ println(0); }} \
                 }} \
                 println(200); \
             }}"
        ),
    );
    differential(
        "drop_order_wild3.stark",
        format!(
            "{decl} enum Three {{ Trip(Tag, Tag, Tag), Empty }} \
             fn main() {{ \
                 let v = Three::Trip(Tag {{ id: 1 }}, Tag {{ id: 2 }}, Tag {{ id: 3 }}); \
                 match v {{ \
                     Three::Trip(a, _, c) => {{ println(900); println(a.id); println(c.id); }} \
                     Three::Empty => {{ println(0); }} \
                 }} \
                 println(200); \
             }}"
        ),
    );
}

/// WP-C4.7-8.3b: a DROPPABLE scrutinee under NESTED patterns — the last recorded MIR residual of
/// the Class-A campaign. A consuming match decomposes the scrutinee completely, so every leaf the
/// pattern discards still owes a destructor; the unbound walk moves those into arm-scoped temps
/// while bindings own what they bind.
///
/// Ordering is the real assertion. Bindings are destroyed first, in reverse binding order, then
/// the discarded leaves — so `Some((a, _, c))` over three tagged values must give `c`, `a`, then
/// the wildcard. That matches neither plain reverse-field order nor declaration order, which is
/// what makes it evidence of the rule rather than a coincidence.
#[test]
fn droppable_nested_pattern_drop_order_agrees() {
    let decl = "struct Tag { id: Int32 } \
         impl Drop for Tag { fn drop(&mut self) { println(self.id); } } ";
    // One binding, one discarded leaf.
    differential(
        "nested_drop_wild.stark",
        format!(
            "{decl} fn main() {{ \
                 println(100); \
                 let pair = Some((Tag {{ id: 1 }}, Tag {{ id: 2 }})); \
                 match pair {{ \
                     Some((a, _)) => {{ println(900); println(a.id); }} \
                     None => {{ println(0); }} \
                 }} \
                 println(200); \
             }}"
        ),
    );
    // Two bindings: reverse binding order.
    differential(
        "nested_drop_both.stark",
        format!(
            "{decl} fn main() {{ \
                 let pair = Some((Tag {{ id: 1 }}, Tag {{ id: 2 }})); \
                 match pair {{ \
                     Some((a, b)) => {{ println(900); println(b.id); println(a.id); }} \
                     None => {{ println(0); }} \
                 }} \
                 println(200); \
             }}"
        ),
    );
    // The discriminating case: bindings first (reverse), then the discarded middle leaf.
    differential(
        "nested_drop_three.stark",
        format!(
            "{decl} fn main() {{ \
                 let t = Some((Tag {{ id: 1 }}, Tag {{ id: 2 }}, Tag {{ id: 3 }})); \
                 match t {{ \
                     Some((a, _, c)) => {{ println(900); println(a.id); println(c.id); }} \
                     None => {{ println(0); }} \
                 }} \
                 println(200); \
             }}"
        ),
    );
    // A whole-payload wildcard: the tuple drops as a unit, in reverse field order.
    differential(
        "nested_drop_whole.stark",
        format!(
            "{decl} fn main() {{ \
                 let t = Some((Tag {{ id: 1 }}, Tag {{ id: 2 }})); \
                 match t {{ Some(_) => {{ println(900); }} None => {{ println(0); }} }} \
                 println(200); \
             }}"
        ),
    );
}

/// Deeper nesting and a mixed runtime/user-Drop payload: the decomposition is recursive, so a
/// two-level pattern and a `String`-plus-`Tag` tuple must behave like the one-level cases.
#[test]
fn droppable_nested_pattern_depth_and_mixed_payloads_agree() {
    let decl = "struct Tag { id: Int32 } \
         impl Drop for Tag { fn drop(&mut self) { println(self.id); } } ";
    differential(
        "nested_two_level.stark",
        format!(
            "{decl} fn main() {{ \
                 let n = Some(Some((Tag {{ id: 1 }}, Tag {{ id: 2 }}))); \
                 match n {{ \
                     Some(Some((a, _))) => {{ println(900); println(a.id); }} \
                     Some(None) => {{ println(1); }} \
                     None => {{ println(0); }} \
                 }} \
                 println(200); \
             }}"
        ),
    );
    differential(
        "nested_mixed_payload.stark",
        format!(
            "{decl} fn main() {{ \
                 let t = Some((String::from(\"s\"), Tag {{ id: 7 }})); \
                 match t {{ \
                     Some((s, _)) => {{ println(900); println(s); }} \
                     None => {{ println(0); }} \
                 }} \
                 println(200); \
             }}"
        ),
    );
}

/// DEV-081 (WP-C4.7-8.3b): a SHORTHAND struct-field binding (`P { a, b }`) owns its moved-in
/// value and must drop at arm end. `bind_shorthand` never registered it, in any mode, so the
/// value was moved out of the scrutinee and then destroyed by nobody — a LEAK, which is why it
/// failed silently: no verifier rule broken, no assertion tripped, and invisible to any program
/// whose destructor does not print. Both the struct-nominal and struct-shaped-enum-variant forms
/// are covered, because the flat path had the same gap before 8.3b existed.
#[test]
fn struct_shorthand_bindings_drop_agrees() {
    let decl = "struct Tag { id: Int32 } \
         impl Drop for Tag { fn drop(&mut self) { println(self.id); } } ";
    differential(
        "shorthand_struct.stark",
        format!(
            "{decl} struct P {{ a: Tag, b: Tag }} \
             fn main() {{ \
                 let p = Some(P {{ a: Tag {{ id: 1 }}, b: Tag {{ id: 2 }} }}); \
                 match p {{ \
                     Some(P {{ a, b }}) => {{ println(900); println(a.id); println(b.id); }} \
                     None => {{ println(0); }} \
                 }} \
                 println(200); \
             }}"
        ),
    );
    differential(
        "shorthand_variant.stark",
        format!(
            "{decl} enum E {{ V {{ a: Tag, b: Tag }}, Empty }} \
             fn main() {{ \
                 let e = E::V {{ a: Tag {{ id: 1 }}, b: Tag {{ id: 2 }} }}; \
                 match e {{ \
                     E::V {{ a, b }} => {{ println(900); println(a.id); println(b.id); }} \
                     E::Empty => {{ println(0); }} \
                 }} \
                 println(200); \
             }}"
        ),
    );
}

/// WP-C4.7-8.6 (surface `0.1-A8`): EXCLUSIVE slice views. REF-SLICE-001 states that "writes
/// through an exclusive slice reference update the original object", and 03-Type-System gives
/// `&mut expr[r]` the type `&mut [T]`, so this is normative Core rather than an optional
/// extension — the owner decided accordingly rather than deferring it past C4.
///
/// Covers the three things that can independently go wrong: write-through reaching the BASE
/// object (array and `Vec`), a view passed to a function that mutates it, and repeated use of a
/// `&mut [T]` local (DEV-082 — a method call on a slice receiver used to consume it, which was
/// invisible while only shared views existed because `&[T]` is `Copy`).
#[test]
fn mutable_slice_views_agree() {
    // Write through a view; the base observes it.
    differential(
        "mut_slice_write.stark",
        "fn main() { \
             let mut a = [1, 2, 3, 4, 5]; \
             let s = &mut a[1..4]; \
             s[0] = 99; \
             println(s.len()); \
             println(s[0]); \
             println(a[1]); \
         }"
        .to_string(),
    );
    // A view as a function argument, mutated through the parameter.
    differential(
        "mut_slice_param.stark",
        "fn bump(s: &mut [Int32]) -> Unit { s[0] = s[0] + 100; } \
         fn main() { \
             let mut a = [1, 2, 3]; \
             bump(&mut a[0..2]); \
             println(a[0]); \
             println(a[1]); \
         }"
        .to_string(),
    );
    // A `Vec` base, writing at a view-relative index that is not zero.
    differential(
        "mut_slice_vec.stark",
        "fn main() { \
             let mut v: Vec<Int32> = Vec::new(); \
             v.push(7); v.push(8); v.push(9); \
             let s = &mut v[1..3]; \
             s[1] = 55; \
             println(v[2]); \
         }"
        .to_string(),
    );
    // DEV-082: a read-only method call must not consume an exclusive view.
    differential(
        "mut_slice_reuse.stark",
        "fn main() { \
             let mut a = [1, 2, 3, 4, 5]; \
             let s = &mut a[1..4]; \
             println(s.len()); \
             println(s.len()); \
             println(s[2]); \
         }"
        .to_string(),
    );
}

/// WP-C4.7-8.5: NON-BARE impl heads — `impl<T> Holder<Option<T>>` applying to
/// `Holder<Option<Int32>>`. `02:117` admits any `Type` as an impl self type, so this is normative
/// Core; C4.7-2 found it front-end-blocked (E0302 "method not found") rather than a MIR gap.
///
/// Both engines needed the same generalization, and they had to agree: the checker's
/// `unify_impl_ty` decides WHICH impls apply, and lowering's `bind_written_impl_arg` recovers the
/// substitution that decision implies. If they disagreed, the front end would admit programs
/// lowering then rejects — the DEV-079 failure shape.
///
/// Two instantiations, so what is exercised is monomorphised dispatch through a non-bare head,
/// not merely that the checker stopped complaining.
#[test]
fn non_bare_impl_heads_agree() {
    differential(
        "nonbare_trait.stark",
        "struct Holder<T> { v: T } \
         trait Wrap { fn wrapped(&self) -> Int32; } \
         impl<T> Wrap for Holder<Option<T>> { fn wrapped(&self) -> Int32 { 1 } } \
         fn main() { let h = Holder { v: Some(3) }; println(h.wrapped()); }"
            .to_string(),
    );
    differential(
        "nonbare_inherent.stark",
        "struct Holder<T> { v: T } \
         impl<T> Holder<Option<T>> { \
             fn pick(&self, fallback: Int32) -> Int32 { fallback } \
         } \
         fn main() { \
             let a = Holder { v: Some(3) }; \
             let b = Holder { v: Some(true) }; \
             println(a.pick(11)); \
             println(b.pick(22)); \
         }"
        .to_string(),
    );
    // A concrete position in the head is fine once the receiver's type is known (DEV-083 records
    // the remaining case, where it is still an inference variable at resolution time).
    differential(
        "nonbare_concrete_pos.stark",
        "struct Pair<A, B> { x: A, y: B } \
         impl<T> Pair<Option<T>, Int32> { fn tag(&self) -> Int32 { self.y } } \
         fn make() -> Pair<Option<Int32>, Int32> { Pair { x: Some(5), y: 42 } } \
         fn main() { let p = make(); println(p.tag()); }"
            .to_string(),
    );
}

/// WP-C4.7-8.4: METHOD-OWN generic parameters — `impl Holder { fn echo<U>(&self, x: U) -> U }`.
/// `02:64` puts `GenericParams?` on every `FunctionSig` and `02:120` makes an impl item a
/// `Function`, so this is normative Core; C4.7-2 found it front-end-blocked rather than a MIR gap.
///
/// Two halves had to meet. The checker instantiated only the IMPL's parameters, leaving `U` a
/// rigid `Ty::Param` that no argument could unify with; and MIR had no way to monomorphise a
/// method at arguments the impl does not mention. `FnKey::ImplFn` now carries `method_args`
/// alongside the impl's `type_args`, filled from a per-call-site record keyed by the call
/// expression — the method equivalent of C4.5c's machinery for top-level generic fns.
///
/// Each case uses MULTIPLE instantiations, so what is exercised is one lowered body per
/// instantiation rather than merely the checker's acceptance.
#[test]
fn method_own_generics_agree() {
    // Two instantiations at different primitive types.
    differential(
        "method_generic_basic.stark",
        "struct Holder { v: Int32 } \
         impl Holder { fn echo<U>(&self, x: U) -> U { x } } \
         fn main() { \
             let h = Holder { v: 1 }; \
             println(h.echo(7)); \
             println(h.echo(true)); \
         }"
        .to_string(),
    );
    // Two method-own parameters in one signature, and a droppable instantiation.
    differential(
        "method_generic_pick.stark",
        "struct Holder { v: Int32 } \
         impl Holder { fn first<U>(&self, a: U, b: U) -> U { a } } \
         fn main() { \
             let h = Holder { v: 1 }; \
             println(h.first(7, 9)); \
             println(h.first(String::from(\"x\"), String::from(\"y\"))); \
         }"
        .to_string(),
    );
    // The combination that matters most: a GENERIC METHOD on a GENERIC NOMINAL. The impl-level
    // and method-level substitutions are separate and must not be conflated — `Holder<Int32>`
    // instantiated at two different `U`s, plus a second nominal instantiation.
    differential(
        "method_generic_on_generic.stark",
        "struct Holder<T> { v: T } \
         impl<T> Holder<T> { fn pair<U>(&self, other: U) -> U { other } } \
         fn main() { \
             let a = Holder { v: 1 }; \
             let b = Holder { v: true }; \
             println(a.pair(5)); \
             println(a.pair(false)); \
             println(b.pair(9)); \
         }"
        .to_string(),
    );
}

/// WP-C4.7-9 audit: `for x in a` over a fixed-length ARRAY. The checker accepted it and the
/// oracle ran it while MIR refused — an internal inconsistency rather than a language boundary,
/// which is exactly the shape the audit exists to find. Lowered as a counting loop reading one
/// element per iteration through the ordinary `CheckIndex` proof discipline.
#[test]
fn for_over_array_agrees() {
    differential(
        "for_array.stark",
        "fn main() { \
             let a = [1, 2, 3]; \
             let mut total = 0; \
             for x in a { println(x); total = total + x; } \
             println(total); \
         }"
        .to_string(),
    );
    // break/continue through the array loop, and a single-element array (boundary).
    differential(
        "for_array_control.stark",
        "fn main() { \
             let a = [10, 20, 30, 40]; \
             for x in a { if x == 30 { break; } if x == 20 { continue; } println(x); } \
             let one = [7]; \
             for y in one { println(y); } \
         }"
        .to_string(),
    );
}

/// WP-C4.7-9 audit: a trait DEFAULT method with its own generic parameters. WP-C4.7-8.4 gave the
/// selected-impl path fresh per-call-site variables; the trait-default path had the same gap in
/// the checker AND no `method_args` on `FnKey::TraitDefault` in lowering. Two instantiations, so
/// the monomorphisation is exercised rather than just the acceptance.
#[test]
fn trait_default_method_own_generics_agree() {
    differential(
        "trait_default_generic.stark",
        "trait Speak { fn say<U>(&self, x: U) -> U { x } } \
         struct D { n: Int32 } \
         impl Speak for D {} \
         fn main() { \
             let d = D { n: 1 }; \
             println(d.say(5)); \
             println(d.say(true)); \
         }"
        .to_string(),
    );
}

/// DEV-086 / MIR amendment A5 (CD-038): consuming array patterns with DROPPABLE elements, which
/// `Projection::ConstIndex` is what made expressible. A proof-backed `Index` cannot name a
/// statically-known sub-place — a dynamic proof forces move analysis to treat the whole array as
/// one unit — so moving one element out used to poison every sibling.
///
/// Drop ORDER is the assertion, and it follows the same rule established for variant payloads:
/// bound bindings first in reverse binding order, then the discarded leaves.
#[test]
fn droppable_array_pattern_agrees() {
    let decl = "struct Tag { id: Int32 } \
         impl Drop for Tag { fn drop(&mut self) { println(self.id); } } ";
    // One binding, one wildcard.
    differential(
        "array_pat_wild.stark",
        format!(
            "{decl} fn main() {{ \
                 let arr = Some([Tag {{ id: 1 }}, Tag {{ id: 2 }}]); \
                 match arr {{ \
                     Some([a, _]) => {{ println(900); println(a.id); }} \
                     None => {{ println(0); }} \
                 }} \
                 println(200); \
             }}"
        ),
    );
    // Both bound: reverse binding order.
    differential(
        "array_pat_both.stark",
        format!(
            "{decl} fn main() {{ \
                 let arr = Some([Tag {{ id: 1 }}, Tag {{ id: 2 }}]); \
                 match arr {{ \
                     Some([a, b]) => {{ println(a.id); println(b.id); }} \
                     None => {{ println(0); }} \
                 }} \
             }}"
        ),
    );
    // Three elements, middle discarded — distinguishes the rule from plain reverse-index order.
    differential(
        "array_pat_three.stark",
        format!(
            "{decl} fn main() {{ \
                 let arr = Some([Tag {{ id: 1 }}, Tag {{ id: 2 }}, Tag {{ id: 3 }}]); \
                 match arr {{ \
                     Some([a, _, c]) => {{ println(900); println(a.id); println(c.id); }} \
                     None => {{ println(0); }} \
                 }} \
                 println(200); \
             }}"
        ),
    );
    // A runtime-type element (String): the drop path with no user destructor to print.
    differential(
        "array_pat_string.stark",
        "fn main() { \
             let arr = Some([String::from(\"x\"), String::from(\"y\")]); \
             match arr { \
                 Some([a, _]) => { println(a); } \
                 None => { println(\"none\"); } \
             } \
         }"
        .to_string(),
    );
}

// ---- DEV-089 (WP-C4.7 close-out): user `Display` controls print/println ----

#[test]
fn dev089_user_struct_display_agrees() {
    // Required test 1 + 3: a user struct's `Display` output differs from its structural form.
    differential(
        "dev089_struct.stark",
        "struct P { x: Int32 } \
         impl Display for P { fn fmt(&self) -> String { String::from(\"P\") } } \
         fn main() { \
             let p = P { x: 1 }; \
             println(p); \
         }"
        .to_string(),
    );
}

#[test]
fn dev089_user_enum_display_agrees() {
    // Required test 2: a user enum's `Display`.
    differential(
        "dev089_enum.stark",
        "enum Color { Red, Green } \
         impl Display for Color { \
             fn fmt(&self) -> String { \
                 match *self { Color::Red => String::from(\"red\"), Color::Green => String::from(\"green\") } \
             } \
         } \
         fn main() { \
             println(Color::Red); \
             println(Color::Green); \
         }"
        .to_string(),
    );
}

#[test]
fn dev089_display_called_once_with_side_effect_agrees() {
    // Required tests 4 + 5: `fmt` runs exactly once, and its side effect is observable in order.
    differential(
        "dev089_once.stark",
        "struct C { n: Int32 } \
         impl Display for C { \
             fn fmt(&self) -> String { println(\"fmt-called\"); String::from(\"C\") } \
         } \
         fn main() { \
             let c = C { n: 5 }; \
             println(c); \
             println(\"done\"); \
         }"
        .to_string(),
    );
}

#[test]
fn dev089_dynamically_constructed_string_agrees() {
    // Required test 6: `fmt` returns a dynamically built String (push_str), not a literal.
    differential(
        "dev089_dynamic.stark",
        "struct Tag { n: Int32 } \
         impl Display for Tag { \
             fn fmt(&self) -> String { \
                 let mut s = String::from(\"tag-\"); \
                 s.push_str(\"x\"); \
                 s \
             } \
         } \
         fn main() { \
             println(Tag { n: 7 }); \
         }"
        .to_string(),
    );
}

#[test]
fn dev089_generic_function_with_display_bound_agrees() {
    // Required test 7: a generic function `show<T: Display>(v: T)` printing its parameter.
    differential(
        "dev089_generic_fn.stark",
        "struct P { x: Int32 } \
         impl Display for P { fn fmt(&self) -> String { String::from(\"P\") } } \
         fn show<T: Display>(v: T) { println(v); } \
         fn main() { \
             show(P { x: 1 }); \
             show(42); \
         }"
        .to_string(),
    );
}

#[test]
fn dev089_generic_nominal_display_agrees() {
    // Required test 8: a `Display` impl on a generic nominal, printed at a concrete instance.
    differential(
        "dev089_generic_nominal.stark",
        "struct Wrap<T> { value: T } \
         impl<T> Display for Wrap<T> { fn fmt(&self) -> String { String::from(\"wrap\") } } \
         fn main() { \
             let w: Wrap<Int32> = Wrap { value: 5 }; \
             println(w); \
         }"
        .to_string(),
    );
}

#[test]
fn dev089_formatter_result_and_argument_drop_timing_agrees() {
    // Required test 9: drop ordering — the argument's destructor runs AFTER its formatted bytes
    // are printed and BEFORE the following statement (ordinary by-value call ownership).
    differential(
        "dev089_drop_timing.stark",
        "struct D { x: Int32 } \
         impl Display for D { fn fmt(&self) -> String { String::from(\"D\") } } \
         impl Drop for D { fn drop(&mut self) { println(\"drop-D\"); } } \
         fn main() { \
             let d = D { x: 1 }; \
             println(d); \
             println(\"after\"); \
         }"
        .to_string(),
    );
}

#[test]
fn dev089_trap_inside_fmt_agrees() {
    // Required test 10: a trap inside `fmt` propagates as a trap in both engines, with the
    // pre-trap prefix identical and no formatted/newline output for the trapping call.
    differential(
        "dev089_trap.stark",
        "struct Boom { x: Int32 } \
         impl Display for Boom { \
             fn fmt(&self) -> String { let z = 0; println(1 / z); String::from(\"unreached\") } \
         } \
         fn main() { \
             println(\"before\"); \
             println(Boom { x: 1 }); \
             println(\"after\"); \
         }"
        .to_string(),
    );
}
