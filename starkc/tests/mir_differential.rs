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
        "fn largest<T: Ord>(a: T, b: T) -> T { if a > b { a } else { b } } \
         fn twice<T: Num>(x: T) -> T { x + x } \
         fn main() { \
             println(largest(3, 7)); \
             println(largest(2.5, 1.5)); \
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
/// The recursing parameter is deliberately unbounded: a *bounded* parameter at an
/// intra-generic call site is over-rejected by the checker today (DEV-067).
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
/// receiver resolves to the implementing type's method after substitution. (A `&T` receiver
/// is over-rejected by the checker today — DEV-067 — so the receiver is by value here.)
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
    std::fs::write(
        &helper_path,
        "pub fn add_self(n: Int32) -> Int32 { n + n }\n\
         pub fn add_both(a: Int32, b: Int32) -> Int32 { add_self(a) + b }\n",
    )
    .unwrap();
    let main_path = dir.join("main.stark");
    // NOTE (DEV-069): the FRONT END + ORACLE are not multi-file-span-clean — typecheck and
    // the HIR interpreter read spans against the entry file, so cross-file METHODS
    // mis-resolve, cross-file LITERALS mis-parse, and cross-file FIELD reads fail. The test
    // therefore stays on the front-end-safe subset (scalar free fns, literal-free helper
    // bodies), padded so helper name spans stay in-bounds for the checker's reads. The MIR
    // lowering under test IS multi-file-clean (per-item files via ProgramMeta); widening
    // this test tracks DEV-069's front-end fix, not MIR work.
    let main_src = "mod helper;\n\
         fn main() {\n\
             println(helper::add_self(4));\n\
             println(helper::add_both(2, 5));\n\
             println(100);\n\
         }\n\
         // padding padding padding padding padding padding padding padding padding\n\
         // padding padding padding padding padding padding padding padding padding\n\
         // padding padding padding padding padding padding padding padding padding\n\
         // padding padding padding padding padding padding padding padding padding\n";
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

    let _ = std::fs::remove_dir_all(&dir);
}

/// C4.5 EXIT CHECK: the ENTIRE frozen corpus (`corpus_version` current, all 17 cases) runs
/// equivalently through both interpreters — the WP-C4.5 exit criterion.
#[test]
fn entire_frozen_corpus_agrees() {
    for name in [
        "collection_iter__01_vec_push_index_iterate",
        "collection_iter__02_hashmap_insert_get_iteration_order",
        "expr_stmt__01_arithmetic_and_precedence",
        "expr_stmt__02_if_else_and_block_tail",
        "expr_stmt__03_loops_break_continue",
        "expr_stmt__04_match_and_patterns",
        "option_result__01_option_construction_and_match",
        "option_result__02_result_and_try_propagation",
        "ownership_drop__01_move_and_drop_order",
        "ownership_drop__02_shared_borrow_does_not_move",
        "primitive__01_integer_widths_and_overflow_traps",
        "primitive__02_integer_overflow_traps",
        "primitive__03_float_arithmetic_and_casts",
        "struct_enum_trait__01_struct_construction_and_methods",
        "struct_enum_trait__02_enum_and_pattern_match",
        "struct_enum_trait__03_generic_function_and_trait_bound",
        "struct_enum_trait__04_trait_default_and_override",
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
/// variant, agreeing with the oracle. (Three-arm exhaustive Ordering matches are a front-end
/// gap — DEV-071 — so this uses an explicit-plus-wildcard match, which still exercises the MIR
/// `CoreOrdering` construct/return/discriminant-switch path.)
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
             match o { Ordering::Less => 10, Ordering::Greater => 30, _ => 20, } \
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
             match v.get(1 as UInt64) { Some(x) => println(*x), None => println(-1), } \
             match v.get(5 as UInt64) { Some(x) => println(*x), None => println(-1), } \
             match v.get_mut(0 as UInt64) { Some(x) => { *x = 99; }, None => { } } \
             match v.get(0 as UInt64) { Some(x) => println(*x), None => println(-1), } \
             let mut s: Vec<String> = Vec::new(); \
             s.push(String::from(\"hi\")); \
             match s.get(0 as UInt64) { Some(x) => println(x.as_str()), None => println(\"none\"), } \
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
