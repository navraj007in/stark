//! **DEV-160: one call, several disjoint accesses to one aggregate.**
//!
//! The third defect in the family DEV-158 and DEV-162 belong to — *whole-value machinery over
//! place-granular semantics* — and the one that reaches furthest, because it is not about a single
//! access at all. It is about what happens when several accesses to one aggregate must be **live at
//! the same instant**.
//!
//! STARK's borrow checker is place-granular (DEV-154), so it correctly accepts
//!
//! ```text
//! consume(&p.name, p.body, p.n)
//! ```
//!
//! — a shared borrow of one field beside a move out of a second and a `Copy` read of a third, all
//! disjoint. The backend then emitted each through a wrapper taking the WHOLE slot:
//!
//! ```text
//! _9 = &(*stark_proj::stark_ref_…f0(&_1));                 // & _1
//! _8 = consume(_9, stark_proj::stark_move_…f1(&mut _1), …); // &mut _1, while _9 lives
//! ```
//!
//! `E0502`. **A correct program refused by its own backend** — the failure mode that matters most,
//! because there is nothing wrong with the user's code and no diagnostic that says so.
//!
//! # Why not the obvious fixes
//!
//! **Reordering the arguments** would work and is prohibited: CD-007 freezes strict left-to-right
//! evaluation of call arguments as *language semantics*. A generated-Rust limitation does not get to
//! change what a STARK program means.
//!
//! **Hoisting each argument into a local first** does not work. The hoisted shared borrow is still
//! live where the call consumes it, so every later `&mut` conflicts exactly as before.
//!
//! # The thunk
//!
//! A generated function takes each participating slot ONCE as `&'a mut ValueSlot<T>`, derives a raw
//! pointer from it, performs every access through that pointer in MIR order, and calls the callee
//! itself. One `&mut` exists, so there is nothing to conflict with; `'a` is a real lifetime carried
//! by a real reference, so a borrow the thunk hands to the callee has honest provenance.
//!
//! It lives in `mod stark_proj` beside the wrappers it calls — §7.8's single `unsafe` boundary.
//! **Generated MIR function bodies still contain none:** the call site is one ordinary safe call.
//!
//! # What it must not disturb
//!
//! A call that does not conflict must reach none of this. Both mechanisms that could touch it — the
//! plan lookup in `emit_call`, the statement suppression in `emit_one_block` — are gated on a plan
//! existing for the block, so `ordinary_calls_plan_nothing` asserts the detector stays silent on
//! shapes one condition away from conflicting. The thunk is a narrow path for a shape that
//! previously did not build at all, not a new default calling convention.

mod support;

/// The reported shape, exactly: a shared field borrow, a sibling move, and a `Copy` sibling read,
/// in one argument list, in that order.
///
/// The `Copy` read is the third argument on purpose. It comes AFTER the move in MIR order, so if
/// the thunk ever reordered its projections the `stark_copyraw` of `f2` would run against storage
/// whose `f1` had already gone — which the raw primitives permit and a whole-value read would not.
/// Ordering is what this asserts, not merely that it compiles.
#[test]
fn a_borrow_a_move_and_a_copy_of_one_aggregate_in_one_call() {
    let source = r#"
struct Parts { name: String, body: String, n: UInt32 }

fn consume(a: &String, b: String, c: UInt32) -> UInt32 {
    (a.len() + b.len()) as UInt32 + c
}

fn main() {
    let p = Parts { name: String::from("abc"), body: String::from("de"), n: 1u32 };
    let n = consume(&p.name, p.body, p.n);
    if n != 6u32 {
        panic("the thunk did not evaluate the arguments it was given");
    }
    println("DEV160_MIXED_OK");
}
"#;
    let done = support::differential::agree_completing("dev160_mixed", source);
    assert_eq!(
        String::from_utf8_lossy(&done.stdout_bytes),
        "DEV160_MIXED_OK\n",
        "every engine must reach the end with the same output"
    );
}

/// Two moves out of one aggregate. Simpler than the mixed case and a different conflict — `&mut`
/// against `&mut` (E0499) rather than `&mut` against `&` — so it is not covered by the first test.
#[test]
fn two_sibling_moves_in_one_call() {
    let source = r#"
struct Parts { name: String, body: String }

fn consume(a: String, b: String) -> UInt32 {
    (a.len() + b.len()) as UInt32
}

fn main() {
    let p = Parts { name: String::from("abc"), body: String::from("de") };
    let n = consume(p.name, p.body);
    if n != 5u32 {
        panic("the two moved fields did not arrive");
    }
    println("DEV160_MOVES_OK");
}
"#;
    let done = support::differential::agree_completing("dev160_moves", source);
    assert_eq!(
        String::from_utf8_lossy(&done.stdout_bytes),
        "DEV160_MOVES_OK\n"
    );
}

/// **The sibling the aggregate keeps must still be droppable afterwards.**
///
/// A thunk moves one field out through a raw projection, leaving the storage `Partial`. The
/// remaining field is destroyed later by MIR's own drop elaboration, through the flag-guarded
/// per-unit `Drop`s — a path the thunk never touches and could silently invalidate. Leaving a
/// survivor that is later dropped is what proves it does not.
#[test]
fn a_field_the_thunk_left_behind_is_still_dropped() {
    let source = r#"
struct Parts { taken: String, kept: String }

fn consume(a: String, n: UInt32) -> UInt32 {
    a.len() as UInt32 + n
}

fn main() {
    let p = Parts { taken: String::from("abc"), kept: String::from("xyzzy") };
    // `&p.kept` borrows the survivor; `p.taken` moves its sibling. `p.kept` is then destroyed by
    // drop elaboration after the call.
    let n = consume(p.taken, p.kept.len() as UInt32);
    if n != 8u32 {
        panic("the surviving field was not readable");
    }
    println("DEV160_SURVIVOR_OK");
}
"#;
    let done = support::differential::agree_completing("dev160_survivor", source);
    assert_eq!(
        String::from_utf8_lossy(&done.stdout_bytes),
        "DEV160_SURVIVOR_OK\n"
    );
}

/// **A named borrow, used twice in one argument list.**
///
/// `let r = &p.name;` does not lower to one statement. It lowers to a PAIR — `_8 = &_1.0` then
/// `_7 = copy _8` — and only `_7` reaches the argument list. A thunk that looked for `RefOf`
/// destinations among the arguments would find nothing, absorb nothing, and leave the conflict
/// exactly where it was. Following the chain, and suppressing every statement along it, is what
/// this case exists to hold in place.
///
/// It is used TWICE as well, which is why absorption counts reads instead of demanding a single
/// one: two shared borrows of the same field are two raw projections and nothing more.
#[test]
fn a_named_borrow_used_twice_beside_a_sibling_move() {
    let source = r#"
struct Parts { name: String, body: String }

fn consume2(a: &String, b: &String, c: String) -> UInt32 {
    (a.len() + b.len() + c.len()) as UInt32
}

fn main() {
    let p = Parts { name: String::from("abc"), body: String::from("de") };
    let r = &p.name;
    let n = consume2(r, r, p.body);
    if n != 8u32 {
        panic("the chained borrow did not survive absorption");
    }
    println("DEV160_DOUBLE_OK");
}
"#;
    let done = support::differential::agree_completing("dev160_double", source);
    assert_eq!(
        String::from_utf8_lossy(&done.stdout_bytes),
        "DEV160_DOUBLE_OK\n"
    );
}

/// **The bound, stated as a refusal rather than discovered as `E0502`.**
///
/// A borrow that is read again AFTER the call cannot be absorbed: suppressing its definition would
/// break the later read, and leaving it in place puts a live borrow beside the thunk's `&mut`.
/// STARK accepts the program — the accesses are disjoint — so this is a real gap in native
/// emission, and the only question is how it is reported.
///
/// It is reported by the backend, naming the local and the field. The alternative is rustc's
/// `E0502` pointing into `mod stark_proj`, which describes a borrow the user never wrote in a
/// function they have never seen. **A known limit that says what it is beats a correct compiler
/// error about the wrong code.**
#[test]
fn a_borrow_outliving_the_call_is_refused_by_name() {
    let source = r#"
struct Parts { name: String, body: String }

fn consume(a: &String, b: String) -> UInt32 {
    (a.len() + b.len()) as UInt32
}

fn main() {
    let p = Parts { name: String::from("abc"), body: String::from("de") };
    let r = &p.name;
    let n = consume(r, p.body);
    if n != 5u32 {
        panic("bad");
    }
    if r.len() != 3u64 {
        panic("the borrow did not survive the call");
    }
    println("unreachable in the native engine");
}
"#;
    let front = support::differential::front_end("dev160_escaping.stark", source);
    let program = starkc::mir::lower::lower_program(&front.hir, &front.tables, front.file.clone())
        .unwrap_or_else(|e| {
            panic!(
                "the program must LOWER; it is the BACKEND that refuses it: {} @ {:?}",
                e.what, e.span
            )
        });

    let versions = starkc::backend::version::build_versions(
        "0.0.0-test".to_string(),
        "x86_64-unknown-linux-gnu".to_string(),
        starkc::backend::generated_rust::Profile::Debug,
    );
    let refusal = match starkc::backend::generated_rust::emit_program::emit(
        &program,
        &versions,
        &starkc::layout::TargetLayout::stark64_v1(),
    ) {
        Err(refusal) => refusal,
        Ok(_) => panic!(
            "the native backend must REFUSE this program. Emitting it produces `E0502` inside \
             `mod stark_proj`, which is a correct compiler error about the wrong code"
        ),
    };

    let message = format!("{refusal:?}");
    assert!(
        message.contains("outlives a call that moves out of a sibling field"),
        "the refusal must say what the program does, not what rustc thinks of the output: {message}"
    );
    assert!(
        message.contains("DEV-160d"),
        "the refusal must carry the SUB-id from the owner's 2026-08-03 taxonomy, not the family \
         name -- three shapes are deferred and a reader hitting one needs to know which: {message}"
    );
}

/// **The invariant that bounds the change: ordinary calls do not go near any of this.**
///
/// Two mechanisms could reach code that has no conflict — the plan lookup in `emit_call`, and the
/// statement suppression in `emit_one_block`. Both are gated on a plan EXISTING for the block, so
/// the whole question reduces to whether the detector ever fires on ordinary code. It is asserted
/// directly, over shapes deliberately adjacent to the conflicting one:
///
/// - two `Copy` reads of one aggregate (two shared borrows, which rustc accepts);
/// - a borrow of one field with no sibling move anywhere near it;
/// - a move out of one field and nothing else;
/// - a whole-value move of one local into a call.
///
/// Each is one condition away from needing a thunk. If the detector widened, this is where it would
/// show up first — as a thunk in a program that never needed one.
#[test]
fn ordinary_calls_plan_nothing() {
    let cases: &[(&str, &str)] = &[
        (
            "two Copy reads of one aggregate",
            r#"
struct P { a: UInt32, b: UInt32 }
fn add(x: UInt32, y: UInt32) -> UInt32 { x + y }
fn main() {
    let p = P { a: 1u32, b: 2u32 };
    if add(p.a, p.b) != 3u32 { panic("no"); }
    println("OK");
}
"#,
        ),
        (
            "a field borrow with no sibling move",
            r#"
struct P { a: String, b: String }
fn read(x: &String) -> UInt64 { x.len() }
fn main() {
    let p = P { a: String::from("abc"), b: String::from("de") };
    if read(&p.a) != 3u64 { panic("no"); }
    println("OK");
}
"#,
        ),
        (
            "a single field move",
            r#"
struct P { a: String, b: String }
fn eat(x: String) -> UInt64 { x.len() }
fn main() {
    let p = P { a: String::from("abc"), b: String::from("de") };
    if eat(p.a) != 3u64 { panic("no"); }
    println("OK");
}
"#,
        ),
        (
            "a whole-value move",
            r#"
fn eat(x: String) -> UInt64 { x.len() }
fn main() {
    let s = String::from("abc");
    if eat(s) != 3u64 { panic("no"); }
    println("OK");
}
"#,
        ),
    ];

    for (what, source) in cases {
        let front = support::differential::front_end("dev160_ordinary.stark", source);
        let program =
            starkc::mir::lower::lower_program(&front.hir, &front.tables, front.file.clone())
                .unwrap_or_else(|e| panic!("{what}: lowering: {} @ {:?}", e.what, e.span));
        let layout = starkc::layout::TargetLayout::stark64_v1();

        let plans =
            starkc::backend::generated_rust::emit_call_thunk::collect_plans(&program, &layout)
                .unwrap_or_else(|e| {
                    panic!("{what}: planning must not refuse an ordinary program: {e:?}")
                });
        assert!(
            plans.is_empty(),
            "{what}: no thunk may be planned -- ordinary calls must keep the emission they had \
             before DEV-160 existed, and a plan is the only thing that can change it"
        );

        let versions = starkc::backend::version::build_versions(
            "0.0.0-test".to_string(),
            "x86_64-unknown-linux-gnu".to_string(),
            starkc::backend::generated_rust::Profile::Debug,
        );
        let source =
            match starkc::backend::generated_rust::emit_program::emit(&program, &versions, &layout)
            {
                Ok(generated) => generated.main_rs,
                Err(e) => panic!("{what}: emission: {e:?}"),
            };
        assert!(
            !source.contains("stark_thunk_"),
            "{what}: the generated program must contain no thunk"
        );
    }
}

/// **The idiom the defect was actually reported as, and what it does now.**
///
/// ```text
/// send_once(builder.url.as_str(), builder.headers, builder.body)
/// ```
///
/// `as_str()` is itself a call. It runs in an EARLIER block, taking `&builder.url`, and returns a
/// `&str` that borrows `builder`. By the time the outer call is reached, that borrow is an ordinary
/// non-slot local carrying no sign of where it came from — and the thunk is about to take `&mut`
/// the very slot it borrows.
///
/// Absorbing it means absorbing the intermediate call, across a block boundary, which is a larger
/// mechanism than this increment carries. So it is **refused, by name**, and the refusal says which
/// local borrows which and what to do instead. Before DEV-160 this shape produced `E0502` inside
/// generated code; a stated limit is not a fix, but it is the difference between a compiler that
/// knows what it cannot do and one that hands the user a stack trace from a file they cannot see.
///
/// **It flipped.** Cross-block absorption landed under WP-ARCH-CLOSE AC1 step 2, and this test now
/// asserts the run it was written to wait for.
///
/// Three arguments, all touching one aggregate: a `&str` produced by `as_str` in an EARLIER block,
/// and two sibling moves. The thunk performs `as_str` itself, so the reference it passes is
/// anchored by the thunk's own `&'a mut` rather than by a borrow that predated it — which is what
/// Stacked Borrows requires and what ruled out laundering the reference through a raw pointer.
#[test]
fn a_borrow_reaching_the_call_through_an_earlier_call_now_builds_and_runs() {
    let source = r#"
struct Req { url: String, headers: String, body: String }

fn send_once(u: &str, h: String, b: String) -> UInt64 {
    (u.len() + h.len() + b.len()) as UInt64
}

fn main() {
    let r = Req { url: String::from("abc"), headers: String::from("de"), body: String::from("f") };
    let n = send_once(r.url.as_str(), r.headers, r.body);
    if n != 6u64 {
        panic("bad");
    }
    println("OK");
}
"#;
    // All four engine configurations, compared on the full normative observation. A native-only
    // assertion would not notice the absorbed call being performed at a different point.
    support::differential::agree_completing_with_stdout("dev160_through_call", source, "OK\n");
}

/// **The Miri fixture is only evidence while it is still a copy of what the generator emits.**
///
/// A thunk is generated code. Miri cannot run what has not been generated, so `stark-runtime`
/// carries a hand-written one (`slot::tests::thunk`) for Miri to check under Stacked Borrows. That
/// arrangement has an obvious failure mode: the generator changes, the fixture does not, and the
/// Miri job keeps passing while proving something about code the compiler no longer produces.
///
/// So the fixture publishes the primitive sequence it stands for, and this test derives the same
/// sequence from a freshly generated thunk — resolving each `stark_proj` wrapper the thunk calls to
/// the `ValueSlot` primitive inside it. The two must agree. A generator change that alters the
/// shape fails HERE, naming the fixture, rather than leaving a green Miri run to reassure nobody.
/// The primitive sequence a fixture constant declares, read from the fixture's own source so there
/// is one authority for it.
fn declared_shape(slot_rs: &str, constant: &str) -> Vec<String> {
    let start = slot_rs
        .find(&format!("pub const {constant}"))
        .unwrap_or_else(|| panic!("the fixture must publish {constant}"));
    // From the `=`, not from the declaration: the TYPE is spelled `&[&str]`, so the first bracket
    // in the item belongs to it rather than to the literal.
    let body = &slot_rs[start..];
    let body = &body[body.find('=').expect("an initialiser") + 1..];
    let open = body.find('[').expect("a slice literal");
    let close = body.find(']').expect("a slice literal");
    body[open + 1..close]
        .split(',')
        .map(|piece| piece.trim().trim_matches('"').to_string())
        .filter(|piece| !piece.is_empty())
        .collect()
}

/// The primitive sequence a freshly generated thunk actually calls, for one program.
///
/// **Nested calls are scanned, not just the outermost one.** DEV-160b's absorbed producer emits
/// `let a0 = <producer>(stark_refraw_…::<'a>(p0));`, so a reader that took the first token of the
/// line would resolve `<producer>` — which is not a generated wrapper — and silently contribute
/// NOTHING, hiding the `field_ref_raw` underneath it. That would have left the new fixture
/// unguarded in exactly the way this whole mechanism exists to prevent.
fn emitted_shape(source: &str, name: &str) -> Vec<String> {
    let front = support::differential::front_end(name, source);
    let program = starkc::mir::lower::lower_program(&front.hir, &front.tables, front.file.clone())
        .unwrap_or_else(|e| panic!("lowering: {} @ {:?}", e.what, e.span));
    let versions = starkc::backend::version::build_versions(
        "0.0.0-test".to_string(),
        "x86_64-unknown-linux-gnu".to_string(),
        starkc::backend::generated_rust::Profile::Debug,
    );
    let generated = match starkc::backend::generated_rust::emit_program::emit(
        &program,
        &versions,
        &starkc::layout::TargetLayout::stark64_v1(),
    ) {
        Ok(generated) => generated.main_rs,
        Err(e) => panic!("emission: {e:?}"),
    };

    let thunk_start = generated
        .find("pub fn stark_thunk_")
        .expect("the program must contain a thunk; if it no longer does, the detector regressed");
    let thunk = &generated[thunk_start..];
    let thunk_end = thunk.find("\n    }\n").expect("the thunk must be closed");
    let thunk = &thunk[..thunk_end];

    const PRIMITIVES: [&str; 4] = [
        "field_ref_raw",
        "move_field_raw",
        "copy_field_raw",
        "take_raw",
    ];

    let mut emitted: Vec<String> = Vec::new();
    for line in thunk.lines() {
        let line = line.trim();
        if line.strip_prefix("let a").is_none() {
            continue;
        }
        // Every generated wrapper named on this line, in order of appearance, plus a direct
        // `ValueSlot::take_raw`. Resolving each to the runtime primitive inside it is what the Miri
        // fixture actually stands for.
        let mut hits: Vec<(usize, String)> = Vec::new();
        for (offset, _) in line.match_indices("stark_") {
            let rest = &line[offset..];
            let ident: String = rest
                .chars()
                .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
                .collect();
            let Some(definition) = generated.find(&format!("pub unsafe fn {ident}")) else {
                continue;
            };
            let body = &generated[definition..];
            let Some(end) = body.find("\n    }") else {
                continue;
            };
            for primitive in PRIMITIVES {
                if body[..end].contains(primitive) {
                    hits.push((offset, primitive.to_string()));
                    break;
                }
            }
        }
        for (offset, _) in line.match_indices("ValueSlot::take_raw") {
            hits.push((offset, "take_raw".to_string()));
        }
        hits.sort_by_key(|(offset, _)| *offset);
        emitted.extend(hits.into_iter().map(|(_, primitive)| primitive));
    }
    emitted
}

#[test]
fn the_miri_fixture_matches_what_the_generator_emits() {
    let slot_rs = std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("stark-runtime/src/slot.rs"),
    )
    .expect("stark-runtime/src/slot.rs must be readable");

    let declared = declared_shape(&slot_rs, "GENERATED_THUNK_SHAPE");
    assert!(
        !declared.is_empty(),
        "GENERATED_THUNK_SHAPE must list the primitives the fixture exercises"
    );

    // The same program the fixture mirrors: a field borrow, a sibling move, a Copy sibling read.
    let emitted = emitted_shape(
        r#"
struct Parts { name: String, body: String, n: UInt32 }
fn consume(a: &String, b: String, c: UInt32) -> UInt32 {
    (a.len() + b.len()) as UInt32 + c
}
fn main() {
    let p = Parts { name: String::from("abc"), body: String::from("de"), n: 1u32 };
    if consume(&p.name, p.body, p.n) != 6u32 { panic("no"); }
    println("OK");
}
"#,
        "dev160_shape.stark",
    );

    assert_eq!(
        emitted, declared,
        "the generated thunk calls a different sequence of raw primitives than the Miri fixture in \
         stark-runtime/src/slot.rs stands for. Update `slot::tests::thunk` AND \
         `GENERATED_THUNK_SHAPE` together -- otherwise the Miri job proves something about code \
         this compiler no longer emits"
    );
}

/// **The same guard, for DEV-160b's absorbed-producer fixture.**
///
/// Added with that fixture rather than after it. A Miri fixture nothing compares against the
/// generator is the precise failure the guard above exists to prevent, and adding an unguarded
/// second one would have reintroduced it while looking like extra coverage.
#[test]
fn the_absorbed_producer_fixture_matches_what_the_generator_emits() {
    let slot_rs = std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("stark-runtime/src/slot.rs"),
    )
    .expect("stark-runtime/src/slot.rs must be readable");

    let declared = declared_shape(&slot_rs, "ABSORBED_PRODUCER_SHAPE");
    assert!(
        !declared.is_empty(),
        "ABSORBED_PRODUCER_SHAPE must list the primitives the fixture exercises"
    );

    let emitted = emitted_shape(
        r#"
struct Req { url: String, body: String }
fn send(u: &str, b: String) -> UInt64 { u.len() + b.len() }
fn main() {
    let r = Req { url: String::from("abc"), body: String::from("de") };
    if send(r.url.as_str(), r.body) != 5u64 { panic("no"); }
    println("OK");
}
"#,
        "dev160b_shape.stark",
    );

    assert_eq!(
        emitted, declared,
        "the generated thunk for a CROSS-BLOCK absorption calls a different sequence of raw \
         primitives than `slot::tests::absorbing_thunk` stands for. Update the fixture AND \
         `ABSORBED_PRODUCER_SHAPE` together"
    );
}
