//! WP-C6.5 §14 — mutation controls: proof that this evidence is capable of failing.
//!
//! Everything else in C6.5 shows the corpus and the comparator *agree*. That is worth nothing unless
//! they would also *disagree* when something is wrong, and a suite of passing tests cannot tell the
//! two apart. So each of §14.3's sixteen mutations takes a **real passing corpus case**, runs it
//! through the **real engines**, clones one normalised observation, applies **one precise
//! test-only mutation**, and requires the **production comparator** to reject it *naming the intended
//! field*.
//!
//! Three rules keep this from being theatre:
//!
//! 1. **The witness must genuinely pass first** (§14.6). Every mutation asserts three-engine
//!    agreement on the unmutated observation before touching it — a mutation "detected" on a case
//!    that was already failing proves nothing.
//! 2. **The mutation must change the intended dimension** (§14.6). Each one asserts the observation
//!    actually differs after mutating, so a no-op mutation cannot pass as a detection. This is the
//!    same trap the metamorphic generator's identity-transform guard exists for.
//! 3. **No mutation is simulated by asserting `false`** (§14.7). The comparator under test is
//!    `compare_observations`, the same function the replay uses; nothing here reimplements
//!    comparison.
//!
//! §14.1 is explicit that this is evidence about **comparator and witness sensitivity** — it does not
//! authorise mutating compiler source. Nothing here modifies an engine; the mutation is applied to a
//! normalised observation after the engines have produced it.

mod support;

use std::path::PathBuf;

use support::corpus::{corpus_root, load, Case};
use support::differential::{
    compare_observations, front_end, run_hir, run_mir, run_native, CanonicalReturnedValue,
    CompletionObservation, DropEvent, Observation, TrapMessageClass, TrapObservation,
};

/// §21.3's per-mutation record.
struct MutationResult {
    mutation_id: &'static str,
    witness_case_id: String,
    expected_field: &'static str,
    unmodified_agrees: bool,
    mutation_detected: bool,
    reported_field: String,
}

fn run_witness(case: &Case) -> Observation {
    let source = std::fs::read_to_string(corpus_root().join(&case.sources[0]))
        .unwrap_or_else(|e| panic!("{}: {e}", case.sources[0]));
    let name = format!("{}.stark", case.case_id);
    let front = front_end(&name, &source);
    let program =
        match starkc::mir::lower::lower_program(&front.hir, &front.tables, front.file.clone()) {
            Ok(program) => program,
            Err(e) => panic!("{name}: lowering failed: {} @ {:?}", e.what, e.span),
        };
    let hir = run_hir(&name, &front);
    let mir = run_mir(&name, &program);
    let native = if case.required_engines.iter().any(|e| e == "native-debug") {
        run_native(&name, &case.case_id, &program)
    } else {
        run_mir(&name, &program)
    };
    // §14.6 clause 1: the witness must agree before it is mutated.
    compare_observations(&name, &hir, &mir, &native).unwrap_or_else(|reason| {
        panic!(
            "witness {} does not agree unmutated: {reason}",
            case.case_id
        )
    });
    hir
}

/// Finds a witness by predicate rather than by hard-coded ID: generated case IDs carry a digest that
/// moves whenever the generator or seed changes, so pinning one here would make this suite fail for
/// a reason that has nothing to do with mutation sensitivity.
fn witness_where<'a>(cases: &'a [Case], what: &str, pred: impl Fn(&Case) -> bool) -> &'a Case {
    cases
        .iter()
        .find(|case| pred(case))
        .unwrap_or_else(|| panic!("no corpus case matches the witness requirement: {what}"))
}

fn clone_observation(observation: &Observation) -> Observation {
    match observation {
        Observation::Completed(done) => Observation::Completed(CompletionObservation {
            stdout_bytes: done.stdout_bytes.clone(),
            stderr_bytes: done.stderr_bytes.clone(),
            exit_status: done.exit_status,
            returned_observation: done.returned_observation.clone(),
            drop_log: done.drop_log.clone(),
        }),
        Observation::Trapped(trap) => Observation::Trapped(TrapObservation {
            category: trap.category,
            source_file: trap.source_file.clone(),
            line: trap.line,
            column: trap.column,
            message_class: trap.message_class.clone(),
            stdout_before_trap: trap.stdout_before_trap.clone(),
            stderr_before_trap: trap.stderr_before_trap.clone(),
            stderr_observation: trap.stderr_observation.clone(),
            exit_status: trap.exit_status,
            drop_log_before_trap: trap.drop_log_before_trap.clone(),
        }),
    }
}

/// The §14.5 mechanism, applied once. Returns the §21.3 record.
#[track_caller]
fn mutation(
    mutation_id: &'static str,
    case: &Case,
    observed: &Observation,
    expected_field: &'static str,
    apply: impl FnOnce(&mut Observation),
) -> MutationResult {
    let mut mutated = clone_observation(observed);
    apply(&mut mutated);
    // §14.6 clause 2: a mutation that changed nothing is not a mutation.
    assert!(
        support::differential::first_difference(observed, &mutated).is_some(),
        "{mutation_id}: the mutation changed nothing, so its detection would be vacuous"
    );

    // The PRODUCTION comparator, in the position the replay uses it: one engine's observation
    // replaced by the mutated one.
    let verdict = compare_observations("mutation.stark", observed, &mutated, observed);
    let reason = match verdict {
        Ok(()) => panic!(
            "{mutation_id}: the comparator ACCEPTED a mutated {expected_field} — the corpus cannot \
             detect this defect class"
        ),
        Err(reason) => reason,
    };
    assert!(
        reason.contains(expected_field),
        "{mutation_id}: rejected, but not for the intended field. Wanted {expected_field:?}, got:\n{reason}"
    );
    MutationResult {
        mutation_id,
        witness_case_id: case.case_id.clone(),
        expected_field,
        unmodified_agrees: true,
        mutation_detected: true,
        reported_field: expected_field.to_string(),
    }
}

fn completion(observation: &mut Observation) -> &mut CompletionObservation {
    match observation {
        Observation::Completed(done) => done,
        Observation::Trapped(_) => panic!("witness was expected to complete"),
    }
}

fn trap(observation: &mut Observation) -> &mut TrapObservation {
    match observation {
        Observation::Trapped(trap) => trap,
        Observation::Completed(_) => panic!("witness was expected to trap"),
    }
}

/// All sixteen §14.3 mutations, each against a real witness.
#[test]
fn every_required_mutation_is_detected() {
    if !support::differential::rustc_available() {
        eprintln!("SKIP: no rustc in this environment.");
        return;
    }
    let (cases, _) = load();
    let mut results = Vec::new();

    // --- witnesses, each chosen because its observation EXPOSES the defect class (§14.4) ---

    let arithmetic = witness_where(
        &cases,
        "an arithmetic case printing a nontrivial value",
        |c| c.template_id.as_deref() == Some("T01") && !c.expected_stdout.is_empty(),
    );
    let arithmetic_observed = run_witness(arithmetic);

    let overflow_trap = witness_where(&cases, "a trap case with pre-trap output", |c| {
        c.expected_outcome == "trap"
            && c.expected_trap_category.as_deref() == Some("IntegerOverflow")
    });
    let trap_observed = run_witness(overflow_trap);

    // WP-C7.9 Packet D: the one case that writes to BOTH streams before trapping. MU24 needs it —
    // clearing a field that was already empty is not a mutation, and every other trap case in the
    // corpus produces no program stderr at all.
    let stderr_trap = witness_where(&cases, "a trap case with pre-trap stderr", |c| {
        c.case_id == "sentinel__22_stderr_before_trap"
    });
    let stderr_trap_observed = run_witness(stderr_trap);

    let drops = witness_where(
        &cases,
        "a Drop witness with at least three distinct events",
        |c| c.case_id == "sentinel__12_drop_identities",
    );
    let drops_observed = run_witness(drops);
    assert!(
        match &drops_observed {
            Observation::Completed(done) => done.drop_log.len() >= 3,
            _ => false,
        },
        "§14.4 requires the Drop witness to carry at least three distinct events"
    );

    let generics = witness_where(
        &cases,
        "two generic instances with different sentinels",
        |c| c.case_id == "sentinel__07_two_generic_instances",
    );
    let generics_observed = run_witness(generics);

    let traits = witness_where(&cases, "two trait impls with different sentinels", |c| {
        c.case_id == "sentinel__08_two_trait_impls"
    });
    let traits_observed = run_witness(traits);

    let fn_values = witness_where(
        &cases,
        "two function-value targets with different sentinels",
        |c| c.case_id == "sentinel__09_two_function_value_targets",
    );
    let fn_values_observed = run_witness(fn_values);

    let collection = witness_where(&cases, "insertion order distinct from sorted order", |c| {
        c.case_id == "sentinel__11_insertion_order_not_sorted"
    });
    let collection_observed = run_witness(collection);

    let slice = witness_where(&cases, "a slice mutated through a view", |c| {
        c.case_id == "sentinel__10_slice_mutation_through_view"
    });
    let slice_observed = run_witness(slice);

    let float32 = witness_where(&cases, "a Float32 rendering unlike Float64", |c| {
        c.case_id == "sentinel__13_float32_rendering"
    });
    let float32_observed = run_witness(float32);

    // --- the sixteen mutations ---

    results.push(mutation(
        "MU01",
        arithmetic,
        &arithmetic_observed,
        "stdout_bytes",
        |o| {
            // A wrong arithmetic result reaches the observation as different printed bytes.
            let done = completion(o);
            done.stdout_bytes = b"999999".to_vec();
        },
    ));
    results.push(mutation(
        "MU02",
        overflow_trap,
        &trap_observed,
        "trap line",
        |o| {
            trap(o).line += 1;
        },
    ));
    results.push(mutation(
        "MU03",
        overflow_trap,
        &trap_observed,
        "trap category",
        |o| {
            trap(o).category = starkc::mir::TrapCategory::CastFailure;
        },
    ));
    results.push(mutation("MU04", drops, &drops_observed, "drop_log", |o| {
        completion(o).drop_log.pop();
    }));
    results.push(mutation("MU05", drops, &drops_observed, "drop_log", |o| {
        let log = &mut completion(o).drop_log;
        let last = log.last().cloned().expect("a Drop event");
        log.push(DropEvent {
            sequence: last.sequence + 1,
            identity: last.identity,
        });
    }));
    results.push(mutation("MU06", drops, &drops_observed, "drop_log", |o| {
        let log = &mut completion(o).drop_log;
        log.reverse();
        // Sequence numbers are positional, so a genuine reversal renumbers them: without this the
        // mutation would also change `sequence`, which is a second dimension.
        for (index, event) in log.iter_mut().enumerate() {
            event.sequence = index as u32 + 1;
        }
    }));
    results.push(mutation("MU07", drops, &drops_observed, "drop_log", |o| {
        // A COPIED move: the value is destroyed once by each owner, so a second event appears with
        // the same identity. Distinct from MU05 in what it models — MU05 is one owner dropped twice.
        let log = &mut completion(o).drop_log;
        let first = log.first().cloned().expect("a Drop event");
        log.insert(
            0,
            DropEvent {
                sequence: 0,
                identity: first.identity,
            },
        );
    }));
    results.push(mutation(
        "MU08",
        generics,
        &generics_observed,
        "stdout_bytes",
        |o| {
            // The wrong generic instance: both calls reach the first instance's sentinel.
            completion(o).stdout_bytes = b"1111".to_vec();
        },
    ));
    results.push(mutation(
        "MU09",
        traits,
        &traits_observed,
        "stdout_bytes",
        |o| {
            completion(o).stdout_bytes = b"3333".to_vec();
        },
    ));
    results.push(mutation(
        "MU10",
        fn_values,
        &fn_values_observed,
        "stdout_bytes",
        |o| {
            completion(o).stdout_bytes = b"5555".to_vec();
        },
    ));
    results.push(mutation(
        "MU11",
        collection,
        &collection_observed,
        "stdout_bytes",
        |o| {
            // Sorted order rather than insertion order — the exact defect STD-HASH-001's CE4 amendment
            // forbids, and the reason the witness inserts 30, 10, 20.
            completion(o).stdout_bytes = b"102030".to_vec();
        },
    ));
    results.push(mutation(
        "MU12",
        slice,
        &slice_observed,
        "stdout_bytes",
        |o| {
            // A view copied instead of borrowed: the owner keeps its original value.
            completion(o).stdout_bytes = b"1".to_vec();
        },
    ));
    results.push(mutation(
        "MU13",
        float32,
        &float32_observed,
        "stdout_bytes",
        |o| {
            // `Float32` widened to `Float64` before rendering — DEV-109's defect.
            completion(o).stdout_bytes = b"0.10000000149011612|0.10000000149011612".to_vec();
        },
    ));
    results.push(mutation(
        "MU14",
        overflow_trap,
        &trap_observed,
        "trap source_file",
        |o| {
            // §14.4: "two plausible paths". The wrong one is the generated crate's own source, which is
            // exactly what a backend leaking its provenance would report.
            trap(o).source_file = "src/main.rs".to_string();
        },
    ));
    results.push(mutation(
        "MU15",
        arithmetic,
        &arithmetic_observed,
        "stdout_bytes",
        |o| {
            completion(o).stdout_bytes.clear();
        },
    ));
    results.push(mutation(
        "MU16",
        arithmetic,
        &arithmetic_observed,
        "exit_status",
        |o| {
            completion(o).exit_status = 1;
        },
    ));

    // --- R-03: the seven comparator fields that had no control at all ---
    //
    // §14.3's sixteen were chosen as DEFECT CLASSES, and between them they exercised eight of the
    // comparator's fifteen fields. The other seven were never mutated, so the comparator's ability
    // to notice them rested on nothing. These seven close that, and `every_comparator_field_has_a_
    // mutation_control` keeps it closed.
    //
    // Four of them can only be INSERTION mutations, and that is a property of the language rather
    // than a shortcut — stated here so the claim is not read as stronger than it is:
    //
    //   * `drop_log_before_trap` is empty in every conformant observation, because DROP-ABORT-001
    //     makes a trap abort WITHOUT running destructors. There is no case whose pre-trap Drop log
    //     is non-empty, and there must never be one. So the mutation adds an event: it models an
    //     engine that ran a destructor while unwinding, which is precisely the violation.
    //   * `stderr_bytes` on a COMPLETION is likewise always empty in the corpus — PROC-EXIT-001's
    //     `Err` path is the only completion that writes stderr, and it lives in
    //     `c65_entry_exit_contract`, not in a corpus case (the manifest has no `expected_stderr`
    //     field). The mutation adds bytes: it models an engine emitting spurious diagnostics.
    //   * `trap exit_status` is 101 for every trap by construction, so the mutation changes it to a
    //     value a non-aborting engine would produce.
    //   * `completion versus trap` has no single-observation witness by definition; it is the shape
    //     mismatch itself.
    //
    // Each still proves the thing that matters — that the comparator READS the field and names it —
    // which is what "the corpus can detect this defect class" means.

    results.push(mutation(
        "MU17",
        arithmetic,
        &arithmetic_observed,
        "stderr_bytes",
        |o| {
            completion(o).stderr_bytes = b"warning: internal backend note\n".to_vec();
        },
    ));
    results.push(mutation(
        "MU18",
        overflow_trap,
        &trap_observed,
        "trap column",
        |o| {
            // The line is right and the column is wrong: a location that looks correct in a summary
            // and points at the wrong expression on the line.
            trap(o).column += 3;
        },
    ));
    results.push(mutation(
        "MU19",
        overflow_trap,
        &trap_observed,
        "stdout_before_trap",
        |o| {
            // Output produced BEFORE the trap must survive it. Losing it is the classic buffered-
            // stdout defect: the program is correct, the trap is correct, and the evidence of what
            // ran first is gone.
            trap(o).stdout_before_trap.clear();
        },
    ));
    results.push(mutation(
        "MU24",
        stderr_trap,
        &stderr_trap_observed,
        "stderr_before_trap",
        |o| {
            // WP-C7.9 Packet D. The program's OWN stderr before a trap must survive it, exactly as
            // its stdout must — and it must stay distinguishable from the runtime's trap
            // diagnostic, which is a separate field. Losing it is the defect that made this channel
            // unobservable for the whole of C6: every engine reported nothing, and nothing
            // reporting nothing compares equal.
            trap(o).stderr_before_trap.clear();
        },
    ));
    results.push(mutation(
        "MU20",
        overflow_trap,
        &trap_observed,
        "stderr_observation",
        |o| {
            // Only the RENDERED text changes; `category` stays correct, so this cannot be caught by
            // the earlier category check and must be caught by this field. Narrowed per R-10: the
            // interpreters' stderr is CONSTRUCTED from the same runtime table, so between HIR and
            // MIR this field is tautological. What it genuinely witnesses is the NATIVE engine's
            // real stderr disagreeing with that construction.
            trap(o).stderr_observation.category_text = "arithmetic problem".to_string();
        },
    ));
    results.push(mutation(
        "MU21",
        overflow_trap,
        &trap_observed,
        "trap exit_status",
        |o| {
            // TRAP-CATEGORY-001 aborts; an engine that returned a normal failure status instead
            // would be observationally different in exactly this field.
            trap(o).exit_status = 1;
        },
    ));
    results.push(mutation(
        "MU22",
        overflow_trap,
        &trap_observed,
        "drop_log_before_trap",
        |o| {
            // DROP-ABORT-001: destructors do NOT run when a trap aborts. A single fabricated event
            // is what an engine that unwound instead of aborting would produce.
            trap(o).drop_log_before_trap.push(DropEvent {
                sequence: 1,
                identity: "Loud#1".to_string(),
            });
        },
    ));
    results.push(mutation(
        "MU23",
        arithmetic,
        &arithmetic_observed,
        "completion versus trap",
        |o| {
            // The coarsest disagreement there is, and the one a per-field comparison could most
            // easily skip: one engine finished, another aborted.
            *o = clone_observation(&trap_observed);
        },
    ));

    assert_eq!(
        results.len(),
        24,
        "§14.3's sixteen mutations, R-03's seven comparator-field controls, and WP-C7.9's \
         pre-trap stderr control (MU24)"
    );
    write_evidence(&results);
}

/// **R-03.** Every field the comparator can report has a mutation control behind it.
///
/// The review found controls for 8 of the 15 fields, and the reason the gap survived is that
/// nothing enumerated the field set — coverage was counted by reading the list. This reads
/// `COMPARATOR_FIELDS`, which lives beside `first_difference`, so a new field added to the
/// comparator fails here until it has a control. That is the durable half of R-03; the seven new
/// mutations are only the current answer.
#[test]
fn every_comparator_field_has_a_mutation_control() {
    /// Fields whose control is a standalone constructed-pair test rather than a `mutation()` entry.
    /// Both need a value no corpus case produces — a second distinct return frame, and a second
    /// distinct trap message class — so they are built directly instead of mutated from a witness.
    /// The test NAME is checked to exist below, so this table cannot cite a control that is gone.
    const CONTROLLED_BY_STANDALONE_TEST: [(&str, &str); 2] = [
        (
            "returned_observation",
            "the_returned_observation_field_is_load_bearing",
        ),
        (
            "trap message_class",
            "the_trap_message_class_is_load_bearing",
        ),
    ];

    let source = std::fs::read_to_string(
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/c6_mutation.rs"),
    )
    .expect("this file");

    for (field, test_name) in CONTROLLED_BY_STANDALONE_TEST {
        assert!(
            source.contains(&format!("fn {test_name}(")),
            "{field} cites `{test_name}` as its control, but no such test exists here — the same \
             fabricated-citation failure as CD-154"
        );
    }

    let missing: Vec<&str> = support::differential::COMPARATOR_FIELDS
        .iter()
        .copied()
        .filter(|field| {
            !source.contains(&format!("\"{field}\","))
                && !CONTROLLED_BY_STANDALONE_TEST
                    .iter()
                    .any(|(f, _)| f == field)
        })
        .collect();
    assert!(
        missing.is_empty(),
        "{} comparator field(s) have no mutation control: {missing:?}\n\
         Every field `first_difference` can name must have a control that provokes it, or the \
         comparator's ability to notice that field rests on nothing.",
        missing.len()
    );
}

/// §14.5 also requires source-level controls for **routing-sensitive** mutations: proof that the
/// witness would observably change if the compiler took the wrong route. Without these, MU09 and
/// MU12 rest on my assertion that the sentinel discriminates — these run the wrong route as a real
/// program and show its observation differs.
#[test]
fn routing_controls_show_the_wrong_route_is_observable() {
    if !support::differential::rustc_available() {
        eprintln!("SKIP: no rustc in this environment.");
        return;
    }
    let (cases, _) = load();

    // MU09 control: calling the OTHER impl really does produce different bytes.
    let traits = witness_where(&cases, "trait sentinel", |c| {
        c.case_id == "sentinel__08_two_trait_impls"
    });
    let correct = run_witness(traits);
    let wrong_route = run_source(
        "trait_wrong_route",
        "trait Speak {\n    fn speak(&self) -> Int32;\n}\nstruct First { v: Int32 }\n\
         struct Second { v: Int32 }\nimpl Speak for First {\n    fn speak(&self) -> Int32 { 33 }\n}\n\
         impl Speak for Second {\n    fn speak(&self) -> Int32 { 44 }\n}\nfn main() {\n    \
         let a: First = First { v: 0 };\n    print(a.speak());\n    print(a.speak());\n}\n",
    );
    assert!(
        support::differential::first_difference(&correct, &wrong_route).is_some(),
        "the trait witness does not distinguish the wrong impl — MU09 would be undetectable in a \
         real compiler even though the observation mutation is caught"
    );

    // MU12 control: a by-value copy instead of a view leaves the owner unchanged.
    let slice = witness_where(&cases, "slice sentinel", |c| {
        c.case_id == "sentinel__10_slice_mutation_through_view"
    });
    let correct = run_witness(slice);
    let wrong_route = run_source(
        "slice_wrong_route",
        "fn bump(mut view: [Int32; 3]) -> Int32 {\n    view[0] = view[0] + 100;\n    view[0]\n}\n\
         fn main() {\n    let xs: [Int32; 3] = [1, 2, 3];\n    let ignored: Int32 = bump(xs);\n    \
         print(xs[0]);\n}\n",
    );
    assert!(
        support::differential::first_difference(&correct, &wrong_route).is_some(),
        "the slice witness does not distinguish a copy from a view — MU12 would be undetectable"
    );
}

fn run_source(tag: &str, source: &str) -> Observation {
    let name = format!("{tag}.stark");
    let front = front_end(&name, source);
    let program =
        match starkc::mir::lower::lower_program(&front.hir, &front.tables, front.file.clone()) {
            Ok(program) => program,
            Err(e) => panic!("{name}: lowering failed: {} @ {:?}", e.what, e.span),
        };
    run_hir(&name, &front).pipe_check(run_mir(&name, &program), &name)
}

trait PipeCheck {
    fn pipe_check(self, other: Observation, name: &str) -> Observation;
}

impl PipeCheck for Observation {
    /// A control program still has to agree with itself across the two interpreters — a control that
    /// diverged would be a finding, not a control.
    fn pipe_check(self, other: Observation, name: &str) -> Observation {
        assert!(
            support::differential::first_difference(&self, &other).is_none(),
            "{name}: the control program itself diverges between HIR and MIR"
        );
        self
    }
}

fn write_evidence(results: &[MutationResult]) {
    let dir = match std::env::var("C6_EVIDENCE_DIR") {
        Ok(dir) if !dir.is_empty() => PathBuf::from(dir),
        _ => PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target/c6.5-evidence"),
    };
    if std::fs::create_dir_all(&dir).is_err() {
        return;
    }
    let records: Vec<String> = results
        .iter()
        .map(|r| {
            format!(
                "{{\"mutation_id\": \"{}\", \"witness_case_id\": \"{}\", \"expected_field\": \"{}\", \
                 \"unmodified_agrees\": {}, \"mutation_detected\": {}, \"reported_field\": \"{}\", \
                 \"result\": \"PASS\"}}",
                r.mutation_id,
                r.witness_case_id,
                r.expected_field,
                r.unmodified_agrees,
                r.mutation_detected,
                r.reported_field,
            )
        })
        .collect();
    let _ = std::fs::write(
        dir.join("mutations.json"),
        format!("[\n  {}\n]\n", records.join(",\n  ")),
    );
}

/// A returned-observation mutation needs a framed-probe witness, which no corpus case is yet — the
/// §8.7 probe cases live in `three_engine_differential.rs`. Rather than skip the dimension, this
/// exercises the comparator's `returned_observation` field directly against a constructed pair, and
/// the gap is recorded: it is comparator evidence, not corpus evidence.
#[test]
fn the_returned_observation_field_is_load_bearing() {
    let base = Observation::Completed(CompletionObservation {
        stdout_bytes: Vec::new(),
        stderr_bytes: Vec::new(),
        exit_status: 0,
        returned_observation: Some(CanonicalReturnedValue {
            type_tag: "Int32".to_string(),
            rendered: b"42".to_vec(),
        }),
        drop_log: Vec::new(),
    });
    let mut mutated = clone_observation(&base);
    if let Observation::Completed(done) = &mut mutated {
        done.returned_observation = Some(CanonicalReturnedValue {
            type_tag: "Int32".to_string(),
            rendered: b"43".to_vec(),
        });
    }
    let reason = compare_observations("returned.stark", &base, &mutated, &base)
        .expect_err("a changed returned observation must be rejected");
    assert!(
        reason.contains("returned_observation"),
        "rejected for the wrong field: {reason}"
    );
}

/// The message class is normative for `panic(msg)` (§8.6), so a mutation of it must be caught too.
#[test]
fn the_trap_message_class_is_load_bearing() {
    let (cases, _) = load();
    if !support::differential::rustc_available() {
        eprintln!("SKIP: no rustc in this environment.");
        return;
    }
    let panic_case = witness_where(&cases, "a panic trap carrying a user message", |c| {
        c.expected_trap_category.as_deref() == Some("Panic")
    });
    let observed = run_witness(panic_case);
    let mut mutated = clone_observation(&observed);
    if let Observation::Trapped(t) = &mut mutated {
        t.message_class = TrapMessageClass::CategoryOnly;
    }
    let reason = compare_observations("panic.stark", &observed, &mutated, &observed)
        .expect_err("a lost user message must be rejected");
    assert!(
        reason.contains("trap message_class"),
        "rejected for the wrong field: {reason}"
    );
}
