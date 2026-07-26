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
    compare_observations, front_end, run_hir, run_mir, run_native, rustc_available,
    CanonicalReturnedValue, CompletionObservation, DropEvent, Observation, TrapMessageClass,
    TrapObservation,
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
    if !rustc_available() {
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

    assert_eq!(results.len(), 16, "§14.3 requires sixteen mutations");
    write_evidence(&results);
}

/// §14.5 also requires source-level controls for **routing-sensitive** mutations: proof that the
/// witness would observably change if the compiler took the wrong route. Without these, MU09 and
/// MU12 rest on my assertion that the sentinel discriminates — these run the wrong route as a real
/// program and show its observation differs.
#[test]
fn routing_controls_show_the_wrong_route_is_observable() {
    if !rustc_available() {
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
    if !rustc_available() {
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
