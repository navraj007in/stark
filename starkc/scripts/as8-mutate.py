#!/usr/bin/env python3
"""AS8 compiler-source mutation harness.

WP-ENGINE-INDEPENDENCE EI5 selects the targets; this applies them.

The distinction that makes this different from `tests/c6_mutation.rs`: that suite mutates a
NORMALISED OBSERVATION after the engines have produced it, and its own §14.1 says so explicitly —
"it does not authorise mutating compiler source. Nothing here modifies an engine." This harness
mutates compiler source, rebuilds, and runs the suites against a genuinely different compiler.

CD-392's evidence invariant is enforced structurally, not by convention:

    a trial declares `expect` = KILLED or SURVIVED before it runs, and the harness reports
    CONFIRMED / UNEXPECTED against that declaration.

A batch whose SURVIVED-expected trials all come back KILLED is a harness that detects edits rather
than defects, which is why Batch 0 exists and why `--batch 0` refuses to be skipped silently.

The source file is ALWAYS restored, including on interrupt or build failure.
"""
import argparse, json, os, re, shutil, subprocess, sys, tempfile, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BATCHES = {
    "0": [
        dict(id="MUT-SELFTEST-LIVE", target="harness self-test", tag="ENGINE_LOCAL",
             authority="n/a — harness calibration", expect="KILLED",
             file="src/typecheck/types.rs",
             find="pub(super) fn is_integer(p: Primitive) -> bool {\n    matches!(\n        p,\n        Primitive::Int8",
             repl="pub(super) fn is_integer(p: Primitive) -> bool {\n    matches!(\n        p,\n        Primitive::Bool",
             tests=["--lib"],
             note="Int8 stops reporting as an integer. A real semantic disturbance; must be detected."),
        dict(id="MUT-SELFTEST-NOOP", target="harness self-test", tag="ENGINE_LOCAL",
             authority="n/a — harness calibration", expect="SURVIVED",
             file="src/mir/drop_plan.rs",
             find="pub fn array_order(len: u64) -> impl Iterator<Item = u64> {\n    (0..len).rev()\n}",
             repl="pub fn array_order(len: u64) -> impl Iterator<Item = u64> {\n    let count = len;\n    (0..count).rev()\n}",
             tests=["--lib"],
             note="Introduces a binding and uses it. Semantically identical; must NOT be detected."),
    ],
    # ---------------------------------------------------------------- EI5 Batch 1 ----------
    # Priority 1: high-risk INVISIBLE shared authorities. EI4's falsifiable prediction is that
    # these SURVIVE the differential suites, because all three engines inherit the front end's
    # decision rather than re-deriving it. `expect` records the prediction, so a kill is reported
    # as UNEXPECTED and forces the question of which control caught it.
    "1": [
        dict(id="AS8-MUT-001", target="ESF-COPY-001", tag="SHARED_AUTHORITY",
             authority="typecheck::traits::copy_eligible_types", expect="SURVIVED",
             file="src/typecheck/traits.rs",
             find="            if eligible.contains(&id) || drop_items.contains(&id) {\n                continue;\n            }",
             repl="            if eligible.contains(&id) {\n                continue;\n            }",
             tests=["--test", "three_engine_differential", "--test", "mir_differential"],
             note="Drops the Copy+Drop exclusion from the fixpoint, so a nominal with a destructor "
                  "can become structurally Copy. 03 forbids Copy+Drop."),
        dict(id="AS8-MUT-002", target="ESF-DROP-001", tag="SHARED_AUTHORITY",
             authority="typecheck::traits::nominals_with_destructor", expect="SURVIVED",
             file="src/typecheck/traits.rs",
             find="        if trait_ref.res != Res::CoreTrait(hir::CoreTrait::Drop) {\n            continue;\n        }",
             repl="        if trait_ref.res != Res::CoreTrait(hir::CoreTrait::Clone) {\n            continue;\n        }",
             tests=["--test", "three_engine_differential", "--test", "mir_differential"],
             note="Collects Clone impls instead of Drop impls, so destructor eligibility is wrong "
                  "for every type. A maximal disturbance of a critical authority."),
        dict(id="AS8-MUT-003", target="ESF-COPY-001", tag="SHARED_AUTHORITY",
             authority="typecheck::traits::copy_eligible_types", expect="SURVIVED",
             file="src/typecheck/traits.rs",
             find="            if eligible.contains(&id) || drop_items.contains(&id) {\n                continue;\n            }",
             repl="            if eligible.contains(&id) {\n                continue;\n            }",
             tests=["--test", "copy_canon_matrix"],
             note="The same mutation as AS8-MUT-001, run against copy_canon_matrix alone. EI2 "
                  "classified that suite IMPLEMENTATION_GENERATED for this question; this trial "
                  "tests whether it is a control or a transcription."),
    ],
    # ------------------------------------------------------- EI5 Batch 1b (diagnostic) ------
    # Batch 1 killed AS8-MUT-001 and AS8-MUT-002, both predicted to survive. The captured
    # divergence explains why, and the explanation is narrower than "the differential works":
    #
    #     HIR oracle   ran the destructors
    #     MIR          did not
    #
    # `copy_eligible_types` CONSULTS `nominals_with_destructor` to exclude Copy+Drop. Both
    # mutations broke that exclusion, producing a type that is simultaneously Copy and Drop.
    # MIR's drop planning then asks "is it Copy?" (ESF-COPY-002) while the HIR interpreter's
    # destruction walk asks "does it have a destructor?" (ESF-DROP-001) -- two DIFFERENT shared
    # authorities, each followed by a different engine. The differential saw the CONTRADICTION
    # BETWEEN two shared authorities, not the WRONGNESS of either one.
    #
    # Batch 1b isolates that. Each trial below is wrong in the same authority but SELF-CONSISTENT:
    # it leaves the Copy/Drop exclusion intact, so no two authorities disagree. EI4's prediction
    # is about this case, and these trials are what actually test it.
    "1b": [
        dict(id="AS8-MUT-004", target="ESF-COPY-001", tag="SHARED_AUTHORITY",
             authority="typecheck::traits::copy_eligible_types", expect="SURVIVED",
             file="src/typecheck/traits.rs",
             find="            if field_tys\n                .iter()\n                .all(|t| field_ty_copy_eligible(hir, *t, &eligible))\n            {",
             repl="            if true\n            {",
             tests=["--test", "three_engine_differential", "--test", "mir_differential"],
             note="Drops the all-fields-Copy requirement, so a struct holding a non-Copy field "
                  "becomes structurally Copy. 03 requires all fields Copy. The Copy+Drop exclusion "
                  "is LEFT INTACT, so this is wrong WITHOUT setting two authorities against each "
                  "other -- the isolated form of MUT-001."),
        dict(id="AS8-MUT-005", target="ESF-COPY-001", tag="SHARED_AUTHORITY",
             authority="typecheck::traits::copy_eligible_types", expect="SURVIVED",
             file="src/typecheck/traits.rs",
             find="                hir::ItemKind::Enum { variants, .. } if variants.is_empty() => continue,",
             repl="                hir::ItemKind::Enum { variants, .. } if variants.is_empty() => Vec::new(),",
             tests=["--test", "three_engine_differential", "--test", "mir_differential"],
             note="Reverts CD-251/OWN-COPY-001: a ZERO-VARIANT enum becomes vacuously Copy again. "
                  "This is a REAL HISTORICAL DEFECT, not an invented one -- the code comment records "
                  "that it broke exactly-once close for host resources. Self-consistent: it sets no "
                  "two authorities against each other."),
        dict(id="AS8-MUT-006", target="ESF-COPY-001", tag="SHARED_AUTHORITY",
             authority="typecheck::traits::is_copy_with_impls", expect="SURVIVED",
             file="src/typecheck/traits.rs",
             find="        Ty::Ref { mutable: false, .. } | Ty::Never | Ty::Error => true,",
             repl="        Ty::Ref { .. } | Ty::Never | Ty::Error => true,",
             tests=["--test", "three_engine_differential", "--test", "mir_differential"],
             note="`&mut T` reports Copy. 03 makes shared references Copy and exclusive references "
                  "NOT Copy -- duplicating a &mut breaks the one-&mut-XOR-many-& rule. Self-consistent "
                  "in the same sense; no destructor authority is contradicted."),
    ],
    # ------------------------------------------------- EI5 Batch 1c (the control EI2 missed) --
    # Batch 1b's survivors were credited to "no independent control exists for ESF-COPY-001".
    # That was EI2's finding and EI5's Selected-tests column repeated it. BOTH ARE WRONG.
    #
    #     starkc/tests/c61f_structural_copy.rs   13 tests, HAND_AUTHORED from OWN-COPY-001
    #         c61g_mutable_reference_field_stays_move           <- the control for MUT-006
    #         c251_a_zero_variant_enum_is_not_structurally_copy <- the control for MUT-005
    #         c61g_mixed_copy_and_non_copy_fields_stays_move    <- the control for MUT-004
    #
    # It pins the NEGATIVE surface by behaviour (reuse after move is E0100), not by enumerating
    # the implementation's arms, so it is a genuine control in EI0's sense. It was never in any
    # selected test set. Batch 1c is Batch 1b with that suite added and the predictions FLIPPED:
    # if these kill, the survivors were an artefact of test selection, not a gap in the tree.
    "1c": [
        dict(id="AS8-MUT-009", target="ESF-COPY-001", tag="SHARED_AUTHORITY",
             authority="typecheck::traits::copy_eligible_types", expect="KILLED",
             file="src/typecheck/traits.rs",
             find="            if field_tys\n                .iter()\n                .all(|t| field_ty_copy_eligible(hir, *t, &eligible))\n            {",
             repl="            if true\n            {",
             tests=["--test", "c61f_structural_copy"],
             note="BATCH 1C RE-RUN AGAINST THE CONTROL SUITE. Drops the all-fields-Copy requirement, so a struct holding a non-Copy field "
                  "becomes structurally Copy. 03 requires all fields Copy. The Copy+Drop exclusion "
                  "is LEFT INTACT, so this is wrong WITHOUT setting two authorities against each "
                  "other -- the isolated form of MUT-001."),
        dict(id="AS8-MUT-010", target="ESF-COPY-001", tag="SHARED_AUTHORITY",
             authority="typecheck::traits::copy_eligible_types", expect="KILLED",
             file="src/typecheck/traits.rs",
             find="                hir::ItemKind::Enum { variants, .. } if variants.is_empty() => continue,",
             repl="                hir::ItemKind::Enum { variants, .. } if variants.is_empty() => Vec::new(),",
             tests=["--test", "c61f_structural_copy"],
             note="BATCH 1C RE-RUN AGAINST THE CONTROL SUITE. Reverts CD-251/OWN-COPY-001: a ZERO-VARIANT enum becomes vacuously Copy again. "
                  "This is a REAL HISTORICAL DEFECT, not an invented one -- the code comment records "
                  "that it broke exactly-once close for host resources. Self-consistent: it sets no "
                  "two authorities against each other."),
        dict(id="AS8-MUT-011", target="ESF-COPY-001", tag="SHARED_AUTHORITY",
             authority="typecheck::traits::is_copy_with_impls", expect="KILLED",
             file="src/typecheck/traits.rs",
             find="        Ty::Ref { mutable: false, .. } | Ty::Never | Ty::Error => true,",
             repl="        Ty::Ref { .. } | Ty::Never | Ty::Error => true,",
             tests=["--test", "c61f_structural_copy"],
             note="BATCH 1C RE-RUN AGAINST THE CONTROL SUITE. `&mut T` reports Copy. 03 makes shared references Copy and exclusive references "
                  "NOT Copy -- duplicating a &mut breaks the one-&mut-XOR-many-& rule. Self-consistent "
                  "in the same sense; no destructor authority is contradicted."),
    ],
    # ---------------------------------- EI5 Batch 2 (shared type and representation predicates) --
    # Selected tests follow the rule AS8 added to EI5: EVERY suite that NAMES the authority, not
    # only the suites that execute it. That rule is why `c61f_structural_copy` appears on the
    # ESF-COPY-002 row -- its omission is exactly how MUT-005/006 were recorded as survivors.
    "2": [
        dict(id="AS8-MUT-012", target="ESF-COPY-002", tag="SHARED_AUTHORITY", expect="KILLED",
             authority="mir::mir_ty_is_copy",
             file="src/mir/mod.rs",
             find="        MirTy::Ref { mutable, .. } => !*mutable,",
             repl="        MirTy::Ref { .. } => true,",
             tests=["--test", "mir_differential", "--test", "three_engine_differential",
                    "--test", "c61f_structural_copy"],
             note="`&mut T` reports Copy OVER MirTy. EI5 predicted KILLED because the HIR engine "
                  "classifies over `Ty`, not `MirTy`, so it should disagree. Note MUT-006 is the "
                  "same rule broken on the FRONT-END side and it survived the differential "
                  "entirely -- if this one is killed, the difference is which engine still holds "
                  "the correct answer, not the rule's visibility."),
        dict(id="AS8-MUT-013", target="ESF-TYPE-001", tag="SHARED_AUTHORITY", expect="KILLED",
             authority="typecheck::types::unit_or_tuple",
             file="src/typecheck/types.rs",
             find="    if elems.is_empty() {\n        Ty::Primitive(Primitive::Unit)\n    } else {\n        Ty::Tuple(elems)\n    }",
             repl="    if elems.is_empty() {\n        Ty::Tuple(Vec::new())\n    } else {\n        Ty::Tuple(elems)\n    }",
             tests=["--test", "conformance", "--test", "three_engine_differential"],
             note="Reverts TYPE-PRIM-001: `()` stops canonicalising to `Unit`. EI5 predicted "
                  "KILLED by EV-SPEC-FIXTURES, the strongest control in the tree. THE CENSUS "
                  "CANNOT CONFIRM THAT CLAIM -- the spec-fixture manifest carries no normative "
                  "rule IDs at all, so no citation links it to TYPE-PRIM-001. This trial decides "
                  "it by measurement instead of by reading the manifest."),
    ],
    # --------------------------------------- EI5 Batch 3 (generic compatibility / trait tables) --
    "3": [
        dict(id="AS8-MUT-014", target="ESF-TRAIT-001", tag="SHARED_AUTHORITY", expect="SURVIVED",
             authority="typecheck::traits::core_trait_contract — receiver",
             file="src/typecheck/traits.rs",
             find='name: "eq",\n                receiver: Some(Ref),',
             repl='name: "eq",\n                receiver: Some(Value),',
             tests=["--test", "copy_canon_matrix", "--test", "conformance",
                    "--test", "gate4a_prelude_traits", "--test", "three_engine_differential"],
             note="`Eq::eq` declared to take `self` by value rather than `&self`. 06 fixes the "
                  "receiver. EI5 expects a survivor because copy_canon_matrix enumerates FROM "
                  "core_method_signature — and MUT-003 has since demonstrated that suite is a "
                  "transcription, so the prediction now rests on measurement, not inference."),
        dict(id="AS8-MUT-015", target="ESF-TRAIT-001", tag="SHARED_AUTHORITY", expect="SURVIVED",
             authority="typecheck::traits::core_trait_contract — return type",
             file="src/typecheck/traits.rs",
             find="ret: Some(ContractTy::Ordering),",
             repl="ret: Some(ContractTy::Bool),",
             tests=["--test", "copy_canon_matrix", "--test", "conformance",
                    "--test", "gate4a_prelude_traits", "--test", "three_engine_differential"],
             note="`Ord::cmp` declared to return Bool rather than Ordering. A maximal disturbance "
                  "of a Core trait contract's return type."),
    ],
    # ------------------------------------------- EI5 Batch 4 (provider and resource mappings) --
    "4": [
        dict(id="AS8-MUT-016", target="ESF-PROV-001", tag="SHARED_AUTHORITY", expect="KILLED",
             authority="mir::provider_sig::signature",
             file="src/mir/provider_sig.rs",
             find="    Ok((tys, PROVIDER_STATUS_TY))",
             repl="    tys.reverse();\n    Ok((tys, PROVIDER_STATUS_TY))",
             tests=["--test", "a10_provider_resolve", "--test", "a10_provider_call",
                    "--test", "a10_provider_verify", "--test", "a10_provider_resource"],
             note="Provider parameter order reversed. EV-PROVIDER-LOOP is EXTERNALLY_DERIVED — "
                  "live peers — so EI5 expects a real kill here. Two engines only (EI2-R2): the "
                  "interpreters have no host access, so there is no third-engine control."),
        dict(id="AS8-MUT-017", target="ESF-RES-001", tag="SHARED_AUTHORITY", expect="KILLED",
             authority="mir::mir_ty_is_copy — HostResource arm",
             file="src/mir/mod.rs",
             find="        MirTy::HostResource(_) => false,",
             repl="        MirTy::HostResource(_) => true,",
             tests=["--test", "dev146_resource_borrow_weakening", "--test", "c788_resource_lifecycle",
                    "--test", "a10_provider_resource", "--test", "a11_host_resource"],
             note="A host resource classified Copy-eligible — THE EXACT A11/CD-234 SHAPE THE CODE "
                  "COMMENT WARNS ABOUT, which records that a wildcard here made a resource Copy "
                  "with three silent consequences. A survivor would mean that warning is "
                  "unenforced by anything."),
    ],
    # ------------------------------------------------ EI5 Batch 5 (canonicalisation helpers) --
    "5": [
        # PREFLIGHT FINDING, 2026-08-09. The original single anchor matched TWICE: `is_integer`
        # and `is_integer_primitive` are BYTE-IDENTICAL, both `pub(super)`, both in types.rs, both
        # answering "is this primitive an integer". A duplicated shared authority inside ONE
        # module -- the CD-065 shape the register exists to catch. The harness replaces the FIRST
        # occurrence, so this trial would silently have mutated the other function and reported
        # the result under the wrong name. Each copy now gets its own trial, and the PAIR of
        # results says which consumers each copy actually reaches.
        dict(id="AS8-MUT-018", target="ESF-TYPE-001", tag="SHARED_AUTHORITY", expect="KILLED",
             authority="typecheck::types::is_integer (copy 1 of 2)",
             file="src/typecheck/types.rs",
             find='pub(super) fn is_integer(p: Primitive) -> bool {\n    matches!(\n        p,\n        Primitive::Int8\n            | Primitive::Int16\n            | Primitive::Int32\n            | Primitive::Int64\n            | Primitive::UInt8\n            | Primitive::UInt16\n            | Primitive::UInt32\n            | Primitive::UInt64\n    )\n}',
             repl='pub(super) fn is_integer(p: Primitive) -> bool {\n    matches!(\n        p,\n        Primitive::Int8\n            | Primitive::Int16\n            | Primitive::Int32\n            | Primitive::Int64\n            | Primitive::UInt8\n            | Primitive::UInt16\n            | Primitive::UInt32\n    )\n}',
             tests=["--test", "conformance", "--test", "three_engine_differential",
                    "--test", "mir_differential"],
             note="UInt64 stops reporting as an integer, in `is_integer` ONLY."),
        dict(id="AS8-MUT-024", target="ESF-TYPE-001", tag="SHARED_AUTHORITY", expect="KILLED",
             authority="typecheck::types::is_integer_primitive (copy 2 of 2)",
             file="src/typecheck/types.rs",
             find='pub(super) fn is_integer_primitive(p: Primitive) -> bool {\n    matches!(\n        p,\n        Primitive::Int8\n            | Primitive::Int16\n            | Primitive::Int32\n            | Primitive::Int64\n            | Primitive::UInt8\n            | Primitive::UInt16\n            | Primitive::UInt32\n            | Primitive::UInt64\n    )\n}',
             repl='pub(super) fn is_integer_primitive(p: Primitive) -> bool {\n    matches!(\n        p,\n        Primitive::Int8\n            | Primitive::Int16\n            | Primitive::Int32\n            | Primitive::Int64\n            | Primitive::UInt8\n            | Primitive::UInt16\n            | Primitive::UInt32\n    )\n}',
             tests=["--test", "conformance", "--test", "three_engine_differential",
                    "--test", "mir_differential"],
             note="The SAME mutation applied to the duplicate. If the two trials differ, the "
                  "copies are reached by different consumers and the duplication is a live "
                  "divergence risk; if either survives, that copy is unguarded."),
        dict(id="AS8-MUT-019", target="ESF-TYPE-001", tag="SHARED_AUTHORITY", expect="KILLED",
             authority="typecheck::types::strip_ref",
             file="src/typecheck/types.rs",
             find="    let mut current = ty;\n    while let Ty::Ref { inner, .. } = current {",
             repl="    let mut current = ty;\n    if let Ty::Ref { inner, .. } = current {",
             tests=["--test", "conformance", "--test", "three_engine_differential",
                    "--test", "mir_differential", "--test", "c61f_nested_refs"],
             note="`strip_ref` stops at ONE level. TYPE-METHOD-002 makes nested-reference "
                  "receivers normative — auto-deref removes one leading `&` AT A TIME — so this "
                  "should break nested-reference method resolution. c61f_nested_refs is selected "
                  "under the AS8 rule: it names the behaviour even though it is not a "
                  "differential suite."),
    ],
    # --------------------------------------- EI5 Batch 7 (rustc-sensitive lowering decisions) --
    # EI3's finding: overflow, shift and drop order were CONVERTED from rustc assumptions into
    # STARK decisions. These mutations hand each one back to rustc, which is the only way to test
    # whether the conversion is load-bearing or decorative.
    "7": [
        dict(id="AS8-MUT-020", target="RA-OVERFLOW", tag="BACKEND_ASSUMPTION", expect="KILLED",
             authority="backend::emit_checked_expr — the destination-width range check",
             file="src/backend/generated_rust/emit_bodies.rs",
             find='"match {checked} {{ Some(__v) if __v >= {min} && __v <= {max} => __v as ',
             repl='"match {checked} {{ Some(__v) => __v as ',
             tests=["--test", "three_engine_differential", "--test", "mir_differential"],
             note="Drops the destination-width range check from arithmetic emission, so an "
                  "i128-representable result is cast down with `as` instead of trapping — i.e. "
                  "overflow behaviour re-delegated to the build. EI3 records `overflow-checks` as "
                  "'RECORDED RATHER THAN RELIED UPON'; this tests that claim. HIR and MIR trap "
                  "independently of the generated Rust, so a survivor would contradict EI3."),
        dict(id="AS8-MUT-021", target="RA-SHIFT", tag="BACKEND_ASSUMPTION", expect="KILLED",
             authority="backend::emit_checked_expr — STARK's own shift-count check",
             file="src/backend/generated_rust/emit_bodies.rs",
             find="if __count < 0 || __count >= {width} {{ ",
             repl="if false {{ ",
             tests=["--test", "three_engine_differential", "--test", "mir_differential"],
             note="Removes STARK's shift-count check, leaving Rust's `checked_shl`, WHICH "
                  "VALIDATES ONLY THE SHIFT COUNT AGAINST i128 and not against the destination "
                  "width. This is EI3's one documented divergence between the two languages. A "
                  "survivor means no test distinguishes STARK's shift rule from Rust's."),
        dict(id="AS8-MUT-022", target="RA-DROP", tag="BACKEND_ASSUMPTION", expect="KILLED",
             authority="mir::drop_plan::array_order",
             file="src/mir/drop_plan.rs",
             find="pub fn array_order(len: u64) -> impl Iterator<Item = u64> {\n    (0..len).rev()\n}",
             repl="pub fn array_order(len: u64) -> impl Iterator<Item = u64> {\n    (0..len).rev().rev()\n}",
             tests=["--test", "three_engine_differential", "--test", "mir_differential"],
             note="Array destruction order reversed. mir and native SHARE this plan (ESF-DROP-002), "
                  "so the HIR engine's independent destruction walk is the ONLY control. A "
                  "survivor means that lone control does not reach array drop order."),
    ],
    # ------------------------------------------- EI5 Batch 8 (correlated evidence generators) --
    "8": [
        dict(id="AS8-MUT-023", target="EV-CORPUS-C6", tag="EVIDENCE_SHARED", expect="KILLED",
             authority="the C6 corpus's expected_trap_category, vs a real semantic change",
             file="src/mir/lower.rs",
             find="            BinOp::Div => (CheckedOp::Div, TrapCategory::DivideByZero),",
             repl="            BinOp::Div => (CheckedOp::Div, TrapCategory::IntegerOverflow),",
             tests=["--test", "c6_generated_corpus"],
             note="The MUT-007 mutation, run against THE CORPUS ALONE. EI5 asks whether the "
                  "corpus 'detects semantic change' or merely 'detects corpus change'. Its cases "
                  "carry expected_trap_category, so a kill here means the corpus pins semantics "
                  "independently of the engines agreeing. EV-COPY-MATRIX, Batch 8's other row, is "
                  "ALREADY ANSWERED by MUT-003 — it survived, so it is a transcription."),
    ],
    # ----------------------------------------------------- EI5 Batch 6 (trap categorisation) --
    # EI2-R3 and the register both say a mis-categorised trap is "invisible to every mechanism in
    # the tree", and rank ESF-TRAP-001 INVISIBLE on that basis. The measurement says otherwise:
    #
    #     interp.rs        28 assignment sites, all 10 categories
    #     mir/lower.rs +   30 assignment sites, all 10 categories
    #     mir/interp.rs
    #     backend           3 assignment sites (the rest are inherited from the runtime)
    #
    # The VOCABULARY is shared -- one enum, and the corpus manifest states expectations in it.
    # The ASSIGNMENT is not: the same operation is categorised twice, independently, in two files.
    # These two trials separate them, and their predictions differ, which is the point.
    "6": [
        dict(id="AS8-MUT-007", target="ESF-TRAP-001b", tag="SHARED_AUTHORITY", expect="KILLED",
             authority="trap category ASSIGNMENT — mir/lower.rs only (one-sided)",
             file="src/mir/lower.rs",
             find="            BinOp::Div => (CheckedOp::Div, TrapCategory::DivideByZero),",
             repl="            BinOp::Div => (CheckedOp::Div, TrapCategory::IntegerOverflow),",
             tests=["--test", "three_engine_differential", "--test", "mir_differential"],
             note="Division by zero reported as IntegerOverflow ON THE MIR PATH ONLY. interp.rs "
                  "still assigns DivideByZero at its own site, so `oracle_category` should "
                  "disagree. Predicted KILLED -- which, if it holds, means assignment is "
                  "PARTIALLY_VISIBLE and the register's INVISIBLE is wrong."),
        dict(id="AS8-MUT-008", target="ESF-TRAP-001a", tag="SHARED_AUTHORITY", expect="SURVIVED",
             authority="trap category VOCABULARY — the enum both engines match on",
             file="src/mir/mod.rs",
             find="    DivideByZero,",
             repl="    DivideByZero, // vocabulary probe\n",
             tests=["--test", "three_engine_differential", "--test", "mir_differential"],
             note="A NO-OP on the vocabulary, paired with MUT-007 to keep the file honest. The "
                  "real vocabulary question cannot be posed as a source mutation at all: if the "
                  "enum names the WRONG CONCEPT, every engine and the corpus manifest are wrong "
                  "together and no in-tree mechanism can disagree. That is the residual EI2-R3 "
                  "should state, and it is NARROWER than what it currently says."),
    ],
}

def run(cmd, **kw):
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, **kw)

# A mutation harness is a PARALLEL WRITER inside your own session, and this checkout is shared.
# On 2026-08-09 a `git add starkc/src/typecheck/` issued while Batch 1 was mid-trial committed
# AS8-MUT-002 to a pushed branch; every C6.5 job failed and the cause read as a refactor
# regression for the better part of an hour. Restoring in `finally` was never enough on its own,
# because nothing PROVED the file was restored and nothing stopped a commit from racing it.
def assert_matches_head(path, when):
    rel = os.path.relpath(path, ROOT)
    r = run(["git", "diff", "--quiet", "HEAD", "--", rel])
    if r.returncode != 0:
        sys.exit(f"as8-mutate: {rel} differs from HEAD {when}.\n"
                 f"  Refusing to continue. A mutated file must never be staged, and an unrelated\n"
                 f"  local edit must never be attributed to a mutation. Commit or stash first,\n"
                 f"  and check `git diff -- {rel}` before trusting any result.")


def extract_killers(text):
    """Which tests failed, and what the first divergence actually SAID.

    EI5 makes `killer independence` a required field, so a bare KILLED is not a usable record:
    a kill by engine disagreement and a kill by the front end rejecting the corpus are different
    results, and they look identical in a pass/fail count."""
    # `cargo test --quiet` prints dots, not "test NAME ... FAILED", so the per-test line is
    # absent and only the trailing `failures:` block names anything. Parse that block.
    failed = sorted(set(re.findall(r"^test (\S+) \.\.\. FAILED$", text, re.M)))
    if not failed:
        for block in re.findall(r"^failures:\n((?:    \S+\n)+)", text, re.M):
            failed.extend(line.strip() for line in block.splitlines() if line.strip())
        failed = sorted(set(failed))
    panic = re.search(r"panicked at [^\n]+\n(.+?)(?=\nnote:|\ntest |\Z)", text, re.S)
    return failed, (panic.group(1).strip()[:600] if panic else "")


def trial(spec, verbose):
    path = os.path.join(ROOT, spec["file"])
    assert_matches_head(path, "BEFORE the trial started")
    original = open(path, encoding="utf-8").read()
    if spec["find"] not in original:
        return dict(spec_id=spec["id"], result="NOT_APPLIED",
                    detail="anchor text not found — the target moved; re-derive before trusting any batch")
    backup = tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8")
    backup.write(original); backup.close()
    started = time.time()
    try:
        open(path, "w", encoding="utf-8").write(original.replace(spec["find"], spec["repl"], 1))
        build = run(["cargo", "build", "--quiet", "-p", "starkc", "--tests"])
        if build.returncode != 0:
            return dict(spec_id=spec["id"], result="BUILD_FAILED",
                        detail="the mutant does not compile; it is not a semantic mutation",
                        stderr=build.stderr[-800:])
        cmd = ["cargo", "test", "--quiet", "-p", "starkc"] + spec["tests"]
        out = run(cmd)
        killed = out.returncode != 0
        failed, divergence = extract_killers(out.stdout + out.stderr)
        return dict(spec_id=spec["id"], result="KILLED" if killed else "SURVIVED",
                    seconds=round(time.time() - started, 1),
                    killers=failed[:12], killer_count=len(failed), divergence=divergence,
                    detail=(out.stdout + out.stderr)[-600:] if verbose else "")
    finally:
        shutil.copyfile(backup.name, path)
        os.unlink(backup.name)
        assert_matches_head(path, "AFTER restoration — THE RESTORE DID NOT TAKE")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", required=True)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--json", help="write the trial record here")
    a = ap.parse_args()
    specs = BATCHES.get(a.batch)
    if not specs:
        sys.exit(f"unknown batch {a.batch}; defined: {sorted(BATCHES)}")
    records, ok = [], True
    for spec in specs:
        r = trial(spec, a.verbose)
        r.update(target=spec["target"], tag=spec["tag"], authority=spec["authority"],
                 expected=spec["expect"], note=spec["note"])
        r["verdict"] = "CONFIRMED" if r["result"] == spec["expect"] else "UNEXPECTED"
        ok &= r["verdict"] == "CONFIRMED"
        print(f"  {r['spec_id']:<22} expected {spec['expect']:<9} got {r['result']:<12} {r['verdict']}")
        if r.get("divergence"):
            print(f"      killed by {r['killer_count']} test(s), first: {r['killers'][0] if r['killers'] else '?'}")
            for line in r["divergence"].splitlines()[:6]:
                print(f"        {line}")
        if r["verdict"] == "UNEXPECTED" and a.verbose and r.get("detail"):
            print(f"      {r['detail'][:400]}")
        records.append(r)
    if a.json:
        open(a.json, "w", encoding="utf-8").write(json.dumps(records, indent=2) + "\n")
    if a.batch == "0":
        print()
        print("  BATCH 0 IS THE PRECONDITION FOR EVERY OTHER BATCH." if ok else
              "  BATCH 0 FAILED — no kill rate from any other batch is interpretable.")
    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
