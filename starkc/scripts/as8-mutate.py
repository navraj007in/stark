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
    # ---------------------------------------------------------------------------------------
    # WP-ARCH-CLOSE AC4. AS8 selected its targets under EI5; AC4 requires at least one trial per
    # CRITICAL AUTHORITY, and the map of AS8's 26 trials against AC4's 11 authorities left two with
    # NO trial at all -- pattern legality and the generic specialization environment -- plus
    # `mir::borrows`, which did not exist when AS8 ran.
    #
    # Pattern legality is first because it has the worst defect history in the tree: DEV-222, 223,
    # 225, 226 and 227 all landed in a single day, and every one was a pattern that compiled,
    # reported nothing, and silently never matched. An authority with that record and zero
    # adversarial coverage is exactly what AC4 exists to find.
    "ac4-pat": [
        dict(id="AC4-MUT-PAT-001", target="pattern legality", tag="FRONT_END",
             authority="resolve::resolution_is_pattern_legal — associated fn admitted as a pattern",
             expect="KILLED",
             file="src/resolve.rs",
             find="            Res::AssociatedFn(_, _)\n            | Res::TraitMember(_, _)",
             repl="            Res::AssociatedFn(_, _) => true,\n            Res::TraitMember(_, _)",
             tests=["--test", "dev222_pattern_only_resolutions",
                    "--test", "dev225_227_resolution_namespaces",
                    "--test", "adversarial_patterns"],
             note="This is DEV-222's exact defect re-injected: `Colour::Blu` resolves to an "
                  "associated function and is accepted as a pattern that never matches. If the "
                  "controls do not kill this, the repair is unguarded."),
        dict(id="AC4-MUT-PAT-002", target="pattern legality", tag="FRONT_END",
             authority="resolve::resolution_is_pattern_legal — any ITEM admitted as a pattern",
             expect="KILLED",
             file="src/resolve.rs",
             find="            Res::Item(item_id) => matches!(\n                self.item_details.get(item_id),\n                Some(ItemDefDetail::Struct { .. }) | Some(ItemDefDetail::Const)\n            ),",
             repl="            Res::Item(_item_id) => true,",
             tests=["--test", "dev222_pattern_only_resolutions",
                    "--test", "dev225_227_resolution_namespaces",
                    "--test", "adversarial_patterns"],
             note="DEV-227's defect: `match n { helper => .., _ => .. }` with `helper` a FUNCTION "
                  "compiles and matches nothing."),
    ],
    # MIR verification. AS8's only trial is `paths_prefix_related` (batch 9, KILLED) -- one
    # predicate out of a verifier carrying 36 distinct MIR-nnnn rules across 60 functions.
    #
    # This authority needs a different mutation strategy from the others, and the difference is the
    # point. Removing a CHECK does not break correct programs: it only matters if something feeds
    # the verifier bad MIR. So a surviving mutation here means "no test constructs the malformed
    # shape", which is a coverage statement about the verifier's NEGATIVE cases -- exactly what a
    # verifier is for.
    #
    # Selected by measurement: 33 of the 36 rule ids are named by at least one test; MIR-0029,
    # MIR-0035 and MIR-0037 are named by NONE. VER-001 takes one of those, VER-002 a well-named
    # rule as the positive control that the method works at all on this authority.
    "ac4-verify": [
        dict(id="AC4-MUT-VER-001", target="MIR verification", tag="MIR",
             authority="mir::verify MIR-0035 — storage_dead on a PROJECTED place is accepted",
             expect="SURVIVED",
             file="src/mir/verify.rs",
             find="                Statement::StorageDead(place, _) => {\n                    if !place.projection.is_empty() {",
             repl="                Statement::StorageDead(place, _) => {\n                    if false {",
             tests=["--lib", "--test", "mir_verify", "--test", "mir_differential",
                    "--test", "a12_storage_end_shapes"],
             note="A12: storage liveness belongs to a WHOLE local, so a projected `storage_dead` is "
                  "a lowering defect. Declared SURVIVED on the census -- no test names MIR-0035 -- "
                  "and a KILLED result is good news requiring re-declaration. A verifier rule no "
                  "test exercises is a rule nothing proves is enforced."),
        dict(id="AC4-MUT-VER-002", target="MIR verification", tag="MIR",
             authority="mir::verify::paths_prefix_related — move-path overlap, AS8's own target",
             expect="KILLED",
             file="src/mir/verify.rs",
             # The anchor first named `&Place` parameters and came back NOT_APPLIED -- the harness
             # refusing to report a verdict on a mutation that never landed, which is the outcome
             # that stops a stale anchor being read as a SURVIVED. The real signature takes
             # `&[MovePathStep]`.
             find="fn paths_prefix_related(a: &[MovePathStep], b: &[MovePathStep]) -> bool {\n    let n = a.len().min(b.len());\n    a[..n] == b[..n]",
             repl="fn paths_prefix_related(a: &[MovePathStep], b: &[MovePathStep]) -> bool {\n    let _ = (a, b);\n    false",
             tests=["--lib", "--test", "mir_verify", "--test", "mir_differential"],
             note="The positive control: AS8 already killed this one, so a SURVIVED here would "
                  "mean the method has stopped working on this authority rather than that the "
                  "rule is unguarded."),
    ],
    # Drop determination. AS8's only trial is `nominals_with_destructor` (batch 1, SURVIVED) —
    # which answers "does this nominal declare a destructor", one input to the real authority.
    # `requires_drop_glue_with` is the authority, a match over NINE arms, and it is the one the
    # module header calls exhaustive on purpose so "a new MirTy variant must be classified at this
    # one authority". One trial per arm, on four of the nine.
    "ac4-drop": [
        dict(id="AC4-MUT-DRP-001", target="Drop determination", tag="MIR",
             authority="mir::drop_rule::requires_drop_glue_with — String stops owning anything",
             expect="KILLED",
             file="src/mir/drop_rule.rs",
             find="        MirTy::String => true,",
             repl="        MirTy::String => false,",
             tests=["--lib", "--test", "three_engine_differential", "--test", "mir_differential",
                    "--test", "native_c6_1_ownership", "--test", "as4_destructor_authority"],
             note="The most-owned type in the language stops being destroyed. If the drop logs the "
                  "differential compares do not notice this, they are not comparing destruction."),
        dict(id="AC4-MUT-DRP-002", target="Drop determination", tag="MIR",
             authority="requires_drop_glue_with — the STRUCT arm stops recursing into fields",
             expect="KILLED",
             file="src/mir/drop_rule.rs",
             find="            facts.has_user_destructor(*item, args)\n                || facts\n                    .struct_fields(*item, args)\n                    .is_some_and(|fields| fields.iter().any(|f| requires_drop_glue_with(f, facts)))",
             repl="            facts.has_user_destructor(*item, args)",
             tests=["--lib", "--test", "three_engine_differential", "--test", "mir_differential",
                    "--test", "native_c6_1_ownership", "--test", "as4_destructor_authority"],
             note="ARM-LEVEL and the realistic defect: a struct with no `Drop` impl but a `String` "
                  "field stops dropping the field. `nominals_with_destructor` -- AS8's only trial "
                  "here -- would still answer correctly, which is why it is not coverage of this."),
        dict(id="AC4-MUT-DRP-003", target="Drop determination", tag="MIR",
             authority="requires_drop_glue_with — a HOST RESOURCE stops needing its close",
             expect="KILLED",
             file="src/mir/drop_rule.rs",
             find="        MirTy::HostResource(_) => true,",
             repl="        MirTy::HostResource(_) => false,",
             tests=["--lib", "--test", "three_engine_differential", "--test", "mir_differential",
                    "--test", "native_c6_1_ownership", "--test", "a11_host_resource"],
             note="A11 §5: a host resource's drop IS its provider close. This leaks the socket, "
                  "the file handle and the TLS session -- the class CD-347/348 built live-peer "
                  "lifecycle evidence for."),
        dict(id="AC4-MUT-DRP-004", target="Drop determination", tag="MIR",
             authority="requires_drop_glue_with — the TUPLE arm stops recursing",
             expect="KILLED",
             file="src/mir/drop_rule.rs",
             find="        MirTy::Tuple(elems) => elems.iter().any(|e| requires_drop_glue_with(e, facts)),",
             repl="        MirTy::Tuple(_elems) => false,",
             tests=["--lib", "--test", "three_engine_differential", "--test", "mir_differential",
                    "--test", "native_c6_1_ownership", "--test", "as4_destructor_authority"],
             note="`(String, Int32)` stops dropping its String. A composite arm, distinct from the "
                  "struct arm above, and the shape DEV-142 already shows is delicate."),
    ],
    # Trait / bound dispatch. AS8's only trial here is batch 3 (`core_trait_contract` receiver),
    # which is one function and SURVIVED. `satisfies_bound_identity` is the authority, and it is a
    # match over FIVE semantic arms — reference forwarding, the primitive matrix, the Core-type
    # rules, nominal impl witness, and the generic-parameter discharge. One trial on one function
    # says nothing about the other four, which is the lesson pattern legality and substitute_ty
    # both taught. One trial per arm.
    "ac4-bound": [
        # DECLARED SURVIVED on measurement, and this is AC4-F3 rather than a pass. An arm census
        # (probe on `satisfies_bound_identity`, whole lib suite) shows the Ref arm executed ZERO
        # times: the suite reaches this authority 26 times and touches only the Primitive and
        # Nominal arms, for Eq/Hash/Ord. This mutation is therefore UNREACHABLE by the controls,
        # and a SURVIVED here means "never run", not "not detected".
        dict(id="AC4-MUT-BND-001", target="trait / bound dispatch", tag="FRONT_END",
             authority="satisfies_bound_identity — the Ty::Ref FORWARDING arm drops Display",
             expect="SURVIVED",
             file="src/typecheck/traits.rs",
             find="                    || bound_name == \"Hash\"\n                    || bound_name == \"Display\"\n                {",
             repl="                    || bound_name == \"Hash\"\n                {",
             tests=["--lib", "--test", "three_engine_differential",
                    "--test", "native_c6_2_generics_traits"],
             note="`&T` stops forwarding Display to `T`, so `fn show<T: Display>(v: &T)` -- the "
                  "shape the file's own comment cites as routine -- loses its bound."),
        # DECLARED SURVIVED on measurement. The Primitive/Ord arm IS reached -- once, in the whole
        # lib suite -- but not with `Bool`, so admitting Bool as Ord changes nothing observable.
        # One reach is not coverage of a matrix with eight primitives in it.
        dict(id="AC4-MUT-BND-002", target="trait / bound dispatch", tag="FRONT_END",
             authority="satisfies_bound_identity — the PRIMITIVE matrix admits Bool as Ord",
             expect="SURVIVED",
             file="src/typecheck/traits.rs",
             find="                    !matches!(p, Primitive::Unit | Primitive::Bool) && !is_float_primitive(*p)",
             repl="                    !matches!(p, Primitive::Unit) && !is_float_primitive(*p)",
             tests=["--lib", "--test", "three_engine_differential",
                    "--test", "native_c6_2_generics_traits"],
             note="DEV-075's matrix says Ord is Eq's set MINUS Bool -- `Char` is ordered, `Bool` is "
                  "not. This makes `Bool` orderable, which is one token's difference from the "
                  "Eq arm directly above it."),
        # DECLARED SURVIVED on measurement, and this is AC4-F3 rather than a pass. An arm census
        # (probe on `satisfies_bound_identity`, whole lib suite) shows the Core arm executed ZERO
        # times: the suite reaches this authority 26 times and touches only the Primitive and
        # Nominal arms, for Eq/Hash/Ord. This mutation is therefore UNREACHABLE by the controls,
        # and a SURVIVED here means "never run", not "not detected".
        dict(id="AC4-MUT-BND-003", target="trait / bound dispatch", tag="FRONT_END",
             authority="satisfies_bound_identity — the Core ITERATOR list drops VecIter",
             expect="SURVIVED",
             file="src/typecheck/traits.rs",
             find="                        || *core_type == CoreType::VecIter\n                        || *core_type == CoreType::KeysIter",
             repl="                        || *core_type == CoreType::KeysIter",
             tests=["--lib", "--test", "three_engine_differential",
                    "--test", "native_c6_2_generics_traits"],
             note="A closed membership list, and the most-used member removed. `for x in v.iter()` "
                  "should stop satisfying Iterator."),
        # DECLARED SURVIVED on measurement, and this is AC4-F3 rather than a pass. An arm census
        # (probe on `satisfies_bound_identity`, whole lib suite) shows the Param arm executed ZERO
        # times: the suite reaches this authority 26 times and touches only the Primitive and
        # Nominal arms, for Eq/Hash/Ord. This mutation is therefore UNREACHABLE by the controls,
        # and a SURVIVED here means "never run", not "not detected".
        dict(id="AC4-MUT-BND-004", target="trait / bound dispatch", tag="FRONT_END",
             authority="satisfies_bound_identity — the Ty::Param arm stops discharging DEV-067(a)",
             expect="SURVIVED",
             file="src/typecheck/traits.rs",
             find="            Ty::Param(param_name) => self.param_declares_bound(param_name, &bound_name, bound_res),",
             repl="            Ty::Param(_param_name) => false,",
             tests=["--lib", "--test", "three_engine_differential",
                    "--test", "native_c6_2_generics_traits"],
             note="DEV-067(a) re-injected: a generic fn calling another with a bounded parameter -- "
                  "including simple recursion -- fails E0500 although the bound is declared right "
                  "there. This arm did not exist once, and its absence was a real defect."),
    ],
    # Resolution / NAMESPACES. AS8 predates DEV-228 entirely: its only resolver trials are
    # `item_is_visible_from` (batches 9, 9b), which is visibility, not namespacing. DEV-228 rebuilt
    # this surface — the resolver now carries the module/type/value namespaces NAME-RESOLVE-001
    # specifies — and the namespaces themselves have never been mutated.
    #
    # All three are ARM-LEVEL, per what pattern legality and substitute_ty both showed: an authority
    # is covered when its arms are, not when it has a trial.
    "ac4-ns": [
        dict(id="AC4-MUT-NS-001", target="resolution / namespaces", tag="FRONT_END",
             authority="resolve::namespace_of_item — a TRAIT filed under Value instead of Type",
             expect="KILLED",
             file="src/resolve.rs",
             find="            | ast::ItemKind::TypeAlias { .. }\n            | ast::ItemKind::Model(_) => Namespace::Type,",
             repl="            | ast::ItemKind::TypeAlias { .. }\n            | ast::ItemKind::Model(_) => Namespace::Value,",
             tests=["--lib", "--test", "dev228_namespaces", "--test", "dev225_227_resolution_namespaces",
                    "--test", "dev229_builtin_spellings"],
             note="Struct, enum, trait, alias and model all move to the VALUE namespace. A type "
                  "annotation should stop finding them, which is the coarsest possible break of "
                  "the distinction DEV-228 built."),
        dict(id="AC4-MUT-NS-002", target="resolution / namespaces", tag="FRONT_END",
             authority="resolve::namespace_of_item — a FUNCTION filed under Type instead of Value",
             expect="KILLED",
             file="src/resolve.rs",
             find="            ast::ItemKind::Fn(_) | ast::ItemKind::Const { .. } => Namespace::Value,",
             repl="            ast::ItemKind::Fn(_) | ast::ItemKind::Const { .. } => Namespace::Type,",
             tests=["--lib", "--test", "dev228_namespaces", "--test", "dev225_227_resolution_namespaces",
                    "--test", "dev229_builtin_spellings"],
             note="DEV-228's motivating case inverted: `struct Pair` alongside `fn Pair()` is legal "
                  "precisely because they occupy different namespaces. Filing functions under Type "
                  "makes them collide."),
        dict(id="AC4-MUT-NS-003", target="resolution / namespaces", tag="FRONT_END",
             authority="resolve::lookup_ns — NsHint::Type reads the VALUE map",
             expect="KILLED",
             file="src/resolve.rs",
             find="            NsHint::Type => self.ns_map(module, Namespace::Type).get(name).copied(),",
             repl="            NsHint::Type => self.ns_map(module, Namespace::Value).get(name).copied(),",
             tests=["--lib", "--test", "dev228_namespaces", "--test", "dev225_227_resolution_namespaces",
                    "--test", "dev229_builtin_spellings"],
             note="The READ side rather than the FILE side: a position that knows it wants a type "
                  "consults the value map. NS-001/002 break where names are put; this breaks where "
                  "they are looked for, and a control that kills one but not the other tells us "
                  "which half is watched."),
    ],
    # The generic specialization environment — the LAST AC4 authority with no trial at all.
    #
    # GEN-003 exists because of what pattern legality taught: an authority is not covered because
    # one of its functions has a trial. `substitute_ty` is a match over ~12 type shapes, and a
    # regression in ONE arm is the realistic defect. GEN-001/002 break the whole authority loudly;
    # GEN-003 breaks a single arm, which is the shape that hides.
    "ac4-gen": [
        # RESOLVED BY DELETION (owner, 2026-08-12). This trial has no target any more, and that is
        # the OUTCOME rather than a gap in the campaign.
        #
        # GEN-001 first mutated `GenericEnvironment::substitutions()` and SURVIVED -- because that
        # method has ZERO callers, not because a control was missing. Retargeted at the copy that IS
        # reached (`bound_dispatch`'s inline map) it STILL survived, across ~850 tests including the
        # three-engine differential, while a probe showed the map built with real bindings six times
        # in two suites. Reached, corrupted, unobserved.
        #
        # That is not "a live semantic fact with no falsifier". It is a CONSTRUCTED fact with no
        # production consumer: the engines read `body` (six sites) and `environment` (one), never
        # `signature`. The owner ruled deletion rather than a manufactured consumer -- architecture
        # documentation should describe what execution needs, not make execution consume something
        # because the documentation promised it. `ResolvedCallable.signature` is gone.
        #
        # `declaration` was censused at the same time and KEPT on different grounds: it is a field
        # copy of an already-made selection, costs nothing, and witnesses the impl-override versus
        # trait-default distinction `as3_callable_use_keying` asserts.
        #
        # A replacement trial belongs here only if the specialiser acquires a new derived fact.
        dict(id="AC4-MUT-GEN-002", target="generic specialization env", tag="FRONT_END",
             authority="typecheck::substitute_ty — the Ty::Param arm never substitutes",
             expect="KILLED",
             file="src/typecheck/types.rs",
             find="        Ty::Param(name) => map.get(name).cloned().unwrap_or_else(|| ty.clone()),",
             repl="        Ty::Param(_name) => ty.clone(),",
             tests=["--lib", "--test", "native_c6_2_generics_traits", "--test", "native_c5_4_generics"],
             note="`T` stays `T` at every instantiation. Same effect as GEN-001 one layer down, "
                  "and a control that kills one but not the other tells us which layer is watched."),
        dict(id="AC4-MUT-GEN-003", target="generic specialization env", tag="FRONT_END",
             authority="typecheck::substitute_ty — ONE arm: Ty::Fn substitutes params, not ret",
             expect="KILLED",
             file="src/typecheck/types.rs",
             find="        Ty::Fn { params, ret } => Ty::Fn {\n            params: params.iter().map(|p| substitute_ty(p, map)).collect(),\n            ret: Box::new(substitute_ty(ret, map)),\n        },",
             repl="        Ty::Fn { params, ret } => Ty::Fn {\n            params: params.iter().map(|p| substitute_ty(p, map)).collect(),\n            ret: ret.clone(),\n        },",
             tests=["--lib", "--test", "native_c6_2_generics_traits", "--test", "native_c5_4_generics",
                    "--test", "native_c5_4_function_values"],
             note="ARM-LEVEL. A function-typed value's RETURN type stops being specialised while "
                  "its parameters still are. This is the shape pattern legality's unguarded arm "
                  "had: the authority looks covered because its other arms are exercised."),
    ],
    # `mir::borrows` is the authority AC1 step 1 created (CE3, 2026-08-12). Its own module-level
    # trials found two of four rules UNCONTROLLED; these promote that finding into the harness so it
    # is tracked mechanically rather than in a doc comment.
    "ac4-borrow": [
        dict(id="AC4-MUT-BOR-001", target="borrow origin", tag="MIR",
             authority="mir::borrows — a call result inherits its arguments' origins unconditionally",
             expect="KILLED",
             file="src/mir/borrows.rs",
             find="                if returns_borrow {",
             repl="                if true {",
             tests=["--lib", "--test", "dev160_call_site_thunk", "--test", "ac1_dev160_probe"],
             note="Removes the precision rule that motivated the module: a scalar result stops "
                  "being excluded and is recorded as borrowing its arguments' storage."),
        dict(id="AC4-MUT-BOR-002", target="borrow origin", tag="MIR",
             authority="mir::borrows — a MOVE stops severing provenance",
             expect="KILLED",
             file="src/mir/borrows.rs",
             find="                    Rvalue::Use(Operand::Move(_)) => Vec::new(),",
             repl="                    Rvalue::Use(Operand::Move(p)) => vec![p.local.0],",
             tests=["--lib", "--test", "dev160_call_site_thunk", "--test", "ac1_dev160_probe"],
             note="The rule whose absence over-refused `stark_http_client::follow` on the first "
                  "DEV-160 repair attempt. Declared KILLED because the module's own pinned "
                  "relation covers it -- if it SURVIVES here the pin is weaker than believed."),
        dict(id="AC4-MUT-BOR-003", target="borrow origin", tag="MIR",
             authority="mir::borrows — the aggregate component filter",
             expect="SURVIVED",
             file="src/mir/borrows.rs",
             find="                        .filter_map(|o| operand_carries(body, o))",
             repl="                        .filter_map(|o| match o { Operand::Copy(p) | Operand::Move(p) => Some(p.local.0), Operand::Const(_) => None })",
             tests=["--lib", "--test", "dev160_call_site_thunk", "--test", "ac1_dev160_probe"],
             note="Declared SURVIVED on the module's own measurement: this rule is precautionary "
                  "and no control reaches it. A trial declared SURVIVED that comes back KILLED is "
                  "GOOD NEWS -- it means a control was found -- and must be re-declared."),
    ],
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
    # ------------------------------- EI5 Batch 4b (Batch 4 re-selected, and why it had to be) --
    # AS8-MUT-016 was recorded SURVIVED and the result is UNINTERPRETABLE, for a reason worth
    # stating rather than quietly fixing: EI5 named `EV-PROVIDER-LOOP` as the control, and THIS
    # HARNESS CANNOT RUN IT. The loopback suites live in `packages/*/native/Cargo.toml` -- separate
    # crates -- and every trial here runs `cargo test -p starkc`. The external control was never
    # invoked, so the trial measured its absence, not the authority's exposure.
    #
    # It is the SECOND time in this packet that a selected test set omitted the only suite that
    # could kill (MUT-005/006 was the first). The rule AS8 added to EI5 -- select every suite that
    # NAMES the authority -- is necessary and was not sufficient: a suite in another CRATE is
    # invisible to "select the right --test flag".
    #
    # In-tree, `provider_sig::signature` has exactly ONE consumer: `mir/verify.rs`. So the honest
    # in-tree question is whether the VERIFIER notices, and these suites are the ones that drive a
    # provider program through it.
    "4b": [
        dict(id="AS8-MUT-025", target="ESF-PROV-001", tag="SHARED_AUTHORITY", expect="KILLED",
             authority="mir::provider_sig::signature — re-selected against its real consumers",
             file="src/mir/provider_sig.rs",
             find="    Ok((tys, PROVIDER_STATUS_TY))",
             repl="    tys.reverse();\n    Ok((tys, PROVIDER_STATUS_TY))",
             tests=["--test", "c788_starkc_build", "--test", "a10_stark_time_e2e",
                    "--test", "a10_provider_emit", "--test", "a10_provider_verify"],
             note="Identical mutation to MUT-016, selected against the suites that actually drive "
                  "a provider program through `mir/verify.rs` and the backend. If this ALSO "
                  "survives, the in-tree evidence does not reach the provider signature mapping "
                  "at all and EI2-R2 understates the gap: not merely 'two engines, not three', "
                  "but no in-tree control of any kind."),
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
    # ---------------------------- EI5 Batch 9 (resolver and MIR verifier rules) -- AS8 SCOPE ---
    # AS8's work section names FIVE rule families for source mutation: "ownership, trap, drop,
    # RESOLVER and MIR VERIFIER rules". Batches 1-8 covered ownership, trap and drop. These are the
    # two that were still missing, and neither appears in EI5's ranked batches -- EI5 ranked the
    # SHARED-FATE register, and the resolver and the verifier are not register entries. That is a
    # gap between two documents rather than in either one, and it is why the packet's own scope has
    # to be read alongside the ranking it delegates to.
    #
    # THE VERIFIER IS AN ODD MUTATION TARGET AND THE PREDICTIONS SAY SO. `mir/verify.rs` is a
    # CHECK, not a producer: WEAKENING it removes a rejection, so nothing fails unless a test
    # asserts the rejection itself. A survivor here means the verifier's rule is unverified, which
    # is a sharper thing than an unverified engine rule -- the verifier is what other evidence
    # leans on.
    "9": [
        dict(id="AS8-MUT-034", target="resolver visibility", tag="SHARED_AUTHORITY", expect="KILLED",
             authority="resolve::item_is_visible_from",
             file="src/resolve.rs",
             find="        matches!(\n            self.ast.item(ast::ItemId(item_id.0)).vis,\n            Some(ast::Vis::Pub)\n        )",
             repl="        true",
             tests=["--test", "conformance", "--test", "three_engine_differential"],
             note="Every item becomes visible from every module — private items stop being private. "
                  "07 governs visibility and resolve.rs carries in-module tests for exactly this "
                  "(`private_item_is_not_visible_from_a_descendant_module`), so `--lib` would also "
                  "kill it; the selection asks whether the SPEC FIXTURES do."),
        dict(id="AS8-MUT-035", target="resolver visibility", tag="SHARED_AUTHORITY", expect="KILLED",
             authority="resolve::name_is_visible_from — re-export visibility",
             file="src/resolve.rs",
             find="        if let Some(vis) = self.reexport_vis.get(&(module_id, name.to_string())) {\n            return matches!(vis, Some(ast::Vis::Pub));\n        }",
             repl="        if self.reexport_vis.contains_key(&(module_id, name.to_string())) {\n            return true;\n        }",
             tests=["--test", "conformance", "--test", "three_engine_differential"],
             note="A NON-`pub` re-export becomes visible outside its module. `use` without `pub` "
                  "must not re-export (07)."),
        dict(id="AS8-MUT-036", target="MIR verifier", tag="SHARED_AUTHORITY", expect="KILLED",
             authority="mir::verify::paths_prefix_related — move-path overlap",
             file="src/mir/verify.rs",
             find="    let n = a.len().min(b.len());\n    a[..n] == b[..n]",
             repl="    let _ = a.len().min(b.len());\n    false",
             tests=["--test", "mir_verify", "--test", "mir_differential",
                    "--test", "three_engine_differential"],
             note="Two move paths are never considered prefix-related, so overlapping partial "
                  "moves stop being detected. WEAKENS the verifier: it removes a rejection rather "
                  "than adding one, so only a test that asserts the rejection can notice."),
        dict(id="AS8-MUT-037", target="MIR verifier", tag="SHARED_AUTHORITY", expect="KILLED",
             authority="mir::verify::may_need_drop — the HostResource arm",
             file="src/mir/verify.rs",
             find="        MirTy::HostResource(_) => true,",
             repl="        MirTy::HostResource(_) => false,",
             tests=["--test", "mir_verify", "--test", "a11_host_resource",
                    "--test", "c788_resource_lifecycle", "--test", "dev146_resource_borrow_weakening"],
             note="A11 §5: a host resource's drop IS its provider close. The source comment records "
                  "that lowering and the verifier were CORRECTED SEPARATELY here — lowering stopped "
                  "emitting the Drop, and when that was fixed the verifier rejected the Drop it now "
                  "emitted. AS8-DA-006 is that pair; this trial disturbs the verifier half."),
    ],
    # -------------------- Batch 9b: the claim Batch 9's notes made, tested instead of asserted --
    # MUT-034/035 selected `conformance` and `three_engine_differential` and BOTH SURVIVED. The
    # trial note asserted "resolve.rs carries in-module tests for exactly this, so --lib would also
    # kill it; the selection asks whether the SPEC FIXTURES do."
    #
    # That second half is a MEASUREMENT and the first half is an ASSERTION, and this packet has now
    # been wrong three times about which suites can kill what. So: same mutations, `--lib` selected.
    # If these kill, the survivors above are a precise statement about spec-fixture coverage. If
    # they DO NOT, the visibility rules have no control anywhere and the finding is far larger.
    "9b": [
        dict(id="AS8-MUT-038", target="resolver visibility", tag="SHARED_AUTHORITY", expect="KILLED",
             authority="resolve::item_is_visible_from — vs the in-module unit tests",
             file="src/resolve.rs",
             find="        matches!(\n            self.ast.item(ast::ItemId(item_id.0)).vis,\n            Some(ast::Vis::Pub)\n        )",
             repl="        true",
             tests=["--lib"],
             note="Every item visible from every module, checked against resolve.rs's own unit "
                  "tests rather than against the fixture corpus."),
        dict(id="AS8-MUT-039", target="resolver visibility", tag="SHARED_AUTHORITY", expect="KILLED",
             authority="resolve::name_is_visible_from — vs the in-module unit tests",
             file="src/resolve.rs",
             find="        if let Some(vis) = self.reexport_vis.get(&(module_id, name.to_string())) {\n            return matches!(vis, Some(ast::Vis::Pub));\n        }",
             repl="        if self.reexport_vis.contains_key(&(module_id, name.to_string())) {\n            return true;\n        }",
             tests=["--lib"],
             note="A non-`pub` re-export becomes visible outside its module, same comparison."),
    ],
    # ----------------------------------- AS8-DA: paired one-sided trials on duplicate authorities --
    # NOT an EI5 batch and NOT an EI0 category. Owner ruling 2026-08-09: EI0's vocabulary answers
    # WHAT KIND of semantic authority something is; duplication answers HOW MANY implementations
    # exist and what relationship they have. Encoding both in one enum would make the model less
    # precise, so these carry AS8-DA-* identifiers and stay orthogonal to the register.
    #
    # THE OWNER ALSO CORRECTED THE INSTINCT TO CONSOLIDATE, and the correction is the whole design
    # of this batch: a verifier can derive its value from implementing a rule INDEPENDENTLY.
    # Replacing both copies with one helper removes drift and CREATES SHARED FATE -- the verifier
    # could then no longer detect a wrong shared predicate. Mutation classifies which it is:
    #
    #     mutate copy A only -> killed?  YES  independent redundancy is useful, KEEP BOTH
    #                                    NO   copy A is unguarded
    #     mutate copy B only -> killed?  YES  useful cross-check
    #                                    NO   copy B can drift silently
    #     both survive -> architectural residual: one authority, or an explicit cross-check
    "da": [
        dict(id="AS8-MUT-026", target="AS8-DA-002", tag="SHARED_AUTHORITY", expect="KILLED",
             authority="is_vec_runtime — implementation A (interpreter)",
             file="src/mir/interp.rs",
             find='fn is_vec_runtime(rt: RuntimeFn) -> bool {\n    use RuntimeFn::*;\n    matches!(\n        rt,\n        VecNew\n            | VecWithCapacity\n            | VecPush\n            | VecPop\n            | VecLen\n            | VecIsEmpty\n            | VecIndexGet\n            | VecReplace\n            | VecRemove\n            | VecClear\n            | VecIterNew\n            | VecIterNext\n            | VecGetRef\n            | VecGetMutRef\n    )\n}',
             repl='fn is_vec_runtime(rt: RuntimeFn) -> bool {\n    use RuntimeFn::*;\n    matches!(\n        rt,\n        VecNew\n            | VecWithCapacity\n            | VecPush\n            | VecPop\n            | VecLen\n            | VecIsEmpty\n            | VecIndexGet\n            | VecReplace\n            | VecRemove\n            | VecIterNew\n            | VecIterNext\n            | VecGetRef\n            | VecGetMutRef\n    )\n}',
             tests=["--test", "mir_differential", "--test", "three_engine_differential", "--test", "mir_verify"],
             note="ONE-SIDED: only implementation A (interpreter) is disturbed, so the other copy still "
                  "holds the correct answer. A KILL means the independent redundancy is doing real "
                  "work and the pair should be KEPT; a SURVIVOR means this copy can drift silently "
                  "and nothing in the tree notices."),
        dict(id="AS8-MUT-027", target="AS8-DA-002", tag="SHARED_AUTHORITY", expect="KILLED",
             authority="is_vec_runtime_fn — implementation B (verifier)",
             file="src/mir/verify.rs",
             find='fn is_vec_runtime_fn(rt: RuntimeFn) -> bool {\n    use RuntimeFn::*;\n    matches!(\n        rt,\n        VecNew\n            | VecWithCapacity\n            | VecPush\n            | VecPop\n            | VecLen\n            | VecIsEmpty\n            | VecIndexGet\n            | VecReplace\n            | VecRemove\n            | VecClear\n            | VecIterNew\n            | VecIterNext\n            | VecGetRef\n            | VecGetMutRef\n    )\n}',
             repl='fn is_vec_runtime_fn(rt: RuntimeFn) -> bool {\n    use RuntimeFn::*;\n    matches!(\n        rt,\n        VecNew\n            | VecWithCapacity\n            | VecPush\n            | VecPop\n            | VecLen\n            | VecIsEmpty\n            | VecIndexGet\n            | VecReplace\n            | VecRemove\n            | VecIterNew\n            | VecIterNext\n            | VecGetRef\n            | VecGetMutRef\n    )\n}',
             tests=["--test", "mir_verify", "--test", "mir_differential", "--test", "three_engine_differential"],
             note="ONE-SIDED: only implementation B (verifier) is disturbed, so the other copy still "
                  "holds the correct answer. A KILL means the independent redundancy is doing real "
                  "work and the pair should be KEPT; a SURVIVOR means this copy can drift silently "
                  "and nothing in the tree notices."),
        dict(id="AS8-MUT-028", target="AS8-DA-003", tag="SHARED_AUTHORITY", expect="KILLED",
             authority="is_box_runtime — implementation A (interpreter)",
             file="src/mir/interp.rs",
             find='fn is_box_runtime(rt: RuntimeFn) -> bool {\n    use RuntimeFn::*;\n    matches!(rt, BoxNew | BoxIntoInner)\n}',
             repl='fn is_box_runtime(rt: RuntimeFn) -> bool {\n    use RuntimeFn::*;\n    matches!(rt, BoxNew)\n}',
             tests=["--test", "mir_differential", "--test", "three_engine_differential", "--test", "mir_verify"],
             note="ONE-SIDED: only implementation A (interpreter) is disturbed, so the other copy still "
                  "holds the correct answer. A KILL means the independent redundancy is doing real "
                  "work and the pair should be KEPT; a SURVIVOR means this copy can drift silently "
                  "and nothing in the tree notices."),
        dict(id="AS8-MUT-029", target="AS8-DA-003", tag="SHARED_AUTHORITY", expect="KILLED",
             authority="is_box_runtime_fn — implementation B (verifier)",
             file="src/mir/verify.rs",
             find='fn is_box_runtime_fn(rt: RuntimeFn) -> bool {\n    use RuntimeFn::*;\n    matches!(rt, BoxNew | BoxIntoInner)\n}',
             repl='fn is_box_runtime_fn(rt: RuntimeFn) -> bool {\n    use RuntimeFn::*;\n    matches!(rt, BoxNew)\n}',
             tests=["--test", "mir_verify", "--test", "mir_differential", "--test", "three_engine_differential"],
             note="ONE-SIDED: only implementation B (verifier) is disturbed, so the other copy still "
                  "holds the correct answer. A KILL means the independent redundancy is doing real "
                  "work and the pair should be KEPT; a SURVIVOR means this copy can drift silently "
                  "and nothing in the tree notices."),
        dict(id="AS8-MUT-030", target="AS8-DA-004", tag="SHARED_AUTHORITY", expect="KILLED",
             authority="is_slice_runtime — implementation A (interpreter)",
             file="src/mir/interp.rs",
             find='fn is_slice_runtime(rt: RuntimeFn) -> bool {\n    use RuntimeFn::*;\n    matches!(rt, SliceNew | SliceNewMut | SliceLen | SliceIsEmpty)\n}',
             repl='fn is_slice_runtime(rt: RuntimeFn) -> bool {\n    use RuntimeFn::*;\n    matches!(rt, SliceNew | SliceNewMut | SliceLen)\n}',
             tests=["--test", "mir_differential", "--test", "three_engine_differential", "--test", "mir_verify"],
             note="ONE-SIDED: only implementation A (interpreter) is disturbed, so the other copy still "
                  "holds the correct answer. A KILL means the independent redundancy is doing real "
                  "work and the pair should be KEPT; a SURVIVOR means this copy can drift silently "
                  "and nothing in the tree notices."),
        dict(id="AS8-MUT-031", target="AS8-DA-004", tag="SHARED_AUTHORITY", expect="KILLED",
             authority="is_slice_runtime_fn — implementation B (verifier)",
             file="src/mir/verify.rs",
             find='fn is_slice_runtime_fn(rt: RuntimeFn) -> bool {\n    use RuntimeFn::*;\n    matches!(rt, SliceNew | SliceNewMut | SliceLen | SliceIsEmpty)\n}',
             repl='fn is_slice_runtime_fn(rt: RuntimeFn) -> bool {\n    use RuntimeFn::*;\n    matches!(rt, SliceNew | SliceNewMut | SliceLen)\n}',
             tests=["--test", "mir_verify", "--test", "mir_differential", "--test", "three_engine_differential"],
             note="ONE-SIDED: only implementation B (verifier) is disturbed, so the other copy still "
                  "holds the correct answer. A KILL means the independent redundancy is doing real "
                  "work and the pair should be KEPT; a SURVIVOR means this copy can drift silently "
                  "and nothing in the tree notices."),
        dict(id="AS8-MUT-032", target="AS8-DA-005", tag="SHARED_AUTHORITY", expect="KILLED",
             authority="scalar_src — implementation A (provider_synth)",
             file="src/provider_synth.rs",
             find='fn scalar_src(s: ScalarTy) -> &\'static str {\n    match s {\n        ScalarTy::U8 => "UInt8",\n        ScalarTy::U16 => "UInt16",\n        ScalarTy::U32 => "UInt32",\n        ScalarTy::U64 => "UInt64",\n        ScalarTy::I8 => "Int8",\n        ScalarTy::I16 => "Int16",\n        ScalarTy::I32 => "Int32",\n        ScalarTy::I64 => "Int64",\n        ScalarTy::Bool => "Bool",\n        ScalarTy::F32 => "Float32",\n        ScalarTy::F64 => "Float64",\n    }\n}',
             repl='fn scalar_src(s: ScalarTy) -> &\'static str {\n    match s {\n        ScalarTy::U8 => "UInt16",\n        ScalarTy::U16 => "UInt16",\n        ScalarTy::U32 => "UInt32",\n        ScalarTy::U64 => "UInt64",\n        ScalarTy::I8 => "Int8",\n        ScalarTy::I16 => "Int16",\n        ScalarTy::I32 => "Int32",\n        ScalarTy::I64 => "Int64",\n        ScalarTy::Bool => "Bool",\n        ScalarTy::F32 => "Float32",\n        ScalarTy::F64 => "Float64",\n    }\n}',
             tests=["--test", "a10_provider_bind", "--test", "a10_provider_emit", "--test", "c788_starkc_build"],
             note="ONE-SIDED: only implementation A (provider_synth) is disturbed, so the other copy still "
                  "holds the correct answer. A KILL means the independent redundancy is doing real "
                  "work and the pair should be KEPT; a SURVIVOR means this copy can drift silently "
                  "and nothing in the tree notices."),
        dict(id="AS8-MUT-033", target="AS8-DA-005", tag="SHARED_AUTHORITY", expect="KILLED",
             authority="scalar_name — implementation B (provider_derive)",
             file="src/provider_derive.rs",
             find='fn scalar_name(s: ScalarTy) -> &\'static str {\n    match s {\n        ScalarTy::U8 => "UInt8",\n        ScalarTy::U16 => "UInt16",\n        ScalarTy::U32 => "UInt32",\n        ScalarTy::U64 => "UInt64",\n        ScalarTy::I8 => "Int8",\n        ScalarTy::I16 => "Int16",\n        ScalarTy::I32 => "Int32",\n        ScalarTy::I64 => "Int64",\n        ScalarTy::Bool => "Bool",\n        ScalarTy::F32 => "Float32",\n        ScalarTy::F64 => "Float64",\n    }\n}',
             repl='fn scalar_name(s: ScalarTy) -> &\'static str {\n    match s {\n        ScalarTy::U8 => "UInt16",\n        ScalarTy::U16 => "UInt16",\n        ScalarTy::U32 => "UInt32",\n        ScalarTy::U64 => "UInt64",\n        ScalarTy::I8 => "Int8",\n        ScalarTy::I16 => "Int16",\n        ScalarTy::I32 => "Int32",\n        ScalarTy::I64 => "Int64",\n        ScalarTy::Bool => "Bool",\n        ScalarTy::F32 => "Float32",\n        ScalarTy::F64 => "Float64",\n    }\n}',
             tests=["--test", "a10_provider_bind", "--test", "a10_provider_emit", "--test", "c788_starkc_build"],
             note="ONE-SIDED: only implementation B (provider_derive) is disturbed, so the other copy still "
                  "holds the correct answer. A KILL means the independent redundancy is doing real "
                  "work and the pair should be KEPT; a SURVIVOR means this copy can drift silently "
                  "and nothing in the tree notices."),
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
# THE GUARD IS NECESSARY AND NOT SUFFICIENT, and the gap is worth stating precisely.
# `finally` restores after a normal exit or an exception. It does NOT run when the process is
# KILLED — and on 2026-08-09 the DA batch was killed mid-trial and left AS8-MUT-030 applied to
# `mir/interp.rs` in the working tree. The pre-trial check below catches that on the NEXT run, which
# is the recovery path: if it refuses, run `git diff -- <file>` and `git restore` the mutation
# before doing anything else. Never commit while a batch is in flight.
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
        # Build ONLY the targets this trial runs. `--tests` builds all 209 integration binaries,
        # each of which links the whole starkc lib, and every trial invalidates the lib — so the
        # harness was relinking ~205 binaries it would never execute. On 2026-08-09 that filled the
        # disk to 99% and a single build stretched to 32 minutes before anyone noticed the cause
        # was space, not code. Building the selected targets keeps BUILD_FAILED a distinct outcome
        # while doing a fraction of the work.
        build = run(["cargo", "build", "--quiet", "-p", "starkc"] + spec["tests"])
        if build.returncode != 0:
            return dict(spec_id=spec["id"], result="BUILD_FAILED",
                        detail="the mutant does not compile; it is not a semantic mutation",
                        stderr=build.stderr[-800:])
        # `--no-fail-fast` is load-bearing for `killer_count`, not tidiness. Without it cargo stops
        # at the FIRST failing target, so a trial listing `--lib` first reported only the lib's
        # failures and never ran the dedicated suites. AC4-MUT-NS-002 read as "killed by 1 test",
        # which was nearly recorded as thin coverage for DEV-228 — while `dev228_namespaces` in
        # fact kills it with two failures, including its own `struct Pair` / `fn Pair()` case.
        #
        # KILLED/SURVIVED verdicts were never affected: a kill is a kill, and a SURVIVED trial ran
        # every target by definition. Only the COUNT was a lower bound, and a count that understates
        # coverage invites exactly the wrong conclusion about which authorities are watched.
        cmd = ["cargo", "test", "--quiet", "--no-fail-fast", "-p", "starkc"] + spec["tests"]
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
    ap.add_argument("--batch", required=True,
                    help='a batch key, or "select" with --only')
    ap.add_argument("--only",
                    help="comma-separated trial ids, selected ACROSS batches. Added for C10's "
                         "§8.2a re-runs: when a merge moves an authority or a control, the trials "
                         "needing re-running do not line up with AS8's batches, and duplicating "
                         "their definitions into a new batch would create a second copy that can "
                         "drift from the first")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--json", help="write the trial record here")
    a = ap.parse_args()
    if a.only:
        wanted = {t.strip() for t in a.only.split(",") if t.strip()}
        specs = [s for batch in BATCHES.values() for s in batch if s["id"] in wanted]
        missing = wanted - {s["id"] for s in specs}
        if missing:
            sys.exit(f"unknown trial id(s): {sorted(missing)}")
    else:
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
