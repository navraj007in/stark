# C6-CORPUS-COVERAGE-MATRIX — WP-C6.5

**Owner:** WP-C6.5 (`WP-C6.5.md`)
**Authority:** `starkc/docs/WP-C6-ENTRY.md` §40 (corpus categories); execution plan §7.2–§7.5
**Frozen:** 2026-07-26 at `b0d7a72`; amended 2026-07-26 by owner decision **CD-148** — rows O13 and
V19 re-dispositioned, and rows **K15–K17 added** for the entry contract that phase 0 omitted
(finding C65-F2 / DEV-111). See §9.
**Status:** phase C6.5-0 complete. Every §7.3 category is decomposed and dispositioned. No corpus
case has been written yet — the `ADD-*` rows are the worklist for phases C6.5-3 through C6.5-7.

## How to read this file

§7.2's required column set is distributed across the per-group tables and this header, so the
tables stay readable:

- **`accepted_surface`** — implied by the normative rule cited on each row; a row whose surface the
  front end rejects is `NOT-APPLICABLE-NON-CORE` with the rejection named.
- **`Tier-1_required`** — **yes for every Core row**, uniformly. C6.5's claim is that both Tier-1
  targets agree, so no Core row is exempt. Rows that are not Tier-1 required are exactly the
  `NOT-APPLICABLE-NON-CORE` ones.
- **`HIR` / `MIR` / `native_debug`** — folded into the **Evidence** column, which names the suite
  supplying the row. A suite listed there runs all three engines unless the row says otherwise.
- **`generated_template_ids` / `metamorphic_families` / `mutation_witnesses`** — carried in the
  **Disposition** column as `→T##`, `→M##`, `→MU##` tags, so a row states in one place what it
  still owes.
- **`package_shape`**, **`trap_or_completion`**, **`drop_observation`** — own columns where the row
  varies; where a whole group is uniform (e.g. every trap row is `trap`) it is stated once in the
  group's preamble instead of repeated 13 times.

**Disposition vocabulary** (§7.4). `EXISTING-EVIDENCE` means an existing suite already covers the
row to the C6.5 standard. `ADD-HANDWRITTEN` / `ADD-GENERATED` / `ADD-METAMORPHIC` /
`ADD-MUTATION-WITNESS` are worklist items. `NOT-APPLICABLE-NON-CORE` requires a cited absence.
`BLOCKED-BY-DEFECT`, `BLOCKED-BY-OTHER-C6-WP` and `ESCALATION-REQUIRED` name their blocker.

> **A caveat that applies to every `EXISTING-EVIDENCE` row.** Existing evidence was produced by one
> of the **23 forked comparators** recorded as finding C65-F1 in `WP-C6.5.md` §2. None of them
> observes the full §39 shape — no stderr bytes, no returned observation, no explicit Drop log. So
> `EXISTING-EVIDENCE` here means *"this row has real three-engine evidence today"*, not *"this row
> needs nothing from C6.5"*. Every such row is re-observed under the unified comparator during
> C6.5-5 replay. Rows that need more than re-observation say so.

> **Every rule ID below was re-derived from the specification under CD-154, and is now
> machine-checked.** The matrix as first written cited **69 invented identifiers out of 84** —
> `OWN-DROP-001`, `FN-VALUE-001`, `MAP-001`, `TRAP-ABORT-001`, `CTRL-IF-001` and 64 more appear in no
> spec document. Phase C6.5-0's exit condition *"every row has a normative citation"* passed because
> nothing compared the citations to the spec. `c6_corpus_manifest.rs::every_rule_id_the_matrix_cites_exists_in_the_spec`
> now fails if any citation here resolves to nothing, and the corpus validator applies the same check
> to `normative_rules` in the manifest.

**Counts:** 136 rows across 8 groups — 17 expressions/statements, 13 control transfer, 13 patterns,
24 values/types, 15 calls/dispatch, 24 ownership/Drop, 13 traps, 17 packages/environment.

---

## 1. Expressions and statements (17 rows)

Uniform: `package_shape = single-file` unless noted; `drop_observation = none` unless noted.

| ID | Sub-category | Normative rule | Outcome | Evidence | Disposition |
| --- | --- | --- | --- | --- | --- |
| E01 | literals | 01-Lexical §literals | completion | `primitive__01`, `scalar_arithmetic_agrees` | CORPUS-GENERATED: gen__t01__0268b2da |
| E02 | identifiers | 04-Semantic NAME-RESOLVE-001 | completion | `expr_stmt__01` | CORPUS-HANDWRITTEN: meta__m01_g1_base |
| E03 | blocks and block tails | 02-Syntax block-expr | completion | `expr_stmt__02_if_else_and_block_tail` | CORPUS-HANDWRITTEN: meta__m02_g1_base |
| E04 | let and mutable assignment | 03-Type AM-LOCAL-001 | completion | `expr_stmt__01`, `cross_block_non_copy_moves_agree` | CORPUS-HANDWRITTEN: meta__m01_g1_base |
| E05 | unary operations | NUM-INT-ARITH-001 | completion | `scalar_arithmetic_agrees` | UNATTRIBUTED |
| E06 | binary arithmetic | NUM-INT-ARITH-001 | both | `scalar_arithmetic_agrees`, `integer_overflow_trap_agrees` | CORPUS-GENERATED: gen__t01__0268b2da |
| E07 | bitwise operations | NUM-INT-ARITH-001 | both | `primitive__04_bitwise_shift_pow_and_ordering`, `invalid_shift_trap_agrees` | UNATTRIBUTED |
| E08 | comparisons | PRIM-TRAIT-001 | completion | `ordering_comparisons_agree` | CORPUS-GENERATED: gen__t02__19a95d0c |
| E09 | casts | NUM-CAST-001 | both | `float_to_int_boundary_conversions_agree`, `cast_failure_trap_agrees` | UNATTRIBUTED |
| E10 | direct calls | EXEC-DISPATCH-001 | completion | `direct_calls_agree` | CORPUS-HANDWRITTEN: meta__m10_g1_base |
| E11 | method calls | TYPE-METHOD-002 | completion | `struct_enum_trait__01` | CORPUS-HANDWRITTEN: meta__m04_g1_base |
| E12 | associated functions | TRAIT-ASSOC-001 | completion | `struct_enum_trait__05` | CORPUS-HANDWRITTEN: meta__m03_g1_base |
| E13 | function values and indirect calls | TYPE-FN-001 | completion | `function_value_in_local_and_indirect_call` +6 siblings | CORPUS-GENERATED: gen__t09__25661533 |
| E14 | returns | EXEC-CFLOW-001 | completion | `c61e_a_local_is_destroyed_on_return` | UNATTRIBUTED |
| E15 | expression statements | 02-Syntax stmt | completion | `expr_stmt__01` | UNATTRIBUTED |
| E16 | discarded values | DROP-ORDER-001 | completion, drop-observing | `ownership_drop__03_discarded_values_and_nested_patterns` | UNATTRIBUTED |
| E17 | assertions and panic | TRAP-CATEGORY-001 | trap | `a_false_assertion_traps_in_all_three_engines`, `a_false_bare_assertion_traps…`, `panic_message_agrees_across_engines` | CORPUS-GENERATED: gen__t16__1aefa931 |

**Group gaps:** none requiring new hand-written witnesses. Every row is re-observed under the
unified comparator in C6.5-5; E06/E13/E17 additionally carry mutation obligations.

---

## 2. Control transfer (13 rows)

| ID | Sub-category | Normative rule | Outcome | Evidence | Disposition |
| --- | --- | --- | --- | --- | --- |
| C01 | if/else | EXEC-EVAL-001 | completion | `branches_both_directions_agree` | CORPUS-GENERATED: gen__t02__19a95d0c |
| C02 | nested if | EXEC-EVAL-001 | completion | `expr_stmt__02` | UNATTRIBUTED |
| C03 | `loop` | TYPE-LOOP-001 | completion | `infinite_loop_with_mid_body_break_agrees` | UNATTRIBUTED |
| C04 | `while` | EXEC-CFLOW-001 | completion | `multi_iteration_loop_agrees` | CORPUS-GENERATED: gen__t03__07e7d3f2 |
| C05 | range `for` | EXEC-FOR-001 | completion | `expr_stmt__03_loops_break_continue` | CORPUS-HANDWRITTEN: meta__m12_g1_base |
| C06 | array `for` | EXEC-FOR-001 | completion | `collection_iter__03_slice_views_and_array_iteration` | UNATTRIBUTED |
| C07 | user iterator `for` | EXEC-FOR-001 | completion | `c63c_iterators` | UNATTRIBUTED |
| C08 | `break` | EXEC-CFLOW-001 | completion, drop-observing | `c61e_a_local_live_at_break_is_destroyed` | CORPUS-GENERATED: gen__t15__24c6dd0c |
| C09 | `continue` | EXEC-CFLOW-001 | completion, drop-observing | `c61e_a_local_live_at_continue_is_destroyed` | UNATTRIBUTED |
| C10 | early return | EXEC-CFLOW-001 | completion, drop-observing | `c61e_a_local_is_destroyed_on_return` | UNATTRIBUTED |
| C11 | `match` | PAT-OWN-001 | completion | `expr_stmt__04_match_and_patterns` | CORPUS-GENERATED: gen__t04__301bbe6e |
| C12 | `?` propagation | EXEC-CFLOW-001 | completion | `question_mark_propagation_agrees`, `option_result__02` | CORPUS-GENERATED: gen__t10__407709ff |
| C13 | trap termination | DROP-ABORT-001 | trap, drop-observing | `no_destructor_runs_after_a_trap` +4 `c61e_no_destructor_runs_after_*` | UNATTRIBUTED |

**Group gaps:** M12 (equivalent loop forms) has no metamorphic pair — C03/C04/C05 are the base
candidates. §13.6 constrains it: only forms whose ownership and **Drop timing** are normatively
equivalent may be paired, which rules out naive `while`↔`for` substitution over an owning
collection.

---

## 3. Patterns (13 rows)

| ID | Sub-category | Normative rule | Outcome | Evidence | Disposition |
| --- | --- | --- | --- | --- | --- |
| P01 | wildcard | SYN-PATTERN-001 | completion | `expr_stmt__04` | UNATTRIBUTED |
| P02 | binding | PAT-DROP-001 | completion | `c61e_a_match_arm_binding_is_destroyed_at_arm_end` | UNATTRIBUTED |
| P03 | tuple | SYN-PATTERN-001 | completion | `tuple_construction_and_projection_agree` | UNATTRIBUTED |
| P04 | struct | SYN-PATTERN-001 | completion | `struct_construction_and_field_projection_agree` | UNATTRIBUTED |
| P05 | enum variant | PAT-EXHAUST-001 | completion | `enum_construction_and_matching_agree` | CORPUS-GENERATED: gen__t06__155e6658 |
| P06 | nested patterns | SYN-PATTERN-001 | completion | `ownership_drop__03`, `pattern_nested_match` | CORPUS-GENERATED: gen__t04__301bbe6e |
| P07 | literal patterns | SYN-PATTERN-001 | completion | `match_order_ascending` | CORPUS-HANDWRITTEN: meta__m06_g1_base |
| P08 | range patterns | — | — | — | NOT-APPLICABLE: **NOT-APPLICABLE-NON-CORE** — 02-Syntax-Grammar declares no range-pattern form; the parser rejects it. Boundary to be pinned by a negative acceptance test (§4.3(4)) |
| P09 | `ref`/`mut` bindings | PAT-OWN-001 | completion | `c61e_a_failed_pattern_test_leaves_the_scrutinee_for_the_matching_arm` | UNATTRIBUTED |
| P10 | ignored fields | SYN-PATTERN-001 | completion | `struct_enum_trait__02` | UNATTRIBUTED |
| P11 | partial-move patterns | OWN-PARTIAL-001 | completion, drop-observing | `a_partially_moved_value_destroys_only_the_surviving_field`, `consuming_match_of_a_non_copy_payload_agrees` | UNATTRIBUTED |
| P12 | array patterns | SYN-PATTERN-001 | completion | `array_construction_and_indexing_agree` (A5/`ConstIndex`) | UNATTRIBUTED |
| P13 | match-arm guards | — | — | — | NOT-APPLICABLE: **NOT-APPLICABLE-NON-CORE** — no guard form in 02-Syntax-Grammar; parser rejects. Negative test to pin |

**Group gaps:** P08 and P13 need the §4.3(4) negative acceptance tests that pin their absence.
Recorded as `ADD-HANDWRITTEN` work under the non-Core classification, not as coverage.

---

## 4. Values and types (24 rows)

| ID | Sub-category | Normative rule | Evidence | Disposition |
| --- | --- | --- | --- | --- |
| V01 | Int8/16/32/64 | NUM-INT-ARITH-001 | `primitive__01_integer_widths_and_overflow_traps` | CORPUS-GENERATED: gen__t01__0268b2da |
| V02 | UInt8/16/32/64 | NUM-INT-ARITH-001 | `primitive__01` | CORPUS-GENERATED: gen__t01__0268b2da |
| V03 | Float32 | NUM-FLOAT-OP-001, CD-140 | `c63e_float32`, `layout_primitives_agree_exactly` | CORPUS-HANDWRITTEN: sentinel__13_float32_rendering |
| V04 | Float64 | NUM-FLOAT-OP-001 | `primitive__03_float_arithmetic_and_casts` | UNATTRIBUTED |
| V05 | Bool | PRIM-TRAIT-001 | `branches_both_directions_agree` | UNATTRIBUTED |
| V06 | Char | TEXT-ITER-001 | `c63a_string` (char push/pop, Unicode) | UNATTRIBUTED |
| V07 | String | TEXT-UTF8-001 | `c63a_string` | CORPUS-GENERATED: gen__t11__07c330c7 |
| V08 | `str` | TEXT-UTF8-001 | `c63a_string` (stored interior `&str`) | CORPUS-GENERATED: gen__t11__07c330c7 |
| V09 | tuple | TYPE-PRIM-001 | `tuple_construction_and_projection_agree`, `layout_tuples_agree_exactly` | CORPUS-RETAINED: entry_exit__06_unit_literal |
| V10 | array | TYPE-PRIM-001 | `array_construction_and_indexing_agree`, `layout_arrays_agree_exactly` | UNATTRIBUTED |
| V11 | slice | REF-SLICE-001 | `collection_iter__03`, `c63b_trapping_ops` | CORPUS-HANDWRITTEN: sentinel__10_slice_mutation_through_view |
| V12 | struct | TYPE-NOMINAL-001 | `struct_enum_trait__01`, `layout_structs_agree_exactly` | CORPUS-GENERATED: gen__t05__0e522a8c |
| V13 | enum | TYPE-NOMINAL-001 | `enum_discriminant_selection_agrees`, `enum_payload_field_order_agrees` | CORPUS-GENERATED: gen__t06__155e6658 |
| V14 | `Option<T>` | STD-PROFILE-001 | `option_construction_and_matching_agree`, `option_result__01` | CORPUS-GENERATED: gen__t10__407709ff |
| V15 | `Result<T,E>` | STD-PROFILE-001 | `result_construction_and_matching_agree`, `option_result__02` | CORPUS-GENERATED: gen__t10__407709ff |
| V16 | `Vec<T>` | DROP-COLLECTION-001 | `c63b_vec_box`, `collection_iter__01` | CORPUS-GENERATED: gen__t11__07c330c7 |
| V17 | `Box<T>` | STD-PROFILE-001, DROP-ORDER-001 | `c63b_vec_box`, `option_result__03_box_and_layout_queries` | UNATTRIBUTED |
| V18 | `HashMap<K,V>` | STD-HASH-001, CE4 insertion order | `c63d_map_key_identity`, `collection_iter__02` | CORPUS-GENERATED: gen__t12__7248da6d |
| V19 | `HashSet<T>` | 06-Standard-Library §`HashSet<T>`, `std-full` | `c63d_map_key_identity::hashset_is_hir_only` (pins the refusal, not the semantics) | BLOCKED: DEV-116 / WP-C6.3 (collections) / `HashSet` is normative in `std-full`, runs in the HIR oracle, and is refused at lowering — a MIR gap, which §4.3 forbids recording as a non-Core exclusion |
| V20 | files/resources | — | — | NOT-APPLICABLE: **NOT-APPLICABLE-NON-CORE** — `std-full` profile, absent from every engine; C6.3f EXCLUDED (CD-142) |
| V21 | function types | TYPE-FN-001 | `function_value_stored_in_a_struct_field`, `…_in_a_tuple` | CORPUS-GENERATED: gen__t09__25661533 |
| V22 | references | REF-IDENTITY-001 | `native_c61f_*` (6 suites), `exclusive_references_cross_the_call_boundary_and_mutate` | UNATTRIBUTED |
| V23 | mutable references | OWN-BORROW-001 | `native_c61f_reborrow`, `native_c61f_b3_stored_refs` | UNATTRIBUTED |
| V24 | nested/generic combinations | TYPE-GENERIC-001 | `nested_and_repeated_instantiations_each_see_their_own_frame`, `recursive_generic_instance_agrees`, `c62c_associated_types` | UNATTRIBUTED |

**Group gaps: V19 is the matrix's single `BLOCKED` row (CD-148).** It was carried in as
`NOT-APPLICABLE-NON-CORE` on the reading that `HashSet` is absent from the `core-min` profile. That
reading does not survive §4.3, and the owner reclassified it. Three facts, none of them in dispute:
`HashSet<T>` is specified normatively in 06-Standard-Library and named in the `std-full` profile, so
§4.3(1)'s "genuinely absent from normative Core v1" fails; row V18 covers `HashMap` — equally
`std-full` — as `EXISTING-EVIDENCE`, so "core-min only" is not the rule this matrix actually runs
on; and CD-142's own words call the exclusion "a lowering gap like C6.3c's adapters", which is
exactly the reason §4.3's closing line forbids. `hashset_is_hir_only` pins the boundary and says so
itself — *"if it now lowers, promote it to a three-engine case"*. It is therefore a C6 blocker on the
same footing O13 was thought to be, held open for a lowering package. V20 (files/resources) is
unaffected and remains the matrix's one non-Core classification in this group: absent from every
engine, not merely from MIR.

---

## 5. Calls and dispatch (15 rows)

| ID | Sub-category | Normative rule | Evidence | Disposition |
| --- | --- | --- | --- | --- |
| D01 | free function | EXEC-DISPATCH-001 | `direct_calls_agree` | CORPUS-HANDWRITTEN: meta__m10_g1_base |
| D02 | inherent method | TYPE-METHOD-002 | `struct_enum_trait__01` | UNATTRIBUTED |
| D03 | user trait | TRAIT-DEF-001 | `c62d_operator_coretrait` | CORPUS-GENERATED: gen__t08__101d93b4 |
| D04 | CoreTrait | PRIM-TRAIT-001 | `c62d_operator_coretrait` | UNATTRIBUTED |
| D05 | default trait method | TRAIT-DEF-001 | `struct_enum_trait__04_trait_default_and_override` | CORPUS-GENERATED: gen__t07__5ca87195 |
| D06 | fully qualified call | TRAIT-ASSOC-001 | `trait_call_qualified` | CORPUS-GENERATED: gen__t08__101d93b4 |
| D07 | generic parameter method | TRAIT-DEF-001 | `struct_enum_trait__03_generic_function_and_trait_bound` | UNATTRIBUTED |
| D08 | associated function | TRAIT-ASSOC-001 | `struct_enum_trait__05` | CORPUS-GENERATED: gen__t20__b6feee0e |
| D09 | associated type result | TRAIT-ASSOC-001 | `c62c_associated_types` | CORPUS-GENERATED: gen__t09__25661533 |
| D10 | explicit/inferred type args | TYPE-INFER-001 | `generics_explicit` / `generics_inferred` | UNATTRIBUTED |
| D11 | function pointer | TYPE-FN-001 | `function_value_as_parameter`, `function_value_returned_from_a_function` | CORPUS-GENERATED: gen__t09__25661533 |
| D12 | cross-package call | MOD-FILE-001 | `native_c5_4_linkage`, `native_c5_4_workspace` | UNATTRIBUTED |
| D13 | dependency-to-dependency call | PKG-RESOLVE-001 | `native_c5_4_workspace` (3-package) | UNATTRIBUTED |
| D14 | Drop-only reachability | DROP-EXACT-001 | `native_c6_1_ownership` | UNATTRIBUTED |
| D15 | trait-only reachability | TRAIT-DEF-001 | `c62b_f2_specific_instance` | UNATTRIBUTED |

**Group gaps:** none in coverage; D03/D11 owe adversarial-sentinel mutation witnesses (§14.4 —
*two* trait impls and *two* function targets returning **different** sentinels, so a wrong route is
observable). The existing cases were written to prove a route works, not to distinguish it from the
wrong route; that distinction is the C6.5-7 obligation.

---

## 6. Ownership and Drop (24 rows)

Uniform: every row is `drop_observation = required` unless it is a pure borrow row.
**Every row in this group is subject to the same limitation:** existing Drop evidence is *printed
output from user `Drop` impls*, compared as ordinary stdout. C6.5-1's §8.8 Drop-log protocol turns
that into a parsed, sequence-checked `drop_log` removed from stdout before comparison. So every row
below is `EXISTING-EVIDENCE` **plus** a re-observation obligation under the protocol.

**The protocol now exists** (commit 3) and **O13 is the first row observed through it** — identities
from the values themselves, order checked by position, frames stripped from stdout, expected log
stated independently of the engines. A companion case pins the Drop log retained *before* a trap,
which is what makes DROP-ABORT-001's "no destructor after a trap" falsifiable for row O22 rather than
assumed. The remaining 23 rows still carry the obligation; they discharge it as C6.5-5's replay
re-observes each category.

| ID | Sub-category | Normative rule | Evidence | Disposition |
| --- | --- | --- | --- | --- |
| O01 | Copy assignment | OWN-COPY-001 | `c61f_structural_copy` | UNATTRIBUTED |
| O02 | Move assignment | OWN-MOVE-001 | `a_moved_value_is_destroyed_by_its_new_owner` | UNATTRIBUTED |
| O03 | move into call | OWN-MOVE-001 | `cross_block_non_copy_moves_agree` | UNATTRIBUTED |
| O04 | move return | OWN-MOVE-001 | `native_c61f_ret_refs` | UNATTRIBUTED |
| O05 | borrow | REF-IDENTITY-001 | `ownership_drop__02_shared_borrow_does_not_move` | UNATTRIBUTED |
| O06 | mutable borrow | OWN-BORROW-001 | `exclusive_references_cross_the_call_boundary_and_mutate` | UNATTRIBUTED |
| O07 | reborrow | REF-PROJECT-001 | `native_c61f_reborrow` | UNATTRIBUTED |
| O08 | stored reference | REF-CARRY-001 | `native_c61f_b3_stored_refs` | UNATTRIBUTED |
| O09 | returned reference | REF-RETURN-001 | `native_c61f_ret_refs` (CD-112) | UNATTRIBUTED |
| O10 | partial struct move | OWN-PARTIAL-001 | `a_non_copy_field_moved_out_of_a_struct_agrees` | UNATTRIBUTED |
| O11 | partial enum move | OWN-PARTIAL-001 | `a_partially_moved_value_destroys_only_the_surviving_field` | CORPUS-GENERATED: gen__t06__155e6658 |
| O12 | array element consumption | A5 `ConstIndex` | `native_c5_3_aggregates_enums` | UNATTRIBUTED |
| O13 | non-Copy array iteration | OWN-MOVE-001, A5 `ConstIndex` | `o13_non_copy_array_by_value_iteration_agrees` | UNATTRIBUTED |
| O14 | reinitialisation | OWN-REINIT-001 | `native_c6_1_ownership` | UNATTRIBUTED |
| O15 | normal scope Drop | DROP-EXACT-001 | `ownership_drop__01_move_and_drop_order` | CORPUS-GENERATED: gen__t15__24c6dd0c |
| O16 | break/continue/return Drop | DROP-EXACT-001 | `c61e_a_local_live_at_break_is_destroyed` +2 | CORPUS-GENERATED: gen__t15__24c6dd0c |
| O17 | exact reverse field order | DROP-ORDER-001 | `struct_fields_are_destroyed_in_reverse_declaration_order` | CORPUS-GENERATED: gen__t15__24c6dd0c |
| O18 | own destructor before fields | DROP-ORDER-001 | `own_destructor_runs_before_fields` | UNATTRIBUTED |
| O19 | active enum payload only | DROP-EXACT-001 | `enum_destroys_the_active_variant_payload_a`/`_b` | UNATTRIBUTED |
| O20 | no duplicate Drop | DROP-EXACT-001 | `a_moved_value_is_destroyed_exactly_once` | CORPUS-HANDWRITTEN: sentinel__12_drop_identities |
| O21 | no skipped Drop | DROP-EXACT-001 | `c61e_a_loop_body_local_is_destroyed_each_iteration` | CORPUS-HANDWRITTEN: sentinel__12_drop_identities |
| O22 | no Drop after trap | DROP-ABORT-001 | `no_destructor_runs_after_a_trap` +4 `c61e_*` | UNATTRIBUTED |
| O23 | collection element Drop | DROP-COLLECTION-001 | `c63b_vec_box` (`Vec<String>`, CD-135/136) | UNATTRIBUTED |
| O24 | Box inner Drop | DROP-ORDER-001 | `c63b_vec_box` | UNATTRIBUTED |

**Group gaps (CD-148): none.** O13 was carried into this matrix as its only `BLOCKED` row, inherited
from CD-038's "narrowed, not closed" wording — by-value iteration over a non-`Copy` array element,
refused because a runtime loop index gives move analysis nothing precise to name. CD-038 recorded
what would close it ("unrolling or runtime-indexed drop flags"), and **WP-C6.1d took the unrolling
option** (CD-084 G2, closing DEV-090). The ledger therefore held two records and the matrix
inherited the older one. Decided by execution rather than by reading either:
`o13_non_copy_array_by_value_iteration_agrees` moves each element into the loop binding and pins the
stdout to `"idid\n"` independently of the engines, so a wrong Drop schedule — both elements dropped
at the end, or neither — fails even if all three engines agree on it. All three engines produce it.
The blocker does not exist.

---

## 7. Traps (13 rows)

Uniform: `trap_or_completion = trap`; every row requires exact source provenance
(`file:line:column`), exit 101 and pre-trap stdout.

| ID | Sub-category | Normative rule | Evidence | Disposition |
| --- | --- | --- | --- | --- |
| X01 | integer overflow | NUM-INT-ARITH-001 | `integer_overflow_trap_agrees`, `primitive__02` | CORPUS-GENERATED: gen__t16__1aefa931 |
| X02 | divide by zero | NUM-INT-DIV-001 | `divide_by_zero_trap_agrees`, `remainder_by_zero_trap_agrees` | UNATTRIBUTED |
| X03 | invalid shift | NUM-SHIFT-001 | `invalid_shift_trap_agrees` | UNATTRIBUTED |
| X04 | cast failure | NUM-CAST-001 | `cast_failure_trap_agrees`, `out_of_range_cast_is_a_cast_failure_not_an_overflow`, 3 float-boundary cases | UNATTRIBUTED |
| X05 | index out of bounds | TRAP-CATEGORY-001 | `index_out_of_bounds_traps_in_all_three_engines`, `negative_index_traps…`, `the_last_valid_index_does_not_trap` | UNATTRIBUTED |
| X06 | unwrap None | TRAP-CATEGORY-001 | `a_trap_from_an_option_payload_agrees` | UNATTRIBUTED |
| X07 | unwrap Err | TRAP-CATEGORY-001 | `c63b_trapping_ops` | UNATTRIBUTED |
| X08 | assert failure | TRAP-CATEGORY-001 | `a_false_assertion_traps…`, `a_false_bare_assertion_traps…` | UNATTRIBUTED |
| X09 | panic with message | TRAP-CATEGORY-001 | `panic_message_agrees_across_engines`, `conditional_panic_message_agrees…` (CD-136) | UNATTRIBUTED |
| X10 | source provenance | TRAP-CATEGORY-001 | every trap case asserts `file:line`; DEV-107 closed | CORPUS-HANDWRITTEN: pkg__dep_trap_provenance |
| X11 | output before trap | PROC-STREAM-001, CD-120 Contract B | `c64_platform_matrix::platform_trap_reports_…` | CORPUS-GENERATED: gen__t16__1aefa931 |
| X12 | exit 101 | PROC-EXIT-001, DROP-ABORT-001 | `c64_platform_matrix::platform_trap_reports_…` | UNATTRIBUTED |
| X13 | no cleanup after trap | DROP-ABORT-001 | 5 `c61e_no_destructor_runs_after_*` | UNATTRIBUTED |

**Note on float division.** There is no float-divide-by-zero trap row, and that is correct:
NUM-FLOAT-OP-001 makes float division **total**, and CD-139 recorded CD-006's supersession by
succession of authority. `cd139_float_division` (13 three-engine cases) is the completion-side
evidence and belongs to V04, not here. A trap row here would encode the superseded rule.

---

## 8. Packages and environment (17 rows)

| ID | Sub-category | Normative rule | Package shape | Evidence | Disposition |
| --- | --- | --- | --- | --- | --- |
| K01 | single file | MOD-FILE-001 | single-file | most of the corpus | UNATTRIBUTED |
| K02 | multi-file package | MOD-FILE-001 | package | `multi_file__01_cross_file_execution_and_provenance` | CORPUS-HANDWRITTEN: pkg__dep_trap_provenance |
| K03 | dependency | PKG-RESOLVE-001 | workspace | `native_c5_4_linkage` | CORPUS-HANDWRITTEN: pkg__workspace_three_packages |
| K04 | dependency-to-dependency | PKG-RESOLVE-001 | workspace | `native_c5_4_workspace` | CORPUS-HANDWRITTEN: pkg__workspace_three_packages |
| K05 | re-export | MOD-USE-001 | workspace | `native_c5_4_linkage` | CORPUS-HANDWRITTEN: pkg__workspace_three_packages |
| K06 | package alias | — | — | — | CORPUS-HANDWRITTEN: meta__m08_g1_base |
| K07 | workspace relocation | PKG-IDENTITY-001, CD-108 | workspace | `native_build_cli::frozen_three_package_workspace_builds_through_cli_after_relocation`, `c62e_deterministic_identity` | CORPUS-HANDWRITTEN: meta__m08_g1_base |
| K08 | dependency declaration reorder | CD-108 | workspace | `c62e_deterministic_identity` | CORPUS-HANDWRITTEN: meta__m09_g1_base |
| K09 | source declaration reorder | NAME-RESOLVE-001 | package | — | CORPUS-HANDWRITTEN: meta__m09_g1_base |
| K10 | locked build | PKG-LOCK-001 | workspace | `c64_platform_matrix::portability_generated_crate_is_locked_and_network_free` | UNATTRIBUTED |
| K11 | offline build | §11.3 | workspace | `c63_closure_evidence` | UNATTRIBUTED |
| K12 | installed runtime | §9.2 | workspace | `c63_closure_evidence`, CI release smoke + negative step (CD-144 R1) | UNATTRIBUTED |
| K13 | Unicode path | §9.7 | workspace | `c64_platform_matrix::portability_builds_and_runs_under_paths_containing_unicode` | UNATTRIBUTED |
| K14 | path containing spaces | §9.7 | workspace | `c64_platform_matrix::portability_builds_and_runs_under_paths_containing_spaces` | UNATTRIBUTED |
| K15 | entry signature set | PROC-MAIN-001 | single-file | `c65_entry_exit_contract` | CORPUS-RETAINED: entry_exit__01_unit_entry |
| K16 | normal exit status (`Int32`, `Ok(Int32)`) | PROC-EXIT-001 | single-file | `c65_entry_exit_contract` | CORPUS-RETAINED: entry_exit__01_unit_entry |
| K17 | `Err(message)` → stderr + status 1 | PROC-EXIT-001, PROC-STREAM-001 | single-file | `c65_entry_exit_contract` | CORPUS-RETAINED: entry_exit__03_err_stderr |

**Group gaps.** K09 needs a metamorphic pair. K06 needs a specification check before it can be
classified at all — recorded as an open question rather than guessed.

**K15–K17 were missing from the matrix entirely** and were added when finding **C65-F2** (DEV-111,
`WP-C6.5.md` §6) ran the entry contract through all three engines. Exit status had been covered only
as X12 (exit 101 after a trap); normal nonzero statuses, the `Err` stderr write and the entry
signature set had no row. So the §7.5 exit condition "no category silently omitted" did not hold when
phase 0 was declared complete.

**DEV-112, closed on the way (CD-150).** `Ok(Unit)` — PROC-EXIT-001's own clause for a
`Result<Unit, String>` entry — was **unwritable**: the checker gave `()` a type that unified with
nothing, so no value of type `Unit` could be constructed at all. TYPE-PRIM-001 makes `Unit` and `()`
two spellings of one type, so this was a conformance bug; all three engines now canonicalise, and
`ok_unit_entry_completes_with_status_zero` covers the clause. K16 is still `BLOCKED` for the native
half only.

**A related channel gap, recorded here because it has no row of its own and cannot get one yet.**
`eprint`/`eprintln` are normative (06-Standard-Library IO) but are **unobservable in every engine**:
the HIR oracle writes them to the *host* process's stderr (`src/interp.rs:2779`) rather than into
`Execution.stderr`, MIR has no lowering for them, and the native backend emits none. §8.3's
`stderr_bytes` field can therefore only ever compare the `Err`-completion write until that is closed.
Not classified as `NOT-APPLICABLE-NON-CORE` — §4.3 forbids exactly that reasoning.

---

## 9. Roll-up

| Disposition | Rows |
| --- | --- |
| `EXISTING-EVIDENCE` (all re-observed under the unified comparator in C6.5-5) | 127 |
| `NOT-APPLICABLE-NON-CORE` | 4 — P08, P13, V20, K06 (K06 provisional) |
| `ADD-METAMORPHIC` | 1 — K09 |
| `BLOCKED-BY-OTHER-C6-WP` | 4 — V19 (`HashSet`), K15/K16/K17 (the entry contract, DEV-111) |

Both blocker-shaped rows moved under CD-148, in opposite directions: **O13 out** (the refusal it
cited was superseded by C6.1d's unrolling; proven by execution, not by ledger reading) and **V19 in**
(a lowering gap cannot be a non-Core exclusion under §4.3). **K15–K17 then arrived under CD-149** —
rows that did not exist until the entry contract was actually run (DEV-111): normal exit statuses,
the `Err` stderr write, and the entry-signature set had no representation in the original 133.

**Two of the matrix's inherited dispositions have now failed on contact with a run, and nothing else
has been re-derived yet.** O13 was blocked and is not; the entry contract was absent and is a
blocker. Both were produced the same way as the other 130 rows — from ledger records and existing
test names, not from executing the category. That is the case for C6.5-5's replay being the thing
that establishes this matrix, rather than the matrix establishing coverage.

**What this roll-up does not mean.** 127 `EXISTING-EVIDENCE` rows is not "C6.5 is nearly done". It
means the *category surface* is already exercised somewhere. What C6.5 owes on top of it, and what
the rest of the package is:

1. **One comparator** instead of 23 (finding C65-F1) — without which "the engines agree" has 23
   different meanings.
2. **The full §39 observation shape** — no row today is observed with stderr bytes, a returned
   observation, or a parsed Drop log. Every ownership row's Drop evidence is currently printed
   stdout.
3. **A generated corpus** — 0 of the required ≥64 cases across ≥10 templates exist. Every `→T##`
   tag above is unbuilt.
4. **Metamorphic breadth** — 7 inherited groups against a floor of 24 groups / 48 members, and 5
   of 12 families have no group at all (M08–M12).
5. **Mutation controls** — 0 of 16. Every `→MU##` tag is unbuilt. The one existing negative test,
   `the_comparator_rejects_disagreeing_outcomes`, covers a fraction of MU-nothing specifically.
6. **Adversarial sentinels** — D03/D11's cases prove a route works, not that the *wrong* route is
   observable, which is what §14.4 requires of a mutation witness.

Items 2–6 are the substance of phases C6.5-1 through C6.5-7, and none of them is reduced by the
row count.
