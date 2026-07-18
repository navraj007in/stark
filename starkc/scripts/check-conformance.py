#!/usr/bin/env python3
import re
import sys
import os
import tomllib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from conformance_lib import parse_toml  # noqa: E402

DEVIATION_ID_RE = re.compile(r'^DEV-\d+$')
GRANULAR_ROW_RE = re.compile(r'^\| ([A-Z][A-Z0-9-]*-\d{3}) \|', re.MULTILINE)
C2_8_RULE_IDS = {
    'SYN-PLACE-001', 'SYN-PATTERN-001',
    'NAME-SCOPE-001', 'NAME-SHADOW-001', 'NAME-RESOLVE-001',
    'TYPE-PRIM-001', 'TYPE-NOMINAL-001', 'TYPE-ALIAS-001', 'TYPE-WF-001',
    'TYPE-INFER-001', 'TYPE-GENERIC-001', 'TYPE-COERCE-003', 'TYPE-LOOP-001',
    'TYPE-METHOD-001', 'TYPE-METHOD-002',
    'TRAIT-DEF-001', 'TRAIT-ASSOC-001', 'TRAIT-COHERENCE-001',
    'TRAIT-COHERENCE-002', 'TRAIT-LAW-001',
    'FLOW-LOOP-001',
    'OWN-COPY-001', 'OWN-BORROW-001', 'OWN-REGION-001', 'OWN-RETURN-001',
    'OWN-CARRY-001',
    'PAT-EXHAUST-001', 'PAT-USEFUL-001',
    'CONST-DECL-001', 'CONST-SUBSET-001', 'CONST-FAIL-001',
    'STD-HOOK-001', 'STD-TRAIT-001',
}
C2_9_RULE_IDS = {
    'LEX-SOURCE-001', 'LEX-IDENT-002', 'LEX-ESCAPE-001', 'SYN-RECOVERY-001',
    'TYPE-CAST-001', 'FLOW-BOUNDS-001',
    'NUM-INT-ARITH-001', 'NUM-INT-DIV-001', 'NUM-SHIFT-001', 'NUM-CAST-001',
    'NUM-FLOAT-FORMAT-001', 'NUM-FLOAT-OP-001', 'NUM-FLOAT-TRAIT-001',
    'NUM-FLOAT-REPRO-001',
    'TEXT-UTF8-001', 'TEXT-INDEX-001', 'TEXT-BOUNDARY-001', 'TEXT-ITER-001',
    'TEXT-CASE-001',
    'MOD-FILE-001', 'MOD-PATH-001', 'MOD-USE-001', 'MOD-VIS-001',
    'MOD-REEXPORT-001', 'MOD-CYCLE-001',
    'PKG-MANIFEST-001', 'PKG-RESOLVE-001', 'PKG-VERSION-001',
    'PKG-IDENTITY-001', 'PKG-MULTIVER-001', 'PKG-LOCK-001',
    'PROC-MAIN-001', 'PROC-EXIT-001', 'PROC-STREAM-001',
    'STD-FORMAT-001', 'STD-PROFILE-001', 'STD-HASH-001', 'STD-IO-001',
    'STD-CONVERT-001', 'STD-MATH-001', 'STD-RANDOM-001',
    'LAYOUT-QUERY-001', 'LAYOUT-ABI-001', 'TRAP-CATEGORY-001',
    'LIMIT-RESOURCE-001', 'LIMIT-COMPILER-001',
}
C2_10_RULE_IDS = {
    'LEX-RESERVED-001', 'FUTURE-SYNTAX-001', 'FUTURE-CLOSURE-001',
    'FUTURE-THREAD-001', 'FUTURE-FFI-001', 'EXT-ISOLATION-001',
}


def validate_evidence_entry(entry, conformance_dir, rule_id, field_name, errors):
    """WP-C1.6: validate one `positive_tests`/`negative_tests` citation. An entry is either a
    bare file path (existence-checked only, same as the legacy `tests` field) or
    `path::function_name` (existence-checked, plus a `fn function_name` grep so a renamed or
    deleted test function is caught, not just a renamed or deleted file)."""
    path_part, _, fn_part = entry.partition('::')
    full_path = os.path.join(conformance_dir, path_part)
    if not os.path.exists(full_path):
        errors.append(f"Rule {rule_id} {field_name} path '{path_part}' does not exist.")
        return
    if fn_part:
        with open(full_path, 'r', encoding='utf-8') as f:
            content = f.read()
        if f'fn {fn_part}' not in content:
            errors.append(
                f"Rule {rule_id} {field_name} cites '{entry}' but no `fn {fn_part}` was found "
                f"in '{path_part}' -- the function may have been renamed or removed."
            )


def main():
    conformance_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    toml_path = os.path.join(conformance_dir, 'STARKLANG', 'conformance', 'core-v1-coverage.toml')
    
    if not os.path.exists(toml_path):
        print(f"Error: Coverage database not found at {toml_path}", file=sys.stderr)
        sys.exit(1)
        
    with open(toml_path, 'r', encoding='utf-8') as f:
        content = f.read()
        
    rules = parse_toml(content)

    valid_statuses = {'implemented', 'partial', 'missing', 'intentionally-deferred', 'spec-defect'}
    errors = []
    warnings = []

    # WP-C2.6: validate the transitional split from every broad legacy rule to stable granular
    # semantic-freeze inventory IDs. The map deliberately does not copy implementation status;
    # C2.11 must attach evidence and status at granular resolution.
    map_path = os.path.join(
        conformance_dir, 'STARKLANG', 'conformance', 'core-v1-rule-id-map.toml'
    )
    inventory_path = os.path.join(
        conformance_dir,
        'STARKLANG',
        'docs',
        'compiler',
        'semantic-freeze',
        'CORE-V1-COMPLETENESS.md',
    )
    if not os.path.exists(map_path):
        errors.append(f"Granular rule map not found at {map_path}.")
    if not os.path.exists(inventory_path):
        errors.append(f"Completeness inventory not found at {inventory_path}.")
    if os.path.exists(map_path) and os.path.exists(inventory_path):
        with open(map_path, 'rb') as f:
            split_map = tomllib.load(f)
        with open(inventory_path, 'r', encoding='utf-8') as f:
            inventory_content = f.read()

        inventory_ids = GRANULAR_ROW_RE.findall(inventory_content)
        inventory_id_set = set(inventory_ids)
        if len(inventory_ids) != len(inventory_id_set):
            duplicates = sorted(
                rule_id for rule_id in inventory_id_set if inventory_ids.count(rule_id) > 1
            )
            errors.append(f"Duplicate granular inventory IDs: {', '.join(duplicates)}.")

        abstract_machine_path = os.path.join(
            conformance_dir,
            'STARKLANG',
            'docs',
            'spec',
            'CORE-V1-ABSTRACT-MACHINE.md',
        )
        if not os.path.exists(abstract_machine_path):
            errors.append(f"Normative abstract machine not found at {abstract_machine_path}.")
        else:
            with open(abstract_machine_path, 'r', encoding='utf-8') as f:
                abstract_machine_content = f.read()
            abstract_ids = re.findall(
                r'\*\*([A-Z][A-Z0-9-]*-\d{3})\.\*\*', abstract_machine_content
            )
            duplicate_abstract_ids = sorted(
                rule_id for rule_id in set(abstract_ids) if abstract_ids.count(rule_id) > 1
            )
            if duplicate_abstract_ids:
                errors.append(
                    "Duplicate abstract-machine rule IDs: "
                    + ', '.join(duplicate_abstract_ids)
                    + "."
                )
            missing_inventory_ids = sorted(set(abstract_ids) - inventory_id_set)
            if missing_inventory_ids:
                errors.append(
                    "Abstract-machine rules missing from the completeness inventory: "
                    + ', '.join(missing_inventory_ids)
                    + "."
                )

        # WP-C2.8: every approved static-semantics rule must occur exactly once in the
        # normative source set and remain marked complete in the granular inventory.
        spec_dir = os.path.join(conformance_dir, 'STARKLANG', 'docs', 'spec')
        normative_id_counts = {}
        for filename in sorted(os.listdir(spec_dir)):
            if not filename.endswith('.md') or filename == 'STARK-Core-v1.md':
                continue
            with open(os.path.join(spec_dir, filename), 'r', encoding='utf-8') as f:
                source_content = f.read()
            for rule_id in re.findall(
                r'\*\*([A-Z][A-Z0-9-]*-\d{3})\.\*\*', source_content
            ):
                normative_id_counts[rule_id] = normative_id_counts.get(rule_id, 0) + 1

        for rule_id in sorted(C2_8_RULE_IDS):
            count = normative_id_counts.get(rule_id, 0)
            if count != 1:
                errors.append(
                    f"C2.8 rule {rule_id} occurs {count} times in normative sources; expected 1."
                )
            row_match = re.search(
                rf'^\| {re.escape(rule_id)} \|.*\| complete/', inventory_content, re.MULTILINE
            )
            if row_match is None:
                errors.append(
                    f"C2.8 rule {rule_id} is not marked complete in the completeness inventory."
                )

        for rule_id in sorted(C2_9_RULE_IDS):
            count = normative_id_counts.get(rule_id, 0)
            if count != 1:
                errors.append(
                    f"C2.9 rule {rule_id} occurs {count} times in normative sources; expected 1."
                )
            row_match = re.search(
                rf'^\| {re.escape(rule_id)} \|.*\| complete/', inventory_content, re.MULTILINE
            )
            if row_match is None:
                errors.append(
                    f"C2.9 rule {rule_id} is not marked complete in the completeness inventory."
                )

        for rule_id in sorted(C2_10_RULE_IDS):
            count = normative_id_counts.get(rule_id, 0)
            if count != 1:
                errors.append(
                    f"C2.10 rule {rule_id} occurs {count} times in normative sources; expected 1."
                )
            row_match = re.search(
                rf'^\| {re.escape(rule_id)} \|.*\| complete/', inventory_content, re.MULTILINE
            )
            if row_match is None:
                errors.append(
                    f"C2.10 rule {rule_id} is not marked complete in the completeness inventory."
                )

        legacy_entries = split_map.get('legacy', [])
        mapped_legacy_ids = [entry.get('id') for entry in legacy_entries]
        coverage_ids = [rule.get('id') for rule in rules]
        for legacy_id in sorted(set(mapped_legacy_ids)):
            if mapped_legacy_ids.count(legacy_id) > 1:
                errors.append(f"Legacy rule '{legacy_id}' occurs more than once in split map.")
        missing_legacy = sorted(set(coverage_ids) - set(mapped_legacy_ids))
        unknown_legacy = sorted(set(mapped_legacy_ids) - set(coverage_ids))
        if missing_legacy:
            errors.append(f"Legacy rules missing from split map: {', '.join(missing_legacy)}.")
        if unknown_legacy:
            errors.append(f"Unknown legacy rules in split map: {', '.join(unknown_legacy)}.")

        for entry in legacy_entries:
            legacy_id = entry.get('id', '<missing>')
            granular = entry.get('granular')
            if not isinstance(granular, list) or not granular:
                errors.append(f"Legacy rule {legacy_id} has no granular ID list.")
                continue
            unknown_granular = sorted(set(granular) - inventory_id_set)
            if unknown_granular:
                errors.append(
                    f"Legacy rule {legacy_id} maps to unknown granular IDs: "
                    f"{', '.join(unknown_granular)}."
                )

        questions_path = os.path.join(
            conformance_dir,
            'STARKLANG',
            'docs',
            'compiler',
            'semantic-freeze',
            'CORE-V1-OPEN-QUESTIONS.md',
        )
        deviations_path = os.path.join(
            conformance_dir, 'starkc', 'docs', 'conformance', 'KNOWN-DEVIATIONS.md'
        )
        if not os.path.exists(questions_path):
            errors.append(f"Open-question register not found at {questions_path}.")
        else:
            with open(questions_path, 'r', encoding='utf-8') as f:
                question_content = f.read()
            known_questions = set(re.findall(r'CORE-Q-(\d{3}[A-Z]?)', question_content))
            used_questions = set(re.findall(r'\bQ(\d{3}[A-Z]?)\b', inventory_content))
            unknown_questions = sorted(used_questions - known_questions)
            if unknown_questions:
                errors.append(
                    "Completeness inventory references unknown questions: "
                    + ', '.join(f"CORE-Q-{qid}" for qid in unknown_questions)
                    + "."
                )

        if not os.path.exists(deviations_path):
            errors.append(f"Known-deviation ledger not found at {deviations_path}.")
        else:
            with open(deviations_path, 'r', encoding='utf-8') as f:
                deviation_content = f.read()
            known_deviations = set(re.findall(r'^## (DEV-\d+)\b', deviation_content, re.MULTILINE))
            used_deviations = set(re.findall(r'\bDEV-\d+\b', inventory_content))
            unknown_deviations = sorted(used_deviations - known_deviations)
            if unknown_deviations:
                errors.append(
                    "Completeness inventory references unknown deviations: "
                    + ', '.join(unknown_deviations)
                    + "."
                )

    # WP-C0.3: valid chapter identifiers are derived from the real spec directory, not a
    # hardcoded list, so a renamed/removed spec file is caught automatically.
    spec_dir = os.path.join(conformance_dir, 'STARKLANG', 'docs', 'spec')
    valid_chapters = set()
    if os.path.isdir(spec_dir):
        for fname in os.listdir(spec_dir):
            if fname.endswith('.md') and fname not in ('STARK-Core-v1.md',):
                valid_chapters.add(fname[:-3])

    # WP-C0.3: duplicate rule ID detection.
    seen_ids = {}
    for rule in rules:
        rid = rule.get('id')
        if rid:
            seen_ids.setdefault(rid, 0)
            seen_ids[rid] += 1
    for rid, count in seen_ids.items():
        if count > 1:
            errors.append(f"Duplicate rule id '{rid}' appears {count} times.")

    # Heuristic keywords suggesting a rule is a semantic *rejection* rule (checks that a
    # malformed program is correctly refused) and therefore needs negative-test evidence, not
    # just positive-test evidence. This is a heuristic over free-text `description` fields, not
    # a structural distinction the schema currently makes (the schema has one `tests` array with
    # no positive/negative tag) -- flagged as a WP-C0.3 known limitation, see the printed report.
    rejection_keywords = ('check', 'error', 'prohibit', 'reject', 'invalid', 'exhaustiveness',
                           'coherence', 'orphan', 'overlap', 'boundary', 'dangling')

    # Validation
    for rule in rules:
        rule_id = rule.get('id')
        if not rule_id:
            errors.append("Rule found without 'id' attribute.")
            continue

        chapter = rule.get('chapter')
        if 'chapter' not in rule:
            errors.append(f"Rule {rule_id} is missing 'chapter' attribute.")
        elif valid_chapters and chapter not in valid_chapters:
            errors.append(
                f"Rule {rule_id} references nonexistent spec chapter '{chapter}' "
                f"(no STARKLANG/docs/spec/{chapter}.md)."
            )
        if 'description' not in rule:
            errors.append(f"Rule {rule_id} is missing 'description' attribute.")
        if 'status' not in rule:
            errors.append(f"Rule {rule_id} is missing 'status' attribute.")
            continue

        status = rule['status']
        if status not in valid_statuses:
            errors.append(f"Rule {rule_id} has invalid status '{status}'; expected one of {valid_statuses}")

        deviation = rule.get('deviation')
        if deviation is not None and not DEVIATION_ID_RE.match(deviation):
            errors.append(
                f"Rule {rule_id} 'deviation' value '{deviation}' does not look like a DEV-NNN id."
            )

        if status == 'implemented':
            source = rule.get('source')
            if not source:
                errors.append(f"Rule {rule_id} has status 'implemented' but is missing 'source' path.")
            else:
                source_path = os.path.join(conformance_dir, source)
                if not os.path.exists(source_path):
                    errors.append(f"Rule {rule_id} source path '{source}' does not exist.")

            tests = rule.get('tests')
            if not tests:
                errors.append(f"Rule {rule_id} has status 'implemented' but is missing 'tests' paths.")
            elif not isinstance(tests, list):
                errors.append(f"Rule {rule_id} 'tests' attribute must be an array of paths.")
            else:
                for test_file in tests:
                    test_path = os.path.join(conformance_dir, test_file)
                    if not os.path.exists(test_path):
                        errors.append(f"Rule {rule_id} test path '{test_file}' does not exist.")

            description = (rule.get('description') or '').lower()
            if any(kw in description for kw in rejection_keywords) and not tests:
                warnings.append(
                    f"Rule {rule_id} description suggests a semantic-rejection rule but has no "
                    f"tests recorded -- verify negative-test (invalid-program) coverage exists, "
                    f"not just positive-test coverage (heuristic, description-keyword based)."
                )

            # WP-C1.6: optional, more precise evidence fields. When present, every citation is
            # validated the same way `tests` is (path existence), plus a function-name check for
            # `path::function` entries. Absent is not an error -- DEV-017 confirmed most rules
            # don't have this precision yet; the report generator surfaces that gap honestly
            # rather than this validator inventing a requirement the schema doesn't mandate yet.
            for field_name in ('positive_tests', 'negative_tests'):
                field = rule.get(field_name)
                if field is None:
                    continue
                if not isinstance(field, list):
                    errors.append(f"Rule {rule_id} '{field_name}' attribute must be an array.")
                    continue
                for entry in field:
                    validate_evidence_entry(entry, conformance_dir, rule_id, field_name, errors)

            if any(kw in description for kw in rejection_keywords) and rule.get(
                'negative_tests'
            ) == []:
                warnings.append(
                    f"Rule {rule_id} description suggests a semantic-rejection rule but "
                    f"'negative_tests' is explicitly empty -- confirm this rule genuinely has no "
                    f"negative-test evidence yet, not an oversight."
                )

        if status == 'missing' and (rule.get('source') or rule.get('tests')):
            warnings.append(
                f"Rule {rule_id} is marked 'missing' but has a 'source' or 'tests' field set -- "
                f"likely stale; a 'missing' rule should have neither."
            )

    if warnings:
        print("Conformance Baseline Warnings (non-fatal):", file=sys.stderr)
        for warning in warnings:
            print(f"  - {warning}", file=sys.stderr)
        print(file=sys.stderr)

    if errors:
        print("Conformance Baseline Verification Failed:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        sys.exit(1)
        
    # Group statistics by chapter
    stats = {}
    total_rules = 0
    total_implemented = 0
    
    for rule in rules:
        chapter = rule.get('chapter', 'Unknown')
        status = rule.get('status', 'missing')
        
        if chapter not in stats:
            stats[chapter] = {s: 0 for s in valid_statuses}
            stats[chapter]['total'] = 0
            
        stats[chapter][status] += 1
        stats[chapter]['total'] += 1
        total_rules += 1
        if status == 'implemented':
            total_implemented += 1
            
    print("=" * 80)
    print(f"{'STARK Core v1 Conformance Report':^80}")
    print("=" * 80)
    print(f"{'Chapter':<30} | {'Total':<6} | {'Impl':<6} | {'Partial':<7} | {'Missing':<7} | {'Coverage':<8}")
    print("-" * 80)
    
    for chapter in sorted(stats.keys()):
        ch_stats = stats[chapter]
        impl = ch_stats['implemented']
        part = ch_stats['partial']
        miss = ch_stats['missing'] + ch_stats['intentionally-deferred'] + ch_stats['spec-defect']
        tot = ch_stats['total']
        coverage = (impl / tot) * 100 if tot > 0 else 0.0
        print(f"{chapter:<30} | {tot:<6} | {impl:<6} | {part:<7} | {miss:<7} | {coverage:>6.1f}%")
        
    print("-" * 80)
    overall_coverage = (total_implemented / total_rules) * 100 if total_rules > 0 else 0.0
    print(f"{'Overall Coverage':<30} | {total_rules:<6} | {total_implemented:<6} | {'-':<7} | {'-':<7} | {overall_coverage:>6.1f}%")
    print("=" * 80)
    
    sys.exit(0)

if __name__ == '__main__':
    main()
