#!/usr/bin/env python3
import sys
import os

def parse_toml(content):
    rules = []
    current_rule = None
    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        if line == '[[rule]]':
            if current_rule:
                rules.append(current_rule)
            current_rule = {}
            continue
        if '=' in line:
            key, val = line.split('=', 1)
            key = key.strip()
            val = val.strip()
            if val.startswith('"') and val.endswith('"'):
                val = val[1:-1]
            elif val.startswith('[') and val.endswith(']'):
                items = val[1:-1].split(',')
                val = [i.strip().strip('"') for i in items if i.strip()]
            if current_rule is not None:
                current_rule[key] = val
    if current_rule:
        rules.append(current_rule)
    return rules

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
