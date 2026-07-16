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
    
    # Validation
    for rule in rules:
        rule_id = rule.get('id')
        if not rule_id:
            errors.append("Rule found without 'id' attribute.")
            continue
            
        if 'chapter' not in rule:
            errors.append(f"Rule {rule_id} is missing 'chapter' attribute.")
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
