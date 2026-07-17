"""Shared parsing for `STARKLANG/conformance/core-v1-coverage.toml`.

Deliberately a flat, dependency-free TOML subset parser (no `tomllib`/`toml` package
requirement), matching the file's own documented schema constraint. Used by both
`check-conformance.py` (the validator) and `generate-conformance-report.py` (WP-C1.6's evidence
generator) so the two never parse the file two different ways.
"""

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


def repo_root():
    """Two levels up from this file: starkc/scripts/ -> starkc/ -> repo root."""
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_rules():
    toml_path = os.path.join(repo_root(), 'STARKLANG', 'conformance', 'core-v1-coverage.toml')
    with open(toml_path, 'r', encoding='utf-8') as f:
        return parse_toml(f.read())
