#!/usr/bin/env python3
"""WP-C1.6 conformance evidence generator.

Emits, per rule in `STARKLANG/conformance/core-v1-coverage.toml`:

    rule id, spec chapter/section, status, source implementation,
    positive tests, negative tests, deviation id if any, last verified commit

Every column is either read directly from the coverage database or computed fresh from git at
generation time -- never hand-asserted prose, which is exactly the failure mode DEV-002 and
DEV-017 both found real instances of (a report field that looks derived but was actually typed
by hand, and silently went stale). `last_verified_commit` in particular is *never* stored in the
TOML: it is `git log -1 --format=%H -- <every path this rule cites>`, so it always reflects the
actual last commit that touched this rule's evidence, with no manual-update step to forget.

Two output formats:
  --format=json      (default) machine-readable, sorted by rule id, deterministic.
  --format=markdown   human-readable table, for a CI step summary (GITHUB_STEP_SUMMARY).

Usage:
    generate-conformance-report.py [--format=json|markdown] [--out=PATH]
"""
import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from conformance_lib import load_rules, repo_root  # noqa: E402


def split_evidence(entry):
    """`path::function` -> (path, function); bare `path` -> (path, None)."""
    path, _, fn = entry.partition('::')
    return path, (fn or None)


def last_verified_commit(root, paths):
    """`git log -1 --format=%H -- <paths>`, i.e. the most recent commit that touched any of this
    rule's cited evidence files. Returns None if no paths are known or none are tracked yet
    (e.g. mid-development, before the first commit that includes them)."""
    if not paths:
        return None
    result = subprocess.run(
        ['git', 'log', '-1', '--format=%H', '--', *sorted(set(paths))],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    sha = result.stdout.strip()
    return sha or None


def evidence_paths(rule):
    """Every file path this rule's evidence fields reference, for the git-log lookup. Source is
    included since a rule whose implementation changed but whose tests didn't is still
    meaningfully "reverified" at that commit."""
    paths = []
    if rule.get('source'):
        paths.append(rule['source'])
    for field in ('tests', 'positive_tests', 'negative_tests'):
        for entry in rule.get(field) or []:
            path, _ = split_evidence(entry)
            paths.append(path)
    return paths


def build_report(root, rules):
    report = []
    for rule in sorted(rules, key=lambda r: r.get('id', '')):
        positive = rule.get('positive_tests')
        negative = rule.get('negative_tests')
        # WP-C1.6 / DEV-017: most rules don't have the positive/negative split yet -- report
        # that honestly as "unclassified", not as an empty list (empty means "checked, there
        # genuinely is none"; unclassified means "not yet attributed at this precision").
        legacy_tests = rule.get('tests') or []
        if positive is None and negative is None:
            positive_out = "unclassified (see 'tests', DEV-017)" if legacy_tests else None
            negative_out = "unclassified (see 'tests', DEV-017)" if legacy_tests else None
        else:
            positive_out = positive or []
            negative_out = negative or []
        report.append({
            "rule_id": rule.get('id'),
            "spec_chapter": rule.get('chapter'),
            "status": rule.get('status'),
            "source": rule.get('source'),
            "positive_tests": positive_out,
            "negative_tests": negative_out,
            "deviation": rule.get('deviation'),
            "last_verified_commit": last_verified_commit(root, evidence_paths(rule)),
        })
    return report


def render_markdown(report):
    lines = [
        "# STARK Core v1 Conformance Evidence Report",
        "",
        "| Rule | Chapter | Status | Source | Positive tests | Negative tests | Deviation | Last verified |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in report:
        pos = row['positive_tests']
        neg = row['negative_tests']
        pos_cell = f"{len(pos)}" if isinstance(pos, list) else (pos or "-")
        neg_cell = f"{len(neg)}" if isinstance(neg, list) else (neg or "-")
        commit = (row['last_verified_commit'] or "-")[:10]
        lines.append(
            f"| {row['rule_id']} | {row['spec_chapter']} | {row['status']} | "
            f"`{row['source'] or '-'}` | {pos_cell} | {neg_cell} | {row['deviation'] or '-'} | "
            f"`{commit}` |"
        )
    unclassified = sum(
        1 for r in report if isinstance(r['positive_tests'], str)
    )
    lines.append("")
    lines.append(
        f"{unclassified} of {len(report)} rules have positive/negative test evidence that is "
        f"still unclassified at file-level precision only (DEV-017)."
    )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--format", choices=["json", "markdown"], default="json")
    parser.add_argument("--out", default=None, help="write to this path instead of stdout")
    args = parser.parse_args()

    root = repo_root()
    rules = load_rules()
    report = build_report(root, rules)

    if args.format == "json":
        output = json.dumps(report, indent=2, sort_keys=True) + "\n"
    else:
        output = render_markdown(report)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(output)
    else:
        sys.stdout.write(output)


if __name__ == "__main__":
    main()
