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
import argparse, json, os, shutil, subprocess, sys, tempfile, time

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
}

def run(cmd, **kw):
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, **kw)

def trial(spec, verbose):
    path = os.path.join(ROOT, spec["file"])
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
        return dict(spec_id=spec["id"], result="KILLED" if killed else "SURVIVED",
                    seconds=round(time.time() - started, 1),
                    detail=(out.stdout + out.stderr)[-600:] if verbose else "")
    finally:
        shutil.copyfile(backup.name, path)
        os.unlink(backup.name)

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
        if r["verdict"] == "UNEXPECTED" and r.get("detail"):
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
