"""Gate 6 (G6-05) — compile/run each typed-Rust comparator case, record results.

For every case in cases/, invoke stable `rustc` and record whether it compiles
and (for the one runtime case) how the compiled binary behaves. Nothing is
asserted — the `outcome` of each row is derived from rustc's exit code and the
program's exit code. Writes JSON to the path given as argv[1] (or stdout).
"""

import json
import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
CASES = os.path.join(HERE, "cases")

# name -> (expectation, what "caught" means)
SPEC = {
    "valid": ("compiles", "baseline correct pipeline"),
    "d1_dim": ("compile_error", "incompatible tensor dimensions"),
    "d2_dtype": ("compile_error", "incorrect element type"),
    "d3_device": ("compile_error", "incompatible device placement"),
    "d4b_artifact": ("compiles_runtime_error", "runtime artifact swap"),
    "limit_reshape": ("compile_error_cannot_express", "shape arithmetic wall (not a defect)"),
}


def first_error(stderr):
    for line in stderr.splitlines():
        s = line.strip()
        if s.startswith("error"):
            return s[:300]
    return (stderr.strip().splitlines() or [""])[0][:300]


def rustc(src, out):
    return subprocess.run(
        ["rustc", "--edition", "2021", "-A", "dead_code", "-A", "unused_variables",
         "--crate-type", "bin", src, "-o", out],
        capture_output=True,
        text=True,
    )


def classify(name, expectation, compiled, cargo_err, run_exit):
    if expectation == "compiles":
        return ("compiles", compiled, "n/a")
    if expectation == "compile_error":
        # caught at compile time iff rustc rejected it
        return ("caught_compile_time" if not compiled else "MISSED", not compiled, "compile")
    if expectation == "compiles_runtime_error":
        if compiled and run_exit not in (0, None):
            return ("caught_runtime", True, "runtime/load")
        return ("MISSED" if compiled else "compile_failed_unexpected", False, "runtime/load")
    if expectation == "compile_error_cannot_express":
        # not a defect: the host cannot express a valid program on stable Rust
        return ("cannot_express_on_stable", not compiled, "n/a")
    return ("unknown", False, "n/a")


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else None
    rustc_ver = subprocess.run(["rustc", "--version"], capture_output=True, text=True).stdout.strip()

    rows = []
    with tempfile.TemporaryDirectory() as td:
        for name, (expectation, roadmap) in SPEC.items():
            src = os.path.join(CASES, f"{name}.rs")
            binout = os.path.join(td, name)
            r = rustc(src, binout)
            compiled = r.returncode == 0
            run_exit = None
            run_stderr = ""
            if compiled and expectation == "compiles_runtime_error":
                rr = subprocess.run([binout], capture_output=True, text=True)
                run_exit = rr.returncode
                run_stderr = rr.stderr.strip()
            outcome, matched_expectation, stage = classify(
                name, expectation, compiled, r.stderr, run_exit
            )
            rows.append(
                {
                    "case": name,
                    "roadmap_class": roadmap,
                    "expectation": expectation,
                    "compiled": compiled,
                    "run_exit": run_exit,
                    "stage": stage,
                    "outcome": outcome,
                    "as_expected": matched_expectation,
                    "diagnostic": first_error(r.stderr) if not compiled else run_stderr[:300],
                }
            )

    report = {
        "harness": "gate6 rust-comparator/run.py",
        "comparator": "strongest generated typed-Rust host (const-generic dims + phantom device + typed elements + load-time digest)",
        "rustc": rustc_ver,
        "channel": "stable",
        "cases": rows,
    }
    print(json.dumps(report, indent=2))
    if out_path:
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nwrote {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
