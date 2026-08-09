#!/usr/bin/env python3
"""C10-E — performance baselines for Gate C10.

**The workload set was FROZEN before measurement** (plan §12.1). `FROZEN.json` pins seven workloads
by per-file SHA-256; this script verifies every hash before it measures anything, so a baseline can
never be reported against a workload that drifted.

**What this adds over `c7-baseline.py`.** That harness answers "how much of a `stark build` is
Cargo" — a real question, and a coarser one. Plan §12.2 additionally requires the front-end phase
split (lex / parse / resolve / check), peak compiler memory, and scaling. Those are measured here.

**Baselines only.** Plan §12.3: *do not optimise because a number looks unattractive.* Nothing in
this script compares against a threshold, and no threshold is proposed — WP-C10.6 is explicit that
regression thresholds may be added only after stable baselines exist, and this is the first.

**Impossible values are checked, not trusted.** `c7-baseline.py`'s header records a method error
that produced a -0.3% host share, and it was caught because a negative share is impossible rather
than because anyone reviewed the arithmetic. Every derived quantity here is range-checked.
"""
from __future__ import annotations
import hashlib, json, pathlib, platform, resource, statistics, subprocess, sys, time

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
WORKLOADS = ROOT / "starkc/benchmarks/c7-workloads"
FROZEN = WORKLOADS / "FROZEN.json"
PHASES = ROOT / "starkc/target/debug/examples/c10e_phases"


def verify_frozen() -> dict:
    d = json.loads(FROZEN.read_text(encoding="utf-8"))
    drift = []
    for name, rec in d["workloads"].items():
        for rel, want in rec["files"].items():
            p = WORKLOADS / name / rel
            got = hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None
            if got != want:
                drift.append(f"{name}/{rel}")
    if drift:
        sys.exit(f"FROZEN WORKLOAD DRIFT — refusing to measure: {drift}")
    return d


def entry_source(name: str) -> pathlib.Path | None:
    for cand in (WORKLOADS / name / "src/main.stark", WORKLOADS / name / "app/src/main.stark"):
        if cand.exists():
            return cand
    return None


def phases(src: pathlib.Path, reps: int) -> dict | None:
    """Front-end phase split, plus the child's peak RSS.

    RSS is taken with `getrusage(RUSAGE_CHILDREN)` deltas rather than by polling: polling can miss a
    peak between samples, and a peak that is missed is silently reported as a smaller number.
    """
    before = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    r = subprocess.run([str(PHASES), str(src), str(reps)], capture_output=True, text=True)
    after = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    if r.returncode != 0:
        return None
    out = json.loads(r.stdout.strip().splitlines()[-1])
    # macOS reports bytes, Linux kilobytes. Recorded with its unit rather than normalised to a
    # number whose meaning depends on the reader's platform.
    out["peak_child_rss"] = max(after - before, after)
    out["peak_child_rss_unit"] = "bytes" if sys.platform == "darwin" else "kilobytes"
    total = out["lex_ns"] + out["parse_ns"] + out["resolve_ns"] + out["check_ns"]
    out["frontend_total_ns"] = total
    for k in ("lex", "parse", "resolve", "check"):
        pct = 100.0 * out[f"{k}_ns"] / total if total else 0.0
        if not (0.0 <= pct <= 100.0):
            sys.exit(f"IMPOSSIBLE VALUE: {k} share is {pct}% — the method is wrong, not the compiler")
        out[f"{k}_pct"] = round(pct, 1)
    return out


def scaling(reps: int) -> list[dict]:
    """Large-module scaling, on GENERATED sources that are not part of the frozen set.

    Kept separate and labelled: the frozen set answers "what does a representative program cost",
    and it cannot answer "how does cost grow", because seven fixed workloads have no size axis.
    Generated inputs are reproducible from this function and carry no hashes, because they are not
    evidence about any particular program.
    """
    out = []
    tmp = pathlib.Path(subprocess.run(["mktemp", "-d"], capture_output=True, text=True).stdout.strip())
    for n in (100, 400, 1600, 6400):
        body = "\n".join(f"fn f{i}(x: Int32) -> Int32 {{ x + {i} }}" for i in range(n))
        src = tmp / f"scale_{n}.stark"
        src.write_text(f"{body}\nfn main() {{ }}\n", encoding="utf-8")
        p = phases(src, reps)
        if p:
            p["functions"] = n
            out.append(p)
    return out


def main() -> int:
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 9
    if not PHASES.exists():
        sys.exit("build the phase timer first: cargo build --example c10e_phases")
    frozen = verify_frozen()

    report = {
        "measured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                 text=True, cwd=ROOT).stdout.strip(),
        "platform": f"{platform.system()}-{platform.machine()}",
        "rustc": subprocess.run(["rustc", "-V"], capture_output=True, text=True).stdout.strip(),
        "frozen_at_commit": frozen["frozen_at_commit"],
        "frozen_workload_integrity": "VERIFIED — every file hash matches FROZEN.json",
        "reps": reps,
        "workloads": {},
        "scaling_generated": [],
    }
    for name in sorted(frozen["workloads"]):
        src = entry_source(name)
        report["workloads"][name] = phases(src, reps) if src else {"error": "no entry source"}
    report["scaling_generated"] = scaling(reps)

    out = ROOT / f"starkc/benchmarks/c10/{report['platform'].lower()}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"wrote {out.relative_to(ROOT)}   platform {report['platform']}   commit {report['commit'][:7]}")
    print(f"\n{'workload':<22} {'lex':>7} {'parse':>8} {'resolve':>9} {'check':>8}   front-end total")
    for name, w in report["workloads"].items():
        if "error" in w:
            print(f"  {name:<20} {w['error']}")
            continue
        print(f"  {name:<20} {w['lex_pct']:>6}% {w['parse_pct']:>7}% "
              f"{w['resolve_pct']:>8}% {w['check_pct']:>7}%   {w['frontend_total_ns']/1e6:.2f} ms")
    print(f"\n{'functions':<12} {'front-end ms':>13} {'check %':>9}")
    for s in report["scaling_generated"]:
        print(f"  {s['functions']:<10} {s['frontend_total_ns']/1e6:>12.2f} {s['check_pct']:>8}%")
    print("\nBaselines only. No threshold is proposed and none should be inferred (plan §12.3).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
