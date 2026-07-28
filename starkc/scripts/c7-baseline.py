#!/usr/bin/env python3
"""WP-C7.0 — the baseline measurement harness.

Two jobs, deliberately separate:

  --measure   stage timings per workload, splitting STARK compiler work from Cargo/rustc work
  --reproduce clean builds from two DISTINCT absolute checkout paths, comparing every artefact class

**Why the split matters.** §2.3 forbids reporting only a combined total. `stark build` shells out to
`cargo`, and if that dominates then a front-end cache cannot fix build latency however fast it is.

**How the split is obtained, and one way it must NOT be.** The first version of this harness timed
`stark build` against `stark build --emit-rust`, assuming the latter stopped before Cargo. It does
not — `--emit-rust` only additionally writes the generated file, so the two timings were the same
run and the "host share" came out as noise, once at -0.3%. A negative share is the useful kind of
wrong: it is impossible, so it exposed the method rather than quietly biasing a number.

The split is measured instead: build once with `--keep-generated`, then `cargo clean` the generated
crate and time `cargo build` on its own. That is the host cost cold, and total minus host is the
STARK compiler's own work.
"""
import argparse, hashlib, json, os, pathlib, shutil, subprocess, statistics, sys, tempfile, time

ROOT = pathlib.Path(__file__).resolve().parent.parent
WORKLOADS = ROOT / "benchmarks" / "c7-workloads"
STARK = ROOT / "target" / "debug" / "stark"

def package_dir(w: pathlib.Path) -> pathlib.Path:
    return w / "app" if (w / "app").is_dir() else w

def sha(p: pathlib.Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()

def source_identity(w: pathlib.Path) -> dict:
    """Content identity of a frozen workload: every .stark and every manifest, path-relative."""
    files = {}
    for p in sorted(w.rglob("*")):
        if p.is_file() and (p.suffix == ".stark" or p.name in ("starkpkg.json", "stark.lock")):
            files[str(p.relative_to(w))] = sha(p)
    digest = hashlib.sha256(
        "\n".join(f"{k}:{v}" for k, v in sorted(files.items())).encode()
    ).hexdigest()
    return {"files": files, "workload_hash": digest}

def clean(w: pathlib.Path):
    for d in w.rglob("target"):
        if d.is_dir():
            shutil.rmtree(d, ignore_errors=True)
    for lock in w.rglob("stark.lock"):
        lock.unlink(missing_ok=True)

def run(cmd, cwd):
    t = time.perf_counter()
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    return time.perf_counter() - t, r

def measure(reps: int) -> dict:
    out = {}
    for w in sorted(WORKLOADS.glob("w0*")):
        pkg = package_dir(w)
        totals, hosts = [], []
        for _ in range(reps):
            clean(w)
            dt_total, r_total = run([str(STARK), "build", "--keep-generated"], pkg)
            if r_total.returncode != 0:
                out[w.name] = {"error": r_total.stderr[-300:]}
                break
            # The generated crate is kept under `target/stark/<profile>/<hash>/`. Cleaning it and
            # rebuilding times the HOST cost alone, cold, for exactly the crate this build produced.
            crates = [c.parent for c in (pkg / "target" / "stark").rglob("Cargo.toml")
                      if "stark-runtime" not in str(c)]
            if not crates:
                out[w.name] = {"error": "generated crate not found under target/stark"}
                break
            subprocess.run(["cargo", "clean", "-q"], cwd=crates[0], capture_output=True)
            dt_host, _ = run(["cargo", "build", "--locked", "--offline", "-q"], crates[0])
            totals.append(dt_total)
            hosts.append(dt_host)
        else:
            gen = sorted((pkg / "target" / "stark").rglob("main.rs"))
            binary = next((p for p in (pkg / "target" / "stark" / "debug").iterdir()
                           if p.is_file() and not p.suffix), None) if (pkg / "target" / "stark" / "debug").is_dir() else None
            out[w.name] = {
                "identity": source_identity(w),
                "total_build_s": {"median": round(statistics.median(totals), 3),
                                   "min": round(min(totals), 3), "max": round(max(totals), 3)},
                "host_cargo_s": {"median": round(statistics.median(hosts), 3),
                                 "min": round(min(hosts), 3), "max": round(max(hosts), 3)},
                "stark_share_s": round(statistics.median(totals) - statistics.median(hosts), 3),
                "host_share_pct": round(100 * statistics.median(hosts)
                                        / statistics.median(totals), 1),
                "generated_rust_bytes": gen[0].stat().st_size if gen else None,
                "executable_bytes": binary.stat().st_size if binary else None,
                "reps": reps,
            }
    return out

ARTEFACT_CLASSES = {
    "generated_rust": "**/main.rs",
    "generated_cargo_toml": "**/Cargo.toml",
    "stark_lock": "stark.lock",
}

def reproduce() -> dict:
    """Clean-build the same logical source from two DIFFERENT absolute paths and compare."""
    results = {}
    for w in sorted(WORKLOADS.glob("w0*")):
        with tempfile.TemporaryDirectory(prefix="c7_repro_a_") as a, \
             tempfile.TemporaryDirectory(prefix="c7_repro_bb_longer_") as b:
            copies = []
            for root in (a, b):
                dst = pathlib.Path(root) / w.name
                shutil.copytree(w, dst)
                clean(dst)
                r = subprocess.run([str(STARK), "build", "--keep-generated"],
                                   cwd=package_dir(dst), capture_output=True, text=True)
                copies.append((dst, r))
            (da, ra), (db, rb) = copies
            if ra.returncode or rb.returncode:
                results[w.name] = {"error": (ra.stderr or rb.stderr)[-300:]}
                continue
            cls = {}
            for name, pattern in ARTEFACT_CLASSES.items():
                fa = sorted(da.rglob(pattern.split("/")[-1]))
                fb = sorted(db.rglob(pattern.split("/")[-1]))
                fa = [p for p in fa if "stark-runtime" not in str(p)]
                fb = [p for p in fb if "stark-runtime" not in str(p)]
                if not fa or not fb:
                    cls[name] = {"verdict": "NOT-PRODUCED"}
                    continue
                ha, hb = sha(fa[0]), sha(fb[0])
                entry = {"verdict": "BYTE-IDENTICAL" if ha == hb else "DIFFERS",
                         "a": ha[:16], "b": hb[:16]}
                if ha != hb:
                    ta, tb = fa[0].read_text(errors="replace"), fb[0].read_text(errors="replace")
                    diffs = [(i, x, y) for i, (x, y) in
                             enumerate(zip(ta.splitlines(), tb.splitlines()), 1) if x != y]
                    entry["differing_lines"] = len(diffs)
                    entry["first_diff"] = ({"line": diffs[0][0], "a": diffs[0][1][:110],
                                            "b": diffs[0][2][:110]} if diffs else None)
                    # Does the difference contain either checkout path? That is the leak test.
                    entry["contains_checkout_path"] = any(
                        str(da) in d[1] or str(db) in d[2] for d in diffs[:50])
                cls[name] = entry
            # The executable: compare bytes.
            bins = []
            for d in (da, db):
                dbg = package_dir(d) / "target" / "stark" / "debug"
                bins.append(next((p for p in dbg.iterdir() if p.is_file() and not p.suffix), None)
                            if dbg.is_dir() else None)
            if all(bins):
                cls["executable"] = {"verdict": "BYTE-IDENTICAL" if sha(bins[0]) == sha(bins[1])
                                     else "DIFFERS",
                                     "a_bytes": bins[0].stat().st_size,
                                     "b_bytes": bins[1].stat().st_size}
            results[w.name] = cls
    return results

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--measure", action="store_true")
    ap.add_argument("--reproduce", action="store_true")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--out", type=pathlib.Path)
    a = ap.parse_args()
    if not STARK.exists():
        sys.exit(f"build `stark` first: {STARK} not found")
    payload = {"stark_binary_sha256": sha(STARK)}
    if a.measure:
        payload["measurements"] = measure(a.reps)
    if a.reproduce:
        payload["reproducibility"] = reproduce()
    text = json.dumps(payload, indent=2)
    print(text)
    if a.out:
        a.out.write_text(text + "\n")

if __name__ == "__main__":
    main()
