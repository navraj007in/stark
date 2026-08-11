#!/usr/bin/env python3
"""Resolution audit, scopes C and D.

C — pattern legality matrix. Scope A probed the pattern kinds a program most often writes. This
    walks the matrix the other way: for each PATH-BEARING pattern kind (`Path`, `TupleVariant`,
    `Struct`) against each resolution category the resolver can produce, plus the non-path kinds
    (`Tuple`, `Array`, `Lit`) against mismatched types.

D — resolution well-formedness. Probed from the surface language: can a program cause a
    `Res::Variant` to be consumed against the wrong enum, or an index the enum does not have?
    The producers all derive the index from the enum's own variant list, so this asks whether any
    program can make producer and consumer disagree.

Every probe runs a whole program. A pattern that compiles but never matches is the failure mode
this whole audit exists for, so REJECT cases carry a wildcard and HIT cases assert the arm fired.
"""
import json
import pathlib
import subprocess
import sys
import tempfile

STARK = pathlib.Path(sys.argv[1]).resolve()
MANIFEST = json.dumps(
    {"name": "probe", "version": "0.1.0", "entry": "src/main.stark", "dependencies": {}}
)


def run(source: str):
    with tempfile.TemporaryDirectory() as d:
        pkg = pathlib.Path(d)
        (pkg / "src").mkdir()
        (pkg / "starkpkg.json").write_text(MANIFEST)
        (pkg / "src" / "main.stark").write_text(source)
        proc = subprocess.run(
            [str(STARK), "run"], cwd=pkg, capture_output=True, text=True, timeout=120
        )
        out = proc.stdout + proc.stderr
        if proc.returncode != 0 or "compilation failed" in out:
            first = next((l for l in out.splitlines() if l.startswith("Error:")), "")
            return (False, "", first.strip()[:72])
        return (True, proc.stdout.strip(), "")


CASES = []


def case(scope, name, expect, source):
    CASES.append({"scope": scope, "name": name, "expect": expect, "source": source})


PRELUDE = """enum Colour { Red, Green }
enum Shape { Dot, Line(Int64) }
enum Rec { One { n: Int64 }, Two }
struct Thing { value: Int64 }
struct Pair { a: Int64, b: Int64 }
trait Describe { fn describe(&self) -> Int64; }
const LIMIT: Int64 = 3i64;
fn helper() -> Int64 { 1i64 }
mod inner { pub fn f() -> Int64 { 1i64 } }
"""


def pat(name, expect, scrutinee, pattern):
    src = (
        PRELUDE
        + "fn main() {\n"
        + f"    let subject = {scrutinee};\n"
        + "    match subject {\n"
        + f"        {pattern} => println(\"HIT\"),\n"
        + "        _ => println(\"MISS\"),\n"
        + "    }\n}\n"
    )
    case("C", name, expect, src)


# --- Struct-shaped patterns against every resolution category ---------------------------------
pat("Struct pat on a struct", "HIT", "Thing { value: 1i64 }", "Thing { value: _v }")
pat("Struct pat on an enum struct-variant", "HIT", "Rec::One { n: 1i64 }", "Rec::One { n: _n }")
pat("Struct pat naming a FUNCTION", "REJECT", "Thing { value: 1i64 }", "helper { value: _v }")
pat("Struct pat naming a MODULE", "REJECT", "Thing { value: 1i64 }", "inner { value: _v }")
pat("Struct pat naming a PRIMITIVE", "REJECT", "Thing { value: 1i64 }", "Int64 { value: _v }")
pat("Struct pat naming a TRAIT", "REJECT", "Thing { value: 1i64 }", "Describe { value: _v }")
pat("Struct pat naming a CONST", "REJECT", "Thing { value: 1i64 }", "LIMIT { value: _v }")
pat("Struct pat, unknown field", "REJECT", "Thing { value: 1i64 }", "Thing { missing: _v }")
pat("Struct pat, tuple-variant path", "REJECT", "Shape::Line(1i64)", "Shape::Line { n: _v }")

# --- TupleVariant patterns against every resolution category ----------------------------------
pat("Tuple-variant pat on a variant", "HIT", "Shape::Line(4i64)", "Shape::Line(_n)")
pat("Tuple-variant pat naming a STRUCT", "REJECT", "Thing { value: 1i64 }", "Thing(_v)")
pat("Tuple-variant pat naming a FUNCTION", "REJECT", "3i64", "helper(_v)")
pat("Tuple-variant pat naming a MODULE", "REJECT", "3i64", "inner(_v)")
pat("Tuple-variant pat naming a CONST", "REJECT", "3i64", "LIMIT(_v)")
pat("Tuple-variant pat on a UNIT variant", "REJECT", "Colour::Red", "Colour::Red(_v)")
pat("Tuple-variant arity too high", "REJECT", "Shape::Line(1i64)", "Shape::Line(_a, _b)")

# --- Path patterns ------------------------------------------------------------------------------
pat("Path pat on a unit variant", "HIT", "Colour::Red", "Colour::Red")
# SYN-PATTERN-001: "Multi-segment `Path` patterns always match by value", and Core v1 has no
# rest patterns, so a bare path is the ONLY way to match a struct variant without binding its
# fields. This probe originally expected a rejection out of Rust intuition; the specification
# disagreed, and the expectation is corrected rather than the compiler.
pat("Path pat on a struct-variant path", "HIT", "Rec::One { n: 1i64 }", "Rec::One")
pat("Path pat naming a trait member", "REJECT", "3i64", "Describe::describe")

# --- non-path kinds against mismatched types ----------------------------------------------------
pat("Tuple pat on a tuple", "HIT", "(1i64, 2i64)", "(_a, _b)")
pat("Tuple pat, wrong arity", "REJECT", "(1i64, 2i64)", "(_a, _b, _c)")
pat("Tuple pat on a struct", "REJECT", "Thing { value: 1i64 }", "(_a, _b)")
pat("Array pat on an array", "HIT", "[1i64, 2i64]", "[_a, _b]")
pat("Array pat, wrong arity", "REJECT", "[1i64, 2i64]", "[_a, _b, _c]")
pat("Lit pat, mismatched type", "REJECT", "3i64", "\"text\"")

# --- D: can producer and consumer disagree about a variant index? -------------------------------
# Both same-named variants across two enums, and a variant index that exists in one enum but not
# the other. If any resolution carried an index from the wrong enum, one of these misbehaves.
case(
    "D",
    "same-named variants in two enums do not cross",
    "OUT:a-second",
    """enum A { First, Second }
enum B { Second }
fn name_a(x: &A) -> String {
    match *x { A::First => String::from("a-first"), A::Second => String::from("a-second") }
}
fn main() { println(name_a(&A::Second).as_str()); }
""",
)
case(
    "D",
    "an index valid in one enum but not the other",
    "OUT:only",
    """enum Wide { V0, V1, V2, V3 }
enum Narrow { Only }
fn narrow(x: &Narrow) -> String { match *x { Narrow::Only => String::from("only") } }
fn main() {
    let _w = Wide::V3;
    println(narrow(&Narrow::Only).as_str());
}
""",
)
case(
    "D",
    "a re-exported variant keeps its own enum's identity",
    "OUT:teal",
    """mod inner { pub enum Hue { Blue, Teal } }
use inner::Hue;
use inner::Hue::Teal;
enum Other { Blue, Teal }
fn hue(h: &Hue) -> String {
    match *h { Hue::Blue => String::from("blue"), Hue::Teal => String::from("teal") }
}
fn main() { println(hue(&Teal).as_str()); }
""",
)
case(
    "D",
    "a generic enum's variant index survives instantiation",
    "OUT:some-7",
    """enum Slot<T> { Empty, Filled(T) }
fn show(s: &Slot<Int64>) -> String {
    match *s {
        Slot::Empty => String::from("empty"),
        Slot::Filled(v) => {
            let mut out = String::from("some-");
            out.push_str(int_text(v).as_str());
            out
        }
    }
}
fn int_text(v: Int64) -> String { if v == 7i64 { String::from("7") } else { String::from("?") } }
fn main() { println(show(&Slot::Filled(7i64)).as_str()); }
""",
)

results = []
for c in CASES:
    compiled, out, diag = run(c["source"])
    exp = c["expect"]
    if exp == "REJECT":
        ok = not compiled
        observed = diag if not compiled else f"COMPILED, printed {out!r}"
    elif exp == "HIT":
        ok = compiled and "HIT" in out
        observed = f"printed {out!r}" if compiled else f"rejected: {diag}"
    else:
        want = exp[4:]
        ok = compiled and want in out
        observed = f"printed {out!r}" if compiled else f"rejected: {diag}"
    results.append({**c, "verdict": "AGREES" if ok else "DISAGREES", "observed": observed})

for scope in ("C", "D"):
    rows = [r for r in results if r["scope"] == scope]
    dis = [r for r in rows if r["verdict"] == "DISAGREES"]
    print(f"\n{'='*80}\nSCOPE {scope}: {len(rows)} probes, {len(dis)} disagreements\n{'='*80}")
    for r in rows:
        mark = "  " if r["verdict"] == "AGREES" else "!!"
        print(f"{mark} [{r['expect']:>10}] {r['name']:<42} {r['observed']}")
print(
    f"\nTOTAL: {len(results)} probes, "
    f"{len([r for r in results if r['verdict']=='DISAGREES'])} disagreements"
)
