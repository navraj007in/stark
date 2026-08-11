#!/usr/bin/env python3
"""Resolution audit harness — scopes A and B.

Each case is a small STARK program plus an expectation. The harness runs `stark run` and compares
what happened with what should have happened. Three verdicts:

  HIT     the construct must compile AND the arm under test must fire
  BIND    the construct must compile and behave as a fresh binding (matches anything)
  REJECT  the construct must NOT compile

Every pattern case carries a wildcard arm, because the wildcard is what turns this defect class
from loud to silent: without it the compiler reports non-exhaustive and the mistake surfaces; with
it the program runs and quietly takes the wrong branch.
"""
import json
import pathlib
import shutil
import subprocess
import sys
import tempfile

STARK = pathlib.Path(sys.argv[1]).resolve()

MANIFEST = json.dumps(
    {"name": "probe", "version": "0.1.0", "entry": "src/main.stark", "dependencies": {}}
)


def run_case(source: str):
    """Returns (compiled: bool, stdout: str, diagnostic: str)."""
    with tempfile.TemporaryDirectory() as d:
        pkg = pathlib.Path(d)
        (pkg / "src").mkdir()
        (pkg / "starkpkg.json").write_text(MANIFEST)
        (pkg / "src" / "main.stark").write_text(source)
        proc = subprocess.run(
            [str(STARK), "run"], cwd=pkg, capture_output=True, text=True, timeout=120
        )
        out = proc.stdout + proc.stderr
        if "compilation failed" in out or "Error:" in out and proc.returncode != 0:
            first = next(
                (ln for ln in out.splitlines() if ln.startswith("Error:")), out.splitlines()[0] if out else ""
            )
            return (False, "", first.strip())
        return (True, proc.stdout.strip(), "")


CASES = []


def case(scope, name, expect, source, note=""):
    CASES.append(
        {"scope": scope, "name": name, "expect": expect, "source": source, "note": note}
    )


# ---------------------------------------------------------------- SCOPE A: silent acceptance --

WRAP = """enum Colour {{ Red, Green }}
struct Thing {{ value: Int64 }}
const LIMIT: Int64 = 3i64;
fn helper() -> Int64 {{ 1i64 }}
trait Describe {{ fn describe(&self) -> Int64; }}
{extra}
fn main() {{
{body}
}}
"""


def pattern_case(name, expect, scrutinee, pattern, extra="", note=""):
    body = f"""    let subject = {scrutinee};
    match subject {{
        {pattern} => println("HIT"),
        _other => println("MISS"),
    }}"""
    case("A", name, expect, WRAP.format(extra=extra, body=body), note)


pattern_case("valid unit variant", "HIT", "Colour::Red", "Colour::Red")
pattern_case("valid tuple variant", "HIT", "Shape::Line(4i64)", "Shape::Line(_n)",
             extra="enum Shape { Dot, Line(Int64) }")
pattern_case("misspelled enum variant", "REJECT", "Colour::Red", "Colour::Blu")
pattern_case("struct with a variant path", "REJECT", "Thing { value: 1i64 }", "Thing::Missing(_n)")
pattern_case("builtin constructor Some", "HIT", "Some(2i64)", "Some(_v)")
pattern_case("builtin constructor None", "HIT", "None", "None",
             note="the scrutinee needs an annotation; see the typed variant below")
pattern_case("builtin FUNCTION as a pattern", "REJECT", "Some(2i64)", "Vec::new(_x)")
pattern_case("builtin function, bare", "BIND", "3i64", "sqrt",
             note="not a constructor, so SYN-PATTERN-001 makes it a binding")
pattern_case("constant", "HIT", "3i64", "LIMIT")
pattern_case("bare function name", "BIND", "3i64", "helper")
pattern_case("bare struct name", "BIND", "3i64", "Thing")
pattern_case("bare enum type name", "BIND", "3i64", "Colour")
pattern_case("bare trait name", "BIND", "3i64", "Describe")
pattern_case("primitive type name", "BIND", "3i64", "Int64")
pattern_case("core trait member path", "REJECT", "3i64", "Eq::eq")
pattern_case("trait member path", "REJECT", "3i64", "Describe::describe")
pattern_case("struct pattern", "HIT", "Thing { value: 1i64 }", "Thing { value: _v }")
pattern_case("struct pattern, unknown field", "REJECT", "Thing { value: 1i64 }",
             "Thing { missing: _v }")
pattern_case("enum struct-variant pattern", "HIT", "Rec::One { n: 1i64 }", "Rec::One { n: _n }",
             extra="enum Rec { One { n: Int64 }, Two }")
pattern_case("module-qualified variant", "HIT", "inner::Hue::Blue", "inner::Hue::Blue",
             extra="mod inner { pub enum Hue { Blue, Teal } }")
pattern_case("module-qualified misspelling", "REJECT", "inner::Hue::Blue", "inner::Hue::Nope",
             extra="mod inner { pub enum Hue { Blue, Teal } }")
pattern_case("imported variant", "HIT", "Hue::Blue", "Blue",
             extra="mod inner { pub enum Hue { Blue, Teal } }\nuse inner::Hue;\nuse inner::Hue::Blue;")
pattern_case("module name as a pattern", "BIND", "3i64", "inner",
             extra="mod inner { pub fn f() -> Int64 { 1i64 } }")

# ------------------------------------------------------------------- SCOPE B: precedence -------

def precedence_case(name, expect_output, source, note=""):
    case("B", name, ("OUT:" + expect_output), source, note)


precedence_case(
    "enum variant vs module fn of the same name", "variant",
    """enum Attr { Flag, Policy }
fn Policy() -> Int64 { 9i64 }
fn main() {
    match Attr::Policy { Attr::Policy => println("variant"), Attr::Flag => println("flag") }
}
""")

precedence_case(
    "enum variant vs module TYPE of the same name", "variant",
    """enum Policy { A, B }
enum Attr { Flag, Policy(Policy) }
fn main() {
    match Attr::Policy(Policy::A) {
        Attr::Policy(_p) => println("variant"),
        Attr::Flag => println("flag"),
    }
}
""")

precedence_case(
    "struct assoc fn vs module fn of the same name", "1",
    """struct Foo { a: Int64 }
impl Foo { pub fn new() -> Foo { Foo { a: 1i64 } } }
fn new() -> Int64 { 99i64 }
fn main() { println(Foo::new().a); }
""")

precedence_case(
    "enum assoc fn vs module fn of the same name", "7",
    """enum Colour { Red, Green }
impl Colour { pub fn count() -> Int64 { 7i64 } }
fn count() -> Int64 { 99i64 }
fn main() { println(Colour::count()); }
""")

precedence_case(
    "module item vs enum variant, qualifier IS a module", "module",
    """mod inner {
    pub enum E { Thing }
    pub fn Thing() -> Int64 { 5i64 }
}
fn main() { let _n = inner::Thing(); println("module"); }
""",
    "a module qualifier must search the MODULE namespace, not some enum's variants")

precedence_case(
    "local binding vs enum variant of the same name", "local",
    """enum E { Value }
fn main() {
    let Value = 4i64;
    if Value == 4i64 { println("local"); } else { println("variant"); }
}
""",
    "a lexical binding must shadow a module-level name")

precedence_case(
    "user type vs hardcoded builtin path spelling", "user",
    """struct Buffer { n: Int64 }
impl Buffer { pub fn new() -> Buffer { Buffer { n: 1i64 } } }
fn main() { let b = Buffer::new(); if b.n == 1i64 { println("user"); } }
""")

precedence_case(
    "user trait named like a core trait", "user",
    """trait Eq2 { fn same(&self) -> Bool; }
struct P { n: Int64 }
impl Eq2 for P { fn same(&self) -> Bool { true } }
fn main() { let p = P { n: 1i64 }; if p.same() { println("user"); } }
""")

precedence_case(
    "nested module shadowing an outer name", "inner",
    """fn tag() -> Int64 { 1i64 }
mod inner {
    pub fn tag() -> Int64 { 2i64 }
    pub fn pick() -> Int64 { tag() }
}
fn main() { if inner::pick() == 2i64 { println("inner"); } else { println("outer"); } }
""")

precedence_case(
    "imported variant vs local fn of the same name", "import",
    """mod inner { pub enum Hue { Blue } }
use inner::Hue::Blue;
fn main() {
    match Blue { Blue => println("import"), }
}
""")

# --------------------------------------------------------------------------- run and report ---

results = []
for c in CASES:
    compiled, out, diag = run_case(c["source"])
    exp = c["expect"]
    if exp == "REJECT":
        verdict = "AGREES" if not compiled else "DISAGREES"
        observed = diag if not compiled else f"compiled, printed {out!r}"
    elif exp == "HIT":
        verdict = "AGREES" if compiled and "HIT" in out else "DISAGREES"
        observed = f"printed {out!r}" if compiled else f"rejected: {diag}"
    elif exp == "BIND":
        verdict = "AGREES" if compiled and "HIT" in out else "DISAGREES"
        observed = f"printed {out!r}" if compiled else f"rejected: {diag}"
    elif exp.startswith("OUT:"):
        want = exp[4:]
        verdict = "AGREES" if compiled and want in out else "DISAGREES"
        observed = f"printed {out!r}" if compiled else f"rejected: {diag}"
    results.append({**c, "verdict": verdict, "observed": observed})

for scope in ("A", "B"):
    rows = [r for r in results if r["scope"] == scope]
    dis = [r for r in rows if r["verdict"] == "DISAGREES"]
    print(f"\n{'='*78}\nSCOPE {scope}: {len(rows)} probes, {len(dis)} disagreements\n{'='*78}")
    for r in rows:
        mark = "  " if r["verdict"] == "AGREES" else "!!"
        print(f"{mark} [{r['expect']:>10}] {r['name']:<46} {r['observed']}")

print(f"\nTOTAL: {len(results)} probes, "
      f"{len([r for r in results if r['verdict']=='DISAGREES'])} disagreements")
