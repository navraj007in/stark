#!/usr/bin/env python3
"""Audit scopes E and F — aimed at type checking, borrow checking and lowering.

E — ENGINE DIFFERENTIAL. Each program runs under the interpreter and as a native binary, and the
    outputs are compared. This needs no oracle: a divergence is wrong whichever engine is right,
    and it is the class DEV-224 belonged to (type-checked, ran, would not build). It also catches
    "builds but behaves differently", which no expectation-based probe finds.

F — REJECTION MATRIX. Programs the ownership and type rules must refuse. The failure mode being
    hunted is the one this whole audit keeps finding: accepted, no diagnostic, wrong behaviour.

Run: scope_ef.py <path-to-stark>
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
WORK = pathlib.Path(tempfile.mkdtemp(prefix="scope_ef_"))


def _pkg(source):
    d = WORK / f"p{abs(hash(source)) % 10**9}"
    shutil.rmtree(d, ignore_errors=True)
    (d / "src").mkdir(parents=True)
    (d / "starkpkg.json").write_text(MANIFEST)
    (d / "src" / "main.stark").write_text(source)
    return d


def interp(source):
    d = _pkg(source)
    p = subprocess.run([str(STARK), "run"], cwd=d, capture_output=True, text=True, timeout=180)
    out = p.stdout + p.stderr
    if p.returncode != 0 or "compilation failed" in out:
        first = next((l for l in out.splitlines() if l.startswith("Error:")), out.strip()[:60])
        return ("FAIL", first.strip()[:78])
    return ("OK", p.stdout.strip())


def native(source):
    d = _pkg(source)
    b = subprocess.run(
        [str(STARK), "build", "--no-build-cache"], cwd=d, capture_output=True, text=True, timeout=900
    )
    if b.returncode != 0:
        out = b.stdout + b.stderr
        first = next(
            (l for l in out.splitlines() if l.startswith("Error:") or l.startswith("error:")),
            out.strip()[:60],
        )
        return ("FAIL", first.strip()[:78])
    exe = d / "target" / "stark" / "debug" / "probe"
    r = subprocess.run([str(exe)], cwd=d, capture_output=True, text=True, timeout=180)
    if r.returncode != 0:
        return ("TRAP", (r.stdout + r.stderr).strip()[:78])
    return ("OK", r.stdout.strip())


E_CASES, F_CASES = [], []


def diff(name, body, prelude=""):
    E_CASES.append((name, prelude + "fn main() {\n" + body + "\n}\n"))


def reject(name, source):
    F_CASES.append((name, source))


# ---------------------------------------------------------------- E: engine differential -------

diff("integer edges", """    let a: Int64 = 9223372036854775807i64;
    println(a);
    let b: UInt8 = 255u8;
    println(b as UInt64);
    let c: Int32 = 0i32 - 2147483647i32;
    println(c as Int64);""")

diff("integer division and remainder", """    let a = 0i64 - 7i64;
    println(a / 2i64);
    println(a - a / 2i64 * 2i64);
    println(7u64 / 2u64);""")

diff("shifts and bitwise", """    let x: UInt32 = 0xF0F0F0F0u32;
    println(x >> 4u32);
    println(x << 4u32);
    println((x & 0x0F0F0F0Fu32) as UInt64);
    println((x | 0x0F0F0F0Fu32) as UInt64);
    println((x ^ 0xFFFFFFFFu32) as UInt64);""")

diff("float formatting and comparison", """    let a = 1.5f64;
    let b = 0.1f64 + 0.2f64;
    println(a);
    if b > 0.3f64 { println("gt"); } else { println("le"); }""")

diff("string building and slicing", """    let mut s = String::new();
    s.push_str("hello");
    s.push(' ');
    s.push_str("world");
    println(s.as_str());
    println(s.len());
    println(s.substring(0u64, 5u64).as_str());""")

diff("vec growth, index, iteration", """    let mut v: Vec<Int64> = Vec::new();
    let mut i = 0i64;
    while i < 5i64 { v.push(i * i); i = i + 1i64; }
    println(v.len());
    let mut j = 0u64;
    while j < v.len() { println(v[j]); j = j + 1u64; }""")

diff("nested struct and enum values", """    let r = Rec::One { n: 3i64 };
    match r { Rec::One { n } => println(n), Rec::Two => println("two") }
    let p = Pair { a: Thing { value: 1i64 }, b: Thing { value: 2i64 } };
    println(p.a.value + p.b.value);""",
    "struct Thing { value: Int64 }\nstruct Pair { a: Thing, b: Thing }\nenum Rec { One { n: Int64 }, Two }\n")

diff("generic function and struct", """    println(first(3i64, 4i64));
    let b = Box2 { one: 7i64, two: 8i64 };
    println(b.one + b.two);""",
    "fn first<T>(a: T, _b: T) -> T { a }\nstruct Box2<T> { one: T, two: T }\n")

diff("trait dispatch through a bound", """    let p = P { n: 5i64 };
    println(describe(&p));""",
    "trait D { fn d(&self) -> Int64; }\nstruct P { n: Int64 }\nimpl D for P { fn d(&self) -> Int64 { self.n * 2i64 } }\nfn describe<T: D>(x: &T) -> Int64 { x.d() }\n")

diff("Drop ordering", """    let _a = N { t: 1i64 };
    { let _b = N { t: 2i64 }; println("inner"); }
    println("outer");""",
    "struct N { t: Int64 }\nimpl Drop for N { fn drop(&mut self) { println(self.t); } }\n")

diff("Drop on move into a function", """    let a = N { t: 1i64 };
    take(a);
    println("after");""",
    "struct N { t: Int64 }\nimpl Drop for N { fn drop(&mut self) { println(self.t); } }\nfn take(_n: N) { println(\"taken\"); }\n")

diff("Option and Result chains", """    println(unwrap_or(Some(4i64), 0i64));
    println(unwrap_or(None, 9i64));
    match parse_ok(1i64) { Ok(v) => println(v), Err(_e) => println("err") }""",
    "fn unwrap_or(o: Option<Int64>, d: Int64) -> Int64 { match o { Some(v) => v, None => d } }\nfn parse_ok(v: Int64) -> Result<Int64, Int64> { Ok(v) }\n")

diff("the ? operator", """    match run() { Ok(v) => println(v), Err(e) => println(e) }""",
    "fn inner(v: Int64) -> Result<Int64, Int64> { if v > 0i64 { Ok(v) } else { Err(0i64 - 1i64) } }\nfn run() -> Result<Int64, Int64> { let a = inner(2i64)?; let b = inner(3i64)?; Ok(a + b) }\n")

diff("by-reference match on a non-Copy enum", """    let v = A::T(String::from("hi"));
    println(tag(&v));
    println(tag(&v));
    match v { A::T(s) => println(s.as_str()), A::F => println("f") }""",
    "enum A { F, T(String) }\nfn tag(a: &A) -> Int64 { match *a { A::F => 0i64, _other => 1i64 } }\n")

diff("mutable borrow through a function", """    let mut v: Vec<Int64> = Vec::new();
    fill(&mut v);
    println(v.len());
    println(v[0u64]);""",
    "fn fill(v: &mut Vec<Int64>) { v.push(42i64); }\n")

diff("shadowing in nested scopes", """    let x = 1i64;
    { let x = 2i64; println(x); }
    println(x);""")

diff("loop, break and continue", """    let mut total = 0i64;
    let mut i = 0i64;
    while i < 10i64 {
        i = i + 1i64;
        if i == 3i64 { continue; }
        if i == 7i64 { break; }
        total = total + i;
    }
    println(total);""")

diff("char and byte round trip", """    let s = String::from("AZaz09");
    let b = s.as_str().bytes();
    let mut i = 0u64;
    let mut sum = 0u64;
    while i < b.len() { sum = sum + b[i] as UInt64; i = i + 1u64; }
    println(sum);""")

diff("array and slice", """    let a: [Int64; 4] = [1i64, 2i64, 3i64, 4i64];
    println(sum(&a));""",
    "fn sum(s: &[Int64]) -> Int64 { let mut t = 0i64; let mut i = 0u64; while i < s.len() { t = t + s[i]; i = i + 1u64; } t }\n")

diff("recursive function", """    println(fib(15i64));""",
    "fn fib(n: Int64) -> Int64 { if n < 2i64 { n } else { fib(n - 1i64) + fib(n - 2i64) } }\n")

# ------------------------------------------------------------------- F: rejection matrix -------

reject("use after move", """struct T { v: Int64 }
fn take(_t: T) {}
fn main() { let t = T { v: 1i64 }; take(t); println(t.v); }
""")

reject("two mutable borrows at once", """fn main() {
    let mut v: Vec<Int64> = Vec::new();
    let a = &mut v;
    let b = &mut v;
    a.push(1i64);
    b.push(2i64);
}
""")

reject("mutable and shared borrow at once", """fn main() {
    let mut v: Vec<Int64> = Vec::new();
    v.push(1i64);
    let a = &v;
    let b = &mut v;
    b.push(2i64);
    println(a.len());
}
""")

reject("move out of a shared borrow", """struct T { v: String }
fn steal(t: &T) -> String { t.v }
fn main() { let t = T { v: String::from("x") }; println(steal(&t).as_str()); }
""")

reject("move out of an indexed place", """struct T { v: String }
fn main() {
    let mut xs: Vec<T> = Vec::new();
    xs.push(T { v: String::from("a") });
    let t = xs[0u64];
    println(t.v.as_str());
}
""")

reject("assign wrong type", """fn main() { let a: Int64 = 1u64; println(a); }
""")

reject("return wrong type", """fn f() -> Int64 { String::from("x") }
fn main() { println(f()); }
""")

reject("call with wrong arity", """fn f(a: Int64, b: Int64) -> Int64 { a + b }
fn main() { println(f(1i64)); }
""")

reject("field that does not exist", """struct T { v: Int64 }
fn main() { let t = T { v: 1i64 }; println(t.missing); }
""")

reject("method that does not exist", """struct T { v: Int64 }
fn main() { let t = T { v: 1i64 }; println(t.nope()); }
""")

reject("unsatisfied trait bound", """trait D { fn d(&self) -> Int64; }
struct P { n: Int64 }
fn need<T: D>(x: &T) -> Int64 { x.d() }
fn main() { let p = P { n: 1i64 }; println(need(&p)); }
""")

reject("mixing integer widths without a cast", """fn main() { let a: Int64 = 1i64; let b: Int32 = 2i32; println(a + b); }
""")

reject("Copy and Drop together", """struct T { v: Int64 }
impl Copy for T {}
impl Drop for T { fn drop(&mut self) {} }
fn main() { let _t = T { v: 1i64 }; }
""")

reject("use of an uninitialised binding", """fn main() { let a: Int64; println(a); }
""")

# ----------------------------------------------------------------------------- run and report --

print(f"SCOPE E — engine differential: {len(E_CASES)} programs")
e_bad = 0
for name, src in E_CASES:
    i_status, i_out = interp(src)
    n_status, n_out = native(src)
    agree = (i_status == n_status) and (i_out == n_out)
    if not agree:
        e_bad += 1
        print(f"!! {name}")
        print(f"     interpreter [{i_status}] {i_out!r}")
        print(f"     native      [{n_status}] {n_out!r}")
    else:
        print(f"   {name:<44} [{i_status}] agree")

print(f"\nSCOPE F — rejection matrix: {len(F_CASES)} programs")
f_bad = 0
for name, src in F_CASES:
    status, out = interp(src)
    if status == "OK":
        f_bad += 1
        print(f"!! {name:<44} ACCEPTED, printed {out!r}")
    else:
        print(f"   {name:<44} rejected: {out[:56]}")

print(f"\nTOTALS: scope E {e_bad} divergences / {len(E_CASES)}, "
      f"scope F {f_bad} wrongly accepted / {len(F_CASES)}")
