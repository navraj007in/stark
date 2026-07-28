//! WP-C6.6 — the normative stdlib surface audit.
//!
//! Gate C6's exit criterion is not "the tests pass". It is:
//!
//! > C6 closes only when native execution is semantically credible, not merely faster.
//! > **Every unsupported normative Core feature must block a full native-conformance claim.**
//!
//! That needs an ENUMERATION, and until now there was none. Coverage was tracked per TYPE in the
//! C6.5 matrix, and DEV-115 and DEV-116 both showed a per-type row hiding a method-level gap:
//! `str` was "covered" while `str::bytes` diverged across engines, and `HashSet<T>` was a single
//! row while all eight of its methods were unlowerable. Neither was found by inspection — one
//! surfaced from reviewing an unrelated diff, the other from a coverage row that had no case.
//!
//! So this walks `06-Standard-Library.md`'s impl blocks method by method, constructs a minimal call
//! for each, and records which of three things happens: it lowers and verifies, MIR refuses it, or
//! no call could be constructed. The last category is reported rather than skipped — an unprobed
//! method is not a passing one.
//!
//! **Owner decisions recorded (CD-181), so no refusal here is merely undescribed.**
//!
//! * **`File` (5 methods) — EXCLUDED from the C6 native executable subset.** It needs an effectful
//!   host/provider contract, filesystem error semantics, and a way to compare or control
//!   environmental observations across engines. Deferred to the I/O/runtime package gate.
//! * **`Random` (4 methods) — EXCLUDED pending a normative PRNG algorithm and a cross-engine
//!   sequence contract.** Explicitly NOT justified as "nondeterminism": a seeded generator is
//!   perfectly reproducible. What is undecided is which algorithm is normative, whether identical
//!   seeds must yield identical sequences, whether the implementation is language- or
//!   provider-defined, and how secure randomness is separated from deterministic PRNG.
//! * The remaining refusals are CARRIED method-level work, not blockers:
//!   `WP-C7-String-Surface` (10), `WP-C7-Vec-Completion` (3), `WP-C7-HashMap-Completion` (4 —
//!   `with_capacity`, `get_mut`, `values`, `iter`; `remove`/`clear` landed in CD-180).
//!
//! **This test does not fail on a refusal.** A refusal is a fact to be recorded and bounded, not a
//! regression: several are deliberate (`File`/`Random` are IO and nondeterminism, outside the
//! native subset). It fails when the EXECUTABLE count drops, or when the surface changes shape
//! without the baseline moving — so the number can only go up, and it cannot go up silently.

mod support;
use support::differential::front_end;

/// (impl block, method, program body). `None` = no call could be constructed; recorded, not skipped.
fn surface() -> Vec<(&'static str, &'static str, Option<&'static str>)> {
    vec![
    ("Box<T>","new",Some("let b: Box<Int32> = Box::new(1);")),
    ("Box<T>","into_inner",Some("let b: Box<Int32> = Box::new(1); let v: Int32 = b.into_inner();")),
    ("Option<T>","is_some",Some("let o: Option<Int32> = Some(1); let a: Bool = o.is_some();")),
    ("Option<T>","is_none",Some("let o: Option<Int32> = Some(1); let a: Bool = o.is_none();")),
    ("Option<T>","unwrap",Some("let o: Option<Int32> = Some(1); let a: Int32 = o.unwrap();")),
    ("Option<T>","unwrap_or",Some("let o: Option<Int32> = Some(1); let a: Int32 = o.unwrap_or(2);")),
    ("Result<T,E>","is_ok",Some("let r: Result<Int32, String> = Ok(1); let a: Bool = r.is_ok();")),
    ("Result<T,E>","is_err",Some("let r: Result<Int32, String> = Ok(1); let a: Bool = r.is_err();")),
    ("Result<T,E>","unwrap",Some("let r: Result<Int32, String> = Ok(1); let a: Int32 = r.unwrap();")),
    ("Result<T,E>","unwrap_or",Some("let r: Result<Int32, String> = Ok(1); let a: Int32 = r.unwrap_or(2);")),
    ("Vec<T>","new",Some("let v: Vec<Int32> = Vec::new();")),
    ("Vec<T>","with_capacity",Some("let v: Vec<Int32> = Vec::with_capacity(4u64);")),
    ("Vec<T>","push",Some("let mut v: Vec<Int32> = Vec::new(); v.push(1);")),
    ("Vec<T>","pop",Some("let mut v: Vec<Int32> = Vec::new(); let a: Option<Int32> = v.pop();")),
    ("Vec<T>","len",Some("let v: Vec<Int32> = Vec::new(); let a: UInt64 = v.len();")),
    ("Vec<T>","capacity",Some("let v: Vec<Int32> = Vec::new(); let a: UInt64 = v.capacity();")),
    ("Vec<T>","is_empty",Some("let v: Vec<Int32> = Vec::new(); let a: Bool = v.is_empty();")),
    ("Vec<T>","get",Some("let v: Vec<Int32> = Vec::new(); let a: Option<&Int32> = v.get(0u64);")),
    ("Vec<T>","get_mut",Some("let mut v: Vec<Int32> = Vec::new(); let a: Option<&mut Int32> = v.get_mut(0u64);")),
    ("Vec<T>","insert",Some("let mut v: Vec<Int32> = Vec::new(); v.insert(0u64, 1);")),
    ("Vec<T>","remove",Some("let mut v: Vec<Int32> = Vec::new(); v.push(1); let a: Int32 = v.remove(0u64);")),
    ("Vec<T>","clear",Some("let mut v: Vec<Int32> = Vec::new(); v.clear();")),
    ("Vec<T>","append",Some("let mut a: Vec<Int32> = Vec::new(); let mut b: Vec<Int32> = Vec::new(); a.append(&mut b);")),
    ("Vec<T>","iter",Some("let v: Vec<Int32> = Vec::new(); for x in v.iter() { print(*x); }")),
    ("Vec<T>","as_slice",Some("let v: Vec<Int32> = Vec::new(); let s = v.as_slice();")),
    ("Index for Vec","index",Some("let mut v: Vec<Int32> = Vec::new(); v.push(1); let a: Int32 = v[0u64];")),
    ("IndexMut for Vec","index_mut",Some("let mut v: Vec<Int32> = Vec::new(); v.push(1); v[0u64] = 2;")),
    ("HashMap<K,V>","new",Some("let m: HashMap<Int32, Int32> = HashMap::new();")),
    ("HashMap<K,V>","with_capacity",Some("let m: HashMap<Int32, Int32> = HashMap::with_capacity(4u64);")),
    ("HashMap<K,V>","insert",Some("let mut m: HashMap<Int32, Int32> = HashMap::new(); m.insert(1, 2);")),
    ("HashMap<K,V>","get",Some("let m: HashMap<Int32, Int32> = HashMap::new(); let a: Option<&Int32> = m.get(&1);")),
    ("HashMap<K,V>","get_mut",Some("let mut m: HashMap<Int32, Int32> = HashMap::new(); let a: Option<&mut Int32> = m.get_mut(&1);")),
    ("HashMap<K,V>","remove",Some("let mut m: HashMap<Int32, Int32> = HashMap::new(); let a: Option<Int32> = m.remove(&1);")),
    ("HashMap<K,V>","contains_key",Some("let m: HashMap<Int32, Int32> = HashMap::new(); let a: Bool = m.contains_key(&1);")),
    ("HashMap<K,V>","len",Some("let m: HashMap<Int32, Int32> = HashMap::new(); let a: UInt64 = m.len();")),
    ("HashMap<K,V>","is_empty",Some("let m: HashMap<Int32, Int32> = HashMap::new(); let a: Bool = m.is_empty();")),
    ("HashMap<K,V>","clear",Some("let mut m: HashMap<Int32, Int32> = HashMap::new(); m.clear();")),
    ("HashMap<K,V>","keys",Some("let m: HashMap<Int32, Int32> = HashMap::new(); for k in m.keys() { print(*k); }")),
    ("HashMap<K,V>","values",Some("let m: HashMap<Int32, Int32> = HashMap::new(); for v in m.values() { print(*v); }")),
    ("HashMap<K,V>","iter",Some("let m: HashMap<Int32, Int32> = HashMap::new(); for e in m.iter() { }")),
    ("HashSet<T>","new",Some("let s: HashSet<Int32> = HashSet::new();")),
    ("HashSet<T>","insert",Some("let mut s: HashSet<Int32> = HashSet::new(); let a: Bool = s.insert(1);")),
    ("HashSet<T>","remove",Some("let mut s: HashSet<Int32> = HashSet::new(); let a: Bool = s.remove(&1);")),
    ("HashSet<T>","contains",Some("let s: HashSet<Int32> = HashSet::new(); let a: Bool = s.contains(&1);")),
    ("HashSet<T>","len",Some("let s: HashSet<Int32> = HashSet::new(); let a: UInt64 = s.len();")),
    ("HashSet<T>","is_empty",Some("let s: HashSet<Int32> = HashSet::new(); let a: Bool = s.is_empty();")),
    ("HashSet<T>","clear",Some("let mut s: HashSet<Int32> = HashSet::new(); s.clear();")),
    ("HashSet<T>","iter",Some("let s: HashSet<Int32> = HashSet::new(); for x in s.iter() { print(*x); }")),
    ("String","new",Some("let s: String = String::new();")),
    ("String","with_capacity",Some("let s: String = String::with_capacity(4u64);")),
    ("String","from",Some("let s: String = String::from(\"a\");")),
    ("String","len",Some("let s: String = String::new(); let a: UInt64 = s.len();")),
    ("String","is_empty",Some("let s: String = String::new(); let a: Bool = s.is_empty();")),
    ("String","push",Some("let mut s: String = String::new(); s.push('a');")),
    ("String","push_str",Some("let mut s: String = String::new(); s.push_str(\"a\");")),
    ("String","pop",Some("let mut s: String = String::new(); let a: Option<Char> = s.pop();")),
    ("String","clear",Some("let mut s: String = String::new(); s.clear();")),
    ("String","chars",Some("let s: String = String::new(); for c in s.chars() { print(c); }")),
    ("String","bytes",Some("let s: String = String::new(); let b = s.bytes();")),
    ("String","as_str",Some("let s: String = String::new(); let a = s.as_str();")),
    ("String","into_bytes",Some("let s: String = String::new(); let b = s.into_bytes();")),
    ("String","substring",Some("let s: String = String::new(); let a = s.substring(0u64, 0u64);")),
    ("String","contains",Some("let s: String = String::new(); let a: Bool = s.contains(\"x\");")),
    ("String","starts_with",Some("let s: String = String::new(); let a: Bool = s.starts_with(\"x\");")),
    ("String","ends_with",Some("let s: String = String::new(); let a: Bool = s.ends_with(\"x\");")),
    ("String","find",Some("let s: String = String::new(); let a: Option<UInt64> = s.find(\"x\");")),
    ("String","replace",Some("let s: String = String::new(); let a: String = s.replace(\"x\", \"y\");")),
    ("String","split",Some("let s: String = String::new(); for p in s.split(\",\") { }")),
    ("String","trim",Some("let s: String = String::new(); let a = s.trim();")),
    ("String","to_lowercase",Some("let s: String = String::new(); let a: String = s.to_lowercase();")),
    ("String","to_uppercase",Some("let s: String = String::new(); let a: String = s.to_uppercase();")),
    ("str","len",Some("let s = \"a\"; let a: UInt64 = s.len();")),
    ("str","is_empty",Some("let s = \"a\"; let a: Bool = s.is_empty();")),
    ("str","chars",Some("let s = \"a\"; for c in s.chars() { print(c); }")),
    ("str","bytes",Some("let s = \"a\"; let b = s.bytes();")),
    ("str","to_string",Some("let s = \"a\"; let a: String = s.to_string();")),
    ("Iterator for Range","next",Some("for x in 0..3 { print(x); }")),
    ("Random","new",Some("let r: Random = Random::new(1u64);")),
    ("Random","next_int",Some("let mut r: Random = Random::new(1u64); let a: UInt64 = r.next_int();")),
    ("Random","next_float",Some("let mut r: Random = Random::new(1u64); let a: Float64 = r.next_float();")),
    ("Random","range",Some("let mut r: Random = Random::new(1u64); let a: Int32 = r.range(0, 2);")),
    ("File","open",Some("let f: Result<File, IOError> = File::open(\"p\");")),
    ("File","create",Some("let f: Result<File, IOError> = File::create(\"p\");")),
    ("File","read_to_string",Some("let mut f: File = File::open(\"p\").unwrap(); let a: Result<String, IOError> = f.read_to_string();")),
    ("File","write",Some("let a: [UInt8; 1] = [1u8]; let d = &a[0..1]; let mut f: File = File::create(\"p\").unwrap(); let r: Result<UInt64, IOError> = f.write(d);")),
    ("File","write_str",Some("let mut f: File = File::create(\"p\").unwrap(); let a: Result<UInt64, IOError> = f.write_str(\"x\");")),
    ("File","close",Some("let f: File = File::open(\"p\").unwrap(); let a: Result<Unit, IOError> = f.close();")),
    ]
}

#[test]
fn audit() {
    let mut lowers = Vec::new();
    let mut refused = Vec::new();
    let mut front_rejected = Vec::new();
    let mut unconstructed = Vec::new();
    for (block, method, body) in surface() {
        let Some(body) = body else {
            unconstructed.push(format!("{block}::{method}"));
            continue;
        };
        let source = format!("fn main() {{ {body} }}\n");
        let name = format!(
            "audit_{}_{}.stark",
            block.replace(['<', '>', ',', '/', ' '], ""),
            method
        );
        let file = std::sync::Arc::new(starkc::source::SourceFile::new(name, source.clone()));
        let (ast, pd) = starkc::parser::parse(&file, starkc::parser::ParseMode::Program);
        let (hir, rd) = starkc::resolve::resolve(&ast, file.clone());
        let checked = starkc::typecheck::analyze(&hir, file.clone());
        let errs: Vec<_> = checked
            .diagnostics
            .iter()
            .filter(|d| d.severity == starkc::diag::Severity::Error)
            .collect();
        if !pd.is_empty() || !rd.is_empty() || !errs.is_empty() {
            let why = errs
                .first()
                .map(|d| d.message.clone())
                .or_else(|| rd.first().map(|d| d.message.clone()))
                .or_else(|| pd.first().map(|d| d.message.clone()))
                .unwrap_or_default();
            front_rejected.push(format!("{block}::{method} — {}", &why[..why.len().min(70)]));
            continue;
        }
        match starkc::mir::lower::lower_program(&hir, &checked.tables, file.clone()) {
            Ok(p) => match starkc::mir::verify::verify_program(&p) {
                Ok(_) => lowers.push(format!("{block}::{method}")),
                Err(e) => refused.push(format!(
                    "{block}::{method} — VERIFY: {}",
                    &e[0].message[..e[0].message.len().min(60)]
                )),
            },
            Err(e) => refused.push(format!(
                "{block}::{method} — LOWER: {}",
                &e.what[..e.what.len().min(60)]
            )),
        }
    }
    let total = lowers.len() + refused.len() + front_rejected.len() + unconstructed.len();
    println!("\n===== NORMATIVE STDLIB SURFACE AUDIT ({total} probes) =====");
    println!("\n-- EXECUTABLE (lowers + verifies): {} --", lowers.len());
    println!("-- REFUSED by MIR: {} --", refused.len());
    for r in &refused {
        println!("     {r}");
    }
    println!("-- REJECTED by the front end: {} --", front_rejected.len());
    for r in &front_rejected {
        println!("     {r}");
    }
    println!("-- NO CALL CONSTRUCTED: {} --", unconstructed.len());
    for r in &unconstructed {
        println!("     {r}");
    }

    // The ratchet. `EXECUTABLE` may only rise; `TOTAL` pins the surface so a probe silently
    // disappearing cannot make the ratio look better.
    const EXECUTABLE: usize = 59;
    const TOTAL: usize = 87;
    assert_eq!(
        total, TOTAL,
        "the probed surface changed size — update TOTAL deliberately"
    );
    assert!(
        lowers.len() >= EXECUTABLE,
        "{} methods lower, down from the recorded {EXECUTABLE} — a previously executable method \
         regressed:\n  {}",
        lowers.len(),
        refused.join("\n  ")
    );
    assert_eq!(
        lowers.len(),
        EXECUTABLE,
        "{} methods now lower, up from {EXECUTABLE}. Raise EXECUTABLE in the same change so the \
         improvement is recorded rather than absorbed.",
        lowers.len()
    );
    assert!(
        front_rejected.is_empty(),
        "a probe was rejected by the FRONT END, which means the probe itself is wrong (a bad \
         signature), not that the method is unsupported:\n  {}",
        front_rejected.join("\n  ")
    );
}
