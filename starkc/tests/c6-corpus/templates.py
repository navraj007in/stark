"""WP-C6.5 §11.5 template registry — the semantic templates the generator instantiates.

Each template is a function of one dimension tuple returning a `Case`. Templates are **valid by
construction** (§11.7): a template only declares dimensions it can instantiate into an
ownership-correct program, and a front-end rejection of a generated case is a GENERATOR DEFECT, not
something to discover by asking the compiler and discarding failures.

Each case carries its own expected observation, computed here from the template's semantic model —
not read back from an engine. That independence is the point: the corpus's claim is that three
engines agree with the SPECIFICATION, and an expectation derived from one of those engines could
only ever prove they agree with each other.

Conventions the templates rely on, each verified against the implementation rather than assumed:
  * integer literals adopt their expected type (DEV-078), so `let x: UInt64 = 7;` needs no suffix;
  * `Float32`/`Float64` do NOT implement `Eq`/`Ord` (NUM-FLOAT-TRAIT-001), so float cases observe
    through `print`, never `assert_eq`;
  * there is no implicit array-to-slice coercion — a view is `&mut xs[a..b]` (TYPE-COERCE-003);
  * `print` of a reference is not lowerable, so iteration prints `*x`;
  * Drop-observing cases emit the §8.8 frame `@@stark-drop:<identity>@@`.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Case:
    """One generated case: its source, and what the specification says it must observe."""

    source: str
    category: str
    subcategories: tuple
    normative_rules: tuple
    expected_stdout: tuple = ()
    expected_outcome: str = "completion"
    expected_trap_category: str = ""
    drop_protocol: bool = False
    expected_drop_log: tuple = ()


# Signed types whose arithmetic the templates keep well inside range, and unsigned ones used only
# with non-negative results — overflow traps (NUM-INT-ARITH-001), so a template that could overflow
# would be generating a trap case by accident.
INT_TYPES = ("Int32", "Int64", "UInt32", "UInt64")
SIGNED_TYPES = ("Int32", "Int64")


def t01_arithmetic(dims):
    """T01 — arithmetic expression tree over a bounded type and operator."""
    ty, (a, b), op = dims
    result = {"+": a + b, "-": a - b, "*": a * b, "/": a // b}[op]
    source = f"""fn main() {{
    let a: {ty} = {a};
    let b: {ty} = {b};
    let c: {ty} = a {op} b;
    assert_eq(c, {result});
    print(c);
}}
"""
    return Case(
        source=source,
        category="expressions-statements",
        subcategories=("E06",),
        normative_rules=("NUM-INT-ARITH-001",),
        expected_stdout=(str(result),),
    )


def t02_comparison_branch(dims):
    """T02 — comparison and branch: both directions are generated, so neither arm is unvisited."""
    ty, (a, b), op = dims
    taken = {"<": a < b, ">": a > b, "==": a == b, "!=": a != b}[op]
    source = f"""fn main() {{
    let a: {ty} = {a};
    let b: {ty} = {b};
    if a {op} b {{
        print("yes");
    }} else {{
        print("no");
    }}
}}
"""
    return Case(
        source=source,
        category="control-transfer",
        subcategories=("C01",),
        normative_rules=("EXEC-EVAL-001", "PRIM-TRAIT-001"),
        expected_stdout=("yes" if taken else "no",),
    )


def t03_bounded_loop(dims):
    """T03 — bounded loop and accumulation. Iteration counts are small and explicit (§11.8)."""
    ty, iterations, step = dims
    total = sum(step * i for i in range(iterations))
    source = f"""fn main() {{
    let mut total: {ty} = 0;
    let mut i: {ty} = 0;
    while i < {iterations} {{
        total = total + i * {step};
        i = i + 1;
    }}
    assert_eq(total, {total});
    print(total);
}}
"""
    return Case(
        source=source,
        category="control-transfer",
        subcategories=("C04",),
        normative_rules=("EXEC-CFLOW-001", "NUM-INT-ARITH-001"),
        expected_stdout=(str(total),),
    )


def t04_match_pattern(dims):
    """T04 — match over an enum with a payload; the selected arm is the dimension."""
    selected, payload = dims
    variants = ("Zero", "One", "Two")
    arms = "\n".join(
        f"        Shape::{name}(v) => v * {index + 1}," for index, name in enumerate(variants)
    )
    result = payload * (variants.index(selected) + 1)
    source = f"""enum Shape {{ Zero(Int32), One(Int32), Two(Int32) }}
fn main() {{
    let value: Shape = Shape::{selected}({payload});
    let scaled: Int32 = match value {{
{arms}
    }};
    assert_eq(scaled, {result});
    print(scaled);
}}
"""
    return Case(
        source=source,
        category="patterns",
        subcategories=("P06", "C11"),
        normative_rules=("PAT-OWN-001", "PAT-EXHAUST-001"),
        expected_stdout=(str(result),),
    )


def t05_struct_projection(dims):
    """T05 — struct construction, projection and field update."""
    ty, (x, y), delta = dims
    source = f"""struct Point {{ x: {ty}, y: {ty} }}
fn main() {{
    let mut p: Point = Point {{ x: {x}, y: {y} }};
    p.x = p.x + {delta};
    assert_eq(p.x, {x + delta});
    assert_eq(p.y, {y});
    print(p.x);
    print(p.y);
}}
"""
    return Case(
        source=source,
        category="values-types",
        subcategories=("V12",),
        normative_rules=("TYPE-NOMINAL-001", "EXEC-ASSIGN-001"),
        expected_stdout=(f"{x + delta}{y}",),
    )


def t06_enum_payload_move(dims):
    """T06 — enum payload movement out of a consuming match."""
    present, payload = dims
    source = f"""enum Held {{ Empty, Full(String) }}
fn describe(held: Held) -> String {{
    match held {{
        Held::Empty => String::from("empty"),
        Held::Full(inner) => inner,
    }}
}}
fn main() {{
    let held: Held = {"Held::Full(String::from(\"" + payload + "\"))" if present else "Held::Empty"};
    let text: String = describe(held);
    print(text);
}}
"""
    return Case(
        source=source,
        category="values-types",
        subcategories=("V13", "O11"),
        normative_rules=("OWN-PARTIAL-001", "OWN-MOVE-001"),
        expected_stdout=(payload if present else "empty",),
    )


def t07_generic_instance(dims):
    """T07 — one generic body, two instances. The instances are the dimension."""
    first, second = dims
    source = f"""fn pick<T>(a: T, b: T, take_first: Bool) -> T {{
    if take_first {{ a }} else {{ b }}
}}
fn main() {{
    let x: {first} = 7;
    let y: {first} = 9;
    let picked: {first} = pick(x, y, true);
    assert_eq(picked, 7);
    let s: Bool = pick(true, false, {"true" if second else "false"});
    assert(s == {"true" if second else "false"});
    print(picked);
}}
"""
    return Case(
        source=source,
        category="calls-dispatch",
        subcategories=("D05",),
        normative_rules=("TYPE-GENERIC-001", "EXEC-DISPATCH-001"),
        expected_stdout=("7",),
    )


def t08_trait_dispatch(dims):
    """T08 — trait dispatch with per-impl sentinel values, so the wrong impl is visible."""
    first, second = dims
    source = f"""trait Score {{
    fn score(&self) -> Int32;
}}
struct Alpha {{ v: Int32 }}
struct Beta {{ v: Int32 }}
impl Score for Alpha {{
    fn score(&self) -> Int32 {{ {first} }}
}}
impl Score for Beta {{
    fn score(&self) -> Int32 {{ {second} }}
}}
fn main() {{
    let a: Alpha = Alpha {{ v: 0 }};
    let b: Beta = Beta {{ v: 0 }};
    assert_eq(a.score(), {first});
    assert_eq(b.score(), {second});
    print(a.score());
    print(b.score());
}}
"""
    return Case(
        source=source,
        category="calls-dispatch",
        subcategories=("D06",),
        normative_rules=("EXEC-DISPATCH-001", "TRAIT-DEF-001"),
        expected_stdout=(f"{first}{second}",),
    )


def t09_function_value(dims):
    """T09 — a function value called indirectly, with a distinct target per dimension."""
    which, argument = dims
    result = argument + 1 if which == "inc" else argument * 2
    source = f"""fn inc(v: Int32) -> Int32 {{ v + 1 }}
fn twice(v: Int32) -> Int32 {{ v * 2 }}
fn apply(f: fn(Int32) -> Int32, v: Int32) -> Int32 {{ f(v) }}
fn main() {{
    let f: fn(Int32) -> Int32 = {which};
    let out: Int32 = apply(f, {argument});
    assert_eq(out, {result});
    print(out);
}}
"""
    return Case(
        source=source,
        category="calls-dispatch",
        subcategories=("D09", "E13"),
        normative_rules=("TYPE-FN-001", "EXEC-DISPATCH-001"),
        expected_stdout=(str(result),),
    )


def t10_option_result(dims):
    """T10 — Option/Result completion, including `?` propagation."""
    kind, present, payload = dims
    if kind == "option":
        constructed = f"Some({payload})" if present else "None"
        source = f"""fn main() {{
    let value: Option<Int32> = {constructed};
    let out: Int32 = match value {{
        Some(v) => v,
        None => -1,
    }};
    assert_eq(out, {payload if present else -1});
    print(out);
}}
"""
    else:
        constructed = f"Ok({payload})" if present else 'Err(String::from("bad"))'
        source = f"""fn step() -> Result<Int32, String> {{
    {constructed}
}}
fn run() -> Result<Int32, String> {{
    let v: Int32 = step()?;
    Ok(v * 2)
}}
fn main() {{
    let out: Int32 = match run() {{
        Ok(v) => v,
        Err(e) => -1,
    }};
    assert_eq(out, {payload * 2 if present else -1});
    print(out);
}}
"""
    expected = (payload if present else -1) if kind == "option" else (payload * 2 if present else -1)
    return Case(
        source=source,
        category="values-types",
        subcategories=("V14", "V15", "C12"),
        normative_rules=("STD-PROFILE-001", "EXEC-CFLOW-001"),
        expected_stdout=(str(expected),),
    )


def t11_value_flow(dims):
    """T11 — String/Vec value flow through a call boundary and back."""
    container, count = dims
    if container == "string":
        pieces = "".join(f'    s.push_str("{i}");\n' for i in range(count))
        text = "".join(str(i) for i in range(count))
        source = f"""fn extend(mut s: String) -> String {{
{pieces}    s
}}
fn main() {{
    let base: String = String::from("x");
    let out: String = extend(base);
    print(out);
}}
"""
        expected = "x" + text
    else:
        pushes = "".join(f"    v.push({i});\n" for i in range(count))
        source = f"""fn fill(mut v: Vec<Int32>) -> Vec<Int32> {{
{pushes}    v
}}
fn main() {{
    let base: Vec<Int32> = Vec::new();
    let out: Vec<Int32> = fill(base);
    assert_eq(out.len(), {count});
    print(out.len());
}}
"""
        expected = str(count)
    return Case(
        source=source,
        category="values-types",
        subcategories=("V07", "V16"),
        normative_rules=("TEXT-UTF8-001", "DROP-COLLECTION-001"),
        expected_stdout=(expected,),
    )


def t12_collection_order(dims):
    """T12 — insertion order preserved on iteration (STD-HASH-001's CE4 amendment)."""
    keys, = dims
    inserts = "".join(f"    m.insert({k}, {i});\n" for i, k in enumerate(keys))
    source = f"""fn main() {{
    let mut m: HashMap<Int32, Int32> = HashMap::new();
{inserts}    assert_eq(m.len(), {len(keys)});
    for k in m.keys() {{
        print(*k);
    }}
}}
"""
    return Case(
        source=source,
        category="values-types",
        subcategories=("V18",),
        normative_rules=("STD-HASH-001",),
        expected_stdout=("".join(str(k) for k in keys),),
    )


def t15_drop_order(dims):
    """T15 — Drop order under normal scope exit and early transfer, observed through the protocol."""
    count, early = dims
    locals_ = "".join(
        f"        let loud{i}: Loud = Loud {{ id: {i} }};\n" for i in range(1, count + 1)
    )
    body = locals_ + ("        break;\n" if early else "")
    source = f"""struct Loud {{ id: Int32 }}
impl Drop for Loud {{
    fn drop(&mut self) {{
        print("@@stark-drop:Loud#");
        print(self.id);
        println("@@");
    }}
}}
fn main() {{
    let mut once: Int32 = 0;
    while once < 1 {{
{body}        once = once + 1;
    }}
    print("done");
}}
"""
    # Reverse declaration order (DROP-ORDER-001), whether the block exits normally or by `break`.
    return Case(
        source=source,
        category="ownership-drop",
        subcategories=("O15", "O16", "O17"),
        normative_rules=("DROP-ORDER-001", "DROP-EXACT-001"),
        expected_stdout=("done",),
        drop_protocol=True,
        expected_drop_log=tuple(f"Loud#{i}" for i in range(count, 0, -1)),
    )


def t16_trap(dims):
    """T16 — one admitted trap category per dimension, with output emitted before the trap."""
    kind, = dims
    prelude = '    print("before");\n'
    bodies = {
        "overflow": (
            "    let a: Int32 = 2147483647;\n    let b: Int32 = a + 1;\n    print(b);\n",
            "IntegerOverflow",
        ),
        "divide_by_zero": (
            "    let a: Int32 = 1;\n    let z: Int32 = 0;\n    let b: Int32 = a / z;\n    print(b);\n",
            "DivideByZero",
        ),
        "index": (
            "    let xs: [Int32; 2] = [1, 2];\n    let i: Int32 = 5;\n    print(xs[i]);\n",
            "IndexOutOfBounds",
        ),
        "cast": (
            "    let big: Int64 = 4294967296;\n    let small: Int32 = big as Int32;\n    print(small);\n",
            "CastFailure",
        ),
        "assert": ("    assert(1 == 2);\n", "AssertFailure"),
        "panic": ('    panic("generated");\n', "Panic"),
        "shift": (
            "    let a: Int32 = 1;\n    let s: Int32 = 64;\n    let b: Int32 = a << s;\n    print(b);\n",
            "InvalidShift",
        ),
    }
    body, category = bodies[kind]
    source = f"fn main() {{\n{prelude}{body}}}\n"
    return Case(
        source=source,
        category="traps",
        subcategories=("X01", "X11"),
        normative_rules=("TRAP-CATEGORY-001",),
        expected_stdout=("before",),
        expected_outcome="trap",
        expected_trap_category=category,
    )


def t20_composite_format(dims):
    """T20 — nested composite formatting: the rendering of a value one level down."""
    shape, = dims
    if shape == "tuple":
        source = """fn main() {
    let pair: (Int32, Bool) = (7, true);
    println(pair);
}
"""
        expected = "(7, true)"
    elif shape == "array":
        source = """fn main() {
    let xs: [Int32; 3] = [1, 2, 3];
    println(xs);
}
"""
        expected = "[1, 2, 3]"
    else:
        source = """fn main() {
    let value: Option<Int32> = Some(5);
    println(value);
}
"""
        expected = "Some(5)"
    return Case(
        source=source,
        category="calls-dispatch",
        subcategories=("D08",),
        normative_rules=("PRINT-DISPLAY-001", "STD-FORMAT-001"),
        expected_stdout=(expected, ""),
    )


# §11.5's registry, restricted to templates that produce valid single-file programs today. T17-T19
# (multi-file, relocation, offline build) need package graphs, which arrive with §15 — recorded as
# absent rather than stubbed, so `--list-templates` never implies coverage that does not exist.
TEMPLATES = {
    "T01": (t01_arithmetic, [
        (ty, values, op)
        for ty in INT_TYPES
        for values in ((12, 4), (100, 7), (9, 3))
        for op in ("+", "-", "*", "/")
        # unsigned subtraction must stay non-negative: a negative result would trap, and this
        # template is a completion template
        if not (op == "-" and ty.startswith("UInt") and values[0] < values[1])
    ]),
    "T02": (t02_comparison_branch, [
        (ty, values, op)
        for ty in INT_TYPES
        for values in ((3, 5), (5, 3), (4, 4))
        for op in ("<", ">", "==", "!=")
    ]),
    "T03": (t03_bounded_loop, [
        (ty, iterations, step)
        for ty in SIGNED_TYPES
        for iterations in (0, 1, 5)
        for step in (1, 3)
    ]),
    "T04": (t04_match_pattern, [
        (selected, payload) for selected in ("Zero", "One", "Two") for payload in (0, 7, -3)
    ]),
    "T05": (t05_struct_projection, [
        (ty, values, delta)
        for ty in SIGNED_TYPES
        for values in ((1, 2), (10, 20))
        for delta in (0, 5)
    ]),
    "T06": (t06_enum_payload_move, [
        (present, payload) for present in (True, False) for payload in ("held", "value")
    ]),
    "T07": (t07_generic_instance, [
        (first, second) for first in SIGNED_TYPES for second in (True, False)
    ]),
    "T08": (t08_trait_dispatch, [
        (first, second) for first in (11, 31) for second in (22, 44)
    ]),
    "T09": (t09_function_value, [
        (which, argument) for which in ("inc", "twice") for argument in (0, 3, 21)
    ]),
    "T10": (t10_option_result, [
        (kind, present, payload)
        for kind in ("option", "result")
        for present in (True, False)
        for payload in (4, 11)
    ]),
    "T11": (t11_value_flow, [
        (container, count) for container in ("string", "vec") for count in (0, 1, 3)
    ]),
    "T12": (t12_collection_order, [
        ((30, 10, 20),), ((1, 2, 3),), ((7,),), ((5, 4),), ((9, 1, 8, 2),),
    ]),
    "T15": (t15_drop_order, [(count, early) for count in (1, 2, 3) for early in (False, True)]),
    "T16": (t16_trap, [
        ("overflow",), ("divide_by_zero",), ("index",), ("cast",), ("assert",), ("panic",),
        ("shift",),
    ]),
    "T20": (t20_composite_format, [("tuple",), ("array",), ("option",)]),
}

MISSING_TEMPLATES = {
    "T13": "borrow/reborrow/reference return — needs provenance dimensions; handwritten today",
    "T14": "partial move and reinitialisation — handwritten today",
    "T17": "multi-file/package call graph — needs package graphs (§15)",
    "T18": "relocation/dependency reorder identity — needs package graphs (§15)",
    "T19": "locked/offline generated build — needs package graphs (§15)",
}
