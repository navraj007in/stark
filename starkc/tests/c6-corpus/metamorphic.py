"""WP-C6.5 §13 — metamorphic families: pairs that must observe identically.

A metamorphic pair is two sources related by a transformation the specification makes
semantics-preserving. The claim is not "both run" but "both produce the *same* normative
observation, in every engine" — so a pair is evidence about the compiler's sensitivity to source
form, which no single case can be.

Two rules shape everything here:

1. **A transformation must be semantics-preserving by a NAMED rule** (§13.3), not by intuition. Each
   group records its precondition, and the preconditions are real constraints: scope insertion is
   applied only to bases with no `Drop` impl, because an extra block changes destruction timing
   (DROP-ORDER-001) and would make the pair unequal for a correct compiler. Arm reordering is applied
   only to non-overlapping arms with no catch-all (§13.5). Loop-form equivalence is applied only where
   ownership and Drop timing are identical (§13.6).
2. **A transformation that changes nothing is a fake pair.** Every transform asserts that it actually
   rewrote the source; an identity transform would produce two identical files that trivially agree
   and prove nothing.

The bases are authored here rather than reused from the generated corpus so that identifier names,
field names and arm order are known exactly — a mechanised rename or reorder over source it does not
control is how a transformation silently becomes a no-op.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Group:
    family_id: str
    group_id: str
    base: str
    transformed: str
    precondition: str
    normative_rules: tuple
    category: str
    expected_stdout: tuple


# --------------------------------------------------------------------- bases --

SCALAR = """fn main() {
    let first: Int32 = 6;
    let second: Int32 = 7;
    let product: Int32 = first * second;
    assert_eq(product, 42);
    print(product);
}
"""

def struct_fields(x: int, y: int) -> str:
    """Parameterised, not string-replaced after the fact: a blind `.replace("3", "8")` over this
    source turned `Int32` into `Int82`, and a blind replace over the loop base produced an identity
    transform. Both were caught, but the fix is to stop generating variants by substring surgery."""
    return f"""struct Point {{ x: Int32, y: Int32 }}
fn main() {{
    let x: Int32 = {x};
    let y: Int32 = {y};
    let p: Point = Point {{ x: x, y: y }};
    assert_eq(p.x + p.y, {x + y});
    print(p.x);
    print(p.y);
}}
"""


STRUCT_FIELDS = struct_fields(3, 4)

def enum_arms(selected: str, expected: int) -> str:
    return f"""enum Colour {{ Red, Green, Blue }}
fn main() {{
    let c: Colour = Colour::{selected};
    let code: Int32 = match c {{
        Colour::Red => 1,
        Colour::Green => 2,
        Colour::Blue => 3,
    }};
    assert_eq(code, {expected});
    print(code);
}}
"""


ENUM_ARMS = enum_arms("Green", 2)

def nested_pattern(scrutinee: str, expected: int) -> str:
    return f"""fn main() {{
    let value: Option<Option<Int32>> = {scrutinee};
    let out: Int32 = match value {{
        Some(inner) => match inner {{
            Some(v) => v,
            None => -1,
        }},
        None => -2,
    }};
    assert_eq(out, {expected});
    print(out);
}}
"""


def sequential_pattern_source(scrutinee: str, expected: int) -> str:
    """M06's transformed member: a nested `match` becomes two sequential matches over the same
    scrutinee shape. PAT-OWN-001 evaluates the scrutinee once in both forms, and no arm binding
    outlives its arm."""
    return f"""fn main() {{
    let value: Option<Option<Int32>> = {scrutinee};
    let inner: Option<Int32> = match value {{
        Some(inner) => inner,
        None => None,
    }};
    let out: Int32 = match inner {{
        Some(v) => v,
        None => -1,
    }};
    assert_eq(out, {expected});
    print(out);
}}
"""


NESTED_PATTERN = nested_pattern("Some(Some(5))", 5)

GENERIC_CALL = """fn identity<T>(value: T) -> T {
    value
}
fn main() {
    let a: Int32 = 9;
    let b: Int32 = identity(a);
    assert_eq(b, 9);
    print(b);
}
"""

def trait_call(sentinel: int) -> str:
    return f"""trait Weight {{
    fn weight(&self) -> Int32;
}}
struct Stone {{ v: Int32 }}
impl Weight for Stone {{
    fn weight(&self) -> Int32 {{ {sentinel} }}
}}
fn main() {{
    let s: Stone = Stone {{ v: 0 }};
    let w: Int32 = s.weight();
    assert_eq(w, {sentinel});
    print(w);
}}
"""


TRAIT_CALL = trait_call(14)

def direct_call(body: str, expected: int) -> str:
    return f"""fn triple(v: Int32) -> Int32 {{
    {body}
}}
fn main() {{
    let out: Int32 = triple(5);
    assert_eq(out, {expected});
    print(out);
}}
"""


DIRECT_CALL = direct_call("v * 3", 15)

def counting_loop(accumulation: str, expected: int) -> str:
    """A `while`-counted accumulation, parameterised so the M12 transform can anchor on the exact
    text it rewrites. The first draft string-replaced the accumulation AFTER building the base, which
    broke the transform's anchor and produced an identity transform — caught by the assertion in
    `add`, which is what that assertion is for."""
    return f"""fn main() {{
    let mut total: Int32 = 0;
    let mut i: Int32 = 0;
    while i < 4 {{
        total = {accumulation};
        i = i + 1;
    }}
    assert_eq(total, {expected});
    print(total);
}}
"""


COUNTING_LOOP = counting_loop("total + i", 6)

STRING_FLOW = """fn main() {
    let mut text: String = String::from("ab");
    text.push_str("cd");
    let length: UInt64 = text.len();
    assert_eq(length, 4);
    print(text);
}
"""

COLLECTION = """fn main() {
    let mut values: Vec<Int32> = Vec::new();
    values.push(2);
    values.push(3);
    let count: UInt64 = values.len();
    assert_eq(count, 2);
    print(count);
}
"""


# ---------------------------------------------------------- transformations --


def rename(source: str, pairs) -> str:
    """M01 alpha-renaming. Applied only to identifiers this module declared, so the rename cannot
    collide with a keyword, a standard item or a field name it does not know about."""
    out = source
    for old, new in pairs:
        out = out.replace(old, new)
    return out


def wrap_body(source: str) -> str:
    """M02 harmless scope insertion: an extra block around the body's statements.

    Precondition: the base declares no `Drop` type. With a `Drop` impl this transformation is NOT
    semantics-preserving — the inner block ends earlier, so destruction moves (DROP-ORDER-001) and a
    correct compiler would produce different observations. That is exactly why the precondition is
    recorded rather than assumed."""
    assert "impl Drop" not in source, "scope insertion is invalid for a Drop-bearing base"
    open_at = source.index("fn main() {") + len("fn main() {")
    close_at = source.rindex("}")
    body = source[open_at:close_at]
    indented = "\n".join(
        ("    " + line if line.strip() else line) for line in body.strip("\n").split("\n")
    )
    return source[:open_at] + "\n    {\n" + indented + "\n    }\n" + source[close_at:]


def reverse_arms(source: str, first: str, last: str) -> str:
    """M07 non-overlapping arm reorder (§13.5).

    Precondition: the arms match distinct enum variants and there is no catch-all, so no arm shadows
    another and order cannot change which arm is selected (PAT-USEFUL-001). A wildcard would make the
    reorder change behaviour, which §13.5 forbids outright."""
    assert " _ =>" not in source and "_ =>" not in source, "cannot reorder past a catch-all"
    lines = source.split("\n")
    start = next(i for i, line in enumerate(lines) if first in line and "=>" in line)
    end = next(i for i, line in enumerate(lines) if last in line and "=>" in line)
    lines[start : end + 1] = list(reversed(lines[start : end + 1]))
    return "\n".join(lines)


def qualify_trait_call(source: str) -> str:
    """M04 qualified versus unqualified trait call: `s.weight()` becomes `Weight::weight(&s)`.
    TRAIT-ASSOC-001: fully qualified syntax selects the named trait, and with one impl in scope the
    two spellings resolve to the same instance."""
    return source.replace("s.weight()", "Weight::weight(&s)")


def add_turbofish(source: str) -> str:
    """M03 explicit versus inferred generics. TYPE-GENERIC-001: explicit arguments bind the same
    parameters inference would have, so the two forms select one instance."""
    return source.replace("identity(a)", "identity::<Int32>(a)")


def use_function_value(source: str) -> str:
    """M11 direct call versus equivalent function value. TYPE-FN-001 makes the value non-capturing
    and EXEC-DISPATCH-001 requires the indirect call to reach the same function."""
    return source.replace(
        "    let out: Int32 = triple(5);",
        "    let f: fn(Int32) -> Int32 = triple;\n    let out: Int32 = f(5);",
    )


def counting_loop_to_for(source: str, accumulation: str = "total + i") -> str:
    """M12 equivalent loop forms (§13.6).

    Precondition: the loop body owns nothing and drops nothing, and the counter is `Copy`, so
    `while` and a range `for` have identical ownership and Drop timing. Over an owning collection the
    two forms differ in destruction timing and the equivalence would be false."""
    assert "String" not in source and "Vec" not in source and "impl Drop" not in source
    before = f"""    let mut i: Int32 = 0;
    while i < 4 {{
        total = {accumulation};
        i = i + 1;
    }}"""
    assert before in source, "the loop transform's anchor does not match its base"
    return source.replace(
        before,
        f"""    for i in 0..4 {{
        total = {accumulation};
    }}""",
    )


def extract_helper(source: str, expression: str, helper: str, call: str) -> str:
    """M10 helper extraction: an expression becomes a call to a function containing it. EXEC-EVAL-001
    keeps evaluation order, and the extracted body is evaluated exactly once in both forms."""
    assert expression in source
    return helper + source.replace(expression, call)


def shorthand_fields(source: str) -> str:
    """M05 shorthand versus explicit fields: `Point { x: x, y: y }` becomes `Point { x, y }`. Both
    forms initialise the same fields from the same locals; declaration order is unchanged, so
    EXEC-AGG-001's field-completion order is identical."""
    return source.replace("Point { x: x, y: y }", "Point { x, y }")


# ------------------------------------------------------------------ registry --

def groups() -> list:
    """The 20 groups this phase delivers: two per family for ten families.

    **M08 (workspace relocation) and M09 (dependency declaration reorder) are absent**, and cannot be
    built here: both transform a PACKAGE GRAPH, and every corpus case is single-file until §15. They
    are recorded rather than approximated — a single-file "relocation" would be a pair that proves
    nothing about relocation."""
    out = []

    def add(family, group, base, transformed, precondition, rules, category, stdout):
        assert transformed != base, (
            f"{family}/{group}: the transformation changed nothing — an identity transform is a "
            f"fake pair that agrees trivially"
        )
        out.append(
            Group(family, group, base, transformed, precondition, rules, category, stdout)
        )

    # M01 alpha-renaming
    add("M01", "g1", SCALAR, rename(SCALAR, [("first", "alpha"), ("second", "beta"),
                                             ("product", "gamma")]),
        "locals only; no shadowing introduced and no name collides with a standard item "
        "(NAME-SHADOW-001)",
        ("NAME-RESOLVE-001", "NAME-SHADOW-001"), "expressions-statements", ("42",))
    add("M01", "g2", STRING_FLOW, rename(STRING_FLOW, [("text", "buffer"), ("length", "size")]),
        "locals only, including an owning `String` local whose Drop point is unchanged by renaming",
        ("NAME-RESOLVE-001", "TEXT-UTF8-001"), "values-types", ("abcd",))

    # M02 harmless scope insertion
    add("M02", "g1", SCALAR, wrap_body(SCALAR),
        "no `Drop` type in the base, so the earlier end of the inserted block moves no destruction "
        "(DROP-ORDER-001)",
        ("DROP-ORDER-001", "NAME-SCOPE-001"), "expressions-statements", ("42",))
    add("M02", "g2", COLLECTION, wrap_body(COLLECTION),
        "the `Vec` local is declared inside the inserted block in both forms, so its destruction "
        "point is the same statement (DROP-COLLECTION-001)",
        ("NAME-SCOPE-001", "DROP-COLLECTION-001"), "values-types", ("2",))

    # M03 explicit versus inferred generics
    add("M03", "g1", GENERIC_CALL, add_turbofish(GENERIC_CALL),
        "the explicit argument is exactly what inference selects (TYPE-GENERIC-001)",
        ("TYPE-GENERIC-001", "TYPE-INFER-001"), "calls-dispatch", ("9",))
    wide = GENERIC_CALL.replace("Int32", "Int64")
    add("M03", "g2", wide, wide.replace("identity(a)", "identity::<Int64>(a)"),
        "same, at a second width — a rule that held only at one type would not be a rule",
        ("TYPE-GENERIC-001", "TYPE-INFER-001"), "calls-dispatch", ("9",))

    # M04 qualified versus unqualified trait call
    add("M04", "g1", TRAIT_CALL, qualify_trait_call(TRAIT_CALL),
        "exactly one impl is in scope, so both spellings select the same instance "
        "(TRAIT-ASSOC-001)",
        ("TRAIT-ASSOC-001", "EXEC-DISPATCH-001"), "calls-dispatch", ("14",))
    add("M04", "g2", trait_call(21), qualify_trait_call(trait_call(21)),
        "same precondition, second sentinel value — so a pair that passed by returning a constant "
        "the harness expected anyway is distinguishable",
        ("TRAIT-ASSOC-001", "EXEC-DISPATCH-001"), "calls-dispatch", ("21",))

    # M05 shorthand versus explicit fields
    add("M05", "g1", STRUCT_FIELDS, shorthand_fields(STRUCT_FIELDS),
        "field names equal the local names and declaration order is unchanged, so field completion "
        "order is identical (EXEC-AGG-001)",
        ("EXEC-AGG-001", "TYPE-NOMINAL-001"), "values-types", ("34",))
    add("M05", "g2", struct_fields(8, 9), shorthand_fields(struct_fields(8, 9)),
        "same, at different values",
        ("EXEC-AGG-001", "TYPE-NOMINAL-001"), "values-types", ("89",))

    # M06 equivalent pattern decomposition
    add("M06", "g1", NESTED_PATTERN, sequential_pattern_source("Some(Some(5))", 5),
        "the scrutinee is `Copy` and evaluated once in both forms; no arm binding outlives its arm "
        "(PAT-OWN-001)",
        ("PAT-OWN-001", "PAT-EXHAUST-001"), "patterns", ("5",))
    add("M06", "g2", nested_pattern("Some(None)", -1), sequential_pattern_source("Some(None)", -1),
        "same, on the inner-None path, so the pair covers a second arm of the decomposition",
        ("PAT-OWN-001", "PAT-EXHAUST-001"), "patterns", ("-1",))

    # M07 non-overlapping arm reorder
    add("M07", "g1", ENUM_ARMS, reverse_arms(ENUM_ARMS, "Colour::Red", "Colour::Blue"),
        "three distinct variants, no catch-all, so no arm shadows another (PAT-USEFUL-001, §13.5)",
        ("PAT-USEFUL-001", "PAT-EXHAUST-001"), "patterns", ("2",))
    add("M07", "g2", enum_arms("Blue", 3),
        reverse_arms(enum_arms("Blue", 3), "Colour::Red", "Colour::Blue"),
        "same, selecting the last arm before the reorder and the first after it — the case that would "
        "expose an order-sensitive selection",
        ("PAT-USEFUL-001", "PAT-EXHAUST-001"), "patterns", ("3",))

    # M10 helper extraction
    add("M10", "g1", SCALAR,
        extract_helper(SCALAR, "first * second",
                       "fn multiply(a: Int32, b: Int32) -> Int32 {\n    a * b\n}\n",
                       "multiply(first, second)"),
        "the extracted expression has no side effects and is evaluated exactly once in both forms "
        "(EXEC-ONCE-001)",
        ("EXEC-ONCE-001", "EXEC-DISPATCH-001"), "calls-dispatch", ("42",))
    add("M10", "g2", COUNTING_LOOP,
        extract_helper(COUNTING_LOOP, "total + i",
                       "fn add(a: Int32, b: Int32) -> Int32 {\n    a + b\n}\n",
                       "add(total, i)"),
        "extraction inside a loop body: the call happens once per iteration in both forms "
        "(EXEC-ONCE-001)",
        ("EXEC-ONCE-001", "EXEC-CFLOW-001"), "control-transfer", ("6",))

    # M11 direct call versus function value
    add("M11", "g1", DIRECT_CALL, use_function_value(DIRECT_CALL),
        "the function value is non-capturing and the indirect call reaches the same function "
        "(TYPE-FN-001, EXEC-DISPATCH-001)",
        ("TYPE-FN-001", "EXEC-DISPATCH-001"), "calls-dispatch", ("15",))
    add("M11", "g2", direct_call("v + 30", 35), use_function_value(direct_call("v + 30", 35)),
        "same, with a different target body so the pair is not satisfied by a constant",
        ("TYPE-FN-001", "EXEC-DISPATCH-001"), "calls-dispatch", ("35",))

    # M12 equivalent loop forms
    add("M12", "g1", COUNTING_LOOP, counting_loop_to_for(COUNTING_LOOP),
        "the body owns and drops nothing and the counter is `Copy`, so `while` and a range `for` have "
        "identical ownership and Drop timing (§13.6, DROP-LOOP-001)",
        ("EXEC-FOR-001", "DROP-LOOP-001"), "control-transfer", ("6",))
    doubled = counting_loop("total + i * 2", 12)
    add("M12", "g2", doubled, counting_loop_to_for(doubled, "total + i * 2"),
        "same precondition, different accumulation",
        ("EXEC-FOR-001", "DROP-LOOP-001"), "control-transfer", ("12",))

    return out


MISSING_FAMILIES = {
    "M08": "workspace relocation — transforms a package graph; every case is single-file until §15",
    "M09": "dependency declaration reorder — same, and needs at least two declared dependencies",
}
