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

import json
import re
from dataclasses import dataclass


def _key_order(manifest_text: str) -> list:
    """The order dependency names are DECLARED in, as text.

    `json.loads` preserves insertion order, but two manifests can parse to equal dicts and still be
    written in a different order — which is exactly the thing M09 varies. Comparing parsed dicts
    proves the graph is unchanged; comparing this proves the declaration really was reordered.
    """
    block = re.search(r'"dependencies"\s*:\s*\{(.*?)\}\s*\}', manifest_text, re.S)
    return re.findall(r'"([A-Za-z_][A-Za-z0-9_]*)"\s*:', block.group(1)) if block else []


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
    #: Matrix ROW ids this pair exercises (R-13). The FAMILY id is `family_id` and belongs in
    #: `metamorphic_family`; conflating the two inflated the coverage count by ten.
    subcategories: tuple = ()
    #: What KIND of transformation this is, which decides how the pair is validated (R-04/R-05).
    #: `source` transformations must change the source text. The two PACKAGE kinds must not — their
    #: whole point is that something outside the logical source changed — so they are validated
    #: against different, stricter invariants rather than by exempting them from validation.
    kind: str = "source"
    #: `{path relative to the member's tree root: contents}` for package kinds; `None` for `source`.
    base_files: dict = None
    transformed_files: dict = None
    #: The root package within the tree, e.g. `app`. Relative to the member's staged root.
    package_root: str = ""
    package_graph: str = "single-file"
    expected_drop_log: tuple = ()
    #: Non-empty when the pair TRAPS rather than completes — M08's second group relocates a trapping
    #: workspace, which is the half a completing pair cannot witness.
    expected_trap: str = ""
    #: For package pairs whose SUBJECT is provenance or symbol identity, the harness additionally
    #: pins those beyond the shared observation — the observation alone would not notice a symbol
    #: set that changed shape while still printing the same bytes.
    pin_canonical_symbols: bool = False
    pin_logical_provenance: bool = False


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


def validate_package_pair(family, group, kind, base_files, transformed_files):
    """The kind-aware half of the identity-transform protection (R-04/R-05).

    The global `transformed != base` rule is NOT weakened for package pairs — it is *replaced by a
    stricter obligation*, because for a package transformation "the source changed" would be the
    wrong assertion: a relocation that edited a file would no longer be a relocation. So each kind
    states what must stay identical AND what must differ, and both are checked.

    A module-level function rather than a closure inside `groups()` so `generate.py
    --self-test-guards` can drive the REAL rule with deliberately invalid pairs. A guard nobody has
    watched refuse is not evidence, and a guard the self-test only *reimplements* is worse — it
    proves the copy works.
    """
    assert base_files and transformed_files, f"{family}/{group}: empty package tree"

    if kind == "relocation":
        # M08. Logical files and contents IDENTICAL; the only difference is the physical root each
        # member is staged at, which is a property of the run rather than of the tree. The harness
        # separately proves the two roots really differ — two references to one staged workspace
        # would agree trivially, which is the relocation-shaped version of an identity transform.
        assert base_files == transformed_files, (
            f"{family}/{group}: a relocation pair must keep every logical file and its contents "
            f"identical — a differing tree is some other transformation wearing its name"
        )
    elif kind == "dependency-reorder":
        # M09. Same graph, same sources, different declaration order.
        assert set(base_files) == set(transformed_files), (
            f"{family}/{group}: a dependency reorder must not add or remove files"
        )
        manifests = [p for p in base_files if p.endswith("starkpkg.json")]
        changed = [p for p in base_files if base_files[p] != transformed_files[p]]
        assert changed, (
            f"{family}/{group}: nothing changed — a reorder that reordered nothing is a fake pair"
        )
        outside = [p for p in changed if p not in manifests]
        assert not outside, (
            f"{family}/{group}: only MANIFESTS may differ in a dependency reorder; these also "
            f"differ: {outside}"
        )
        for path in changed:
            before = json.loads(base_files[path]).get("dependencies", {})
            after = json.loads(transformed_files[path]).get("dependencies", {})
            assert before == after, (
                f"{family}/{group}: {path} changed the dependency SET, not just its order — that "
                f"is an edit to the graph, not a reorder of its declaration"
            )
            assert _key_order(base_files[path]) != _key_order(transformed_files[path]), (
                f"{family}/{group}: {path} differs but its dependency declaration order does not"
            )
    else:
        raise AssertionError(f"{family}/{group}: unknown package transformation kind {kind!r}")


# ----------------------------------------------------------- package fixtures --
#
# Authored here for the same reason the single-file bases are: a reorder or relocation applied to a
# tree this module does not control is how a transformation silently becomes a no-op.


def _manifest(name, deps=(), entry="src/main.stark"):
    """A `starkpkg.json` with dependencies written in the order given — the order IS the variable."""
    body = {"name": name, "version": "0.1.0", "entry": entry}
    if deps:
        body["dependencies"] = {dep: {"path": f"../{dep}"} for dep in deps}
    return json.dumps(body, indent=4) + "\n"


def reorder_dependencies(tree, manifest_path):
    """Reverse the DECLARATION order of one manifest's dependencies, changing nothing else."""
    manifest = json.loads(tree[manifest_path])
    deps = manifest.get("dependencies", {})
    assert len(deps) >= 2, f"{manifest_path}: a reorder needs at least two dependencies"
    manifest["dependencies"] = {k: deps[k] for k in reversed(list(deps))}
    out = dict(tree)
    out[manifest_path] = json.dumps(manifest, indent=4) + "\n"
    return out


#: M08 g1 — a two-package workspace that completes, with a `Drop` type destroyed in the root so the
#: pair witnesses the Drop log across relocation as well as the printed bytes.
RELOCATABLE_WORKSPACE = {
    "app/starkpkg.json": _manifest("app", ["core"]),
    "app/src/main.stark": """use core::scaled;
use core::Marked;
use core::mark;

fn main() {
    print(scaled(3));
    print("|");
    println(4);
    {
        let held: Marked = mark(1);
    }
    print("end");
}
""",
    "core/starkpkg.json": _manifest("core"),
    "core/src/main.stark": """pub struct Marked { id: Int32 }

impl Drop for Marked {
    fn drop(&mut self) {
        print("@@stark-drop:Marked#");
        print(self.id);
        println("@@");
    }
}

pub fn mark(id: Int32) -> Marked { Marked { id: id } }

pub fn scaled(v: Int32) -> Int32 { v * 4 + 1 }
""",
}

#: M08 g2 — the trapping half. A completing pair cannot witness trap provenance, which is precisely
#: what DEV-113 got wrong: the trap named an absolute checkout path, so the same workspace in two
#: directories reported two different source files.
RELOCATABLE_TRAP_WORKSPACE = {
    "app/starkpkg.json": _manifest("app", ["core"]),
    "app/src/main.stark": """use core::burst;

fn main() {
    print("before");
    burst(2147483647);
}
""",
    "core/starkpkg.json": _manifest("core"),
    "core/src/main.stark": """pub fn burst(v: Int32) -> Int32 { v + 1 }
""",
}

#: M09 g1 — a diamond: `app` declares BOTH `left` and `right`, which share `base`. Two declared
#: dependencies are what makes an order exist to reorder, and the shared leaf is what made DEV-114's
#: nondeterminism observable — whichever path reached `base` first named its items.
DIAMOND_WORKSPACE = {
    "app/starkpkg.json": _manifest("app", ["left", "right"]),
    "app/src/main.stark": """use left::via_left;
use right::via_right;

fn main() {
    print(via_left(4));
    print("|");
    print(via_right(4));
}
""",
    "left/starkpkg.json": _manifest("left", ["base"]),
    "left/src/main.stark": """use base::shared;

pub fn via_left(v: Int32) -> Int32 { shared(v) + 1 }
""",
    "right/starkpkg.json": _manifest("right", ["base"]),
    "right/src/main.stark": """use base::shared;

pub fn via_right(v: Int32) -> Int32 { shared(v) + 1 }
""",
    "base/starkpkg.json": _manifest("base"),
    "base/src/main.stark": """pub fn shared(v: Int32) -> Int32 { v * 2 }
""",
}

#: M09 g2 — the same diamond whose shared leaf carries a `Drop` type, so the pair witnesses
#: destruction ORDER against declaration order, not only the printed bytes.
DIAMOND_DROP_WORKSPACE = {
    "app/starkpkg.json": _manifest("app", ["left", "right"]),
    # `app` does NOT import `base` — it declares only `left` and `right`, and PKG-RESOLVE-001 makes
    # a package's dependencies exactly what it declares. Naming `Leaf` here would have required a
    # third edge and turned the diamond into a different graph; the bindings are inferred instead.
    "app/src/main.stark": """use left::leaf_from_left;
use right::leaf_from_right;

fn main() {
    println("ready");
    {
        let first = leaf_from_left(1);
        let second = leaf_from_right(2);
    }
}
""",
    "left/starkpkg.json": _manifest("left", ["base"]),
    "left/src/main.stark": """use base::Leaf;
use base::leaf;

pub fn leaf_from_left(id: Int32) -> Leaf { leaf(id) }
""",
    "right/starkpkg.json": _manifest("right", ["base"]),
    "right/src/main.stark": """use base::Leaf;
use base::leaf;

pub fn leaf_from_right(id: Int32) -> Leaf { leaf(id) }
""",
    "base/starkpkg.json": _manifest("base"),
    "base/src/main.stark": """pub struct Leaf { id: Int32 }

impl Drop for Leaf {
    fn drop(&mut self) {
        print("@@stark-drop:Leaf#");
        print(self.id);
        println("@@");
    }
}

pub fn leaf(id: Int32) -> Leaf { Leaf { id: id } }
""",
}


# ------------------------------------------------------------------ registry --

def groups() -> list:
    """All 24 groups: two per family for twelve families (§13.2's floor, R-04).

    M08 (workspace relocation) and M09 (dependency declaration reorder) were previously recorded as
    unbuildable — "both transform a PACKAGE GRAPH, and every corpus case is single-file until §15".
    That stopped being true when DEV-113/DEV-114 added package cases to the corpus, and DEV-114's fix
    is what makes M09 comparable at all: before it, a diamond graph produced different canonical
    symbols run to run, so a reorder pair would have disagreed for a reason that had nothing to do
    with the reorder.
    """
    out = []

    #: Which matrix rows each family's pairs actually exercise. Stated per family rather than per
    #: group because both groups of a family transform the same construct.
    rows = {
        "M01": ("E02", "E04"),
        "M02": ("E03", "O15"),
        "M03": ("D05", "E12"),
        "M04": ("D06", "E11"),
        "M05": ("V12", "E04"),
        "M06": ("P07", "C11"),
        "M07": ("C11", "P05"),
        "M08": ("K06", "K07"),
        "M09": ("K08", "K09"),
        "M10": ("E10", "D01"),
        "M11": ("E13", "D09"),
        "M12": ("C04", "C05"),
    }

    def add(family, group, base, transformed, precondition, rules, category, stdout):
        assert transformed != base, (
            f"{family}/{group}: the transformation changed nothing — an identity transform is a "
            f"fake pair that agrees trivially"
        )
        out.append(
            Group(
                family,
                group,
                base,
                transformed,
                precondition,
                rules,
                category,
                stdout,
                rows.get(family, ()),
            )
        )

    def add_package(family, group, kind, base_files, transformed_files, package_root,
                    package_graph, precondition, rules, category, stdout, drop_log=(),
                    trap="", pin_symbols=False, pin_provenance=False):
        """A package-graph pair, validated by rules specific to its KIND (R-04/R-05).

        The global `transformed != base` protection is NOT weakened for these. It is replaced by a
        stricter obligation, because for a package transformation "the source changed" would be the
        WRONG assertion — a relocation that edited a file would no longer be a relocation. So each
        kind states what must stay identical and what must differ, and both are checked."""
        validate_package_pair(family, group, kind, base_files, transformed_files)
        out.append(
            Group(
                family_id=family, group_id=group, base="", transformed="",
                precondition=precondition, normative_rules=rules, category=category,
                expected_stdout=stdout, subcategories=rows.get(family, ()), kind=kind,
                base_files=base_files, transformed_files=transformed_files,
                package_root=package_root, package_graph=package_graph,
                expected_drop_log=drop_log, expected_trap=trap,
                pin_canonical_symbols=pin_symbols, pin_logical_provenance=pin_provenance,
            )
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

    # ---------------------------------------------------------- package pairs --

    # M08 workspace relocation. PKG-IDENTITY-001: a package token is "never an absolute checkout
    # path", so the same workspace compiled in two directories must observe identically — same
    # bytes, same trap provenance, same canonical symbols. This is the property DEV-113 broke and
    # fixed; the pair is what keeps it fixed.
    add_package(
        "M08", "g1", "relocation", RELOCATABLE_WORKSPACE, RELOCATABLE_WORKSPACE, "app", "workspace",
        "identical logical files and contents, staged at two independent physical roots; the "
        "harness proves the roots differ (PKG-IDENTITY-001, §15.2)",
        ("PKG-IDENTITY-001", "TYPE-NOMINAL-001"), "packages-environment", ("13|4", "end"),
        drop_log=("Marked#1",), pin_symbols=True, pin_provenance=True)
    add_package(
        "M08", "g2", "relocation", RELOCATABLE_TRAP_WORKSPACE, RELOCATABLE_TRAP_WORKSPACE, "app",
        "workspace",
        "same, for a TRAPPING workspace: relocation must not move the reported trap location either, "
        "which is the half a completing pair cannot witness (DEV-113)",
        ("PKG-IDENTITY-001", "TRAP-CATEGORY-001"), "packages-environment", ("before",),
        trap="IntegerOverflow", pin_symbols=True, pin_provenance=True)

    # M09 dependency declaration reorder. TYPE-NOMINAL-001 makes identity "canonical package
    # instance + module path + item name", so a dependency EDGE is not part of it and the order the
    # edges are written in cannot be either. DEV-114 is why this needs a pair: the graph walk
    # followed a per-process-seeded HashMap, so a diamond produced different symbols run to run.
    add_package(
        "M09", "g1", "dependency-reorder", DIAMOND_WORKSPACE,
        reorder_dependencies(DIAMOND_WORKSPACE, "app/starkpkg.json"), "app", "workspace",
        "same package graph and identical sources; only the ORDER of the root's two dependency "
        "declarations differs (TYPE-NOMINAL-001, PKG-RESOLVE-001)",
        ("TYPE-NOMINAL-001", "PKG-RESOLVE-001"), "packages-environment", ("9|9",),
        pin_symbols=True)
    add_package(
        "M09", "g2", "dependency-reorder", DIAMOND_DROP_WORKSPACE,
        reorder_dependencies(DIAMOND_DROP_WORKSPACE, "app/starkpkg.json"), "app", "workspace",
        "same, for a graph whose shared leaf carries a `Drop` type: declaration order must not "
        "change destruction order either (DROP-ORDER-001)",
        # `expected_stdout` is LINES joined by "\n", and the body's last output op is `println`, so
        # the trailing newline makes a final empty line. The Drop frames that follow are stripped by
        # the §8.8 protocol scan and compared as `expected_drop_log`, not as stdout.
        ("TYPE-NOMINAL-001", "DROP-ORDER-001"), "packages-environment", ("ready", ""),
        drop_log=("Leaf#2", "Leaf#1"), pin_symbols=True)

    return out


#: Every family is now built. Kept as an empty registry rather than deleted: `the_metamorphic_floor_
#: is_reported_honestly` reads it, and a silently absent structure is how an unmet floor stops being
#: reported at all.
MISSING_FAMILIES = {}
