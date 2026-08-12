# STARK pre-alpha — known limitations

**For participants in the controlled pre-alpha cohort.** Current as of the tree this file ships
with; the facts below are checked against their sources on every CI run
(`starkc/tests/cohort_limitations_are_current.rs`), so a stale page fails the build rather than
misleading you.

**This page is not an authority.** It presents facts owned elsewhere:

```text
starkc/docs/conformance/NATIVE-CONFORMANCE-MATRIX.md   what the native compiler supports
starkc/docs/conformance/KNOWN-DEVIATIONS.md            live deviations, in full detail
STARKLANG/docs/compiler/plans/WP-ARCH-CLOSE-*.md       qualification status
```

Where this page and one of those disagree, **that source is right and this page is stale** — and the
CI check exists so that cannot happen quietly.

---

## 1. Qualification status — read this first

```text
Architecture closure          PROVISIONAL
Native compilation            done, over a QUALIFIED SUBSET (see §2)
Distribution                  integrity-verified, NOT authenticated (archives are unsigned)
```

**Provisional means the closure verdict is not yet taken.** The pre-alpha gate opened on two
criteria — an executable native conformance contract and trustworthy qualification infrastructure —
and the remaining architecture work continues while you use it. **You are part of that evidence:**
anything you hit that looks like a compiler defect enters the same triage as an internally found
one, and some of what you find will change the verdict.

Concretely, what is *not* yet established: the adversarial campaign over the compiler's semantic
authorities is partway through, and until it finishes no unconditional claim about architectural
stability is being made.

## 2. Unsupported native constructs

**Read `NATIVE-CONFORMANCE-MATRIX.md`.** It is generated from a live compiler run and validated on
Linux, macOS and Windows every CI run; it is not a hand-maintained list, and it will not drift from
the compiler you are using.

It answers, per construct: *does this build natively, and if not, what happens when I write it?*
Six constructs are `KNOWN-DEVIATION` — valid STARK this compiler refuses, each with a
**deterministic, named refusal at compile time** rather than a miscompile:

```text
DEV-140   Vec::insert, and extend/truncate/sort/reverse/contains/dedup/split_off/drain/retain
          push, pop, len and indexing ARE supported
DEV-141   HashMap<K, V> where V has a destructor. Maps of values without destructors are fine
DEV-142   a composite mixing an owned droppable and a borrow -- (String, &str)
DEV-143   assert_eq on a user type implementing Eq
DEV-144   `for` over an iterator that is neither a range nor a Vec cursor
DEV-145   String::to_uppercase, and to_lowercase/trim/replace/starts_with/ends_with/find/
          split_at/repeat. len, as_str and push_str ARE supported
```

**Workarounds, where one exists:**

```text
DEV-143   `a == b` works on a user nominal in every engine
DEV-142   print or pass the parts separately
DEV-141   a HashMap whose values carry no destructor is unaffected
```

## 3. Other reachable limitations

Two live deviations that are not native-subset boundaries:

```text
DEV-221   a qualified core-trait call on a BOUNDED generic parameter is refused --
          `<T as Display>::fmt(x)` where T: Display.
          WORKAROUND: the ordinary method form `x.fmt()` works
DEV-233   the interpreter discards output already written when a trap follows, so a
          `println` before a trap may not appear under `stark run`.
          This is a debugging tax, not a correctness one -- the native binary is unaffected.
          WORKAROUND: reproduce under `stark build` when output ordering matters
```

## 4. Capability-backed packages do not run under `stark run`

The interpreters have **no host access at all**. A package declaring any capability —
`filesystem-read`, `network-client`, `clock`, and the rest — builds with `stark build` and runs as a
native binary. `stark run` cannot execute it, and this is a design boundary rather than a defect.

## 5. What to do when you hit something

Report it. Externally discovered cases enter the same triage as internally found ones, including the
architecture-trigger classification that decides whether a defect is ordinary maintenance or
evidence that a boundary is wrong. A case that looks trivial to you may be the one that moves the
verdict — DEV-160, the last reachable capability gap, was reported from ordinary application code.

---

*Facts on this page are validated against their sources by
`starkc/tests/cohort_limitations_are_current.rs`. It fails if a deviation named here has been
resolved, if a live user-reachable deviation is missing, or if the conformance matrix disagrees.*
