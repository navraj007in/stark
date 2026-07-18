# Core v1 Open Questions

Status: **WP-C2.6 skeleton — recommendations are not approved semantics**
Created by: Gate C2 semantic-freeze preflight, 2026-07-18

## Register rules

Every unresolved semantic question must record urgency, rationale, cost of delay, recommended
default, alternatives, compatibility impact, owner, and approval state. Implementation behavior
may be cited as evidence, but it does not approve a language rule. An approved answer must be
moved into its normative home and linked to stable completeness/conformance rule IDs.

Approval states are `pending`, `approved`, `rejected`, and `superseded`.

## Decision register

| ID | Question | Urgency / delay cost | Recommended default | Material alternatives | Compatibility impact | Owner | Approval |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CORE-Q-001 | Which document owns ownership legality, execution, memory guarantees, and layout? | Immediate; duplicate authority makes every later decision ambiguous. | Type System: legality; Abstract Machine: execution; Memory Model: safety summary; C2.9: observable layout. | Keep duplicate normative chapters; merge all into one chapter. | High documentation and implementation ambiguity if delayed. | C2.6/C2.7 | Pending |
| CORE-Q-002 | Are type aliases transparent or nominal? | Before MIR type identity; late change breaks APIs and coherence. | Transparent aliases; distinct identity requires a distinct type declaration. | Nominal aliases; context-dependent aliases. | Ecosystem-breaking if changed later. | C2.8 | Pending |
| CORE-Q-003 | Which recursive and unsized types are legal? | Before layout/MIR; late change alters well-formedness and ABI assumptions. | Require finite sized values; permit only explicitly supported unsized pointees behind references/views. | All generics implicitly sized; general structural unsized types. | High. | C2.8 | Pending |
| CORE-Q-004 | What inference and normalization algorithm is normative? | Before a second frontend; implementation behavior otherwise becomes accidental law. | Deterministic local constraint solving with specified expected-type flow, defaulting, ambiguity rejection, and trait normalization. | Leave algorithm implementation-defined; global inference. | High source compatibility. | C2.8 | Pending |
| CORE-Q-005 | How is trait coherence computed across packages and versions? | Before package identity and ecosystem growth. | Selection independent of implementation order; no specialization/negative impls; overlap uses canonical package/type identity. | Ordered selection; limited specialization. | Ecosystem-breaking. | C2.8/C2.9 | Pending |
| CORE-Q-006 | What is the language-level model for places, temporaries, references, and destruction? | Immediate prerequisite for MIR and differential comparison. | Implementation-independent abstract machine covering success, early exit, and partial failure. | Treat interpreter behavior as normative. | High soundness and portability risk. | C2.7 | Pending |
| CORE-Q-007 | What expressions are allowed in constants? | Before type/layout lowering and package metadata. | Bounded deterministic side-effect-free subset with explicit cycles, overflow, traps, and cross-package rules. | General interpreter evaluation; literals only. | High source compatibility. | C2.8 | Pending |
| CORE-Q-008 | What are the complete integer semantics? | Before optimizer/backend work. | Fixed-width two's-complement values with explicit checked arithmetic, division/remainder, shift, negation, and cast rules invariant across build modes. | Wrapping defaults; target/native behavior. | High correctness and security impact. | C2.9 | Pending |
| CORE-Q-009 | Which equality, ordering, and hashing traits do floats implement? | Before collection and generic contracts freeze. | Do not imply total `Eq`/`Ord`/`Hash`; provide explicitly named partial or bitwise behavior where specified. | IEEE comparison plus total wrappers; canonicalized NaN hashing. | High generic/API compatibility. | C2.9 | Pending |
| CORE-Q-010 | Which floating-point results are reproducible? | Before multiple backends. | Specify required IEEE behavior and permitted target variation; prohibit unapproved contraction/reassociation. | Bit-for-bit all targets; fully target-defined. | High optimizer portability. | C2.9 | Pending |
| CORE-Q-011 | What units and boundary failures do text APIs use? | Before public APIs and serialized positions spread. | UTF-8 bytes for low-level offsets; scalar-boundary operations reject or trap deterministically; document Unicode-data version policy. | Scalar indices; grapheme indices. | Ecosystem-breaking API behavior. | C2.9 | Pending |
| CORE-Q-012 | Which layout facts are observable? | Before native ABI assumptions arise. | No stable Core v1 ABI; only explicitly exposed facts are target-defined, and pointers/stacks/tagged unions are not language promises. | Stable ABI; entirely opaque layout. | Ecosystem-breaking. | C2.9 | Pending |
| CORE-Q-013 | What is canonical package and public-item identity? | Before multi-version dependencies and package-level coherence. | Canonical name + resolved version + locked source/content identity; aliases do not change identity. | Name/version only; source URL identity; content hash only. | Ecosystem-breaking. | C2.9 | Pending |
| CORE-Q-014 | Which executable entry signatures and process mappings are valid? | Before native compilation and test harness equivalence. | Small explicit `main` signature set with deterministic return/Result-to-exit and stream/trap mapping. | One signature only; host-defined signatures. | High tooling/runtime compatibility. | C2.9 | Pending |
| CORE-Q-015 | Which standard-library items are language hooks? | Before replacing interpreter builtins and stabilizing std. | Enumerate a minimal hook set; all other names use ordinary resolution and trait dispatch. | Recognize APIs by name broadly; make all hooks compiler intrinsics. | High library evolution cost. | C2.8/C2.9 | Pending |
| CORE-Q-016 | What future boundary must Core v1 preserve? | Before closures, concurrency, and native providers constrain semantics. | Safe single-threaded Core; reserve ownership-aware capturing closures/lifetimes; unsafe/FFI stay non-Core; native providers declare ABI/provenance/capabilities. | Add concurrency/FFI now; leave future behavior unconstrained. | Ecosystem-breaking if delayed. | C2.10 | Pending |

## Known deviation links

WP-C2.6 must link at least DEV-009, DEV-017, DEV-018, DEV-022, DEV-023, and DEV-024 to the
corresponding questions and rule rows. DEV-005 and DEV-019 are C2.11 alignment work. DEV-036 is
C2.12 harness work. Closed C2.2 deviations remain evidence unless a newly specified rule proves
a residual mismatch.
