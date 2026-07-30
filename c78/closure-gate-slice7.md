# C7.8 Slice 7 Closure Gate

This matrix deliberately separates the layers that C7.8.7 gates. A single "supported" flag would
over-claim: a capability can have valid provider metadata without a source-level API, a source-level
API without resource lifecycle lowering, or native execution without cross-platform verification.

**Every row cites evidence that is committed on `main`.** Work in a session's working tree is not
evidence — the matrix records what a fresh clone can reproduce.

That rule did real work rather than being decorative: the TCP bind/accept/echo row sat at
**not yet** while the path existed and passed in a working tree, and only moved once the test was
committed (CD-258). The gap between "it works on my machine" and "the matrix may claim it" was a
real commit, which is the point.

## What "cross-platform verified" means here

CI runs `cargo test --workspace --all-targets --all-features --no-fail-fast` on **macos-arm64,
linux-x64 and windows-x64**, plus the `C7.8 Native Capabilities` workflow's provider metadata, unit,
resource and loopback jobs on the same three. A committed test therefore executes on all three
platforms on every push, and the column below says `yes` only where that is true of committed tests.

`--no-fail-fast` matters for reading this column. Before CD-245 cargo stopped at the first failing
test binary, so a green-looking Windows job could simply never have reached a later suite. It had
been hiding a real `STATUS_STACK_OVERFLOW` (fixed as `LIMIT-MIR-TYPE-DEPTH`, CD-255) for an unknown
number of runs.

## Capability Matrix

| Capability | Frontend | HIR | MIR-Lowered | Native-Runtime | Cross-Platform Verified | Evidence (committed) |
| --- | --- | --- | --- | --- | --- | --- |
| `clock` (`stark-time`) | yes | yes | yes | yes | yes | `c788_source_time_e2e` (source → `Callee::Provider` → link → execute), `a10_stark_time_e2e` (ABI/emission) |
| `process.args` (`stark-env`) | yes | yes | yes | yes | yes | `c783_env_e2e`, `c788_starkc_build::args_len_executes_from_source_through_build_command` |
| `process.env` (`stark-env`) | yes | yes | yes | yes | yes | `c78_buffer_e2e`; `c788_starkc_build::env_var_success_and_recoverable_invalid_name_...` — the first **executing** proof of the recoverable-status `Err` arm (CD-233) |
| `filesystem` (`stark-file`) | n/a | n/a | legacy path | yes | yes | `c784_file_e2e`. **Not a gap — SELECT-C (CD-253).** `File` stays on `MirTy::Core`; conditional migration would make MIR identity depend on build configuration |
| `tcp` (`stark-net`) — resource lifecycle | yes | yes | yes | yes | yes | `c788_lifecycle_e2e` (four lifecycle cases, executed against the real provider) |
| `tcp` (`stark-net`) — bind/accept/echo | yes | yes | yes | yes | yes | `c788_starkc_build::tcp_bind_accept_connect_and_echo_execute_from_source_through_build_command` (CD-258) |

## Lifecycle Set

CD-234's runtime guarantees. **Observed** means executed against a real provider on `main`, not
argued from the parts.

| Case | Status | Evidence |
| --- | --- | --- |
| never-initialised resource does not close | **observed** | `c788_lifecycle_e2e::a_never_initialised_resource_does_not_close` |
| failed `HandleOut` does not close | **observed** | `c788_lifecycle_e2e::a_failed_handle_out_does_not_close` |
| successful `HandleOut` closes exactly once | **observed** | `c788_lifecycle_e2e::a_successful_handle_out_closes_exactly_once` |
| move then drop closes only the destination | **observed** | `c788_lifecycle_e2e::move_then_drop_closes_only_the_destination` |
| explicit close then destructor path | **unreachable by construction** | A package may not bind a close (design §2) and `MIR-0033` rejects a direct call, so MIR owns the only close path. Pinned by `a11_host_resource`'s rule-4 tests |
| accept/release (two resources, closed independently) | **observed** | `c788_lifecycle_e2e::accept_and_release_close_two_resources_independently` — asserts the listener closes through `stark_tcp_listener_close` and the accepted stream through `stark_tcp_stream_close`, the pairing `MIR-0030` enforces |
| early return with a live resource | **observed** | `c788_lifecycle_e2e::an_early_return_with_a_live_resource_closes_it` |
| resource moved through a **call** | **observed** | `c788_lifecycle_e2e::a_resource_moved_through_a_call_closes_once_in_the_callee` — the caller's local is dead after the move, so exactly one close runs, in the callee |
| `?` propagation with a live resource | **observed** | `c788_lifecycle_e2e::question_mark_propagation_closes_a_live_resource` — a live first resource is closed on the desugared error path, which exits differently from an explicit `return` |
| repeated connect/release | **written, and it found a defect** | `DEFECT-C788-LOOP-TEMP` — see below. Test committed `#[ignore]`d with a classification, per CD-247. Admitted as a **non-blocking C7 deviation** at P1 compiler priority (CD-264) |
| repeated open/release (`filesystem`) | defined, **and blocked by SELECT-C** | `File` is not on the `HostResource` path, so it has no A11 close arena to exercise |

### How the observed cases detect a violation

`stark_tcp_stream_close` aborts when the handle is not in the provider's live table, so closing an
unopened resource — or closing one twice — kills the process. Each test asserts a clean exit and its
marker, which makes the provider itself the detector rather than a compiler diagnostic.

A clean exit alone proves only *at most once*. The successful case therefore also asserts the close
appears in the generated Rust, and **that assertion is what caught a total leak**: before CD-256 no
close was emitted at all, and every runtime check would have passed a program that never closed
anything.

## P1 unblock assessment

§5.7's claim requires each mandatory capability reachable from ordinary STARK source through a
package API, typed HIR, provider MIR lowering and native execution on all three platforms.

**Met for `clock`, `process.args`, `process.env`, and for `tcp` — both its resource lifecycle and
its listener/stream surface**, which P1 explicitly requires. `filesystem` reaches native execution
through its legacy path by decision (SELECT-C), not by omission.

So **C7.8 has removed P1's host-capability precondition.** The residue is not capability coverage but
a single lifecycle case — repeated connect/release — which is blocked by `DEFECT-C788-LOOP-TEMP`.
That is a property of resource *handling*, not of whether a capability is reachable, and it does not
block P1: the affected shape is a compiler-generated temporary, the P1 REST workload does not emit
it, and the workload is already built and running on this surface.

## DEFECT-C788-LOOP-TEMP — found by the last lifecycle case

`repeated_connect_and_release_reuses_slot_state` connects and releases three times in a `while` loop.
It aborts on the second iteration:

```
generated-code invariant violated: write to a live slot
(MIR must Drop or move out before reassigning a live place)
(STARK compiler defect, not a program fault)
```

The runtime classifies it itself. The generated program contains **exactly one `drop_with`** — on the
match binding — and **none for the scrutinee temporary** holding `Result<TcpStream, E>`. That temp is
written every iteration and never dropped, so the second write lands on a live slot.

**This is the state-reuse case CD-262 predicted**, and the only one of the nine that exercises a slot
going live → dead → live. Every other lifecycle case is straight-line or single-exit, which is why
eight passed and this did not.

Scope: it affects a **temporary**, not a user local — user bindings are drop-tracked and close
correctly, which is why the P1 REST workload runs its 24-accept loop without tripping this. The
defect needs a resource-bearing temporary re-written across iterations.

Recorded as an ignore with a `CLASSIFIED_IGNORES` entry rather than deleted or left red, so C6.4
tier-1 keeps it visible. Un-ignore with the fix.

### Disposition (CD-264): non-blocking C7 deviation, mandatory near-term correction

> **DEFECT-C788-LOOP-TEMP is admitted as a non-blocking C7 deviation.** A resource-bearing
> match-scrutinee temporary reused across loop iterations is not dropped before reassignment, causing
> the runtime to abort on the second write to the live slot. The runtime detects the compiler
> violation and fails closed; no silent ownership corruption has been observed.
>
> The defect does not invalidate the frozen P1 workload or its qualification because P1 does not
> generate the affected temporary-reuse shape and its user-bound resources close correctly. It
> therefore does not block Gate C7.
>
> The defect is nevertheless a mandatory compiler correction before STARK claims general native
> support for repeated resource-producing expressions or recommends such expressions for application
> use. The classified ignored regression remains committed and must be unignored by the fixing
> change.

| question | ruling |
| --- | --- |
| blocks P1 qualification? | **no** |
| blocks C7 closure? | **no** |
| lifecycle matrix fully complete? | **no** — 8 observed, 1 unreachable, 1 blocked |
| may remain indefinitely? | **no** |
| must be fixed before a broad native-resource completeness claim? | **yes** |
| must be fixed before a public release recommending resource-producing calls in loops? | **yes** |

Priority: **P1 compiler priority** — high priority, *not* the P1 workload. The two senses of "P1"
are unrelated and are kept distinct deliberately.

**Why it does not block.** C7's admitted closure question is whether the selected native path is
usable and qualified for the frozen workloads. Making this defect blocking would retroactively
change the gate from *prove the admitted native workload and its required resource surface* to
*prove every valid looping shape involving resource-bearing intermediate values*. The broader
guarantee matters, but it was not the frozen criterion.

**Why it is not a minor deferral.** The failing program is valid source:

```stark
while condition {
    match connect(address) {
        Ok(stream) => { use(stream); }
        Err(error) => { handle(error); }
    }
}
```

A correct program aborts on its second iteration. TCP merely exposed the defect; it is generic, and
reaches repeated provider operations, resource-producing expressions in loops, any future
`Result<Resource, E>` or `Option<Resource>` API, and confidence in exactly-once lowering for
reusable control-flow regions.

**Safety reading.** The compiler does not silently overwrite a live resource and continue — it fails
closed with a compiler-defect diagnostic. So this is a language-correctness and availability defect,
**not** a demonstrated silent double-close, use-after-move or ownership corruption. That fail-closed
behaviour is the reason it can be admitted as a known limitation rather than stopping C7.

### Fix boundary

Compiler-wide, not TCP-specific:

> Every non-`Copy` compiler-generated temporary that may be assigned again must be proven dead
> before the next assignment. If live, lowering must emit the appropriate Drop or move-out
> transition on **every** predecessor edge.

The investigation must examine:

1. lifetime scope assigned to match scrutinee temporaries;
2. drop scheduling at the loop back-edge;
3. drop flags for compiler-generated locals;
4. whether the temp is statement-scoped or loop-body-scoped;
5. cleanup on `continue`, `break`, `return`, `?`, panic and the normal back-edge;
6. `Option<Resource>` as well as `Result<Resource, E>`;
7. nested matches and multiple resource temporaries;
8. HIR/MIR/native parity.

`repeated_connect_and_release_reuses_slot_state` becomes the primary regression test and is
unignored by the fixing change (removing its `CLASSIFIED_IGNORES` entry in the same commit).

## A finding the gate should record

Six `MirTy` catch-all arms silently swallowed `HostResource` as it was introduced —
`dump_ty`, `emit_ty`, `default_value_expr`, `TypeContext::is_copy`, `FnLowerer::is_copy`,
`lower::ty_needs_drop`, `verify::may_need_drop`. Each compiled cleanly. Each was found by something
failing downstream, one at a time, over several days.

The last was the most serious: `ty_needs_drop`'s `_ => false` meant no `Drop` terminator was ever
emitted for a resource, so **every resource leaked while every unit test on the close machinery
passed**. It was found only by a test that inspected generated code, not by any test of the parts.

Two consequences worth carrying past this gate:

1. **"The parts pass" repeatedly failed to mean "the whole works."** Slice 7's evidence column
   should keep recording *how* a claim was verified, not only that a test exists.
2. The predicates that decide semantics — Copy-ness, drop-ness, representation — should not have
   `_ =>` fallbacks. Removing them would have turned all six defects into compile errors when
   `HostResource` was added. That change is mechanical and is recommended before the next `MirTy`
   variant.
