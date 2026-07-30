# C7.8 Slice 7 Closure Gate

This matrix deliberately separates the layers that C7.8.7 gates. A single "supported" flag would
over-claim: a capability can have valid provider metadata without a source-level API, a source-level
API without resource lifecycle lowering, or native execution without cross-platform verification.

**Every row cites evidence that is committed on `main`.** Work in a session's working tree is not
evidence — the matrix records what a fresh clone can reproduce. That distinction is doing real work
here: the TCP bind/accept/echo path exists and passes locally, and is deliberately *not* claimed
below, because it is not yet on `main`.

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
| `tcp` (`stark-net`) — bind/accept/echo | yes | yes | yes | yes | **not yet** | Exists and passes locally; **uncommitted**, so not claimed |

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
| repeated connect/release | defined | Single connect/release is observed; the repeated form is not |
| accept/release (two resources, closed independently) | defined | Needs the uncommitted TCP path |
| repeated open/release (`filesystem`) | defined, **and blocked by SELECT-C** | `File` is not on the `HostResource` path, so it has no A11 close arena to exercise |
| early return with a live resource | defined | |
| `?` propagation with a live resource | defined | |
| resource moved through a **call** | defined | Move *within* a scope is observed; through a call boundary is not |

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

**Met for `clock`, `process.args`, `process.env`.** Met for `tcp`'s resource lifecycle; **not yet
claimable for TCP's listener/stream surface**, which P1 explicitly requires — the path works but is
uncommitted. `filesystem` reaches native execution through its legacy path by decision, not by
omission.

So C7.8 has **substantially** removed P1's host-capability precondition, and the honest residue is
one commit plus the lifecycle cases still marked `defined`.

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
