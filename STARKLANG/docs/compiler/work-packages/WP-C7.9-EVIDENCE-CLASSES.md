# WP-C7.9 Packet H — evidence classes for provider-backed capabilities

**Status:** the classification is normative for how this project states qualification claims. The
deferred item in §4 is recorded, not scheduled.

---

## 1. The problem this fixes

"Three-engine qualified" is a strong claim and it is not available for every capability. The MIR
interpreter deliberately does not execute `Callee::Provider`: no provider is linked into it, and
its own source says so — *"provider call cannot execute in the MIR interpreter: no provider is
linked into it (A10)"*. That is a design decision, not a gap to be closed by accident.

The consequence is that a provider-backed capability **cannot** be three-engine qualified, however
thoroughly it is tested. Args and environment, time, TCP, `File`, DNS, TLS, database handles — all
of them get a different, and differently strong, kind of evidence. Describing that as three-engine
qualification would be describing evidence nobody has.

## 2. The classes

```text
Pure language semantics       :  HIR == MIR == native-debug == native-release
Provider binding / lifecycle  :  MIR verifier + provider ABI tests + native execution
Real operating-system effects :  native platform qualification, per target
```

Read as: the top line is the strongest claim and is available only for programs that call no
provider. The second is what a provider-backed program's *structure* gets — the verifier proves the
call shape, ownership transfer and exactly-once release; the ABI tests prove the boundary contract;
native execution proves it runs. The third is what the underlying OS behaviour gets, and it is
per-target: a capability qualified on macOS is not thereby qualified on Linux.

## 3. How to state a claim

| Claim | Permitted for |
| --- | --- |
| "agrees across HIR, MIR and native" | programs calling no provider |
| "verifier + ABI + native qualified" | provider binding, lifecycle, release, failure channels |
| "qualified on `<target>`" | real OS effects, naming the target |
| "three-engine qualified" | **never** for a provider-backed execution path |

A package that has both halves states both, separately. A package whose pure-STARK logic could be
three-engine qualified over a deterministic adapter may not claim that until such an adapter exists
(§4).

## 4. Deferred: the deterministic interpreter-side provider

**Owner ruling D5 for WP-C7.9: do not build it in this work package.** Recorded here with one owner
and one future work item, rather than left as an open question that each package answers for itself.

What it would be: an in-memory provider the MIR interpreter can execute, scripting success and
failure status codes, borrowed and consumed handles, output buffers, a failed `HandleOut`, exact
close events, and short reads and writes. That would make provider *call semantics* comparable
between MIR and native — ownership transfer, release timing, failure-channel mapping — while real
OS behaviour stayed native-only.

What it would not do: make filesystem, clock, or socket behaviour comparable. Those are the host's,
and no fake changes that.

- **Owner:** the I/O gate (the same one holding `File`, excluded at C6 closure for the related
  reason that it needs "a way to compare environmental observations across engines").
- **Blocks:** any claim that a pure-STARK convenience layer over a provider is three-engine
  qualified.

## 5. Audit performed

| Artifact | Finding |
| --- | --- |
| `stark-time/EVIDENCE.md` | **Already correct.** Its "Cross-engine evidence" section states outright that the comparison is *not applicable* while the clock-reading members are unimplemented, and that the pure arithmetic runs only through the HIR interpreter under `stark test`. No overclaim to fix. |
| `stark-time/BLOCKERS.md`, `native/README.md` | Consistent with the above; the provider is classified `READY_PACKAGE_PROVIDER` with linkage recorded as the blocker. |
| `c78/` capability records | State per-capability status; no three-engine claim for a provider path. |
| `starkc/src/mir/interp.rs` | The refusal is explicit and documented at the refusal site, which is why the classification above is a statement of existing behaviour rather than a change to it. |
| `stark-io` | **Not audited here — not present in this branch** (untracked work in another checkout). The rule in §3 applies to it: its provider calls are verifier/ABI/native qualified, and its pure-STARK layer is not three-engine qualified until a deterministic adapter exists. |

The audit's outcome is that the *documents* were largely honest already, and what was missing was a
stated rule — so the next capability does not have to re-derive it, and a reviewer has something to
check a claim against.
