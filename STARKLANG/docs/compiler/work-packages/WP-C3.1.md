# WP-C3.1 — Architecture Hypothesis and Workload Freeze

Gate: C3 (Native Compiler Architecture and Backend Selection Spike). Scope extracted from
`STARKLANG/docs/compiler/COMPILER-ROADMAP.md`, WP-C3.1, with the workload extended to 23 items
per CD-021/CD-022.

## Scope

Write `STARKLANG/docs/compiler/proposals/NATIVE-CORE-ARCHITECTURE.md`. Freeze the representative
workload set (23 items). Define the measurement set. State the architecture hypothesis and its
falsifiers.

This WP does **not**: implement a backend (that is WP-C3.2 generated Rust/C and WP-C3.3 direct
Cranelift), select a production backend (that is WP-C3.4, escalation CE5), or freeze the MIR
contract or runtime ABI (that is Gate C4/CE3 and WP-C5.1/CE4). It sets up the comparison; it
does not conclude it.

## Deliverables

- `STARKLANG/docs/compiler/proposals/NATIVE-CORE-ARCHITECTURE.md` — architecture hypothesis,
  the frozen 23-item workload, the measurement framework, the risk register, and the WP-C3.4
  decision-framework preview.
- Workload freeze: each of the 23 items mapped either to a frozen `exec_snapshots` corpus case
  (`corpus_version = 1.0.0`, the semantic oracle) or to a specified reference program the spikes
  must implement.

## Semantic comparator vs. architecture comparator (charter/roadmap distinction)

- **Semantic comparator:** the existing reference interpreter. It already executes all 23
  workload items (function values via `Value::Function`, file I/O via `Value::File`/DEV-009,
  references/slices, Drop). Native output must match it (charter §1.6 rule 6).
- **Architecture comparator:** *not* "no native compiler" (native compilation is mandatory,
  CD-004). It is the strongest practical candidate implementation paths — generated Rust/C and a
  direct Cranelift backend.

## Scope-control answers (charter §2.6)

- **Exact claim this WP completes:** a frozen, representative workload and a defined measurement
  framework exist, so the C3.2/C3.3 spikes produce attributable, comparable evidence and C3.4
  can select on evidence rather than impression.
- **Later mechanism that would make the result unattributable:** an unrepresentative or unfrozen
  workload — spike results would not generalize, and a backend could look adequate only because
  the workload dodged its weak points (monomorphization, trait dispatch, reference/slice ABI,
  Drop-bearing resources, function values). The workload deliberately includes all five.
- **Strongest existing implementation path / comparator:** for generated Rust, old Gate 5
  already lowers to a generated Rust host (`deploy/`) — partial precedent. For direct, Cranelift
  is the roadmap's named candidate.
- **Negative result that would stop this WP:** none — WP-C3.1 is setup. `BLOCKED`/`REVISE` are
  WP-C3.4 outcomes, not C3.1's. The only failure mode here is an inadequate workload, which is
  corrected by revising this WP, not by stopping the gate.

## Done when

- [ ] `NATIVE-CORE-ARCHITECTURE.md` exists with all four sections (hypothesis, frozen workload,
      measurements, risk register + decision preview).
- [ ] Every one of the 23 workload items is mapped to a corpus case or a specified reference
      program, with its semantic-oracle source identified.
- [ ] The 13 measurement dimensions from the roadmap each have a defined measurement method.
- [ ] `COMPILER-STATE.md` Native-backend-selection status reads `SPIKING`; a WP-C3.1 session
      record is added; next is WP-C3.2/C3.3.
