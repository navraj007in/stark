//! WP-C7.8.2b — provider selection and validated call-record construction.
//!
//! Covers the A10 §3 binding sequence, Packet 1 §1.3 (symbols validated verbatim, never
//! sanitised) and §1.4 (capability + target selection, ambiguity is a hard error with no priority
//! mechanism).
//!
//! The `stark-time` cases matter most: its metadata is the real thing this seam exists to consume,
//! and it must resolve **with no change to that crate**.

use starkc::mir::ProviderCallId;
use starkc::provider_abi::{AbiParam, FunctionDecl, ProviderIdentity, ProviderMetadata, ScalarTy};
use starkc::provider_resolve::{
    DeclaredProvider, ProviderCallArena, ProviderSet, ResolveError, SymbolProblem,
};

const LINUX: &str = "x86_64-unknown-linux-gnu";
const MAC: &str = "aarch64-apple-darwin";
const WINDOWS: &str = "x86_64-pc-windows-msvc";

// ------------------------------------------------------------------ fixtures --

/// `stark-time`'s metadata **exactly as `stark-time/native/src/lib.rs` declares it**, including
/// the identity name `stark-std-time` and the `stark_time_*` symbols. Copied rather than
/// paraphrased: this fixture's job is to be that provider, and a tidied-up version would prove
/// nothing about the crate that actually has to work.
fn stark_time() -> DeclaredProvider {
    DeclaredProvider {
        metadata: ProviderMetadata {
            identity: ProviderIdentity {
                name: "stark-std-time".to_string(),
                semver: (0, 1, 0),
                abi_version: "0.1".to_string(),
            },
            target_triples: vec![
                MAC.to_string(),
                "x86_64-apple-darwin".to_string(),
                LINUX.to_string(),
                WINDOWS.to_string(),
            ],
            capabilities: vec!["clock".to_string()],
            resource_types: vec![],
            functions: vec![
                FunctionDecl {
                    name: "stark_time_monotonic_now_ns".to_string(),
                    capability: "clock".to_string(),
                    params: vec![AbiParam::ScalarOut(ScalarTy::U64)],
                    is_close_for: None,
                    may_block: false,
                },
                FunctionDecl {
                    name: "stark_time_unix_now".to_string(),
                    capability: "clock".to_string(),
                    params: vec![AbiParam::ScalarOut(ScalarTy::I64)],
                    is_close_for: None,
                    may_block: false,
                },
            ],
        },
        crate_name: "stark-time-native".to_string(),
        origin: "stark-time/native/Cargo.toml".to_string(),
    }
}

/// A second, independent clock provider — the ambiguity case. Distinct symbols, so ambiguity is
/// isolated from symbol collision.
fn rival_clock() -> DeclaredProvider {
    DeclaredProvider {
        metadata: ProviderMetadata {
            identity: ProviderIdentity {
                name: "rival-clock".to_string(),
                semver: (1, 0, 0),
                abi_version: "0.1".to_string(),
            },
            target_triples: vec![LINUX.to_string()],
            capabilities: vec!["clock".to_string()],
            resource_types: vec![],
            functions: vec![FunctionDecl {
                name: "rival_clock_now".to_string(),
                capability: "clock".to_string(),
                params: vec![AbiParam::ScalarOut(ScalarTy::U64)],
                is_close_for: None,
                may_block: false,
            }],
        },
        crate_name: "rival-clock-native".to_string(),
        origin: "vendor/rival-clock/manifest.json".to_string(),
    }
}

fn clock() -> Vec<String> {
    vec!["clock".to_string()]
}

// ---------------------------------------------------------------- selection --

/// The whole point of the slice: `stark-time` resolves, unmodified, to a validated call record
/// carrying its declaration and its verbatim symbol.
#[test]
fn stark_time_resolves_unmodified() {
    let set = ProviderSet::select(vec![stark_time()], LINUX, &clock())
        .expect("stark-time must select cleanly for a target it declares");

    let call = set
        .resolve("clock", "stark_time_monotonic_now_ns")
        .expect("the declared clock function must resolve");

    assert_eq!(call.provider.name, "stark-std-time");
    assert_eq!(call.capability, "clock");
    assert_eq!(call.target_triple, LINUX);
    assert_eq!(call.symbol(), "stark_time_monotonic_now_ns");
    assert!(matches!(
        call.function.params.as_slice(),
        [AbiParam::ScalarOut(ScalarTy::U64)]
    ));
}

/// Selection is per target. The same provider set resolves on every triple it declares, and the
/// record records which one it was resolved for — A10 §4 invariant 2 re-checks it rather than
/// trusting that resolution ran.
#[test]
fn resolution_records_the_target_it_selected_for() {
    for target in [MAC, LINUX, WINDOWS] {
        let set = ProviderSet::select(vec![stark_time()], target, &clock()).expect("selects");
        let call = set
            .resolve("clock", "stark_time_unix_now")
            .expect("resolves");
        assert_eq!(call.target_triple, target);
    }
}

/// ABI §16 check 2: a provider that does not admit the target is excluded, and the capability is
/// then reported unavailable **with the targets it does declare** — so the diagnostic says
/// "declared, but not for your target" rather than "unknown capability".
#[test]
fn capability_unavailable_for_an_undeclared_target() {
    let errors = ProviderSet::select(vec![stark_time()], "riscv64gc-unknown-linux-gnu", &clock())
        .expect_err("a target no provider declares must fail selection");

    match errors.as_slice() {
        [ResolveError::CapabilityUnavailable {
            capability,
            target,
            declared_for_other_targets,
        }] => {
            assert_eq!(capability, "clock");
            assert_eq!(target, "riscv64gc-unknown-linux-gnu");
            let (name, triples) = &declared_for_other_targets[0];
            assert_eq!(name, "stark-std-time");
            assert!(triples.contains(&LINUX.to_string()));
        }
        other => panic!("expected a single CapabilityUnavailable, got {other:#?}"),
    }
}

/// Packet 1 §1.4: two providers applicable to one capability and target is a **hard error**.
/// Silent first-match would make the binary depend on declaration order, so the test asserts the
/// failure in **both** declaration orders — the exact property a first-match implementation would
/// pass one of and fail the other.
#[test]
fn two_providers_for_one_capability_is_a_hard_error_in_either_order() {
    for declared in [
        vec![stark_time(), rival_clock()],
        vec![rival_clock(), stark_time()],
    ] {
        let errors =
            ProviderSet::select(declared, LINUX, &clock()).expect_err("ambiguity must not resolve");

        match errors.as_slice() {
            [ResolveError::CapabilityAmbiguous {
                capability,
                target,
                providers,
            }] => {
                assert_eq!(capability, "clock");
                assert_eq!(target, LINUX);
                // Both conflicting providers AND their metadata locations are named, so the
                // remediation ("remove one, or narrow its target declarations") is actionable.
                assert_eq!(
                    providers,
                    &[
                        (
                            "rival-clock".to_string(),
                            "vendor/rival-clock/manifest.json".to_string()
                        ),
                        (
                            "stark-std-time".to_string(),
                            "stark-time/native/Cargo.toml".to_string()
                        ),
                    ]
                );
            }
            other => panic!("expected a single CapabilityAmbiguous, got {other:#?}"),
        }
    }
}

/// Ambiguity is per target. The rival declares only Linux, so macOS has exactly one supplier and
/// must still resolve — ambiguity must not be treated as a property of the provider *set*.
#[test]
fn ambiguity_on_one_target_does_not_block_another() {
    let set = ProviderSet::select(vec![stark_time(), rival_clock()], MAC, &clock())
        .expect("macOS has exactly one clock supplier");
    assert_eq!(set.providers().len(), 1);
    assert!(set.resolve("clock", "stark_time_unix_now").is_ok());
}

/// Resolving a capability that was never selected must not search — otherwise a caller could
/// bypass the ambiguity check by resolving directly.
#[test]
fn resolving_an_unselected_capability_does_not_search() {
    let set = ProviderSet::select(vec![stark_time()], LINUX, &clock()).expect("selects");
    assert!(matches!(
        set.resolve("filesystem", "anything"),
        Err(ResolveError::CapabilityUnavailable { .. })
    ));
}

// ------------------------------------------------------------------ symbols --

/// Packet 1 §1.3: an invalid symbol is **rejected, never repaired**. Sanitising would make the
/// metadata name differ from the linkage name, which must never be true when the same field has to
/// resolve under a future `dlsym`.
#[test]
fn invalid_symbols_are_rejected_not_sanitised() {
    for (bad, expect_leading_digit) in [
        ("", false),
        ("9lives", true),
        ("has space", false),
        ("has-hyphen", false),
        ("has.dot", false),
        ("has\0nul", false),
        ("héllo", false),
    ] {
        let mut p = stark_time();
        p.metadata.functions[0].name = bad.to_string();

        let errors = ProviderSet::select(vec![p], LINUX, &clock())
            .expect_err("an invalid symbol must fail selection");

        let found = errors.iter().find_map(|e| match e {
            ResolveError::InvalidSymbol {
                symbol, problem, ..
            } => Some((symbol, problem)),
            _ => None,
        });
        let (symbol, problem) = found.unwrap_or_else(|| panic!("no InvalidSymbol for {bad:?}"));

        // The rejected symbol is reported EXACTLY as declared -- no repaired form anywhere.
        assert_eq!(symbol, bad, "the symbol must be reported verbatim");
        match (expect_leading_digit, problem) {
            (true, SymbolProblem::LeadingDigit) => {}
            (false, SymbolProblem::Empty | SymbolProblem::IllegalByte { .. }) => {}
            (_, other) => panic!("unexpected problem for {bad:?}: {other:?}"),
        }
    }
}

/// The admitted grammar accepts what a C identifier accepts, including a leading underscore and
/// digits after the first byte.
#[test]
fn valid_symbols_are_accepted() {
    // Deliberately excludes names already declared by the fixture -- reusing one would collide
    // with its sibling function and fail on duplication rather than on grammar.
    for good in ["_", "_x", "a", "stark_time_something_new", "f9", "A_1"] {
        let mut p = stark_time();
        p.metadata.functions[0].name = good.to_string();
        ProviderSet::select(vec![p], LINUX, &clock())
            .unwrap_or_else(|e| panic!("{good:?} must be accepted, got {e:#?}"));
    }
}

/// Packet 1 §1.3: two selected providers exporting the same symbol is rejected. This is the
/// *exact* anti-collision guarantee, as opposed to a prefix convention which only makes collision
/// unlikely.
#[test]
fn duplicate_symbols_across_selected_providers_are_rejected() {
    let mut rival = rival_clock();
    rival.metadata.functions[0].name = "stark_time_unix_now".to_string();
    // Keep the capabilities distinct so this fails on the symbol, not on ambiguity.
    rival.metadata.capabilities = vec!["rival".to_string()];
    rival.metadata.functions[0].capability = "rival".to_string();

    // Both capabilities are required, because Packet 5's admission rule means an unrequested
    // provider is never selected -- and a symbol collision is only possible between providers that
    // are actually linked together.
    let errors = ProviderSet::select(
        vec![stark_time(), rival],
        LINUX,
        &["clock".to_string(), "rival".to_string()],
    )
    .expect_err("a duplicate symbol must fail selection");

    match errors.as_slice() {
        [ResolveError::DuplicateSymbol { symbol, providers }] => {
            assert_eq!(symbol, "stark_time_unix_now");
            assert_eq!(providers.len(), 2);
        }
        other => panic!("expected a single DuplicateSymbol, got {other:#?}"),
    }
}

/// A provider declaring the same symbol twice is a **provider** defect, reported separately from
/// a cross-provider conflict. Folding the two together would name one provider twice and send the
/// reader hunting for a second one that does not exist.
#[test]
fn intra_provider_duplicate_is_distinct_from_a_cross_provider_conflict() {
    let mut p = stark_time();
    p.metadata.functions[0].name = "stark_time_unix_now".to_string();

    let errors =
        ProviderSet::select(vec![p], LINUX, &clock()).expect_err("a repeated symbol must fail");

    match errors.as_slice() {
        [ResolveError::DuplicateSymbolWithinProvider {
            provider,
            symbol,
            origin,
        }] => {
            assert_eq!(provider, "stark-std-time");
            assert_eq!(symbol, "stark_time_unix_now");
            assert_eq!(origin, "stark-time/native/Cargo.toml");
        }
        other => panic!("expected a single DuplicateSymbolWithinProvider, got {other:#?}"),
    }
}

/// Symbol uniqueness is scoped to the **selected** set: two providers for different targets can
/// never be linked into one binary, so their identical symbols cannot collide.
#[test]
fn duplicate_symbols_on_unselectable_targets_do_not_collide() {
    let mut other_target = rival_clock();
    other_target.metadata.functions[0].name = "stark_time_unix_now".to_string();
    other_target.metadata.capabilities = vec!["rival".to_string()];
    other_target.metadata.functions[0].capability = "rival".to_string();
    other_target.metadata.target_triples = vec![WINDOWS.to_string()];

    ProviderSet::select(vec![stark_time(), other_target], LINUX, &clock())
        .expect("a provider excluded by target cannot collide on this build");
}

// --------------------------------------------------------------- resolution --

#[test]
fn unknown_function_is_reported_against_the_selected_provider() {
    let set = ProviderSet::select(vec![stark_time()], LINUX, &clock()).expect("selects");
    match set.resolve("clock", "stark_time_no_such_fn") {
        Err(ResolveError::UnknownFunction {
            function, provider, ..
        }) => {
            assert_eq!(function, "stark_time_no_such_fn");
            assert_eq!(provider, "stark-std-time");
        }
        other => panic!("expected UnknownFunction, got {other:#?}"),
    }
}

/// A10 §4 invariant 3: membership is checked against the declaration, not assumed from a
/// successful lookup. A function reachable by name from the right provider can still belong to a
/// different capability, and that must not silently widen what the capability admits.
#[test]
fn function_belonging_to_another_capability_is_rejected() {
    let mut p = stark_time();
    p.metadata.capabilities.push("env".to_string());
    p.metadata.functions.push(FunctionDecl {
        name: "stark_time_env_peek".to_string(),
        capability: "env".to_string(),
        params: vec![AbiParam::ScalarOut(ScalarTy::U64)],
        is_close_for: None,
        may_block: false,
    });

    let set = ProviderSet::select(vec![p], LINUX, &["clock".to_string(), "env".to_string()])
        .expect("selects");

    match set.resolve("clock", "stark_time_env_peek") {
        Err(ResolveError::FunctionCapabilityMismatch {
            declared,
            requested,
            ..
        }) => {
            assert_eq!(declared, "env");
            assert_eq!(requested, "clock");
        }
        other => panic!("expected FunctionCapabilityMismatch, got {other:#?}"),
    }
}

/// Selection reports **every** failure, not the first — three misconfigured providers should
/// produce three findings, not three rebuilds.
#[test]
fn selection_reports_every_failure() {
    let mut bad_symbol = stark_time();
    bad_symbol.metadata.functions[0].name = "has space".to_string();

    let errors = ProviderSet::select(vec![bad_symbol, rival_clock()], LINUX, &clock())
        .expect_err("must fail");

    assert!(
        errors
            .iter()
            .any(|e| matches!(e, ResolveError::InvalidSymbol { .. })),
        "expected the symbol failure: {errors:#?}"
    );
    assert!(
        errors
            .iter()
            .any(|e| matches!(e, ResolveError::CapabilityAmbiguous { .. })),
        "expected the ambiguity failure too: {errors:#?}"
    );
}

// -------------------------------------------------------------------- arena --

/// Interning deduplicates: one provider function called from many bodies produces one record and
/// one id, so the arena is deterministic regardless of body iteration order.
#[test]
fn arena_interning_is_deduplicating_and_stable() {
    let set = ProviderSet::select(vec![stark_time()], LINUX, &clock()).expect("selects");
    let mono = set.resolve("clock", "stark_time_monotonic_now_ns").unwrap();
    let unix = set.resolve("clock", "stark_time_unix_now").unwrap();

    let mut arena = ProviderCallArena::new();
    assert_eq!(arena.intern(mono.clone()), ProviderCallId(0));
    assert_eq!(arena.intern(unix.clone()), ProviderCallId(1));
    // Re-interning either returns the SAME id rather than growing the arena.
    assert_eq!(arena.intern(mono), ProviderCallId(0));
    assert_eq!(arena.intern(unix), ProviderCallId(1));
    assert_eq!(arena.len(), 2);

    let calls = arena.into_calls();
    assert_eq!(calls[0].symbol(), "stark_time_monotonic_now_ns");
    assert_eq!(calls[1].symbol(), "stark_time_unix_now");
}

/// The same call resolved for two different targets is two distinct records: the target is part of
/// the validated contract (A10 §4 invariant 2), not incidental metadata.
#[test]
fn records_resolved_for_different_targets_do_not_dedup() {
    let linux = ProviderSet::select(vec![stark_time()], LINUX, &clock())
        .unwrap()
        .resolve("clock", "stark_time_unix_now")
        .unwrap();
    let mac = ProviderSet::select(vec![stark_time()], MAC, &clock())
        .unwrap()
        .resolve("clock", "stark_time_unix_now")
        .unwrap();

    let mut arena = ProviderCallArena::new();
    assert_eq!(arena.intern(linux), ProviderCallId(0));
    assert_eq!(arena.intern(mac), ProviderCallId(1));
}
