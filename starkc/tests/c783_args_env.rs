//! WP-C7.8.3 — process arguments and environment.
//!
//! Packet 4 places these as **package capabilities**, never Core: nothing here adds a Core symbol,
//! and `06-Standard-Library.md` is untouched. Packet 5 makes them **read-only** — there is no
//! environment-mutating function in the provider, and none may be added in C7.8.
//!
//! `stark-env` is also the first provider with a **non-empty status vocabulary**, so this is where
//! ABI §12's channel one stops being vacuous. Every earlier slice had only `stark-time`, which
//! declares no recoverable status and therefore treats every nonzero code as a contract violation.

use starkc::mir::ProviderCallId;
use starkc::provider_abi::AbiParam;
use starkc::provider_bind::{plan, ResourceRegistry, StatusOutcome, STATUS_SUCCESS};
use starkc::provider_registry;
use starkc::provider_resolve::ProviderSet;

const LINUX: &str = "x86_64-unknown-linux-gnu";

fn args_env() -> Vec<String> {
    vec!["process.args".to_string(), "process.env".to_string()]
}

fn selected() -> ProviderSet {
    ProviderSet::select(provider_registry::first_party(), LINUX, &args_env())
        .expect("process.args and process.env must be available on a Tier-1 target")
}

// ------------------------------------------------------------- selection --

/// Both capabilities resolve, and they resolve to the **same** provider — `stark-env` supplies
/// both, so requiring either selects one crate rather than two.
#[test]
fn both_capabilities_resolve_to_one_provider() {
    let set = selected();
    assert_eq!(set.providers().len(), 1);
    assert_eq!(set.providers()[0].crate_name, "stark-env-native");

    for (capability, function) in [
        ("process.args", "stark_env_args_len"),
        ("process.args", "stark_env_args_fill"),
        ("process.env", "stark_env_var_len"),
        ("process.env", "stark_env_var_fill"),
    ] {
        set.resolve(capability, function)
            .unwrap_or_else(|e| panic!("{capability}/{function} must resolve: {e:#?}"));
    }
}

/// Requiring only `process.env` still selects the provider — a package that reads an environment
/// variable but never inspects its arguments does not have to declare both.
#[test]
fn one_capability_is_enough_to_select() {
    let set = ProviderSet::select(
        provider_registry::first_party(),
        LINUX,
        &["process.env".to_string()],
    )
    .expect("selects");
    assert_eq!(set.providers().len(), 1);
}

/// Requiring `clock` alone must **not** drag in `stark-env`. Packet 5's admission rule again: a
/// provider is linked only if something declared its capability.
#[test]
fn an_undeclared_capability_does_not_select_its_provider() {
    let set = ProviderSet::select(
        provider_registry::first_party(),
        LINUX,
        &["clock".to_string()],
    )
    .expect("selects");
    assert_eq!(set.providers().len(), 1);
    assert_eq!(set.providers()[0].crate_name, "stark-time-native");
    assert!(
        set.resolve("process.env", "stark_env_var_len").is_err(),
        "an unrequested capability must not be resolvable"
    );
}

// --------------------------------------------------- the two-call shape --

/// ABI §9 gives the provider no way to allocate for the caller, so a value of unknown size takes
/// two calls: ask the length, allocate, then have the provider fill a `BufferInOut`.
///
/// The plan must classify that correctly — the fill buffer is an **input** (caller-owned, §11.1),
/// while the byte count is an **output** (uninitialised until success).
#[test]
fn the_fill_call_treats_its_buffer_as_caller_owned() {
    let call = selected()
        .resolve("process.args", "stark_env_args_fill")
        .expect("resolves");
    let p = plan(
        ProviderCallId(0),
        &call,
        &ResourceRegistry::builtin(),
        call.status_binding.clone(),
    )
    .expect("plans");

    assert!(
        matches!(
            call.function.params.as_slice(),
            [AbiParam::BufferInOut, AbiParam::ScalarOut(_)]
        ),
        "{:#?}",
        call.function.params
    );
    // The buffer is caller-owned: an input, never a MaybeUninit output slot.
    assert_eq!(p.inputs.len(), 1);
    assert_eq!(p.outputs.len(), 1);
    assert_eq!(p.inputs[0].index(), 0);
    assert_eq!(p.outputs[0].index(), 1);
    assert!(p.inputs[0].requires_live_borrow());
}

/// `stark_env_var_len` reports presence through a `ScalarOut(Bool)` **separately** from the length,
/// so "absent" is distinguishable from "present and empty" without a sentinel length.
#[test]
fn presence_is_reported_separately_from_length() {
    let call = selected()
        .resolve("process.env", "stark_env_var_len")
        .expect("resolves");
    assert!(
        matches!(
            call.function.params.as_slice(),
            [
                AbiParam::BufferIn,
                AbiParam::ScalarOut(starkc::provider_abi::ScalarTy::Bool),
                AbiParam::ScalarOut(starkc::provider_abi::ScalarTy::U64),
            ]
        ),
        "{:#?}",
        call.function.params
    );
}

// --------------------------------------------- channel one, for real --

/// **The first non-vacuous channel-one test.** `stark-env` declares four recoverable codes, so
/// those map to package errors while everything else remains a contract violation.
///
/// Until this slice every provider declared none, which meant the declared-error arm was never
/// exercised against a real vocabulary — only its absence was.
#[test]
fn declared_codes_are_recoverable_and_the_rest_are_violations() {
    let call = selected()
        .resolve("process.env", "stark_env_var_len")
        .expect("resolves");
    let p = plan(
        ProviderCallId(0),
        &call,
        &ResourceRegistry::builtin(),
        call.status_binding.clone(),
    )
    .expect("plans");

    assert!(
        !call.status_binding.is_empty(),
        "stark-env must declare a status vocabulary"
    );
    assert_eq!(p.classify(STATUS_SUCCESS), StatusOutcome::Success);

    for (code, expected) in [
        (1u32, "ProcessError::InvalidName"),
        (2, "ProcessError::InvalidEncoding"),
        (3, "ProcessError::BufferTooSmall"),
        (4, "ProcessError::Unsupported"),
    ] {
        assert_eq!(
            p.classify(code),
            StatusOutcome::RecoverableError {
                code,
                package_error: expected.to_string()
            },
            "code {code} must be recoverable"
        );
    }

    // Anything the package did not declare stays channel two, however plausible it looks.
    for code in [5u32, 99, u32::MAX] {
        assert_eq!(
            p.classify(code),
            StatusOutcome::ContractViolation { code },
            "code {code} is undeclared and must be a contract violation"
        );
    }
}

/// `stark-time` still declares nothing, so the two providers demonstrate both shapes side by side.
/// The vocabulary is per provider, not global.
#[test]
fn the_status_vocabulary_is_per_provider() {
    let clock = ProviderSet::select(
        provider_registry::first_party(),
        LINUX,
        &["clock".to_string()],
    )
    .expect("selects")
    .resolve("clock", "stark_time_monotonic_now_ns")
    .expect("resolves");

    assert!(
        clock.status_binding.is_empty(),
        "stark-time declares no recoverable status"
    );

    let env = selected()
        .resolve("process.env", "stark_env_var_fill")
        .expect("resolves");
    assert!(!env.status_binding.is_empty());
}

// ------------------------------------------------------- trust boundary --

/// **Packet 5: the environment is read-only, and no provider executes a process.**
///
/// Two rules with different scopes, which this used to conflate by applying one substring list to
/// every provider:
///
/// - **Process execution is forbidden everywhere.** No provider may spawn or exec, whatever it is
///   for. That is a whole-registry rule and stays one.
/// - **Mutation is forbidden for the PROCESS capabilities**, which is what "the environment is
///   read-only" means. It was never a whole-registry rule: `stark-std-file` has declared
///   `stark_file_write` and `stark_file_create` since C7.8.4, and a filesystem provider that
///   cannot write is not a filesystem provider.
///
/// Applied globally, the mutation list was a NAME filter standing in for a semantic rule — it
/// passed `write` and `create` (which mutate) while it would reject `set_len` and `remove` (which
/// mutate no more). CD-292's expanded file surface is what surfaced the inconsistency: the guard
/// failed on `stark_iofile_set_len` while the equally-mutating `stark_iofile_write` beside it went
/// through. Scoped to the capabilities the rule is actually about, it tests the rule again.
#[test]
fn no_environment_mutating_function_is_declared() {
    for provider in provider_registry::first_party() {
        for f in &provider.metadata.functions {
            let name = f.name.to_ascii_lowercase();
            // Whole-registry: nothing runs a process.
            for forbidden in ["exec", "spawn"] {
                assert!(
                    !name.contains(forbidden),
                    "{} declares `{}`, which looks like a process-executing operation; Packet 5 \
                     admits none in C7.8, from any provider",
                    provider.metadata.identity.name,
                    f.name
                );
            }
            // Process capabilities only: the environment is observed, never altered.
            if !f.capability.starts_with("process.") {
                continue;
            }
            for forbidden in ["set", "put", "unset", "remove", "clear"] {
                assert!(
                    !name.contains(forbidden),
                    "{} declares `{}` under capability `{}`, which looks like a mutating \
                     operation; Packet 5 makes the process environment read-only in C7.8",
                    provider.metadata.identity.name,
                    f.name,
                    f.capability
                );
            }
        }
    }
}

/// Every resource type any registered provider declares is either **bound**, or **refused by name**
/// when a call carrying it is planned. What is ruled out is the middle case: a declared resource
/// type that is silently accepted without a MIR type behind it.
///
/// `file` is bound (C7.8.4). `tcp_listener`, `tcp_stream` and `tls_stream` are not, and deliberately
/// so — Packet 4 makes them package types, so binding them needs a package declaration rather than a
/// Core change. This asserts the refusal is precise rather than asserting the binding is complete.
#[test]
fn every_declared_resource_type_is_bound_or_precisely_refused() {
    use starkc::provider_bind::{plan, PlanError, ResourceRegistry};
    let registry = ResourceRegistry::builtin();
    let all = provider_registry::first_party();

    for provider in provider_registry::first_party() {
        for resource in &provider.metadata.resource_types {
            if registry.lookup(resource).is_some() {
                continue;
            }
            // Unbound: find a function carrying it and confirm the refusal names it.
            let carrier = provider
                .metadata
                .functions
                .iter()
                .find(|f| {
                    f.params.iter().any(|p| match p {
                        AbiParam::HandleBorrowed { resource_type }
                        | AbiParam::HandleConsumed { resource_type }
                        | AbiParam::HandleOut { resource_type } => resource_type == resource,
                        _ => false,
                    })
                })
                .unwrap_or_else(|| {
                    panic!(
                        "{} declares resource type `{resource}` but no function uses it",
                        provider.metadata.identity.name
                    )
                });

            // HC9: a provider that CONSUMES another's resource cannot be selected alone —
            // CD-360's rule in `ProviderSet::select` requires the owner in the same set, so
            // asking for `tls` without `net` is refused before any planning happens. Requiring
            // the owners' capabilities alongside is what a real build of such a package does,
            // and it leaves the assertion below untouched: the carrier's resource type is still
            // unbound, so the refusal must still name it.
            let mut required = provider.metadata.capabilities.clone();
            for foreign in &provider.metadata.foreign_resources {
                let owner = all
                    .iter()
                    .find(|p| p.metadata.identity.name == foreign.provider)
                    .unwrap_or_else(|| {
                        panic!(
                            "{} consumes `{}` from `{}`, which is not a registered provider",
                            provider.metadata.identity.name, foreign.resource, foreign.provider
                        )
                    });
                required.extend(owner.metadata.capabilities.iter().cloned());
            }
            required.sort();
            required.dedup();

            let set = ProviderSet::select(provider_registry::first_party(), LINUX, &required)
                .expect("selects");
            let call = set
                .resolve(&carrier.capability, &carrier.name)
                .expect("resolves");

            assert!(
                matches!(
                    plan(
                        ProviderCallId(0),
                        &call,
                        &registry,
                        call.status_binding.clone()
                    ),
                    Err(PlanError::UnboundResourceType { .. })
                ),
                "{}: `{resource}` is unbound, so a call carrying it must be refused by name",
                provider.metadata.identity.name
            );
        }
    }
}
