//! WP-C7.8.8 step 3 — synthesizing the raw binding layer as STARK source.
//!
//! **Why source text rather than constructed HIR.** Every name in HIR is a `Span` into a
//! `SourceFile` (`hir::FnSig::name`, `ItemKind::Struct::name`, …). A synthesized item with no source
//! has no span, so constructing HIR directly would mean either fabricating spans that point
//! nowhere — and surface in every diagnostic that touches them — or threading an alternative name
//! representation through the whole front end.
//!
//! Generating source instead means name resolution, type checking and visibility are **ordinary**.
//! That was the design's claim (§3: "by the time they run, these are ordinary items"), and this is
//! what makes it literally true rather than approximately.
//!
//! **The body is never lowered.** A generated function's body is `panic(…)`, which has type `!` and
//! so satisfies any return type. Lowering ignores it and emits `Callee::Provider` from the binding
//! (step 6). The body exists because Core v1 has no bodyless-`fn` grammar and adding one would be
//! the CE1 grammar change Packet 4 avoids — not because anything runs it.
//!
//! **The raw error types are generated here too**, one per raw error name the bound signatures
//! reference, with one variant per declared status code. §7.2 approved that status mapping stays in
//! ordinary STARK and that the manifest carries no code→variant table — so the compiler has to own
//! the raw type, because otherwise nothing could say which variant means status 3. A capability
//! declaring no recoverable status gets an **uninhabited** enum, which is the type-system statement
//! of "every nonzero status here is a contract violation".
//!
//! **Resource nominals ARE synthesized here, as zero-variant enums** (CD-234). See
//! [`RESOURCE_SYNTHESIS_LIMIT`] for why every other source form was unusable and why this one is not.

use crate::provider_abi::ScalarTy;
use crate::provider_bind::StatusBinding;
use crate::provider_derive::{DerivedSignature, DerivedTy};
use std::collections::BTreeMap;

/// Why resource nominals need a different mechanism from functions.
///
/// A host resource must be **opaque**: no fields, no constructor (design §6, A11 §6). Every source
/// form that declares a nominal is constructible — `struct S;` and `struct S {}` both admit `S` or
/// `S {}` at a use site — so generating source for a resource would hand programs a way to forge a
/// handle that no provider ever produced.
///
/// Function synthesis has no such problem: a generated `fn` is called, never constructed.
///
/// So resource nominals are deferred to the slice that needs them (`File`, then TCP), and will need
/// either a compiler-injected opaque item form or a source form the checker refuses to construct.
/// **This is recorded rather than worked around**, because a forged handle reaching
/// `from_raw_checked` would pass its `resource_type` check — the id would be whatever the forger
/// wrote — and the ABI's validation would not save it.
pub const RESOURCE_SYNTHESIS_LIMIT: &str =
    "host-resource nominals cannot be synthesized as an ordinary constructible nominal: every such \
     source form admits a value at a use site, and a host resource must not";

/// **CD-234's answer: a zero-variant enum.**
///
/// `enum TcpStream {}` is opaque *structurally* rather than by prohibition — there is no variant to
/// name and no struct-literal form, so no expression and no pattern can manufacture a value. It is
/// still an ordinary parsed item with real spans and ordinary name resolution, and it needs no marker
/// that a future construction path could forget.
///
/// The nominal supplies **source identity only**. A provider-bound instance lowers to
/// `MirTy::HostResource` and must never take an ordinary zero-variant enum's backend representation
/// or default-initialisation (CD-234's soundness condition).
fn resource_nominal_source(nominal: &str, provider_resource: &str) -> String {
    format!(
        "// Host resource nominal for the provider resource `{provider_resource}`.\n\
         // ZERO VARIANTS, deliberately: no expression and no pattern can manufacture a value, so\n\
         // opacity is structural. Instances come only from a provider call, and lower to\n\
         // MirTy::HostResource -- never to this enum's own representation (CD-234).\n\
         pub enum {nominal} {{ }}\n"
    )
}

/// The **raw** variant name for a declared status, from the vocabulary's name for it.
///
/// A vocabulary names a status with the *package-facing* error it corresponds to, and the first-party
/// registry writes those qualified: `stark-env` declares code 1 as `"ProcessError::InvalidName"`.
/// `ProcessError` is by §7.2's own account the **public** type package code maps the raw result *to*,
/// so the raw variant is the final segment — `InvalidName`.
///
/// Taking the last segment is interpretation, so the result is **validated** rather than trusted: a
/// name that is not a legal STARK identifier is refused here, where the vocabulary can be pointed at,
/// instead of emitting source that fails to parse with no indication of where it came from.
fn raw_variant_name(declared: &str, error_ty: &str, code: u32) -> Result<String, String> {
    let leaf = declared.rsplit("::").next().unwrap_or(declared).trim();
    let legal = !leaf.is_empty()
        && !leaf.starts_with(|c: char| c.is_ascii_digit())
        && leaf.chars().all(|c| c.is_ascii_alphanumeric() || c == '_');
    if !legal {
        return Err(format!(
            "raw error type `{error_ty}` cannot name status {code}: the vocabulary calls it \
             `{declared}`, whose final segment `{leaf}` is not a legal STARK variant name"
        ));
    }
    Ok(leaf.to_string())
}

/// Raw error type name → declared status code → that code's variant index in the generated enum.
pub type ErrorVariants = BTreeMap<String, BTreeMap<u32, u32>>;

/// A package's synthesized raw layer: the source text, and the binding each item carries.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SynthesizedLayer {
    /// STARK source for the raw layer. Empty when the package binds nothing.
    pub source: String,
    /// Item path → `(capability, symbol)`, for lowering to consult (step 6).
    ///
    /// A side table rather than a field on the HIR item, because HIR is built by the ordinary
    /// parser from the source above and carries no provider vocabulary. The binding is **carried,
    /// not consulted**, until lowering — which is what keeps name resolution and type checking free
    /// of special cases.
    pub bindings: BTreeMap<String, (String, String)>,
    /// Raw error type name → declared status code → that code's **variant index** in the generated
    /// enum.
    ///
    /// Lowering needs the index, not the name, to build `Err(RawE::V)` as an
    /// `AggKind::EnumVariant`. Computed here because this is where the enum's declaration order is
    /// decided, and a second computation elsewhere could disagree with it.
    pub error_variants: ErrorVariants,
    /// Provider resource name → the package nominal bound to it, for the registry to resolve to an
    /// `ItemId` once the generated source has been parsed (A11 §4's implementation note).
    pub resource_nominals: BTreeMap<String, String>,
}

fn scalar_src(s: ScalarTy) -> &'static str {
    match s {
        ScalarTy::U8 => "UInt8",
        ScalarTy::U16 => "UInt16",
        ScalarTy::U32 => "UInt32",
        ScalarTy::U64 => "UInt64",
        ScalarTy::I8 => "Int8",
        ScalarTy::I16 => "Int16",
        ScalarTy::I32 => "Int32",
        ScalarTy::I64 => "Int64",
        ScalarTy::Bool => "Bool",
        ScalarTy::F32 => "Float32",
        ScalarTy::F64 => "Float64",
    }
}

fn ty_src(t: &DerivedTy) -> String {
    match t {
        DerivedTy::Scalar(s) => scalar_src(*s).to_string(),
        DerivedTy::ScalarSlot(s) => format!("&mut {}", scalar_src(*s)),
        DerivedTy::SharedBytes => "&[UInt8]".to_string(),
        DerivedTy::ExclusiveBytes => "&mut [UInt8]".to_string(),
        DerivedTy::SharedResource { nominal } => format!("&{nominal}"),
        DerivedTy::OwnedResource { nominal } => nominal.clone(),
    }
}

/// The function name from an item path: `Instant::now_ns` → `now_ns`.
fn leaf(item_path: &str) -> &str {
    item_path.rsplit_once("::").map_or(item_path, |(_, n)| n)
}

/// Generates the raw layer for a set of derived signatures.
///
/// Free functions only for now: associated placement (CD-225 §7.1) needs the receiver's nominal to
/// exist, and resource nominals are not synthesizable (see [`RESOURCE_SYNTHESIS_LIMIT`]). A
/// signature with a receiver is therefore **refused** rather than emitted as a free function with a
/// silently different call shape.
pub fn synthesize(
    signatures: &[DerivedSignature],
    vocabularies: &BTreeMap<String, StatusBinding>,
) -> Result<SynthesizedLayer, String> {
    synthesize_with_resources(signatures, vocabularies, &BTreeMap::new(), &BTreeMap::new())
}

/// `synthesize`, plus the resource nominals in play.
///
/// `resources` maps a **provider resource name** to the package **nominal** bound to it, as
/// `provider_api.resources` declares it. Each becomes a zero-variant enum (CD-234).
///
/// **HC9 — `foreign` is the other half, and the two must not be merged.** It maps a provider
/// resource name to a QUALIFIED path at the package that owns it (`stark_net::TcpStream`), as
/// `provider_api.foreign_resources` declares it. A signature may name one, and nothing is generated
/// for it: the nominal already exists in the owning package, so generating a second would produce a
/// distinct `ItemId` — a type with the same spelling that no provider call could ever produce a
/// value of. Merging the two maps is therefore not a simplification, it is that bug.
pub fn synthesize_with_resources(
    signatures: &[DerivedSignature],
    vocabularies: &BTreeMap<String, StatusBinding>,
    resources: &BTreeMap<String, String>,
    foreign: &BTreeMap<String, String>,
) -> Result<SynthesizedLayer, String> {
    if signatures.is_empty() && resources.is_empty() {
        return Ok(SynthesizedLayer::default());
    }

    let (error_source, error_variants) = synthesize_error_types(signatures, vocabularies)?;

    let mut source = String::from(
        "// GENERATED by the STARK compiler (WP-C7.8.8). The raw provider binding layer.\n\
         // Package-private: a package exports its own API over this, never this itself.\n\
         // Bodies are never lowered -- a provider call is emitted from the binding (CD-225).\n",
    );
    source.push_str(&error_source);

    // Resource nominals first: a function signature may name one, and although STARK items are
    // order-independent, emitting them first keeps the generated unit readable. `resources` is a
    // `BTreeMap`, so the order is name-derived rather than iteration-derived.
    let mut resource_nominals = BTreeMap::new();
    for (provider_resource, nominal) in resources {
        source.push_str(&resource_nominal_source(nominal, provider_resource));
        resource_nominals.insert(provider_resource.clone(), nominal.clone());
    }

    let mut bindings = BTreeMap::new();

    for sig in signatures {
        // A receiver is still refused: associated placement (CD-225 §7.1) needs an `impl` block on the
        // nominal, and emitting the item as a free function instead would silently change the call
        // shape a programmer writes. Resource TYPES are now fine -- CD-234 gave them a form.
        if sig.receiver.is_some() {
            return Err(format!(
                "{}: associated placement on a host-resource nominal is not synthesized yet; the \
                 nominal itself is (CD-234)",
                sig.item_path
            ));
        }
        // Every resource a signature names must be a nominal this package bound, or one it
        // declared as foreign. Otherwise the generated source would reference a type that does not
        // exist -- caught here rather than as an unresolved-name diagnostic in generated code
        // nobody wrote.
        for t in sig.params.iter().chain(sig.results.iter()) {
            if let DerivedTy::SharedResource { nominal } | DerivedTy::OwnedResource { nominal } = t
            {
                let bound = resource_nominals.values().any(|n| n == nominal);
                let borrowed = foreign.values().any(|n| n == nominal);
                if !bound && !borrowed {
                    return Err(format!(
                        "{}: references host-resource nominal `{nominal}`, which this package \
                         neither binds nor declares in `provider_api.foreign_resources`",
                        sig.item_path
                    ));
                }
            }
        }

        let params = sig
            .params
            .iter()
            .enumerate()
            .map(|(i, t)| format!("a{i}: {}", ty_src(t)))
            .collect::<Vec<_>>()
            .join(", ");

        let ok = match sig.results.len() {
            0 => "Unit".to_string(),
            1 => ty_src(&sig.results[0]),
            _ => format!(
                "({})",
                sig.results
                    .iter()
                    .map(ty_src)
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        };

        // `panic` has type `!`, so it satisfies any return type -- which is what lets a body exist
        // without being meaningful. Lowering never reaches it.
        //
        // **No trailing semicolon.** `panic(...)` as a statement makes the block's value `Unit` and
        // the body stops typechecking; only as the tail expression does `!` reach the return type.
        source.push_str(&format!(
            "fn {}({params}) -> Result<{ok}, {}> {{ panic(\"provider binding not lowered\") }}\n",
            leaf(&sig.item_path),
            sig.error
        ));

        bindings.insert(
            sig.item_path.clone(),
            (sig.capability.clone(), sig.symbol.clone()),
        );
    }

    Ok(SynthesizedLayer {
        source,
        bindings,
        error_variants,
        resource_nominals,
    })
}

/// Generates the **raw error types**, one per raw error name the signatures reference.
///
/// **Why the compiler generates these rather than the package declaring them.** §7.2 approved that
/// status mapping stays in ordinary STARK, and that the manifest carries "only the minimum raw error
/// identity" with **no** status-code→variant table. Those two together leave no way for a
/// package-declared enum to tell the compiler which variant means status 3 — so the raw type has to
/// be derived from the same validated vocabulary the emitter dispatches on. §7.2's division survives
/// exactly: the compiler produces the raw typed result, and package code maps it to a public
/// `IOError`/`ProcessError` in ordinary STARK.
///
/// **An empty vocabulary generates an uninhabited enum**, and that is the point rather than a
/// degenerate case. A capability that declares no recoverable status gets
/// `enum RawProviderError { }`, and `Result<T, RawProviderError>`'s `Err` arm is *provably*
/// unreachable — the type system says what Packet 1 §1.2 says, which is that every nonzero status
/// from that provider is a contract violation and no package code runs on it.
fn synthesize_error_types(
    signatures: &[DerivedSignature],
    vocabularies: &BTreeMap<String, StatusBinding>,
) -> Result<(String, ErrorVariants), String> {
    // Error type name -> code -> variant name. A `BTreeMap` on the outer key so declaration order
    // is name-ordered, and on the inner so variant order is code-ordered: generated source and the
    // variant indices below are then a function of the vocabulary alone, never of iteration order.
    let mut by_type: BTreeMap<String, BTreeMap<u32, String>> = BTreeMap::new();

    for sig in signatures {
        let vocabulary = vocabularies.get(&sig.capability).ok_or_else(|| {
            format!(
                "{}: capability `{}` has no status vocabulary, so its raw error type cannot be \
                 derived -- the vocabulary is what the variants ARE",
                sig.item_path, sig.capability
            )
        })?;

        let entry = by_type.entry(sig.error.clone()).or_default();
        for (code, name) in vocabulary.declared_codes() {
            // Status 0 is SUCCESS (ABI 11). A vocabulary declaring it as a recoverable error is a
            // contradiction, and it is refused here rather than tolerated: `classify` tests success
            // first, so the declaration would be silently shadowed, and lowering would emit two
            // `SwitchInt` arms for 0. Neither failure names the actual mistake.
            if *code == crate::provider_bind::STATUS_SUCCESS {
                return Err(format!(
                    "raw error type `{}` declares status 0 as the recoverable error `{name}`, but 0 \
                     is SUCCESS",
                    sig.error
                ));
            }
            // Two capabilities may name the same raw error type. That is fine while they agree; a
            // disagreement would silently give one code two meanings in one enum, so it is refused.
            let derived = raw_variant_name(name, &sig.error, *code)?;
            if let Some(existing) = entry.get(code) {
                if *existing != derived {
                    return Err(format!(
                        "raw error type `{}` is given two different names for status {code}: \
                         `{existing}` and `{derived}`",
                        sig.error
                    ));
                }
            }
            entry.insert(*code, derived);
        }
    }

    let mut source = String::new();
    let mut variants = BTreeMap::new();
    for (ty, cases) in &by_type {
        let mut index_of = BTreeMap::new();
        let names: Vec<&str> = cases
            .iter()
            .enumerate()
            .map(|(i, (code, name))| {
                index_of.insert(*code, i as u32);
                name.as_str()
            })
            .collect();

        if names.is_empty() {
            source.push_str(&format!(
                "// `{ty}` is UNINHABITED: this capability declares no recoverable status, so every\n\
                 // nonzero code is a contract violation and the `Err` arm cannot be constructed.\n\
                 enum {ty} {{ }}\n"
            ));
        } else {
            source.push_str(&format!("enum {ty} {{ {} }}\n", names.join(", ")));
        }
        variants.insert(ty.clone(), index_of);
    }

    Ok((source, variants))
}
