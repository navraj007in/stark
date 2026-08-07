//! Library-owned orchestration for `stark build`.

use crate::analysis::{analyze_project, ProjectInput};
use crate::backend::generated_rust::{
    emit_native_debug_with_toolchain, BackendDiagnostic, NativeBuildOptions,
    NativeToolchainOptions, Profile,
};
use crate::mir::{
    lower::lower_program_with_providers, provider_lower::ProviderLowering, verify::verify_program,
};
use crate::native_toolchain::{self, ToolchainError, ToolchainInfo};
use crate::options::LanguageOptions;
use crate::package::{find_package_root, PackageGraph};
use crate::provider_derive::DerivedSignature;
use crate::provider_resolve::ProviderSet;
use crate::provider_synth::SynthesizedLayer;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct BuildCommandOptions {
    pub locked: bool,
    pub offline: bool,
    pub keep_generated: bool,
    pub emit_rust: bool,
    pub verbose: bool,
    /// WP-C7.1 `--release`.
    pub release: bool,
    /// WP-C7.1 `--target <triple>`. `None` builds for the host.
    pub target: Option<String>,
    /// WP-C7.3 `--no-build-cache`: delete the generated crate after the build, as every build did
    /// before the cache existed. This is the qualification path — a run that must not benefit from,
    /// or be influenced by, anything a previous build left behind.
    pub no_build_cache: bool,
    /// WP-C7.4 `--no-mir-opt`: lower to MIR and hand it to the backend exactly as lowered.
    ///
    /// The baseline optimisations are on by default because they are required to be observationally
    /// transparent — if one is not, that is a defect to fix, not a reason to keep the pass off. The
    /// flag exists so a divergence can be BISECTED against unoptimised MIR, which Gate C7 makes the
    /// higher authority, without rebuilding the compiler.
    pub no_mir_opt: bool,
}

impl BuildCommandOptions {
    pub fn profile(&self) -> Profile {
        if self.release {
            Profile::Release
        } else {
            Profile::Debug
        }
    }
}

#[derive(Clone, Debug)]
pub struct BuildCommandResult {
    pub package_name: String,
    /// WP-C7.1: the profile this build actually used. Reported rather than re-derived: the CLI
    /// printed a hard-coded "[debug]" for a `--release` build until this field existed, so the
    /// message and the artefact path disagreed about the same build.
    pub profile: Profile,
    pub package_root: PathBuf,
    pub artifact_path: PathBuf,
    pub generated_dir: Option<PathBuf>,
    pub generated_rust: Option<PathBuf>,
    pub backend_artifact: Option<PathBuf>,
    pub mir_bodies: usize,
    pub toolchain: ToolchainInfo,
    /// WP-C7.3: what the post-build LRU sweep removed, or `None` when the cache was disabled.
    pub cache_eviction: Option<crate::build_cache::EvictionReport>,
    /// WP-C7.4: what the MIR optimiser changed, or `None` when `--no-mir-opt` was passed.
    pub mir_opt: Option<crate::mir::opt::OptStats>,
}

/// Every capability declared across the package graph, sorted and deduplicated.
///
/// The union across the graph, not just the root: a dependency that needs a clock needs it whether
/// or not the root package mentions one. Sorted so the requirement set — and therefore the
/// selected provider set, and therefore the generated manifest — cannot depend on graph iteration
/// order.
/// The `provider_api` source overlay a package graph needs, for callers that ANALYSE rather than
/// build — `stark test` above all.
///
/// **Why this exists.** `provider_api` bindings are not real source: `provider_synth` generates
/// them as items and merges them into the entry file before the front end runs. A caller that skips
/// that step sees every generated `*_raw` function as E0200 "undefined variable" and the package
/// fails to compile. `stark test` skipped it, so no package declaring `provider_api` could run its
/// own tests at all — `stark-io` reported eighteen undefined variables before discovering a single
/// test.
///
/// **The triple comes from `target::host_triple_of_this_build`, not from probing `rustc`.** A native
/// build needs the toolchain regardless, so `native_toolchain::discover` is right there. Analysis
/// does not compile anything, and requiring a Rust toolchain to run interpreter-only tests would
/// make a machine without one unable to test a STARK package. The triple is used solely to gate
/// which providers are available, and for a host-run test the host's providers are the correct
/// answer.
///
/// **What this does NOT change:** the reference interpreter cannot PERFORM a provider call. This
/// makes a provider-bound package compile and its provider-free tests run; a test that actually
/// reaches a provider still cannot execute under `stark test`, and that is a property of the
/// interpreter rather than of synthesis.
///
/// Returns an empty map when the graph declares no capability — such a package needs no provider
/// set, no triple, and no overlay.
pub fn provider_overlays_for_analysis(
    graph: &PackageGraph,
) -> Result<HashMap<PathBuf, String>, BuildCommandError> {
    let required = required_capabilities(graph);
    if required.is_empty() {
        return Ok(HashMap::new());
    }
    let Some(triple) = crate::target::host_triple_of_this_build() else {
        return Err(BuildCommandError::Capability(format!(
            "this compiler was built for {}-{}, which is not a target it knows, so the providers \
             for capabilities {:?} cannot be selected",
            std::env::consts::ARCH,
            std::env::consts::OS,
            required
        )));
    };
    let set = ProviderSet::select(crate::provider_registry::first_party(), triple, &required)
        .map_err(|errors| {
            BuildCommandError::Capability(render_capability_errors(&errors, triple))
        })?;
    Ok(provider_layer_for_build(graph, Some(&set))?.overlays)
}

fn required_capabilities(graph: &PackageGraph) -> Vec<String> {
    let mut caps: Vec<String> = graph
        .packages
        .values()
        .flat_map(|p| p.capabilities.iter().cloned())
        .collect();
    caps.sort();
    caps.dedup();
    caps
}

/// Selects the providers for the declared capabilities, or fails the build.
///
/// **Packet 5: `stark build` must fail when a required capability has no unique selected
/// provider.** Ambiguity is not resolved by a priority rule, declaration order, or a fallback —
/// each of those would make the produced binary depend on something other than what the program
/// asked for.
fn select_provider_set(
    required: &[String],
    target: Option<&str>,
    toolchain: &crate::native_toolchain::ToolchainInfo,
) -> Result<ProviderSet, BuildCommandError> {
    let triple = target.unwrap_or(toolchain.host_triple.as_str());
    ProviderSet::select(crate::provider_registry::first_party(), triple, required)
        .map_err(|errors| BuildCommandError::Capability(render_capability_errors(&errors, triple)))
}

fn provider_crates_for_set(
    set: &ProviderSet,
    package_root: &Path,
    runtime_crate: &Path,
) -> Result<std::collections::BTreeMap<String, PathBuf>, BuildCommandError> {
    let repo_root = provider_repo_root(package_root, runtime_crate);
    let mut out = std::collections::BTreeMap::new();
    for provider in set.providers() {
        let name = &provider.crate_name;
        // CD-363: the provider's MANIFEST says where its crate lives, resolved against a root the
        // caller supplies. This replaced a hardcoded match over five names — the last piece of the
        // mechanism that made every native capability a compiler-source change.
        //
        // `crate_path` is constrained at parse time to be relative and free of `..`, so joining it
        // here cannot escape the root. For an external provider that root is the only containment
        // there is.
        let path = repo_root.join(&provider.crate_path);
        if !path.join("Cargo.toml").is_file() {
            return Err(BuildCommandError::Capability(format!(
                "provider crate `{name}` is not present at {}",
                path.display()
            )));
        }
        out.insert(name.clone(), path);
    }
    Ok(out)
}

struct ProviderBuildLayer {
    overlays: HashMap<PathBuf, String>,
    lowering: ProviderLowering,
    resource_nominals: std::collections::BTreeMap<String, String>,
}

fn provider_layer_for_build(
    graph: &PackageGraph,
    set: Option<&ProviderSet>,
) -> Result<ProviderBuildLayer, BuildCommandError> {
    let Some(set) = set else {
        return Ok(ProviderBuildLayer {
            overlays: HashMap::new(),
            lowering: ProviderLowering::default(),
            resource_nominals: std::collections::BTreeMap::new(),
        });
    };

    let mut tables = ProviderLayerTables::default();

    let mut package_names: Vec<_> = graph.packages.keys().cloned().collect();
    package_names.sort();
    for package_name in package_names {
        let package = &graph.packages[&package_name];
        let api = &package.provider_api;
        if api.functions.is_empty() && api.resources.is_empty() {
            continue;
        }

        // What this package OWNS: each becomes a synthesized zero-variant nominal (CD-234).
        let owned_nominal: std::collections::BTreeMap<String, String> = api
            .resources
            .iter()
            .map(|r| (r.resource.clone(), r.nominal.clone()))
            .collect();
        // HC9 — what this package may NAME but does not own. These render as qualified paths
        // (`stark_net::TcpStream`) and synthesize nothing: the nominal exists in the owning
        // package, and a second one would be a second type the program could not pass a handle
        // between.
        let foreign_nominal: std::collections::BTreeMap<String, String> = api
            .foreign_resources
            .iter()
            .map(|f| (f.resource.clone(), f.qualified_nominal()))
            .collect();
        // Derivation sees both: a signature naming either must resolve.
        let mut resource_nominal = owned_nominal.clone();
        resource_nominal.extend(foreign_nominal.iter().map(|(k, v)| (k.clone(), v.clone())));
        let errors: std::collections::BTreeMap<String, String> =
            api.errors.iter().cloned().collect();
        let mut raw_bindings = Vec::new();
        let mut vocabularies = std::collections::BTreeMap::new();

        for binding in &api.functions {
            let call = set
                .resolve(&binding.capability, &binding.symbol)
                .map_err(|error| BuildCommandError::Capability(format!("{error:?}")))?;
            raw_bindings.push((
                binding.item_path.clone(),
                binding.capability.clone(),
                call.function.clone(),
            ));
            vocabularies.insert(binding.capability.clone(), call.status_binding.clone());
        }

        let signatures =
            crate::provider_derive::derive_all(&raw_bindings, &resource_nominal, &errors).map_err(
                |errors| {
                    BuildCommandError::Capability(format!(
                        "provider_api for package `{package_name}` cannot be derived: {errors:?}"
                    ))
                },
            )?;

        // Synthesis gets the two sets SEPARATELY: `owned_nominal` is what it generates, and
        // `foreign_nominal` is what it merely accepts as already existing. Passing the union would
        // generate a duplicate nominal for every foreign resource, which is the bug this split
        // exists to prevent.
        let layer = crate::provider_synth::synthesize_with_resources(
            &signatures,
            &vocabularies,
            &owned_nominal,
            &foreign_nominal,
        )
        .map_err(|error| {
            BuildCommandError::Capability(format!(
                "provider_api for package `{package_name}` cannot be synthesized: {error}"
            ))
        })?;
        merge_layer(
            &package_name,
            package.entry.clone(),
            &mut tables,
            &signatures,
            layer,
        )?;
    }

    let mut lowering = ProviderLowering::build_with_errors(
        &tables.bindings,
        &tables.error_variants,
        &tables.error_ty_by_item,
        |capability, symbol| {
            set.resolve(capability, symbol)
                .map_err(|error| format!("{error:?}"))
        },
    )
    .map_err(BuildCommandError::Capability)?;

    // A11 §5: every bound resource needs its close selected HERE, where the provider set is, and
    // recorded against the resource name. Lowering completes it once the nominal has an item id.
    // A resource with no `is_close_for` function is refused rather than left closeless: §5
    // obligation 5 -- a resource reaching emission without a close is a leak the ABI cannot detect,
    // because the provider never learns the handle was abandoned.
    lowering.resource_nominal_names = tables.resource_nominals.clone();
    lowering
        .select_closes(|resource| {
            set.close_for(resource)
                .map_err(|error| format!("resource `{resource}` has no usable close: {error:?}"))
        })
        .map_err(BuildCommandError::Capability)?;

    Ok(ProviderBuildLayer {
        overlays: tables.overlays,
        lowering,
        resource_nominals: tables.resource_nominals,
    })
}

/// The four tables a synthesized layer contributes to, bundled.
///
/// Threaded as one value rather than four `&mut` parameters: they are always passed together, always
/// to the same place, and separating them only made the arity grow with each addition.
#[derive(Default)]
struct ProviderLayerTables {
    overlays: HashMap<PathBuf, String>,
    bindings: std::collections::BTreeMap<String, (String, String)>,
    error_variants: std::collections::BTreeMap<String, std::collections::BTreeMap<u32, u32>>,
    error_ty_by_item: std::collections::BTreeMap<String, String>,
    resource_nominals: std::collections::BTreeMap<String, String>,
}

fn merge_layer(
    package_name: &str,
    entry: PathBuf,
    tables: &mut ProviderLayerTables,
    signatures: &[DerivedSignature],
    layer: SynthesizedLayer,
) -> Result<(), BuildCommandError> {
    let ProviderLayerTables {
        overlays,
        bindings,
        error_variants,
        error_ty_by_item,
        resource_nominals,
    } = tables;
    let original = std::fs::read_to_string(&entry).map_err(|error| BuildCommandError::Io {
        action: "reading package entry for provider synthesis".into(),
        path: Some(entry.clone()),
        detail: error.to_string(),
    })?;
    overlays.insert(entry, format!("{original}\n{}", layer.source));

    for (item, binding) in layer.bindings {
        if let Some(previous) = bindings.insert(item.clone(), binding) {
            return Err(BuildCommandError::Capability(format!(
                "provider_api item `{item}` is bound more than once in the package graph; previous \
                 binding was capability `{}` symbol `{}`",
                previous.0, previous.1
            )));
        }
    }
    for (ty, variants) in layer.error_variants {
        if let Some(previous) = error_variants.insert(ty.clone(), variants.clone()) {
            if previous != variants {
                return Err(BuildCommandError::Capability(format!(
                    "provider_api raw error type `{ty}` is synthesized with conflicting status \
                     mappings in the package graph"
                )));
            }
        }
    }
    for sig in signatures {
        error_ty_by_item.insert(sig.item_path.clone(), sig.error.clone());
    }
    for (resource, nominal) in layer.resource_nominals {
        if let Some(previous) = resource_nominals.insert(resource.clone(), nominal.clone()) {
            if previous != nominal {
                return Err(BuildCommandError::Capability(format!(
                    "provider resource `{resource}` is bound to conflicting package nominals \
                     `{previous}` and `{nominal}` in the package graph"
                )));
            }
        }
    }

    let _ = package_name;
    Ok(())
}

fn resolve_resource_items(
    hir: &crate::hir::Hir,
    file: &crate::source::SourceFile,
    resource_nominals: &std::collections::BTreeMap<String, String>,
) -> Result<std::collections::BTreeMap<String, crate::hir::ItemId>, BuildCommandError> {
    let mut out = std::collections::BTreeMap::new();
    for (resource, nominal) in resource_nominals {
        let item = hir
            .items
            .iter()
            .enumerate()
            .find_map(|(idx, item)| match &item.kind {
                crate::hir::ItemKind::Enum { name, .. }
                | crate::hir::ItemKind::Struct { name, .. }
                    if span_text(
                        hir.item_file(crate::hir::ItemId(idx as u32))
                            .map(|f| f.as_ref())
                            .unwrap_or(file),
                        *name,
                    ) == nominal =>
                {
                    Some(crate::hir::ItemId(idx as u32))
                }
                _ => None,
            })
            .ok_or_else(|| {
                BuildCommandError::Lowering(format!(
                    "provider resource `{resource}` is bound to nominal `{nominal}`, but that \
                     nominal was not found after provider API synthesis"
                ))
            })?;
        out.insert(resource.clone(), item);
    }
    Ok(out)
}

fn span_text(file: &crate::source::SourceFile, span: crate::source::Span) -> &str {
    file.src
        .get(span.lo as usize..span.hi as usize)
        .unwrap_or("")
}

fn select_provider_closes(
    lowering: &mut ProviderLowering,
    set: Option<&ProviderSet>,
) -> Result<(), BuildCommandError> {
    let Some(set) = set else {
        return Ok(());
    };
    lowering
        .select_closes(|resource| {
            set.providers()
                .iter()
                .find_map(|provider| {
                    provider
                        .metadata
                        .functions
                        .iter()
                        .find(|function| function.is_close_for.as_deref() == Some(resource))
                        .cloned()
                        .map(|function| crate::mir::ValidatedProviderCall {
                            // CD-360: closes never consume a foreign resource — a provider may not
                            // declare a close for a resource it does not own.
                            foreign_resources: Vec::new(),
                            provider: provider.metadata.identity.clone(),
                            capability: function.capability.clone(),
                            function,
                            target_triple: set.target().to_string(),
                            status_binding: provider.status_binding.clone(),
                            provider_crate: provider.crate_name.clone(),
                            provider_resource_types: provider.metadata.resource_types.clone(),
                            provider_target_triples: provider.metadata.target_triples.clone(),
                        })
                })
                .ok_or_else(|| format!("no close provider function declared for `{resource}`"))
        })
        .map(|_| ())
        .map_err(BuildCommandError::Capability)
}

/// Where first-party provider crates live relative to a package being built.
///
/// Two locations, tried in this order:
///
/// 1. **A checkout containing them**, found by walking up from the package. This is how in-repo
///    development works and is unchanged.
/// 2. **The installed toolchain's own provider root**, `<exe>/../lib/stark/providers`. This is what
///    makes provider-backed capabilities usable by a package that lives anywhere on the machine;
///    without it, an installed `stark` could compile pure Core programs and nothing that reads a
///    clock, a file, the environment or a socket.
///
/// **Still not implicit discovery, which is the constraint that shaped this.** Packet 5 forbids the
/// *environment* deciding which code gets linked, and this deliberately remains an environment-free
/// search: no variable is consulted, and the second location is a fixed path inside the toolchain
/// that the compiler binary is part of — the same mechanism, and the same reasoning, as
/// `native_toolchain::discover_runtime`'s `<exe>/../lib/stark/stark-runtime`. A user cannot point
/// it somewhere else without replacing the installation.
///
/// The installed root mirrors the repository's shape — `<root>/stark-time/native`, reachable from
/// a `<root>/../starkc/stark-provider-abi` for the `../../../` dependency each provider crate
/// writes — so [`crate::provider_registry::crate_location`] needs no knowledge of which of the two
/// it got. In a checkout that root is `<repo>/packages`; installed it is
/// `<prefix>/lib/stark/packages`. The *relative* depth from a provider crate to `starkc/` is the
/// invariant, not either absolute path.
fn provider_repo_root(package_root: &Path, runtime_crate: &Path) -> PathBuf {
    // **The runtime decides.** Providers and the runtime both depend on `stark-provider-abi`, and
    // Cargo will not write a lockfile naming one package at two different paths. Taking the
    // runtime from an installed prefix and the providers from a checkout produces exactly that:
    //
    //     error: package collision in the lockfile: packages stark-provider-abi v0.1.0
    //     (~/.local/lib/stark/...) and stark-provider-abi v0.1.0 (…/starkc/…) are different
    //
    // So the tree that supplied the runtime supplies the providers. Anything else is a build that
    // fails late, in Cargo, with a message about neither.
    if let Some(root) = provider_root_beside_runtime(runtime_crate) {
        return root;
    }
    let mut current = Some(package_root);
    while let Some(dir) = current {
        if has_provider_layout(dir) {
            return dir.to_path_buf();
        }
        current = dir.parent();
    }
    if let Some(installed) = installed_provider_root() {
        return installed;
    }
    package_root.to_path_buf()
}

/// The provider root belonging to the same installation as `runtime_crate`.
///
/// **The canonical installed layout mirrors the repository**: the runtime sits at
/// `<prefix>/lib/stark/starkc/stark-runtime`, and the providers sit wherever the repository puts
/// them relative to `starkc/`. That correspondence is what lets one `stark-provider-abi` satisfy
/// both the runtime's `../` dependency and each provider's `../../../starkc/` one.
///
/// Three candidates, newest first:
///
/// 1. `<prefix>/lib/stark/packages` — the current shape. The packages moved under `packages/` in
///    the repository, so each provider's ABI dependency gained a level and the installed tree
///    gained the same one. These must move together: a provider crate written for one depth and
///    installed at the other resolves `stark-provider-abi` to a directory that does not exist.
/// 2. `<prefix>/lib/stark` — providers beside `starkc/`, the flat mirror layout. Kept because an
///    installation made before the move carries provider crates that still say `../../starkc/`,
///    and at that depth they are still right.
/// 3. `<prefix>/lib/stark/providers` — older still, from before the mirror layout.
///
/// A runtime resolved out of a checkout matches none of them and returns `None`, leaving the
/// checkout walk to decide — which keeps in-repo development on repo providers.
fn provider_root_beside_runtime(runtime_crate: &Path) -> Option<PathBuf> {
    let parent = runtime_crate.parent()?;
    [
        parent.parent().map(|root| root.join("packages")),
        parent.parent().map(Path::to_path_buf),
        Some(parent.join("providers")),
    ]
    .into_iter()
    .flatten()
    .find(|candidate| has_provider_layout(candidate))
}

/// Whether `dir` is a root holding first-party provider crates in the layout
/// `crate_location` expects.
fn has_provider_layout(dir: &Path) -> bool {
    dir.join("stark-time")
        .join("native")
        .join("Cargo.toml")
        .is_file()
}

/// `<exe>/../lib/stark/{packages,providers}`, when the running compiler has one.
///
/// The last resort, reached only when neither the runtime nor the checkout walk settled it. Same
/// ordering rule as [`provider_root_beside_runtime`]: the current shape first, the older one kept
/// so an installation made before the move still resolves.
fn installed_provider_root() -> Option<PathBuf> {
    let exe = std::env::current_exe().ok()?;
    let bin = exe.parent()?;
    ["../lib/stark/packages", "../lib/stark/providers"]
        .into_iter()
        .map(|relative| bin.join(relative))
        .find(|root| has_provider_layout(root))
        .map(|root| root.canonicalize().unwrap_or(root))
}

/// Renders selection failures with the remediation Packet 5's diagnostic requirement names.
fn render_capability_errors(
    errors: &[crate::provider_resolve::ResolveError],
    triple: &str,
) -> String {
    use crate::provider_resolve::ResolveError as E;
    let mut out = Vec::new();
    for error in errors {
        out.push(match error {
            E::CapabilityUnavailable { capability, .. } => format!(
                "no provider supplies capability `{capability}` for target `{triple}`\n  \
                 STARK knows these capabilities: {}",
                crate::provider_registry::known_capabilities().join(", ")
            ),
            E::CapabilityAmbiguous {
                capability,
                providers,
                ..
            } => format!(
                "capability `{capability}` is supplied by more than one provider for target \
                 `{triple}`:\n{}\n  remove one provider, or narrow its declared targets",
                providers
                    .iter()
                    .map(|(name, origin)| format!("    {name} ({origin})"))
                    .collect::<Vec<_>>()
                    .join("\n")
            ),
            other => format!("{other:?}"),
        });
    }
    out.join("\n")
}

#[derive(Clone, Debug)]
pub enum BuildCommandError {
    Package(String),
    /// WP-C7.1: a target this compiler does not name. Distinct from a target it names but whose
    /// toolchain is missing locally, which `target::preflight` reports separately.
    Target(String),
    Analysis {
        rendered: String,
        package_name: String,
    },
    Lowering(String),
    MirVerification(String),
    Toolchain(ToolchainError),
    UnsupportedNative(String),
    /// WP-C6.4a: target preflight refused the build, before any crate was generated. Kept
    /// distinct from `UnsupportedNative` (an unlowered construct) and from `BackendBuild` (a
    /// rustc/linker failure) because §8.4 requires an unsupported target to be rejected on its
    /// own terms rather than surfacing as a downstream tool failure.
    TargetRejected(crate::target::TargetError),
    /// WP-C7.8 (CD-212, Packet 5): a declared capability requirement could not be satisfied --
    /// no provider supplies it for this target, two do, or a selected provider's metadata is
    /// invalid.
    ///
    /// Its own variant because Packet 5 makes this a **build failure on its own terms**: a
    /// capability with no unique provider must not be reported as a missing feature, an
    /// unsupported target, or a downstream linker error.
    Capability(String),
    BackendBuild(Box<NativeBackendBuildError>),
    ArtifactMissing(PathBuf),
    ArtifactInstall {
        from: PathBuf,
        to: PathBuf,
        detail: String,
    },
    Io {
        action: String,
        path: Option<PathBuf>,
        detail: String,
    },
}

#[derive(Clone, Debug)]
pub struct NativeBackendBuildError {
    pub failure: crate::backend::generated_rust::BackendBuildFailure,
    pub toolchain: ToolchainInfo,
}

pub fn build_current_package(
    current_dir: &Path,
    options: &BuildCommandOptions,
) -> Result<BuildCommandResult, BuildCommandError> {
    // WP-C7.1 (§3.3): the target is validated BEFORE any expensive work — before the package graph
    // is loaded, before analysis, before a crate is emitted. A bad triple must cost nothing.
    if let Some(requested) = &options.target {
        if crate::target::classify(requested).is_none() {
            return Err(BuildCommandError::Target(format!(
                "unsupported target `{requested}`\n  STARK names these targets: {}",
                crate::target::known_targets()
                    .iter()
                    .map(|t| format!("{} ({})", t.triple, t.tier))
                    .collect::<Vec<_>>()
                    .join(", ")
            )));
        }
    }
    let manifest = find_package_root(current_dir).map_err(BuildCommandError::Package)?;
    let package_root = manifest
        .parent()
        .ok_or_else(|| {
            BuildCommandError::Package("package manifest has no parent directory".into())
        })?
        .to_path_buf();
    let graph = PackageGraph::load_from_root_with_modes(&manifest, options.locked, options.offline)
        .map_err(BuildCommandError::Package)?;
    let package_name = graph.root_package_name.clone();
    // WP-C7.8 step 3 (CD-212): read the capability requirements BEFORE the graph is consumed by
    // analysis. They come from the package manifests and nowhere else -- Packet 5 forbids implicit
    // discovery, so a program declaring none links no provider and builds byte-identically to a
    // pre-C7.8 one.
    let required = required_capabilities(&graph);
    validate_binary_name(&package_name).map_err(BuildCommandError::Package)?;

    // Provider API synthesis must happen before the ordinary front end runs: generated functions
    // are intentionally just source-level items, and lowering receives the side table that says
    // which of those items are provider calls.
    //
    // **The toolchain is probed only if a capability actually needs it**, and this ordering is
    // load-bearing rather than an optimisation. `source_errors_precede_toolchain_probes_...` pins
    // it: a program with a syntax error must report the syntax error, not "rustc not found". Probing
    // unconditionally here -- which is what synthesis needing a target triple quietly introduced --
    // makes every source error on a machine without a Rust toolchain come back as a toolchain error.
    //
    // A package declaring no capability needs no provider set, so it needs no triple, so it needs no
    // probe until the actual build below. That is the pre-C7.8 ordering restored for exactly the
    // programs that had it.
    let mut probed: Option<ToolchainInfo> = None;
    let provider_set = if required.is_empty() {
        None
    } else {
        let toolchain = native_toolchain::discover(std::env::current_exe().ok().as_deref())
            .map_err(BuildCommandError::Toolchain)?;
        let set = select_provider_set(&required, options.target.as_deref(), &toolchain)?;
        probed = Some(toolchain);
        Some(set)
    };
    let mut provider_layer = provider_layer_for_build(&graph, provider_set.as_ref())?;

    let analysis = if provider_layer.overlays.is_empty() {
        analyze_project(ProjectInput::package(graph), LanguageOptions::CORE)
    } else {
        analyze_project(
            ProjectInput::package_with_overlays(graph, provider_layer.overlays),
            LanguageOptions::CORE,
        )
    };
    if analysis.has_errors() {
        return Err(BuildCommandError::Analysis {
            rendered: analysis
                .diagnostic_batch(&HashMap::new())
                .render(&analysis.source_map),
            package_name,
        });
    }
    // The build itself needs the toolchain whether or not a capability did. Reuse the probe if
    // provider selection already made one -- probing twice would be wasteful and could, on a machine
    // whose PATH changes mid-build, disagree with itself.
    let toolchain = match probed {
        Some(toolchain) => toolchain,
        None => native_toolchain::discover(std::env::current_exe().ok().as_deref())
            .map_err(BuildCommandError::Toolchain)?,
    };

    let hir = analysis.hir.as_ref().ok_or_else(|| {
        BuildCommandError::Lowering("successful analysis did not produce HIR".into())
    })?;
    let tables = analysis.type_tables.as_ref().ok_or_else(|| {
        BuildCommandError::Lowering("successful analysis did not produce type tables".into())
    })?;
    provider_layer.lowering.resource_items = resolve_resource_items(
        hir,
        analysis.root_file.as_ref(),
        &provider_layer.resource_nominals,
    )?;
    select_provider_closes(&mut provider_layer.lowering, provider_set.as_ref())?;
    let mut mir = lower_program_with_providers(
        hir,
        tables,
        analysis
            .ast
            .sources
            .id_for_name(&analysis.root_file.name)
            .and_then(|id| analysis.ast.sources.get(id))
            .expect("the analysis registered its own root")
            .clone(),
        &provider_layer.lowering,
    )
    .map_err(|error| BuildCommandError::Lowering(error.what))?;
    let mir_bodies = mir.bodies.len();
    // WP-C7.4. Optimise BEFORE verifying, deliberately: the verifier then checks the program that
    // is actually compiled and executed, rather than a form the backend never sees. An optimiser
    // that produced ill-formed MIR would otherwise be caught only by whatever the backend happened
    // to notice downstream.
    let mir_opt = (!options.no_mir_opt).then(|| crate::mir::opt::optimise(&mut mir));
    let verified = verify_program(&mir).map_err(|errors| {
        {
            if std::env::var("STARK_DUMP_MIR_ON_VERIFY_FAIL").is_ok() {
                eprintln!("{}", mir.dump());
            }
        }
        BuildCommandError::MirVerification(
            errors
                .into_iter()
                .map(|error| {
                    format!(
                        "{} {} bb{}: {}",
                        error.code, error.symbol, error.block, error.message
                    )
                })
                .collect::<Vec<_>>()
                .join("\n"),
        )
    })?;

    // WP-C7.1 (§3.4): the layout is parameterised by TARGET and PROFILE, so a debug build, a
    // release build and a cross-target build of one package cannot overwrite each other. The old
    // layout was `target/stark/debug/` with both components fixed.
    let profile = options.profile();
    let target_root = package_root.join("target/stark");
    let final_dir = match &options.target {
        Some(triple) => target_root.join(triple).join(profile.as_str()),
        None => target_root.join(profile.as_str()),
    };
    // WP-C7.3: the cache root is where the backend puts content-addressed crate directories —
    // `target/stark/<profile>/`. Per-profile by construction, so a debug entry can never be reused
    // for a release build; TARGET separation comes from the build key, which carries the triple.
    let cache_root = target_root.join(profile.as_str());
    let provider_crates = if let Some(provider_set) = provider_set.as_ref() {
        provider_crates_for_set(provider_set, &package_root, &toolchain.runtime_crate)?
    } else {
        std::collections::BTreeMap::new()
    };

    let artifact = emit_native_debug_with_toolchain(
        &verified,
        &NativeBuildOptions {
            target_dir: target_root.clone(),
            profile,
            target_triple: options.target.clone(),
            ..NativeBuildOptions::default()
        },
        &NativeToolchainOptions {
            rustc: toolchain.rustc.clone(),
            cargo: toolchain.cargo.clone(),
            runtime_crate: toolchain.runtime_crate.clone(),
            provider_crates,
        },
    )
    .map_err(|error| map_backend_error(error, &toolchain))?;
    if !artifact.binary_path.is_file() {
        return Err(BuildCommandError::ArtifactMissing(artifact.binary_path));
    }
    let final_path = final_dir.join(binary_filename(&package_name, &artifact.binary_path));
    install_artifact(&artifact.binary_path, &final_path)?;

    let generated_rust_path = artifact.build_dir.join("src/main.rs");
    if options.emit_rust && !generated_rust_path.is_file() {
        return Err(BuildCommandError::ArtifactMissing(generated_rust_path));
    }
    // WP-C7.3. The generated crate is RETAINED by default — it is content-addressed, so keeping it
    // makes the next build of the same source a cache hit rather than a rebuild. Before this, it was
    // deleted immediately and every rebuild paid the full cost including recompiling the runtime.
    //
    // `--keep-generated`/`--emit-rust` still mean something distinct from "cached": they PIN the
    // entry, so eviction never removes something the user explicitly asked to keep. An ordinary
    // cached entry is evictable; a requested one is not.
    let pinned = options.keep_generated || options.emit_rust;
    let generated_dir = pinned.then(|| artifact.build_dir.clone());
    let generated_rust = options.emit_rust.then_some(generated_rust_path);
    let backend_artifact = pinned.then(|| artifact.binary_path.clone());
    let mut cache_eviction = None;
    if options.no_build_cache {
        std::fs::remove_dir_all(&artifact.build_dir).map_err(|error| BuildCommandError::Io {
            action: "removing generated crate".into(),
            path: Some(artifact.build_dir.clone()),
            detail: error.to_string(),
        })?;
    } else {
        crate::build_cache::touch(&artifact.build_dir, pinned);
        // Sweep AFTER a successful build, never before: an eviction that ran first could remove the
        // very entry this build was about to reuse.
        cache_eviction = Some(crate::build_cache::evict(
            &cache_root,
            Some(&artifact.build_dir),
            crate::build_cache::DEFAULT_MAX_BYTES,
            crate::build_cache::DEFAULT_MAX_AGE,
        ));
    }
    Ok(BuildCommandResult {
        package_name,
        profile,
        package_root,
        artifact_path: final_path,
        generated_dir,
        generated_rust,
        backend_artifact,
        mir_bodies,
        toolchain,
        cache_eviction,
        mir_opt,
    })
}

fn map_backend_error(error: BackendDiagnostic, toolchain: &ToolchainInfo) -> BuildCommandError {
    match error {
        BackendDiagnostic::Unsupported(message) => BuildCommandError::UnsupportedNative(message),
        BackendDiagnostic::TargetRejected(error) => BuildCommandError::TargetRejected(error),
        BackendDiagnostic::BuildFailed(failure) => {
            BuildCommandError::BackendBuild(Box::new(NativeBackendBuildError {
                failure: *failure,
                toolchain: toolchain.clone(),
            }))
        }
        BackendDiagnostic::Io(detail) => BuildCommandError::Io {
            action: "running the native backend".to_string(),
            path: None,
            detail,
        },
    }
}

pub fn validate_binary_name(name: &str) -> Result<(), String> {
    if name.is_empty()
        || matches!(name, "." | "..")
        || name.contains('/')
        || name.contains('\\')
        || !name
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_' | '.'))
    {
        return Err(format!(
            "package name '{name}' is not a safe executable name"
        ));
    }
    Ok(())
}

fn binary_filename(package: &str, backend: &Path) -> String {
    backend
        .extension()
        .and_then(|value| value.to_str())
        .map(|suffix| format!("{package}.{suffix}"))
        .unwrap_or_else(|| package.to_string())
}

static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

fn install_artifact(source: &Path, destination: &Path) -> Result<(), BuildCommandError> {
    std::fs::create_dir_all(destination.parent().expect("final artifact has parent")).map_err(
        |error| BuildCommandError::ArtifactInstall {
            from: source.to_path_buf(),
            to: destination.to_path_buf(),
            detail: error.to_string(),
        },
    )?;
    let temp = destination.with_file_name(format!(
        "{}.tmp-{}-{}",
        destination.file_name().unwrap().to_string_lossy(),
        std::process::id(),
        TEMP_COUNTER.fetch_add(1, Ordering::Relaxed)
    ));
    if temp.exists() {
        std::fs::remove_file(&temp).map_err(|error| BuildCommandError::ArtifactInstall {
            from: source.to_path_buf(),
            to: temp.clone(),
            detail: error.to_string(),
        })?;
    }
    std::fs::copy(source, &temp).map_err(|error| BuildCommandError::ArtifactInstall {
        from: source.to_path_buf(),
        to: destination.to_path_buf(),
        detail: error.to_string(),
    })?;
    if !temp.is_file() {
        return Err(BuildCommandError::ArtifactMissing(temp));
    }
    replace_artifact(&temp, destination, source)?;
    if !destination.is_file() {
        return Err(BuildCommandError::ArtifactMissing(
            destination.to_path_buf(),
        ));
    }
    Ok(())
}

#[cfg(not(windows))]
fn replace_artifact(
    temp: &Path,
    destination: &Path,
    source: &Path,
) -> Result<(), BuildCommandError> {
    std::fs::rename(temp, destination).map_err(|error| BuildCommandError::ArtifactInstall {
        from: source.to_path_buf(),
        to: destination.to_path_buf(),
        detail: error.to_string(),
    })
}

#[cfg(windows)]
fn replace_artifact(
    temp: &Path,
    destination: &Path,
    source: &Path,
) -> Result<(), BuildCommandError> {
    // Windows rename does not replace an existing destination. Preserve the old executable
    // until the new one is ready, and roll back if the second half of the swap fails.
    let backup = destination.with_file_name(format!(
        "{}.old-{}-{}",
        destination.file_name().unwrap().to_string_lossy(),
        std::process::id(),
        TEMP_COUNTER.fetch_add(1, Ordering::Relaxed)
    ));
    let had_destination = destination.is_file();
    if had_destination {
        std::fs::rename(destination, &backup).map_err(|error| {
            BuildCommandError::ArtifactInstall {
                from: source.to_path_buf(),
                to: destination.to_path_buf(),
                detail: format!("preserving previous artifact: {error}"),
            }
        })?;
    }
    match std::fs::rename(temp, destination) {
        Ok(()) => {
            if had_destination {
                std::fs::remove_file(&backup).map_err(|error| {
                    BuildCommandError::ArtifactInstall {
                        from: source.to_path_buf(),
                        to: destination.to_path_buf(),
                        detail: format!("removing previous artifact backup: {error}"),
                    }
                })?;
            }
            Ok(())
        }
        Err(error) => {
            if had_destination {
                let _ = std::fs::rename(&backup, destination);
            }
            Err(BuildCommandError::ArtifactInstall {
                from: source.to_path_buf(),
                to: destination.to_path_buf(),
                detail: error.to_string(),
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_safe_binary_names() {
        for valid in ["app", "my-app", "my_app", "app.v1"] {
            assert!(validate_binary_name(valid).is_ok());
        }
        for invalid in ["", ".", "..", "../app", "a/b", "a\\b", "bad name"] {
            assert!(validate_binary_name(invalid).is_err());
        }
    }

    #[test]
    fn preserves_backend_executable_suffix() {
        assert_eq!(binary_filename("demo", Path::new("stark_program")), "demo");
        assert_eq!(
            binary_filename("demo", Path::new("stark_program.exe")),
            "demo.exe"
        );
    }

    /// Builds a directory holding first-party providers in the layout `crate_location` expects.
    fn provider_layout_at(root: &Path) {
        std::fs::create_dir_all(root.join("stark-time/native")).unwrap();
        std::fs::write(root.join("stark-time/native/Cargo.toml"), "").unwrap();
    }

    /// **The flat mirror layout**: the runtime under `starkc/`, providers beside it.
    ///
    /// This was canonical before the packages moved under `packages/`, and it stays supported
    /// because an installation made then carries provider crates that still say `../../starkc/` —
    /// which is correct at *that* depth. Upgrading the compiler must not strand them. The newer
    /// arrangement is covered by `providers_resolve_under_a_packages_directory` above; between the
    /// two, a layout regression fails here in milliseconds rather than in a Cargo lockfile error at
    /// the end of a native build.
    #[test]
    fn providers_follow_a_mirrored_installed_runtime() {
        let prefix = std::env::temp_dir().join(format!("stark_prov_mirror_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&prefix);
        let lib = prefix.join("lib/stark");
        // Repository-shaped: `starkc/stark-runtime` + `starkc/stark-provider-abi`, providers beside.
        std::fs::create_dir_all(lib.join("starkc/stark-runtime")).unwrap();
        std::fs::create_dir_all(lib.join("starkc/stark-provider-abi")).unwrap();
        provider_layout_at(&lib);

        let package = prefix.join("elsewhere/app");
        std::fs::create_dir_all(&package).unwrap();

        let chosen = provider_repo_root(&package, &lib.join("starkc/stark-runtime"));
        assert_eq!(
            chosen.canonicalize().unwrap(),
            lib.canonicalize().unwrap(),
            "a mirrored installation must resolve providers beside `starkc/`, so that the runtime's \
             `../stark-provider-abi` and a provider's `../../starkc/stark-provider-abi` name ONE crate"
        );

        // The property that matters, stated as the paths Cargo will see.
        let abi_from_runtime = lib.join("starkc/stark-runtime/../stark-provider-abi");
        let abi_from_provider = chosen.join("stark-time/native/../../starkc/stark-provider-abi");
        assert_eq!(
            abi_from_runtime.canonicalize().unwrap(),
            abi_from_provider.canonicalize().unwrap(),
            "both relative paths must land on the same ABI crate; two copies is the lockfile \
             collision this layout exists to prevent, and a symlink does not help because Cargo \
             does not canonicalise symlinked path dependencies"
        );
        let _ = std::fs::remove_dir_all(&prefix);
    }

    /// **The current installed layout**: providers under `lib/stark/packages`, matching the
    /// repository's `packages/` directory.
    ///
    /// The depth is the whole point. A provider crate writes `../../../starkc/stark-provider-abi`,
    /// which is correct at `<root>/packages/stark-time/native` and wrong one level up — so the
    /// repository move and the installed layout have to change together. This test fails if either
    /// one moves alone, which is the failure that would otherwise surface as a Cargo lockfile
    /// collision at the end of a native build.
    #[test]
    fn providers_resolve_under_a_packages_directory() {
        let prefix = std::env::temp_dir().join(format!("stark_prov_pkgs_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&prefix);
        let lib = prefix.join("lib/stark");
        std::fs::create_dir_all(lib.join("starkc/stark-runtime")).unwrap();
        std::fs::create_dir_all(lib.join("starkc/stark-provider-abi")).unwrap();
        provider_layout_at(&lib.join("packages"));

        let package = prefix.join("elsewhere/app");
        std::fs::create_dir_all(&package).unwrap();

        let chosen = provider_repo_root(&package, &lib.join("starkc/stark-runtime"));
        assert_eq!(
            chosen.canonicalize().unwrap(),
            lib.join("packages").canonicalize().unwrap(),
            "an installation carrying `packages/` must resolve providers there, in preference to \
             the flat layout kept for older installs"
        );

        let abi_from_runtime = lib.join("starkc/stark-runtime/../stark-provider-abi");
        let abi_from_provider = chosen.join("stark-time/native/../../../starkc/stark-provider-abi");
        assert_eq!(
            abi_from_runtime.canonicalize().unwrap(),
            abi_from_provider.canonicalize().unwrap(),
            "at this depth a provider's `../../../starkc/stark-provider-abi` and the runtime's \
             `../stark-provider-abi` must name ONE crate"
        );
        let _ = std::fs::remove_dir_all(&prefix);
    }

    /// The legacy flat installation stays supported: runtime at `lib/stark/stark-runtime`,
    /// providers at `lib/stark/providers`. An existing install must not break on upgrade.
    #[test]
    fn legacy_flat_installation_remains_supported() {
        let prefix = std::env::temp_dir().join(format!("stark_prov_prefix_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&prefix);
        let lib = prefix.join("lib/stark");
        std::fs::create_dir_all(lib.join("stark-runtime")).unwrap();
        provider_layout_at(&lib.join("providers"));

        // A package that IS inside a checkout with its own providers...
        let checkout = prefix.join("checkout");
        provider_layout_at(&checkout);
        let package = checkout.join("app");
        std::fs::create_dir_all(&package).unwrap();

        // ...still gets the installed providers, because that is where its runtime came from.
        let chosen = provider_repo_root(&package, &lib.join("stark-runtime"));
        assert_eq!(
            chosen.canonicalize().unwrap(),
            lib.join("providers").canonicalize().unwrap(),
            "mixing an installed runtime with checkout providers is the lockfile collision this \
             rule exists to prevent"
        );
        let _ = std::fs::remove_dir_all(&prefix);
    }

    /// A package inside a checkout resolves providers from that checkout — the in-repo path,
    /// which must keep working exactly as before.
    #[test]
    fn a_package_inside_a_checkout_uses_the_checkout() {
        let root = std::env::temp_dir().join(format!("stark_prov_repo_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        provider_layout_at(&root);
        let package = root.join("examples/demo");
        std::fs::create_dir_all(&package).unwrap();

        assert_eq!(
            provider_repo_root(&package, Path::new("/nonexistent/stark-runtime"))
                .canonicalize()
                .unwrap(),
            root.canonicalize().unwrap(),
            "the enclosing checkout must win: an in-repo build links the checkout's own providers"
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    /// A package with no checkout above it falls back to the package root only when the running
    /// compiler has no installed provider root either.
    ///
    /// The installed half cannot be exercised here without relocating the test binary, so what is
    /// pinned is the SHAPE both halves share: `has_provider_layout` is the single predicate that
    /// decides whether a candidate root is usable, and `crate_location` reads the same layout from
    /// whichever root it is given.
    #[test]
    fn a_root_is_usable_only_when_it_has_the_provider_layout() {
        let root = std::env::temp_dir().join(format!("stark_prov_empty_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        assert!(
            !has_provider_layout(&root),
            "an empty directory is not a provider root"
        );

        provider_layout_at(&root);
        assert!(
            has_provider_layout(&root),
            "a directory holding stark-time/native/Cargo.toml is one"
        );
        assert_eq!(
            crate::provider_registry::built_in_crate_location("stark-time-native", &root),
            Some(root.join("stark-time").join("native")),
            "the installed root mirrors the repository shape, so one locator serves both"
        );
        let _ = std::fs::remove_dir_all(&root);
    }
}
