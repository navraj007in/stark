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
use crate::provider_derive::{DerivedSignature, DerivedTy};
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
) -> Result<std::collections::BTreeMap<String, PathBuf>, BuildCommandError> {
    let repo_root = provider_repo_root(package_root);
    let mut out = std::collections::BTreeMap::new();
    for provider in set.providers() {
        let name = &provider.crate_name;
        let path = crate::provider_registry::crate_location(name, &repo_root).ok_or_else(|| {
            BuildCommandError::Capability(format!(
                "provider `{}` needs crate `{name}`, which this build has no location for",
                provider.metadata.identity.name
            ))
        })?;
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
}

fn provider_layer_for_build(
    graph: &PackageGraph,
    set: Option<&ProviderSet>,
) -> Result<ProviderBuildLayer, BuildCommandError> {
    let Some(set) = set else {
        return Ok(ProviderBuildLayer {
            overlays: HashMap::new(),
            lowering: ProviderLowering::default(),
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

        let resource_nominal: std::collections::BTreeMap<String, String> = api
            .resources
            .iter()
            .map(|r| (r.resource.clone(), r.nominal.clone()))
            .collect();
        let errors: std::collections::BTreeMap<String, String> =
            api.errors.iter().cloned().collect();
        let mut raw_bindings = Vec::new();
        let mut vocabularies = std::collections::BTreeMap::new();

        for binding in &api.functions {
            let call = set
                .resolve(&binding.capability, &binding.symbol)
                .map_err(|error| BuildCommandError::Capability(format!("{error:?}")))?;
            if call.function.params.iter().any(is_resource_abi_param) {
                return Err(BuildCommandError::Capability(format!(
                    "provider_api for package `{package_name}` binds `{}` to a resource-bearing \
                     provider signature. Synthesis and lowering handle host resources as of \
                     CD-234/CD-235 -- the nominal is a zero-variant enum and the type is \
                     MirTy::HostResource -- but the close arena and the Drop-terminator close are \
                     not implemented, so a resource obtained here could never be released. Refused \
                     until that lands rather than built with a leak.",
                    binding.item_path
                )));
            }
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
        reject_resource_signatures(&package_name, &signatures)?;

        let layer =
            crate::provider_synth::synthesize(&signatures, &vocabularies).map_err(|error| {
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

    let lowering = ProviderLowering::build_with_errors(
        &tables.bindings,
        &tables.error_variants,
        &tables.error_ty_by_item,
        |capability, symbol| {
            set.resolve(capability, symbol)
                .map_err(|error| format!("{error:?}"))
        },
    )
    .map_err(BuildCommandError::Capability)?;

    Ok(ProviderBuildLayer {
        overlays: tables.overlays,
        lowering,
    })
}

fn reject_resource_signatures(
    package_name: &str,
    signatures: &[DerivedSignature],
) -> Result<(), BuildCommandError> {
    for sig in signatures {
        if sig.receiver.as_ref().is_some_and(is_resource_ty)
            || sig.params.iter().any(is_resource_ty)
            || sig.results.iter().any(is_resource_ty)
        {
            return Err(BuildCommandError::Capability(format!(
                "provider_api for package `{package_name}` binds `{}` to a resource-bearing \
                 provider signature. Refused because the close arena and the Drop-terminator close \
                 are not implemented yet (CD-234/CD-235 gave the nominal and the type, not the \
                 lifecycle), so a resource obtained here could never be released.",
                sig.item_path
            )));
        }
    }
    Ok(())
}

fn is_resource_ty(ty: &DerivedTy) -> bool {
    matches!(
        ty,
        DerivedTy::SharedResource { .. } | DerivedTy::OwnedResource { .. }
    )
}

fn is_resource_abi_param(param: &crate::provider_abi::AbiParam) -> bool {
    matches!(
        param,
        crate::provider_abi::AbiParam::HandleBorrowed { .. }
            | crate::provider_abi::AbiParam::HandleConsumed { .. }
            | crate::provider_abi::AbiParam::HandleOut { .. }
    )
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

    let _ = package_name;
    Ok(())
}

/// Where first-party provider crates live relative to a package being built.
///
/// Walks up from the package looking for the checkout that contains them. Deliberately a *search
/// for a known layout* rather than an environment variable: Packet 5 forbids implicit discovery,
/// and an env-var override would be exactly that — a way for the environment, rather than the
/// program's declarations, to decide which code gets linked.
fn provider_repo_root(package_root: &Path) -> PathBuf {
    let mut current = Some(package_root);
    while let Some(dir) = current {
        if dir
            .join("stark-time")
            .join("native")
            .join("Cargo.toml")
            .is_file()
        {
            return dir.to_path_buf();
        }
        current = dir.parent();
    }
    package_root.to_path_buf()
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
    let toolchain = native_toolchain::discover(std::env::current_exe().ok().as_deref())
        .map_err(BuildCommandError::Toolchain)?;
    let provider_set = if required.is_empty() {
        None
    } else {
        Some(select_provider_set(
            &required,
            options.target.as_deref(),
            &toolchain,
        )?)
    };
    let provider_layer = provider_layer_for_build(&graph, provider_set.as_ref())?;

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
    let hir = analysis.hir.as_ref().ok_or_else(|| {
        BuildCommandError::Lowering("successful analysis did not produce HIR".into())
    })?;
    let tables = analysis.type_tables.as_ref().ok_or_else(|| {
        BuildCommandError::Lowering("successful analysis did not produce type tables".into())
    })?;
    let mut mir = lower_program_with_providers(
        hir,
        tables,
        analysis.root_file.clone(),
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
        provider_crates_for_set(provider_set, &package_root)?
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
}
