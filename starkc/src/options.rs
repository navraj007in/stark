//! Language configuration for optional, non-Core extensions (Gate 4+).
//!
//! Core v1 is the default: `LanguageOptions::default()` enables no extensions,
//! so every existing entry point (`parse`, `resolve`, `typecheck::check`)
//! behaves exactly as before. Extension syntax and semantics are gated behind
//! an explicit, deterministic [`ExtensionSet`] that is threaded through parse,
//! resolve, and type analysis rather than read from global process state.
//!
//! The only extension in Gate 4 is `tensor` v0.1
//! (`STARKLANG/docs/extensions/Tensor-Model-Types.md`).

/// A single optional language extension.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum ExtensionId {
    /// The `tensor` v0.1 extension: tensor & model types, ONNX import.
    Tensor,
}

impl ExtensionId {
    /// The extension's stable identifier as written on the CLI (`--extension
    /// tensor`) and in `use tensor::*;`.
    pub fn name(self) -> &'static str {
        match self {
            ExtensionId::Tensor => "tensor",
        }
    }

    /// Parse an extension id from its CLI/`use` spelling.
    pub fn from_name(name: &str) -> Option<ExtensionId> {
        match name {
            "tensor" => Some(ExtensionId::Tensor),
            _ => None,
        }
    }
}

/// The set of enabled extensions. Defaults to Core-only (empty).
#[derive(Clone, Copy, Default, PartialEq, Eq, Debug)]
pub struct ExtensionSet {
    tensor: bool,
}

impl ExtensionSet {
    /// A Core-only set with no extensions enabled.
    pub const CORE_ONLY: ExtensionSet = ExtensionSet { tensor: false };

    /// Whether `id` is enabled.
    pub fn contains(self, id: ExtensionId) -> bool {
        match id {
            ExtensionId::Tensor => self.tensor,
        }
    }

    /// Enable `id`. Returns `false` if it was already enabled (used by the CLI
    /// to reject a duplicate `--extension` flag).
    pub fn enable(&mut self, id: ExtensionId) -> bool {
        let slot = match id {
            ExtensionId::Tensor => &mut self.tensor,
        };
        if *slot {
            return false;
        }
        *slot = true;
        true
    }
}

/// Compilation options shared across every front-end and semantic stage.
#[derive(Clone, Copy, Default, PartialEq, Eq, Debug)]
pub struct LanguageOptions {
    pub extensions: ExtensionSet,
}

impl LanguageOptions {
    /// Core v1 only — the default for every existing consumer.
    pub const CORE: LanguageOptions = LanguageOptions {
        extensions: ExtensionSet::CORE_ONLY,
    };

    /// Convenience: a set with only the `tensor` extension enabled.
    pub fn with_tensor() -> LanguageOptions {
        let mut opts = LanguageOptions::default();
        opts.extensions.enable(ExtensionId::Tensor);
        opts
    }

    /// Whether `id` is enabled.
    pub fn has(&self, id: ExtensionId) -> bool {
        self.extensions.contains(id)
    }

    /// Whether the `tensor` extension is enabled.
    pub fn tensor(&self) -> bool {
        self.has(ExtensionId::Tensor)
    }
}

/// An error parsing `--extension` values from the command line.
#[derive(Debug, PartialEq, Eq)]
pub enum ExtensionCliError {
    /// `--extension <name>` named an extension that does not exist.
    Unknown(String),
    /// `--extension <name>` was passed more than once for the same extension.
    Duplicate(&'static str),
}

impl std::fmt::Display for ExtensionCliError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ExtensionCliError::Unknown(name) => {
                write!(f, "unknown extension `{name}` (known extensions: tensor)")
            }
            ExtensionCliError::Duplicate(name) => {
                write!(f, "extension `{name}` enabled more than once")
            }
        }
    }
}

/// Build a [`LanguageOptions`] from a list of `--extension` values, rejecting
/// unknown and duplicate ids (CLI usage errors).
pub fn options_from_extension_flags(
    names: &[String],
) -> Result<LanguageOptions, ExtensionCliError> {
    let mut opts = LanguageOptions::default();
    for name in names {
        let id =
            ExtensionId::from_name(name).ok_or_else(|| ExtensionCliError::Unknown(name.clone()))?;
        if !opts.extensions.enable(id) {
            return Err(ExtensionCliError::Duplicate(id.name()));
        }
    }
    Ok(opts)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_core_only() {
        let opts = LanguageOptions::default();
        assert!(!opts.tensor());
        assert!(!opts.has(ExtensionId::Tensor));
    }

    #[test]
    fn with_tensor_enables_only_tensor() {
        let opts = LanguageOptions::with_tensor();
        assert!(opts.tensor());
    }

    #[test]
    fn flags_parse_tensor() {
        let opts = options_from_extension_flags(&["tensor".to_string()]).unwrap();
        assert!(opts.tensor());
    }

    #[test]
    fn flags_reject_unknown() {
        let err = options_from_extension_flags(&["nonsense".to_string()]).unwrap_err();
        assert_eq!(err, ExtensionCliError::Unknown("nonsense".to_string()));
    }

    #[test]
    fn flags_reject_duplicate() {
        let err = options_from_extension_flags(&["tensor".to_string(), "tensor".to_string()])
            .unwrap_err();
        assert_eq!(err, ExtensionCliError::Duplicate("tensor"));
    }

    #[test]
    fn empty_flags_are_core_only() {
        let opts = options_from_extension_flags(&[]).unwrap();
        assert_eq!(opts, LanguageOptions::CORE);
    }
}
