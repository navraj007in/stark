use std::path::Path;

pub const CANONICAL_STARK_EXTENSION: &str = "stark";
pub const SHORT_STARK_EXTENSION: &str = "st";

pub fn is_stark_source(path: &Path) -> bool {
    matches!(
        path.extension().and_then(|value| value.to_str()),
        Some(CANONICAL_STARK_EXTENSION | SHORT_STARK_EXTENSION)
    )
}

pub fn supported_stark_source_extensions() -> &'static [&'static str] {
    &[CANONICAL_STARK_EXTENSION, SHORT_STARK_EXTENSION]
}
