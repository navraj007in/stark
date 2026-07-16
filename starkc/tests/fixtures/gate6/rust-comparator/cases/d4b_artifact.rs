// Defect 4b — runtime artifact swap (Gate 5 defect #5).
// The host embeds the expected artifact digest and verifies the bytes at load.
// A swapped artifact fails verification. This is a RUNTIME/load check in both
// STARK and any competent Rust host — rustc cannot catch it, but the host
// refuses to run inference. Expected: COMPILES, exits non-zero at runtime.
//
// Defect 4a (declaration/signature drift) is structural: the model types above
// are *generated from the artifact's signature*, so a declaration that
// disagrees with the artifact is regenerated; a stale hand-edit is caught by
// this same load-time digest check. There is no separate compile-time story.
include!("../lib.rs");

fn main() {
    let expected = fnv1a(b"the-pinned-resnet50-artifact-bytes");
    let swapped_artifact = b"a-different-model-version-bytes";
    match verify_artifact(swapped_artifact, expected) {
        Ok(()) => {
            println!("artifact ok");
        }
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
    }
}
