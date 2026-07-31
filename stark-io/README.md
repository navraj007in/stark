# stark-io

`stark-io` is the proposed host-backed synchronous file and filesystem package for STARK.

This package currently records the v0.1 public API shape and implements only pure validation helpers.
The host-backed methods intentionally trap until source-level provider calls are available for
package APIs. The existing `stark-file/native` crate is the first native provider candidate for the
minimal file-handle slice.

The attached v0.1 spec names the required capability `file`; this checkout's first-party registry
currently exposes the same provider under `filesystem`, so the manifest uses `filesystem`.

See `../STARK-IO-v0.1-Codex-Implementation-Spec.md` for the implementation target and current
compiler boundary.
