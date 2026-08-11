# stark-urlencoded Evidence

Created during the ASCII / percent / URL-encoded consolidation packet.

Baseline SHA: `6e7fa959594691f56ff00ee9b1922e66d480cb93`

The package consolidates the duplicated `stark-query` and `stark-form` pair scanner/serializer
mechanics while preserving their distinct `+` semantics.

The behavioural properties the removed packages pinned are carried here: only the first `=`
separates, empty segments are preserved as pairs, each of the four limits is enforced separately,
the limits measure decoded rather than escaped length, and serialize/parse round-trips in both
modes. See `TEST-MATRIX.md`.
