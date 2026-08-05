# stark-time v0.1 blockers

## Current status

No package-local blocker remains for v0.1.

The original provider-execution blocker was discharged by the compiler/provider integration track:
generated STARK binaries can link and call `stark-time-native` through the provider ABI. WP-TIME-B
then wired that seam into the public STARK package APIs:

- `Instant::now() -> Result<Instant, TimeError>`
- `Instant::elapsed(&self) -> Result<Duration, TimeError>`
- `UnixTimestamp::now() -> Result<UnixTimestamp, TimeError>`

`stark run` still cannot execute provider-backed programs because the interpreter has no provider
layer. That is an execution-mode limitation, not a `stark-time` package blocker; the qualified path
for clock reads is native `stark build` plus the generated binary.

## Historical note

Before WP-TIME-B, this file recorded a split state: the Rust provider and compiler seam existed,
but the STARK package still exposed no public clock-reading wrapper. That split is now closed by
the provider API entries in `starkpkg.json`, the generated raw bindings, and `map_raw_error` in
`src/lib.stark`.
