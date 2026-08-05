# stark-get

`stark-get` fetches a URL over HTTP/1.1 or HTTPS and writes the body to stdout. It is a proving
application: it exists to drive the package stack — `stark-args`, `stark-env`, `stark-http-client`,
`stark-io`, `stark-json` — the way a real program would, end to end, against real peers.

```
stark-get [options] <url>
```

| option | effect |
| --- | --- |
| `-h`, `--help` | print help and exit `0` |
| `--version` | print the version and exit `0` |
| `-H`, `--header <h>` | add a request header, `Name: Value`; repeatable |
| `--max-body <n>` | maximum response body in bytes (default `8388608`, 8 MiB) |
| `-o`, `--output <path>` | write the body to a file instead of stdout |
| `--json` | require the body to be valid JSON |
| `-i`, `--include` | print the status line and headers before the body |
| `-q`, `--quiet` | suppress diagnostics; the exit code still reports |
| `--fail-status` | exit `9` on a 4xx or 5xx status |

**stdout carries the body and nothing else**, so redirecting it yields exactly the bytes the peer
sent; every diagnostic goes to stderr. `--quiet` silences stderr only — it does not touch stdout,
and it does not touch the exit code, which is what a quiet caller reads instead.

## Exit codes

| code | meaning |
| --- | --- |
| 0 | the exchange completed |
| 2 | command-line usage error |
| 3 | invalid URL, header, or option value |
| 4 | DNS, TCP or TLS failure — no exchange took place |
| 5 | the response was not valid HTTP, or the body was not text |
| 6 | the body exceeded `--max-body` |
| 7 | `--json` was given and the body is not valid JSON |
| 8 | `--output` was given and the file could not be written |
| 9 | `--fail-status` was given and the status was 4xx or 5xx |
| 10 | an internal failure |

**A transport failure and an HTTP error status are different things.** A `404` that arrived intact
is a completed exchange and exits `0`; `--fail-status` is how a caller asks for the other reading.
Code `4` means the request never happened and is worth retrying in a way code `5` is not.

## What it will not do

- **There is no `--insecure` or `-k`.** Certificate verification cannot be disabled. A flag is
  exactly how that gets used in anger, so the flag does not exist; asking for it produces that
  explanation rather than "unknown option".
- **There is no `--connect-timeout`.** `ClientConfig` accepts a connect timeout and does not yet
  enforce it (DEV-165). The option would parse, appear to work, and do nothing — worse than absent.
- **Credentials are never printed.** No diagnostic echoes a request header's value, and
  `Authorization`, `Proxy-Authorization`, `Cookie` and `X-API-Key` are recognised case-insensitively
  as credential-bearing. A `--header` that fails validation is reported by position and reason, not
  by quoting it back.
- Credentials are not forwarded across origins: the client's redirect policy carries
  `preserve_authorization_same_origin_only`.

A default `User-Agent: stark-get/0.1.0` is sent unless `-H` supplies one; real peers (api.github.com
among them) answer `403` to a request without one.

## Limits worth knowing

**A body that is not valid UTF-8 cannot go to stdout at all.** There is no byte-level stdout in the
language — `println` takes a `Display` value, and nothing converts `Vec<UInt8>` to `String` — so the
body is decoded before it is printed, strictly: over-long encodings, surrogates and anything above
U+10FFFF are rejected rather than widened. A binary response exits `5` and says to use `--output`,
which writes the exact bytes.

**This package is one file, and that is not a style choice.** A dependency cannot be reached from a
non-entry file of a package — neither `use stark_http_core::Header;` (E0205) nor the fully-qualified
`stark_http_core::Header` (E0202) resolves outside `src/main.stark` (DEV-175). Every other package
in this repo is a single `lib.stark`, so nothing had hit this before.

Being capability-backed (`process.args`, `tcp`, `dns`, `tls`, `filesystem`), it builds with
`stark build` and cannot run under `stark run`: the interpreters have no host access.
