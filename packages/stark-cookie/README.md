# stark-cookie

Bounded parsing and deterministic formatting of HTTP cookie header values: `Cookie` on the request
side, `Set-Cookie` on the response side.

Pure STARK. Depends only on `stark-ascii`. Declares no host capability and needs no native provider,
so it runs under `stark run` as well as `stark build`.

## What it is not

A value package, not an engine. It holds no cookie jar, does no domain or path matching, consults
no Public Suffix List, schedules no expiry, and enforces no SameSite, secure-transport or
third-party policy. Those are decisions a client or server makes *with* a parsed cookie; they are
not properties of the cookie itself, and putting them here would mean a syntax package quietly
making security choices on a caller's behalf.

## Public API

```stark
pub fn default_limits() -> CookieLimits;

pub fn parse_cookie(input: &String, limits: CookieLimits) -> Result<Cookie, CookieError>;
pub fn format_cookie(cookie: &Cookie) -> String;

pub fn parse_set_cookie(input: &String, limits: CookieLimits) -> Result<SetCookie, CookieError>;
pub fn format_set_cookie(set_cookie: &SetCookie) -> String;

pub fn cookie_value(cookie: &Cookie, name: &String) -> Option<String>;
pub fn same_site_text(value: SameSite) -> String;

pub fn attribute_expires(value: String) -> CookieAttribute;
pub fn attribute_max_age(seconds: Int64) -> CookieAttribute;
pub fn attribute_domain(value: String) -> CookieAttribute;
pub fn attribute_path(value: String) -> CookieAttribute;
pub fn attribute_secure() -> CookieAttribute;
pub fn attribute_http_only() -> CookieAttribute;
pub fn attribute_same_site(policy: SameSite) -> CookieAttribute;
pub fn attribute_extension(name: String, value: String) -> CookieAttribute;
pub fn attribute_extension_flag(name: String) -> CookieAttribute;
```

Types: `Cookie`, `CookiePair`, `SetCookie`, `CookieAttribute`, `CookieAttributeKind`, `SameSite`,
`CookieLimits`, `CookieError`.

## Examples

```stark
let cookie = parse_cookie(&String::from("session=abc123; theme=dark"), default_limits())?;
// cookie.pairs[0].name == "session", cookie.pairs[1].value == "dark"

let set_cookie = parse_set_cookie(
    &String::from("session=abc; Path=/; Secure; HttpOnly; SameSite=Lax"),
    default_limits(),
)?;
// set_cookie.attributes.len() == 4
```

## Syntax

Request cookies are `name=value` pairs separated by `;`, with optional spaces or tabs around the
separators. Response cookies are one such pair followed by zero or more attributes.

Cookie names are RFC 6265 tokens, validated with `stark_ascii::is_tchar` — the token rule lives in
`stark-ascii` and is never restated here. Cookie *values* are `cookie-octet`, a different and wider
set that `stark-ascii` does not classify, so this package carries that one classifier itself.

A value may be bare or wrapped in double quotes. **Inside quotes a space is also accepted**, which
is a deliberate extension over RFC 6265's `cookie-octet`: `session="hello world"` is the case it
exists for. The quotes are syntax rather than value — they are stripped on parse and re-added on
format only when the bare form would not parse back.

Whitespace terminates a bare value, but whatever follows must be a separator. Trailing junk is an
error rather than something silently dropped, and a browser tolerating it is not a reason to.

## Attributes

`Expires`, `Max-Age`, `Domain`, `Path`, `Secure`, `HttpOnly` and `SameSite` are recognised;
attribute names compare ASCII case-insensitively, so `secure`, `Secure` and `SECURE` are the same
attribute. Anything else is preserved as an extension — with a value (`Priority=High`) or without
(`Foo`) — because an unknown attribute is a legal extension point and must never fail the whole
value.

- **`Max-Age`** parses to `Int64`. Overflow is an error, never a clamp. STARK traps on integer
  overflow in every build mode, so the accumulator is bounded *before* it multiplies — a hostile
  thirty-digit `Max-Age` returns `NumberOverflow` instead of aborting the process.
- **`Expires`** is kept as a validated opaque string. v0.1 does not parse cookie dates: the
  repository's `stark-time` offers no date parser, and depending on it would pull in its `clock`
  capability and end this package's purity for nothing.
- **`SameSite`** accepts `Strict`, `Lax` and `None` case-insensitively. An unrecognised value is
  `InvalidSameSite`, never quietly folded onto a recognised one — a caller told `Lax` when the
  server said something else would be making a security decision on invented data.
- **`Secure` and `HttpOnly`** carry no value. `Secure=yes` is `MalformedAttribute`, not a `Secure`
  attribute with the value ignored.

## Duplicates and ordering

Duplicate cookie names and duplicate attributes are preserved in appearance order and never
collapsed. `Cookie.pairs` and `SetCookie.attributes` are the authoritative representation; nothing
here is a map, so no hash iteration order can influence behaviour.

`cookie_value` returns the **first** matching pair. A caller that needs every occurrence reads
`pairs` directly.

## Formatting

Deterministic: pairs and attributes are emitted in stored order and never sorted, recognised
attributes use canonical casing (`Max-Age`, `HttpOnly`, `SameSite`), extensions keep the casing they
arrived with, and repeated formatting of the same value is byte-identical.

A value is quoted only when the bare form would not parse back, so `a="plain"` formats as `a=plain`
while `a="two words"` keeps its quotes. `parse(format(value))` is equivalent to `value` for every
representable value.

## Limits

Every parse entry point takes an explicit `CookieLimits`; there are no hard-coded bounds and the
caller is in charge of all seven:

```stark
max_total_bytes            max_attribute_count
max_cookie_pairs           max_attribute_name_bytes
max_name_bytes             max_attribute_value_bytes
max_value_bytes
```

Each has its own error, and each is checked before the work it bounds. A quoted value is measured
by its content, not by the quotes carrying it.

## Security model

Cookie headers are untrusted input and this package treats them as hostile. The limits are the
boundary; a parser without them is a denial of service waiting for a long enough header. Errors
carry a byte offset and at most one offending byte, never a copy of the input, so no error value can
be used to echo an attacker's header into a log. No network access, no filesystem access, no
environment-dependent behaviour.
