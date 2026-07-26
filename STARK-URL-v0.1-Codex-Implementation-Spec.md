# STARK URL v0.1 Implementation Specification

## Package

`stark-url` v0.1 provides bounded, deterministic parsing and encoding for HTTP origin-form request targets. It is intentionally not a browser URL implementation.

## Supported Inputs

The parser accepts authority-free request targets required by a sequential HTTP server:

```text
/health
/users/123
/search?q=stark&page=2
/files/a%20b.txt
```

The input is split into path and query before percent decoding. Structural separators are therefore preserved:

- `/` separates path segments in the path.
- `?` separates path from query.
- `&` separates query pairs.
- `=` separates a query parameter name from its value.
- Encoded separators such as `%2F`, `%26`, and `%3D` are decoded as data after splitting.

## Public API

```stark
pub enum UrlErrorKind {
    InvalidPercentEscape,
    InvalidUtf8,
    PercentDecodedNonAsciiBlocked,
    InvalidQueryPair,
    TooManyQueryParameters,
    InputTooLarge,
    PositionOverflow,
}

pub struct UrlError {
    pub kind: UrlErrorKind,
    pub offset: UInt64,
}

pub struct UrlLimits {
    pub max_input_bytes: UInt64,
    pub max_query_parameters: UInt64,
}

pub struct QueryParameter {
    pub name: String,
    pub value: String,
}

pub struct RequestTarget {
    pub path: String,
    pub query: Vec<QueryParameter>,
}

pub fn default_limits() -> UrlLimits;

pub fn percent_encode_component(input: &str) -> String;

pub fn percent_decode(input: &str) -> Result<String, UrlError>;

pub fn parse_request_target(input: &str) -> Result<RequestTarget, UrlError>;

pub fn parse_request_target_with_limits(
    input: &str,
    limits: &UrlLimits,
) -> Result<RequestTarget, UrlError>;

pub fn encode_query(parameters: &Vec<QueryParameter>) -> String;
```

## Encoding Policy

`percent_encode_component` leaves RFC 3986 unreserved ASCII bytes unchanged:

```text
ALPHA DIGIT - . _ ~
```

All other UTF-8 bytes are encoded as uppercase `%HH`. This means `/`, `?`, `&`, `=`, space, `%`, and non-ASCII UTF-8 bytes are encoded when passed as component data.

`encode_query` emits parameters in input order, joining each encoded `name=value` pair with `&`. Empty values are encoded as `name=`.

## Decoding Policy

`percent_decode` accepts only strict `%HH` escapes and validates the decoded byte stream as UTF-8. Errors report the original input byte offset:

- malformed or incomplete `%HH`: `InvalidPercentEscape`;
- decoded invalid UTF-8: `InvalidUtf8`;
- offset arithmetic overflow: `PositionOverflow`.

Direct UTF-8 input is accepted when valid. No normalization is performed.

Implementation note: the current STARK package surface does not expose a stable way to construct
a `String` from a validated UTF-8 byte vector or from a runtime Unicode scalar value. v0.1 source
therefore validates percent-decoded UTF-8 but returns `PercentDecodedNonAsciiBlocked` for escaped
non-ASCII decoded bytes until that compiler/runtime API lands. Direct non-ASCII input that does
not require byte reconstruction is still valid UTF-8 input for the language, but this package's
current decoder rebuilds output byte-by-byte and is constrained by the same blocker.

The conceptual API used `&[QueryParameter]` for `encode_query`; the current compiler does not
coerce `&Vec<QueryParameter>` to a slice at package call sites, so v0.1 freezes the implemented
surface as `&Vec<QueryParameter>` until slice coercion is available.

## Request Target Policy

`parse_request_target`:

- rejects inputs larger than `default_limits().max_input_bytes`;
- decodes the path after splitting off the query;
- preserves repeated query keys and parameter order;
- accepts empty query values (`a=`);
- rejects query components without `=` as `InvalidQueryPair`;
- rejects more than `max_query_parameters` parameters as `TooManyQueryParameters`;
- does not normalize `.`, `..`, repeated slashes, case, or Unicode.

## Explicit Exclusions

The package does not parse schemes, authorities, usernames, passwords, hosts, ports, DNS, IDNA, punycode, fragments, browser correction rules, base URL resolution, filesystem paths, `+` as space, routing, or HTTP framing.
