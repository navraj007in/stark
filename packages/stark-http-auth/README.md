# stark-http-auth

`stark-http-auth` formats and parses HTTP `Authorization` header values for two schemes:

- `Basic`
- `Bearer`

It is pure STARK, depends on `stark-base64`, declares no host capability, and requires no native
provider.

Public API:

```stark
pub fn basic(username: &str, password: &str) -> Result<String, AuthError>;
pub fn bearer(token: &str) -> Result<String, AuthError>;
pub fn parse(value: &str) -> Result<Authorization, AuthError>;
pub fn to_string(auth: &Authorization) -> String;
```

Basic example:

```text
basic("Aladdin", "open sesame") -> Ok(Basic QWxhZGRpbjpvcGVuIHNlc2FtZQ==)
```

Bearer example:

```text
bearer("abc.def") -> Bearer abc.def
```

Parsing accepts `Basic` and `Bearer` schemes case-insensitively. Rendering always uses canonical
scheme casing: `Basic` and `Bearer`.

`stark-http-auth` formats Authorization header values. It does not securely store credentials,
authenticate users, validate access tokens, verify JWT signatures, refresh OAuth tokens, or protect
transport confidentiality.

Authorization credentials should only be transmitted over an appropriately authenticated secure
transport such as HTTPS.

Basic authentication transmits reversible credentials. Bearer tokens are opaque to this package.
HTTPS is outside this package but normally required for safe use.
