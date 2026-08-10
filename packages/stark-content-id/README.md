# stark-content-id

`stark-content-id` represents canonical content identities. It does not calculate cryptographic hashes.

Canonical syntax:

```text
sha256:<64 lowercase hex>
```

Example:

```text
sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
```

Public API:

```stark
pub fn from_digest(digest: Digest) -> ContentId;
pub fn parse(text: &str) -> Result<ContentId, ContentIdError>;
pub fn digest(id: &ContentId) -> &Digest;
pub fn to_string(id: &ContentId) -> String;
pub fn equals(left: &ContentId, right: &ContentId) -> Bool;
```

The package depends on `stark-digest` for digest representation, parsing, hexadecimal
canonicalisation, and equality. It does not depend directly on `stark-hex`.

`stark-content-id` is pure STARK. It has no native provider and declares no host capability.

Supported algorithm in v0.1: SHA-256.

Composition example:

```stark
let digest = stark_sha256::hash(bytes);
let id = stark_content_id::from_digest(digest);
```
