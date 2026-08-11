# stark-urlencoded

Shared URL `name=value` parsing and serialization for STARK.

`stark-urlencoded` owns the common pair scanning, serialization bounds, and percent-error
propagation used by ordinary URL queries and `application/x-www-form-urlencoded` form bodies.

- Query mode treats `+` as a literal plus.
- Form mode decodes a bare `+` as space and serializes space as `+`.
- Percent encoding and decoding are delegated to `stark-percent`.
- ASCII byte conversion is delegated to `stark-ascii`.

Principal API:

```stark
pub fn default_limits() -> Limits;
pub fn parse_query(input: &String, limits: Limits) -> Result<Vec<Pair>, UrlEncodedError>;
pub fn serialize_query(pairs: &[Pair], limits: Limits) -> Result<String, UrlEncodedError>;
pub fn parse_form(input: &String, limits: Limits) -> Result<Vec<Pair>, UrlEncodedError>;
pub fn serialize_form(pairs: &[Pair], limits: Limits) -> Result<String, UrlEncodedError>;
```
