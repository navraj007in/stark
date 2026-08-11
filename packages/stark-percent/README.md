# stark-percent

Strict RFC percent encoding and decoding for STARK URLs and query components.

Encode sets. `PathSegment` encodes every byte outside the unreserved set, which is also the
strictest generic-component encoding, so `stark-url` reuses it for query parameter names and
values rather than declaring a second set with identical output.

- `PathSegment`
- `Path`
- `QueryComponent`

## API Summary

```stark
pub enum PercentEncodeSet {
    PathSegment,
    Path,
    QueryComponent,
}

pub enum PercentError {
    IncompleteEscape(UInt64),
    InvalidHexDigit(UInt64, UInt8),
    OutputTooLarge,
}

pub fn encode(input: &[UInt8], set: PercentEncodeSet) -> String;
pub fn decode(input: &str) -> Result<Vec<UInt8>, PercentError>;
```
